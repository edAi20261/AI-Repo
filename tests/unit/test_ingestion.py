"""Unit tests for the ingestion pipeline — no keys, no services.

Covers normalisation, chunking, config resolution, topic extraction/dedupe, the
vision JSON parsing and factory, the backend resolver, and one end-to-end run.
"""

from __future__ import annotations

import os

import pytest

from ingestion.config import Config
from ingestion.extractors import LLMDependencyDeriver, LLMTopicExtractor
from ingestion.llm_provider import (
    VISION_STUB_SENTINEL,
    HashEmbedder,
    OpenRouterVisionClient,
    StubVisionClient,
    get_vision_client,
)
from ingestion.backend import (
    FixtureBackendClient,
    discipline_of,
    resolve_lesson,
)
from ingestion.context import PipelineContext
from ingestion.schemas import FigureCaption, PageTranscription
from ingestion.graph import run_ingestion
from ingestion.state import Chunk, Dependency, Discipline, Element, new_state
from ingestion.stages.chunk import run as run_chunk
from ingestion.stages.dependencies import _clean, run as run_dependencies
from ingestion.stages.extract import run as run_extract
from ingestion.stages.topics import _match_one_to_one, run as run_topics
from ingestion.stores import DryRunChunkStore
from ingestion.utils.normalise import normalise_text
from shared.tokens import count_tokens


# --- normalisation ---------------------------------------------------------

def test_nfkc_folds_arabic_presentation_forms():
    # Presentation-form glyphs (what the PDF text layer stores) must fold to the
    # logical characters a student actually types, or retrieval never matches.
    presentation = "ﻣﻭﺎ"  # standalone/isolated Arabic glyph forms
    out = normalise_text(presentation)
    assert out != presentation
    import unicodedata

    assert out == unicodedata.normalize("NFKC", presentation)


def test_strip_tatweel_removed():
    assert normalise_text("مدرﺳـــــة") == normalise_text("مدرسة").replace("ـ", "")
    assert "ـ" not in normalise_text("كتاـــب")


def test_latin_ligature_folded():
    assert normalise_text("ﬁle") == "file"  # U+FB01 fi ligature


def test_newlines_preserved_spaces_collapsed():
    out = normalise_text("Heading\n\n\n\nbody   text\there")
    assert "\n\n" in out
    assert "\n\n\n" not in out
    assert "   " not in out


# --- token counting --------------------------------------------------------

def test_count_tokens_scales_with_length():
    assert count_tokens("") == 0
    small = count_tokens("one two three")
    big = count_tokens("one two three " * 20)
    assert big > small > 0


# --- chunking --------------------------------------------------------------

def _state_with(elements):
    st = new_state(
        lecture_id="lec-1", subject_id="sub-1", source_uri="x.pdf", grade="prep_2",
        subject_name="Science EN", lesson_title="Combustion", lesson_order=3,
    )
    st["elements"] = elements
    return st


class _Ctx:
    """Minimal ctx exposing just the config the chunk stage reads."""

    def __init__(self, config):
        self.config = config


def _config():
    return Config.load()


def test_chunk_respects_max_tokens():
    cfg = _config()
    max_tokens = cfg.get("ingestion.chunk.max_tokens")
    # 30 paragraphs of ~60 tokens each => must split into several chunks.
    para = " ".join(["word"] * 45)
    elements = [Element(text=para, page=i + 1, language="en") for i in range(30)]
    st = run_chunk(_state_with(elements), _Ctx(cfg))
    chunks = st["chunks"]
    assert len(chunks) > 1
    for c in chunks:
        assert count_tokens(c.text) <= max_tokens * 1.1  # small tolerance for overlap


def test_chunk_carries_heading_path_and_pages():
    cfg = _config()
    elements = [
        Element(text="Unit 3", page=2, is_heading=True, level=1, language="en"),
        Element(text="Fractions", page=2, is_heading=True, level=2, language="en"),
        Element(text="A fraction represents a part of a whole. " * 8, page=2, language="en"),
        Element(text="More on fractions. " * 8, page=3, language="en"),
    ]
    st = run_chunk(_state_with(elements), _Ctx(cfg))
    c = st["chunks"][0]
    assert "Unit 3" in c.heading_path and "Fractions" in c.heading_path
    assert c.page_from == 2 and c.page_to == 3
    assert c.lecture_id == "lec-1"
    assert c.namespace == "syllabus" and c.subject_id == "sub-1"


def test_chunk_strips_trailing_punctuation_from_heading_path():
    # A heading carrying a trailing comma must not reach heading_path as "Oxidation," (U3).
    cfg = _config()
    elements = [
        Element(text="Combustion,", page=1, is_heading=True, level=1, language="en"),
        Element(text="Oxidation، ", page=1, is_heading=True, level=2, language="en"),
        Element(text="Oxidation is a reaction with oxygen. " * 8, page=1, language="en"),
    ]
    st = run_chunk(_state_with(elements), _Ctx(cfg))
    path = st["chunks"][0].heading_path
    assert path == "Combustion > Oxidation"
    assert "," not in path and "،" not in path


def test_chunk_empty_elements_yields_no_chunks():
    st = run_chunk(_state_with([]), _Ctx(_config()))
    assert st["chunks"] == []


def test_chunk_carries_lecture_citation_labels():
    # subject_name/lesson_title/lesson_order flow state -> chunk -> Qdrant payload,
    # so citations can render "Subject · Lesson N: Title".
    elements = [Element(text="A paragraph about fire. " * 30, page=1, language="en")]
    c = run_chunk(_state_with(elements), _Ctx(_config()))["chunks"][0]
    assert c.subject_name == "Science EN"
    assert c.lesson_title == "Combustion"
    assert c.lesson_order == 3


# --- config ----------------------------------------------------------------

def test_llm_override_merges_over_base():
    cfg = _config()
    base = cfg.llm_for()
    et = cfg.llm_for("extract_topics")
    # extract_topics overrides provider/model/temperature, inherits the rest.
    # It is vertex now: ingestion's two chat calls follow vision onto the
    # account that holds this project's credit, while every task that answers
    # a student stays on OpenRouter.
    assert et["temperature"] == 0.0
    assert et["provider"] == "vertex"
    assert et["max_tokens"] == base["max_tokens"]


def test_unknown_task_falls_back_to_base():
    cfg = _config()
    assert cfg.llm_for("no_such_task") == cfg.llm_for()


def test_secrets_never_come_from_yaml():
    # A behavioural key is readable; a secret name is not in the yaml at all.
    cfg = _config()
    assert cfg.get("ingestion.ocr.render_dpi") == 300
    assert cfg.get("ANTHROPIC_API_KEY") is None


# --- topic extraction (LLM-agnostic) ---------------------------------------

class _FakeMsg:
    def __init__(self, content):
        self.content = content


class _FakeLLM:
    """Stands in for any chat model — just needs .invoke(str) -> message."""

    def __init__(self, content):
        self._content = content
        self.last_prompt = None

    def invoke(self, prompt):
        self.last_prompt = prompt
        return _FakeMsg(self._content)


def test_topic_extractor_parses_fenced_json():
    payload = """Sure, here you go:
```json
[
  {"label": "Factoring quadratics", "description": "Split into binomials", "bloom_level": "apply"},
  {"label": "Solving linear equations", "description": "Isolate x", "bloom_level": "APPLY"},
  {"label": "Naming shapes", "description": "identify", "bloom_level": "banana"}
]
```"""
    ex = LLMTopicExtractor(_FakeLLM(payload), min_topics=3, max_topics=8)
    topics = ex.extract(["some chunk text"])
    assert [t["label"] for t in topics][:2] == [
        "Factoring quadratics",
        "Solving linear equations",
    ]
    assert topics[1]["bloom_level"] == "apply"  # case-normalised
    assert topics[2]["bloom_level"] is None  # invalid bloom dropped, topic kept


def test_topic_extractor_clamps_to_max():
    arr = ",".join(
        f'{{"label": "t{i}", "description": "d", "bloom_level": "understand"}}'
        for i in range(12)
    )
    ex = LLMTopicExtractor(_FakeLLM(f"[{arr}]"), min_topics=3, max_topics=8)
    assert len(ex.extract(["x"])) == 8


def test_topic_extractor_reports_prose_instead_of_swallowing_it():
    """Prose is a failure to comply, not a lesson with no skills in it."""
    from shared.structured import StructuredOutputError

    ex = LLMTopicExtractor(_FakeLLM("I could not find any topics."),
                           min_topics=3, max_topics=8)
    with pytest.raises(StructuredOutputError):
        ex.extract(["x"])
    assert ex.extract([]) == []  # no chunks -> no LLM call needed


def test_topic_extractor_accepts_a_genuinely_empty_answer():
    ex = LLMTopicExtractor(_FakeLLM('{"topics": []}'), min_topics=3, max_topics=8)
    assert ex.extract(["x"]) == []


class _SeqLLM:
    """Returns a different payload per invoke; records the call count."""

    def __init__(self, payloads):
        self._payloads = payloads
        self.calls = 0

    def invoke(self, prompt):
        payload = self._payloads[min(self.calls, len(self._payloads) - 1)]
        self.calls += 1
        return _FakeMsg(payload)


def test_topic_extractor_batches_over_token_budget():
    # Three big chunks, a tiny budget => one batch per chunk => three calls, unioned.
    big = " ".join(["word"] * 400)
    payloads = [
        '[{"label": "Alpha", "description": "d", "bloom_level": "apply"}]',
        '[{"label": "Beta", "description": "d", "bloom_level": "apply"}]',
        '[{"label": "Alpha", "description": "d", "bloom_level": "apply"}]',  # dup
    ]
    llm = _SeqLLM(payloads)
    ex = LLMTopicExtractor(llm, min_topics=1, max_topics=8, max_input_tokens=50)
    topics = ex.extract([big, big, big])
    assert llm.calls == 3               # batched, not one oversized prompt
    assert [t["label"] for t in topics] == ["Alpha", "Beta"]  # label-deduped union


def test_topic_extractor_uses_structured_output_when_available():
    from ingestion.schemas import TopicItem, TopicList

    container = TopicList(topics=[TopicItem(label="Fractions", description="d", bloom_level="apply")])

    class _StructuredLLM:
        def __init__(self):
            self.structured_called = False

        def with_structured_output(self, model_cls):
            self.structured_called = True
            self._model = model_cls
            return self

        def invoke(self, prompt):
            return container

    llm = _StructuredLLM()
    ex = LLMTopicExtractor(llm, min_topics=1, max_topics=8)
    out = ex.extract(["chunk text"])
    assert llm.structured_called is True
    assert out == [{"label": "Fractions", "description": "d", "bloom_level": "apply"}]


class _FakeExtractor:
    def __init__(self, topics):
        self._topics = topics
        self.is_stub = False

    def extract(self, chunk_texts, language="en"):
        return list(self._topics)


class _FakeCtx:
    def __init__(self, config, extractor, embedder, store):
        from ingestion.stores import DryRunVectorStore
        self.config = config
        self.topic_extractor = extractor
        self.embedder = embedder
        self.chunk_store = store
        self.vector_store = DryRunVectorStore()   # captures topic upserts


def _embedded_chunk(text, embedder, **kw):
    c = Chunk(
        lecture_id="lec-1", text=text, heading_path="", page_from=1, page_to=1,
        language="en", order_index=0, subject_id="sub-1", **kw
    )
    c.embedding = embedder.embed([text])[0]
    return c


def test_topics_stage_tags_chunks_and_writes_new_topics():
    cfg = _config()
    emb = HashEmbedder(dimension=64)
    store = DryRunChunkStore()
    extractor = _FakeExtractor(
        [{"label": "Fractions", "description": "parts of a whole", "bloom_level": "understand"}]
    )
    chunks = [_embedded_chunk("adding fractions together", emb),
              _embedded_chunk("subtracting fractions", emb)]
    st = new_state(lecture_id="lec-1", subject_id="sub-1", source_uri="x.pdf")
    st["chunks"] = chunks
    ctx = _FakeCtx(cfg, extractor, emb, store)
    st = run_topics(st, ctx)

    assert st["stats"]["topics"]["extracted"] == 1
    assert st["stats"]["topics"]["new"] == 1
    # Every chunk gets at least its best-matching topic id.
    assert all(c.topic_ids for c in chunks)
    assert len(store.topics) == 1  # the new topic was persisted
    # The topic's label vector is indexed in the Qdrant topics collection.
    assert len(ctx.vector_store.topic_points) == 1
    assert ctx.vector_store.topic_points[0]["payload"]["label"] == "Fractions"


def test_topics_stage_reuses_lecture_topic_id():
    cfg = _config()
    emb = HashEmbedder(dimension=64)
    existing_vec = emb.embed(["Fractions"])[0]

    class _StoreWithLectureTopics(DryRunChunkStore):
        def read_lecture_topics(self, lecture_id):
            return [("existing-topic-id", "Fractions", existing_vec)]

    store = _StoreWithLectureTopics()
    extractor = _FakeExtractor([{"label": "Fractions", "description": "d", "bloom_level": "understand"}])
    chunks = [_embedded_chunk("fractions", emb)]
    st = new_state(lecture_id="lec-1", subject_id="sub-1", source_uri="x.pdf")
    st["chunks"] = chunks
    st = run_topics(st, _FakeCtx(cfg, extractor, emb, store))

    # Same label -> cosine 1.0 -> reuse the lecture's old id, do not re-write it.
    assert st["topics"][0]["topic_id"] == "existing-topic-id"
    assert st["topics"][0]["reused"] is True
    assert store.topics == []
    assert chunks[0].topic_ids == ["existing-topic-id"]


# --- dependency derivation -------------------------------------------------

def test_dependency_deriver_parses_and_filters():
    payload = """[
      {"prereq": "adding fractions", "strength": "required", "rationale": "combine terms"},
      {"prereq": "not a candidate", "strength": "required"},
      {"prereq": "multiplying fractions", "strength": "bogus"}
    ]"""
    deriver = LLMDependencyDeriver(_FakeLLM(payload))
    out = deriver.derive(
        "solving equations", ["adding fractions", "multiplying fractions"]
    )
    # Unknown candidate dropped; invalid strength -> required.
    assert [d["prereq"] for d in out] == ["adding fractions", "multiplying fractions"]
    assert out[0]["strength"] == "required"
    assert out[1]["strength"] == "required"


def _dep(a, b):
    return Dependency(topic_id=a, prereq_topic_id=b)


def test_dependency_strength_matches_the_platform_enum():
    """ai.dependency_strength is (required, related) in the platform database."""
    from ingestion.schemas import DEPENDENCY_STRENGTHS, DependencyItem

    assert DEPENDENCY_STRENGTHS == {"required", "related"}
    # A model reaching for another word lands on the nearest real value, not a default.
    assert DependencyItem(prereq="p", strength="recommended").strength == "related"
    assert DependencyItem(prereq="p", strength="Optional").strength == "related"
    assert DependencyItem(prereq="p", strength="essential").strength == "required"
    assert DependencyItem(prereq="p", strength="banana").strength == "required"


def test_clean_drops_self_edges_and_duplicates():
    edges = [_dep("A", "A"), _dep("A", "B"), _dep("A", "B")]
    kept = _clean(edges, drop_self=True, max_out_degree=5, break_cycles=True)
    assert [(e.topic_id, e.prereq_topic_id) for e in kept] == [("A", "B")]


def test_clean_caps_out_degree():
    edges = [_dep("A", "B"), _dep("A", "C"), _dep("A", "D")]
    kept = _clean(edges, drop_self=True, max_out_degree=2, break_cycles=True)
    assert len(kept) == 2 and all(e.topic_id == "A" for e in kept)


def test_clean_breaks_cycles():
    # A->B kept; B->A would close a cycle, so it is dropped.
    edges = [_dep("A", "B"), _dep("B", "A")]
    kept = _clean(edges, drop_self=True, max_out_degree=5, break_cycles=True)
    assert [(e.topic_id, e.prereq_topic_id) for e in kept] == [("A", "B")]


class _FakeDeriver:
    def __init__(self, mapping):
        self._mapping = mapping   # topic label -> list of prereq dicts
        self.is_stub = False

    def derive(self, topic_label, candidate_labels):
        return list(self._mapping.get(topic_label, []))


def test_match_one_to_one_no_double_claim():
    # Two new topics, one existing: only the closest reuses the id, the other mints.
    a = [1.0, 0.0]
    b = [0.0, 1.0]
    existing = [("old-a", "A", a)]
    # new[0] == a (sim 1.0), new[1] == a too (both want old-a); only one may win.
    got = _match_one_to_one([a, a], existing, 0.6)
    assert sorted(got, key=str) == [None, "old-a"]  # exactly one reuses, no double-claim
    # Below-threshold match mints new; distinct topics keep distinct ids.
    assert _match_one_to_one([b], existing, 0.6) == [None]
    # No existing topics -> everything is new.
    assert _match_one_to_one([a, b], [], 0.6) == [None, None]


def test_dependencies_stage_writes_edges():
    cfg = _config()
    store = DryRunChunkStore()
    deriver = _FakeDeriver(
        {"solving equations": [
            {"prereq": "adding fractions", "strength": "required"},
            {"prereq": "multiplying fractions", "strength": "recommended"},  # alias -> related
        ]}
    )

    class _Ctx2:
        def __init__(self):
            self.config = cfg
            self.dependency_deriver = deriver
            self.chunk_store = store

    st = new_state(lecture_id="l", subject_id="s", source_uri="x.pdf")
    st["topics"] = [
        {"topic_id": "T-solve", "label": "solving equations"},
        {"topic_id": "T-add", "label": "adding fractions"},
        {"topic_id": "T-mul", "label": "multiplying fractions"},
    ]
    st = run_dependencies(st, _Ctx2())

    assert st["stats"]["dependencies"]["edges"] == 2
    assert {(d["topic_id"], d["prereq_topic_id"]) for d in st["dependencies"]} == {
        ("T-solve", "T-add"),
        ("T-solve", "T-mul"),
    }
    assert len(store.dependencies) == 2


def test_dependencies_stage_caps_candidates():
    cfg = _config()
    cfg._data["ingestion"]["dependencies"]["max_candidates"] = 3
    emb = HashEmbedder(dimension=64)
    store = DryRunChunkStore()

    seen: dict[str, list[str]] = {}

    class _RecordingDeriver:
        is_stub = False

        def derive(self, topic_label, candidate_labels):
            seen[topic_label] = list(candidate_labels)
            return []

    class _CtxCap:
        def __init__(self):
            self.config = cfg
            self.dependency_deriver = _RecordingDeriver()
            self.chunk_store = store
            self.embedder = emb

    st = new_state(lecture_id="l", subject_id="s", source_uri="x.pdf")
    st["topics"] = [{"topic_id": f"T{i}", "label": f"topic {i}"} for i in range(8)]
    run_dependencies(st, _CtxCap())

    # Every target sees exactly the cap: the 3 nearest, itself excluded.
    all_labels = {f"topic {i}" for i in range(8)}
    assert len(seen) == 8
    for target, cands in seen.items():
        assert len(cands) == 3
        assert target not in cands
        assert set(cands).issubset(all_labels)


# --- vision stub contract --------------------------------------------------

def test_vision_stub_is_marked_and_safe():
    vc = StubVisionClient()
    assert vc.is_stub is True

    # One call per page returns text + figure boxes together.
    page = vc.transcribe_page(b"", language_hint="ar")
    assert VISION_STUB_SENTINEL in page.text
    assert page.figures == []  # no fabricated boxes until a real model runs

    # math=True selects the maths-fidelity prompt but keeps the same shape.
    maths = vc.transcribe_page(b"", math=True)
    assert VISION_STUB_SENTINEL in maths.text
    assert maths.figures == []

    # Captioning stays a separate call on the crop.
    cap = vc.caption_figure(b"", heading_path="Unit 3")
    assert VISION_STUB_SENTINEL in cap.caption and VISION_STUB_SENTINEL in cap.alt_text

    # The old split methods are gone.
    assert not hasattr(vc, "transcribe_math")
    assert not hasattr(vc, "detect_figures")
    # Stub never flags a page unreadable — it has nothing to judge.
    assert page.unreadable is False


# --- real vision client: JSON parsing + factory ----------------------------

def test_a_vision_reply_wrapped_in_prose_is_still_read():
    """Vision models fence their JSON or introduce it. The parsing lives in
    shared.structured now, so the vision client no longer carries a private
    scraper of its own — there were two of those."""
    from shared.structured import recover

    fenced = 'Here you go:\n```json\n{"unreadable": false, "text": "hi"}\n```'
    assert recover(fenced, PageTranscription).text == "hi"
    bare = 'prose {"text": "x", "figures": []} trailing'
    assert recover(bare, PageTranscription).text == "x"
    assert recover("no json here", PageTranscription) is None


def test_a_malformed_figure_box_costs_the_figure_not_the_page():
    """The regression this guards. The old hand-written parser dropped a bad box
    and kept the page; validating the object whole would fail the page instead,
    sending a page of readable text to manual review over one bad crop."""
    page = PageTranscription.model_validate({
        "unreadable": False,
        "text": "the page",
        "figures": [{"bbox": [1, 2, 3, 4]}, {"bbox": "bad"}, {"nope": 1}],
    })
    assert page.text == "the page"
    assert page.unreadable is False
    assert [f.bbox for f in page.figures] == [(1.0, 2.0, 3.0, 4.0)]


def test_an_unreadable_page_says_so():
    page = PageTranscription.model_validate(
        {"unreadable": True, "text": "", "figures": []})
    assert page.unreadable is True
    assert page.text == ""


def test_a_caption_with_nothing_in_it_is_empty_not_absent():
    cap = FigureCaption.model_validate({"caption": "a heart", "alt_text": "diagram"})
    assert cap.caption == "a heart" and cap.alt_text == "diagram"
    empty = FigureCaption.model_validate({})
    assert empty.caption == "" and empty.alt_text == ""


def test_vision_model_id_is_locked():
    """Pinned in config, not code, and it must move deliberately.

    Was qwen/qwen3-vl-235b-a22b-instruct via OpenRouter. Measured on eight
    pages of a scanned Arabic question bank, gemini via vertex reads the same
    pages for $0.0065 against $0.0127 and answers in ~6.9s against ~60s — and
    bills to the Cloud account where this project's credit is.
    """
    assert Config.load().get("vision.model") == "google/gemini-2.5-flash"


def test_vision_factory_selects_real_client_with_key():
    """A real client whenever the gateway vision points at has its credential.

    The credential is no longer always OPENROUTER_API_KEY: vertex authenticates
    with a service account, so the factory has to ask the provider rather than
    one hardcoded variable.
    """
    cfg = Config.load()
    cfg.override("vision.provider", "openrouter")
    saved = os.environ.get("OPENROUTER_API_KEY")
    try:
        os.environ["OPENROUTER_API_KEY"] = "test-key"
        assert isinstance(get_vision_client(cfg), OpenRouterVisionClient)
    finally:
        if saved is None:
            os.environ.pop("OPENROUTER_API_KEY", None)
        else:
            os.environ["OPENROUTER_API_KEY"] = saved


def test_vision_factory_falls_back_to_stub_without_key():
    cfg = Config.load()
    saved = os.environ.get("OPENROUTER_API_KEY")
    try:
        os.environ.pop("OPENROUTER_API_KEY", None)
        assert isinstance(get_vision_client(cfg), StubVisionClient)
    finally:
        if saved is not None:
            os.environ["OPENROUTER_API_KEY"] = saved


# --- discipline + backend resolver (fixture, no backend/DB/S3) --------------

def _make_pdf(path, text=None):
    """A tiny born-digital text PDF with no drawings (no fraction bars)."""
    import fitz

    text = text or ("The plant uses the light of the sun and it is the energy "
                    "that is stored in the cell for the growth of the plant. " * 4)
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 100), text, fontsize=11)
    doc.save(str(path))
    doc.close()


def _write_fixture(directory, discipline_name="Mathematics", code="MATH8"):
    _make_pdf(directory / "lesson.pdf")
    manifest = directory / "fixtures.yaml"
    manifest.write_text(
        "lessons:\n"
        "  lesson-1:\n"
        "    path: lesson.pdf\n"
        "    subject_id: subj-1\n"
        "    subject:\n"
        "      grade_level: g8\n"
        "      language: en\n"
        f"      name: {discipline_name}\n"
        f"      code: {code}\n",
        encoding="utf-8",
    )
    return manifest


def test_discipline_derivation_from_name_code():
    assert discipline_of({"name": "Mathematics", "code": "MATH8"}) is Discipline.MATH
    assert discipline_of({"name": "علوم"}) is Discipline.SCIENCE
    assert discipline_of({"name": "Social Studies"}) is Discipline.SOCIAL_STUDIES
    assert discipline_of({"name": "English Language"}) is Discipline.LANGUAGE
    assert discipline_of({"name": "Philosophy"}) is Discipline.OTHER
    # An explicit backend field, if one ever appears, wins over the keywords.
    assert discipline_of({"name": "x", "discipline": "science"}) is Discipline.SCIENCE
    assert discipline_of({"discipline": "bogus"}) is Discipline.OTHER


def test_grade_labels_normalise_to_the_platform_enum():
    """The API serialises the enum by name, the column stores its value."""
    from ingestion.utils.grades import to_platform_grade

    assert to_platform_grade("GRADE_7") == "Grade 7"
    assert to_platform_grade("Grade 7") == "Grade 7"
    assert to_platform_grade("g8") == "Grade 8"
    assert to_platform_grade("12") == "Grade 12"
    assert to_platform_grade("year 3") == "Grade 3"
    assert to_platform_grade(None) is None
    assert to_platform_grade("") is None
    assert to_platform_grade("KG1") is None
    assert to_platform_grade("GRADE_13") is None


def test_fixture_backend_resolves_lesson(tmp_path):
    manifest = _write_fixture(tmp_path)
    client = FixtureBackendClient(manifest)

    assert client.list_pending_lesson_ids() == ["lesson-1"]
    src = resolve_lesson(client, "lesson-1")
    assert src.subject_id == "subj-1"
    assert src.grade == "Grade 8" and src.language == "en"
    assert src.discipline is Discipline.MATH
    assert src.subject_is_math is True
    assert src.source_uri.startswith("file://") and src.source_uri.endswith("lesson.pdf")


def test_resolver_uses_the_platform_declared_body_text(tmp_path):
    """A text-only lesson has no media at all; the body text is the source."""

    class _TextLessonClient(FixtureBackendClient):
        def get_lesson(self, lesson_id):
            return {
                "id": lesson_id,
                "subject_id": "subj-1",
                "text_content": "# Lab Safety\n\nWear goggles.",
                "attachments": [],
                "ingestible_source": {"type": "text_content", "media_id": None},
            }

        def get_media_url(self, media_id):
            raise AssertionError("a text-only lesson must not ask for a media URL")

    src = resolve_lesson(_TextLessonClient(_write_fixture(tmp_path)), "lesson-9")

    assert src.text_content.startswith("# Lab Safety")
    assert src.source_uri == ""


def test_resolver_uses_the_platform_declared_attachment(tmp_path):
    """With an attachment declared, the media URL is resolved as before."""

    class _AttachmentLessonClient(FixtureBackendClient):
        def get_lesson(self, lesson_id):
            return {
                "id": lesson_id,
                "subject_id": "subj-1",
                "text_content": None,
                "attachments": [{"media_id": "media-2"}, {"media_id": "media-3"}],
                "ingestible_source": {"type": "attachment", "media_id": "media-2"},
            }

        def get_media_url(self, media_id):
            return f"https://signed.example/{media_id}.pdf?X-Amz-Signature=abc"

    src = resolve_lesson(_AttachmentLessonClient(_write_fixture(tmp_path)), "lesson-9")

    assert src.source_uri.startswith("https://signed.example/media-2.pdf")
    assert src.text_content is None


def test_resolver_falls_back_when_the_backend_declares_nothing(tmp_path):
    """Fixtures and older backends send no ingestible_source; attachments still win."""

    class _LegacyClient(FixtureBackendClient):
        def __init__(self, manifest, lesson):
            super().__init__(manifest)
            self._lesson = lesson

        def get_lesson(self, lesson_id):
            return dict(self._lesson, id=lesson_id, subject_id="subj-1")

        def get_media_url(self, media_id):
            return f"file:///media/{media_id}.pdf"

    manifest = _write_fixture(tmp_path)
    with_file = _LegacyClient(manifest, {"attachments": [{"media_id": "m-1"}]})
    with_text = _LegacyClient(manifest, {"attachments": [], "text_content": "Body only"})
    with_nothing = _LegacyClient(manifest, {"attachments": [], "text_content": None})

    assert resolve_lesson(with_file, "l").source_uri == "file:///media/m-1.pdf"
    assert resolve_lesson(with_text, "l").text_content == "Body only"
    with pytest.raises(ValueError, match="no ingestible source"):
        resolve_lesson(with_nothing, "l")


def test_extract_reads_a_lesson_body_without_any_file():
    ctx = PipelineContext.build(Config.load())
    st = new_state(
        lecture_id="l",
        subject_id="s",
        source_text="<h1>Addition</h1><p>Combining quantities of objects.</p>",
    )

    result = run_extract(st, ctx)

    assert [(e.text, e.is_heading) for e in result["elements"]] == [
        ("Addition", True),
        ("Combining quantities of objects.", False),
    ]
    assert result["figures"] == []
    assert result["stats"]["extract"]["text"] == 1
    assert result["pages"][0].route == "text"


def test_extract_flags_a_lesson_whose_body_holds_no_content():
    ctx = PipelineContext.build(Config.load())
    st = new_state(lecture_id="l", subject_id="s", source_text="<p>&nbsp;</p>")

    result = run_extract(st, ctx)

    assert result["elements"] == []
    assert result["ingestion_status"] == "needs_review"
    assert result["errors"]


def test_a_lesson_needs_a_source():
    with pytest.raises(ValueError, match="source_uri or source_text"):
        new_state(lecture_id="l", subject_id="s")


def _has_httpx():
    try:
        import httpx  # noqa: F401

        return True
    except ImportError:
        return False


def test_resolver_pipeline_end_to_end(tmp_path):
    """Resolve a lesson from a fixture and run the whole pipeline into dry-run
    stores — real Qwen vision when OPENROUTER_API_KEY (and httpx) are available,
    stub otherwise."""
    from ingestion.extractors import StubTopicExtractor

    manifest = _write_fixture(tmp_path)  # maths subject -> vision route forced
    saved = {
        k: os.environ.get(k)
        for k in ("DATABASE_URL", "QDRANT_URL", "MOONSHOT_API_KEY",
                  "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY")
    }
    try:
        # Force the placeholder so the dry-run store is selected even when a real
        # DATABASE_URL is set in .env (load_dotenv won't override an existing var).
        os.environ["DATABASE_URL"] = "postgresql+asyncpg://user:password@localhost"
        os.environ["QDRANT_URL"] = "dryrun"       # -> dry-run vector store
        for k in ("MOONSHOT_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY"):
            os.environ.pop(k, None)

        cfg = Config.load()
        cfg._data["embeddings"]["provider"] = "hash"  # no torch

        client = FixtureBackendClient(manifest)
        src = resolve_lesson(client, "lesson-1")
        assert src.subject_is_math is True

        ctx = PipelineContext.build(cfg)
        ctx._topic_extractor = StubTopicExtractor()   # not under test here
        if not (os.environ.get("OPENROUTER_API_KEY") and _has_httpx()):
            ctx.vision = StubVisionClient()

        state = new_state(
            lecture_id=src.lesson_id,
            subject_id=src.subject_id,
            source_uri=src.source_uri,
            grade=src.grade,
            language_hint=src.language,
            subject_is_math=src.subject_is_math,
        )
        result = run_ingestion(state, ctx=ctx, config=cfg)

        # Maths subject forced every page down the vision route, and the pipeline
        # completed into the dry-run index.
        assert result["stats"]["extract"]["math"] >= 1
        assert result["stats"]["extract"]["digital"] == 0
        assert result["stats"]["index"]["points_upserted"] >= 1
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_body_text_lesson_runs_end_to_end(tmp_path):
    """A lesson with no file at all still reaches the index — Farah's bug 1."""
    from ingestion.extractors import StubTopicExtractor

    saved = {
        k: os.environ.get(k)
        for k in ("DATABASE_URL", "QDRANT_URL", "MOONSHOT_API_KEY",
                  "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY")
    }
    try:
        os.environ["DATABASE_URL"] = "postgresql+asyncpg://user:password@localhost"
        os.environ["QDRANT_URL"] = "dryrun"
        for k in ("MOONSHOT_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY"):
            os.environ.pop(k, None)

        cfg = Config.load()
        cfg._data["embeddings"]["provider"] = "hash"
        ctx = PipelineContext.build(cfg)
        ctx._topic_extractor = StubTopicExtractor()

        body = (
            "<h1>Addition</h1>"
            "<p>" + ("Addition combines two quantities into a single total. " * 20) + "</p>"
            "<h2>Counting on</h2>"
            "<p>" + ("Start from the larger number and count on. " * 20) + "</p>"
        )
        state = new_state(lecture_id="lesson-text", subject_id="subj-1", source_text=body)

        result = run_ingestion(state, ctx=ctx, config=cfg)

        assert result["stats"]["extract"]["text"] == 1
        assert result["chunks"]
        assert result["chunks"][0].heading_path
        assert result["stats"]["index"]["points_upserted"] >= 1
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_subject_is_math_forces_vision_route(tmp_path):
    """A maths page with NO fraction bars still goes to vision when the subject
    is maths — the whole point of subject_is_math."""
    pdf = tmp_path / "algebra.pdf"
    _make_pdf(pdf)  # plain text, zero drawings -> horizontal_rule_count == 0

    cfg = Config.load()
    ctx = PipelineContext.build(cfg)
    ctx.vision = StubVisionClient()  # deterministic, no network

    def _routes(subject_is_math):
        st = new_state(
            lecture_id="l", subject_id="s",
            source_uri=pdf.resolve().as_uri(), subject_is_math=subject_is_math,
        )
        return run_extract(st, ctx)["stats"]["extract"]

    forced = _routes(True)
    assert forced["math"] == 1 and forced["digital"] == 0

    # Same page, non-maths subject: the fraction-bar heuristic finds nothing, so
    # it stays on the plain-text digital route.
    plain = _routes(False)
    assert plain["digital"] == 1 and plain["math"] == 0


# --- born-digital garbage gate ---------------------------------------------

_GARBAGE_CFG = {
    "symbol_ratio": 0.45,
    "word_check_min_chars": 80,
    "min_word_hits": 3,
    "stopwords": {
        "ar": ["في", "من", "على", "التي", "كان", "إلى"],
        "en": ["the", "of", "and", "is", "to", "in", "a", "are", "for", "with", "that"],
        "fr": ["le", "la", "les", "de", "et", "un", "une", "des", "est", "dans", "que"],
        "de": ["der", "die", "und", "ist", "den", "das", "mit", "für", "ein", "eine", "im"],
    },
}


def test_garbage_symbol_junk_flagged():
    from ingestion.utils import is_garbage

    assert is_garbage("!@#$%^&*(){}[]<>|~+=;:/?.," * 5, _GARBAGE_CFG) is True


def test_garbage_latin_cipher_flagged():
    from ingestion.utils import is_garbage

    # Letter-shaped substitution noise: no common words hit.
    assert is_garbage("Xqzwbf kjlmpv gtrhnd czxwvb pflmqr " * 4, _GARBAGE_CFG) is True


def test_garbage_arabic_cipher_flagged():
    from ingestion.utils import is_garbage

    # Arabic letters, no common function words.
    assert is_garbage("ثقخ شضظ غعذ خصث ضظغ ذقش ثخض غعذ شقخ ظضث خذق " * 3, _GARBAGE_CFG) is True


def test_garbage_passes_clean_text_in_each_language():
    from ingestion.utils import is_garbage

    clean = {
        "en": "The cell is the unit of life and it is found in all of the plants that grow.",
        "fr": "La cellule est une unite de la vie et elle est dans toutes les plantes des jardins.",
        "de": "Die Zelle ist der Baustein und das Leben ist in der Pflanze mit ein und eine im Garten.",
        "ar": "الخلية هي وحدة الحياة التي توجد في كل النباتات من حولنا وعلى الأرض كان لها دور مهم إلى اليوم.",
    }
    for lang, text in clean.items():
        assert is_garbage(text, _GARBAGE_CFG) is False, lang


def test_garbage_short_page_not_flagged():
    from ingestion.utils import is_garbage

    # Below the word-check min length: a sparse header is not garbage.
    assert is_garbage("Chapter notes", _GARBAGE_CFG) is False


class _CountingExtractor:
    """Returns a scripted sequence of results, counting extract() calls."""

    def __init__(self, results, is_stub=False):
        self._results = list(results)
        self.is_stub = is_stub
        self.calls = 0

    def extract(self, chunk_texts, language="en"):
        i = min(self.calls, len(self._results) - 1)
        self.calls += 1
        return list(self._results[i])


def _topics_state(emb):
    st = new_state(lecture_id="lec-1", subject_id="sub-1", source_uri="x.pdf")
    st["chunks"] = [_embedded_chunk("some lesson text", emb)]
    return st


def test_topics_empty_answer_flags_needs_review_without_re_asking():
    """An empty answer is a real answer: the model read the lesson and named
    nothing. Compliance retries live in the structured-call contract now, so the
    stage asks once and flags rather than paying for the same reply three times."""
    cfg = _config()
    emb = HashEmbedder(dimension=64)
    ex = _CountingExtractor([[]])
    st = run_topics(_topics_state(emb), _FakeCtx(cfg, ex, emb, DryRunChunkStore()))

    assert ex.calls == 1
    assert st["ingestion_status"] == "needs_review"
    assert any("no topics" in e for e in st.get("errors", []))


def test_topics_extraction_failure_is_flagged_not_lost():
    """A lesson keeps its chunks and its OCR when topic naming fails."""
    from shared.structured import StructuredValidationFailure

    class _Failing:
        is_stub = False
        calls = 0

        def extract(self, _texts, language="en"):
            self.calls += 1
            raise StructuredValidationFailure(
                "topics: the model did not return valid output after 2 attempts.",
                task="extract_topics", label="topics", attempts=2)

    cfg = _config()
    emb = HashEmbedder(dimension=64)
    st = run_topics(_topics_state(emb), _FakeCtx(cfg, _Failing(), emb, DryRunChunkStore()))

    assert st["ingestion_status"] == "needs_review"
    assert any("topic extraction failed" in e for e in st.get("errors", []))
    assert st["topics"] == []


def test_topics_stub_empty_stays_silent():
    cfg = _config()
    emb = HashEmbedder(dimension=64)
    ex = _CountingExtractor([[]], is_stub=True)  # stub: no key, no retries, no flag
    st = run_topics(_topics_state(emb), _FakeCtx(cfg, ex, emb, DryRunChunkStore()))

    assert ex.calls == 1
    assert st.get("ingestion_status") != "needs_review"


def test_reingest_replaces_not_duplicates():
    """Running the index stage twice for a lecture must leave one copy in each
    store, not two (delete-before-write idempotency)."""
    from ingestion.stages.index import run as run_index
    from ingestion.stores import DryRunChunkStore, DryRunVectorStore

    cs, vs = DryRunChunkStore(), DryRunVectorStore()

    class _Ctx:
        chunk_store = cs
        vector_store = vs
        config = _config()

    def _chunks():
        return [
            Chunk(lecture_id="lec-1", text="a", heading_path="", page_from=1,
                  page_to=1, language="en", order_index=0, subject_id="s",
                  embedding=[0.1, 0.2]),
            Chunk(lecture_id="lec-1", text="b", heading_path="", page_from=1,
                  page_to=1, language="en", order_index=1, subject_id="s",
                  embedding=[0.3, 0.4]),
        ]

    for _ in range(2):
        st = new_state(lecture_id="lec-1", subject_id="s", source_uri="x.pdf")
        st["chunks"] = _chunks()
        run_index(st, _Ctx())

    assert len(cs.chunks) == 2
    assert len(cs.index) == 2
    assert len(vs.points) == 2
    # One tracking row, overwritten (not duplicated) on the second ingest.
    assert len(cs.ingestion_records) == 1
    rec = cs.ingestion_records[0]
    assert rec["lecture_id"] == "lec-1" and rec["status"] == "ready"
    assert rec["chunk_count"] == 2


def test_index_records_needs_review_status():
    """A flagged lesson is tracked as needs_review with the reason."""
    from ingestion.stages.index import run as run_index
    from ingestion.stores import DryRunChunkStore, DryRunVectorStore

    cs, vs = DryRunChunkStore(), DryRunVectorStore()

    class _Ctx:
        chunk_store = cs
        vector_store = vs
        config = _config()

    st = new_state(lecture_id="lec-9", subject_id="s", source_uri="x.pdf")
    st["chunks"] = []
    st["ingestion_status"] = "needs_review"
    st["errors"] = ["1 page(s) were unreadable"]
    run_index(st, _Ctx())

    rec = cs.ingestion_records[0]
    assert rec["status"] == "needs_review"
    assert rec["needs_review_reason"] == "1 page(s) were unreadable"
    assert rec["chunk_count"] == 0


def test_figure_bbox_scaling():
    """Vision figure boxes come back normalised; they must scale to the rendered
    page's pixel size before cropping (the bug that made every scanned crop junk)."""
    from ingestion.stages.extract import _to_pixels

    W, H = 2480, 3508

    def rounded(bbox):
        return tuple(round(v) for v in _to_pixels(bbox, W, H))

    # 0..1 fractions -> pixels
    assert rounded([0.1, 0.2, 0.5, 0.6]) == (248, 702, 1240, 2105)
    # 0..1000 normalised on a large render -> same pixels
    assert rounded([100, 200, 500, 600]) == (248, 702, 1240, 2105)
    # already-pixel coordinates pass through unchanged
    assert rounded([300, 400, 1200, 1500]) == (300, 400, 1200, 1500)


def test_figure_bbox_overlap_dedupe():
    """Two images stacked at the same spot (a layered duplicate) overlap ~fully;
    distinct figures do not."""
    from ingestion.stages.extract import _bbox_overlap

    # Same map embedded twice (near-identical placement) -> ~1.0
    assert _bbox_overlap((63, 80, 341, 337), (62, 78, 341, 338)) > 0.9
    # Two different figures on the page (no vertical overlap) -> 0
    assert _bbox_overlap((63, 80, 341, 337), (62, 524, 230, 715)) == 0.0


def test_vision_fill_adds_only_missed_figures():
    """The math hybrid adds a vision figure only where get_images has nothing;
    a vision box overlapping an existing get_images figure is skipped."""
    from io import BytesIO

    from PIL import Image

    from ingestion.stages.extract import _vision_fill_figures
    from ingestion.state import Figure

    buf = BytesIO()
    Image.new("RGB", (1000, 1000), "white").save(buf, "PNG")
    png = buf.getvalue()

    # get_images already found a figure at (0,0,100,100) in PDF points (dpi 72 -> scale 1).
    existing = [Figure(lecture_id="l", page=1, s3_uri="x", bbox=(0, 0, 100, 100))]

    class _Box:
        def __init__(self, bbox):
            self.bbox = bbox

    boxes = [_Box((0.0, 0.0, 0.1, 0.1)),   # overlaps existing -> skipped
             _Box((0.5, 0.5, 0.8, 0.8))]   # empty region -> kept

    class _Up:
        def upload(self, data, key, *a, **k):
            return "file://x"

    class _Ctx:
        uploader = _Up()
        vision = None  # not used: generate_alt_text False

    ing = {"figures": {"dedupe_overlap": 0.7, "generate_alt_text": False}}
    out = _vision_fill_figures({"lecture_id": "l"}, _Ctx(), png, 1, boxes,
                               existing, 72, "en", ing)
    assert len(out) == 1  # only the non-overlapping box


def test_garbage_digital_page_routes_to_vision(tmp_path):
    """A born-digital page whose extracted text is cipher noise is routed to vision
    instead of being indexed as clean text."""
    pdf_path = tmp_path / "cipher.pdf"
    _make_pdf(pdf_path, text="Xqzwbf kjlmpv gtrhnd czxwvb pflmqr wknztd " * 4)

    cfg = Config.load()
    ctx = PipelineContext.build(cfg)
    ctx.vision = StubVisionClient()

    st = new_state(lecture_id="l", subject_id="s", source_uri=pdf_path.resolve().as_uri())
    stats = run_extract(st, ctx)["stats"]["extract"]
    assert stats["garbage"] == 1 and stats["digital"] == 0


def test_local_uploader_delete_prefix_scopes_to_lecture(tmp_path):
    """delete_prefix clears one lecture's figures and leaves others; idempotent."""
    from ingestion.utils.s3 import LocalUploader

    up = LocalUploader(str(tmp_path))
    up.upload(b"a", "figures/lec-1/p001_aa.png")
    up.upload(b"b", "figures/lec-1/p002_bb.png")
    up.upload(b"c", "figures/lec-2/p001_cc.png")

    assert up.delete_prefix("figures/lec-1/") == 2
    assert not (tmp_path / "figures" / "lec-1").exists()
    assert (tmp_path / "figures" / "lec-2" / "p001_cc.png").exists()  # untouched
    assert up.delete_prefix("figures/lec-1/") == 0  # nothing left -> no-op


def test_figure_keys_live_under_their_lecture_prefix():
    from ingestion.utils.s3 import figure_key, figure_prefix

    key = figure_key("lec-1", 3, b"crop")

    assert figure_prefix("lec-1") == "figures/lec-1/"
    assert key.startswith(figure_prefix("lec-1"))
    assert key == figure_key("lec-1", 3, b"crop")


def test_extract_downloads_a_signed_url_before_opening_it(tmp_path, monkeypatch):
    """The platform hands the pipeline a presigned URL, not a local path."""
    import httpx

    pdf_path = tmp_path / "lesson.pdf"
    _make_pdf(pdf_path)
    body = pdf_path.read_bytes()

    class _Response:
        def __init__(self):
            self.headers = {"content-length": str(len(body))}

        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

        def raise_for_status(self):
            return None

        def iter_bytes(self, chunk_size):
            yield body

    monkeypatch.setattr(httpx, "stream", lambda method, url, **kwargs: _Response())

    ctx = PipelineContext.build(Config.load())
    ctx.vision = StubVisionClient()
    st = new_state(
        lecture_id="l",
        subject_id="s",
        source_uri="https://bucket.s3.amazonaws.com/uploads/lesson.pdf?X-Amz-Signature=abc",
    )

    result = run_extract(st, ctx)

    assert result["stats"]["extract"]["pages"] == 1
    assert result["elements"]


def test_extract_continues_when_old_figures_cannot_be_cleared(tmp_path, caplog):
    from ingestion.utils.s3 import StorageError

    class _DenyingUploader:
        def delete_prefix(self, prefix):
            raise StorageError("AccessDenied")

        def upload(self, data, key, content_type="image/png"):
            return f"mem://{key}"

    pdf_path = tmp_path / "lesson.pdf"
    _make_pdf(pdf_path)
    ctx = PipelineContext.build(Config.load())
    ctx.vision = StubVisionClient()
    ctx.uploader = _DenyingUploader()
    st = new_state(lecture_id="l", subject_id="s", source_uri=pdf_path.resolve().as_uri())

    with caplog.at_level("WARNING", logger="ingestion.stages.extract"):
        result = run_extract(st, ctx)

    assert result["stats"]["extract"]
    assert "could not be cleared" in caplog.text


def test_a_failed_caption_is_counted_not_indexed_as_empty(tmp_path):
    """An empty caption used to be indistinguishable from a model with nothing to
    say — and captions are what make a figure findable."""
    class _NoCaption(StubVisionClient):
        def caption_figure(self, image_png, *, heading_path="", language_hint=None):
            return None

    pdf_path = tmp_path / "lesson.pdf"
    _make_pdf(pdf_path)
    ctx = PipelineContext.build(Config.load())
    ctx.vision = _NoCaption()
    st = new_state(lecture_id="l", subject_id="s", source_uri=pdf_path.resolve().as_uri())

    result = run_extract(st, ctx)

    assert "caption_failed" in result["stats"]["extract"]

# --- what a chunk is embedded as --------------------------------------------

def _chunk(**kw):
    from ingestion.state import Chunk
    base = dict(lecture_id="lec1", text="الكيمياء التجريبية مهدت لنهضة علمية.",
                heading_path="", page_from=3, page_to=3, language="ar", order_index=0)
    base.update(kw)
    return Chunk(**base)


def test_a_chunk_is_embedded_with_the_lesson_it_came_from():
    """A short chunk is unfindable on its own. "مهدت لنهضة علمية" says nothing
    about which lesson, unit or page it belongs to, so a query naming the lesson
    has nothing in the vector to match."""
    chunk = _chunk(subject_name="اللغة العربية",
                   lesson_title="Unit 2 · Lesson 2: Jabir ibn Hayyan",
                   heading_path="ثقافة النجاح")
    embedded = chunk.for_embedding()

    assert embedded.endswith(chunk.text)
    assert "اللغة العربية" in embedded
    assert "Unit 2 · Lesson 2: Jabir ibn Hayyan" in embedded
    assert "ثقافة النجاح" in embedded
    assert "3" in embedded.split(chunk.text)[0]


def test_the_text_a_student_reads_is_never_the_prefixed_one():
    """The prefix exists to be searched, not to be read. It must not reach the
    stored row, the citation, or the passage handed to the answering model."""
    chunk = _chunk(subject_name="Science", lesson_title="Unit 1: Matter")

    assert chunk.for_embedding() != chunk.text
    assert chunk.text == "الكيمياء التجريبية مهدت لنهضة علمية."


def test_a_chunk_that_knows_nothing_about_itself_is_embedded_as_it_is():
    """A prefix of empty fields would embed punctuation, which is worse than
    no prefix at all."""
    bare = _chunk(page_from=0, page_to=0)
    assert bare.for_embedding() == bare.text


def test_the_prefix_names_only_the_parts_that_exist():
    chunk = _chunk(lesson_title="Unit 1: Matter", page_from=0)
    prefix = chunk.for_embedding().split(chunk.text)[0]

    assert "Unit 1: Matter" in prefix
    assert "Section" not in prefix
    assert "0" not in prefix


def test_a_direct_run_labels_its_chunks_like_a_resolved_one(monkeypatch):
    """A lecture ingested from the command line carried no lesson title or
    subject name, so every citation from it was labelled by whatever the layout
    parser took for a heading: "Visualize It!", "is a straight line"."""
    import ingestion.__main__ as cli
    import ingestion.lesson as lesson

    captured = {}

    class _Backend:
        def get_subject(self, subject_id):
            return {"name": "Biology", "language": "en"}

    monkeypatch.setattr(lesson, "get_backend_client", lambda config, **k: _Backend())
    monkeypatch.setattr(cli, "run_ingestion",
                        lambda state, **k: captured.update(state) or {"stats": {}})
    cli.main(["lesson.pdf", "--lecture-id", "l1", "--subject-id", "s1",
              "--lesson-title", "The Water Cycle", "--lesson-order", "4",
              "--no-read-ahead"])
    assert captured["subject_name"] == "Biology"
    assert captured["lesson_title"] == "The Water Cycle"
    assert captured["lesson_order"] == 4
    assert captured["language_hint"] == "en"


# --- a heading longer than its column -----------------------------------------

def test_a_heading_path_never_outgrows_the_column_it_is_written_to():
    """A French textbook died on its last stage with `value too long for type
    character varying(500)`. One page had a paragraph styled large enough to
    read as a heading, the path grew past the column, and the insert took the
    whole book with it — 106 pages read, nothing written."""
    from ingestion.state import Chunk

    chunk = Chunk(
        lecture_id="l1",
        text="body",
        heading_path=" > ".join(["Apprenons le français avec des phrases longues"] * 40),
        page_from=1,
        page_to=1,
        language="fr",
        order_index=0,
    )

    assert len(chunk.heading_path) <= 500
    assert chunk.heading_path.endswith("…")
    assert chunk.heading_path.startswith("Apprenons")


def test_a_heading_path_that_fits_is_left_exactly_as_it_is():
    from ingestion.state import Chunk

    path = "Unité 1 > Leçon 2 > Les salutations"
    chunk = Chunk(lecture_id="l1", text="body", heading_path=path,
                  page_from=1, page_to=1, language="fr", order_index=0)
    assert chunk.heading_path == path
