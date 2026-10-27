from __future__ import annotations

import pytest

from shared.namespaces import BACKGROUND, SYLLABUS

SUBJECT = "7b1e5c40-0a6e-4d1a-9f3c-5d2a91c00007"
MINISTRY_LESSON = "9d3f1b20-0002-4a00-9000-000000000014"


def _conditions(flt) -> list[tuple[str, object]]:
    out = []
    for condition in flt.must or []:
        match = condition.match
        out.append((condition.key, getattr(match, "any", None) or getattr(match, "value", None)))
    return out


@pytest.fixture
def searched(monkeypatch):
    import agent.tools as tools

    seen = {}

    class _Embedder:
        def embed_query(self, _query):
            return [0.0] * 4

    def candidates(query, filters, config, **kwargs):
        seen["filter"] = filters[0]
        return {}, {}

    monkeypatch.setattr(tools, "get_embedder", lambda config: _Embedder())
    monkeypatch.setattr(tools, "_candidate_scores", candidates)
    monkeypatch.setattr(tools, "_subjects_of", lambda ids, config: [SUBJECT])
    return seen


def test_a_ministry_lesson_is_also_answered_from_its_subjects_external_books(searched):
    """External books are filed under lessons of their own, so a search scoped
    to the ministry lesson a student is on never reached them: the background
    they were ingested to be was never read."""
    from agent.tools import rag_search
    from shared.config import get_config

    rag_search("ما هي الذرة", [SYLLABUS], lecture_ids=[MINISTRY_LESSON], config=get_config())
    own, background = searched["filter"].should
    assert ("lecture_id", [MINISTRY_LESSON]) in _conditions(own)
    assert ("namespace", [SYLLABUS]) in _conditions(own)
    assert ("namespace", BACKGROUND) in _conditions(background)
    assert ("subject_id", [SUBJECT]) in _conditions(background)


def test_background_is_never_searched_beyond_the_lessons_subject(searched, monkeypatch):
    """A lesson whose subject cannot be found must not open the background of
    every subject to it."""
    import agent.tools as tools
    from agent.tools import rag_search
    from shared.config import get_config

    monkeypatch.setattr(tools, "_subjects_of", lambda ids, config: [])
    rag_search("q", [SYLLABUS], lecture_ids=[MINISTRY_LESSON], config=get_config())
    assert not searched["filter"].should
    assert ("namespace", [SYLLABUS]) in _conditions(searched["filter"])


def test_an_external_book_is_never_named_to_a_student_as_a_source():
    """Students must not learn the external books exist; a citation carried the
    book's lesson title and pages into the sources under every answer."""
    from agent.models import RetrievedChunk
    from agent.retrieval import cite

    chunk = RetrievedChunk(
        chunk_id="c1", score=0.9, text="...", lecture_id="external-lesson",
        topic_ids=[], page_from=41, page_to=42, heading_path="الأضواء › الدرس 1",
        language="ar", subject_name="العلوم", lesson_title="الدرس 1: تركيب الذرة",
        lesson_order=0, background=True)
    citation = cite(chunk, 3)
    assert citation.subject_name == "العلوم"
    assert (citation.lecture_id, citation.lesson_title, citation.page_from,
            citation.page_to, citation.heading_path) == (None, None, None, None, None)


def test_an_external_book_lesson_is_filed_so_the_app_can_hide_it(fake_postgres):
    """Lesson rows of external books were filed as ministry_official, the same
    as the lessons the app is meant to show."""
    from ingestion.books import LessonCatalog, lesson_id_for
    from shared.config import get_config

    db = fake_postgres(fetchone=[(False,), (0,), (False,), (0,)])
    lessons = [{"lesson_id": lesson_id_for("book", 0), "ordinal": 0, "title": "L1",
                "unit_number": 1}]
    LessonCatalog(get_config()).register(subject_id=SUBJECT, book_title="Book",
                                         lessons=lessons, namespace=BACKGROUND)
    insert = next(args for sql, args in db.statements if "INSERT INTO content.lessons" in sql)
    assert insert[2] == "ai_generated"
    LessonCatalog(get_config()).register(subject_id=SUBJECT, book_title="Book",
                                         lessons=lessons, namespace=SYLLABUS)
    inserts = [args for sql, args in db.statements if "INSERT INTO content.lessons" in sql]
    assert inserts[-1][2] == "ministry_official"


def test_a_ministry_lessons_prerequisites_never_come_from_an_external_book(monkeypatch):
    """A prerequisite is shown to a student by name, so a ministry lesson must
    not be given one taken from a background book's topics."""
    import importlib

    dependencies = importlib.import_module("ingestion.stages.dependencies")

    asked = {}

    class _Store:
        def delete_lecture_dependencies(self, _lecture_id):
            pass

        def read_topic_catalog(self, subject_id, *, include_background=True):
            asked["include_background"] = include_background
            return []

        def write_dependencies(self, *_a):
            pass

    class _Deriver:
        is_stub = False

        def derive(self, *_a):
            return []

    class _Ctx:
        chunk_store = _Store()
        dependency_deriver = _Deriver()
        config = None

    monkeypatch.setattr(dependencies, "_candidate_vectors", lambda *a: {})
    state = {"lecture_id": "l", "subject_id": SUBJECT, "namespace": SYLLABUS,
             "topics": [{"label": "a", "topic_id": "1"}, {"label": "b", "topic_id": "2"}]}

    class _Config:
        def get(self, _key, default=None):
            return {"max_candidates": 0}

    _Ctx.config = _Config()
    dependencies.run(state, _Ctx())
    assert asked["include_background"] is False
    dependencies.run({**state, "namespace": BACKGROUND}, _Ctx())
    assert asked["include_background"] is True


def test_the_ingest_workflow_files_a_book_as_external_when_told():
    """The kind of book is chosen when it is started; nothing about the PDF
    tells a ministry book from an external one."""
    import subprocess

    out = subprocess.run(
        ["bash", "-c", "source scripts/ci_lesson_args.sh && echo \"${LESSON_ARGS[@]}\""],
        env={"BOOK_KIND": "external", "SUBJECT_ID": SUBJECT, "PATH": "/usr/bin:/bin"},
        capture_output=True, text=True, check=True)
    assert "--namespace external" in out.stdout
    bad = subprocess.run(
        ["bash", "-c", "source scripts/ci_lesson_args.sh"],
        env={"BOOK_KIND": "guide", "PATH": "/usr/bin:/bin"}, capture_output=True, text=True)
    assert bad.returncode != 0


def test_the_topics_collection_is_indexed_on_namespace():
    """Topic lookups leave background topics out with a namespace filter, and
    Qdrant refuses to filter on a field it has no index for: the production
    topics collection had none, and every lookup would have failed."""
    from ingestion.stores import QdrantVectorStore

    created = []

    class _Info:
        payload_schema = {"subject_id": {}, "lecture_id": {}}

    class _Client:
        def collection_exists(self, _name):
            return True

        def get_collection(self, _name):
            return _Info()

        def create_payload_index(self, collection, *, field_name, field_schema):
            created.append(field_name)

    store = QdrantVectorStore.__new__(QdrantVectorStore)
    store._client = _Client()
    store._topics_collection = "eduai_topics"
    store._distance = "cosine"
    store._hnsw = {}
    store.ensure_topics_collection(1024)
    assert created == ["namespace"]
