"""Citation markers: what the model writes, and what reaches the student.

The named regression is the reason this module exists — the marker pattern
matched only ``[1]``, so an answer citing ``[1, 2]`` was judged ungrounded, its
citations were discarded, the sources block never rendered, and the student read
bracketed numbers that pointed at nothing.
"""

from __future__ import annotations

import pytest

from agent.citations import (
    cited_numbers,
    has_citation,
    keep_cited,
    renumber,
    strip_markers,
)
from agent.graph.nodes import _renumber_citations, _sources_block
from agent.models import Citation

# --- the reported bug -------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("Subtraction borrows from the next column [1].", [1]),
    ("Both sources agree [1, 2].", [1, 2]),
    ("No space either [1,2].", [1, 2]),
    ("Adjacent markers [1][3].", [1, 3]),
    ("Padded [ 4 ].", [4]),
    ("Repeated [1] and again [1].", [1]),
    ("A refusal cites nothing.", []),
])
def test_every_form_a_model_writes_is_read(text, expected):
    assert cited_numbers(text) == expected


def test_a_grouped_citation_counts_as_grounded():
    """This returned False, which discarded the citations for the whole answer."""
    assert has_citation("The lesson defines it this way [1, 2].") is True


def test_prose_without_markers_is_not_grounded():
    assert has_citation("The syllabus does not cover this topic.") is False


# --- publishing -------------------------------------------------------------

def _cite(n: int, lecture: str, task: str = "a") -> Citation:
    return Citation(n=n, task_id=task, lecture_id=lecture,
                    heading_path=f"H{n}", page_from=n)


def test_only_cited_sources_are_published():
    """Retrieval returns k chunks; the answer leans on one. The other four are
    lessons the student was never pointed at."""
    answers = [{"task_id": "a", "text": "Borrowing works like this [2]."}]
    citations = [_cite(n, f"lec{n}") for n in range(1, 6)]
    new_answers, ordered = _renumber_citations(answers, citations)
    assert [c.lecture_id for c in ordered] == ["lec2"]
    assert new_answers[0]["text"].endswith("[1].")


def test_grouped_markers_become_one_marker_per_source():
    answers = [{"task_id": "a", "text": "Both agree [1, 2]."}]
    citations = [_cite(1, "lecA"), _cite(2, "lecB")]
    new_answers, ordered = _renumber_citations(answers, citations)
    assert new_answers[0]["text"] == "Both agree [1][2]."
    assert len(ordered) == 2


def test_numbering_follows_the_answer_not_retrieval():
    answers = [{"task_id": "a", "text": "Second first [3], then [1]."}]
    citations = [_cite(1, "lecA"), _cite(3, "lecC")]
    new_answers, ordered = _renumber_citations(answers, citations)
    assert new_answers[0]["text"] == "Second first [1], then [2]."
    assert [c.lecture_id for c in ordered] == ["lecC", "lecA"]


def test_sources_beyond_the_cap_drop_their_markers_too():
    """A marker that outlives its source points at the wrong lesson."""
    answers = [{"task_id": "a", "text": " ".join(f"point [{n}]." for n in range(1, 9))}]
    citations = [_cite(n, f"lec{n}") for n in range(1, 9)]
    new_answers, ordered = _renumber_citations(answers, citations, limit=3)
    assert len(ordered) == 3
    assert cited_numbers(new_answers[0]["text"]) == [1, 2, 3]


def test_a_marker_with_no_retrieved_source_is_removed():
    """The model invents [7]; nothing in the sources block would explain it."""
    answers = [{"task_id": "a", "text": "Grounded [1] and invented [7]."}]
    new_answers, ordered = _renumber_citations(answers, [_cite(1, "lecA")])
    assert new_answers[0]["text"] == "Grounded [1] and invented."
    assert len(ordered) == 1


def test_published_sources_and_markers_always_agree():
    """The invariant behind all of the above: every marker the student sees has a
    line in the sources block, and every line is referenced."""
    answers = [{"task_id": "a", "text": "A [1, 4] and B [9]."}]
    citations = [_cite(n, f"lec{n}") for n in (1, 2, 4)]
    new_answers, ordered = _renumber_citations(answers, citations)
    block = _sources_block(ordered, "en")
    shown = set(cited_numbers(new_answers[0]["text"]))
    assert shown == {c.n for c in ordered}
    for n in shown:
        assert f"[{n}]" in block


# --- helpers ----------------------------------------------------------------

def test_keep_cited_filters_by_number():
    kept = keep_cited([_cite(1, "a"), _cite(2, "b"), _cite(3, "c")], [1, 3])
    assert [c.n for c in kept] == [1, 3]


@pytest.mark.parametrize("text,expected", [
    ("away [1][2][1].", "away [1][2]."),
    ("triple [1][1][1] here", "triple [1] here"),
    ("both agree [1, 2].", "both agree [1][2]."),
    ("separate [1]. and [2].", "separate [1]. and [2]."),
])
def test_a_source_is_not_cited_twice_in_one_breath(text, expected):
    """A model leaning on the same passages writes them after every sentence;
    split one per source that reads as "[1][2][1]"."""
    assert renumber(text, {1: 1, 2: 2}) == expected


def test_renumber_leaves_text_without_markers_alone():
    assert renumber("plain prose", {1: 2}) == "plain prose"


def test_strip_markers_leaves_readable_prose():
    assert strip_markers("Light is needed [1, 2]. So is water [3].") == (
        "Light is needed. So is water."
    )


# --- prerequisites ----------------------------------------------------------

def test_the_prerequisite_walk_keeps_the_nearest_topics(monkeypatch):
    """Two hops over a connected subject reached dozens of topics, each adding
    chunks fetched by id — no score to filter on — until prerequisite material
    outweighed the lesson the student asked about."""
    from agent import retrieval
    from agent.models import PrereqTopic

    found = [
        PrereqTopic(topic_id="far", label="far", strength="recommended", depth=2),
        PrereqTopic(topic_id="near", label="near", strength="required", depth=1),
        PrereqTopic(topic_id="near2", label="near2", strength="recommended", depth=1),
        PrereqTopic(topic_id="far2", label="far2", strength="required", depth=2),
    ]
    asked: list[list[str]] = []
    monkeypatch.setattr(retrieval, "get_topic_deps", lambda *a, **k: found)
    monkeypatch.setattr(retrieval, "get_topic_chunks",
                        lambda ids, ns, **k: asked.append(list(ids)) or [])

    prereqs, _ = retrieval.resolve_prereqs(["t1"], ["ns"], limit=2)
    assert [p.topic_id for p in prereqs] == ["near", "near2"]
    assert asked == [["near", "near2"]]


def test_an_unlimited_walk_is_still_possible(monkeypatch):
    from agent import retrieval
    from agent.models import PrereqTopic

    found = [PrereqTopic(topic_id=f"t{i}", label=f"t{i}", strength="required", depth=1)
             for i in range(9)]
    monkeypatch.setattr(retrieval, "get_topic_deps", lambda *a, **k: found)
    monkeypatch.setattr(retrieval, "get_topic_chunks", lambda ids, ns, **k: [])
    prereqs, _ = retrieval.resolve_prereqs(["t"], ["ns"], limit=0)
    assert len(prereqs) == 9


# --- through the real answer node -------------------------------------------

class _FakeLLM:
    """Answers with the grouped-citation form that made the whole system inert."""

    def __init__(self, reply: str = "Column subtraction lines the digits up [1, 2]."):
        self.reply = reply
        self.prompt = ""

    def invoke(self, messages):
        self.prompt = messages[0].content
        return type("R", (), {"content": self.reply})()


def _chunk(n: int):
    from agent.models import RetrievedChunk
    return RetrievedChunk(
        chunk_id=f"c{n}", text="Line the digits up.", lecture_id=f"lec{n}",
        heading_path=f"Unit 1 > Part {n}", page_from=n, page_to=n,
        topic_ids=[], language="en", score=0.9)


def _answer(mode: str, llm) -> dict:
    import agent.graph.subgraphs.explain_summarize_sg as sg
    from agent.config import Config
    node = sg._build_answer(mode, llm, Config.load())
    return node({"chunks": [_chunk(1), _chunk(2), _chunk(3)], "prereq_chunks": [],
                 "query": "explain column subtraction", "language": "en",
                 "namespaces": ["ns"], "topic_ids": []})


@pytest.mark.parametrize("mode", ["explain", "summarize"])
def test_a_grouped_citation_answer_is_grounded_and_publishes_only_its_sources(mode):
    out = _answer(mode, _FakeLLM())
    assert out["grounded"] is True
    assert [c.n for c in out["citations"]] == [1, 2]


@pytest.mark.parametrize("mode", ["explain", "summarize"])
def test_the_formatting_contract_reaches_the_model(mode):
    llm = _FakeLLM()
    _answer(mode, llm)
    assert "Write in Markdown" in llm.prompt
    assert "$x^2$" in llm.prompt


def test_explain_renders_the_length_it_is_given():
    """``length`` was passed to the explain prompt, which had no slot for it."""
    llm = _FakeLLM()
    _answer("explain", llm)
    assert "Length: medium" in llm.prompt


@pytest.mark.parametrize("mode", ["explain", "summarize"])
def test_a_refusal_publishes_nothing(mode):
    out = _answer(mode, _FakeLLM("The syllabus does not cover this topic."))
    assert out["grounded"] is False
    assert out["citations"] == []


# --- the API contract -------------------------------------------------------

def test_the_api_publishes_the_sources_the_reply_cites():
    """A client renders the [n] markers as badges, so the structured list must be
    the renumbered one compose produced — not the subgraphs' local numbering."""
    from system.schemas.chat import ChatResponse

    state = {
        "response": "Borrowing works like this [1][2].",
        "language": "en",
        "published_citations": [_cite(1, "lecA"), _cite(2, "lecB")],
        "citations": [_cite(1, "lecA"), _cite(1, "lecB")],  # pre-renumbering, collides
    }
    payload = ChatResponse.from_state(state, "req-1")
    assert [c.n for c in payload.citations] == [1, 2]
    assert [c.lecture_id for c in payload.citations] == ["lecA", "lecB"]
    assert set(cited_numbers(payload.response)) == {c.n for c in payload.citations}


def test_a_turn_with_no_sources_returns_an_empty_list_not_an_error():
    from system.schemas.chat import ChatResponse

    payload = ChatResponse.from_state({"response": "Hello!", "language": "en"}, "req-2")
    assert payload.citations == []


# --- citation labels --------------------------------------------------------

def test_topic_chunks_are_named_not_just_identified():
    """A prerequisite chunk is fetched by id straight from Postgres, where the
    lesson name does not live — so the citation rendered as a raw uuid."""
    from agent.tools import _apply_labels

    chunks = [_retrieved("c1"), _retrieved("c2")]
    named = _apply_labels(chunks, {
        "c1": {"subject_name": "Mathematics", "lesson_title": "Subtraction",
               "lesson_order": 4},
    })
    assert (named[0].subject_name, named[0].lesson_title) == ("Mathematics", "Subtraction")
    assert named[0].lesson_order == 4
    # A chunk the payload does not cover keeps its lecture id and cites by that.
    assert named[1].lesson_title is None and named[1].lecture_id == "lec-c2"


def _retrieved(chunk_id: str):
    from agent.models import RetrievedChunk
    return RetrievedChunk(
        chunk_id=chunk_id, score=0.0, text="t", heading_path=None, page_from=1,
        page_to=1, lecture_id=f"lec-{chunk_id}", language="en", topic_ids=[])


def test_missing_labels_degrade_to_the_lecture(monkeypatch):
    """Qdrant being unreachable must cost the label, not the answer."""
    from agent import tools

    class Boom:
        def __init__(self, *a, **k):
            raise RuntimeError("qdrant down")

    monkeypatch.setattr("qdrant_client.QdrantClient", Boom)

    class Cfg:
        def __init__(self):
            self.vector_store = {"collection": "x"}

        def secret(self, *a):
            return "http://localhost:6333"

    assert tools._citation_labels(["c1"], Cfg()) == {}
