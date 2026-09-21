"""grade_sg tests — pure helpers, the stub + LLM grade paths, mastery write, and dispatch
routing. No real LLM or DB: tools are monkeypatched."""

from __future__ import annotations

from typing import ClassVar

import pytest

from agent.config import get_config
from agent.graph import nodes as nmod
from agent.graph.subgraphs import grade_sg as gmod
from agent.graph.subgraphs.grade_sg import (
    _build_graded,
    _correct,
    _event_from,
    _merge_rubric,
    _rollup_mastery,
    _stub_draft,
    _summary,
    _to_items,
    build_grade_sg,
)
from agent.models import GradeDraft, SubmissionItem


def _sub(**over):
    base = {"question": "2+2?", "student_answer": "4", "correct_answer": "4"}
    base.update(over)
    return SubmissionItem(**base)


# --- pure helpers -----------------------------------------------------------

def test_to_items_coerces_and_drops_questionless():
    items = _to_items([
        {"question": "q1", "student_answer": "a"},
        {"student_answer": "no question"},          # dropped
        {"question": "q2"},                         # blank answer is fine
    ])
    assert [i.question for i in items] == ["q1", "q2"]
    assert items[1].student_answer == ""            # default, not None


def test_merge_rubric_fills_spine_without_overwriting():
    items = [_sub(correct_answer=None, topic_id=None, lecture_id=None),
             _sub(correct_answer="keep")]
    rubric = [
        {"correct_answer_text": "4", "lesson_id": "l1", "question_type": "mcq"},
        {"correct_answer_text": "override?", "lesson_id": "l2"},
    ]
    _merge_rubric(items, rubric)
    assert items[0].correct_answer == "4" and items[0].lecture_id == "l1"
    assert items[0].type == "mcq"
    assert items[1].correct_answer == "keep"        # already set -> not overwritten
    assert items[1].lecture_id == "l2"              # was empty -> filled


def test_correct_prefers_partial_score_then_flag():
    assert _correct(GradeDraft(partial_score=0.8), 0.5) is True
    assert _correct(GradeDraft(partial_score=0.3), 0.5) is False
    assert _correct(GradeDraft(correct=True), 0.5) is True   # no partial -> flag


def test_build_graded_carries_spine_and_clears_labels_when_correct():
    item = _sub(topic_id="t1", lecture_id="l1", question_id="q9", difficulty="hard")
    g = _build_graded(item, GradeDraft(partial_score=1.0, mistakes=["x"]), 0.5)
    assert g.correct and g.mistake_labels == []      # correct -> no mistake labels
    assert g.topic_id == "t1" and g.lecture_id == "l1" and g.question_id == "q9"
    wrong = _build_graded(item, GradeDraft(partial_score=0.0, mistakes=["sign error"]), 0.5)
    assert not wrong.correct and wrong.mistake_labels == ["sign error"]


def test_stub_draft_exact_match():
    ok = _stub_draft(_sub(student_answer="4", correct_answer="4"))
    assert ok.correct and ok.partial_score == 1.0 and ok.mistakes == []
    bad = _stub_draft(_sub(student_answer="5", correct_answer="4"))
    assert not bad.correct and bad.partial_score == 0.0 and bad.mistakes


def test_event_from_requires_invariant_12_spine():
    g = _build_graded(_sub(topic_id="t1", lecture_id=None), GradeDraft(partial_score=1.0), 0.5)
    assert _event_from(g, "u1", "practice") is None            # no lecture_id -> skipped
    g2 = _build_graded(
        _sub(topic_id="t1", lecture_id="l1", difficulty="easy"),
        GradeDraft(partial_score=0.5), 0.5)
    e = _event_from(g2, "u1", "quiz")
    assert e["topic_id"] == "t1" and e["lecture_id"] == "l1"
    assert e["partial_score"] == 0.5 and e["original_score"] == 0.5  # AI score preserved
    assert e["source"] == "quiz" and e["user_id"] == "u1"


def _one_chunk(text="photosynthesis needs light"):
    from agent.models import RetrievedChunk
    return [RetrievedChunk(chunk_id="c1", score=0.9, text=text, lecture_id="lec1",
                           topic_ids=["t1"], page_from=1, page_to=1, heading_path="H",
                           language="en")]


def test_ground_answers_grounds_only_keyless(monkeypatch):
    # Rubric/key FIRST: a keyed answer (any type) is graded against its key — RAG grounds ONLY
    # a keyless answer (the fallback).
    monkeypatch.setattr(gmod, "rag_search", lambda q, ns, **k: _one_chunk())
    keyless = SubmissionItem(question="what does a plant need?", student_answer="water")
    keyed_essay = SubmissionItem(question="explain photosynthesis", student_answer="...",
                                 correct_answer="model answer", type="essay")
    keyed_mcq = SubmissionItem(question="q3", student_answer="a", correct_answer="b", type="mcq")
    gmod._ground_answers([keyless, keyed_essay, keyed_mcq],
                         {"namespaces": ["syllabus"], "min_score": 0.0}, get_config())
    assert "photosynthesis" in keyless.context        # keyless -> grounded (fallback)
    assert keyed_essay.context is None                 # keyed -> use the key/rubric, no RAG
    assert keyed_mcq.context is None                   # keyed exact type -> strict, no context


def test_ground_answers_noop_without_namespaces(monkeypatch):
    called = {"n": 0}
    monkeypatch.setattr(gmod, "rag_search", lambda *a, **k: called.__setitem__("n", called["n"] + 1))
    item = SubmissionItem(question="q", student_answer="a")
    gmod._ground_answers([item], {}, get_config())     # no namespaces (e.g. endpoint path)
    assert item.context is None and called["n"] == 0


def test_grade_block_shows_context_when_present():
    withctx = SubmissionItem(question="q", student_answer="a", context="the syllabus says X")
    noctx = SubmissionItem(question="q2", student_answer="a", correct_answer="b")
    assert "Syllabus context: the syllabus says X" in gmod._grade_blocks([withctx])
    assert "Syllabus context" not in gmod._grade_blocks([noctx])   # no context -> no line


def test_summary_totals():
    graded = [
        _build_graded(_sub(), GradeDraft(partial_score=1.0), 0.5),
        _build_graded(_sub(), GradeDraft(partial_score=0.0), 0.5),
    ]
    s = _summary(graded)
    assert s == {"count": 2, "score": 1.0, "max_score": 2.0, "correct": 1, "declined": 0}


# --- canonicalize (mistake tags) -------------------------------------------

def test_canonicalize_fills_tag_ids_and_caches(monkeypatch):
    calls = []
    monkeypatch.setattr(gmod, "create_mistake_tag",
                        lambda label, subject_id=None, config=None:
                        calls.append(label) or f"tag:{label.lower()}")
    graded = [
        _build_graded(_sub(), GradeDraft(partial_score=0.0, mistakes=["Sign error"]), 0.5),
        _build_graded(_sub(), GradeDraft(partial_score=0.0, mistakes=["sign error"]), 0.5),
    ]
    gmod._canonicalize(graded, "sub1", get_config())
    assert graded[0].mistake_tag_ids == ["tag:sign error"]
    assert graded[1].mistake_tag_ids == ["tag:sign error"]
    assert calls == ["Sign error"]                 # cached: created once for the same label


# --- mastery rollup ---------------------------------------------------------

def test_rollup_mastery_averages_within_assessment(monkeypatch):
    monkeypatch.setattr(gmod, "get_mastery", lambda uid, ids, config=None: [])
    updates = {}
    monkeypatch.setattr(gmod, "update_mastery",
                        lambda uid, tid, m, config=None: updates.__setitem__(tid, m))
    events = [
        {"topic_id": "t1", "correct": True, "partial_score": 1.0, "difficulty": "easy"},
        {"topic_id": "t1", "correct": False, "partial_score": 0.0, "difficulty": "easy"},
        {"topic_id": "t2", "correct": True, "partial_score": 1.0, "difficulty": "hard"},
    ]
    n = _rollup_mastery(events, "u1", get_config())
    assert n == 2 and set(updates) == {"t1", "t2"}
    # ONE EMA step per topic per assessment: t1's two answers average to 0.5 first.
    assert updates["t1"].attempts == 1             # one assessment, not two questions
    assert updates["t2"].attempts == 1
    # t1: prior 0, avg obs 0.5. Asymmetric weights per event: easy-correct 0.7, easy-wrong
    # 1.3 -> avg weight 1.0 -> alpha_eff 0.4*1.0=0.4 -> 0.4*0.5=0.2.
    assert round(updates["t1"].score, 4) == 0.2
    # t2: hard-correct weight 1.3 -> alpha_eff min(1, 0.4*1.3=0.52) -> 0.52*1.0=0.52.
    assert round(updates["t2"].score, 4) == 0.52


# --- LLM batch grade --------------------------------------------------------

class _BatchLLM:
    def __init__(self, content):
        self._content = content

    def invoke(self, _msgs):
        class _R:
            pass
        r = _R()
        r.content = self._content
        return r


def test_batch_grade_matches_answers_by_block_id():
    """Grades carry the id of the block they grade; order is not trusted."""
    from agent.graph.subgraphs.grade_sg import _batch_grade
    llm = _BatchLLM(
        '{"items": ['
        '{"id": "b2", "correct": false, "partial_score": 0, "feedback": "no",'
        ' "mistakes": ["sign error"]},'
        '{"id": "b1", "correct": true, "partial_score": 1, "feedback": "good",'
        ' "mistakes": []}]}')
    items = [_sub(), _sub(student_answer="5")]
    drafts = _batch_grade(llm, items, {"language": "en"})
    assert len(drafts) == 2
    assert drafts[0].correct and drafts[0].feedback == "good"
    assert not drafts[1].correct and drafts[1].mistakes == ["sign error"]


def test_batch_grade_refuses_to_guess_when_an_answer_is_missing():
    """A short reply used to be padded by position, grading the wrong question."""
    from agent.graph.subgraphs.grade_sg import _batch_grade
    from shared.structured import StructuredOutputError

    llm = _BatchLLM('{"items": [{"id": "b1", "correct": true, "partial_score": 1}]}')
    with pytest.raises(StructuredOutputError):
        _batch_grade(llm, [_sub(), _sub(student_answer="5")], {"language": "en"})


def test_batch_grade_rejects_an_out_of_range_score():
    """The prompt promised [0,1]; only mastery clamped, so the student saw 5/1."""
    from agent.graph.subgraphs.grade_sg import _batch_grade
    from shared.structured import StructuredOutputError

    llm = _BatchLLM('{"items": [{"id": "b1", "correct": true, "partial_score": 5}]}')
    with pytest.raises(StructuredOutputError):
        _batch_grade(llm, [_sub()], {"language": "en"})


def test_batch_grade_structured_output():
    from agent.graph.subgraphs.grade_sg import _batch_grade
    from agent.models import GradeBatch

    class _Structured:
        def invoke(self, _msgs):
            return GradeBatch(items=[GradeDraft(id="b1", correct=True, partial_score=1.0)])

    class _StructLLM:
        def with_structured_output(self, model_cls, **kwargs):
            assert model_cls is GradeBatch
            return _Structured()

        def invoke(self, _msgs):
            raise AssertionError("plain invoke should not run on the structured path")

    drafts = _batch_grade(_StructLLM(), [_sub()], {"language": "en"})
    assert len(drafts) == 1 and drafts[0].correct


def test_batch_grade_raises_on_error():
    # A runtime LLM failure propagates (no stub grade); the grade node degrades.
    from agent.graph.subgraphs.grade_sg import _batch_grade

    class _BadLLM:
        def invoke(self, _msgs):
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        _batch_grade(_BadLLM(), [_sub(student_answer="4", correct_answer="4")],
                     {"language": "en"})


def test_grade_degrades_on_llm_failure(monkeypatch):
    # LLM failure while a key exists -> the answer is NOT graded and NO mastery moves.
    called = {"n": 0}
    monkeypatch.setattr(gmod, "write_mastery_events",
                        lambda events, config=None: called.__setitem__("n", called["n"] + 1))

    class _BadLLM:
        def invoke(self, _msgs):
            raise RuntimeError("boom")

    graph = build_grade_sg(config=get_config(), llm=_BadLLM())
    out = graph.invoke({
        "submissions": [{"question": "2+2?", "student_answer": "4", "correct_answer": "4",
                         "topic_id": "t1", "lecture_id": "l1"}],
        "user_id": "u1", "source": "quiz",
    })
    res = out["result"]
    assert res["degraded"] is True and res["graded"][0]["graded"] is False
    assert res["mastery_events"] == 0 and called["n"] == 0   # no mastery on a failed grade
    assert "couldn't complete" in res["graded"][0]["feedback"].lower()


# --- end-to-end stub --------------------------------------------------------

def test_grade_sg_stub_grades_and_writes_mastery(monkeypatch):
    written = {}
    monkeypatch.setattr(gmod, "write_mastery_events",
                        lambda events, config=None: written.update(events=events) or ["e1"])
    monkeypatch.setattr(gmod, "get_mastery", lambda uid, ids, config=None: [])
    monkeypatch.setattr(gmod, "update_mastery", lambda *a, **k: None)
    graph = build_grade_sg(config=get_config(), llm=None)   # no LLM -> stub grader
    out = graph.invoke({
        "submissions": [
            {"question": "2+2?", "student_answer": "4", "correct_answer": "4",
             "topic_id": "t1", "lecture_id": "l1", "difficulty": "easy"},
            {"question": "Cap of France?", "student_answer": "Berlin",
             "correct_answer": "Paris", "topic_id": "t1", "lecture_id": "l1"},
        ],
        "user_id": "u1", "source": "quiz", "pass_threshold": 0.5,
    })
    res = out["result"]
    assert res["summary"] == {"count": 2, "score": 1.0, "max_score": 2.0, "correct": 1,
                              "declined": 0}
    assert res["mastery_events"] == 2 and res["source"] == "quiz"
    assert len(written["events"]) == 2               # both grounded -> both written


def test_grade_sg_declines_without_key_or_context(monkeypatch):
    # No reference answer and no syllabus context (no namespaces) -> declined, never graded on
    # the model's own knowledge, and no mastery moves.
    monkeypatch.setattr(gmod, "write_mastery_events",
                        lambda events, config=None: ["e1"])
    graph = build_grade_sg(config=get_config(), llm=None)
    out = graph.invoke({
        "submissions": [{"question": "explain X", "student_answer": "my essay",
                         "topic_id": "t1", "lecture_id": "l1", "type": "essay"}],
        "user_id": "u1", "source": "practice",
    })
    res = out["result"]
    assert res["summary"]["declined"] == 1 and res["summary"]["count"] == 0
    assert res["mastery_events"] == 0                 # declined -> no mastery
    assert res["graded"][0]["graded"] is False and "syllabus" in res["graded"][0]["feedback"].lower()


def test_grade_sg_no_user_id_feedback_only(monkeypatch):
    monkeypatch.setattr(gmod, "write_mastery_events",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no write")))
    graph = build_grade_sg(config=get_config(), llm=None)
    out = graph.invoke({
        "submissions": [{"question": "q", "student_answer": "a", "correct_answer": "a",
                         "topic_id": "t1", "lecture_id": "l1"}],
        "user_id": None,
    })
    assert out["result"]["mastery_events"] == 0      # no user -> graded, no mastery


def test_the_grade_endpoint_grounds_a_keyless_answer_in_the_syllabus(monkeypatch):
    """Every answer sent to POST /ai/grade with no reference key was declined.

    grade_sg grounds a keyless answer by retrieving the syllabus, and returns
    early when no namespaces were given. The chat path passes them; run_grade,
    the endpoint's entrypoint, never did — so a student pasting their own
    homework question got "there is no model answer or enough syllabus content"
    even where the syllabus answers it plainly.
    """
    from agent.graph import grade as grade_entrypoint

    seen: dict = {}

    class _Sg:
        def invoke(self, state):
            seen.update(state)
            return {"result": {}}

    monkeypatch.setattr(grade_entrypoint, "build_grade_sg", lambda **k: _Sg())
    monkeypatch.setattr(grade_entrypoint, "get_backend_client", lambda config: None)
    grade_entrypoint.run_grade(
        {"submissions": [{"question": "ما معنى ريادة الأعمال؟", "student_answer": "..."}],
         "user_id": "u1", "subject_id": "s1"},
        config=get_config())

    assert seen["namespaces"] == ["syllabus"]
    assert seen["min_score"] == get_config().get("agent.min_score")


# --- source: student practice vs teacher assigned ---------------------------

def test_grade_adapter_source_by_pathway():
    from agent.graph.graph import _build_grade_adapter
    from agent.graph.subgraphs import build_grade_sg
    cfg = get_config()
    adapter = _build_grade_adapter(build_grade_sg(config=cfg, llm=None), cfg, client=None)
    base = {"subject_id": None, "user_id": None, "language": "en"}
    # A student chat grade is self-practice even when a quiz is referenced.
    student = adapter({**base, "pathway": "student",
                       "task": {"params": {"query": "a", "grade": {"kind": "quiz"}}}})
    assert student["grade_result"]["source"] == "practice"
    # A teacher grading in chat is assigned work -> source by kind.
    teacher = adapter({**base, "pathway": "teacher",
                       "task": {"params": {"query": "a", "grade": {"kind": "homework"}}}})
    assert teacher["grade_result"]["source"] == "homework"


# --- dispatch wiring --------------------------------------------------------

def test_dispatch_routes_grade_and_forwards_user_id():
    sends = nmod.dispatch({
        "tasks": [{"type": "grade", "params": {"query": "my answer"}, "task_id": "x"}],
        "namespaces": ["syllabus"], "language": "en", "subject_id": "sub1", "user_id": "u1",
    })
    assert len(sends) == 1
    assert sends[0].node == "grade_adapter"
    assert sends[0].arg["user_id"] == "u1"


def test_unparseable_grading_moves_no_mastery():
    """The regression this contract exists for.

    A grading reply that does not satisfy the schema used to fall through to
    case-insensitive string matching, mark every essay wrong, report success, and
    write the result to mastery. The degrade path already existed — the silent
    empty dict is what stopped it firing.
    """
    from agent.graph.subgraphs.grade_sg import build_grade_sg
    from shared.config import Config

    class _Prose:
        """The real shape of the failure: schema decoding errors, and the model's
        actual reply is prose that no schema can be read out of."""

        class _Structured:
            def invoke(self, _msgs):
                raise ValueError("Failed to parse GradeBatch from completion")

        def with_structured_output(self, schema, **kwargs):
            return self._Structured()

        def invoke(self, _msgs):
            class _Reply:
                content = "I'm afraid I can't mark these for you."
                additional_kwargs: ClassVar[dict] = {}
                response_metadata: ClassVar[dict] = {}
            return _Reply()

    config = Config.load()
    sg = build_grade_sg(config=config, llm=_Prose())
    state = sg.invoke({
        "submissions": [
            {"question": "Why does ice float?", "student_answer": "Because it is less dense.",
             "correct_answer": "Ice is less dense than water."},
        ],
        "user_id": "u-1",
        "language": "en",
    })

    result = state.get("result", {})
    assert result.get("degraded") is True, "a failed grading call must be reported, not hidden"
    assert result.get("mastery_events", 0) == 0, "no mastery may move on a grade we did not compute"
    assert not any(g.get("correct") for g in result.get("graded", [])), \
        "a correct answer must never be marked wrong by a fallback the caller cannot see"
