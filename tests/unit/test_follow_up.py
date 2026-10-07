"""A follow-up is taught differently, and a question about the subject is taught
even when the lessons leave it out.

Students asked "I didn't get that", "make it simpler", "start from the very
beginning, step by step" and were told their lessons had nothing on it, or were
given the same explanation again. Two things were missing. The turn reading
recorded what to look up but not how the student wanted it taught, so the answer
prompt had one generic line for every kind of follow-up. And grounding was all
or nothing: whatever the book did not state was refused, so the tutor of a maths
subject would not teach the basics of subtraction.
"""

from __future__ import annotations

import pytest

from agent import retrieval as rmod
from agent.config import get_config
from agent.graph import nodes as nmod
from agent.graph.subgraphs.explain_summarize_sg import build_explain_sg, build_summarize_sg
from agent.models import RetrievedChunk, TurnUnderstanding

LESSON = "11111111-1111-1111-1111-111111111111"


class _Writer:
    """A chat model that records the prompt it was given."""

    def __init__(self, reply="An answer."):
        self.reply, self.prompts = reply, []

    def invoke(self, messages):
        self.prompts.append(" ".join(messages[0].content.split()))

        class _Msg:
            content = self.reply
        return _Msg()


def _chunk(cid="c1", score=0.8):
    return RetrievedChunk(chunk_id=cid, score=score, text="Borrow from the tens column.",
                          lecture_id="lec1", topic_ids=["t1"], page_from=1, page_to=1,
                          heading_path="Ch2 > Column subtraction", language="en")


def _turn(**fields):
    turn = {"query": "column subtraction from the beginning",
            "message": "I didn't get it, start from the very beginning",
            "namespaces": ["syllabus"], "language": "en", "subject_id": "math",
            "subject_name": "Mathematics", "beyond_lessons": True,
            "capabilities": '- "visual": a diagram', "approach": "direct",
            "scope_fit": "topic"}
    turn.update(fields)
    return turn


def _lessons(monkeypatch, chunks):
    monkeypatch.setattr(rmod, "rag_search", lambda *a, **k: list(chunks))
    monkeypatch.setattr(rmod, "get_topic_deps", lambda ids, **k: [])


# --- the lessons hold nothing on it ------------------------------------------

def test_a_question_inside_the_subject_is_taught_when_the_lessons_leave_it_out(monkeypatch):
    """The reported failure: a maths student asking for subtraction from the very
    beginning was told the lessons had nothing on it."""
    _lessons(monkeypatch, [])
    writer = _Writer("Subtraction means taking away. This goes beyond your lesson.")
    out = build_explain_sg(config=get_config(), llm=writer).invoke(
        _turn(scope_fit="foundation"))
    assert out["basis"] == "beyond"
    assert out["explanation"].startswith("Subtraction means taking away")
    assert out["grounded"] is False and out["citations"] == []
    assert "Mathematics" in writer.prompts[0]
    assert "Do not answer the question from your own knowledge" not in writer.prompts[0]


def test_a_question_outside_the_subject_is_still_declined(monkeypatch):
    _lessons(monkeypatch, [])
    writer = _Writer("Your lessons don't cover that.")
    out = build_explain_sg(config=get_config(), llm=writer).invoke(
        _turn(scope_fit="outside", message="who won the football last night?"))
    assert out["basis"] == "declined"
    assert "Do not answer the question from your own knowledge" in writer.prompts[0]


def test_a_deployment_can_keep_answers_to_the_lessons(monkeypatch):
    _lessons(monkeypatch, [])
    writer = _Writer("Your lessons don't cover that.")
    out = build_explain_sg(config=get_config(), llm=writer).invoke(
        _turn(beyond_lessons=False))
    assert out["basis"] == "declined"


def test_a_summary_never_leaves_the_material(monkeypatch):
    """A summary is of the lessons; one of something they do not hold is invented."""
    _lessons(monkeypatch, [])
    out = build_summarize_sg(config=get_config(), llm=_Writer("Nothing on that.")).invoke(
        _turn())
    assert out["basis"] == "declined"


def test_without_a_subject_there_is_nothing_to_stay_inside(monkeypatch):
    _lessons(monkeypatch, [])
    out = build_explain_sg(config=get_config(), llm=_Writer("Nothing on that.")).invoke(
        _turn(subject_id=None))
    assert out["basis"] == "declined"


def test_inside_a_selected_lesson_another_topic_is_still_refused_by_name(monkeypatch):
    """Teaching beyond the lessons must not undo the lesson boundary: a student in
    "Column subtraction" asking about fractions is told it is outside that lesson."""
    _lessons(monkeypatch, [])
    writer = _Writer("That's outside Column subtraction.")
    scope = {"lecture_id": LESSON, "lesson_title": "Column subtraction"}
    out = build_explain_sg(config=get_config(), llm=writer).invoke(
        _turn(study_scope=scope, query="fractions", message="what is a fraction?"))
    assert out["basis"] == "declined"
    assert "Column subtraction" in writer.prompts[0]


def test_inside_a_selected_lesson_its_basics_are_still_taught(monkeypatch):
    _lessons(monkeypatch, [])
    scope = {"lecture_id": LESSON, "lesson_title": "Column subtraction"}
    out = build_explain_sg(config=get_config(), llm=_Writer("Taking away first.")).invoke(
        _turn(study_scope=scope, scope_fit="foundation"))
    assert out["basis"] == "beyond"


# --- the lessons answer part of it -------------------------------------------

def test_inside_the_subject_an_explanation_may_fill_the_step_the_book_assumes(monkeypatch):
    _lessons(monkeypatch, [_chunk()])
    writer = _Writer("Borrow from the tens [1].")
    out = build_explain_sg(config=get_config(), llm=writer).invoke(_turn())
    assert "goes beyond their lesson" in writer.prompts[0]
    assert "using ONLY the context" not in writer.prompts[0]
    assert out["basis"] == "lesson"


def test_outside_the_subject_an_explanation_keeps_to_the_context(monkeypatch):
    _lessons(monkeypatch, [_chunk()])
    writer = _Writer("The context does not cover that.")
    out = build_explain_sg(config=get_config(), llm=writer).invoke(_turn(scope_fit="outside"))
    assert "using ONLY the context" in writer.prompts[0]
    assert out["basis"] == "declined"


def test_an_uncited_answer_inside_the_subject_is_teaching_not_a_refusal(monkeypatch):
    _lessons(monkeypatch, [_chunk()])
    out = build_explain_sg(config=get_config(), llm=_Writer("Think of ten sweets.")).invoke(
        _turn())
    assert out["basis"] == "beyond" and out["citations"] == []


def test_the_basics_of_a_selected_lesson_are_looked_for_across_the_subject(monkeypatch):
    """The basics a lesson assumes are taught in an earlier lesson. Searching only
    the selected one, then comparing it with the rest of the subject, declined them."""
    seen = []

    def spy(query, namespaces, **kwargs):
        seen.append(kwargs.get("lecture_ids"))
        return [_chunk()]

    monkeypatch.setattr(rmod, "rag_search", spy)
    monkeypatch.setattr(rmod, "get_topic_deps", lambda ids, **k: [])
    monkeypatch.setattr(rmod, "answers_within_scope",
                        lambda *a, **k: pytest.fail("a foundation is not a scope check"))
    out = build_explain_sg(config=get_config(), llm=_Writer("Borrow [1].")).invoke(
        _turn(scope_fit="foundation", study_scope={"lecture_id": LESSON}))
    assert seen == [None]
    assert out["basis"] == "lesson"


# --- how it is taught ----------------------------------------------------------

def test_every_teaching_approach_has_a_strategy():
    from agent.models import TEACHING_APPROACHES as VALUES
    from agent.prompts import TEACHING_APPROACHES
    assert set(TEACHING_APPROACHES) == set(VALUES)


@pytest.mark.parametrize("chunks", [[], [_chunk()]])
def test_an_explanation_that_did_not_land_is_taught_another_way(monkeypatch, chunks):
    from agent.prompts import TEACHING_APPROACHES
    _lessons(monkeypatch, chunks)
    writer = _Writer("Think of ten sweets [1].")
    build_explain_sg(config=get_config(), llm=writer).invoke(_turn(approach="another_way"))
    assert TEACHING_APPROACHES["another_way"] in writer.prompts[0]


def test_the_tutor_offers_only_a_next_step_it_can_take(monkeypatch):
    _lessons(monkeypatch, [_chunk()])
    writer = _Writer("Borrow [1].")
    build_explain_sg(config=get_config(), llm=writer).invoke(_turn())
    assert '- "visual": a diagram' in writer.prompts[0]


def test_capabilities_are_the_tasks_this_user_can_run():
    text = nmod.capabilities(get_config(), "student")
    assert '"explain"' in text and '"visual"' in text


# --- reading the turn ----------------------------------------------------------

def test_the_turn_reading_carries_how_to_teach_it_into_the_task(monkeypatch):
    from agent.models import PlannedTask
    monkeypatch.setattr(nmod, "resolve_topics", lambda *a, **k: [])
    reading = TurnUnderstanding(
        needs_syllabus=True, continues_previous=True, approach="simpler",
        scope_fit="foundation",
        tasks=[PlannedTask(type="explain", query="column subtraction")])
    plan = nmod.build_plan(get_config(), _NoSubjects())
    out = plan({"pathway": "student", "message": "simpler please",
                "understanding": reading.model_dump(),
                "planned_tasks": [{"type": "explain", "query": "column subtraction",
                                   "level": "basic", "length": "medium"}],
                "subject_id": "math"})
    params = out["tasks"][0]["params"]
    assert params["approach"] == "simpler" and params["scope_fit"] == "foundation"


def test_a_follow_up_cannot_be_outside_the_subject(monkeypatch):
    from agent.models import PlannedTask

    captured = {}

    def fake_call(schema, messages, **kwargs):
        captured["validate"] = kwargs["validate"]
        captured["prompt"] = messages[0].content
        return TurnUnderstanding(needs_syllabus=False, student_words="hi")

    monkeypatch.setattr(nmod, "call_structured", fake_call)
    node = nmod.build_understand(get_config(), object(), _NoSubjects())
    node({"pathway": "student", "message": "hi", "history": "", "conversation": {}})
    assert "approach:" in captured["prompt"] and "scope_fit:" in captured["prompt"]
    with pytest.raises(ValueError):
        captured["validate"](TurnUnderstanding(
            needs_syllabus=True, continues_previous=True, scope_fit="outside",
            student_words="hi", tasks=[PlannedTask(type="explain", query="x")]))


class _NoSubjects:
    def facts(self, subject_id):
        from agent.models import SubjectFacts
        return SubjectFacts()


def test_the_api_says_where_an_answer_came_from():
    """An uncited reply was either a refusal or teaching beyond the lesson, and
    nothing a client or an eval could read told the two apart."""
    from shared.metering import Meter
    from system.schemas.chat import ChatResponse

    payload = ChatResponse.from_state(
        {"response": "Subtraction means taking away.", "language": "en",
         "answers": [{"type": "explain", "text": "x", "basis": "beyond"}]},
        "req-1", Meter())
    assert payload.answer_basis == ["beyond"]
