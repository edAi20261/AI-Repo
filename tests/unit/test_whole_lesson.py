"""A request about the selected lesson as a whole reads that lesson.

"Summarize this lesson", the portal's own quick prompt, was searched for as the
words "this lesson". Against a subtraction lesson they scored 0.472, the whole
corpus scored higher, and the turn was refused as outside the lesson the student
had open: "This lesson is about subtraction, and it doesn't contain a summary of
itself." An Arabic lesson passed only because its pages happened to mention the
word for lesson.
"""

from __future__ import annotations

import pytest

from agent.models import PlannedTask, RetrievedChunk, TurnUnderstanding

LESSON = "11111111-1111-1111-1111-111111111111"


def _chunk(n: int, score: float = 0.0) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=f"c{n}", score=score, text=f"passage {n}", topic_ids=[],
                          page_from=n, page_to=n, heading_path=None, language="en",
                          lecture_id=LESSON, order_index=n)


def _state(**over) -> dict:
    state = {"query": "this lesson", "message": "Summarize this lesson",
             "namespaces": ["syllabus"], "k": 8, "context_limit": 14, "min_score": 0.47,
             "study_scope": {"lecture_id": LESSON, "lesson_title": "Subtraction"}}
    state.update(over)
    return state


def _searches_find_only_a_weak_match(monkeypatch, sg):
    seen = {"searched": 0}

    def search(*a, **k):
        seen["searched"] += 1
        return [_chunk(1, score=0.472)]

    monkeypatch.setattr(sg.retrieval, "retrieve", search)
    monkeypatch.setattr(sg.retrieval, "answers_within_scope", lambda *a, **k: False)
    monkeypatch.setattr(sg.retrieval, "with_neighbours", lambda found, **k: found)
    return seen


def test_a_summary_of_the_selected_lesson_reads_the_lesson_instead_of_searching(monkeypatch):
    from agent.graph.subgraphs import explain_summarize_sg as sg

    seen = _searches_find_only_a_weak_match(monkeypatch, sg)
    asked = {}

    def lesson(lecture_id, namespaces, *, limit):
        asked.update(lecture_id=lecture_id, limit=limit)
        return [_chunk(n) for n in range(3)]

    monkeypatch.setattr(sg.retrieval, "whole_lesson", lesson)
    chunks = sg._retrieve(_state(whole_lesson=True), config=None)["chunks"]

    assert [c.chunk_id for c in chunks] == ["c0", "c1", "c2"]
    assert asked == {"lecture_id": LESSON, "limit": 14} and seen["searched"] == 0


def test_without_a_selected_lesson_the_flag_changes_nothing(monkeypatch):
    from agent.graph.subgraphs import explain_summarize_sg as sg

    seen = _searches_find_only_a_weak_match(monkeypatch, sg)
    monkeypatch.setattr(sg.retrieval, "answers_within_scope", lambda *a, **k: True)
    sg._retrieve(_state(whole_lesson=True, study_scope=None), config=None)
    assert seen["searched"] == 1


def test_a_topic_inside_the_lesson_is_still_searched_for(monkeypatch):
    from agent.graph.subgraphs import explain_summarize_sg as sg

    seen = _searches_find_only_a_weak_match(monkeypatch, sg)
    sg._retrieve(_state(whole_lesson=False), config=None)
    assert seen["searched"] == 1


def _searches_find_nothing(monkeypatch, sg, *, within_scope=True):
    seen = {"read": []}
    monkeypatch.setattr(sg.retrieval, "retrieve",
                        lambda *a, **k: [] if within_scope else [_chunk(1, score=0.6)])
    monkeypatch.setattr(sg.retrieval, "answers_within_scope", lambda *a, **k: within_scope)
    monkeypatch.setattr(sg.retrieval, "with_neighbours", lambda found, **k: found)

    def lesson(lecture_id, namespaces, *, limit):
        seen["read"].append(lecture_id)
        return [_chunk(n) for n in range(2)]

    monkeypatch.setattr(sg.retrieval, "whole_lesson", lesson)
    return seen


def test_a_summary_the_search_finds_nothing_for_reads_the_selected_lesson(monkeypatch):
    """"Summarize it" in Subtraction scored under the floor against the lesson's
    only passage, and the summary was refused as outside the lesson the student
    had open, although the turn reading never marked it as about the whole lesson."""
    from agent.graph.subgraphs import explain_summarize_sg as sg

    seen = _searches_find_nothing(monkeypatch, sg)
    chunks = sg._retrieve(_state(whole_lesson=False, mode="summarize"), config=None)["chunks"]

    assert seen["read"] == [LESSON] and [c.chunk_id for c in chunks] == ["c0", "c1"]


def test_a_question_judged_outside_the_lesson_is_still_refused(monkeypatch):
    from agent.graph.subgraphs import explain_summarize_sg as sg

    seen = _searches_find_nothing(monkeypatch, sg, within_scope=False)
    chunks = sg._retrieve(_state(whole_lesson=False, mode="summarize"), config=None)["chunks"]

    assert chunks == [] and seen["read"] == []


def test_an_explanation_the_search_finds_nothing_for_is_not_given_the_whole_lesson(monkeypatch):
    from agent.graph.subgraphs import explain_summarize_sg as sg

    seen = _searches_find_nothing(monkeypatch, sg)
    chunks = sg._retrieve(_state(whole_lesson=False, mode="explain"), config=None)["chunks"]

    assert chunks == [] and seen["read"] == []


def test_the_turn_reading_is_told_which_lesson_is_selected():
    from agent.graph.nodes import _session_facts
    from agent.models import SubjectFacts
    from agent.scope import StudyScope
    from shared.language import LanguagePolicy

    facts = _session_facts(SubjectFacts(name="Mathematics", language="en"), None,
                           LanguagePolicy(supported=("ar", "en"), default="en"), None,
                           StudyScope(lecture_id=LESSON, lesson_title="Subtraction"))
    assert '"Subtraction"' in facts and "this lesson" in facts


def test_the_flag_travels_from_the_turn_reading_to_the_task():
    from agent.graph import nodes

    reading = TurnUnderstanding(
        student_words="Summarize this lesson", needs_syllabus=True,
        tasks=[PlannedTask(type="summarize", query="this lesson", whole_lesson=True)])
    assert nodes._planned(reading)[0]["whole_lesson"] is True


def _config():
    from agent.config import get_config

    return get_config()


def test_a_long_lesson_is_sampled_from_its_first_passage_to_its_last(fake_postgres):
    """The first fourteen passages of a forty-page lesson are its opening pages; a
    summary built from them never reaches the lesson's end."""
    from agent import tools

    rows = [(f"c{n}", f"passage {n}", None, n, n, LESSON, "en", [], n) for n in range(30)]
    db = fake_postgres(fetchall=[rows])
    chunks = tools.get_lesson_chunks(LESSON, ["syllabus"], limit=6, ready_only=False,
                                     config=_config())
    orders = [c.order_index for c in chunks]
    assert orders[0] == 0 and orders[-1] == 29 and orders == sorted(orders) and len(orders) == 6
    assert "v.namespace = ANY" in db.sql[0]


def test_a_lessons_topics_include_those_its_text_carries(fake_postgres):
    """A topic reused from an earlier lesson is owned by that lesson; reading a
    lesson's topics from the owner column alone left the second lesson with none,
    and "quiz me on this lesson" there had nothing to ask about."""
    from agent import tools

    db = fake_postgres(fetchall=[[("t-sub", "Subtraction", "sub1")]])
    topics = tools.get_lecture_topics([LESSON], config=_config())

    assert [t.topic_id for t in topics] == ["t-sub"]
    assert "unnest(c.topic_ids)" in db.sql[0] and "t.lecture_id = ANY(%s::uuid[])" in db.sql[0]


class _NoSubjects:
    def facts(self, subject_id):
        from agent.models import SubjectFacts
        return SubjectFacts(name="English", language="en")


def _read_with(monkeypatch, study_scope):
    from agent.graph import nodes

    captured = {}

    def fake_call(schema, messages, **kwargs):
        captured["validate"] = kwargs["validate"]
        captured["prompt"] = " ".join(messages[0].content.split())
        return TurnUnderstanding(needs_syllabus=False, student_words="hi")

    monkeypatch.setattr(nodes, "call_structured", fake_call)
    node = nodes.build_understand(_config(), object(), _NoSubjects())
    node({"pathway": "student", "message": _CHIP, "history": "", "conversation": {},
          "study_scope": study_scope})
    return captured


_SELECTED = {"lecture_id": LESSON, "lesson_title": "Subtraction"}
_CHIP = "Explain this topic"


def _pointing(**task) -> TurnUnderstanding:
    return TurnUnderstanding(needs_syllabus=True, student_words=_CHIP,
                             tasks=[PlannedTask(type="explain", **task)])


def test_a_request_that_only_points_at_the_selected_lesson_is_read_as_about_it(monkeypatch):
    """The lesson screen's starter chips say "Explain this topic". The turn reading
    was told only that "this lesson" meant the selected lesson, so it rewrote the
    query to "the topic", and the reply was "The topic you're asking about is not
    specified", recorded as a decline."""
    prompt = _read_with(monkeypatch, _SELECTED)["prompt"]

    assert "in any language" in prompt and "means that lesson" in prompt
    assert "names no subject matter" in prompt
    assert "names_subject_matter:" in prompt


def test_a_pointing_request_inside_a_selected_lesson_is_sent_back_until_it_reads_the_lesson(
        monkeypatch):
    """A query of "the topic" with a lesson selected searched for nothing and
    reached the answer with no subject to explain, so the tutor declined."""
    validate = _read_with(monkeypatch, _SELECTED)["validate"]

    with pytest.raises(ValueError, match="whole_lesson"):
        validate(_pointing(query="the topic", names_subject_matter=False))
    validate(_pointing(query="the topic", names_subject_matter=False, whole_lesson=True))


def test_a_whole_lesson_reading_is_not_held_to_the_lessons_exact_title(monkeypatch):
    """A whole-lesson task reads the lesson whatever its query says. Requiring the
    title verbatim would fail every retry for a model that translated it, and the
    turn would end with the tutor unavailable instead of an answer."""
    captured = _read_with(monkeypatch, _SELECTED)

    assert "write the lesson's title in the query" in captured["prompt"]
    captured["validate"](_pointing(query="الطرح", names_subject_matter=False,
                                   whole_lesson=True))


def test_without_a_selected_lesson_a_pointing_request_is_not_forced_onto_one(monkeypatch):
    validate = _read_with(monkeypatch, None)["validate"]

    validate(_pointing(query="the topic", names_subject_matter=False))
