"""A grade the model is unsure of waits for a teacher before it counts.

Grading was AI-final: every grade the model returned moved mastery, however
little it believed it. A wrong mark the model itself doubted became a weak topic
on the student's homepage, steered their next quiz and went into the parent's
report, and nobody had looked at it. The confidence it reported was stored and
read by nothing.
"""

from __future__ import annotations

import json

import pytest

from agent.config import get_config
from agent.graph.subgraphs import grade_sg as gmod
from agent.graph.subgraphs.grade_sg import build_grade_sg, record_reviewed
from agent.models import ReviewedGrade


class _Grades:
    """A model that grades every block it is shown with the given confidence."""

    def __init__(self, *confidences, partial_score=0.0):
        self._confidences = confidences
        self._score = partial_score

    def invoke(self, _msgs):
        items = []
        for i, confidence in enumerate(self._confidences, 1):
            item = {"id": f"b{i}", "correct": self._score >= 0.5, "partial_score": self._score,
                    "feedback": "f", "mistakes": [] if self._score >= 0.5 else ["sign error"]}
            if confidence is not None:
                item["confidence"] = confidence
            items.append(item)

        class _Reply:
            content = json.dumps({"items": items})

        return _Reply()


@pytest.fixture
def written(monkeypatch):
    seen: dict = {"events": []}
    monkeypatch.setattr(gmod, "get_mastery", lambda uid, ids, config=None: [])
    monkeypatch.setattr(gmod, "create_mistake_tag", lambda label, **k: f"tag:{label}")
    monkeypatch.setattr(gmod, "record_graded_evidence",
                        lambda uid, events, aggs, config=None:
                        seen.update(events=events) or [f"e{i}" for i in range(len(events))])
    return seen


def _grade(llm, *submissions):
    return build_grade_sg(config=get_config(), llm=llm).invoke({
        "submissions": [{"question": f"q{i}", "student_answer": "x", "correct_answer": "y",
                         "topic_id": "t1", "lecture_id": "l1", **extra}
                        for i, extra in enumerate(submissions)],
        "user_id": "u1", "source": "uploaded", "pass_threshold": 0.5,
    })["result"]


def _floor() -> float:
    return get_config().get("grade.review_below")


def test_a_grade_the_model_doubts_is_held_and_moves_no_mastery(written):
    result = _grade(_Grades(_floor() - 0.2), {})

    held = result["graded"][0]
    assert held["needs_review"] is True and held["review"] == ["low_confidence"]
    assert held["graded"] is True and held["partial_score"] == 0.0
    assert written["events"] == [] and result["mastery_events"] == 0
    assert result["summary"]["needs_review"] == 1


def test_a_grade_with_no_confidence_is_held_rather_than_trusted(written):
    """A model that left confidence out has told us nothing about its certainty."""
    result = _grade(_Grades(None), {})

    assert result["graded"][0]["review"] == ["low_confidence"]
    assert result["mastery_events"] == 0


def test_a_grade_the_model_is_sure_of_still_moves_mastery_at_once(written):
    result = _grade(_Grades(0.95), {})

    assert result["graded"][0]["needs_review"] is False
    assert result["mastery_events"] == 1 and len(written["events"]) == 1


def test_only_the_doubted_grade_is_held_out_of_a_submission(written):
    result = _grade(_Grades(0.95, _floor() - 0.1), {}, {})

    assert [g["needs_review"] for g in result["graded"]] == [False, True]
    assert result["mastery_events"] == 1
    assert result["summary"]["count"] == 2 and result["summary"]["needs_review"] == 1


def test_an_answer_read_unclearly_off_a_photo_is_held_however_sure_the_grade(written):
    """A confident mark on a half-read answer is confident about the wrong thing."""
    result = _grade(_Grades(0.99), {"unclear_reading": True})

    assert result["graded"][0]["review"] == ["unclear_reading"]
    assert result["mastery_events"] == 0


def test_a_held_wrong_answer_keeps_its_mistake_tags_for_the_teacher(written):
    """The teacher's decision is sent back with the tags, so a confirmed mistake is
    counted in the class's commonest mistakes like any other."""
    result = _grade(_Grades(_floor() - 0.3), {})

    assert result["graded"][0]["mistake_tag_ids"] == ["tag:sign error"]


def test_an_answer_the_platform_already_scored_is_never_held(written):
    result = _grade(None, {"known_score": 1.0})

    assert result["graded"][0]["needs_review"] is False
    assert result["mastery_events"] == 1


# --- the teacher's decision -------------------------------------------------

def _reviewed(**over) -> ReviewedGrade:
    return ReviewedGrade(**{"topic_id": "t1", "lecture_id": "l1", "question_id": "q1",
                            "ai_score": 0.0, "score": 1.0, **over})


def test_a_teacher_decision_moves_mastery_and_keeps_the_model_s_score(written):
    """Both scores are kept so how often teachers overturn held grades can be read
    back from the evidence, and the review threshold re-set from it."""
    result = record_reviewed("u1", [_reviewed(mistake_tag_ids=["tag:x"])],
                             source="uploaded", pass_threshold=0.5, config=get_config())

    event = written["events"][0]
    assert event["partial_score"] == 1.0 and event["original_score"] == 0.0
    assert event["correct"] is True and event["source"] == "uploaded"
    assert event["item_id"] == "q1" and event["mistake_tag_ids"] == ["tag:x"]
    assert result == {"mastery_events": 1, "topics_updated": 1, "mastery_recorded": True}


def test_a_teacher_decision_that_could_not_be_stored_is_not_reported_as_recorded(monkeypatch):
    monkeypatch.setattr(gmod, "get_mastery", lambda uid, ids, config=None: [])

    def unreachable(*a, **k):
        raise ConnectionError("db down")

    monkeypatch.setattr(gmod, "record_graded_evidence", unreachable)
    result = record_reviewed("u1", [_reviewed()], source="uploaded", pass_threshold=0.5,
                             config=get_config())

    assert result == {"mastery_events": 0, "topics_updated": 0, "mastery_recorded": False}


def test_a_teacher_score_outside_zero_to_one_is_refused():
    with pytest.raises(ValueError):
        _reviewed(score=4)
