"""The student analysis behind the homepage: where a student stands in every subject,
what they are strong and weak in, and what to practise next. No model calls."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from agent import analytics
from agent.config import get_config

NOW = datetime(2026, 9, 27, tzinfo=timezone.utc)

TOPICS = [
    {"topic_id": "fact", "label": "Factoring", "lecture_id": "l1", "subject_id": "math"},
    {"topic_id": "quad", "label": "Quadratics", "lecture_id": "l2", "subject_id": "math"},
    {"topic_id": "frac", "label": "Fractions", "lecture_id": "l1", "subject_id": "math"},
    {"topic_id": "ineq", "label": "Inequalities", "lecture_id": "l2", "subject_id": "math"},
    {"topic_id": "cell", "label": "Cells", "lecture_id": "b1", "subject_id": "bio"},
]


def _row(tid, score, attempts, status, days_ago=1):
    return {"topic_id": tid, "score": score, "attempts": attempts, "status": status,
            "last_seen": NOW - timedelta(days=days_ago), "trend": None, "confidence": 1.0}


MASTERY = [_row("fact", 0.35, 3, "weak"), _row("quad", 0.30, 4, "weak"),
           _row("frac", 0.85, 5, "strong"), _row("cell", 0.60, 2, "developing")]


@pytest.fixture
def reads(monkeypatch):
    calls = {}

    def _set(name, fn):
        monkeypatch.setattr(analytics, name, fn)

    _set("get_scope_topics", lambda uid, **k: TOPICS)
    _set("get_mastery", lambda uid, *a, **k: MASTERY)
    _set("get_recent_outcomes",
         lambda uid, ids, **k: {"quad": [0.0, 0.0, 1.0], "fact": [0.0]})
    _set("get_direct_prereqs",
         lambda ids, **k: {"quad": [("fact", "Factoring"), ("frac", "Fractions")]})
    _set("get_weekly_activity",
         lambda uid, **k: [{"week_start": "2026-09-21", "answers": 6, "correct": 3,
                            "mean_score": 0.5}])
    _set("get_mastery_trend", lambda uid, **k: {"direction": "declining", "count": 9,
                                                "events": []})
    _set("get_top_mistakes", lambda ids, **k: calls.setdefault("mistakes", (ids, k)) and [
        {"tag_id": "g1", "label": "Sign error", "occurrences": 4, "students": 1}])
    return calls


def _summary(**kw):
    return analytics.learner_summary("u1", now=NOW, config=get_config(), **kw)


def test_each_subject_counts_every_topic_including_the_untouched(reads):
    """An average over practised topics hides how much of a subject is untouched; the
    counts say "1 of 4 mastered", which an average of 0.47 never could."""
    out = _summary(subject_ids=["math", "bio"])

    subjects = {s["subject_id"]: s for s in out["subjects"]}
    assert subjects["math"] == {"subject_id": "math", "topics": 4, "strong": 1,
                                "developing": 0, "weak": 2, "unknown": 0,
                                "not_practised": 1}
    assert subjects["bio"]["developing"] == 1 and subjects["bio"]["topics"] == 1
    assert out["available"] is True and out["degraded"] is False


def test_strengths_and_weaknesses_are_ordered_and_carry_their_evidence(reads):
    out = _summary()
    assert [t["topic_id"] for t in out["strengths"]] == ["frac"]
    assert [t["topic_id"] for t in out["weaknesses"]] == ["quad", "fact"]
    assert out["weaknesses"][0]["recent"] == [0.0, 0.0, 1.0]


def test_only_a_prerequisite_that_is_itself_weak_is_named_as_the_cause(reads):
    """Fractions is a prerequisite of Quadratics but is mastered; sending the student
    back to it would waste their time."""
    out = _summary()
    first = out["recommendations"][0]
    assert (first["action"], first["topic_id"]) == ("review_prerequisite", "fact")
    quad = next(r for r in out["recommendations"] if r["topic_id"] == "quad")
    assert {"code": "weak_prerequisite", "topic_id": "fact", "label": "Factoring"} \
        in quad["reasons"]
    assert not any(r.get("topic_id") == "frac" for r in quad["reasons"])


def test_untouched_topics_in_the_lessons_the_student_is_in_are_offered(reads):
    """With no lessons named, the lessons the student has already answered in stand in
    for "current": an untouched topic there is the next thing on the page."""
    out = _summary()
    assert [t["topic_id"] for t in out["not_practised"]] == ["ineq"]
    assert ("start", "ineq") in [(r["action"], r["topic_id"]) for r in out["recommendations"]]

    named = _summary(lecture_ids=["b1"])
    assert named["not_practised"] == []


def test_a_student_with_no_graded_work_is_unavailable_not_all_clear(reads, monkeypatch):
    monkeypatch.setattr(analytics, "get_mastery", lambda uid, *a, **k: [])
    out = _summary(subject_ids=["math"])
    assert out["available"] is False
    assert out["degraded"] is False
    assert out["subjects"][0]["not_practised"] == 4


def test_an_unreadable_learner_model_is_degraded_not_empty(reads, monkeypatch):
    """A failed mastery read that came back as an empty analysis would tell a student
    with months of practice that they have not started."""
    def _unreachable(*a, **k):
        raise RuntimeError("database unreachable")

    monkeypatch.setattr(analytics, "get_mastery", _unreachable)
    out = _summary()
    assert out["degraded"] is True and out["available"] is False
    assert out["recommendations"] == [] and out["subjects"] == []


def test_a_failed_side_read_keeps_the_analysis_but_marks_it_degraded(reads, monkeypatch):
    def _unreachable(*a, **k):
        raise RuntimeError("database unreachable")

    monkeypatch.setattr(analytics, "get_direct_prereqs", _unreachable)
    monkeypatch.setattr(analytics, "get_weekly_activity", _unreachable)
    out = _summary()
    assert out["degraded"] is True and out["available"] is True
    assert out["recommendations"][0]["action"] == "practise"
    assert out["activity"] == []


def test_the_trend_and_activity_cover_the_same_subjects(reads, monkeypatch):
    seen = {}

    def _activity(uid, **k):
        seen["activity"] = k
        return []

    def _trend(uid, **k):
        seen["trend"] = k
        return {"direction": "flat", "count": 4, "events": []}

    monkeypatch.setattr(analytics, "get_weekly_activity", _activity)
    monkeypatch.setattr(analytics, "get_mastery_trend", _trend)
    out = _summary(subject_ids=["math"])
    assert seen["activity"]["subject_ids"] == ["math"]
    assert seen["trend"]["subject_id"] == "math"
    assert out["trend"] == {"direction": "flat", "count": 4}


def test_the_students_own_mistake_patterns_are_part_of_the_analysis(reads):
    """Every wrong answer is tagged with a mistake category at grading time, and none
    of it reached the student: "you often drop the minus sign" is the most direct
    advice the evidence can give."""
    out = _summary(subject_ids=["math"])
    assert out["mistakes"] == [{"tag_id": "g1", "label": "Sign error", "occurrences": 4}]
    ids, kwargs = reads["mistakes"]
    assert ids == ["u1"] and kwargs["subject_ids"] == ["math"]


def test_topics_likely_to_be_fading_form_a_review_queue(reads, monkeypatch):
    """A mastered topic dropped off the homepage for good; the queue brings it back
    when the student has probably started to forget it, most-faded first."""
    monkeypatch.setattr(analytics, "get_mastery", lambda uid, *a, **k: [
        _row("frac", 0.85, 3, "strong", days_ago=40),
        _row("cell", 0.60, 2, "developing", days_ago=9),
        _row("quad", 0.90, 6, "strong", days_ago=2),
    ])
    out = _summary()

    assert [t["topic_id"] for t in out["review_queue"]] == ["frac", "cell"]
    first = out["review_queue"][0]
    assert first["retention"] < out["review_queue"][1]["retention"] < 0.8
    assert first["overdue_days"] > 0
