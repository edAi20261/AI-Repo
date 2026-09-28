"""A teacher's view of one class: where the class stands per topic, who needs
attention and why, and the mistakes and questions that trip it up. No model calls;
the roster comes from the caller, which owns who may see whom."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from agent import class_analytics as cmod
from agent.config import get_config

NOW = datetime(2026, 9, 27, tzinfo=timezone.utc)
TOPICS = [
    {"topic_id": "t1", "label": "Fractions", "lecture_id": "l1", "subject_id": "math"},
    {"topic_id": "t2", "label": "Decimals", "lecture_id": "l1", "subject_id": "math"},
    {"topic_id": "t3", "label": "Ratios", "lecture_id": "l2", "subject_id": "math"},
]


def _m(uid, tid, score, status):
    return {"user_id": uid, "topic_id": tid, "score": score, "attempts": 4,
            "last_seen": NOW, "status": status}


MASTERY = [
    _m("s1", "t1", 0.2, "weak"), _m("s1", "t2", 0.3, "weak"), _m("s1", "t3", 0.8, "strong"),
    _m("s2", "t1", 0.9, "strong"), _m("s2", "t2", 0.6, "developing"),
    _m("s2", "t3", 0.85, "strong"),
    _m("s3", "t1", 0.35, "weak"),
]
HISTORY = {
    "s1": {"last_active": NOW - timedelta(days=1), "scores": [1, 1, 1, 0, 0, 0]},
    "s2": {"last_active": NOW - timedelta(days=2), "scores": [0.5, 1, 1, 1]},
    "s3": {"last_active": NOW - timedelta(days=20), "scores": [0.3]},
}


@pytest.fixture(autouse=True)
def reads(monkeypatch):
    monkeypatch.setattr(cmod, "get_scope_topics", lambda ids, **k: TOPICS)
    monkeypatch.setattr(cmod, "get_roster_mastery", lambda ids, **k: MASTERY)
    monkeypatch.setattr(cmod, "get_roster_history", lambda ids, **k: HISTORY)
    monkeypatch.setattr(cmod, "get_top_mistakes",
                        lambda ids, **k: [{"tag_id": "g1", "label": "Sign error",
                                           "occurrences": 7, "students": 2}])
    monkeypatch.setattr(cmod, "get_most_missed",
                        lambda ids, **k: [{"item_id": "q1", "topic_id": "t1", "answers": 5,
                                           "wrong": 4, "students": 3}])
    monkeypatch.setattr(cmod, "get_topic_labels", lambda ids, **k: {"t1": "Fractions"})
    monkeypatch.setattr(cmod, "get_weekly_activity",
                        lambda ids, **k: [{"week_start": "2026-09-21", "answers": 20,
                                           "correct": 12, "mean_score": 0.6}])


def _run(roster=("s1", "s2", "s3", "s4"), **kw):
    return cmod.class_analytics(list(roster), subject_ids=["math"], now=NOW,
                                config=get_config(), **kw)


def test_each_topic_shows_how_the_class_is_spread_across_it():
    """An average hides a split class; a teacher needs to see that four students are
    weak on a topic two others have mastered."""
    out = _run()
    t1 = next(t for t in out["topics"] if t["topic_id"] == "t1")
    assert (t1["strong"], t1["developing"], t1["weak"], t1["not_practised"]) == (1, 0, 2, 1)
    assert t1["mean_score"] == pytest.approx((0.2 + 0.9 + 0.35) / 3, abs=1e-4)
    assert out["topics"][0]["topic_id"] == "t1"
    assert out["students"] == {"total": 4, "with_evidence": 3}


def test_a_student_is_flagged_with_the_reasons_that_flagged_them():
    out = _run()
    flagged = {s["user_id"]: s for s in out["at_risk"]}

    assert {"code": "many_weak_topics", "weak": 2, "of": 3} in flagged["s1"]["reasons"]
    assert {"code": "declining"} in flagged["s1"]["reasons"]
    assert {"code": "inactive", "days": 20} in flagged["s3"]["reasons"]
    assert "s2" not in flagged
    assert out["at_risk"][0]["user_id"] == "s1"


def test_too_few_judged_topics_never_flag_a_student_as_mostly_weak():
    """One weak topic out of one is not a pattern."""
    out = _run()
    s3 = next(s for s in out["at_risk"] if s["user_id"] == "s3")
    assert not any(r["code"] == "many_weak_topics" for r in s3["reasons"])


def test_a_student_with_no_graded_work_is_listed_as_not_started_not_at_risk():
    out = _run()
    assert out["not_started"] == ["s4"]
    assert "s4" not in {s["user_id"] for s in out["at_risk"]}


def test_missed_questions_and_mistakes_come_with_topic_labels():
    out = _run()
    assert out["most_missed"][0]["label"] == "Fractions"
    assert out["mistakes"][0]["label"] == "Sign error"


def test_an_unreadable_roster_model_is_degraded_not_an_empty_class(monkeypatch):
    def _unreachable(*a, **k):
        raise RuntimeError("database unreachable")

    monkeypatch.setattr(cmod, "get_roster_mastery", _unreachable)
    out = _run()
    assert out["degraded"] is True and out["available"] is False
    assert out["topics"] == [] and out["at_risk"] == []


def test_a_failed_side_read_keeps_the_heatmap_and_says_it_is_partial(monkeypatch):
    def _unreachable(*a, **k):
        raise RuntimeError("database unreachable")

    monkeypatch.setattr(cmod, "get_top_mistakes", _unreachable)
    out = _run()
    assert out["degraded"] is True and out["available"] is True
    assert out["mistakes"] == [] and out["topics"]


def test_an_oversized_roster_is_refused():
    with pytest.raises(ValueError):
        _run(roster=[f"s{i}" for i in range(501)])


def test_students_are_spread_into_score_bands():
    """The Analytics tab draws a score distribution; a class average of 0.5 is two very
    different classes depending on whether everyone is at 0.5 or half at 0 and half at 1."""
    out = _run()
    bands = {b["band"]: b["students"] for b in out["score_distribution"]}
    assert bands == {"0-20": 0, "20-40": 1, "40-60": 1, "60-80": 1, "80-100": 0}
    assert sum(bands.values()) == 3


def test_engagement_counts_who_answered_recently():
    """The homepage showed an invented "engagement" figure; this is who actually
    answered something in the last week, out of the whole class."""
    out = _run()
    assert out["engagement"] == {"active": 2, "total": 4, "share": 0.5, "window_days": 7,
                                 "answers_this_week": 20}


def test_students_weak_on_the_same_topic_are_suggested_as_a_group():
    """A teacher with 30 students cannot re-teach each one; students stuck on the same
    topic can be taught together, and the group comes with the topic to target."""
    out = _run()
    assert out["groups"] == [{"topic_id": "t1", "label": "Fractions", "lecture_id": "l1",
                              "subject_id": "math", "user_ids": ["s1", "s3"], "size": 2}]


def test_a_topic_only_one_student_is_weak_on_is_not_a_group():
    out = _run()
    assert all(g["size"] >= 2 for g in out["groups"])
    assert "t2" not in {g["topic_id"] for g in out["groups"]}
