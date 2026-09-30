"""What a student should practise next, and why. A pure policy: the same evidence
always gives the same list, so a homepage and a tutor never disagree."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from agent.mastery import RetentionParams
from agent.recommend import RecommendParams, TopicStanding, recommend

NOW = datetime(2026, 9, 27, tzinfo=timezone.utc)
PARAMS = RecommendParams(limit=5, stale_days=14, stale_weight=0.15, prereq_weight=0.2,
                         retention=RetentionParams(base_stability_days=7.0, growth=1.5,
                                                   max_stability_days=180.0, target=0.8))


def _t(tid, status, score=0.5, attempts=3, days_ago=1, recent=(), lecture="l1", label=None):
    return TopicStanding(
        topic_id=tid, label=label or tid.title(), subject_id="math", lecture_id=lecture,
        status=status, score=score, attempts=attempts,
        last_seen=NOW - timedelta(days=days_ago) if days_ago is not None else None,
        recent=list(recent))


def _ids(recs):
    return [(r.action, r.topic_id) for r in recs]


def test_a_weak_prerequisite_is_recommended_before_the_topic_built_on_it():
    """Practising a topic whose prerequisite is itself weak repeats the failure; the
    report already names the deeper cause, so the next step must start there."""
    standings = [_t("quad", "weak", score=0.3), _t("fact", "weak", score=0.4)]
    recs = recommend(standings, weak_prereqs={"quad": [("fact", "Factoring")]},
                     current_lectures=set(), now=NOW, params=PARAMS)

    assert _ids(recs)[:2] == [("review_prerequisite", "fact"), ("practise", "quad")]
    assert {"code": "prerequisite_of", "topic_id": "quad", "label": "Quad"} in recs[0].reasons
    assert {"code": "weak_prerequisite", "topic_id": "fact", "label": "Factoring"} \
        in recs[1].reasons


def test_weaker_and_longer_neglected_topics_come_first():
    standings = [_t("a", "weak", score=0.40, days_ago=1),
                 _t("b", "weak", score=0.20, days_ago=1),
                 _t("c", "weak", score=0.40, days_ago=30)]
    recs = recommend(standings, weak_prereqs={}, current_lectures=set(), now=NOW,
                     params=PARAMS)
    assert [r.topic_id for r in recs] == ["b", "c", "a"]


def test_weak_then_developing_then_new_lesson_topics_then_stale_strengths():
    standings = [_t("strong-old", "strong", score=0.9, days_ago=45),
                 _t("new", "not_practised", attempts=0, days_ago=None, lecture="l9"),
                 _t("dev", "developing", score=0.6),
                 _t("weak", "weak", score=0.3),
                 _t("strong-fresh", "strong", score=0.9, days_ago=2)]
    recs = recommend(standings, weak_prereqs={}, current_lectures={"l9"}, now=NOW,
                     params=PARAMS)
    assert _ids(recs) == [("practise", "weak"), ("strengthen", "dev"), ("start", "new"),
                          ("refresh", "strong-old")]


def test_every_recommendation_says_why_in_facts_not_prose():
    """Reasons are data so each client words them in the student's language; a
    sentence written here would be English in an Arabic homepage."""
    standings = [_t("frac", "weak", score=0.25, days_ago=20, recent=[0.0, 1.0, 0.0])]
    rec = recommend(standings, weak_prereqs={}, current_lectures=set(), now=NOW,
                    params=PARAMS)[0]
    assert {"code": "recent_misses", "missed": 2, "of": 3} in rec.reasons
    assert {"code": "low_mastery", "score": 0.25} in rec.reasons
    assert {"code": "not_practised_for", "days": 20} in rec.reasons


def test_a_topic_outside_the_current_lessons_is_not_suggested_as_new():
    standings = [_t("elsewhere", "not_practised", attempts=0, days_ago=None, lecture="l2")]
    assert recommend(standings, weak_prereqs={}, current_lectures={"l1"}, now=NOW,
                     params=PARAMS) == []


def test_a_topic_with_too_little_evidence_is_never_called_weak_or_strong():
    """Two lucky answers are not mastery and one slip is not a gap; an unknown topic
    is offered as more practice, never as a weakness."""
    standings = [_t("thin", "unknown", score=0.1, attempts=1)]
    recs = recommend(standings, weak_prereqs={}, current_lectures=set(), now=NOW,
                     params=PARAMS)
    assert _ids(recs) == [("strengthen", "thin")]
    assert {"code": "little_evidence", "attempts": 1} in recs[0].reasons


def test_the_list_is_capped_and_never_names_a_topic_twice():
    standings = [_t(f"w{i}", "weak", score=0.1 + i / 100) for i in range(8)]
    standings.append(_t("p", "weak", score=0.05))
    prereqs = {f"w{i}": [("p", "P")] for i in range(8)}
    recs = recommend(standings, weak_prereqs=prereqs, current_lectures=set(), now=NOW,
                     params=PARAMS)
    assert len(recs) == 5
    assert len({r.topic_id for r in recs}) == 5


def test_a_strong_topic_is_offered_for_review_when_it_is_likely_fading():
    """A flat 30-day rule reviewed a topic mastered once and a topic mastered ten times
    on the same day; how long a topic stays remembered depends on how it was learned."""
    once = _t("once", "strong", score=0.9, attempts=2, days_ago=12)
    often = _t("often", "strong", score=0.9, attempts=8, days_ago=12)
    recs = recommend([once, often], weak_prereqs={}, current_lectures=set(), now=NOW,
                     params=PARAMS)

    assert [(r.action, r.topic_id) for r in recs] == [("refresh", "once")]
    due = next(r for r in recs[0].reasons if r["code"] == "due_for_review")
    assert due["retention"] < 0.8 and due["overdue_days"] > 0
