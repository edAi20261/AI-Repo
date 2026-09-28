"""The student analysis: where a student stands in every subject, what they are
strong and weak in, and what to practise next.

Read-only and model-free, so a homepage can ask for it on every load. The words
around it (the narrative report) are generated elsewhere and stored; this is the
evidence they are written from.

One subject asked for is a subject page and everything is scoped to it. Otherwise
it is the homepage: every subject the student is enrolled in or has answered in.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any, Optional

from shared.config import Config

from .recommend import RecommendParams, TopicStanding, recommend
from .tools import (
    get_direct_prereqs,
    get_mastery,
    get_mastery_trend,
    get_recent_outcomes,
    get_scope_topics,
    get_top_mistakes,
    get_weekly_activity,
)

logger = logging.getLogger(__name__)

_STATUSES = ("strong", "developing", "weak", "unknown", "not_practised")
_JUDGED = ("strong", "developing", "weak")


def learner_summary(
    user_id: str,
    *,
    subject_ids: Sequence[str] | None = None,
    lecture_ids: Sequence[str] | None = None,
    now: Optional[datetime] = None,
    config: Config | None = None,
) -> dict[str, Any]:
    config = config or Config.load()
    now = now or datetime.now(timezone.utc)
    limits = _Limits(config)
    focus = subject_ids[0] if subject_ids and len(subject_ids) == 1 else None

    try:
        topics = get_scope_topics(user_id, subject_ids=subject_ids, config=config)
        rows = {r["topic_id"]: r for r in get_mastery(user_id, config=config)}
    except Exception:
        logger.error("learner summary: the learner model could not be read", exc_info=True)
        return _unreadable()
    if focus:
        topics = [t for t in topics if t["subject_id"] == focus]

    degraded = False
    uncertain = [t["topic_id"] for t in topics
                 if rows.get(t["topic_id"], {}).get("status") in ("weak", "developing",
                                                                   "unknown")]
    recent: dict[str, list[float]] = {}
    if uncertain:
        try:
            recent = get_recent_outcomes(user_id, uncertain,
                                         per_topic=limits.recent_per_topic, config=config)
        except Exception:
            degraded = True
            logger.error("learner summary: recent outcomes unreadable", exc_info=True)

    standings = [_standing(t, rows.get(t["topic_id"]), recent) for t in topics]
    by_id = {s.topic_id: s for s in standings}
    weak_ids = [s.topic_id for s in standings if s.status == "weak"]
    weak_prereqs: dict[str, list[tuple[str, str]]] = {}
    if weak_ids:
        try:
            direct = get_direct_prereqs(weak_ids, config=config)
            weak_prereqs = {tid: [(pid, label) for pid, label in pre
                                  if by_id.get(pid) and by_id[pid].status == "weak"]
                            for tid, pre in direct.items()}
        except Exception:
            degraded = True
            logger.error("learner summary: prerequisites unreadable", exc_info=True)

    current = set(lecture_ids or {s.lecture_id for s in standings
                                  if s.status != "not_practised" and s.lecture_id})
    recs = recommend(standings, weak_prereqs=weak_prereqs, current_lectures=current,
                     now=now, params=RecommendParams.from_config(config))

    activity: list[dict[str, Any]] = []
    trend: dict[str, Any] = {"direction": "insufficient", "count": 0}
    try:
        activity = get_weekly_activity(user_id, weeks=limits.activity_weeks,
                                       subject_ids=[focus] if focus else None, config=config)
    except Exception:
        degraded = True
        logger.error("learner summary: weekly activity unreadable", exc_info=True)
    try:
        t = get_mastery_trend(user_id, subject_id=focus, config=config)
        trend = {"direction": t.get("direction", "insufficient"), "count": t.get("count", 0)}
    except Exception:
        degraded = True
        logger.error("learner summary: trend unreadable", exc_info=True)

    mistakes: list[dict[str, Any]] = []
    try:
        mistakes = [{k: m[k] for k in ("tag_id", "label", "occurrences")}
                    for m in get_top_mistakes([user_id], since_days=limits.mistake_window_days,
                                              limit=limits.mistakes,
                                              subject_ids=[focus] if focus else subject_ids,
                                              config=config)]
    except Exception:
        degraded = True
        logger.error("learner summary: mistake patterns unreadable", exc_info=True)

    judged = [s for s in standings if s.status in _JUDGED]
    strengths = sorted((s for s in judged if s.status == "strong"),
                       key=lambda s: (-s.score, s.label))[:limits.strengths]
    weaknesses = sorted((s for s in judged if s.status == "weak"),
                        key=lambda s: (s.score, s.label))[:limits.weaknesses]
    untouched = [s for s in standings
                 if s.status == "not_practised" and s.lecture_id in current]

    logger.info("learner summary", extra={
        "topics": len(standings), "judged": len(judged), "recommendations": len(recs),
        "degraded": degraded, "focused": bool(focus)})
    return {
        "available": bool(judged),
        "degraded": degraded,
        "subjects": _subject_counts(standings),
        "strengths": [_topic(s) for s in strengths],
        "weaknesses": [_topic(s) for s in weaknesses],
        "not_practised": [_topic(s) for s in untouched[:limits.not_practised]],
        "recommendations": [r.to_dict() for r in recs],
        "mistakes": mistakes,
        "activity": activity,
        "trend": trend,
    }


class _Limits:
    def __init__(self, config: Config) -> None:
        self.strengths = config.get("learner_summary.strengths_limit", 5)
        self.weaknesses = config.get("learner_summary.weaknesses_limit", 5)
        self.not_practised = config.get("learner_summary.not_practised_limit", 5)
        self.recent_per_topic = config.get("learner_summary.recent_per_topic", 5)
        self.activity_weeks = config.get("learner_summary.activity_weeks", 8)
        self.mistakes = config.get("learner_summary.mistakes_limit", 3)
        self.mistake_window_days = config.get("learner_summary.mistake_window_days", 60)


def _standing(topic: dict, row: Optional[dict], recent: dict[str, list[float]]) -> TopicStanding:
    return TopicStanding(
        topic_id=topic["topic_id"], label=topic.get("label") or "",
        subject_id=topic.get("subject_id"), lecture_id=topic.get("lecture_id"),
        status=row["status"] if row else "not_practised",
        score=float((row or {}).get("score") or 0.0),
        attempts=int((row or {}).get("attempts") or 0),
        last_seen=(row or {}).get("last_seen"),
        recent=recent.get(topic["topic_id"], []))


def _topic(s: TopicStanding) -> dict[str, Any]:
    return {"topic_id": s.topic_id, "label": s.label, "subject_id": s.subject_id,
            "lecture_id": s.lecture_id, "status": s.status, "score": round(s.score, 4),
            "attempts": s.attempts,
            "last_seen": s.last_seen.isoformat() if s.last_seen else None,
            "recent": s.recent}


def _subject_counts(standings: Sequence[TopicStanding]) -> list[dict[str, Any]]:
    counts: dict[str, dict[str, Any]] = {}
    for s in standings:
        c = counts.setdefault(s.subject_id or "", {"subject_id": s.subject_id, "topics": 0,
                                                   **dict.fromkeys(_STATUSES, 0)})
        c["topics"] += 1
        c[s.status] += 1
    return list(counts.values())


def _unreadable() -> dict[str, Any]:
    return {"available": False, "degraded": True, "subjects": [], "strengths": [],
            "weaknesses": [], "not_practised": [], "recommendations": [], "mistakes": [],
            "activity": [],
            "trend": {"direction": "insufficient", "count": 0}}
