"""A teacher's view of one class: where the class stands on each topic, who needs
attention and why, and the mistakes and questions that trip it up.

Read-only and model-free. The roster is the caller's: the platform decides which
students a teacher may see, and this service only aggregates the ids it is given.
A student is flagged with the facts that flagged them, never with a bare label,
because a teacher acts on the reason, not on the flag.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any, Optional

from shared.config import Config

from .tools import (
    get_most_missed,
    get_roster_history,
    get_roster_mastery,
    get_scope_topics,
    get_top_mistakes,
    get_topic_labels,
    get_weekly_activity,
    trend_direction,
)

logger = logging.getLogger(__name__)

_JUDGED = ("strong", "developing", "weak")
_STATUSES = ("strong", "developing", "weak", "unknown", "not_practised")


def class_analytics(
    user_ids: Sequence[str],
    *,
    subject_ids: Sequence[str] | None = None,
    now: Optional[datetime] = None,
    config: Config | None = None,
) -> dict[str, Any]:
    config = config or Config.load()
    now = now or datetime.now(timezone.utc)
    p = _Params(config)
    roster = list(dict.fromkeys(str(u) for u in user_ids))
    if len(roster) > p.max_students:
        raise ValueError(f"a roster of {len(roster)} exceeds {p.max_students} students.")

    try:
        topics = get_scope_topics(roster, subject_ids=subject_ids, config=config)
        rows = get_roster_mastery(roster, subject_ids=subject_ids, config=config)
    except Exception:
        logger.error("class analytics: the roster's learner model could not be read",
                     exc_info=True)
        return _unreadable(len(roster))
    if subject_ids:
        wanted = set(subject_ids)
        topics = [t for t in topics if t["subject_id"] in wanted]
    in_scope = {t["topic_id"] for t in topics}
    rows = [r for r in rows if r["topic_id"] in in_scope]

    degraded = False
    history: dict[str, dict[str, Any]] = {}
    try:
        history = get_roster_history(roster, per_user=p.history_per_student, config=config)
    except Exception:
        degraded = True
        logger.error("class analytics: roster history unreadable", exc_info=True)

    by_student: dict[str, dict[str, dict]] = {}
    for r in rows:
        by_student.setdefault(r["user_id"], {})[r["topic_id"]] = r

    mistakes, missed, activity = [], [], []
    try:
        mistakes = get_top_mistakes(roster, since_days=p.window_days, limit=p.mistakes_limit,
                                    subject_ids=subject_ids, config=config)
    except Exception:
        degraded = True
        logger.error("class analytics: mistakes unreadable", exc_info=True)
    try:
        missed = get_most_missed(roster, since_days=p.window_days, limit=p.most_missed_limit,
                                 subject_ids=subject_ids, config=config)
        labels = get_topic_labels(sorted({m["topic_id"] for m in missed if m["topic_id"]}),
                                  config=config) if missed else {}
        missed = [{**m, "label": labels.get(m["topic_id"], "")} for m in missed]
    except Exception:
        degraded = True
        logger.error("class analytics: missed questions unreadable", exc_info=True)
    try:
        activity = get_weekly_activity(roster, weeks=p.activity_weeks,
                                       subject_ids=subject_ids, config=config)
    except Exception:
        degraded = True
        logger.error("class analytics: weekly activity unreadable", exc_info=True)

    started = set(by_student) | set(history)
    at_risk = sorted((a for u in roster if u in started
                      if (a := _assess(u, by_student.get(u, {}), history.get(u), now, p))),
                     key=lambda a: (-len(a["reasons"]), -a["weak_share"], a["user_id"]))
    logger.info("class analytics", extra={
        "students": len(roster), "started": len(started), "topics": len(topics),
        "at_risk": len(at_risk), "degraded": degraded})
    return {
        "available": any(r["status"] in _JUDGED for r in rows),
        "degraded": degraded,
        "students": {"total": len(roster), "with_evidence": len(started)},
        "topics": _topic_spread(topics, roster, by_student),
        "at_risk": at_risk[:p.at_risk_limit],
        "not_started": [u for u in roster if u not in started],
        "mistakes": mistakes,
        "most_missed": missed,
        "activity": activity,
    }


class _Params:
    def __init__(self, config: Config) -> None:
        self.weak_share = config.get("class_analytics.at_risk_weak_share", 0.4)
        self.min_judged = config.get("class_analytics.at_risk_min_judged", 3)
        self.inactive_days = config.get("class_analytics.inactive_days", 14)
        self.history_per_student = config.get("class_analytics.history_per_student", 20)
        self.window_days = config.get("class_analytics.window_days", 30)
        self.mistakes_limit = config.get("class_analytics.mistakes_limit", 5)
        self.most_missed_limit = config.get("class_analytics.most_missed_limit", 5)
        self.at_risk_limit = config.get("class_analytics.at_risk_limit", 10)
        self.max_students = config.get("class_analytics.max_students", 500)
        self.activity_weeks = config.get("learner_summary.activity_weeks", 8)
        self.trend_min_events = config.get("report.trend.min_events", 4)
        self.trend_flat_band = config.get("report.trend.flat_band", 0.05)


def _topic_spread(topics: list[dict], roster: list[str],
                  by_student: dict[str, dict[str, dict]]) -> list[dict[str, Any]]:
    spread = []
    for t in topics:
        counts = dict.fromkeys(_STATUSES, 0)
        scores = []
        for u in roster:
            row = by_student.get(u, {}).get(t["topic_id"])
            counts[row["status"] if row else "not_practised"] += 1
            if row:
                scores.append(float(row["score"] or 0.0))
        spread.append({**t, **counts,
                       "mean_score": round(sum(scores) / len(scores), 4) if scores else None})
    return sorted(spread, key=lambda t: (-t["weak"], t["mean_score"] if t["mean_score"]
                                         is not None else 2.0, t["label"]))


def _assess(user_id: str, topics: dict[str, dict], history: Optional[dict],
            now: datetime, p: _Params) -> Optional[dict[str, Any]]:
    judged = [r for r in topics.values() if r["status"] in _JUDGED]
    weak = sum(1 for r in judged if r["status"] == "weak")
    share = weak / len(judged) if judged else 0.0
    scores = (history or {}).get("scores") or []
    direction = trend_direction(scores, p.trend_min_events, p.trend_flat_band)
    last_active = (history or {}).get("last_active")
    idle = (now - last_active).days if last_active else None

    reasons: list[dict[str, Any]] = []
    if len(judged) >= p.min_judged and share >= p.weak_share:
        reasons.append({"code": "many_weak_topics", "weak": weak, "of": len(judged)})
    if direction == "declining":
        reasons.append({"code": "declining"})
    if idle is not None and idle >= p.inactive_days:
        reasons.append({"code": "inactive", "days": idle})
    if not reasons:
        return None
    return {"user_id": user_id, "reasons": reasons, "weak": weak, "judged": len(judged),
            "weak_share": round(share, 4),
            "mean_score": round(sum(scores) / len(scores), 4) if scores else None,
            "direction": direction,
            "last_active": last_active.isoformat() if last_active else None}


def _unreadable(total: int) -> dict[str, Any]:
    return {"available": False, "degraded": True,
            "students": {"total": total, "with_evidence": 0}, "topics": [], "at_risk": [],
            "not_started": [], "mistakes": [], "most_missed": [], "activity": []}
