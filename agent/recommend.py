"""What a student should practise next, and why.

A pure policy over evidence already read: the same standings always produce the
same list, so the homepage, the tutor and a teacher's view of the student can
never disagree about it. Reasons are returned as facts (``{"code": ...}``), never
sentences, because the reader's language is the client's to render.

Order, strongest claim first:
  1. weak topics, each preceded by any weak prerequisite it rests on
  2. developing topics, and topics with too little evidence to judge
  3. untouched topics in the lessons the student is working through now
  4. strong topics whose estimated retention has fallen below the target
Within a tier, the larger the gap and the longer the neglect, the sooner it comes.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from .mastery import Mastery, RetentionParams, days_until_due, retention

_TIER = {"review_prerequisite": 0, "practise": 0, "strengthen": 1, "start": 2, "refresh": 3}


@dataclass(frozen=True)
class RecommendParams:
    limit: int = 5
    stale_days: int = 14
    stale_weight: float = 0.15
    prereq_weight: float = 0.2
    pass_threshold: float = 0.5
    retention: RetentionParams = field(default_factory=RetentionParams)

    @classmethod
    def from_config(cls, config) -> RecommendParams:
        return cls(
            limit=config.get("recommend.limit", 5),
            stale_days=config.get("recommend.stale_days", 14),
            stale_weight=config.get("recommend.stale_weight", 0.15),
            prereq_weight=config.get("recommend.prereq_weight", 0.2),
            pass_threshold=config.get("grade.pass_threshold", 0.5),
            retention=RetentionParams.from_config(config),
        )


@dataclass(frozen=True)
class TopicStanding:
    topic_id: str
    label: str
    subject_id: Optional[str]
    lecture_id: Optional[str]
    status: str
    score: float = 0.0
    attempts: int = 0
    last_seen: Optional[datetime] = None
    recent: list[float] = field(default_factory=list)


@dataclass
class Recommendation:
    topic_id: str
    label: str
    subject_id: Optional[str]
    lecture_id: Optional[str]
    action: str
    priority: float
    reasons: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {"topic_id": self.topic_id, "label": self.label,
                "subject_id": self.subject_id, "lecture_id": self.lecture_id,
                "action": self.action, "priority": round(self.priority, 4),
                "reasons": self.reasons}


def recommend(
    standings: Sequence[TopicStanding],
    *,
    weak_prereqs: Mapping[str, Sequence[tuple[str, str]]],
    current_lectures: Iterable[str],
    now: datetime,
    params: RecommendParams,
) -> list[Recommendation]:
    by_id = {s.topic_id: s for s in standings}
    current = set(current_lectures)
    candidates: list[Recommendation] = []

    for s in standings:
        if s.status == "weak":
            risk = _gap(s) + _neglect(s, now, params)
            prereqs = [(pid, label) for pid, label in weak_prereqs.get(s.topic_id, ())
                       if pid != s.topic_id]
            for pid, plabel in prereqs:
                p = by_id.get(pid)
                candidates.append(_rec(
                    p or s, "review_prerequisite", risk + params.prereq_weight,
                    [{"code": "prerequisite_of", "topic_id": s.topic_id, "label": s.label},
                     *(_evidence(p, now, params) if p else [])],
                    topic_id=pid, label=(p.label if p else plabel)))
            candidates.append(_rec(
                s, "practise", risk,
                [*_evidence(s, now, params),
                 *({"code": "weak_prerequisite", "topic_id": pid, "label": plabel}
                   for pid, plabel in prereqs)]))
        elif s.status in ("developing", "unknown"):
            candidates.append(_rec(s, "strengthen", _gap(s) + _neglect(s, now, params),
                                   _evidence(s, now, params)))
        elif s.status == "not_practised" and s.lecture_id in current:
            candidates.append(_rec(s, "start", 0.0,
                                   [{"code": "in_current_lesson", "lecture_id": s.lecture_id}]))
        elif s.status == "strong":
            due = review_due(s, now, params.retention)
            if due:
                candidates.append(_rec(s, "refresh", 1.0 - due["retention"],
                                       [{"code": "mastered", "score": round(s.score, 2)},
                                        due]))

    candidates.sort(key=lambda r: (_TIER[r.action], -r.priority, r.topic_id))
    chosen: list[Recommendation] = []
    seen: set[str] = set()
    for r in candidates:
        if r.topic_id in seen:
            continue
        seen.add(r.topic_id)
        chosen.append(r)
        if len(chosen) == params.limit:
            break
    return chosen


def review_due(s: TopicStanding, now: datetime,
               params: RetentionParams) -> Optional[dict[str, Any]]:
    """The ``due_for_review`` reason when this topic's estimated retention is below the
    target, else None. A topic never assessed has nothing to forget yet."""
    days = _days_since_exact(s, now)
    if days is None or not s.attempts:
        return None
    m = Mastery(score=s.score, attempts=s.attempts)
    left = days_until_due(m, days, params)
    if left > 0:
        return None
    return {"code": "due_for_review", "retention": round(retention(m, days, params), 2),
            "overdue_days": round(-left, 1)}


def _days_since_exact(s: TopicStanding, now: datetime) -> Optional[float]:
    if s.last_seen is None:
        return None
    return max(0.0, (now - s.last_seen).total_seconds() / 86400)


def _rec(s: TopicStanding, action: str, priority: float, reasons: list[dict[str, Any]],
         *, topic_id: Optional[str] = None, label: Optional[str] = None) -> Recommendation:
    return Recommendation(topic_id=topic_id or s.topic_id, label=label or s.label,
                          subject_id=s.subject_id, lecture_id=s.lecture_id,
                          action=action, priority=priority, reasons=reasons)


def _gap(s: TopicStanding) -> float:
    return 1.0 - max(0.0, min(1.0, s.score or 0.0))


def _days_since(s: TopicStanding, now: datetime) -> Optional[int]:
    if s.last_seen is None:
        return None
    return max(0, (now - s.last_seen).days)


def _neglect(s: TopicStanding, now: datetime, params: RecommendParams) -> float:
    days = _days_since(s, now)
    if days is None or params.stale_days <= 0:
        return 0.0
    return params.stale_weight * min(1.0, days / params.stale_days)


def _evidence(s: TopicStanding, now: datetime, params: RecommendParams) -> list[dict[str, Any]]:
    reasons: list[dict[str, Any]] = []
    missed = sum(1 for score in s.recent if score < params.pass_threshold)
    if missed:
        reasons.append({"code": "recent_misses", "missed": missed, "of": len(s.recent)})
    if s.status == "unknown":
        reasons.append({"code": "little_evidence", "attempts": s.attempts})
    elif s.status == "weak":
        reasons.append({"code": "low_mastery", "score": round(s.score, 2)})
    elif s.status == "developing":
        reasons.append({"code": "partial_mastery", "score": round(s.score, 2)})
    days = _days_since(s, now)
    if days is not None and days >= params.stale_days:
        reasons.append({"code": "not_practised_for", "days": days})
    return reasons
