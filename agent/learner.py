"""What the tutor knows about this student before it answers them.

``ai.mastery`` has held a per-topic estimate of what a student can do since
grading was built, and until now the only things that read it were the parts
that produce a number: assessment weighting, grading, the report. The part that
actually talks to the student read the conversation and nothing else — so a
student could miss three questions on fractions and have the next explanation
pitched exactly as if they had never been assessed.

This is that estimate, scoped to the subject in play and shaped for a prompt:
the topics the evidence says are weak, the ones it says are secure, and nothing
about the topics it has not seen enough of. ``unknown`` is not knowledge, and a
tutor told a student is weak on something it has one data point for will teach
to a fact that is not true.

Nothing here may cost a student their answer. The read is bounded and its
failure degrades to an empty standing — the tutor then knows what it knew
before this module existed — but never silently: an operator can see in the log
that the learner model was unavailable, which is the difference between a
student with no history and a database that is down.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Sequence

from .tools import get_mastery, get_topic_labels

logger = logging.getLogger(__name__)

# Enough to steer an explanation, not enough to write an essay from. A model
# handed twenty topic names starts ranging over all of them instead of
# answering the question in front of it.
MAX_WEAK = 5
MAX_STRONG = 3

NOTHING_RECORDED = "(nothing recorded for this student yet)"


@dataclass(frozen=True)
class LearnerStanding:
    """What the evidence says this student can and cannot do, in one subject."""

    weak: list[str] = field(default_factory=list)
    strong: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.weak or self.strong)

    def render(self) -> str:
        """The standing as a prompt quotes it."""
        if not self:
            return NOTHING_RECORDED
        lines = []
        if self.weak:
            lines.append("Answered incorrectly before: " + ", ".join(self.weak) + ".")
        if self.strong:
            lines.append("Answered correctly before: " + ", ".join(self.strong) + ".")
        return "\n".join(lines)

    def log_fields(self) -> dict:
        return {"weak_topics": len(self.weak), "strong_topics": len(self.strong)}


def standing(
    user_id: Optional[str],
    subject_id: Optional[str],
    *,
    topic_ids: Optional[Sequence[str]] = None,
    config=None,
) -> LearnerStanding:
    """This student's standing in ``subject_id``, or an empty one.

    ``topic_ids`` narrows it to the topics this turn is about, which is what a
    turn wants; without them the whole subject is read, which is what a turn
    that resolved no topic wants.
    """
    if not (user_id and subject_id):
        return LearnerStanding()
    try:
        rows = get_mastery(user_id, topic_ids or None, subject_id=subject_id, config=config)
    except Exception:
        logger.warning("the learner model could not be read; answering without it",
                       exc_info=True, extra={"subject_id": subject_id})
        return LearnerStanding()

    ranked = {"weak": [], "strong": []}
    for row in rows:
        bucket = ranked.get(row.get("status"))
        if bucket is not None:
            bucket.append(row)
    # Weakest first, strongest first: what a tutor would raise before what it
    # can take for granted.
    ranked["weak"].sort(key=lambda row: row.get("score") or 0.0)
    ranked["strong"].sort(key=lambda row: row.get("score") or 0.0, reverse=True)
    weak = ranked["weak"][:MAX_WEAK]
    strong = ranked["strong"][:MAX_STRONG]

    labels = _labels([row["topic_id"] for row in (*weak, *strong)], config)
    return LearnerStanding(weak=_named(weak, labels), strong=_named(strong, labels))


def _labels(topic_ids: list[str], config) -> dict[str, str]:
    if not topic_ids:
        return {}
    try:
        return get_topic_labels(topic_ids, config=config)
    except Exception:
        logger.warning("topic labels for the learner model could not be read",
                       exc_info=True)
        return {}


def _named(rows: list[dict], labels: dict[str, str]) -> list[str]:
    """Topics the tutor can name. A topic whose label is missing is dropped:
    a bare uuid in a prompt is noise the model has to ignore or, worse, quote."""
    return [label for row in rows if (label := (labels.get(row["topic_id"]) or "").strip())]
