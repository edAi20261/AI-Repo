"""Topic suggestions for questions written outside ingestion.

A quiz a teacher writes carries no topic, and an answer with no topic moves no
mastery. Suggesting the nearest syllabus topics per question lets the teacher
confirm them once, when the question is saved, instead of the grader guessing on
every submission.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

from shared.config import Config

from .tools import get_topic_lectures, resolve_topics_many

logger = logging.getLogger(__name__)


def suggest_topics(
    questions: Sequence[Mapping[str, Any]],
    subject_id: str,
    *,
    lecture_ids: Sequence[str] | None = None,
    config: Config | None = None,
) -> dict[str, Any]:
    """Nearest topics for each question, in request order.

    ``resolved`` is False only when the lookup itself failed, so "nothing in the
    syllabus matches" and "the lookup broke" never read the same.
    """
    config = config or Config.load()
    k = config.get("tagging.k", 3)
    threshold = config.get("tagging.threshold", 0.40)
    try:
        found = resolve_topics_many([q["text"] for q in questions], subject_id,
                                    lecture_ids=lecture_ids, k=k, threshold=threshold,
                                    config=config)
        resolved = True
    except Exception:
        logger.error("topic suggestion failed", exc_info=True,
                     extra={"questions": len(questions)})
        found, resolved = [[] for _ in questions], False
    rows = [{"index": index, "question_id": question.get("question_id"),
             "resolved": resolved, "hits": hits}
            for index, (question, hits) in enumerate(zip(questions, found))]

    lectures, lectures_ok = _lectures_of({h.topic_id for r in rows for h in r["hits"]}, config)
    suggestions = [{
        "index": r["index"], "question_id": r["question_id"], "resolved": r["resolved"],
        "topics": [{"topic_id": h.topic_id, "label": h.label, "score": round(h.score, 4),
                    "lecture_id": lectures.get(h.topic_id)} for h in r["hits"]],
    } for r in rows]
    failed = sum(1 for r in rows if not r["resolved"])
    logger.info("suggest_topics", extra={
        "questions": len(rows), "failed": failed,
        "untagged": sum(1 for s in suggestions if s["resolved"] and not s["topics"])})
    return {"suggestions": suggestions, "degraded": failed > 0 or not lectures_ok}


def _lectures_of(topic_ids: set[str], config: Config) -> tuple[dict[str, str], bool]:
    if not topic_ids:
        return {}, True
    try:
        return get_topic_lectures(sorted(topic_ids), config=config), True
    except Exception:
        logger.error("lesson lookup for suggested topics failed", exc_info=True)
        return {}, False
