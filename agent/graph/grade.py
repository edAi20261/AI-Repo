"""Grading entrypoint (endpoint path).

The backend's ``POST /ai/grade`` calls ``run_grade`` with a self-contained payload
(invariant 4): the student's answers plus the questions/rubric, or an assessment
reference to fetch the answer key. AI-final — grade_sg writes mastery itself
and returns the per-answer suggestions; the backend persists the grade row. This is the
structured, multi-answer path; the chat 'grade me this' turn uses the grade_adapter.
"""

from __future__ import annotations

from typing import Any, Optional

from shared.backend import get_backend_client
from shared.llm import maybe_llm

from ..config import Config, get_config
from .graph import (
    _kind_source, _subject_facts, _subject_instructions)
from .subgraphs import build_grade_sg


def run_grade(payload: dict[str, Any], *, config: Optional[Config] = None) -> dict:
    """Grade ``payload['submissions']`` and move mastery. Returns
    ``{graded, summary, mastery_events, topics_updated, source}``.

    Payload: ``submissions`` (each ``{question, student_answer, correct_answer?,
    topic_id?, lecture_id?, question_id?, type?, difficulty?, bloom?, points?}``),
    ``user_id``, ``subject_id?``, ``kind?`` (quiz|homework), ``assessment_id?`` (to
    fetch the rubric), ``source?``, ``language?``.

    Inputs supported: inline free-text answers and a DB quiz/homework reference.
    TODO: attachment answers (photo/PDF of solved work) — needs the upload/extraction
    (vision/OCR) pipeline before they can be graded."""
    config = config or get_config()
    try:
        client = get_backend_client(config)
    except ValueError:
        client = None
    facts = _subject_facts(client, payload.get("subject_id"))
    g = config.get("grade", {}) or {}

    sg = build_grade_sg(config=config, llm=maybe_llm(config, "grade"))
    result = sg.invoke({
        "submissions": payload.get("submissions") or [],
        "subject_id": payload.get("subject_id"),
        "lecture_ids": payload.get("lecture_ids") or None,
        "user_id": payload.get("user_id"),
        "language": payload.get("language", "en"),
        "instructions": _subject_instructions(facts),
        "kind": payload.get("kind"),
        "assessment_id": payload.get("assessment_id"),
        "source": payload.get("source") or _kind_source(payload.get("kind"), g),
        "pass_threshold": g.get("pass_threshold", 0.5),
    })
    return result.get("result", {})
