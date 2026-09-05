"""Reporting entrypoint (endpoint path).

The backend's ``GET /ai/reports/{user_id}`` calls ``run_report`` with a scope + the
student's ``user_id``. Read-only: report_sg reads mastery, walks prerequisites for the
root cause, and narrates — it moves no mastery and (for now) is not persisted to
``ai.reports``. This is the structured path (returns the full payload); the chat 'how am
I doing' turn uses the report_adapter, which surfaces only the narrative + trend chart.

Only ``student_progress`` is served today; other scopes return an ``available: False``
shell (they need classroom/parent scope — Sprint 3).
"""

from __future__ import annotations

from typing import Any, Optional

from shared.backend import get_backend_client
from shared.llm import maybe_llm

from ..config import Config, get_config
from .graph import _subject_facts
from .subgraphs import build_report_sg


def run_report(payload: dict[str, Any], *, config: Optional[Config] = None) -> dict:
    """Build a progress report. Returns the ``payload_json``-shaped dict
    ``{scope, available, narrative, weak_topics[], root_cause, trend, degraded}`` plus a
    ``visual_spec`` (trend chart) when there are enough points.

    Payload: ``user_id``, ``scope?`` (default ``student_progress``), ``subject_id?``,
    ``lecture_ids?``, ``language?``.

    TODO (deferred): persist the report to ``ai.reports`` (agent owns the table,
    invariant 3) — INSERT ``{scope, payload_json, artifact_refs, generated_at}`` so reports
    have history; the return is already payload_json-shaped."""
    config = config or get_config()
    try:
        client = get_backend_client(config)
    except ValueError:
        client = None
    facts = _subject_facts(client, payload.get("subject_id"))

    sg = build_report_sg(config=config, llm=maybe_llm(config, "report"))
    state = sg.invoke({
        "scope": payload.get("scope") or "student_progress",
        "user_id": payload.get("user_id"),
        "subject_id": payload.get("subject_id"),
        "lecture_ids": payload.get("lecture_ids") or None,
        "language": facts.language or payload.get("language", "en"),
    })
    result = state.get("result", {})
    if state.get("visual_spec"):
        result = {**result, "visual_spec": state["visual_spec"]}
    return result
