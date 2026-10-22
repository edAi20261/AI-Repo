"""Reporting entrypoint (endpoint path).

The backend's ``GET /ai/reports/{user_id}`` calls ``run_report`` with a scope + the
student's ``user_id``. report_sg reads mastery, walks prerequisites for the root
cause, and narrates; it moves no mastery. The finished report is stored in
``ai.reports`` so a homepage can serve it without another model call. This is the
structured path (returns the full payload); the chat 'how am
I doing' turn uses the report_adapter, which surfaces only the narrative + trend chart.

``student_progress``, ``parent`` and ``teacher`` are served; ``teacher`` is about one
subject, so it needs ``subject_id``. Other scopes return an ``available: False`` shell.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from shared.llm import maybe_llm

from ..config import Config, get_config
from ..class_analytics import class_analytics
from ..prompts import REPORT_CLASS_PROMPT
from ..tools import save_report
from .subgraphs import build_report_sg
from .subgraphs.report_sg import class_evidence, narrate_claims

logger = logging.getLogger(__name__)

_STORED_SCOPES = frozenset({"student_progress", "parent"})
SUBJECT_SCOPES = frozenset({"teacher"})


def stored_scopes(config: Config) -> frozenset[str]:
    """The student scopes ``ai.reports`` can hold. ``teacher`` waits for the backend's
    migration adding it to ``ai.report_scope``: until then an insert would fail."""
    if config.get("report.teacher_scope_stored"):
        return _STORED_SCOPES | {"teacher"}
    return _STORED_SCOPES


def run_report(payload: dict[str, Any], *, config: Optional[Config] = None) -> dict:
    """Build a progress report. Returns the ``payload_json``-shaped dict
    ``{scope, available, narrative, weak_topics[], root_cause, trend, degraded}`` plus a
    ``visual_spec`` (trend chart) when there are enough points.

    Payload: ``user_id``, ``scope?`` (default ``student_progress``), ``subject_id``
    (required for ``teacher``, else optional), ``lecture_ids?``, ``language?``.

    A degraded report is returned but not stored: it would become the latest one and
    shadow a good report. ``persisted`` says whether this one was stored."""
    config = config or get_config()
    scope = payload.get("scope") or "student_progress"
    if scope in SUBJECT_SCOPES and not payload.get("subject_id"):
        raise ValueError(f"a {scope} report is about one subject: subject_id is required.")
    sg = build_report_sg(config=config, llm=maybe_llm(config, "report"))
    state = sg.invoke({
        "scope": scope,
        "user_id": payload.get("user_id"),
        "subject_id": payload.get("subject_id"),
        "lecture_ids": payload.get("lecture_ids") or None,
        "language": _language(payload, config),
    })
    result = dict(state.get("result", {}))
    result.update(_store(result, payload.get("user_id"), config))
    if state.get("visual_spec"):
        result["visual_spec"] = state["visual_spec"]
    return result


def _language(payload: dict[str, Any], config: Config) -> str:
    # The router passes the key whether or not the caller sent one, so a .get
    # default never applied and None reached the prompt.
    return payload.get("language") or config.get("language.default")


def _store(result: dict[str, Any], user_id: Optional[str], config: Config) -> dict[str, Any]:
    scope = result.get("scope")
    if not user_id or scope not in stored_scopes(config) or result.get("degraded"):
        return {"persisted": False}
    try:
        return {**save_report(scope, dict(result), user_id=user_id, config=config),
                "persisted": True}
    except Exception:
        logger.error("report not stored", exc_info=True, extra={"scope": scope})
        return {"persisted": False}


def run_class_report(payload: dict[str, Any], *, config: Optional[Config] = None) -> dict:
    """A teacher's written summary of one class, from ``class_analytics``, stored
    under the class. Payload: ``class_id``, ``user_ids`` (the roster), ``subject_ids?``,
    ``language?``. Returns ``{scope, available, narrative, claims, analysis, degraded,
    persisted, report_id?}``."""
    config = config or get_config()
    analysis = class_analytics(payload["user_ids"], subject_ids=payload.get("subject_ids"),
                               config=config)
    narrative, claims = "", []
    degraded = bool(analysis.get("degraded"))
    llm = maybe_llm(config, "report")
    if llm is not None and not degraded:
        evidence = class_evidence(analysis,
                                  topic_limit=config.get("report.weak_topic_limit", 5))
        try:
            narrative, claims = narrate_claims(llm, REPORT_CLASS_PROMPT, evidence,
                                               language=_language(payload, config),
                                               config=config)
        except Exception:
            logger.error("class report narrate failed; degrading", exc_info=True)
            degraded = True
    result = {"scope": "class_analytics", "available": bool(analysis.get("available")),
              "narrative": narrative, "claims": claims, "analysis": analysis,
              "degraded": degraded}
    result.update(_store_class(result, payload.get("class_id"), config))
    return result


def _store_class(result: dict[str, Any], class_id: Optional[str], config: Config) -> dict:
    if not class_id or result.get("degraded"):
        return {"persisted": False}
    try:
        return {**save_report("class_analytics", dict(result), class_id=class_id,
                              config=config), "persisted": True}
    except Exception:
        logger.error("class report not stored", exc_info=True)
        return {"persisted": False}

