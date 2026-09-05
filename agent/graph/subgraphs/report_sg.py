"""report_sg — student progress report (deterministic analysis, narrated).

    fetch_metrics -> root_cause -> narrate

Private state (``ReportState``); the parent adapter (chat ``report`` task) or
``run_report`` (endpoint #10) hands it a scope + ``user_id``. The ANALYSIS is
deterministic — mastery and trend are DB reads, and root cause is a prerequisite graph
walk (the deepest weak prerequisite = the cause, not the symptom). Only ``narrate`` uses
the LLM, and strictly on the fetched numbers: it never invents a topic, a score, or a
cause, and degrades to a friendly message on failure. Read-only — a report moves no
mastery and (for now) is not persisted to ``ai.reports``.

Only ``student_progress`` is built. ``class_analytics`` / ``at_risk`` / ``parent`` /
``homework_batch`` are shelled — they need classroom rosters / parent linkage (Sprint 3)
and the school-scoping decision before they can aggregate beyond one student.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, TypedDict

from langchain_core.messages import HumanMessage

from ...prompts import REPORT_STUDENT_PROMPT
from ...tools import get_mastery, get_mastery_trend, get_topic_deps, get_topic_labels
from ...utils import degraded_message

logger = logging.getLogger(__name__)

_SUPPORTED = {"student_progress"}
# Scopes that need classroom/parent scope (Sprint 3) + the school-scoping PO decision.
# TODO (Sprint 3): build class_analytics / at_risk / parent / homework_batch — each needs a
# classroom roster or parent->student linkage and the get_class_analytics / get_at_risk tools.
_SHELLED = {"class_analytics", "at_risk", "parent", "homework_batch"}

# No-weak-topics and no-data notes (the LLM is only used when there IS something to explain).
_ALL_GOOD = {
    "en": "You're on track — no weak spots are showing up right now. Keep it up.",
    "ar": "أداؤك جيد — لا توجد نقاط ضعف واضحة حاليًا. واصل التقدّم.",
    "fr": "Tu es sur la bonne voie — aucun point faible pour l'instant. Continue.",
    "de": "Du bist auf einem guten Weg — derzeit keine Schwachstellen. Weiter so.",
}
_NO_DATA = {
    "en": "There isn't enough graded work yet to report on your progress.",
    "ar": "لا يوجد قدر كافٍ من الأعمال المصحّحة بعد لإعداد تقرير عن تقدّمك.",
    "fr": "Il n'y a pas encore assez de travail noté pour faire le point sur tes progrès.",
    "de": "Es gibt noch nicht genug bewertete Arbeit, um deinen Fortschritt zu berichten.",
}


class ReportState(TypedDict, total=False):
    # inputs
    scope: str                          # student_progress (others shelled)
    user_id: Optional[str]
    subject_id: Optional[str]
    lecture_ids: Optional[list[str]]    # optional lecture scope; first is used
    language: str
    # working
    weak_topics: list[dict[str, Any]]   # weakest-first, label-hydrated, capped
    weak_ids: list[str]                 # ALL weak topic ids (root-cause seeds)
    scores: dict[str, float]            # topic_id -> mastery score
    trend: dict[str, Any]               # get_mastery_trend output
    root_cause: Optional[dict[str, Any]]
    done: bool                          # shelled scope -> skip straight to the result
    # produced
    result: dict[str, Any]              # payload_json-shaped report
    visual_spec: dict[str, Any]         # trend chart (rendered by the parent)


def _shelled_result(scope: str) -> dict[str, Any]:
    return {"scope": scope, "available": False, "narrative": "", "weak_topics": [],
            "root_cause": None, "trend": {"direction": "insufficient", "count": 0},
            "degraded": False}


# --- nodes ------------------------------------------------------------------

def _build_fetch_metrics(config, weak_limit):
    def _node(state: ReportState) -> dict:
        scope = state.get("scope") or "student_progress"
        if scope not in _SUPPORTED:
            logger.info("report scope %r not available yet (shelled)", scope)
            return {"result": _shelled_result(scope), "done": True}
        user_id = state.get("user_id")
        subject_id = state.get("subject_id")
        lecture_id = (state.get("lecture_ids") or [None])[0]
        rows: list[dict] = []
        trend: dict = {"direction": "insufficient", "count": 0, "events": []}
        if user_id:
            try:
                rows = get_mastery(user_id, subject_id=subject_id,
                                   lecture_id=lecture_id, config=config)
            except Exception:
                logger.error("report get_mastery failed", exc_info=True)
            try:
                trend = get_mastery_trend(user_id, subject_id=subject_id, config=config)
            except Exception:
                logger.error("report get_mastery_trend failed", exc_info=True)
        scores = {r["topic_id"]: (r.get("score") or 0.0) for r in rows}
        weak = sorted((r for r in rows if r.get("status") == "weak"),
                      key=lambda r: r.get("score") or 0.0)
        weak_ids = [r["topic_id"] for r in weak]
        labels = get_topic_labels(weak_ids, config=config) if weak_ids else {}
        weak_topics = [{
            "topic_id": r["topic_id"], "label": labels.get(r["topic_id"], ""),
            "score": r.get("score"), "status": r.get("status"),
            "attempts": r.get("attempts"),
        } for r in weak[:weak_limit]]
        logger.info("report fetch_metrics", extra={
            "topics": len(rows), "weak": len(weak_ids), "trend": trend.get("direction")})
        return {"weak_topics": weak_topics, "weak_ids": weak_ids, "scores": scores,
                "trend": trend}

    return _node


def _build_root_cause(config, depth):
    def _node(state: ReportState) -> dict:
        weak_ids = state.get("weak_ids") or []
        if not weak_ids:
            return {"root_cause": None}
        weak_set = set(weak_ids)
        scores = state.get("scores") or {}
        try:
            prereqs = get_topic_deps(weak_ids, depth=depth, config=config)
        except Exception:
            logger.error("report get_topic_deps failed", exc_info=True)
            return {"root_cause": None}
        # A prerequisite that is ITSELF weak is a candidate cause; the deepest one (furthest
        # back in the chain) is the root, tie-broken by the lowest mastery score.
        weak_prereqs = [p for p in prereqs if p.topic_id in weak_set]
        if not weak_prereqs:
            return {"root_cause": None}
        root = max(weak_prereqs, key=lambda p: (p.depth, -(scores.get(p.topic_id, 0.0))))
        logger.info("report root_cause", extra={
            "candidates": len(weak_prereqs), "depth": root.depth})
        return {"root_cause": {"topic_id": root.topic_id, "label": root.label,
                               "depth": root.depth, "score": scores.get(root.topic_id)}}

    return _node


def _describe(score: Optional[float]) -> str:
    """A word for a score — the model is told not to quote raw numbers."""
    s = score or 0.0
    return "very weak" if s < 0.3 else "shaky" if s < 0.45 else "developing"


def _weak_block(weak_topics: list[dict]) -> str:
    return "\n".join(
        f"- {w.get('label') or w['topic_id']} — {_describe(w.get('score'))}, "
        f"{w.get('attempts') or 0} attempts"
        for w in weak_topics) or "- (none)"


def _trend_chart(trend: dict, language: str) -> Optional[dict]:
    """A line chart of mastery score over recent events, or None when too few points.
    Data-tied — the values are the real graded scores, never invented."""
    events = trend.get("events") or []
    if len(events) < 2:
        return None
    values = [round(float(e.get("score") or 0.0), 3) for e in events]
    title = "تقدّم الإتقان" if language == "ar" else "Mastery over time"
    return {
        "kind": "chart",
        "description": title,
        "params": {"type": "line", "title": title,
                   "x_label": "", "y_label": "score",
                   "labels": [str(i + 1) for i in range(len(values))],
                   "series": [{"name": "mastery", "values": values}]},
    }


def _build_narrate(llm, config):
    def _node(state: ReportState) -> dict:
        language = state.get("language", "en")
        weak_topics = state.get("weak_topics") or []
        trend = state.get("trend") or {}
        root_cause = state.get("root_cause")
        degraded = False

        if not state.get("weak_ids") and not (trend.get("events")):
            narrative = _NO_DATA.get(language, _NO_DATA["en"])
        elif not weak_topics:
            narrative = _ALL_GOOD.get(language, _ALL_GOOD["en"])
        elif llm is None:                       # dev/test: no key -> deterministic stub
            focus = (root_cause or {}).get("label") or weak_topics[0].get("label") or ""
            narrative = f"Focus next on: {focus}." if focus else _ALL_GOOD["en"]
        else:
            prompt_text = REPORT_STUDENT_PROMPT.format(
                language=language,
                weak_block=_weak_block(weak_topics),
                root_cause=(root_cause or {}).get("label") or "none",
                trend=trend.get("direction", "insufficient"))
            try:
                narrative = (llm.invoke([HumanMessage(content=prompt_text)]).content or "").strip()
                if not narrative:
                    raise ValueError("empty narrative")
            except Exception:
                logger.error("report narrate failed; degrading", exc_info=True)
                narrative = degraded_message(language)
                degraded = True

        result = {
            "scope": "student_progress", "available": True, "narrative": narrative,
            "weak_topics": weak_topics, "root_cause": root_cause,
            "trend": {"direction": trend.get("direction", "insufficient"),
                      "count": trend.get("count", 0)},
            "degraded": degraded,
        }
        out: dict[str, Any] = {"result": result}
        chart = _trend_chart(trend, language)
        if chart:
            out["visual_spec"] = chart
        logger.info("report narrate", extra={
            "weak": len(weak_topics), "root_cause": bool(root_cause), "degraded": degraded})
        return out

    return _node


# --- build ------------------------------------------------------------------

def build_report_sg(*, config, llm=None):
    from langgraph.graph import END, START, StateGraph

    weak_limit = config.get("report.weak_topic_limit", 5)
    depth = config.get("report.prereq_depth", config.get("agent.prereq_depth", 2))

    graph = StateGraph(ReportState)
    graph.add_node("fetch_metrics", _build_fetch_metrics(config, weak_limit))
    graph.add_node("root_cause", _build_root_cause(config, depth))
    graph.add_node("narrate", _build_narrate(llm, config))

    graph.add_edge(START, "fetch_metrics")
    # A shelled scope short-circuits: the result is already set, skip the student-only work.
    graph.add_conditional_edges(
        "fetch_metrics", lambda s: "done" if s.get("done") else "continue",
        {"done": END, "continue": "root_cause"})
    graph.add_edge("root_cause", "narrate")
    graph.add_edge("narrate", END)
    return graph.compile()
