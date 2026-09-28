"""report_sg — a student's progress note (deterministic analysis, narrated).

    fetch_metrics -> root_cause -> narrate

Private state (``ReportState``); the parent adapter (chat ``report`` task) or
``run_report`` hands it a scope + ``user_id``. The ANALYSIS is deterministic —
mastery and trend are DB reads, and root cause is a prerequisite graph walk (the
deepest weak prerequisite = the cause, not the symptom). Only ``narrate`` uses the
LLM: it is given numbered evidence and every claim it writes must cite it, or the
claim is sent back. A report moves no mastery.

``student_progress`` addresses the student and ``parent`` the student's parent, from
the same evidence. A class summary is built from ``class_analytics`` by
``run_class_report``; ``at_risk`` and ``homework_batch`` stay shelled here.
"""

from __future__ import annotations

import logging
from typing import Any, TypedDict

from langchain_core.messages import HumanMessage

from shared.structured import call_structured, schema_hint

from ...models import ReportNarrative
from ...prompts import REPORT_PARENT_PROMPT, REPORT_STUDENT_PROMPT
from ...tools import get_mastery, get_mastery_trend, get_topic_deps, get_topic_labels

logger = logging.getLogger(__name__)

_SUPPORTED = {"student_progress", "parent"}
_PROMPTS = {"student_progress": REPORT_STUDENT_PROMPT, "parent": REPORT_PARENT_PROMPT}
# A class summary is run_class_report's (it needs the roster the caller holds); at-risk
# students are part of that analysis rather than a report of their own.
_SHELLED = {"class_analytics", "at_risk", "homework_batch"}


class ReportState(TypedDict, total=False):
    # inputs
    scope: str                          # student_progress (others shelled)
    user_id: str | None
    subject_id: str | None
    lecture_ids: list[str] | None    # optional lecture scope; first is used
    language: str
    # working
    weak_topics: list[dict[str, Any]]   # weakest-first, label-hydrated, capped
    strengths: list[dict[str, Any]]
    weak_ids: list[str]                 # ALL weak topic ids (root-cause seeds)
    scores: dict[str, float]            # topic_id -> mastery score
    trend: dict[str, Any]               # get_mastery_trend output
    root_cause: dict[str, Any] | None
    evidence: bool
    read_failed: bool
    done: bool                          # shelled scope -> skip straight to the result
    # produced
    result: dict[str, Any]              # payload_json-shaped report
    visual_spec: dict[str, Any]         # trend chart (rendered by the parent)


def _shelled_result(scope: str) -> dict[str, Any]:
    return {"scope": scope, "available": False, "narrative": "", "weak_topics": [],
            "root_cause": None, "trend": {"direction": "insufficient", "count": 0},
            "degraded": False}


# --- nodes ------------------------------------------------------------------

def _build_fetch_metrics(config, weak_limit, strength_limit):
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
        read_failed = False
        if user_id:
            try:
                rows = get_mastery(user_id, subject_id=subject_id,
                                   lecture_id=lecture_id, config=config)
            except Exception:
                read_failed = True
                logger.error("report get_mastery failed", exc_info=True)
            try:
                trend = get_mastery_trend(user_id, subject_id=subject_id,
                                          lecture_id=lecture_id, config=config)
            except Exception:
                read_failed = True
                logger.error("report get_mastery_trend failed", exc_info=True)
        scores = {r["topic_id"]: (r.get("score") or 0.0) for r in rows}
        weak = sorted((r for r in rows if r.get("status") == "weak"),
                      key=lambda r: r.get("score") or 0.0)
        weak_ids = [r["topic_id"] for r in weak]
        strong = sorted((r for r in rows if r.get("status") == "strong"),
                        key=lambda r: -(r.get("score") or 0.0))[:strength_limit]
        labelled = weak_ids[:weak_limit] + [r["topic_id"] for r in strong]
        labels: dict[str, str] = {}
        if labelled:
            try:
                labels = get_topic_labels(labelled, config=config)
            except Exception:
                read_failed = True
                logger.error("report get_topic_labels failed", exc_info=True)
        weak_topics = [{
            "topic_id": r["topic_id"], "label": labels.get(r["topic_id"], ""),
            "score": r.get("score"), "status": r.get("status"),
            "attempts": r.get("attempts"),
        } for r in weak[:weak_limit]]
        strengths = [{
            "topic_id": r["topic_id"], "label": labels.get(r["topic_id"], ""),
            "score": r.get("score"), "status": r.get("status"),
            "attempts": r.get("attempts"),
        } for r in strong]
        logger.info("report fetch_metrics", extra={
            "topics": len(rows), "weak": len(weak_ids), "trend": trend.get("direction")})
        return {"weak_topics": weak_topics, "strengths": strengths,
                "weak_ids": weak_ids, "scores": scores,
                "trend": trend, "read_failed": read_failed,
                "evidence": any(r.get("status") not in (None, "unknown") for r in rows)}

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


def _describe(score: float | None) -> str:
    """A word for a score — the model is told not to quote raw numbers."""
    s = score or 0.0
    return "very weak" if s < 0.3 else "shaky" if s < 0.45 else "developing"


_TREND_WORDS = {"improving": "improving", "flat": "holding steady", "declining": "slipping"}


def _evidence(weak_topics: list[dict], strengths: list[dict], root_cause: dict | None,
              trend: dict, has_rows: bool) -> list[dict[str, Any]]:
    """Numbered facts a note may cite: ``{id, kind, topic_id, line}``."""
    items: list[dict[str, Any]] = []

    def add(kind: str, line: str, topic_id: str | None = None) -> None:
        items.append({"id": f"E{len(items) + 1}", "kind": kind, "topic_id": topic_id,
                      "line": line})

    for w in weak_topics:
        add("weak_topic", f"{w.get('label') or w['topic_id']} — weak "
            f"({_describe(w.get('score'))}), {w.get('attempts') or 0} assessments",
            w["topic_id"])
    for st in strengths:
        add("strength", f"{st.get('label') or st['topic_id']} — strong, "
            f"{st.get('attempts') or 0} assessments", st["topic_id"])
    if root_cause:
        add("root_cause", f"{root_cause.get('label') or root_cause['topic_id']} — a weak "
            "prerequisite underneath the weak topics (likely root cause)",
            root_cause["topic_id"])
    direction = trend.get("direction")
    if direction in _TREND_WORDS:
        add("trend", f"recent results are {_TREND_WORDS[direction]} "
            f"over {trend.get('count', 0)} answers")
    if not items:
        add("no_evidence", "too little graded practice to judge yet" if has_rows
            else "no graded practice yet")
    return items


def _check_citations(evidence_ids: set[str]):
    def _validate(narrative: ReportNarrative) -> None:
        bad = [(i + 1, ref) for i, c in enumerate(narrative.claims)
               for ref in c.evidence if ref not in evidence_ids]
        if bad:
            listed = ", ".join(sorted(evidence_ids))
            raise ValueError("; ".join(
                f"claim {n} cites {ref}, which is not in the evidence (use only {listed})"
                for n, ref in bad))
    return _validate


def narrate_claims(llm, prompt, evidence: list[dict[str, Any]], *, language: str,
                   config) -> tuple[str, list[dict[str, Any]]]:
    """A note as claims that each cite the evidence they rest on. Raises when the model
    cannot produce one that cites only what it was given."""
    by_id = {e["id"]: e for e in evidence}
    prompt_text = prompt.render(
        language=language,
        evidence_block="\n".join(f"{e['id']}: {e['line']}" for e in evidence),
        schema=schema_hint(ReportNarrative))
    note = call_structured(
        ReportNarrative, [HumanMessage(content=prompt_text)], task="report",
        config=config, llm=llm, label="report", validate=_check_citations(set(by_id)))
    claims = [{"text": c.text.strip(), "evidence": [
        {"id": ref, "kind": by_id[ref]["kind"], "topic_id": by_id[ref]["topic_id"]}
        for ref in c.evidence]} for c in note.claims]
    return " ".join(c["text"] for c in claims), claims


def class_evidence(analysis: dict[str, Any], *, topic_limit: int) -> list[dict[str, Any]]:
    """Numbered facts about a class, from ``class_analytics``, for a class note to cite."""
    items: list[dict[str, Any]] = []
    total = analysis.get("students", {}).get("total", 0)

    def add(kind: str, line: str, topic_id: str | None = None) -> None:
        items.append({"id": f"E{len(items) + 1}", "kind": kind, "topic_id": topic_id,
                      "line": line})

    for t in [t for t in analysis.get("topics", []) if t.get("weak")][:topic_limit]:
        add("class_topic", f"{t.get('label') or t['topic_id']} — {t['weak']} of {total} "
            f"students weak, {t.get('strong', 0)} strong, "
            f"{t.get('not_practised', 0)} not started", t["topic_id"])
    at_risk = analysis.get("at_risk", [])
    codes = [r["code"] for a in at_risk for r in a.get("reasons", [])]
    add("at_risk", f"{len(at_risk)} of {total} students need attention "
        f"({codes.count('declining')} declining, {codes.count('inactive')} inactive, "
        f"{codes.count('many_weak_topics')} weak on many topics)")
    for m in analysis.get("mistakes", [])[:1]:
        add("mistake", f"{m.get('label')} — most common mistake, {m.get('occurrences')} "
            f"times across {m.get('students')} students")
    if analysis.get("not_started"):
        add("not_started", f"{len(analysis['not_started'])} of {total} students have "
            "no graded work yet")
    weeks = analysis.get("activity", [])
    if len(weeks) >= 2:
        add("activity", f"graded answers: {weeks[-1]['answers']} in the latest week, "
            f"{weeks[-2]['answers']} the week before")
    return items


def _trend_chart(trend: dict) -> dict | None:
    """A line chart of mastery score over recent events, or None when too few points.
    Data-tied — the values are the real graded scores, never invented."""
    events = trend.get("events") or []
    if len(events) < 2:
        return None
    values = [round(float(e.get("score") or 0.0), 3) for e in events]
    return {
        "kind": "chart",
        "description": "",
        "params": {"type": "line", "title": "",
                   "x_label": "", "y_label": "",
                   "labels": [str(i + 1) for i in range(len(values))],
                   "series": [{"name": "mastery", "values": values}]},
    }


def _build_narrate(llm, config):
    def _node(state: ReportState) -> dict:
        language = state.get("language", "en")
        scope = state.get("scope") or "student_progress"
        weak_topics = state.get("weak_topics") or []
        strengths = state.get("strengths") or []
        trend = state.get("trend") or {}
        root_cause = state.get("root_cause")
        degraded = bool(state.get("read_failed"))
        claims: list[dict[str, Any]] = []

        if llm is None:
            focus = (root_cause or {}).get("label") or (
                weak_topics[0].get("label") if weak_topics else "")
            narrative = f"🎯 {focus}" if focus else ("✓" if trend.get("events") else "")
        else:
            evidence = _evidence(weak_topics, strengths, root_cause, trend,
                                 bool(state.get("scores")))
            try:
                narrative, claims = narrate_claims(llm, _PROMPTS[scope], evidence,
                                                   language=language, config=config)
            except Exception:
                logger.error("report narrate failed; degrading", exc_info=True)
                narrative = ""
                degraded = True

        result = {
            "scope": scope, "available": bool(state.get("evidence")),
            "narrative": narrative, "claims": claims,
            "weak_topics": weak_topics, "strengths": strengths, "root_cause": root_cause,
            "trend": {"direction": trend.get("direction", "insufficient"),
                      "count": trend.get("count", 0)},
            "degraded": degraded,
        }
        out: dict[str, Any] = {"result": result}
        chart = _trend_chart(trend)
        if chart:
            out["visual_spec"] = chart
        logger.info("report narrate", extra={
            "weak": len(weak_topics), "strengths": len(strengths), "claims": len(claims),
            "root_cause": bool(root_cause), "degraded": degraded})
        return out

    return _node


def build_report_sg(*, config, llm=None):
    from langgraph.graph import END, START, StateGraph

    weak_limit = config.get("report.weak_topic_limit", 5)
    strength_limit = config.get("report.strength_limit", 3)
    depth = config.get("report.prereq_depth", config.get("agent.prereq_depth", 2))

    graph = StateGraph(ReportState)
    graph.add_node("fetch_metrics", _build_fetch_metrics(config, weak_limit, strength_limit))
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
