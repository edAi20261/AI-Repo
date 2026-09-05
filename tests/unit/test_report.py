"""report_sg — student progress report: deterministic root-cause (deepest weak
prerequisite), trend chart emission, and the shelled non-student scopes. No DB, no LLM."""

from __future__ import annotations

from agent.config import get_config
from agent.graph.subgraphs import build_report_sg
from agent.graph.subgraphs import report_sg as rmod
from agent.models import PrereqTopic


def _patch(monkeypatch, *, mastery, trend, deps, labels):
    monkeypatch.setattr(rmod, "get_mastery", lambda uid, **k: mastery)
    monkeypatch.setattr(rmod, "get_mastery_trend", lambda uid, **k: trend)
    monkeypatch.setattr(rmod, "get_topic_deps", lambda ids, **k: deps)
    monkeypatch.setattr(rmod, "get_topic_labels", lambda ids, **k: labels)


def test_root_cause_is_deepest_weak_prerequisite(monkeypatch):
    # tBuffer (weak) depends on tAcid (weak, depth 1) which depends on tEquilibrium
    # (weak, depth 2). The report should name the deepest weak prereq, not the symptom.
    mastery = [
        {"topic_id": "tBuffer", "score": 0.4, "attempts": 3, "status": "weak"},
        {"topic_id": "tAcid", "score": 0.35, "attempts": 3, "status": "weak"},
        {"topic_id": "tEquilibrium", "score": 0.3, "attempts": 3, "status": "weak"},
        {"topic_id": "tStrong", "score": 0.9, "attempts": 5, "status": "strong"},
    ]
    deps = [
        PrereqTopic(topic_id="tAcid", label="Acids", strength="required", depth=1),
        PrereqTopic(topic_id="tEquilibrium", label="Equilibrium", strength="required", depth=2),
    ]
    labels = {"tBuffer": "Buffers", "tAcid": "Acids", "tEquilibrium": "Equilibrium"}
    trend = {"direction": "declining", "count": 3,
             "events": [{"score": 0.6}, {"score": 0.4}, {"score": 0.3}]}
    _patch(monkeypatch, mastery=mastery, trend=trend, deps=deps, labels=labels)

    sg = build_report_sg(config=get_config(), llm=None)
    state = sg.invoke({"scope": "student_progress", "user_id": "u1", "language": "en"})
    result = state["result"]

    assert result["available"] is True
    assert result["root_cause"]["topic_id"] == "tEquilibrium"   # deepest weak, not tBuffer
    assert result["trend"]["direction"] == "declining"
    assert "Equilibrium" in result["narrative"]                 # stub focuses on the root cause
    # A trend with >= 2 points emits a data-tied line chart for the parent to render.
    assert state["visual_spec"]["kind"] == "chart"
    assert state["visual_spec"]["params"]["series"][0]["values"] == [0.6, 0.4, 0.3]


def test_no_weak_topics_reports_all_good(monkeypatch):
    mastery = [{"topic_id": "t1", "score": 0.9, "attempts": 5, "status": "strong"}]
    _patch(monkeypatch, mastery=mastery,
           trend={"direction": "improving", "count": 5, "events": [{"score": 0.8}]},
           deps=[], labels={})
    sg = build_report_sg(config=get_config(), llm=None)
    result = sg.invoke({"scope": "student_progress", "user_id": "u1", "language": "en"})["result"]
    assert result["available"] is True
    assert result["root_cause"] is None
    assert result["weak_topics"] == []
    assert result["narrative"]                                   # a positive, localized note


def test_shelled_scope_is_unavailable(monkeypatch):
    # class_analytics needs classroom scoping (Sprint 3) -> shelled, no DB touched.
    sg = build_report_sg(config=get_config(), llm=None)
    result = sg.invoke({"scope": "class_analytics", "user_id": "u1"})["result"]
    assert result["available"] is False
    assert result["weak_topics"] == []
