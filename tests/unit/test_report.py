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


class _Writer:
    """A plain chat client: its reply is JSON text, validated like any structured call."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.prompts: list[str] = []

    def invoke(self, messages):
        import json

        last = messages[-1]
        self.prompts.append(last["content"] if isinstance(last, dict) else last.content)
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]

        class _Msg:
            content = json.dumps(reply)
        return _Msg()


def test_a_student_with_no_practice_yet_is_told_so_by_the_model(monkeypatch):
    """The no-data and all-good notes were fixed sentences; the model writes them
    now from the same numbers, so they read like the rest of the report."""
    _patch(monkeypatch, mastery=[], trend={"direction": "insufficient", "count": 0,
                                           "events": []}, deps=[], labels={})
    writer = _Writer({"claims": [{"text": "There isn't enough practice yet to judge.",
                                  "evidence": ["E1"]}]})
    sg = build_report_sg(config=get_config(), llm=writer)
    result = sg.invoke({"scope": "student_progress", "user_id": "u1", "language": "en"})["result"]
    assert result["narrative"].startswith("There isn't enough practice yet")
    assert "E1: no graded practice yet" in writer.prompts[0]


def test_every_sentence_of_the_note_cites_the_evidence_it_rests_on(monkeypatch):
    """The note was free prose the model could fill with anything; a claim that names
    its evidence can be checked, shown next to the numbers, and rejected when it
    points at nothing."""
    _patch(monkeypatch,
           mastery=[{"topic_id": "tq", "score": 0.3, "attempts": 4, "status": "weak"},
                    {"topic_id": "tf", "score": 0.9, "attempts": 5, "status": "strong"}],
           trend={"direction": "improving", "count": 8, "events": []},
           deps=[], labels={"tq": "Quadratics", "tf": "Fractions"})
    writer = _Writer({"claims": [
        {"text": "Fractions are going well.", "evidence": ["E2"]},
        {"text": "Quadratics need more work.", "evidence": ["E1"]}]})
    sg = build_report_sg(config=get_config(), llm=writer)
    result = sg.invoke({"scope": "student_progress", "user_id": "u1", "language": "en"})["result"]

    assert result["narrative"] == "Fractions are going well. Quadratics need more work."
    assert result["claims"][0]["evidence"] == [
        {"id": "E2", "kind": "strength", "topic_id": "tf"}]
    assert result["strengths"][0]["label"] == "Fractions"
    assert "E1: Quadratics" in writer.prompts[0] and "E2: Fractions" in writer.prompts[0]


def test_a_claim_citing_evidence_that_was_not_given_is_sent_back(monkeypatch):
    _patch(monkeypatch,
           mastery=[{"topic_id": "tq", "score": 0.3, "attempts": 4, "status": "weak"}],
           trend={"direction": "flat", "count": 4, "events": []},
           deps=[], labels={"tq": "Quadratics"})
    writer = _Writer({"claims": [{"text": "Geometry is weak.", "evidence": ["E9"]}]},
                     {"claims": [{"text": "Quadratics need work.", "evidence": ["E1"]}]})
    sg = build_report_sg(config=get_config(), llm=writer)
    result = sg.invoke({"scope": "student_progress", "user_id": "u1", "language": "en"})["result"]

    assert result["narrative"] == "Quadratics need work."
    assert len(writer.prompts) == 2 and result["degraded"] is False


def test_a_note_that_never_cites_its_evidence_is_dropped_not_shown(monkeypatch):
    _patch(monkeypatch,
           mastery=[{"topic_id": "tq", "score": 0.3, "attempts": 4, "status": "weak"}],
           trend={"direction": "flat", "count": 4, "events": []},
           deps=[], labels={"tq": "Quadratics"})
    writer = _Writer({"claims": [{"text": "You are a genius.", "evidence": ["E7"]}]})
    sg = build_report_sg(config=get_config(), llm=writer)
    result = sg.invoke({"scope": "student_progress", "user_id": "u1", "language": "en"})["result"]

    assert result["narrative"] == "" and result["claims"] == []
    assert result["degraded"] is True


def test_a_report_with_no_graded_work_says_it_is_unavailable(monkeypatch):
    """``available`` was hard-coded True, so a dashboard could not tell a student with
    no graded work from one whose work was all fine, although the contract promises
    False until there is enough evidence."""
    _patch(monkeypatch,
           mastery=[{"topic_id": "t1", "score": 0.4, "attempts": 1, "status": "unknown"}],
           trend={"direction": "insufficient", "count": 1, "events": [{"score": 0.4}]},
           deps=[], labels={})
    sg = build_report_sg(config=get_config(), llm=None)
    result = sg.invoke({"scope": "student_progress", "user_id": "u1", "language": "en"})["result"]
    assert result["available"] is False


def test_an_unreadable_learner_model_is_not_reported_as_no_practice(monkeypatch):
    """A failed mastery read was logged and then read as an empty history, so the
    student was told they had not practised yet and nothing marked the report degraded."""
    def _unreachable(*a, **k):
        raise RuntimeError("database unreachable")

    _patch(monkeypatch, mastery=[], trend={"direction": "insufficient", "count": 0,
                                           "events": []}, deps=[], labels={})
    monkeypatch.setattr(rmod, "get_mastery", _unreachable)
    sg = build_report_sg(config=get_config(), llm=None)
    result = sg.invoke({"scope": "student_progress", "user_id": "u1", "language": "en"})["result"]
    assert result["degraded"] is True
    assert result["available"] is False


def test_a_failed_label_lookup_still_returns_the_report(monkeypatch):
    """get_topic_labels was the one unguarded read in fetch_metrics, so a label lookup
    failure turned the whole report into a 500."""
    def _unreachable(*a, **k):
        raise RuntimeError("database unreachable")

    _patch(monkeypatch,
           mastery=[{"topic_id": "t1", "score": 0.2, "attempts": 4, "status": "weak"}],
           trend={"direction": "flat", "count": 4, "events": []}, deps=[], labels={})
    monkeypatch.setattr(rmod, "get_topic_labels", _unreachable)
    sg = build_report_sg(config=get_config(), llm=None)
    result = sg.invoke({"scope": "student_progress", "user_id": "u1", "language": "en"})["result"]
    assert result["weak_topics"][0]["topic_id"] == "t1"
    assert result["degraded"] is True


def test_the_trend_is_scoped_to_the_same_lecture_as_the_weak_topics(monkeypatch):
    """Weak topics were read for the requested lecture but the trend for the whole
    subject, so one report described two different scopes."""
    seen = {}
    _patch(monkeypatch, mastery=[], trend={"direction": "insufficient", "count": 0,
                                           "events": []}, deps=[], labels={})
    monkeypatch.setattr(rmod, "get_mastery_trend",
                        lambda uid, **k: seen.update(k) or {"direction": "insufficient",
                                                            "count": 0, "events": []})
    sg = build_report_sg(config=get_config(), llm=None)
    sg.invoke({"scope": "student_progress", "user_id": "u1", "subject_id": "s1",
               "lecture_ids": ["lec7"], "language": "en"})
    assert seen.get("lecture_id") == "lec7"


def test_a_built_report_is_stored_so_a_homepage_need_not_rebuild_it(monkeypatch):
    """Every report was rebuilt from scratch, one model call per page view, and
    nothing kept what a student had been told."""
    from agent.graph import report as report_mod

    _patch(monkeypatch,
           mastery=[{"topic_id": "t1", "score": 0.2, "attempts": 4, "status": "weak"}],
           trend={"direction": "flat", "count": 4, "events": [{"score": 0.2},
                                                              {"score": 0.3}]},
           deps=[], labels={"t1": "Fractions"})
    saved = {}
    monkeypatch.setattr(report_mod, "save_report",
                        lambda scope, payload, **k: saved.update(scope=scope, payload=payload,
                                                                 **k)
                        or {"report_id": "r1", "generated_at": "2026-09-27T08:00:00+00:00"})
    monkeypatch.setattr(report_mod, "maybe_llm", lambda config, task: None)

    result = report_mod.run_report({"user_id": "u1"}, config=get_config())

    assert saved["scope"] == "student_progress" and saved["user_id"] == "u1"
    assert "visual_spec" not in saved["payload"]
    assert saved["payload"]["weak_topics"][0]["label"] == "Fractions"
    assert result["report_id"] == "r1" and result["persisted"] is True


def test_a_degraded_report_is_returned_but_never_stored(monkeypatch):
    """A degraded note would become the latest report and shadow a good one."""
    from agent.graph import report as report_mod

    def _unreachable(*a, **k):
        raise RuntimeError("database unreachable")

    _patch(monkeypatch, mastery=[], trend={"direction": "insufficient", "count": 0,
                                           "events": []}, deps=[], labels={})
    monkeypatch.setattr(rmod, "get_mastery", _unreachable)
    monkeypatch.setattr(report_mod, "save_report",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("stored")))
    monkeypatch.setattr(report_mod, "maybe_llm", lambda config, task: None)

    result = report_mod.run_report({"user_id": "u1"}, config=get_config())
    assert result["degraded"] is True and result["persisted"] is False


def test_a_failed_save_still_returns_the_report_and_says_so(monkeypatch):
    from agent.graph import report as report_mod

    def _fails(*a, **k):
        raise RuntimeError("write failed")

    _patch(monkeypatch, mastery=[], trend={"direction": "insufficient", "count": 0,
                                           "events": []}, deps=[], labels={})
    monkeypatch.setattr(report_mod, "save_report", _fails)
    monkeypatch.setattr(report_mod, "maybe_llm", lambda config, task: None)

    result = report_mod.run_report({"user_id": "u1"}, config=get_config())
    assert result["persisted"] is False and "report_id" not in result


def test_a_parent_report_speaks_to_the_parent_about_their_child(monkeypatch):
    """The parent scope returned an empty shell, so a parent saw nothing about how
    their child was doing even though the same evidence served the child's own report."""
    _patch(monkeypatch,
           mastery=[{"topic_id": "tq", "score": 0.3, "attempts": 4, "status": "weak"}],
           trend={"direction": "improving", "count": 6, "events": []},
           deps=[], labels={"tq": "Quadratics"})
    writer = _Writer({"claims": [{"text": "Your child is working on Quadratics.",
                                  "evidence": ["E1"]}]})
    sg = build_report_sg(config=get_config(), llm=writer)
    result = sg.invoke({"scope": "parent", "user_id": "u1", "language": "en"})["result"]

    assert result["scope"] == "parent" and result["available"] is True
    assert result["narrative"] == "Your child is working on Quadratics."
    assert "parent" in writer.prompts[0].lower()


def test_a_class_report_is_narrated_from_the_class_analysis_and_stored(monkeypatch):
    """The class scope was a shell: a teacher had no written summary of their class."""
    from agent.graph import report as report_mod

    analysis = {
        "available": True, "degraded": False,
        "students": {"total": 3, "with_evidence": 3},
        "topics": [{"topic_id": "t1", "label": "Fractions", "strong": 1, "developing": 0,
                    "weak": 2, "unknown": 0, "not_practised": 0, "mean_score": 0.4}],
        "at_risk": [{"user_id": "s1", "reasons": [{"code": "declining"}]}],
        "not_started": [], "mistakes": [{"tag_id": "g", "label": "Sign error",
                                         "occurrences": 5, "students": 2}],
        "most_missed": [], "activity": []}
    monkeypatch.setattr(report_mod, "class_analytics", lambda ids, **k: analysis)
    saved = {}
    monkeypatch.setattr(report_mod, "save_report",
                        lambda scope, payload, **k: saved.update(scope=scope, **k)
                        or {"report_id": "r9", "generated_at": "2026-09-27T08:00:00+00:00"})
    writer = _Writer({"claims": [
        {"text": "Two students are struggling with Fractions.", "evidence": ["E1"]},
        {"text": "Sign errors are common.", "evidence": ["E3"]}]})
    monkeypatch.setattr(report_mod, "maybe_llm", lambda config, task: writer)

    result = report_mod.run_class_report(
        {"class_id": "c1", "user_ids": ["s1", "s2", "s3"], "language": "en"},
        config=get_config())

    assert "E1: Fractions — 2 of 3 students weak" in writer.prompts[0]
    assert "E2: 1 of 3 students need attention" in writer.prompts[0]
    assert "E3: Sign error" in writer.prompts[0]
    assert "could be taught together" not in writer.prompts[0]
    assert result["narrative"].startswith("Two students")
    assert result["analysis"]["topics"][0]["topic_id"] == "t1"
    assert (saved["scope"], saved["class_id"]) == ("class_analytics", "c1")
    assert result["report_id"] == "r9" and result["persisted"] is True


def test_a_class_report_on_an_unreadable_roster_is_degraded_and_not_stored(monkeypatch):
    from agent.graph import report as report_mod

    monkeypatch.setattr(report_mod, "class_analytics",
                        lambda ids, **k: {"available": False, "degraded": True,
                                          "students": {"total": 2, "with_evidence": 0},
                                          "topics": [], "at_risk": [], "not_started": [],
                                          "mistakes": [], "most_missed": [], "activity": []})
    monkeypatch.setattr(report_mod, "save_report",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("stored")))
    monkeypatch.setattr(report_mod, "maybe_llm", lambda config, task: None)

    result = report_mod.run_class_report({"class_id": "c1", "user_ids": ["s1", "s2"]},
                                         config=get_config())
    assert result["degraded"] is True and result["persisted"] is False


def test_a_report_asked_for_without_a_language_is_still_written(monkeypatch):
    """GET /ai/reports/{id} without ?language= passes language=None, and
    payload.get("language", "en") kept the None. The prompt refused to render, every
    such report came back available with an empty narrative and was never stored —
    in production, every student and parent report the platform asked for."""
    from agent.graph import report as report_mod

    _patch(monkeypatch,
           mastery=[{"topic_id": "t1", "score": 0.2, "attempts": 4, "status": "weak"}],
           trend={"direction": "flat", "count": 4, "events": []},
           deps=[], labels={"t1": "Fractions"})
    monkeypatch.setattr(report_mod, "save_report",
                        lambda scope, payload, **k: {"report_id": "r1",
                                                     "generated_at": "2026-10-04T15:00:00+00:00"})
    writer = _Writer({"claims": [{"text": "Fractions need more practice.", "evidence": ["E1"]}]})
    monkeypatch.setattr(report_mod, "maybe_llm", lambda config, task: writer)

    for scope in ("student_progress", "parent"):
        result = report_mod.run_report(
            {"user_id": "u1", "scope": scope, "subject_id": None, "lecture_ids": None,
             "language": None}, config=get_config())
        assert result["narrative"] == "Fractions need more practice.", scope
        assert result["degraded"] is False and result["persisted"] is True


def test_a_class_report_asked_for_without_a_language_is_still_written(monkeypatch):
    from agent.graph import report as report_mod

    analysis = {"available": True, "degraded": False,
                "students": {"total": 2, "with_evidence": 2},
                "topics": [{"topic_id": "t1", "label": "Fractions", "strong": 0,
                            "developing": 0, "weak": 2, "unknown": 0, "not_practised": 0,
                            "mean_score": 0.3}],
                "at_risk": [], "not_started": [], "mistakes": [], "most_missed": [],
                "activity": []}
    monkeypatch.setattr(report_mod, "class_analytics", lambda ids, **k: analysis)
    monkeypatch.setattr(report_mod, "save_report",
                        lambda scope, payload, **k: {"report_id": "r2",
                                                     "generated_at": "2026-10-04T15:00:00+00:00"})
    writer = _Writer({"claims": [{"text": "Both students find Fractions hard.",
                                  "evidence": ["E1"]}]})
    monkeypatch.setattr(report_mod, "maybe_llm", lambda config, task: writer)

    result = report_mod.run_class_report(
        {"class_id": "c1", "user_ids": ["s1", "s2"], "language": None}, config=get_config())
    assert result["narrative"] == "Both students find Fractions hard."
    assert result["persisted"] is True


def test_a_teacher_report_speaks_about_the_student_in_one_subject(monkeypatch):
    """A teacher opening a student saw the student's own note, written to "you" and
    across every subject, with nothing saying which subject it was about."""
    seen = {}
    _patch(monkeypatch,
           mastery=[{"topic_id": "tq", "score": 0.3, "attempts": 4, "status": "weak"}],
           trend={"direction": "flat", "count": 4, "events": []},
           deps=[], labels={"tq": "Quadratics"})
    monkeypatch.setattr(rmod, "get_mastery",
                        lambda uid, **k: seen.update(k) or [
                            {"topic_id": "tq", "score": 0.3, "attempts": 4,
                             "status": "weak"}])
    writer = _Writer({"claims": [{"text": "The student needs work on Quadratics.",
                                  "evidence": ["E1"]}]})
    sg = build_report_sg(config=get_config(), llm=writer)
    result = sg.invoke({"scope": "teacher", "user_id": "u1", "subject_id": "math",
                        "language": "en"})["result"]

    assert result["scope"] == "teacher" and result["subject_id"] == "math"
    assert result["narrative"] == "The student needs work on Quadratics."
    assert seen["subject_id"] == "math"
    assert "teacher" in writer.prompts[0].lower()
    assert "third person" in writer.prompts[0]


def test_a_teacher_report_needs_its_subject(monkeypatch):
    import pytest

    from agent.graph import report as report_mod

    monkeypatch.setattr(report_mod, "save_report",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("stored")))
    with pytest.raises(ValueError):
        report_mod.run_report({"user_id": "u1", "scope": "teacher"}, config=get_config())


def _teacher_report(monkeypatch, config):
    from agent.graph import report as report_mod

    _patch(monkeypatch,
           mastery=[{"topic_id": "t1", "score": 0.2, "attempts": 4, "status": "weak"}],
           trend={"direction": "flat", "count": 4, "events": []},
           deps=[], labels={"t1": "Fractions"})
    saved = {}
    monkeypatch.setattr(report_mod, "save_report",
                        lambda scope, payload, **k: saved.update(scope=scope, payload=payload)
                        or {"report_id": "r7", "generated_at": "2026-10-22T08:00:00+00:00"})
    monkeypatch.setattr(report_mod, "maybe_llm", lambda config, task: None)
    result = report_mod.run_report({"user_id": "u1", "scope": "teacher", "subject_id": "math"},
                                   config=config)
    return result, saved


def test_a_teacher_report_is_not_stored_until_the_database_accepts_its_scope(monkeypatch):
    """ai.report_scope has no 'teacher' value yet, so an insert would fail; the report
    is built and returned, and says it was not stored."""
    result, saved = _teacher_report(monkeypatch, get_config())
    assert result["persisted"] is False and "report_id" not in result
    assert saved == {}


def test_a_teacher_report_is_stored_with_its_subject_once_the_scope_exists(monkeypatch):
    config = get_config()
    config.override("report.teacher_scope_stored", True)
    result, saved = _teacher_report(monkeypatch, config)
    assert result["persisted"] is True and result["report_id"] == "r7"
    assert saved["scope"] == "teacher" and saved["payload"]["subject_id"] == "math"
