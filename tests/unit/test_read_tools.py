"""Read/write tool tests. No DB, no backend: psycopg.connect is faked and
the backend client is monkeypatched. Asserts SQL targets and row mapping."""

from __future__ import annotations

import pytest

import agent.tools as tools
from tests.conftest import FakePostgres
from agent.config import get_config
from agent.mastery import Mastery
from agent.tools import (
    create_mistake_tag,
    get_figures,
    get_mastery,
    get_mastery_rollup,
    list_mistake_tags,
    update_mastery,
    write_mastery_events,
)


# --- get_figures ------------------------------------------------------------

def test_get_figures_requires_a_filter():
    with pytest.raises(ValueError):
        get_figures(config=get_config())


def test_get_figures_by_topic(fake_postgres):
    cur = FakePostgres(fetchall=[[
        ("f1", "lec1", "s3://b/f1.png", "cap", "alt", 2, ["t1"]),
    ]])
    fake_postgres(cur)
    out = get_figures(topic_ids=["t1"], config=get_config())
    assert out[0]["s3_uri"] == "s3://b/f1.png"
    assert "topic_ids && " in cur.sql[0]


# --- mistake_tags -----------------------------------------------------------

def test_list_mistake_tags(fake_postgres):
    fake_postgres(FakePostgres(fetchall=[[("tag1", "sign error", None)]]))
    out = list_mistake_tags(config=get_config())
    assert out == [{"tag_id": "tag1", "label": "sign error", "description": None}]


def test_create_mistake_tag_reuses_existing(fake_postgres):
    cur = FakePostgres(fetchone=[("existing-id",)])  # SELECT finds a match
    fake_postgres(cur)
    assert create_mistake_tag("Sign Error", config=get_config()) == "existing-id"
    assert len(cur.sql) == 1  # only the SELECT, no INSERT


def test_create_mistake_tag_inserts_when_absent(fake_postgres):
    cur = FakePostgres(fetchone=[None, ("new-id",)])  # SELECT miss, INSERT returns
    conn = fake_postgres(cur)
    assert create_mistake_tag("brand new", config=get_config()) == "new-id"
    assert any("INSERT INTO ai.mistake_tags" in s for s in cur.sql)
    assert conn.committed


def test_create_mistake_tag_rejects_blank():
    with pytest.raises(ValueError):
        create_mistake_tag("   ", config=get_config())


# --- mastery store ----------------------------------------------------------

def test_get_mastery_adds_status(fake_postgres):
    # attempts >= min_attempts (3) and score >= strong_threshold -> 'strong'
    fake_postgres(FakePostgres(fetchall=[[("t1", 0.9, 5, None, 0.1, 1.0)]]))
    out = get_mastery("u1", ["t1"], config=get_config())
    assert out[0]["topic_id"] == "t1"
    assert out[0]["status"] == "strong"


def test_get_mastery_scopes_by_subject(fake_postgres):
    cur = FakePostgres(fetchall=[[("t1", 0.9, 5, None, 0.1, 1.0)]])
    fake_postgres(cur)
    get_mastery("u1", subject_id="sub1", config=get_config())
    # subject scope joins ai.topics and filters on it
    assert "JOIN ai.topics" in cur.sql[0] and "t.subject_id = %s" in cur.sql[0]
    assert cur.last_args == ("u1", "sub1")


def test_get_mastery_rollup_weighted_mean_and_counts(fake_postgres):
    # subject sub1: two topics — strong (0.9, 5 attempts) + weak (0.2, 3 attempts).
    fake_postgres(FakePostgres(fetchall=[[
        ("sub1", 0.9, 5, 0.1, 1.0),
        ("sub1", 0.2, 3, -0.1, 1.0),
    ]]))
    out = get_mastery_rollup("u1", "subject", config=get_config())
    assert len(out) == 1
    row = out[0]
    assert row["subject_id"] == "sub1" and row["topics"] == 2
    assert row["attempts"] == 8                       # 5 + 3
    assert row["score"] == round((0.9 * 5 + 0.2 * 3) / 8, 4)   # attempts-weighted mean
    assert row["strong"] == 1 and row["weak"] == 1


def test_get_mastery_rollup_rejects_bad_group():
    with pytest.raises(ValueError):
        get_mastery_rollup("u1", "topic", config=get_config())


def test_write_mastery_events_inserts_and_commits(fake_postgres):
    cur = FakePostgres(fetchone=[("e1",), ("e2",)])
    conn = fake_postgres(cur)
    ids = write_mastery_events([
        {"user_id": "u1", "topic_id": "t1", "source": "practice", "correct": True},
        {"user_id": "u1", "topic_id": "t2", "source": "quiz", "correct": False},
    ], config=get_config())
    assert ids == ["e1", "e2"]
    assert conn.committed


def test_write_mastery_events_requires_topic_and_source(fake_postgres):
    fake_postgres(FakePostgres())
    with pytest.raises(ValueError):
        write_mastery_events([{"user_id": "u1", "correct": True}], config=get_config())


def test_update_mastery_upserts(fake_postgres):
    cur = FakePostgres()
    conn = fake_postgres(cur)
    update_mastery("u1", "t1", Mastery(score=0.7, attempts=4, trend=0.1,
                                       confidence=1.0), config=get_config())
    assert "ON CONFLICT (user_id, topic_id) DO UPDATE" in cur.sql[0]
    assert conn.committed


# --- backend-wrapping tools -------------------------------------------------

class _FakeBackend:
    def get_question_bank(self, subject_id, topic_ids=None, count=10, type_mix=None):
        return [{"id": "qb1", "subject_id": subject_id, "count": count}]

    def get_questions(self, kind, assessment_id):
        return [{
            "id": "q1", "question_type": "mcq", "correct_answer_text": "oxygen",
            "rubric": [{"criterion": "c", "points": 1}], "points": 1,
            "topic_ids": ["t1"], "lecture_id": "lec1", "question_text": "…",
        }]


def test_get_question_bank_wraps_client(monkeypatch):
    monkeypatch.setattr(tools, "get_backend_client", lambda config: _FakeBackend())
    out = tools.get_question_bank("subjA", count=3, config=get_config())
    assert out[0]["id"] == "qb1" and out[0]["count"] == 3


def test_fetch_rubric_projects_answer_key(monkeypatch):
    monkeypatch.setattr(tools, "get_backend_client", lambda config: _FakeBackend())
    rub = tools.fetch_rubric("quiz", "qz1", config=get_config())
    assert rub[0]["correct_answer_text"] == "oxygen"
    assert "question_text" not in rub[0]  # projected out
    full = tools.get_item("quiz", "qz1", config=get_config())
    assert "question_text" in full[0]


def test_graded_evidence_and_its_rollup_commit_together(fake_postgres):
    """Events and the mastery aggregate were committed separately, so a failure
    between them left events with no rollup, or a rollup with no evidence."""
    from agent.tools import record_graded_evidence

    cur = FakePostgres(fetchone=[("e1",)])
    conn = fake_postgres(cur)
    ids = record_graded_evidence(
        "u1",
        [{"user_id": "u1", "topic_id": "t1", "source": "quiz", "correct": True}],
        {"t1": Mastery(score=0.4, attempts=1, trend=0.4, confidence=0.5)},
        config=get_config())

    assert ids == ["e1"]
    assert "INSERT INTO ai.mastery_events" in cur.sql[0]
    assert "ON CONFLICT (user_id, topic_id) DO UPDATE" in cur.sql[1]
    assert conn.committed


def test_scope_topics_are_read_for_the_subjects_asked_and_those_with_evidence(fake_postgres):
    """A subject the student has answered in but the caller did not list still belongs
    in their analysis, and a listed subject with no answers yet must show its topics as
    not practised rather than disappear."""
    from agent.tools import get_scope_topics

    cur = FakePostgres(rows=[("t1", "Fractions", "l1", "math"), ("t2", "Cells", "l7", "bio")])
    fake_postgres(cur)
    rows = get_scope_topics("u1", subject_ids=["math"], config=get_config())

    sql, args = cur.statements[0]
    assert "t.subject_id = ANY(%s)" in sql and "m.user_id = ANY(%s)" in sql
    assert args == (["math"], ["u1"])
    assert rows[1] == {"topic_id": "t2", "label": "Cells", "lecture_id": "l7",
                       "subject_id": "bio"}


def test_recent_outcomes_keep_the_newest_answers_per_topic(fake_postgres):
    from agent.tools import get_recent_outcomes

    cur = FakePostgres(rows=[("t1", 1.0, None), ("t1", None, False), ("t2", 0.5, True)])
    fake_postgres(cur)
    out = get_recent_outcomes("u1", ["t1", "t2"], per_topic=5, config=get_config())

    sql, args = cur.statements[0]
    assert "row_number() OVER (PARTITION BY e.topic_id ORDER BY e.created_at DESC)" in sql
    assert args == ("u1", ["t1", "t2"], 5)
    assert out == {"t1": [1.0, 0.0], "t2": [0.5]}


def test_weekly_activity_is_bucketed_in_the_database(fake_postgres):
    from datetime import datetime, timezone

    from agent.tools import get_weekly_activity

    week = datetime(2026, 9, 21, tzinfo=timezone.utc)
    cur = FakePostgres(rows=[(week, 12, 9, 0.71234)])
    fake_postgres(cur)
    out = get_weekly_activity("u1", weeks=8, subject_ids=["math"], config=get_config())

    sql, args = cur.statements[0]
    assert "date_trunc('week', e.created_at)" in sql and "t.subject_id = ANY(%s)" in sql
    assert args == (["u1"], 8, ["math"])
    assert out == [{"week_start": "2026-09-21", "answers": 12, "correct": 9,
                    "mean_score": 0.7123}]


def test_direct_prerequisites_are_grouped_per_topic(fake_postgres):
    from agent.tools import get_direct_prereqs

    cur = FakePostgres(rows=[("quad", "fact", "Factoring"), ("quad", "arith", "Arithmetic"),
                             ("frac", "div", None)])
    fake_postgres(cur)
    out = get_direct_prereqs(["quad", "frac"], config=get_config())
    assert out == {"quad": [("fact", "Factoring"), ("arith", "Arithmetic")],
                   "frac": [("div", "")]}


def test_a_report_is_saved_with_its_scope_and_owner(fake_postgres):
    import json
    from datetime import datetime, timezone

    from agent.tools import save_report

    at = datetime(2026, 9, 27, 8, 0, tzinfo=timezone.utc)
    cur = FakePostgres(fetchone=[("r1", at)])
    conn = fake_postgres(cur)
    saved = save_report("student_progress", {"narrative": "ok"}, user_id="u1",
                        config=get_config())

    sql, args = cur.statements[0]
    assert "INSERT INTO ai.reports" in sql and "%s::ai.report_scope" in sql
    assert args[0] == "u1" and args[1] is None and args[2] == "student_progress"
    assert json.loads(args[3]) == {"narrative": "ok"}
    assert saved == {"report_id": "r1", "generated_at": at.isoformat()}
    assert conn.committed


def test_the_latest_report_says_whether_newer_evidence_has_arrived(fake_postgres):
    """A stored report is only worth serving while it still describes the student;
    newer graded answers mean the note is out of date, and the caller should know."""
    from datetime import datetime, timezone

    from agent.tools import get_latest_report

    at = datetime(2026, 9, 27, 8, 0, tzinfo=timezone.utc)
    cur = FakePostgres(fetchone=[("r1", at, {"narrative": "ok"}, 3)])
    fake_postgres(cur)
    latest = get_latest_report("student_progress", user_id="u1", config=get_config())

    sql, _ = cur.statements[0]
    assert "ORDER BY r.generated_at DESC LIMIT 1" in sql and "ai.mastery_events" in sql
    assert latest == {"report_id": "r1", "generated_at": at.isoformat(),
                      "payload": {"narrative": "ok"}, "newer_evidence": 3, "stale": True}


def test_no_stored_report_is_none_not_an_empty_report(fake_postgres):
    from agent.tools import get_latest_report

    fake_postgres(FakePostgres(fetchone=[None]))
    assert get_latest_report("student_progress", user_id="u1", config=get_config()) is None


def test_a_rosters_mastery_is_read_in_one_query(fake_postgres):
    """One query per student would make a 40-student class 40 round trips on every
    teacher page load."""
    from agent.tools import get_roster_mastery

    cur = FakePostgres(rows=[("s1", "t1", 0.3, 4, None), ("s2", "t1", 0.9, 5, None),
                             ("s2", "t2", 0.2, 1, None)])
    fake_postgres(cur)
    rows = get_roster_mastery(["s1", "s2"], config=get_config())

    sql, args = cur.statements[0]
    assert "m.user_id = ANY(%s)" in sql and args == (["s1", "s2"],)
    assert [(r["user_id"], r["topic_id"], r["status"]) for r in rows] == [
        ("s1", "t1", "weak"), ("s2", "t1", "strong"), ("s2", "t2", "unknown")]


def test_each_students_recent_history_comes_back_oldest_first(fake_postgres):
    from datetime import datetime, timedelta, timezone

    from agent.tools import get_roster_history

    t0 = datetime(2026, 9, 1, tzinfo=timezone.utc)
    cur = FakePostgres(rows=[("s1", t0 + timedelta(days=2), 0.0, False),
                             ("s1", t0, 1.0, True), ("s2", t0, None, True)])
    fake_postgres(cur)
    out = get_roster_history(["s1", "s2"], per_user=20, config=get_config())

    sql, args = cur.statements[0]
    assert "PARTITION BY e.user_id ORDER BY e.created_at DESC" in sql
    assert args == (["s1", "s2"], 20)
    assert out["s1"] == {"last_active": t0 + timedelta(days=2), "scores": [1.0, 0.0]}
    assert out["s2"]["scores"] == [1.0]


def test_the_commonest_mistakes_count_answers_and_students(fake_postgres):
    from agent.tools import get_top_mistakes

    cur = FakePostgres(rows=[("tag1", "Sign error", 9, 4)])
    fake_postgres(cur)
    out = get_top_mistakes(["s1", "s2"], since_days=30, limit=5, subject_ids=["math"],
                           config=get_config())

    sql, args = cur.statements[0]
    assert "unnest(e.mistake_tag_ids)" in sql and "t.subject_id = ANY(%s)" in sql
    assert args == (["s1", "s2"], 30, ["math"], 5)
    assert out == [{"tag_id": "tag1", "label": "Sign error", "occurrences": 9,
                    "students": 4}]


def test_the_most_missed_questions_rank_by_wrong_answers(fake_postgres):
    from agent.tools import get_most_missed

    cur = FakePostgres(rows=[("q1", "t1", 12, 9, 7)])
    fake_postgres(cur)
    out = get_most_missed(["s1"], since_days=30, limit=5, config=get_config())

    sql, args = cur.statements[0]
    assert "e.item_id IS NOT NULL" in sql and "ORDER BY 4 DESC" in sql
    assert args == (["s1"], 30, 5)
    assert out == [{"item_id": "q1", "topic_id": "t1", "answers": 12, "wrong": 9,
                    "students": 7}]
