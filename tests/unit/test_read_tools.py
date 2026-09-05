"""Read/write tool tests. No DB, no backend: psycopg.connect is faked and
the backend client is monkeypatched. Asserts SQL targets and row mapping."""

from __future__ import annotations

from collections import deque

import psycopg
import pytest

import agent.tools as tools
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


# --- fake psycopg -----------------------------------------------------------

class _FakeCursor:
    def __init__(self, fetchone=None, fetchall=None):
        self.sql: list[str] = []
        self._one = deque(fetchone or [])
        self._all = deque(fetchall or [])

    def execute(self, sql, args=None):
        self.sql.append(sql)
        self.last_args = args

    def fetchone(self):
        return self._one.popleft() if self._one else None

    def fetchall(self):
        return self._all.popleft() if self._all else []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _FakeConn:
    def __init__(self, cursor):
        self._cursor = cursor
        self.committed = False

    def cursor(self):
        return self._cursor

    def commit(self):
        self.committed = True

    def close(self):
        pass


@pytest.fixture()
def fake_db(monkeypatch):
    """Install a fake psycopg.connect returning a conn wrapping a preset cursor.
    Returns a setter the test calls with the cursor it wants."""
    holder = {}

    def install(cursor):
        conn = _FakeConn(cursor)
        holder["conn"] = conn
        monkeypatch.setattr(psycopg, "connect", lambda url: conn)
        return conn

    return install


# --- get_figures ------------------------------------------------------------

def test_get_figures_requires_a_filter():
    with pytest.raises(ValueError):
        get_figures(config=get_config())


def test_get_figures_by_topic(fake_db):
    cur = _FakeCursor(fetchall=[[
        ("f1", "lec1", "s3://b/f1.png", "cap", "alt", 2, ["t1"]),
    ]])
    fake_db(cur)
    out = get_figures(topic_ids=["t1"], config=get_config())
    assert out[0]["s3_uri"] == "s3://b/f1.png"
    assert "topic_ids && " in cur.sql[0]


# --- mistake_tags -----------------------------------------------------------

def test_list_mistake_tags(fake_db):
    fake_db(_FakeCursor(fetchall=[[("tag1", "sign error", None)]]))
    out = list_mistake_tags(config=get_config())
    assert out == [{"tag_id": "tag1", "label": "sign error", "description": None}]


def test_create_mistake_tag_reuses_existing(fake_db):
    cur = _FakeCursor(fetchone=[("existing-id",)])  # SELECT finds a match
    fake_db(cur)
    assert create_mistake_tag("Sign Error", config=get_config()) == "existing-id"
    assert len(cur.sql) == 1  # only the SELECT, no INSERT


def test_create_mistake_tag_inserts_when_absent(fake_db):
    cur = _FakeCursor(fetchone=[None, ("new-id",)])  # SELECT miss, INSERT returns
    conn = fake_db(cur)
    assert create_mistake_tag("brand new", config=get_config()) == "new-id"
    assert any("INSERT INTO ai.mistake_tags" in s for s in cur.sql)
    assert conn.committed


def test_create_mistake_tag_rejects_blank():
    with pytest.raises(ValueError):
        create_mistake_tag("   ", config=get_config())


# --- mastery store ----------------------------------------------------------

def test_get_mastery_adds_status(fake_db):
    # attempts >= min_attempts (3) and score >= strong_threshold -> 'strong'
    fake_db(_FakeCursor(fetchall=[[("t1", 0.9, 5, None, 0.1, 1.0)]]))
    out = get_mastery("u1", ["t1"], config=get_config())
    assert out[0]["topic_id"] == "t1"
    assert out[0]["status"] == "strong"


def test_get_mastery_scopes_by_subject(fake_db):
    cur = _FakeCursor(fetchall=[[("t1", 0.9, 5, None, 0.1, 1.0)]])
    fake_db(cur)
    get_mastery("u1", subject_id="sub1", config=get_config())
    # subject scope joins ai.topics and filters on it
    assert "JOIN ai.topics" in cur.sql[0] and "t.subject_id = %s" in cur.sql[0]
    assert cur.last_args == ("u1", "sub1")


def test_get_mastery_rollup_weighted_mean_and_counts(fake_db):
    # subject sub1: two topics — strong (0.9, 5 attempts) + weak (0.2, 3 attempts).
    fake_db(_FakeCursor(fetchall=[[
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


def test_write_mastery_events_inserts_and_commits(fake_db):
    cur = _FakeCursor(fetchone=[("e1",), ("e2",)])
    conn = fake_db(cur)
    ids = write_mastery_events([
        {"user_id": "u1", "topic_id": "t1", "source": "practice", "correct": True},
        {"user_id": "u1", "topic_id": "t2", "source": "quiz", "correct": False},
    ], config=get_config())
    assert ids == ["e1", "e2"]
    assert conn.committed


def test_write_mastery_events_requires_topic_and_source(fake_db):
    fake_db(_FakeCursor())
    with pytest.raises(ValueError):
        write_mastery_events([{"user_id": "u1", "correct": True}], config=get_config())


def test_update_mastery_upserts(fake_db):
    cur = _FakeCursor()
    conn = fake_db(cur)
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
