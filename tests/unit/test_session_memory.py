"""What the tutor remembers about a conversation, and what happens when it can't.

The service is stateless per turn, so an instruction like "reply in Arabic from
now on" was honoured once and forgotten. This is the one field that survives —
and because it now sits at the front of every turn, the failure paths matter as
much as the happy one.
"""

from __future__ import annotations

import pytest

from agent.config import Config
from agent.session_memory import KIND, SessionLanguage


class _Cursor:
    def __init__(self, conn):
        self._conn = conn

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params):
        if self._conn.explode:
            raise RuntimeError("database unreachable")
        self._conn.statements.append((" ".join(sql.split()), params))

    def fetchall(self):
        return self._conn.rows


class _Connection:
    """Stands in for psycopg: records statements, or fails on demand."""

    def __init__(self, rows=None, explode=False):
        self.rows = rows or []
        self.explode = explode
        self.statements: list = []
        self.committed = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def cursor(self):
        return _Cursor(self)

    def commit(self):
        self.committed = True


@pytest.fixture
def config() -> Config:
    return Config.load()


def _store(monkeypatch, config, conn) -> SessionLanguage:
    store = SessionLanguage(config)
    monkeypatch.setattr(store, "_run", store._run)          # keep the real body
    import psycopg
    monkeypatch.setattr(psycopg, "connect", lambda *a, **k: conn)
    return store


USER = "11111111-1111-1111-1111-111111111111"
SESSION = "22222222-2222-2222-2222-222222222222"


# --- reading ----------------------------------------------------------------

def test_a_stored_instruction_is_returned(monkeypatch, config):
    conn = _Connection(rows=[("ar",)])
    store = _store(monkeypatch, config, conn)
    assert store.get(USER, SESSION) == "ar"
    assert KIND in str(conn.statements[0][1])


def test_a_session_with_no_instruction_returns_nothing(monkeypatch, config):
    store = _store(monkeypatch, config, _Connection(rows=[]))
    assert store.get(USER, SESSION) is None


@pytest.mark.parametrize("user_id,session_id", [
    (None, SESSION), (USER, None), (None, None), ("", ""),
])
def test_without_both_ids_nothing_is_read(monkeypatch, config, user_id, session_id):
    """An anonymous or one-shot turn costs no round trip at all."""
    conn = _Connection(rows=[("ar",)])
    store = _store(monkeypatch, config, conn)
    assert store.get(user_id, session_id) is None
    assert conn.statements == []


# --- writing ----------------------------------------------------------------

def test_setting_replaces_what_the_session_held(monkeypatch, config):
    """One row per session: the read stays a single lookup and the table stays
    the size of live conversations."""
    conn = _Connection()
    store = _store(monkeypatch, config, conn)
    assert store.set(USER, SESSION, code="ar", user_type="student") is True
    verbs = [s.split()[0] for s, _ in conn.statements]
    assert verbs == ["DELETE", "INSERT"]
    assert conn.committed is True


def test_clearing_removes_the_instruction(monkeypatch, config):
    conn = _Connection()
    store = _store(monkeypatch, config, conn)
    assert store.clear(USER, SESSION) is True
    assert [s.split()[0] for s, _ in conn.statements] == ["DELETE"]


def test_a_parent_turn_remembers_nothing(monkeypatch, config):
    """ai.memory_user_type covers student and teacher only."""
    conn = _Connection()
    store = _store(monkeypatch, config, conn)
    assert store.set(USER, SESSION, code="ar", user_type="parent") is False
    assert conn.statements == []


# --- when the database is not there -----------------------------------------

def test_a_read_failure_looks_like_no_instruction(monkeypatch, config):
    """The turn must answer exactly as it did before this feature existed."""
    store = _store(monkeypatch, config, _Connection(explode=True))
    assert store.get(USER, SESSION) is None


def test_a_write_failure_is_reported_not_raised(monkeypatch, config):
    store = _store(monkeypatch, config, _Connection(explode=True))
    assert store.set(USER, SESSION, code="ar", user_type="student") is False


def test_repeated_failures_stop_costing_the_turn(monkeypatch, config):
    """Without the breaker, a dead database adds the connect timeout to every
    single turn for as long as the outage lasts."""
    conn = _Connection(explode=True)
    store = SessionLanguage(config, failure_threshold=2, breaker_seconds=60)
    import psycopg
    monkeypatch.setattr(psycopg, "connect", lambda *a, **k: conn)

    for _ in range(2):
        store.get(USER, SESSION)
    attempts = len(conn.statements)

    def fail(*a, **k):
        raise AssertionError("the breaker should have stopped this")

    monkeypatch.setattr(psycopg, "connect", fail)
    assert store.get(USER, SESSION) is None
    assert len(conn.statements) == attempts


def test_a_success_resets_the_failure_count(monkeypatch, config):
    store = SessionLanguage(config, failure_threshold=3)
    import psycopg
    monkeypatch.setattr(psycopg, "connect", lambda *a, **k: _Connection(explode=True))
    store.get(USER, SESSION)
    assert store._failures == 1
    monkeypatch.setattr(psycopg, "connect", lambda *a, **k: _Connection(rows=[("ar",)]))
    assert store.get(USER, SESSION) == "ar"
    assert store._failures == 0


# --- switching it off -------------------------------------------------------

def test_the_feature_can_be_turned_off(config, monkeypatch):
    monkeypatch.setattr(config, "get",
                        lambda key, default=None: False if key == "language.sticky.enabled"
                        else Config.load().get(key, default))
    assert SessionLanguage.from_config(config) is None
