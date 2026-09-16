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
from tests.conftest import FakePostgres


@pytest.fixture
def config() -> Config:
    return Config.load()


def _store(fake_postgres, config, conn) -> SessionLanguage:
    fake_postgres(conn)
    return SessionLanguage(config)


USER = "11111111-1111-1111-1111-111111111111"
SESSION = "22222222-2222-2222-2222-222222222222"


# --- reading ----------------------------------------------------------------

def test_a_stored_instruction_is_returned(fake_postgres, config):
    conn = FakePostgres(rows=[("ar",)])
    store = _store(fake_postgres, config, conn)
    assert store.get(USER, SESSION) == "ar"
    assert KIND in str(conn.statements[0][1])


def test_a_session_with_no_instruction_returns_nothing(fake_postgres, config):
    store = _store(fake_postgres, config, FakePostgres(rows=[]))
    assert store.get(USER, SESSION) is None


@pytest.mark.parametrize("user_id,session_id", [
    (None, SESSION), (USER, None), (None, None), ("", ""),
])
def test_without_both_ids_nothing_is_read(fake_postgres, config, user_id, session_id):
    """An anonymous or one-shot turn costs no round trip at all."""
    conn = FakePostgres(rows=[("ar",)])
    store = _store(fake_postgres, config, conn)
    assert store.get(user_id, session_id) is None
    assert conn.statements == []


# --- writing ----------------------------------------------------------------

def test_setting_replaces_what_the_session_held(fake_postgres, config):
    """One row per session: the read stays a single lookup and the table stays
    the size of live conversations."""
    conn = FakePostgres()
    store = _store(fake_postgres, config, conn)
    assert store.set(USER, SESSION, code="ar", user_type="student") is True
    verbs = [s.split()[0] for s, _ in conn.statements]
    assert verbs == ["DELETE", "INSERT"]
    assert conn.committed is True


def test_clearing_removes_the_instruction(fake_postgres, config):
    conn = FakePostgres()
    store = _store(fake_postgres, config, conn)
    assert store.clear(USER, SESSION) is True
    assert [s.split()[0] for s, _ in conn.statements] == ["DELETE"]


def test_a_parent_turn_remembers_nothing(fake_postgres, config):
    """ai.memory_user_type covers student and teacher only."""
    conn = FakePostgres()
    store = _store(fake_postgres, config, conn)
    assert store.set(USER, SESSION, code="ar", user_type="parent") is False
    assert conn.statements == []


# --- when the database is not there -----------------------------------------

def test_a_read_failure_looks_like_no_instruction(fake_postgres, config):
    """The turn must answer exactly as it did before this feature existed."""
    store = _store(fake_postgres, config, FakePostgres(explode=True))
    assert store.get(USER, SESSION) is None


def test_a_write_failure_is_reported_not_raised(fake_postgres, config):
    store = _store(fake_postgres, config, FakePostgres(explode=True))
    assert store.set(USER, SESSION, code="ar", user_type="student") is False


def test_repeated_failures_stop_costing_the_turn(fake_postgres, monkeypatch, config):
    """Without the breaker, a dead database adds the connect timeout to every
    single turn for as long as the outage lasts."""
    conn = fake_postgres(explode=True)
    store = SessionLanguage(config, failure_threshold=2, breaker_seconds=60)

    for _ in range(2):
        store.get(USER, SESSION)
    attempts = len(conn.statements)

    def fail(*a, **k):
        raise AssertionError("the breaker should have stopped this")

    import psycopg
    monkeypatch.setattr(psycopg, "connect", fail)
    assert store.get(USER, SESSION) is None
    assert len(conn.statements) == attempts


def test_a_success_resets_the_failure_count(fake_postgres, config):
    store = SessionLanguage(config, failure_threshold=3)
    fake_postgres(explode=True)
    store.get(USER, SESSION)
    assert store._failures == 1
    fake_postgres(rows=[("ar",)])
    assert store.get(USER, SESSION) == "ar"
    assert store._failures == 0


# --- switching it off -------------------------------------------------------

def test_the_feature_can_be_turned_off(config, monkeypatch):
    monkeypatch.setattr(config, "get",
                        lambda key, default=None: False if key == "language.sticky.enabled"
                        else Config.load().get(key, default))
    assert SessionLanguage.from_config(config) is None


# --- the feature, through the nodes -----------------------------------------

class _FakeStore:
    """The store without a database. Also stands in for one that is down, by
    returning the same neutral values every method promises on failure."""

    def __init__(self, working: bool = True):
        self.rows: dict = {}
        self.working = working

    def get(self, user_id, session_id):
        return self.rows.get((user_id, session_id)) if self.working else None

    def set(self, user_id, session_id, *, code, user_type, evidence=None):
        if not self.working:
            return False
        self.rows[(user_id, session_id)] = code
        return True

    def clear(self, user_id, session_id):
        if not self.working:
            return False
        self.rows.pop((user_id, session_id), None)
        return True


def _conversation(store, session_id="s1"):
    from agent.graph import nodes
    config = Config.load()
    classify = nodes.build_classify(config, sticky=store)
    persist = nodes.build_persist(config, sticky=store)

    def say(message):
        state = {"message": message, "user_id": USER, "user_type": "student",
                 "session_id": session_id}
        out = classify(state)
        persist({**state, **out})
        return out["language"]

    return say


def test_from_now_on_means_from_now_on():
    """The whole point. Turn 2 is plain English and must still come back Arabic."""
    store = _FakeStore()
    say = _conversation(store)
    assert say("explain subtraction, and reply in Arabic from now on") == "ar"
    assert say("and what about addition?") == "ar"
    assert say("2 + 2 = ?") == "ar"


def test_the_student_can_get_out_again():
    store = _FakeStore()
    say = _conversation(store)
    say("reply in Arabic from now on")
    assert say("answer in the language I write in") == "en"
    assert say("and what about multiplication?") == "en"


def test_an_instruction_does_not_leak_into_another_conversation():
    store = _FakeStore()
    _conversation(store, "s1")("reply in Arabic from now on")
    assert _conversation(store, "s2")("and what about addition?") == "en"


def test_a_newer_instruction_replaces_the_standing_one():
    store = _FakeStore()
    say = _conversation(store)
    say("reply in Arabic from now on")
    assert say("actually, answer in French") == "fr"
    assert say("and what about addition?") == "fr"


def test_a_database_that_is_down_costs_the_memory_not_the_answer():
    """Degraded behaviour is exactly the behaviour the service shipped with."""
    say = _conversation(_FakeStore(working=False))
    assert say("explain subtraction, and reply in Arabic from now on") == "ar"
    assert say("and what about addition?") == "en"
    assert say("اشرح لي عملية الطرح") == "ar"


def test_memory_switched_off_changes_nothing():
    say = _conversation(None)
    assert say("reply in Arabic from now on") == "ar"
    assert say("and what about addition?") == "en"


def test_an_instruction_turn_does_not_read_the_store():
    """A message that gives or revokes an instruction settles it by itself."""
    class _Counting(_FakeStore):
        reads = 0

        def get(self, user_id, session_id):
            type(self).reads += 1
            return super().get(user_id, session_id)

    store = _Counting()
    say = _conversation(store)
    say("reply in Arabic from now on")
    say("answer in the language I write in")
    assert _Counting.reads == 0
    say("and what about addition?")
    assert _Counting.reads == 1
