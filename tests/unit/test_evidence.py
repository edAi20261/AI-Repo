"""What an answer was built from, remembered for the turn that follows it.

A student who says "explain that more simply" is asking about the passages the
last answer used. Searching for their words finds something else; the only
right answer is the one already on the table, said differently.

So a turn records the chunks behind its answer, and the next turn can reach
them by pointer instead of by search. Evidence that does not outlive the turn
is why re-presentation cannot work.
"""

from __future__ import annotations

import json

import pytest

from agent.evidence import EvidenceSet, EvidenceStore

SESSION = "11111111-2222-3333-4444-555555555555"
USER = "99999999-8888-7777-6666-555555555555"


class _Rows:
    """Stands in for ai.user_memory."""

    def __init__(self, stored: str | None = None, explode: bool = False):
        self.stored = stored
        self.explode = explode
        self.writes: list[tuple] = []

    def run(self, statements, *, fetch=False, label=""):
        if self.explode:
            raise RuntimeError("no database today")
        if fetch:
            return [(self.stored,)] if self.stored else []
        self.writes.extend(statements)
        return True


def _store(rows: _Rows) -> EvidenceStore:
    store = EvidenceStore(config=None, user_id=USER, user_type="student",
                          session_id=SESSION)
    store._run = rows.run           # the one seam: where the rows come from
    return store


def _unreachable(monkeypatch) -> EvidenceStore:
    """A real store whose database is not there.

    The failure has to come from the connection, not from a stubbed _run —
    replacing _run replaces the very code under test.
    """
    from agent import evidence as module

    def _no_database(*_a, **_k):
        raise RuntimeError("no database today")

    monkeypatch.setattr(module, "connect", _no_database)
    return EvidenceStore(config=None, user_id=USER, user_type="student",
                         session_id=SESSION)


def test_what_an_answer_was_built_from_comes_back_on_the_next_turn():
    evidence = EvidenceSet(chunk_ids=["c1", "c2"], query="What is evaporation?",
                           lecture_id="lec1")
    rows = _Rows(stored=evidence.model_dump_json())

    recalled = _store(rows).recall()

    assert recalled is not None
    assert recalled.chunk_ids == ["c1", "c2"]
    assert recalled.query == "What is evaporation?"


def test_a_session_that_has_answered_nothing_remembers_nothing():
    assert _store(_Rows(stored=None)).recall() is None


def test_remembering_replaces_what_the_session_held():
    """Only the last answer can be followed up on. Keeping every turn's
    evidence would grow without bound and answer "explain more" from a
    question three turns ago."""
    rows = _Rows()
    _store(rows).remember(EvidenceSet(chunk_ids=["c9"], query="q", lecture_id=None))

    assert len(rows.writes) == 2, "a delete and an insert, in one transaction"
    assert "DELETE" in rows.writes[0][0]
    assert "INSERT" in rows.writes[1][0]


def test_an_answer_with_no_evidence_is_not_remembered():
    """A declined turn grounded nothing. Remembering an empty set would let the
    next turn re-present a refusal as though it were an explanation."""
    rows = _Rows()
    _store(rows).remember(EvidenceSet(chunk_ids=[], query="q", lecture_id=None))

    assert rows.writes == []


# --- nothing here may cost a student their answer ---------------------------

def test_a_store_that_cannot_be_reached_remembers_nothing_and_says_so(
        monkeypatch, caplog):
    """Indistinguishable, to the caller, from a session with no previous
    answer — which is the behaviour the service already has. The reason is
    logged rather than swallowed."""
    import logging

    with caplog.at_level(logging.WARNING):
        assert _unreachable(monkeypatch).recall() is None
    assert any("evidence" in r.message.lower() for r in caplog.records)


def test_a_store_that_cannot_be_written_to_does_not_fail_the_turn(
        monkeypatch, caplog):
    import logging

    with caplog.at_level(logging.WARNING):
        assert _unreachable(monkeypatch).remember(
            EvidenceSet(chunk_ids=["c1"], query="q", lecture_id=None)) is False


def test_a_store_that_keeps_failing_stops_trying(monkeypatch, caplog):
    """A database that is down stays down for a while. Retrying it on every
    turn spends a student's latency to learn the same thing again."""
    import logging

    store = _unreachable(monkeypatch)
    with caplog.at_level(logging.WARNING):
        for _ in range(4):
            store.recall()

    assert any("paused" in r.message for r in caplog.records)


def test_a_stored_set_that_no_longer_fits_the_shape_is_forgotten(caplog):
    """The shape changed under it. That is stale, not fatal: the turn searches
    the way it always did."""
    import logging

    with caplog.at_level(logging.WARNING):
        assert _store(_Rows(stored=json.dumps({"chunk_ids": "not a list"}))).recall() is None


@pytest.mark.parametrize("user_type", ["parent", "guest"])
def test_only_the_people_the_service_remembers_for_are_remembered_for(user_type):
    """ai.memory_user_type covers student and teacher. A parent turn stores
    nothing, exactly as session language does."""
    rows = _Rows()
    store = EvidenceStore(config=None, user_id=USER, user_type=user_type,
                          session_id=SESSION)
    store._run = rows.run
    store.remember(EvidenceSet(chunk_ids=["c1"], query="q", lecture_id=None))

    assert rows.writes == []
    assert store.recall() is None


def test_a_turn_with_no_session_remembers_nothing():
    """Without a conversation there is nothing to follow up on."""
    store = EvidenceStore(config=None, user_id=USER, user_type="student",
                          session_id=None)
    rows = _Rows(stored='{"chunk_ids":["c1"],"query":"q","lecture_id":null}')
    store._run = rows.run

    assert store.recall() is None
