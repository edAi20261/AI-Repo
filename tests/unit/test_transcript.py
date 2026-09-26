"""A thread the student can still see, that this service can no longer remember.

What we keep expires at ttl_hours; the platform's transcript never does. Nobody
could reach an old thread before, so nobody met the result — a history panel is
exactly a feature for reaching old threads, and without this the student
reopens a conversation, sees the whole of it on screen, and is answered by a
tutor for whom it is the first message ever sent.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

from agent.config import get_config
from agent.conversation import HistoryWindow, InMemoryConversationStore, Turn, recover
from agent.conversation.records import SessionState
from agent.graph import nodes as nmod
from agent.subjects import SubjectDirectory

USER = str(uuid.uuid4())
SESSION = str(uuid.uuid4())
SUBJECT = str(uuid.uuid4())

WHEN = datetime(2026, 8, 1, 9, 0, tzinfo=timezone.utc)


def _messages(*pairs):
    """Transcript rows newest-first, as the query returns them."""
    rows, clock = [], WHEN
    for sender, text, metadata in pairs:
        rows.append((sender, text, metadata, clock))
        clock += timedelta(seconds=30)
    return list(reversed(rows))


def _db(fake_postgres, rows, subject=SUBJECT):
    return fake_postgres(fetchall=[rows], fetchone=[(subject,)])


def test_an_expired_thread_is_read_back_from_the_transcript(fake_postgres):
    _db(fake_postgres, _messages(
        ("student", "What is evaporation?", None),
        ("ai", "Evaporation is ...", {"tasks": ["explain"]}),
        ("student", "Give me an example.", None),
        ("ai", "A puddle ...", {"tasks": ["explain"]}),
    ))
    turns, state = recover(SESSION, turns=6, config=get_config())
    assert [t.message for t in turns] == ["What is evaporation?", "Give me an example."]
    assert turns[0].reply == "Evaporation is ..." and turns[0].kind == "study"
    assert state.subject_id == SUBJECT


def test_a_question_that_was_never_answered_is_not_recovered(fake_postgres):
    """A turn that produced no reply is never written in the first place, so
    what comes back is what would have been remembered — not everything typed."""
    _db(fake_postgres, _messages(
        ("student", "this one died", None),
        ("student", "What is evaporation?", None),
        ("ai", "Evaporation is ...", {"tasks": ["explain"]}),
    ))
    turns, _ = recover(SESSION, turns=6, config=get_config())
    assert [t.message for t in turns] == ["What is evaporation?"]


def test_small_talk_is_recovered_as_small_talk(fake_postgres):
    _db(fake_postgres, _messages(
        ("student", "hello", None),
        ("ai", "Hi there!", {"tasks": []}),
    ))
    turns, _ = recover(SESSION, turns=6, config=get_config())
    assert turns[0].kind == "conversation"


def test_a_recovered_turn_lends_its_words_but_not_its_passages(fake_postgres):
    """The transcript keeps which lesson was cited, not which chunks, so a
    recovered answer cannot be followed up from its own evidence."""
    _db(fake_postgres, _messages(
        ("student", "What is evaporation?", None),
        ("ai", "Evaporation is ...", {"tasks": ["explain"],
                                      "citations": [{"lectureId": "l1"}]}),
    ))
    turns, _ = recover(SESSION, turns=6, config=get_config())
    assert turns[0].cited_chunk_ids == []


def test_only_the_most_recent_exchanges_are_read_back(fake_postgres):
    _db(fake_postgres, _messages(*[
        item for index in range(8)
        for item in (("student", f"q{index}", None), ("ai", f"a{index}", {"tasks": []}))
    ]))
    turns, _ = recover(SESSION, turns=3, config=get_config())
    assert [t.message for t in turns] == ["q5", "q6", "q7"]


def test_a_thread_with_no_transcript_recovers_nothing(fake_postgres):
    _db(fake_postgres, [], subject=None)
    turns, state = recover(SESSION, turns=6, config=get_config())
    assert turns == [] and state.subject_id is None


def test_an_unreadable_transcript_does_not_cost_the_student_an_answer(fake_postgres):
    fake_postgres(explode=True)
    assert recover(SESSION, turns=6, config=get_config()) == ([], SessionState())


def test_nothing_is_read_without_a_session():
    assert recover(None, turns=6) == ([], SessionState())


# --- putting it back ----------------------------------------------------------

def test_a_recovered_conversation_is_written_back_in_order():
    """Otherwise every later turn recovers again, and the turn after the first
    one sees only itself."""
    store = InMemoryConversationStore()
    recovered = [Turn(message=f"q{i}", reply=f"a{i}") for i in range(3)]
    assert store.restore(USER, "student", SESSION, recovered, SessionState(subject_id=SUBJECT))
    loaded = store.load(USER, SESSION)
    assert [t.message for t in loaded.turns] == ["q0", "q1", "q2"]
    assert loaded.state.subject_id == SUBJECT
    assert not loaded.is_new


def test_restoring_nothing_writes_nothing():
    store = InMemoryConversationStore()
    assert store.restore(USER, "student", SESSION, [], SessionState()) is False
    assert store.load(USER, SESSION).is_new


def test_a_remembered_conversation_is_not_recovered_over(monkeypatch):
    store = InMemoryConversationStore()
    store.append(USER, "student", SESSION, Turn(message="just now", reply="ok"))
    called = []
    monkeypatch.setattr(nmod, "recover",
                        lambda *a, **k: called.append(1) or ([], SessionState()))
    classify = nmod.build_classify(get_config(), store, HistoryWindow(),
                                   SubjectDirectory(get_config()))
    classify({"user_type": "student", "user_id": USER, "session_id": SESSION})
    assert not called


def test_a_recovered_thread_is_in_view_of_the_turn_that_recovered_it(monkeypatch):
    store = InMemoryConversationStore()
    monkeypatch.setattr(nmod, "recover", lambda *a, **k: (
        [Turn(message="What is evaporation?", reply="Evaporation is ...")],
        SessionState(subject_id=SUBJECT)))
    classify = nmod.build_classify(get_config(), store, HistoryWindow(),
                                   SubjectDirectory(get_config()))
    out = classify({"user_type": "student", "user_id": USER, "session_id": SESSION})
    assert "What is evaporation?" in out["history"]
    assert out["conversation"]["state"]["subject_id"] == SUBJECT
