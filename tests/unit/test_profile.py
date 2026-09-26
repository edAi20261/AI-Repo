"""What the tutor still knows when the chat thread is new.

Every fact the tutor held was scoped to one conversation id. A student who
closed the tab and opened another chat met a stranger — same student, same
subject, same lesson they were stuck on an hour before, and nothing carried.
"""

from __future__ import annotations

import json
import uuid

from agent.config import get_config
from agent.conversation import HistoryWindow, InMemoryConversationStore
from agent.conversation.history import NO_HISTORY
from agent.graph import nodes as nmod
from agent.models import StatedPreference, SubjectFacts
from agent.profile import Carryover, carried_over, record
from agent.subjects import SubjectDirectory

USER = str(uuid.uuid4())
SCIENCE = str(uuid.uuid4())
FRENCH = str(uuid.uuid4())


class _Subjects:
    """A subject directory that can name two subjects and no others.

    Returns real ``SubjectFacts``, not an object with whatever attribute the
    test happened to want. The first version of this defined its own class with
    a ``title`` on it; SubjectFacts calls that field ``name``, so the test
    passed while production raised AttributeError on the first turn of every
    new conversation.
    """

    def __init__(self, names):
        self.names = names

    def facts(self, subject_id) -> SubjectFacts:
        return SubjectFacts(name=self.names.get(subject_id))


def _turn(language=None, subject_id=None):
    return json.dumps({"message": "q", "reply": "a", "kind": "study",
                       "language": language, "subject_id": subject_id})


def _db(fake_postgres, turns, stated=None):
    return fake_postgres(fetchall=[[(fact,) for fact in turns]],
                         fetchone=[stated] if stated is not None else [])


# --- what is carried ----------------------------------------------------------

def test_a_student_carries_the_subject_they_were_working_on(fake_postgres):
    _db(fake_postgres, [_turn("ar", SCIENCE), _turn("ar", SCIENCE), _turn("fr", FRENCH)])
    carried = carried_over(USER, _Subjects({SCIENCE: "العلوم", FRENCH: "Sciences"}),
                           config=get_config())
    assert carried.subjects[0] == "العلوم"          # the one they were in most
    assert carried.languages[0] == "ar"
    assert "العلوم" in carried.render()


def test_a_subject_the_tutor_cannot_name_is_left_out(fake_postgres):
    _db(fake_postgres, [_turn("en", SCIENCE)])
    carried = carried_over(USER, _Subjects({}), config=get_config())
    assert carried.subjects == [] and carried.languages == ["en"]


def test_a_stated_preference_outlives_the_conversation_it_was_stated_in(fake_postgres):
    _db(fake_postgres, [_turn("en", SCIENCE)], stated=("short", True, False, "slow"))
    carried = carried_over(USER, _Subjects({SCIENCE: "Science"}), config=get_config())
    assert carried.stated.length == "short" and carried.stated.examples is True
    assert carried.stated.pace == "slow"
    assert carried.stated.visuals is None        # false in the column is "never said"
    text = carried.render()
    assert "short answers" in text and "worked examples" in text


def test_a_student_with_no_past_carries_nothing(fake_postgres):
    _db(fake_postgres, [])
    assert not carried_over(USER, _Subjects({}), config=get_config())


def test_an_unreadable_profile_starts_the_conversation_cold(fake_postgres):
    fake_postgres(explode=True)
    assert not carried_over(USER, _Subjects({}), config=get_config())


def test_nothing_is_read_without_a_student():
    assert not carried_over(None, _Subjects({}))


# --- what is written ----------------------------------------------------------

def test_only_a_stated_preference_is_written(fake_postgres):
    db = fake_postgres()
    assert record(USER, StatedPreference()) is False
    assert not db.statements


def test_a_stated_preference_is_written_once_per_student(fake_postgres):
    db = fake_postgres()
    assert record(USER, StatedPreference(length="short"), config=get_config())
    sql, params = db.statements[0]
    assert "INSERT INTO ai.adaptive_profile" in sql and "ON CONFLICT (user_id)" in sql
    assert params[0] == USER and "short" in params


def test_a_preference_not_restated_is_not_overwritten(fake_postgres):
    """Saying "keep it short" must not erase the pace they asked for last week."""
    db = fake_postgres()
    record(USER, StatedPreference(length="short"), config=get_config())
    sql, _ = db.statements[0]
    assert "COALESCE(EXCLUDED.explanation_length, p.explanation_length)" in sql
    assert "COALESCE(%s, p.pace)" in sql or "COALESCE(EXCLUDED.pace, p.pace)" in sql


def test_a_preference_the_column_cannot_hold_is_not_written_silently(fake_postgres, caplog):
    """prefers_examples is NOT NULL DEFAULT false, so a stated "no examples"
    reads back as "never mentioned". Until that is migrated, it says so."""
    fake_postgres()
    with caplog.at_level("WARNING"):
        record(USER, StatedPreference(examples=False), config=get_config())
    assert "cannot hold a stated 'no'" in caplog.text


# --- the turn -----------------------------------------------------------------

def test_the_carryover_stands_in_for_a_history_that_does_not_exist_yet():
    conversation = InMemoryConversationStore().load(USER, str(uuid.uuid4()))
    carried = Carryover(subjects=["Science"], languages=["ar"])
    text = HistoryWindow().render(conversation, carried_over=carried.render())
    assert NO_HISTORY in text and "Science" in text


def test_a_conversation_with_turns_in_it_does_not_reach_back(monkeypatch):
    """The last six turns are better evidence than what the student did last
    week, and reading both is two queries for the worse one."""
    store = InMemoryConversationStore()
    session = str(uuid.uuid4())
    store.append(USER, "student", session, nmod.Turn(message="hello", reply="hi"))
    reached = []
    monkeypatch.setattr(nmod, "carried_over",
                        lambda *a, **k: reached.append(1) or Carryover())
    classify = nmod.build_classify(get_config(), store, HistoryWindow(),
                                   SubjectDirectory(get_config()))
    classify({"user_type": "student", "user_id": USER, "session_id": session})
    assert not reached
