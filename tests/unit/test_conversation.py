"""The tutor remembers a conversation, and every turn reads it.

A turn used to be answered from its own message alone. The platform stored the
transcript and never sent it; the service kept a language and a pointer to the
last answer's passages, and overwrote the pointer's question with each rewrite,
so "give me an example" became the remembered question and the turn after it
had nothing left to be about.
"""

from __future__ import annotations

import json
import uuid

from agent import retrieval as rmod
from agent.config import get_config
from agent.conversation import (
    Conversation,
    HistoryWindow,
    InMemoryConversationStore,
    SessionState,
    Turn,
    fold_if_due,
)
from agent.conversation.store import NOTE, PostgresConversationStore
from agent.graph import nodes as nmod
from agent.models import Citation, PrereqTopic, ResolvedTopic, RetrievedChunk
from shared.tokens import count_tokens

USER = str(uuid.uuid4())
SESSION = str(uuid.uuid4())


def _turn(message, reply="ok", **fields):
    return Turn(message=message, reply=reply, **fields)


# --- the store ----------------------------------------------------------------

def test_a_conversation_is_remembered_in_order():
    store = InMemoryConversationStore()
    store.append(USER, "student", SESSION, _turn("first"))
    store.append(USER, "student", SESSION, _turn("second"))
    loaded = store.load(USER, SESSION)
    assert [t.message for t in loaded.turns] == ["first", "second"]
    assert all(t.id for t in loaded.turns)


def test_sessions_do_not_share_memory():
    store = InMemoryConversationStore()
    store.append(USER, "student", SESSION, _turn("mine"))
    assert store.load(USER, str(uuid.uuid4())).is_new


def test_a_role_the_memory_table_cannot_hold_remembers_nothing():
    store = InMemoryConversationStore()
    assert store.append(USER, "parent", SESSION, _turn("hi")) is False
    assert store.load(USER, SESSION).is_new


def test_the_standing_state_is_replaced_not_accumulated():
    store = InMemoryConversationStore()
    store.save_state(USER, "student", SESSION, SessionState(language="fr"))
    store.save_state(USER, "student", SESSION, SessionState(language=None, subject_id="bio"))
    assert store.load(USER, SESSION).state == SessionState(subject_id="bio")


def test_folding_takes_old_turns_out_of_the_window_but_keeps_them():
    """The fold used to delete the turns it summarised, which left the next
    summary nothing to be rewritten from but the summary before it."""
    store = InMemoryConversationStore()
    for index in range(4):
        store.append(USER, "student", SESSION, _turn(f"q{index}"))
    old = store.load(USER, SESSION).turns[:2]
    store.fold(USER, "student", SESSION, summary="Covered: q0, q1", note="n", folded=old, discard=[])
    loaded = store.load(USER, SESSION)
    assert loaded.summary == "Covered: q0, q1"
    assert [t.message for t in loaded.turns] == ["q2", "q3"]
    assert [t.message for t in loaded.folded] == ["q0", "q1"]
    assert [t.message for t in loaded.said_so_far] == ["q0", "q1", "q2", "q3"]


def test_only_a_discarded_turn_is_destroyed():
    store = InMemoryConversationStore()
    for index in range(4):
        store.append(USER, "student", SESSION, _turn(f"q{index}"))
    turns = store.load(USER, SESSION).turns
    store.fold(USER, "student", SESSION, summary="s", note="n", folded=turns[:2], discard=turns[:1])
    loaded = store.load(USER, SESSION)
    assert [t.message for t in loaded.folded] == ["q1"]
    assert [t.message for t in loaded.turns] == ["q2", "q3"]


def test_a_store_that_cannot_be_reached_reads_as_a_new_conversation(fake_postgres):
    fake_postgres(explode=True)
    store = PostgresConversationStore(get_config(), {"failure_threshold": 1})
    assert store.load(USER, SESSION).is_new
    assert store.append(USER, "student", SESSION, _turn("x")) is False


def test_the_database_rows_become_one_conversation(fake_postgres):
    fake_postgres(rows=[
        ("1", "conversation_state", SessionState(subject_id="bio").model_dump_json()),
        ("2", "conversation_summary", "Covered: evaporation"),
        ("3", "conversation_folded_turn", _turn("what is evaporation?").model_dump_json()),
        ("4", "conversation_turn", _turn("explain more").model_dump_json()),
    ])
    loaded = PostgresConversationStore(get_config(), {}).load(USER, SESSION)
    assert loaded.state.subject_id == "bio"
    assert loaded.summary == "Covered: evaporation"
    assert loaded.turns[0].message == "explain more" and loaded.turns[0].id == "4"
    assert loaded.folded[0].message == "what is evaporation?"


# --- the window ---------------------------------------------------------------

def test_the_window_ships_the_values_config_declares():
    """``settings.get(name, DEFAULT)`` is the same second-value-in-the-source
    that test_config_contract exists for, one level below the dotted path it
    can see. A deployment with no overlay must get what config.yaml says."""
    declared = get_config().get("agent.conversation") or {}
    shipped = HistoryWindow.from_config(get_config())
    for name in ("recent_turns", "fold_after", "history_tokens", "reply_tokens",
                 "rebuild_tokens"):
        assert declared[name] == getattr(shipped, name) == getattr(
            HistoryWindow(), name), name


def test_the_history_quotes_the_recent_turns_oldest_first():
    conversation = Conversation(turns=[_turn("what is evaporation?", "Evaporation is ..."),
                                       _turn("give me an example", "A puddle ...")])
    text = HistoryWindow().render(conversation)
    assert text.index("what is evaporation?") < text.index("give me an example")
    assert "Tutor: A puddle" in text


def test_the_history_carries_the_summary_of_what_was_folded():
    text = HistoryWindow().render(Conversation(summary="Covered: watersheds"))
    assert "Covered: watersheds" in text


def test_a_long_answer_cannot_push_the_question_out_of_view():
    window = HistoryWindow(reply_tokens=50)
    text = window.render(Conversation(turns=[_turn("q", "x " * 5000)]))
    assert "Student: q" in text and count_tokens(text) < 60


def test_the_window_is_bounded_in_tokens_not_characters():
    """Six turns within 6000 characters was four turns on any conversation
    above the median: a p90 turn is 1305 characters."""
    conversation = Conversation(turns=[_turn(f"q{i}", "word " * 260) for i in range(12)])
    quoted = HistoryWindow().render(conversation)
    assert all(f"q{i}" in quoted for i in range(12))


def test_a_folded_turn_is_held_but_not_quoted_back_into_the_window():
    """Kept for the next rewrite to read, not for the next prompt to carry:
    a turn quoted verbatim and summarised is paid for twice."""
    store = InMemoryConversationStore()
    store.append(USER, "student", SESSION, _turn("long ago"))
    store.append(USER, "student", SESSION, _turn("just now"))
    turns = store.load(USER, SESSION).turns
    store.fold(USER, "student", SESSION, summary="Covered: long ago", note="n",
               folded=turns[:1], discard=[])
    conversation = store.load(USER, SESSION)
    assert [t.message for t in conversation.folded] == ["long ago"]
    quoted = HistoryWindow().render(conversation)
    assert "Student: long ago" not in quoted
    assert "Covered: long ago" in quoted and "Student: just now" in quoted


def test_folding_waits_until_the_window_is_full():
    window = HistoryWindow(recent_turns=2, fold_after=3)
    assert window.overflow(Conversation(turns=[_turn("a"), _turn("b"), _turn("c")])) == []
    full = Conversation(turns=[_turn(x) for x in "abcd"])
    assert [t.message for t in window.overflow(full)] == ["a", "b"]


class _Summariser:
    def __init__(self, reply=None, error=False, note="We talked about evaporation.",
                 replies=None):
        self.error, self.prompt, self.calls = error, None, 0
        self.replies = replies or [json.dumps({"summary": reply, "note": note})]

    def invoke(self, messages):
        self.prompt = messages[0].content
        if self.error:
            raise RuntimeError("model unavailable")
        content = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1

        class _Msg:
            pass
        message = _Msg()
        message.content = content
        return message


def _fold(store, window, llm, language="en"):
    return fold_if_due(store, window, llm, user_id=USER, user_type="student",
                       session_id=SESSION, language=language)


def _conversation_of(store, count, reply="ok"):
    for index in range(count):
        store.append(USER, "student", SESSION, _turn(f"question {index}", reply))
    return store


def test_old_turns_are_folded_into_a_running_summary():
    store = _conversation_of(InMemoryConversationStore(), 4)
    llm = _Summariser("Covered: a and b")
    window = HistoryWindow(recent_turns=2, fold_after=3)
    assert _fold(store, window, llm)
    assert "question 0" in llm.prompt
    loaded = store.load(USER, SESSION)
    assert loaded.summary == "Covered: a and b" and len(loaded.turns) == 2


def _note_of(store):
    return [fact for _, kind, fact in store._rows[(USER, SESSION)] if kind == NOTE]


def test_a_fold_keeps_a_plain_note_for_the_student_beside_the_summary():
    """The memory page showed the internal summary, headings and bullets included."""
    store = _conversation_of(InMemoryConversationStore(), 4)
    llm = _Summariser("Goals:\n- evaporation", note="We worked on evaporation and condensation.")
    assert _fold(store, HistoryWindow(recent_turns=2, fold_after=3), llm)
    assert _note_of(store) == ["We worked on evaporation and condensation."]
    assert store.load(USER, SESSION).summary == "Goals:\n- evaporation"


def test_a_markdown_note_is_sent_back_to_be_rewritten():
    store = _conversation_of(InMemoryConversationStore(), 4)
    llm = _Summariser(replies=[
        json.dumps({"summary": "Covered: a", "note": "- **Evaporation**\n- Condensation"}),
        json.dumps({"summary": "Covered: a", "note": "We covered evaporation."})])
    assert _fold(store, HistoryWindow(recent_turns=2, fold_after=3), llm)
    assert llm.calls == 2
    assert _note_of(store) == ["We covered evaporation."]


def test_a_note_that_never_becomes_plain_folds_nothing():
    store = _conversation_of(InMemoryConversationStore(), 4)
    window = HistoryWindow(recent_turns=2, fold_after=3, note_chars=40)
    llm = _Summariser("Covered: a", note="word " * 30)
    assert not _fold(store, window, llm)
    assert not _note_of(store) and len(store.load(USER, SESSION).turns) == 4


def test_a_failed_summary_keeps_every_turn():
    store = _conversation_of(InMemoryConversationStore(), 4)
    window = HistoryWindow(recent_turns=2, fold_after=3)
    assert not _fold(store, window, _Summariser(error=True))
    loaded = store.load(USER, SESSION)
    assert len(loaded.turns) == 4 and not loaded.folded


def test_the_summary_is_written_in_the_language_of_the_conversation():
    """Unpinned, it came back in English for conversations held in Arabic, and
    was then quoted at the head of every prompt that answered one."""
    store = _conversation_of(InMemoryConversationStore(), 4)
    llm = _Summariser("تمت تغطية: التبخر")
    assert _fold(store, HistoryWindow(recent_turns=2, fold_after=3), llm, language="ar")
    assert "Arabic (العربية)" in llm.prompt


def test_a_second_fold_rewrites_from_the_conversation_not_from_its_own_summary():
    """A summary of a summary loses detail every time it is taken."""
    store = _conversation_of(InMemoryConversationStore(), 4)
    window = HistoryWindow(recent_turns=2, fold_after=3)
    assert _fold(store, window, _Summariser("first summary"))
    store.append(USER, "student", SESSION, _turn("question 4"))
    store.append(USER, "student", SESSION, _turn("question 5"))
    llm = _Summariser("second summary")
    assert _fold(store, window, llm)
    assert "question 0" in llm.prompt          # the oldest turn, read again
    assert "first summary" not in llm.prompt   # and not its own last answer


def test_past_the_rewrite_budget_the_old_summary_is_carried_forward():
    """Once turns have to be dropped for good, what was said in them survives
    only in the summary — so the summary has to come with."""
    store = _conversation_of(InMemoryConversationStore(), 4, reply="word " * 400)
    window = HistoryWindow(recent_turns=2, fold_after=3, rebuild_tokens=800)
    assert _fold(store, window, _Summariser("first summary"))
    store.append(USER, "student", SESSION, _turn("question 4", "word " * 400))
    store.append(USER, "student", SESSION, _turn("question 5", "word " * 400))
    llm = _Summariser("second summary")
    assert _fold(store, window, llm)
    assert "first summary" in llm.prompt
    assert len(store.load(USER, SESSION).folded) < 4


# --- persist --------------------------------------------------------------------

def _persisted(state, *, standing=None):
    store = InMemoryConversationStore()
    if standing is not None:
        store.save_state(USER, "student", SESSION, standing)
        state = {**state, "conversation": {"state": standing.model_dump()}}
    node = nmod.build_persist(get_config(), store, HistoryWindow(), None)
    base = {"user_id": USER, "session_id": SESSION, "pathway": "student",
            "message": "Explain this: what is evaporation?", "conversation": {}}
    base.update(state)
    node(base)
    return store.load(USER, SESSION)


def test_a_grounded_answer_remembers_its_passages_and_subject():
    loaded = _persisted({
        "turn_kind": "study", "subject_id": "bio", "reply_body": "Evaporation is ... [1]",
        "tasks": [{"type": "explain", "params": {"query": "evaporation"}}],
        "published_citations": [Citation(n=1, chunk_id="c1"), Citation(n=2)]})
    turn = loaded.turns[0]
    assert turn.cited_chunk_ids == ["c1"] and turn.query == "evaporation"
    assert turn.reply == "Evaporation is ... [1]"
    assert loaded.state.subject_id == "bio"


def test_a_declined_answer_does_not_settle_the_subject():
    loaded = _persisted({"turn_kind": "study", "subject_id": "bio",
                         "reply_body": "Not in your lessons.", "published_citations": []})
    assert loaded.state.subject_id is None


def test_a_requested_language_is_kept_for_the_next_turn():
    loaded = _persisted({"turn_kind": "conversation", "reply_body": "حسنًا",
                         "requested_language": "ar"})
    assert loaded.state.language == "ar"


# --- through the graph --------------------------------------------------------------

def test_the_next_turn_sees_the_one_before_it(monkeypatch):
    """The whole point: the second turn is read with the first in view."""
    from agent.graph import build_graph
    from agent.graph import graph as gmod

    store = InMemoryConversationStore()
    monkeypatch.setattr(gmod, "maybe_llm", lambda config, task: None)
    monkeypatch.setattr(gmod.ConversationStore, "from_config", classmethod(lambda cls, c: store))
    monkeypatch.setattr(nmod, "resolve_topics",
                        lambda q, sid=None, **k: [ResolvedTopic("t1", "Evaporation", 0.9)])
    chunk = RetrievedChunk(chunk_id="c1", score=0.8, text="Evaporation is ...",
                           lecture_id="lec1", topic_ids=["t1"], page_from=3, page_to=3,
                           heading_path="Water", language="en")
    monkeypatch.setattr(rmod, "rag_search", lambda *a, **k: [chunk])
    monkeypatch.setattr(rmod, "get_topic_deps",
                        lambda ids, **k: [PrereqTopic("t0", "States of matter", "required", 1)])
    graph = build_graph(get_config())
    turn = {"user_id": USER, "user_type": "student", "session_id": SESSION,
            "subject_id": "bio"}

    graph.invoke({**turn, "message": "explain evaporation"})
    second = graph.invoke({**turn, "message": "give me an example"})

    assert "explain evaporation" in second["history"]
    assert [t.message for t in store.load(USER, SESSION).turns] == [
        "explain evaporation", "give me an example"]


def test_prerequisites_come_from_the_passages_the_answer_cited(monkeypatch):
    """A question about wind turbines listed the states of water: the walk was
    seeded from topics guessed off the question's words, not from the answer."""
    from agent.graph.subgraphs.explain_summarize_sg import _resolve_prereqs

    seen = {}

    def walk(ids, **k):
        seen["seeds"] = list(ids)
        return []

    monkeypatch.setattr(rmod, "resolve_prereqs", walk)
    cited = RetrievedChunk(chunk_id="c1", score=0.8, text="", lecture_id="l",
                           topic_ids=["cited-topic"], page_from=1, page_to=1,
                           heading_path="", language="en")
    uncited = RetrievedChunk(chunk_id="c2", score=0.7, text="", lecture_id="l",
                             topic_ids=["other-topic"], page_from=1, page_to=1,
                             heading_path="", language="en")
    _resolve_prereqs({"grounded": True, "chunks": [cited, uncited],
                      "topic_ids": ["guessed-topic"],
                      "citations": [Citation(n=1, chunk_id="c1")]})
    assert seen["seeds"] == ["cited-topic"]
    assert _resolve_prereqs({"grounded": False, "chunks": [cited],
                             "citations": []}) == {"prereq_topics": []}


def test_a_continuing_turn_answers_from_the_last_passages_and_new_ones(monkeypatch):
    from agent.graph.subgraphs import explain_summarize_sg as esg

    def chunk(cid):
        return RetrievedChunk(chunk_id=cid, score=0.8, text=cid, lecture_id="l",
                              topic_ids=[], page_from=1, page_to=1, heading_path="",
                              language="en")

    monkeypatch.setattr(esg, "get_chunks_by_id", lambda ids: [chunk(i) for i in ids])
    monkeypatch.setattr(rmod, "retrieve", lambda *a, **k: [chunk("old"), chunk("new")])
    out = esg._retrieve({"query": "examples of evaporation", "namespaces": ["syllabus"],
                         "previous_chunk_ids": ["old"], "k": 8}, get_config())
    assert [c.chunk_id for c in out["chunks"]] == ["old", "new"]


# --- an open quiz ---------------------------------------------------------------

def _quiz_state(**over):
    state = {
        "turn_kind": "study", "subject_id": "bio",
        "reply_body": "1. Which process moves water into the air?\n   a) condensation\n   d) evaporation",
        "tasks": [{"type": "assessment", "params": {"query": "water cycle quiz"}}],
        "assessment": {"items": [
            {"type": "mcq", "question": "Which process moves water into the air?",
             "topic_id": "t1", "lecture_id": "l1", "answer": "d) evaporation",
             "options": [{"text": "condensation", "is_correct": False},
                         {"text": "evaporation", "is_correct": True}]}]},
    }
    state.update(over)
    return state


def test_an_issued_quiz_is_remembered_with_its_answer_key():
    """The items were sent to the student and dropped. Nothing could be marked
    afterwards, so an answer came back explained instead of graded."""
    loaded = _persisted(_quiz_state())
    open_quiz = loaded.state.pending
    assert open_quiz and len(open_quiz.items) == 1
    item = open_quiz.items[0]
    assert item.answer == "d) evaporation"
    assert item.question.startswith("Which process")
    assert item.topic_id == "t1" and item.lecture_id == "l1"


def test_a_marked_quiz_is_closed():
    """A quiz already marked must not swallow the next message as an answer."""
    from agent.conversation import PendingAssessment, PendingQuestion, SessionState

    open_quiz = PendingAssessment(items=[PendingQuestion(
        id="q1", question="Which process moves water into the air?", answer="d) evaporation")])
    loaded = _persisted({
        "turn_kind": "study", "subject_id": "bio", "reply_body": "Correct — 1 out of 1.",
        "tasks": [{"type": "grade", "params": {"query": "d) evaporation"}}],
        "grade_result": {"graded": 1},
    }, standing=SessionState(subject_id="bio", pending=open_quiz))
    assert loaded.state.pending is None


def test_a_question_asked_while_a_quiz_is_open_leaves_it_open():
    from agent.conversation import PendingAssessment, PendingQuestion, SessionState

    open_quiz = PendingAssessment(items=[PendingQuestion(
        id="q1", question="Which process moves water into the air?", answer="d) evaporation")])
    loaded = _persisted({
        "turn_kind": "study", "subject_id": "bio", "reply_body": "Evaporation is ...",
        "tasks": [{"type": "explain", "params": {"query": "evaporation"}}],
    }, standing=SessionState(subject_id="bio", pending=open_quiz))
    assert loaded.state.pending is not None
    assert loaded.state.pending.items[0].answer == "d) evaporation"


def test_a_turn_nobody_kept_says_which_part_of_the_request_was_missing(caplog):
    """`remembered: false` says it happened, not whose problem it is: a caller
    that stopped sending a session id, a role nothing is kept for, and a store
    that is down have three different people fixing them."""
    with caplog.at_level("WARNING"):
        _persisted({"session_id": None, "response": "ok", "reply_body": "ok"})
    assert "carried no session_id" in caplog.text


def test_a_role_nothing_is_kept_for_is_not_reported_as_a_fault(caplog):
    with caplog.at_level("INFO"):
        _persisted({"pathway": "parent", "response": "ok", "reply_body": "ok"})
    assert "nothing is kept for a parent" in caplog.text
