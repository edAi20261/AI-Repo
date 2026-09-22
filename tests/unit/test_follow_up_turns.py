"""A turn that continues reuses what the last answer was built from.

The three pieces — telling a follow-up from a new question, remembering the
evidence, fetching it back — meet here. This is the behaviour a student sees:
ask about evaporation, say "explain more simply", and be answered from the
evaporation pages rather than from wherever those words happen to land.
"""

from __future__ import annotations

from agent.evidence import EvidenceSet
from agent.models import RetrievedChunk
from agent.search_decision import SearchDecision
from shared.config import get_config


def _decides(monkeypatch, sg, *, opens: bool, question: str = "carried question"):
    """Fix what the decision says, so these tests measure the wiring under it.

    Whether the decision itself is right on real phrasings is not something a
    fake can answer — that is measured against a live model over whole
    conversations in scripts/eval_conversations.py.
    """
    monkeypatch.setattr(sg, "decide", lambda *a, **k: SearchDecision(
        opens_new_subject=opens, question=question))


def _chunk(chunk_id: str, text: str = "Evaporation is …") -> RetrievedChunk:
    return RetrievedChunk(chunk_id=chunk_id, score=0.7, text=text,
                          heading_path="", page_from=31, page_to=31,
                          lecture_id="lec1", language="en", topic_ids=[])


def _retrieve(monkeypatch, *, message, evidence, searched=None, remembered=None,
              opens=False):
    """Run the retrieve node with the search and the lookup both observable."""
    from agent.graph.subgraphs import explain_summarize_sg as sg

    _decides(monkeypatch, sg, opens=opens, question=message)
    calls = {"searched": 0, "looked_up": []}

    def _search(*a, **k):
        calls["searched"] += 1
        return searched if searched is not None else []

    def _by_id(chunk_ids, **k):
        calls["looked_up"] = list(chunk_ids)
        return remembered if remembered is not None else []

    monkeypatch.setattr(sg.retrieval, "retrieve", _search)
    monkeypatch.setattr(sg, "get_chunks_by_id", _by_id)

    state = {
        "message": message, "query": message, "evidence": evidence,
        "namespaces": ["syllabus"], "subject_id": "sub1", "k": 8,
        "min_score": get_config().get("agent.min_score"),
    }
    return sg._retrieve(state, get_config()), calls


def test_a_follow_up_reuses_the_last_answers_passages_without_searching(monkeypatch):
    """The whole point. "Explain more simply" names nothing to search for, and
    the right passages are the ones already on the table."""
    evidence = EvidenceSet(chunk_ids=["c1", "c2"], query="What is evaporation?")
    out, calls = _retrieve(
        monkeypatch, message="I didn't understand. Explain more simply.",
        evidence=evidence, remembered=[_chunk("c1"), _chunk("c2")])

    assert calls["searched"] == 0, "a follow-up must not search"
    assert calls["looked_up"] == ["c1", "c2"]
    assert [c.chunk_id for c in out["chunks"]] == ["c1", "c2"]


def test_a_new_question_searches_the_way_it_always_did(monkeypatch):
    evidence = EvidenceSet(chunk_ids=["c1"], query="What is evaporation?")
    out, calls = _retrieve(monkeypatch, message="What about tornadoes?",
                           evidence=evidence, searched=[_chunk("t9")], opens=True)

    assert calls["searched"] == 1
    assert calls["looked_up"] == []
    assert [c.chunk_id for c in out["chunks"]] == ["t9"]


def test_the_first_turn_searches(monkeypatch):
    _, calls = _retrieve(monkeypatch, message="explain more", evidence=None,
                         searched=[_chunk("c1")], opens=True)

    assert calls["searched"] == 1


def test_a_follow_up_whose_evidence_is_gone_searches_rather_than_answering_from_nothing(
        monkeypatch):
    """A lecture re-ingested between turns re-mints every chunk_id. Reusing
    pointers that no longer resolve would answer from an empty context; the
    turn falls back to the search it would have done anyway."""
    evidence = EvidenceSet(chunk_ids=["gone"], query="What is evaporation?")
    out, calls = _retrieve(monkeypatch, message="explain more simply",
                           evidence=evidence, remembered=[], searched=[_chunk("c1")])

    assert calls["looked_up"] == ["gone"]
    assert calls["searched"] == 1, "nothing to reuse means search"
    assert [c.chunk_id for c in out["chunks"]] == ["c1"]


def test_the_words_the_student_typed_reach_the_turn_that_classifies_them():
    """The planner rewrites a message into a search query — "I didn't
    understand, explain more simply" becomes "explain evaporation simply".
    Deciding whether a turn is a follow-up from that rewrite is deciding from
    the wrong text: the rewrite always names a subject, because that is what a
    rewrite is for."""
    from agent.graph import nodes as nmod

    sends = nmod.dispatch({
        "tasks": [{"type": "explain", "params": {"query": "explain evaporation simply"},
                   "task_id": "x"}],
        "message": "I didn't understand. Explain more simply.",
        "namespaces": ["syllabus"], "language": "en", "subject_id": "sub1",
        "user_id": "u1", "session_id": "s1", "pathway": "student",
    })
    assert sends[0].arg["message"] == "I didn't understand. Explain more simply."


# --- small talk is only small talk at the start of a conversation ------------

def _route(monkeypatch, *, message, intent, has_evidence):
    """The route node, with the classifier and the evidence store observable."""
    from agent.graph import nodes as nmod
    from agent.intent import IntentDecision

    class _Router:
        def classify(self, _m):
            return IntentDecision(intent=intent, source="similarity", confidence=0.76)

    class _Store:
        def __init__(self, **_kw):
            pass

        def recall(self):
            return EvidenceSet(chunk_ids=["c1"], query="evaporation") if has_evidence else None

    monkeypatch.setattr(nmod.IntentRouter, "from_config",
                        classmethod(lambda cls, *a, **k: _Router()))
    monkeypatch.setattr(nmod, "EvidenceStore", _Store)
    route = nmod.build_route(get_config(), embedder=None)
    return route({"message": message, "language": "en",
                  "user_id": "u1", "session_id": "s1", "pathway": "student"})


def test_asking_for_an_example_mid_lesson_is_not_a_question_about_the_tutor(monkeypatch):
    """"Can you give me an example?" sits 0.76 from "can you help me" in the
    small-talk seeds, so it was answered with the capability blurb and no
    citations — in the middle of a lesson about evaporation. Mid-conversation
    those words are about the lesson, not about the tutor."""
    out = _route(monkeypatch, message="Can you give me an example?",
                 intent="meta", has_evidence=True)

    assert out["intent"] == "question", "a follow-up must reach the syllabus"
    assert "response" not in out, "it must not be answered before retrieval"


def test_the_same_words_opening_a_conversation_are_still_small_talk(monkeypatch):
    """With nothing answered yet there is nothing to give an example of."""
    out = _route(monkeypatch, message="Can you give me an example?",
                 intent="meta", has_evidence=False)

    assert out["intent"] == "meta"
    assert out["response"]


def test_a_greeting_is_a_greeting_whenever_it_arrives(monkeypatch):
    """Hello is not a follow-up. Only the intents that could be asking about
    the lesson are reconsidered."""
    out = _route(monkeypatch, message="thanks!", intent="thanks", has_evidence=True)

    assert out["intent"] == "thanks"
    assert out["response"]


# --- what the answering model is asked --------------------------------------

def test_a_follow_up_is_answered_about_the_subject_it_follows(monkeypatch):
    """Retrieval was fixed and generation still failed: the planner rewrote
    "Can you give me an example?" into the standalone query "example", and the
    tutor answered "the context is insufficient to explain the concept of
    example" — over the evaporation passages it had correctly reused.

    A follow-up carries its subject from the turn before it. The model is
    asked about that subject, in the student's own words."""
    from agent.graph.subgraphs import explain_summarize_sg as sg

    evidence = EvidenceSet(chunk_ids=["c1"], query="What is evaporation?")
    monkeypatch.setattr(sg, "get_chunks_by_id", lambda ids, **k: [_chunk("c1")])
    _decides(monkeypatch, sg, opens=False,
             question="Give an example of evaporation")

    out = sg._retrieve({
        "message": "Can you give me an example?",
        "query": "example",                     # what the planner made of it
        "evidence": evidence, "namespaces": ["syllabus"], "k": 8,
    }, get_config())

    assert out["query"] == "Give an example of evaporation", (
        "the answering model is asked the restated question, not the planner's "
        "subjectless rewrite")


def test_a_new_question_is_answered_about_itself(monkeypatch):
    from agent.graph.subgraphs import explain_summarize_sg as sg

    searched = {}

    def _search(query, *a, **k):
        searched["q"] = query
        return [_chunk("t9")]

    monkeypatch.setattr(sg.retrieval, "retrieve", _search)
    _decides(monkeypatch, sg, opens=True, question="How do tornadoes form?")
    out = sg._retrieve({
        "message": "What about tornadoes?", "query": "tornadoes",
        "evidence": EvidenceSet(chunk_ids=["c1"], query="evaporation"),
        "namespaces": ["syllabus"], "k": 8,
    }, get_config())
    assert searched["q"] == "How do tornadoes form?", (
        "an opening turn is searched for what it asked, standing on its own")
    assert [c.chunk_id for c in out["chunks"]] == ["t9"]


# --- a follow-up inherits where the answer came from ------------------------

def test_a_follow_up_keeps_the_subject_the_answer_came_from(monkeypatch):
    """Subject and topics were re-derived from every message. "Can you explain
    more?" names no subject, so an English science answer was followed by
    prerequisite topics in Arabic and sources from a lesson about Jabir ibn
    Hayyan — the wrong corpus entirely, under the right explanation."""
    from agent.graph.subgraphs import explain_summarize_sg as sg

    monkeypatch.setattr(sg, "get_chunks_by_id", lambda ids, **k: [_chunk("c1")])
    _decides(monkeypatch, sg, opens=False)
    out = sg._retrieve({
        "message": "Can you explain more?", "query": "more",
        "evidence": EvidenceSet(chunk_ids=["c1"], query="What is evaporation?",
                                subject_id="science", topic_ids=["t1", "t2"]),
        "subject_id": "arabic",          # what this turn's words inferred
        "namespaces": ["syllabus"], "k": 8,
    }, get_config())

    assert out["subject_id"] == "science", "the subject travels with the evidence"
    assert out["topic_ids"] == ["t1", "t2"], "and so do the topics the prereqs walk"
