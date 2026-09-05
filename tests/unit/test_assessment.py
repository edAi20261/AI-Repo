"""assessment_sg tests — pure helpers, the stub generate path, adaptivity, and dispatch
routing. No LLM, no DB: retrieval and mastery are monkeypatched."""

from __future__ import annotations

import pytest

from agent.config import get_config
from agent.graph import nodes as nmod
from agent.graph.subgraphs import assessment_sg as qmod
from agent.graph.subgraphs.assessment_sg import (
    _assign_slots,
    _build_item,
    _enforce_type_mix,
    _flatten_type_mix,
    _order_topics_by_weakness,
    _plan_coverage,
    _well_formed,
    build_assessment_sg,
)
from agent.models import AssessmentItem, AssessmentOption, RetrievedChunk


def _chunk(topic="t1", lecture="lec1"):
    return RetrievedChunk(
        chunk_id="c1", score=0.5, text="Fire needs fuel, oxygen and heat.",
        lecture_id=lecture, topic_ids=[topic], page_from=1, page_to=1,
        heading_path="Ch1 > Fire", language="en")


# --- pure helpers -----------------------------------------------------------

def test_flatten_type_mix_orders_and_counts():
    assert _flatten_type_mix({"mcq": 2, "essay": 1}) == ["mcq", "mcq", "essay"]
    assert _flatten_type_mix({"mcq": 0}) == []


def test_order_topics_weakest_first():
    rows = [{"topic_id": "t1", "status": "strong"},
            {"topic_id": "t2", "status": "weak"}]
    assert _order_topics_by_weakness(["t1", "t2"], rows) == ["t2", "t1"]
    # unknown (no row) sorts ahead of strong
    assert _order_topics_by_weakness(["t1", "t3"], rows) == ["t3", "t1"]


def test_assign_slots_cycles_topics_and_handles_none():
    diffs = ["easy", "medium", "hard"]
    slots = _assign_slots(["mcq", "true_false", "mcq"], ["tA", "tB"], diffs)
    assert [s["topic_id"] for s in slots] == ["tA", "tB", "tA"]
    assert [s["difficulty"] for s in slots] == diffs
    assert _assign_slots(["mcq"], [], ["easy"]) == \
        [{"topic_id": None, "type": "mcq", "difficulty": "easy"}]


def test_spread_difficulty_mix_vs_fixed():
    from agent.graph.subgraphs.assessment_sg import _spread_difficulty
    assert _spread_difficulty(4, "mixed") == ["easy", "medium", "hard", "easy"]
    assert _spread_difficulty(3, "hard") == ["hard", "hard", "hard"]


def test_stamp_topic_prefers_slot_topic():
    from agent.graph.subgraphs.assessment_sg import _stamp_topic

    class _C:
        topic_ids = ["chunk-topic"]
    assert _stamp_topic(_C(), "slot-topic") == "slot-topic"   # slot wins
    assert _stamp_topic(_C(), None) == "chunk-topic"          # fallback


class _BatchLLM:
    def __init__(self, content):
        self._content = content

    def invoke(self, _msgs):
        class _R:
            pass
        r = _R()
        r.content = self._content
        return r


def test_batch_generate_maps_items_and_multicorrect():
    from agent.graph.subgraphs.assessment_sg import _batch_generate
    llm = _BatchLLM(
        '{"items": ['
        '{"question": "Q1", "type": "mcq", "options": ["A","B","C","D"], '
        '"answer": ["A","B"], "explanation": "e1"},'
        '{"question": "Q2", "type": "true_false", "answer": "true", "explanation": "e2"}]}')
    c = _chunk()
    bound = [({"type": "mcq", "topic_id": "tA", "difficulty": "easy"}, c),
             ({"type": "true_false", "topic_id": "tB", "difficulty": "hard"}, c)]
    items = _batch_generate(llm, bound, {"language": "en", "query": "fire"})
    assert len(items) == 2
    assert items[0].topic_id == "tA" and items[0].lecture_id == "lec1"
    assert items[0].answer == ["A", "B"]              # multi-correct answer preserved
    assert items[0].multiple_correct                  # two options flagged correct
    assert [o.text for o in items[0].options if o.is_correct] == ["A", "B"]
    assert items[0].difficulty == "easy"
    assert items[1].type == "true_false" and items[1].topic_id == "tB"


def test_batch_generate_uses_structured_output():
    from agent.graph.subgraphs.assessment_sg import _batch_generate
    from agent.models import DraftBatch, ItemDraft

    class _Structured:
        def invoke(self, _msgs):
            return DraftBatch(items=[
                ItemDraft(question="Q1", type="mcq", options=["A", "B"], answer="A")])

    class _StructLLM:
        def with_structured_output(self, model_cls):
            assert model_cls is DraftBatch
            return _Structured()

        def invoke(self, _msgs):  # must NOT run on the structured path
            raise AssertionError("plain invoke should not run on the structured path")

    bound = [({"type": "mcq", "topic_id": "tA", "difficulty": "easy"}, _chunk())]
    items = _batch_generate(_StructLLM(), bound, {"language": "en", "query": "x"})
    assert len(items) == 1 and items[0].type == "mcq" and items[0].topic_id == "tA"


def test_batch_generate_raises_on_error():
    # A runtime LLM failure propagates (no placeholder items); the generate node degrades.
    from agent.graph.subgraphs.assessment_sg import _batch_generate

    class _BadLLM:
        def invoke(self, _msgs):
            raise RuntimeError("boom")

    c = _chunk()
    bound = [({"type": "mcq", "topic_id": "tA", "difficulty": "easy"}, c)]
    with pytest.raises(RuntimeError):
        _batch_generate(_BadLLM(), bound, {"language": "en", "query": "x"})


def _item(**over):
    base = {"type": "short_answer", "question": "q", "answer": "a",
            "topic_id": "t1", "lecture_id": "l1"}
    base.update(over)
    return AssessmentItem(**base)


def test_build_item_enforces_invariant_12():
    # Missing lecture_id -> None (invariant 12 enforced at construction).
    assert _build_item(
        {"type": "short_answer", "topic_id": "t1", "lecture_id": None, "difficulty": "easy"},
        {"question": "q", "answer": "a"}) is None
    # No question -> None.
    assert _build_item(
        {"type": "short_answer", "topic_id": "t1", "lecture_id": "l1", "difficulty": "easy"},
        {"answer": "a"}) is None


def test_well_formed_type_rules():
    assert _well_formed(_item())                                   # short_answer + answer
    assert not _well_formed(_item(type="short_answer", answer=None))  # no answer
    assert _well_formed(_item(type="mcq", options=[
        AssessmentOption(text="A"), AssessmentOption(text="B", is_correct=True)]))
    assert not _well_formed(_item(type="mcq", options=[AssessmentOption(text="A")]))  # <2


def test_enforce_type_mix_caps_per_type():
    items = [
        _item(type="mcq", options=[AssessmentOption(text="A"), AssessmentOption(text="B")]),
        _item(type="mcq", options=[AssessmentOption(text="A"), AssessmentOption(text="B")]),
        _item(type="true_false", answer="true"),
        _item(type="mcq", options=[AssessmentOption(text="A")]),  # <2 options -> dropped
    ]
    kept = _enforce_type_mix(items, {"mcq": 1, "true_false": 1})
    assert sum(1 for i in kept if i.type == "mcq") == 1            # capped
    assert sum(1 for i in kept if i.type == "true_false") == 1


# --- plan_coverage adaptivity ----------------------------------------------

def test_plan_coverage_weights_weak_topics(monkeypatch):
    monkeypatch.setattr(qmod, "get_mastery", lambda uid, ids: [
        {"topic_id": "t1", "status": "strong"},
        {"topic_id": "t2", "status": "weak"}])
    out = _plan_coverage({
        "topic_ids": ["t1", "t2"], "type_mix": {"mcq": 2}, "adaptive": True,
        "user_id": "u1"})
    assert out["coverage"][0]["topic_id"] == "t2"  # weak topic gets the first slot


# --- stub generate end-to-end ----------------------------------------------

def test_assessment_sg_stub_generates_mixed_items_with_ids(monkeypatch):
    monkeypatch.setattr(qmod.retrieval, "retrieve", lambda *a, **k: [_chunk()])
    graph = build_assessment_sg(config=get_config(), llm=None)  # no LLM -> stub items
    out = graph.invoke({
        "query": "fire triangle", "subject_id": "sub1",
        "topic_ids": ["t1"], "namespaces": ["syllabus"], "language": "en",
        "item_source": "generated",
        "type_mix": {"mcq": 2, "true_false": 1, "short_answer": 0, "essay": 0},
        "k_per_topic": 2, "adaptive": False,
    })
    assessment = out["assessment"]
    assert assessment["count"] == 3 and assessment["complete"]
    assert assessment["type_mix"] == {"mcq": 2, "true_false": 1}
    for item in assessment["items"]:
        assert item["topic_id"] == "t1" and item["lecture_id"] == "lec1"  # inv 12


def test_assessment_sg_resolves_topics_when_absent(monkeypatch):
    called = {"resolve": False}

    def fake_resolve(query, subject_id=None, **k):
        called["resolve"] = True
        from agent.models import ResolvedTopic
        return [ResolvedTopic("t9", "Fire", 0.9, "sub1")]

    monkeypatch.setattr(qmod, "resolve_topics", fake_resolve)
    monkeypatch.setattr(qmod.retrieval, "retrieve", lambda *a, **k: [_chunk("t9")])
    graph = build_assessment_sg(config=get_config(), llm=None)
    out = graph.invoke({
        "query": "fire", "subject_id": "sub1", "namespaces": ["syllabus"],
        "language": "en", "item_source": "generated",
        "type_mix": {"mcq": 1}, "k_per_topic": 1, "adaptive": False,
    })
    assert called["resolve"]  # invariant 8: resolve ran because topic_ids was empty
    assert out["assessment"]["items"][0]["topic_id"] == "t9"


# --- dispatch wiring --------------------------------------------------------

def test_dispatch_routes_assessment_and_forwards_user_id():
    sends = nmod.dispatch({
        "tasks": [{"type": "assessment", "params": {"query": "fire"}, "task_id": "x"}],
        "namespaces": ["syllabus"], "language": "en", "subject_id": "sub1",
        "user_id": "u1",
    })
    assert len(sends) == 1
    assert sends[0].node == "assessment_adapter"
    assert sends[0].arg["user_id"] == "u1"
