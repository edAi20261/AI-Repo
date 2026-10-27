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
    """Questions come back carrying their block id, and mark correctness by index."""
    from agent.graph.subgraphs.assessment_sg import _batch_generate
    llm = _BatchLLM(
        '{"items": ['
        '{"id": "b1", "question": "Q1", "type": "mcq", "options": ["A","B","C","D"], '
        '"correct_indices": [0, 1], "explanation": "e1"},'
        '{"id": "b2", "question": "Q2", "type": "true_false", "answer": "true", '
        '"explanation": "e2"}]}')
    c = _chunk()
    bound = [({"type": "mcq", "topic_id": "tA", "difficulty": "easy"}, c),
             ({"type": "true_false", "topic_id": "tB", "difficulty": "hard"}, c)]
    items, _ = _batch_generate(llm, bound, {"language": "en", "query": "fire"})
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
                ItemDraft(id="b1", question="Q1", type="mcq", options=["A", "B"],
                          correct_indices=[0])])

    class _StructLLM:
        def with_structured_output(self, model_cls, **kwargs):
            assert model_cls is DraftBatch
            return _Structured()

        def invoke(self, _msgs):  # must NOT run on the structured path
            raise AssertionError("plain invoke should not run on the structured path")

    bound = [({"type": "mcq", "topic_id": "tA", "difficulty": "easy"}, _chunk())]
    items, _ = _batch_generate(_StructLLM(), bound, {"language": "en", "query": "x"})
    assert len(items) == 1 and items[0].type == "mcq" and items[0].topic_id == "tA"


def test_a_quiz_comes_with_an_intro_in_the_students_language():
    """The reply printed every question while the platform showed them again on the
    quiz card. The reply now carries one line instead, written for the student even
    when the questions follow the subject's language."""
    from agent.graph.subgraphs.assessment_sg import _batch_generate

    prompts = []

    class _Recording(_BatchLLM):
        def invoke(self, msgs):
            prompts.append(msgs[0].content)
            return super().invoke(msgs)

    llm = _Recording(
        '{"intro": " Here are questions on fire. ", "items": ['
        '{"id": "b1", "question": "Q1", "type": "true_false", "answer": "true"}]}')
    bound = [({"type": "true_false", "topic_id": "tA", "difficulty": "easy"}, _chunk())]
    items, intro = _batch_generate(
        llm, bound, {"language": "en", "reply_language": "ar", "query": "fire"})

    assert len(items) == 1 and intro == "Here are questions on fire."
    assert "written in Arabic" in prompts[0]


def test_the_assessment_hands_its_intro_to_the_reply():
    out = qmod._validate({"items": [_item()], "intro": "Three questions on fire.",
                          "type_mix": {"short_answer": 1}})
    assert out["assessment"]["intro"] == "Three questions on fire."
    empty = qmod._validate({"items": [], "intro": "Three questions on fire.",
                            "type_mix": {"short_answer": 1}})
    assert empty["assessment"]["intro"] == ""


def test_an_mcq_answered_by_letter_is_rejected_not_saved():
    """"answer": "A" against full-sentence options produced a question with no
    correct option — which then passed validation and was saved."""
    from shared.structured import StructuredOutputError
    from agent.graph.subgraphs.assessment_sg import _batch_generate

    llm = _BatchLLM(
        '{"items": [{"id": "b1", "question": "Q1", "type": "mcq", '
        '"options": ["Ice is less dense", "Ice is more dense"], "answer": "A"}]}')
    bound = [({"type": "mcq", "topic_id": "tA", "difficulty": "easy"}, _chunk())]
    with pytest.raises(StructuredOutputError):
        _batch_generate(llm, bound, {"language": "en", "query": "x"})


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
    # An mcq whose options are all wrong is unanswerable, and used to pass.
    assert not _well_formed(_item(type="mcq", options=[
        AssessmentOption(text="A"), AssessmentOption(text="B")]))


def test_enforce_type_mix_caps_per_type():
    items = [
        _item(type="mcq", options=[AssessmentOption(text="A", is_correct=True),
                                   AssessmentOption(text="B")]),
        _item(type="mcq", options=[AssessmentOption(text="A"),
                                   AssessmentOption(text="B", is_correct=True)]),
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


def test_a_quiz_falls_back_to_the_topic_when_the_request_has_nothing_to_match(monkeypatch):
    """'Quiz me on this lesson' produced no questions at all.

    A quiz slot already carries a topic_id, which is an exact filter. Requiring
    the slot's chunks to also clear a similarity floor against the request text
    meant a request with no subject matter in it — "quiz me on this lesson", or
    an English request over an Arabic lesson — retrieved nothing and the student
    got "there isn't enough syllabus material" instead of their quiz.
    """
    from agent.models import ResolvedTopic

    monkeypatch.setattr(
        qmod, "resolve_topics",
        lambda query, subject_id=None, **k: [ResolvedTopic("t9", "Fire", 0.9, "sub1")])
    monkeypatch.setattr(qmod.retrieval, "retrieve", lambda *a, **k: [])
    monkeypatch.setattr(qmod, "get_topic_chunks", lambda *a, **k: [_chunk("t9")])

    graph = build_assessment_sg(config=get_config(), llm=None)
    out = graph.invoke({
        "query": "quiz me on this lesson", "subject_id": "sub1",
        "namespaces": ["syllabus"], "language": "en", "item_source": "generated",
        "type_mix": {"mcq": 1}, "k_per_topic": 1, "adaptive": False,
    })
    assert out["assessment"]["items"], "the lesson's own chunks should still ground a quiz"
    assert out["assessment"]["items"][0]["topic_id"] == "t9"


def test_a_quiz_in_a_lesson_uses_that_lesson_when_the_request_names_no_topic(monkeypatch):
    """An English "quiz me on this lesson" over an Arabic lesson produced nothing.

    Topic resolution is a similarity search over topic labels, so a request that
    names no topic — and a request in a language the labels are not written in —
    resolves nothing, leaves the coverage plan empty, and the student gets no
    questions. Inside one lesson there is no ambiguity about what to ask about.
    """
    from agent.models import ResolvedTopic

    monkeypatch.setattr(qmod, "resolve_topics", lambda *a, **k: [])
    monkeypatch.setattr(
        qmod, "get_lecture_topics",
        lambda lecture_ids, **k: [ResolvedTopic("t9", "السجع", 0.0, "sub1")])
    monkeypatch.setattr(qmod.retrieval, "retrieve", lambda *a, **k: [_chunk("t9")])

    graph = build_assessment_sg(config=get_config(), llm=None)
    out = graph.invoke({
        "query": "give me a quiz on this lesson", "subject_id": "sub1",
        "lecture_ids": ["lec1"], "namespaces": ["syllabus"], "language": "en",
        "item_source": "generated", "type_mix": {"mcq": 1}, "k_per_topic": 1,
        "adaptive": False,
    })
    assert out["assessment"]["items"], "the lesson's own topics should carry the quiz"
    assert out["assessment"]["items"][0]["topic_id"] == "t9"


def _nothing_found_by_text_or_tag(monkeypatch):
    from agent.models import ResolvedTopic

    monkeypatch.setattr(qmod, "resolve_topics", lambda *a, **k: [])
    monkeypatch.setattr(
        qmod, "get_lecture_topics",
        lambda lecture_ids, **k: [ResolvedTopic("t-sub", "Subtraction", 0.0, "sub1")])
    monkeypatch.setattr(qmod.retrieval, "retrieve", lambda *a, **k: [])
    monkeypatch.setattr(qmod, "get_topic_chunks", lambda *a, **k: [])
    monkeypatch.setattr(qmod, "get_topic_labels", lambda ids, **k: {"t-sub": "Subtraction"})
    read = {"lessons": []}

    def whole_lesson(lecture_id, namespaces, *, limit):
        read["lessons"].append(lecture_id)
        return [RetrievedChunk(
            chunk_id="c-sub", score=0.0, text="15 - 7 = 8", lecture_id=lecture_id,
            topic_ids=[], page_from=1, page_to=1, heading_path=None, language="en")]

    monkeypatch.setattr(qmod.retrieval, "whole_lesson", whole_lesson)
    return read


def test_a_quiz_on_a_lesson_whose_text_carries_no_topic_reads_the_lesson(monkeypatch):
    """"Quiz me on this lesson" in Subtraction returned no questions: its passage
    carried no topic after a re-ingest, its topics pointed at no text, and the
    request scored 0.40-0.45 against the passage, under the 0.47 floor."""
    read = _nothing_found_by_text_or_tag(monkeypatch)

    graph = build_assessment_sg(config=get_config(), llm=None)
    out = graph.invoke({
        "query": "quiz me on this lesson", "subject_id": "sub1", "lecture_ids": ["lec-sub"],
        "namespaces": ["syllabus"], "language": "en", "item_source": "generated",
        "type_mix": {"mcq": 1}, "k_per_topic": 1, "adaptive": False,
    })

    assert read["lessons"] == ["lec-sub"]
    item = out["assessment"]["items"][0]
    assert item["topic_id"] == "t-sub" and item["lecture_id"] == "lec-sub"


def test_a_quiz_over_several_lessons_does_not_read_any_one_of_them_whole(monkeypatch):
    read = _nothing_found_by_text_or_tag(monkeypatch)

    graph = build_assessment_sg(config=get_config(), llm=None)
    graph.invoke({
        "query": "quiz me", "subject_id": "sub1", "lecture_ids": ["lec-a", "lec-b"],
        "namespaces": ["syllabus"], "language": "en", "item_source": "generated",
        "type_mix": {"mcq": 1}, "k_per_topic": 1, "adaptive": False,
    })

    assert read["lessons"] == []


def test_a_quiz_across_a_whole_subject_still_needs_the_request_to_resolve(monkeypatch):
    """Without a lesson there is no lesson to fall back to — resolution still rules."""
    monkeypatch.setattr(qmod, "resolve_topics", lambda *a, **k: [])
    called = {"lecture_topics": False}

    def _spy(lecture_ids, **k):
        called["lecture_topics"] = True
        return []

    monkeypatch.setattr(qmod, "get_lecture_topics", _spy)
    monkeypatch.setattr(qmod.retrieval, "retrieve", lambda *a, **k: [])
    graph = build_assessment_sg(config=get_config(), llm=None)
    graph.invoke({
        "query": "quiz me", "subject_id": "sub1", "namespaces": ["syllabus"],
        "language": "en", "item_source": "generated", "type_mix": {"mcq": 1},
        "k_per_topic": 1, "adaptive": False,
    })
    assert not called["lecture_topics"]


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


def test_a_weak_topic_gets_more_of_the_questions_not_just_the_first(monkeypatch):
    """Coverage only ordered topics weakest-first and then dealt the questions round
    robin, so a strong topic got as many questions as a weak one."""
    monkeypatch.setattr(qmod, "get_mastery", lambda uid, ids: [
        {"topic_id": "t1", "status": "strong"},
        {"topic_id": "t2", "status": "weak"}])
    out = _plan_coverage({"topic_ids": ["t1", "t2"], "type_mix": {"mcq": 4},
                          "adaptive": True, "user_id": "u1"})
    topics = [slot["topic_id"] for slot in out["coverage"]]
    assert topics.count("t2") == 3 and topics.count("t1") == 1
    assert topics[0] == "t2"


def test_every_asked_for_topic_keeps_at_least_one_question(monkeypatch):
    monkeypatch.setattr(qmod, "get_mastery", lambda uid, ids: [
        {"topic_id": "a", "status": "weak"}, {"topic_id": "b", "status": "strong"},
        {"topic_id": "c", "status": "strong"}])
    out = _plan_coverage({"topic_ids": ["a", "b", "c"], "type_mix": {"mcq": 5},
                          "adaptive": True, "user_id": "u1"})
    topics = [slot["topic_id"] for slot in out["coverage"]]
    assert set(topics) == {"a", "b", "c"} and topics.count("a") == 3


def test_a_quiz_that_names_no_topic_practises_the_students_weak_spots(monkeypatch):
    """"Give me a practice quiz" resolved no topic, so the questions came from whatever
    the words happened to retrieve instead of what the student needs."""
    from agent.graph.subgraphs.assessment_sg import _resolve

    monkeypatch.setattr(qmod, "resolve_topics", lambda *a, **k: [])
    seen = {}
    monkeypatch.setattr(qmod, "get_mastery", lambda uid, *a, **k: seen.update(k) or [
        {"topic_id": "s", "status": "strong", "score": 0.9},
        {"topic_id": "d", "status": "developing", "score": 0.6},
        {"topic_id": "w", "status": "weak", "score": 0.2},
        {"topic_id": "u", "status": "unknown", "score": 0.1}])
    out = _resolve({"query": "practise my weak spots", "subject_id": "math",
                    "adaptive": True, "user_id": "u1", "focus": "weak"})
    assert out["topic_ids"] == ["w", "u", "d"]
    assert seen["subject_id"] == "math"


def test_with_no_evidence_a_topicless_quiz_is_left_to_retrieval(monkeypatch):
    from agent.graph.subgraphs.assessment_sg import _resolve

    monkeypatch.setattr(qmod, "resolve_topics", lambda *a, **k: [])
    monkeypatch.setattr(qmod, "get_mastery", lambda uid, *a, **k: [])
    out = _resolve({"query": "quiz me", "subject_id": "math", "adaptive": True,
                    "user_id": "teacher-1", "focus": "weak"})
    assert out["topic_ids"] == []


def test_an_off_syllabus_quiz_request_is_not_turned_into_weak_spot_practice(monkeypatch):
    """Nothing resolving is also what an off-syllabus request looks like; only a student
    who asked for their weak spots gets them, the rest still decline on retrieval."""
    from agent.graph.subgraphs.assessment_sg import _resolve

    monkeypatch.setattr(qmod, "resolve_topics", lambda *a, **k: [])
    monkeypatch.setattr(qmod, "get_mastery", lambda uid, *a, **k: [
        {"topic_id": "w", "status": "weak", "score": 0.2}])
    out = _resolve({"query": "quiz me on the French Revolution", "subject_id": "math",
                    "adaptive": True, "user_id": "u1"})
    assert out["topic_ids"] == []


def test_the_planner_can_ask_for_weak_spot_practice():
    from agent.graph.nodes import _validate_assessment_spec

    assert _validate_assessment_spec({"focus": "weak"}).focus == "weak"
    assert _validate_assessment_spec({"focus": "everything"}) is None


def test_a_topic_no_passage_is_tagged_with_is_found_by_its_name(monkeypatch):
    """Three Arabic multiple-choice questions on entrepreneurship came back as one.

    Two of the three topics the request resolved to were carried by no passage's
    tags, so neither the tagged search nor the tag lookup found anything for
    them, their slots were dropped, and the student got a third of the quiz."""
    from agent.models import ResolvedTopic

    topics = [ResolvedTopic("t1", "Understand entrepreneurship", 0.9, "sub1"),
              ResolvedTopic("t2", "Traits of an entrepreneur", 0.8, "sub1"),
              ResolvedTopic("t3", "Entrepreneurs serve society", 0.7, "sub1")]
    monkeypatch.setattr(qmod, "resolve_topics", lambda query, subject_id=None, **k: topics)
    monkeypatch.setattr(qmod, "get_topic_chunks",
                        lambda ids, *a, **k: [_chunk("t1")] if ids == ["t1"] else [])
    monkeypatch.setattr(qmod, "get_topic_labels",
                        lambda ids, **k: {t.topic_id: t.label for t in topics}, raising=False)

    def search(query, namespaces, *, topic_ids=None, **k):
        if topic_ids:
            return []
        named = {"Traits of an entrepreneur": "t2", "Entrepreneurs serve society": "t3"}
        if query in named:
            chunk = _chunk(named[query])
            return [RetrievedChunk(**{**chunk.__dict__, "chunk_id": f"c-{named[query]}"})]
        return []

    monkeypatch.setattr(qmod.retrieval, "retrieve", search)
    graph = build_assessment_sg(config=get_config(), llm=None)
    out = graph.invoke({
        "query": "3 multiple choice questions on entrepreneurship", "subject_id": "sub1",
        "namespaces": ["syllabus"], "language": "ar", "item_source": "generated",
        "type_mix": {"mcq": 3}, "k_per_topic": 2, "adaptive": False,
    })
    assert sorted(i["topic_id"] for i in out["assessment"]["items"]) == ["t1", "t2", "t3"]


def test_a_quiz_on_a_lesson_without_topics_asks_from_more_than_one_passage(monkeypatch):
    """213 published lessons have no topics. A quiz on one searched the request,
    found a single passage, never read the lesson because something had been found,
    and the model wrote no questions from it, so the turn produced nothing."""
    lone = RetrievedChunk(chunk_id="c-1", score=0.6, text="Plants make food.",
                          lecture_id="lec-x", topic_ids=[], page_from=1, page_to=1,
                          heading_path=None, language="en")
    monkeypatch.setattr(qmod.retrieval, "retrieve", lambda *a, **k: [lone])
    read = {"lessons": []}

    def whole_lesson(lecture_id, namespaces, *, limit):
        read["lessons"].append(lecture_id)
        return [RetrievedChunk(
            chunk_id=f"c-{n}", score=0.0, text=f"Passage {n}.", lecture_id=lecture_id,
            topic_ids=[], page_from=n, page_to=n, heading_path=None, language="en")
            for n in range(1, 4)]

    monkeypatch.setattr(qmod.retrieval, "whole_lesson", whole_lesson)
    out = qmod._retrieve_batch({
        "query": "quiz me on this lesson", "subject_id": "sub1", "lecture_ids": ["lec-x"],
        "namespaces": ["syllabus"], "k_per_topic": 4,
        "coverage": [{"topic_id": None, "type": "mcq", "difficulty": "easy"}] * 3,
    })

    assert read["lessons"] == ["lec-x"]
    assert [c.chunk_id for c in out["chunks_by_topic"][""]] == ["c-1", "c-2", "c-3"]


def test_a_lesson_quiz_with_a_passage_for_every_question_does_not_read_the_lesson(monkeypatch):
    found = [RetrievedChunk(chunk_id=f"c-{n}", score=0.6, text=f"Passage {n}.",
                            lecture_id="lec-x", topic_ids=[], page_from=n, page_to=n,
                            heading_path=None, language="en") for n in range(1, 3)]
    monkeypatch.setattr(qmod.retrieval, "retrieve", lambda *a, **k: found)
    monkeypatch.setattr(qmod.retrieval, "whole_lesson",
                        lambda *a, **k: pytest.fail("the lesson was read"))
    out = qmod._retrieve_batch({
        "query": "quiz me", "subject_id": "sub1", "lecture_ids": ["lec-x"],
        "namespaces": ["syllabus"], "k_per_topic": 4,
        "coverage": [{"topic_id": None, "type": "mcq", "difficulty": "easy"}] * 2,
    })
    assert out["chunks_by_topic"][""] == found
