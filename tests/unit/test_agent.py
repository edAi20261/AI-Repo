"""Agent runtime-graph unit tests. No DB, no API keys: tools are monkeypatched and
the chat model is forced off so nodes take their deterministic grounded path."""

from __future__ import annotations

from typing import ClassVar

import pytest

from agent import retrieval as rmod
from agent.config import get_config
from agent.graph import build_graph
from agent.graph import nodes as nmod
from agent.graph.nodes import (
    _infer_subject,
    _plan_tasks,
    _prereq_line,
    _renumber_citations,
    _sources_block,
    build_load_context,
)
from agent.graph.pathways import build_namespaces, resolve_style
from agent.models import Citation, PrereqTopic, ResolvedTopic, RetrievedChunk
from agent.tools import AuthorizationError, assert_user_type

# --- utils ------------------------------------------------------------------

# --- pathways ---------------------------------------------------------------

def test_build_namespaces():
    assert build_namespaces() == ["syllabus"]
    assert build_namespaces(session_id="s1") == ["syllabus", "session_s1"]


def test_resolve_style_teacher_vs_student():
    assert resolve_style("teacher").tone == "professional"
    assert resolve_style("student").voice == "neutral"


# --- load_context -----------------------------------------------------------

def test_load_context_builds_namespaces_and_style():
    load_context = build_load_context(get_config())
    out = load_context({"pathway": "student", "user_id": "u1",
                        "user_type": "student", "subject_id": "sub1"})
    assert out["namespaces"] == ["syllabus"]
    assert out["subject_id"] == "sub1"
    assert out["style"].voice == "neutral"


def test_load_context_teacher_style():
    load_context = build_load_context(get_config())
    out = load_context({"pathway": "teacher", "user_id": "t1", "user_type": "teacher"})
    assert out["style"].tone == "professional"


# --- tool-layer auth --------------------------------------------------------

def test_assert_user_type_allows_and_rejects():
    assert_user_type({"user_type": "student"}, "student")           # ok
    assert_user_type({"user_type": "teacher"}, "student", "teacher")  # ok
    with pytest.raises(AuthorizationError):
        assert_user_type({"user_type": "student"}, "teacher")
    with pytest.raises(AuthorizationError):
        assert_user_type(None, "student")  # missing ctx


# --- planner ----------------------------------------------------------------

class _FakeLLM:
    def __init__(self, content):
        self._content = content

    def invoke(self, _messages):
        class _Msg:
            content = self._content
        return _Msg()


def test_plan_tasks_uses_llm_json_within_allowlist():
    llm = _FakeLLM('{"tasks": [{"type": "summarize", "query": "cells"}]}')
    tasks, lang = _plan_tasks("summarize cells", ["explain", "summarize"], llm)
    assert tasks == [{"type": "summarize", "query": "cells",
                      "level": "intermediate", "length": "medium"}]
    assert lang is None  # no explicit language request


def test_plan_tasks_injects_role_and_languages():
    # The planner prompt is filled with the caller's role and the configured
    # language set (both parametrised, not hardcoded).
    class _CaptureLLM:
        prompt = None

        def invoke(self, messages):
            _CaptureLLM.prompt = messages[0].content
            class _Msg:
                content = '{"tasks": [{"type": "explain", "query": "x"}]}'
            return _Msg()

    _plan_tasks("x", ["explain"], _CaptureLLM(), role="teacher", languages="ar, en")
    assert "teacher" in _CaptureLLM.prompt
    assert "ar, en" in _CaptureLLM.prompt


def test_plan_tasks_uses_structured_output():
    from agent.models import PlannedTask, PlannerOutput

    class _Structured:
        def invoke(self, _msgs):
            return PlannerOutput(tasks=[PlannedTask(type="summarize", query="cells")])

    class _StructLLM:
        def with_structured_output(self, model_cls):
            assert model_cls is PlannerOutput
            return _Structured()

        def invoke(self, _msgs):  # must NOT be used when structured output works
            raise AssertionError("plain invoke should not run on the structured path")

    tasks, _ = _plan_tasks("summarize cells", ["explain", "summarize"], _StructLLM())
    assert tasks[0]["type"] == "summarize" and tasks[0]["query"] == "cells"


def test_plan_tasks_falls_back_to_json_without_structured_output():
    # A model without `with_structured_output` -> plain call + extract_json fallback.
    llm = _FakeLLM('{"tasks": [{"type": "explain", "query": "cells"}]}')
    tasks, _ = _plan_tasks("explain cells", ["explain"], llm)
    assert tasks[0]["type"] == "explain"


def test_plan_tasks_drops_disallowed_and_falls_back():
    llm = _FakeLLM('{"tasks": [{"type": "grade", "query": "x"}]}')  # not allowed
    tasks, _ = _plan_tasks("do something", ["explain", "summarize"], llm)
    assert tasks[0]["type"] == "explain"  # deterministic fallback


def test_plan_tasks_raises_on_llm_failure():
    # A real LLM failure PROPAGATES (no silently-picked task); the plan node degrades.
    class _BadLLM:
        def invoke(self, _messages):
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        _plan_tasks("x", ["explain"], _BadLLM())


def test_plan_degrades_on_planner_failure():
    class _BadLLM:
        def invoke(self, _messages):
            raise RuntimeError("boom")

    plan = nmod.build_plan(get_config(), _BadLLM())
    out = plan({"pathway": "student", "message": "hi", "language": "en"})
    assert out["tasks"] == [] and "try again" in out["response"].lower()


def test_dispatch_routes_empty_tasks_to_persist():
    # Planner degraded (no tasks) -> dispatch sends the turn straight to the end.
    assert nmod.dispatch({"tasks": []}) == "persist"


def test_plan_tasks_no_llm_is_deterministic():
    tasks, lang = _plan_tasks("explain photosynthesis", ["explain", "summarize"], None)
    assert tasks == [{"type": "explain", "query": "explain photosynthesis",
                      "level": "intermediate", "length": "medium"}]
    assert lang is None


def test_infer_subject_picks_top_topic_subject(monkeypatch):
    from agent.graph import nodes as nmod
    monkeypatch.setattr(
        nmod, "resolve_topics",
        lambda q, sid=None, **k: [ResolvedTopic("t1", "Fractions", 0.8, "subjX")])
    assert _infer_subject("what is a fraction", object()) == "subjX"
    # Nothing clears the threshold -> None (caller searches unscoped).
    monkeypatch.setattr(nmod, "resolve_topics", lambda q, sid=None, **k: [])
    assert _infer_subject("totally unrelated", object()) is None


class _SubjectCfg:
    """Minimal config for plan() with a controllable default subject."""
    def __init__(self, default_subject_id=None):
        self._default = default_subject_id

    def get(self, key, default=None):
        if key == "agent.default_subject_id":
            return self._default
        if key == "agent.default_tasks":
            return ["explain"]
        if key == "language.supported":
            return ["ar", "en", "fr", "de"]
        return default  # agent.pathways.* -> None -> falls back to default_tasks


def test_plan_prefers_inferred_subject_over_default(monkeypatch):
    # A default is set, but the message infers a subject -> inference wins.
    monkeypatch.setattr(
        nmod, "resolve_topics",
        lambda q, sid=None, **k: [ResolvedTopic("t1", "Fractions", 0.8, "subjINFER")])
    plan = nmod.build_plan(_SubjectCfg(default_subject_id="subjDEFAULT"), None)
    out = plan({"pathway": "student", "message": "what is a fraction"})
    assert out["subject_id"] == "subjINFER"


def test_plan_falls_back_to_default_when_inference_empty(monkeypatch):
    # Nothing clears the resolve threshold -> the default is the last resort.
    monkeypatch.setattr(nmod, "resolve_topics", lambda q, sid=None, **k: [])
    plan = nmod.build_plan(_SubjectCfg(default_subject_id="subjDEFAULT"), None)
    out = plan({"pathway": "student", "message": "totally unrelated"})
    assert out["subject_id"] == "subjDEFAULT"


def test_plan_keeps_explicit_subject_and_skips_default(monkeypatch):
    # An explicit subject is authoritative: neither inference nor default touch it,
    # and plan does not re-emit it (already in state).
    monkeypatch.setattr(
        nmod, "resolve_topics",
        lambda q, sid=None, **k: [ResolvedTopic("t1", "Fractions", 0.8, "subjINFER")])
    plan = nmod.build_plan(_SubjectCfg(default_subject_id="subjDEFAULT"), None)
    out = plan({"pathway": "student", "message": "explain fractions",
                "subject_id": "subjEXPLICIT"})
    assert "subject_id" not in out


def test_infer_subject_constrained_to_enrolled(monkeypatch):
    # The top topic hit is a subject the student isn't enrolled in; inference must pick the
    # best hit that IS enrolled, never a subject the student can't access.
    monkeypatch.setattr(nmod, "resolve_topics", lambda q, sid=None, **k: [
        ResolvedTopic("t1", "A", 0.9, "subjOther"),
        ResolvedTopic("t2", "B", 0.7, "subjEnrolled")])
    assert _infer_subject("q", _SubjectCfg(), {"subjEnrolled"}) == "subjEnrolled"
    assert _infer_subject("q", _SubjectCfg(), {"subjNope"}) is None   # no enrolled match -> unscoped


def test_plan_constrains_inference_to_enrolled_subjects(monkeypatch):
    monkeypatch.setattr(nmod, "resolve_topics", lambda q, sid=None, **k: [
        ResolvedTopic("t1", "A", 0.9, "subjOther"),
        ResolvedTopic("t2", "B", 0.7, "subjEnrolled")])

    class _Client:
        def get_student_subjects(self, uid):
            return [{"id": "subjEnrolled", "name": "Chem"}]

    plan = nmod.build_plan(_SubjectCfg(), None, _Client())
    out = plan({"pathway": "student", "message": "q", "user_id": "u1"})
    assert out["subject_id"] == "subjEnrolled"        # the non-enrolled top hit was skipped


def test_merge_same_type_collapses_only_explain_summarize():
    from agent.graph.nodes import _merge_same_type
    tasks = [
        {"type": "explain", "query": "X", "level": "basic", "length": "short"},
        {"type": "explain", "query": "Y", "level": "basic", "length": "short"},
        {"type": "grade", "query": "a"},
        {"type": "grade", "query": "b"},
    ]
    out = _merge_same_type(tasks)
    explains = [t for t in out if t["type"] == "explain"]
    grades = [t for t in out if t["type"] == "grade"]
    assert len(explains) == 1 and "X" in explains[0]["query"] and "Y" in explains[0]["query"]
    assert len(grades) == 2                            # grade tasks are never merged


def test_validate_assessment_spec_sanitizes():
    from agent.graph.nodes import _validate_assessment_spec
    assert _validate_assessment_spec(None) is None
    assert _validate_assessment_spec({}) is None
    assert _validate_assessment_spec({"count": 5}).count == 5
    assert _validate_assessment_spec({"types": ["mcq", "bogus"]}).types == ["mcq"]
    assert _validate_assessment_spec({"count": "five"}) is None     # bad type -> None
    assert _validate_assessment_spec({"count": 0}) is None          # out of range -> None
    assert _validate_assessment_spec({"types": ["bogus"]}) is None  # no valid types -> None


def test_assessment_type_mix_distribution():
    from agent.graph.nodes import AssessmentSpec, _assessment_type_mix
    default = {"mcq": 3, "true_false": 1, "short_answer": 1}
    assert _assessment_type_mix(AssessmentSpec(count=4, types=["mcq"]), default) == {"mcq": 4}
    assert _assessment_type_mix(AssessmentSpec(count=3, types=["mcq", "true_false"]), default) == \
        {"mcq": 2, "true_false": 1}
    # types only -> spread the default total across requested types
    assert sum(_assessment_type_mix(AssessmentSpec(types=["mcq", "essay"]), default).values()) == \
        sum(default.values())
    # count only -> spread across the default's types
    assert sum(_assessment_type_mix(AssessmentSpec(count=6), default).values()) == 6


def test_plan_applies_requested_assessment_shape(monkeypatch):
    monkeypatch.setattr(nmod, "resolve_topics",
                        lambda q, sid=None, **k: [ResolvedTopic("t1", "T", 0.9)])
    llm = _FakeLLM('{"tasks": [{"type": "assessment", "query": "fractions", '
                   '"assessment": {"count": 4, "types": ["mcq"]}}]}')
    plan = nmod.build_plan(get_config(), llm)
    out = plan({"pathway": "student", "message": "give me 4 mcqs on fractions",
                "subject_id": "s"})
    qtask = next(t for t in out["tasks"] if t["type"] == "assessment")
    assert qtask["params"]["type_mix"] == {"mcq": 4}


def test_allowed_descriptions_lists_only_allowed():
    # A teacher pathway without "visual" must not see the visual bullet.
    text = nmod._allowed_descriptions(["explain", "grade"])
    assert '"explain"' in text and '"grade"' in text
    assert '"visual"' not in text and '"summarize"' not in text


def test_plan_grade_ref_overrides_nl_reference(monkeypatch):
    # A structured grade_ref on the request wins over the NL-parsed reference.
    monkeypatch.setattr(nmod, "resolve_topics", lambda q, sid=None, **k: [])
    llm = _FakeLLM('{"tasks": [{"type": "grade", "query": "my answer", '
                   '"grade": {"assessment_id": "from-nl", "kind": "quiz"}}]}')
    plan = nmod.build_plan(get_config(), llm)
    out = plan({"pathway": "student", "message": "grade my saved quiz", "subject_id": "s",
                "grade_ref": {"assessment_id": "from-frontend", "kind": "homework"}})
    gtask = next(t for t in out["tasks"] if t["type"] == "grade")
    assert gtask["params"]["grade"]["assessment_id"] == "from-frontend"
    assert gtask["params"]["grade"]["kind"] == "homework"


def test_plan_tasks_extracts_explicit_output_language():
    llm = _FakeLLM('{"output_language": "ar", '
                   '"tasks": [{"type": "explain", "query": "fire triangle"}]}')
    tasks, lang = _plan_tasks("explain the fire triangle, answer in Arabic",
                              ["explain", "summarize"], llm)
    assert tasks[0]["type"] == "explain" and lang == "ar"


# --- compose helpers --------------------------------------------------------

def _pt(topic_id="t0", label="Cells", strength="weak", depth=1):
    return PrereqTopic(topic_id=topic_id, label=label, strength=strength, depth=depth)


def test_prereq_line_bilingual():
    prereqs = [_pt()]
    assert _prereq_line(prereqs, "en") == "Prerequisite topics: Cells"


def test_an_empty_prereq_list_renders_no_heading():
    """A bare "Prerequisite topics:" with nothing after it reads as a bug."""
    assert _prereq_line([], "ar") == ""


def test_an_arabic_prereq_is_shown_under_an_arabic_answer():
    arabic = PrereqTopic(topic_id="t2", label="تحليل النص الشعري",
                         strength="required", depth=1)
    assert _prereq_line([arabic], "ar").startswith("مواضيع تمهيدية")


def test_prereq_and_sources_headers_fr_de():
    prereqs = [_pt()]
    assert _prereq_line(prereqs, "fr").startswith("Sujets prérequis")
    assert _prereq_line(prereqs, "de").startswith("Voraussetzungen")
    cites = [Citation(n=1, lecture_id="lec1", heading_path="Ch1", page_from=3)]
    assert _sources_block(cites, "fr").startswith("Sources:")
    assert _sources_block(cites, "de").startswith("Quellen:")


def test_sources_block_dedupes_and_numbers():
    cites = [
        Citation(n=1, lecture_id="lec1", heading_path="Ch1", page_from=3, page_to=3),
        Citation(n=1, lecture_id="lec1", heading_path="Ch1", page_from=3, page_to=3),
    ]
    block = _sources_block(cites, "en")
    assert block.count("[1]") == 1
    assert "Ch1, p.3" in block


def test_sources_block_renders_readable_lecture_label():
    cites = [Citation(n=1, lecture_id="lec1", subject_name="Science FR",
                      lesson_order=4, lesson_title="Mendel's Laws",
                      page_from=6, page_to=7)]
    block = _sources_block(cites, "en")
    assert "[1] Science FR · Lesson 4: Mendel's Laws, p.6–7" in block


def test_sources_block_lecture_label_partial_and_localized():
    # order but no title
    b1 = _sources_block(
        [Citation(n=1, lecture_id="l", subject_name="Math EN", lesson_order=2,
                  page_from=1)], "en")
    assert "[1] Math EN · Lesson 2, p.1" in b1
    # localized lesson word (fr)
    b2 = _sources_block(
        [Citation(n=1, lecture_id="l", subject_name="Sciences", lesson_order=5,
                  page_from=2)], "fr")
    assert "Leçon 5" in b2
    # no metadata -> heading_path fallback (unchanged behavior)
    b3 = _sources_block([Citation(n=1, lecture_id="l", heading_path="Ch1",
                                  page_from=3)], "en")
    assert "[1] Ch1, p.3" in b3


def test_sources_block_strips_trailing_punctuation():
    # A heading carrying a trailing comma must not render "Oxidation,, p.3".
    cites = [Citation(n=1, lecture_id="lec1", heading_path="Oxidation,", page_from=3)]
    block = _sources_block(cites, "en")
    assert "[1] Oxidation, p.3" in block
    assert ",," not in block


def test_renumber_citations_global_across_tasks():
    # Two tasks each number their own source [1]; compose must renumber globally.
    answers = [
        {"task_id": "a", "text": "Photosynthesis needs light [1]."},
        {"task_id": "b", "text": "Cells are the unit of life [1]."},
    ]
    citations = [
        Citation(n=1, task_id="a", lecture_id="lecA", heading_path="H1", page_from=1),
        Citation(n=1, task_id="b", lecture_id="lecB", heading_path="H2", page_from=2),
    ]
    new_answers, ordered = _renumber_citations(answers, citations)
    assert new_answers[0]["text"].endswith("[1].")
    assert new_answers[1]["text"].endswith("[2].")
    assert [c.n for c in ordered] == [1, 2]


def test_renumber_citations_dedupes_shared_source():
    # The same source cited by both tasks gets one global number; both markers map to it.
    answers = [
        {"task_id": "a", "text": "First [1]."},
        {"task_id": "b", "text": "Second [1]."},
    ]
    shared = {"lecture_id": "lecA", "heading_path": "H1", "page_from": 1}
    citations = [
        Citation(n=1, task_id="a", **shared),
        Citation(n=1, task_id="b", **shared),
    ]
    new_answers, ordered = _renumber_citations(answers, citations)
    assert new_answers[0]["text"].endswith("[1].")
    assert new_answers[1]["text"].endswith("[1].")
    assert len(ordered) == 1


def _chunk(cid, score):
    return RetrievedChunk(
        chunk_id=cid, score=score, text="x", lecture_id="l" + cid, topic_ids=[],
        page_from=1, page_to=1, heading_path="H" + cid, language="en")


def test_retrieve_applies_min_score_floor(monkeypatch):
    # Below-floor chunks are dropped so an off-syllabus query grounds in nothing.
    monkeypatch.setattr(rmod, "rag_search",
                        lambda *a, **k: [_chunk("1", 0.10), _chunk("2", 0.50)])
    kept = rmod.retrieve("q", ["syllabus"], min_score=0.30)
    assert [c.chunk_id for c in kept] == ["2"]
    # No floor -> everything passes through unchanged.
    assert len(rmod.retrieve("q", ["syllabus"], min_score=0.0)) == 2


def test_retrieve_soft_topic_fallback(monkeypatch):
    # A topic-filtered pass that grounds NOTHING is a likely wrong topic guess. Soft retries
    # unscoped; hard leaves it empty (the silent-miss the soft mode exists to prevent).
    def fake(query, namespaces, **k):
        return [] if k.get("topic_ids") else [_chunk("2", 0.9)]
    monkeypatch.setattr(rmod, "rag_search", fake)
    assert rmod.retrieve("q", ["syllabus"], topic_ids=["tX"], topic_filter="hard") == []
    kept = rmod.retrieve("q", ["syllabus"], topic_ids=["tX"], topic_filter="soft")
    assert [c.chunk_id for c in kept] == ["2"]
    # Soft does NOT fall back when the topic-filtered pass already grounded something.
    monkeypatch.setattr(rmod, "rag_search",
                        lambda q, ns, **k: [_chunk("t", 0.9)] if k.get("topic_ids") else [_chunk("u", 0.9)])
    kept = rmod.retrieve("q", ["syllabus"], topic_ids=["tX"], topic_filter="soft")
    assert [c.chunk_id for c in kept] == ["t"]


def test_recursion_limit_wired_from_config():
    graph = build_graph(get_config())
    assert getattr(graph, "config", {}).get("recursion_limit") == 25


def test_subject_facts_and_instructions(monkeypatch):
    from agent.graph.graph import _subject_facts, _subject_instructions
    from agent.models import SubjectFacts
    from agent.prompts import MATH_PROMPT_INSTRUCTIONS

    class _Client:
        def get_subject(self, sid):
            # Extra backend fields must be ignored by the model.
            return ({"language": "ar", "is_math": True, "grade_level": "9"}
                    if sid == "math-ar" else {})

    c = _Client()
    # One lookup feeds both the subject language and the prompt addendum; typed + extras ignored.
    facts = _subject_facts(c, "math-ar")
    assert facts.language == "ar" and facts.is_math is True
    assert _subject_facts(None, "math-ar") == SubjectFacts()   # no backend -> defaults
    assert _subject_facts(c, None) == SubjectFacts()           # no subject -> defaults
    # Math subject -> math addendum; anything else -> empty (append nothing).
    assert _subject_instructions(SubjectFacts(is_math=True)) == MATH_PROMPT_INSTRUCTIONS
    assert _subject_instructions(SubjectFacts(language="en")) == ""


# --- full graph (stub path) -------------------------------------------------

def _patch_tools(monkeypatch):
    from agent.graph import graph as gmod
    monkeypatch.setattr(gmod, "maybe_llm", lambda config, task: None)

    monkeypatch.setattr(
        nmod, "resolve_topics",
        lambda q, sid=None, **k: [ResolvedTopic("t1", "Photosynthesis", 0.9)])

    primary = RetrievedChunk(
        chunk_id="c1", score=0.7, text="Plants convert light into energy.",
        lecture_id="lec1", topic_ids=["t1"], page_from=3, page_to=3,
        heading_path="Ch1 > Photosynthesis", language="en")
    prereq = RetrievedChunk(
        chunk_id="c0", score=0.0, text="A cell is the basic unit of life.",
        lecture_id="lec0", topic_ids=["t0"], page_from=1, page_to=1,
        heading_path="Ch0 > Cells", language="en")

    monkeypatch.setattr(rmod, "rag_search", lambda *a, **k: [primary])
    monkeypatch.setattr(rmod, "get_topic_deps",
                        lambda ids, **k: [PrereqTopic("t0", "Cells", "required", 1)])
    monkeypatch.setattr(rmod, "get_topic_chunks", lambda ids, ns, **k: [prereq])


def test_graph_halts_on_unknown_role(monkeypatch):
    # No valid role -> the graph does NOT run; it returns a sign-in message.
    _patch_tools(monkeypatch)
    graph = build_graph(get_config())
    out = graph.invoke({"message": "explain photosynthesis", "user_type": "guest"})
    assert out["pathway"] is None
    assert "sign in" in out["response"].lower()
    assert "tasks" not in out                     # plan never ran


def test_classify_recognizes_parent(monkeypatch):
    # Parent is a configured pathway -> recognized, not mis-routed to student.
    classify = nmod.build_classify(get_config())
    out = classify({"message": "how is my child doing?", "user_type": "parent"})
    assert out["pathway"] == "parent" and "response" not in out


def test_require_llm_in_prod_fails_without_key(monkeypatch):
    from agent.graph import graph as gmod
    monkeypatch.setattr(gmod, "llm_key_available", lambda config, spec: False)

    class _Cfg:
        app_env = "prod"

        def llm_for(self, task):
            return {}

    with pytest.raises(RuntimeError):
        gmod._require_llm_in_prod(_Cfg())        # prod + no key -> refuse to start
    _Cfg.app_env = "local"
    gmod._require_llm_in_prod(_Cfg())            # dev env -> allowed (stub path)


def test_shipped_config_starts_in_production_with_openrouter_key(monkeypatch):
    from agent.graph import graph as gmod
    from shared.config import Config

    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")

    gmod._require_llm_in_prod(Config.load())


def test_explain_sg_degrades_on_llm_failure(monkeypatch):
    # A runtime LLM failure -> a friendly message, no stub text, no citations.
    from agent.graph.subgraphs.explain_summarize_sg import build_explain_sg

    chunk = RetrievedChunk(chunk_id="c1", score=0.9, text="t", lecture_id="lec1",
                           topic_ids=["t1"], page_from=1, page_to=1, heading_path="H",
                           language="en")
    monkeypatch.setattr(rmod, "retrieve", lambda *a, **k: [chunk])
    monkeypatch.setattr(rmod, "resolve_prereqs", lambda *a, **k: ([], []))

    class _BadLLM:
        def invoke(self, _messages):
            raise RuntimeError("boom")

    sg = build_explain_sg(config=get_config(), llm=_BadLLM())
    out = sg.invoke({"query": "x", "topic_ids": ["t1"], "namespaces": ["syllabus"],
                     "language": "en"})
    assert "couldn't complete" in out["explanation"].lower()
    assert out["citations"] == [] and "pending chat model" not in out["explanation"]


def test_graph_end_to_end_stub_english(monkeypatch):
    _patch_tools(monkeypatch)
    graph = build_graph(get_config())
    out = graph.invoke({"message": "explain photosynthesis",
                        "user_type": "student", "subject_id": "sub1"})

    assert out["pathway"] == "student"
    assert out["language"] == "en"
    assert out["tasks"][0]["params"]["topic_ids"] == ["t1"]
    assert "grounded in" in out["response"]
    assert "Prerequisite topics: Cells" in out["response"]
    assert "Sources" in out["response"]
    assert out["citations"] and out["citations"][0].lecture_id == "lec1"


def test_graph_routes_teacher_and_arabic(monkeypatch):
    _patch_tools(monkeypatch)
    graph = build_graph(get_config())
    out = graph.invoke({"message": "لخص لي الدرس", "user_type": "teacher",
                        "subject_id": "sub1"})
    assert out["pathway"] == "teacher"
    assert out["language"] == "ar"
    assert out["response"]


# --- compose: grade line, visual caption, decline suppression, render error ----

def test_grade_line_shows_correct_answer_and_explanation():
    line = nmod._grade_line({
        "summary": {"score": 0, "max_score": 1},
        "graded": [{"correct": False, "feedback": "Not quite.",
                    "correct_answer": "6", "explanation": "b = sqrt(a*c)."}],
    }, "en")
    assert "Correct answer: 6" in line and "Why: b = sqrt(a*c)." in line


def test_grade_line_omits_key_when_correct():
    line = nmod._grade_line({
        "summary": {"score": 1, "max_score": 1},
        "graded": [{"correct": True, "feedback": "Right.",
                    "correct_answer": "6", "explanation": "because."}],
    }, "en")
    assert "Correct answer" not in line and "Why" not in line


def test_visual_caption_by_kind_and_language():
    assert nmod._visual_caption([{"kind": "mindmap", "s3_uri": "u"}], "en") == "Here's your mind map:"
    assert nmod._visual_caption([{"kind": "flashcards", "s3_uri": "u"}], "ar").startswith("إليك")
    # slides carry their own text -> no caption
    assert nmod._visual_caption([{"kind": "slides", "text": "S"}], "en") == ""


def test_compose_render_error_over_decline():
    compose = nmod.build_compose(get_config(), None)
    out = compose({"answers": [], "visual_error": True, "language": "en"})
    assert "went wrong" in out["response"].lower()
    assert "syllabus material" not in out["response"].lower()


def test_compose_suppresses_prereqs_on_decline():
    compose = nmod.build_compose(get_config(), None)
    out = compose({
        "answers": [{"type": "explain", "text": "The context does not cover this.",
                     "grounded": False}],
        "prereq_topics": [PrereqTopic(topic_id="t1", label="Fractions",
                                      strength="strong", depth=1)],
        "citations": [], "language": "en",
    })
    assert "Prerequisite topics" not in out["response"] and "Fractions" not in out["response"]


def test_an_unreadable_plan_does_not_become_an_explain_task():
    """A parse failure used to be indistinguishable from "explain this".

    A student who asked to be graded got an explanation instead, with nothing in
    the response saying the planner never ran.
    """
    from agent.graph.nodes import build_plan
    from shared.config import Config

    class _Prose:
        class _Structured:
            def invoke(self, _msgs):
                raise ValueError("Failed to parse PlannerOutput from completion")

        def with_structured_output(self, schema, **kwargs):
            return self._Structured()

        def invoke(self, _msgs):
            class _Reply:
                content = "Sure, I can help with that!"
                additional_kwargs: ClassVar[dict] = {}
                response_metadata: ClassVar[dict] = {}
            return _Reply()

    config = Config.load()
    node = build_plan(config, _Prose())
    out = node({"pathway": "student", "message": "grade my answer: 42", "language": "en"})

    assert out["tasks"] == [], "a failed plan must not invent a task"
    assert out["response"], "the student must be told the turn degraded"
