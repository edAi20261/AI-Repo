"""Agent runtime-graph unit tests. No DB, no API keys: tools are monkeypatched and
the chat model is forced off so nodes take their deterministic grounded path."""

from __future__ import annotations

import pytest

from agent import retrieval as rmod
from agent.config import get_config
from agent.graph import build_graph
from agent.graph import nodes as nmod
from agent.graph.nodes import (
    _renumber_citations,
    build_load_context,
)
from agent.graph.pathways import build_namespaces, resolve_style
from agent.models import Citation, PrereqTopic, ResolvedTopic, RetrievedChunk
from agent.persona.voice import PersonaDirectory, Voices
from agent.tools import AuthorizationError, assert_user_type

NO_PERSONAS = PersonaDirectory(None, ttl_seconds=0)
NO_VOICES = Voices(NO_PERSONAS, per_turn=3, embedder=lambda: None)

# --- utils ------------------------------------------------------------------

# --- pathways ---------------------------------------------------------------

def test_build_namespaces():
    assert build_namespaces() == ["syllabus"]
    assert build_namespaces(session_id="s1") == ["syllabus", "session_s1"]


def test_resolve_style_teacher_vs_student():
    assert resolve_style("teacher").tone == "professional"


# --- load_context -----------------------------------------------------------

def test_load_context_builds_namespaces_and_style():
    load_context = build_load_context(get_config(), NO_PERSONAS)
    out = load_context({"pathway": "student", "user_id": "u1",
                        "user_type": "student", "subject_id": "sub1"})
    assert out["namespaces"] == ["syllabus"]
    assert out["subject_id"] == "sub1"


def test_load_context_teacher_style():
    load_context = build_load_context(get_config(), NO_PERSONAS)
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


# --- understand ---------------------------------------------------------------

class _Subjects:
    """Subject facts without a backend."""

    def __init__(self, **facts):
        self._facts = facts

    def facts(self, subject_id):
        from agent.models import SubjectFacts
        return SubjectFacts(**self._facts.get(subject_id, {})) if subject_id else SubjectFacts()


def _reading(**fields):
    from agent.models import TurnUnderstanding
    fields.setdefault("needs_syllabus", True)
    return TurnUnderstanding(**fields)


def _understand(monkeypatch, reading, *, state=None, subjects=None, llm=object()):
    captured = {}

    def fake_call(schema, messages, **kwargs):
        captured["prompt"] = messages[0].content
        captured["validate"] = kwargs.get("validate")
        if isinstance(reading, Exception):
            raise reading
        return reading

    monkeypatch.setattr(nmod, "call_structured", fake_call)
    node = nmod.build_understand(get_config(), llm, subjects or _Subjects())
    turn = {"pathway": "student", "message": "hi", "history": "", "conversation": {}}
    turn.update(state or {})
    return node(turn), captured


def test_a_social_message_is_answered_without_searching_the_syllabus(monkeypatch):
    """A greeting used to be planned as an explain task and searched for in the book."""
    out, _ = _understand(monkeypatch, _reading(needs_syllabus=False, message_language="en"))
    assert out["turn_kind"] == "conversation"
    assert out["planned_tasks"] == []
    assert nmod._after_understand(out) == "converse"


def test_a_follow_up_is_read_against_the_conversation(monkeypatch):
    """"Give me an example" names nothing; the model reads it with the history
    and writes a query that names what it refers to."""
    from agent.models import PlannedTask
    reading = _reading(tasks=[PlannedTask(type="explain", query="examples of evaporation")],
                       continues_previous=True, message_language="en")
    out, captured = _understand(monkeypatch, reading, state={
        "message": "Give me an example.",
        "history": "Student: explain evaporation\nTutor: Evaporation is ..."})
    assert "Tutor: Evaporation is" in captured["prompt"]
    assert out["planned_tasks"][0]["query"] == "examples of evaporation"
    assert nmod._after_understand(out) == "plan"


def test_asking_for_a_language_sets_it_for_the_rest_of_the_conversation(monkeypatch):
    from agent.models import LanguageRequest
    out, _ = _understand(monkeypatch, _reading(
        needs_syllabus=False, message_language="ar",
        language_request=LanguageRequest(action="set", code="ar")))
    assert out["requested_language"] == "ar"
    assert out["language"] == "ar"


def test_objecting_to_a_language_clears_it(monkeypatch):
    """The stuck-in-French bug: nothing an English speaker typed could undo it."""
    from agent.models import LanguageRequest
    out, _ = _understand(monkeypatch, _reading(
        needs_syllabus=False, message_language="en",
        language_request=LanguageRequest(action="clear")),
        state={"conversation": {"state": {"language": "fr"}}})
    assert out["requested_language"] is None
    assert out["language"] == "en"


def test_mentioning_a_language_changes_nothing(monkeypatch):
    out, _ = _understand(monkeypatch, _reading(needs_syllabus=False, message_language="en"),
                         state={"message": "why did you answer in french?"})
    assert out["requested_language"] is None
    assert out["language"] == "en"


def test_a_subject_is_answered_in_its_own_language(monkeypatch):
    out, captured = _understand(
        monkeypatch, _reading(needs_syllabus=False, message_language="ar"),
        state={"conversation": {"state": {"subject_id": "bio"}}},
        subjects=_Subjects(bio={"language": "en", "name": "Biology"}))
    assert out["language"] == "en"
    assert "Biology" in captured["prompt"]


def test_an_unreadable_turn_fails_instead_of_guessing_a_task(monkeypatch):
    """A parse failure used to be indistinguishable from "explain this"; it
    then became an apology in one of four languages. It is a failed request."""
    from agent.errors import TutorUnavailable
    with pytest.raises(TutorUnavailable):
        _understand(monkeypatch, RuntimeError("unparseable"))


def test_without_a_model_every_turn_is_an_explain_task(monkeypatch):
    node = nmod.build_understand(get_config(), None, _Subjects())
    out = node({"pathway": "student", "message": "explain photosynthesis",
                "conversation": {}, "history": ""})
    assert out["planned_tasks"] == [{"type": "explain", "query": "explain photosynthesis",
                                     "level": "intermediate", "length": "medium",
                                     "whole_lesson": False}]


def test_a_reading_that_contradicts_itself_goes_back_to_the_model(monkeypatch):
    from agent.models import LanguageRequest, PlannedTask
    _, captured = _understand(monkeypatch, _reading(needs_syllabus=False))
    check = captured["validate"]
    with pytest.raises(ValueError):
        check(_reading(needs_syllabus=True, tasks=[], student_words="hi"))
    with pytest.raises(ValueError):
        check(_reading(tasks=[PlannedTask(type="explain", query="  ")], student_words="hi"))
    with pytest.raises(ValueError):
        check(_reading(needs_syllabus=False,
                       language_request=LanguageRequest(action="set", code="it")))


def test_the_task_descriptions_route_by_intent_not_by_quoted_phrases():
    """The planner was told "how am I doing" -> report; a phrase list covers
    only the phrasings someone thought to write down."""
    from agent.prompts import TASK_DESCRIPTIONS
    for description in TASK_DESCRIPTIONS.values():
        body = description.split(":", 1)[1]
        assert '("' not in body


def test_report_is_a_task_the_model_is_allowed_to_choose():
    """"report" was missing from the task-type enum, so structured output could
    never return it and progress questions fell through to other tasks."""
    from agent.models import PlannedTask
    assert PlannedTask(type="report", query="my progress").type == "report"


def test_allowed_descriptions_lists_only_allowed():
    text = nmod._allowed_descriptions(["explain", "grade"])
    assert '"explain"' in text and '"grade"' in text
    assert '"visual"' not in text and '"summarize"' not in text


def test_dispatch_routes_empty_tasks_to_persist():
    assert nmod.dispatch({"tasks": []}) == "persist"


# --- plan -------------------------------------------------------------------

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
        return default


def _plan_state(**extra):
    state = {"pathway": "student", "message": "q", "namespaces": ["syllabus"],
             "planned_tasks": [{"type": "explain", "query": "q", "level": "intermediate",
                                "length": "medium"}],
             "understanding": {"needs_syllabus": True}, "conversation": {}}
    state.update(extra)
    return state


def _hit(subject_id, score):
    return RetrievedChunk(chunk_id=f"c-{subject_id}-{score}", score=score, text="x",
                          lecture_id="l", topic_ids=[], page_from=1, page_to=1,
                          heading_path="H", language="en", subject_id=subject_id)


def test_the_subject_is_found_by_searching_the_material(monkeypatch):
    monkeypatch.setattr(nmod, "resolve_topics", lambda q, sid=None, **k: [])
    monkeypatch.setattr(nmod, "rag_search", lambda *a, **k: [
        _hit("bio", 0.7), _hit("bio", 0.6), _hit("chem", 0.8)])
    plan = nmod.build_plan(_SubjectCfg(default_subject_id="dflt"), _Subjects())
    assert plan(_plan_state())["subject_id"] == "bio"


def test_passages_below_the_floor_do_not_vote(monkeypatch):
    monkeypatch.setattr(nmod, "resolve_topics", lambda q, sid=None, **k: [])
    monkeypatch.setattr(nmod, "rag_search", lambda *a, **k: [_hit("bio", 0.1)])
    plan = nmod.build_plan(_SubjectCfg(default_subject_id="dflt"), _Subjects())
    assert plan(_plan_state())["subject_id"] == "dflt"


def test_inference_is_confined_to_the_student_s_subjects(monkeypatch):
    monkeypatch.setattr(nmod, "resolve_topics", lambda q, sid=None, **k: [])
    monkeypatch.setattr(nmod, "rag_search", lambda *a, **k: [
        _hit("other", 0.9), _hit("enrolled", 0.6)])

    class _Client:
        def get_student_subjects(self, uid):
            return [{"id": "enrolled"}]

    plan = nmod.build_plan(_SubjectCfg(), _Subjects(), _Client())
    assert plan(_plan_state(user_id="u1"))["subject_id"] == "enrolled"


def test_the_caller_s_subject_is_authoritative(monkeypatch):
    monkeypatch.setattr(nmod, "resolve_topics", lambda q, sid=None, **k: [])
    monkeypatch.setattr(nmod, "rag_search", lambda *a, **k: [_hit("bio", 0.9)])
    plan = nmod.build_plan(_SubjectCfg(), _Subjects())
    assert "subject_id" not in plan(_plan_state(subject_id="explicit"))


def test_the_conversation_keeps_its_subject_until_the_student_moves_on(monkeypatch):
    """A follow-up names no subject; re-inferring it from "can you explain more?"
    sent an English science answer to the Arabic corpus."""
    monkeypatch.setattr(nmod, "resolve_topics", lambda q, sid=None, **k: [])
    monkeypatch.setattr(nmod, "rag_search", lambda *a, **k: [_hit("arabic", 0.9)])
    plan = nmod.build_plan(_SubjectCfg(), _Subjects())
    kept = plan(_plan_state(conversation={"state": {"subject_id": "bio"}}))
    assert kept["subject_id"] == "bio"
    moved = plan(_plan_state(conversation={"state": {"subject_id": "bio"}},
                             understanding={"needs_syllabus": True, "changes_subject": True}))
    assert moved["subject_id"] == "arabic"


def test_a_continuing_turn_brings_the_last_answer_s_passages(monkeypatch):
    monkeypatch.setattr(nmod, "resolve_topics", lambda q, sid=None, **k: [])
    conversation = {"state": {"subject_id": "bio"}, "turns": [
        {"message": "explain evaporation", "reply": "...", "kind": "study",
         "subject_id": "bio", "cited_chunk_ids": ["c1", "c2"]}]}
    plan = nmod.build_plan(_SubjectCfg(), _Subjects())
    carried = plan(_plan_state(conversation=conversation, understanding={
        "needs_syllabus": True, "continues_previous": True}))
    assert carried["previous_chunk_ids"] == ["c1", "c2"]
    fresh = plan(_plan_state(conversation=conversation))
    assert fresh["previous_chunk_ids"] == []


def test_the_reply_language_follows_a_subject_the_student_opened(monkeypatch):
    """A lesson opened in the app teaches in its own language: an English
    literature lesson answers in English whatever the student types in.

    It used to hold for a guessed subject too, and that is how a student who
    wrote 'ممكن تشرح لي الكهربية الساكنة؟' was answered in French — one earlier
    English question had settled the conversation on the French science book.
    A subject nobody chose does not get to overrule the student."""
    monkeypatch.setattr(nmod, "resolve_topics", lambda q, sid=None, **k: [])
    monkeypatch.setattr(nmod, "rag_search", lambda *a, **k: [_hit("bio", 0.9)])
    plan = nmod.build_plan(get_config(), _Subjects(bio={"language": "en"}))

    opened = plan(_plan_state(message_language="ar", subject_id="bio"))
    assert opened["language"] == "en"

    guessed = plan(_plan_state(message_language="ar"))
    assert guessed["language"] == "ar"

    assert plan(_plan_state(message_language="ar", requested_language="ar"))["language"] == "ar"


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
    assert len(grades) == 2


def test_validate_assessment_spec_sanitizes():
    from agent.graph.nodes import _validate_assessment_spec
    assert _validate_assessment_spec(None) is None
    assert _validate_assessment_spec({}) is None
    assert _validate_assessment_spec({"count": 5}).count == 5
    assert _validate_assessment_spec({"types": ["mcq", "bogus"]}).types == ["mcq"]
    assert _validate_assessment_spec({"count": "five"}) is None
    assert _validate_assessment_spec({"count": 0}) is None
    assert _validate_assessment_spec({"types": ["bogus"]}) is None


def test_assessment_type_mix_distribution():
    from agent.graph.nodes import AssessmentSpec, _assessment_type_mix
    default = {"mcq": 3, "true_false": 1, "short_answer": 1}
    assert _assessment_type_mix(AssessmentSpec(count=4, types=["mcq"]), default) == {"mcq": 4}
    assert _assessment_type_mix(AssessmentSpec(count=3, types=["mcq", "true_false"]), default) == \
        {"mcq": 2, "true_false": 1}
    assert sum(_assessment_type_mix(AssessmentSpec(types=["mcq", "essay"]), default).values()) == \
        sum(default.values())
    assert sum(_assessment_type_mix(AssessmentSpec(count=6), default).values()) == 6


def test_plan_applies_requested_assessment_shape(monkeypatch):
    monkeypatch.setattr(nmod, "resolve_topics",
                        lambda q, sid=None, **k: [ResolvedTopic("t1", "T", 0.9)])
    plan = nmod.build_plan(get_config(), _Subjects())
    out = plan(_plan_state(subject_id="s", planned_tasks=[{
        "type": "assessment", "query": "fractions", "level": "intermediate",
        "length": "medium", "assessment": {"count": 4, "types": ["mcq"]}}]))
    qtask = next(t for t in out["tasks"] if t["type"] == "assessment")
    assert qtask["params"]["type_mix"] == {"mcq": 4}


def test_plan_grade_ref_overrides_nl_reference(monkeypatch):
    monkeypatch.setattr(nmod, "resolve_topics", lambda q, sid=None, **k: [])
    plan = nmod.build_plan(get_config(), _Subjects())
    out = plan(_plan_state(
        subject_id="s",
        grade_ref={"assessment_id": "from-frontend", "kind": "homework"},
        planned_tasks=[{"type": "grade", "query": "my answer", "level": "intermediate",
                        "length": "medium",
                        "grade": {"assessment_id": "from-nl", "kind": "quiz"}}]))
    gtask = next(t for t in out["tasks"] if t["type"] == "grade")
    assert gtask["params"]["grade"]["assessment_id"] == "from-frontend"
    assert gtask["params"]["grade"]["kind"] == "homework"


# --- compose helpers --------------------------------------------------------

def _pt(topic_id="t0", label="Cells", strength="weak", depth=1):
    return PrereqTopic(topic_id=topic_id, label=label, strength=strength, depth=depth)


def test_what_follows_an_answer_is_marked_not_worded():
    """"Prerequisite topics" and "Sources" were sentences kept in four languages,
    and wrong the day a fifth was added. A mark reads the same in every one."""
    assert nmod._prereq_line([_pt()]) == "🧩 Cells"
    cites = [Citation(n=1, lecture_id="lec1", heading_path="Ch1", page_from=3)]
    assert nmod._sources_block(cites).startswith("📚\n[1]")


def test_an_empty_prereq_list_renders_nothing():
    assert nmod._prereq_line([]) == ""


def test_an_arabic_prereq_is_shown_as_written():
    arabic = PrereqTopic(topic_id="t2", label="تحليل النص الشعري",
                         strength="required", depth=1)
    assert "تحليل النص الشعري" in nmod._prereq_line([arabic])


def test_sources_block_dedupes_and_numbers():
    cites = [
        Citation(n=1, lecture_id="lec1", heading_path="Ch1", page_from=3, page_to=3),
        Citation(n=1, lecture_id="lec1", heading_path="Ch1", page_from=3, page_to=3),
    ]
    block = nmod._sources_block(cites)
    assert block.count("[1]") == 1
    assert "Ch1, p.3" in block


def test_sources_block_renders_readable_lecture_label():
    cites = [Citation(n=1, lecture_id="lec1", subject_name="Science FR",
                      lesson_order=4, lesson_title="Mendel's Laws",
                      page_from=6, page_to=7)]
    assert "[1] Science FR · 4: Mendel's Laws, p.6–7" in nmod._sources_block(cites)


def test_sources_block_lecture_label_partial():
    b1 = nmod._sources_block([Citation(n=1, lecture_id="l", subject_name="Math EN",
                                       lesson_order=2, page_from=1)])
    assert "[1] Math EN · #2, p.1" in b1
    b3 = nmod._sources_block([Citation(n=1, lecture_id="l", heading_path="Ch1", page_from=3)])
    assert "[1] Ch1, p.3" in b3


def test_sources_block_strips_trailing_punctuation():
    cites = [Citation(n=1, lecture_id="lec1", heading_path="Oxidation,", page_from=3)]
    block = nmod._sources_block(cites)
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
    monkeypatch.setattr(rmod, "rag_search", lambda *a, **k: [primary])
    monkeypatch.setattr(rmod, "get_topic_deps",
                        lambda ids, **k: [PrereqTopic("t0", "Cells", "required", 1)])


def test_graph_halts_on_unknown_role(monkeypatch):
    from shared.errors import UnprocessableError
    _patch_tools(monkeypatch)
    graph = build_graph(get_config())
    with pytest.raises(UnprocessableError):
        graph.invoke({"message": "explain photosynthesis", "user_type": "guest"})


def test_classify_recognizes_parent(monkeypatch):
    # Parent is a configured pathway -> recognized, not mis-routed to student.
    from agent.conversation import HistoryWindow, InMemoryConversationStore
    classify = nmod.build_classify(get_config(), InMemoryConversationStore(), HistoryWindow())
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


def test_explain_sg_fails_the_turn_when_the_model_fails(monkeypatch):
    from agent.errors import TutorUnavailable
    from agent.graph.subgraphs.explain_summarize_sg import build_explain_sg

    chunk = RetrievedChunk(chunk_id="c1", score=0.9, text="t", lecture_id="lec1",
                           topic_ids=["t1"], page_from=1, page_to=1, heading_path="H",
                           language="en")
    monkeypatch.setattr(rmod, "retrieve", lambda *a, **k: [chunk])

    class _BadLLM:
        def invoke(self, _messages):
            raise RuntimeError("boom")

    sg = build_explain_sg(config=get_config(), llm=_BadLLM())
    with pytest.raises(TutorUnavailable):
        sg.invoke({"query": "x", "topic_ids": ["t1"], "namespaces": ["syllabus"],
                   "language": "en"})


def test_graph_end_to_end_stub_english(monkeypatch):
    _patch_tools(monkeypatch)
    graph = build_graph(get_config())
    out = graph.invoke({"message": "explain photosynthesis",
                        "user_type": "student", "subject_id": "sub1"})

    assert out["pathway"] == "student"
    assert out["language"] == "en"
    assert out["tasks"][0]["params"]["topic_ids"] == ["t1"]
    assert "grounded in" in out["response"]
    assert "🧩" not in out["response"] and "📚" not in out["response"]
    assert "[1]" not in out["response"]
    assert out["published_citations"][0].lecture_id == "lec1"


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
    })
    assert "→ 6" in line and "💡 b = sqrt(a*c)." in line


def test_grade_line_omits_key_when_correct():
    line = nmod._grade_line({
        "summary": {"score": 1, "max_score": 1},
        "graded": [{"correct": True, "feedback": "Right.",
                    "correct_answer": "6", "explanation": "because."}],
    })
    assert "→" not in line and "💡" not in line


def test_an_answer_that_could_not_be_graded_is_marked():
    line = nmod._grade_line({"summary": {"score": 0, "max_score": 0},
                             "graded": [{"graded": False, "feedback": ""}]})
    assert "∅ 1." in line


def test_a_visual_is_linked_by_its_mark():
    line = nmod._visual_line([{"kind": "mindmap", "s3_uri": "u"},
                              {"kind": "figure", "s3_uri": "f", "reused": True}])
    assert "[🧠](u)" in line and "[🖼️ 📖](f)" in line
    assert nmod._visual_line([{"kind": "slides", "text": "S"}]) == "S"


def test_a_quiz_reply_introduces_the_questions_without_printing_them():
    """The reply listed every question and the quiz card under it listed them again,
    so a student saw each question twice."""
    compose = nmod.build_compose(get_config(), None, NO_VOICES)
    assessment = {"intro": "Here are two questions on evaporation.", "items": [
        {"question": "What is evaporation?", "answer": "gas",
         "options": [{"text": "gas", "is_correct": True}, {"text": "ice"}]},
        {"question": "Does ice float?", "answer": "true"}]}

    alone = compose({"answers": [], "assessment": assessment, "language": "en"})
    beside = compose({"answers": [{"type": "explain", "text": "Water turns to gas.",
                                   "task_id": "a", "grounded": True}],
                      "assessment": assessment, "language": "en"})

    for out in (alone, beside):
        assert "Here are two questions on evaporation." in out["response"]
        assert "What is evaporation?" not in out["response"]
        assert "Does ice float?" not in out["response"]
    assert alone["response"] == "Here are two questions on evaporation."


def test_a_quiz_without_an_intro_still_has_a_reply():
    compose = nmod.build_compose(get_config(), None, NO_VOICES)
    out = compose({"answers": [], "language": "en",
                   "assessment": {"items": [{"question": "Does ice float?"}]}})
    assert out["response"] == nmod.QUIZ_MARK


def test_a_visual_that_failed_alone_fails_the_turn():
    from agent.errors import TutorUnavailable
    compose = nmod.build_compose(get_config(), None, NO_VOICES)
    with pytest.raises(TutorUnavailable):
        compose({"answers": [], "visual_error": True, "language": "en"})


def test_a_visual_that_failed_beside_an_answer_leaves_the_answer():
    compose = nmod.build_compose(get_config(), None, NO_VOICES)
    out = compose({"answers": [{"type": "explain", "text": "Plants make food [1].",
                                "task_id": "a", "grounded": True}],
                   "citations": [Citation(n=1, task_id="a", lecture_id="l", page_from=1)],
                   "visual_error": True, "language": "en"})
    assert out["response"].startswith("Plants make food.")


def test_compose_suppresses_prereqs_on_decline():
    compose = nmod.build_compose(get_config(), None, NO_VOICES)
    out = compose({
        "answers": [{"type": "explain", "text": "The context does not cover this.",
                     "grounded": False}],
        "prereq_topics": [PrereqTopic(topic_id="t1", label="Fractions",
                                      strength="strong", depth=1)],
        "citations": [], "language": "en",
    })
    assert "🧩" not in out["response"] and "Fractions" not in out["response"]


def test_every_later_step_sees_only_the_student_s_own_words(monkeypatch):
    """The app sends "Explain this: Hellllllo". The reply step read the prefix as the
    student's request and asked what part of "Hellllllo" to explain."""
    out, _ = _understand(monkeypatch, _reading(
        needs_syllabus=False, student_words="Hellllllo", message_language="en"),
        state={"message": "Explain this: Hellllllo"})
    assert out["message"] == "Hellllllo"
    assert out["raw_message"] == "Explain this: Hellllllo"


def test_words_the_student_did_not_write_go_back_to_the_model(monkeypatch):
    _, captured = _understand(monkeypatch, _reading(needs_syllabus=False, student_words="hi"),
                              state={"message": "Explain this: hi"})
    check = captured["validate"]
    with pytest.raises(ValueError):
        check(_reading(needs_syllabus=False, student_words="Hello there"))
    check(_reading(needs_syllabus=False, student_words="hi"))


def test_without_a_topic_filter_the_best_match_is_not_excluded(monkeypatch):
    """"clothes catch fire" guessed "Scientific inquiry skills" and two water-cycle
    topics; filtered to those, the fire-safety passage was never a candidate."""
    seen = {}

    def search(query, namespaces, **k):
        seen["topic_ids"] = k.get("topic_ids")
        return [_chunk("fire", 0.66)]

    monkeypatch.setattr(rmod, "rag_search", search)
    rmod.retrieve("q", ["syllabus"], topic_ids=["guessed"], topic_filter="off")
    assert seen["topic_ids"] is None


def _placed(cid, score, order, lecture="L"):
    return RetrievedChunk(chunk_id=cid, score=score, text=cid, lecture_id=lecture,
                          topic_ids=[], page_from=1, page_to=1, heading_path="",
                          language="en", order_index=order)


def test_a_section_split_into_chunks_is_read_whole(monkeypatch):
    """"What causes ocean waves?" found the wind chunk and never the landslide and
    meteorite chunks two and three places after it on the same page."""
    section = {o: _placed(f"s{o}", 0.0, o) for o in range(10, 15)}
    monkeypatch.setattr(rmod, "get_neighbours", lambda seeds, window: [
        section[o] for s in seeds for o in range(s.order_index - window, s.order_index + window + 1)
        if o in section])
    wind, far = _placed("s11", 0.72, 11), _placed("x", 0.60, 90)
    out = rmod.with_neighbours([far, wind], anchors=1, window=2, limit=10)
    assert [c.chunk_id for c in out] == ["s10", "s11", "s12", "s13", "x"]
    assert out[1].score == 0.72


# --- a quiz waiting for its answers -------------------------------------------

def _open_quiz():
    from agent.conversation import PendingAssessment, PendingQuestion

    return PendingAssessment(items=[
        PendingQuestion(id="q1", question="Which process moves water into the air?",
                        type="mcq", options=["condensation", "evaporation"],
                        answer="evaporation", topic_id="t1", lecture_id="l1"),
        PendingQuestion(id="q2", question="Most ocean wave energy comes from the sun.",
                        type="true_false", answer="True", topic_id="t2", lecture_id="l1"),
    ])


def test_the_open_questions_are_put_in_front_of_the_model(monkeypatch):
    """A student who answers a quiz says only "d) evaporation". Without the
    questions in view the turn was read as "explain evaporation" and the answers
    came back explained instead of marked."""
    _, captured = _understand(
        monkeypatch, _reading(message_language="en"),
        state={"message": "d) evaporation\nTrue",
               "conversation": {"state": {"subject_id": "bio", "pending": _open_quiz().model_dump()}}},
        subjects=_Subjects(bio={"language": "en", "name": "Biology"}))

    prompt = captured["prompt"]
    assert "Which process moves water into the air?" in prompt
    assert "Most ocean wave energy comes from the sun." in prompt
    assert "evaporation" not in prompt.split("Which process")[0]   # never the answer key


def test_the_answer_key_is_never_shown_to_the_model(monkeypatch):
    """The key is the server's. A model that can see it can hand it to the
    student, and a student who asks "what was the answer?" would be told."""
    from agent.conversation import PendingAssessment, PendingQuestion

    quiz = PendingAssessment(items=[PendingQuestion(
        id="q1", question="Name the process that moves water into the air.",
        answer="sublimation-of-the-hydrosphere")])
    _, captured = _understand(
        monkeypatch, _reading(message_language="en"),
        state={"message": "evaporation",
               "conversation": {"state": {"pending": quiz.model_dump()}}})

    assert "Name the process" in captured["prompt"]
    assert "sublimation-of-the-hydrosphere" not in captured["prompt"]


# --- a term that exists in two languages ---------------------------------------

class _Subjects2:
    """Subject facts for a corpus that holds the same term twice."""

    def __init__(self, **facts):
        from agent.models import SubjectFacts
        self._facts = {k: SubjectFacts(subject_id=k, **v) for k, v in facts.items()}

    def facts(self, subject_id):
        from agent.models import SubjectFacts
        return self._facts.get(subject_id) or SubjectFacts(subject_id=subject_id)


def test_a_settled_subject_is_left_behind_when_the_student_changes_language(monkeypatch):
    """One English question settled the conversation on the French science
    subject; every Arabic question after it was answered from the French book,
    in French. A student who switches language is asking for their own."""
    from agent.conversation import Conversation, SessionState

    seen = {}

    def fake_infer(query, namespaces, config, enrolled, language=None):
        seen["language"] = language
        return "sci-ar"

    monkeypatch.setattr(nmod, "_infer_subject", fake_infer)
    conversation = Conversation(state=SessionState(subject_id="sci-fr"))
    subject_id, source = nmod._resolve_subject(
        {"message": "ما هي الكهربية الساكنة؟", "message_language": "ar", "namespaces": []},
        _reading(message_language="ar"), conversation, [], get_config(), None,
        subjects=_Subjects2(**{"sci-fr": {"language": "fr"}, "sci-ar": {"language": "ar"}}))

    assert subject_id == "sci-ar"
    assert source == "inferred"
    assert seen["language"] == "ar"


def test_a_settled_subject_is_kept_when_the_language_still_matches(monkeypatch):
    from agent.conversation import Conversation, SessionState

    monkeypatch.setattr(nmod, "_infer_subject",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not re-infer")))
    conversation = Conversation(state=SessionState(subject_id="sci-ar"))
    subject_id, source = nmod._resolve_subject(
        {"message": "اشرح أكثر", "message_language": "ar", "namespaces": []},
        _reading(message_language="ar"), conversation, [], get_config(), None,
        subjects=_Subjects2(**{"sci-ar": {"language": "ar"}}))

    assert subject_id == "sci-ar" and source == "conversation"


def test_the_vote_prefers_material_in_the_language_the_student_wrote_in(monkeypatch):
    """Both books answer the question; the one the student can read wins."""
    from agent.models import RetrievedChunk

    def chunk(subject, language, score):
        return RetrievedChunk(chunk_id=f"{subject}-{score}", score=score, text="x",
                              lecture_id="l", topic_ids=[], page_from=1, page_to=1,
                              heading_path=None, language=language, subject_id=subject)

    monkeypatch.setattr(nmod, "rag_search", lambda *a, **k: [
        chunk("sci-fr", "fr", 0.72), chunk("sci-fr", "fr", 0.70),
        chunk("sci-ar", "ar", 0.66), chunk("sci-ar", "ar", 0.64)])

    assert nmod._infer_subject("الكهربية الساكنة", [], get_config(), None,
                               language="ar") == "sci-ar"
    assert nmod._infer_subject("électricité statique", [], get_config(), None,
                               language="fr") == "sci-fr"
    assert nmod._infer_subject("static electricity", [], get_config(), None,
                               language="en") == "sci-fr"
