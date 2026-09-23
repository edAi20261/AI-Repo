"""Compile the runtime graph.

    classify -> load_context -> understand -> converse -------------------------------+
                                           -> plan -> dispatch ⇉ {task}_sg              |
                                                   -> render_visuals -> compose --------+
                                                                  -> verify_language -> persist

Subgraphs are compiled once and wrapped by adapter nodes that translate a task +
context into the subgraph's private state and map its result back onto MainState
(answers / citations / prereq_topics / visual_specs merge via the state's reducers).
All task adapters fan in to render_visuals (which renders every collected visual_spec)
before compose.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from shared.backend import get_backend_client
from shared.llm import llm_key_available, maybe_llm

from .. import retrieval
from ..config import Config, get_config
from ..conversation import Conversation, ConversationStore, HistoryWindow
from ..errors import TutorUnavailable
from ..models import Style, SubjectFacts
from ..scope import lecture_filter
from ..state import MainState
from ..subjects import SubjectDirectory
from . import nodes
from .subgraphs import (
    build_assessment_sg,
    build_explain_sg,
    build_grade_sg,
    build_report_sg,
    build_summarize_sg,
    build_visual_sg,
)

logger = logging.getLogger(__name__)


# Environments where a missing LLM key is allowed (the nodes take the deterministic stub
# path). Any other environment (e.g. production) must fail fast at startup instead of
# silently stubbing answers to real users.
_DEV_ENVS = {"local", "test", "dev", "ci"}


def _require_llm_in_prod(config: Config) -> None:
    """Fail fast: a non-dev environment with no chat-model key would serve stub answers to
    real users. Refuse to compile instead (the stub path is dev/test only)."""
    if config.app_env in _DEV_ENVS:
        return
    if not llm_key_available(config, config.llm_for(None)):
        raise RuntimeError(
            f"No chat-model API key configured for APP_ENV={config.app_env!r}. "
            "Refusing to start with stub answers — set the provider key or use a dev env.")


# Adapter = the bridge between MainState and a subgraph's PRIVATE state. Every adapter does
# the same three steps: (1) translate the task + shared context INTO the subgraph's input
# state, (2) run the compiled subgraph, (3) map its output BACK onto MainState fields. The
# subgraph never sees MainState, so it stays self-contained. (LangGraph: a subgraph with a
# different state schema is invoked inside a node function that transforms state in and out.)
def _build_explain_summarize_adapter(compiled, mode: str, config: Config,
                                     subjects: SubjectDirectory):
    # Read the config knobs once, at build time (not per turn).
    k = config.get("agent.k", 8)
    depth = config.get("agent.prereq_depth", 2)
    prereq_limit = config.get("agent.max_prereqs", retrieval.DEFAULT_MAX_PREREQS)
    min_score = config.get("agent.min_score", retrieval.DEFAULT_MIN_SCORE)
    topic_filter = config.get("agent.topic_filter", "off")
    scope_ratio = config.get("agent.scope.relevance_ratio", retrieval.DEFAULT_SCOPE_RATIO)
    with_figure = config.get("agent.explain_figures", False)
    expand = config.get("agent.expand") or {}

    def adapter(payload) -> dict:                    # payload = {task, ...shared context}
        task = payload["task"]
        task_id = task.get("task_id")
        params = task.get("params", {})
        facts = subjects.facts(payload.get("subject_id"))
        # (1) TRANSLATE IN: build the explain/summarize subgraph's private input state.
        result = compiled.invoke({
            "mode": mode,
            "query": params.get("query", ""),
            "message": payload.get("message") or params.get("query", ""),
            "history": payload.get("history") or "",
            "previous_chunk_ids": payload.get("previous_chunk_ids") or [],
            "level": params.get("level", "intermediate"),
            "length": params.get("length", "medium"),
            "topic_ids": params.get("topic_ids") or [],
            "lecture_ids": lecture_filter(payload),
            "study_scope": payload.get("study_scope"),
            "namespaces": payload.get("namespaces") or ["syllabus"],
            "style": payload.get("style") or Style(),
            "language": payload.get("language", "en"),
            "instructions": _subject_instructions(facts, with_figure=with_figure),
            "subject_id": payload.get("subject_id"),
            "grade": payload.get("grade"),
            "k": k,
            "prereq_depth": depth,
            "prereq_limit": prereq_limit,
            "min_score": min_score,
            "topic_filter": topic_filter,
            "scope_relevance_ratio": scope_ratio,
            "expand_anchors": expand.get("anchors", 0),
            "expand_window": expand.get("window", 0),
            "context_limit": expand.get("context_limit", k),
        })
        # (2) the subgraph ran; `result` is its private output. (3) TRANSLATE OUT below.
        # Tag citations with their task so compose can renumber [n] globally across a
        # multi-task turn (each subgraph numbers from [1] independently).
        citations = [c.model_copy(update={"task_id": task_id})
                     for c in result.get("citations", [])]
        # A declined answer (no inline citation / figure) carries grounded=False; compose
        # then suppresses the trailing sources/prereqs so a refusal doesn't read as answered.
        # (Citations and the figure are already dropped by the subgraph on a decline.)
        out = {                                      # only these keys land on MainState
            "answers": [{
                "type": mode,
                "task_id": task_id,
                "text": result.get("explanation", ""),
                "topic_labels": params.get("topic_labels", []),
                "grounded": result.get("grounded", True),
            }],
            "citations": citations,
            "prereq_topics": result.get("prereq_topics", []),
        }
        # Forward a visual_spec the subgraph emitted onto MainState (reducer-merged);
        # render_visuals renders it after the fan-in.
        if result.get("visual_spec"):
            out["visual_specs"] = [result["visual_spec"]]
        return out

    return adapter


def _subject_facts(client, subject_id: Optional[str]) -> SubjectFacts:
    """The subject's facts (language, is_math, …) from the backend, or defaults. One lookup
    feeds both the subject-language choice and the per-subject prompt addendum. Any failure
    degrades to empty defaults so the turn still runs."""
    if client is None or not subject_id:
        return SubjectFacts()
    try:
        return SubjectFacts.model_validate(client.get_subject(subject_id) or {})
    except Exception:
        return SubjectFacts()


def _subject_instructions(facts: SubjectFacts, *, with_figure: bool = False) -> str:
    """The prompt addendum for a subject, appended to any subgraph via its ``{instructions}``
    slot. Math is the first; add more subject kinds here. ``with_figure`` adds the figure-emit
    directive — only for explain/summarize, whose answer is parsed for the block."""
    if not facts.is_math:
        return ""
    from ..prompts import MATH_FIGURE_INSTRUCTION, MATH_PROMPT_INSTRUCTIONS

    return (MATH_PROMPT_INSTRUCTIONS + "\n" + MATH_FIGURE_INSTRUCTION
            if with_figure else MATH_PROMPT_INSTRUCTIONS)


def _build_assessment_adapter(compiled, config: Config, subjects: SubjectDirectory):
    q = config.get("assessment", {}) or {}
    quiz_follows_subject = config.get("language.quiz_single_language", True)

    def adapter(payload) -> dict:
        task = payload["task"]
        params = task.get("params", {})
        facts = subjects.facts(payload.get("subject_id"))
        # Questions are material a student will sit, so they follow the subject's
        # language when it is known — an IGCSE physics quiz stays in English even
        # for an Arabic-speaking student. Everything addressed TO the student
        # (feedback, reports, captions) follows the student instead.
        language = (facts.language or payload.get("language", "en")
                    if quiz_follows_subject else payload.get("language", "en"))
        # (1) translate in: each `params.X or q.get(X)` = per-request override, else config default.
        result = compiled.invoke({
            "query": params.get("query", ""),
            "topic_ids": params.get("topic_ids") or [],
            "lecture_ids": lecture_filter(payload),
            "subject_id": payload.get("subject_id"),
            "namespaces": payload.get("namespaces") or ["syllabus"],
            "language": language,
            "instructions": _subject_instructions(facts),
            "grade": payload.get("grade"),
            "user_id": payload.get("user_id"),
            "item_source": params.get("item_source") or q.get("item_source", "generated"),
            "type_mix": params.get("type_mix") or q.get("type_mix", {}),
            "difficulty": params.get("difficulty") or q.get("difficulty", "mixed"),
            "k_per_topic": q.get("k_per_topic", 4),
            "min_score": config.get("agent.min_score", retrieval.DEFAULT_MIN_SCORE),
            "adaptive": q.get("adaptive", True),
        })
        return {"assessment": result.get("assessment", {})}   # (2) translate out

    return adapter


def _kind_source(kind: Optional[str], grade_cfg: dict) -> str:
    """Map a grading kind to a ``mastery_event_source``: a referenced quiz/homework keeps
    its kind; a plain chat grade turn defaults to ``practice`` (config ``grade.source``)."""
    if kind in ("quiz", "homework"):
        return kind
    return grade_cfg.get("source", "practice")


def _question_key(reference: str) -> str:
    """A question named as "2", "2." or "2. Which process ...", reduced to one key.

    Which of those a model returns is not worth constraining: it varies by model
    and by how the question was phrased, and pairing an answer onto the wrong
    question marks the student down for someone else's mistake.
    """
    reference = (reference or "").strip()
    leading = re.match(r"^\s*(\d+)[.)\s]", reference) or re.fullmatch(r"\s*(\d+)\s*", reference)
    if leading:
        return leading.group(1)
    return " ".join(reference.lower().split())


def _answered_questions(payload, spec: dict, answer: str) -> list[dict]:
    """The tutor's own open questions, paired with what the student answered them with.

    The questions the tutor asked in chat are not in any rubric table — they were
    generated for this conversation — so the answer key is the one kept on the
    session. Without it a student's "d) evaporation" was marked on its merits, or
    read as a new question and explained back at them.
    """
    conversation = Conversation.model_validate(payload.get("conversation") or {})
    pending = conversation.state.pending
    if not (pending and pending.items):
        return []

    paired = {_question_key(a.get("question", "")): a.get("answer", "")
              for a in (spec.get("answers") or []) if isinstance(a, dict)}
    lines = [line.strip() for line in (answer or "").splitlines() if line.strip()]
    submissions = []
    for index, item in enumerate(pending.items, 1):
        given = (paired.get(str(index)) or paired.get(item.id)
                 or paired.get(_question_key(item.question)) or "")
        if not given and len(lines) == len(pending.items):
            given = lines[index - 1]
        if not given:
            continue
        submissions.append({
            "question": item.question,
            "student_answer": given,
            "correct_answer": item.answer,
            "type": item.type,
            "topic_id": item.topic_id,
            "lecture_id": item.lecture_id,
        })
    return submissions


def _build_grade_adapter(compiled, config: Config, subjects: SubjectDirectory):
    """Chat 'grade me this' path. Normalizes the turn into grade_sg's private state:
    structured ``submissions`` when the request carries them, else a single inline item
    from the message (graded on merits, or against a referenced quiz's rubric)."""
    g = config.get("grade", {}) or {}

    def adapter(payload) -> dict:
        task = payload["task"]
        params = task.get("params", {})
        spec = params.get("grade") or {}
        facts = subjects.facts(payload.get("subject_id"))
        answer = params.get("query", "")
        # Inline "grade my answer": the message is both the question and the student's answer.
        submissions = (params.get("submissions")
                       or _answered_questions(payload, spec, answer)
                       or [{"question": answer, "student_answer": answer}])
        # A student chat grade is self-practice (source=practice) even if it references a
        # quiz; a teacher grading in chat is assigned work (source=quiz/homework by kind).
        source = ("practice" if payload.get("pathway") == "student"
                  else _kind_source(spec.get("kind"), g))
        result = compiled.invoke({       # (1) translate in
            "submissions": submissions,
            "subject_id": payload.get("subject_id"),
            "lecture_ids": lecture_filter(payload),
            "user_id": payload.get("user_id"),
            "language": payload.get("language", "en"),
            "instructions": _subject_instructions(facts),
            "kind": spec.get("kind"),
            "assessment_id": spec.get("assessment_id"),
            "source": source,
            "pass_threshold": g.get("pass_threshold", 0.5),
            "namespaces": payload.get("namespaces") or ["syllabus"],
            "min_score": config.get("agent.min_score", retrieval.DEFAULT_MIN_SCORE),
        })
        return {"grade_result": result.get("result", {})}   # (2) translate out

    return adapter


def _build_visual_adapter(config: Config):
    """Chat 'show me a visual' task: emit ONE VisualSpec onto MainState.visual_specs.
    Rendering happens once in render_visuals (after the fan-in), the same path a
    subgraph-emitted spec takes."""
    def adapter(payload) -> dict:
        params = payload["task"].get("params", {})
        spec = dict(params.get("visual") or {})
        spec.setdefault("description", params.get("query", ""))
        spec.setdefault("topic_ids", params.get("topic_ids") or None)
        # Image generation is deferred; a kindless visual request has nothing safe to render,
        # so emit no spec rather than fabricate a picture.
        if not spec.get("kind"):
            return {}
        return {"visual_specs": [spec]}

    return adapter


def _build_report_adapter(compiled, config: Config):
    """Chat 'how am I doing' task -> report_sg. Students are forced to their OWN progress
    (scope + user_id = self); the report narrative becomes an answer (prose) and any trend
    chart is emitted as a visual_spec the parent renders. The structured payload is returned
    only by the /ai/reports endpoint (run_report), not surfaced in the chat response."""
    def adapter(payload) -> dict:
        task = payload["task"]
        task_id = task.get("task_id")
        # Scope is forced by pathway: a student can only see their own progress. Teacher/parent
        # report scopes are shelled (Sprint 3), so no report task reaches them today.
        state = compiled.invoke({
            "scope": "student_progress",
            "user_id": payload.get("user_id"),
            "subject_id": payload.get("subject_id"),
            "lecture_ids": lecture_filter(payload),
            "language": payload.get("language", "en"),
        })
        result = state.get("result", {})
        if result.get("degraded"):
            raise TutorUnavailable()
        out: dict = {"answers": [{
            "type": "report", "task_id": task_id,
            "text": result.get("narrative", ""), "topic_labels": [],
        }]}
        if state.get("visual_spec"):
            out["visual_specs"] = [state["visual_spec"]]
        return out

    return adapter


def _build_render_visuals(compiled, config: Config, subjects: SubjectDirectory):
    """Parent node after the fan-in: render every collected visual_spec (chat task +
    subgraph-emitted) through visual_sg, once. get_figures-first, capped."""
    max_per = config.get("render.max_per_turn", 3)

    def render_visuals(state) -> dict:
        specs = state.get("visual_specs") or []
        if not specs:
            return {}
        facts = subjects.facts(state.get("subject_id"))
        arts = state.get("answers") or []
        # Gather what the turn already produced so a visual can be built from real content
        # (e.g. flashcards from the quiz, a chart from the grade) rather than invented.
        content = {
            "explanations": [a.get("text") for a in arts
                             if a.get("type") in ("explain", "summarize") and a.get("text")],
            "assessment": state.get("assessment") or {},
            "grade_result": state.get("grade_result") or {},
        }
        result = compiled.invoke({
            "specs": specs,
            "subject_id": state.get("subject_id"),
            "lecture_ids": lecture_filter(state),
            "user_id": state.get("user_id"),
            "conversation_id": state.get("session_id"),
            "language": state.get("language", "en"),
            "namespaces": state.get("namespaces"),
            "content": content,
            "instructions": _subject_instructions(facts),
            "max_per_turn": max_per,
        })
        return {"visuals": result.get("visuals", []),
                "visual_error": bool(result.get("errors"))}

    return render_visuals


def build_graph(config: Optional[Config] = None, *, checkpointer=None):
    """Compile the parent StateGraph. Pass a ``checkpointer`` (e.g. MemorySaver)
    for session memory; omit for a stateless one-shot invoke."""
    from langgraph.graph import END, START, StateGraph

    config = config or get_config()
    _require_llm_in_prod(config)   # fail fast rather than serve stub answers in production

    # Compile each subgraph once, wiring its per-task chat model (None -> deterministic stub).
    explain_sg = build_explain_sg(config=config, llm=maybe_llm(config, "explain"))
    summarize_sg = build_summarize_sg(config=config, llm=maybe_llm(config, "summarize"))
    assessment_sg = build_assessment_sg(config=config, llm=maybe_llm(config, "assessment"))
    grade_sg = build_grade_sg(config=config, llm=maybe_llm(config, "grade"))
    report_sg = build_report_sg(config=config, llm=maybe_llm(config, "report"))
    visual_sg = build_visual_sg(config=config)   # per-task LLM resolution inside the subgraph

    # One backend read-client per compile, shared by every adapter for subject facts
    # (language, is_math); None when no backend is configured -> turn language/defaults are used.
    try:
        backend_client = get_backend_client(config)
    except ValueError:
        backend_client = None
    subjects = SubjectDirectory(backend_client)
    store = ConversationStore.from_config(config)
    window = HistoryWindow.from_config(config)

    graph = StateGraph(MainState)
    graph.add_node("classify", nodes.build_classify(config, store, window))
    graph.add_node("load_context", nodes.build_load_context(config))
    graph.add_node("understand", nodes.build_understand(
        config, maybe_llm(config, "understand"), subjects))
    graph.add_node("converse", nodes.build_converse(config, maybe_llm(config, "converse")))
    graph.add_node("plan", nodes.build_plan(config, subjects, backend_client))
    graph.add_node("explain_adapter", _build_explain_summarize_adapter(explain_sg, "explain", config, subjects))
    graph.add_node("summarize_adapter", _build_explain_summarize_adapter(summarize_sg, "summarize", config, subjects))
    graph.add_node("assessment_adapter", _build_assessment_adapter(assessment_sg, config, subjects))
    graph.add_node("grade_adapter", _build_grade_adapter(grade_sg, config, subjects))
    graph.add_node("report_adapter", _build_report_adapter(report_sg, config))
    graph.add_node("visual_adapter", _build_visual_adapter(config))
    graph.add_node("render_visuals", _build_render_visuals(visual_sg, config, subjects))
    graph.add_node("compose", nodes.build_compose(config, maybe_llm(config, "compose")))
    graph.add_node("verify_language", nodes.build_verify_language(
        config, maybe_llm(config, "compose")))
    graph.add_node("persist", nodes.build_persist(
        config, store, window, maybe_llm(config, "conversation_summary")))

    graph.add_edge(START, "classify")
    graph.add_edge("classify", "load_context")
    graph.add_edge("load_context", "understand")
    graph.add_conditional_edges("understand", nodes._after_understand, ["converse", "plan"])
    graph.add_edge("converse", "verify_language")
    graph.add_conditional_edges(
        "plan", nodes.dispatch,
        ["explain_adapter", "summarize_adapter", "assessment_adapter", "grade_adapter",
         "report_adapter", "visual_adapter", "persist"])
    for adapter in ("explain_adapter", "summarize_adapter", "assessment_adapter",
                    "grade_adapter", "report_adapter", "visual_adapter"):
        graph.add_edge(adapter, "render_visuals")
    graph.add_edge("render_visuals", "compose")
    graph.add_edge("compose", "verify_language")
    graph.add_edge("verify_language", "persist")
    graph.add_edge("persist", END)

    compiled = graph.compile(checkpointer=checkpointer)
    # Wire the configured recursion backstop so every invoke inherits it.
    return compiled.with_config({"recursion_limit": config.get("agent.recursion_limit", 25)})
