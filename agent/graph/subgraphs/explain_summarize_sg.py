"""explain_sg / summarize_sg — structurally identical RAG subgraphs.

    retrieve -> answer | answer_beyond | decline -> resolve_prereqs

RAG is hard-wired: no branch goes around ``retrieve``. What happens after it is
decided in one place, ``_route_grounding``: the lessons answer, or the question
belongs to the subject and the tutor teaches it beyond them, or it is declined. Both modes
share this builder; ``mode`` picks the prompt and the artifact label. The subgraph
has private state (``ExplainSummarizeState``) and never sees the pathway — it is handed concrete
namespaces/style/language by the parent's adapter node. Retrieval itself lives in
``agent/retrieval.py`` (shared across subgraphs); this file keeps only the answer
step and the explain/summarize prompts.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, TypedDict

from langchain_core.messages import HumanMessage
from pydantic import ValidationError


from ... import render, retrieval
from ...citations import cited_numbers, has_citation, keep_cited
from ...errors import TutorUnavailable
from ...learner import NOTHING_RECORDED
from ...models import AnswerBasis, Citation, MathFigureSpec, PrereqTopic, RetrievedChunk, Style
from ...prompts import (
    BEYOND_LESSON_PROMPT,
    EXPLAIN_LESSON_ONLY,
    EXPLAIN_PROMPT,
    EXPLAIN_WITHIN_SUBJECT,
    FORMATTING_CONTRACT,
    NEXT_STEP,
    NO_MATERIAL_PROMPT,
    SUMMARIZE_PROMPT,
    TEACHING_APPROACHES,
)
from ...retrieval import DEFAULT_MIN_SCORE, DEFAULT_SCOPE_RATIO
from ...scope import StudyScope, lecture_filter, scope_of
from ...tools import get_chunks_by_id

# A trailing ```json math-figure block the math prompt may append (see MATH_PROMPT_INSTRUCTIONS).
_MATH_BLOCK = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)

logger = logging.getLogger(__name__)


class ExplainSummarizeState(TypedDict, total=False):
    mode: str                       # explain | summarize
    query: str
    level: str
    length: str
    topic_ids: list[str]
    namespaces: list[str]
    style: Style | None
    voice: str
    language: str
    subject_id: str | None
    lecture_ids: list[str] | None  # optional lecture scope; None -> whole subject
    study_scope: dict | None       # the UI selection this turn is bounded by
    message: str                   # what the student typed, before the planner reworded it
    history: str
    learner: str                   # graded evidence about this student (agent/learner.py)
    previous_chunk_ids: list[str]
    grade: str | None
    instructions: str               # subject-specific prompt addendum (e.g. math), or ""
    k: int
    prereq_depth: int
    prereq_limit: int
    min_score: float
    topic_filter: str               # soft = a topic-filtered miss retries unscoped (config agent.topic_filter)
    scope_relevance_ratio: float    # in-scope share of the unscoped best score (config agent.scope.relevance_ratio)
    expand_anchors: int
    expand_window: int
    context_limit: int
    whole_lesson: bool
    approach: str                   # models.TeachingApproach, from the turn reading
    scope_fit: str                  # models.ScopeFit: topic | foundation | outside
    subject_name: str | None
    capabilities: str               # the tasks this user can run, for the next-step offer
    beyond_lessons: bool            # config agent.beyond_lessons
    # produced
    chunks: list[RetrievedChunk]
    prereq_topics: list[PrereqTopic]
    explanation: str
    grounded: bool                  # answer used the context (inline [n] or a math figure); a decline is False
    basis: AnswerBasis              # lesson | beyond | declined
    outside_lesson: bool            # declined because the selected lesson does not cover it
    citations: list[Citation]
    visual_spec: dict[str, Any]     # a figure request the parent's visual_sg renders


def _retrieve(state: ExplainSummarizeState, config) -> dict:
    """The passages this turn answers from.

    A request about the selected lesson as a whole reads that lesson; there is
    nothing in "this lesson" for a search to match.
    A turn that carries the last answer on starts from the passages that answer
    rested on, reached by id, and adds whatever a search for the rewritten
    question finds. The rewrite names its subject, so the search is a real one;
    the passages already on the table keep the answer continuous. New passages
    pass the lesson scope, the score floor and the relevance ratio; reused ones
    passed them on the turn that found them.
    """
    scope = scope_of(state)
    if scope.is_narrow and state.get("whole_lesson"):
        return {"chunks": retrieval.whole_lesson(
            scope.lecture_id, state["namespaces"],
            limit=state.get("context_limit", state.get("k", 8)))}
    subject_id = scope.subject_id or state.get("subject_id")
    query = state.get("query") or state.get("message") or ""
    k = state.get("k", 8)
    # The basics of a lesson are usually taught in an earlier one, so a question
    # about them searches the whole subject rather than the lesson selected.
    foundation = state.get("scope_fit") == "foundation"
    found = retrieval.retrieve(
        query,
        state["namespaces"],
        k=k,
        subject_id=subject_id,
        topic_ids=state.get("topic_ids") or None,
        lecture_ids=None if foundation else lecture_filter(state),
        min_score=state.get("min_score", DEFAULT_MIN_SCORE),
        topic_filter=state.get("topic_filter", "hard"),
    )
    outside = False
    if found and scope.is_narrow and not foundation:
        ratio = state.get("scope_relevance_ratio") or DEFAULT_SCOPE_RATIO
        if not retrieval.answers_within_scope(
            query, state["namespaces"], found[0].score,
            subject_id=subject_id, ratio=ratio,
        ):
            logger.info("question falls outside the selected lesson", extra={
                "lecture_id": scope.lecture_id, "scoped_best": round(found[0].score, 3)})
            found, outside = [], True
    found = retrieval.with_neighbours(
        found,
        anchors=state.get("expand_anchors", 0),
        window=state.get("expand_window", 0),
        limit=state.get("context_limit", k),
    )
    # A summary of the selected lesson that the turn reading did not mark as
    # being about the whole lesson still has the lesson to summarize: "summarize
    # it" scores under the floor against every passage, and a lesson whose
    # passages carry no topic finds nothing by tag.
    if (not found and not outside and not foundation and scope.is_narrow
            and state.get("mode") == "summarize" and state.get("scope_fit") != "outside"):
        found = retrieval.whole_lesson(scope.lecture_id, state["namespaces"],
                                       limit=state.get("context_limit", k))
    carried = _carried_passages(state.get("previous_chunk_ids") or [])
    if not carried:
        return {"chunks": found}
    known = {chunk.chunk_id for chunk in carried}
    merged = carried + [chunk for chunk in found if chunk.chunk_id not in known]
    limit = state.get("context_limit", k)
    logger.info("turn continues the last answer", extra={
        "carried": len(carried), "found": len(found), "kept": min(len(merged), limit)})
    return {"chunks": merged[:limit]}


def _carried_passages(chunk_ids: list[str]) -> list[RetrievedChunk]:
    """The last answer's passages, or none when a re-ingest has since replaced them."""
    if not chunk_ids:
        return []
    chunks = get_chunks_by_id(chunk_ids)
    if not chunks:
        logger.info("the last answer's passages no longer resolve",
                    extra={"chunks": len(chunk_ids)})
    return chunks


def _resolve_prereqs(state: ExplainSummarizeState) -> dict:
    """What the student should know first, from the passages the answer cited.

    Walking from topics guessed off the words of the question named whatever the
    guess touched — a question about wind turbines listed the states of water.
    The passages an answer actually cites are the only evidence of what it was
    about, so their topics seed the walk, and an answer that cited nothing has
    no prerequisites.
    """
    cited = {c.chunk_id for c in state.get("citations") or [] if c.chunk_id}
    if not (state.get("grounded") and cited):
        return {"prereq_topics": []}
    seeds: list[str] = []
    for chunk in state.get("chunks") or []:
        if chunk.chunk_id in cited:
            seeds.extend(t for t in chunk.topic_ids if t not in seeds)
    if not seeds:
        return {"prereq_topics": []}
    prereq_topics = retrieval.resolve_prereqs(
        seeds,
        depth=state.get("prereq_depth", 2),
        limit=state.get("prereq_limit", retrieval.DEFAULT_MAX_PREREQS),
    )
    return {"prereq_topics": prereq_topics}


def _within_subject(mode: str, state: ExplainSummarizeState) -> bool:
    """Whether this answer may teach what the lessons leave out.

    Only an explanation, only when the deployment allows it, only for a question
    the turn reading placed inside the subject, and only once a subject is known
    to stay inside. A summary is of the material, so it never goes beyond it.
    Inside a selected lesson only that lesson's foundations may: another topic
    of the subject is still refused by the lesson's name.
    """
    scope = scope_of(state)
    fit = state.get("scope_fit", "topic")
    if scope.is_narrow and fit != "foundation":
        return False
    return (mode == "explain"
            and bool(state.get("beyond_lessons"))
            and fit != "outside"
            and bool(scope.subject_id or state.get("subject_id")))


def _route_grounding(mode: str):
    def route(state: ExplainSummarizeState) -> str:
        if state.get("chunks"):
            return "answer"
        if _within_subject(mode, state):
            return "answer_beyond"
        return "decline"
    return route


def _style(state: ExplainSummarizeState) -> str:
    st = state.get("style")
    return st.tone if st else "neutral"


def _subject(state: ExplainSummarizeState) -> str:
    return state.get("subject_name") or "the school subject this conversation is about"


def _teaching(state: ExplainSummarizeState) -> dict:
    """The slots both explanation prompts share: how to teach it and what to offer next."""
    capabilities = state.get("capabilities") or ""
    return {
        "approach": TEACHING_APPROACHES[state.get("approach") or "direct"],
        "next_step": NEXT_STEP.format(capabilities=capabilities) if capabilities else "",
    }


def _build_decline(llm):
    def decline(state: ExplainSummarizeState) -> dict:
        scope = scope_of(state)
        logger.info("answer declined (no grounding)", extra={
            "mode": state.get("mode"), "chunks": 0, "lecture_id": scope.lecture_id,
            "scope_fit": state.get("scope_fit")})
        return {"explanation": _declined(scope, state, llm, _style(state)),
                "citations": [], "grounded": False, "basis": "declined",
                "outside_lesson": scope.is_narrow and not state.get("whole_lesson")}
    return decline


def _build_answer_beyond(mode: str, llm, config):
    """Teach a question about the subject that the lessons hold nothing on.

    A follow-up never lands here: it carries the passages the last answer rested
    on, so it is answered from them, filling only the steps they leave out.
    """
    def answer_beyond(state: ExplainSummarizeState) -> dict:
        logger.info("answer beyond the lessons", extra={
            "mode": mode, "scope_fit": state.get("scope_fit"),
            "approach": state.get("approach")})
        if llm is None:
            return {"explanation": "[Explanation pending chat model] beyond the lessons",
                    "citations": [], "grounded": False, "basis": "beyond"}
        grade = state.get("grade")
        prompt_text = BEYOND_LESSON_PROMPT.render(
            language=state.get("language", "en"),
            subject=_subject(state),
            grade=f", grade {grade}" if grade else "",
            level=state.get("level", "intermediate"),
            length=state.get("length", "medium"),
            style=_style(state),
            instructions=state.get("instructions", ""),
            formatting=FORMATTING_CONTRACT,
            voice=state.get("voice") or "",
            history=state.get("history") or "",
            learner=state.get("learner") or NOTHING_RECORDED,
            message=state.get("message") or state.get("query", ""),
            query=state.get("query", ""),
            **_teaching(state),
        )
        started = time.perf_counter()
        try:
            content = llm.invoke([HumanMessage(content=prompt_text)]).content
        except Exception as error:
            logger.error("answer LLM call failed", exc_info=True)
            raise TutorUnavailable() from error
        if not (content or "").strip():
            raise TutorUnavailable()
        logger.info("answer finished", extra={
            "mode": mode, "basis": "beyond",
            "duration_s": round(time.perf_counter() - started, 3)})
        clean, math_spec = _math_figure_from_answer(content, config)
        out = {"explanation": clean, "citations": [], "grounded": False, "basis": "beyond"}
        if math_spec:
            out["visual_spec"] = math_spec
        return out
    return answer_beyond


def _build_answer(mode: str, llm, config):
    prompt = EXPLAIN_PROMPT if mode == "explain" else SUMMARIZE_PROMPT

    def _answer(state: ExplainSummarizeState) -> dict:
        chunks = state.get("chunks", [])
        primary_ctx, citations = retrieval.render_context(chunks, start=1)
        style = _style(state)

        if llm is None:  # no chat key: deterministic grounded stub
            text = _stub_answer(mode, chunks, state.get("language", "en"))
            return _finalize(text, citations, mode, state, chunks, config)

        logger.info("answer started", extra={"mode": mode, "chunks": len(chunks)})
        started = time.perf_counter()

        instructions = state.get("instructions", "")
        grounding = (EXPLAIN_WITHIN_SUBJECT.format(subject=_subject(state))
                     if _within_subject(mode, state) else EXPLAIN_LESSON_ONLY)
        prompt_text = prompt.render(language=state.get("language", "en"),
            grounding=grounding,
            **_teaching(state),
            level=state.get("level", "intermediate"),
            length=state.get("length", "medium"),
            style=style,
            instructions=instructions,
            formatting=FORMATTING_CONTRACT,
            voice=state.get("voice") or "",
            history=state.get("history") or "",
            learner=state.get("learner") or NOTHING_RECORDED,
            context=primary_ctx,
            message=state.get("message") or state.get("query", ""),
            query=state.get("query", ""),
        ) if mode == "explain" else prompt.render(language=state.get("language", "en"),
            length=state.get("length", "medium"),
            style=style,
            instructions=instructions,
            formatting=FORMATTING_CONTRACT,
            voice=state.get("voice") or "",
            history=state.get("history") or "",
            learner=state.get("learner") or NOTHING_RECORDED,
            context=primary_ctx,
            message=state.get("message") or state.get("query", ""),
            query=state.get("query", ""),
        )
        try:
            content = llm.invoke([HumanMessage(content=prompt_text)]).content
        except Exception as error:
            logger.error("answer LLM call failed", exc_info=True)
            raise TutorUnavailable() from error
        logger.info("answer finished", extra={
            "mode": mode, "duration_s": round(time.perf_counter() - started, 3)})
        return _finalize(content, citations, mode, state, chunks, config)

    return _answer


def _finalize(text, citations, mode, state, chunks, config) -> dict:
    """Assemble the answer result. A math figure the model emitted (parsed + sampled) is the
    visual_spec; otherwise fall back to the figure-reuse spec. The json block is stripped from
    the answer text so the reader never sees it.

    ``grounded`` = the answer actually used the context (it carries at least one inline [n]
    citation). A refusal ("the context does not contain...") cites nothing; the parent then
    suppresses the sources/prerequisites/figure that would make a decline look answered."""
    clean, math_spec = _math_figure_from_answer(text, config)
    # Grounded = the answer used the material: an inline [n] citation, or an emitted math
    # figure (a refusal has neither). Ungrounded -> suppress sources/prereqs/figure.
    grounded = has_citation(clean) or bool(math_spec)
    # Retrieval hands over every chunk it found; the answer usually leans on two or
    # three. Publishing the rest lists lessons the student was never pointed at.
    cited = keep_cited(citations, cited_numbers(clean)) if grounded else []
    basis = ("lesson" if grounded
             else "beyond" if _within_subject(mode, state) else "declined")
    out = {"explanation": clean, "grounded": grounded, "citations": cited, "basis": basis}
    if grounded:
        spec = math_spec or _figure_spec(mode, state, chunks, config)
        if spec:
            out["visual_spec"] = spec
    return out


def _math_figure_from_answer(text, config) -> tuple[str, dict | None]:
    """Extract a trailing ```json math-figure block; return (clean_text, visual_spec|None).
    The spec's params are sampled here (expr+domain -> points) so prepare renders directly."""
    if not text:
        return text, None
    matches = list(_MATH_BLOCK.finditer(text))
    if not matches:
        return text, None
    m = matches[-1]
    # This block is the one piece of model output a student's own message can
    # steer into the renderer, so it is validated like any other reply rather
    # than taken as whatever JSON happens to parse.
    try:
        figure = MathFigureSpec.model_validate_json(m.group(1))
    except ValidationError:
        logger.warning("discarding a malformed math figure block")
        return text, None
    if not (figure.functions or figure.geometry):
        return text, None
    data = figure.model_dump(exclude_none=True)
    spec = render.sample_math_spec(data, config)
    if not (spec.get("series") or spec.get("geometry")):
        return text, None
    kind = "geometry" if spec.get("geometry") else "math"
    clean = (text[:m.start()] + text[m.end():]).rstrip()
    return clean, {"kind": kind, "params": spec}


def _figure_spec(mode, state, chunks, config) -> dict | None:
    """Emit a ``figure`` visual_spec so an explanation also surfaces the lecture's diagram
    (visual_sg reuses a syllabus figure — no generation). Config-gated; explain only."""
    if mode != "explain" or not config.get("render.figures_with_explain", False):
        return None
    topic_ids = state.get("topic_ids") or None
    lecture_id = chunks[0].lecture_id if chunks else None
    if not topic_ids and not lecture_id:
        return None
    # The primary grounding page — visual_sg prefers the figure nearest it.
    page = chunks[0].page_from if chunks else None
    return {"kind": "figure", "topic_ids": topic_ids, "lecture_id": lecture_id,
            "page": page}


def _declined(scope: StudyScope, state: ExplainSummarizeState, llm, style: str) -> str:
    """Why nothing was found, told by the tutor in the conversation's own terms.

    A fixed sentence per language answered every miss the same way, whatever was
    asked and however the conversation had gone. The model writes it instead,
    told only that the lessons hold nothing on the question and, inside a lesson,
    which lesson that is, so it cannot answer from its own knowledge. A model
    that fails to write it fails the turn rather than falling back to a sentence
    in some language.
    """
    language = state.get("language", "en")
    if llm is None:
        return _decline_without_model(scope)
    lesson = scope.label()
    where = (f' They are studying the lesson "{lesson}", and this is outside it; say so.'
             if scope.is_narrow and lesson else "")
    prompt = NO_MATERIAL_PROMPT.render(
        language=language, style=style, scope=where,
        suggest=retrieval.what_to_suggest(scope, state.get("subject_id")),
        history=state.get("history") or "",
        message=state.get("message") or state.get("query", ""),
        voice=state.get("voice") or "")
    try:
        text = (llm.invoke([HumanMessage(content=prompt)]).content or "").strip()
    except Exception as error:
        raise TutorUnavailable() from error
    if not text:
        raise TutorUnavailable()
    return text


def _decline_without_model(scope: StudyScope) -> str:
    """The decline for a turn with no model to write one: the lesson that bounded
    it, when there was one, and nothing a reader must know a language to read."""
    lesson = scope.label()
    return f"∅ {lesson}" if scope.is_narrow and lesson else "∅"


def _stub_answer(mode: str, chunks: list[RetrievedChunk], language: str) -> str:
    heads = [c.heading_path or c.lecture_id for c in chunks[:3]]
    joined = "; ".join(h for h in heads if h)
    verb = "Explanation" if mode == "explain" else "Summary"
    return f"[{verb} pending chat model] grounded in: {joined} [1]"


def _build(mode: str, *, config, llm=None):
    from langgraph.graph import END, START, StateGraph

    graph = StateGraph(ExplainSummarizeState)
    graph.add_node("retrieve", lambda state: _retrieve(state, config))
    graph.add_node("resolve_prereqs", _resolve_prereqs)
    graph.add_node("answer", _build_answer(mode, llm, config))
    graph.add_node("answer_beyond", _build_answer_beyond(mode, llm, config))
    graph.add_node("decline", _build_decline(llm))
    graph.add_edge(START, "retrieve")
    graph.add_conditional_edges("retrieve", _route_grounding(mode),
                                ["answer", "answer_beyond", "decline"])
    for node in ("answer", "answer_beyond", "decline"):
        graph.add_edge(node, "resolve_prereqs")
    graph.add_edge("resolve_prereqs", END)
    return graph.compile()


def build_explain_sg(*, config, llm=None):
    return _build("explain", config=config, llm=llm)


def build_summarize_sg(*, config, llm=None):
    return _build("summarize", config=config, llm=llm)
