"""explain_sg / summarize_sg — structurally identical RAG subgraphs.

    retrieve -> resolve_prereqs -> answer

RAG is hard-wired: no branch goes around ``retrieve``. Both modes
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
from typing import Any, Optional, TypedDict

from langchain_core.messages import HumanMessage
from pydantic import ValidationError

from shared.language import language_directive, translate

from ... import render, retrieval
from ...citations import cited_numbers, has_citation, keep_cited
from ...models import MathFigureSpec, Citation, PrereqTopic, RetrievedChunk, Style
from ...prompts import EXPLAIN_PROMPT, FORMATTING_CONTRACT, SUMMARIZE_PROMPT

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
    style: Optional[Style]
    language: str
    subject_id: Optional[str]
    lecture_ids: Optional[list[str]]  # optional lecture scope; None -> whole subject
    grade: Optional[str]
    instructions: str               # subject-specific prompt addendum (e.g. math), or ""
    k: int
    prereq_depth: int
    prereq_limit: int
    min_score: float
    topic_filter: str               # soft = a topic-filtered miss retries unscoped (config agent.topic_filter)
    # produced
    chunks: list[RetrievedChunk]
    prereq_chunks: list[RetrievedChunk]
    prereq_topics: list[PrereqTopic]
    explanation: str
    grounded: bool                  # answer used the context (inline [n] or a math figure); a decline is False
    citations: list[Citation]
    visual_spec: dict[str, Any]     # a figure request the parent's visual_sg renders


def _retrieve(state: ExplainSummarizeState) -> dict:
    chunks = retrieval.retrieve(
        state.get("query") or "",
        state["namespaces"],
        k=state.get("k", 8),
        subject_id=state.get("subject_id"),
        topic_ids=state.get("topic_ids") or None,
        lecture_ids=state.get("lecture_ids") or None,
        min_score=state.get("min_score", 0.0),
        topic_filter=state.get("topic_filter", "hard"),
    )
    return {"chunks": chunks}


def _resolve_prereqs(state: ExplainSummarizeState) -> dict:
    prereq_topics, prereq_chunks = retrieval.resolve_prereqs(
        state.get("topic_ids") or [],
        state["namespaces"],
        depth=state.get("prereq_depth", 2),
        limit=state.get("prereq_limit", retrieval.DEFAULT_MAX_PREREQS),
    )
    return {"prereq_topics": prereq_topics, "prereq_chunks": prereq_chunks}


def _build_answer(mode: str, llm, config):
    prompt = EXPLAIN_PROMPT if mode == "explain" else SUMMARIZE_PROMPT

    def _answer(state: ExplainSummarizeState) -> dict:
        chunks = state.get("chunks", [])
        prereq_chunks = state.get("prereq_chunks", [])
        primary_ctx, citations = retrieval.render_context(chunks, start=1)
        prereq_ctx, prereq_cites = retrieval.render_context(
            prereq_chunks, start=len(chunks) + 1)
        st = state.get("style")
        style = st.tone if st else "neutral"

        if not chunks and not prereq_chunks:
            logger.info("answer declined (no grounding)", extra={
                "mode": mode, "chunks": 0, "prereq_chunks": 0})
            return {"explanation": _no_context_message(state.get("language", "en")),
                    "citations": [], "grounded": False}

        if llm is None:  # no chat key: deterministic grounded stub
            text = _stub_answer(mode, chunks, state.get("language", "en"))
            return _finalize(text, citations, mode, state, chunks, config)

        logger.info("answer started", extra={
            "mode": mode, "chunks": len(chunks), "prereq_chunks": len(prereq_chunks)})
        started = time.perf_counter()

        instructions = state.get("instructions", "")
        prompt_text = prompt.format(
            language=language_directive(state.get("language", "en")),
            level=state.get("level", "intermediate"),
            length=state.get("length", "medium"),
            style=style,
            instructions=instructions,
            formatting=FORMATTING_CONTRACT,
            context=primary_ctx,
            prereq_context=prereq_ctx or "(none)",
            query=state.get("query", ""),
        ) if mode == "explain" else prompt.format(
            language=language_directive(state.get("language", "en")),
            length=state.get("length", "medium"),
            style=style,
            instructions=instructions,
            formatting=FORMATTING_CONTRACT,
            context=primary_ctx,
            query=state.get("query", ""),
        )
        try:
            content = llm.invoke([HumanMessage(content=prompt_text)]).content
        except Exception:
            # Runtime LLM failure (key present, call failed): a friendly message, not a stub
            # or a stack trace. No citations/figure on a degraded answer.
            logger.error("answer LLM call failed", exc_info=True)
            return {"explanation": translate("common.degraded", state.get("language", "en")),
                    "citations": [], "grounded": False}
        logger.info("answer finished", extra={
            "mode": mode, "duration_s": round(time.perf_counter() - started, 3)})
        return _finalize(content, citations + prereq_cites, mode, state, chunks, config)

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
    out = {"explanation": clean, "grounded": grounded, "citations": cited}
    if grounded:
        spec = math_spec or _figure_spec(mode, state, chunks, config)
        if spec:
            out["visual_spec"] = spec
    return out


def _math_figure_from_answer(text, config) -> tuple[str, Optional[dict]]:
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


def _figure_spec(mode, state, chunks, config) -> Optional[dict]:
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


def _no_context_message(language: str) -> str:
    return translate("explain.no_context", language)


def _stub_answer(mode: str, chunks: list[RetrievedChunk], language: str) -> str:
    heads = [c.heading_path or c.lecture_id for c in chunks[:3]]
    joined = "; ".join(h for h in heads if h)
    verb = "Explanation" if mode == "explain" else "Summary"
    return f"[{verb} pending chat model] grounded in: {joined} [1]"


def _build(mode: str, *, config, llm=None):
    from langgraph.graph import END, START, StateGraph

    graph = StateGraph(ExplainSummarizeState)
    graph.add_node("retrieve", _retrieve)
    graph.add_node("resolve_prereqs", _resolve_prereqs)
    graph.add_node("answer", _build_answer(mode, llm, config))
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "resolve_prereqs")
    graph.add_edge("resolve_prereqs", "answer")
    graph.add_edge("answer", END)
    return graph.compile()


def build_explain_sg(*, config, llm=None):
    return _build("explain", config=config, llm=llm)


def build_summarize_sg(*, config, llm=None):
    return _build("summarize", config=config, llm=llm)
