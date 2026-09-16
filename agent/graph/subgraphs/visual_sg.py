"""visual_sg — turn visual_specs into artifact refs.

    collect -> prepare -> render

Private state (``VisualState``). ``collect`` dedupes and caps to ``render.max_per_turn``
(config: 3; the ``1`` in ``_collect`` is only the fallback if the parent passes no cap).
``prepare`` fills each spec's render input (``params``) when it
is missing — building a mind-map outline, flashcards, a PDF's blocks, chart data or a math
spec from the turn's already-produced content (explanation/quiz/grade in ``content``), grounded
by that content or, for a standalone visual, by a min-score RAG on the topic (invariant 1/9).
``render`` tries ``get_figures`` FIRST for a grounded spec (reuse a syllabus figure before
generating), else routes ``kind`` to a deterministic render tool (image models only for the
``image`` kind). User-agnostic — same for student, teacher and parent. Precise math/geometry
never goes through an image model.
"""

from __future__ import annotations

import logging
from typing import Any, TypedDict

from langchain_core.messages import HumanMessage

from shared.structured import StructuredOutputError, call_structured, schema_hint

from ... import prompts as R_prompts
from ... import render as R
from ...models import (
    ArtifactRef,
    FlashcardSet,
    MathFigureSpec,
    MindmapNode,
    PdfDocument,
)
from ...retrieval import DEFAULT_MIN_SCORE
from ...tools import get_figures, rag_search

logger = logging.getLogger(__name__)


class VisualState(TypedDict, total=False):
    # inputs
    specs: list[dict[str, Any]]      # VisualSpec dicts
    subject_id: str | None
    lecture_ids: list[str] | None  # optional lecture scope for grounding RAG; None -> whole subject
    user_id: str | None
    conversation_id: str | None
    language: str
    max_per_turn: int
    namespaces: list[str]            # for standalone-visual grounding RAG
    content: dict[str, Any]          # the turn's produced content {explanations, assessment, grade_result}
    instructions: str                # subject-specific prompt addendum (e.g. math), or ""
    # produced
    visuals: list[dict[str, Any]]
    errors: int                      # count of specs that RAISED during render


def _spec_key(s: dict) -> tuple:
    return (s.get("kind"), s.get("item_id"),
            str(s.get("params")), s.get("description"))


def _collect(state: VisualState) -> dict:
    """Dedupe specs and cap to ``max_per_turn`` (default 1)."""
    seen: set = set()
    kept: list[dict] = []
    for s in state.get("specs") or []:
        k = _spec_key(s)
        if k in seen:
            continue
        seen.add(k)
        kept.append(s)
    cap = state.get("max_per_turn", 1)
    return {"specs": kept[:cap]}


# --- prepare: build each spec's render input ------------------------------------------

# Kinds whose render tool needs its input prepared first; image/figure already work from a
# description or a reused figure. slides needs grounded CONTENT (topic + syllabus), not a bare topic.
_NEEDS_BUILD = {"mindmap", "flashcards", "pdf", "chart", "math", "geometry", "slides"}


def _resolve_llm(config, task: str, override):
    """The LLM for a visual-builder task: an explicit ``override`` (tests) wins; otherwise the
    configured per-task model, or None when no key is set (builder then degrades). Shares the
    one 'LLM if available, else None' helper with the parent graph."""
    from shared.llm import maybe_llm
    return override if override is not None else maybe_llm(config, task)


def _has_input(kind: str, params: dict) -> bool:
    """True when the spec already carries a usable render input (subgraph-emitted or
    caller-supplied) — prepare then leaves it untouched."""
    if kind == "mindmap":
        return bool(params.get("label") or params.get("children"))
    if kind == "flashcards":
        return bool(params.get("cards"))
    if kind == "pdf":
        return bool(params.get("blocks"))
    if kind == "chart":
        return bool(params.get("series"))
    if kind in ("math", "geometry"):
        return bool(params.get("series") or params.get("geometry"))
    if kind == "slides":
        return bool(params.get("content"))
    return True


def _context_text(spec: dict, content: dict, state: VisualState, config) -> str:
    """Grounding text for an LLM builder: the turn's already-produced (grounded) explanation
    text when a sibling task ran, else a min-score RAG on the topic (invariant 1/9), else the
    bare description."""
    parts = [p for p in (content.get("explanations") or []) if p]
    if parts:
        return "\n\n".join(parts)
    query = spec.get("description") or ""
    namespaces = state.get("namespaces")
    if namespaces and query:
        try:
            floor = config.get("agent.min_score", DEFAULT_MIN_SCORE)
            chunks = rag_search(query, namespaces, subject_id=state.get("subject_id"),
                                topic_ids=spec.get("topic_ids"),
                                lecture_ids=state.get("lecture_ids") or None, config=config)
            grounded = [c.text for c in chunks if c.score >= floor]
            if grounded:
                return "\n\n".join(grounded)
        except Exception:
            logger.warning("visual grounding RAG failed", exc_info=True)
    return query


def _build_mindmap(desc: str, ctx: str, language: str, config, llm) -> dict | None:
    """A mind-map outline, or None to skip. A map with no branches is meaningless, so a
    failed/degenerate build renders nothing rather than a lone box (no visual > a wrong one)."""
    if llm is None:
        return None
    prompt_text = R_prompts.MINDMAP_PROMPT.render(
        topic=desc, context=ctx, language=language,
        schema=schema_hint(MindmapNode))
    try:
        node = call_structured(MindmapNode, [HumanMessage(content=prompt_text)],
                               task="visual_mindmap", config=config, llm=llm, label="mindmap")
    except StructuredOutputError as exc:
        logger.warning("mindmap build failed", extra=exc.log_fields())
        return None
    data = node.model_dump()
    return data if (data.get("label") and data.get("children")) else None


def _cards_from_assessment(content: dict) -> list[dict]:
    """Flashcards straight from this turn's quiz items (no LLM) — front=question, back=answer."""
    items = (content.get("assessment") or {}).get("items") or []
    cards = []
    for it in items:
        front = it.get("question")
        back = it.get("answer") or it.get("explanation") or ""
        if front:
            cards.append({"front": front, "back": _as_text(back)})
    return cards


def _build_flashcards(desc: str, content: dict, spec, state, config, llm) -> list[dict]:
    cards = _cards_from_assessment(content)
    if cards or llm is None:
        return cards
    ctx = _context_text(spec, content, state, config)
    n = config.get("render.flashcards.count", 8)
    prompt_text = R_prompts.FLASHCARDS_PROMPT.render(
        topic=desc, context=ctx, n=n,
        language=state.get("language", "en"),
        schema=schema_hint(FlashcardSet))
    try:
        deck = call_structured(FlashcardSet, [HumanMessage(content=prompt_text)],
                               task="visual_flashcards", config=config, llm=llm,
                               label="flashcards")
    except StructuredOutputError as exc:
        logger.warning("flashcards build failed", extra=exc.log_fields())
        return []
    return [c.model_dump() for c in deck.cards if c.front]


def _blocks_from_content(content: dict, desc: str) -> list[dict]:
    """PDF blocks assembled from the turn's content: a quiz's questions, else the explanation
    text, else the bare description."""
    items = (content.get("assessment") or {}).get("items") or []
    if items:
        blocks = []
        for i, it in enumerate(items, 1):
            blocks.append({"type": "heading", "text": f"{i}. {it.get('question', '')}"})
            if it.get("options"):
                for o in it["options"]:
                    blocks.append({"type": "text", "text": f"- {o.get('text', '')}"})
        return blocks
    parts = [p for p in (content.get("explanations") or []) if p]
    if parts:
        return [{"type": "text", "text": p} for p in parts]
    return [{"type": "text", "text": desc}] if desc else []


def _chart_from_content(content: dict, desc: str) -> dict | None:
    """Chart data from the turn's grade result (per-question scores). None when there is no
    data to plot — the spec is then skipped rather than drawing an empty axis.

    Charts are DATA-TIED, never LLM-fabricated. TODO: add real-data sources as they land —
    mastery_trend (get_mastery_trend, line over time), per-topic mastery (get_mastery, bar),
    report_sg aggregates (class/at-risk distributions). Each is a deterministic map like below."""
    graded = (content.get("grade_result") or {}).get("graded") or []
    if graded:
        labels = [f"Q{i}" for i in range(1, len(graded) + 1)]
        values = [float(g.get("partial_score") if g.get("partial_score") is not None
                        else (1.0 if g.get("correct") else 0.0)) for g in graded]
        return {"type": "bar", "title": desc or "Results", "labels": labels,
                "y_label": "score", "series": [{"name": "score", "values": values}]}
    return None


def _build_math(desc: str, language: str, config, llm) -> dict | None:
    if llm is None:
        return None
    prompt_text = R_prompts.MATH_VISUAL_PROMPT.render(
        topic=desc, language=language, schema=schema_hint(MathFigureSpec))
    try:
        figure = call_structured(MathFigureSpec, [HumanMessage(content=prompt_text)],
                                 task="visual_math", config=config, llm=llm, label="math figure")
    except StructuredOutputError as exc:
        logger.warning("math-visual build failed", extra=exc.log_fields())
        return None
    data = figure.model_dump(exclude_none=True)
    spec = R.sample_math_spec(data, config) if data else {}
    return spec if (spec.get("series") or spec.get("geometry")) else None


def _build_pdf(desc: str, ctx: str, language: str, llm, config=None) -> dict | None:
    """Structure grounded content into a titled set of heading/text blocks, or None to fall
    back to a single block. Grounded ONLY in ``ctx`` — the prompt reorganises, never invents."""
    if llm is None:
        return None
    prompt_text = R_prompts.PDF_STRUCTURE_PROMPT.render(
        topic=desc, content=ctx, language=language,
        schema=schema_hint(PdfDocument))
    try:
        document = call_structured(PdfDocument, [HumanMessage(content=prompt_text)],
                                   task="visual_pdf", config=config, llm=llm, label="pdf")
    except StructuredOutputError as exc:
        logger.warning("pdf build failed", extra=exc.log_fields())
        return None
    blocks = [b.model_dump() for b in document.blocks if b.text]
    if not blocks:
        return None
    return {"title": document.title, "blocks": blocks}


def _as_text(v: Any) -> str:
    return ", ".join(map(str, v)) if isinstance(v, list) else str(v or "")


def _build_input(kind: str, spec: dict, params: dict, content: dict,
                 state: VisualState, config, llm_override) -> dict | None:
    """Produce the render input for one spec, or None to skip it."""
    desc = spec.get("description") or ""
    lang = state.get("language", "en")
    if kind == "mindmap":
        llm = _resolve_llm(config, "visual_mindmap", llm_override)
        return _build_mindmap(desc, _context_text(spec, content, state, config), lang, config, llm)
    if kind == "flashcards":
        llm = _resolve_llm(config, "visual_flashcards", llm_override)
        cards = _build_flashcards(desc, content, spec, state, config, llm)
        return {"cards": cards, "title": params.get("title") or "Flashcards"} if cards else None
    if kind == "pdf":
        # A quiz PDF is assembled deterministically from the turn's items.
        items = (content.get("assessment") or {}).get("items") or []
        if items:
            return {"title": params.get("title") or "Document",
                    "blocks": _blocks_from_content(content, desc)}
        # Otherwise structure grounded content (the turn's explanation, else a min-score RAG on
        # the topic — invariant 1/9) into headed sections via the LLM, so the PDF has real
        # structure instead of one wall of text. Fall back to a single block if that fails.
        text = _context_text(spec, content, state, config)
        if not text or text == desc:
            return None
        llm = _resolve_llm(config, "visual_pdf", llm_override)
        built = _build_pdf(desc, text, state.get("language", "en"), llm, config)
        if built:
            built.setdefault("title", params.get("title") or "Document")
            return built
        return {"title": params.get("title") or "Document",
                "blocks": [{"type": "text", "text": text}]}
    if kind == "chart":
        return _chart_from_content(content, desc)
    if kind in ("math", "geometry"):
        llm = _resolve_llm(config, "visual_math", llm_override)
        return _build_math(desc, state.get("language", "en"), config, llm)
    if kind == "slides":
        # Slides are built from grounded CONTENT (the turn's explanation, else a RAG on the
        # topic), not the bare topic — powerpoint_points then outlines that content.
        text = _context_text(spec, content, state, config)
        return {"content": text} if text else None
    return None


def _build_prepare(config, llm_override):
    def prepare(state: VisualState) -> dict:
        content = state.get("content") or {}
        out: list[dict] = []
        for spec in state.get("specs") or []:
            spec = dict(spec)
            kind = spec.get("kind")
            params = dict(spec.get("params") or {})
            if kind in _NEEDS_BUILD and not _has_input(kind, params):
                built = _build_input(kind, spec, params, content, state, config, llm_override)
                if built is None:
                    logger.info("visual prepare: no input for kind %r; skipped", kind)
                    continue
                params = built
            spec["params"] = params
            out.append(spec)
        return {"specs": out}

    return prepare


def _reuse_figure(spec: dict, config) -> dict | None:
    """A syllabus figure for a grounded spec, or None. Reuse beats generation. When several
    figures match, prefer the one on the page nearest the spec's reference ``page`` (the
    grounding chunk's page) — a cheap, deterministic relevance proxy. (Caption-embedding
    ranking is the planned upgrade — see the pre-ingestion checklist.)"""
    if not (spec.get("topic_ids") or spec.get("lecture_id")):
        return None
    try:
        figs = get_figures(topic_ids=spec.get("topic_ids"),
                           lecture_id=spec.get("lecture_id"), config=config)
    except Exception:
        logger.warning("get_figures failed; will generate instead", exc_info=True)
        return None
    if not figs:
        return None
    f = _nearest_page(figs, spec.get("page"))
    return ArtifactRef(
        kind="figure", s3_uri=f.get("s3_uri"), figure_id=f.get("figure_id"),
        caption=f.get("caption"), reused=True, item_id=spec.get("item_id"),
    ).model_dump(exclude_none=True)


def _nearest_page(figs: list[dict], page: int | None) -> dict:
    """The figure on the page closest to ``page``; the first figure when no page hint (or
    no figure carries a page)."""
    if page is None:
        return figs[0]
    with_page = [f for f in figs if f.get("page") is not None]
    if not with_page:
        return figs[0]
    return min(with_page, key=lambda f: abs(f["page"] - page))


def _build_render(config, llm):
    def _render(state: VisualState) -> dict:
        specs = state.get("specs") or []
        # slides is the only render-time LLM; resolve per-task (or use the passed override).
        slides_llm = _resolve_llm(config, "slides", llm)
        common = {"user_id": state.get("user_id"),
                  "conversation_id": state.get("conversation_id"), "config": config}
        out: list[dict] = []
        errors = 0
        for spec in specs:
            kind = spec.get("kind")
            # `figure` = fetch a syllabus figure, never generate. Every other kind is a
            # specific generated artifact — never substitute a figure for it. (Image
            # generation is deferred, so there is no generate-a-picture path.)
            if kind == "figure":
                ref = _reuse_figure(spec, config)
                if not ref:
                    logger.info("no syllabus figure for a figure spec; skipped")
                    continue
            else:
                ref, err = _safe_render(spec, common, state, slides_llm)
                errors += err
            if ref:
                if spec.get("item_id"):
                    ref["item_id"] = spec["item_id"]
                out.append(ref)
        logger.info("visual render", extra={"specs": len(specs), "produced": len(out),
                                            "errors": errors})
        # `errors` = specs that RAISED during render (distinct from a spec that legitimately
        # produced nothing); the parent surfaces a "something went wrong" note vs a decline.
        return {"visuals": out, "errors": errors}

    return _render


def _safe_render(spec, common, state, llm) -> tuple[dict | None, int]:
    """(ref, errored) — errored is 1 when the render RAISED, so the caller can tell a real
    failure apart from a spec that simply produced nothing."""
    try:
        return _dispatch(spec, common, state, llm), 0
    except Exception:
        logger.error("visual render failed for kind %r", spec.get("kind"), exc_info=True)
        return None, 1


def _dispatch(spec: dict, common: dict, state: VisualState, llm) -> dict | None:
    """Route one spec's ``kind`` to its render tool."""
    kind = spec.get("kind")
    params = spec.get("params") or {}
    desc = spec.get("description") or ""
    if kind == "chart":
        return R.gen_chart(params, **common)
    if kind == "mindmap":
        return R.gen_mindmap(params, **common)
    if kind in ("math", "geometry"):
        return R.gen_math_visual(params, **common)
    if kind == "flashcards":
        return R.gen_flashcards(params.get("cards") or [],
                                title=params.get("title", "Flashcards"), **common)
    if kind == "pdf":
        return R.gen_pdf(params.get("title", "Document"), params.get("blocks") or [],
                         **common)
    if kind == "slides":
        # TODO: honour an explicit user-requested slide count (needs a count field threaded
        # from the planner's visual spec); until then the config cap keeps outlines digestible.
        max_slides = common["config"].get("render.slides.max_slides", 10)
        text = R.powerpoint_points(params.get("content") or desc, llm,
                                   language=state.get("language", "en"),
                                   instructions=state.get("instructions", ""),
                                   max_slides=max_slides)
        return ArtifactRef(kind="slides", text=text).model_dump(exclude_none=True)
    logger.warning("unknown visual kind %r; skipped", kind)
    return None


def build_visual_sg(*, config, llm=None):
    from langgraph.graph import END, START, StateGraph

    graph = StateGraph(VisualState)
    graph.add_node("collect", _collect)
    graph.add_node("prepare", _build_prepare(config, llm))
    graph.add_node("render", _build_render(config, llm))
    graph.add_edge(START, "collect")
    graph.add_edge("collect", "prepare")
    graph.add_edge("prepare", "render")
    graph.add_edge("render", END)
    return graph.compile()
