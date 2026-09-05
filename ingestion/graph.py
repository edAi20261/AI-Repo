"""Ingestion graph — extract -> chunk -> embed -> topics -> dependencies -> index.

Linear, but expressed as a LangGraph ``StateGraph`` for tracing/checkpointing.
LangGraph is imported lazily; if absent, the stages run sequentially instead.
"""

from __future__ import annotations

import logging
from typing import Optional

from shared.logging import redact_uri
from shared.metering import start_meter

from . import stages
from .config import Config, get_config
from .context import PipelineContext
from .state import IngestionState

logger = logging.getLogger(__name__)

_STAGE_ORDER = ("extract", "chunk", "embed", "topics", "dependencies", "index")


def build_graph(ctx: PipelineContext):
    """Compile the LangGraph ``StateGraph`` for ingestion."""
    from langgraph.graph import END, START, StateGraph

    graph = StateGraph(IngestionState)
    for name in _STAGE_ORDER:
        graph.add_node(name, _node(getattr(stages, name), ctx, name))

    graph.add_edge(START, _STAGE_ORDER[0])
    for prev, nxt in zip(_STAGE_ORDER, _STAGE_ORDER[1:]):
        graph.add_edge(prev, nxt)
    graph.add_edge(_STAGE_ORDER[-1], END)
    return graph.compile()


def _node(stage_fn, ctx: PipelineContext, name: str):
    def _run(state: IngestionState) -> IngestionState:
        return _run_stage(stage_fn, state, ctx, name)

    return _run


def _run_stage(stage_fn, state: IngestionState, ctx: PipelineContext, name: str) -> IngestionState:
    state = stage_fn(state, ctx)
    logger.info("stage complete", extra={
        "stage": name, "lecture_id": state.get("lecture_id"),
        "stats": state.get("stats", {}).get(name),
    })
    return state


def run_ingestion(
    state: IngestionState,
    *,
    ctx: Optional[PipelineContext] = None,
    config: Optional[Config] = None,
) -> IngestionState:
    """Run one lesson through the pipeline. Uses the compiled LangGraph when
    available, else the sequential fallback."""
    cfg = config or get_config()
    ctx = ctx or PipelineContext.build(cfg)
    lecture_id = state.get("lecture_id")
    logger.info("lesson ingest start", extra={
        "lecture_id": lecture_id, "source_uri": redact_uri(state.get("source_uri"))})
    meter = start_meter()  # accumulates LLM cost/tokens across all stages of this run
    try:
        graph = build_graph(ctx)
    except ImportError:
        result = _run_sequential(state, ctx)
    else:
        result = graph.invoke(state)
    cost = meter.as_dict()
    result.setdefault("stats", {})["cost"] = cost
    logger.info("lesson ingest done", extra={
        "lecture_id": lecture_id,
        "ingestion_status": result.get("ingestion_status") or "ready",
        "cost_usd": cost["cost_usd"], "llm_calls": cost["calls"]})
    return result


def _run_sequential(state: IngestionState, ctx: PipelineContext) -> IngestionState:
    for name in _STAGE_ORDER:
        state = _run_stage(getattr(stages, name), state, ctx, name)
    return state
