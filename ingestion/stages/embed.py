"""embed stage — one vector per chunk (BGE-M3).

The embedding is attached to the chunk in transit; it goes to Qdrant in the
index stage, never to Postgres.
"""

from __future__ import annotations

from ..context import PipelineContext
from ..state import IngestionState


def run(state: IngestionState, ctx: PipelineContext) -> IngestionState:
    chunks = state.get("chunks", [])
    if not chunks:
        state.setdefault("stats", {})["embed"] = {"embedded": 0}
        return state

    prefixed = bool(ctx.config.get("ingestion.context_prefix"))
    texts = [c.for_embedding() if prefixed else c.text for c in chunks]
    vectors = ctx.embedder.embed(texts)
    for chunk, vector in zip(chunks, vectors):
        chunk.embedding = vector

    # The lexical half, from the same text the dense half was built from: a
    # chunk searched by words it was not indexed by is a chunk nobody finds.
    hybrid = bool(ctx.config.get("vector_store.hybrid"))
    if hybrid:
        for chunk, terms in zip(chunks, ctx.embedder.embed_sparse(texts)):
            chunk.sparse = terms

    state.setdefault("stats", {})["embed"] = {
        "embedded": len(vectors),
        "dimension": ctx.embedder.dimension,
        "context_prefix": prefixed,
        "sparse": hybrid,
    }
    return state
