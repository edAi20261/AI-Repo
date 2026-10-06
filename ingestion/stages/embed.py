"""embed stage — one vector per chunk (BGE-M3).

The embedding is attached to the chunk in transit; it goes to Qdrant in the
index stage, never to Postgres.

What is embedded and how the result is attached are separate functions because
a book can be embedded on several machines at once (``ingestion.distributed``):
each embeds a share of the same texts, and the vectors are attached afterwards.
Both paths go through these two functions, so a chunk embedded on one machine
is indistinguishable from one embedded here.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from ..context import PipelineContext
from ..state import Chunk, IngestionState


def embedding_texts(chunks: Sequence[Chunk], config: Any) -> list[str]:
    """What each chunk is searched by, in chunk order."""
    prefixed = bool(config.get("ingestion.context_prefix"))
    return [c.for_embedding() if prefixed else c.text for c in chunks]


def attach(
    state: IngestionState,
    config: Any,
    vectors: Sequence[list[float]],
    sparse: Optional[Sequence[dict[int, float]]],
    *,
    dimension: int,
) -> IngestionState:
    """Put each chunk's vectors on it and record what was embedded."""
    chunks = state.get("chunks", [])
    if len(vectors) != len(chunks):
        raise ValueError(f"{len(vectors)} vectors for {len(chunks)} chunks")
    hybrid = bool(config.get("vector_store.hybrid"))
    if hybrid and (sparse is None or len(sparse) != len(chunks)):
        raise ValueError("the index is hybrid and the lexical vectors are missing")
    for chunk, vector in zip(chunks, vectors):
        chunk.embedding = vector
    # The lexical half, from the same text the dense half was built from: a
    # chunk searched by words it was not indexed by is a chunk nobody finds.
    if hybrid:
        for chunk, terms in zip(chunks, sparse):
            chunk.sparse = terms

    state.setdefault("stats", {})["embed"] = {
        "embedded": len(vectors),
        "dimension": dimension,
        "context_prefix": bool(config.get("ingestion.context_prefix")),
        "sparse": hybrid,
    }
    return state


def run(state: IngestionState, ctx: PipelineContext) -> IngestionState:
    chunks = state.get("chunks", [])
    if not chunks:
        state.setdefault("stats", {})["embed"] = {"embedded": 0}
        return state

    texts = embedding_texts(chunks, ctx.config)
    vectors = ctx.embedder.embed(texts)
    sparse = (ctx.embedder.embed_sparse(texts)
              if ctx.config.get("vector_store.hybrid") else None)
    return attach(state, ctx.config, vectors, sparse, dimension=ctx.embedder.dimension)
