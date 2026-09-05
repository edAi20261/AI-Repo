"""index stage — persist chunks, figures, Qdrant points and the manifest.

Order matters: ``ai.chunks`` first assigns each ``chunk_id``, which then keys the
figure rows, the Qdrant points and the ``ai.vector_index`` manifest. Writes only
tables ingestion owns (chunks, figures, vector_index).
"""

from __future__ import annotations

from ..context import PipelineContext
from ..state import Chunk, Figure, IngestionState


def run(state: IngestionState, ctx: PipelineContext) -> IngestionState:
    chunks: list[Chunk] = state.get("chunks", [])
    figures: list[Figure] = state.get("figures", [])

    # 0. Replace, don't append: drop any prior rows/points for this lecture so a
    #    re-ingest is idempotent. Topics are subject-owned and kept (invariant 11).
    #    Ensure the collection + payload indexes FIRST — the Qdrant delete filters
    #    on lecture_id, which 400s on an unindexed collection.
    lecture_id = state["lecture_id"]
    ctx.vector_store.ensure_collection(ctx.config.get("embeddings.dimensions", 1024))
    ctx.chunk_store.delete_lecture(lecture_id)
    ctx.vector_store.delete_lecture(lecture_id)

    # 1. Postgres: ai.chunks -> assigns and returns chunk_id on each chunk.
    chunk_ids = ctx.chunk_store.write_chunks(chunks)

    # 2. Link figures to the chunk covering their page, then write ai.figures.
    _link_figures_to_chunks(figures, chunks)
    ctx.chunk_store.write_figures(figures)

    # 3. Qdrant: one point per embedded chunk (skip if embed stage was skipped).
    upserted = 0
    embedded = [c for c in chunks if c.embedding is not None]
    if embedded:
        upserted = ctx.vector_store.upsert(embedded)

    # 4. Postgres: ai.vector_index manifest (idempotent on chunk_id, namespace).
    ctx.chunk_store.write_vector_index(chunks)

    # 5. Per-lecture tracking row: current status, counts and time (upsert).
    errors = state.get("errors", [])
    status = state.get("ingestion_status") or "ready"
    ctx.chunk_store.write_ingestion_record({
        "lecture_id": lecture_id,
        "subject_id": state.get("subject_id"),
        "version": state.get("version", 1),
        "status": status,
        "chunk_count": len(chunk_ids),
        "figure_count": len(figures),
        "topic_count": state.get("stats", {}).get("topics", {}).get("extracted", 0),
        "needs_review_reason": "; ".join(errors) if status != "ready" and errors else None,
    })

    state.setdefault("stats", {})["index"] = {
        "chunks_written": len(chunk_ids),
        "figures_written": len(figures),
        "points_upserted": upserted,
    }
    return state


def _link_figures_to_chunks(figures: list[Figure], chunks: list[Chunk]) -> None:
    """Attach each figure to the first chunk whose page range covers it, and
    inherit that chunk's topic_ids so ``get_figures`` by topic finds it."""
    for fig in figures:
        for chunk in chunks:
            if chunk.page_from <= fig.page <= chunk.page_to:
                fig.chunk_id = chunk.chunk_id
                if not fig.topic_ids:
                    fig.topic_ids = list(chunk.topic_ids)
                break
