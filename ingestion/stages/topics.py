"""topics stage — extract a lesson's skills, reuse ids on re-ingest, tag chunks.

Runs after ``embed`` (reuses the embedder for labels and the chunk vectors for
tagging): an LLM names 3-8 skills; each is matched one-to-one against this
lecture's own existing topics by label cosine (a match reuses that topic_id so
mastery history stays attached — invariant 11); each chunk is tagged with its
closest topics. Writes ``ai.topics``. Stub returns none when no chat key is set.
"""

from __future__ import annotations

import uuid

from ..context import PipelineContext
from ..state import Chunk, IngestionState, Topic


def run(state: IngestionState, ctx: PipelineContext) -> IngestionState:
    chunks: list[Chunk] = state.get("chunks", [])
    stats = {"extracted": 0, "reused": 0, "new": 0, "chunks_tagged": 0}
    if not chunks:
        state["topics"] = []
        state.setdefault("stats", {})["topics"] = stats
        return state

    raw = ctx.topic_extractor.extract([c.text for c in chunks])
    if not raw and not ctx.topic_extractor.is_stub:
        # A non-stub extractor returning nothing for a non-empty lesson is a likely
        # transient LLM failure. Retry, then flag needs_review rather than silently
        # indexing untagged chunks (invariant 12).
        for _ in range(ctx.config.get("ingestion.topics.empty_retries", 2)):
            raw = ctx.topic_extractor.extract([c.text for c in chunks])
            if raw:
                break
        if not raw:
            state["ingestion_status"] = "needs_review"
            state.setdefault("errors", []).append(
                "topic extraction returned no topics for a non-empty lesson; "
                "chunks indexed without topic_ids."
            )
    if not raw:
        # Stub path (no chat key) or retries exhausted: leave topic_ids empty.
        state["topics"] = []
        state.setdefault("stats", {})["topics"] = stats
        return state

    cfg = ctx.config.get("ingestion.topics", {})
    reuse_threshold = cfg.get("reuse_threshold", 0.60)
    assign_threshold = cfg.get("assign_threshold", 0.45)

    labels = [t["label"] for t in raw]
    label_vecs = ctx.embedder.embed(labels)
    # Match against THIS lecture's own existing topics so re-ingest reuses ids
    # (lecture-scoped, not subject-wide: no cross-lecture merges).
    existing = ctx.chunk_store.read_lecture_topics(state["lecture_id"])
    reuse_ids = _match_one_to_one(label_vecs, existing, reuse_threshold)

    topics: list[Topic] = []
    for item, vec, match_id in zip(raw, label_vecs, reuse_ids):
        reused = match_id is not None
        topics.append(
            Topic(
                label=item["label"],
                subject_id=state["subject_id"],
                lecture_id=state["lecture_id"],
                description=item.get("description"),
                bloom_level=item.get("bloom_level"),
                topic_id=match_id if reused else str(uuid.uuid4()),
                label_embedding=vec,
                reused=reused,
            )
        )
    stats["extracted"] = len(topics)
    stats["reused"] = sum(1 for t in topics if t.reused)
    stats["new"] = len(topics) - stats["reused"]

    # Tag each chunk with the topics it is closest to (at least its best match).
    for chunk in chunks:
        if chunk.embedding is None:
            continue
        sims = [
            (_dot(chunk.embedding, vec), topic.topic_id)
            for topic, vec in zip(topics, label_vecs)
        ]
        chosen = [tid for score, tid in sims if score >= assign_threshold]
        if not chosen and sims:
            chosen = [max(sims, key=lambda s: s[0])[1]]
        chunk.topic_ids = list(dict.fromkeys(chosen))
        if chunk.topic_ids:
            stats["chunks_tagged"] += 1

    # Persist only genuinely new topics; reused ones keep their existing row.
    ctx.chunk_store.write_topics([t for t in topics if not t.reused])
    # Index every topic's label vector in Qdrant (reused too — idempotent on topic_id)
    # so runtime resolve_topics does an indexed top-k, not a Python brute-force.
    ctx.vector_store.ensure_topics_collection(ctx.config.get("embeddings.dimensions", 1024))
    ctx.vector_store.upsert_topics(topics)

    state["topics"] = [
        {
            "topic_id": t.topic_id,
            "label": t.label,
            "bloom_level": t.bloom_level,
            "reused": t.reused,
        }
        for t in topics
    ]
    state.setdefault("stats", {})["topics"] = stats
    return state


def _dot(a: list[float], b: list[float]) -> float:
    """Cosine similarity for L2-normalised vectors (the embedder normalises)."""
    return sum(x * y for x, y in zip(a, b))


def _match_one_to_one(
    new_vecs: list[list[float]],
    existing: list[tuple[str, str, list[float]]],
    threshold: float,
) -> list[str | None]:
    """Assign each new topic at most one existing ``topic_id`` to reuse, greedily
    by descending similarity so no two new topics claim the same old id and no old
    id is reused twice. Returns a per-new-topic list of ids (``None`` = mint new)."""
    result: list[str | None] = [None] * len(new_vecs)
    if not existing:
        return result
    pairs = [
        (_dot(vec, emb), i, tid)
        for i, vec in enumerate(new_vecs)
        for tid, _label, emb in existing
    ]
    pairs.sort(reverse=True)
    used_new: set[int] = set()
    used_old: set[str] = set()
    for sim, i, tid in pairs:
        if sim < threshold:
            break
        if i in used_new or tid in used_old:
            continue
        result[i] = tid
        used_new.add(i)
        used_old.add(tid)
    return result
