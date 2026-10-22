"""topics stage — extract a lesson's skills, reuse ids on re-ingest, tag chunks.

Runs after ``embed`` (reuses the embedder for labels and the chunk vectors for
tagging): an LLM names 3-8 skills; labels that name the same skill twice are
merged; each is matched one-to-one against this lecture's own existing topics,
then against the subject's syllabus catalog, by label cosine (a match reuses
that topic_id so mastery history stays attached — invariant 11, and a skill
taught in two lessons stays one topic); each chunk is tagged with its closest
topics. Writes ``ai.topics``. A lesson with text that ends with no topics is
flagged ``needs_review``; a stub extractor outside a dev environment fails the
run.
"""

from __future__ import annotations

import logging
import uuid

from shared.config import DEV_ENVIRONMENTS
from shared.namespaces import SYLLABUS
from shared.structured import StructuredOutputError

from ..context import PipelineContext
from ..state import Chunk, IngestionState, Topic

logger = logging.getLogger(__name__)


def _lesson_language(chunks: list[Chunk]) -> str:
    """The language most of the lesson is written in — what its topics are named in."""
    counts: dict[str, int] = {}
    for chunk in chunks:
        if chunk.language:
            counts[chunk.language] = counts.get(chunk.language, 0) + 1
    return max(counts, key=counts.get) if counts else "en"


def run(state: IngestionState, ctx: PipelineContext) -> IngestionState:
    chunks: list[Chunk] = state.get("chunks", [])
    stats = {"extracted": 0, "reused": 0, "new": 0, "chunks_tagged": 0}
    if not chunks:
        state["topics"] = []
        state.setdefault("stats", {})["topics"] = stats
        return state

    if ctx.topic_extractor.is_stub and ctx.config.app_env not in DEV_ENVIRONMENTS:
        raise RuntimeError(
            f"No topic model configured for APP_ENV={ctx.config.app_env!r}: "
            f"lesson {state['lecture_id']} would be indexed without topics.")

    try:
        raw = ctx.topic_extractor.extract(
            [c.text for c in chunks], _lesson_language(chunks))
    except StructuredOutputError as exc:
        # Compliance is retried one layer down; reaching here means the model
        # never produced a usable answer. Index the lesson without topic_ids and
        # flag it, rather than losing the extraction and the OCR behind it.
        logger.error("topic extraction failed", extra=exc.log_fields())
        _flag(state, f"topic extraction failed: {exc}")
        raw = []
    if not raw and ctx.topic_extractor.is_stub:
        _flag(state, "no topic model is configured; chunks indexed without topic_ids.")
    elif not raw:
        _flag(state, "topic extraction returned no topics for a non-empty lesson; "
                     "chunks indexed without topic_ids.")
    if not raw:
        state["topics"] = []
        state.setdefault("stats", {})["topics"] = stats
        return state

    cfg = ctx.config.get("ingestion.topics", {})
    reuse_threshold = cfg.get("reuse_threshold", 0.60)
    merge_threshold = cfg["merge_threshold"]
    assign_threshold = cfg.get("assign_threshold", 0.45)

    raw, label_vecs = _merge_near_duplicates(
        raw, ctx.embedder.embed([t["label"] for t in raw]), merge_threshold)
    own = ctx.chunk_store.read_lecture_topics(state["lecture_id"])
    reuse_ids = _match_one_to_one(label_vecs, own, reuse_threshold)
    if (state.get("namespace") or SYLLABUS) == SYLLABUS:
        claimed = {tid for tid in reuse_ids if tid}
        open_slots = [i for i, tid in enumerate(reuse_ids) if tid is None]
        catalog = [entry for entry in ctx.chunk_store.read_topic_catalog(
            state["subject_id"], include_background=False) if entry[0] not in claimed]
        for i, tid in zip(open_slots, _match_one_to_one(
                [label_vecs[i] for i in open_slots], catalog, merge_threshold)):
            if tid is not None:
                reuse_ids[i] = tid
    owners = ctx.chunk_store.read_topic_owners([tid for tid in reuse_ids if tid])

    topics: list[Topic] = []
    for item, vec, match_id in zip(raw, label_vecs, reuse_ids):
        reused = match_id is not None
        topics.append(
            Topic(
                label=item["label"],
                subject_id=state["subject_id"],
                lecture_id=owners.get(match_id) or state["lecture_id"],
                description=item.get("description"),
                bloom_level=item.get("bloom_level"),
                topic_id=match_id if reused else str(uuid.uuid4()),
                label_embedding=vec,
                reused=reused,
                namespace=state.get("namespace") or SYLLABUS,
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
    if not stats["chunks_tagged"]:
        _flag(state, "no chunk of the lesson was tagged with a topic.")

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


def _flag(state: IngestionState, reason: str) -> None:
    state["ingestion_status"] = "needs_review"
    state.setdefault("errors", []).append(reason)


def _merge_near_duplicates(
    items: list[dict], vecs: list[list[float]], threshold: float,
) -> tuple[list[dict], list[list[float]]]:
    """One topic per skill: a label within ``threshold`` of one already kept names
    the same skill, and is folded into it rather than minted as a second topic."""
    kept: list[dict] = []
    kept_vecs: list[list[float]] = []
    for item, vec in zip(items, vecs):
        if any(_dot(vec, other) >= threshold for other in kept_vecs):
            continue
        kept.append(item)
        kept_vecs.append(vec)
    return kept, kept_vecs


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
