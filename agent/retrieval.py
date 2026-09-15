"""Shared RAG composition — pure functions over the read tools, no LLM, no state.

Layering: ``tools`` (atomic reads) -> ``retrieval`` (composition) ->
``subgraphs`` (orchestration). Every subgraph that needs grounding imports this
module rather than another subgraph, keeping subgraphs independent.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from .models import Citation, PrereqTopic, RetrievedChunk
from .tools import get_topic_chunks, get_topic_deps, rag_search

logger = logging.getLogger(__name__)


def retrieve(
    query: str,
    namespaces: list[str],
    *,
    subject_id: Optional[str] = None,
    topic_ids: Optional[list[str]] = None,
    lecture_ids: Optional[list[str]] = None,
    k: int = 8,
    min_score: float = 0.0,
    topic_filter: str = "hard",
) -> list[RetrievedChunk]:
    # No language filter: retrieval is cross-lingual over one multilingual index, so
    # a French/Arabic query must still reach English (or any-language) chunks.
    def _search(topics):
        chunks = rag_search(
            query or "", namespaces, k=k, subject_id=subject_id,
            topic_ids=topics or None,
            lecture_ids=lecture_ids or None,   # optional lecture scoping (None -> not scoped)
        )
        # Similarity floor: drop weakly-matched chunks so an off-syllabus query grounds in
        # nothing and the answer declines, rather than citing unrelated material.
        return [c for c in chunks if c.score >= min_score] if min_score else chunks

    kept = _search(topic_ids)
    # Soft topic scoping: if the topic-filtered pass grounded NOTHING, the topic guess was
    # likely wrong/partial and hard-filtered out the right chunks (invariant 8 in reverse).
    # Retry unscoped so the material isn't silently excluded — a decline should mean "not in
    # the syllabus", never "not under the guessed topic". Assessment keeps the hard default.
    if topic_filter == "soft" and topic_ids and not kept:
        kept = _search(None)
        logger.info("retrieve soft-fallback: topic filter dropped", extra={"kept": len(kept)})
    logger.info("retrieve", extra={"kept": len(kept), "min_score": min_score})
    return kept


DEFAULT_MAX_PREREQS = 5

# Nearest first, and a required prerequisite before a merely recommended one: the
# topic you must know to follow this lesson outranks one that would help.
_STRENGTH_RANK = {"required": 0, "recommended": 1}


def resolve_prereqs(
    topic_ids: list[str],
    namespaces: list[str],
    *,
    depth: int = 2,
    limit: int = DEFAULT_MAX_PREREQS,
) -> tuple[list[PrereqTopic], list[RetrievedChunk]]:
    """Prerequisite topics for ``topic_ids`` and the chunks that teach them.

    The walk fans out: two hops over a well-connected subject reaches dozens of
    topics, each contributing chunks that are fetched by id and so carry no
    relevance score to filter on. Unbounded, prerequisite material outweighed the
    lesson the student actually asked about. ``limit`` keeps the nearest few.
    """
    if not topic_ids:
        return [], []
    found = get_topic_deps(topic_ids, depth=depth)
    prereqs = sorted(
        found, key=lambda p: (p.depth, _STRENGTH_RANK.get(p.strength, 9), p.label)
    )[:limit] if limit else found
    prereq_ids = [p.topic_id for p in prereqs]
    prereq_chunks = get_topic_chunks(prereq_ids, namespaces) if prereq_ids else []
    logger.info("resolve_prereqs", extra={
        "found": len(found), "prereqs": len(prereqs),
        "prereq_chunks": len(prereq_chunks)})
    return prereqs, prereq_chunks


def cite(chunk: RetrievedChunk, n: int) -> Citation:
    return Citation(
        n=n,
        lecture_id=chunk.lecture_id,
        heading_path=chunk.heading_path,
        page_from=chunk.page_from,
        page_to=chunk.page_to,
        subject_name=chunk.subject_name,
        lesson_title=chunk.lesson_title,
        lesson_order=chunk.lesson_order,
    )


def render_context(chunks: list[RetrievedChunk], start: int = 1) -> tuple[str, list[Citation]]:
    """Number chunks [n] and return the block text plus citation records."""
    lines, citations = [], []
    for i, c in enumerate(chunks, start=start):
        head = c.heading_path or c.lecture_id
        lines.append(f"[{i}] ({head}) {c.text}")
        citations.append(cite(c, i))
    return "\n\n".join(lines), citations
