"""Shared RAG composition — pure functions over the read tools, no LLM, no state.

Layering: ``tools`` (atomic reads) -> ``retrieval`` (composition) ->
``subgraphs`` (orchestration). Every subgraph that needs grounding imports this
module rather than another subgraph, keeping subgraphs independent.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from shared.logging import clip

from .models import Citation, PrereqTopic, RetrievedChunk
from .tools import (
    get_lecture_topics,
    get_lesson_chunks,
    get_neighbours,
    get_subject_topics,
    get_topic_deps,
    rag_search,
)

logger = logging.getLogger(__name__)


def retrieve(
    query: str,
    namespaces: list[str],
    *,
    subject_id: str | None = None,
    topic_ids: list[str] | None = None,
    lecture_ids: list[str] | None = None,
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

    if topic_filter == "off":
        topic_ids = None
    kept = _search(topic_ids)
    # Soft topic scoping: if the topic-filtered pass grounded NOTHING, the topic guess was
    # likely wrong/partial and hard-filtered out the right chunks (invariant 8 in reverse).
    # Retry unscoped so the material isn't silently excluded — a decline should mean "not in
    # the syllabus", never "not under the guessed topic". Assessment keeps the hard default.
    if topic_filter == "soft" and topic_ids and not kept:
        kept = _search(None)
        logger.info("retrieve soft-fallback: topic filter dropped", extra={"kept": len(kept)})
    # "kept: 8" cannot explain an answer about the wrong lesson. The query and
    # what it actually matched, with scores, is the line you read when a student
    # says the tutor answered about history.
    logger.info("retrieve", extra={
        "query": clip(query),
        "kept": len(kept),
        "min_score": min_score,
        "subject_scoped": bool(subject_id),
        "matched": describe(kept),
    })
    return kept


def answers_within_scope(
    query: str,
    namespaces: list[str],
    scoped_best: float,
    *,
    subject_id: str | None = None,
    ratio: float,
) -> bool:
    """Whether the selected lesson is where this question is actually answered.

    Inside one lesson a similarity floor cannot tell a question the lesson
    covers from one it does not: with the rest of the corpus filtered out there
    is nothing left to lose to, so the lesson's own best chunk clears the floor
    whatever was asked. Both "who was Jabir ibn Hayyan" and "explain the Youth
    poem" score around 0.50 against the Jabir lesson.

    What separates them is the rest of the corpus. A question the lesson owns
    scores the same with or without the filter; a question belonging to another
    lesson scores markedly worse once that lesson is excluded.
    """
    if scoped_best <= 0:
        return False
    unscoped = rag_search(query or "", namespaces, k=1, subject_id=subject_id)
    if not unscoped:
        return True
    return scoped_best / unscoped[0].score >= ratio


def topics_on_offer(*, lecture_id: str | None, subject_id: str | None,
                    limit: int) -> list[str]:
    """What the student's lessons do cover, for a tutor that has to say they do not
    cover what was asked: the selected lesson's topics, else the subject's."""
    try:
        if lecture_id:
            return [t.label for t in get_lecture_topics([lecture_id], limit=limit) if t.label]
        if subject_id:
            return get_subject_topics(subject_id, limit=limit)
    except Exception:
        logger.warning("could not read what the lessons cover; the refusal will name none",
                       extra={"lecture_id": lecture_id, "subject_id": subject_id},
                       exc_info=True)
    return []
def whole_lesson(lecture_id: str, namespaces: list[str], *, limit: int) -> list[RetrievedChunk]:
    """The selected lesson's own passages, for a request about the lesson as a whole."""
    chunks = get_lesson_chunks(lecture_id, namespaces, limit=limit)
    logger.info("read the whole lesson", extra={"lecture_id": lecture_id, "kept": len(chunks)})
    return chunks


def with_neighbours(
    chunks: list[RetrievedChunk],
    *,
    anchors: int,
    window: int,
    limit: int,
) -> list[RetrievedChunk]:
    """Search small, read big: the best matches, each with the rest of its section.

    A layout chunker cuts a section at every sub-heading and every page, so
    "what causes ocean waves?" found the chunk about wind and never the ones
    about landslides and meteorites, three lines further down the same page.
    Their words match the question poorly; their place next to the chunk that
    matched it well is what says they belong to the answer. The ``anchors``
    closest matches by meaning are read in their lesson's order with ``window``
    chunks either side, then the remaining matches, up to ``limit``. Closeness
    is the dense score, not the fused rank: a question's function words lift
    whole lessons written in its language up the fused order.
    """
    if not chunks or anchors < 1 or window < 1:
        return chunks
    seeds = sorted(chunks, key=lambda c: c.score, reverse=True)[:anchors]
    around = get_neighbours(seeds, window=window)
    scored = {c.chunk_id: c for c in chunks}
    out: list[RetrievedChunk] = []
    seen: set[str] = set()
    for seed in seeds:
        run = [c for c in around
               if c.lecture_id == seed.lecture_id and seed.order_index is not None
               and abs(c.order_index - seed.order_index) <= window]
        for chunk in run or [seed]:
            if chunk.chunk_id not in seen:
                seen.add(chunk.chunk_id)
                out.append(scored.get(chunk.chunk_id, chunk))
    for chunk in chunks:
        if chunk.chunk_id not in seen:
            seen.add(chunk.chunk_id)
            out.append(chunk)
    logger.info("retrieve with neighbours", extra={
        "matched": len(chunks), "added": len(out) - len(chunks), "kept": min(len(out), limit)})
    return out[:limit]


def describe(chunks: Sequence[RetrievedChunk], limit: int = 5) -> list[str]:
    """``"Subtraction Part 2 0.715"`` per chunk — what matched, and how well."""
    return [
        f"{c.lesson_title or c.lecture_id or '?'} {c.score:.3f}"
        for c in chunks[:limit]
    ]


# The shipped floor, repeated here so a subgraph reached without one does not
# silently fall back to "keep everything". config.yaml agent.min_score is the
# source; this is the safety net when a payload omits it.
DEFAULT_MIN_SCORE = 0.44

# The shipped ratio, repeated here so a subgraph reached without one does not
# silently fall back to answering everything. config.yaml is the source.
DEFAULT_SCOPE_RATIO = 0.90

DEFAULT_MAX_PREREQS = 5

# Nearest first, and a required prerequisite before a merely recommended one: the
# topic you must know to follow this lesson outranks one that would help.
# Ingestion writes exactly these two (ingestion/schemas.py DEPENDENCY_STRENGTHS),
# folding "recommended", "optional" and "helpful" into "related" on the way in.
# This ranked "recommended", which ingestion never writes, so every non-required
# edge fell to the unknown sentinel and the ordering worked by accident.
_STRENGTH_RANK = {"required": 0, "related": 1}


def resolve_prereqs(
    topic_ids: list[str],
    *,
    depth: int = 2,
    limit: int = DEFAULT_MAX_PREREQS,
) -> list[PrereqTopic]:
    """Prerequisite topics for ``topic_ids``, nearest and most required first.

    The walk fans out: two hops over a well-connected subject reaches dozens of
    topics. ``limit`` keeps the nearest few. Only the topics are returned; their
    passages were once fetched by position in the lesson and handed to the model
    as citable context, which is how a book's title page came to be cited.
    """
    if not topic_ids:
        return []
    found = get_topic_deps(topic_ids, depth=depth)
    prereqs = sorted(
        found, key=lambda p: (p.depth, _STRENGTH_RANK.get(p.strength, 9), p.label)
    )[:limit] if limit else found
    logger.info("resolve_prereqs", extra={"found": len(found), "prereqs": len(prereqs)})
    return prereqs


def cite(chunk: RetrievedChunk, n: int) -> Citation:
    if chunk.background:
        return Citation(n=n, subject_name=chunk.subject_name, chunk_id=chunk.chunk_id)
    return Citation(
        n=n,
        lecture_id=chunk.lecture_id,
        heading_path=chunk.heading_path,
        page_from=chunk.page_from,
        page_to=chunk.page_to,
        subject_name=chunk.subject_name,
        lesson_title=chunk.lesson_title,
        lesson_order=chunk.lesson_order,
        chunk_id=chunk.chunk_id,
    )


def render_context(chunks: list[RetrievedChunk], start: int = 1) -> tuple[str, list[Citation]]:
    """Number chunks [n] and return the block text plus citation records."""
    lines, citations = [], []
    for i, c in enumerate(chunks, start=start):
        head = c.heading_path or c.lecture_id
        lines.append(f"[{i}] ({head}) {c.text}")
        citations.append(cite(c, i))
    return "\n\n".join(lines), citations
