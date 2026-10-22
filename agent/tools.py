"""Agent tools — read-only, deterministic access to the index and catalog, plus
the agent-owned ai.* writes (mastery, mistake tags).

Layering: ``tools`` (atomic reads/writes) -> ``retrieval`` (composition) ->
``subgraphs`` (orchestration). Tools never call agents (strict layering); psycopg
and qdrant are imported lazily so importing a tool never requires them.

The data shapes these return (``RetrievedChunk``, ``ResolvedTopic``, ``PrereqTopic``)
live in ``agent/models.py``.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from typing import Any

from shared.backend import get_backend_client
from shared.config import Config
from shared.lexical import bm25_scores, ranks_from_scores, reciprocal_rank_fusion
from shared.postgres import connect
from shared.vector_store import query_client
from shared.embeddings import get_embedder

logger = logging.getLogger(__name__)

from .mastery import Mastery, MasteryParams, status
from shared.matching import NOT_CONTENT, chunk_kind
from shared.namespaces import BACKGROUND, NOT_BACKGROUND_TOPIC, SYLLABUS
from shared.vectors import DENSE_VECTOR, SPARSE_VECTOR, has_named_vectors
from .models import PrereqTopic, ResolvedTopic, RetrievedChunk

__all__ = [
    "AuthorizationError",
    "assert_user_type",
    "create_mistake_tag",
    "fetch_rubric",
    "get_figures",
    "get_item",
    "get_mastery",
    "get_mastery_rollup",
    "get_neighbours",
    "get_question_bank",
    "get_topic_chunks",
    "get_topic_deps",
    "list_mistake_tags",
    "rag_search",
    "resolve_topics",
    "update_mastery",
    "write_mastery_events",
]


# --- authorization ----------------------------------------------------------

class AuthorizationError(PermissionError):
    """Raised when a tool runs under a user_type it does not permit."""


def assert_user_type(ctx: Mapping[str, Any] | None, *allowed: str) -> None:
    """Assert ``ctx['user_type']`` is one of ``allowed``.

    ``ctx`` is the resolved context bundle from ``load_context``. Raises
    :class:`AuthorizationError` when it is missing or not permitted.

    TODO: not yet called anywhere — wire into the permission-sensitive tools when
    teacher/classroom scoping lands (backend still checks ownership independently)."""
    user_type = (ctx or {}).get("user_type")
    if user_type not in allowed:
        raise AuthorizationError(
            f"user_type {user_type!r} not permitted (allowed: {', '.join(allowed)})"
        )


# --- rag_search: namespaced semantic retrieval ------------------------------

def rag_search(
    query: str,
    namespaces: Sequence[str],
    *,
    k: int = 8,
    subject_id: str | None = None,
    grade: str | None = None,
    language: str | None = None,
    topic_ids: Sequence[str] | None = None,
    lecture_ids: Sequence[str] | None = None,
    ready_only: bool = True,
    config: Config | None = None,
) -> list[RetrievedChunk]:
    """Return the top-``k`` chunks most similar to ``query`` within ``namespaces``.

    Embeds the query with the same BGE-M3 embedder ingestion used, runs a
    namespace-filtered vector search in Qdrant (invariant 1), then hydrates chunk
    text from ``ai.chunks`` excluding non-``ready`` lectures (invariant 9).
    ``subject_id``/``grade``/``language``/``topic_ids``/``lecture_ids`` narrow the search
    further — all OPTIONAL payload filters on top of the namespace isolation; ``None`` means
    that dimension is not scoped.
    """
    if not namespaces:
        raise ValueError("rag_search requires at least one namespace (invariant 1).")
    config = config or Config.load()

    from qdrant_client import models

    vector = get_embedder(config).embed_query(query)

    own = [models.FieldCondition(
        key="namespace", match=models.MatchAny(any=list(namespaces)))]
    must = []
    if subject_id:
        must.append(models.FieldCondition(
            key="subject_id", match=models.MatchValue(value=subject_id)))
    if grade:
        must.append(models.FieldCondition(
            key="grade", match=models.MatchValue(value=grade)))
    if language:
        must.append(models.FieldCondition(
            key="language", match=models.MatchValue(value=language)))
    if topic_ids:
        must.append(models.FieldCondition(
            key="topic_ids", match=models.MatchAny(any=list(topic_ids))))
    # Optional lecture scoping: applied only when lecture_ids are supplied; None -> whole subject.
    # Fed from the request `lecture_ids` field, threaded state -> subgraph -> here.
    if lecture_ids:
        own.append(models.FieldCondition(
            key="lecture_id", match=models.MatchAny(any=list(lecture_ids))))
    # TODO (pending decision: school/content scoping) filter on `source_type` / `school_id` /
    # `classroom_id` payload fields so retrieval is limited to the
    # student's general + school + joined-classroom content once the model is confirmed.

    # A contents page lists every lesson title, so it beats the lessons
    # themselves whenever a student names one. It can never be an answer.
    must_not = [models.FieldCondition(
        key="chunk_kind", match=models.MatchAny(any=list(NOT_CONTENT)))]

    collection = config.vector_store.get("collection", "eduai_chunks")
    # Over-fetch for two reasons: ready_only may drop hits, and the lexical
    # stage can only reorder what the dense stage brought back.
    fetch = k * 3 if ready_only else k
    if config.get("agent.hybrid.enabled"):
        fetch = max(fetch, config.get("agent.hybrid.candidates"))
    background = _background_scope(namespaces, subject_id, lecture_ids, config)
    if background is None:
        scope = models.Filter(must=must + own, must_not=must_not)
    else:
        scope = models.Filter(must=must, must_not=must_not, should=[
            models.Filter(must=own), models.Filter(must=background)])
    scores, meta = _candidate_scores(
        query, [scope], config,
        collection=collection, limit=fetch, vector=vector, with_meta=True)
    if not scores:
        return []

    ready_clause = ""
    if ready_only:
        ready_clause = (" AND c.lecture_id IN (SELECT lecture_id FROM "
                        "ai.lecture_ingestion WHERE status = 'ready')")
    sql = (
        "SELECT c.chunk_id::text, c.text, c.heading_path, c.page_from, c.page_to, "
        "c.lecture_id::text, c.language, c.topic_ids, c.order_index "
        "FROM ai.chunks c WHERE c.chunk_id::text = ANY(%s)" + ready_clause
    )

    with connect(config) as conn, conn.cursor() as cur:
        cur.execute(sql, (list(scores),))
        rows = cur.fetchall()

    out = [
        RetrievedChunk(
            chunk_id=r[0], score=scores.get(r[0], 0.0), text=r[1],
            heading_path=r[2], page_from=r[3], page_to=r[4],
            lecture_id=r[5], language=r[6],
            topic_ids=[str(t) for t in (r[7] or [])],
            subject_name=meta.get(r[0], {}).get("subject_name"),
            lesson_title=meta.get(r[0], {}).get("lesson_title"),
            lesson_order=meta.get(r[0], {}).get("lesson_order"),
            subject_id=meta.get(r[0], {}).get("subject_id"),
            order_index=r[8],
            background=meta.get(r[0], {}).get("namespace") == BACKGROUND,
        )
        for r in rows
    ]
    out.sort(key=lambda c: c.score, reverse=True)
    if config.get("agent.hybrid.enabled"):
        out = _fuse_lexically(query, out, config)
    return out[:k]



def _background_scope(namespaces: Sequence[str], subject_id: str | None,
                      lecture_ids: Sequence[str] | None, config: Config) -> list | None:
    """Where the background a syllabus search also reads lives: the external
    books of the same subject. A lesson's own id never matches them, since they
    are filed under lessons of their own, so a lesson-scoped search reaches them
    through its subject."""
    from qdrant_client import models

    if SYLLABUS not in namespaces or BACKGROUND in namespaces:
        return None
    subjects = [subject_id] if subject_id else (
        _subjects_of(lecture_ids, config) if lecture_ids else None)
    if lecture_ids and not subjects:
        return None
    scope = [models.FieldCondition(key="namespace", match=models.MatchValue(value=BACKGROUND))]
    if subjects:
        scope.append(models.FieldCondition(key="subject_id", match=models.MatchAny(any=subjects)))
    return scope


def _subjects_of(lecture_ids: Sequence[str], config: Config) -> list[str]:
    with connect(config) as conn, conn.cursor() as cur:
        cur.execute("SELECT DISTINCT subject_id::text FROM content.lessons "
                    "WHERE id::text = ANY(%s)", (list(lecture_ids),))
        return [row[0] for row in cur.fetchall()]


def _candidate_scores(
    query: str,
    filters: list,
    config: Config,
    *,
    collection: str,
    limit: int,
    vector: list[float] | None = None,
    with_meta: bool = False,
):
    """Candidate chunk ids with their DENSE cosine, however they were found.

    A collection holding both vectors is searched twice: once by meaning, once
    by the words themselves. The second is what earns the round trip — a chunk
    the dense vector averaged away was never a candidate before, so no amount of
    re-ranking could surface it. A student typing a name, a formula or a date is
    exactly that case.

    What comes back is always a similarity. ``agent.min_score`` is measured
    against one, and a fused rank score in its place would read as "everything
    clears the floor" — the tutor would stop declining what the syllabus cannot
    answer. So a chunk only the sparse half found is scored by asking the dense
    half about it, not by its sparse score.
    """
    client = query_client(config)
    query_filter = filters[0] if filters else None
    vector = vector if vector is not None else get_embedder(config).embed_query(query)
    hybrid = has_named_vectors(client, collection)

    # A collection with named vectors has to be told which one to search; the
    # collection we have today has one unnamed vector, and naming it finds
    # nothing. Same call, one argument apart.
    named = {"using": DENSE_VECTOR} if hybrid else {}
    dense = client.query_points(
        collection, query=vector, query_filter=query_filter,
        limit=limit, with_payload=True, **named,
    ).points
    scores = {str(h.id): h.score for h in dense}
    meta = {str(h.id): (h.payload or {}) for h in dense}

    if hybrid:
        terms = get_embedder(config).embed_sparse([query])[0]
        if terms:
            scores, meta = _add_lexical_candidates(
                client, collection, terms, vector, query_filter, limit,
                scores, meta, config)

    return (scores, meta) if with_meta else scores


def _add_lexical_candidates(
    client, collection, terms, vector, query_filter, limit, scores, meta, config,
):
    """Chunks the words found, scored by meaning so the floor still applies."""
    from qdrant_client import models

    lexical = client.query_points(
        collection,
        query=models.SparseVector(indices=list(terms), values=list(terms.values())),
        using=SPARSE_VECTOR, query_filter=query_filter,
        limit=limit, with_payload=True,
    ).points
    unseen = [str(h.id) for h in lexical if str(h.id) not in scores]
    if not unseen:
        return scores, meta

    rescored = client.query_points(
        collection, query=vector, using=DENSE_VECTOR,
        query_filter=models.Filter(must=[models.HasIdCondition(has_id=unseen)]),
        limit=len(unseen), with_payload=True,
    ).points
    logger.info("lexical candidates", extra={
        "found_only_by_words": len(unseen), "rescored": len(rescored)})
    for hit in rescored:
        scores[str(hit.id)] = hit.score
        meta[str(hit.id)] = hit.payload or {}
    return scores, meta


def _fuse_lexically(
    query: str, candidates: list[RetrievedChunk], config: Config,
) -> list[RetrievedChunk]:
    """Reorder dense candidates by fusing their rank with a BM25 rank.

    ``score`` is deliberately left as the dense cosine. It is what
    ``agent.min_score`` is measured against, and a fused rank-based score is not
    comparable to a similarity — swapping it in would silently disable the floor
    that stops the tutor answering questions the syllabus cannot answer.
    """
    if len(candidates) < 2:
        return candidates
    dense_ranks = list(range(1, len(candidates) + 1))   # already sorted by score
    lexical = bm25_scores(query, [c.text or "" for c in candidates])
    # A document that matches none of the query's words should not be pushed up
    # merely for being in the pool, so it places in the dense ranking only.
    lexical_ranks = ranks_from_scores(lexical, skip_zero=True)
    fused = reciprocal_rank_fusion(
        dense_ranks, lexical_ranks, k=config.get("agent.hybrid.rrf_k"))
    order = sorted(range(len(candidates)), key=lambda i: -fused[i])
    return [candidates[i] for i in order]


# --- resolve_topics: free-text -> topic_ids ---------------------------------

def _dot(a, b) -> float:
    return sum(x * y for x, y in zip(a, b))


def resolve_topics(
    query: str,
    subject_id: str | None = None,
    *,
    lecture_ids: Sequence[str] | None = None,
    k: int = 5,
    threshold: float = 0.40,
    config: Config | None = None,
) -> list[ResolvedTopic]:
    """Top-``k`` topics whose label is closest to ``query`` (cosine >= ``threshold``).

    Scopes to ``subject_id`` and, when given, ``lecture_ids`` (invariant 8). Returns
    ``[]`` when nothing clears the threshold, so the planner attaches no topic_ids
    rather than guessing. Backed by the Qdrant topics collection (indexed top-k) when
    ``agent.resolve_topics_qdrant`` is on; otherwise a Postgres brute-force over
    ``ai.topics.label_embedding`` (the default until the topics collection is populated
    at re-ingest)."""
    config = config or Config.load()
    vector = get_embedder(config).embed_query(query)
    # TODO: once the topics collection is populated and validated, delete the brute-force
    # `_resolve_topics_pg` and this gate so the indexed path is the only one.
    if config.get("agent.resolve_topics_qdrant", True):
        return _resolve_topics_qdrant(vector, subject_id, lecture_ids, k, threshold, config)
    return _resolve_topics_pg(vector, subject_id, lecture_ids, k, threshold, config)


def _resolve_topics_qdrant(vector, subject_id, lecture_ids, k, threshold, config):
    """Indexed top-k over the Qdrant topics collection (label vectors + subject/lecture
    payload). Scalable — no full-catalog scan."""
    from qdrant_client import models

    must = []
    if subject_id:
        must.append(models.FieldCondition(
            key="subject_id", match=models.MatchValue(value=subject_id)))
    if lecture_ids:
        must.append(models.FieldCondition(
            key="lecture_id", match=models.MatchAny(any=list(lecture_ids))))
    not_background = [models.FieldCondition(
        key="namespace", match=models.MatchValue(value=BACKGROUND))]
    client = query_client(config)
    hits = client.query_points(
        config.vector_store.get("topics_collection", "eduai_topics"),
        query=vector, query_filter=models.Filter(must=must, must_not=not_background),
        limit=k, score_threshold=threshold, with_payload=True,
    ).points
    return [
        ResolvedTopic(
            topic_id=(h.payload or {}).get("topic_id") or str(h.id),
            label=(h.payload or {}).get("label", ""),
            score=h.score,
            subject_id=(h.payload or {}).get("subject_id", ""),
        )
        for h in hits
    ]


def _resolve_topics_pg(vector, subject_id, lecture_ids, k, threshold, config):
    """Brute-force cosine over ``ai.topics`` in Python. Correct but O(N) — the Qdrant
    path replaces it once the topics collection is populated (re-ingest)."""
    sql = ("SELECT topic_id::text, label, subject_id::text, label_embedding FROM ai.topics "
           "WHERE label_embedding IS NOT NULL AND " + NOT_BACKGROUND_TOPIC)
    params: list = []
    if subject_id:
        sql += " AND subject_id = %s"
        params.append(subject_id)
    if lecture_ids:
        sql += " AND lecture_id::text = ANY(%s)"
        params.append([str(x) for x in lecture_ids])

    with connect(config) as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()

    scored, mismatched = [], 0
    for tid, label, sid, emb in rows:
        if not emb:
            continue
        if len(emb) != len(vector):
            mismatched += 1          # stale embedding (different model) — don't score it wrong
            continue
        scored.append(
            ResolvedTopic(topic_id=tid, label=label, score=_dot(vector, emb), subject_id=sid))
    if mismatched:
        logger.warning("resolve_topics skipped %d topics with a mismatched-dimension "
                       "embedding (re-embed the topic catalog)", mismatched)
    scored = [t for t in scored if t.score >= threshold]
    scored.sort(key=lambda t: t.score, reverse=True)
    return scored[:k]


# --- get_topic_deps: a topic's prerequisites --------------------------------

def get_topic_deps(
    topic_ids: Sequence[str],
    *,
    depth: int = 2,
    config: Config | None = None,
) -> list[PrereqTopic]:
    """Prerequisite topics reachable from ``topic_ids`` within ``depth`` hops.

    Walks ``ai.topic_dependencies`` breadth-first, so ``depth`` is the shortest-path
    distance; a topic already seen (including the seeds) is never revisited, which
    also breaks cycles. Labels are hydrated from ``ai.topics``.
    """
    if not topic_ids or depth < 1:
        return []
    config = config or Config.load()

    found: dict[str, PrereqTopic] = {}
    with connect(config) as conn, conn.cursor() as cur:
        seen: set[str] = {str(t) for t in topic_ids}
        frontier: list[str] = list(seen)
        for level in range(1, depth + 1):
            if not frontier:
                break
            cur.execute(
                "SELECT d.prereq_topic_id::text, d.strength::text, t.label "
                "FROM ai.topic_dependencies d "
                "LEFT JOIN ai.topics t ON t.topic_id = d.prereq_topic_id "
                "WHERE d.topic_id::text = ANY(%s)",
                (frontier,),
            )
            next_frontier: list[str] = []
            for prereq_id, strength, label in cur.fetchall():
                if prereq_id in seen:
                    continue
                seen.add(prereq_id)
                next_frontier.append(prereq_id)
                found[prereq_id] = PrereqTopic(
                    topic_id=prereq_id, label=label or "", strength=strength,
                    depth=level,
                )
            frontier = next_frontier

    return sorted(found.values(), key=lambda p: (p.depth, p.label))


def get_topic_labels(
    topic_ids: Sequence[str],
    *,
    config: Config | None = None,
) -> dict[str, str]:
    """``{topic_id: label}`` for the given ids, from ``ai.topics``. Missing ids are
    absent from the result. For hydrating labels onto id-only rows (e.g. mastery)."""
    if not topic_ids:
        return {}
    config = config or Config.load()
    with connect(config) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT topic_id::text, label FROM ai.topics WHERE topic_id::text = ANY(%s)",
                ([str(t) for t in topic_ids],))
            return {r[0]: r[1] or "" for r in cur.fetchall()}


def get_chunks_by_id(
    chunk_ids: Sequence[str],
    *,
    ready_only: bool = True,
    config: Config | None = None,
) -> list[RetrievedChunk]:
    """The named chunks, in the order they were named.

    A lookup, not a search: ``score`` is 0.0 because nothing was ranked. It is
    how a turn that continues the one before it reaches the passages that
    answer already used — by pointer, rather than by searching for words that
    name nothing.

    Order is preserved because it is the order the previous answer read them
    in, and a re-presentation that reshuffles its evidence reads as a different
    answer to the same question.
    """
    if not chunk_ids:
        return []
    config = config or Config.load()
    wanted = [str(c) for c in chunk_ids]

    ready_clause = ""
    if ready_only:
        ready_clause = (" AND c.lecture_id IN (SELECT lecture_id FROM "
                        "ai.lecture_ingestion WHERE status = 'ready')")
    sql = (
        "SELECT c.chunk_id::text, c.text, c.heading_path, c.page_from, c.page_to, "
        "c.lecture_id::text, c.language, c.topic_ids, c.order_index "
        "FROM ai.chunks c WHERE c.chunk_id::text = ANY(%s)" + ready_clause
    )
    with connect(config) as conn, conn.cursor() as cur:
        cur.execute(sql, (wanted,))
        rows = {r[0]: r for r in cur.fetchall()}

    found = [rows[cid] for cid in wanted if cid in rows]
    chunks = [
        RetrievedChunk(
            chunk_id=r[0], score=0.0, text=r[1], heading_path=r[2],
            page_from=r[3], page_to=r[4], lecture_id=r[5], language=r[6],
            topic_ids=[str(t) for t in (r[7] or [])], order_index=r[8],
        )
        for r in found
    ]
    labelled = _apply_labels(
        chunks, _citation_labels([c.chunk_id for c in chunks], config))
    if len(labelled) < len(wanted):
        # A lecture re-ingested since the last turn re-mints every chunk_id, so
        # the evidence points at rows that no longer exist. Saying so beats a
        # re-presentation that quietly drops half of what it is re-presenting.
        logger.info("evidence partly gone", extra={
            "asked": len(wanted), "found": len(labelled)})
    return labelled


def get_neighbours(
    anchors: Sequence[RetrievedChunk],
    *,
    window: int,
    ready_only: bool = True,
    config: Config | None = None,
) -> list[RetrievedChunk]:
    """The chunks written next to ``anchors`` in their lesson, in reading order.

    A section a student asks about is often split across several chunks — one
    per sub-heading, one per page — and only the chunk that best matches the
    question's words is found. Its neighbours hold the rest of the section. A
    lookup, not a search: ``score`` is 0.0. Navigation and apparatus are left
    out, as they are from a search.
    """
    wanted = [(c.lecture_id, c.order_index) for c in anchors
              if c.lecture_id and c.order_index is not None]
    if not wanted or window < 1:
        return []
    config = config or Config.load()
    ready_clause = ""
    if ready_only:
        ready_clause = (" AND c.lecture_id IN (SELECT lecture_id FROM "
                        "ai.lecture_ingestion WHERE status = 'ready')")
    ranges = " OR ".join(
        "(c.lecture_id::text = %s AND c.order_index BETWEEN %s AND %s)" for _ in wanted)
    params: list[Any] = []
    for lecture_id, order in wanted:
        params += [str(lecture_id), order - window, order + window]
    sql = (
        "SELECT c.chunk_id::text, c.text, c.heading_path, c.page_from, c.page_to, "
        "c.lecture_id::text, c.language, c.topic_ids, c.order_index "
        f"FROM ai.chunks c WHERE ({ranges})" + ready_clause
        + " ORDER BY c.lecture_id, c.order_index"
    )
    with connect(config) as conn, conn.cursor() as cur:
        cur.execute(sql, tuple(params))
        rows = cur.fetchall()
    chunks = [
        RetrievedChunk(
            chunk_id=r[0], score=0.0, text=r[1], heading_path=r[2],
            page_from=r[3], page_to=r[4], lecture_id=r[5], language=r[6],
            topic_ids=[str(t) for t in (r[7] or [])], order_index=r[8],
        )
        for r in rows if chunk_kind(r[1] or "") not in NOT_CONTENT
    ]
    return _apply_labels(chunks, _citation_labels([c.chunk_id for c in chunks], config))


def get_lesson_chunks(
    lecture_id: str,
    namespaces: Sequence[str],
    *,
    limit: int,
    ready_only: bool = True,
    config: Config | None = None,
) -> list[RetrievedChunk]:
    """Passages from across one lesson, in reading order, at most ``limit``.

    A request about a lesson as a whole names nothing a search can match: "this
    lesson" scored 0.47 against a subtraction lesson, and the turn was refused as
    outside the lesson the student was sitting in. A lookup reads the lesson
    itself. A lesson longer than ``limit`` is sampled at even steps from its first
    passage to its last, so a summary covers the end of the lesson and not only
    its opening pages. ``score`` is 0.0: nothing was ranked.
    """
    if not namespaces:
        raise ValueError("get_lesson_chunks requires at least one namespace (invariant 1).")
    if limit < 1:
        return []
    config = config or Config.load()
    ready_clause = ""
    if ready_only:
        ready_clause = (" AND c.lecture_id IN (SELECT lecture_id FROM "
                        "ai.lecture_ingestion WHERE status = 'ready')")
    sql = (
        "SELECT DISTINCT c.chunk_id::text, c.text, c.heading_path, c.page_from, "
        "c.page_to, c.lecture_id::text, c.language, c.topic_ids, c.order_index "
        "FROM ai.chunks c JOIN ai.vector_index v ON v.chunk_id = c.chunk_id "
        "WHERE c.lecture_id::text = %s AND v.namespace = ANY(%s)"
        + ready_clause + " ORDER BY c.order_index"
    )
    with connect(config) as conn, conn.cursor() as cur:
        cur.execute(sql, (str(lecture_id), list(namespaces)))
        rows = [r for r in cur.fetchall() if chunk_kind(r[1] or "") not in NOT_CONTENT]
    if len(rows) > limit:
        step = (len(rows) - 1) / (limit - 1) if limit > 1 else 0
        rows = [rows[round(i * step)] for i in range(limit)]
    chunks = [
        RetrievedChunk(
            chunk_id=r[0], score=0.0, text=r[1], heading_path=r[2],
            page_from=r[3], page_to=r[4], lecture_id=r[5], language=r[6],
            topic_ids=[str(t) for t in (r[7] or [])], order_index=r[8],
        )
        for r in rows
    ]
    return _apply_labels(chunks, _citation_labels([c.chunk_id for c in chunks], config))


def get_lecture_topics(
    lecture_ids: Sequence[str],
    *,
    limit: int = 20,
    config: Config | None = None,
) -> list[ResolvedTopic]:
    """Every topic belonging to ``lecture_ids``, in lesson order.

    A lookup, not a search: ``score`` is 0.0. It is what a lesson is *about*
    when the request itself says nothing — "quiz me on this lesson" carries no
    subject matter for an embedding to match, and the lesson the student is
    sitting in is the answer.
    """
    if not lecture_ids:
        return []
    config = config or Config.load()
    with connect(config) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT topic_id::text, label, subject_id::text FROM ai.topics "
            "WHERE lecture_id::text = ANY(%s) ORDER BY created_at LIMIT %s",
            ([str(x) for x in lecture_ids], limit))
        return [ResolvedTopic(topic_id=r[0], label=r[1] or "", score=0.0, subject_id=r[2])
                for r in cur.fetchall()]


def get_subject_topics(
    subject_id: str,
    *,
    limit: int = 20,
    config: Config | None = None,
) -> list[str]:
    """The labels of a subject's topics, in the order they were taught. A lookup."""
    config = config or Config.load()
    with connect(config) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT label FROM ai.topics WHERE subject_id::text = %s AND label <> '' "
            "AND " + NOT_BACKGROUND_TOPIC + " GROUP BY label ORDER BY min(created_at) LIMIT %s",
            (str(subject_id), limit))
        return [r[0] for r in cur.fetchall()]


def get_topic_lectures(
    topic_ids: Sequence[str],
    *,
    config: Config | None = None,
) -> dict[str, str]:
    """``{topic_id: lecture_id}`` for the given ids, from ``ai.topics`` (a topic belongs to a
    lecture). Used to backfill an item's lecture_id from its resolved topic so a graded answer
    can move mastery (invariant 12) even when the caller didn't supply the lecture."""
    if not topic_ids:
        return {}
    config = config or Config.load()
    with connect(config) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT topic_id::text, lecture_id::text FROM ai.topics "
            "WHERE topic_id::text = ANY(%s)", ([str(t) for t in topic_ids],))
        return {r[0]: r[1] for r in cur.fetchall() if r[1]}


# --- get_topic_chunks: representative chunks for topics ----------------------

def _citation_labels(
    chunk_ids: Sequence[str], config: Config
) -> dict[str, dict[str, Any]]:
    """Readable labels (subject, lesson, order) for chunks fetched without a search.

    These live on the Qdrant payload, not on ``ai.chunks``, so a lookup that goes
    straight to Postgres has no way to name the lesson it found — the citation
    renders as a raw id. One retrieve by point id fills them in.

    Best effort: a citation with no label still resolves to a lecture, so a
    missing payload degrades the label rather than failing the turn.
    """
    if not chunk_ids:
        return {}

    collection = config.vector_store.get("collection", "eduai_chunks")
    try:
        client = query_client(config)
        points = client.retrieve(collection, ids=list(chunk_ids), with_payload=True,
                                 with_vectors=False)
    except Exception:
        logger.warning("citation labels unavailable; citing by lecture id",
                       exc_info=True)
        return {}
    return {str(point.id): (point.payload or {}) for point in points}


def _apply_labels(
    chunks: list[RetrievedChunk], labels: dict[str, dict[str, Any]]
) -> list[RetrievedChunk]:
    """Attach readable citation labels to chunks that were fetched without them."""
    for chunk in chunks:
        label = labels.get(chunk.chunk_id) or {}
        chunk.subject_name = label.get("subject_name")
        chunk.lesson_title = label.get("lesson_title")
        chunk.lesson_order = label.get("lesson_order")
        chunk.subject_id = label.get("subject_id")
    return chunks


def get_topic_chunks(
    topic_ids: Sequence[str],
    namespaces: Sequence[str],
    *,
    per_topic: int = 2,
    ready_only: bool = True,
    config: Config | None = None,
) -> list[RetrievedChunk]:
    """Up to ``per_topic`` chunks for each of ``topic_ids``, within ``namespaces``.

    Fetches by ``topic_ids`` directly (not a vector search), namespace-scoped via
    ``ai.vector_index`` (invariant 1) and excluding non-``ready`` lectures
    (invariant 9). Ordered by the chunk's position in its lecture so the earliest
    (usually the most defining) chunks win. ``score`` is 0.0 — a lookup, not a
    ranked search.
    """
    if not topic_ids:
        return []
    if not namespaces:
        raise ValueError("get_topic_chunks requires at least one namespace (invariant 1).")
    config = config or Config.load()

    ready_clause = ""
    if ready_only:
        ready_clause = (" AND c.lecture_id IN (SELECT lecture_id FROM "
                        "ai.lecture_ingestion WHERE status = 'ready')")
    sql = (
        "SELECT DISTINCT c.chunk_id::text, c.text, c.heading_path, c.page_from, "
        "c.page_to, c.lecture_id::text, c.language, c.topic_ids, c.order_index "
        "FROM ai.chunks c "
        "JOIN ai.vector_index v ON v.chunk_id = c.chunk_id "
        "WHERE c.topic_ids && %s::uuid[] AND v.namespace = ANY(%s)"
        + ready_clause + " ORDER BY c.order_index"
    )
    wanted = [str(t) for t in topic_ids]

    with connect(config) as conn, conn.cursor() as cur:
        cur.execute(sql, (wanted, list(namespaces)))
        rows = cur.fetchall()

    # Cap to ``per_topic`` chunks per requested topic.
    counts: dict[str, int] = {}
    out: list[RetrievedChunk] = []
    for r in rows:
        if chunk_kind(r[1] or "") in NOT_CONTENT:
            continue
        chunk_topics = [str(t) for t in (r[7] or [])]
        keys = [t for t in wanted if t in chunk_topics] or ["_"]
        if all(counts.get(key, 0) >= per_topic for key in keys):
            continue
        for key in keys:
            counts[key] = counts.get(key, 0) + 1
        out.append(RetrievedChunk(
            chunk_id=r[0], score=0.0, text=r[1], heading_path=r[2],
            page_from=r[3], page_to=r[4], lecture_id=r[5], language=r[6],
            topic_ids=chunk_topics,
        ))

    return _apply_labels(out, _citation_labels([c.chunk_id for c in out], config))


# --- get_figures: syllabus figures for topics/lectures -----------------------

def get_figures(
    *,
    topic_ids: Sequence[str] | None = None,
    lecture_id: str | None = None,
    config: Config | None = None,
) -> list[dict[str, Any]]:
    """Figures matching ``topic_ids`` (overlap) and/or ``lecture_id``.

    Each row: ``{figure_id, lecture_id, s3_uri, caption, alt_text, page,
    topic_ids}``. Raises when neither filter is given (an unscoped scan is never
    intended)."""
    # TODO: accept an optional query and, when captions are embedded, rank the matched
    # figures by caption-embedding similarity (small set -> brute-force, no ANN index).
    if not topic_ids and not lecture_id:
        raise ValueError("get_figures requires topic_ids and/or lecture_id.")

    clauses, params = [], []
    if topic_ids:
        clauses.append("f.topic_ids && %s::uuid[]")
        params.append([str(t) for t in topic_ids])
    if lecture_id:
        clauses.append("f.lecture_id::text = %s")
        params.append(str(lecture_id))
    sql = (
        "SELECT f.figure_id::text, f.lecture_id::text, f.s3_uri, f.caption, "
        "f.alt_text, f.page, f.topic_ids FROM ai.figures f WHERE "
        + " AND ".join(clauses)
    )
    with connect(config) as conn, conn.cursor() as cur:
        cur.execute(sql, tuple(params))
        rows = cur.fetchall()
    return [
        {
            "figure_id": r[0],
            "lecture_id": r[1],
            "s3_uri": r[2],
            "caption": r[3],
            "alt_text": r[4],
            "page": r[5],
            "topic_ids": [str(t) for t in (r[6] or [])],
        }
        for r in rows
    ]


# --- get_question_bank: existing questions from the backend ------------------

def get_question_bank(
    subject_id: str,
    topic_ids: Sequence[str] | None = None,
    *,
    count: int = 10,
    type_mix: dict[str, int] | None = None,
    config: Config | None = None,
) -> list[dict[str, Any]]:
    """Bank questions for ``subject_id``/``topic_ids`` (up to ``count``). Feeds the
    BANK/HYBRID assessment sources (off by default; GENERATED is the default)."""
    config = config or Config.load()
    client = get_backend_client(config)
    return client.get_question_bank(
        subject_id,
        topic_ids=list(topic_ids) if topic_ids else None,
        count=count,
        type_mix=type_mix,
    )


# --- assessment reads: questions + grading rubric ----------------------------

# Answer-key fields fetch_rubric keeps from each question.
_RUBRIC_FIELDS = ("id", "question_type", "correct_answer_text", "rubric",
                  "explanation", "points", "topic_ids", "lecture_id")


def get_item(
    kind: str,
    assessment_id: str,
    *,
    config: Config | None = None,
) -> list[dict[str, Any]]:
    """All questions of an assessment. ``kind`` is ``homework`` or ``quiz``. The AI
    never reads the student's submission from the DB (invariant 4) — these fetch the
    *correct* answers only."""
    config = config or Config.load()
    client = get_backend_client(config)
    return client.get_questions(kind, assessment_id)


def fetch_rubric(
    kind: str,
    assessment_id: str,
    *,
    config: Config | None = None,
) -> list[dict[str, Any]]:
    """The answer key for grading: each question projected to its rubric fields."""
    questions = get_item(kind, assessment_id, config=config)
    return [{f: q.get(f) for f in _RUBRIC_FIELDS if f in q} for q in questions]


# --- mistake_tags: shared vocabulary of mistake categories -------------------

def list_mistake_tags(
    subject_id: str | None = None, config: Config | None = None
) -> list[dict[str, Any]]:
    """All mistake tags: ``[{tag_id, label, description}]``.

    Scoped to ``subject_id`` when given (plus cross-subject tags where
    ``subject_id IS NULL``); unscoped returns the whole catalog.
    """
    where, params = "", ()
    if subject_id is not None:
        where = " WHERE subject_id = %s OR subject_id IS NULL"
        params = (subject_id,)
    with connect(config) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT tag_id::text, label, description FROM ai.mistake_tags"
            + where + " ORDER BY label", params)
        rows = cur.fetchall()
    return [{"tag_id": r[0], "label": r[1], "description": r[2]} for r in rows]


def create_mistake_tag(
    label: str,
    *,
    description: str | None = None,
    subject_id: str | None = None,
    created_by: str = "agent",
    config: Config | None = None,
) -> str:
    """Return the id of the tag named ``label``, creating it if absent.

    Idempotent on ``(subject_id, lower(label))``: a prior/concurrent identical
    tag in the same subject wins rather than a duplicate being inserted."""
    import psycopg

    label = label.strip()
    if not label:
        raise ValueError("mistake tag label must be non-empty.")
    try:
        return _create_mistake_tag(label, description, subject_id, created_by, config)
    except psycopg.errors.UndefinedColumn:
        # `subject_id` is a pending schema add — absent on a partial dev DB. Retry unscoped
        # (a fresh connection: the failed one is in an aborted transaction).
        if subject_id is None:
            raise
        logger.warning("ai.mistake_tags.subject_id absent; creating tag unscoped")
        return _create_mistake_tag(label, description, None, created_by, config)


def _create_mistake_tag(label, description, subject_id, created_by, config) -> str:
    sel = "SELECT tag_id::text FROM ai.mistake_tags WHERE lower(label) = lower(%s)"
    sel_params: tuple = (label,)
    if subject_id is not None:
        sel += " AND subject_id IS NOT DISTINCT FROM %s"
        sel_params = (label, subject_id)
    cols, ph, ins_params = "label, description, created_by", "%s, %s, %s", (
        label, description, created_by)
    if subject_id is not None:
        cols = "label, description, subject_id, created_by"
        ph = "%s, %s, %s, %s"
        ins_params = (label, description, subject_id, created_by)
    with connect(config) as conn:
        with conn.cursor() as cur:
            cur.execute(sel + " LIMIT 1", sel_params)
            hit = cur.fetchone()
            if hit:
                return hit[0]
            cur.execute(
                f"INSERT INTO ai.mistake_tags ({cols}) "
                f"VALUES ({ph}) RETURNING tag_id::text", ins_params)
            tag_id = cur.fetchone()[0]
        conn.commit()
    return tag_id


# --- mastery store: ai.mastery, ai.mastery_events (agent-owned) --------------

# mastery_events columns written by write_mastery_events (order matters).
_EVENT_COLS = (
    "user_id", "item_id", "topic_id", "lecture_id", "correct", "partial_score",
    "original_score", "difficulty", "bloom_level", "mistake_tag_ids", "source",
)


def get_mastery(
    user_id: str,
    topic_ids: Sequence[str] | None = None,
    *,
    subject_id: str | None = None,
    lecture_id: str | None = None,
    config: Config | None = None,
) -> list[dict[str, Any]]:
    """Per-topic mastery rows for ``user_id``, each with a derived ``status``
    (strong|weak|developing|unknown). Optionally restricted to ``topic_ids`` and/or scoped
    to a ``subject_id`` / ``lecture_id`` (joined via ``ai.topics`` — subject and lecture
    are not stored on ``ai.mastery``)."""
    config = config or Config.load()
    params = MasteryParams.from_config(config)
    join = " JOIN ai.topics t ON t.topic_id = m.topic_id" if (subject_id or lecture_id) else ""
    sql = ("SELECT m.topic_id::text, m.score, m.attempts, m.last_seen, m.trend, m.confidence "
           "FROM ai.mastery m" + join + " WHERE m.user_id = %s")
    args: list[Any] = [str(user_id)]
    if topic_ids:
        sql += " AND m.topic_id::text = ANY(%s)"
        args.append([str(t) for t in topic_ids])
    if subject_id:
        sql += " AND t.subject_id = %s"
        args.append(str(subject_id))
    if lecture_id:
        sql += " AND t.lecture_id = %s"
        args.append(str(lecture_id))
    with connect(config) as conn, conn.cursor() as cur:
        cur.execute(sql, tuple(args))
        rows = cur.fetchall()
    out = []
    for r in rows:
        m = Mastery(score=r[1], attempts=r[2], trend=r[4], confidence=r[5])
        out.append({
            "topic_id": r[0], "score": r[1], "attempts": r[2], "last_seen": r[3],
            "trend": r[4], "confidence": r[5], "status": status(m, params),
        })
    return out


# group_by -> the ai.topics column mastery rolls up over.
_ROLLUP_COL = {"subject": "subject_id", "lecture": "lecture_id"}


def get_mastery_rollup(
    user_id: str,
    group_by: str = "subject",
    *,
    config: Config | None = None,
) -> list[dict[str, Any]]:
    """Mastery rolled up per subject or per lecture — for the teacher/parent views, which
    want a subject-level number, not per-topic. Derived by joining ``ai.topics`` (no
    per-subject/lesson storage). Each row: ``{subject_id|lecture_id, score, attempts,
    topics, strong, developing, weak, unknown, status}`` where ``score`` is the
    attempts-weighted mean of the group's topic scores and ``status`` buckets that mean."""
    col = _ROLLUP_COL.get(group_by)
    if not col:
        raise ValueError("group_by must be 'subject' or 'lecture'.")
    config = config or Config.load()
    params = MasteryParams.from_config(config)
    sql = (f"SELECT t.{col}::text, m.score, m.attempts, m.trend, m.confidence "
           "FROM ai.mastery m JOIN ai.topics t ON t.topic_id = m.topic_id "
           f"WHERE m.user_id = %s AND t.{col} IS NOT NULL")
    with connect(config) as conn, conn.cursor() as cur:
        cur.execute(sql, (str(user_id),))
        rows = cur.fetchall()
    groups: dict[str, dict] = {}
    for gid, score, attempts, trend, confidence in rows:
        g = groups.setdefault(gid, {"num": 0.0, "den": 0, "topics": 0,
                                    "strong": 0, "developing": 0, "weak": 0, "unknown": 0})
        w = max(1, attempts or 0)
        g["num"] += (score or 0.0) * w
        g["den"] += w
        g["topics"] += 1
        st = status(Mastery(score=score or 0.0, attempts=attempts or 0,
                            trend=trend, confidence=confidence or 0.0), params)
        g[st] += 1
    out = []
    for gid, g in groups.items():
        mean = g["num"] / g["den"] if g["den"] else 0.0
        out.append({
            col: gid, "score": round(mean, 4), "attempts": g["den"], "topics": g["topics"],
            "strong": g["strong"], "developing": g["developing"], "weak": g["weak"],
            "unknown": g["unknown"],
            "status": status(Mastery(score=mean, attempts=g["den"]), params),
        })
    return out


def _user_ids(user_ids: str | Sequence[str]) -> list[str]:
    return [str(user_ids)] if isinstance(user_ids, str) else [str(u) for u in user_ids]


def get_scope_topics(
    user_ids: str | Sequence[str],
    *,
    subject_ids: Sequence[str] | None = None,
    config: Config | None = None,
) -> list[dict[str, Any]]:
    """Every topic in ``subject_ids`` and in any subject the student (or any student of
    a roster) has graded evidence in: ``[{topic_id, label, lecture_id, subject_id}]``.
    The universe an analysis counts against, so an untouched topic is counted as not
    practised, not left out."""
    config = config or Config.load()
    sql = ("SELECT t.topic_id::text, t.label, t.lecture_id::text, t.subject_id::text "
           "FROM ai.topics t WHERE t.subject_id = ANY(%s) OR t.subject_id IN ("
           "SELECT DISTINCT s.subject_id FROM ai.mastery m "
           "JOIN ai.topics s ON s.topic_id = m.topic_id WHERE m.user_id = ANY(%s)) "
           "ORDER BY t.subject_id, t.lecture_id, t.label")
    with connect(config) as conn, conn.cursor() as cur:
        cur.execute(sql, ([str(s) for s in subject_ids or []], _user_ids(user_ids)))
        rows = cur.fetchall()
    return [{"topic_id": r[0], "label": r[1] or "", "lecture_id": r[2], "subject_id": r[3]}
            for r in rows]


def get_recent_outcomes(
    user_id: str,
    topic_ids: Sequence[str],
    *,
    per_topic: int,
    config: Config | None = None,
) -> dict[str, list[float]]:
    """The newest ``per_topic`` answer scores per topic, newest first."""
    if not topic_ids:
        return {}
    config = config or Config.load()
    sql = ("SELECT topic_id, partial_score, correct FROM ("
           "SELECT e.topic_id::text AS topic_id, e.partial_score, e.correct, "
           "row_number() OVER (PARTITION BY e.topic_id ORDER BY e.created_at DESC) AS n "
           "FROM ai.mastery_events e "
           "WHERE e.user_id = %s AND e.topic_id::text = ANY(%s)) ranked "
           "WHERE n <= %s ORDER BY topic_id, n")
    with connect(config) as conn, conn.cursor() as cur:
        cur.execute(sql, (str(user_id), [str(t) for t in topic_ids], int(per_topic)))
        rows = cur.fetchall()
    out: dict[str, list[float]] = {}
    for tid, partial, correct in rows:
        out.setdefault(tid, []).append(
            float(partial) if partial is not None else (1.0 if correct else 0.0))
    return out


def get_weekly_activity(
    user_ids: str | Sequence[str],
    *,
    weeks: int,
    subject_ids: Sequence[str] | None = None,
    config: Config | None = None,
) -> list[dict[str, Any]]:
    """Graded answers per calendar week over the last ``weeks`` weeks, oldest first:
    ``[{week_start, answers, correct, mean_score}]``, for one student or a roster.
    Weeks with no answers are absent."""
    config = config or Config.load()
    join = " JOIN ai.topics t ON t.topic_id = e.topic_id" if subject_ids else ""
    sql = ("SELECT date_trunc('week', e.created_at) AS wk, count(*), "
           "count(*) FILTER (WHERE e.correct), "
           "avg(coalesce(e.partial_score, CASE WHEN e.correct THEN 1.0 ELSE 0.0 END)) "
           "FROM ai.mastery_events e" + join + " WHERE e.user_id = ANY(%s) "
           "AND e.created_at >= date_trunc('week', now()) - make_interval(weeks => %s - 1)")
    args: list[Any] = [_user_ids(user_ids), int(weeks)]
    if subject_ids:
        sql += " AND t.subject_id = ANY(%s)"
        args.append([str(s) for s in subject_ids])
    sql += " GROUP BY wk ORDER BY wk"
    with connect(config) as conn, conn.cursor() as cur:
        cur.execute(sql, tuple(args))
        rows = cur.fetchall()
    return [{"week_start": wk.date().isoformat(), "answers": int(n), "correct": int(c or 0),
             "mean_score": round(float(mean or 0.0), 4)} for wk, n, c, mean in rows]


def get_roster_mastery(
    user_ids: Sequence[str],
    *,
    subject_ids: Sequence[str] | None = None,
    config: Config | None = None,
) -> list[dict[str, Any]]:
    """Per-(student, topic) mastery for a whole roster in one query, each with its
    derived ``status``."""
    if not user_ids:
        return []
    config = config or Config.load()
    params = MasteryParams.from_config(config)
    join = " JOIN ai.topics t ON t.topic_id = m.topic_id" if subject_ids else ""
    sql = ("SELECT m.user_id::text, m.topic_id::text, m.score, m.attempts, m.last_seen "
           "FROM ai.mastery m" + join + " WHERE m.user_id = ANY(%s)")
    args: list[Any] = [_user_ids(user_ids)]
    if subject_ids:
        sql += " AND t.subject_id = ANY(%s)"
        args.append([str(s) for s in subject_ids])
    with connect(config) as conn, conn.cursor() as cur:
        cur.execute(sql, tuple(args))
        rows = cur.fetchall()
    return [{"user_id": uid, "topic_id": tid, "score": score, "attempts": attempts,
             "last_seen": last_seen,
             "status": status(Mastery(score=score or 0.0, attempts=attempts or 0), params)}
            for uid, tid, score, attempts, last_seen in rows]


def get_roster_history(
    user_ids: Sequence[str],
    *,
    per_user: int,
    config: Config | None = None,
) -> dict[str, dict[str, Any]]:
    """Each student's newest ``per_user`` answer scores (returned oldest first, the
    order a trend reads them in) and when they last answered anything."""
    if not user_ids:
        return {}
    config = config or Config.load()
    sql = ("SELECT user_id, created_at, partial_score, correct FROM ("
           "SELECT e.user_id::text AS user_id, e.created_at, e.partial_score, e.correct, "
           "row_number() OVER (PARTITION BY e.user_id ORDER BY e.created_at DESC) AS n "
           "FROM ai.mastery_events e WHERE e.user_id = ANY(%s)) ranked "
           "WHERE n <= %s ORDER BY user_id, created_at DESC")
    with connect(config) as conn, conn.cursor() as cur:
        cur.execute(sql, (_user_ids(user_ids), int(per_user)))
        rows = cur.fetchall()
    out: dict[str, dict[str, Any]] = {}
    for uid, created_at, partial, correct in rows:
        entry = out.setdefault(uid, {"last_active": created_at, "scores": []})
        entry["scores"].insert(
            0, float(partial) if partial is not None else (1.0 if correct else 0.0))
    return out


def get_top_mistakes(
    user_ids: Sequence[str],
    *,
    since_days: int,
    limit: int,
    subject_ids: Sequence[str] | None = None,
    config: Config | None = None,
) -> list[dict[str, Any]]:
    """The mistake categories a roster made most often in the last ``since_days``:
    ``[{tag_id, label, occurrences, students}]``."""
    if not user_ids:
        return []
    config = config or Config.load()
    join = " JOIN ai.topics t ON t.topic_id = e.topic_id" if subject_ids else ""
    sql = ("SELECT tag.tag_id::text, tag.label, count(*), count(DISTINCT x.user_id) FROM ("
           "SELECT e.user_id, unnest(e.mistake_tag_ids) AS tag_id "
           "FROM ai.mastery_events e" + join + " WHERE e.user_id = ANY(%s) "
           "AND e.created_at >= now() - make_interval(days => %s)")
    args: list[Any] = [_user_ids(user_ids), int(since_days)]
    if subject_ids:
        sql += " AND t.subject_id = ANY(%s)"
        args.append([str(s) for s in subject_ids])
    sql += (") x JOIN ai.mistake_tags tag ON tag.tag_id::text = x.tag_id::text "
            "GROUP BY tag.tag_id, tag.label ORDER BY 3 DESC, 2 LIMIT %s")
    args.append(int(limit))
    with connect(config) as conn, conn.cursor() as cur:
        cur.execute(sql, tuple(args))
        rows = cur.fetchall()
    return [{"tag_id": r[0], "label": r[1] or "", "occurrences": int(r[2]),
             "students": int(r[3])} for r in rows]


def get_most_missed(
    user_ids: Sequence[str],
    *,
    since_days: int,
    limit: int,
    subject_ids: Sequence[str] | None = None,
    config: Config | None = None,
) -> list[dict[str, Any]]:
    """The questions a roster answered wrong most often in the last ``since_days``:
    ``[{item_id, topic_id, answers, wrong, students}]``."""
    if not user_ids:
        return []
    config = config or Config.load()
    join = " JOIN ai.topics t ON t.topic_id = e.topic_id" if subject_ids else ""
    sql = ("SELECT e.item_id::text, e.topic_id::text, count(*), "
           "count(*) FILTER (WHERE NOT e.correct), count(DISTINCT e.user_id) "
           "FROM ai.mastery_events e" + join + " WHERE e.user_id = ANY(%s) "
           "AND e.item_id IS NOT NULL AND e.created_at >= now() - make_interval(days => %s)")
    args: list[Any] = [_user_ids(user_ids), int(since_days)]
    if subject_ids:
        sql += " AND t.subject_id = ANY(%s)"
        args.append([str(s) for s in subject_ids])
    sql += (" GROUP BY e.item_id, e.topic_id HAVING count(*) FILTER (WHERE NOT e.correct) > 0 "
            "ORDER BY 4 DESC, 3 DESC LIMIT %s")
    args.append(int(limit))
    with connect(config) as conn, conn.cursor() as cur:
        cur.execute(sql, tuple(args))
        rows = cur.fetchall()
    return [{"item_id": r[0], "topic_id": r[1], "answers": int(r[2]), "wrong": int(r[3]),
             "students": int(r[4])} for r in rows]


def get_direct_prereqs(
    topic_ids: Sequence[str],
    *,
    config: Config | None = None,
) -> dict[str, list[tuple[str, str]]]:
    """``{topic_id: [(prereq_topic_id, label), ...]}``: one hop, for many topics in one
    query. ``get_topic_deps`` walks further but loses which seed a prerequisite serves."""
    if not topic_ids:
        return {}
    config = config or Config.load()
    with connect(config) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT d.topic_id::text, d.prereq_topic_id::text, t.label "
            "FROM ai.topic_dependencies d "
            "LEFT JOIN ai.topics t ON t.topic_id = d.prereq_topic_id "
            "WHERE d.topic_id::text = ANY(%s)",
            ([str(t) for t in topic_ids],))
        rows = cur.fetchall()
    out: dict[str, list[tuple[str, str]]] = {}
    for tid, pid, label in rows:
        out.setdefault(tid, []).append((pid, label or ""))
    return out


def get_mastery_trend(
    user_id: str,
    *,
    topic_id: str | None = None,
    subject_id: str | None = None,
    lecture_id: str | None = None,
    config: Config | None = None,
) -> dict[str, Any]:
    """Performance over time for ``user_id`` from ``ai.mastery_events`` (ordered by
    ``created_at``). Optionally narrowed to one ``topic_id``, one ``lecture_id``, or a
    ``subject_id`` (joined via ``ai.topics``). Backs endpoint #9 and report_sg's trend chart.

    Returns ``{events: [{created_at, score, correct, difficulty, source, topic_id}],
    count, direction}`` where ``score`` is ``partial_score`` (else 1.0/0.0 from
    ``correct``) and ``direction`` (improving|flat|declining|insufficient) compares the
    mean score of the earliest vs latest half of the window. Read-only."""
    config = config or Config.load()
    window = config.get("report.trend.window", 50)       # newest N events considered
    min_events = config.get("report.trend.min_events", 4)  # below -> direction unknown
    flat_band = config.get("report.trend.flat_band", 0.05)  # |delta| within -> flat
    join = " JOIN ai.topics t ON t.topic_id = e.topic_id" if subject_id else ""
    sql = ("SELECT e.created_at, e.correct, e.partial_score, e.difficulty, "
           "e.source::text, e.topic_id::text "
           "FROM ai.mastery_events e" + join + " WHERE e.user_id = %s")
    args: list[Any] = [str(user_id)]
    if topic_id:
        sql += " AND e.topic_id = %s"
        args.append(str(topic_id))
    if lecture_id:
        sql += " AND e.lecture_id = %s"
        args.append(str(lecture_id))
    if subject_id:
        sql += " AND t.subject_id = %s"
        args.append(str(subject_id))
    # Take the newest `window` rows, then present oldest-first for a left-to-right series.
    sql += " ORDER BY e.created_at DESC LIMIT %s"
    args.append(int(window))
    with connect(config) as conn, conn.cursor() as cur:
        cur.execute(sql, tuple(args))
        rows = cur.fetchall()
    rows = list(reversed(rows))
    events = []
    for created_at, correct, partial, difficulty, source, tid in rows:
        score = partial if partial is not None else (1.0 if correct else 0.0)
        events.append({
            "created_at": created_at, "score": score, "correct": correct,
            "difficulty": difficulty, "source": source, "topic_id": tid,
        })
    return {"events": events, "count": len(events),
            "direction": trend_direction([e["score"] for e in events],
                                          min_events, flat_band)}


def trend_direction(scores: list[float], min_events: int, flat_band: float) -> str:
    """improving|flat|declining from earliest-half vs latest-half mean; ``insufficient``
    below ``min_events``. Purely deterministic — no model picks the direction."""
    if len(scores) < min_events:
        return "insufficient"
    mid = len(scores) // 2
    early = sum(scores[:mid]) / mid
    late = sum(scores[mid:]) / (len(scores) - mid)
    delta = late - early
    if abs(delta) <= flat_band:
        return "flat"
    return "improving" if delta > 0 else "declining"


_INSERT_EVENT_SQL = (
    f"INSERT INTO ai.mastery_events ({', '.join(_EVENT_COLS)}) "
    f"VALUES ({', '.join(['%s'] * len(_EVENT_COLS))}) RETURNING event_id::text"
)

_UPSERT_MASTERY_SQL = (
    "INSERT INTO ai.mastery "
    "(user_id, topic_id, score, attempts, last_seen, trend, confidence) "
    "VALUES (%s, %s, %s, %s, now(), %s, %s) "
    "ON CONFLICT (user_id, topic_id) DO UPDATE SET "
    "score = EXCLUDED.score, attempts = EXCLUDED.attempts, "
    "last_seen = EXCLUDED.last_seen, trend = EXCLUDED.trend, "
    "confidence = EXCLUDED.confidence"
)


def _insert_events(cur, events: Sequence[Mapping[str, Any]]) -> list[str]:
    ids: list[str] = []
    for e in events:
        if not e.get("topic_id") or not e.get("source"):
            raise ValueError("mastery event requires topic_id and source.")
        cur.execute(_INSERT_EVENT_SQL, tuple(e.get(c) for c in _EVENT_COLS))
        ids.append(cur.fetchone()[0])
    return ids


def _upsert_mastery(cur, user_id: str, topic_id: str, mastery: Mastery) -> None:
    cur.execute(_UPSERT_MASTERY_SQL, (str(user_id), str(topic_id), mastery.score,
                                      mastery.attempts, mastery.trend, mastery.confidence))


def write_mastery_events(
    events: Sequence[Mapping[str, Any]],
    *,
    config: Config | None = None,
) -> list[str]:
    """Append graded events to ``ai.mastery_events`` (immutable). Returns the new
    ``event_id``s. A teacher correction is just another event with ``original_score``
    set — never an update to an existing row."""
    if not events:
        return []
    with connect(config) as conn:
        with conn.cursor() as cur:
            ids = _insert_events(cur, events)
        conn.commit()
    return ids


def update_mastery(
    user_id: str,
    topic_id: str,
    mastery: Mastery,
    *,
    config: Config | None = None,
) -> None:
    """Upsert the rolled-up aggregate for one (user, topic). Writes absolute values
    from ``mastery`` (the model already incremented attempts); ``last_seen`` = now."""
    with connect(config) as conn:
        with conn.cursor() as cur:
            _upsert_mastery(cur, user_id, topic_id, mastery)
        conn.commit()


def record_graded_evidence(
    user_id: str,
    events: Sequence[Mapping[str, Any]],
    aggregates: Mapping[str, Mastery],
    *,
    config: Config | None = None,
) -> list[str]:
    """Append ``events`` and upsert the aggregates they produced in one transaction.

    A rollup without its evidence cannot be audited, and evidence without its rollup
    never reaches the student, so either both land or neither does.
    """
    with connect(config) as conn:
        with conn.cursor() as cur:
            ids = _insert_events(cur, events)
            for topic_id, mastery in aggregates.items():
                _upsert_mastery(cur, user_id, topic_id, mastery)
        conn.commit()
    return ids


# --- reports: ai.reports (agent-owned) ---------------------------------------

def save_report(
    scope: str,
    payload: Mapping[str, Any],
    *,
    user_id: str | None = None,
    class_id: str | None = None,
    config: Config | None = None,
) -> dict[str, str]:
    """Store one generated report. Reports are history: a new one never replaces an
    old row, so what a student was told on a given day can still be read back."""
    if not user_id and not class_id:
        raise ValueError("a report belongs to a user or a class.")
    with connect(config) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO ai.reports (user_id, class_id, scope, payload_json) "
                "VALUES (%s, %s, %s::ai.report_scope, %s::jsonb) "
                "RETURNING report_id::text, generated_at",
                (user_id, class_id, scope, json.dumps(payload, ensure_ascii=False,
                                                      default=str)))
            report_id, generated_at = cur.fetchone()
        conn.commit()
    return {"report_id": report_id, "generated_at": generated_at.isoformat()}


def attach_report_visual(
    report_id: str,
    visual: Mapping[str, Any],
    *,
    config: Config | None = None,
) -> bool:
    """Add a rendered chart to a stored report, so serving it later shows the chart
    built with it. False when no report has that id."""
    artifact_id = visual.get("artifact_id")
    refs = ("artifact_refs = array_append(coalesce(artifact_refs, '{}'), %s::uuid), "
            if artifact_id else "")
    args = ((artifact_id,) if artifact_id else ()) + (
        json.dumps(dict(visual), ensure_ascii=False, default=str), report_id)
    with connect(config) as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"UPDATE ai.reports SET {refs}"
                "payload_json = payload_json || jsonb_build_object('visual', %s::jsonb) "
                "WHERE report_id = %s::uuid",
                args)
            attached = cur.rowcount == 1
        conn.commit()
    return attached


def get_latest_report(
    scope: str,
    *,
    user_id: str | None = None,
    class_id: str | None = None,
    subject_id: str | None = None,
    config: Config | None = None,
) -> dict[str, Any] | None:
    """The newest stored report for ``user_id`` or ``class_id``, or None when there is
    none; with ``subject_id``, the newest one built for that subject. ``newer_evidence``
    counts the student's graded answers since it was written; it is None for a class
    report, whose roster this service does not hold."""
    if not user_id and not class_id:
        raise ValueError("a report belongs to a user or a class.")
    owner, value = ("r.user_id", user_id) if user_id else ("r.class_id", class_id)
    newer = ("(SELECT count(*) FROM ai.mastery_events e "
             "WHERE e.user_id = r.user_id AND e.created_at > r.generated_at)"
             if user_id else "NULL")
    subject = " AND r.payload_json->>'subject_id' = %s" if subject_id else ""
    with connect(config) as conn, conn.cursor() as cur:
        cur.execute(
            f"SELECT r.report_id::text, r.generated_at, r.payload_json, {newer} "
            f"FROM ai.reports r WHERE r.scope = %s::ai.report_scope AND {owner} = %s"
            f"{subject} ORDER BY r.generated_at DESC LIMIT 1",
            (scope, value, subject_id) if subject_id else (scope, value))
        row = cur.fetchone()
    if row is None:
        return None
    report_id, generated_at, payload, newer_count = row
    return {"report_id": report_id, "generated_at": generated_at.isoformat(),
            "payload": payload, "newer_evidence": newer_count,
            "stale": None if newer_count is None else newer_count > 0}

