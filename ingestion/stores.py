"""Write-side stores — the two-store model.

Ingestion owns ``ai.chunks``, ``ai.figures`` and ``ai.vector_index`` and writes
them directly (no API). Chunk text/metadata -> Postgres; chunk vector -> Qdrant;
``ai.vector_index`` is a Postgres manifest of what is in Qdrant (for idempotent
re-ingestion), never searched. Both stores have dry-run implementations so a run
needs neither service. A single ``chunk_id`` keys the row, the manifest and the
point.
"""

from __future__ import annotations

import logging
import uuid
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from shared.matching import chunk_kind
from shared.namespaces import NOT_BACKGROUND_TOPIC
from shared.vectors import DENSE_VECTOR, SPARSE_VECTOR
from shared.vector_store import upsert_client

from .config import Config
from .state import Chunk, Dependency, Figure, Topic

logger = logging.getLogger(__name__)


# ===========================================================================
# Chunk store (Postgres: ai.chunks, ai.figures, ai.vector_index)
# ===========================================================================

class ChunkStore(ABC):
    @abstractmethod
    def delete_lecture(self, lecture_id: str) -> None:
        """Remove a lecture's chunks, figures and manifest rows so a re-ingest
        replaces rather than duplicates. Topics are subject-owned and untouched."""

    @abstractmethod
    def write_chunks(self, chunks: Sequence[Chunk]) -> list[str]:
        """Insert ``ai.chunks`` rows; set + return each ``chunk_id``."""

    @abstractmethod
    def write_figures(self, figures: Sequence[Figure]) -> None:
        """Insert ``ai.figures`` rows."""

    @abstractmethod
    def write_vector_index(self, chunks: Sequence[Chunk]) -> None:
        """Upsert the ``ai.vector_index`` manifest (ON CONFLICT chunk_id,namespace)."""

    @abstractmethod
    def read_topic_catalog(self, subject_id: str, *, include_background: bool = True,
                           ) -> list[tuple[str, str, list[float]]]:
        """Existing ``(topic_id, label, label_embedding)`` for a subject — the
        dependency stage's prerequisite candidate pool. A ministry lesson leaves
        background topics out: a prerequisite is shown to a student by name."""

    @abstractmethod
    def read_lecture_topics(self, lecture_id: str) -> list[tuple[str, str, list[float]]]:
        """Existing ``(topic_id, label, label_embedding)`` for one lecture — what
        re-ingest matches new topics against to reuse a ``topic_id``."""

    @abstractmethod
    def write_topics(self, topics: Sequence[Topic]) -> None:
        """Insert new ``ai.topics`` rows (reused topics are not re-written)."""

    @abstractmethod
    def delete_lecture_dependencies(self, lecture_id: str) -> None:
        """Remove the ``ai.topic_dependencies`` edges owned by a lecture's topics,
        so a re-ingest replaces them rather than leaving stale edges. Safe: edges
        are ingestion-owned facts, not student history."""

    @abstractmethod
    def write_dependencies(self, deps: Sequence[Dependency]) -> None:
        """Upsert ``ai.topic_dependencies`` rows (ON CONFLICT topic,prereq)."""

    @abstractmethod
    def write_ingestion_record(self, record: dict[str, Any]) -> None:
        """Upsert the per-lecture ``ai.lecture_ingestion`` tracking row."""


_INSERT_CHUNK = """
INSERT INTO ai.chunks
    (chunk_id, lecture_id, topic_ids, text, heading_path,
     page_from, page_to, language, order_index, version)
VALUES
    (:chunk_id, :lecture_id, :topic_ids, :text, :heading_path,
     :page_from, :page_to, :language, :order_index, :version)
"""

_INSERT_FIGURE = """
INSERT INTO ai.figures
    (lecture_id, chunk_id, topic_ids, s3_uri, caption, alt_text, page)
VALUES
    (:lecture_id, :chunk_id, :topic_ids, :s3_uri, :caption, :alt_text, :page)
"""

_UPSERT_INDEX = """
INSERT INTO ai.vector_index
    (chunk_id, lecture_id, topic_ids, subject_id, grade, namespace, language, version)
VALUES
    (:chunk_id, :lecture_id, :topic_ids, :subject_id, :grade, :namespace, :language, :version)
ON CONFLICT (chunk_id, namespace)
DO UPDATE SET version = EXCLUDED.version,
              topic_ids = EXCLUDED.topic_ids,
              language = EXCLUDED.language
"""

_DELETE_LECTURE = (
    "DELETE FROM ai.figures WHERE lecture_id = :lecture_id",
    "DELETE FROM ai.vector_index WHERE lecture_id = :lecture_id",
    "DELETE FROM ai.chunks WHERE lecture_id = :lecture_id",
)

_UPSERT_INGESTION = """
INSERT INTO ai.lecture_ingestion
    (lecture_id, subject_id, version, status, chunk_count, figure_count,
     topic_count, needs_review_reason, ingested_at)
VALUES
    (:lecture_id, :subject_id, :version, :status, :chunk_count, :figure_count,
     :topic_count, :needs_review_reason, CURRENT_TIMESTAMP)
ON CONFLICT (lecture_id) DO UPDATE SET
    subject_id = EXCLUDED.subject_id,
    version = EXCLUDED.version,
    status = EXCLUDED.status,
    chunk_count = EXCLUDED.chunk_count,
    figure_count = EXCLUDED.figure_count,
    topic_count = EXCLUDED.topic_count,
    needs_review_reason = EXCLUDED.needs_review_reason,
    ingested_at = CURRENT_TIMESTAMP
"""

_SELECT_TOPICS = """
SELECT topic_id, label, label_embedding
FROM ai.topics
WHERE subject_id = :subject_id
"""

_SELECT_SYLLABUS_TOPICS = _SELECT_TOPICS + "AND " + NOT_BACKGROUND_TOPIC + "\n"

_SELECT_LECTURE_TOPICS = """
SELECT topic_id, label, label_embedding
FROM ai.topics
WHERE lecture_id = :lecture_id
"""

_DELETE_LECTURE_DEPS = """
DELETE FROM ai.topic_dependencies
WHERE topic_id IN (SELECT topic_id FROM ai.topics WHERE lecture_id = :lecture_id)
"""

_INSERT_TOPIC = """
INSERT INTO ai.topics
    (topic_id, label, description, lecture_id, subject_id, bloom_level, label_embedding)
VALUES
    (:topic_id, :label, :description, :lecture_id, :subject_id, :bloom_level,
     CAST(:label_embedding AS jsonb))
"""

_UPSERT_DEP = """
INSERT INTO ai.topic_dependencies (topic_id, prereq_topic_id, strength, rationale)
VALUES (:topic_id, :prereq_topic_id, CAST(:strength AS ai.dependency_strength), :rationale)
ON CONFLICT (topic_id, prereq_topic_id)
DO UPDATE SET strength = EXCLUDED.strength, rationale = EXCLUDED.rationale
"""


def _sync_url(database_url: str) -> str:
    """Normalise any Postgres URL to the synchronous psycopg (v3) driver, so a
    plain ``postgresql://`` or an async ``+asyncpg`` URL all resolve to psycopg
    rather than the (uninstalled) default psycopg2."""
    for prefix in ("postgresql+asyncpg://", "postgresql://", "postgres://"):
        if database_url.startswith(prefix):
            return "postgresql+psycopg://" + database_url[len(prefix):]
    return database_url


class PostgresChunkStore(ChunkStore):
    """SQLAlchemy Core writer for the ai.* tables."""

    def __init__(self, database_url: str) -> None:
        from sqlalchemy import create_engine

        # Sync engine (offline batch); normalise DATABASE_URL to psycopg (v3).
        self._engine = create_engine(_sync_url(database_url), future=True)

    def delete_lecture(self, lecture_id: str) -> None:
        from sqlalchemy import text

        with self._engine.begin() as conn:
            for stmt in _DELETE_LECTURE:
                conn.execute(text(stmt), {"lecture_id": lecture_id})

    def write_chunks(self, chunks: Sequence[Chunk]) -> list[str]:
        from sqlalchemy import text

        ids: list[str] = []
        rows = []
        for c in chunks:
            c.chunk_id = c.chunk_id or str(uuid.uuid4())
            ids.append(c.chunk_id)
            rows.append(
                {
                    "chunk_id": c.chunk_id,
                    "lecture_id": c.lecture_id,
                    "topic_ids": c.topic_ids or [],
                    "text": c.text,
                    "heading_path": c.heading_path or None,
                    "page_from": c.page_from,
                    "page_to": c.page_to,
                    "language": c.language,
                    "order_index": c.order_index,
                    "version": c.version,
                }
            )
        with self._engine.begin() as conn:
            if rows:
                conn.execute(text(_INSERT_CHUNK), rows)
        return ids

    def write_figures(self, figures: Sequence[Figure]) -> None:
        from sqlalchemy import text

        rows = [
            {
                "lecture_id": f.lecture_id,
                "chunk_id": f.chunk_id,
                "topic_ids": f.topic_ids or [],
                "s3_uri": f.s3_uri,
                "caption": f.caption,
                "alt_text": f.alt_text,
                "page": f.page,
            }
            for f in figures
        ]
        with self._engine.begin() as conn:
            if rows:
                conn.execute(text(_INSERT_FIGURE), rows)

    def write_vector_index(self, chunks: Sequence[Chunk]) -> None:
        from sqlalchemy import text

        rows = [
            {
                "chunk_id": c.chunk_id,
                "lecture_id": c.lecture_id,
                "topic_ids": c.topic_ids or [],
                "subject_id": c.subject_id,
                "grade": c.grade,
                "namespace": c.namespace,
                "language": c.language,
                "version": c.version,
            }
            for c in chunks
        ]
        with self._engine.begin() as conn:
            if rows:
                conn.execute(text(_UPSERT_INDEX), rows)

    def read_topic_catalog(self, subject_id: str, *, include_background: bool = True,
                           ) -> list[tuple[str, str, list[float]]]:
        from sqlalchemy import text

        sql = _SELECT_TOPICS if include_background else _SELECT_SYLLABUS_TOPICS
        with self._engine.begin() as conn:
            result = conn.execute(text(sql), {"subject_id": subject_id})
            out = []
            for tid, label, emb in result:
                if emb:  # skip catalog topics that never got an embedding
                    out.append((str(tid), label, list(emb)))
            return out

    def read_lecture_topics(self, lecture_id: str) -> list[tuple[str, str, list[float]]]:
        from sqlalchemy import text

        with self._engine.begin() as conn:
            result = conn.execute(text(_SELECT_LECTURE_TOPICS), {"lecture_id": lecture_id})
            return [(str(tid), label, list(emb)) for tid, label, emb in result if emb]

    def write_topics(self, topics: Sequence[Topic]) -> None:
        import json

        from sqlalchemy import text

        rows = [
            {
                "topic_id": t.topic_id,
                "label": t.label,
                "description": t.description,
                "lecture_id": t.lecture_id,
                "subject_id": t.subject_id,
                "bloom_level": t.bloom_level,
                "label_embedding": json.dumps(t.label_embedding)
                if t.label_embedding is not None
                else None,
            }
            for t in topics
        ]
        with self._engine.begin() as conn:
            if rows:
                conn.execute(text(_INSERT_TOPIC), rows)

    def delete_lecture_dependencies(self, lecture_id: str) -> None:
        from sqlalchemy import text

        with self._engine.begin() as conn:
            conn.execute(text(_DELETE_LECTURE_DEPS), {"lecture_id": lecture_id})

    def write_dependencies(self, deps: Sequence[Dependency]) -> None:
        from sqlalchemy import text

        rows = [
            {
                "topic_id": d.topic_id,
                "prereq_topic_id": d.prereq_topic_id,
                "strength": d.strength,
                "rationale": d.rationale,
            }
            for d in deps
        ]
        with self._engine.begin() as conn:
            if rows:
                conn.execute(text(_UPSERT_DEP), rows)

    def write_ingestion_record(self, record: dict[str, Any]) -> None:
        from sqlalchemy import text

        with self._engine.begin() as conn:
            conn.execute(text(_UPSERT_INGESTION), record)


class DryRunChunkStore(ChunkStore):
    """No Postgres: assign ids, collect rows in memory (tests)."""

    def __init__(self) -> None:
        self.chunks: list[dict[str, Any]] = []
        self.figures: list[dict[str, Any]] = []
        self.index: list[dict[str, Any]] = []
        self.topics: list[dict[str, Any]] = []
        self.dependencies: list[dict[str, Any]] = []
        self.ingestion_records: list[dict[str, Any]] = []

    def delete_lecture(self, lecture_id: str) -> None:
        self.chunks = [r for r in self.chunks if r.get("lecture_id") != lecture_id]
        self.figures = [r for r in self.figures if r.get("lecture_id") != lecture_id]
        self.index = [r for r in self.index if r.get("lecture_id") != lecture_id]

    def write_chunks(self, chunks: Sequence[Chunk]) -> list[str]:
        ids = []
        for c in chunks:
            c.chunk_id = c.chunk_id or str(uuid.uuid4())
            ids.append(c.chunk_id)
            self.chunks.append(
                {"chunk_id": c.chunk_id, "order_index": c.order_index,
                 "lecture_id": c.lecture_id}
            )
        return ids

    def write_figures(self, figures: Sequence[Figure]) -> None:
        self.figures.extend(
            {"s3_uri": f.s3_uri, "chunk_id": f.chunk_id, "lecture_id": f.lecture_id}
            for f in figures
        )

    def write_vector_index(self, chunks: Sequence[Chunk]) -> None:
        self.index.extend(
            {"chunk_id": c.chunk_id, "namespace": c.namespace, "version": c.version,
             "lecture_id": c.lecture_id}
            for c in chunks
        )

    def read_topic_catalog(self, subject_id: str, *, include_background: bool = True,
                           ) -> list[tuple[str, str, list[float]]]:
        return []  # no persistent catalog in dry-run

    def read_lecture_topics(self, lecture_id: str) -> list[tuple[str, str, list[float]]]:
        return []  # no persistent catalog in dry-run

    def write_topics(self, topics: Sequence[Topic]) -> None:
        self.topics.extend(
            {"topic_id": t.topic_id, "label": t.label} for t in topics
        )

    def delete_lecture_dependencies(self, lecture_id: str) -> None:
        pass  # no persistent edges in dry-run

    def write_dependencies(self, deps: Sequence[Dependency]) -> None:
        self.dependencies.extend(
            {"topic_id": d.topic_id, "prereq_topic_id": d.prereq_topic_id} for d in deps
        )

    def write_ingestion_record(self, record: dict[str, Any]) -> None:
        self.ingestion_records = [
            r for r in self.ingestion_records
            if r.get("lecture_id") != record.get("lecture_id")
        ]
        self.ingestion_records.append(record)


def get_chunk_store(config: Config) -> ChunkStore:
    url = config.secret("DATABASE_URL")
    if url and not url.startswith("postgresql+asyncpg://user:password@localhost"):
        # A real URL (not the untouched .env.example placeholder) -> write to it.
        return PostgresChunkStore(url)
    return DryRunChunkStore()


# ===========================================================================
# Vector store (Qdrant)
# ===========================================================================

class VectorStore(ABC):
    @abstractmethod
    def ensure_collection(self, dimension: int) -> None:
        ...

    @abstractmethod
    def delete_lecture(self, lecture_id: str) -> None:
        """Delete all points for a lecture so a re-ingest replaces them."""

    @abstractmethod
    def upsert(self, chunks: Sequence[Chunk]) -> int:
        """Upsert one point per chunk (id=chunk_id, vector, filter payload)."""

    @abstractmethod
    def ensure_topics_collection(self, dimension: int) -> None:
        """Create the topics collection (separate from chunks) if absent."""

    @abstractmethod
    def upsert_topics(self, topics: Sequence[Any]) -> int:
        """Upsert one point per topic (id=topic_id, label vector, subject/lecture
        payload) so ``resolve_topics`` can do an indexed top-k instead of a Python
        brute-force over ``ai.topics``."""


def _payload(c: Chunk) -> dict[str, Any]:
    # Query-filter fields, plus readable citation labels (not filtered on, just
    # returned with the hit so the agent can render Subject · Lesson N: Title).
    payload = {
        "namespace": c.namespace,
        "subject_id": c.subject_id,
        "grade": c.grade,
        "language": c.language,
        "topic_ids": c.topic_ids or [],
        "lecture_id": c.lecture_id,
        "version": c.version,
        "subject_name": c.subject_name,
        "lesson_title": c.lesson_title,
        "lesson_order": c.lesson_order,
        # Navigation, not material. A contents page is a list of lesson titles
        # and so outmatches the lessons themselves on any title-shaped question.
        "chunk_kind": chunk_kind(c.text),
    }
    return {k: v for k, v in payload.items() if v is not None}


def _topic_payload(t: Any) -> dict[str, Any]:
    # label rides along so resolve_topics returns it without a Postgres hydrate.
    payload = {"topic_id": t.topic_id, "label": t.label, "subject_id": t.subject_id,
               "lecture_id": t.lecture_id, "namespace": t.namespace}
    return {k: v for k, v in payload.items() if v is not None}


DEFAULT_UPSERT_BATCH = 64




class QdrantVectorStore(VectorStore):
    def __init__(self, config: Config, location: str) -> None:
        vs = config.vector_store
        self._collection = vs.get("collection", "eduai_chunks")
        self._topics_collection = vs.get("topics_collection", "eduai_topics")
        self._distance = vs.get("distance", "cosine")
        self._hnsw = vs.get("hnsw", {})
        self._indexed_fields = vs.get("indexed_payload_fields", [])
        # A whole lesson in one request times out on a small cluster, so points
        # go up in batches with an explicit budget per request.
        self._batch_size = max(1, int(vs.get("upsert_batch_size", DEFAULT_UPSERT_BATCH)))
        # Named vectors are a migration, not a switch: every point in an
        # unnamed collection stops being found the moment a query names one.
        self._hybrid = bool(vs.get("hybrid", False))
        self._client = upsert_client(config, location)

    def _upsert_batched(self, collection: str, points: list) -> int:
        """Upsert in bounded batches; a partial failure leaves earlier batches in place."""
        for start in range(0, len(points), self._batch_size):
            batch = points[start : start + self._batch_size]
            self._client.upsert(collection_name=collection, points=batch, wait=True)
            logger.debug(
                "qdrant upsert batch",
                extra={"collection": collection, "points": len(batch),
                       "done": min(start + self._batch_size, len(points)),
                       "total": len(points)},
            )
        return len(points)

    def ensure_collection(self, dimension: int) -> None:
        from qdrant_client import models

        distance = {
            "cosine": models.Distance.COSINE,
            "dot": models.Distance.DOT,
            "euclid": models.Distance.EUCLID,
        }[self._distance]
        if not self._client.collection_exists(self._collection):
            dense = models.VectorParams(size=dimension, distance=distance)
            hnsw = models.HnswConfigDiff(
                m=self._hnsw.get("m", 16),
                ef_construct=self._hnsw.get("ef_construct", 128),
            )
            if self._hybrid:
                self._client.create_collection(
                    collection_name=self._collection,
                    vectors_config={DENSE_VECTOR: dense},
                    sparse_vectors_config={
                        SPARSE_VECTOR: models.SparseVectorParams()},
                    hnsw_config=hnsw,
                )
            else:
                self._client.create_collection(
                    collection_name=self._collection,
                    vectors_config=dense,
                    hnsw_config=hnsw,
                )
        # Ensure every filter field is indexed on each call, not only at creation:
        # a filtered delete or query on an unindexed field fails with 400, and a
        # collection created before a field was added to indexed_payload_fields
        # would otherwise never get it. Only missing indexes are created.
        existing = set(
            (self._client.get_collection(self._collection).payload_schema or {}).keys())
        for field in self._indexed_fields:
            if field in existing:
                continue
            schema = "integer" if field == "version" else "keyword"
            try:
                self._client.create_payload_index(
                    self._collection, field_name=field, field_schema=schema
                )
            except Exception:  # pragma: no cover - concurrent create / transient
                pass

    def delete_lecture(self, lecture_id: str) -> None:
        from qdrant_client import models

        if not self._client.collection_exists(self._collection):
            return
        self._client.delete(
            collection_name=self._collection,
            points_selector=models.Filter(must=[models.FieldCondition(
                key="lecture_id", match=models.MatchValue(value=lecture_id))]),
        )

    def upsert(self, chunks: Sequence[Chunk]) -> int:
        from qdrant_client import models

        points = [
            models.PointStruct(id=c.chunk_id, vector=self._vector(c, models),
                               payload=_payload(c))
            for c in chunks
            if c.embedding is not None and c.chunk_id
        ]
        return self._upsert_batched(self._collection, points)

    def _vector(self, chunk: Chunk, models) -> Any:
        """What goes on the point: one unnamed vector, or the named pair.

        A chunk of pure notation has no words to match on and its sparse half is
        empty. It is written anyway — findable by meaning rather than dropped.
        """
        if not self._hybrid:
            return chunk.embedding
        terms = chunk.sparse or {}
        return {
            DENSE_VECTOR: chunk.embedding,
            SPARSE_VECTOR: models.SparseVector(
                indices=list(terms.keys()), values=list(terms.values())),
        }

    def ensure_topics_collection(self, dimension: int) -> None:
        from qdrant_client import models

        distance = {
            "cosine": models.Distance.COSINE,
            "dot": models.Distance.DOT,
            "euclid": models.Distance.EUCLID,
        }[self._distance]
        if not self._client.collection_exists(self._topics_collection):
            self._client.create_collection(
                collection_name=self._topics_collection,
                vectors_config=models.VectorParams(size=dimension, distance=distance),
                hnsw_config=models.HnswConfigDiff(
                    m=self._hnsw.get("m", 16),
                    ef_construct=self._hnsw.get("ef_construct", 128),
                ),
            )
        existing = set(
            (self._client.get_collection(self._topics_collection).payload_schema or {}).keys())
        for field in ("subject_id", "lecture_id"):  # filtered on in resolve_topics
            if field in existing:
                continue
            try:
                self._client.create_payload_index(
                    self._topics_collection, field_name=field, field_schema="keyword")
            except Exception:  # pragma: no cover - concurrent create / transient
                pass

    def upsert_topics(self, topics: Sequence[Any]) -> int:
        from qdrant_client import models

        points = [
            models.PointStruct(id=t.topic_id, vector=t.label_embedding,
                               payload=_topic_payload(t))
            for t in topics
            if t.label_embedding is not None and t.topic_id
        ]
        return self._upsert_batched(self._topics_collection, points)


class DryRunVectorStore(VectorStore):
    """No Qdrant: collect points in memory (tests)."""

    def __init__(self) -> None:
        self.points: list[dict[str, Any]] = []
        self.topic_points: list[dict[str, Any]] = []
        self.dimension: int | None = None

    def ensure_collection(self, dimension: int) -> None:
        self.dimension = dimension

    def ensure_topics_collection(self, dimension: int) -> None:
        self.dimension = dimension

    def delete_lecture(self, lecture_id: str) -> None:
        self.points = [p for p in self.points
                       if p["payload"].get("lecture_id") != lecture_id]

    def upsert_topics(self, topics: Sequence[Any]) -> int:
        n = 0
        for t in topics:
            if t.label_embedding is None or not t.topic_id:
                continue
            self.topic_points.append({"id": t.topic_id, "payload": _topic_payload(t)})
            n += 1
        return n

    def upsert(self, chunks: Sequence[Chunk]) -> int:
        n = 0
        for c in chunks:
            if c.embedding is None or not c.chunk_id:
                continue
            self.points.append({"id": c.chunk_id, "payload": _payload(c)})
            n += 1
        return n


def get_vector_store(config: Config) -> VectorStore:
    url = config.secret("QDRANT_URL", "http://localhost:6333")
    if url in ("dryrun", "none", "", None):
        return DryRunVectorStore()
    try:
        return QdrantVectorStore(config, location=url)
    except ImportError:
        return DryRunVectorStore()
