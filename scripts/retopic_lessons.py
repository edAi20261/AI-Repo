"""Find the topics of lessons again, from the passages they already have.

A book read whole had its topics found once, for the whole book, and when it
was split into lessons the passages were filed under each lesson but the topics
were not: every lesson of a subject kept the first lessons' topics. "Lesson Two
Chemical Equations" was tagged with states of matter and heating curves, and a
quiz on it asked about those.

Nothing is read or embedded again. ``plan`` runs the topics and prerequisites
stages on each lesson's stored passages and vectors and writes what it found to
a file, touching no store; ``apply`` writes exactly that file, so what is applied
is what was reviewed.

    python -m scripts.retopic_lessons plan --lessons ids.txt --out plan.json
    python -m scripts.retopic_lessons apply --plan plan.json            # dry run
    python -m scripts.retopic_lessons apply --plan plan.json --apply    # writes a backup first
    python -m scripts.retopic_lessons restore --backup BACKUP.json.gz --apply

``plan`` needs the embedder and the topic model, so it runs where ingestion runs
(.github/workflows/retopic.yml). The topics a lesson carries today are never
deleted: students' mastery, quiz and homework questions point at them.
"""

from __future__ import annotations

import argparse
import datetime
import gzip
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Sequence

from shared.config import get_config
from shared.namespaces import NOT_BACKGROUND_TOPIC, SYLLABUS
from shared.postgres import connect_once

_LESSONS = """
SELECT l.id::text, l.subject_id::text, l.title, l.order_index,
       (SELECT v.namespace FROM ai.vector_index v WHERE v.lecture_id = l.id LIMIT 1)
FROM content.lessons l WHERE l.id = ANY(%s::uuid[]) AND l.status <> 'archived'
ORDER BY l.subject_id, l.order_index
"""
_PASSAGES = """
SELECT chunk_id::text, text, heading_path, page_from, page_to, language, order_index, version
FROM ai.chunks WHERE lecture_id = %s ORDER BY order_index
"""
_OWNED = "SELECT topic_id::text, label, label_embedding FROM ai.topics WHERE lecture_id = %s"
# A topic owned by an archived lesson would hand the mastery a student earns
# here to a lesson nobody can open.
_CATALOG = """
SELECT topic_id::text, label, label_embedding FROM ai.topics
WHERE subject_id = %s AND label_embedding IS NOT NULL
  AND lecture_id NOT IN (SELECT id FROM content.lessons WHERE status = 'archived')
"""
_OWNERS = """
SELECT topic_id::text, lecture_id::text FROM ai.topics
WHERE topic_id = ANY(%s::uuid[]) AND lecture_id IS NOT NULL
"""

_TAGS = "SELECT chunk_id::text, topic_ids::text[] FROM ai.chunks WHERE lecture_id = ANY(%s::uuid[])"
_INDEX_TAGS = ("SELECT chunk_id::text, namespace, topic_ids::text[] FROM ai.vector_index "
               "WHERE lecture_id = ANY(%s::uuid[])")
_FIGURE_TAGS = "SELECT figure_id::text, topic_ids::text[] FROM ai.figures WHERE lecture_id = ANY(%s::uuid[])"
_COUNTS = "SELECT lecture_id::text, topic_count FROM ai.lecture_ingestion WHERE lecture_id = ANY(%s::uuid[])"

_INSERT_TOPIC = """
INSERT INTO ai.topics (topic_id, label, description, lecture_id, subject_id, bloom_level, label_embedding)
VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb) ON CONFLICT (topic_id) DO NOTHING
"""
_TAG_PASSAGE = "UPDATE ai.chunks SET topic_ids = %s::uuid[] WHERE chunk_id = %s::uuid"
_TAG_INDEX = """
UPDATE ai.vector_index v SET topic_ids = c.topic_ids FROM ai.chunks c
WHERE v.chunk_id = c.chunk_id AND c.lecture_id = ANY(%s::uuid[])
"""
_TAG_FIGURES = """
UPDATE ai.figures f SET topic_ids = c.topic_ids FROM ai.chunks c
WHERE f.chunk_id = c.chunk_id AND f.lecture_id = ANY(%s::uuid[])
"""
_DROP_EDGES = """
DELETE FROM ai.topic_dependencies
WHERE topic_id IN (SELECT topic_id FROM ai.topics WHERE lecture_id = %s)
"""
_EDGE = """
INSERT INTO ai.topic_dependencies (topic_id, prereq_topic_id, strength, rationale)
VALUES (%s, %s, CAST(%s AS ai.dependency_strength), %s)
ON CONFLICT (topic_id, prereq_topic_id) DO UPDATE SET strength = EXCLUDED.strength
"""
_COUNT = "UPDATE ai.lecture_ingestion SET topic_count = %s WHERE lecture_id = %s"
_RESTORE_PASSAGE = "UPDATE ai.chunks SET topic_ids = %s::uuid[] WHERE chunk_id = %s::uuid"
_RESTORE_INDEX = ("UPDATE ai.vector_index SET topic_ids = %s::uuid[] "
                  "WHERE chunk_id = %s::uuid AND namespace = %s")
_RESTORE_FIGURE = "UPDATE ai.figures SET topic_ids = %s::uuid[] WHERE figure_id = %s::uuid"

_PAYLOAD_BATCH = 500


class PlanStore:
    """What the topics and prerequisites stages read and write, during a plan.

    Reads come from the database, with one change: a lesson's own topics are
    the ones it owns, not the ones its passages carry, since what they carry is
    exactly what is wrong. Writes are kept, never sent."""

    def __init__(self, read: Callable[[str, tuple], list[tuple]]) -> None:
        self._read = read
        self.planned: list[Any] = []
        self.edges: list[Any] = []

    def _planned(self, subject_id: str) -> list[tuple[str, str, list[float]]]:
        return [(t.topic_id, t.label, t.label_embedding) for t in self.planned
                if t.subject_id == subject_id]

    def read_lecture_topics(self, lecture_id: str) -> list[tuple[str, str, list[float]]]:
        return [(tid, label, list(emb)) for tid, label, emb in self._read(_OWNED, (lecture_id,)) if emb]

    def read_topic_catalog(self, subject_id: str, *, include_background: bool = True,
                           ) -> list[tuple[str, str, list[float]]]:
        sql = _CATALOG if include_background else _CATALOG + " AND " + NOT_BACKGROUND_TOPIC
        stored = [(tid, label, list(emb)) for tid, label, emb in self._read(sql, (subject_id,))]
        return stored + self._planned(subject_id)

    def read_topic_owners(self, topic_ids: Sequence[str]) -> dict[str, str]:
        owners = dict(self._read(_OWNERS, (list(topic_ids),))) if topic_ids else {}
        owners.update({t.topic_id: t.lecture_id for t in self.planned if t.topic_id in topic_ids})
        return owners

    def write_topics(self, topics: Sequence[Any]) -> None:
        self.planned += list(topics)

    def delete_lecture_dependencies(self, lecture_id: str) -> None:
        pass

    def write_dependencies(self, deps: Sequence[Any]) -> None:
        self.edges += list(deps)


class PlanVectors:
    """Topic points are written by ``apply``, from the plan."""

    def ensure_topics_collection(self, dimension: int) -> None:
        pass

    def upsert_topics(self, topics: Sequence[Any]) -> int:
        return 0


def _vectors(config, chunk_ids: list[str]) -> dict[str, list[float]]:
    from shared.vector_store import query_client
    from shared.vectors import DENSE_VECTOR

    client = query_client(config, timeout=120)
    found = {}
    for start in range(0, len(chunk_ids), 256):
        for point in client.retrieve(config.vector_store.get("collection"),
                                     ids=chunk_ids[start:start + 256], with_vectors=True):
            vector = point.vector
            found[str(point.id)] = vector[DENSE_VECTOR] if isinstance(vector, dict) else vector
    return found


def plan_lesson(lesson: tuple, passages: list[tuple], vectors: dict[str, list[float]],
                ctx: Any, store: PlanStore) -> dict:
    """One lesson through the topics and prerequisites stages, on what is stored."""
    from ingestion.stages.dependencies import run as derive_prerequisites
    from ingestion.stages.topics import run as find_topics
    from ingestion.state import Chunk

    lecture_id, subject_id, title, _order, namespace = lesson
    chunks = [Chunk(lecture_id=lecture_id, text=text, heading_path=heading or "",
                    page_from=page_from, page_to=page_to, language=language or "",
                    order_index=order, version=version, chunk_id=chunk_id,
                    embedding=vectors.get(chunk_id), namespace=namespace or SYLLABUS,
                    subject_id=subject_id)
              for chunk_id, text, heading, page_from, page_to, language, order, version in passages]
    state = {"lecture_id": lecture_id, "subject_id": subject_id,
             "namespace": namespace or SYLLABUS, "chunks": chunks}
    edges_before = len(store.edges)
    state = find_topics(state, ctx)
    state = derive_prerequisites(state, ctx)
    failed = state.get("ingestion_status") == "needs_review" or not state.get("topics")
    return {
        "lecture_id": lecture_id, "subject_id": subject_id, "title": title,
        "namespace": namespace or SYLLABUS,
        "status": "failed" if failed else "planned",
        "reason": "; ".join(state.get("errors", [])),
        "topic_count": len(state.get("topics", [])),
        "labels": [t["label"] for t in state.get("topics", [])],
        "passages": {c.chunk_id: c.topic_ids for c in chunks},
        "edges": [{"topic_id": e.topic_id, "prereq_topic_id": e.prereq_topic_id,
                   "strength": e.strength, "rationale": e.rationale}
                  for e in store.edges[edges_before:]],
    }


def plan(config, lesson_ids: list[str], out: Path) -> int:
    from ingestion.context import PipelineContext

    with connect_once(config) as conn, conn.cursor() as cur:
        def read(sql: str, params: tuple) -> list[tuple]:
            cur.execute(sql, params)
            return cur.fetchall()

        lessons = read(_LESSONS, (lesson_ids,))
        store = PlanStore(read)
        ctx = PipelineContext(config=config, vision=None, uploader=None,
                              _chunk_store=store, _vector_store=PlanVectors())
        planned = []
        for n, lesson in enumerate(lessons, 1):
            passages = read(_PASSAGES, (lesson[0],))
            vectors = _vectors(config, [p[0] for p in passages])
            try:
                planned.append(plan_lesson(lesson, passages, vectors, ctx, store))
            except Exception as error:
                planned.append({"lecture_id": lesson[0], "title": lesson[2], "status": "failed",
                                "reason": f"{type(error).__name__}: {error}"})
            last = planned[-1]
            print(f"{n}/{len(lessons)} {last['status']:<8} {lesson[2][:60]} | "
                  f"{', '.join(last.get('labels', []))[:120] or last['reason'][:120]}", flush=True)
    topics = [{"topic_id": t.topic_id, "label": t.label, "description": t.description,
               "lecture_id": t.lecture_id, "subject_id": t.subject_id,
               "bloom_level": t.bloom_level, "label_embedding": t.label_embedding,
               "namespace": t.namespace} for t in store.planned]
    out.write_text(json.dumps({"lessons": planned, "topics": topics}, ensure_ascii=False))
    failed = [p for p in planned if p["status"] != "planned"]
    print(f"planned {len(planned) - len(failed)} lessons, {len(topics)} new topics; "
          f"{len(failed)} failed → {out}")
    return 1 if failed and len(failed) == len(planned) else 0


def payload_groups(passages: dict[str, list[str]]) -> dict[tuple, list[str]]:
    """Passages grouped by the topics they carry: one payload write per group."""
    groups: dict[tuple, list[str]] = defaultdict(list)
    for chunk_id, topic_ids in passages.items():
        groups[tuple(topic_ids)].append(chunk_id)
    return groups


def _write_payloads(config, passages: dict[str, list[str]]) -> None:
    from shared.vector_store import upsert_client

    client = upsert_client(config, config.secret("QDRANT_URL"))
    collection = config.vector_store.get("collection")
    for topic_ids, chunk_ids in payload_groups(passages).items():
        for start in range(0, len(chunk_ids), _PAYLOAD_BATCH):
            client.set_payload(collection, payload={"topic_ids": list(topic_ids)},
                               points=chunk_ids[start:start + _PAYLOAD_BATCH], wait=True)


def apply(config, plan_path: Path, write: bool) -> int:
    from ingestion.state import Topic
    from ingestion.stores import QdrantVectorStore

    document = json.loads(plan_path.read_text())
    lessons = [p for p in document["lessons"] if p["status"] == "planned"]
    ids = [p["lecture_id"] for p in lessons]
    owners = {p["lecture_id"] for p in lessons}
    topics = [t for t in document["topics"] if t["lecture_id"] in owners]
    passages = {cid: tids for p in lessons for cid, tids in p["passages"].items()}
    print(f"{len(lessons)} lessons to retag ({len(document['lessons']) - len(lessons)} failed in the "
          f"plan, left as they are), {len(passages)} passages, {len(topics)} new topics, "
          f"{sum(len(p['edges']) for p in lessons)} prerequisite edges")
    if not write:
        print("dry run: nothing changed; pass --apply")
        return 0

    with connect_once(config) as conn, conn.cursor() as cur:
        backup = {}
        for name, sql in (("passages", _TAGS), ("index", _INDEX_TAGS), ("figures", _FIGURE_TAGS),
                          ("counts", _COUNTS)):
            cur.execute(sql, (ids,))
            backup[name] = cur.fetchall()
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = Path(f"retopic-backup-{stamp}.json.gz")
    with gzip.open(path, "wt") as out:
        json.dump(backup, out)
    print("backup:", path)

    with connect_once(config) as conn, conn.cursor() as cur:
        cur.executemany(_INSERT_TOPIC, [
            (t["topic_id"], t["label"], t["description"], t["lecture_id"], t["subject_id"],
             t["bloom_level"], json.dumps(t["label_embedding"])) for t in topics])
        cur.executemany(_TAG_PASSAGE, [(tids, cid) for cid, tids in passages.items()])
        cur.execute(_TAG_INDEX, (ids,))
        cur.execute(_TAG_FIGURES, (ids,))
        for lesson in lessons:
            cur.execute(_DROP_EDGES, (lesson["lecture_id"],))
            cur.executemany(_EDGE, [(e["topic_id"], e["prereq_topic_id"], e["strength"],
                                     e["rationale"]) for e in lesson["edges"]])
            cur.execute(_COUNT, (lesson["topic_count"], lesson["lecture_id"]))
        conn.commit()
    _write_payloads(config, passages)
    QdrantVectorStore(config, config.secret("QDRANT_URL")).upsert_topics(
        [Topic(**{k: t[k] for k in ("label", "subject_id", "lecture_id", "description",
                                    "bloom_level", "topic_id", "label_embedding", "namespace")})
         for t in topics])
    print(f"retagged {len(passages)} passages of {len(lessons)} lessons; {len(topics)} topics added")
    return 0


def restore(config, backup_path: Path, write: bool) -> int:
    with gzip.open(backup_path, "rt") as source:
        backup = json.load(source)
    print(f"{len(backup['passages'])} passages and {len(backup['counts'])} lessons to put back")
    if not write:
        print("dry run: nothing changed; pass --apply")
        return 0
    with connect_once(config) as conn, conn.cursor() as cur:
        cur.executemany(_RESTORE_PASSAGE, [(tids or [], cid) for cid, tids in backup["passages"]])
        cur.executemany(_RESTORE_INDEX, [(tids or [], cid, ns) for cid, ns, tids in backup["index"]])
        cur.executemany(_RESTORE_FIGURE, [(tids or [], fid) for fid, tids in backup["figures"]])
        cur.executemany(_COUNT, [(count, lid) for lid, count in backup["counts"]])
        conn.commit()
    _write_payloads(config, {cid: tids or [] for cid, tids in backup["passages"]})
    print("restored")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    planning = sub.add_parser("plan")
    planning.add_argument("--lessons", type=Path, required=True, help="one lesson id per line")
    planning.add_argument("--out", type=Path, required=True)
    applying = sub.add_parser("apply")
    applying.add_argument("--plan", type=Path, required=True)
    applying.add_argument("--apply", action="store_true")
    restoring = sub.add_parser("restore")
    restoring.add_argument("--backup", type=Path, required=True)
    restoring.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)

    config = get_config()
    if args.command == "plan":
        ids = [line.strip() for line in args.lessons.read_text().replace(",", "\n").splitlines()
               if line.strip()]
        return plan(config, ids, args.out)
    if args.command == "apply":
        return apply(config, args.plan, args.apply)
    return restore(config, args.backup, args.apply)


if __name__ == "__main__":
    raise SystemExit(main())
