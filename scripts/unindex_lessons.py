"""Take archived lessons, and points with no passage behind them, out of search.

An archived lesson keeps its passages, topics and index entries — it can be
shown again — but nothing searched may find it. Its points and ``ai.vector_index``
rows move to the ``archived`` namespace, which no search reads; the namespace
each one had is kept in the backup, so ``--restore`` puts them back.

    python -m scripts.unindex_lessons                         # dry run
    python -m scripts.unindex_lessons --apply                 # writes a backup first
    python -m scripts.unindex_lessons --restore BACKUP.json.gz --apply

Without ``--lessons`` it takes every archived lesson that still has a point in a
searched namespace, and every point whose passage no longer exists (they were
left by test lessons deleted from the database).
"""

from __future__ import annotations

import argparse
import datetime
import gzip
import json
from collections import defaultdict
from pathlib import Path

from shared.config import get_config
from shared.namespaces import ARCHIVED, BACKGROUND, SYLLABUS
from shared.postgres import connect_once

SEARCHED = [SYLLABUS, BACKGROUND]
_LESSONS_PER_WRITE = 25

_ARCHIVED_IN_SEARCH = """
SELECT DISTINCT v.lecture_id::text FROM ai.vector_index v
JOIN content.lessons l ON l.id = v.lecture_id
WHERE l.status = 'archived' AND v.namespace = ANY(%s)
"""
_ROWS = """
SELECT v.chunk_id::text, v.lecture_id::text, v.namespace FROM ai.vector_index v
WHERE v.lecture_id = ANY(%s::uuid[]) AND v.namespace = ANY(%s)
"""
_NOT_ARCHIVED = "SELECT id::text FROM content.lessons WHERE id = ANY(%s::uuid[]) AND status <> 'archived'"
_MOVE = """
UPDATE ai.vector_index SET namespace = %s
WHERE lecture_id = ANY(%s::uuid[]) AND namespace = ANY(%s)
"""
_RESTORE = "UPDATE ai.vector_index SET namespace = %s WHERE chunk_id = %s::uuid AND namespace = %s"
_PASSAGE_IDS = "SELECT chunk_id::text FROM ai.chunks"


def by_namespace(rows: list[tuple]) -> dict[str, list[str]]:
    """Lessons grouped by the namespace their rows had, from (chunk, lesson, namespace)."""
    grouped: dict[str, set] = defaultdict(set)
    for _chunk, lesson, namespace in rows:
        grouped[namespace].add(lesson)
    return {namespace: sorted(lessons) for namespace, lessons in grouped.items()}


def as_json(vector):
    """A point's vectors as JSON: the sparse half is an object, not a list."""
    if not isinstance(vector, dict):
        return vector
    return {name: (value if isinstance(value, list) else
                   {"indices": list(value.indices), "values": list(value.values)})
            for name, value in vector.items()}


def orphans(point_ids: set[str], passage_ids: set[str]) -> list[str]:
    return sorted(point_ids - passage_ids)


def _client(config):
    from shared.vector_store import upsert_client

    return upsert_client(config, config.secret("QDRANT_URL"))


def _lesson_filter(lessons: list[str], namespaces: list[str]):
    from qdrant_client import models

    return models.Filter(must=[
        models.FieldCondition(key="lecture_id", match=models.MatchAny(any=lessons)),
        models.FieldCondition(key="namespace", match=models.MatchAny(any=namespaces))])


def _collections(config) -> list[str]:
    return [config.vector_store.get("collection"), config.vector_store.get("topics_collection")]


def _point_ids(client, collection: str) -> set[str]:
    ids, offset = set(), None
    while True:
        batch, offset = client.scroll(collection, limit=10000, offset=offset,
                                      with_payload=False, with_vectors=False)
        ids.update(str(p.id) for p in batch)
        if offset is None:
            return ids


def _move_points(client, config, lessons: list[str], namespaces: list[str], target: str) -> None:
    for collection in _collections(config):
        for start in range(0, len(lessons), _LESSONS_PER_WRITE):
            client.set_payload(collection, payload={"namespace": target},
                               points=_lesson_filter(lessons[start:start + _LESSONS_PER_WRITE], namespaces),
                               wait=True)


def unindex(config, lessons: list[str] | None, apply: bool) -> int:
    client = _client(config)
    with connect_once(config) as conn, conn.cursor() as cur:
        if lessons is None:
            cur.execute(_ARCHIVED_IN_SEARCH, (SEARCHED,))
            lessons = [r[0] for r in cur.fetchall()]
        cur.execute(_NOT_ARCHIVED, (lessons,))
        live = [r[0] for r in cur.fetchall()]
        cur.execute(_ROWS, (lessons, SEARCHED))
        rows = cur.fetchall()
        cur.execute(_PASSAGE_IDS)
        passages = {r[0] for r in cur.fetchall()}
    if live:
        print(f"refused: {len(live)} of these lessons are not archived: {live[:5]}")
        return 1
    stray = orphans(_point_ids(client, config.vector_store.get("collection")), passages)
    print(f"{len(lessons)} archived lessons, {len(rows)} index rows to move to '{ARCHIVED}'")
    for namespace, ids in by_namespace(rows).items():
        print(f"  from {namespace}: {len(ids)} lessons")
    print(f"{len(stray)} points with no passage: {stray[:5]}")
    if not apply:
        print("dry run: nothing changed; pass --apply")
        return 0

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = Path(f"unindex-backup-{stamp}.json.gz")
    stray_points = client.retrieve(config.vector_store.get("collection"), ids=stray,
                                   with_payload=True, with_vectors=True) if stray else []
    with gzip.open(backup, "wt") as out:
        json.dump({"rows": rows, "orphans": [
            {"id": str(p.id), "payload": p.payload, "vector": as_json(p.vector)} for p in stray_points]}, out)
    print("backup:", backup)

    with connect_once(config) as conn, conn.cursor() as cur:
        cur.execute(_MOVE, (ARCHIVED, lessons, SEARCHED))
        moved = cur.rowcount
        conn.commit()
    _move_points(client, config, lessons, SEARCHED, ARCHIVED)
    if stray:
        client.delete(config.vector_store.get("collection"), points_selector=stray, wait=True)
    print(f"moved {moved} index rows and their points; deleted {len(stray)} points with no passage")
    return 0


def restore(config, backup: Path, apply: bool) -> int:
    with gzip.open(backup, "rt") as source:
        rows = json.load(source)["rows"]
    grouped = by_namespace([tuple(r) for r in rows])
    print(f"{len(rows)} index rows to put back: "
          + ", ".join(f"{len(v)} lessons to {k}" for k, v in grouped.items()))
    if not apply:
        print("dry run: nothing changed; pass --apply")
        return 0
    with connect_once(config) as conn, conn.cursor() as cur:
        for chunk, _lesson, namespace in rows:
            cur.execute(_RESTORE, (namespace, chunk, ARCHIVED))
        conn.commit()
    client = _client(config)
    for namespace, lessons in grouped.items():
        _move_points(client, config, lessons, [ARCHIVED], namespace)
    print("restored")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lessons", type=Path, help="one archived lesson id per line")
    parser.add_argument("--restore", type=Path, help="a backup this script wrote")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)

    config = get_config()
    if args.restore:
        return restore(config, args.restore, args.apply)
    lessons = ([line.strip() for line in args.lessons.read_text().splitlines() if line.strip()]
               if args.lessons else None)
    return unindex(config, lessons, args.apply)


if __name__ == "__main__":
    raise SystemExit(main())
