"""File books that were ingested as syllabus as background instead.

Nothing is re-read or re-embedded: the chunks, topics and lessons stay where
they are and only their labels change, in every place a label lives:

- Qdrant chunk and topic points get ``namespace = external``
- ``ai.vector_index`` rows get ``namespace = external``
- ``content.lessons`` rows get ``source = ai_generated`` and lose their author,
  so no teacher sees them as their own drafts

    python -m scripts.relabel_background --lessons lessons.txt            # dry run
    python -m scripts.relabel_background --lessons lessons.txt --apply

``lessons.txt`` holds one lesson id (``content.lessons.id``) per line.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from shared.config import get_config
from shared.namespaces import BACKGROUND
from shared.postgres import connect_once

_LESSONS = """
SELECT l.id::text, l.source, l.created_by IS NOT NULL, left(l.title, 60),
       (SELECT count(*) FROM ai.vector_index v
         WHERE v.lecture_id = l.id AND v.namespace <> %(background)s)
FROM content.lessons l WHERE l.id = ANY(%(ids)s::uuid[]) ORDER BY l.order_index
"""
_VECTOR_INDEX = ("UPDATE ai.vector_index SET namespace = %(background)s "
                 "WHERE lecture_id = ANY(%(ids)s::uuid[]) AND namespace <> %(background)s")
_LESSON_ROWS = ("UPDATE content.lessons SET source = 'ai_generated', created_by = NULL, "
                "updated_at = now() WHERE id = ANY(%(ids)s::uuid[])")


def _qdrant(config, ids: list[str], apply: bool) -> dict[str, int]:
    from qdrant_client import models

    from shared.vector_store import query_client

    client = query_client(config)
    scope = models.Filter(must=[models.FieldCondition(
        key="lecture_id", match=models.MatchAny(any=ids))])
    counts = {}
    for collection in (config.vector_store.get("collection"),
                       config.vector_store.get("topics_collection")):
        counts[collection] = client.count(collection, count_filter=scope, exact=True).count
        if apply and counts[collection]:
            client.set_payload(collection, payload={"namespace": BACKGROUND},
                               points=scope, wait=True)
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lessons", required=True, type=Path)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)

    ids = [line.strip() for line in args.lessons.read_text().splitlines() if line.strip()]
    config = get_config()
    params = {"ids": ids, "background": BACKGROUND}
    with connect_once(config) as conn, conn.cursor() as cur:
        cur.execute(_LESSONS, params)
        rows = cur.fetchall()
    missing = set(ids) - {row[0] for row in rows}
    for lesson_id, source, authored, title, unlabelled in rows:
        print(f"{lesson_id}  {source:<17} {'author ' if authored else '       '}"
              f"{unlabelled:>5} to relabel  {title}")
    if missing:
        print(f"{len(missing)} id(s) are not lessons: {sorted(missing)}")
        return 1
    print(f"{len(rows)} lessons, {sum(r[4] for r in rows)} vector_index rows to relabel")
    print("qdrant points:", _qdrant(config, ids, apply=False))
    if not args.apply:
        print("dry run: nothing changed; pass --apply to relabel")
        return 0

    with connect_once(config) as conn, conn.cursor() as cur:
        cur.execute(_VECTOR_INDEX, params)
        relabelled = cur.rowcount
        cur.execute(_LESSON_ROWS, params)
        lessons = cur.rowcount
        conn.commit()
    print(f"relabelled {relabelled} vector_index rows and {lessons} lessons")
    print("qdrant points relabelled:", _qdrant(config, ids, apply=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
