"""Publish the ministry lessons ingested before ingestion published them itself.

Ingestion left every lesson of a book a draft, and students only see published
lessons. This publishes what ingestion would publish now: a ministry lesson that
is still a draft and finished ready, not one that needs review.

    python -m scripts.publish_ready_lessons --subject <id> [--subject <id> ...]           # dry run
    python -m scripts.publish_ready_lessons --subject <id> [--subject <id> ...] --apply
"""

from __future__ import annotations

import argparse

from shared.config import get_config
from shared.namespaces import SYLLABUS
from shared.postgres import connect_once

_READY = """
FROM content.lessons l JOIN ai.lecture_ingestion i ON i.lecture_id = l.id
WHERE l.subject_id = ANY(%(subjects)s::uuid[]) AND l.source = %(shown_source)s
  AND l.status = 'draft' AND l.ingestion_status = 'ready' AND i.status = 'ready'
"""
_LIST = ("SELECT l.subject_id::text, l.id::text, left(l.title, 70) " + _READY
         + "ORDER BY l.subject_id, l.order_index")
_PUBLISH = ("UPDATE content.lessons SET status = 'published', published_at = now(), "
            "updated_at = now() WHERE id IN (SELECT l.id " + _READY + ")")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--subject", action="append", required=True, dest="subjects")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)

    config = get_config()
    params = {"subjects": args.subjects,
              "shown_source": config.get("ingestion.books.lesson_source")[SYLLABUS]}
    with connect_once(config) as conn, conn.cursor() as cur:
        cur.execute(_LIST, params)
        rows = cur.fetchall()
        for subject_id, lesson_id, title in rows:
            print(f"{subject_id}  {lesson_id}  {title}")
        print(f"{len(rows)} lessons to publish")
        if not args.apply:
            print("dry run: nothing changed; pass --apply to publish")
            return 0
        cur.execute(_PUBLISH, params)
        published = cur.rowcount
        conn.commit()
    print(f"published {published} lessons")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
