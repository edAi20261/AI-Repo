"""Number a subject's shown lessons from the first again, in the order they have.

A student sees each lesson's place in its subject. Lessons were numbered after
every lesson already in the subject, background ones too, so the textbook of a
subject that had twelve other books ingested first read "Lesson 583" onwards.
Background and archived lessons are not shown and are not numbered.

    python -m scripts.renumber_lessons --subject <id> [--subject <id> ...]          # dry run
    python -m scripts.renumber_lessons --subject <id> [--subject <id> ...] --apply
"""

from __future__ import annotations

import argparse

from shared.config import get_config
from shared.namespaces import BACKGROUND
from shared.postgres import connect_once

_ORDERED = """
WITH ordered AS (
    SELECT id, subject_id, order_index,
           row_number() OVER (PARTITION BY subject_id ORDER BY order_index, created_at, id) - 1
               AS place
    FROM content.lessons
    WHERE subject_id = ANY(%(subjects)s::uuid[]) AND status <> 'archived'
      AND source <> %(background_source)s
)
"""
_PLAN = _ORDERED + """
SELECT subject_id::text, count(*), min(order_index), max(order_index),
       count(*) FILTER (WHERE order_index <> place)
FROM ordered GROUP BY subject_id ORDER BY subject_id
"""
_RENUMBER = _ORDERED + """
UPDATE content.lessons l SET order_index = o.place, updated_at = now()
FROM ordered o WHERE l.id = o.id AND l.order_index <> o.place
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--subject", action="append", required=True, dest="subjects")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)

    config = get_config()
    params = {"subjects": args.subjects,
              "background_source": config.get("ingestion.books.lesson_source")[BACKGROUND]}
    with connect_once(config) as conn, conn.cursor() as cur:
        cur.execute(_PLAN, params)
        for subject_id, shown, low, high, moved in cur.fetchall():
            print(f"{subject_id}  {shown} shown, numbered {low}-{high}: "
                  f"{moved} renumbered to 0-{shown - 1}")
        if not args.apply:
            print("dry run: nothing changed; pass --apply to renumber")
            return 0
        cur.execute(_RENUMBER, params)
        renumbered = cur.rowcount
        conn.commit()
    print(f"renumbered {renumbered} lessons")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
