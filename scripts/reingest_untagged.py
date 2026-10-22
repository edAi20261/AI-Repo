"""Re-ingest the lessons recorded ready that cannot answer anything.

A lesson recorded ready with no topics has passages no quiz can be built on,
and one with no passages has nothing to search, summarize or quiz on. Each is
read again through the path the platform uses when a teacher publishes it.

    python -m scripts.reingest_untagged                         # dry run: list them
    python -m scripts.reingest_untagged --lecture <id>          # dry run: one lesson
    python -m scripts.reingest_untagged [--lecture <id> ...] --apply

A lesson the platform cannot hand over as one source (a lesson cut from a book
has no file of its own) is reported and left alone: its book is what re-reads it.
"""

from __future__ import annotations

import argparse
from typing import Callable, Sequence

from shared.config import get_config
from shared.postgres import connect_once

_UNTAGGED = """
SELECT i.lecture_id::text, s.name, l.title, l.source::text, i.chunk_count, i.topic_count
FROM ai.lecture_ingestion i
JOIN content.lessons l ON l.id = i.lecture_id
LEFT JOIN academics.subjects s ON s.id = l.subject_id
WHERE i.status = 'ready' AND (i.topic_count = 0 OR i.chunk_count = 0)
  AND l.status <> 'archived'
  AND (cardinality(%(lectures)s::uuid[]) = 0 OR i.lecture_id = ANY(%(lectures)s::uuid[]))
ORDER BY s.name, l.order_index
"""


def untagged(config, lectures: Sequence[str]) -> list[tuple]:
    with connect_once(config) as conn, conn.cursor() as cur:
        cur.execute(_UNTAGGED, {"lectures": list(lectures)})
        return cur.fetchall()


def reingest(config, lesson_ids: Sequence[str], *,
             resolve: Callable[[object, str], dict],
             ingest: Callable[..., dict]) -> dict[str, list[str]]:
    """Each lesson through ``resolve`` then ``ingest``, one at a time, so one
    lesson that fails does not stop the rest."""
    outcome: dict[str, list[str]] = {"ready": [], "needs_review": [], "skipped": [],
                                     "failed": []}
    for lesson_id in lesson_ids:
        try:
            state = resolve(config, lesson_id)
        except Exception as error:
            print(f"skipped {lesson_id}: {error}")
            outcome["skipped"].append(lesson_id)
            continue
        try:
            result = ingest(state, config=config)
        except Exception as error:
            print(f"failed  {lesson_id}: {type(error).__name__}: {error}")
            outcome["failed"].append(lesson_id)
            continue
        status = result.get("ingestion_status") or "ready"
        stats = result.get("stats", {})
        print(f"{status:<12} {lesson_id}  chunks={stats.get('index', {}).get('chunks_written', 0)}"
              f"  topics={stats.get('topics', {}).get('extracted', 0)}"
              + (f"  ({'; '.join(result.get('errors') or [])})" if status != "ready" else ""))
        outcome.setdefault(status, []).append(lesson_id)
    return outcome


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lecture", action="append", default=[], dest="lectures")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)

    config = get_config()
    rows = untagged(config, args.lectures)
    for lecture_id, subject, title, source, chunks, topics in rows:
        print(f"{lecture_id}  {subject or '-':<24.24}  {source:<16}  "
              f"chunks={chunks:<4} topics={topics:<3} {title}")
    print(f"{len(rows)} ready lessons with no topics or no passages")
    if not args.apply:
        print("dry run: nothing changed; pass --apply to re-ingest them")
        return 0

    from ingestion import run_ingestion
    from system.ingestion_jobs import lesson_state

    outcome = reingest(config, [row[0] for row in rows],
                       resolve=lesson_state, ingest=run_ingestion)
    print("  ".join(f"{status}={len(ids)}" for status, ids in outcome.items()))
    return 1 if outcome["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
