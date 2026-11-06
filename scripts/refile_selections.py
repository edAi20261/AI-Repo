"""Give each of a book's paired selections only its own pages.

A literature book teaches some texts in pairs on consecutive pages ("Thank You,
M'am" then "A Police Stop Changed This Teenager's Life"), and the split gave
both selections the pair's whole span. Each lesson held both texts and the
unit's writing task: 21 pairs, about 150 passages each, filed twice.

Where a selection starts is decided by a person reading the book, not guessed:
the opening page of a pair prints both titles, and a pair either repeats the
pre-reading pages for each text or shares them. ``survey`` lists each pair with
the pages the book prints each title, QUICK START and BACKGROUND on as
headings, from the reading stored when the book was ingested; the reviewed
start pages go in a plan, one ``{"lesson": id, "start": page}`` per selection.

Each lesson keeps its passages and figures from its start to the page before
the next selection's; the last keeps the rest of the span, as the book runs on
into the unit's task. Its other copies are removed, from that lesson only.

    python -m scripts.refile_selections survey
    python -m scripts.refile_selections refile --plan starts.json            # dry run
    python -m scripts.refile_selections refile --plan starts.json --apply    # writes a backup first
    python -m scripts.refile_selections restore --backup BACKUP.json.gz --apply
"""

from __future__ import annotations

import argparse
import datetime
import gzip
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

from shared.config import get_config
from shared.postgres import connect_once

# As in scripts/check_integrity.py: three passages this long, on more than one
# page, is a copy. Two sections that share one page both carry it on purpose.
SHARED_PASSAGE_CHARS = 300
SHARED_PASSAGES = 3
HEADINGS = ("heading", "subheading", "title", "header")
LANDMARKS = ("quick start", "background")

_SHARED = f"""
WITH p AS (
    SELECT c.lecture_id, l.subject_id, c.page_from, md5(regexp_replace(c.text, '\\s+', '', 'g')) h
    FROM ai.chunks c JOIN content.lessons l ON l.id = c.lecture_id
    WHERE l.status = 'published' AND length(c.text) > {SHARED_PASSAGE_CHARS}
)
SELECT a.lecture_id::text, b.lecture_id::text
FROM p a JOIN p b ON a.subject_id = b.subject_id AND a.h = b.h AND a.lecture_id < b.lecture_id
GROUP BY 1, 2 HAVING count(*) >= {SHARED_PASSAGES} AND count(DISTINCT a.page_from) > 1
"""
_LESSON = "SELECT title FROM content.lessons WHERE id = %s"
_PASSAGES = "SELECT chunk_id::text, page_from FROM ai.chunks WHERE lecture_id = %s"
_FIGURES = "SELECT figure_id::text, page FROM ai.figures WHERE lecture_id = %s"
_BOOKS = "SELECT DISTINCT lecture_id::text FROM ai.page_transcriptions"
_PAGES = """
SELECT DISTINCT ON (page) page, raw_output FROM ai.page_transcriptions
WHERE lecture_id = %s AND page BETWEEN %s AND %s AND raw_output ? 'blocks'
ORDER BY page, created_at DESC
"""

_BACKUP = {
    "ai.chunks": "SELECT row_to_json(x)::text FROM ai.chunks x WHERE chunk_id = ANY(%s::uuid[])",
    "ai.vector_index": ("SELECT row_to_json(x)::text FROM ai.vector_index x "
                        "WHERE chunk_id = ANY(%s::uuid[])"),
    "ai.figures": "SELECT row_to_json(x)::text FROM ai.figures x WHERE figure_id = ANY(%s::uuid[])",
}
_DELETE = (
    "DELETE FROM ai.figures WHERE figure_id = ANY(%(figures)s::uuid[])",
    "DELETE FROM ai.vector_index WHERE chunk_id = ANY(%(chunks)s::uuid[])",
    "DELETE FROM ai.chunks WHERE chunk_id = ANY(%(chunks)s::uuid[])",
)
_COUNTS = """
UPDATE ai.lecture_ingestion SET
    chunk_count = (SELECT count(*) FROM ai.chunks WHERE lecture_id = %(lesson)s),
    figure_count = (SELECT count(*) FROM ai.figures WHERE lecture_id = %(lesson)s)
WHERE lecture_id = %(lesson)s
"""
_RESTORE = ("INSERT INTO {table} SELECT * FROM json_populate_record(NULL::{table}, %s::json) "
            "ON CONFLICT DO NOTHING")


def words(text: str) -> str:
    """Letters and digits only, lower case: a title as the book prints it and as
    the lesson names it differ in case, quotes and punctuation."""
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", text).lower()))


def groups_of(pairs: list[tuple]) -> list[list[str]]:
    """Lessons that share passages, joined into groups."""
    linked: dict[str, set] = defaultdict(set)
    for a, b, *_ in pairs:
        linked[a].add(b)
        linked[b].add(a)
    seen, groups = set(), []
    for lesson in sorted(linked):
        if lesson in seen:
            continue
        group, stack = set(), [lesson]
        while stack:
            node = stack.pop()
            if node not in group:
                group.add(node)
                stack.extend(linked[node] - group)
        seen |= group
        groups.append(sorted(group))
    return groups


def headed(pages: list[tuple[int, dict]], wanted: str) -> list[int]:
    """The pages printing ``wanted`` as a heading, from their stored reading."""
    return [page for page, reading in pages
            if any(block.get("type") in HEADINGS and words(block.get("text") or "") == words(wanted)
                   for block in reading.get("blocks", []))]


def spans(starts: dict[str, int]) -> dict[str, tuple[int, float]]:
    """Each selection from its start to the page before the next; the last to the end."""
    ordered = sorted(starts.items(), key=lambda item: item[1])
    if len({page for _, page in ordered}) < len(ordered):
        raise ValueError("two selections start on the same page")
    return {lesson: (page, ordered[n + 1][1] - 1 if n + 1 < len(ordered) else float("inf"))
            for n, (lesson, page) in enumerate(ordered)}


def outside(items: list[tuple[str, int]], span: tuple[int, float]) -> list[str]:
    return [item_id for item_id, page in items if not span[0] <= page <= span[1]]


def _book_of(lesson: str, books: list[str]) -> str | None:
    """The book a lesson was cut from: its id is the book's id and its place in it."""
    from ingestion.books import STALE_SEARCH, lesson_id_for

    return next((book for book in books
                 if lesson in {lesson_id_for(book, n) for n in range(STALE_SEARCH)}), None)


def _groups(cur) -> list[list[str]]:
    cur.execute(_SHARED)
    return groups_of(cur.fetchall())


def survey(config) -> int:
    with connect_once(config) as conn, conn.cursor() as cur:
        groups = _groups(cur)
        cur.execute(_BOOKS)
        books = [r[0] for r in cur.fetchall()]
        for group in groups:
            titles, first, last = {}, None, None
            for lesson in group:
                cur.execute(_LESSON, (lesson,))
                titles[lesson] = cur.fetchone()[0]
                cur.execute(_PASSAGES, (lesson,))
                pages = [page for _c, page in cur.fetchall()]
                first = min(pages + ([first] if first else []))
                last = max(pages + ([last] if last else []))
            cur.execute(_PAGES, (_book_of(group[0], books), first, last))
            read = cur.fetchall()
            print(f"== p.{first}-{last}  "
                  + "  ".join(f"{mark.upper()} {headed(read, mark)}" for mark in LANDMARKS))
            for lesson, title in titles.items():
                print(f"   {lesson}  {title[:60]:<60} printed {headed(read, title)}")
    return 0


def _plan(cur, plan: list[dict]) -> list[dict]:
    starts = {entry["lesson"]: int(entry["start"]) for entry in plan}
    work = []
    for group in _groups(cur):
        if not set(group) & set(starts):
            continue
        if not set(group) <= set(starts):
            raise SystemExit(f"the plan names only part of a group: {sorted(set(group) - set(starts))}")
        for lesson, span in spans({lesson: starts[lesson] for lesson in group}).items():
            cur.execute(_LESSON, (lesson,))
            title = cur.fetchone()[0]
            cur.execute(_PASSAGES, (lesson,))
            passages = cur.fetchall()
            cur.execute(_FIGURES, (lesson,))
            figures = cur.fetchall()
            removed = outside(passages, span)
            work.append({"lesson": lesson, "title": title, "span": span,
                         "keep": len(passages) - len(removed), "chunks": removed,
                         "figures": outside(figures, span)})
    unknown = set(starts) - {w["lesson"] for w in work}
    if unknown:
        raise SystemExit(f"the plan names lessons that share no passages: {sorted(unknown)}")
    return work


def refile(config, plan_path: Path, apply: bool) -> int:
    with connect_once(config) as conn, conn.cursor() as cur:
        work = _plan(cur, json.loads(plan_path.read_text()))
    for item in work:
        end = "end" if item["span"][1] == float("inf") else item["span"][1]
        print(f"{item['title'][:60]:<60} p.{item['span'][0]}-{end}  keeps {item['keep']}, "
              f"removes {len(item['chunks'])} passages and {len(item['figures'])} figures")
    if any(not item["keep"] for item in work):
        raise SystemExit("refused: a lesson would keep no passage")
    chunks = [c for item in work for c in item["chunks"]]
    figures = [f for item in work for f in item["figures"]]
    print(f"{len(work)} lessons: {len(chunks)} passages and {len(figures)} figures to remove")
    if not apply:
        print("dry run: nothing changed; pass --apply")
        return 0

    from shared.vector_store import upsert_client

    client = upsert_client(config, config.secret("QDRANT_URL"))
    collection = config.vector_store.get("collection")
    backup = {"lessons": [item["lesson"] for item in work], "rows": {}, "points": []}
    with connect_once(config) as conn, conn.cursor() as cur:
        for table, sql in _BACKUP.items():
            cur.execute(sql, (figures if table == "ai.figures" else chunks,))
            backup["rows"][table] = [r[0] for r in cur.fetchall()]
    for start in range(0, len(chunks), 256):
        backup["points"] += [{"id": str(p.id), "vector": _plain(p.vector), "payload": p.payload}
                             for p in client.retrieve(collection, ids=chunks[start:start + 256],
                                                      with_payload=True, with_vectors=True)]
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = Path(f"refile-backup-{stamp}.json.gz")
    with gzip.open(path, "wt") as out:
        json.dump(backup, out)
    print("backup:", path)

    for start in range(0, len(chunks), 500):
        client.delete(collection, points_selector=chunks[start:start + 500], wait=True)
    with connect_once(config) as conn, conn.cursor() as cur:
        for sql in _DELETE:
            cur.execute(sql, {"chunks": chunks, "figures": figures})
        for item in work:
            cur.execute(_COUNTS, {"lesson": item["lesson"]})
        conn.commit()
    print(f"removed {len(chunks)} passages and {len(figures)} figures from {len(work)} lessons")
    return 0


def _plain(vector):
    """A point's vectors as JSON: the sparse half is an object, not a list."""
    if not isinstance(vector, dict):
        return vector
    return {name: (value if isinstance(value, list) else
                   {"indices": list(value.indices), "values": list(value.values)})
            for name, value in vector.items()}


def restore(config, backup_path: Path, apply: bool) -> int:
    from qdrant_client import models

    from shared.vector_store import upsert_client

    with gzip.open(backup_path, "rt") as source:
        backup = json.load(source)
    print({table: len(rows) for table, rows in backup["rows"].items()}, len(backup["points"]), "points")
    if not apply:
        print("dry run: nothing changed; pass --apply")
        return 0
    with connect_once(config) as conn, conn.cursor() as cur:
        for table in ("ai.chunks", "ai.vector_index", "ai.figures"):
            for row in backup["rows"][table]:
                cur.execute(_RESTORE.format(table=table), (row,))
        for lesson in backup["lessons"]:
            cur.execute(_COUNTS, {"lesson": lesson})
        conn.commit()

    def vector(value):
        if not isinstance(value, dict):
            return value
        return {name: models.SparseVector(**v) if isinstance(v, dict) else v for name, v in value.items()}

    client = upsert_client(config, config.secret("QDRANT_URL"))
    points = [models.PointStruct(id=p["id"], vector=vector(p["vector"]), payload=p["payload"])
              for p in backup["points"]]
    for start in range(0, len(points), 64):
        client.upsert(config.vector_store.get("collection"), points=points[start:start + 64], wait=True)
    print("restored")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("survey")
    refiling = sub.add_parser("refile")
    refiling.add_argument("--plan", type=Path, required=True)
    refiling.add_argument("--apply", action="store_true")
    restoring = sub.add_parser("restore")
    restoring.add_argument("--backup", type=Path, required=True)
    restoring.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)

    config = get_config()
    if args.command == "survey":
        return survey(config)
    if args.command == "refile":
        return refile(config, args.plan, args.apply)
    return restore(config, args.backup, args.apply)


if __name__ == "__main__":
    raise SystemExit(main())
