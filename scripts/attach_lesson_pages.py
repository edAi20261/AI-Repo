"""Attach to each lesson of a book the pages of the book it was cut from.

The lesson page shows a lesson's PDF attachment, and a lesson ingestion split
out of a book had none: a student opened it to an empty page. Ingestion now
attaches them itself (``ingestion.lesson_pages``); this does the same, with the
same functions, for the lessons ingested before it did. A lesson that already
has an attachment, and a lesson of the book that is not shown (its glossary or
index, filed as background), is left alone.

A lesson's pages run from the first page any of its passages came from to the
page before the next lesson's, counting every lesson of the book, shown or not;
the last runs to the end of the book.

    python -m scripts.attach_lesson_pages --source s3://bucket/book.pdf          # dry run
    python -m scripts.attach_lesson_pages --source s3://bucket/book.pdf --apply

The pages are stored as the user ``LESSON_PAGES_UPLOADER`` names, unless
``--uploader`` names another.

``--book`` names the book's id when it was not ingested from ``--source`` itself
(a book ingested before ingestion split books is filed under its own id).
"""

from __future__ import annotations

import argparse
import tempfile
from dataclasses import asdict
from pathlib import Path
from urllib.parse import urlparse

from ingestion import lesson_pages
from ingestion.books import STALE_SEARCH, lesson_id_for
from ingestion.lesson import book_id_for
from shared.config import get_config
from shared.namespaces import SYLLABUS
from shared.postgres import connect_once
from shared.storage import get_uploader, s3_client

_LESSONS = """
SELECT l.id::text, l.subject_id::text, l.title,
       (SELECT min(c.page_from) FROM ai.chunks c WHERE c.lecture_id = l.id),
       EXISTS (SELECT 1 FROM content.lesson_attachments a WHERE a.lesson_id = l.id)
       OR l.source <> %(shown_source)s
FROM content.lessons l
WHERE l.id = ANY(%(ids)s::uuid[]) AND l.status <> 'archived'
ORDER BY l.order_index
"""


def _spans(rows: list[tuple], page_count: int) -> list[tuple[tuple, int, int]]:
    """Each lesson with the pages it runs over; one with no passages has none."""
    located = sorted((row for row in rows if row[3] is not None), key=lambda row: row[3])
    spans = []
    for n, row in enumerate(located):
        later = [other[3] for other in located[n + 1:] if other[3] > row[3]]
        spans.append((row, row[3], later[0] - 1 if later else page_count))
    return spans


def main(argv: list[str] | None = None) -> int:
    import fitz

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", required=True, help="the book's PDF, s3://bucket/key")
    parser.add_argument("--book", help="the book's id, when it is not the one --source gives")
    parser.add_argument("--uploader", help="the user the pages are stored as, when not "
                                           "the one LESSON_PAGES_UPLOADER names")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)

    config = get_config()
    book_id = args.book or book_id_for(args.source)
    params = {"ids": [book_id] + [lesson_id_for(book_id, n) for n in range(STALE_SEARCH)],
              "shown_source": config.get("ingestion.books.lesson_source")[SYLLABUS]}
    with connect_once(config) as conn, conn.cursor() as cur:
        cur.execute(_LESSONS, params)
        rows = cur.fetchall()
    if not rows:
        raise SystemExit(f"no ministry lessons are filed under book {book_id}")

    location = urlparse(args.source)
    bucket, key = location.netloc, location.path.lstrip("/")
    s3 = s3_client(config)
    with tempfile.TemporaryDirectory() as scratch:
        local = Path(scratch) / "book.pdf"
        s3.download_file(bucket, key, str(local))
        book = fitz.open(local)
        spans = _spans(rows, book.page_count)
        todo = [(row, first, last) for row, first, last in spans if not row[4]]
        for (lesson_id, _, title, _, attached), first, last in spans:
            state = "skipped" if attached else "to attach"
            print(f"{lesson_id}  p{first}-{last:<4} {state:<9}  {title[:70]}")
        print(f"{len(rows)} lessons, {len(todo)} to attach, "
              f"{len(rows) - len(spans)} with no passages to place")
        if not args.apply:
            print("dry run: nothing changed; pass --apply to attach")
            return 0

        uploader = args.uploader or lesson_pages.uploader_id(config)
        storage = get_uploader(config)
        for (lesson_id, subject_id, title, _, _), first, last in todo:
            stored = lesson_pages.store(storage, book, subject_id=subject_id,
                                        lesson_id=lesson_id, title=title,
                                        first_page=first, last_page=last)
            with connect_once(config) as conn, conn.cursor() as cur:
                lesson_pages.record(cur, asdict(stored), lesson_id=lesson_id, uploader=uploader)
                conn.commit()
            print(f"attached p{first}-{last} to {lesson_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
