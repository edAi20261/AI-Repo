"""Give every lesson of a book split before ingestion read terms the term it is in.

A subject holds both terms' books, and nothing said which was which: Christian
Religion G7 showed the second term's lessons first, and the G9 books for the
whole year ran unit 1 to 5 twice. Each book's term is read again from its own
first pages and the openings of its lessons, into a plan to look over; a plan is
only stored when it is given back with --apply.

    python -m scripts.backfill_lesson_terms --out terms.csv [--subject <id> ...]   # read only
    python -m scripts.backfill_lesson_terms --apply terms.csv                      # store
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

from ingestion.books import (
    OPENING_PAGE_CHARS,
    OPENING_PAGES,
    STALE_SEARCH,
    LessonCatalog,
    SplitLesson,
    lesson_id_for,
    read_lesson_terms,
)
from ingestion.layout.blocks import ParsedPage
from shared.config import get_config
from shared.postgres import connect_once

logger = logging.getLogger("backfill_lesson_terms")

LESSON_OPENING_CHARS = 240
FIELDS = ("subject_id", "book_id", "lesson_id", "ordinal", "term", "title")

# A book cut into one PDF per lesson was read slice by slice, each under its
# lesson's own id, and only the slice holding page 1 has the cover that says
# which term the book is: Christian Religion G7 read as no term in 35 of its 38
# lessons until the slices were put back together under the book they came from.
_BOOKS = """
SELECT lecture_id::text,
       COALESCE(NULLIF(MIN(raw_output->'meta'->>'doc_id'), lecture_id::text),
                lecture_id::text) AS book,
       MIN(page) AS first_page
FROM ai.page_transcriptions GROUP BY lecture_id ORDER BY 2, 3
"""
_OPENING = """
SELECT page, raw_output FROM (
    SELECT DISTINCT ON (page) page, raw_output FROM ai.page_transcriptions
    WHERE lecture_id = ANY(%s::uuid[]) ORDER BY page, created_at DESC) AS pages
ORDER BY page LIMIT %s
"""
_LESSONS = """
SELECT id::text, subject_id::text, title FROM content.lessons
WHERE id = ANY(%(ids)s::uuid[]) AND status <> 'archived'
  AND (%(subjects)s::uuid[] IS NULL OR subject_id = ANY(%(subjects)s::uuid[]))
"""
_LESSON_OPENINGS = """
SELECT DISTINCT ON (lecture_id) lecture_id::text, text FROM ai.chunks
WHERE lecture_id = ANY(%s::uuid[]) ORDER BY lecture_id, order_index
"""


@dataclass(frozen=True)
class Book:
    book_id: str
    subject_id: str
    opening_pages: str
    lessons: list[tuple[str, SplitLesson]]


def page_text(raw: dict[str, Any]) -> str:
    """A transcribed page, in either form the page cache holds."""
    if "blocks" in raw:
        return " ".join(block.text for block in ParsedPage.model_validate(raw).content)
    return str(raw.get("text") or "")


def _one_line(text: str, limit: int) -> str:
    return " ".join(str(text or "").split())[:limit]


def books_to_read(cur, subjects: Optional[list[str]]) -> list[Book]:
    cur.execute(_BOOKS)
    slices: dict[str, list[str]] = defaultdict(list)
    for slice_id, book_id, _ in cur.fetchall():
        slices[book_id].append(slice_id)
    books = []
    for book_id, slice_ids in slices.items():
        book = _book(cur, book_id, slice_ids, subjects)
        if book is not None:
            books.append(book)
    return books


def _book(cur, book_id: str, slice_ids: list[str],
          subjects: Optional[list[str]]) -> Optional[Book]:
    place = {}
    for position, slice_id in enumerate(slice_ids):
        place[slice_id] = (position, 0)
        place.update({lesson_id_for(slice_id, n): (position, n) for n in range(STALE_SEARCH)})
    cur.execute(_LESSONS, {"ids": list(place), "subjects": subjects})
    rows = sorted(cur.fetchall(), key=lambda row: place[row[0]])
    if not rows:
        return None
    by_subject = Counter(subject_id for _, subject_id, _ in rows)
    if len(by_subject) > 1:
        logger.error("book %s has lessons in %d subjects; left out", book_id, len(by_subject))
        return None
    cur.execute(_LESSON_OPENINGS, ([lesson_id for lesson_id, _, _ in rows],))
    openings = dict(cur.fetchall())
    cur.execute(_OPENING, (slice_ids, OPENING_PAGES))
    opening = "\n\n".join(f"[{page}]\n{page_text(raw)[:OPENING_PAGE_CHARS]}"
                          for page, raw in cur.fetchall())
    lessons = [(lesson_id, SplitLesson(ordinal, _one_line(title, 200),
                                       _one_line(openings.get(lesson_id, ""),
                                                 LESSON_OPENING_CHARS)))
               for ordinal, (lesson_id, _, title) in enumerate(rows)]
    return Book(book_id, rows[0][1], opening, lessons)


def plan_rows(book: Book, terms: dict[int, Optional[int]]) -> list[dict[str, Any]]:
    return [{"subject_id": book.subject_id, "book_id": book.book_id, "lesson_id": lesson_id,
             "ordinal": lesson.ordinal, "term": "" if terms[lesson.ordinal] is None
             else terms[lesson.ordinal], "title": lesson.title}
            for lesson_id, lesson in book.lessons]


def read_plan(books: Sequence[Book], read: Callable[[Book], dict[int, Optional[int]]],
              workers: int) -> tuple[list[dict[str, Any]], list[str]]:
    def one(book: Book):
        try:
            return book, read(book), None
        except Exception as error:      # noqa: BLE001 — one book's failure is reported, not fatal
            logger.error("could not read the terms of book %s: %s", book.book_id, error)
            return book, None, error

    rows: list[dict[str, Any]] = []
    failed: list[str] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for book, terms, error in pool.map(one, books):
            if error is not None:
                failed.append(book.book_id)
            else:
                rows.extend(plan_rows(book, terms))
    return rows, failed


def apply_plan(rows: Sequence[dict[str, str]], catalog: LessonCatalog) -> int:
    by_subject: dict[str, dict[str, Optional[int]]] = defaultdict(dict)
    for row in rows:
        by_subject[row["subject_id"]][row["lesson_id"]] = (
            int(row["term"]) if str(row["term"]).strip() else None)
    return sum(catalog.set_terms(subject_id, terms) for subject_id, terms in by_subject.items())


def _summary(rows: Sequence[dict[str, Any]]) -> None:
    counts: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        counts[row["subject_id"]][str(row["term"]) or "none"] += 1
    for subject_id, terms in sorted(counts.items()):
        print(f"{subject_id}  " + "  ".join(f"term {t}: {n}" for t, n in sorted(terms.items())))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--out", help="read the terms and write the plan to this CSV")
    action.add_argument("--apply", help="store the terms of a plan this script wrote")
    parser.add_argument("--subject", action="append", dest="subjects")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    config = get_config()

    if args.apply:
        with open(args.apply, newline="", encoding="utf-8") as plan:
            rows = list(csv.DictReader(plan))
        print(f"stored the term of {apply_plan(rows, LessonCatalog(config))} lessons")
        return 0

    with connect_once(config) as conn, conn.cursor() as cur:
        books = books_to_read(cur, args.subjects)
    print(f"{len(books)} books to read")
    rows, failed = read_plan(
        books, lambda book: read_lesson_terms(book.opening_pages,
                                              [lesson for _, lesson in book.lessons], config),
        args.workers)
    with open(args.out, "w", newline="", encoding="utf-8") as plan:
        writer = csv.DictWriter(plan, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    _summary(rows)
    print(f"plan for {len(rows)} lessons written to {args.out}; nothing stored")
    for book_id in failed:
        print(f"not read: {book_id}")
    return 1 if failed else 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    sys.exit(main())
