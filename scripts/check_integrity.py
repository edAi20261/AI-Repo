"""Check the live stores against what the platform promises a student.

    python -m scripts.check_integrity                 # every rule, PDFs by suspicion
    python -m scripts.check_integrity --pdfs all      # open every shown lesson's PDF
    python -m scripts.check_integrity --json out.json

Each rule prints its count and the first offenders, and any failed rule makes
the exit status 1. Nothing is written.

Every rule is a failure that reached a student or a reviewer before anything
noticed: archived lessons still answering questions, a lesson tagged with
another lesson's topics, two lessons holding the same passages, a 14-page lesson
shown as a 133-page PDF. Each was the leftover of a step that did its own job
(a split, a re-read, an archive) and was checked only against that job.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable

from shared.config import get_config
from shared.matching import NOT_CONTENT
from shared.namespaces import BACKGROUND, SYLLABUS
from shared.postgres import connect_once
from shared.vector_store import query_client

SEARCHED = (SYLLABUS, BACKGROUND)
SHOWN = "published"
# A passage this long is a paragraph of the book, not a heading or a running
# head every page repeats; three of them shared is a copied section.
SHARED_PASSAGE_CHARS = 300
SHARED_PASSAGES = 3
# The lesson PDF may hold the figure-only pages that end its last passage's page
# range without a passage of their own.
PDF_SLACK_PAGES = 2
# A PDF whose bytes per lesson page are this many times its subject's median is
# opened; the 133-page PDF of a 14-page lesson stood at 9x.
PDF_SUSPECT_RATIO = 3.0
MIN_LESSON_CHARS = 200
SHOWN_EXAMPLES = 8


@dataclass
class Result:
    rule: str
    promise: str
    offenders: list = field(default_factory=list)
    warning: bool = False

    @property
    def failed(self) -> bool:
        return bool(self.offenders) and not self.warning


def shared_passages(rows: Iterable[tuple]) -> list[tuple]:
    """Pairs of lessons of one subject holding the same long passages.

    ``rows``: (subject_id, lesson_id, title, passage_hash)."""
    holders: dict[tuple, set] = defaultdict(set)
    titles = {}
    for subject, lesson, title, digest in rows:
        holders[(subject, digest)].add(lesson)
        titles[lesson] = title
    pairs: dict[tuple, int] = defaultdict(int)
    for lessons in holders.values():
        ordered = sorted(lessons)
        for i, first in enumerate(ordered):
            for second in ordered[i + 1:]:
                pairs[(first, second)] += 1
    return sorted(((titles[a], titles[b], n) for (a, b), n in pairs.items()
                   if n >= SHARED_PASSAGES), key=lambda p: -p[2])


def out_of_order(rows: Iterable[tuple]) -> list[tuple]:
    """Lessons listed after a lesson that comes later in the same book.

    ``rows``: (subject_id, book, order_index, first_page, title), in any order.
    A lesson cut from a book carries the book's title as its description."""
    books: dict[tuple, list] = defaultdict(list)
    for subject, book, order, first_page, title in rows:
        books[(subject, book)].append((order, first_page, title))
    found = []
    for lessons in books.values():
        lessons.sort()
        highest = None
        for _order, first_page, title in lessons:
            if highest is not None and first_page < highest[0]:
                found.append((title, first_page, highest[1], highest[0]))
            if highest is None or first_page > highest[0]:
                highest = (first_page, title)
    return found


def pdf_suspects(rows: Iterable[tuple]) -> list[tuple]:
    """Lessons whose PDF is heavy for the pages they teach, against their subject.

    ``rows``: (subject_id, lesson_id, title, pdf_bytes, page_span, s3_bucket, s3_key)."""
    by_subject: dict[str, list] = defaultdict(list)
    for row in rows:
        if row[3] and row[4]:
            by_subject[row[0]].append(row)
    suspects = []
    for lessons in by_subject.values():
        median = statistics.median(r[3] / r[4] for r in lessons)
        suspects += [r for r in lessons if r[3] / r[4] > PDF_SUSPECT_RATIO * median]
    return suspects


_COUNTS = """
SELECT (SELECT count(*) FROM ai.chunks), (SELECT count(*) FROM ai.vector_index)
"""
_INDEXED = "SELECT chunk_id::text, lecture_id::text, namespace FROM ai.vector_index"
_ARCHIVED = "SELECT id::text FROM content.lessons WHERE status = 'archived'"

_SHOWN = f"""
SELECT l.id::text, s.name, l.title, coalesce(i.status::text, 'none'),
       count(c.chunk_id), coalesce(sum(length(c.text)), 0),
       EXISTS (SELECT 1 FROM content.lesson_attachments a WHERE a.lesson_id = l.id)
FROM content.lessons l
JOIN academics.subjects s ON s.id = l.subject_id
LEFT JOIN ai.lecture_ingestion i ON i.lecture_id = l.id
LEFT JOIN ai.chunks c ON c.lecture_id = l.id
WHERE l.status = '{SHOWN}'
GROUP BY l.id, s.name, l.title, i.status
"""

_BORROWED_TOPICS = f"""
SELECT s.name, l.title, count(c.chunk_id),
       count(c.chunk_id) FILTER (WHERE cardinality(coalesce(c.topic_ids, '{{}}')) = 0),
       coalesce(i.topic_count, 0),
       (SELECT count(*) FROM ai.topics t WHERE t.lecture_id = l.id)
FROM content.lessons l
JOIN academics.subjects s ON s.id = l.subject_id
JOIN ai.chunks c ON c.lecture_id = l.id
LEFT JOIN ai.lecture_ingestion i ON i.lecture_id = l.id
WHERE l.status = '{SHOWN}'
  AND EXISTS (SELECT 1 FROM ai.vector_index v WHERE v.lecture_id = l.id AND v.namespace = '{SYLLABUS}')
GROUP BY s.name, l.id, l.title, i.topic_count
"""

_PASSAGES = f"""
SELECT l.subject_id::text, l.id::text, l.title, md5(regexp_replace(c.text, '\\s+', '', 'g'))
FROM ai.chunks c JOIN content.lessons l ON l.id = c.lecture_id
WHERE l.status <> 'archived' AND length(c.text) > {SHARED_PASSAGE_CHARS}
  AND EXISTS (SELECT 1 FROM ai.vector_index v WHERE v.chunk_id = c.chunk_id AND v.namespace = '{SYLLABUS}')
"""

_ORDER = f"""
SELECT l.subject_id::text, l.description, l.order_index, min(c.page_from), l.title
FROM content.lessons l JOIN ai.chunks c ON c.lecture_id = l.id
WHERE l.status = '{SHOWN}' AND l.description IS NOT NULL
GROUP BY l.id
"""

_PDFS = f"""
SELECT l.subject_id::text, l.id::text, l.title, m.size_bytes,
       max(c.page_to) - min(c.page_from) + 1, m.s3_bucket, m.s3_key
FROM content.lessons l
JOIN content.lesson_attachments a ON a.lesson_id = l.id
JOIN content.media_assets m ON m.id = a.media_id AND m.mime_type = 'application/pdf'
JOIN ai.chunks c ON c.lecture_id = l.id
WHERE l.status = '{SHOWN}'
GROUP BY l.id, m.id
"""

_EMPTY_SUBJECTS = f"""
SELECT cu.name, s.grade_level::text, s.name
FROM academics.subjects s JOIN academics.curricula cu ON cu.id = s.curriculum_id
WHERE NOT EXISTS (SELECT 1 FROM content.lessons l WHERE l.subject_id = s.id AND l.status = '{SHOWN}')
ORDER BY 1, 2, 3
"""


def _points(config) -> dict[str, dict]:
    client = query_client(config, timeout=120)
    points, offset = {}, None
    while True:
        batch, offset = client.scroll(
            config.vector_store.get("collection"), limit=5000, offset=offset,
            with_payload=["lecture_id", "namespace", "chunk_kind"], with_vectors=False)
        points.update({str(p.id): p.payload for p in batch})
        if offset is None:
            return points


def _pdf_pages(config, bucket: str, key: str) -> int:
    import fitz

    from shared.storage import s3_client

    body = s3_client(config).get_object(Bucket=bucket, Key=key)["Body"].read()
    with fitz.open(stream=body, filetype="pdf") as document:
        return document.page_count


def check(config, *, all_pdfs: bool) -> list[Result]:
    with connect_once(config) as conn, conn.cursor() as cur:
        def rows(sql):
            cur.execute(sql)
            return cur.fetchall()

        chunk_count, index_count = rows(_COUNTS)[0]
        indexed = {chunk: (lesson, namespace) for chunk, lesson, namespace in rows(_INDEXED)}
        archived = {r[0] for r in rows(_ARCHIVED)}
        shown = rows(_SHOWN)
        borrowed = rows(_BORROWED_TOPICS)
        passages = rows(_PASSAGES)
        order = rows(_ORDER)
        pdfs = rows(_PDFS)
        empty = rows(_EMPTY_SUBJECTS)
    points = _points(config)

    parity = Result("R1 index parity",
                    "every passage is in ai.vector_index and Qdrant once, and nothing else is")
    if not chunk_count == index_count == len(points):
        parity.offenders.append(f"chunks={chunk_count} vector_index={index_count} points={len(points)}")
    parity.offenders += [f"point {pid} ({p.get('lecture_id')}) has no passage"
                         for pid, p in points.items() if pid not in indexed]
    parity.offenders += [f"passage {cid} has no point" for cid in indexed if cid not in points]

    searchable = Result("R2 archived is never searched",
                        "no archived lesson has a point in a searched namespace")
    by_lesson: dict[str, int] = defaultdict(int)
    for payload in points.values():
        if payload.get("lecture_id") in archived and payload.get("namespace") in SEARCHED:
            by_lesson[payload["lecture_id"]] += 1
    searchable.offenders = [f"{lesson}: {n} points" for lesson, n in
                            sorted(by_lesson.items(), key=lambda x: -x[1])]

    complete = Result("R3 every shown lesson is complete",
                      f"ready, has passages, at least {MIN_LESSON_CHARS} characters and a PDF")
    for _lesson, subject, title, status, chunks, chars, pdf in shown:
        missing = [what for what, bad in (("not ready", status != "ready"), ("no passages", not chunks),
                                          ("too little text", chars < MIN_LESSON_CHARS), ("no PDF", not pdf))
                   if bad]
        if missing:
            complete.offenders.append(f"{subject} | {title} | {', '.join(missing)}")

    topics = Result("R4 a lesson's topics are its own",
                    "every passage carries a topic, and every shown lesson's topics were found in it")
    for subject, title, chunks, untagged, topic_count, owned in borrowed:
        if untagged:
            topics.offenders.append(f"{subject} | {title} | {untagged}/{chunks} passages without a topic")
        elif not owned and not topic_count:
            topics.offenders.append(f"{subject} | {title} | only other lessons' topics")

    copies = Result("R5 no lesson holds another's passages",
                    f"no two lessons share {SHARED_PASSAGES}+ passages over {SHARED_PASSAGE_CHARS} characters")
    copies.offenders = [f"{a} | {b} | {n} shared" for a, b, n in shared_passages(passages)]

    fit = Result("R6 a lesson's PDF is its own pages",
                 f"the PDF has the lesson's page span (+{PDF_SLACK_PAGES})")
    for _subject, _lesson, title, _bytes, span, bucket, key in (pdfs if all_pdfs else pdf_suspects(pdfs)):
        try:
            pages = _pdf_pages(config, bucket, key)
        except Exception as error:
            fit.offenders.append(f"{title} | PDF unreadable: {type(error).__name__}: {error}")
            continue
        if pages > span + PDF_SLACK_PAGES:
            fit.offenders.append(f"{title} | PDF {pages} pages for a {span}-page lesson")

    apparatus = Result("R7 the tutor can answer from every shown lesson",
                       "a shown lesson has passages retrieval answers from, not only contents or index pages",
                       warning=True)
    kinds: dict[str, set] = defaultdict(set)
    for payload in points.values():
        kinds[payload.get("lecture_id")].add(payload.get("chunk_kind"))
    apparatus.offenders = [f"{subject} | {title}" for lesson, subject, title, *_ in shown
                           if kinds.get(lesson) and kinds[lesson] <= set(NOT_CONTENT)]

    ordered = Result("R8 lessons follow the book",
                     "no lesson is listed after a lesson that comes later in its book")
    ordered.offenders = [f"{title} (p.{page}) after {later} (p.{later_page})"
                         for title, page, later, later_page in out_of_order(order)]

    empties = Result("R9 subjects with nothing shown",
                     "a subject with no shown lesson is not listed to students (backend)", warning=True)
    empties.offenders = [" | ".join(r) for r in empty]

    return [parity, searchable, complete, topics, copies, fit, apparatus, ordered, empties]


def report(results: list[Result]) -> int:
    for result in results:
        mark = "WARN" if result.warning and result.offenders else ("FAIL" if result.failed else "OK  ")
        print(f"{mark} {result.rule}: {len(result.offenders)} — {result.promise}")
        for offender in result.offenders[:SHOWN_EXAMPLES]:
            print(f"       {offender}")
        if len(result.offenders) > SHOWN_EXAMPLES:
            print(f"       … and {len(result.offenders) - SHOWN_EXAMPLES} more")
    failed = [r.rule for r in results if r.failed]
    print("\n" + ("ALL RULES HOLD" if not failed else f"FAILED: {', '.join(failed)}"))
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pdfs", choices=("suspects", "all"), default="suspects")
    parser.add_argument("--json", help="also write every offender to this file")
    args = parser.parse_args(argv)

    results = check(get_config(), all_pdfs=args.pdfs == "all")
    if args.json:
        with open(args.json, "w") as out:
            json.dump([{"rule": r.rule, "failed": r.failed, "offenders": r.offenders}
                       for r in results], out, ensure_ascii=False, indent=1)
    return report(results)


if __name__ == "__main__":
    sys.exit(main())
