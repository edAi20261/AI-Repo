"""Split a book that was ingested whole into the lessons it teaches, in place.

Books ingested before ingestion split them are one lesson each: the subject
shows the whole book as its only lesson. Their pages are already read and
indexed, so nothing is read or embedded again. The lessons are found in the
stored page text by the breakdown ingestion uses, every passage, figure, page
and topic is filed under the lesson its page belongs to, and the whole-book
lesson is archived.

    python -m scripts.split_whole_books --lesson <id> --plan plan.json            # dry run
    python -m scripts.split_whole_books --lesson <id> --plan plan.json --apply

The dry run asks the model and writes the lessons it found to ``plan.json``;
``--apply`` files the book under exactly those lessons, so what is applied is
what was reviewed.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from ingestion.books import Lesson, LessonCatalog, detect_lessons, lessons_at
from ingestion.layout.blocks import Block, PageMeta, ParsedPage
from shared.config import get_config
from shared.namespaces import SYLLABUS
from shared.postgres import connect_once

_BOOK = """
SELECT subject_id::text, title, source, status, version FROM content.lessons WHERE id = %s
"""
_CHUNKS = """
SELECT chunk_id::text, page_from, heading_path, text, topic_ids::text[]
FROM ai.chunks WHERE lecture_id = ANY(%s::uuid[]) ORDER BY page_from, order_index
"""
_REFILE = (
    "UPDATE ai.chunks SET lecture_id = %(lesson)s WHERE chunk_id = ANY(%(chunks)s::uuid[])",
    "UPDATE ai.vector_index SET lecture_id = %(lesson)s WHERE chunk_id = ANY(%(chunks)s::uuid[])",
    "UPDATE ai.figures SET lecture_id = %(lesson)s "
    "WHERE lecture_id = %(book)s AND page BETWEEN %(page_from)s AND %(page_to)s",
    "UPDATE ai.page_transcriptions SET lecture_id = %(lesson)s "
    "WHERE lecture_id = %(book)s AND page BETWEEN %(page_from)s AND %(page_to)s",
    "UPDATE ai.topics SET lecture_id = %(lesson)s WHERE topic_id = ANY(%(topics)s::uuid[])",
)
_RECORD = """
INSERT INTO ai.lecture_ingestion
    (lecture_id, subject_id, version, status, chunk_count, figure_count, topic_count, ingested_at)
VALUES (%(lesson)s, %(subject)s, %(version)s, 'ready', %(chunk_count)s,
        (SELECT count(*) FROM ai.figures WHERE lecture_id = %(lesson)s), %(topic_count)s, now())
ON CONFLICT (lecture_id) DO UPDATE SET
    chunk_count = EXCLUDED.chunk_count, figure_count = EXCLUDED.figure_count,
    topic_count = EXCLUDED.topic_count, status = 'ready', ingested_at = now()
"""
_BOOK_TOPICS = "SELECT topic_id::text FROM ai.topics WHERE lecture_id = ANY(%s::uuid[])"
_ARCHIVE_BOOK = ("UPDATE content.lessons SET status = 'archived', ingestion_status = 'ready', "
                 "updated_at = now() WHERE id = %s")


def _pages(book_id: str, chunks: list[tuple]) -> list[dict]:
    blocks: dict[int, list[Block]] = defaultdict(list)
    for _, page, heading, text, _ in chunks:
        if heading and not any(b.text == heading for b in blocks[page]):
            blocks[page].append(Block(text=heading, type="heading", bbox=(0, 0, 0, 0)))
        blocks[page].append(Block(text=text, type="body", bbox=(0, 0, 0, 0)))
    return [ParsedPage(meta=PageMeta(doc_id=book_id, page=page, width=0, height=0),
                       blocks=page_blocks).model_dump()
            for page, page_blocks in sorted(blocks.items())]


def _filing(lessons: list[Lesson], chunks: list[tuple], book_topics: list[str]):
    """Each passage under the lesson its first page is in, and each topic under
    the lesson most of its passages went to."""
    passages: dict[str, list[str]] = defaultdict(list)
    votes: dict[str, Counter] = defaultdict(Counter)
    for chunk_id, page, _, _, topic_ids in chunks:
        owner = lessons_at(lessons, page)[0].lesson_id
        passages[owner].append(chunk_id)
        for topic_id in topic_ids or []:
            votes[topic_id][owner] += 1
    topics: dict[str, list[str]] = defaultdict(list)
    for topic_id in book_topics:
        owner = votes[topic_id].most_common(1)[0][0] if votes[topic_id] else lessons[0].lesson_id
        topics[owner].append(topic_id)
    return passages, topics


def _read(config, book_id: str, lesson_ids: Sequence[str] = ()):
    """The book and everything filed under it, or under its lessons already: a
    run that stopped after the database was refiled is finished by running
    again."""
    owners = [book_id, *lesson_ids]
    with connect_once(config) as conn, conn.cursor() as cur:
        cur.execute(_BOOK, (book_id,))
        book = cur.fetchone()
        cur.execute(_CHUNKS, (owners,))
        chunks = cur.fetchall()
        cur.execute(_BOOK_TOPICS, (owners,))
        topics = [row[0] for row in cur.fetchall()]
    return book, chunks, topics


def _plan(config, book_id: str) -> dict:
    book, chunks, _ = _read(config, book_id)
    if book is None:
        raise SystemExit(f"{book_id} is not a lesson")
    subject_id, title, source, status, _ = book
    if source != config.get("ingestion.books.lesson_source")[SYLLABUS]:
        raise SystemExit(f"{book_id} is a {source} lesson; only a ministry book is split")
    if status == "archived":
        raise SystemExit(f"{book_id} is archived: it has been split already")
    state = {"lecture_id": book_id, "parsed_pages": _pages(book_id, chunks),
             "namespace": SYLLABUS}
    lessons = detect_lessons(state, config)
    if not lessons:
        raise SystemExit(f"no lessons found in {book_id}; it stays one lesson")
    return {"book": book_id, "subject": subject_id, "title": title,
            "lessons": [asdict(lesson) for lesson in lessons]}


def _apply(config, plan: dict) -> None:
    from qdrant_client import models

    from shared.vector_store import upsert_client

    book_id = plan["book"]
    lessons = [Lesson(**lesson) for lesson in plan["lessons"]]
    book, chunks, book_topics = _read(config, book_id, [lesson.lesson_id for lesson in lessons])
    version = book[4]
    passages, topics = _filing(lessons, chunks, book_topics)
    catalog = LessonCatalog(config)
    catalog.register(subject_id=plan["subject"], book_title=plan["title"],
                     lessons=plan["lessons"])
    with connect_once(config) as conn, conn.cursor() as cur:
        for lesson in lessons:
            params = {"lesson": lesson.lesson_id, "book": book_id, "subject": plan["subject"],
                      "chunks": passages[lesson.lesson_id], "topics": topics[lesson.lesson_id],
                      "page_from": lesson.page_from, "page_to": lesson.page_to,
                      "version": version, "chunk_count": len(passages[lesson.lesson_id]),
                      "topic_count": len(topics[lesson.lesson_id])}
            for statement in _REFILE:
                cur.execute(statement, params)
            cur.execute(_RECORD, params)
        cur.execute(_ARCHIVE_BOOK, (book_id,))
        conn.commit()

    client = upsert_client(config, config.secret("QDRANT_URL"))
    for lesson in lessons:
        if passages[lesson.lesson_id]:
            client.set_payload(
                config.vector_store.get("collection"),
                payload={"lecture_id": lesson.lesson_id, "lesson_title": lesson.label,
                         "lesson_order": lesson.ordinal},
                points=passages[lesson.lesson_id], wait=True)
        if topics[lesson.lesson_id]:
            client.set_payload(config.vector_store.get("topics_collection"),
                               payload={"lecture_id": lesson.lesson_id},
                               points=topics[lesson.lesson_id], wait=True)
        catalog.finished(lesson.lesson_id, status="ready", errors=[], version=version)
    left = client.count(config.vector_store.get("collection"), exact=True,
                        count_filter=models.Filter(must=[models.FieldCondition(
                            key="lecture_id", match=models.MatchValue(value=book_id))])).count
    if left:
        raise SystemExit(f"{left} index points still name the whole book {book_id}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lesson", required=True, help="the whole-book lesson id")
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--provider", help="call the breakdown model through this provider "
                        "for this run, e.g. openrouter when Vertex billing is off")
    args = parser.parse_args(argv)
    config = get_config()
    if args.provider:
        config.override("llm.overrides.split_lessons.provider", args.provider)

    if not args.apply:
        plan = _plan(config, args.lesson)
        args.plan.write_text(json.dumps(plan, ensure_ascii=False, indent=1))
        for lesson in plan["lessons"]:
            print(f"{lesson['ordinal']:>3}  p{lesson['page_from']}-{lesson['page_to']:<4} "
                  f"u{lesson['unit_number'] or '-'}  {lesson['title']}")
        print(f"{len(plan['lessons'])} lessons; plan written to {args.plan}")
        print("dry run: nothing changed; pass --apply to file the book under them")
        return 0

    plan = json.loads(args.plan.read_text())
    if plan["book"] != args.lesson:
        raise SystemExit(f"{args.plan} is the plan for {plan['book']}, not {args.lesson}")
    _apply(config, plan)
    print(f"filed {plan['book']} under {len(plan['lessons'])} published lessons; "
          "the whole-book lesson is archived")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
