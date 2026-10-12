"""A book, split into the lessons it teaches.

A book is uploaded under its subject. Each lesson in it becomes its own row in
``content.lessons``, and every chunk, figure and topic is filed under the lesson
whose pages it came from, so a student on a lesson's page is answered from that
lesson.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import asdict, dataclass, replace
from typing import Any, Optional, Sequence

from shared.llm import get_llm, llm_key_available
from shared.structured import call_structured, schema_hint

from .layout.blocks import ParsedPage
from .layout.bridge import parsed_pages
from .lesson import ID_NAMESPACE
from .prompts import LESSON_BREAKDOWN_PROMPT
from .schemas import LessonBreakdown
from .state import IngestionState

logger = logging.getLogger(__name__)

TASK = "split_lessons"
OPENING_PAGES = 12
OPENING_PAGE_CHARS = 2500
OUTLINE_LINE_CHARS = 420
OUTLINE_BODY_BLOCKS = 2
HEADING_TYPES = frozenset({"heading", "subheading"})
STALE_SEARCH = 400

_INGESTION_STATUS = {"ready": "ready", "needs_review": "ready", "failed": "failed"}


class BreakdownFailed(RuntimeError):
    pass


@dataclass
class Lesson:
    lesson_id: str
    ordinal: int
    title: str
    unit_number: Optional[int]
    unit_title: str
    page_from: int
    page_to: int

    @property
    def label(self) -> str:
        return f"{self.unit_title} — {self.title}" if self.unit_title else self.title


def lesson_id_for(book_id: str, ordinal: int) -> str:
    return str(uuid.uuid5(ID_NAMESPACE, f"lesson:{book_id}:{ordinal}"))


def _page_text(page: ParsedPage) -> str:
    return " ".join(b.text for b in page.blocks if not b.is_furniture)


def _outline_line(page: ParsedPage) -> str:
    content = [b for b in page.blocks if not b.is_furniture]
    headings = [b.text for b in content if b.type in HEADING_TYPES]
    opening = [b.text for b in content if b.type not in HEADING_TYPES][:OUTLINE_BODY_BLOCKS]
    line = " | ".join([*headings, *opening])
    return f"[{page.meta.page}] {line[:OUTLINE_LINE_CHARS]}"


def book_outline(pages: Sequence[ParsedPage]) -> tuple[str, str]:
    ordered = sorted(pages, key=lambda p: p.meta.page)
    opening = "\n\n".join(
        f"[{p.meta.page}]\n{_page_text(p)[:OPENING_PAGE_CHARS]}" for p in ordered[:OPENING_PAGES])
    outline = "\n".join(_outline_line(p) for p in ordered)
    return opening, outline


def _validator(first_page: int, last_page: int):
    def check(breakdown: LessonBreakdown) -> None:
        items = [item for unit in breakdown.units for item in unit.lessons]
        if not items:
            raise ValueError("a book teaches at least one lesson; none were listed")
        empty = [unit.title for unit in breakdown.units if not unit.lessons]
        if empty:
            raise ValueError(f"these units list no lessons: {empty}. List the lessons "
                             "inside every unit")
        posing = [unit.title for unit in breakdown.units
                  if unit.title and any(_same(item.title, unit.title) for item in unit.lessons)]
        if posing:
            raise ValueError(f"a unit is not a lesson, but these units are listed as their "
                             f"own lesson: {posing}. List the lessons inside them")
        starts = [item.start_page for item in items]
        outside = [s for s in starts if not first_page <= s <= last_page]
        if outside:
            raise ValueError(f"start pages {outside} are outside the book "
                             f"(pages {first_page}-{last_page})")
        if any(b < a for a, b in zip(starts, starts[1:])):
            raise ValueError(f"start pages must not go back in book order; got {starts}")
        if any(not item.title for item in items):
            raise ValueError("every lesson needs its title as printed in the book")
    return check


def _same(a: str, b: str) -> bool:
    return " ".join(a.casefold().split()) == " ".join(b.casefold().split())


def detect_lessons(state: IngestionState, config, *, llm: Any = None) -> list[Lesson]:
    pages = parsed_pages(state.get("parsed_pages") or [])
    if not pages:
        return []
    spec = config.llm_for(TASK)
    if llm is None:
        if not llm_key_available(config, spec):
            logger.warning("no model for the lesson breakdown; the book stays one lesson",
                           extra={"lecture_id": state["lecture_id"]})
            return []
        llm = get_llm(config, TASK)
    numbers = sorted(p.meta.page for p in pages)
    opening, outline = book_outline(pages)
    prompt = LESSON_BREAKDOWN_PROMPT.render(
        page_count=len(numbers), opening_pages=opening, outline=outline,
        schema=schema_hint(LessonBreakdown))
    try:
        breakdown = call_structured(
            LessonBreakdown, prompt, task=TASK, config=config, llm=llm,
            label="lesson_breakdown", validate=_validator(numbers[0], numbers[-1]))
    except Exception as error:
        raise BreakdownFailed(f"could not split the book into lessons: {error}") from error
    return lessons_from(breakdown, book_id=state["lecture_id"],
                        first_page=numbers[0], last_page=numbers[-1])


def lessons_from(breakdown: LessonBreakdown, *, book_id: str,
                 first_page: int, last_page: int) -> list[Lesson]:
    items = [(unit, item) for unit in breakdown.units for item in unit.lessons]
    opening = items[0][1].start_page if items else first_page
    lessons: list[Lesson] = []
    for ordinal, (unit, item) in enumerate(items):
        start = item.start_page
        page_from = first_page if start == opening else start
        later = [other.start_page for _, other in items[ordinal + 1:] if other.start_page > start]
        page_to = later[0] - 1 if later else last_page
        lessons.append(Lesson(
            lesson_id=lesson_id_for(book_id, ordinal), ordinal=ordinal, title=item.title,
            unit_number=unit.unit_number, unit_title=unit.title,
            page_from=page_from, page_to=page_to))
    return lessons


def lessons_at(lessons: Sequence[Lesson], page: int) -> list[Lesson]:
    owners = [lesson for lesson in lessons if lesson.page_from <= page <= lesson.page_to]
    if owners:
        return owners
    return [lessons[0] if page < lessons[0].page_from else lessons[-1]]


def _chunk_for(chunk, lesson: Lesson):
    return replace(chunk, lecture_id=lesson.lesson_id, lesson_title=lesson.label,
                   lesson_order=lesson.ordinal, chunk_id=None,
                   topic_ids=list(chunk.topic_ids))


def file_under_lessons(state: IngestionState, lessons: Sequence[Lesson]) -> IngestionState:
    if not lessons:
        return state
    state["chunks"] = [_chunk_for(chunk, lesson)
                       for chunk in state.get("chunks", [])
                       for lesson in lessons_at(lessons, chunk.page_from)]
    state["figures"] = [replace(figure, lecture_id=lesson.lesson_id, chunk_id=None,
                                topic_ids=list(figure.topic_ids))
                        for figure in state.get("figures", [])
                        for lesson in lessons_at(lessons, figure.page)]
    state["lessons"] = [{**asdict(lesson), "label": lesson.label} for lesson in lessons]
    return state


_PER_BOOK = frozenset({"chunks", "figures", "topics", "dependencies", "stats", "errors",
                       "ingestion_status", "lessons", "pages", "elements", "parsed_pages"})


def lesson_states(state: IngestionState) -> list[IngestionState]:
    shared_fields = {k: v for k, v in state.items() if k not in _PER_BOOK}
    states: list[IngestionState] = []
    for lesson in state.get("lessons", []):
        lesson_id = lesson["lesson_id"]
        states.append({
            **shared_fields,
            "lecture_id": lesson_id,
            "lesson_title": lesson["label"],
            "lesson_order": lesson["ordinal"],
            "chunks": [c for c in state.get("chunks", []) if c.lecture_id == lesson_id],
            "figures": [f for f in state.get("figures", []) if f.lecture_id == lesson_id],
            "stats": {},
            "errors": [],
        })
    return states


_UPSERT_LESSON = """
INSERT INTO content.lessons
    (id, subject_id, source, title, description, unit_number, order_index,
     status, ingestion_status)
VALUES (%s, %s, %s, %s, %s, %s, %s, 'draft', 'processing')
ON CONFLICT (id) DO UPDATE SET
    title = EXCLUDED.title,
    description = EXCLUDED.description,
    unit_number = EXCLUDED.unit_number,
    ingestion_status = 'processing',
    ingestion_error = NULL,
    updated_at = now()
"""

_LOCK_SUBJECT = "SELECT pg_advisory_xact_lock(hashtext(%s))"

_NEXT_ORDER = """
SELECT COALESCE(MAX(order_index), -1) + 1 FROM content.lessons
WHERE subject_id = %s AND NOT (id = ANY(%s::uuid[]))
"""

_FINISH_LESSON = """
UPDATE content.lessons
SET ingestion_status = %s, ingestion_error = %s, ingested_at = now(),
    version = %s, updated_at = now()
WHERE id = %s
"""

_EXISTING = "SELECT id::text FROM content.lessons WHERE id = ANY(%s::uuid[])"

_ARCHIVE = """
UPDATE content.lessons SET status = 'archived', updated_at = now()
WHERE id = ANY(%s::uuid[])
"""


class LessonCatalog:
    def __init__(self, config) -> None:
        self._config = config
        self._source = config.get("ingestion.books.lesson_source", "ministry_official")

    def _connect(self):
        from shared.postgres import connect_once

        return connect_once(self._config)

    def register(self, *, subject_id: str, book_title: Optional[str],
                 lessons: Sequence[dict]) -> None:
        ids = [lesson["lesson_id"] for lesson in lessons]
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(_LOCK_SUBJECT, (subject_id,))
                cur.execute(_NEXT_ORDER, (subject_id, ids))
                base = cur.fetchone()[0]
                for lesson in lessons:
                    cur.execute(_UPSERT_LESSON, (
                        lesson["lesson_id"], subject_id, self._source, lesson["title"],
                        book_title, lesson["unit_number"], base + lesson["ordinal"]))
            conn.commit()

    def finished(self, lesson_id: str, *, status: str, errors: Sequence[str],
                 version: int) -> None:
        settled = _INGESTION_STATUS.get(status, "ready")
        error = "; ".join(errors) or None
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(_FINISH_LESSON, (settled, error, version, lesson_id))
            conn.commit()

    def stale(self, book_id: str, kept: int) -> list[str]:
        candidates = [lesson_id_for(book_id, n) for n in range(kept, kept + STALE_SEARCH)]
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(_EXISTING, (candidates,))
                return [row[0] for row in cur.fetchall()]

    def archive(self, lesson_ids: Sequence[str]) -> None:
        if not lesson_ids:
            return
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(_ARCHIVE, (list(lesson_ids),))
            conn.commit()
