"""A book, split into the lessons it teaches.

A book is uploaded under its subject. Each lesson in it becomes its own row in
``content.lessons``, and every chunk, figure and topic is filed under the lesson
whose pages it came from, so a student on a lesson's page is answered from that
lesson.
"""

from __future__ import annotations

import logging
import statistics
import uuid
from dataclasses import asdict, dataclass, replace
from typing import Any, Optional, Sequence

from shared.llm import get_llm, llm_key_available
from shared.namespaces import BACKGROUND, SYLLABUS
from shared.structured import StructuredValidationFailure, call_structured, schema_hint

from .layout.blocks import ParsedPage
from .layout.bridge import parsed_pages
from .lesson import ID_NAMESPACE, UNTITLED
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


@dataclass(frozen=True)
class _Entry:
    title: str
    start_page: int
    unit_title: str = ""
    unit_number: Optional[int] = None
    section: bool = False


def _entries(breakdown: LessonBreakdown) -> list[_Entry]:
    lessons = [_Entry(item.title, item.start_page, unit.title, unit.unit_number)
               for unit in breakdown.units for item in unit.lessons]
    sections = [_Entry(item.title, item.start_page, section=True)
                for item in breakdown.sections]
    return sorted(lessons + sections, key=lambda entry: entry.start_page)


def _spans(entries: Sequence[_Entry], first_page: int,
           last_page: int) -> list[tuple[int, int]]:
    opening = entries[0].start_page if entries else first_page
    spans = []
    for n, entry in enumerate(entries):
        start = entry.start_page
        later = [other.start_page for other in entries[n + 1:] if other.start_page > start]
        spans.append((first_page if start == opening else start,
                      later[0] - 1 if later else last_page))
    return spans


def _oversized(entries: Sequence[_Entry], first_page: int, last_page: int,
               ratio: float) -> list[str]:
    spans = _spans(entries, first_page, last_page)
    lessons: dict[tuple[int, int], list[str]] = {}
    for entry, span in zip(entries, spans):
        if not entry.section:
            lessons.setdefault(span, []).append(entry.title)
    if len(lessons) < 3:
        return []
    typical = statistics.median(to - start + 1 for start, to in lessons)
    return [f"{' / '.join(titles)} (pages {start}-{to}, {to - start + 1} pages)"
            for (start, to), titles in lessons.items()
            if to - start + 1 > ratio * typical]


def _validator(first_page: int, last_page: int, oversized_ratio: float):
    def check(breakdown: LessonBreakdown) -> None:
        items = [item for unit in breakdown.units for item in unit.lessons]
        if not items and not breakdown.sections:
            raise ValueError("a book holds at least one lesson or section; none were listed")
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
        section_starts = [item.start_page for item in breakdown.sections]
        outside = [s for s in starts + section_starts if not first_page <= s <= last_page]
        if outside:
            raise ValueError(f"start pages {outside} are outside the book "
                             f"(pages {first_page}-{last_page})")
        if any(b < a for a, b in zip(starts, starts[1:])):
            raise ValueError(f"start pages must not go back in book order; got {starts}")
        if any(b <= a for a, b in zip(section_starts, section_starts[1:])):
            raise ValueError("each section starts on its own page, after the one before "
                             f"it; got {section_starts}")
        both = sorted(set(starts) & set(section_starts))
        if both:
            raise ValueError(f"pages {both} are given as the start of a lesson and of a "
                             "section; a page starts one or the other")
        if any(not item.title for item in [*items, *breakdown.sections]):
            raise ValueError("every lesson and section needs its title as printed in the "
                             "book; a lesson whose pages show only its number is titled "
                             "with that number as printed")
        long = _oversized(_entries(breakdown), first_page, last_page, oversized_ratio)
        if long:
            raise ValueError(
                "these lessons run far longer than a lesson in this book usually does: "
                f"{long}. Pages that belong to no single lesson — revision of the whole "
                "term, model exams, weekly or monthly assessments, answer keys — are not "
                "part of the lesson before them: list them under sections, each with the "
                "page it starts on")
    return check


def _same(a: str, b: str) -> bool:
    return " ".join(a.casefold().split()) == " ".join(b.casefold().split())


def detect_lessons(state: IngestionState, config, *, llm: Any = None) -> list[Lesson]:
    pages = parsed_pages(state.get("parsed_pages") or [])
    if not pages:
        return []
    spec = config.llm_for(TASK)
    settings = config.get("ingestion.books")
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
            label="lesson_breakdown",
            validate=_validator(numbers[0], numbers[-1], settings["oversized_lesson_ratio"]))
    except StructuredValidationFailure as error:
        raise BreakdownFailed(f"could not split the book into lessons: {error}") from error
    shared = settings["shared_pages"][state.get("namespace") or SYLLABUS]
    return lessons_from(breakdown, book_id=state["lecture_id"],
                        first_page=numbers[0], last_page=numbers[-1], shared_pages=shared)


def lessons_for(state: IngestionState, config, *, llm: Any = None) -> list[Lesson]:
    """The book's lessons; for a background book that has none to find, itself.

    A question bank or a revision booklet is often not split into lessons at
    all. As background it is searched across its whole subject and never shown,
    so it is filed as one lesson rather than failing the book. A ministry book
    whose lessons cannot be found still fails: its lessons are what the app
    shows."""
    try:
        return detect_lessons(state, config, llm=llm)
    except BreakdownFailed:
        if state.get("namespace") != BACKGROUND:
            raise
        logger.warning("background book has no lessons to find; filed as one",
                       extra={"lecture_id": state["lecture_id"]}, exc_info=True)
    pages = sorted(p.meta.page for p in parsed_pages(state.get("parsed_pages") or []))
    if not pages:
        return []
    return [Lesson(lesson_id=lesson_id_for(state["lecture_id"], 0), ordinal=0,
                   title=state.get("lesson_title") or UNTITLED, unit_number=None,
                   unit_title="", page_from=pages[0], page_to=pages[-1])]


COPY, MERGE = "copy", "merge"


def _merged(entries: Sequence[_Entry]) -> list[_Entry]:
    merged: list[_Entry] = []
    for entry in entries:
        last = merged[-1] if merged else None
        if last and not entry.section and not last.section \
                and last.start_page == entry.start_page:
            merged[-1] = replace(last, title=f"{last.title} / {entry.title}")
        else:
            merged.append(entry)
    return merged


def lessons_from(breakdown: LessonBreakdown, *, book_id: str, first_page: int,
                 last_page: int, shared_pages: str = COPY) -> list[Lesson]:
    if shared_pages not in (COPY, MERGE):
        raise ValueError(f"shared_pages is {COPY!r} or {MERGE!r}, not {shared_pages!r}")
    entries = _entries(breakdown)
    if shared_pages == MERGE:
        entries = _merged(entries)
    return [Lesson(lesson_id=lesson_id_for(book_id, ordinal), ordinal=ordinal,
                   title=entry.title, unit_number=entry.unit_number,
                   unit_title=entry.unit_title, page_from=page_from, page_to=page_to)
            for ordinal, (entry, (page_from, page_to))
            in enumerate(zip(entries, _spans(entries, first_page, last_page)))]


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
        self._sources = config.get("ingestion.books.lesson_source")

    def _connect(self):
        from shared.postgres import connect_once

        return connect_once(self._config)

    def register(self, *, subject_id: str, book_title: Optional[str],
                 lessons: Sequence[dict], namespace: str = SYLLABUS) -> None:
        source = self._sources[namespace]
        ids = [lesson["lesson_id"] for lesson in lessons]
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(_LOCK_SUBJECT, (subject_id,))
                cur.execute(_NEXT_ORDER, (subject_id, ids))
                base = cur.fetchone()[0]
                for lesson in lessons:
                    cur.execute(_UPSERT_LESSON, (
                        lesson["lesson_id"], subject_id, source, lesson["title"],
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
