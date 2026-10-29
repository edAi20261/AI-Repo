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
from typing import Any, Mapping, Optional, Sequence

from shared.llm import get_llm, llm_key_available
from shared.namespaces import BACKGROUND, SYLLABUS
from shared.storage import get_uploader
from shared.structured import StructuredValidationFailure, call_structured, schema_hint

from . import lesson_pages
from .layout.blocks import ParsedPage
from .layout.bridge import parsed_pages
from .lesson import ID_NAMESPACE, UNTITLED
from .prompts import LESSON_BREAKDOWN_PROMPT, LESSON_TERMS_PROMPT
from .schemas import LessonBreakdown, LessonTerms
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


class TermColumnMissing(RuntimeError):
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
    term: Optional[int] = None
    apparatus: bool = False

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
    term: Optional[int] = None
    apparatus: bool = False


def _entries(breakdown: LessonBreakdown) -> list[_Entry]:
    lessons = [_Entry(item.title, item.start_page, unit.title, unit.unit_number,
                      term=unit.term)
               for unit in breakdown.units for item in unit.lessons]
    sections = [_Entry(item.title, item.start_page, section=True, term=item.term,
                       apparatus=item.apparatus)
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
               ratio: float, min_pages: int) -> list[str]:
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
            if to - start + 1 > max(ratio * typical, min_pages)]


def _validator(first_page: int, last_page: int, oversized_ratio: float,
               oversized_min_pages: int, max_term: int):
    questioned: set[str] = set()

    def check(breakdown: LessonBreakdown) -> None:
        items = [item for unit in breakdown.units for item in unit.lessons]
        if not items and not breakdown.sections:
            raise ValueError("a book holds at least one lesson or section; none were listed")
        apparatus = [item.title for item in items if item.apparatus]
        if apparatus:
            raise ValueError(f"these are not lessons: {apparatus}. Pages that only help a "
                             "reader find their way or check their work teach nothing of "
                             "their own; list each under sections with apparatus: true and "
                             "the page it starts on, and remove it from the lessons")
        empty = [unit.title for unit in breakdown.units if not unit.lessons]
        if empty:
            raise ValueError(f"these units list no lessons: {empty}. List the lessons "
                             "inside every unit")
        posing = [unit.title for unit in breakdown.units
                  if unit.title and len(unit.lessons) == 1
                  and _same(unit.lessons[0].title, unit.title)]
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
        if any(not item.title for item in [*items, *breakdown.sections]):
            raise ValueError("every lesson and section needs its title as printed in the "
                             "book; a lesson whose pages show only its number is titled "
                             "with that number as printed")
        check_terms([entry.term for entry in _entries(breakdown)], max_term)
        long = _oversized(_entries(breakdown), first_page, last_page, oversized_ratio,
                          oversized_min_pages)
        if long and questioned:
            logger.warning("long lessons kept after being questioned",
                           extra={"lessons": long})
        elif long:
            questioned.update(long)
            raise ValueError(
                "these lessons run far longer than a lesson in this book usually does: "
                f"{long}. Pages that belong to no single lesson — revision of the whole "
                "term, model exams, weekly or monthly assessments, answer keys — are not "
                "part of the lesson before them: list them under sections, each with the "
                "page it starts on")
    return check


def check_terms(terms: Sequence[Optional[int]], max_term: int) -> None:
    """Raise when the terms of a book's parts, in book order, cannot be right."""
    given = [term for term in terms if term is not None]
    unknown = sorted({term for term in given if not 1 <= term <= max_term})
    if unknown:
        raise ValueError(f"terms {unknown} are not terms of a school year: a term is a "
                         f"number from 1 to {max_term}, or null when the book does not say")
    if any(b < a for a, b in zip(given, given[1:])):
        raise ValueError("a book's terms run in order, the first term's parts before the "
                         f"second's; in book order the terms given were {given}")


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
            validate=_validator(numbers[0], numbers[-1], settings["oversized_lesson_ratio"],
                                settings["oversized_lesson_min_pages"], settings["max_term"]))
    except StructuredValidationFailure as error:
        raise BreakdownFailed(f"could not split the book into lessons: {error}") from error
    shared = settings["shared_pages"][state.get("namespace") or SYLLABUS]
    return lessons_from(breakdown, book_id=state["lecture_id"],
                        first_page=numbers[0], last_page=numbers[-1], shared_pages=shared,
                        term=state.get("term"))


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
                   unit_title="", page_from=pages[0], page_to=pages[-1],
                   term=state.get("term"))]


@dataclass(frozen=True)
class SplitLesson:
    ordinal: int
    title: str
    opening: str
    start_page: Optional[int] = None


def read_lesson_terms(opening_pages: str, outline: str, lessons: Sequence[SplitLesson],
                      config, *, llm: Any = None) -> dict[int, Optional[int]]:
    """The term of each lesson of a book that was split before terms were read."""
    ordinals = [lesson.ordinal for lesson in lessons]
    max_term = config.get("ingestion.books.max_term")

    def check(answer: LessonTerms) -> None:
        listed = sorted(item.ordinal for item in answer.lessons)
        if listed != sorted(ordinals):
            raise ValueError(f"list every lesson once, by its number: {sorted(ordinals)}; "
                             f"got {listed}")
        check_terms([answer.book_term], max_term)
        other = sorted({item.term for item in answer.lessons
                        if answer.book_term is not None and item.term is not None
                        and item.term != answer.book_term})
        if other:
            raise ValueError(f"the book is for term {answer.book_term}, yet lessons were given "
                             f"terms {other}; a book with parts in two terms is for the whole "
                             "year, and its book_term is null")
        check_terms([_term_of(item, answer) for item in
                     sorted(answer.lessons, key=lambda i: i.ordinal)], max_term)

    listing = "\n".join(f"[{lesson.ordinal}] page {lesson.start_page}: {lesson.title} | "
                         f"{lesson.opening}" for lesson in lessons)
    prompt = LESSON_TERMS_PROMPT.render(opening_pages=opening_pages, outline=outline,
                                        lessons=listing, schema=schema_hint(LessonTerms))
    answer = call_structured(LessonTerms, prompt, task=TASK, config=config,
                             llm=llm if llm is not None else get_llm(config, TASK),
                             label="lesson_terms", validate=check)
    return {item.ordinal: _term_of(item, answer) for item in answer.lessons}


def _term_of(item, answer: LessonTerms) -> Optional[int]:
    """A lesson of a book for one term is in that term, whether or not it was said
    again: the second-term Christian Religion G7 book came back with a term for
    six of its nineteen lessons."""
    return item.term if item.term is not None else answer.book_term


COPY, MERGE = "copy", "merge"


def _merged(entries: Sequence[_Entry]) -> list[_Entry]:
    merged: list[_Entry] = []
    for entry in entries:
        last = merged[-1] if merged else None
        if last and last.start_page == entry.start_page:
            merged[-1] = replace(last, title=f"{last.title} / {entry.title}",
                                 section=last.section and entry.section,
                                 apparatus=last.apparatus and entry.apparatus)
        else:
            merged.append(entry)
    return merged


def lessons_from(breakdown: LessonBreakdown, *, book_id: str, first_page: int,
                 last_page: int, shared_pages: str = COPY,
                 term: Optional[int] = None) -> list[Lesson]:
    """The book's lessons. ``term`` is the operator's word for the whole book, and
    wins over what the model read from it."""
    if shared_pages not in (COPY, MERGE):
        raise ValueError(f"shared_pages is {COPY!r} or {MERGE!r}, not {shared_pages!r}")
    entries = _entries(breakdown)
    if shared_pages == MERGE:
        entries = _merged(entries)
    return [Lesson(lesson_id=lesson_id_for(book_id, ordinal), ordinal=ordinal,
                   title=entry.title, unit_number=entry.unit_number,
                   unit_title=entry.unit_title, page_from=page_from, page_to=page_to,
                   term=entry.term if term is None else term, apparatus=entry.apparatus)
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


def shown(lesson: Mapping[str, Any], namespace: str) -> bool:
    """Whether students are shown this lesson of a book: a ministry book's lessons
    and sections are (a term's revision, its exams), its apparatus never is
    (contents, index, glossary, credits, answer keys)."""
    return namespace == SYLLABUS and not lesson.get("apparatus")


def store_lesson_pages(state: IngestionState, book_path: str, config, *,
                       uploader: Any = None) -> IngestionState:
    """Each shown lesson's pages, cut from the book and stored for ``finish`` to
    attach. Only ``prepare`` holds the book; a lesson whose pages could not be
    stored carries the reason instead, and is not published."""
    from .utils import pdf

    namespace = state.get("namespace") or SYLLABUS
    todo = [lesson for lesson in state.get("lessons", []) if shown(lesson, namespace)]
    if not todo:
        return state
    uploader = uploader if uploader is not None else get_uploader(config)
    book = pdf.open_document(book_path)
    try:
        for lesson in todo:
            try:
                stored = lesson_pages.store(
                    uploader, book, subject_id=state["subject_id"],
                    lesson_id=lesson["lesson_id"], title=lesson["title"],
                    first_page=lesson["page_from"], last_page=lesson["page_to"])
            except Exception as error:
                lesson["pdf_error"] = f"{type(error).__name__}: {error}"
                logger.error("lesson pages not stored: %s", lesson["pdf_error"],
                             extra={"lecture_id": lesson["lesson_id"]})
            else:
                lesson["pdf"] = asdict(stored)
    finally:
        book.close()
    return state


_PER_BOOK = frozenset({"chunks", "figures", "topics", "dependencies", "stats", "errors",
                       "ingestion_status", "lessons", "pages", "elements", "parsed_pages"})


def lesson_states(state: IngestionState) -> list[IngestionState]:
    shared_fields = {k: v for k, v in state.items() if k not in _PER_BOOK}
    namespace = state.get("namespace") or SYLLABUS
    states: list[IngestionState] = []
    for lesson in state.get("lessons", []):
        lesson_id = lesson["lesson_id"]
        states.append({
            **shared_fields,
            "lecture_id": lesson_id,
            "lesson_title": lesson["label"],
            "lesson_order": lesson["ordinal"],
            "shown": shown(lesson, namespace),
            "lesson_pdf": lesson.get("pdf"),
            "lesson_pdf_error": lesson.get("pdf_error"),
            "chunks": [c for c in state.get("chunks", []) if c.lecture_id == lesson_id],
            "figures": [f for f in state.get("figures", []) if f.lecture_id == lesson_id],
            "stats": {},
            "errors": [],
        })
    return states


def _upsert_lesson(with_term: bool) -> str:
    term_column, term_value, term_update = (
        (", term", ", %s", "\n    term = EXCLUDED.term,") if with_term else ("", "", ""))
    return f"""
INSERT INTO content.lessons
    (id, subject_id, source, title, description, unit_number, order_index{term_column},
     status, ingestion_status)
VALUES (%s, %s, %s, %s, %s, %s, %s{term_value}, 'draft', 'processing')
ON CONFLICT (id) DO UPDATE SET
    subject_id = EXCLUDED.subject_id,
    order_index = EXCLUDED.order_index,
    source = EXCLUDED.source,
    status = CASE WHEN content.lessons.status = 'published' AND EXCLUDED.source <> %s
                  THEN 'draft' ELSE content.lessons.status END,
    title = EXCLUDED.title,
    description = EXCLUDED.description,
    unit_number = EXCLUDED.unit_number,{term_update}
    ingestion_status = 'processing',
    ingestion_error = NULL,
    updated_at = now()
"""


# The backend owns content.lessons; its term column arrives with a Prisma
# migration (docs/BACKEND_INTEGRATION.md, "Lesson terms"), and ingestion must keep
# working on a database that does not have it yet.
_HAS_TERM = """
SELECT EXISTS (SELECT 1 FROM information_schema.columns
               WHERE table_schema = 'content' AND table_name = 'lessons'
                 AND column_name = 'term')
"""

_LOCK_SUBJECT = "SELECT pg_advisory_xact_lock(hashtext(%s))"

_SHOWN_IN_SUBJECT = """
subject_id = %(subject)s AND status <> 'archived' AND source <> %(background_source)s
"""

_NEXT_ORDER = f"""
SELECT COALESCE(MAX(order_index), -1) + 1 FROM content.lessons
WHERE {_SHOWN_IN_SUBJECT} AND NOT (id = ANY(%(ids)s::uuid[]))
"""

# Both terms' books share a subject, and each book was appended after the last:
# Christian Religion G7 showed the second term's lessons 0-18 and the first's
# 19-37. A book in the middle of a term keeps its place among that term's books.
_ORDER_BY_TERM = f"""
UPDATE content.lessons AS l SET order_index = o.position, updated_at = now()
FROM (SELECT id, row_number() OVER (ORDER BY term NULLS LAST, order_index, id) - 1 AS position
      FROM content.lessons WHERE {_SHOWN_IN_SUBJECT}) AS o
WHERE l.id = o.id AND l.order_index <> o.position
"""

_SET_TERM = """
UPDATE content.lessons SET term = %(term)s, updated_at = now()
WHERE id = %(lesson_id)s AND subject_id = %(subject)s AND term IS DISTINCT FROM %(term)s
"""

# Checked where the lesson is published, not by the caller: a lesson filed by any
# path (a book's finish, split_whole_books) is never shown to an empty page.
_PUBLISHES = """%(publish)s AND status = 'draft' AND source = %(shown_source)s
    AND EXISTS (SELECT 1 FROM content.lesson_attachments a
                WHERE a.lesson_id = content.lessons.id)"""

_FINISH_LESSON = f"""
UPDATE content.lessons
SET ingestion_status = %(settled)s, ingestion_error = %(error)s, ingested_at = now(),
    version = %(version)s, updated_at = now(),
    status = CASE WHEN {_PUBLISHES} THEN 'published' ELSE status END,
    published_at = CASE WHEN {_PUBLISHES} THEN now() ELSE published_at END
WHERE id = %(lesson_id)s
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
        shown_source = self._sources[SYLLABUS]
        ids = [lesson["lesson_id"] for lesson in lessons]
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(_LOCK_SUBJECT, (subject_id,))
                with_term = self._has_term(cur, subject_id, lessons)
                # A student sees a lesson's place among the lessons shown: the
                # first book after twelve background books in a subject was
                # numbered from 583.
                cur.execute(_NEXT_ORDER, {"subject": subject_id, "ids": ids,
                                          "background_source": self._sources[BACKGROUND]})
                base = cur.fetchone()[0]
                upsert = _upsert_lesson(with_term)
                for lesson in lessons:
                    source = (shown_source if shown(lesson, namespace)
                              else self._sources[BACKGROUND])
                    row = (lesson["lesson_id"], subject_id, source, lesson["title"],
                           book_title, lesson["unit_number"], base + lesson["ordinal"])
                    if with_term:
                        row = (*row, lesson.get("term"))
                    cur.execute(upsert, (*row, shown_source))
                if with_term:
                    self._order_by_term(cur, subject_id)
            conn.commit()

    def attach(self, lesson_id: str, *, pdf: Optional[Mapping[str, Any]],
               reason: Optional[str]) -> None:
        if pdf is None:
            raise lesson_pages.PagesNotAttached(
                f"the lesson's pages were not stored: {reason or 'no reason recorded'}")
        uploader = lesson_pages.uploader_id(self._config)
        with self._connect() as conn:
            with conn.cursor() as cur:
                lesson_pages.record(cur, pdf, lesson_id=lesson_id, uploader=uploader)
            conn.commit()

    def set_terms(self, subject_id: str, terms: dict[str, Optional[int]]) -> int:
        """Store the terms of a subject's lessons and order the subject by them."""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(_LOCK_SUBJECT, (subject_id,))
                cur.execute(_HAS_TERM)
                row = cur.fetchone()
                if not (row and row[0]):
                    raise TermColumnMissing(
                        "content.lessons has no term column yet; the backend migration in "
                        "docs/BACKEND_INTEGRATION.md (\"Lesson terms\") adds it")
                changed = 0
                for lesson_id, term in terms.items():
                    cur.execute(_SET_TERM, {"term": term, "lesson_id": lesson_id,
                                            "subject": subject_id})
                    changed += max(cur.rowcount, 0)
                self._order_by_term(cur, subject_id)
            conn.commit()
        return changed

    def _order_by_term(self, cur, subject_id: str) -> None:
        cur.execute(_ORDER_BY_TERM, {"subject": subject_id,
                                     "background_source": self._sources[BACKGROUND]})

    @staticmethod
    def _has_term(cur, subject_id: str, lessons: Sequence[dict]) -> bool:
        cur.execute(_HAS_TERM)
        row = cur.fetchone()
        present = bool(row and row[0])
        termed = sum(1 for lesson in lessons if lesson.get("term") is not None)
        if not present and termed:
            logger.warning("content.lessons has no term column; the term of these lessons "
                           "is not stored and the subject is not ordered by term",
                           extra={"subject_id": subject_id, "lessons_with_term": termed})
        return present

    def finished(self, lesson_id: str, *, status: str, errors: Sequence[str],
                 version: int) -> None:
        # A lesson that needs review is kept from students until someone has
        # looked; a teacher's draft or archive is never overridden.
        params = {"settled": _INGESTION_STATUS.get(status, "ready"),
                  "error": "; ".join(errors) or None, "version": version,
                  "lesson_id": lesson_id, "publish": status == "ready",
                  "shown_source": self._sources[SYLLABUS]}
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(_FINISH_LESSON, params)
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
