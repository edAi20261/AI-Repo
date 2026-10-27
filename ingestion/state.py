"""Pipeline state and the domain objects carried through it.

One lesson flows through extract -> chunk -> embed -> topics -> index as a single
:class:`IngestionState`. The dataclasses map onto the ``ai.*`` tables; field names
match the columns so the store layer is a straight copy.
"""

from __future__ import annotations

from shared.namespaces import SYLLABUS

import logging

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, Optional, TypedDict


class Discipline(str, Enum):
    """Kind of subject a lesson belongs to. Derived from the subject name/code
    (the backend has no discipline field). ``math`` forces every page to vision."""

    MATH = "math"
    SCIENCE = "science"
    LANGUAGE = "language"
    SOCIAL_STUDIES = "social_studies"
    OTHER = "other"


# How a page's text was recovered, set by the extract routing:
# "digital" = PyMuPDF; "scanned"/"math" = render + vision.
PageRoute = Literal["digital", "scanned", "math", "text"]


@dataclass
class Element:
    """A heading or paragraph from a page. Headings build the ``heading_path``
    stored on each chunk."""

    text: str
    page: int
    is_heading: bool = False
    level: int = 0  # heading depth; 0 for body text
    font_size: float = 0.0
    route: PageRoute = "digital"
    language: Optional[str] = None


@dataclass
class PageResult:
    """The extract stage's outcome for one page."""

    number: int
    route: PageRoute
    language: str
    char_count: int
    elements: list[Element] = field(default_factory=list)


@dataclass
class Figure:
    """A cropped figure for ``ai.figures``. ``caption``/``alt_text`` come from the
    vision model; ``chunk_id``/``topic_ids`` are filled in at index time."""

    lecture_id: str
    page: int
    s3_uri: str
    bbox: tuple[float, float, float, float]
    caption: Optional[str] = None
    alt_text: Optional[str] = None
    chunk_id: Optional[str] = None
    topic_ids: list[str] = field(default_factory=list)


# ai.chunks.heading_path is varchar(500). A heading longer than that is not a
# heading — it is a paragraph a page styled large enough to look like one — but
# the database does not care which it is, and the insert takes the whole book
# with it: a 106-page French textbook was read in full and wrote nothing.
logger = logging.getLogger(__name__)

MAX_HEADING_PATH = 500


@dataclass
class Chunk:
    """A ~500-800 token unit of lesson text. Mirrors ``ai.chunks`` plus the filter
    fields the Qdrant payload needs. ``embedding`` is transient (Qdrant only)."""

    lecture_id: str
    text: str
    heading_path: str
    page_from: int
    page_to: int
    language: str
    order_index: int
    version: int = 1
    topic_ids: list[str] = field(default_factory=list)
    chunk_id: Optional[str] = None       # set at write; keys row, manifest and point
    embedding: Optional[list[float]] = None
    # {term_id: weight} — the lexical half of a hybrid search. Transient, like
    # embedding: it goes to Qdrant and never to ai.chunks.
    sparse: Optional[dict[int, float]] = None
    namespace: str = "syllabus"          # below: carried for indexing, not on ai.chunks
    subject_id: Optional[str] = None
    grade: Optional[str] = None
    # Readable citation labels (Qdrant payload only, not on ai.chunks).
    subject_name: Optional[str] = None
    lesson_title: Optional[str] = None
    lesson_order: Optional[int] = None

    def __post_init__(self) -> None:
        if self.heading_path and len(self.heading_path) > MAX_HEADING_PATH:
            logger.warning(
                "heading path too long for its column; keeping the head of it",
                extra={"lecture_id": self.lecture_id, "page": self.page_from,
                       "length": len(self.heading_path)})
            self.heading_path = self.heading_path[:MAX_HEADING_PATH - 1].rstrip() + "…"

    def for_embedding(self) -> str:
        """What this chunk is searched by — never what a student reads.

        A chunk carries no sign of where it came from. "مهدت لنهضة علمية" is
        four words that name no lesson, no unit and no page, so a query naming
        any of them has nothing in the vector to match. Prefixing the context
        the chunk already knows about itself puts it there.

        The prefix is built here rather than at the call sites that embed, so a
        second one cannot embed the bare text and quietly index a chunk the
        rest of the corpus is not comparable to.
        """
        parts = [self.subject_name, self.lesson_title]
        if self.heading_path:
            parts.append(f"Section: {self.heading_path}")
        if self.page_from:
            parts.append(f"Page {self.page_from}")
        named = [p for p in parts if p]
        if not named:
            return self.text
        return ". ".join(named) + ".\n" + self.text


@dataclass
class Topic:
    """A skill for ``ai.topics``. ``reused`` marks a dedupe match that keeps its
    existing ``topic_id`` (so mastery history stays attached)."""

    label: str
    subject_id: str
    lecture_id: str
    description: Optional[str] = None
    bloom_level: Optional[str] = None
    topic_id: Optional[str] = None
    label_embedding: Optional[list[float]] = None
    reused: bool = False
    namespace: str = SYLLABUS


@dataclass
class Dependency:
    """A prerequisite edge for ``ai.topic_dependencies``: ``topic_id`` requires
    ``prereq_topic_id`` first."""

    topic_id: str
    prereq_topic_id: str
    strength: str = "required"       # ai.dependency_strength: required | related
    rationale: Optional[str] = None


class IngestionState(TypedDict, total=False):
    """State for one lesson. ``total=False``; required inputs are set by
    :func:`new_state`."""

    # inputs
    lecture_id: str          # content.lessons.id
    subject_id: str
    grade: Optional[str]
    subject_name: Optional[str]   # readable citation labels (payload only)
    lesson_title: Optional[str]
    lesson_order: Optional[int]
    source_uri: str          # S3 or local URI; empty when the lesson body is the source
    source_text: Optional[str]    # content.lessons.text_content, when there is no file
    namespace: str           # "syllabus" for permanent content; session_* for TTL
    version: int             # bumped on re-ingestion
    language_hint: Optional[str]
    subject_is_math: bool    # force every page to the vision route
    term: Optional[int]      # the operator's term for the whole book; None to read it
    ingestion_status: Optional[str]   # set to needs_review on an unreadable page

    # accumulated by the stages
    pages: list[PageResult]
    parsed_pages: list[dict]        # layout-parsed pages, when ingestion.layout is on
    elements: list[Element]  # flattened across pages, in reading order
    figures: list[Figure]
    chunks: list[Chunk]
    topics: list[dict[str, Any]]
    dependencies: list[dict[str, Any]]
    stats: dict[str, Any]
    errors: list[str]
    book_id: str
    lessons: list[dict[str, Any]]


def new_state(
    *,
    lecture_id: str,
    subject_id: str,
    source_uri: str = "",
    source_text: Optional[str] = None,
    grade: Optional[str] = None,
    subject_name: Optional[str] = None,
    lesson_title: Optional[str] = None,
    lesson_order: Optional[int] = None,
    namespace: str = "syllabus",
    version: int = 1,
    language_hint: Optional[str] = None,
    subject_is_math: bool = False,
    term: Optional[int] = None,
) -> IngestionState:
    """Build a fresh state for one lesson with its lists initialised empty."""
    if not source_uri and not source_text:
        raise ValueError("a lesson needs either a source_uri or source_text to ingest")
    return IngestionState(
        lecture_id=lecture_id,
        subject_id=subject_id,
        grade=grade,
        subject_name=subject_name,
        lesson_title=lesson_title,
        lesson_order=lesson_order,
        source_uri=source_uri,
        source_text=source_text,
        namespace=namespace,
        version=version,
        language_hint=language_hint,
        subject_is_math=subject_is_math,
        term=term,
        ingestion_status=None,
        pages=[],
        elements=[],
        figures=[],
        chunks=[],
        topics=[],
        dependencies=[],
        stats={},
        errors=[],
    )
