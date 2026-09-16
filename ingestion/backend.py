"""Backend access + the lesson-id resolver.

The pipeline needs a source URI + subject facts, not a lesson id, so the id is
resolved first (lesson -> source media -> URL, subject -> grade/language/
discipline). Behind :class:`BackendClient`: a real HTTP impl and a fixture impl
(local manifest) so the resolver runs with no backend, database or S3.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from shared.service_http import DEFAULT_TIMEOUT, BackendTransport

from .config import Config
from .state import Discipline
from .utils.grades import to_platform_grade


@dataclass
class LessonSource:
    """Everything the pipeline needs about one lesson, resolved from its id."""

    lesson_id: str
    source_uri: str            # signed S3 URL, or a local file:// URI (fixtures)
    subject_id: str
    grade: Optional[str]
    language: Optional[str]
    discipline: Discipline
    # Readable labels for citations (denormalised from the lesson/subject reads).
    subject_name: Optional[str] = None
    lesson_title: Optional[str] = None
    lesson_order: Optional[int] = None
    text_content: Optional[str] = None   # set instead of source_uri for body-text lessons

    @property
    def subject_is_math(self) -> bool:
        return self.discipline is Discipline.MATH


# ===========================================================================
# Client interface
# ===========================================================================

class BackendClient(ABC):
    """The four backend reads ingestion needs."""

    @abstractmethod
    def list_pending_lesson_ids(self, subject_id: str | None = None) -> list[str]:
        """Lessons awaiting ingestion (published, not yet ingested) — backfill."""

    @abstractmethod
    def get_lesson(self, lesson_id: str) -> dict[str, Any]:
        """One lesson record (subject_id, status, attachments, ...)."""

    @abstractmethod
    def get_media_url(self, media_id: str) -> str:
        """The source file's URI — a signed S3 URL, or a local file:// URI."""

    @abstractmethod
    def get_subject(self, subject_id: str) -> dict[str, Any]:
        """Subject facts: grade_level, language, name, code (+ discipline if any)."""


class HttpBackendClient(BackendClient):
    """Calls the real platform APIs."""

    def __init__(self, base_url: str, service_token: str | None = None,
                 timeout: float = DEFAULT_TIMEOUT) -> None:
        self._http = BackendTransport(base_url, service_token, timeout=timeout)

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        return self._http.get(path, params)

    def list_pending_lesson_ids(self, subject_id: str | None = None) -> list[str]:
        params: dict[str, Any] = {"status": "published", "ingested": False}
        if subject_id:
            params["subject_id"] = subject_id
        return [row["id"] for row in self._get("/lessons", params)]

    def get_lesson(self, lesson_id: str) -> dict[str, Any]:
        return self._get(f"/lessons/{lesson_id}")

    def get_media_url(self, media_id: str) -> str:
        return self._get(f"/media/{media_id}/url")["signed_url"]

    def get_subject(self, subject_id: str) -> dict[str, Any]:
        return self._get(f"/subjects/{subject_id}")


class FixtureBackendClient(BackendClient):
    """Reads a local YAML/JSON manifest instead of a backend. Makes the resolver
    and the whole pipeline runnable offline.

    Manifest shape::

        lessons:
          lesson-1:
            path: samples/math_g8_lesson3.pdf   # local source PDF
            subject_id: subj-math-8
            status: published                    # optional (default published)
            ingested: false                      # optional (default false)
            subject:
              grade_level: g8
              language: ar
              name: Mathematics                  # discipline derived from name/code
              code: MATH8
              discipline: math                   # optional; wins over name/code

    Local ``path`` is resolved relative to the manifest file's directory.
    """

    def __init__(self, manifest_path: str | Path) -> None:
        self._path = Path(manifest_path)
        self._root = self._path.parent
        self._lessons = self._load().get("lessons", {})

    def _load(self) -> dict[str, Any]:
        text = self._path.read_text(encoding="utf-8")
        if self._path.suffix.lower() in (".yaml", ".yml"):
            import yaml

            return yaml.safe_load(text) or {}
        return json.loads(text)

    def _lesson(self, lesson_id: str) -> dict[str, Any]:
        if lesson_id not in self._lessons:
            raise KeyError(f"lesson {lesson_id!r} not in fixture {self._path}")
        return self._lessons[lesson_id]

    def list_pending_lesson_ids(self, subject_id: str | None = None) -> list[str]:
        out = []
        for lid, lesson in self._lessons.items():
            if lesson.get("status", "published") != "published":
                continue
            if lesson.get("ingested", False):
                continue
            if subject_id and lesson.get("subject_id") != subject_id:
                continue
            out.append(lid)
        return out

    def get_lesson(self, lesson_id: str) -> dict[str, Any]:
        lesson = self._lesson(lesson_id)
        # Shape it like the real GET /lessons/{id}: the source is the sole
        # attachment, so the same resolver code path serves both clients.
        return {
            "id": lesson_id,
            "subject_id": lesson["subject_id"],
            "status": lesson.get("status", "published"),
            "title": lesson.get("title"),
            "order_index": lesson.get("order_index"),
            "attachments": [{"media_id": lesson_id, "order_index": 0}],
        }

    def get_media_url(self, media_id: str) -> str:
        # media_id == lesson_id in the fixture; return the local file as a URI.
        path = Path(self._lesson(media_id)["path"])
        if not path.is_absolute():
            path = (self._root / path).resolve()
        return path.as_uri()

    def get_subject(self, subject_id: str) -> dict[str, Any]:
        for lesson in self._lessons.values():
            if lesson.get("subject_id") == subject_id:
                return dict(lesson.get("subject", {}))
        raise KeyError(f"subject {subject_id!r} not in fixture {self._path}")


def get_backend_client(
    config: Config, fixtures_path: str | None = None
) -> BackendClient:
    """Pick the backend client: explicit ``fixtures_path`` wins (deliberate
    offline), else HTTP when ``BACKEND_API_BASE_URL`` is set, else the config
    fixtures path."""
    if fixtures_path:
        return FixtureBackendClient(fixtures_path)
    base = config.secret("BACKEND_API_BASE_URL")
    if base:
        return HttpBackendClient(base, service_token=config.secret("AI_WEBHOOK_SHARED_SECRET"))
    path = config.get("backend.fixtures_path")
    if not path:
        raise ValueError(
            "No BACKEND_API_BASE_URL and no fixtures manifest — set one, or pass "
            "--fixtures / backend.fixtures_path for the offline resolver."
        )
    return FixtureBackendClient(path)


# ===========================================================================
# Resolver
# ===========================================================================

def resolve_lesson(client: BackendClient, lesson_id: str) -> LessonSource:
    """Resolve a lesson id to a :class:`LessonSource` the pipeline can run."""
    lesson = client.get_lesson(lesson_id)
    subject = client.get_subject(lesson["subject_id"])
    text_content, media_id = _ingestible_source(lesson)
    return LessonSource(
        lesson_id=lesson_id,
        source_uri="" if media_id is None else client.get_media_url(media_id),
        text_content=text_content,
        subject_id=lesson["subject_id"],
        grade=to_platform_grade(subject.get("grade_level")),
        language=subject.get("language"),
        discipline=discipline_of(subject),
        subject_name=subject.get("name"),
        lesson_title=lesson.get("title"),
        lesson_order=lesson.get("order_index"),
    )


def _ingestible_source(lesson: dict[str, Any]) -> tuple[Optional[str], Optional[str]]:
    """The lesson's source as ``(text_content, media_id)`` — exactly one is set.

    The platform computes ``ingestible_source`` for this pipeline: body text wins
    over attachments, and the first attachment by order wins among files. Lessons
    served by older backends and by the fixture manifest carry no such field, so
    the attachment list is still honoured.
    """
    declared = lesson.get("ingestible_source") or {}
    kind = declared.get("type")
    if kind == "text_content":
        text = lesson.get("text_content")
        if text and text.strip():
            return text, None
        raise ValueError(f"lesson {lesson.get('id')!r} declares body text but has none")
    if kind == "attachment" and declared.get("media_id"):
        return None, declared["media_id"]

    attachments = lesson.get("attachments") or []
    if attachments:
        return None, attachments[0]["media_id"]
    text = lesson.get("text_content")
    if text and text.strip():
        return text, None
    raise ValueError(f"lesson {lesson.get('id')!r} has no ingestible source")


# Deriving discipline from subject name/code (AR + EN keyword substrings).
_DISCIPLINE_HINTS: list[tuple[Discipline, tuple[str, ...]]] = [
    (Discipline.MATH, ("math", "algebra", "geometry", "calculus", "رياضيات", "جبر", "هندسة")),
    (Discipline.SCIENCE, ("science", "physics", "chemistry", "biology", "علوم", "فيزياء", "كيمياء", "أحياء")),
    (Discipline.SOCIAL_STUDIES, ("social", "history", "geography", "civics", "دراسات", "اجتماعية", "تاريخ", "جغرافيا")),
    (Discipline.LANGUAGE, ("language", "arabic", "english", "french", "grammar", "لغة", "عربية", "انجليزية", "نحو")),
]


def discipline_of(subject: dict[str, Any]) -> Discipline:
    """Discipline from an explicit field if present, else name/code keywords."""
    explicit = subject.get("discipline") or subject.get("type")
    if explicit:
        try:
            return Discipline(str(explicit).strip().lower())
        except ValueError:
            return Discipline.OTHER

    haystack = f"{subject.get('name', '')} {subject.get('code', '')}".lower()
    for discipline, hints in _DISCIPLINE_HINTS:
        if any(hint in haystack for hint in hints):
            return discipline
    return Discipline.OTHER
