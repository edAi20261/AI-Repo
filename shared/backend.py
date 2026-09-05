"""Backend read-client for the runtime agent.

The agent reads reference data from the platform (a student's subjects, subject
facts, the question bank, an assessment's answer key) but never writes it and
never reads a student submission from the DB (that arrives in the grade payload).

Behind :class:`BackendClient` are two impls sharing the same methods: a real HTTP
client and a fixture client backed by a local YAML manifest, so the whole agent
runs offline. The factory picks HTTP when ``BACKEND_API_BASE_URL`` is set, else the
fixtures manifest at ``agent.backend.fixtures_path``.

Transport mirrors ``ingestion/backend.py`` (lazy httpx, one shared-secret header);
the two clients stay separate because their method sets differ. Converging the
transport is a later cleanup.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Optional, Protocol, runtime_checkable

from .config import Config

logger = logging.getLogger(__name__)


@runtime_checkable
class BackendClient(Protocol):
    """The Part A reads the runtime agent needs."""

    def get_student_subjects(self, user_id: str) -> list[dict[str, Any]]:
        """A student's enrolled subjects: ``[{id, name, grade_level, language}]``."""

    def get_subject(self, subject_id: str) -> dict[str, Any]:
        """One subject's facts:
        ``{id, curriculum_id, grade_level, name, code, language, is_math}``.
        ``is_math`` gates the math prompt addendum (append math instructions)."""

    def get_question_bank(
        self,
        subject_id: str,
        topic_ids: Optional[list[str]] = None,
        count: int = 10,
        type_mix: Optional[dict[str, int]] = None,
    ) -> list[dict[str, Any]]:
        """Existing questions for a subject/topics (quiz BANK/HYBRID sources).

        ``type_mix`` is accepted for a stable signature but is not implemented
        backend-side yet; it is ignored by the HTTP client."""

    def get_questions(self, kind: str, assessment_id: str) -> list[dict[str, Any]]:
        """An assessment's questions with model answer + rubric — the answer key
        used by grading. ``kind`` is ``homework`` or ``quiz``."""


# ===========================================================================
# HTTP client
# ===========================================================================

class HttpBackendClient:
    """Calls the platform's ``/api/ai-integration`` endpoints. httpx is imported
    lazily so importing this module never requires it."""

    def __init__(
        self,
        base_url: str,
        service_token: str | None = None,
        *,
        timeout: float = 30,
        max_retries: int = 3,
    ) -> None:
        # base_url points at the service-to-service prefix (…/api/ai-integration).
        self._base = base_url.rstrip("/")
        # Service-to-service auth: one shared secret, same header both directions.
        self._headers = {"x-service-token": service_token} if service_token else {}
        self._timeout = timeout
        self._max_retries = max(1, max_retries)

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        import httpx  # lazy

        # Drop None params so optional query args (topic_ids, count) stay absent.
        clean = {k: v for k, v in (params or {}).items() if v is not None}
        last_exc: Exception | None = None
        for attempt in range(1, self._max_retries + 1):
            try:
                resp = httpx.get(
                    f"{self._base}{path}",
                    params=clean or None,
                    headers=self._headers,
                    timeout=self._timeout,
                )
                # Retry only transient server errors; 4xx is a caller bug — raise.
                if resp.status_code >= 500:
                    resp.raise_for_status()
                resp.raise_for_status()
                return resp.json()
            except httpx.HTTPStatusError as exc:
                if exc.response is not None and exc.response.status_code < 500:
                    raise
                last_exc = exc
            except httpx.TransportError as exc:
                last_exc = exc
            if attempt < self._max_retries:
                backoff = 0.2 * (2 ** (attempt - 1))
                logger.warning("backend GET retry", extra={
                    "path": path, "attempt": attempt, "backoff": backoff})
                time.sleep(backoff)
        logger.error("backend GET failed", extra={"path": path}, exc_info=last_exc)
        assert last_exc is not None
        raise last_exc

    def get_student_subjects(self, user_id: str) -> list[dict[str, Any]]:
        return self._get(f"/students/{user_id}/subjects")

    def get_subject(self, subject_id: str) -> dict[str, Any]:
        return self._get(f"/subjects/{subject_id}")

    def get_question_bank(
        self,
        subject_id: str,
        topic_ids: Optional[list[str]] = None,
        count: int = 10,
        type_mix: Optional[dict[str, int]] = None,
    ) -> list[dict[str, Any]]:
        # type_mix is not a backend param yet (documented); enforced by us for
        # GENERATED. Send subject_id/topic_ids/count only.
        params: dict[str, Any] = {"subject_id": subject_id, "count": count}
        if topic_ids:
            params["topic_ids"] = ",".join(topic_ids)
        return self._get("/question-bank", params)

    def get_questions(self, kind: str, assessment_id: str) -> list[dict[str, Any]]:
        prefix = _questions_prefix(kind)
        return self._get(f"/{prefix}/{assessment_id}/questions")


# ===========================================================================
# Fixture client
# ===========================================================================

class FixtureBackendClient:
    """Reads a local YAML/JSON manifest instead of a backend, so the agent runs
    with no platform, database or network.

    Manifest shape::

        students:
          student-1:
            subjects: [{id, name, grade_level, language}, ...]
        subjects:
          <subject_id>: {id, curriculum_id, grade_level, name, code, language}
        question_bank:
          <subject_id>: [{id, question_text, question_type, topic_ids,
                          lecture_id, options?, correct_answer_text, points,
                          difficulty, bloom}, ...]
        homework:
          <assessment_id>: {questions: [{... correct_answer_text, explanation,
                                         points, rubric?}, ...]}
        quizzes:
          <assessment_id>: {questions: [...]}
    """

    def __init__(self, manifest_path: str | Path) -> None:
        self._path = Path(manifest_path)
        self._data = self._load()

    def _load(self) -> dict[str, Any]:
        text = self._path.read_text(encoding="utf-8")
        if self._path.suffix.lower() in (".yaml", ".yml"):
            import yaml

            return yaml.safe_load(text) or {}
        return json.loads(text)

    def get_student_subjects(self, user_id: str) -> list[dict[str, Any]]:
        student = self._data.get("students", {}).get(user_id)
        if student is None:
            raise KeyError(f"student {user_id!r} not in fixture {self._path}")
        return list(student.get("subjects", []))

    def get_subject(self, subject_id: str) -> dict[str, Any]:
        subject = self._data.get("subjects", {}).get(subject_id)
        if subject is None:
            raise KeyError(f"subject {subject_id!r} not in fixture {self._path}")
        return dict(subject)

    def get_question_bank(
        self,
        subject_id: str,
        topic_ids: Optional[list[str]] = None,
        count: int = 10,
        type_mix: Optional[dict[str, int]] = None,
    ) -> list[dict[str, Any]]:
        items = list(self._data.get("question_bank", {}).get(subject_id, []))
        if topic_ids:
            wanted = set(topic_ids)
            items = [q for q in items if wanted & set(q.get("topic_ids", []))]
        return items[:count]

    def get_questions(self, kind: str, assessment_id: str) -> list[dict[str, Any]]:
        section = _questions_prefix(kind)
        record = self._data.get(section, {}).get(assessment_id)
        if record is None:
            raise KeyError(
                f"{kind} {assessment_id!r} not in fixture {self._path}")
        return list(record.get("questions", []))


# ===========================================================================
# Factory + helpers
# ===========================================================================

def _questions_prefix(kind: str) -> str:
    """Map an assessment kind to its endpoint/section: homework|quiz."""
    if kind == "homework":
        return "homework"
    if kind in ("quiz", "quizzes"):
        return "quizzes"
    raise ValueError(f"unknown assessment kind {kind!r} (expected homework|quiz)")


def get_backend_client(
    config: Config, fixtures_path: str | None = None
) -> BackendClient:
    """Pick the backend client: explicit ``fixtures_path`` wins (deliberate
    offline), else the ``agent.backend.prefer_fixtures`` flag forces the configured
    fixtures (offline-by-default in the ``local`` overlay even when a base URL is set
    in ``.env``), else HTTP when ``BACKEND_API_BASE_URL`` is set, else the configured
    ``agent.backend.fixtures_path``."""
    if fixtures_path:
        return FixtureBackendClient(fixtures_path)
    configured = config.get("agent.backend.fixtures_path")
    if config.get("agent.backend.prefer_fixtures", False):
        if not configured:
            raise ValueError(
                "agent.backend.prefer_fixtures is set but agent.backend.fixtures_path "
                "is empty — point it at a fixtures manifest."
            )
        return FixtureBackendClient(configured)
    base = config.secret("BACKEND_API_BASE_URL")
    if base:
        return HttpBackendClient(
            base,
            service_token=config.secret("AI_WEBHOOK_SHARED_SECRET"),
            timeout=config.get("agent.backend.timeout", 30),
            max_retries=config.get("agent.backend.max_retries", 3),
        )
    if not configured:
        raise ValueError(
            "No BACKEND_API_BASE_URL and no agent.backend.fixtures_path — set one "
            "for the offline agent."
        )
    return FixtureBackendClient(configured)
