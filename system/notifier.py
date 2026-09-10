"""Ingestion status reported back to the platform.

The platform owns a lesson's ``ingestion_status``; this service is what moves it.
The contract is theirs: ``PATCH /lessons/{id}/ingestion`` with a status drawn
from ``content.ingestion_status``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional, Protocol

import httpx

from .settings import Settings

if TYPE_CHECKING:
    from .ingestion_jobs import IngestionJob

logger = logging.getLogger(__name__)

# The platform's content.ingestion_status enum.
PENDING, PROCESSING, READY, FAILED = "pending", "processing", "ready", "failed"


def platform_status(job: IngestionJob) -> str:
    """Map a job to the platform's vocabulary.

    A ``needs_review`` run is reported ``ready``: pages the model could not read
    were never indexed, but what was indexed is usable and the lesson should go
    live. The platform has no value for a partial ingest.
    """
    if job.status.value == "failed":
        return FAILED
    if job.status.value == "succeeded":
        return READY
    return PROCESSING if job.status.value == "running" else PENDING


class StatusNotifier(Protocol):
    async def report(self, job: IngestionJob) -> None:
        ...


class HttpStatusNotifier:
    """PATCHes lesson ingestion status, authenticated with the shared secret."""

    def __init__(
        self,
        base_url: str,
        path_template: str,
        service_token: Optional[str],
        *,
        timeout: float = 10.0,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._path_template = path_template
        self._headers = {"x-service-token": service_token} if service_token else {}
        self._timeout = timeout

    async def report(self, job: IngestionJob) -> None:
        status = platform_status(job)
        body: dict[str, Any] = {"ingestion_status": status}
        if status == FAILED and job.errors:
            body["ingestion_error"] = job.errors[0][:1000]

        path = self._path_template.format(lesson_id=job.lecture_id, lecture_id=job.lecture_id)
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.patch(
                f"{self._base}{path}", json=body, headers=self._headers
            )
            response.raise_for_status()
        logger.info(
            "ingestion status reported",
            extra={"job_id": job.job_id, "lecture_id": job.lecture_id, "status": status},
        )


def build_notifier(settings: Settings) -> Optional[HttpStatusNotifier]:
    """A notifier only when reporting is enabled and the platform URL is known."""
    if not settings.ingestion.callback_enabled:
        return None
    if not settings.backend_base_url:
        logger.warning(
            "ingestion status reporting is enabled but BACKEND_API_BASE_URL is unset; disabling"
        )
        return None
    return HttpStatusNotifier(
        settings.backend_base_url,
        settings.ingestion.callback_path,
        settings.service_token,
    )
