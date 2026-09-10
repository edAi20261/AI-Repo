"""Outbound notification to the platform backend.

When a lesson finishes ingesting, the platform needs to know so it can mark the
lesson available. The callback path is configuration, not a constant, because
the platform owns that contract — see ``api.ingestion.callback_path``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol

import httpx

from .settings import Settings

if TYPE_CHECKING:
    from .ingestion_jobs import IngestionJob

logger = logging.getLogger(__name__)


class CompletionNotifier(Protocol):
    async def completed(self, job: IngestionJob) -> None:
        ...


class HttpCompletionNotifier:
    """POSTs a terminal job state to the platform, authenticated with the same
    shared secret the platform uses to call this service."""

    def __init__(
        self,
        base_url: str,
        path_template: str,
        service_token: str | None,
        *,
        timeout: float = 10.0,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._path_template = path_template
        self._headers = {"x-service-token": service_token} if service_token else {}
        self._timeout = timeout

    async def completed(self, job: IngestionJob) -> None:
        path = self._path_template.format(lesson_id=job.lecture_id, lecture_id=job.lecture_id)
        payload = {
            "lecture_id": job.lecture_id,
            "job_id": job.job_id,
            "status": job.status.value,
            "ingestion_status": job.ingestion_status,
            "stats": job.stats,
            "errors": job.errors,
        }
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(f"{self._base}{path}", json=payload, headers=self._headers)
            response.raise_for_status()
        logger.info("ingestion callback delivered", extra={"job_id": job.job_id})


def build_notifier(settings: Settings) -> HttpCompletionNotifier | None:
    """A notifier only when a callback is configured and a backend URL is known."""
    if not settings.ingestion.callback_enabled:
        return None
    if not settings.backend_base_url:
        logger.warning("ingestion callback enabled but BACKEND_API_BASE_URL is unset; disabling")
        return None
    return HttpCompletionNotifier(
        settings.backend_base_url,
        settings.ingestion.callback_path,
        settings.service_token,
    )
