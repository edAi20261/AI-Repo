"""Background ingestion jobs.

Ingesting a lesson takes minutes, so it is never run inside a request. The
caller receives a job id immediately and polls for status; on completion the
platform backend is notified when a callback is configured.

Job records are per-process and exist for progress reporting only. The durable
record of what has been ingested is ``ai.lecture_ingestion``, written by the
pipeline itself.
"""

from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional
from uuid import uuid4

import anyio
from anyio import CapacityLimiter

from shared.config import Config
from shared.logging import redact_uri

from .errors import ConflictError, NotFoundError
from .settings import IngestionSettings

if TYPE_CHECKING:
    from .notifier import CompletionNotifier

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass
class IngestionJob:
    job_id: str
    lecture_id: str
    subject_id: str
    status: JobStatus = JobStatus.QUEUED
    stats: dict[str, Any] = field(default_factory=dict)
    ingestion_status: Optional[str] = None
    errors: list[str] = field(default_factory=list)
    submitted_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None

    @property
    def is_terminal(self) -> bool:
        return self.status in (JobStatus.SUCCEEDED, JobStatus.FAILED)


class IngestionJobManager:
    """Submits pipeline runs to a bounded worker pool and tracks their outcome."""

    def __init__(
        self,
        config: Config,
        settings: IngestionSettings,
        *,
        notifier: Optional[CompletionNotifier] = None,
    ) -> None:
        self._config = config
        self._settings = settings
        self._notifier = notifier
        self._limiter = CapacityLimiter(settings.max_workers)
        self._jobs: OrderedDict[str, IngestionJob] = OrderedDict()
        self._active: dict[str, str] = {}
        self._tasks: set[asyncio.Task] = set()
        self._lock = asyncio.Lock()

    async def submit(
        self,
        *,
        lecture_id: str,
        subject_id: str,
        source_uri: str,
        grade: Optional[str] = None,
        namespace: str = "syllabus",
        version: int = 1,
        subject_is_math: bool = False,
    ) -> IngestionJob:
        """Queue one lesson. Raises :class:`ConflictError` if it is already running."""
        async with self._lock:
            if lecture_id in self._active:
                raise ConflictError(
                    "This lecture is already being ingested.",
                    details={"job_id": self._active[lecture_id], "lecture_id": lecture_id},
                )
            job = IngestionJob(job_id=uuid4().hex, lecture_id=lecture_id, subject_id=subject_id)
            self._jobs[job.job_id] = job
            self._active[lecture_id] = job.job_id
            self._evict_old()

        from ingestion import new_state  # lazy: keeps PDF/vision deps off the import path

        state = new_state(
            lecture_id=lecture_id,
            subject_id=subject_id,
            source_uri=source_uri,
            grade=grade,
            namespace=namespace,
            version=version,
            subject_is_math=subject_is_math,
        )
        task = asyncio.create_task(self._run(job, state))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

        logger.info(
            "ingestion queued",
            extra={"job_id": job.job_id, "lecture_id": lecture_id,
                   "source_uri": redact_uri(source_uri)},
        )
        return job

    def get(self, job_id: str) -> IngestionJob:
        job = self._jobs.get(job_id)
        if job is None:
            raise NotFoundError("No such ingestion job.", details={"job_id": job_id})
        return job

    def latest_for_lecture(self, lecture_id: str) -> IngestionJob:
        for job in reversed(self._jobs.values()):
            if job.lecture_id == lecture_id:
                return job
        raise NotFoundError(
            "No ingestion job for this lecture in the current process.",
            details={"lecture_id": lecture_id},
        )

    def list(self, *, limit: int = 50) -> list[IngestionJob]:
        return list(reversed(self._jobs.values()))[:limit]

    async def drain(self, timeout: float = 30.0) -> None:
        """Let in-flight jobs finish during shutdown."""
        if not self._tasks:
            return
        logger.info("draining ingestion jobs", extra={"in_flight": len(self._tasks)})
        with anyio.move_on_after(timeout):
            await asyncio.gather(*tuple(self._tasks), return_exceptions=True)

    async def _run(self, job: IngestionJob, state: dict[str, Any]) -> None:
        job.status = JobStatus.RUNNING
        job.started_at = datetime.now(timezone.utc)
        try:
            result = await anyio.to_thread.run_sync(self._pipeline(state), limiter=self._limiter)
            job.stats = result.get("stats", {}) or {}
            job.ingestion_status = result.get("ingestion_status") or "ready"
            job.errors = list(result.get("errors") or [])
            job.status = JobStatus.SUCCEEDED
            logger.info(
                "ingestion finished",
                extra={"job_id": job.job_id, "lecture_id": job.lecture_id,
                       "ingestion_status": job.ingestion_status},
            )
        except Exception as exc:
            job.status = JobStatus.FAILED
            job.errors = [str(exc)]
            logger.exception(
                "ingestion failed",
                extra={"job_id": job.job_id, "lecture_id": job.lecture_id},
            )
        finally:
            job.finished_at = datetime.now(timezone.utc)
            async with self._lock:
                self._active.pop(job.lecture_id, None)
            await self._notify(job)

    def _pipeline(self, state: dict[str, Any]):
        def run() -> dict[str, Any]:
            from ingestion import run_ingestion

            return run_ingestion(state, config=self._config)

        return run

    async def _notify(self, job: IngestionJob) -> None:
        if self._notifier is None:
            return
        try:
            await self._notifier.completed(job)
        except Exception:
            logger.exception("ingestion callback failed", extra={"job_id": job.job_id})

    def _evict_old(self) -> None:
        while len(self._jobs) > self._settings.job_retention:
            _, evicted = self._jobs.popitem(last=False)
            if not evicted.is_terminal:
                self._jobs[evicted.job_id] = evicted
                return
