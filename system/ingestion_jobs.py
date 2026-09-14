"""Background ingestion jobs.

Ingesting a lesson takes minutes, so it never runs inside a request. The caller
receives a job id immediately and polls for status, and the platform is told
when a run starts and when it ends.

Job records are per-process and exist for progress reporting only. The durable
record of what has been ingested is ``ai.lecture_ingestion``, written by the
pipeline itself.
"""

from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict
from collections.abc import Awaitable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from functools import partial
from typing import TYPE_CHECKING, Any, Optional
from uuid import uuid4

import anyio
from anyio import CapacityLimiter

from shared.config import Config
from shared.logging import redact_uri

from .errors import ConflictError, NotFoundError
from .settings import IngestionSettings

if TYPE_CHECKING:
    from .notifier import StatusNotifier

logger = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.now(timezone.utc)


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass
class IngestionJob:
    job_id: str
    lecture_id: str
    subject_id: Optional[str] = None
    status: JobStatus = JobStatus.QUEUED
    stats: dict[str, Any] = field(default_factory=dict)
    ingestion_status: Optional[str] = None
    errors: list[str] = field(default_factory=list)
    submitted_at: datetime = field(default_factory=_now)
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
        notifier: Optional[StatusNotifier] = None,
    ) -> None:
        self._config = config
        self._settings = settings
        self._notifier = notifier
        self._limiter = CapacityLimiter(settings.max_workers)
        self._jobs: OrderedDict[str, IngestionJob] = OrderedDict()
        self._active: dict[str, str] = {}
        self._tasks: set[asyncio.Task] = set()
        self._lock = asyncio.Lock()

    # --- submission ---------------------------------------------------------

    async def submit_lesson(self, lesson_id: str) -> IngestionJob:
        """Queue by platform lesson id — the shape the platform's trigger sends.

        Resolving the lesson to a source URI and subject costs three calls back
        to the platform, and the trigger is fire-and-forget on a short timeout,
        so resolution happens inside the job rather than in the request.
        """
        job = await self._reserve(lesson_id)
        self._spawn(self._resolve_and_run(job))
        logger.info("ingestion queued", extra={"job_id": job.job_id, "lecture_id": lesson_id})
        return job

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
        """Queue a lesson whose source is already known. Raises
        :class:`ConflictError` if that lecture is already running."""
        from ingestion import new_state  # lazy: keeps PDF/vision deps off the import path

        job = await self._reserve(lecture_id, subject_id)
        state = new_state(
            lecture_id=lecture_id,
            subject_id=subject_id,
            source_uri=source_uri,
            grade=grade,
            namespace=namespace,
            version=version,
            subject_is_math=subject_is_math,
        )
        self._spawn(self._run(job, state))
        logger.info(
            "ingestion queued",
            extra={"job_id": job.job_id, "lecture_id": lecture_id,
                   "source_uri": redact_uri(source_uri)},
        )
        return job

    # --- reads --------------------------------------------------------------

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

    # --- execution ----------------------------------------------------------

    async def _resolve_and_run(self, job: IngestionJob) -> None:
        try:
            state = await anyio.to_thread.run_sync(partial(self._resolve, job.lecture_id))
        except Exception as exc:
            await self._fail(job, f"could not resolve lesson: {exc}", exc)
            return
        job.subject_id = state.get("subject_id")
        await self._run(job, state)

    async def _run(self, job: IngestionJob, state: dict[str, Any]) -> None:
        job.status = JobStatus.RUNNING
        job.started_at = _now()
        await self._notify(job)

        try:
            result = await anyio.to_thread.run_sync(
                partial(self._ingest, state), limiter=self._limiter
            )
        except Exception as exc:
            await self._fail(job, str(exc), exc)
            return

        job.stats = result.get("stats") or {}
        job.ingestion_status = result.get("ingestion_status") or "ready"
        job.errors = list(result.get("errors") or [])
        job.status = JobStatus.SUCCEEDED
        logger.info(
            "ingestion finished",
            extra={"job_id": job.job_id, "lecture_id": job.lecture_id,
                   "ingestion_status": job.ingestion_status},
        )
        await self._finish(job)

    def _resolve(self, lesson_id: str) -> dict[str, Any]:
        """Platform lesson id -> a state the pipeline can run. Blocking."""
        from ingestion import get_backend_client, new_state, resolve_lesson

        source = resolve_lesson(get_backend_client(self._config), lesson_id)
        return new_state(
            lecture_id=source.lesson_id,
            subject_id=source.subject_id,
            source_uri=source.source_uri,
            source_text=source.text_content,
            grade=source.grade,
            subject_name=source.subject_name,
            lesson_title=source.lesson_title,
            lesson_order=source.lesson_order,
            language_hint=source.language,
            subject_is_math=source.subject_is_math,
        )

    def _ingest(self, state: dict[str, Any]) -> dict[str, Any]:
        """Run the pipeline. Blocking — always called in a worker thread."""
        from ingestion import run_ingestion

        return run_ingestion(state, config=self._config)

    # --- bookkeeping --------------------------------------------------------

    async def _reserve(
        self, lecture_id: str, subject_id: Optional[str] = None
    ) -> IngestionJob:
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
        return job

    def _spawn(self, coro: Awaitable[None]) -> None:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _fail(self, job: IngestionJob, message: str, exc: BaseException) -> None:
        job.status = JobStatus.FAILED
        job.errors = [message]
        logger.error(
            "ingestion failed",
            extra={"job_id": job.job_id, "lecture_id": job.lecture_id},
            exc_info=exc,
        )
        await self._finish(job)

    async def _finish(self, job: IngestionJob) -> None:
        """The one place a job becomes terminal: release the lecture, report out."""
        job.finished_at = _now()
        async with self._lock:
            self._active.pop(job.lecture_id, None)
        await self._notify(job)

    async def _notify(self, job: IngestionJob) -> None:
        if self._notifier is None:
            return
        try:
            await self._notifier.report(job)
        except Exception:
            logger.exception("ingestion status report failed", extra={"job_id": job.job_id})

    def _evict_old(self) -> None:
        while len(self._jobs) > self._settings.job_retention:
            _, evicted = self._jobs.popitem(last=False)
            if not evicted.is_terminal:
                self._jobs[evicted.job_id] = evicted
                return
