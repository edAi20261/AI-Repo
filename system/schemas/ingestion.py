"""The ingestion contract.

Submission is asynchronous: the response is a job handle, not a result.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional

from pydantic import Field

from .common import Schema

if TYPE_CHECKING:
    from ..ingestion_jobs import IngestionJob


class IngestionRequest(Schema):
    lecture_id: str = Field(description="Platform lesson id; also the pipeline's lecture id.")
    subject_id: str
    source_uri: str = Field(description="file:// or s3:// URI of the lesson PDF.")
    grade: Optional[str] = None
    namespace: str = "syllabus"
    version: int = Field(default=1, ge=1)
    subject_is_math: bool = Field(
        default=False, description="Routes every page through vision; maths flattens under text extraction."
    )


class IngestionJobResponse(Schema):
    job_id: str
    lecture_id: str
    subject_id: str
    status: str
    ingestion_status: Optional[str] = None
    stats: dict[str, Any] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)
    submitted_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None

    @classmethod
    def from_job(cls, job: IngestionJob) -> IngestionJobResponse:
        return cls(
            job_id=job.job_id,
            lecture_id=job.lecture_id,
            subject_id=job.subject_id,
            status=job.status.value,
            ingestion_status=job.ingestion_status,
            stats=job.stats,
            errors=job.errors,
            submitted_at=job.submitted_at,
            started_at=job.started_at,
            finished_at=job.finished_at,
        )
