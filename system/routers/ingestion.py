"""Ingestion — queue a lesson for processing and poll its progress.

A pipeline run takes minutes, so submission returns ``202 Accepted`` with a job
handle. The durable record of what is ingested is ``ai.lecture_ingestion``.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Query, status

from ..dependencies import Authenticated, JobsDep
from ..schemas.ingestion import IngestionJobResponse, IngestionRequest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ai/ingest", tags=["ingestion"], dependencies=[Authenticated])


@router.post(
    "",
    response_model=IngestionJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Queue a lesson for ingestion",
)
async def submit(payload: IngestionRequest, jobs: JobsDep) -> IngestionJobResponse:
    job = await jobs.submit(**payload.model_dump())
    return IngestionJobResponse.from_job(job)


@router.get("", response_model=list[IngestionJobResponse], summary="List recent ingestion jobs")
async def list_jobs(jobs: JobsDep, limit: int = Query(default=50, ge=1, le=200)):
    return [IngestionJobResponse.from_job(job) for job in jobs.list(limit=limit)]


@router.get("/{job_id}", response_model=IngestionJobResponse, summary="Read one job")
async def get_job(job_id: str, jobs: JobsDep) -> IngestionJobResponse:
    return IngestionJobResponse.from_job(jobs.get(job_id))


@router.get(
    "/lectures/{lecture_id}",
    response_model=IngestionJobResponse,
    summary="Read the latest job for a lecture",
)
async def get_lecture_job(lecture_id: str, jobs: JobsDep) -> IngestionJobResponse:
    return IngestionJobResponse.from_job(jobs.latest_for_lecture(lecture_id))
