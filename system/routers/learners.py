"""The student analysis a homepage shows: standing, strengths, weak spots, next steps."""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Query

from ..dependencies import Authenticated, RequestId, RuntimeDep
from ..schemas.common import ERROR_RESPONSES
from ..schemas.learners import LearnerSummaryResponse

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/ai", tags=["learners"], dependencies=[Authenticated], responses=ERROR_RESPONSES
)


@router.get(
    "/learners/{user_id}/summary",
    response_model=LearnerSummaryResponse,
    summary="A student's standing, strengths, weak spots and next steps",
)
async def learner_summary(
    user_id: str,
    runtime: RuntimeDep,
    request_id: RequestId,
    subject_ids: Optional[list[str]] = Query(
        default=None,
        description="Subjects the student is enrolled in, counted even when untouched. "
                    "Exactly one scopes the whole analysis to that subject."),
    lecture_ids: Optional[list[str]] = Query(
        default=None, description="Lessons the student is working through now."),
) -> LearnerSummaryResponse:
    result = await runtime.learner_summary({
        "user_id": user_id, "subject_ids": subject_ids, "lecture_ids": lecture_ids})
    response = LearnerSummaryResponse.from_result(result, request_id)
    logger.info("learner summary", extra={
        "available": response.available, "degraded": response.degraded,
        "recommendations": len(response.recommendations)})
    return response
