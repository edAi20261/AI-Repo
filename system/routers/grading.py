"""Grading — grade a batch of answers and move the student's mastery."""

from __future__ import annotations

import logging

from fastapi import APIRouter

from ..dependencies import Authenticated, RequestId, RuntimeDep
from ..schemas.common import PAID_RESPONSES
from ..schemas.grading import GradeRequest, GradeResponse, ReviewedRequest, ReviewedResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ai", tags=["grading"], dependencies=[Authenticated])


@router.post("/grade", response_model=GradeResponse, summary="Grade submitted answers",
             responses=PAID_RESPONSES)
async def grade(payload: GradeRequest, runtime: RuntimeDep, request_id: RequestId) -> GradeResponse:
    logger.info(
        "grade",
        extra={"items": len(payload.submissions), "kind": payload.kind,
               "has_assessment": payload.assessment_id is not None},
    )
    graded = await runtime.grade(payload.to_payload(), billed_user_id=payload.billed_user_id)
    return GradeResponse.from_result(graded.result, request_id, graded.meter)


@router.post(
    "/grade/reviewed",
    response_model=ReviewedResponse,
    summary="Record a teacher's decision on grades held for review",
)
async def reviewed(payload: ReviewedRequest, runtime: RuntimeDep,
                   request_id: RequestId) -> ReviewedResponse:
    logger.info("reviewed grades", extra={"items": len(payload.grades), "source": payload.source})
    recorded = await runtime.record_reviewed(payload.model_dump())
    return ReviewedResponse(request_id=request_id, **recorded)
