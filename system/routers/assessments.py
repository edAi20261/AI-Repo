"""Assessments — generate a practice quiz from a lesson scope."""

from __future__ import annotations

import logging

from fastapi import APIRouter

from shared.errors import UnprocessableError

from ..dependencies import Authenticated, RequestId, RuntimeDep
from ..schemas.assessments import AssessmentRequest, AssessmentResponse
from ..schemas.common import PAID_RESPONSES

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/ai/assessments", tags=["assessments"],
    dependencies=[Authenticated], responses=PAID_RESPONSES,
)


@router.post(
    "/generate",
    response_model=AssessmentResponse,
    summary="Generate a quiz of a given size, types and difficulty from the scoped lessons",
)
async def generate_assessment(payload: AssessmentRequest, runtime: RuntimeDep,
                              request_id: RequestId) -> AssessmentResponse:
    limit = runtime.config.get(f"assessment.max_count.{payload.user_type}")
    if payload.count > limit:
        raise UnprocessableError(
            f"A {payload.user_type} quiz can have at most {limit} questions.",
            details={"count": payload.count, "limit": limit})
    generated = await runtime.generate_assessment(payload.to_payload(),
                                                  billed_user_id=payload.billed_user_id)
    assessment = generated.result
    logger.info("assessment generated", extra={
        "requested": payload.count, "returned": assessment["count"],
        "shortfall": assessment["shortfall"], "lectures": len(payload.lecture_ids or [])})
    return AssessmentResponse.from_assessment(assessment, request_id, generated.meter)
