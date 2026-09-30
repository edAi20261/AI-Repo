"""Grading — grade a batch of answers and move the student's mastery."""

from __future__ import annotations

import logging

from fastapi import APIRouter

from ..dependencies import Authenticated, RequestId, RuntimeDep
from ..schemas.common import PAID_RESPONSES
from ..schemas.grading import GradeRequest, GradeResponse

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/ai", tags=["grading"], dependencies=[Authenticated], responses=PAID_RESPONSES
)


@router.post("/grade", response_model=GradeResponse, summary="Grade submitted answers")
async def grade(payload: GradeRequest, runtime: RuntimeDep, request_id: RequestId) -> GradeResponse:
    logger.info(
        "grade",
        extra={"items": len(payload.submissions), "kind": payload.kind,
               "has_assessment": payload.assessment_id is not None},
    )
    graded = await runtime.grade(payload.to_payload(), billed_user_id=payload.billed_user_id)
    return GradeResponse.from_result(graded.result, request_id, graded.meter)
