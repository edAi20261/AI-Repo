"""Grading — grade a batch of answers and move the student's mastery."""

from __future__ import annotations

import logging

from fastapi import APIRouter

from ..dependencies import Authenticated, RequestId, RuntimeDep
from ..schemas.grading import GradeRequest, GradeResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ai", tags=["grading"], dependencies=[Authenticated])


@router.post("/grade", response_model=GradeResponse, summary="Grade submitted answers")
async def grade(payload: GradeRequest, runtime: RuntimeDep, request_id: RequestId) -> GradeResponse:
    logger.info(
        "grade",
        extra={"items": len(payload.submissions), "kind": payload.kind,
               "has_assessment": payload.assessment_id is not None},
    )
    result = await runtime.grade(payload.to_payload())
    return GradeResponse.from_result(result, request_id)
