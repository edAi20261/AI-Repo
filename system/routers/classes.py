"""A teacher's view of one class: topic spread, students needing attention, common mistakes."""

from __future__ import annotations

import logging

from fastapi import APIRouter

from ..dependencies import Authenticated, RequestId, RuntimeDep
from ..errors import NotFoundError
from ..schemas.classes import (
    ClassAnalyticsRequest,
    ClassAnalyticsResponse,
    ClassReportRequest,
    ClassReportResponse,
)
from ..schemas.reports import StoredReportResponse
from ..schemas.common import ERROR_RESPONSES

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/ai", tags=["classes"], dependencies=[Authenticated], responses=ERROR_RESPONSES
)


@router.post(
    "/classes/analytics",
    response_model=ClassAnalyticsResponse,
    summary="Analyse a class roster",
)
async def analyse_class(
    payload: ClassAnalyticsRequest, runtime: RuntimeDep, request_id: RequestId
) -> ClassAnalyticsResponse:
    result = await runtime.class_analytics(payload.to_payload())
    response = ClassAnalyticsResponse.from_result(result, request_id)
    logger.info("class analytics", extra={
        "class_id": payload.class_id, "students": len(payload.user_ids),
        "at_risk": len(response.at_risk), "degraded": response.degraded})
    return response


@router.post(
    "/classes/report",
    response_model=ClassReportResponse,
    summary="Write and store a teacher's summary of a class",
)
async def class_report(
    payload: ClassReportRequest, runtime: RuntimeDep, request_id: RequestId
) -> ClassReportResponse:
    result = await runtime.class_report(payload.to_payload())
    return ClassReportResponse.from_result(result, request_id)


@router.get(
    "/classes/{class_id}/report/latest",
    response_model=StoredReportResponse,
    summary="The newest stored class summary, without writing a new one",
)
async def latest_class_report(
    class_id: str, runtime: RuntimeDep, request_id: RequestId
) -> StoredReportResponse:
    stored = await runtime.latest_report({"class_id": class_id, "scope": "class_analytics"})
    if stored is None:
        raise NotFoundError("No report has been written for this class yet.",
                            details={"class_id": class_id})
    return StoredReportResponse(
        request_id=request_id, report_id=stored["report_id"],
        generated_at=stored["generated_at"], stale=stored["stale"],
        newer_evidence=stored["newer_evidence"], report=stored["payload"])

