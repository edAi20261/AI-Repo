"""Progress reports — a student's trend, weak topics, and root cause."""

from __future__ import annotations

import logging
from typing import Literal, Optional

from fastapi import APIRouter, Query

from agent.graph.report import SUBJECT_SCOPES

from ..dependencies import Authenticated, RequestId, RuntimeDep
from ..schemas.common import ERROR_RESPONSES, PAID_RESPONSES, Language, Visual
from ..errors import NotFoundError, UnprocessableError
from ..schemas.reports import ReportResponse, StoredReportResponse

logger = logging.getLogger(__name__)

StudentScope = Literal["student_progress", "parent", "teacher"]
_SCOPE_DESCRIPTION = ("`parent` writes the same evidence as a note to the student's parent; "
                      "`teacher` as a note to their teacher about one subject, and needs "
                      "`subject_id`.")

router = APIRouter(
    prefix="/ai", tags=["reports"], dependencies=[Authenticated], responses=ERROR_RESPONSES
)


@router.get(
    "/reports/{user_id}/latest",
    response_model=StoredReportResponse,
    summary="The newest stored report, without building a new one",
)
async def latest_report(
    user_id: str,
    runtime: RuntimeDep,
    request_id: RequestId,
    scope: StudentScope = Query(default="student_progress", description=_SCOPE_DESCRIPTION),
    subject_id: Optional[str] = Query(
        default=None, description="The newest report built for this subject."),
) -> StoredReportResponse:
    _require_subject(scope, subject_id)
    stored = await runtime.latest_report({"user_id": user_id, "scope": scope,
                                          "subject_id": subject_id})
    if stored is None:
        raise NotFoundError("No report has been built for this student yet.",
                            details={"user_id": user_id, "scope": scope,
                                     "subject_id": subject_id})
    return StoredReportResponse(
        request_id=request_id, report_id=stored["report_id"],
        generated_at=stored["generated_at"], stale=stored["stale"],
        newer_evidence=stored["newer_evidence"], report=stored["payload"])


@router.get(
    "/reports/{user_id}",
    response_model=ReportResponse,
    summary="Build a student's progress report",
    responses=PAID_RESPONSES,
)
async def student_report(
    user_id: str,
    runtime: RuntimeDep,
    request_id: RequestId,
    billed_user_id: str = Query(
        description="The account the credits are taken from: the student, or the parent."),
    subject_id: Optional[str] = Query(default=None, description="Scopes the report to one subject."),
    lecture_ids: Optional[list[str]] = Query(default=None, description="Scopes it to specific lessons."),
    language: Optional[Language] = Query(default=None),
    scope: StudentScope = Query(default="student_progress", description=_SCOPE_DESCRIPTION),
) -> ReportResponse:
    _require_subject(scope, subject_id)
    built = await runtime.report({
        "user_id": user_id,
        "scope": scope,
        "subject_id": subject_id,
        "lecture_ids": lecture_ids,
        "language": language,
    }, billed_user_id=billed_user_id)

    result = built.result
    spec = result.pop("visual_spec", None)
    response = ReportResponse.from_result(result, request_id, built.meter)
    if spec:
        response.visual = await _render_trend_chart(
            runtime, spec, user_id=user_id, subject_id=subject_id,
            language=result.get("language") or language or "en",
        )
    if response.visual is not None and response.persisted and response.report_id:
        await _attach_visual(runtime, response.report_id, response.visual)
    logger.info(
        "report",
        extra={"available": response.available, "weak_topics": len(response.weak_topics),
               "charted": response.visual is not None},
    )
    return response


def _require_subject(scope: str, subject_id: Optional[str]) -> None:
    if scope in SUBJECT_SCOPES and not subject_id:
        raise UnprocessableError(f"A {scope} report is about one subject; send subject_id.",
                                 details={"scope": scope})


async def _attach_visual(runtime, report_id: str, visual: Visual) -> None:
    """The stored report is what a refresh serves, and it was written before its chart
    was rendered. Failing to attach leaves the report stored without a chart."""
    try:
        attached = await runtime.attach_report_visual(
            report_id, visual.model_dump(exclude_none=True))
    except Exception:
        logger.exception("trend chart not attached to the stored report",
                         extra={"report_id": report_id})
        return
    if not attached:
        logger.error("trend chart not attached: no stored report with that id",
                     extra={"report_id": report_id})


async def _render_trend_chart(runtime, spec, *, user_id, subject_id, language) -> Optional[Visual]:
    """``run_report`` returns the chart unrendered; a report without its chart is
    still useful, so a render failure degrades rather than fails the request."""
    try:
        rendered = await runtime.render([spec], user_id=user_id, subject_id=subject_id,
                                        language=language)
    except Exception:
        logger.exception("trend chart render failed")
        return None
    return Visual.model_validate(rendered[0]) if rendered else None
