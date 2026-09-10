"""Progress reports — a student's trend, weak topics, and root cause."""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Query

from ..dependencies import Authenticated, RequestId, RuntimeDep
from ..schemas.common import Language, Visual
from ..schemas.reports import ReportResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ai", tags=["reports"], dependencies=[Authenticated])


@router.get(
    "/reports/{user_id}",
    response_model=ReportResponse,
    summary="Build a student's progress report",
)
async def student_report(
    user_id: str,
    runtime: RuntimeDep,
    request_id: RequestId,
    subject_id: Optional[str] = Query(default=None, description="Scopes the report to one subject."),
    lecture_ids: Optional[list[str]] = Query(default=None, description="Scopes it to specific lessons."),
    language: Optional[Language] = Query(default=None),
) -> ReportResponse:
    result = await runtime.report({
        "user_id": user_id,
        "scope": "student_progress",
        "subject_id": subject_id,
        "lecture_ids": lecture_ids,
        "language": language,
    })

    spec = result.pop("visual_spec", None)
    response = ReportResponse.from_result(result, request_id)
    if spec:
        response.visual = await _render_trend_chart(
            runtime, spec, user_id=user_id, subject_id=subject_id,
            language=result.get("language") or language or "en",
        )
    logger.info(
        "report",
        extra={"available": response.available, "weak_topics": len(response.weak_topics),
               "charted": response.visual is not None},
    )
    return response


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
