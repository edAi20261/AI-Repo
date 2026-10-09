"""Homework — read a photographed page into structured questions, or mark it."""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, File, Query, UploadFile

from shared.errors import UnprocessableError

from ..dependencies import Authenticated, RequestId, RuntimeDep
from ..schemas.common import PAID_RESPONSES, Language
from ..schemas.homework import CaptureResponse, MarkResponse

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/ai/homework", tags=["homework"],
    dependencies=[Authenticated], responses=PAID_RESPONSES,
)

ACCEPTED = ("image/png", "image/jpeg", "image/webp")


async def _photo(image: UploadFile, runtime) -> bytes:
    """The upload's bytes, refused before any model is paid for when it cannot be a page."""
    if image.content_type not in ACCEPTED:
        raise UnprocessableError(
            f"Send a photo as one of {', '.join(ACCEPTED)}.",
            details={"received": image.content_type})
    limit = runtime.config.get("api.homework.max_upload_bytes")
    content = await image.read()
    if len(content) > limit:
        raise UnprocessableError(
            "That photo is too large. Send one page, not a scanned booklet.",
            details={"bytes": len(content), "limit": limit})
    if not content:
        raise UnprocessableError("The upload was empty.")
    return content


@router.post(
    "/capture",
    response_model=CaptureResponse,
    summary="Read the questions off a photo of a homework page",
)
async def capture_homework(
    request_id: RequestId,
    runtime: RuntimeDep,
    billed_user_id: str = Query(description="The account the credits are taken from."),
    image: UploadFile = File(description="A photo of one homework page."),
    language: Language | None = Query(
        default=None,
        description="Helps the model read the script; it never translates."),
) -> CaptureResponse:
    content = await _photo(image, runtime)
    captured = await runtime.capture_homework(content, language=language,
                                              billed_user_id=billed_user_id)
    result = captured.result
    logger.info("homework capture", extra={
        "bytes": len(content), "questions": len(result.questions),
        "unreadable": result.unreadable})
    return CaptureResponse.from_capture(result, request_id, captured.meter)


@router.post(
    "/mark",
    response_model=MarkResponse,
    summary="Read a photo of a homework page and mark the student's answers",
)
async def mark_homework(
    request_id: RequestId,
    runtime: RuntimeDep,
    user_id: str = Query(description="The student whose page this is, and whose mastery moves."),
    billed_user_id: str = Query(description="The account the credits are taken from."),
    image: UploadFile = File(description="A photo of one homework page."),
    subject_id: Optional[str] = Query(
        default=None, description="The subject the homework is in; marks against its lessons."),
    lecture_ids: Optional[list[str]] = Query(
        default=None, description="The lessons it was set on, when known."),
    language: Language | None = Query(
        default=None,
        description=("Helps the model read the script, and is the feedback language only "
                     "when the page's own cannot be told.")),
) -> MarkResponse:
    content = await _photo(image, runtime)
    marked = await runtime.mark_homework(
        content,
        {"user_id": user_id, "subject_id": subject_id, "lecture_ids": lecture_ids,
         "language": language},
        billed_user_id=billed_user_id)
    result = marked.result
    logger.info("homework mark", extra={
        "bytes": len(content), "questions": len(result.questions),
        "unreadable": result.unreadable, "degraded": result.degraded})
    return MarkResponse.from_mark(result, request_id, marked.meter)
