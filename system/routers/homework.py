"""Homework — read a photographed page into structured questions."""

from __future__ import annotations

import logging

from fastapi import APIRouter, File, Query, UploadFile

from shared.errors import UnprocessableError

from ..dependencies import Authenticated, RequestId, RuntimeDep
from ..schemas.common import PAID_RESPONSES, Language
from ..schemas.homework import CaptureResponse

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/ai/homework", tags=["homework"],
    dependencies=[Authenticated], responses=PAID_RESPONSES,
)

ACCEPTED = ("image/png", "image/jpeg", "image/webp")


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

    captured = await runtime.capture_homework(content, language=language,
                                              billed_user_id=billed_user_id)
    result = captured.result
    logger.info("homework capture", extra={
        "bytes": len(content), "questions": len(result.questions),
        "unreadable": result.unreadable})
    return CaptureResponse.from_capture(result, request_id, captured.meter)
