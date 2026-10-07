"""Voice: a spoken question becomes text, and a reply can be heard.

Neither endpoint answers anything. The transcript is sent to /ai/chat like any
typed message, and /ai/voice/speak reads back a reply the chat already wrote.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, File, Query, UploadFile
from fastapi.responses import Response

from agent.voice.service import Capabilities, audio_type
from shared.errors import UnprocessableError

from ..dependencies import Authenticated, RequestId, RuntimeDep
from ..schemas.common import PAID_RESPONSES, Language
from ..schemas.voice import CapabilitiesResponse, SpeakRequest, TranscriptResponse

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/ai/voice", tags=["voice"], dependencies=[Authenticated], responses=PAID_RESPONSES,
)

ACCEPTED = ("audio/webm", "audio/ogg", "audio/mp4", "audio/mpeg", "audio/mp3", "audio/wav",
            "audio/x-wav", "audio/wave", "audio/m4a", "audio/x-m4a", "audio/aac", "audio/flac",
            "video/webm")


@router.get(
    "/capabilities",
    response_model=CapabilitiesResponse,
    summary="Which voice features this service offers, for showing the mic and speaker buttons",
)
async def capabilities(runtime: RuntimeDep, request_id: RequestId) -> CapabilitiesResponse:
    return CapabilitiesResponse.of(Capabilities.of(runtime.config), request_id)


@router.post(
    "/transcribe",
    response_model=TranscriptResponse,
    summary="Turn a recorded question into text",
)
async def transcribe(
    request_id: RequestId,
    runtime: RuntimeDep,
    billed_user_id: str = Query(description="The account the credits are taken from."),
    audio: UploadFile = File(description="One recording, as the browser's MediaRecorder made it."),
    language: Language | None = Query(
        default=None,
        description=("Only when the student will certainly speak this language. Omit it "
                     "otherwise: the language is detected, including Arabic and English "
                     "mixed in one sentence.")),
) -> TranscriptResponse:
    content = await audio.read()
    limit = runtime.config.get("voice.transcribe.max_upload_bytes")
    if not content:
        raise UnprocessableError("The recording was empty.")
    content_type = audio_type(content, audio.content_type)
    if content_type not in ACCEPTED:
        raise UnprocessableError(f"Send the recording as one of {', '.join(ACCEPTED)}.",
                                 details={"received": audio.content_type})
    if len(content) > limit:
        raise UnprocessableError("That recording is too long. Ask one question at a time.",
                                 details={"bytes": len(content), "limit": limit})
    heard = await runtime.transcribe(content, content_type,
                                     language=language, billed_user_id=billed_user_id)
    return TranscriptResponse.of(heard.result, request_id, heard.meter)


@router.post(
    "/speak",
    response_class=Response,
    summary="Read a tutor reply aloud",
    responses={200: {"content": {"audio/mpeg": {}},
                     "description": "The reply as speech. Usage is in the x-usage-* headers."}},
)
async def speak(payload: SpeakRequest, runtime: RuntimeDep, request_id: RequestId) -> Response:
    spoken = await runtime.speak(payload.text, payload.language,
                                 billed_user_id=payload.billed_user_id)
    usage = spoken.meter.as_dict()
    return Response(
        content=spoken.result.audio, media_type=spoken.result.content_type,
        headers={"x-request-id": request_id,
                 "x-usage-cost-usd": str(usage["cost_usd"]),
                 "x-voice": spoken.result.voice,
                 "x-spoken-characters": str(spoken.result.characters),
                 "cache-control": "private, max-age=86400"})
