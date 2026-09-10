"""Chat — the primary endpoint. One message becomes one grounded reply."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from ..dependencies import Authenticated, RequestId, RuntimeDep
from ..runtime import AgentRuntime
from ..schemas.chat import ChatRequest, ChatResponse
from ..schemas.common import ERROR_RESPONSES

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/ai", tags=["chat"], dependencies=[Authenticated], responses=ERROR_RESPONSES
)


@router.post("/chat", response_model=ChatResponse, summary="Answer a message from the syllabus")
async def chat(payload: ChatRequest, runtime: RuntimeDep, request_id: RequestId) -> ChatResponse:
    logger.info(
        "chat",
        extra={"user_type": payload.user_type, "subject_id": payload.subject_id,
               "has_grade_ref": payload.grade_ref is not None},
    )
    state = await runtime.chat(payload.to_state())
    return ChatResponse.from_state(state, request_id)


@router.post(
    "/chat/stream",
    summary="Answer a message, streaming stage-by-stage progress",
    response_class=StreamingResponse,
)
async def chat_stream(
    payload: ChatRequest, runtime: RuntimeDep, request_id: RequestId
) -> StreamingResponse:
    """Server-sent events: a ``progress`` event per graph stage, then one
    ``result`` event carrying the same body as ``POST /ai/chat``."""
    return StreamingResponse(
        _stream(runtime, payload, request_id),
        media_type="text/event-stream",
        headers={"cache-control": "no-cache", "x-accel-buffering": "no"},
    )


async def _stream(
    runtime: AgentRuntime, payload: ChatRequest, request_id: str
) -> AsyncIterator[str]:
    state: dict[str, Any] = {}
    try:
        async for stage, delta in runtime.stream_chat(payload.to_state()):
            if stage == "__final__":
                state = delta
                break
            yield _event("progress", {"stage": stage})
    except Exception:
        logger.exception("chat stream failed")
        yield _event("error", {"code": "internal_error", "request_id": request_id})
        return
    yield _event("result", ChatResponse.from_state(state, request_id).model_dump(mode="json"))


def _event(name: str, data: dict[str, Any]) -> str:
    return f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
