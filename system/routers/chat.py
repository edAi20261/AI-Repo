"""Chat — the primary endpoint. One message becomes one grounded reply."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from ..dependencies import Authenticated, RequestId, RuntimeDep
from ..errors import code_of
from ..schemas.chat import ChatRequest, ChatResponse
from ..schemas.common import PAID_RESPONSES

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/ai", tags=["chat"], dependencies=[Authenticated], responses=PAID_RESPONSES
)


@router.post("/chat", response_model=ChatResponse, summary="Answer a message from the syllabus")
async def chat(payload: ChatRequest, runtime: RuntimeDep, request_id: RequestId) -> ChatResponse:
    logger.info(
        "chat",
        extra={"user_type": payload.user_type, "subject_id": payload.subject_id,
               "has_grade_ref": payload.grade_ref is not None},
    )
    turn = await runtime.chat(payload.to_state())
    return ChatResponse.from_state(turn.result, request_id, turn.meter)


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
    updates = await runtime.stream_chat(payload.to_state())
    return StreamingResponse(
        _stream(updates, request_id),
        media_type="text/event-stream",
        headers={"cache-control": "no-cache", "x-accel-buffering": "no"},
    )


async def _stream(
    updates: AsyncIterator[tuple[str, Any]], request_id: str
) -> AsyncIterator[str]:
    try:
        async for stage, delta in updates:
            if stage == "__final__":
                result = ChatResponse.from_state(delta.result, request_id, delta.meter)
                break
            yield _event("progress", {"stage": stage})
    except Exception as error:
        logger.exception("chat stream failed")
        yield _event("error", {"code": code_of(error), "request_id": request_id})
        return
    yield _event("result", result.model_dump(mode="json"))


def _event(name: str, data: dict[str, Any]) -> str:
    return f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
