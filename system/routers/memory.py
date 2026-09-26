"""Memory — what the tutor holds about a student, and how it is destroyed.

Until now nothing this service remembered could be read back or erased by
anybody. The users are children; "what do you know about my child, and delete
it" is a question that gets asked, and it had no answer that was not a database
console.

Who may ask it is not decided here. Every endpoint on this service is behind
the same service token and takes the student it acts on from the caller, which
makes the platform the one place that decides whether this parent may see this
child — the same arrangement chat, grading and reports already run on. Deciding
it twice, in two services, is how the two come to disagree.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Query

from agent.profile import forget, held

from ..dependencies import Authenticated, RequestId
from ..schemas.common import ERROR_RESPONSES
from ..schemas.memory import ForgetResponse, MemoryResponse

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/ai", tags=["memory"], dependencies=[Authenticated], responses=ERROR_RESPONSES
)


@router.get(
    "/memory/{user_id}",
    response_model=MemoryResponse,
    summary="What the tutor remembers about a student",
)
async def read_memory(user_id: str, request_id: RequestId) -> MemoryResponse:
    record = held(user_id)
    logger.info("memory read", extra={"conversations": record["conversations"],
                                      "summaries": len(record["summaries"])})
    return MemoryResponse(**record, request_id=request_id)


@router.delete(
    "/memory/{user_id}",
    response_model=ForgetResponse,
    summary="Destroy what the tutor remembers about a student",
)
async def forget_memory(
    user_id: str,
    request_id: RequestId,
    conversation_id: Optional[str] = Query(
        default=None,
        description=("Forget one conversation. Omit to forget the student entirely, "
                     "which also destroys the preferences they stated.")),
) -> ForgetResponse:
    destroyed = forget(user_id, conversation_id=conversation_id)
    logger.info("memory forgotten", extra={"conversation_id": conversation_id, **destroyed})
    return ForgetResponse(user_id=user_id, conversation_id=conversation_id,
                          request_id=request_id, **destroyed)
