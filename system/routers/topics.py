"""Topic suggestions for questions written outside ingestion."""

from __future__ import annotations

import logging

from fastapi import APIRouter

from ..dependencies import Authenticated, RequestId, RuntimeDep
from ..schemas.common import ERROR_RESPONSES
from ..schemas.topics import TopicSuggestRequest, TopicSuggestResponse

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/ai", tags=["topics"], dependencies=[Authenticated], responses=ERROR_RESPONSES
)


@router.post(
    "/topics/suggest",
    response_model=TopicSuggestResponse,
    summary="Suggest syllabus topics for questions",
)
async def suggest(
    payload: TopicSuggestRequest, runtime: RuntimeDep, request_id: RequestId
) -> TopicSuggestResponse:
    result = await runtime.suggest_topics(payload.to_payload())
    return TopicSuggestResponse.from_result(result, request_id)
