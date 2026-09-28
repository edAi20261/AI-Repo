"""The topic-suggestion contract."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import Field

from .common import Schema


class QuestionText(Schema):
    text: str = Field(min_length=1)
    question_id: Optional[str] = None


class TopicSuggestRequest(Schema):
    subject_id: str = Field(description="Suggestions come from this subject's syllabus only.")
    lecture_ids: Optional[list[str]] = Field(
        default=None, description="Narrows suggestions to these lessons.")
    questions: list[QuestionText] = Field(min_length=1, max_length=200)

    def to_payload(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=True)


class TopicSuggestResponse(Schema):
    request_id: str
    suggestions: list[dict[str, Any]] = Field(
        default_factory=list,
        description="One entry per question, in request order. `resolved: false` means "
                    "the lookup failed, not that nothing matched.",
    )
    degraded: bool = False

    @classmethod
    def from_result(cls, result: dict[str, Any], request_id: str) -> TopicSuggestResponse:
        known = {k: v for k, v in result.items() if k in cls.model_fields}
        return cls(request_id=request_id, **known)
