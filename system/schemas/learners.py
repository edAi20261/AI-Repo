"""The student-analysis contract."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from .common import Schema


class LearnerSummaryResponse(Schema):
    request_id: str
    available: bool = Field(description="False until some topic has enough graded work to judge.")
    degraded: bool = Field(
        default=False, description="A read failed; what is present is true but incomplete.")
    subjects: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Per subject: topic count and how many are strong, developing, weak, "
                    "unknown (too little evidence) and not_practised.")
    strengths: list[dict[str, Any]] = Field(default_factory=list)
    weaknesses: list[dict[str, Any]] = Field(default_factory=list)
    not_practised: list[dict[str, Any]] = Field(
        default_factory=list, description="Untouched topics in the lessons the student is in.")
    recommendations: list[dict[str, Any]] = Field(
        default_factory=list,
        description="What to practise next, in order. `reasons` are facts with a `code`; "
                    "the client words them in the reader's language.")
    mistakes: list[dict[str, Any]] = Field(
        default_factory=list, description="The student's commonest mistake categories.")
    activity: list[dict[str, Any]] = Field(
        default_factory=list, description="Graded answers per week, oldest first.")
    trend: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_result(cls, result: dict[str, Any], request_id: str) -> LearnerSummaryResponse:
        known = {k: v for k, v in result.items() if k in cls.model_fields}
        return cls(request_id=request_id, **known)
