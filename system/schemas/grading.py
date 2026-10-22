"""The grading contract.

A submission is self-contained: the answer text arrives in the payload, never
read from the platform's database by this service. An ``assessment_id`` may be
supplied so the answer key is fetched and used ahead of the syllabus.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import Field, model_validator

from agent.models import ReviewedGrade
from shared.metering import Meter

from .common import Language, Schema, Submission, Usage


class GradeRequest(Schema):
    user_id: str = Field(description="Whose mastery this grading moves.")
    billed_user_id: str = Field(description="The account the credits are taken from.")
    submissions: list[Submission] = Field(min_length=1, max_length=200)
    subject_id: Optional[str] = None
    lecture_ids: Optional[list[str]] = None
    kind: Optional[Literal["quiz", "homework"]] = None
    assessment_id: Optional[str] = Field(
        default=None, description="Fetches the stored rubric for this assessment."
    )
    source: Optional[Literal["practice", "quiz", "homework", "uploaded"]] = None
    language: Optional[Language] = None

    @model_validator(mode="after")
    def _assessment_requires_kind(self) -> GradeRequest:
        if self.assessment_id and not self.kind:
            raise ValueError("kind is required when assessment_id is supplied")
        return self

    def to_payload(self) -> dict[str, Any]:
        payload = self.model_dump(exclude_none=True, exclude={"billed_user_id"})
        payload["submissions"] = [s.model_dump(exclude_none=True) for s in self.submissions]
        return payload


class GradeResponse(Schema):
    request_id: str
    graded: list[dict[str, Any]] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)
    mastery_events: int = 0
    topics_updated: int = 0
    mastery_recorded: bool = Field(
        default=True,
        description="False when grading succeeded but the evidence could not be stored.",
    )
    source: Optional[str] = None
    degraded: bool = Field(
        default=False, description="True when the model failed and output is a safe fallback."
    )

    usage: Usage

    @classmethod
    def from_result(cls, result: dict[str, Any], request_id: str, meter: Meter) -> GradeResponse:
        known = {k: v for k, v in result.items() if k in cls.model_fields}
        return cls(request_id=request_id, usage=Usage.from_meter(meter), **known)


class ReviewedRequest(Schema):
    """The teacher's decision on grades that came back with ``needs_review``."""

    user_id: str = Field(description="Whose mastery these grades move.")
    source: Literal["practice", "quiz", "homework", "uploaded"] = Field(
        description="The source the held grade came from: the grade response's source.")
    grades: list[ReviewedGrade] = Field(
        min_length=1, max_length=200,
        description=("One per decided grade. ai_score is the held grade's partial_score, "
                     "score the teacher's; topic_id, lecture_id, question_id, difficulty, "
                     "bloom and mistake_tag_ids are copied from the held grade."))


class ReviewedResponse(Schema):
    request_id: str
    mastery_events: int
    topics_updated: int
    mastery_recorded: bool = Field(
        description="False when the evidence could not be stored. Retry; nothing was written.")
