"""The grading contract.

A submission is self-contained: the answer text arrives in the payload, never
read from the platform's database by this service. An ``assessment_id`` may be
supplied so the answer key is fetched and used ahead of the syllabus.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import Field, model_validator

from .common import Language, Schema

QuestionType = Literal["mcq", "true_false", "short_answer", "essay"]
Difficulty = Literal["easy", "medium", "hard"]


class Submission(Schema):
    question: str = Field(min_length=1)
    student_answer: str
    correct_answer: Optional[str] = Field(
        default=None, description="Reference answer; used ahead of the syllabus when present."
    )
    question_id: Optional[str] = None
    topic_id: Optional[str] = None
    lecture_id: Optional[str] = None
    type: Optional[QuestionType] = None
    difficulty: Optional[Difficulty] = None
    bloom: Optional[str] = None
    points: Optional[float] = Field(default=None, ge=0)


class GradeRequest(Schema):
    user_id: str = Field(description="Whose mastery this grading moves.")
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
        payload = self.model_dump(exclude_none=True)
        payload["submissions"] = [s.model_dump(exclude_none=True) for s in self.submissions]
        return payload


class GradeResponse(Schema):
    request_id: str
    graded: list[dict[str, Any]] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)
    mastery_events: int = 0
    topics_updated: int = 0
    source: Optional[str] = None
    degraded: bool = Field(
        default=False, description="True when the model failed and output is a safe fallback."
    )

    @classmethod
    def from_result(cls, result: dict[str, Any], request_id: str) -> GradeResponse:
        return cls(request_id=request_id, **{k: v for k, v in result.items() if k in cls.model_fields})
