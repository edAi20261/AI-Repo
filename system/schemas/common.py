"""Shapes shared by more than one endpoint."""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from shared.language import Lang
from shared.metering import Meter

Pathway = Literal["student", "teacher", "parent"]
# One definition of the supported set, so the API, the resolver and the string
# catalogue cannot drift into disagreeing about what "supported" means.
Language = Lang


class Schema(BaseModel):
    """Base for request bodies: unknown fields are rejected rather than ignored,
    so a client typo surfaces as a 422 instead of silently doing nothing."""

    model_config = ConfigDict(extra="forbid")


class Usage(Schema):
    cost_usd: float = Field(description="What the model gateway billed for every call in the turn.")
    prompt_tokens: int
    completion_tokens: int
    calls: int = Field(description="Model calls the turn made, retries included.")

    @classmethod
    def from_meter(cls, meter: Meter) -> Usage:
        return cls.model_validate(meter.as_dict())


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
    known_score: Optional[float] = Field(
        default=None, ge=0, le=1,
        description="The platform already scored this answer (a fraction of full marks). "
                    "It moves mastery as given and is not regraded by the model.",
    )


class GradeReference(Schema):
    """Points a grading turn at a saved assessment so its answer key is used.

    A turn carrying one is the student reviewing work the platform already marked,
    so it records nothing: that attempt's evidence was written when it was marked.
    """

    assessment_id: str
    kind: Literal["quiz", "homework"]
    submissions: Optional[list[Submission]] = Field(
        default=None, max_length=200,
        description=("The student's answers to this assessment, one per question, in the "
                     "shape POST /ai/grade takes. Without them the turn walks through every "
                     "question with its correct answer and says the student's own answers "
                     "were not available."))


class Visual(BaseModel):
    """A rendered study visual. Only the fields relevant to ``kind`` are present."""

    kind: str
    artifact_id: Optional[str] = None
    s3_uri: Optional[str] = None
    cards: Optional[list[dict[str, Any]]] = None
    text: Optional[str] = None
    figure_id: Optional[str] = None
    caption: Optional[str] = None
    item_id: Optional[str] = None


class ErrorDetail(BaseModel):
    code: str
    message: str
    request_id: str = ""
    details: dict[str, Any] = Field(default_factory=dict)


class ErrorEnvelope(BaseModel):
    """The body of every non-2xx response."""

    error: ErrorDetail


# Attached to every authenticated router so the error shape appears in /docs.
ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    401: {"model": ErrorEnvelope, "description": "Missing or invalid service token"},
    422: {"model": ErrorEnvelope, "description": "Request failed validation"},
    500: {"model": ErrorEnvelope, "description": "Unexpected error"},
}

PAID_RESPONSES: dict[int | str, dict[str, Any]] = {
    **ERROR_RESPONSES,
    402: {"model": ErrorEnvelope, "description": "The paying account has too few credits"},
}
