"""The assessment-generation contract.

A lesson scope and a shape in, that many questions out, each grounded in the scoped
lessons and carrying what ``POST /ai/grade`` needs to mark an answer to it. A quiz
that comes back short says so: ``complete`` is false and ``shortfall`` names why.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, model_validator

from agent.graph.subgraphs.grade_sg import answer_key_of
from agent.models import AssessmentOption
from shared.metering import Meter

from .common import Difficulty, Language, QuestionType, Schema, Usage

Shortfall = Literal["generation_failed", "not_enough_content", "items_rejected"]


class AssessmentRequest(Schema):
    user_id: str = Field(description="The student the quiz is for; weak topics get more questions.")
    user_type: Literal["student", "teacher"] = Field(
        description="Sets the largest count allowed (config assessment.max_count).")
    billed_user_id: str = Field(description="The account the credits are taken from.")
    subject_id: Optional[str] = Field(
        default=None, description="Scopes the questions to a subject; required without lecture_ids.")
    lecture_ids: Optional[list[str]] = Field(
        default=None, min_length=1, max_length=50,
        description="The lessons the questions are drawn from.")
    topic_ids: Optional[list[str]] = Field(
        default=None, min_length=1, max_length=50,
        description="Narrows the questions to these topics within the scope.")
    focus: Optional[str] = Field(
        default=None, min_length=1, max_length=500,
        description="What the questions should cover, in the student's words.")
    count: int = Field(ge=1, description="How many questions to generate.")
    types: Optional[list[QuestionType]] = Field(
        default=None, min_length=1,
        description="Question types, shared out evenly over count. Omitted: the configured mix.")
    difficulty: Optional[Difficulty | Literal["mixed"]] = Field(
        default=None, description="Omitted: the configured difficulty.")
    language: Optional[Language] = Field(
        default=None,
        description=("The student's language. The language resolver ranks the subject's "
                     "own language first, so this decides only when that is unknown."))

    @model_validator(mode="after")
    def _scoped(self) -> AssessmentRequest:
        if not (self.subject_id or self.lecture_ids):
            raise ValueError("send lecture_ids, subject_id or both")
        if self.types and len(set(self.types)) != len(self.types):
            raise ValueError("types must not repeat")
        return self

    def to_payload(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=True, exclude={"billed_user_id", "user_type"})


class GeneratedQuestion(BaseModel):
    id: str = Field(description="This question's position in the quiz, q1, q2, ...")
    type: QuestionType
    question: str
    options: Optional[list[AssessmentOption]] = Field(
        default=None, description="For mcq: the options, each marked correct or not.")
    multiple_correct: bool = False
    answer: Optional[str | list[str]] = Field(
        default=None, description="The reference answer; for mcq, the correct options' text.")
    correct_answer: str = Field(
        description="The answer key as POST /ai/grade takes it in a submission's correct_answer.")
    explanation: Optional[str] = None
    difficulty: Optional[Difficulty] = None
    topic_id: str = Field(description="The topic the question tests; send it with the answer to /ai/grade.")
    lecture_id: str = Field(description="The lesson whose passage the question was written from.")

    @classmethod
    def from_item(cls, position: int, item: dict[str, Any]) -> GeneratedQuestion:
        answer = item.get("answer")
        key = answer_key_of({
            "correct_answer_text": answer if isinstance(answer, str) else None,
            "options": item.get("options") or [],
        })
        return cls(id=f"q{position}", correct_answer=key, **item)


class AssessmentResponse(Schema):
    request_id: str
    questions: list[GeneratedQuestion]
    requested_count: int
    count: int = Field(description="Questions returned. Below requested_count only when complete is false.")
    requested_types: dict[str, int] = Field(description="Questions asked for, by type.")
    types: dict[str, int] = Field(description="Questions returned, by type.")
    complete: bool = Field(description="False when fewer questions came back than were asked for.")
    shortfall: Optional[Shortfall] = Field(
        default=None,
        description=("Why the quiz is short, when it is: generation_failed (the model call "
                     "failed; retry), not_enough_content (the scope has too few passages "
                     "for that many questions; widen it or ask for fewer), items_rejected "
                     "(questions failed validation)."))
    degraded: bool = Field(default=False, description="True when the model call failed.")
    language: str = Field(description="The language the questions are written in.")
    intro: str = Field(default="", description="One line introducing the quiz, in the student's language.")
    usage: Usage

    @classmethod
    def from_assessment(cls, assessment: dict[str, Any], request_id: str,
                        meter: Meter) -> AssessmentResponse:
        items = assessment["items"]
        requested = assessment["requested_type_mix"]
        return cls(
            request_id=request_id,
            questions=[GeneratedQuestion.from_item(i, item) for i, item in enumerate(items, 1)],
            requested_count=sum(requested.values()),
            count=len(items),
            requested_types=requested,
            types=assessment["type_mix"],
            complete=assessment["complete"],
            shortfall=assessment["shortfall"],
            degraded=assessment["degraded"],
            language=assessment["language"],
            intro=assessment["intro"],
            usage=Usage.from_meter(meter),
        )
