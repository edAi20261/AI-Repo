"""Reading a student's homework off a photo.

A student who is stuck rarely types the question out. They photograph the page,
and what arrives is an image holding several numbered questions, sometimes with
their own handwritten answers beside them.

Turning that into structured questions is what makes the rest possible: each one
can then be explained from the syllabus or marked against it. Guessing at a
blurred page is the one thing this must not do — a misread question produces a
confident answer to something the student never asked.
"""

from __future__ import annotations

import logging
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from shared.config import Config
from shared.language import LanguagePolicy
from shared.structured import schema_hint
from shared.vision import read_image

from .graph.grade import run_grade
from .models import GradedItem

logger = logging.getLogger(__name__)

MAX_QUESTIONS = 20


class CapturedQuestion(BaseModel):
    """One question read off the page, with the student's answer if they wrote one."""

    number: Optional[str] = Field(
        default=None, description="As printed — '3', 'Q4', 'ثانيًا'.")
    question: str = Field(description="The question text, copied exactly.")
    student_answer: Optional[str] = Field(
        default=None,
        description="What the student wrote, or null where they left it blank.")
    marks: Optional[float] = Field(default=None, ge=0)
    unclear: bool = Field(
        default=False,
        description="True when any word or symbol of the question or the answer was a guess.")

    @field_validator("unclear", mode="before")
    @classmethod
    def _null_is_clear(cls, v):
        return bool(v)

    @field_validator("number", "student_answer", mode="before")
    @classmethod
    def _blank_is_absent(cls, v):
        return (str(v).strip() or None) if v is not None else None

    @field_validator("question", mode="before")
    @classmethod
    def _clean_question(cls, v):
        return " ".join(str(v or "").split())


class HomeworkCapture(BaseModel):
    """What one photograph of a homework page contained.

    ``unreadable`` exists so a bad photo can be answered honestly. Without it a
    blurred page and a blank one are the same empty list, and the student is
    told their homework has no questions on it.
    """

    questions: list[CapturedQuestion] = Field(default_factory=list)
    unreadable: bool = False

    @field_validator("questions", mode="before")
    @classmethod
    def _drop_questionless_entries(cls, v):
        if not isinstance(v, list):
            return []
        return [q for q in v
                if isinstance(q, dict) and str(q.get("question", "")).strip()][:MAX_QUESTIONS]

    @property
    def answered(self) -> list[CapturedQuestion]:
        """The questions the student attempted — the ones there is work to mark."""
        return [q for q in self.questions if q.student_answer]


def capture(
    image_png: bytes, *, config: Config, language_hint: Optional[str] = None
) -> HomeworkCapture:
    """The questions on a photographed homework page.

    A page no model could read comes back ``unreadable`` rather than empty, so
    the tutor can ask for a clearer photo instead of insisting there is nothing
    on the page.
    """
    from .prompts import HOMEWORK_CAPTURE_PROMPT

    result = read_image(
        image_png,
        HOMEWORK_CAPTURE_PROMPT.render(
            language_hint=language_hint or "", schema=schema_hint(HomeworkCapture)),
        HomeworkCapture,
        config=config,
        label="homework_capture",
    )
    if result is None:
        return HomeworkCapture(unreadable=True)
    if not result.questions and not result.unreadable:
        logger.info("homework photo held no questions")
    return result


MarkStatus = Literal["marked", "blank", "not_marked"]


class MarkedQuestion(CapturedQuestion):
    """A question off the page with what became of the student's answer.

    ``blank`` is a question the student left empty; it is reported, never scored
    as wrong, because a page photographed half-done is not a page answered badly.
    ``not_marked`` is an answer nothing in the syllabus could settle.
    """

    status: MarkStatus
    grade: Optional[GradedItem] = None


class MarkedHomework(BaseModel):
    """One photographed page, read and marked."""

    unreadable: bool = False
    language: Optional[str] = None
    questions: list[MarkedQuestion] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)
    mastery_events: int = 0
    topics_updated: int = 0
    mastery_recorded: bool = True
    degraded: bool = False


def mark(
    image_png: bytes,
    *,
    config: Config,
    user_id: Optional[str],
    subject_id: Optional[str] = None,
    lecture_ids: Optional[list[str]] = None,
    language: Optional[str] = None,
) -> MarkedHomework:
    """Read a photographed page and mark every answer on it against the syllabus.

    The questions and answers are the student's own, copied off the page, so they
    are graded as any other submission is: grounded in the lessons, never in the
    model's own knowledge, and held for a teacher when the grade or the reading
    is in doubt.
    """
    page = capture(image_png, config=config, language_hint=language)
    if page.unreadable:
        return MarkedHomework(unreadable=True)

    answered = page.answered
    feedback_language = _feedback_language(page, language, config)
    result = run_grade({
        "submissions": [{"question": q.question, "student_answer": q.student_answer,
                         "points": q.marks, "unclear_reading": q.unclear}
                        for q in answered],
        "user_id": user_id,
        "subject_id": subject_id,
        "lecture_ids": lecture_ids,
        "language": feedback_language,
        "source": "uploaded",
    }, config=config)

    grades = [GradedItem.model_validate(g) for g in result.get("graded") or []]
    if len(grades) != len(answered):
        raise RuntimeError(
            f"{len(answered)} answers were sent for marking and {len(grades)} came back")
    by_question = dict(zip(map(id, answered), grades))
    questions = [_marked(q, by_question.get(id(q))) for q in page.questions]
    blank = sum(1 for q in questions if q.status == "blank")
    logger.info("homework marked", extra={
        "questions": len(questions), "blank": blank,
        "not_marked": sum(1 for q in questions if q.status == "not_marked"),
        "needs_review": sum(1 for g in grades if g.needs_review),
        "unclear": sum(1 for q in page.questions if q.unclear),
        "degraded": result.get("degraded", False)})
    return MarkedHomework(
        language=feedback_language,
        questions=questions,
        summary={**(result.get("summary") or {}), "blank": blank},
        mastery_events=result.get("mastery_events", 0),
        topics_updated=result.get("topics_updated", 0),
        mastery_recorded=result.get("mastery_recorded", True),
        degraded=result.get("degraded", False),
    )


def _marked(question: CapturedQuestion, grade: Optional[GradedItem]) -> MarkedQuestion:
    if grade is None:
        return MarkedQuestion(**question.model_dump(), status="blank")
    status: MarkStatus = "marked" if grade.graded else "not_marked"
    return MarkedQuestion(**question.model_dump(), status=status, grade=grade)


def _feedback_language(page: HomeworkCapture, language: Optional[str], config: Config) -> str:
    """The page's own language, read from its questions; the caller's only when the
    page cannot tell. A student who wrote their homework in Arabic is marked in it."""
    policy = LanguagePolicy.from_config(config)
    printed = " ".join(q.question for q in page.questions)
    return policy.reply(message_language=policy.written_in(printed), api_language=language,
                        subject_chosen=False).code
