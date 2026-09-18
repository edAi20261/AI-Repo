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
from typing import Optional

from pydantic import BaseModel, Field, field_validator

from shared.config import Config
from shared.structured import schema_hint
from shared.vision import read_image

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
