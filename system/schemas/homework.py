"""The homework-capture contract.

An image in, the questions on it out. Capture reads the page and does not answer
it. Mark reads it and grades the student's own answers against the syllabus.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import Field

from agent.homework import CapturedQuestion, HomeworkCapture, MarkedHomework, MarkedQuestion
from shared.metering import Meter

from .common import Language, Schema, Usage


class CaptureResponse(Schema):
    """What one photographed page held."""

    request_id: str
    questions: list[CapturedQuestion] = Field(default_factory=list)
    unreadable: bool = Field(
        default=False,
        description=("True when the photo could not be read. Ask for a clearer one; "
                     "an empty question list means the page had none."),
    )

    usage: Usage

    @classmethod
    def from_capture(cls, capture: HomeworkCapture, request_id: str,
                     meter: Meter) -> CaptureResponse:
        return cls(request_id=request_id, questions=capture.questions,
                   unreadable=capture.unreadable, usage=Usage.from_meter(meter))


class MarkResponse(Schema):
    """One photographed page, read and marked."""

    request_id: str
    unreadable: bool = Field(
        description="True when the photo could not be read. Nothing was marked; ask for a clearer one.")
    language: Optional[Language] = Field(
        default=None, description="The language the feedback is written in: the page's own.")
    questions: list[MarkedQuestion] = Field(
        default_factory=list,
        description=("Every question on the page in order. status is marked, blank (left "
                     "empty, never scored) or not_marked (nothing in the syllabus settles "
                     "it). A grade with needs_review true moved no mastery: show it to the "
                     "teacher and send their decision to POST /ai/grade/reviewed."))
    summary: dict[str, Any] = Field(default_factory=dict)
    mastery_events: int = 0
    topics_updated: int = 0
    mastery_recorded: bool = Field(
        default=True,
        description="False when the answers were marked but the evidence could not be stored.")
    degraded: bool = Field(
        default=False, description="True when grading failed: nothing was marked or charged.")
    usage: Usage

    @classmethod
    def from_mark(cls, marked: MarkedHomework, request_id: str, meter: Meter) -> MarkResponse:
        return cls(request_id=request_id, usage=Usage.from_meter(meter), **marked.model_dump())


class CaptureQuery(Schema):
    """Query parameters for a capture. The image itself is the request body."""

    language: Optional[Language] = Field(
        default=None,
        description=("Helps the model read the script. It never translates — the "
                     "questions come back in the language they are printed in."),
    )
