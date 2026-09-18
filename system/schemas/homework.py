"""The homework-capture contract.

An image in, the questions on it out. The service reads the page; it does not
answer it — the answer is a separate turn, grounded in the syllabus.
"""

from __future__ import annotations

from typing import Optional

from pydantic import Field

from agent.homework import CapturedQuestion, HomeworkCapture

from .common import Language, Schema


class CaptureResponse(Schema):
    """What one photographed page held."""

    request_id: str
    questions: list[CapturedQuestion] = Field(default_factory=list)
    unreadable: bool = Field(
        default=False,
        description=("True when the photo could not be read. Ask for a clearer one; "
                     "an empty question list means the page had none."),
    )

    @classmethod
    def from_capture(cls, capture: HomeworkCapture, request_id: str) -> CaptureResponse:
        return cls(request_id=request_id, questions=capture.questions,
                   unreadable=capture.unreadable)


class CaptureQuery(Schema):
    """Query parameters for a capture. The image itself is the request body."""

    language: Optional[Language] = Field(
        default=None,
        description=("Helps the model read the script. It never translates — the "
                     "questions come back in the language they are printed in."),
    )
