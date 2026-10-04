"""The voice contract: a recording in, its words out; a reply in, its audio out."""

from __future__ import annotations

from typing import Optional

from pydantic import Field

from agent.voice.service import Heard
from shared.metering import Meter

from .common import Language, Schema, Usage


class TranscriptResponse(Schema):
    request_id: str
    text: str = Field(description="What the student said. Put it in the message box for "
                                  "them to read and send.")
    heard: bool = Field(description="False when the recording held no speech. Ask them "
                                    "to try again; nothing was charged.")
    language: Optional[Language] = Field(
        default=None, description="The language it was spoken in, when it is one we teach in.")
    seconds: float = Field(description="How long the speech ran.")
    usage: Usage

    @classmethod
    def of(cls, heard: Heard, request_id: str, meter: Meter) -> TranscriptResponse:
        return cls(request_id=request_id, text=heard.text, heard=heard.heard,
                   language=heard.language, seconds=round(heard.seconds, 2),
                   usage=Usage.from_meter(meter))


class SpeakRequest(Schema):
    text: str = Field(min_length=1, max_length=20000,
                      description="The tutor reply as the chat returned it, Markdown and all.")
    language: Language = Field(description="The reply's `language` from the chat response.")
    billed_user_id: str = Field(description="The account the credits are taken from.")
