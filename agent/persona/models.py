"""The shapes a persona is built from and stored as."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator

from ..models import PersonaCard, PersonaVoice


class SourceKind(StrEnum):
    EXPLANATION = "explanation"
    TRANSCRIPT = "transcript"
    CHAT = "chat"


class PersonaStatus(StrEnum):
    DRAFT = "draft"
    ACTIVE = "active"
    INACTIVE = "inactive"


class PersonaSource(BaseModel):
    """One piece of material a teacher uploaded about how they teach.

    ``teacher_speaker`` is the label the teacher appears under in a conversation.
    A conversation with more than one speaker and no label is refused rather
    than guessed at: taking a student's words for the teacher's would teach the
    tutor to speak like the student.
    """

    source_id: str = Field(min_length=1, max_length=120)
    kind: SourceKind
    text: Optional[str] = None
    uri: Optional[str] = None
    teacher_speaker: Optional[str] = Field(default=None, max_length=120)
    title: Optional[str] = Field(default=None, max_length=200)

    @model_validator(mode="after")
    def _one_body(self) -> PersonaSource:
        if bool(self.text) == bool(self.uri):
            raise ValueError("a source carries exactly one of text or uri")
        return self


class QuestionnaireAnswer(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    answer: str = Field(min_length=1, max_length=4000)


@dataclass(frozen=True)
class Utterance:
    speaker: Optional[str]
    text: str
    by_teacher: bool


@dataclass(frozen=True)
class Passage:
    """A stretch of the teacher's own words, with what the student had just said."""

    index: int
    source_id: str
    text: str
    prompt: str = ""


@dataclass(frozen=True)
class SkippedSource:
    source_id: str
    reason: str


@dataclass(frozen=True)
class Material:
    passages: list[Passage]
    corpus: str
    used: list[str] = field(default_factory=list)
    skipped: list[SkippedSource] = field(default_factory=list)


class Exemplar(BaseModel):
    """A passage kept as an example of how the teacher explains."""

    text: str
    prompt: str = ""
    move: str = ""
    source_id: str
    embedding: list[float] = Field(default_factory=list, repr=False)


class Persona(BaseModel):
    persona_id: str
    teacher_id: str
    version: int
    status: PersonaStatus
    teacher_name: str
    card: PersonaCard
    exemplar_count: int = 0
    sources: dict[str, Any] = Field(default_factory=dict)
    consented_at: datetime
    created_at: datetime
    updated_at: datetime
    activated_at: Optional[datetime] = None

    def voice(self) -> PersonaVoice:
        return PersonaVoice(persona_id=self.persona_id, teacher_id=self.teacher_id,
                            version=self.version, teacher_name=self.teacher_name,
                            card=self.card)
