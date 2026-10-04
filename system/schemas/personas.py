"""The teacher-persona contract."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import Field, model_validator

from agent.persona.build import BuildRequest, BuildResult
from agent.persona.models import Exemplar, Persona, PersonaSource, QuestionnaireAnswer
from shared.metering import Meter

from .common import Language, Schema, Usage


class PersonaBuildRequest(Schema):
    teacher_id: str = Field(description="The teacher's platform user id.")
    teacher_name: str = Field(min_length=1, max_length=200,
                              description="How students address the teacher, e.g. 'Ms Mona'.")
    billed_user_id: str = Field(description="The account the credits are taken from.")
    consented_at: datetime = Field(
        description=("When the teacher agreed to have a persona built from this material. "
                     "The platform keeps the record of that agreement; it is stored here "
                     "beside the persona it allowed."))
    sources: list[PersonaSource] = Field(
        default_factory=list, max_length=50,
        description=("Material in the teacher's own words: transcripts and chat exports "
                     "(say which speaker is the teacher) and written explanations."))
    questionnaire: list[QuestionnaireAnswer] = Field(
        default_factory=list, max_length=30,
        description="The teacher's own answers about how they teach.")
    language: Language = Field(default="en",
                               description="The language the card is written in for review.")

    @model_validator(mode="after")
    def _has_material(self) -> PersonaBuildRequest:
        if not self.sources and not self.questionnaire:
            raise ValueError("a persona needs sources, questionnaire answers, or both")
        ids = [s.source_id for s in self.sources]
        if len(ids) != len(set(ids)):
            raise ValueError("source_id must be unique within a request")
        return self

    def to_request(self) -> BuildRequest:
        return BuildRequest(
            teacher_id=self.teacher_id, teacher_name=" ".join(self.teacher_name.split()),
            sources=list(self.sources), questionnaire=list(self.questionnaire),
            language=self.language, consented_at=self.consented_at)


class ExemplarView(Schema):
    text: str
    prompt: str = Field(description="What the student had said just before, if anything.")
    move: str = Field(description="What the passage shows about how the teacher explains.")
    source_id: str

    @classmethod
    def of(cls, exemplar: Exemplar) -> ExemplarView:
        return cls(text=exemplar.text, prompt=exemplar.prompt, move=exemplar.move,
                   source_id=exemplar.source_id)


class SkippedSourceView(Schema):
    source_id: str
    reason: str


class PersonaBuildResponse(Schema):
    request_id: str
    persona: Persona = Field(description="The new version, stored as a draft.")
    exemplars: list[ExemplarView]
    skipped: list[SkippedSourceView] = Field(
        description="Sources that could not be used, and why.")
    usage: Usage

    @classmethod
    def of(cls, result: BuildResult, request_id: str, meter: Meter) -> PersonaBuildResponse:
        return cls(
            request_id=request_id, persona=result.persona,
            exemplars=[ExemplarView.of(e) for e in result.exemplars],
            skipped=[SkippedSourceView(source_id=s.source_id, reason=s.reason)
                     for s in result.skipped],
            usage=Usage.from_meter(meter))


class PersonaListResponse(Schema):
    request_id: str
    active_version: Optional[int] = Field(
        description="The version students hear, or null when none is approved.")
    versions: list[Persona] = Field(description="Every version, newest first.")


class PersonaDetailResponse(Schema):
    request_id: str
    persona: Persona
    exemplars: list[ExemplarView]


class PersonaResponse(Schema):
    request_id: str
    persona: Persona


class PersonaDeactivateResponse(Schema):
    request_id: str
    deactivated_version: Optional[int] = Field(
        description="The version taken off students, or null when none was active.")


class PersonaDeleteResponse(Schema):
    request_id: str
    versions_deleted: int
