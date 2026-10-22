"""The chat contract — one message in, one composed reply out."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import Field

from agent.models import AnswerBasis
from agent.scope import StudyScope
from shared.metering import Meter

from .common import GradeReference, Language, Pathway, Schema, Usage, Visual


class ChatRequest(Schema):
    user_id: str = Field(description="Platform user id; scopes mastery reads and writes.")
    user_type: Pathway = Field(description="Determines the allowed tasks and the voice.")
    message: str = Field(min_length=1, max_length=8000)
    subject_id: Optional[str] = Field(
        default=None,
        description=("Scopes retrieval. Falls back to study_scope.subject_id, then is "
                     "inferred from the message when neither is sent."),
    )
    lecture_ids: Optional[list[str]] = Field(
        default=None, description="Narrows retrieval to specific lessons."
    )
    study_scope: Optional[StudyScope] = Field(
        default=None,
        description=("What the student has selected in the UI. With a lecture_id the "
                     "turn answers from that lesson only, and a question outside it is "
                     "refused by name rather than as a missing-material message."),
    )
    session_id: Optional[str] = Field(
        default=None, description="Conversation id; also the ephemeral upload namespace."
    )
    grade: Optional[str] = Field(default=None, description="Academic year level, e.g. 'g8'.")
    grade_ref: Optional[GradeReference] = Field(
        default=None, description="Set when the turn grades a saved assessment."
    )
    teacher_id: Optional[str] = Field(
        default=None,
        description=("The teacher of the class this student is studying the subject in. "
                     "When they have an approved persona, the tutor explains the way they do."),
    )
    persona_version: Optional[int] = Field(
        default=None, ge=1,
        description=("Previews one version of a persona before it is approved. Only the "
                     "teacher it belongs to may send it: user_type teacher, with user_id "
                     "equal to teacher_id."),
    )
    language: Optional[Language] = Field(
        default=None,
        description=("Fallback answer language, used only when the message itself "
                     "cannot decide — too short to detect, Arabizi, a bare equation. "
                     "The message always wins: Arabic in, Arabic out, and an "
                     "in-message 'reply in Arabic' wins over both. Do not send an "
                     "interface or UI language here."),
    )

    def to_state(self) -> dict[str, Any]:
        """Project onto the graph's ``MainState`` input keys."""
        state: dict[str, Any] = {
            "user_id": self.user_id,
            "user_type": self.user_type,
            "message": self.message,
            "subject_id": self.subject_id or (self.study_scope.subject_id
                                              if self.study_scope else None),
            "lecture_ids": self.lecture_ids,
            "session_id": self.session_id,
            "grade": self.grade,
            "teacher_id": self.teacher_id,
            "persona_version": self.persona_version,
        }
        if self.grade_ref:
            state["grade_ref"] = self.grade_ref.model_dump()
        if self.study_scope:
            state["study_scope"] = self.study_scope.model_dump()
        if self.language:
            state["language"] = self.language
        return {key: value for key, value in state.items() if value is not None}


class SourceRef(Schema):
    """One source the reply cites, as structured data.

    The reply text carries ``[n]`` markers; this is what each ``n`` refers to. A
    client that renders the markers as badges needs the label and the page, and
    should not have to parse them back out of the prose.
    """

    n: int = Field(description="The inline marker this source is referenced by.")
    lecture_id: Optional[str] = None
    lesson_title: Optional[str] = None
    lesson_order: Optional[int] = None
    subject_name: Optional[str] = None
    heading_path: Optional[str] = Field(
        default=None, description="Where in the lesson, e.g. 'Unit 2 > Subtraction'."
    )
    page_from: Optional[int] = None
    page_to: Optional[int] = None

    @classmethod
    def from_citation(cls, citation: Any) -> SourceRef:
        """Project an internal citation onto the published fields.

        Explicit rather than a wholesale validate: the internal record also carries
        routing detail (which task cited it) that the contract forbids and a client
        has no use for.
        """
        data = _as_dict(citation)
        return cls(**{field: data.get(field) for field in cls.model_fields
                      if field != "n"}, n=data.get("n", 0))


class ChatResponse(Schema):
    request_id: str
    response: str = Field(description="The composed reply, with inline citations.")
    language: Language
    pathway: Optional[str] = None
    tasks: list[str] = Field(default_factory=list, description="Tasks the planner selected.")
    visuals: list[Visual] = Field(default_factory=list)
    assessment: dict[str, Any] = Field(default_factory=dict)
    grade_result: dict[str, Any] = Field(default_factory=dict)
    prereq_topics: list[dict[str, Any]] = Field(default_factory=list)
    citations: list[SourceRef] = Field(
        default_factory=list,
        description="Sources the reply cites, numbered to match its [n] markers.",
    )
    answer_basis: list[AnswerBasis] = Field(
        default_factory=list,
        description=("Where each explanation or summary in the reply came from: lesson "
                     "(it cites the student's lessons), beyond (it teaches, inside the "
                     "subject, what the lessons leave out) or declined."),
    )
    outside_lesson: bool = Field(
        default=False,
        description=("True when the student had a lesson selected and the question was "
                     "declined because that lesson does not cover it. Show a notice that "
                     "the question is outside the selected lesson."),
    )
    persona_version: Optional[int] = Field(
        default=None,
        description="The version of the teacher's persona the reply was written in, if any.")
    usage: Usage

    @classmethod
    def from_state(cls, state: dict[str, Any], request_id: str, meter: Meter) -> ChatResponse:
        return cls(
            request_id=request_id,
            usage=Usage.from_meter(meter),
            response=state.get("response", ""),
            language=state.get("language", "en"),
            pathway=state.get("pathway"),
            tasks=[task.get("type", "") for task in state.get("tasks") or []],
            visuals=[Visual.model_validate(v) for v in state.get("visuals") or []],
            assessment=state.get("assessment") or {},
            grade_result=state.get("grade_result") or {},
            prereq_topics=[_as_dict(topic) for topic in state.get("prereq_topics") or []],
            citations=[SourceRef.from_citation(c)
                       for c in state.get("published_citations") or []],
            answer_basis=[a["basis"] for a in state.get("answers") or [] if a.get("basis")],
            outside_lesson=any(a.get("outside_lesson") for a in state.get("answers") or []),
            persona_version=_persona_version(state.get("style")),
        )


def _persona_version(style: Any) -> Optional[int]:
    persona = getattr(style, "persona", None)
    return persona.version if persona else None


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return getattr(value, "__dict__", {})
