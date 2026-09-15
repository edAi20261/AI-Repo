"""The chat contract — one message in, one composed reply out."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import Field

from .common import GradeReference, Language, Pathway, Schema, Visual


class ChatRequest(Schema):
    user_id: str = Field(description="Platform user id; scopes mastery reads and writes.")
    user_type: Pathway = Field(description="Determines the allowed tasks and the voice.")
    message: str = Field(min_length=1, max_length=8000)
    subject_id: Optional[str] = Field(
        default=None, description="Scopes retrieval. Inferred from the message when omitted."
    )
    lecture_ids: Optional[list[str]] = Field(
        default=None, description="Narrows retrieval to specific lessons."
    )
    session_id: Optional[str] = Field(
        default=None, description="Conversation id; also the ephemeral upload namespace."
    )
    grade: Optional[str] = Field(default=None, description="Academic year level, e.g. 'g8'.")
    grade_ref: Optional[GradeReference] = Field(
        default=None, description="Set when the turn grades a saved assessment."
    )
    language: Optional[Language] = Field(
        default=None,
        description=("Answer in this language. Outranks detection — pass it when the "
                     "platform knows the learner's language."),
    )

    def to_state(self) -> dict[str, Any]:
        """Project onto the graph's ``MainState`` input keys."""
        state: dict[str, Any] = {
            "user_id": self.user_id,
            "user_type": self.user_type,
            "message": self.message,
            "subject_id": self.subject_id,
            "lecture_ids": self.lecture_ids,
            "session_id": self.session_id,
            "grade": self.grade,
        }
        if self.grade_ref:
            state["grade_ref"] = self.grade_ref.model_dump()
        if self.language:
            state["language"] = self.language
        return {key: value for key, value in state.items() if value is not None}


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

    @classmethod
    def from_state(cls, state: dict[str, Any], request_id: str) -> ChatResponse:
        return cls(
            request_id=request_id,
            response=state.get("response", ""),
            language=state.get("language", "en"),
            pathway=state.get("pathway"),
            tasks=[task.get("type", "") for task in state.get("tasks") or []],
            visuals=[Visual.model_validate(v) for v in state.get("visuals") or []],
            assessment=state.get("assessment") or {},
            grade_result=state.get("grade_result") or {},
            prereq_topics=[_as_dict(topic) for topic in state.get("prereq_topics") or []],
        )


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return getattr(value, "__dict__", {})
