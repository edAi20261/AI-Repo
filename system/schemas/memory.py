"""What the tutor holds about a student, and the receipt for destroying it."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class RememberedSummary(BaseModel):
    """One conversation as the tutor summarised it to itself."""

    conversation_id: Optional[str] = None
    note: Optional[str] = Field(
        default=None,
        description=("What the tutor remembers from this conversation, in one or two plain "
                     "sentences for the student. Show this. Null for a conversation "
                     "summarised before notes were written."))
    summary: str = Field(
        description="The tutor's own working summary, written for the next turn's prompt.")
    written_at: Optional[datetime] = None
    expires_at: Optional[datetime] = Field(
        default=None,
        description="When this is discarded on its own. Null means it is kept until deleted.")


class MemoryResponse(BaseModel):
    """Everything this service holds about one student.

    Deliberately not the transcript: the words of a conversation belong to the
    platform and the student is already shown them. This is what the tutor made
    of those words, which is the part nobody could see.
    """

    user_id: str
    conversations: int = Field(description="Conversations this student has memory for.")
    summaries: list[RememberedSummary] = Field(default_factory=list)
    stated_preferences: Optional[dict[str, Any]] = Field(
        default=None,
        description="Preferences the student asked for in so many words. Never inferred.")
    request_id: str


class ForgetResponse(BaseModel):
    """What was destroyed. Irreversible."""

    user_id: str
    conversation_id: Optional[str] = Field(
        default=None, description="Null when the whole student was forgotten.")
    facts_deleted: int
    profile_deleted: bool = Field(
        description="Stated preferences are deleted only when the whole student is.")
    transcript_retained: bool = Field(
        default=True,
        description=("The platform owns the conversation transcript and this endpoint "
                     "cannot reach it. A student asking to be forgotten has to be "
                     "forgotten there too."))
    request_id: str
