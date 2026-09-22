"""What the tutor keeps of a conversation between turns.

A turn used to be answered from its own message alone. The platform stored the
transcript and never sent it; the service kept a language and a pointer to the
last answer's passages, and every other turn started from nothing. "Give me an
example" could only be answered by guessing what it was an example of.

These are the facts a later turn needs: what was said, what the answer rested
on, which subject and language were in force. They are the model's record of the
conversation, not the transcript the student sees, which the platform owns.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

TurnKind = Literal["conversation", "study"]


class Turn(BaseModel):
    """One exchange, as a later turn needs to see it."""

    id: Optional[str] = None
    message: str
    reply: str = ""
    kind: TurnKind = "study"
    language: Optional[str] = None
    subject_id: Optional[str] = None
    tasks: list[str] = Field(default_factory=list)
    query: str = ""
    cited_chunk_ids: list[str] = Field(default_factory=list)


class SessionState(BaseModel):
    """What stays in force across turns until the student changes it."""

    subject_id: Optional[str] = None
    language: Optional[str] = None


class Conversation(BaseModel):
    """Everything remembered for one session: state, the folded past, the recent turns."""

    state: SessionState = Field(default_factory=SessionState)
    summary: str = ""
    turns: list[Turn] = Field(default_factory=list)

    @property
    def last_study_turn(self) -> Optional[Turn]:
        """The most recent turn that answered from the syllabus."""
        for turn in reversed(self.turns):
            if turn.kind == "study" and turn.cited_chunk_ids:
                return turn
        return None

    @property
    def is_new(self) -> bool:
        return not (self.turns or self.summary)
