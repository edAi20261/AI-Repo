"""The language decision — what language a turn is answered in, and why.

A turn's language is resolved once and carried, never re-derived. The decision
records its own provenance so a wrong answer can be traced to the rule that
produced it rather than guessed at.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

Lang = Literal["ar", "en", "fr", "de"]

LanguageSource = Literal[
    "explicit",   # the message itself asked for a language
    "session",    # an earlier message in this session asked, and it still stands
    "api",        # the caller passed one
    "script",     # decided by writing system
    "detected",   # statistical detection, above the confidence floor
    "planner",    # the planner read a request the matcher did not
    "profile",    # the learner's stored preference
    "previous",   # the language of the previous turn
    "default",    # configured fallback
]

Script = Literal["arabic", "latin", "arabizi", "mixed", "none"]

ENDONYMS: dict[str, str] = {
    "ar": "العربية",
    "en": "English",
    "fr": "français",
    "de": "Deutsch",
}

ENGLISH_NAMES: dict[str, str] = {
    "ar": "Arabic",
    "en": "English",
    "fr": "French",
    "de": "German",
}


class LanguageDecision(BaseModel):
    """The resolved language for one turn, with the evidence behind it."""

    model_config = ConfigDict(frozen=True)

    code: str
    source: LanguageSource
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    script: Script = "none"
    detector: Optional[str] = None
    refused: bool = False
    evidence: Optional[str] = None

    @property
    def is_explicit(self) -> bool:
        """True when the user or caller named the language outright."""
        return self.source in ("explicit", "api")

    @property
    def verifiable(self) -> bool:
        """Whether the produced answer may be checked against this decision.

        A refused detection means we never knew the language; holding the answer
        to it would repair correct replies into the wrong language.
        """
        return not self.refused

    def log_fields(self) -> dict[str, object]:
        return {
            "language": self.code,
            "language_source": self.source,
            "language_confidence": round(self.confidence, 3),
            "language_script": self.script,
        }
