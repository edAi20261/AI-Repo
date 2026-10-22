"""The one place a turn's reply language is decided.

Two different questions used to be answered by one chain of regexes: what
language the student wrote in, and what language they asked to be answered in.
Word lists cannot tell "answer in French" from "why did you answer in French?",
and the second was stored as an instruction for a month. Reading a request is a
judgement about meaning, so it belongs to the model that reads the conversation;
this module only ranks the facts it is handed:

    a request the student made in this conversation
    -> the language of the subject being studied
    -> the language the student wrote in
    -> the caller's parameter
    -> the configured default

Every subject is taught in its own language. A student who writes Arabic in an
English science lesson is answered in English unless they ask otherwise, and
that request holds until they change it. ``reply_follows`` switches the second
and third ranks for a deployment that answers in the student's language.

Detection by script and statistics stays, as evidence the model and the
no-model path can use, and for ingested documents where there is more text.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Literal, Optional

from .detect import (
    DEFAULT_MIN_CHARS,
    DEFAULT_MIN_CONFIDENCE,
    DEFAULT_MIN_MARGIN,
    DOCUMENT_MIN_CHARS,
    detect,
    is_trustworthy,
)
from .script import MIN_MIXED_SHARE, classify
from .types import LanguageDecision

logger = logging.getLogger(__name__)

DEFAULT_SUPPORTED: tuple[str, ...] = ("ar", "en", "fr", "de")
DEFAULT_LANGUAGE = "en"
DEFAULT_REPLY_FOLLOWS = "subject"

ReplyFollows = Literal["subject", "message"]


class LanguagePolicy:
    """Decides the language of a turn, or of a document, from its configuration."""

    def __init__(
        self,
        *,
        supported: Sequence[str] = DEFAULT_SUPPORTED,
        default: str = DEFAULT_LANGUAGE,
        arabic_ratio: float = MIN_MIXED_SHARE,
        min_chars: int = DEFAULT_MIN_CHARS,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
        min_margin: float = DEFAULT_MIN_MARGIN,
        reply_follows: ReplyFollows = DEFAULT_REPLY_FOLLOWS,
    ) -> None:
        self.supported = tuple(code for code in supported if code)
        self.default = default if default in self.supported else (
            self.supported[0] if self.supported else DEFAULT_LANGUAGE
        )
        self.arabic_ratio = arabic_ratio
        self.min_chars = min_chars
        self.min_confidence = min_confidence
        self.min_margin = min_margin
        self.reply_follows = reply_follows

    @classmethod
    def from_config(cls, config) -> LanguagePolicy:
        get = config.get
        return cls(
            supported=tuple(get("language.supported", DEFAULT_SUPPORTED)),
            default=get("language.default", DEFAULT_LANGUAGE),
            arabic_ratio=get("language.arabic_ratio", MIN_MIXED_SHARE),
            min_chars=get("language.min_chars", DEFAULT_MIN_CHARS),
            min_confidence=get("language.min_confidence", DEFAULT_MIN_CONFIDENCE),
            min_margin=get("language.min_margin", DEFAULT_MIN_MARGIN),
            reply_follows=get("language.reply_follows", DEFAULT_REPLY_FOLLOWS),
        )

    def supports(self, code: Optional[str]) -> bool:
        return bool(code) and code in self.supported

    def written_in(self, message: str) -> Optional[str]:
        """The language ``message`` is written in, when the text can tell."""
        text = message or ""
        writing_system = classify(text, arabic_threshold=self.arabic_ratio)
        if writing_system == "arabic" and self.supports("ar"):
            return "ar"
        if writing_system != "latin":
            return None
        detection = detect(text, self.supported)
        if detection and self.supports(detection.code) and is_trustworthy(
            detection, text, min_chars=self.min_chars,
            min_confidence=self.min_confidence, min_margin=self.min_margin,
        ):
            return detection.code
        return None

    def reply(
        self,
        *,
        requested: Optional[str] = None,
        subject_language: Optional[str] = None,
        message_language: Optional[str] = None,
        api_language: Optional[str] = None,
        subject_chosen: bool = True,
    ) -> LanguageDecision:
        """The language this turn is answered in, and the fact that decided it.

        ``subject_chosen`` is whether the student is in that subject or the
        system guessed it. A subject opened in the app teaches in its own
        language — an English literature lesson answers in English however the
        student types. A guessed one does not get to overrule them: the same
        term now exists in Arabic and in French, so one English question can
        settle the conversation on the French book and answer a student writing
        Arabic in French, which is what it did.
        """
        ranked = [("session", requested)]
        if self.reply_follows == "subject" and subject_chosen:
            ranked += [("subject", subject_language), ("message", message_language)]
        else:
            ranked += [("message", message_language), ("subject", subject_language)]
        ranked.append(("api", api_language))
        for source, code in ranked:
            if self.supports(code):
                return self._decide(code, source)
        return self._decide(self.default, "default", refused=True)

    def resolve_document(
        self, text: str, *, hint: Optional[str] = None
    ) -> LanguageDecision:
        """The language of ingested material, where more text is available."""
        writing_system = classify(text or "", arabic_threshold=self.arabic_ratio)
        if writing_system == "arabic" and self.supports("ar"):
            return self._decide("ar", "script", script=writing_system)

        detection = detect(text or "", self.supported)
        if detection and self.supports(detection.code) and is_trustworthy(
            detection,
            text or "",
            min_chars=DOCUMENT_MIN_CHARS,
            min_confidence=self.min_confidence,
            min_margin=self.min_margin,
        ):
            return self._decide(
                detection.code,
                "detected",
                script=writing_system,
                confidence=detection.confidence,
                detector=detection.detector,
            )

        if self.supports(hint):
            return self._decide(hint, "profile", script=writing_system, refused=True)
        return self._decide(self.default, "default", script=writing_system, refused=True)

    def _decide(
        self,
        code: str,
        source: str,
        *,
        script: str = "none",
        confidence: float = 1.0,
        detector: Optional[str] = None,
        refused: bool = False,
        evidence: Optional[str] = None,
    ) -> LanguageDecision:
        decision = LanguageDecision(
            code=code,
            source=source,
            confidence=confidence,
            script=script,
            detector=detector,
            refused=refused,
            evidence=evidence,
        )
        logger.info("language resolved", extra=decision.log_fields())
        return decision
