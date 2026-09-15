"""The one place a turn's language is decided.

The rule every assistant people already use converges on: answer in the language
of the message, unless the message itself asks for another one. Anthropic states
it as "follow the language of the query unless explicitly requested otherwise" —
and applies it even over a stored "my native language is Spanish". OpenAI's voice
prompt says "respond in the same language the user is speaking unless directed
otherwise".

So precedence runs from what this message says, to what it is written in, to what
we were told about the learner beforehand:

    explicit request in the message
    -> Arabic script
    -> confident detection
    -> the caller's parameter
    -> planner hint -> profile -> previous turn -> default

A caller's ``language`` parameter is a default for when the message cannot decide
— a greeting too short to detect, Arabizi, a bare equation — never an override of
what the student actually typed. A stored preference is the weakest evidence
there is about what someone wants right now.

Every node downstream reads the result; none of them decide again.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Optional

from .detect import (
    DEFAULT_MIN_CHARS,
    DEFAULT_MIN_CONFIDENCE,
    DEFAULT_MIN_MARGIN,
    DOCUMENT_MIN_CHARS,
    detect,
    is_trustworthy,
)
from .request import explicit_request
from .script import MIN_MIXED_SHARE, classify
from .types import LanguageDecision

logger = logging.getLogger(__name__)

DEFAULT_SUPPORTED: tuple[str, ...] = ("ar", "en", "fr", "de")
DEFAULT_LANGUAGE = "en"


class LanguagePolicy:
    """Resolves the language of a turn, or of a document, from its configuration."""

    def __init__(
        self,
        *,
        supported: Sequence[str] = DEFAULT_SUPPORTED,
        default: str = DEFAULT_LANGUAGE,
        arabic_ratio: float = MIN_MIXED_SHARE,
        min_chars: int = DEFAULT_MIN_CHARS,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
        min_margin: float = DEFAULT_MIN_MARGIN,
        arabizi_hint: Optional[str] = None,
    ) -> None:
        self.supported = tuple(code for code in supported if code)
        self.default = default if default in self.supported else (
            self.supported[0] if self.supported else DEFAULT_LANGUAGE
        )
        self.arabic_ratio = arabic_ratio
        self.min_chars = min_chars
        self.min_confidence = min_confidence
        self.min_margin = min_margin
        self.arabizi_hint = arabizi_hint if arabizi_hint in self.supported else None

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
            arabizi_hint=get("language.arabizi.hint_to", None),
        )

    def resolve(
        self,
        message: str = "",
        *,
        api_language: Optional[str] = None,
        planner_hint: Optional[str] = None,
        profile_language: Optional[str] = None,
        previous_language: Optional[str] = None,
    ) -> LanguageDecision:
        """The language this turn is answered in, and the rule that decided it."""
        text = message or ""
        writing_system = classify(text, arabic_threshold=self.arabic_ratio)

        requested = explicit_request(text)
        if requested and self._supported(requested[0]):
            return self._decide(
                requested[0], "explicit", script=writing_system, evidence=requested[1]
            )

        if writing_system == "arabic" and self._supported("ar"):
            return self._decide("ar", "script", script=writing_system)

        if writing_system in ("arabizi", "mixed"):
            return self._fallback(
                writing_system, api_language, planner_hint, profile_language,
                previous_language,
            )

        detection = detect(text, self.supported)
        trustworthy = is_trustworthy(
            detection,
            text,
            min_chars=self.min_chars,
            min_confidence=self.min_confidence,
            min_margin=self.min_margin,
        )
        if detection and trustworthy and self._supported(detection.code):
            return self._decide(
                detection.code,
                "detected",
                script=writing_system,
                confidence=detection.confidence,
                detector=detection.detector,
            )

        return self._fallback(
            writing_system, api_language, planner_hint, profile_language,
            previous_language, detection,
        )

    def resolve_document(
        self, text: str, *, hint: Optional[str] = None
    ) -> LanguageDecision:
        """The language of ingested material, where more text is available."""
        writing_system = classify(text or "", arabic_threshold=self.arabic_ratio)
        if writing_system == "arabic" and self._supported("ar"):
            return self._decide("ar", "script", script=writing_system)

        detection = detect(text or "", self.supported)
        if detection and is_trustworthy(
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

        if self._supported(hint):
            return self._decide(hint, "profile", script=writing_system, refused=True)
        return self._decide(self.default, "default", script=writing_system, refused=True)

    def _fallback(
        self,
        writing_system: str,
        api_language: Optional[str],
        planner_hint: Optional[str],
        profile_language: Optional[str],
        previous_language: Optional[str],
        detection=None,
    ) -> LanguageDecision:
        """The message could not decide: use what we were told, and say we guessed."""
        if writing_system == "arabizi" and self.arabizi_hint:
            return self._decide(
                self.arabizi_hint, "script", script=writing_system, refused=True
            )
        # The caller's parameter is an instruction, not a guess, so the answer is
        # still checked against it — unlike everything below, which is memory.
        if self._supported(api_language):
            return self._decide(api_language, "api", script=writing_system)
        for code, source in (
            (planner_hint, "planner"),
            (profile_language, "profile"),
            (previous_language, "previous"),
        ):
            if self._supported(code):
                return self._decide(code, source, script=writing_system, refused=True)

        confidence = detection.confidence if detection else 0.0
        return self._decide(
            self.default, "default", script=writing_system, confidence=confidence, refused=True
        )

    def _supported(self, code: Optional[str]) -> bool:
        return bool(code) and code in self.supported

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
