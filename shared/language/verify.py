"""Checking that the answer came back in the language we asked for.

A directive in the prompt is an instruction, not a guarantee — retrieved material
in another language pulls the model across, and the student is the one who finds
out. Detection on the finished answer costs microseconds, so the guarantee is
cheap to actually hold rather than hope for.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional, Sequence

from .detect import detect
from .types import LanguageDecision

logger = logging.getLogger(__name__)

DEFAULT_MIN_ANSWER_CHARS = 40
DEFAULT_MIN_CONFIDENCE = 0.60

_CITATION = re.compile(r"\[\d+(?:,\s*\d+)*\]")
_FENCED = re.compile(r"```.*?```", re.S)
_INLINE_CODE = re.compile(r"`[^`]*`")
_LATEX = re.compile(r"\$[^$]*\$|\\\(.*?\\\)|\\\[.*?\\]", re.S)
_URL = re.compile(r"https?://\S+|s3://\S+")


@dataclass(frozen=True)
class LanguageCheck:
    """The outcome of verifying one answer against its intended language."""

    matches: bool
    detected: Optional[str] = None
    confidence: float = 0.0
    checked: bool = False

    @property
    def needs_repair(self) -> bool:
        return self.checked and not self.matches


def prose_of(answer: str) -> str:
    """The answer with the parts that carry no language signal removed."""
    text = _FENCED.sub(" ", answer or "")
    text = _LATEX.sub(" ", text)
    text = _INLINE_CODE.sub(" ", text)
    text = _URL.sub(" ", text)
    text = _CITATION.sub(" ", text)
    return text.strip()


def verify_answer(
    answer: str,
    decision: LanguageDecision,
    *,
    languages: Sequence[str],
    min_chars: int = DEFAULT_MIN_ANSWER_CHARS,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
) -> LanguageCheck:
    """Whether ``answer`` is written in the language ``decision`` asked for.

    Only a confident reading of a decision we actually trusted can fail: when the
    language was never known (Arabizi, mixed script, a two-word question), holding
    the answer to a guess would rewrite correct replies into the wrong language.
    """
    if not decision.verifiable:
        return LanguageCheck(matches=True)

    prose = prose_of(answer)
    if sum(1 for char in prose if char.isalpha()) < min_chars:
        return LanguageCheck(matches=True)

    detection = detect(prose, languages)
    if detection is None or detection.confidence < min_confidence:
        return LanguageCheck(matches=True, detected=detection.code if detection else None,
                             confidence=detection.confidence if detection else 0.0)

    matches = detection.code == decision.code
    if not matches:
        logger.warning(
            "answer language mismatch",
            extra={"expected": decision.code, "detected": detection.code,
                   "confidence": round(detection.confidence, 3),
                   "language_source": decision.source},
        )
    return LanguageCheck(
        matches=matches,
        detected=detection.code,
        confidence=detection.confidence,
        checked=True,
    )
