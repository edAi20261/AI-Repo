"""Statistical language detection that is allowed to say "I don't know".

A short question carries too little signal to classify: "explain column
subtraction" reads as French to a trigram model. Detection therefore returns a
confidence and a margin over the runner-up, and the caller declines whenever
either is thin — a wrong guess sends a child an answer in a language they cannot
read, which is worse than falling back to what we already knew about them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Sequence

logger = logging.getLogger(__name__)

DEFAULT_MIN_CHARS = 12
DEFAULT_MIN_CONFIDENCE = 0.55
DEFAULT_MIN_MARGIN = 0.15
DOCUMENT_MIN_CHARS = 200

_ISO_BY_NAME = {"ARABIC": "ar", "ENGLISH": "en", "FRENCH": "fr", "GERMAN": "de"}


@dataclass(frozen=True)
class Detection:
    """A detector's best guess with the evidence needed to trust or reject it."""

    code: str
    confidence: float
    margin: float
    detector: str


@lru_cache(maxsize=4)
def _detector(languages: tuple[str, ...]):
    from lingua import Language, LanguageDetectorBuilder

    wanted = {code: name for name, code in _ISO_BY_NAME.items()}
    selected = [getattr(Language, wanted[code]) for code in languages if code in wanted]
    if len(selected) < 2:
        selected = [Language.ARABIC, Language.ENGLISH]
    return LanguageDetectorBuilder.from_languages(*selected).with_preloaded_language_models().build()


def detect(text: str, languages: Sequence[str]) -> Optional[Detection]:
    """Best guess for ``text`` among ``languages``, or ``None`` when unavailable."""
    stripped = (text or "").strip()
    if not stripped:
        return None

    try:
        values = _detector(tuple(languages)).compute_language_confidence_values(stripped)
    except ImportError:
        logger.warning("language detector unavailable; falling back to configured default")
        return None
    except Exception:
        logger.exception("language detection failed")
        return None

    if not values:
        return None

    top = values[0]
    runner_up = values[1].value if len(values) > 1 else 0.0
    code = _ISO_BY_NAME.get(top.language.name, "")
    if not code:
        return None
    return Detection(
        code=code,
        confidence=float(top.value),
        margin=float(top.value) - float(runner_up),
        detector="lingua",
    )


def is_trustworthy(
    detection: Optional[Detection],
    text: str,
    *,
    min_chars: int = DEFAULT_MIN_CHARS,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    min_margin: float = DEFAULT_MIN_MARGIN,
) -> bool:
    """Whether a detection carries enough evidence to decide a turn's language."""
    if detection is None:
        return False
    letters = sum(1 for char in text if char.isalpha())
    return (
        letters >= min_chars
        and detection.confidence >= min_confidence
        and detection.margin >= min_margin
    )
