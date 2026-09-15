"""Statistical language detection that is allowed to say "I don't know".

A short question carries too little signal to classify: "explain column
subtraction" reads as French to a trigram model. Detection therefore returns a
confidence and a margin over the runner-up, and the caller declines whenever
either is thin — a wrong guess sends a child an answer in a language they cannot
read, which is worse than falling back to what we already knew about them.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_MIN_CHARS = 12
DEFAULT_MIN_CONFIDENCE = 0.55
DEFAULT_MIN_MARGIN = 0.15
DOCUMENT_MIN_CHARS = 200

_ISO_BY_NAME = {"ARABIC": "ar", "ENGLISH": "en", "FRENCH": "fr", "GERMAN": "de"}

# Languages we do not answer in, but must be able to recognise. A detector built
# only from the languages we support has no way to say "this is Italian" — it has
# to pick the nearest of the four it knows, and does so confidently: Turkish
# scores 1.00 for German, Dutch 0.82. Naming them lets the resolver decline and
# fall back, instead of answering a Turkish child in German.
_ALSO_RECOGNISED = {
    "SPANISH": "es", "ITALIAN": "it", "PORTUGUESE": "pt", "DUTCH": "nl",
    "TURKISH": "tr", "RUSSIAN": "ru", "PERSIAN": "fa", "URDU": "ur",
    "HINDI": "hi", "CHINESE": "zh", "INDONESIAN": "id", "ROMANIAN": "ro",
    "POLISH": "pl", "GREEK": "el", "HEBREW": "he",
}

_ALL_NAMES = {**_ISO_BY_NAME, **_ALSO_RECOGNISED}


@dataclass(frozen=True)
class Detection:
    """A detector's best guess with the evidence needed to trust or reject it."""

    code: str
    confidence: float
    margin: float
    detector: str


@lru_cache(maxsize=4)
def _detector(languages: tuple[str, ...]):
    """A detector over the languages we answer in, plus the ones we only need to
    recognise. Costs about 5 MB and a fifth of a millisecond per call."""
    from lingua import Language, LanguageDetectorBuilder

    by_code = {code: name for name, code in _ISO_BY_NAME.items()}
    names = [by_code[code] for code in languages if code in by_code]
    if len(names) < 2:
        names = ["ARABIC", "ENGLISH"]
    names += list(_ALSO_RECOGNISED)
    selected = [getattr(Language, name) for name in names if hasattr(Language, name)]
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
    # The code may be one we do not answer in; the caller checks that and
    # declines, rather than being handed the nearest language we do support.
    code = _ALL_NAMES.get(top.language.name, "")
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
