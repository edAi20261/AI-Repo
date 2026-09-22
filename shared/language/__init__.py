"""Language policy shared by the runtime agent and the ingestion pipeline.

One resolver decides a turn's language once, from the strongest evidence
available, and records why. Everything downstream reads that decision.
"""

from .detect import Detection, detect, is_trustworthy
from .directive import content_directive, language_directive, name_of
from .resolver import DEFAULT_LANGUAGE, DEFAULT_SUPPORTED, LanguagePolicy
from .script import arabic_ratio, classify, looks_arabizi, written_in
from .types import (
    ENDONYMS,
    ENGLISH_NAMES,
    Lang,
    LanguageDecision,
    LanguageSource,
    Script,
)
from .verify import LanguageCheck, prose_of, verify_answer

__all__ = [
    "DEFAULT_LANGUAGE",
    "DEFAULT_SUPPORTED",
    "ENDONYMS",
    "ENGLISH_NAMES",
    "Detection",
    "Lang",
    "LanguageCheck",
    "LanguageDecision",
    "LanguagePolicy",
    "LanguageSource",
    "Script",
    "arabic_ratio",
    "classify",
    "content_directive",
    "detect",
    "is_trustworthy",
    "language_directive",
    "looks_arabizi",
    "name_of",
    "prose_of",
    "verify_answer",
    "written_in",
]
