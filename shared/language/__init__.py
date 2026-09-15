"""Language policy shared by the runtime agent and the ingestion pipeline.

One resolver decides a turn's language once, from the strongest evidence
available, and records why. Everything downstream reads that decision.
"""

from .catalog import translate
from .detect import Detection, detect, is_trustworthy
from .directive import content_directive, language_directive, name_of
from .request import explicit_request
from .resolver import DEFAULT_LANGUAGE, DEFAULT_SUPPORTED, LanguagePolicy
from .script import arabic_ratio, classify, looks_arabizi
from .types import ENDONYMS, ENGLISH_NAMES, Lang, LanguageDecision, LanguageSource, Script
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
    "explicit_request",
    "is_trustworthy",
    "language_directive",
    "looks_arabizi",
    "name_of",
    "prose_of",
    "translate",
    "verify_answer",
]
