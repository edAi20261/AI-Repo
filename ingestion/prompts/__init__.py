"""Ingestion prompt templates (kept out of stage code)."""

from .dependency_derivation import DERIVE_DEPS_PROMPT
from .figure_caption import FIGURE_CAPTION_PROMPT
from .math_transcription import MATH_PAGE_TRANSCRIPTION_PROMPT
from .page_transcription import PAGE_TRANSCRIPTION_PROMPT
from .topic_extraction import TOPIC_EXTRACTION_PROMPT
from .topic_relabel import TOPIC_RELABEL_PROMPT

__all__ = [
    "DERIVE_DEPS_PROMPT",
    "FIGURE_CAPTION_PROMPT",
    "MATH_PAGE_TRANSCRIPTION_PROMPT",
    "PAGE_TRANSCRIPTION_PROMPT",
    "TOPIC_EXTRACTION_PROMPT",
    "TOPIC_RELABEL_PROMPT",
]
