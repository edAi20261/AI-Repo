"""The instruction that pins a model's output language.

Inferring the language from the conversation is unreliable once retrieved
material is in the prompt: an Arabic textbook passage pulls an English answer
into Arabic and the reverse. The target is therefore named outright, in English
and in its own script, and the pull from the source material is called out.
"""

from __future__ import annotations

from .types import ENDONYMS, ENGLISH_NAMES

KEEP_TERMS = (
    "Keep technical terms, formulas, numbers and proper nouns in their source form."
)
CONTEXT_PULL = (
    "The source material below may be in another language; that must not change "
    "the language you reply in."
)


def name_of(code: str) -> str:
    """``ar`` -> ``Arabic (العربية)`` — how a language is named to a model."""
    english = ENGLISH_NAMES.get(code, code)
    endonym = ENDONYMS.get(code)
    return f"{english} ({endonym})" if endonym and endonym != english else english


def language_directive(
    code: str, *, keep_terms: bool = True, context_may_differ: bool = True
) -> str:
    """The reply-language instruction appended to every user-facing prompt."""
    target = name_of(code)
    lines = [
        f"The student is writing in {target}. You MUST write your ENTIRE reply in "
        f"{target} — every sentence, heading, label and caption.",
    ]
    if context_may_differ:
        lines.append(CONTEXT_PULL)
    if keep_terms:
        lines.append(KEEP_TERMS)
    return " ".join(lines)


def content_directive(code: str, *, keep_terms: bool = True) -> str:
    """The language instruction for generated material rather than conversation."""
    target = name_of(code)
    line = f"Write the content in {target}."
    return f"{line} {KEEP_TERMS}" if keep_terms else line
