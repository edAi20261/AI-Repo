"""Unicode normalisation for extracted text.

The Arabic text layer stores presentation-form glyphs, not logical characters,
so a typed Arabic word matches nothing without NFKC. Runs before chunking (the
stored chunk text must match too). Also folds Latin ligatures, so it runs on
every language.
"""

from __future__ import annotations

import re

import unicodedata

_TATWEEL = "ـ"                        # Arabic kashida; decorative, no meaning
_MULTISPACE = re.compile(r"[ \t ]+")
_MULTINEWLINE = re.compile(r"\n{3,}")
_MARK_RUN = 5
_REPEATED_MARK = re.compile(rf"([^\w\s]|_)\1{{{_MARK_RUN},}}")
# A text layer can carry NUL and other C0 controls; Postgres refuses NUL in a
# text column, and a Geometry textbook's lessons failed to store on 19 Oct 2026.
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def collapse_repeated_marks(text: str) -> str:
    if not text:
        return ""
    return _REPEATED_MARK.sub(lambda m: m.group(1) * _MARK_RUN, _CONTROL.sub(" ", text))


def normalise_text(
    text: str,
    *,
    nfkc: bool = True,
    strip_tatweel: bool = True,
    collapse_whitespace: bool = True,
) -> str:
    """Normalise extracted text. Newlines are preserved (they carry structure the
    chunker relies on); only runs of spaces and 3+ blank lines are collapsed."""
    if not text:
        return ""

    if nfkc:
        text = unicodedata.normalize("NFKC", text)
    if strip_tatweel:
        text = text.replace(_TATWEEL, "")
    text = collapse_repeated_marks(text)
    if collapse_whitespace:
        text = _MULTISPACE.sub(" ", text)
        text = _MULTINEWLINE.sub("\n\n", text)
        text = "\n".join(line.strip() for line in text.split("\n"))
    return text.strip()
