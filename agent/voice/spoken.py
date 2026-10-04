"""A tutor reply with everything only an eye can use taken out.

Structure is syntax, the same in every language: Markdown markers, citation
markers, links, emoji, and the sources and prerequisites blocks. Notation is
left in place; saying it is language, and that is the speech-script model's job.
"""

from __future__ import annotations

import re
import unicodedata

from ..citations import MARKER, RUN

_FENCE = re.compile(r"```.*?```", re.DOTALL)
_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_BARE_URL = re.compile(r"(?:https?://|s3://|www\.)\S+", re.IGNORECASE)
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s*", re.MULTILINE)
_QUOTE = re.compile(r"^\s*>\s?", re.MULTILINE)
_BULLET = re.compile(r"^\s*[-*+•]\s+", re.MULTILINE)
_RULE = re.compile(r"^\s*(?:[-*_]\s*){3,}$", re.MULTILINE)
_TABLE_DIVIDER = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(?:\|\s*:?-{2,}:?\s*)*\|?\s*$", re.MULTILINE)
_EMPHASIS = re.compile(r"(\*\*|__|~~|`)(?=\S)(.+?)(?<=\S)\1")
_NOTATION = re.compile(r"\$|\\[A-Za-z(\[]|\^|(?<![\w)\]])[-−]\d")
_TRAILER = ("📚", "🧩")
_SPACES = re.compile(r"[ \t]+")
_SPACE_BEFORE_PUNCTUATION = re.compile(r"\s+([.,;:!?؟،])")
_BLANK_LINES = re.compile(r"\n{2,}")


def readable(text: str) -> str:
    """``text`` without the parts of a reply that are only for the eye."""
    text = _without_trailers(_FENCE.sub(" ", text or ""))
    text = RUN.sub(" ", text)
    text = MARKER.sub(" ", text)
    text = _IMAGE.sub(" ", text)
    text = _LINK.sub(lambda m: m.group(1), text)
    text = _BARE_URL.sub(" ", text)
    text = _TABLE_DIVIDER.sub("", text)
    text = "\n".join(_table_row(line) for line in text.splitlines())
    text = _RULE.sub("", text)
    text = _HEADING.sub("", text)
    text = _QUOTE.sub("", text)
    text = _BULLET.sub("", text)
    for _ in range(3):
        text = _EMPHASIS.sub(lambda m: m.group(2), text)
    text = _without_pictographs(text)
    lines = [_SPACE_BEFORE_PUNCTUATION.sub(r"\1", _SPACES.sub(" ", line)).strip()
             for line in text.splitlines()]
    return _BLANK_LINES.sub("\n", "\n".join(lines)).strip()


def has_notation(text: str) -> bool:
    """Whether ``text`` holds anything a voice would read as symbols: LaTeX, or a
    mathematical symbol by its Unicode category, in any language."""
    return bool(_NOTATION.search(text)) or any(
        unicodedata.category(ch) == "Sm" for ch in text)


def _without_trailers(text: str) -> str:
    kept, skipping = [], False
    for line in text.splitlines():
        if line.lstrip().startswith(_TRAILER):
            skipping = True
        elif skipping and not line.strip():
            skipping = False
        if not skipping:
            kept.append(line)
    return "\n".join(kept)


def _table_row(line: str) -> str:
    stripped = line.strip()
    if stripped.startswith("|") and stripped.endswith("|") and stripped.count("|") >= 2:
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        return ", ".join(cell for cell in cells if cell)
    return line


def _without_pictographs(text: str) -> str:
    return "".join(" " if unicodedata.category(ch) == "So" else ch
                   for ch in text if ch not in "️‍")
