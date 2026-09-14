"""Read a lesson's stored body text into headings and paragraphs.

``content.lessons.text_content`` is authored in the platform's editor, so it
arrives as HTML or as Markdown. Both are reduced to the same ordered blocks the
extract stage turns into elements, keeping heading depth so chunking can still
break on sections.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser

BLOCK_TAGS = frozenset(
    {"p", "div", "section", "article", "li", "tr", "blockquote", "pre", "figcaption"}
)
HEADING_TAGS = {f"h{level}": level for level in range(1, 7)}
SKIP_TAGS = frozenset({"script", "style", "head"})
MARKDOWN_HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
HTML_TAG = re.compile(r"<[a-zA-Z!/][^>]*>")


@dataclass(frozen=True)
class TextBlock:
    """One heading or paragraph of a lesson body."""

    text: str
    level: int = 0

    @property
    def is_heading(self) -> bool:
        return self.level > 0


def to_blocks(text: str) -> list[TextBlock]:
    """Ordered blocks of a lesson body, whichever format it was authored in."""
    if not text or not text.strip():
        return []
    return _from_html(text) if HTML_TAG.search(text) else _from_markdown(text)


class _HtmlReader(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[TextBlock] = []
        self._buffer: list[str] = []
        self._level = 0
        self._skipping = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in SKIP_TAGS:
            self._skipping += 1
            return
        if tag == "br":
            self._flush()
            return
        if tag in HEADING_TAGS or tag in BLOCK_TAGS:
            self._flush()
            self._level = HEADING_TAGS.get(tag, 0)

    def handle_endtag(self, tag: str) -> None:
        if tag in SKIP_TAGS:
            self._skipping = max(0, self._skipping - 1)
            return
        if tag in HEADING_TAGS or tag in BLOCK_TAGS:
            self._flush()

    def handle_data(self, data: str) -> None:
        if not self._skipping:
            self._buffer.append(data)

    def close(self) -> None:
        super().close()
        self._flush()

    def _flush(self) -> None:
        text = _collapse("".join(self._buffer))
        self._buffer.clear()
        if text:
            self.blocks.append(TextBlock(text=text, level=self._level))
        self._level = 0


def _from_html(text: str) -> list[TextBlock]:
    reader = _HtmlReader()
    reader.feed(text)
    reader.close()
    return reader.blocks


def _from_markdown(text: str) -> list[TextBlock]:
    blocks: list[TextBlock] = []
    paragraph: list[str] = []

    def flush() -> None:
        joined = _collapse(" ".join(paragraph))
        paragraph.clear()
        if joined:
            blocks.append(TextBlock(text=joined))

    for line in unescape(text).splitlines():
        heading = MARKDOWN_HEADING.match(line.strip())
        if heading:
            flush()
            blocks.append(TextBlock(text=_collapse(heading.group(2)), level=len(heading.group(1))))
        elif line.strip():
            paragraph.append(line.strip())
        else:
            flush()
    flush()
    return blocks


def _collapse(text: str) -> str:
    return re.sub(r"\s+", " ", unescape(text)).strip()
