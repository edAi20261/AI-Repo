"""Reading a page from the PDF's own text layer.

Free and instant, and correct whenever the publisher embedded real text. It
cannot see a photograph or tell a caption from a paragraph by meaning, so the
type of each block is inferred from how it is set: size and weight against the
page's own median, and position against the page's own margins.

Relative to the page, not to absolute point sizes — a textbook, a worksheet and
a slide deck disagree about what 14pt means.
"""

from __future__ import annotations

from statistics import median
from typing import Any

from .blocks import Block, PageMeta, ParsedPage, ReadingDirection
from .reading_order import sort_blocks

HEADING_RATIO = 1.4
SUBHEADING_RATIO = 1.15
MARGIN_FRACTION = 0.05
# A folio is a handful of characters; a running head is a short line. Longer
# than this near an edge and it is content that happens to sit there.
FURNITURE_MAX_WORDS = 12
# A copyright or phonogram mark means the line is a rights notice wherever it
# sits on the page. The characters mean that by definition, so this is not a
# guess about one publisher's house style.
RIGHTS_MARKS = ("\u00a9", "\u2117")
# An image this large is the page's background, not a figure on it. Publishers
# of illustrated textbooks flatten the whole spread into one image, and without
# this every block on every page reports that it sits over a figure — which
# makes the signal useless for telling a caption from a paragraph.
#
# Measured over 75 sampled pages of a Grade 6 science module: image rectangles
# are sharply bimodal. 632 of them cover under 10% of the page, ten fall between
# 10% and 80%, and 72 cover 80% or more — of which 68 cover 90% or more. The
# gap is wide enough that the exact cut barely matters; 0.85 sits inside it.
BACKGROUND_FRACTION = 0.85


def parse_page(
    page: Any, meta: PageMeta, *, direction: ReadingDirection = "ltr"
) -> ParsedPage:
    """The page's text layer as ordered, typed blocks.

    Grouped by line rather than by the extractor's own blocks. PyMuPDF merges a
    heading with the paragraph that follows it whenever they share a text block,
    and a paragraph whose first line is set larger is not a heading — it is a
    heading and then a paragraph, which the chunker has to be able to tell apart.
    """
    raw = page.get_text("dict")
    figures = figure_rects(page, meta.area)

    spans = [
        (block, line, span)
        for block in raw.get("blocks", [])
        if block.get("type") == 0
        for line in block.get("lines", [])
        for span in line.get("spans", [])
    ]
    typical = body_size(spans)

    blocks: list[Block] = []
    for raw_block in raw.get("blocks", []):
        if raw_block.get("type") != 0:
            continue
        blocks.extend(_blocks_from_lines(raw_block, typical, meta, figures))

    ordered = sort_blocks(_deduplicate(blocks), page_height=meta.height,
                          direction=direction)
    return ParsedPage(meta=meta, blocks=ordered)


def _deduplicate(blocks: list[Block]) -> list[Block]:
    """Drop blocks that repeat the same text at the same place.

    Illustrated PDFs stack overlapping copies of a glyph to fake a typographic
    effect. One page of a Grade 6 science module emits the letter "i" 528 times
    at identical coordinates, which on its own accounted for most of the
    headings found across the whole book.
    """
    seen: set[tuple[str, tuple[int, int, int, int]]] = set()
    kept = []
    for block in blocks:
        fingerprint = (block.text, tuple(round(v) for v in block.bbox))
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        kept.append(block)
    return kept


def _blocks_from_lines(
    raw_block: dict,
    typical: float,
    meta: PageMeta,
    figures: list[tuple[float, float, float, float]],
) -> list[Block]:
    """One block per run of consecutive lines that are the same kind of text."""
    runs: list[tuple[str, list[dict]]] = []
    for line in raw_block.get("lines", []):
        text = "".join(span.get("text", "") for span in line.get("spans", []))
        if not text.strip():
            continue
        kind = _line_type(line, typical, meta, text)
        if runs and runs[-1][0] == kind:
            runs[-1][1].append(line)
        else:
            runs.append((kind, [line]))

    rotation = _rotation(raw_block)
    out = []
    for kind, lines in runs:
        text = " ".join(
            "".join(span.get("text", "") for span in line.get("spans", []))
            for line in lines
        )
        bbox = _union(line["bbox"] for line in lines)
        out.append(Block(
            text=text,
            type=kind,
            bbox=bbox,
            rotation=rotation,
            overlaps_figure=any(_overlaps(bbox, rect) for rect in figures),
            source_tier=1,
        ))
    return out


def _union(boxes) -> tuple[float, float, float, float]:
    boxes = [tuple(float(v) for v in b) for b in boxes]
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


def body_size(spans: list[tuple[Any, Any, dict]]) -> float:
    """The size the page's body text is set in.

    Weighted by how many characters are set at each size rather than by how many
    spans use it. A plain median over spans is decided by a page's headings and
    labels, which outnumber its paragraphs — on a page holding one 22pt title
    and one 10pt paragraph it returns 16, and then nothing is 1.4x it and the
    title is not a heading.
    """
    weighted: list[float] = []
    for _, _, span in spans:
        text = span.get("text", "").strip()
        if text:
            weighted.extend([span["size"]] * len(text))
    return median(weighted) if weighted else 0.0


def figure_rects(page: Any, page_area: float) -> list[tuple[float, float, float, float]]:
    """The images on the page that are figures rather than its background."""
    rects = []
    for image in page.get_images(full=True):
        for rect in page.get_image_rects(image[0]):
            if page_area and (rect.width * rect.height) / page_area >= BACKGROUND_FRACTION:
                continue
            rects.append((rect.x0, rect.y0, rect.x1, rect.y1))
    return rects


def _overlaps(a: tuple[float, ...], b: tuple[float, ...]) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _rotation(raw_block: dict) -> float:
    """Degrees off horizontal, from the writing direction of the first line.

    A credit set vertically up the side of a photograph is rotated text, and
    reading it as a paragraph puts a publisher's copyright notice in the middle
    of a lesson.
    """
    lines = raw_block.get("lines") or []
    if not lines:
        return 0.0
    dx, dy = lines[0].get("dir", (1.0, 0.0))
    if abs(dx) >= abs(dy):
        return 0.0 if dx >= 0 else 180.0
    return 90.0 if dy < 0 else 270.0


def _line_type(line: dict, typical: float, meta: PageMeta, text: str) -> str:
    spans = line.get("spans") or []
    if not spans or not typical:
        return "body"
    bbox = tuple(float(v) for v in line["bbox"])

    if any(mark in text for mark in RIGHTS_MARKS):
        return "boilerplate"

    # A lone glyph is a drop cap or a rendering artefact: it cannot introduce a
    # section, so it never takes the heading branch below. It can still be a
    # folio, which is why this is a flag rather than an early return.
    lone_glyph = len(text.strip()) < 2

    size = body_size([(None, None, span) for span in spans]) or 0.0
    bold = any("bold" in str(s.get("font", "")).lower() for s in spans)
    # Prominence is checked before position. A section heading set large at the
    # top of the page is a heading, not a running head — "Ask a Question" sits
    # inside the top margin of a Grade 6 science page and is the most important
    # block on it.
    if size >= typical * HEADING_RATIO and not lone_glyph:
        return "heading"

    words = len(text.split())
    margin = meta.height * MARGIN_FRACTION
    near_edge = bbox[1] < margin or bbox[3] > meta.height - margin
    if near_edge and words <= FURNITURE_MAX_WORDS:
        if text.strip().isdigit():
            return "page_number"
        return "header" if bbox[1] < margin else "footer"

    if lone_glyph:
        return "unknown"
    if size >= typical * SUBHEADING_RATIO or bold:
        return "subheading"
    return "body"
