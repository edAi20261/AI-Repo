"""Choosing how to read a page, one page at a time.

A single PDF can hold all three kinds: a text layer the publisher embedded, a
page whose text sits in scattered boxes that only a layout model orders
correctly, and a scan with no text at all. Deciding per document would send
clean pages to the expensive tier and hard pages to the cheap one.

The cheap tier is free and instant, so the question is only ever whether it can
be trusted with this page.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

logger = logging.getLogger(__name__)

TEXT_LAYER = 1
LAYOUT_MODEL = 2
VISION = 3

# Below this many characters a page has no usable text layer, whatever the
# extractor reports. The same threshold the pipeline already uses to decide a
# page is scanned.
MIN_CHARS = 40
# A page of prose has a handful of wide blocks. Many narrow ones scattered over
# it is a magazine spread, and their drawing order is not their reading order.
SIMPLE_MAX_BLOCKS = 12
SIMPLE_MIN_WIDTH_FRACTION = 0.40
SCATTERED_MIN_BLOCKS = 6
SCATTERED_FIGURE_OVERLAP = 0.40


def choose_tier(page: Any, *, char_count: int, blocks: Sequence[Any]) -> int:
    """Which parser this page needs.

    ``blocks`` are the already-parsed text-layer blocks, so the decision costs
    nothing beyond work the caller has done anyway.

    Page furniture is excluded from every signal. An illustrated textbook sets
    its image credit vertically up the side of the photograph, so every page in
    the book carries a rotated block — and reading that as evidence of scattered
    text sent all forty pages of a slice to the vision tier, which is both the
    slowest answer and the wrong one.
    """
    blocks = [b for b in blocks if not b.is_furniture]
    if char_count < MIN_CHARS or not blocks:
        return VISION
    if _is_simple(page, blocks):
        return TEXT_LAYER
    if _is_scattered(blocks):
        return VISION
    return LAYOUT_MODEL


def _is_simple(page: Any, blocks: Sequence[Any]) -> bool:
    """One or two clean columns of wide blocks, upright, none over a picture.

    Rotation disqualifies a page here rather than only in ``_is_scattered``:
    a self-check key printed upside down at the foot of an otherwise ordinary
    page is still text the cheap tier reads in the wrong order, and it is
    precisely the text that must never be embedded with the lesson.
    """
    if len(blocks) > SIMPLE_MAX_BLOCKS:
        return False
    if any(b.overlaps_figure or b.rotation for b in blocks):
        return False
    width = float(getattr(page, "rect", None).width) if getattr(page, "rect", None) else 0.0
    if not width:
        return False
    widths = sorted(b.bbox[2] - b.bbox[0] for b in blocks)
    median_width = widths[len(widths) // 2]
    return median_width / width > SIMPLE_MIN_WIDTH_FRACTION


def _is_scattered(blocks: Sequence[Any]) -> bool:
    """Text that only a model can put in order.

    Any rotated block settles it: text printed sideways or upside down is a
    credit, a callout or a self-check answer key, and its position on the page
    says nothing about where it belongs in the reading.
    """
    if any(b.rotation for b in blocks):
        return True
    if len(blocks) <= SCATTERED_MIN_BLOCKS:
        return False
    over_figures = sum(1 for b in blocks if b.overlaps_figure) / len(blocks)
    return over_figures > SCATTERED_FIGURE_OVERLAP
