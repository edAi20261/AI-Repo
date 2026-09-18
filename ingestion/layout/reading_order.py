"""Putting a page's blocks into the order a human reads them.

Sorting by ``y`` alone fails on anything with columns; sorting by ``x`` alone
fails on everything. The rule that works is to band the page into rows, order
the rows down the page, and order within a row along the direction the script
runs — left to right for Latin, right to left for Arabic.

The band is what stops a box three points higher than its neighbour sorting
ahead of it and breaking the line.
"""

from __future__ import annotations

from statistics import median
from typing import Iterable, Sequence

from .blocks import Block, ReadingDirection

# A row is a band of the page height. Too tall and two columns merge into one
# row; too short and a single line of large type splits across bands. A twelfth
# of the page is about two lines of body text on a school textbook page.
BAND_DIVISOR = 12
MIN_BAND_POINTS = 24.0


def band_height(page_height: float, blocks: Sequence[Block] = ()) -> float:
    """How tall a row is on this page.

    Derived from the blocks when there are any — a page set in large type needs
    taller bands than one set in small — and from the page otherwise.
    """
    heights = [b.bbox[3] - b.bbox[1] for b in blocks if b.bbox[3] > b.bbox[1]]
    if heights:
        return max(MIN_BAND_POINTS, median(heights))
    return max(MIN_BAND_POINTS, page_height / BAND_DIVISOR)


def assign_rows(blocks: Sequence[Block], band: float) -> dict[int, int]:
    """Which row of the page each block sits in.

    Blocks are clustered by how far apart their tops are, not by rounding the
    top to a fixed grid. Rounding puts a boundary somewhere on every page, and
    two boxes three points apart that straddle it end up in different rows —
    which is exactly the case the banding exists to prevent.
    """
    ordered = sorted(blocks, key=lambda b: b.bbox[1])
    rows: dict[int, int] = {}
    row = 0
    top = ordered[0].bbox[1] if ordered else 0.0
    for block in ordered:
        if block.bbox[1] - top > band:
            row += 1
            top = block.bbox[1]
        rows[id(block)] = row
    return rows


def sort_blocks(
    blocks: Iterable[Block],
    *,
    page_height: float,
    direction: ReadingDirection = "ltr",
) -> list[Block]:
    """The blocks in reading order, each stamped with its ``reading_order``."""
    blocks = list(blocks)
    if not blocks:
        return []

    rows = assign_rows(blocks, band_height(page_height, blocks))

    def key(block: Block) -> tuple[int, float]:
        x0, _, x1, _ = block.bbox
        # Arabic reads from the right, so the rightmost edge comes first.
        return (rows[id(block)], -x1 if direction == "rtl" else x0)

    ordered = sorted(blocks, key=key)
    for position, block in enumerate(ordered, start=1):
        block.reading_order = position
    return ordered
