"""Layout-aware parsing: a page becomes blocks, and blocks become chunks."""

from .blocks import Block, BlockType, PageLayout, PageMeta, ParsedPage, ReadingDirection
from .chunker import PageChunk, page_to_chunks
from .reading_order import sort_blocks
from .router import LAYOUT_MODEL, TEXT_LAYER, VISION, choose_tier
from .tier1 import parse_page

__all__ = [
    "LAYOUT_MODEL",
    "TEXT_LAYER",
    "VISION",
    "Block",
    "BlockType",
    "PageLayout",
    "PageMeta",
    "ParsedPage",
    "PageChunk",
    "ReadingDirection",
    "choose_tier",
    "page_to_chunks",
    "parse_page",
    "sort_blocks",
]
