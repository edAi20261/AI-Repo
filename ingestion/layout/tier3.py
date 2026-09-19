"""Reading a page with a vision model, as blocks.

The only tier that can read the Arabic corpus, which has no text layer at all.
The pipeline already sends each of those pages to a model once; this asks that
same call for the page's structure rather than a wall of text, which costs
output tokens and not another request.

A model asked for structure can also be confidently wrong in ways a model asked
for text cannot — a plausible block in the wrong place reads as data. So what
comes back is checked against the page it came from before it is believed.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from pydantic import BaseModel, Field, field_validator

from shared.config import Config
from shared.structured import schema_hint
from shared.vision import read_image

from ..prompts import PAGE_LAYOUT_PROMPT
from .blocks import Block, BlockType, PageLayout, PageMeta, ParsedPage, ReadingDirection
from .reading_order import sort_blocks

logger = logging.getLogger(__name__)

# A page with this much ink on it is not blank, whatever came back.
INK_FLOOR = 0.03
MIN_CHARS_FOR_INKY_PAGE = 200
INKY = 0.05
REPEAT_NGRAM = 8
REPEAT_TIMES = 3

_ARABIC = r"؀-ۿ"
# A Latin letter welded to an Arabic one inside a word: the model has drifted
# out of the script it was reading.
_SCRIPT_DRIFT = re.compile(f"[{_ARABIC}][A-Za-z]|[A-Za-z][{_ARABIC}]")
_WORD = re.compile(r"\S+")


class RawBlock(BaseModel):
    """One block as the model reports it, in fractions of the page."""

    text: str = ""
    type: BlockType = "body"
    bbox: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)
    reading_order: int = 0
    rotation: float = 0.0
    belongs_to: Optional[int] = None

    @field_validator("bbox", mode="before")
    @classmethod
    def _four_numbers_or_the_whole_page(cls, v):
        try:
            x0, y0, x1, y1 = (float(n) for n in v)
        except (TypeError, ValueError):
            return (0.0, 0.0, 1.0, 1.0)
        return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


class LayoutTranscription(BaseModel):
    """A page as the model read it."""

    page_layout: PageLayout = "standard"
    blocks: list[RawBlock] = Field(default_factory=list)
    unreadable: bool = False

    @field_validator("blocks", mode="before")
    @classmethod
    def _drop_malformed(cls, v):
        """A block the model mangled costs that block, not the page."""
        if not isinstance(v, list):
            return []
        kept = []
        for raw in v:
            try:
                block = RawBlock.model_validate(raw)
            except Exception:
                continue
            if block.text.strip():
                kept.append(block)
        return kept


def parse_page(
    image_png: bytes,
    meta: PageMeta,
    *,
    config: Config,
    language_hint: Optional[str] = None,
    direction: ReadingDirection = "ltr",
) -> Optional[ParsedPage]:
    """The page image as ordered, typed blocks, or ``None`` if it could not be read."""
    result = read_image(
        image_png,
        PAGE_LAYOUT_PROMPT.render(
            language_hint=language_hint or "",
            schema=schema_hint(LayoutTranscription)),
        LayoutTranscription,
        config=config,
        label="page_layout",
    )
    if result is None or result.unreadable:
        return None

    blocks = [_to_block(raw, meta) for raw in result.blocks]
    ordered = sort_blocks(blocks, page_height=meta.height, direction=direction)
    return ParsedPage(
        meta=meta.model_copy(update={"layout": result.page_layout, "source_tier": 3}),
        blocks=ordered,
    )


def _to_block(raw: RawBlock, meta: PageMeta) -> Block:
    """One reported block, with its box converted back to page points."""
    x0, y0, x1, y1 = raw.bbox
    return Block(
        text=raw.text,
        type=raw.type,
        bbox=(x0 * meta.width, y0 * meta.height, x1 * meta.width, y1 * meta.height),
        rotation=raw.rotation,
        belongs_to=raw.belongs_to,
        source_tier=3,
    )


def problems(page: Optional[ParsedPage], ink: float) -> list[str]:
    """What is wrong with this transcription, judged against the page itself.

    There is no text layer to compare against on a scanned corpus, so these are
    the checks that need only the image and the answer. Each one is a way a
    vision model fails that produces output which parses perfectly.
    """
    if page is None:
        return ["unreadable"]

    text = " ".join(b.text for b in page.blocks)
    found: list[str] = []

    if ink > INKY and len(text) < MIN_CHARS_FOR_INKY_PAGE:
        found.append("short_output")
    if ink > INK_FLOOR and not text.strip():
        found.append("empty")
    if _SCRIPT_DRIFT.search(text):
        found.append("script_drift")
    if _repeats(text):
        found.append("repetition")
    return found


def _repeats(text: str, n: int = REPEAT_NGRAM, times: int = REPEAT_TIMES) -> bool:
    """Whether the same run of words comes back over and over.

    The classic vision-model collapse: it finds a phrase it likes and emits it
    until the token budget runs out. The output is well-formed and the page is
    gone.
    """
    words = _WORD.findall(text)
    if len(words) < n * times:
        return False
    seen: dict[tuple[str, ...], int] = {}
    for start in range(len(words) - n + 1):
        gram = tuple(words[start:start + n])
        seen[gram] = seen.get(gram, 0) + 1
        if seen[gram] >= times:
            return True
    return False


def ink_ratio(image_png: bytes) -> float:
    """The share of the page that is not blank paper."""
    try:
        import io

        from PIL import Image
    except ImportError:      # pragma: no cover - Pillow ships with the pipeline
        return 0.0
    with Image.open(io.BytesIO(image_png)) as image:
        grey = image.convert("L")
        # Downscale first: this is a coverage estimate, and a full-size scan of
        # every pixel of a 300 dpi page buys nothing.
        grey.thumbnail((400, 400))
        pixels = list(grey.getdata())
    if not pixels:
        return 0.0
    return sum(1 for p in pixels if p < 250) / len(pixels)
