"""A page as regions of text, in the order a human reads them.

Plain extraction returns text in *drawing* order, which is the order the PDF
writer happened to emit it. On page 17 of a Grade 6 science module that order
opens with the take-home box from the bottom-right corner, gives the second
section's heading before the first section's body, and hands the first
section's heading over after the body it introduces.

Splitting that by character count produces chunks that read like nonsense and
attach the wrong heading to the wrong text. So nothing downstream splits text:
the page is parsed into blocks, and the chunker decides which blocks travel
together.

Every tier — text layer, layout model, vision model — emits this same shape, so
no consumer can tell which one produced a page.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

BlockType = Literal[
    "heading", "subheading", "body", "caption", "sidebar",
    "question", "answer_key", "table", "figure", "list",
    "header", "footer", "boilerplate", "page_number", "unknown",
]

PageLayout = Literal["standard", "image_dominant", "scattered", "diagram", "mixed"]

ReadingDirection = Literal["ltr", "rtl"]

# Text that frames a page rather than saying anything: running heads, folios,
# and the image credits set in 4pt up the side of a photograph.
FURNITURE: frozenset[str] = frozenset(
    {"header", "footer", "page_number", "boilerplate"})


class Block(BaseModel):
    """One region of a page: what it says, where it sits, and what kind it is."""

    text: str
    type: BlockType = "body"
    bbox: tuple[float, float, float, float] = Field(
        description="x0, y0, x1, y1 in page points, origin top-left.")
    reading_order: int = 0
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    rotation: float = Field(default=0.0, description="Degrees; 180 is upside down.")
    overlaps_figure: bool = False
    source_tier: int = 1
    belongs_to: Optional[int] = Field(
        default=None,
        description="reading_order of the block this one annotates, when it "
                    "only means something beside that block.")

    @field_validator("text", mode="before")
    @classmethod
    def _collapse_whitespace(cls, v):
        return " ".join(str(v or "").split())

    @property
    def is_furniture(self) -> bool:
        """Whether this frames the page rather than saying anything."""
        return self.type in FURNITURE

    @property
    def words(self) -> int:
        return len(self.text.split())

    @property
    def area(self) -> float:
        x0, y0, x1, y1 = self.bbox
        return max(0.0, x1 - x0) * max(0.0, y1 - y0)


class PageMeta(BaseModel):
    """What the chunker needs to know about the page a block came from."""

    doc_id: str
    page: int
    width: float
    height: float
    layout: PageLayout = "standard"
    reading_direction: ReadingDirection = "ltr"
    image_uri: Optional[str] = None
    source_tier: int = 1
    figure_area: float = Field(
        default=0.0,
        description="Combined area of the figures on the page, backgrounds excluded.")

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def figure_fraction(self) -> float:
        """How much of the page is illustration."""
        return (self.figure_area / self.area) if self.area else 0.0


class ParsedPage(BaseModel):
    """One page, parsed. The unit the page cache stores and the chunker consumes."""

    meta: PageMeta
    blocks: list[Block] = Field(default_factory=list)

    @property
    def content(self) -> list[Block]:
        """The blocks that say something — furniture removed."""
        return [b for b in self.blocks if not b.is_furniture]

    @property
    def words(self) -> int:
        return sum(b.words for b in self.content)
