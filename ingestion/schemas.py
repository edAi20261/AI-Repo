"""Pydantic contracts for LLM and vision JSON outputs.

Model responses are validated through these before the pipeline uses them, so the
clamping and vocabulary rules live in one place (field validators) rather than
scattered across the parse helpers. The ``*List`` wrappers are the targets for
provider-native structured output; the tolerant text parsers reuse the item models.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, field_validator

BLOOM_LEVELS = {"remember", "understand", "apply", "analyze", "evaluate", "create"}
# The platform owns ai.dependency_strength; these are its values. A model asked for
# a prerequisite's strength reaches for other words, so near-synonyms are mapped
# rather than dropped.
DEPENDENCY_STRENGTHS = {"required", "related"}
DEPENDENCY_STRENGTH_ALIASES = {
    "recommended": "related",
    "optional": "related",
    "helpful": "related",
    "useful": "related",
    "supporting": "related",
    "mandatory": "required",
    "essential": "required",
    "prerequisite": "required",
}


# --- vision ----------------------------------------------------------------

class FigureBox(BaseModel):
    """A figure region in pixel coordinates of the rendered page (the crop box)."""

    bbox: tuple[float, float, float, float]


class PageTranscription(BaseModel):
    """One page-vision result: page text plus figure boxes. ``unreadable`` is set
    when the page is too degraded to read; such a page is flagged, not indexed."""

    text: str = ""
    figures: list[FigureBox] = Field(default_factory=list)
    unreadable: bool = False


class FigureCaption(BaseModel):
    caption: str = ""
    alt_text: str = ""


# --- topics ----------------------------------------------------------------

class TopicItem(BaseModel):
    """One extracted skill. ``label`` is clamped to the ``ai.topics.label`` width;
    an out-of-vocabulary ``bloom_level`` folds to ``None`` rather than failing."""

    label: str
    description: Optional[str] = None
    bloom_level: Optional[str] = None

    @field_validator("label", mode="before")
    @classmethod
    def _clamp_label(cls, v):
        return str(v or "").strip()[:255]

    @field_validator("description", mode="before")
    @classmethod
    def _empty_description_to_none(cls, v):
        return (str(v).strip() or None) if v is not None else None

    @field_validator("bloom_level", mode="before")
    @classmethod
    def _known_bloom_only(cls, v):
        s = str(v or "").strip().lower()
        return s if s in BLOOM_LEVELS else None


class TopicList(BaseModel):
    """Structured-output wrapper: a lesson's topics."""

    topics: list[TopicItem] = Field(default_factory=list)


# --- dependencies ----------------------------------------------------------

class DependencyItem(BaseModel):
    """One prerequisite edge proposed for a topic. ``prereq`` is a candidate label;
    an unknown ``strength`` folds to ``required``."""

    prereq: str
    strength: str = "required"
    rationale: Optional[str] = None

    @field_validator("prereq", mode="before")
    @classmethod
    def _strip_prereq(cls, v):
        return str(v or "").strip()

    @field_validator("strength", mode="before")
    @classmethod
    def _known_strength_only(cls, v):
        s = str(v or "").strip().lower()
        s = DEPENDENCY_STRENGTH_ALIASES.get(s, s)
        return s if s in DEPENDENCY_STRENGTHS else "required"

    @field_validator("rationale", mode="before")
    @classmethod
    def _empty_rationale_to_none(cls, v):
        return (str(v).strip() or None) if v is not None else None


class DependencyList(BaseModel):
    """Structured-output wrapper: one topic's prerequisites."""

    dependencies: list[DependencyItem] = Field(default_factory=list)
