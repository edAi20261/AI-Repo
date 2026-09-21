"""Deciding which blocks travel together.

A block is a region of a page. A chunk is one row in the vector store. They are
not the same thing, and the difference is where retrieval quality comes from.

The test every block has to pass is whether it can stand alone and still mean
something. "A liquid that dissolves substances is called a solvent" can. "Only
this one doesn't dissolve quickly in water" cannot — on its own it is six words
that match nothing and answer nobody. It travels with the chunk it belongs to
and is never embedded by itself.

That rule is what stops an illustrated textbook producing thousands of
meaningless six-word vectors.
"""

from __future__ import annotations

import re
from statistics import median
from typing import Iterable, Optional, Sequence

from pydantic import BaseModel, Field

from ..utils.tokens import count_tokens
from .blocks import Block, PageLayout, PageMeta, ParsedPage
from .reading_order import sort_blocks

# Retrieval precision, not a model limit: BGE-M3 accepts 8192 tokens, but a
# vector averaged over many topics matches all of them weakly.
MAX_TOKENS = 800
# Below this a chunk is a fragment. It joins the one before it under the same
# heading rather than becoming a row of its own.
MIN_TOKENS = 40
# A block shorter than this, that is not itself a heading or a question, is a
# label rather than a statement.
PASSENGER_MAX_WORDS = 25
# Left edges within this share of the page width belong to the same column.
COLUMN_TOLERANCE = 0.10
# A page whose blocks are mostly single words is a table or a diagram: its text
# is labels, not prose. Measured over 121 pages of a Grade 6 science module, the
# median words per block is 1.0 on the periodic table and on graph pages, and
# 5.0 to 11.0 on every page of prose — including the magazine-style openers.
#
# This is the signal because the usual one cannot work here: the publisher
# flattens each page into a single full-page image, which the parser correctly
# treats as a background, so the fraction of the page covered by figures is zero
# everywhere. A vision tier reports the layout directly and is preferred when it
# ran; this is what Tier 1 can see on its own.
LABEL_MEDIAN_WORDS = 2
LABEL_MIN_BLOCKS = 8

# A sentence longer than this is not carried across a split as overlap.
OVERLAP_MAX = MAX_TOKENS // 4

# Where a sentence ends, across the scripts a syllabus arrives in: Latin,
# Arabic (؟), Urdu (۔), Devanagari (।॥), Armenian (։), Ethiopic (።) and the CJK
# full stops. CJK sets no space after its stop, so there the mark itself is the
# break rather than the whitespace that follows one.
_SENTENCE_END = re.compile(r"(?<=[\u3002\uFF01\uFF1F])|(?<=[.!?؟۔।॥։።])\s+")

# Types that never become a chunk on their own: page furniture, and the image
# itself, which carries no text.
SKIP = frozenset({"header", "footer", "page_number", "boilerplate", "figure"})
# Types that open a new group.
OPENS_GROUP = frozenset({"heading", "subheading"})
# Types that are self-contained: someone may search for one directly.
STANDS_ALONE = frozenset({"sidebar", "question"})
# Types that annotate whatever they sit beside.
ANNOTATES = frozenset({"caption", "label"})


class PageChunk(BaseModel):
    """One page's worth of blocks that belong together.

    Named apart from ``ingestion.state.Chunk``, which is the row written to
    ``ai.chunks``. This is the layout decision; that is the record. Keeping one
    name for both invites code that converts a thing into itself.
    """

    text: str
    block_type: str = "section"
    heading: Optional[str] = None
    page: int
    doc_id: str
    bbox: Optional[tuple[float, float, float, float]] = None
    figure_id: Optional[str] = None
    has_figure: bool = False
    page_image: Optional[str] = None
    answer_key: Optional[str] = Field(
        default=None,
        description="Stored, never embedded — otherwise searching for a concept "
                    "retrieves the answer sheet.")
    passenger_count: int = 0

    @property
    def tokens(self) -> int:
        return count_tokens(self.text)


def page_to_chunks(page: ParsedPage) -> list[PageChunk]:
    """The chunks one parsed page produces.

    Four outcomes, and every page in every document falls into one of them.
    """
    blocks = content_blocks(page.blocks)
    if not blocks:
        return []

    layout = classify_layout(blocks, page.meta)
    if layout == "image_dominant":
        chunks = [whole_page_chunk(blocks, page.meta, "page_opener")]
    elif layout == "scattered":
        chunks = [whole_page_chunk(blocks, page.meta, "visual_summary")]
    elif layout == "diagram":
        chunks = diagram_chunks(blocks, page.meta)
    else:
        chunks = standard_chunks(blocks, page.meta)
    # The size guard applies to every outcome. A whole page of scattered boxes
    # is still one thought, but a 1,250-token vector averages too many topics to
    # match any of them well.
    return split_oversized(chunks)


def content_blocks(blocks: Sequence[Block]) -> list[Block]:
    """The blocks that can become, or join, a chunk.

    Applied at every entry point rather than once at the top: a running head or
    a copyright line reaching a chunk is the kind of defect nobody notices until
    a student reads it, and standard_chunks is called directly.
    """
    return [b for b in blocks if b.type not in SKIP and b.text.strip()]


def classify_layout(blocks: Sequence[Block], meta: PageMeta) -> PageLayout:
    """What kind of page this is.

    The vision tier reports this directly when it ran; this derives it from the
    geometry when it did not.
    """
    if meta.layout != "standard":
        return meta.layout

    words = sum(b.words for b in blocks)
    figures = meta.figure_fraction

    if _is_label_dominant(blocks):
        return "scattered"
    if figures > 0.55 and words < 120:
        return "image_dominant"
    if any(b.type in OPENS_GROUP for b in blocks) and _has_columns(blocks, meta.width):
        return "standard"
    if figures > 0.35 and sum(1 for b in blocks if _annotates(b)) >= 3:
        return "diagram"
    if len(blocks) > 4 and not _has_columns(blocks, meta.width):
        return "scattered"
    return "standard"


# --- the ordinary page ------------------------------------------------------

def standard_chunks(blocks: Sequence[Block], meta: PageMeta) -> list[PageChunk]:
    """A page of prose under headings: one chunk per heading and its body."""
    blocks = content_blocks(blocks)
    groups: list[dict] = []
    current: Optional[dict] = None

    for block in blocks:
        if block.type in OPENS_GROUP:
            if current:
                groups.append(current)
            current = {"heading": block, "body": [], "extras": []}
        elif block.type in STANDS_ALONE:
            groups.append({"heading": None, "body": [block], "extras": [],
                           "kind": block.type})
        elif block.type in ANNOTATES or block.type in ("list", "table"):
            current = current or {"heading": None, "body": [], "extras": []}
            current["extras"].append(block)
        else:
            current = current or {"heading": None, "body": [], "extras": []}
            current["body"].append(block)
    if current:
        groups.append(current)

    # A heading with nothing under it was not a heading. The cells of a periodic
    # table and the labels on a graph axis are set large and bold, so they open a
    # group each and then produce a chunk that is two characters long. They are
    # labels, and they travel as passengers like any other label.
    sections = [g for g in groups if g["body"] or g["extras"]]
    labels = [g["heading"] for g in groups
              if g["heading"] and not (g["body"] or g["extras"])]

    consumed = {id(b) for g in sections
                for b in ([g["heading"]] if g["heading"] else []) + g["body"] + g["extras"]}
    consumed |= {id(b) for b in labels}
    chunks = [c for c in (_group_to_chunk(g, meta) for g in sections) if c.text.strip()]
    chunks = attach_passengers(chunks, [b for b in blocks if id(b) not in consumed])
    chunks = attach_passengers(chunks, labels, test=False)
    return merge_undersized(chunks)


def _group_to_chunk(group: dict, meta: PageMeta) -> PageChunk:
    """One group of blocks as a chunk."""
    heading: Optional[Block] = group["heading"]
    parts = [heading.text] if heading else []
    parts += [b.text for b in group["body"]]
    parts += [b.text for b in group["extras"]]
    members = ([heading] if heading else []) + group["body"] + group["extras"]
    return PageChunk(
        text="\n".join(p for p in parts if p),
        block_type=group.get("kind", "section"),
        heading=heading.text if heading else None,
        bbox=_union(members),
        has_figure=any(b.overlaps_figure for b in members),
        page_image=meta.image_uri if any(b.overlaps_figure for b in members) else None,
        page=meta.page,
        doc_id=meta.doc_id,
    )


# --- the rule that saves retrieval quality ----------------------------------

def is_passenger(block: Block) -> bool:
    """Whether this block means nothing on its own.

    A caption or a label always does. Anything short does too, unless it is the
    kind of thing someone searches for directly — a heading, a question, a
    sidebar.
    """
    if block.type in ANNOTATES or block.type == "page_number":
        return True
    if block.type in OPENS_GROUP or block.type in STANDS_ALONE:
        return False
    return block.words < PASSENGER_MAX_WORDS


def attach_passengers(
    chunks: list[PageChunk], orphans: Sequence[Block], *, test: bool = True
) -> list[PageChunk]:
    """Give every fragment no group claimed to the chunk it sits nearest.

    Written into the chunk text in brackets so it is embedded *with* its parent,
    while a reader can still see it was separate on the page.

    ``test`` is off for blocks the caller has already judged — a heading with
    nothing under it is a label, and would fail the passenger test that headings
    exist to fail.
    """
    if not chunks:
        return chunks

    for orphan in (b for b in orphans if not test or is_passenger(b)):
        target = _nearest(chunks, orphan)
        if target is None:
            continue
        target.text += f"\n[{orphan.type}: {orphan.text}]"
        target.passenger_count += 1
    return chunks


def _nearest(chunks: list[PageChunk], block: Block) -> Optional[PageChunk]:
    """The chunk whose region sits closest to this block on the page."""
    placed = [c for c in chunks if c.bbox]
    if not placed:
        return chunks[0] if chunks else None
    return min(placed, key=lambda c: _centre_distance(c.bbox, block.bbox))


# --- pages that are mostly picture ------------------------------------------

def whole_page_chunk(
    blocks: Sequence[Block], meta: PageMeta, block_type: str
) -> PageChunk:
    """One chunk for a page whose text only means anything together.

    An opener with a title over a photograph, or a spread of scattered boxes
    that are a single thought between them. Cutting either into pieces produces
    fragments; keeping them whole produces one findable thing.
    """
    ordered = _banded(content_blocks(blocks), meta)
    body = [b for b in ordered if b.type != "answer_key"]
    hidden = [b for b in ordered if b.type == "answer_key"]
    return PageChunk(
        text="\n".join(b.text for b in body),
        block_type=block_type,
        bbox=_union(list(blocks)),
        has_figure=True,
        page_image=meta.image_uri,
        # Never embedded: a search for a concept must not return the answer sheet.
        answer_key="\n".join(b.text for b in hidden) or None,
        page=meta.page,
        doc_id=meta.doc_id,
    )


def diagram_chunks(blocks: Sequence[Block], meta: PageMeta) -> list[PageChunk]:
    """One chunk per annotation on an illustration.

    Each callout on a diagram is usually self-contained — a small heading and a
    sentence — so each is findable on its own, and all of them share a figure id
    so a retriever can pull the siblings and show the whole illustration.
    """
    figure_id = f"{meta.doc_id}-p{meta.page}-fig1"
    return [
        PageChunk(
            text=block.text,
            block_type="diagram_callout",
            figure_id=figure_id,
            has_figure=True,
            page_image=meta.image_uri,
            bbox=block.bbox,
            page=meta.page,
            doc_id=meta.doc_id,
        )
        for block in content_blocks(blocks)
        if _annotates(block) or block.words >= 3
    ]


# --- size guards ------------------------------------------------------------

def split_oversized(chunks: list[PageChunk], limit: int = MAX_TOKENS) -> list[PageChunk]:
    """Break a chunk that grew past the limit, on sentence boundaries only."""
    out: list[PageChunk] = []
    for chunk in chunks:
        if chunk.tokens <= limit:
            out.append(chunk)
            continue
        for piece in _split_text(chunk.text, limit):
            out.append(chunk.model_copy(update={"text": piece}))
    return out


def merge_undersized(chunks: list[PageChunk], floor: int = MIN_TOKENS) -> list[PageChunk]:
    """Fold a fragment into the chunk before it, under the same heading.

    A fragment with nothing before it survives only if it is the kind of thing
    someone searches for.
    """
    out: list[PageChunk] = []
    for chunk in chunks:
        if chunk.tokens >= floor:
            out.append(chunk)
            continue
        if out and out[-1].heading == chunk.heading:
            out[-1].text += "\n" + chunk.text
            continue
        if chunk.heading or chunk.block_type != "section":
            out.append(chunk)
    return out


def _split_text(text: str, limit: int) -> list[str]:
    pieces: list[str] = []
    current: list[str] = []
    for sentence in _sentences(text, limit):
        candidate = current + [sentence]
        if current and count_tokens(" ".join(candidate)) > limit:
            pieces.append(" ".join(current))
            # One sentence of overlap, so a statement split across the join is
            # still whole in one of the pieces — but only while that sentence is
            # small. Carrying a long one forward doubles the piece it starts,
            # which is how a 3,000-token run came back as pieces of 1,600.
            tail = current[-1]
            current = [tail, sentence] if count_tokens(tail) <= OVERLAP_MAX else [sentence]
        else:
            current = candidate
    if current:
        pieces.append(" ".join(current))
    return pieces


def _sentences(text: str, limit: int) -> list[str]:
    """The text as sentences, and any single sentence longer than a whole chunk
    broken on word boundaries.

    Sentence boundaries are the right place to cut, and sometimes there are
    none: a transcribed Arabic page came back as one 3,000-token run with no
    punctuation at all, and the size guard passed it through whole. A vector
    averaged over 3,000 tokens matches nothing in particular, so a cut on a word
    boundary is worse than a sentence and much better than no cut.
    """
    out: list[str] = []
    sentences = [s for s in _SENTENCE_END.split(text) if s.strip()] or [text]
    for sentence in sentences:
        if count_tokens(sentence) <= limit:
            out.append(sentence)
            continue
        words, run = sentence.split(), []
        for word in words:
            run.append(word)
            if count_tokens(" ".join(run)) >= limit:
                out.append(" ".join(run))
                run = []
        if run:
            out.append(" ".join(run))
    return out or [text]


# --- geometry ---------------------------------------------------------------

def _is_label_dominant(blocks: Sequence[Block]) -> bool:
    """Whether the page's text is labels rather than sentences.

    The cells of a periodic table and the tick marks on a graph axis mean
    nothing apart; the page means something whole. Chunking each label produces
    a hundred rows reading "Cm 96".
    """
    if len(blocks) < LABEL_MIN_BLOCKS:
        return False
    return median(b.words for b in blocks) <= LABEL_MEDIAN_WORDS


def _annotates(block: Block) -> bool:
    return block.type in ANNOTATES or block.belongs_to is not None


def _has_columns(blocks: Sequence[Block], page_width: float) -> bool:
    """Whether the blocks line up on one or two left edges.

    The edges are clustered against the page, not against each other. Measured
    against the spread of the edges themselves, an indented list item reads as a
    third column — one Grade 6 science page has its left edges at 63, 77, 120
    and 204 points, which is two columns with indents and was being called
    scattered.
    """
    if len(blocks) < 3:
        return True
    tolerance = page_width * COLUMN_TOLERANCE
    lefts = sorted(b.bbox[0] for b in blocks)
    columns = 1
    for previous, current in zip(lefts, lefts[1:]):
        if current - previous > tolerance:
            columns += 1
    return columns <= 2


def _banded(blocks: Sequence[Block], meta: PageMeta) -> list[Block]:
    """Scattered boxes in a sane order: rows down the page, then along the row."""
    return sort_blocks(blocks, page_height=meta.height,
                       direction=meta.reading_direction)


def _union(blocks: Iterable[Block]) -> Optional[tuple[float, float, float, float]]:
    boxes = [b.bbox for b in blocks]
    if not boxes:
        return None
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


def _centre_distance(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    ax, ay = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
    bx, by = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
