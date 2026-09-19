"""Which blocks travel together.

A block is a region of a page; a chunk is one row in the vector store. The
difference between them is where retrieval quality comes from.

The test every block has to pass is whether it can stand alone and still mean
something. "A liquid that dissolves substances is called a solvent" can. "Only
this one doesn't dissolve quickly in water" cannot — alone it is six words that
match nothing and answer nobody. It travels with the chunk it belongs to.
"""

from __future__ import annotations

from ingestion.layout.blocks import Block, PageMeta, ParsedPage
from ingestion.layout.chunker import (
    classify_layout,
    is_passenger,
    merge_undersized,
    page_to_chunks,
    split_oversized,
    standard_chunks,
)

META = PageMeta(doc_id="d", page=7, width=600, height=800)
PROSE = ("Water covers most of the planet and moves constantly between the sea, "
         "the air and the land, driven by heat from the sun. ") * 3


def _block(text, type="body", y=100, x=50, **kw):
    return Block(text=text, type=type, bbox=(x, y, x + 200, y + 40), **kw)


# --- grouping ---------------------------------------------------------------

def test_a_heading_travels_with_the_body_it_introduces():
    chunks = standard_chunks(
        [_block("The Water Cycle", "heading", y=50), _block(PROSE, y=100)], META)
    assert len(chunks) == 1
    assert chunks[0].heading == "The Water Cycle"
    assert "Water covers most" in chunks[0].text


def test_a_second_heading_starts_a_second_chunk():
    chunks = standard_chunks([
        _block("Evaporation", "heading", y=50), _block(PROSE, y=100),
        _block("Condensation", "heading", y=300), _block(PROSE, y=350),
    ], META)
    assert [c.heading for c in chunks] == ["Evaporation", "Condensation"]


def test_a_question_is_its_own_chunk_because_someone_searches_for_it():
    chunks = standard_chunks([
        _block("The Water Cycle", "heading", y=50), _block(PROSE, y=100),
        _block("What drives the water cycle and why does it never stop running?",
               "question", y=400),
    ], META)
    assert any(c.text.startswith("What drives") for c in chunks)


# --- the passenger rule -----------------------------------------------------

def test_a_caption_is_never_embedded_on_its_own():
    """The rule that stops an illustrated textbook producing thousands of
    meaningless six-word vectors."""
    assert is_passenger(_block("Figure 3: the water cycle", "caption"))
    assert is_passenger(_block("Only this one doesn't dissolve quickly."))
    assert not is_passenger(_block("The Water Cycle", "heading"))
    assert not is_passenger(_block("Why does it rain?", "question"))


def test_a_caption_inside_a_section_is_part_of_that_section():
    chunks = standard_chunks([
        _block("The Water Cycle", "heading", y=50), _block(PROSE, y=100),
        _block("Evaporation happens here", "caption", y=140),
    ], META)
    assert len(chunks) == 1
    assert "Evaporation happens here" in chunks[0].text


def test_a_label_no_section_claimed_is_attached_to_the_chunk_nearest_it():
    """Bracketed, so it is embedded with its parent while a reader can still see
    it was separate on the page."""
    chunks = standard_chunks([
        _block("The Water Cycle", "heading", y=50), _block(PROSE, y=100),
        _block("Cm", "heading", y=600),
    ], META)
    assert len(chunks) == 1
    assert "[heading: Cm]" in chunks[0].text
    assert chunks[0].passenger_count == 1


def test_a_passenger_is_attached_once():
    chunks = standard_chunks([
        _block("Ratios", "heading", y=50), _block(PROSE, y=100),
        _block("Step 1 Write the ratio.", "caption", y=150),
    ], META)
    assert chunks[0].text.count("Step 1 Write the ratio.") == 1


def test_a_heading_with_nothing_under_it_was_not_a_heading():
    """The cells of a periodic table and the labels on a graph axis are set
    large and bold, so each opens a group and then produces a chunk two
    characters long. They are labels, and they travel as passengers."""
    chunks = standard_chunks([
        _block("Solvents", "heading", y=50), _block(PROSE, y=100),
        _block("Cm", "heading", y=400), _block("Eu", "heading", y=450),
    ], META)
    assert len(chunks) == 1
    assert "[heading: Cm]" in chunks[0].text


# --- size guards ------------------------------------------------------------

def test_an_oversized_chunk_is_split_on_sentence_boundaries():
    chunks = split_oversized(standard_chunks(
        [_block("Long", "heading", y=50), _block(PROSE * 12, y=100)], META), limit=200)
    assert len(chunks) > 1
    assert all(c.tokens <= 320 for c in chunks)
    # Never mid-sentence: every piece ends where a sentence does.
    assert all(c.text.rstrip().endswith((".", "!", "?")) for c in chunks)


def test_a_fragment_joins_the_chunk_before_it_under_the_same_heading():
    from ingestion.layout.chunker import PageChunk

    merged = merge_undersized([
        PageChunk(text=PROSE, heading="Tides", page=7, doc_id="d"),
        PageChunk(text="Short tail.", heading="Tides", page=7, doc_id="d"),
    ])
    assert len(merged) == 1
    assert merged[0].text.endswith("Short tail.")


def test_a_fragment_with_nothing_before_it_survives_only_if_it_is_findable():
    from ingestion.layout.chunker import PageChunk

    kept = merge_undersized([PageChunk(text="Tiny.", heading="Tides", page=7, doc_id="d")])
    dropped = merge_undersized([PageChunk(text="Tiny.", page=7, doc_id="d")])
    assert len(kept) == 1
    assert dropped == []


# --- whole pages ------------------------------------------------------------

def test_a_page_of_labels_becomes_one_chunk_not_a_hundred():
    """The periodic table. Its cells mean nothing apart; the page means
    something whole. Measured over 121 pages of a Grade 6 science module, the
    median words per block is 1.0 there and 5.0 to 11.0 on every page of prose."""
    cells = [_block(f"E{n}", "heading", y=50 + n * 10, x=40 + (n % 8) * 60)
             for n in range(40)]
    chunks = page_to_chunks(ParsedPage(meta=META, blocks=cells))
    assert len(chunks) == 1
    assert chunks[0].block_type == "visual_summary"


def test_an_answer_key_is_stored_but_never_embedded():
    """Otherwise a search for a concept retrieves the answer sheet."""
    blocks = [_block(f"Box {n}", y=50 + n * 60, x=40 + (n % 3) * 150) for n in range(6)]
    blocks.append(_block("1. b 2. d 3. a", "answer_key", y=700, rotation=180.0))
    page = ParsedPage(meta=META.model_copy(update={"layout": "scattered"}), blocks=blocks)

    chunk = page_to_chunks(page)[0]
    assert "1. b" not in chunk.text
    assert chunk.answer_key == "1. b 2. d 3. a"


def test_a_page_that_is_mostly_picture_is_one_opener_chunk():
    meta = META.model_copy(update={"figure_area": 600 * 800 * 0.7})
    page = ParsedPage(meta=meta, blocks=[
        _block("Unit 2: Oceanography", "heading", y=60),
        _block("How deep is the sea?", y=120),
    ])
    assert classify_layout(page.blocks, meta) == "image_dominant"
    assert page_to_chunks(page)[0].block_type == "page_opener"


def test_scattered_boxes_are_read_in_rows_not_in_the_order_they_were_drawn():
    """Without the banding a box three points higher than its neighbour sorts
    ahead of it and the page reads like nonsense."""
    meta = META.model_copy(update={"layout": "scattered"})
    page = ParsedPage(meta=meta, blocks=[
        _block("second", y=100, x=400), _block("first", y=103, x=40),
        _block("third", y=300, x=40),
    ])
    assert page_to_chunks(page)[0].text.splitlines() == ["first", "second", "third"]


def test_arabic_scattered_boxes_are_read_from_the_right():
    meta = META.model_copy(update={"layout": "scattered", "reading_direction": "rtl"})
    page = ParsedPage(meta=meta, blocks=[
        _block("left", y=100, x=40), _block("right", y=100, x=400),
        _block("below", y=300, x=40),
    ])
    assert page_to_chunks(page)[0].text.splitlines() == ["right", "left", "below"]


# --- furniture --------------------------------------------------------------

def test_page_furniture_never_becomes_a_chunk():
    chunks = standard_chunks([
        _block("The Water Cycle", "heading", y=50), _block(PROSE, y=100),
        _block("7", "page_number", y=770), _block("© A Publisher", "boilerplate", y=780),
    ], META)
    assert all("Publisher" not in c.text for c in chunks)
    assert all("\n7" not in c.text for c in chunks)
