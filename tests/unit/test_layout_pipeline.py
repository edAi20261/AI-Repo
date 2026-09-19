"""The layout parser and chunker, wired into the pipeline that writes the rows.

``PageChunk`` is what the chunker decided; ``Chunk`` is the record written to
``ai.chunks`` and indexed in Qdrant. These cover the bridge between them, and
the router that decides which parser a page needs.
"""

from __future__ import annotations

from ingestion.layout.blocks import Block, PageMeta, ParsedPage
from ingestion.layout.bridge import chunks_for, payload_for
from ingestion.layout.chunker import PageChunk
from ingestion.layout.router import TEXT_LAYER, VISION, choose_tier

PROSE = ("Water covers most of the planet and moves constantly between the sea, "
         "the air and the land, driven by heat from the sun. ") * 3


class _Rect:
    width, height = 600.0, 800.0


class _Page:
    rect = _Rect()


def _block(text, type="body", y=100, x=50, **kw):
    return Block(text=text, type=type, bbox=(x, y, x + 300, y + 40), **kw)


def _page(number, blocks):
    return ParsedPage(meta=PageMeta(doc_id="d", page=number, width=600, height=800),
                      blocks=blocks)


# --- the bridge -------------------------------------------------------------

def test_chunks_are_numbered_across_the_whole_lesson_not_per_page():
    """order_index keys the row, the manifest and the Qdrant point. Restarting
    it on every page collides them."""
    rows = chunks_for(
        [_page(1, [_block("Tides", "heading", y=50), _block(PROSE)]),
         _page(2, [_block("Currents", "heading", y=50), _block(PROSE)])],
        lecture_id="lec", language="en")
    assert [r.order_index for r in rows] == list(range(len(rows)))
    assert [r.page_from for r in rows] == [1, 2]


def test_a_section_keeps_its_heading_onto_the_next_page():
    """A section runs past the page it started on. Without carrying the heading
    forward, every chunk after the first on a spread has none at all — which
    costs the context prefix its section name."""
    rows = chunks_for(
        [_page(1, [_block("Ocean Floor", "heading", y=50), _block(PROSE)]),
         _page(2, [_block(PROSE)])],
        lecture_id="lec", language="en")
    assert [r.heading_path for r in rows] == ["Ocean Floor", "Ocean Floor"]


def test_the_row_carries_the_fields_the_index_stage_needs():
    rows = chunks_for([_page(3, [_block("Tides", "heading", y=50), _block(PROSE)])],
                      lecture_id="lec", language="ar", version=2,
                      namespace="syllabus", subject_id="subj", grade="g7",
                      lesson_title="Unit 2 · Lesson 1")
    row = rows[0]
    assert (row.lecture_id, row.language, row.version) == ("lec", "ar", 2)
    assert (row.subject_id, row.grade) == ("subj", "g7")
    assert row.lesson_title == "Unit 2 · Lesson 1"
    assert row.page_from == row.page_to == 3


def test_an_answer_key_rides_in_the_payload_and_not_the_text():
    """Stored so a teacher view can show it; never embedded, or a search for the
    concept returns the answer sheet."""
    chunk = PageChunk(text="the lesson", page=1, doc_id="d",
                      answer_key="1. b  2. d", block_type="visual_summary")
    payload = payload_for(chunk)
    assert payload["answer_key"] == "1. b  2. d"
    assert payload["block_type"] == "visual_summary"
    assert "the lesson" not in payload.values()


def test_a_diagram_callout_carries_its_figure_id_so_siblings_can_be_pulled():
    chunk = PageChunk(text="The trench forms here", page=1, doc_id="d",
                      block_type="diagram_callout", figure_id="d-p1-fig1",
                      has_figure=True)
    assert payload_for(chunk)["figure_id"] == "d-p1-fig1"


# --- the router -------------------------------------------------------------

def test_a_page_with_no_text_layer_needs_vision():
    assert choose_tier(_Page(), char_count=0, blocks=[]) == VISION


def test_a_clean_page_of_prose_is_read_by_the_text_layer():
    blocks = [_block("Tides", "heading", y=50), _block(PROSE, y=120)]
    assert choose_tier(_Page(), char_count=900, blocks=blocks) == TEXT_LAYER


def test_a_rotated_image_credit_does_not_make_a_page_scattered():
    """An illustrated textbook sets its credit vertically up the side of the
    photograph, so every page carries a rotated block. Reading that as scattered
    text sent all forty pages of a slice to the vision tier — the slowest answer
    and the wrong one."""
    blocks = [
        _block("Tides", "heading", y=50),
        _block(PROSE, y=120),
        _block("© A Publisher. Image credits: Alamy.", "boilerplate",
               y=400, rotation=90.0),
    ]
    assert choose_tier(_Page(), char_count=900, blocks=blocks) == TEXT_LAYER


def test_rotated_content_still_means_the_page_needs_a_better_parser():
    """A callout printed sideways or a self-check key printed upside down is
    content, and its position says nothing about where it belongs in the
    reading."""
    blocks = [_block("Tides", "heading", y=50), _block(PROSE, y=120),
              _block("Answers: 1b 2d", "answer_key", y=700, rotation=180.0)]
    assert choose_tier(_Page(), char_count=900, blocks=blocks) == VISION
