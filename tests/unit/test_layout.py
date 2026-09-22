"""A page becomes blocks in the order a human reads them.

Plain extraction returns text in *drawing* order — the order the PDF writer
happened to emit it. On page 17 of a Grade 6 science module that order opens
with the take-home box from the bottom-right corner, gives the second section's
heading before the first section's body, and hands over the first section's
heading after the body it introduces.

Splitting that by character count attaches the wrong heading to the wrong text.
The tests below build the same shape of page so the failure is reproducible
without the book.
"""

from __future__ import annotations

import pytest

fitz = pytest.importorskip("fitz")

from ingestion.layout.blocks import Block, PageMeta, ParsedPage
from ingestion.layout.reading_order import sort_blocks
from ingestion.layout.tier1 import parse_page


def _parse(page) -> ParsedPage:
    meta = PageMeta(doc_id="d", page=1,
                    width=page.rect.width, height=page.rect.height)
    return parse_page(page, meta)


def _drawing_order_page() -> list[Block]:
    """The shape of page 17 of a Grade 6 science module, in drawing order.

    Blocks rather than a generated PDF on purpose. PyMuPDF groups text that
    shares a baseline into one block whatever the horizontal gap, so a synthetic
    two-column page collapses into a single row and cannot reproduce the bug —
    the real page escapes that only because its columns start at different
    heights. The ordering rule is what is under test here; the parser that
    produces these blocks is covered separately.
    """
    return [
        Block(text="Take it home with an adult.", bbox=(390, 680, 560, 700)),
        Block(text="Make a Plan", type="heading", bbox=(390, 40, 560, 70)),
        Block(text="Make a list of five ways.", bbox=(390, 100, 560, 130)),
        Block(text="Water is an essential resource.", bbox=(40, 104, 210, 134)),
        Block(text="Ask a Question", type="heading", bbox=(40, 44, 210, 74)),
    ]


# --- reading order ----------------------------------------------------------

def test_a_heading_comes_before_the_body_it_introduces():
    """The bug this whole module exists for. In drawing order "Ask a Question"
    arrives after its own paragraph, so a chunker that trusts that order glues
    the heading to the next section's text."""
    order = [b.text for b in sort_blocks(_drawing_order_page(), page_height=800)]

    assert order.index("Ask a Question") < order.index("Water is an essential resource.")
    assert order.index("Make a Plan") < order.index("Make a list of five ways.")


def test_the_closing_callout_does_not_open_the_page():
    """It is written first and read last."""
    order = [b.text for b in sort_blocks(_drawing_order_page(), page_height=800)]
    assert order[-1] == "Take it home with an adult."


def test_the_left_column_is_read_before_the_right():
    order = [b.text for b in sort_blocks(_drawing_order_page(), page_height=800)]
    assert order.index("Ask a Question") < order.index("Make a Plan")


def test_arabic_reads_the_rightmost_column_first():
    """Every sort in this pipeline flips for a right-to-left script. Reading the
    left column first turns a two-column Arabic page into alternating nonsense."""
    left = Block(text="left", bbox=(50, 100, 200, 120))
    right = Block(text="right", bbox=(400, 100, 550, 120))

    ltr = [b.text for b in sort_blocks([left, right], page_height=800)]
    rtl = [b.text for b in sort_blocks([left, right], page_height=800, direction="rtl")]

    assert ltr == ["left", "right"]
    assert rtl == ["right", "left"]


def test_a_box_a_few_points_higher_does_not_jump_the_row():
    """Banding is what makes column order survive typesetting. Without it a box
    three points above its neighbour sorts ahead of it and breaks the line."""
    later = Block(text="second", bbox=(400, 103, 550, 123))
    earlier = Block(text="first", bbox=(50, 100, 200, 120))

    assert [b.text for b in sort_blocks([later, earlier], page_height=800)] \
        == ["first", "second"]


def test_reading_order_is_stamped_onto_every_block():
    blocks = [Block(text=str(n), bbox=(50, n * 100, 200, n * 100 + 20)) for n in (3, 1, 2)]
    assert [b.reading_order for b in sort_blocks(blocks, page_height=800)] == [1, 2, 3]


# --- what kind of block it is -----------------------------------------------

def test_a_section_heading_at_the_top_of_the_page_is_not_a_running_head(tmp_path):
    """"Ask a Question" sits inside the top margin of a Grade 6 science page and
    is the most important block on it. Position alone would discard it."""
    doc = fitz.open()
    page = doc.new_page(width=600, height=800)
    page.insert_textbox(fitz.Rect(40, 20, 400, 60), "Ask a Question", fontsize=22)
    page.insert_textbox(fitz.Rect(40, 200, 400, 300),
                        "Water is an essential resource for everyone.", fontsize=10)
    path = tmp_path / "heading.pdf"
    doc.save(path)

    parsed = _parse(fitz.open(path)[0])
    heading = next(b for b in parsed.blocks if "Ask a Question" in b.text)
    assert heading.type == "heading"
    assert not heading.is_furniture


def test_a_rights_notice_is_boilerplate_wherever_it_sits(tmp_path):
    """The credit set vertically up the side of a photograph is not a paragraph
    of the lesson. The copyright mark means that by definition."""
    doc = fitz.open()
    page = doc.new_page(width=600, height=800)
    page.insert_text((60, 400), "© Some Publisher. Image credits: Alamy.", fontsize=9)
    path = tmp_path / "credit.pdf"
    doc.save(path)

    parsed = _parse(fitz.open(path)[0])
    assert parsed.blocks[0].type == "boilerplate"


def test_furniture_is_kept_but_excluded_from_content(tmp_path):
    """Running heads carry the unit and lesson title, which the context prefix
    needs — so they are tagged, not discarded."""
    doc = fitz.open()
    page = doc.new_page(width=600, height=800)
    page.insert_text((60, 300), "Body text of the lesson goes here.", fontsize=11)
    page.insert_text((300, 780), "7", fontsize=9)
    path = tmp_path / "folio.pdf"
    doc.save(path)

    parsed = _parse(fitz.open(path)[0])
    folio = next(b for b in parsed.blocks if b.text == "7")
    assert folio.type == "page_number"
    assert folio not in parsed.content
    assert parsed.words == 7


def test_the_body_size_is_weighted_by_how_much_text_is_set_in_it():
    """A plain median over spans is decided by a page's headings and labels,
    which outnumber its paragraphs. One 22pt title beside one 10pt paragraph
    gives a median of 16, nothing clears 1.4x it, and the title stops being a
    heading."""
    from ingestion.layout.tier1 import body_size

    spans = [
        (None, None, {"size": 22.0, "text": "Ask a Question"}),
        (None, None, {"size": 10.0, "text": "Water is an essential resource for everyone."}),
    ]
    assert body_size(spans) == 10.0


def test_a_glyph_stacked_on_itself_is_counted_once(tmp_path):
    """Illustrated PDFs stack overlapping copies of a glyph to fake a
    typographic effect. One page of a Grade 6 science module emits the letter
    "i" 528 times at identical coordinates — on its own that accounted for most
    of the headings found across the whole book."""
    doc = fitz.open()
    page = doc.new_page(width=600, height=800)
    for _ in range(20):
        page.insert_text((100, 70), "i", fontsize=40)
    page.insert_textbox(fitz.Rect(40, 200, 500, 300),
                        "The ocean floor can be mapped with sonar.", fontsize=11)
    path = tmp_path / "stacked.pdf"
    doc.save(path)

    parsed = _parse(fitz.open(path)[0])
    assert sum(1 for b in parsed.blocks if b.text == "i") == 1


def test_a_lone_glyph_is_never_a_heading(tmp_path):
    """A drop cap cannot introduce a section, and letting it be a heading makes
    the chunker start a group on a single letter."""
    doc = fitz.open()
    page = doc.new_page(width=600, height=800)
    page.insert_text((40, 150), "W", fontsize=44)
    page.insert_textbox(fitz.Rect(40, 200, 500, 320),
                        "ater covers most of the planet's surface and moves "
                        "constantly between the sea, the air and the land.",
                        fontsize=11)
    path = tmp_path / "dropcap.pdf"
    doc.save(path)

    parsed = _parse(fitz.open(path)[0])
    cap = next(b for b in parsed.blocks if b.text == "W")
    assert cap.type == "unknown"


def test_a_heading_is_split_from_the_paragraph_it_introduces(tmp_path):
    """PyMuPDF merges the two whenever they share a text block, and a paragraph
    whose first line is set larger is not a heading — it is a heading and then a
    paragraph, which the chunker has to be able to tell apart."""
    doc = fitz.open()
    page = doc.new_page(width=600, height=800)
    page.insert_textbox(
        fitz.Rect(40, 100, 520, 260),
        "What powers surface currents?\n"
        "The sun heats air near the equator more than it heats air near the "
        "poles, and that difference drives the winds which drive the currents.",
        fontsize=11)
    path = tmp_path / "merged.pdf"
    doc.save(path)
    # The heading line is set larger than the body it introduces.
    doc2 = fitz.open()
    page2 = doc2.new_page(width=600, height=800)
    page2.insert_textbox(fitz.Rect(40, 100, 520, 140),
                         "What powers surface currents?", fontsize=17)
    page2.insert_textbox(fitz.Rect(40, 145, 520, 300),
                         "The sun heats air near the equator more than it heats "
                         "air near the poles, and that difference drives the "
                         "winds which drive the currents.", fontsize=11)
    path2 = tmp_path / "split.pdf"
    doc2.save(path2)

    parsed = _parse(fitz.open(path2)[0])
    kinds = [(b.type, b.text[:20]) for b in parsed.blocks]
    assert kinds[0][0] == "heading"
    assert kinds[1][0] == "body"
