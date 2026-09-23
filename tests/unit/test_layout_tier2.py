"""The layout-model tier: a magazine page read as regions, not as prose.

A 334-page illustrated science textbook went into the index as 545 chunks, 145
of them ending mid-sentence, because every page whose text layer returned
anything at all was read linearly — the router asked for a richer parser and
the extractor overruled it. This tier is what the router was asking for.
"""

from __future__ import annotations

import io

import pytest

from ingestion.layout import tier2
from ingestion.layout.blocks import PageMeta


class FakePipeline:
    """Stands in for PaddleOCR-VL: replays one page of regions."""

    def __init__(self, result=None, explode=False):
        self.result = result
        self.explode = explode
        self.calls = 0

    def predict(self, _image):
        self.calls += 1
        if self.explode:
            raise RuntimeError("model unavailable")
        return [self.result] if self.result is not None else []


def _meta(**over) -> PageMeta:
    fields = {"doc_id": "book", "page": 17, "width": 612.0, "height": 792.0}
    fields.update(over)
    return PageMeta(**fields)


def _page_image() -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (8, 10), "white").save(buffer, format="PNG")
    return buffer.getvalue()


def _result(regions, width=1224, height=1584) -> dict:
    return {"width": width, "height": height, "parsing_res_list": regions}


def test_a_page_comes_back_as_typed_blocks_in_the_models_reading_order():
    """Drawing order opens page 17 with the take-home box from the corner. The
    model reports the order a human reads, and it is kept."""
    pipeline = FakePipeline(_result([
        {"block_label": "doc_title", "block_content": "Ocean Waves",
         "block_bbox": [0, 0, 1224, 100], "block_order": 0},
        {"block_label": "text", "block_content": "Waves carry energy, not water.",
         "block_bbox": [0, 100, 600, 400], "block_order": 1},
        {"block_label": "aside_text", "block_content": "Take home: energy moves.",
         "block_bbox": [620, 100, 1224, 400], "block_order": 2},
    ]))

    page = tier2.parse_page(_page_image(), _meta(), config=None, pipeline=pipeline)

    assert [b.type for b in page.blocks] == ["heading", "body", "sidebar"]
    assert [b.reading_order for b in page.blocks] == [0, 1, 2]
    assert page.blocks[0].text == "Ocean Waves"
    assert all(b.source_tier == tier2.TIER for b in page.blocks)
    assert page.meta.source_tier == tier2.TIER


def test_boxes_arrive_in_page_points_not_render_pixels():
    """The page is rendered at 300 dpi to be read; a box in those pixels puts
    every block off the page, and the chunker decides by geometry."""
    pipeline = FakePipeline(_result([
        {"block_label": "text", "block_content": "half the page",
         "block_bbox": [0, 0, 612, 792], "block_order": 0}]))

    page = tier2.parse_page(_page_image(), _meta(), config=None, pipeline=pipeline)

    assert page.blocks[0].bbox == (0.0, 0.0, 306.0, 396.0)


def test_page_furniture_keeps_its_own_type_so_it_never_reaches_a_chunk():
    """The copyright line was welded into 25 chunks of 25 before blocks had types."""
    pipeline = FakePipeline(_result([
        {"block_label": "header", "block_content": "Unit 3 · Earth's Water",
         "block_bbox": [0, 0, 1224, 40], "block_order": 0},
        {"block_label": "number", "block_content": "217",
         "block_bbox": [1100, 1540, 1224, 1584], "block_order": 1},
        {"block_label": "text", "block_content": "A watershed drains to one river.",
         "block_bbox": [0, 60, 1224, 900], "block_order": 2},
    ]))

    page = tier2.parse_page(_page_image(), _meta(), config=None, pipeline=pipeline)

    assert [b.type for b in page.blocks] == ["header", "page_number", "body"]
    assert [b.text for b in page.content] == ["A watershed drains to one river."]


def test_what_the_model_read_inside_a_diagram_is_kept_as_text():
    """The thing a magazine page hides from a text layer: the words are inside
    the picture. A caption that says 'an illustration of a pencil' is not it."""
    pipeline = FakePipeline(_result([
        {"block_label": "chart", "block_order": 0, "block_bbox": [0, 0, 1224, 800],
         "block_content": "| Layer | Height |\n| Troposphere | 0-12 km |"},
    ]))

    page = tier2.parse_page(_page_image(), _meta(), config=None, pipeline=pipeline)

    assert page.blocks[0].type == "figure"
    assert "Troposphere" in page.blocks[0].text
    assert page.meta.figure_fraction > 0


def test_an_unlabelled_region_is_body_text_rather_than_lost():
    """Losing a paragraph is worse than mistyping one."""
    pipeline = FakePipeline(_result([
        {"block_label": "something_new", "block_content": "Still a sentence.",
         "block_bbox": [0, 0, 1224, 200], "block_order": 0}]))

    page = tier2.parse_page(_page_image(), _meta(), config=None, pipeline=pipeline)
    assert page.blocks[0].type == "body"


def test_a_model_that_fails_hands_the_page_back_rather_than_emptying_it():
    """None is 'I could not read this', and the caller falls back. An empty page
    would be indistinguishable from a blank one and would be indexed as blank."""
    assert tier2.parse_page(_page_image(), _meta(), config=None,
                            pipeline=FakePipeline(explode=True)) is None
    assert tier2.parse_page(_page_image(), _meta(), config=None,
                            pipeline=FakePipeline()) is None


def test_a_failure_says_which_page_it_was(caplog):
    import logging

    with caplog.at_level(logging.ERROR, logger="ingestion.layout.tier2"):
        tier2.parse_page(_page_image(), _meta(page=17), config=None,
                         pipeline=FakePipeline(explode=True))
    record = next(r for r in caplog.records if r.name == "ingestion.layout.tier2")
    assert record.page == 17 and record.doc_id == "book"


def test_the_tier_is_off_until_a_deployment_turns_it_on():
    class _Config:
        def __init__(self, settings):
            self.settings = settings

        def get(self, key, default=None):
            return self.settings if key == "ingestion.layout.tier2" else default

    assert tier2.enabled(_Config({"enabled": True})) is True
    assert tier2.enabled(_Config({"enabled": False})) is False
    assert tier2.enabled(_Config(None)) is False


@pytest.mark.parametrize("raw", [None, "nonsense", [1, 2]])
def test_a_box_that_cannot_be_read_covers_the_page_rather_than_a_corner(raw):
    pipeline = FakePipeline(_result([
        {"block_label": "text", "block_content": "somewhere", "block_bbox": raw}]))
    page = tier2.parse_page(_page_image(), _meta(), config=None, pipeline=pipeline)
    assert page.blocks[0].bbox == (0.0, 0.0, 612.0, 792.0)
