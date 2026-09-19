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


# --- the page cache ---------------------------------------------------------

def test_a_cached_page_is_not_read_again(fake_postgres):
    """One vision call per page is the most expensive thing the pipeline does,
    and chunking is the thing most often changed. Coupling them means every
    adjustment to how blocks are grouped re-pays for reading every page."""
    from ingestion.layout.page_cache import PageCache
    from shared.config import Config

    stored = _page(4, [_block("Tides", "heading", y=50), _block(PROSE)])
    fake_postgres(rows=[(stored.model_dump(mode="json"),)])

    cache = PageCache(Config.load(), model="text-layer", prompt_version="abc123")
    got = cache.get("11111111-1111-1111-1111-111111111111", 4)

    assert got is not None
    assert got.meta.page == 4
    assert [b.text for b in got.blocks] == [b.text for b in stored.blocks]
    assert cache.report()["cache_hits"] == 1


def test_a_page_nobody_has_read_is_a_miss(fake_postgres):
    from ingestion.layout.page_cache import PageCache
    from shared.config import Config

    fake_postgres(rows=[])
    cache = PageCache(Config.load(), model="text-layer", prompt_version="abc123")

    assert cache.get("11111111-1111-1111-1111-111111111111", 4) is None
    assert cache.report() == {"cache_hits": 0, "cache_misses": 1, "cache_enabled": True}


def test_a_cache_that_cannot_be_reached_never_fails_the_ingest(fake_postgres, caplog):
    """A missing table costs money, not correctness. It is reported once, with
    what to do about it, rather than per page or not at all."""
    import logging

    from ingestion.layout.page_cache import PageCache
    from shared.config import Config

    fake_postgres(explode=True)
    cache = PageCache(Config.load(), model="text-layer", prompt_version="abc123")

    with caplog.at_level(logging.ERROR):
        assert cache.get("11111111-1111-1111-1111-111111111111", 4) is None
        assert cache.get("11111111-1111-1111-1111-111111111111", 5) is None

    assert cache.report()["cache_enabled"] is False
    # An outage is not a miss: reporting one would describe a working cache
    # that happened to be cold.
    assert cache.report()["cache_misses"] == 0
    assert sum(1 for r in caplog.records if "page cache unavailable" in r.message) == 1


def test_a_stored_page_that_no_longer_fits_the_schema_is_re_read(fake_postgres):
    """The shape changed under it. That is stale, not fatal."""
    from ingestion.layout.page_cache import PageCache
    from shared.config import Config

    fake_postgres(rows=[({"meta": {"doc_id": "d"}, "blocks": "not a list"},)])
    cache = PageCache(Config.load(), model="text-layer", prompt_version="abc123")

    assert cache.get("11111111-1111-1111-1111-111111111111", 4) is None
    assert cache.report()["cache_misses"] == 1


def test_changing_the_prompt_changes_what_the_cache_is_keyed_on():
    """A hand-maintained version string is a step someone forgets, and the cache
    then serves pages the previous prompt produced — the one failure a cache
    exists to be trusted not to have."""
    from shared.prompting import Audience, LanguageRule, Prompt

    def _prompt(text):
        return Prompt(name="p", template=text, audience=Audience.INTERNAL,
                      language_rule=LanguageRule.NONE)

    assert _prompt("read this page").fingerprint == _prompt("read this page").fingerprint
    assert _prompt("read this page").fingerprint != _prompt("read this page.").fingerprint


# --- reading a page with a vision model -------------------------------------

def test_a_reported_box_is_converted_from_fractions_to_page_points():
    """Models place boxes far more reliably in fractions of the page than in
    absolute points, and everything downstream measures points."""
    from ingestion.layout.tier3 import RawBlock, _to_block

    meta = PageMeta(doc_id="d", page=1, width=600, height=800)
    block = _to_block(RawBlock(text="x", bbox=(0.0, 0.25, 0.5, 0.5)), meta)
    assert block.bbox == (0.0, 200.0, 300.0, 400.0)


def test_a_reversed_box_is_read_rather_than_refused():
    from ingestion.layout.tier3 import RawBlock

    assert RawBlock(text="x", bbox=(0.9, 0.8, 0.1, 0.2)).bbox == (0.1, 0.2, 0.9, 0.8)


def test_a_block_the_model_mangled_costs_that_block_not_the_page():
    from ingestion.layout.tier3 import LayoutTranscription

    page = LayoutTranscription.model_validate({
        "page_layout": "standard",
        "blocks": [{"text": "real", "bbox": [0, 0, 1, 0.2]},
                   {"bbox": "nonsense"},
                   {"text": "   ", "bbox": [0, 0, 1, 1]}],
    })
    assert [b.text for b in page.blocks] == ["real"]


# --- what a confidently wrong transcription looks like ----------------------

def _parsed(text: str) -> ParsedPage:
    return _page(1, [_block(text)])


def test_a_page_of_ink_that_came_back_almost_empty_is_flagged():
    """The output parses perfectly and the page is gone."""
    from ingestion.layout.tier3 import problems

    assert "short_output" in problems(_parsed("a few words"), ink=0.40)
    assert problems(_parsed("a few words"), ink=0.001) == []


def test_latin_letters_welded_into_arabic_words_are_flagged():
    """Script drift: the model has left the script it was reading."""
    from ingestion.layout.tier3 import problems

    assert "script_drift" in problems(_parsed("جابرd بن حيان"), ink=0.01)
    assert "script_drift" not in problems(_parsed("جابر بن حيان (Jabir)"), ink=0.01)


def test_a_phrase_repeated_until_the_budget_ran_out_is_flagged():
    """The classic collapse: it finds a phrase it likes and emits it forever."""
    from ingestion.layout.tier3 import problems

    loop = "the sun heats the air near the equator " * 8
    assert "repetition" in problems(_parsed(loop), ink=0.20)


def test_a_page_that_could_not_be_read_at_all_says_so():
    from ingestion.layout.tier3 import problems

    assert problems(None, ink=0.3) == ["unreadable"]
