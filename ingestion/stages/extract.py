"""extract stage — text + figures from one lesson PDF, routed per page.

  - digital page: PyMuPDF text; embedded images -> crop -> S3 -> caption.
  - scanned page (< ocr.min_chars_per_page): render + vision (text + figure boxes).
  - maths subject: every page rendered + vision (maths prompt), PyMuPDF text dropped.

NFKC + tatweel normalisation runs on all extracted text. An unreadable page flags
the lesson needs_review and is not indexed.
"""

from __future__ import annotations

import logging
from collections import Counter
from functools import lru_cache
from typing import Any, Optional

from shared.language import LanguagePolicy
from shared.language.script import reading_direction

from ..context import PipelineContext
from ..schemas import FigureCaption
from ..state import Element, Figure, IngestionState, PageResult

from ..utils import is_garbage, normalise_text, pdf
from ..utils.richtext import to_blocks
from ..utils.s3 import StorageError, figure_key, figure_prefix
from ..layout import (
    LAYOUT_MODEL,
    TEXT_LAYER,
    VISION,
    PageMeta,
    ParsedPage,
    choose_tier,
    parse_page,
    tier2,
    tier3,
)
from ..layout.chunker import content_blocks
from ..concurrency import DEFAULT_WORKERS, in_order, streamed
from ..layout.page_cache import PageCache
from ..prompts import MATH_PAGE_TRANSCRIPTION_PROMPT, PAGE_TRANSCRIPTION_PROMPT
from ..prompts.page_layout import PAGE_LAYOUT_PROMPT
from ..schemas import PageTranscription
from shared.source import materialise

logger = logging.getLogger(__name__)


@lru_cache(maxsize=4)
def _policy(supported: tuple[str, ...], default: str) -> LanguagePolicy:
    return LanguagePolicy(supported=supported, default=default)


def run(state: IngestionState, ctx: PipelineContext) -> IngestionState:
    """Extract one lesson from its body text, or from its file once fetched."""
    if state.get("source_text"):
        return _extract_body_text(state, ctx)

    with materialise(state["source_uri"], ctx.config) as source_path:
        doc = pdf.open_document(source_path)
        try:
            return _extract(state, ctx, doc)
        finally:
            doc.close()


def _extract(state: IngestionState, ctx: PipelineContext, doc: Any) -> IngestionState:
    cfg = ctx.config
    ing = cfg.ingestion
    ocr = ing.get("ocr", {})
    norm = ing.get("normalise", {})
    supported = tuple(cfg.get("language.supported", ["ar", "en", "fr", "de"]))
    default_lang = state.get("language_hint") or cfg.get("language.default", "en")

    min_chars = ocr.get("min_chars_per_page", 40)
    render_dpi = ocr.get("render_dpi", 300)
    vision_enabled = ocr.get("engine") == "vision"

    # Re-ingest housekeeping: drop this lecture's prior figure objects so the run
    # leaves exactly the figures it produces. Keys are content-addressed, so a
    # re-uploaded identical crop overwrites; only crops whose bytes changed would
    # otherwise orphan. (The index stage replaces DB rows/points but not S3.)
    _clear_previous_figures(ctx, state["lecture_id"])

    # Pass 1: gather digital text blocks so heading detection has a document-wide
    # body font size to compare against (a single big glyph shouldn't define it).
    page_blocks: dict[int, list[pdf.Block]] = {}
    for page_no, page in pdf.iter_pages(doc):
        if pdf.page_char_count(page) >= min_chars:
            page_blocks[page_no] = pdf.extract_blocks(page, page_no)
    body_size = _body_font_size(page_blocks)

    layout_cfg = ing.get("layout", {})
    layout_on = bool(layout_cfg.get("enabled"))
    parsed_pages: list[dict] = []
    cache = _page_cache(cfg, layout_cfg) if layout_on else None

    pages: list[PageResult] = []
    elements: list[Element] = []
    figures: list[Figure] = []
    counts = {
        "digital": 0, "scanned": 0, "math": 0, "garbage": 0,
        "figures": 0, "stub_vision": 0, "unreadable": 0, "caption_failed": 0,
    }
    garbage_cfg = ocr.get("garbage", {})
    is_math = state.get("subject_is_math", False)

    # Read every page that needs a model before walking them, several at a
    # time. A page is seconds of arithmetic and a minute of waiting on HTTP,
    # and Fargate bills for the container existing rather than working — so
    # one page at a time meant a book held a machine for ten hours to do
    # twenty minutes of work.
    transcribed: dict[int, Any] = {}
    if vision_enabled:
        transcribed = _prefetch_transcriptions(
            doc, state, ctx, min_chars=min_chars, render_dpi=render_dpi,
            is_math=is_math, garbage_cfg=garbage_cfg, default_lang=default_lang,
            workers=ocr.get("page_workers", DEFAULT_WORKERS))

    for page_no, page in pdf.iter_pages(doc):
        char_count = pdf.page_char_count(page)

        if layout_on:
            parsed = _parse_layout(
                page, page_no, state, layout_cfg, char_count, counts, cache,
                render_dpi=render_dpi, config=cfg)
            if parsed is not None:
                parsed_pages.append(parsed)

        # --- scanned page: one vision call -> text + figure boxes -----------
        if char_count < min_chars:
            counts["scanned"] += 1
            page_elems: list[Element] = []
            page_figs: list[Figure] = []
            if vision_enabled:
                page_elems, page_figs = _vision_transcribe(
                    state, ctx, page, page_no, default_lang, "scanned",
                    render_dpi, norm, supported, counts,
                    prefetched=transcribed.get(page_no),
                )
            figures.extend(page_figs)
            elements.extend(page_elems)
            pages.append(
                PageResult(
                    number=page_no,
                    route="scanned",
                    language=default_lang,
                    char_count=char_count,
                    elements=page_elems,
                )
            )
            continue

        # --- born-digital garbage page: glyph-noise text (subset Identity-H, no
        # ToUnicode) is unrecoverable, but the embedded images are intact. Transcribe
        # the TEXT via vision; take FIGURES from get_images (full-fidelity embedded
        # rasters, unaffected by the text-encoding garbage) — not vision boxes.
        if not is_math and garbage_cfg and is_garbage(pdf.page_text(page), garbage_cfg):
            counts["garbage"] += 1
            page_elems = []
            if vision_enabled:
                page_elems, _ = _vision_transcribe(
                    state, ctx, page, page_no, default_lang, "garbage",
                    render_dpi, norm, supported, counts, with_figures=False,
                    prefetched=transcribed.get(page_no),
                )
            else:
                counts["unreadable"] += 1
            page_figs = _digital_figures(state, ctx, doc, page, page_no, ing, default_lang,
                                         counts)
            figures.extend(page_figs)
            counts["figures"] += len(page_figs)
            elements.extend(page_elems)
            pages.append(
                PageResult(
                    number=page_no,
                    route="garbage",
                    language=default_lang,
                    char_count=char_count,
                    elements=page_elems,
                )
            )
            continue

        # --- born-digital page ---------------------------------------------
        blocks = page_blocks.get(page_no, [])
        page_lang = _language(
            " ".join(b.text for b in blocks[:5]), default_lang, supported
        )

        # Maths subjects route every page through vision (render + transcribe),
        # discarding the PyMuPDF text: plain extraction flattens notation
        # (8/12 -> "8" then "12") and would index the corrupted version.
        math_png = None
        math_boxes = None
        math_png = None
        if is_math:
            route = "math"
            counts["math"] += 1
            page_elems = []
            if vision_enabled:
                result = transcribed.get(page_no)
                if result is None:
                    math_png = pdf.render_page_png(page, dpi=render_dpi)
                    result = ctx.vision.transcribe_page(
                        math_png, language_hint=page_lang, math=True)
                if ctx.vision.is_stub:
                    counts["stub_vision"] += 1
                if result.unreadable:
                    counts["unreadable"] += 1
                else:
                    text = _norm(result.text, norm)
                    if text:
                        page_elems.append(
                            Element(text=text, page=page_no, route="math", language=page_lang)
                        )
                    math_boxes = result.figures  # kept for the hybrid fill below
        else:
            route = "digital"
            counts["digital"] += 1
            page_elems = _digital_elements(blocks, body_size, ing, page_lang, norm)

        # figures: embedded rasters (get_images), deduped
        page_figs = _digital_figures(state, ctx, doc, page, page_no, ing, page_lang, counts)
        # math hybrid: add vision-detected figures get_images missed (vector geometry
        # diagrams have no embedded raster), deduped against the get_images figures.
        if math_boxes and ing.get("figures", {}).get("vision_fill_math", True):
            # Rendered here rather than carried from the prefetch, for the same
            # reason as the scanned path: one image per page held for the length
            # of the run is what ran the task out of memory.
            if math_png is None:
                math_png = pdf.render_page_png(page, dpi=render_dpi)
            page_figs += _vision_fill_figures(
                state, ctx, math_png, page_no, math_boxes, page_figs,
                render_dpi, page_lang, ing,
            )
        figures.extend(page_figs)
        counts["figures"] += len(page_figs)

        elements.extend(page_elems)
        pages.append(
            PageResult(
                number=page_no,
                route=route,
                language=page_lang,
                char_count=char_count,
                elements=page_elems,
            )
        )

    state["pages"] = pages
    state["elements"] = elements
    state["figures"] = figures
    if layout_on:
        state["parsed_pages"] = parsed_pages
        if cache is not None:
            counts.update(cache.report())
    state.setdefault("stats", {})["extract"] = {
        "pages": len(pages),
        **counts,
        "elements": len(elements),
    }
    state["stats"]["extract"]["ingestion_status"] = state.get("ingestion_status")
    if counts["stub_vision"]:
        state.setdefault("errors", []).append(
            f"{counts['stub_vision']} page(s) used STUB vision output — not real "
            "transcription; re-run once the vision key is configured."
        )
    state["ingestion_status"] = review_status(
        unreadable=counts["unreadable"], pages=len(pages),
        ratio=ing.get("needs_review_ratio"))
    state["stats"]["extract"]["ingestion_status"] = state["ingestion_status"]
    if counts["unreadable"]:
        state.setdefault("errors", []).append(
            f"{counts['unreadable']} of {len(pages)} page(s) were unreadable; lesson "
            f"is {state['ingestion_status']}."
        )
    return state


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _extract_body_text(state: IngestionState, ctx: PipelineContext) -> IngestionState:
    """Lessons authored in the platform editor: the body text is the whole source."""
    cfg = ctx.config
    norm = cfg.ingestion.get("normalise", {})
    supported = tuple(cfg.get("language.supported", ["ar", "en", "fr", "de"]))
    default_lang = state.get("language_hint") or cfg.get("language.default", "en")

    blocks = to_blocks(state["source_text"] or "")
    elements = [
        Element(
            text=text,
            page=1,
            is_heading=block.is_heading,
            level=block.level,
            route="text",
            language=_language(text, default_lang, supported),
        )
        for block in blocks
        if (text := _norm(block.text, norm))
    ]

    language = elements[0].language if elements else default_lang
    char_count = sum(len(el.text) for el in elements)
    state["pages"] = [
        PageResult(
            number=1,
            route="text",
            language=language or default_lang,
            char_count=char_count,
            elements=elements,
        )
    ]
    state["elements"] = elements
    state["figures"] = []
    state.setdefault("stats", {})["extract"] = {
        "pages": 1,
        "text": 1,
        "elements": len(elements),
        "characters": char_count,
        "ingestion_status": state.get("ingestion_status"),
    }
    if not elements:
        state["ingestion_status"] = "needs_review"
        state.setdefault("errors", []).append(
            "lesson body text held no readable content after parsing"
        )
    logger.info(
        "extracted body text",
        extra={"lecture_id": state["lecture_id"], "elements": len(elements)},
    )
    return state



def review_status(*, unreadable: int, pages: int, ratio: float) -> str:
    """Whether a lesson is fit to answer from, or has to be looked at first.

    Proportional, because any-page-failed does not scale. A 371-page textbook
    read into 545 good chunks answered nothing at all: one page came back
    unreadable, the lesson was held back, and retrieval only reads lectures
    marked ready. At that length one bad page is near-certain, so the rule
    quarantined every long document it was given while letting a three-page
    lesson through on the same evidence.

    A share also needs no special case for short lessons: one page of three is a
    third of the lesson and fails the same test.
    """
    if not pages:
        return "needs_review"
    return "needs_review" if unreadable / pages >= ratio else "ready"


def _page_cache(cfg: Any, layout_cfg: dict) -> Optional[PageCache]:
    """The cache for the pages this run has to pay a model to read.

    Keyed on the model that read the page and on the prompt's own digest, so
    changing either sends the page back to a model without anyone having to
    remember to invalidate anything. Text-layer pages are never stored: they
    cost nothing to parse, and a cache of free work is a table no one saves
    anything by reading.
    """
    if not layout_cfg.get("cache", True):
        return None
    return PageCache(cfg, model=cfg.get("vision.model"),
                     prompt_version=PAGE_LAYOUT_PROMPT.fingerprint)


@lru_cache(maxsize=8)
def _transcription_cache(config: Any, math: bool):
    """The cache for the page a model was paid to read word by word.

    This is the most expensive call in the pipeline and it was the only one
    without a cache: the layout parse of a page was stored and its
    transcription was not, so a run that died on page 300 of 418 — or was
    stopped to pick up a config change — paid for all 300 again. Three external
    books were re-read twice for exactly that reason.

    Keyed per prompt, not per document: a maths book holds scanned pages read
    with the ordinary prompt and text-layer pages read with the maths one, and a
    page stored under the wrong prompt is served to the run that asked for the
    other.
    """
    prompt = MATH_PAGE_TRANSCRIPTION_PROMPT if math else PAGE_TRANSCRIPTION_PROMPT
    return PageCache(config, model=config.get("vision.model"),
                     prompt_version=prompt.fingerprint, shape=PageTranscription)


@lru_cache(maxsize=4)
def _layout_model_cache(config: Any):
    """The cache for pages the layout model read.

    Its own entry per page, keyed on the layout model rather than the vision
    one: the two read the same page differently, and a parse stored under the
    wrong reader is served to a run that asked for the other.
    """
    settings = (config.get("ingestion.layout.tier2") if config else None) or {}
    if not settings.get("cache", True):
        return None
    version = str(settings.get("version", "v1.6"))
    return PageCache(config, model=f"paddleocr-vl:{version}", prompt_version=version)


def _page_direction(page: Any, layout_cfg: dict) -> str:
    """Which side this page's columns are read from.

    Read off the page itself, because one direction for a whole corpus reads
    every Latin, Cyrillic and CJK page backwards the moment the corpus stops
    being Arabic — and a single document can hold both. ``detect_direction:
    false`` hands the decision back to whoever set it.
    """
    if layout_cfg.get("detect_direction", True):
        return reading_direction(pdf.page_text(page))
    return layout_cfg.get("reading_direction", "ltr")


def _parse_layout(
    page: Any,
    page_no: int,
    state: IngestionState,
    layout_cfg: dict,
    char_count: int,
    counts: dict,
    cache: Optional[PageCache],
    *,
    render_dpi: int,
    config: Any,
) -> Optional[dict]:
    """This page as typed blocks in reading order, or ``None`` when no tier can.

    The router decides per page rather than per document: one PDF holds clean
    text-layer pages, magazine spreads whose drawing order is not their reading
    order, and scans with no text at all.
    """
    meta = PageMeta(
        doc_id=state["lecture_id"],
        page=page_no,
        width=float(page.rect.width),
        height=float(page.rect.height),
        reading_direction=_page_direction(page, layout_cfg),
    )
    blocks = parse_page(page, meta, direction=meta.reading_direction).blocks
    wanted = choose_tier(page, char_count=char_count, blocks=blocks)
    counts[f"layout_wants_tier{wanted}"] = counts.get(f"layout_wants_tier{wanted}", 0) + 1

    if wanted == TEXT_LAYER:
        return ParsedPage(meta=meta, blocks=blocks).model_dump()

    # A page the router sent here has a text layer that cannot be trusted with
    # it: scattered boxes whose drawing order is not their reading order, or no
    # text at all. The layout model reads the page from pixels, so neither
    # hurts it; the vision model is what answers a page it cannot read, and a
    # page with no text layer at all has always gone straight there.
    if wanted == LAYOUT_MODEL and tier2.enabled(config):
        read = _read_page_with_layout_model(
            page, page_no, state, meta, render_dpi, config, counts)
        if read is not None:
            return read.model_dump()

    if not content_blocks(blocks) or wanted == VISION:
        read = _read_page_with_vision(
            page, page_no, state, meta, render_dpi, config, cache, counts)
        if read is not None:
            return read.model_dump()
        if not blocks:
            # Nothing the text layer can see and nothing a model could read:
            # leave the page to the older extract path rather than index it empty.
            return None

    counts["layout_downgraded"] = counts.get("layout_downgraded", 0) + 1
    logger.info("page read by the text layer though it wants a richer parser",
                extra={"lecture_id": state["lecture_id"], "page": page_no,
                       "wanted_tier": wanted})
    return ParsedPage(meta=meta, blocks=blocks).model_dump()


def _read_page_with_layout_model(
    page: Any,
    page_no: int,
    state: IngestionState,
    meta: PageMeta,
    render_dpi: int,
    config: Any,
    counts: dict,
) -> Optional[ParsedPage]:
    """The page as the layout model read it, from the cache when it has been read."""
    layout_cache = _layout_model_cache(config)
    if layout_cache is not None:
        stored = layout_cache.get(state["lecture_id"], page_no)
        if stored is not None:
            return stored

    parsed = tier2.parse_page(
        pdf.render_page_png(page, dpi=render_dpi), meta, config=config)
    if parsed is None:
        counts["layout_model_unread"] = counts.get("layout_model_unread", 0) + 1
        return None

    counts["layout_model_read"] = counts.get("layout_model_read", 0) + 1
    if layout_cache is not None:
        layout_cache.put(state["lecture_id"], page_no, parsed, dpi=render_dpi)
    return parsed


def _prefetch_transcriptions(
    doc: Any, state: IngestionState, ctx: PipelineContext, *,
    min_chars: int, render_dpi: int, is_math: bool, garbage_cfg: dict,
    default_lang: str, workers: int,
) -> dict[int, tuple[bytes, Any]]:
    """Every page that needs a model, read ahead of the loop that consumes it.

    Returns ``{page_no: transcription}``. Rendering happens here, on one thread,
    because a PyMuPDF ``Document`` is not thread-safe; only the HTTP call is
    handed out.

    The rendered page is deliberately **not** kept. It used to be returned
    alongside the transcription so the loop could crop figures out of it without
    rendering twice, which held every page of the book — about a gigabyte at 300
    dpi for four hundred pages — until the run ended. The embedder then loaded
    its 2.3 GB on top, inside an 8 GB task, and the kernel killed the run: two
    books died that way on 5 October after being read in full, one at 740 chunks
    and one at 1,269, writing nothing. Re-rendering a page costs a fraction of a
    second and only happens for pages that actually have a figure to crop.
    """
    wanted: list[tuple[int, Any, str]] = []
    for page_no, page in pdf.iter_pages(doc):
        char_count = pdf.page_char_count(page)
        if char_count < min_chars:
            wanted.append((page_no, page, "scanned"))
        elif is_math:
            wanted.append((page_no, page, "math"))
        elif garbage_cfg and is_garbage(pdf.page_text(page), garbage_cfg):
            wanted.append((page_no, page, "garbage"))
    if not wanted:
        return {}

    def render(item):
        no, page, route = item
        return no, pdf.render_page_png(page, dpi=render_dpi), route

    def read(item):
        no, png, route = item
        cache = _transcription_cache(ctx.config, route == "math")
        stored = cache.get(state["lecture_id"], no) if cache else None
        if stored is not None:
            return no, stored
        result = ctx.vision.transcribe_page(
            png, language_hint=default_lang, math=(route == "math"))
        # An unreadable page is not stored: it is the one answer worth paying
        # for again, because a quota rejection and a ruined scan reach here
        # looking identical and only one of them will read next time.
        if cache and not result.unreadable:
            cache.put(state["lecture_id"], no, result, dpi=render_dpi)
        return no, result

    out: dict[int, Any] = dict(
        streamed(wanted, render, read, workers=workers, label="transcribe"))
    logger.info("pages transcribed ahead of the loop",
                extra={"lecture_id": state["lecture_id"], "pages": len(out),
                       "workers": workers})
    return out


def _read_page_with_vision(
    page: Any,
    page_no: int,
    state: IngestionState,
    meta: PageMeta,
    render_dpi: int,
    config: Any,
    cache: Optional[PageCache],
    counts: dict,
) -> Optional[ParsedPage]:
    """The page as a model read it, from the cache when it has been read before."""
    if cache is not None:
        stored = cache.get(state["lecture_id"], page_no)
        if stored is not None:
            return stored

    parsed = tier3.parse_page(
        pdf.render_page_png(page, dpi=render_dpi), meta,
        config=config,
        language_hint=state.get("language_hint"),
        direction=meta.reading_direction,
    )
    if parsed is None:
        counts["layout_unread"] = counts.get("layout_unread", 0) + 1
        return None

    counts["layout_read_by_vision"] = counts.get("layout_read_by_vision", 0) + 1
    if cache is not None:
        cache.put(state["lecture_id"], page_no, parsed, dpi=render_dpi)
    return parsed


def _clear_previous_figures(ctx: PipelineContext, lecture_id: str) -> None:
    try:
        ctx.uploader.delete_prefix(figure_prefix(lecture_id))
    except StorageError:
        logger.warning(
            "previous figure crops could not be cleared; stale objects may remain",
            extra={"lecture_id": lecture_id},
            exc_info=True,
        )


def _vision_transcribe(
    state, ctx, page, page_no, lang, route,
    render_dpi, norm, supported, counts, with_figures=True, prefetched=None,
) -> tuple[list[Element], list[Figure]]:
    """Render a page and transcribe its text via vision. Figures come from the
    returned boxes only when ``with_figures`` (scanned pages); a garbage page passes
    False and takes figures from get_images instead. Caller guards on vision_enabled;
    an unreadable page flags needs_review and is not indexed."""
    # The prefetch has usually read this page already, several pages at a time
    # while this loop was still on an earlier one. Falling back to reading it
    # here keeps one path for a page the prefetch skipped.
    png: Optional[bytes] = None
    if prefetched is not None:
        result = prefetched
    else:
        png = pdf.render_page_png(page, dpi=render_dpi)
        result = ctx.vision.transcribe_page(png, language_hint=lang)
    if ctx.vision.is_stub:
        counts["stub_vision"] += 1
    if result.unreadable:
        counts["unreadable"] += 1
        state["ingestion_status"] = "needs_review"
        return [], []
    page_elems: list[Element] = []
    text = _norm(result.text, norm)
    if text:
        elem_lang = _language(text, lang, supported)
        page_elems.append(Element(text=text, page=page_no, route=route, language=elem_lang))
    page_figs: list[Figure] = []
    if with_figures and result.figures:
        # Rendered now rather than carried from the prefetch: holding one image
        # per page for the length of the run is what ran the task out of memory.
        if png is None:
            png = pdf.render_page_png(page, dpi=render_dpi)
        page_figs = _figures_from_boxes(state, ctx, png, page_no, result.figures, lang,
                                        counts)
        counts["figures"] += len(page_figs)
    return page_elems, page_figs


def _digital_elements(
    blocks: list[pdf.Block],
    body_size: float,
    ing: dict,
    page_lang: str,
    norm: dict,
) -> list[Element]:
    """Turn PyMuPDF blocks into headings/paragraphs with NFKC applied."""
    heading_ratio = 1.15
    strong_heading_ratio = 1.4
    max_heading_chars = 140
    out: list[Element] = []
    for blk in blocks:
        text = _norm(blk.text, norm)
        if not text:
            continue
        single_line = "\n" not in blk.text.strip()
        is_heading = (
            body_size > 0
            and blk.size >= body_size * heading_ratio
            and len(text) <= max_heading_chars
            and single_line
        )
        level = 0
        if is_heading:
            level = 1 if blk.size >= body_size * strong_heading_ratio else 2
        out.append(
            Element(
                text=text,
                page=blk.page,
                is_heading=is_heading,
                level=level,
                font_size=blk.size,
                route="digital",
                language=page_lang,
            )
        )
    return out


def _note_caption_failure(counts: dict | None) -> None:
    """A figure whose caption failed is still worth keeping, but not worth hiding:
    captions are what make a figure findable."""
    if counts is not None:
        counts["caption_failed"] = counts.get("caption_failed", 0) + 1


def _digital_figures(state, ctx, doc, page, page_no, ing, page_lang,
                     counts: dict | None = None) -> list[Figure]:
    """Extract, filter, dedupe, upload and caption embedded images on a page."""
    figs_cfg = ing.get("figures", {})
    if not figs_cfg.get("extract_embedded", True):
        return []

    images = pdf.extract_embedded_images(doc, page, page_no)
    page_area = float(page.rect.width * page.rect.height) or 1.0

    # Discard tiled scan fragments: many identically-sized images on one page.
    size_counts = Counter((im.width, im.height) for im in images)
    max_identical = figs_cfg.get("max_identical_sizes_per_page", 8)

    kept: list[pdf.EmbeddedImage] = []
    for im in images:
        if im.width < figs_cfg.get("min_width_px", 120):
            continue
        if im.height < figs_cfg.get("min_height_px", 120):
            continue
        if size_counts[(im.width, im.height)] > max_identical:
            continue
        x0, y0, x1, y1 = im.bbox
        if abs(x1 - x0) * abs(y1 - y0) / page_area > figs_cfg.get("max_page_area_ratio", 0.6):
            continue  # full-page scan, not a diagram
        kept.append(im)

    # Layered-duplicate dedupe: the same figure is often embedded twice at one spot.
    # Keep the higher-resolution copy of any group whose placement boxes overlap.
    overlap_cut = figs_cfg.get("dedupe_overlap", 0.7)
    survivors: list[pdf.EmbeddedImage] = []
    for im in sorted(kept, key=lambda i: i.width * i.height, reverse=True):
        if any(_bbox_overlap(im.bbox, s.bbox) >= overlap_cut for s in survivors):
            continue
        survivors.append(im)

    out: list[Figure] = []
    for im in survivors:
        key = figure_key(state["lecture_id"], page_no, im.image_bytes)
        s3_uri = ctx.uploader.upload(im.image_bytes, key, f"image/{im.ext}")
        cap = None
        alt = None
        if figs_cfg.get("generate_alt_text", True):
            caption = ctx.vision.caption_figure(im.image_bytes, language_hint=page_lang)
            if caption is None:
                _note_caption_failure(counts)
            else:
                cap, alt = caption.caption, caption.alt_text
        out.append(
            Figure(
                lecture_id=state["lecture_id"],
                page=page_no,
                s3_uri=s3_uri,
                bbox=tuple(im.bbox),
                caption=cap,
                alt_text=alt,
            )
        )
    return out


def _vision_fill_figures(state, ctx, page_png, page_no, boxes, existing,
                         render_dpi, lang, ing, counts: dict | None = None) -> list[Figure]:
    """Math hybrid: add vision-detected figures that get_images missed (vector
    diagrams). A box overlapping an existing get_images figure is skipped (get_images
    is higher fidelity). Kept boxes are cropped from the rendered page."""
    if not page_png or not boxes:
        return []
    from io import BytesIO

    from PIL import Image

    figs_cfg = ing.get("figures", {})
    overlap_cut = figs_cfg.get("dedupe_overlap", 0.7)
    with Image.open(BytesIO(page_png)) as img:
        width, height = img.width, img.height
    scale = render_dpi / 72.0  # get_images bbox is PDF points; render is at dpi
    existing_px = [tuple(v * scale for v in f.bbox) for f in existing]

    out: list[Figure] = []
    for box in boxes:
        vb = _to_pixels(box.bbox, width, height)
        if any(_bbox_overlap(vb, e) >= overlap_cut for e in existing_px):
            continue  # already have it from get_images
        crop = _crop_png_pixels(page_png, box.bbox)
        if crop is None:
            continue
        key = figure_key(state["lecture_id"], page_no, crop)
        s3_uri = ctx.uploader.upload(crop, key)
        cap = alt = None
        if figs_cfg.get("generate_alt_text", True):
            caption = ctx.vision.caption_figure(crop, language_hint=lang)
            if caption is None:
                _note_caption_failure(counts)
            else:
                cap, alt = caption.caption, caption.alt_text
        out.append(
            Figure(lecture_id=state["lecture_id"], page=page_no, s3_uri=s3_uri,
                   bbox=tuple(vb), caption=cap, alt_text=alt)
        )
    return out


def _bbox_overlap(a, b) -> float:
    """Intersection area over the smaller box's area (0..1). Robust to one box being
    slightly larger than the other, as with a layered duplicate image."""
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    if inter <= 0:
        return 0.0
    area_a = abs(a[2] - a[0]) * abs(a[3] - a[1])
    area_b = abs(b[2] - b[0]) * abs(b[3] - b[1])
    smaller = min(area_a, area_b)
    return inter / smaller if smaller > 0 else 0.0


def _figures_from_boxes(state, ctx, page_png, page_no, boxes, lang,
                        counts: dict | None = None) -> list[Figure]:
    """Figures on a scanned page: the page-vision call already returned the boxes
    (stub -> none). Crop each from the rendered page image (boxes are in that
    image's pixel space) at full resolution, upload, then caption the crop."""
    if not boxes:
        return []

    # Cropping is arithmetic on bytes already in memory and the upload and the
    # caption are both network, so the figures of a page go together. Measured
    # over twelve pages of a question bank: thirteen figures captioned one
    # after another cost 148s of a 219s run — two thirds of the time, for calls
    # that have nothing to do with each other.
    workers = ctx.config.get("ingestion.ocr.page_workers", DEFAULT_WORKERS)

    def build(box):
        crop = _crop_png_pixels(page_png, box.bbox)
        if crop is None:
            return None
        key = figure_key(state["lecture_id"], page_no, crop)
        s3_uri = ctx.uploader.upload(crop, key)
        # Caption the full-res crop (better than a region of the whole page).
        caption = ctx.vision.caption_figure(crop, language_hint=lang)
        if caption is None:
            _note_caption_failure(counts)
            caption = FigureCaption(caption="", alt_text="")
        return Figure(
            lecture_id=state["lecture_id"],
            page=page_no,
            s3_uri=s3_uri,
            bbox=tuple(box.bbox),
            caption=caption.caption,
            alt_text=caption.alt_text,
        )

    return [f for f in in_order(list(boxes), build, workers=workers,
                                label="caption") if f is not None]


def _to_pixels(bbox, width: int, height: int) -> tuple[float, float, float, float]:
    """Convert a figure box to pixel coordinates of the rendered page. The vision
    prompt asks for 0..1 fractions of page width/height; some models emit 0..1000
    instead, and a few return raw pixels. Detect which by magnitude and scale."""
    x0, y0, x1, y1 = (float(v) for v in bbox)
    m = max(x0, y0, x1, y1)
    if m <= 1.5:                                    # fractions of the page
        return x0 * width, y0 * height, x1 * width, y1 * height
    if m <= 1000.0 and (width > 1000 or height > 1000):  # 0..1000 normalised
        return (x0 / 1000 * width, y0 / 1000 * height,
                x1 / 1000 * width, y1 / 1000 * height)
    return x0, y0, x1, y1                            # already pixels


def _crop_png_pixels(page_png: bytes, bbox) -> bytes | None:
    """Crop a figure box out of a rendered page PNG, scaling normalised box
    coordinates to the image's pixel size first. Returns None for a degenerate or
    out-of-range box rather than raising."""
    from io import BytesIO

    from PIL import Image

    try:
        vals = [float(v) for v in bbox]
    except (TypeError, ValueError):
        return None
    if len(vals) != 4:
        return None
    with Image.open(BytesIO(page_png)) as img:
        x0, y0, x1, y1 = (int(round(v)) for v in _to_pixels(vals, img.width, img.height))
        if x1 <= x0 or y1 <= y0:
            return None
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(img.width, x1), min(img.height, y1)
        if x1 <= x0 or y1 <= y0:
            return None
        crop = img.crop((x0, y0, x1, y1))
        buf = BytesIO()
        crop.save(buf, "PNG")
        return buf.getvalue()


def _body_font_size(page_blocks: dict[int, list[pdf.Block]]) -> float:
    """Most common rounded block size across the document = body text size."""
    counter: Counter = Counter()
    for blocks in page_blocks.values():
        for blk in blocks:
            counter[round(blk.size * 2) / 2] += len(blk.text)  # weight by text length
    if not counter:
        return 0.0
    return counter.most_common(1)[0][0]


def _norm(text: str, norm: dict) -> str:
    return normalise_text(
        text,
        nfkc=norm.get("nfkc", True),
        strip_tatweel=norm.get("strip_tatweel", True),
        collapse_whitespace=norm.get("collapse_whitespace", True),
    )


def _language(text: str, default: str, supported: tuple[str, ...]) -> str:
    """The language of one extracted block, through the shared policy.

    Ingestion ran its own detector with a ten-character floor and no confidence
    check, so a heading could label a whole page. The resolver declines instead,
    and the block keeps the language it inherited.
    """
    return _policy(tuple(supported), default).resolve_document(text, hint=default).code
