"""Reading a book's pages into the page cache before the pipeline walks them.

The extract stage reads a scanned page twice — once for its layout, once for its
words — in two passes over the book, rendering the page again for each pass on
the one thread that owns the document. A scanned 1033 x 1375 pt page takes
0.83 s to render at 300 dpi on a 2-vCPU machine, so a 306-page book spent eight
and a half minutes of one core rendering — each page twice — while the other
core idled and the calls waited to be handed out.

This renders each page once, on every core, and makes both calls from that one
image, many pages in flight. What it reads goes into ``ai.page_transcriptions``
under exactly the keys the extract stage looks up, so the stage that follows
finds every page already read and only cuts figures and chunks.

It also takes a subset of the pages, which is what lets several machines read
one book at once: each reads its share into the same table, and the run that
builds the index afterwards reads none.

It decides nothing about a page itself. Which calls a page needs comes from the
extract stage's own helpers, so the two cannot disagree about it.
"""

from __future__ import annotations

import concurrent.futures as futures
import logging
import threading
import time
from collections.abc import Iterable
from importlib import import_module
from typing import Any, Optional

from .context import PipelineContext
from .layout import tier3
from .layout.page_cache import PageCache
from .state import IngestionState
from .utils import pdf
from .utils.render_pool import PageRenderer, default_processes

logger = logging.getLogger(__name__)

# The module, not the stage function ``ingestion.stages`` exports under the same
# name, and looked up at call time: the decisions read here are the extract
# stage's own, whatever that stage currently holds.
_extract = import_module(".stages.extract", __package__)

PROGRESS_EVERY = 25


class ReadAheadUnavailable(RuntimeError):
    """Reading ahead would pay for pages that nothing could keep."""


def read_ahead(
    source_path: str,
    state: IngestionState,
    ctx: PipelineContext,
    *,
    pages: Optional[Iterable[int]] = None,
    workers: Optional[int] = None,
    processes: Optional[int] = None,
) -> dict:
    """Read ``pages`` of the book (1-based; all when ``None``) into the cache.

    Returns what it did. Raises ``ReadAheadUnavailable`` when the cache cannot
    be written or no model is configured: a read that is thrown away is the
    full price of the book for nothing, and the caller is better off letting the
    pipeline read the pages itself.
    """
    started = time.monotonic()
    cfg = ctx.config
    ing = cfg.ingestion
    ocr = ing.get("ocr", {})
    layout_cfg = ing.get("layout", {})
    layout_on = bool(layout_cfg.get("enabled"))
    vision_on = ocr.get("engine") == "vision"
    render_dpi = ocr.get("render_dpi", 300)
    workers = workers or cfg.get("ingestion.ocr.read_ahead.workers")
    processes = default_processes(
        processes if processes is not None
        else cfg.get("ingestion.ocr.read_ahead.render_processes"))
    lecture_id = state["lecture_id"]

    if not (layout_on or vision_on):
        return _report(started, pages=0, workers=workers, processes=processes)
    if ctx.vision.is_stub:
        raise ReadAheadUnavailable(
            "no vision credentials: the stub reader's output must never be cached")

    layout_meta, routes, numbers = _plan(source_path, state, cfg, pages,
                                         layout_on=layout_on, vision_on=vision_on)

    layout_cache = _extract._page_cache(cfg, layout_cfg) if layout_meta else None
    if layout_meta and layout_cache is None:
        raise ReadAheadUnavailable("ingestion.layout.cache is off: a layout read "
                                   "made now would be made again by the pipeline")
    have_layout = (layout_cache.get_many(lecture_id, layout_meta)
                   if layout_cache is not None else {})
    have_words = _extract.cached_transcriptions(cfg, lecture_id, routes)
    _require_writable(layout_cache, *_transcription_caches(cfg, routes))

    todo = sorted({no for no in layout_meta if no not in have_layout}
                  | {no for no in routes if no not in have_words})
    already = len(have_layout) + len(have_words)
    logger.info("reading ahead", extra={
        "lecture_id": lecture_id, "pages": len(numbers), "to_read": len(todo),
        "calls_already_paid_for": already, "workers": workers,
        "render_processes": processes})

    tally: dict[str, int] = {}
    lock = threading.Lock()
    done = [0]

    def count(key: str) -> None:
        with lock:
            tally[key] = tally.get(key, 0) + 1

    def work(page_no: int) -> None:
        try:
            png = renderer.render(page_no)
            count("rendered")
            meta = layout_meta.get(page_no)
            if meta is not None and page_no not in have_layout:
                parsed = tier3.parse_page(
                    png, meta, config=cfg, language_hint=state.get("language_hint"),
                    direction=meta.reading_direction)
                if parsed is None:
                    count("layout_unread")
                else:
                    layout_cache.put(lecture_id, page_no, parsed, dpi=render_dpi)
                    count("layout_read")
            route = routes.get(page_no)
            if route is not None and page_no not in have_words:
                math = route == "math"
                # No hint unless one was given: the book's language is decided
                # afterwards from what these reads return, and the prompt takes
                # an empty hint as "read whatever is there".
                result = ctx.vision.transcribe_page(
                    png, language_hint=state.get("language_hint") or "", math=math)
                if result.unreadable:
                    # Not stored, exactly as the extract stage does not store
                    # it: a quota rejection and a ruined scan look identical
                    # here, and only one of them reads on the next attempt.
                    count("unreadable")
                    if result.answered:
                        count("judged_unreadable")
                else:
                    _extract._transcription_cache(cfg, math).put(
                        lecture_id, page_no, result, dpi=render_dpi)
                    count("transcribed")
        except Exception:      # noqa: BLE001 — one page must not cost the others
            logger.exception("a page could not be read ahead; the pipeline will "
                             "read it itself", extra={"lecture_id": lecture_id,
                                                      "page": page_no})
            count("failed")
        with lock:
            done[0] += 1
            if done[0] % PROGRESS_EVERY == 0 or done[0] == len(todo):
                logger.info("read-ahead progress", extra={
                    "lecture_id": lecture_id, "done": done[0], "of": len(todo),
                    "seconds": round(time.monotonic() - started, 1)})

    if todo:
        with PageRenderer(source_path, dpi=render_dpi, processes=processes) as renderer:
            with futures.ThreadPoolExecutor(max_workers=workers,
                                            thread_name_prefix="read-ahead") as pool:
                list(pool.map(work, todo))

    # A cache that went down mid-run kept nothing it was handed after that.
    _require_writable(layout_cache, *_transcription_caches(cfg, routes))
    return _report(started, pages=len(numbers), workers=workers, processes=processes,
                   to_read=len(todo), already_paid_for=already, **tally)


def _plan(
    source_path: str, state: IngestionState, cfg: Any, pages: Optional[Iterable[int]],
    *, layout_on: bool, vision_on: bool,
) -> tuple[dict[int, Any], dict[int, str], list[int]]:
    """Which model calls each page needs, from its text layer alone."""
    ing = cfg.ingestion
    ocr = ing.get("ocr", {})
    layout_cfg = ing.get("layout", {})
    min_chars = ocr.get("min_chars_per_page", 40)
    garbage_cfg = ocr.get("garbage", {})
    is_math = bool(state.get("subject_is_math", False))

    layout_meta: dict[int, Any] = {}
    routes: dict[int, str] = {}
    doc = pdf.open_document(source_path)
    try:
        total = len(doc)
        numbers = sorted({int(n) for n in pages}) if pages is not None else list(
            range(1, total + 1))
        outside = [n for n in numbers if not 1 <= n <= total]
        if outside:
            raise ValueError(f"pages {outside[:5]} are not in a {total}-page book")
        for page_no in numbers:
            page = doc[page_no - 1]
            chars = pdf.page_char_count(page)
            if layout_on:
                meta, blocks, wanted = _extract._layout_plan(
                    page, page_no, state, layout_cfg, chars)
                if _extract.layout_needs_model(blocks, wanted):
                    layout_meta[page_no] = meta
            if vision_on:
                route = _extract.transcription_route(
                    page, chars, min_chars=min_chars, is_math=is_math,
                    garbage_cfg=garbage_cfg)
                if route:
                    routes[page_no] = route
    finally:
        doc.close()
    return layout_meta, routes, numbers


def _transcription_caches(cfg: Any, routes: dict[int, str]) -> list[Optional[PageCache]]:
    kinds = {route == "math" for route in routes.values()}
    return [_extract._transcription_cache(cfg, math) for math in sorted(kinds)]


def _require_writable(*caches: Optional[PageCache]) -> None:
    for cache in caches:
        if cache is not None and not cache.enabled:
            raise ReadAheadUnavailable(
                "the page cache cannot be reached, so nothing read now would be kept")


def _report(started: float, **fields) -> dict:
    fields["seconds"] = round(time.monotonic() - started, 1)
    return fields


def pages_of_shard(total: int, shard: int, of: int) -> list[int]:
    """The 1-based pages one of ``of`` machines reads.

    Interleaved rather than in blocks: a textbook's front matter and its answer
    pages are light and its exercise spreads are heavy, and a block of either
    leaves one machine finished while another is still reading.
    """
    if of < 1 or not 0 <= shard < of:
        raise ValueError(f"shard {shard} of {of} does not exist")
    return list(range(shard + 1, total + 1, of))
