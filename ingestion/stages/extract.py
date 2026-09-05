"""extract stage — text + figures from one lesson PDF, routed per page.

  - digital page: PyMuPDF text; embedded images -> crop -> S3 -> caption.
  - scanned page (< ocr.min_chars_per_page): render + vision (text + figure boxes).
  - maths subject: every page rendered + vision (maths prompt), PyMuPDF text dropped.

NFKC + tatweel normalisation runs on all extracted text. An unreadable page flags
the lesson needs_review and is not indexed.
"""

from __future__ import annotations

from collections import Counter

from ..context import PipelineContext
from ..state import Element, Figure, IngestionState, PageResult
from ..utils import is_garbage, normalise_text
from ..utils import pdf
from ..utils.s3 import figure_key

try:
    from langdetect import DetectorFactory, detect
    DetectorFactory.seed = 0  # deterministic language detection
except ImportError:  # pragma: no cover
    detect = None  # type: ignore[assignment]


def run(state: IngestionState, ctx: PipelineContext) -> IngestionState:
    cfg = ctx.config
    ing = cfg.ingestion
    ocr = ing.get("ocr", {})
    norm = ing.get("normalise", {})
    supported = tuple(cfg.get("language.supported", ["ar", "en"]))
    default_lang = state.get("language_hint") or cfg.get("language.default", "en")

    min_chars = ocr.get("min_chars_per_page", 40)
    render_dpi = ocr.get("render_dpi", 300)
    vision_enabled = ocr.get("engine") == "vision"

    source_path = _local_path(state["source_uri"])
    doc = pdf.open_document(source_path)

    # Re-ingest housekeeping: drop this lecture's prior figure objects so the run
    # leaves exactly the figures it produces. Keys are content-addressed, so a
    # re-uploaded identical crop overwrites; only crops whose bytes changed would
    # otherwise orphan. (The index stage replaces DB rows/points but not S3.)
    ctx.uploader.delete_prefix(f"figures/{state['lecture_id']}/")

    # Pass 1: gather digital text blocks so heading detection has a document-wide
    # body font size to compare against (a single big glyph shouldn't define it).
    page_blocks: dict[int, list[pdf.Block]] = {}
    for page_no, page in pdf.iter_pages(doc):
        if pdf.page_char_count(page) >= min_chars:
            page_blocks[page_no] = pdf.extract_blocks(page, page_no)
    body_size = _body_font_size(page_blocks)

    pages: list[PageResult] = []
    elements: list[Element] = []
    figures: list[Figure] = []
    counts = {
        "digital": 0, "scanned": 0, "math": 0, "garbage": 0,
        "figures": 0, "stub_vision": 0, "unreadable": 0,
    }
    garbage_cfg = ocr.get("garbage", {})

    for page_no, page in pdf.iter_pages(doc):
        char_count = pdf.page_char_count(page)

        # --- scanned page: one vision call -> text + figure boxes -----------
        if char_count < min_chars:
            counts["scanned"] += 1
            page_elems: list[Element] = []
            page_figs: list[Figure] = []
            if vision_enabled:
                page_elems, page_figs = _vision_transcribe(
                    state, ctx, page, page_no, default_lang, "scanned",
                    render_dpi, norm, supported, counts,
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

        is_math = state.get("subject_is_math", False)

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
                )
            else:
                counts["unreadable"] += 1
                state["ingestion_status"] = "needs_review"
            page_figs = _digital_figures(state, ctx, doc, page, page_no, ing, default_lang)
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
        if is_math:
            route = "math"
            counts["math"] += 1
            page_elems = []
            if vision_enabled:
                math_png = pdf.render_page_png(page, dpi=render_dpi)
                result = ctx.vision.transcribe_page(
                    math_png, language_hint=page_lang, math=True
                )
                if ctx.vision.is_stub:
                    counts["stub_vision"] += 1
                if result.unreadable:
                    counts["unreadable"] += 1
                    state["ingestion_status"] = "needs_review"
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
        page_figs = _digital_figures(state, ctx, doc, page, page_no, ing, page_lang)
        # math hybrid: add vision-detected figures get_images missed (vector geometry
        # diagrams have no embedded raster), deduped against the get_images figures.
        if math_boxes and ing.get("figures", {}).get("vision_fill_math", True):
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

    doc.close()

    state["pages"] = pages
    state["elements"] = elements
    state["figures"] = figures
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
    if counts["unreadable"]:
        state.setdefault("errors", []).append(
            f"{counts['unreadable']} page(s) were unreadable and skipped; lesson "
            "flagged ingestion_status=needs_review."
        )
    return state


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _vision_transcribe(
    state, ctx, page, page_no, lang, route,
    render_dpi, norm, supported, counts, with_figures=True,
) -> tuple[list[Element], list[Figure]]:
    """Render a page and transcribe its text via vision. Figures come from the
    returned boxes only when ``with_figures`` (scanned pages); a garbage page passes
    False and takes figures from get_images instead. Caller guards on vision_enabled;
    an unreadable page flags needs_review and is not indexed."""
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
    if with_figures:
        page_figs = _figures_from_boxes(state, ctx, png, page_no, result.figures, lang)
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


def _digital_figures(state, ctx, doc, page, page_no, ing, page_lang) -> list[Figure]:
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
                         render_dpi, lang, ing) -> list[Figure]:
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


def _figures_from_boxes(state, ctx, page_png, page_no, boxes, lang) -> list[Figure]:
    """Figures on a scanned page: the page-vision call already returned the boxes
    (stub -> none). Crop each from the rendered page image (boxes are in that
    image's pixel space) at full resolution, upload, then caption the crop."""
    if not boxes:
        return []
    out: list[Figure] = []
    for box in boxes:
        crop = _crop_png_pixels(page_png, box.bbox)
        if crop is None:
            continue
        key = figure_key(state["lecture_id"], page_no, crop)
        s3_uri = ctx.uploader.upload(crop, key)
        # Caption the full-res crop (better than a region of the whole page).
        caption = ctx.vision.caption_figure(crop, language_hint=lang)
        out.append(
            Figure(
                lecture_id=state["lecture_id"],
                page=page_no,
                s3_uri=s3_uri,
                bbox=tuple(box.bbox),
                caption=caption.caption,
                alt_text=caption.alt_text,
            )
        )
    return out


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
    if detect is None or len(text.strip()) < 10:
        return default
    try:
        code = detect(text)
    except Exception:
        return default
    return code if code in supported else default


def _local_path(source_uri: str) -> str:
    """Resolve a source URI to a local path. S3 fetch-to-temp is not wired yet;
    a plain path or file:// URI is used directly."""
    if source_uri.startswith("file://"):
        from urllib.parse import urlparse
        from urllib.request import url2pathname

        return url2pathname(urlparse(source_uri).path)
    if source_uri.startswith("s3://"):
        raise NotImplementedError(
            "S3 fetch-to-temp not wired yet; pass a local path or file:// URI."
        )
    return source_uri
