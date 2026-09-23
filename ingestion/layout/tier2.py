"""Reading a page with a document layout model.

The tier between a text layer that cannot be trusted and a general vision model
that reads the page as prose. A magazine-style textbook page is neither: its
text layer is present but scattered, and asking a chat model to transcribe it
costs a second per page and invents a block now and then.

A document layout model answers the question this tier actually asks — where do
the regions sit, what kind is each, and in what order are they read — and it
answers it from pixels, so a page whose text layer is scattered is no harder
than a clean one. What it returns is mapped onto the same ``Block`` shape every
other tier emits, so nothing downstream can tell which tier read a page.

The model is loaded on first use and kept, never at import and never on a
health-check path: the service image carries this module too.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Optional

from .blocks import Block, BlockType, PageMeta, ParsedPage

logger = logging.getLogger(__name__)

TIER = 2

# PP-DocLayout's labels, mapped onto the block types the chunker knows. A label
# that is not here is a region the model found and we have no rule for: it is
# body text rather than dropped, because losing a paragraph is worse than
# mistyping one.
LABELS: dict[str, BlockType] = {
    "doc_title": "heading",
    "title": "heading",
    "paragraph_title": "subheading",
    "text": "body",
    "abstract": "body",
    "algorithm": "body",
    "formula": "body",
    "content": "list",
    "list": "list",
    "table": "table",
    "image": "figure",
    "figure": "figure",
    "chart": "figure",
    "figure_title": "caption",
    "table_title": "caption",
    "chart_title": "caption",
    "vision_footnote": "caption",
    "aside_text": "sidebar",
    "header": "header",
    "header_image": "header",
    "footer": "footer",
    "footer_image": "footer",
    "number": "page_number",
    "footnote": "boilerplate",
    "reference": "boilerplate",
    "seal": "boilerplate",
    "formula_number": "boilerplate",
}

FIGURES = frozenset({"figure"})

_LOCK = threading.Lock()
_PIPELINE: Any = None
_PIPELINE_KEY: Optional[tuple] = None


def enabled(config: Any) -> bool:
    """Whether this deployment reads pages with a layout model at all."""
    return bool(_settings(config).get("enabled", False))


def _settings(config: Any) -> dict:
    return (config.get("ingestion.layout.tier2") if config else None) or {}


def _pipeline(config: Any) -> Any:
    """The layout model, loaded once per process.

    A page is parsed inside one pipeline call; the weights are hundreds of
    megabytes and loading them per page would cost more than the parse.
    """
    global _PIPELINE, _PIPELINE_KEY

    settings = _settings(config)
    key = (settings.get("pipeline", "structure"), settings.get("device", "cpu"),
           settings.get("version", "v1.6"), settings.get("language"),
           bool(settings.get("read_charts", True)),
           bool(settings.get("read_images", True)))
    if _PIPELINE is not None and _PIPELINE_KEY == key:
        return _PIPELINE

    with _LOCK:
        if _PIPELINE is None or _PIPELINE_KEY != key:
            _PIPELINE = _build(key)
            _PIPELINE_KEY = key
        return _PIPELINE


def _build(key: tuple) -> Any:
    """The pipeline this deployment can actually afford to run.

    ``vl`` is one 0.9B vision-language model that reads the whole page and is
    the better reader; on a CPU it wants gigabytes of memory and minutes per
    page. ``structure`` is the older stack of small detection and recognition
    models: it reads the same regions in seconds on the same machine, which is
    what makes ingesting a shelf of books in an afternoon possible at all.
    """
    pipeline, device, version, language, charts, images = key
    if pipeline == "vl":
        from paddleocr import PaddleOCRVL  # lazy, heavy

        built = PaddleOCRVL(device=device, pipeline_version=version,
                            use_chart_recognition=charts,
                            use_ocr_for_image_block=images)
    else:
        from paddleocr import PPStructureV3  # lazy, heavy

        built = PPStructureV3(device=device, use_chart_recognition=charts,
                              **({"lang": language} if language else {}))
    logger.info("layout model ready",
                extra={"pipeline": pipeline, "device": device, "language": language})
    return built


def parse_page(
    image: bytes,
    meta: PageMeta,
    *,
    config: Any,
    pipeline: Any = None,
) -> Optional[ParsedPage]:
    """The page as the layout model read it, or ``None`` when it could not.

    ``None`` is a failure the caller falls back from, never an empty page: a
    page that genuinely holds nothing still comes back as a page with no blocks.
    """
    try:
        runner = pipeline if pipeline is not None else _pipeline(config)
        results = list(runner.predict(_as_array(image)))
    except Exception:
        logger.error("the layout model could not read the page",
                     extra={"doc_id": meta.doc_id, "page": meta.page}, exc_info=True)
        return None

    if not results:
        logger.warning("the layout model returned nothing for a page",
                       extra={"doc_id": meta.doc_id, "page": meta.page})
        return None

    data = _as_dict(results[0])
    blocks = _blocks(data, meta)
    figure_area = sum(b.area for b in blocks if b.type in FIGURES)
    read = meta.model_copy(update={"source_tier": TIER, "figure_area": figure_area})
    logger.info("page read by the layout model",
                extra={"doc_id": meta.doc_id, "page": meta.page, "blocks": len(blocks)})
    return ParsedPage(meta=read, blocks=blocks)


def _as_array(image: bytes):
    """The rendered page as the array the pipeline expects (BGR, as OpenCV has it)."""
    import io

    import numpy as np
    from PIL import Image

    with Image.open(io.BytesIO(image)) as handle:
        rgb = np.array(handle.convert("RGB"))
    return rgb[:, :, ::-1]


def _as_dict(result: Any) -> dict:
    data = getattr(result, "json", result)
    if isinstance(data, dict) and "res" in data and isinstance(data["res"], dict):
        return data["res"]
    return data if isinstance(data, dict) else {}


def _blocks(data: dict, meta: PageMeta) -> list[Block]:
    """The model's regions as blocks, in its reading order, in page points."""
    scale_x = meta.width / (data.get("width") or meta.width or 1)
    scale_y = meta.height / (data.get("height") or meta.height or 1)

    blocks: list[Block] = []
    for index, region in enumerate(data.get("parsing_res_list") or []):
        text = _text_of(region)
        label = str(region.get("block_label") or "").lower()
        kind = LABELS.get(label, "body")
        if not text and kind not in FIGURES:
            continue
        blocks.append(Block(
            text=text,
            type=kind,
            bbox=_bbox(region.get("block_bbox"), scale_x, scale_y, meta),
            reading_order=_order(region, index),
            source_tier=TIER,
        ))
    return blocks


def _text_of(region: dict) -> str:
    return " ".join(str(region.get("block_content") or "").split())


def _order(region: dict, fallback: int) -> int:
    for key in ("block_order", "block_id"):
        value = region.get(key)
        if isinstance(value, int):
            return value
    return fallback


def _bbox(raw: Any, scale_x: float, scale_y: float,
          meta: PageMeta) -> tuple[float, float, float, float]:
    """A region's box in page points, or the whole page when it cannot be read."""
    try:
        x0, y0, x1, y1 = (float(v) for v in list(raw)[:4])
    except (TypeError, ValueError):
        return (0.0, 0.0, meta.width, meta.height)
    box = (min(x0, x1) * scale_x, min(y0, y1) * scale_y,
           max(x0, x1) * scale_x, max(y0, y1) * scale_y)
    return (max(0.0, box[0]), max(0.0, box[1]),
            min(meta.width, box[2]), min(meta.height, box[3]))
