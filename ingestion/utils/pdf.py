"""PyMuPDF access — text, page render, embedded images.

Pure extraction geometry: no routing, no normalisation. The extract stage
decides what to do with the results.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import fitz  # PyMuPDF


@dataclass
class Block:
    """A text block with its median font size (for heading detection). Coords are
    PDF points."""

    text: str
    size: float
    bbox: tuple[float, float, float, float]
    page: int


@dataclass
class EmbeddedImage:
    """A raw embedded image: bytes (already the crop), placement bbox (points),
    pixel size."""

    xref: int
    image_bytes: bytes
    ext: str
    bbox: tuple[float, float, float, float]
    width: int
    height: int


def open_document(path: str) -> fitz.Document:
    """Open a local PDF path. (S3 URIs are fetched to a temp path upstream.)"""
    return fitz.open(path)


def page_text(page: fitz.Page) -> str:
    """Raw text of a page (no normalisation)."""
    return page.get_text("text")


def page_char_count(page: fitz.Page) -> int:
    """Non-whitespace character count — the scanned-page trigger metric."""
    return len("".join(page_text(page).split()))


def extract_blocks(page: fitz.Page, page_number: int) -> list[Block]:
    """Text blocks in reading order, each tagged with its median span size
    (median resists one large glyph faking a heading)."""
    data = page.get_text("dict")
    blocks: list[Block] = []
    for blk in data.get("blocks", []):
        if blk.get("type", 0) != 0:  # 0 = text; 1 = image (handled separately)
            continue
        sizes: list[float] = []
        lines: list[str] = []
        for line in blk.get("lines", []):
            spans = line.get("spans", [])
            line_text = "".join(s.get("text", "") for s in spans)
            if line_text.strip():
                lines.append(line_text)
                sizes.extend(s.get("size", 0.0) for s in spans if s.get("text", "").strip())
        text = "\n".join(lines).strip()
        if not text:
            continue
        blocks.append(
            Block(
                text=text,
                size=_median(sizes),
                bbox=tuple(blk.get("bbox", (0, 0, 0, 0))),  # type: ignore[arg-type]
                page=page_number,
            )
        )
    return blocks


def render_page_png(page: fitz.Page, dpi: int = 300) -> bytes:
    """Render a full page to PNG bytes for a vision transcription call."""
    pix = page.get_pixmap(dpi=dpi)
    return pix.tobytes("png")


def extract_embedded_images(doc: fitz.Document, page: fitz.Page, page_number: int) -> list[EmbeddedImage]:
    """Embedded raster images on a digital page (bytes + placement + pixel size).
    Scan-tile filtering happens downstream in the extract stage."""
    images: list[EmbeddedImage] = []
    for img in page.get_images(full=True):
        xref = img[0]
        try:
            rects = page.get_image_rects(xref)
        except Exception:  # pragma: no cover - malformed xref
            rects = []
        bbox = (
            tuple(rects[0]) if rects else (0.0, 0.0, 0.0, 0.0)  # type: ignore[arg-type]
        )
        try:
            extracted = doc.extract_image(xref)
        except Exception:  # pragma: no cover
            continue
        images.append(
            EmbeddedImage(
                xref=xref,
                image_bytes=extracted["image"],
                ext=extracted.get("ext", "png"),
                bbox=bbox,  # type: ignore[arg-type]
                width=extracted.get("width", 0),
                height=extracted.get("height", 0),
            )
        )
    return images


def iter_pages(doc: fitz.Document) -> Iterator[tuple[int, fitz.Page]]:
    """Yield ``(1-based page number, page)`` pairs."""
    for i, page in enumerate(doc):
        yield i + 1, page


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0
