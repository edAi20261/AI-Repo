"""Render tools — turn structured content into a stored visual artifact.

User-agnostic: student, teacher and parent use the same renderers (a progress chart for a
parent is the same code as a quiz chart for a student). Precise educational visuals are
rendered DETERMINISTICALLY (matplotlib / SVG / reportlab) — image models are unreliable for
exact figures; only ``gen_image`` uses an image model, for illustrative pictures.

Each tool returns an ``artifact_ref`` ``{artifact_id, kind, s3_uri}`` after uploading the
bytes (``shared.storage``) and recording the row in ``ai.artifacts``. ``powerpoint_points``
is the exception: it returns free text (a slide outline), not a file.
"""

from __future__ import annotations

import io
import logging
import unicodedata
import uuid
from collections.abc import Mapping, Sequence
from typing import Any

import matplotlib

matplotlib.use("Agg")   # headless: no display needed
import matplotlib.pyplot as plt
from langchain_core.messages import HumanMessage

from shared.image_gen import get_image_generator
from shared.storage import get_uploader

from .config import Config
from .models import ArtifactRef, Flashcard
from .prompts import POWERPOINT_POINTS_PROMPT
from .tools import connect

logger = logging.getLogger(__name__)

# artifact kind -> file extension + content type.
_MEDIA = {
    "chart": ("png", "image/png"),
    "image": ("png", "image/png"),
    "mindmap": ("svg", "image/svg+xml"),
    "flashcards": ("pdf", "application/pdf"),
    "pdf": ("pdf", "application/pdf"),
}


# --- artifact persistence ---------------------------------------------------

# Register one generated artifact in ai.artifacts — the durable record/index; the bytes
# themselves live in object storage. No user -> nothing to own, so no row is written.
def _record_artifact(user_id, kind, s3_uri, conversation_id, config) -> str | None:
    """INSERT the ``ai.artifacts`` row; returns the id, or None when there is no user to
    own it (an anonymous/demo turn still uploads the file, just records no row). A DB failure
    (e.g. the table absent on a partial dev DB) is non-fatal: the bytes are already stored, so
    we log and return None rather than lose the produced visual."""
    if not user_id:
        return None
    try:
        with connect(config) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO ai.artifacts (user_id, kind, s3_uri, conversation_id) "
                    "VALUES (%s, %s, %s, %s) RETURNING artifact_id::text",
                    (str(user_id), kind, s3_uri, conversation_id))
                aid = cur.fetchone()[0]
            conn.commit()
        return aid
    except Exception:
        logger.warning("ai.artifacts insert skipped (%s); artifact kept at %s", kind, s3_uri)
        return None


def _artifact_key(user_id: str | None, kind: str, ext: str) -> str:
    return f"artifacts/{user_id or 'anon'}/{kind}/{uuid.uuid4().hex}.{ext}"


# The shared tail of every render tool: upload the bytes to object storage, register the
# ai.artifacts row, and return the artifact_ref {artifact_id, kind, s3_uri} the tool hands back.
def _persist(
    data: bytes, kind: str, *, user_id, conversation_id, config
) -> dict[str, Any]:
    ext, content_type = _MEDIA[kind]
    s3_uri = get_uploader(config).upload(data, _artifact_key(user_id, kind, ext), content_type)
    artifact_id = _record_artifact(user_id, kind, s3_uri, conversation_id, config)
    logger.info("render", extra={"kind": kind, "artifact_id": artifact_id})
    return ArtifactRef(artifact_id=artifact_id, kind=kind, s3_uri=s3_uri).model_dump(
        exclude_none=True)


# Render a matplotlib figure to PNG bytes and close it (figures must be freed, not leaked
# across calls). Used by every matplotlib-based tool.
def _fig_bytes(fig, config) -> bytes:
    r = (config.get("render.chart", {}) or {})
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=r.get("dpi", 120), bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


# --- gen_chart: analytics / progress / data charts --------------------------

def gen_chart(
    spec: Mapping[str, Any], *,
    user_id: str | None = None, conversation_id: str | None = None,
    config: Config | None = None,
) -> dict[str, Any]:
    """A bar / line / pie chart from ``spec`` — for analytics, progress, any data.

    ``spec``: ``{type: bar|line|pie, title?, x_label?, y_label?, labels: [...],
    series: [{name, values: [...]}]}`` (pie uses the first series). Deterministic."""
    config = config or Config.load()
    r = config.get("render.chart", {}) or {}
    ctype = (spec.get("type") or "bar").lower()
    labels = list(spec.get("labels") or [])
    series = spec.get("series") or []
    fig, ax = plt.subplots(figsize=(r.get("width", 8.0), r.get("height", 5.0)))
    if ctype == "pie" and series:
        ax.pie(series[0].get("values", []), labels=labels, autopct="%1.0f%%")
    elif ctype == "line":
        x = range(len(labels))
        for s in series:
            ax.plot(x, s.get("values", []), marker="o", label=s.get("name"))
        ax.set_xticks(list(x)); ax.set_xticklabels(labels)
        if any(s.get("name") for s in series):
            ax.legend()
    else:  # bar (grouped when multiple series)
        n = max(1, len(series))
        width = 0.8 / n
        base = range(len(labels))
        for i, s in enumerate(series):
            ax.bar([b + i * width for b in base], s.get("values", []),
                   width=width, label=s.get("name"))
        ax.set_xticks([b + width * (n - 1) / 2 for b in base])
        ax.set_xticklabels(labels)
        if any(s.get("name") for s in series):
            ax.legend()
    if spec.get("title"):
        ax.set_title(spec["title"])
    if spec.get("x_label"):
        ax.set_xlabel(spec["x_label"])
    if spec.get("y_label"):
        ax.set_ylabel(spec["y_label"])
    return _persist(_fig_bytes(fig, config), "chart",
                    user_id=user_id, conversation_id=conversation_id, config=config)


# --- gen_math_visual: function plots + geometry -----------------------------

def gen_math_visual(
    spec: Mapping[str, Any], *,
    user_id: str | None = None, conversation_id: str | None = None,
    config: Config | None = None,
) -> dict[str, Any]:
    """A precise math figure. ``spec`` either plots functions
    (``{series: [{name, x: [...], y: [...]}], title?, x_label?, y_label?}`` — axes through
    the origin, grid on) or draws geometry as a matplotlib figure with exact coordinates
    (``{geometry: {segments: [[[x1,y1],[x2,y2]], ...], points: {label: [x,y]}}}``).
    Deterministic — never an image model (they mis-render exact figures)."""
    config = config or Config.load()
    if spec.get("geometry"):
        return _persist(_geometry_png(spec["geometry"], spec.get("title"), config), "chart",
                        user_id=user_id, conversation_id=conversation_id, config=config)
    r = config.get("render.chart", {}) or {}
    fig, ax = plt.subplots(figsize=(r.get("width", 8.0), r.get("height", 5.0)))
    for s in spec.get("series") or []:
        ax.plot(s.get("x", []), s.get("y", []), label=s.get("name"))
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.grid(True, linestyle=":", alpha=0.6)
    if spec.get("title"):
        ax.set_title(spec["title"])
    ax.set_xlabel(spec.get("x_label", "x"))
    ax.set_ylabel(spec.get("y_label", "y"))
    if any(s.get("name") for s in (spec.get("series") or [])):
        ax.legend()
    return _persist(_fig_bytes(fig, config), "chart",
                    user_id=user_id, conversation_id=conversation_id, config=config)


# Turn an LLM-emitted math figure ({functions: [{name, expr, domain}], geometry, title}) into a
# gen_math_visual spec by sampling each expression DETERMINISTICALLY — the model gives the formula
# and range, our code computes the points (accurate curves, no LLM arithmetic).
import math as _math

_SAFE_MATH = {k: getattr(_math, k) for k in (
    "sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh", "tanh",
    "exp", "log", "log10", "sqrt", "fabs", "floor", "ceil", "pi", "e")}
_SAFE_MATH["abs"] = abs


def _sample_expr(expr: str, domain, n: int) -> tuple[list[float], list[float]]:
    """Evaluate ``expr`` (a formula in ``x``) at ``n`` evenly-spaced points over ``domain``.
    Sandboxed eval: no builtins, math functions only. Non-finite/erroring points are dropped."""
    try:
        a, b = float(domain[0]), float(domain[1])
    except Exception:
        return [], []
    step = (b - a) / max(1, n - 1)
    xs: list[float] = []
    ys: list[float] = []
    for i in range(n):
        x = a + i * step
        try:
            y = eval(expr, {"__builtins__": {}}, {**_SAFE_MATH, "x": x})
        except Exception:
            continue
        if isinstance(y, (int, float)) and _math.isfinite(y):
            xs.append(x)
            ys.append(float(y))
    return xs, ys


def sample_math_spec(spec: Mapping[str, Any], config: Config | None = None) -> dict[str, Any]:
    """Normalize a math figure spec into what ``gen_math_visual`` renders: geometry and
    pre-sampled ``series`` pass through; ``functions`` (expr + domain) are sampled here.
    Returns ``{}`` when nothing plottable can be built."""
    if spec.get("geometry"):
        return {"geometry": spec["geometry"], "title": spec.get("title")}
    if spec.get("series"):                      # already sampled -> passthrough
        return dict(spec)
    n = (config.get("render.math.samples", 200) if config else 200)
    series = []
    for f in spec.get("functions") or []:
        xs, ys = _sample_expr(f.get("expr", ""), f.get("domain") or [-10, 10], n)
        if xs:
            series.append({"name": f.get("name") or f.get("expr"), "x": xs, "y": ys})
    if not series:
        return {}
    return {"series": series, "title": spec.get("title"),
            "x_label": spec.get("x_label", "x"), "y_label": spec.get("y_label", "y")}


def _geometry_png(geo: Mapping[str, Any], title, config) -> bytes:
    """A precise geometry figure (labelled points + segments) via matplotlib — exact
    coordinates, equal aspect, no image model. A segment endpoint may be a raw ``[x, y]``
    pair or a string label that references ``points`` (the natural way a model names a
    triangle's corners); malformed endpoints are skipped rather than raising."""
    r = config.get("render.chart", {}) or {}
    fig, ax = plt.subplots(figsize=(r.get("width", 8.0), r.get("height", 5.0)))
    points = geo.get("points") or {}

    def _xy(pt):
        if isinstance(pt, str):            # a label into points
            pt = points.get(pt)
        if isinstance(pt, (list, tuple)) and len(pt) >= 2:
            try:
                return float(pt[0]), float(pt[1])
            except (TypeError, ValueError):
                return None
        return None

    for seg in geo.get("segments") or []:
        if not (isinstance(seg, (list, tuple)) and len(seg) >= 2):
            continue
        a, b = _xy(seg[0]), _xy(seg[1])
        if a and b:
            ax.plot([a[0], b[0]], [a[1], b[1]], color="#1f4e79", linewidth=2)
    for label, pt in points.items():
        p = _xy(pt)
        if p:
            ax.plot(p[0], p[1], "o", color="#c0392b")
            ax.annotate(str(label), (p[0], p[1]), textcoords="offset points", xytext=(6, 6))
    ax.set_aspect("equal", adjustable="datalim")
    ax.axis("off")
    if title:
        ax.set_title(title)
    return _fig_bytes(fig, config)


# --- gen_mindmap: SVG tree --------------------------------------------------

def gen_mindmap(
    outline: Mapping[str, Any], *,
    user_id: str | None = None, conversation_id: str | None = None,
    config: Config | None = None,
) -> dict[str, Any]:
    """A left-to-right mind-map tree as SVG. ``outline``: ``{label, children: [{label,
    children: [...]}, ...]}`` (any depth). Deterministic layout — no image model."""
    config = config or Config.load()
    r = config.section("render.mindmap")
    svg = _mindmap_svg(outline, r["width"], r["height"],
                       line_chars=r["line_chars"], max_label_chars=r["max_label_chars"])
    return _persist(svg.encode("utf-8"), "mindmap",
                    user_id=user_id, conversation_id=conversation_id, config=config)


_BOX_W = 150
_LINE_H = 16
_BOX_PAD = 8
_ROW_GAP = 10
_COL_GAP = 40
_MARGIN = 20
_ELLIPSIS = "\u2026"


def _graphemes(text: str) -> list[str]:
    clusters: list[str] = []
    for ch in text:
        if clusters and (_extends_cluster(ch) or clusters[-1].endswith("\u200d")):
            clusters[-1] += ch
        else:
            clusters.append(ch)
    return clusters


def _extends_cluster(ch: str) -> bool:
    code = ord(ch)
    return (unicodedata.category(ch) in ("Mn", "Me", "Mc") or code == 0x200D
            or 0xFE00 <= code <= 0xFE0F or 0x1F3FB <= code <= 0x1F3FF
            or 0xE0020 <= code <= 0xE007F)


def _is_rtl(text: str) -> bool:
    for ch in text:
        bidi = unicodedata.bidirectional(ch)
        if bidi in ("R", "AL"):
            return True
        if bidi == "L":
            return False
    return False


def _wrap_label(label: str, line_chars: int, max_label_chars: int) -> list[str]:
    clusters = _graphemes(" ".join(label.split()))
    if len(clusters) > max_label_chars:
        clusters = [*_graphemes("".join(clusters[:max_label_chars - 1]).rstrip()), _ELLIPSIS]
    lines: list[str] = []
    current: list[str] = []
    for word in _graphemes_by_word(clusters):
        while len(word) > line_chars:
            if current:
                lines.append("".join(current))
                current = []
            lines.append("".join(word[:line_chars]))
            word = word[line_chars:]
        if current and len(current) + 1 + len(word) > line_chars:
            lines.append("".join(current))
            current = []
        current = [*current, " ", *word] if current else list(word)
    if current or not lines:
        lines.append("".join(current))
    return lines


def _graphemes_by_word(clusters: list[str]) -> list[list[str]]:
    words: list[list[str]] = [[]]
    for c in clusters:
        if c.isspace():
            words.append([])
        else:
            words[-1].append(c)
    return [w for w in words if w]


def _box_h(lines: list[str]) -> float:
    return len(lines) * _LINE_H + 2 * _BOX_PAD


def _mindmap_svg(root: Mapping[str, Any], width: int, height: int, *,
                 line_chars: int, max_label_chars: int) -> str:
    depth = _depth(root)
    labels: dict[int, list[str]] = {}
    bands: dict[int, float] = {}

    def measure(node) -> float:
        labels[id(node)] = _wrap_label(str(node.get("label", "")), line_chars, max_label_chars)
        kids = node.get("children") or []
        stacked = sum(measure(k) for k in kids) + _ROW_GAP * max(0, len(kids) - 1)
        bands[id(node)] = max(_box_h(labels[id(node)]), stacked)
        return bands[id(node)]

    total = measure(root)
    height = max(height, int(total + 2 * _MARGIN + 0.5))
    width = max(width, 2 * _MARGIN + _BOX_W + (depth - 1) * (_BOX_W + _COL_GAP))
    col_w = (width - 2 * _MARGIN - _BOX_W) / max(1, depth - 1)
    ys: dict[int, float] = {}

    def place(node, top: float) -> None:
        band = bands[id(node)]
        ys[id(node)] = top + band / 2
        kids = node.get("children") or []
        stacked = sum(bands[id(k)] for k in kids) + _ROW_GAP * max(0, len(kids) - 1)
        cursor = top + (band - stacked) / 2
        for k in kids:
            place(k, cursor)
            cursor += bands[id(k)] + _ROW_GAP

    place(root, _MARGIN + (height - 2 * _MARGIN - total) / 2)
    boxes: list[str] = []
    lines: list[str] = []

    def draw(node, d: int):
        x = _MARGIN + d * col_w
        y = ys[id(node)]
        for k in node.get("children") or []:
            kx, ky = _MARGIN + (d + 1) * col_w, ys[id(k)]
            lines.append(f'<line x1="{x + _BOX_W:.1f}" y1="{y:.1f}" x2="{kx:.1f}" '
                         f'y2="{ky:.1f}" stroke="#9aa5b1" stroke-width="1.5"/>')
            draw(k, d + 1)
        boxes.append(_mindmap_box(labels[id(node)], x, y))

    draw(root, 0)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}"><rect width="{width}" height="{height}" '
            f'fill="white"/>{"".join(lines)}{"".join(boxes)}</svg>')


def _mindmap_box(label_lines: list[str], x: float, y: float) -> str:
    h = _box_h(label_lines)
    top = y - h / 2
    rtl = _is_rtl("".join(label_lines))
    tx = x + _BOX_W - _BOX_PAD if rtl else x + _BOX_PAD
    anchor = ' direction="rtl" text-anchor="end"' if rtl else ""
    spans = "".join(
        f'<tspan x="{tx:.1f}" y="{top + _BOX_PAD + (i + 1) * _LINE_H - 4:.1f}">{_esc(line)}</tspan>'
        for i, line in enumerate(label_lines))
    return (f'<rect x="{x:.1f}" y="{top:.1f}" width="{_BOX_W}" height="{h:.1f}" rx="6" '
            'fill="#eaf2fb" stroke="#1f4e79"/>'
            f'<text font-family="sans-serif" font-size="13"{anchor}>{spans}</text>')


def _depth(node: Mapping[str, Any]) -> int:
    kids = node.get("children") or []
    return 1 + (max((_depth(k) for k in kids), default=0) if kids else 0)


# --- gen_flashcards ---------------------------------------------------------

def gen_flashcards(
    cards: Sequence[Mapping[str, str]], *,
    title: str = "Flashcards",
    user_id: str | None = None, conversation_id: str | None = None,
    config: Config | None = None,
) -> dict[str, Any]:
    """A flashcards PDF (one Q/A per card) plus the structured cards echoed back for a
    flip UI. ``cards``: ``[{front, back}]``. Deterministic."""
    config = config or Config.load()
    blocks: list[tuple[str, str]] = [("h", title)]
    for i, c in enumerate(cards, 1):
        blocks.append(("h", f"{i}. {c.get('front', '')}"))
        blocks.append(("p", c.get("back", "")))
    ref = _persist(_pdf_bytes(blocks, config), "flashcards",
                   user_id=user_id, conversation_id=conversation_id, config=config)
    ref["cards"] = [Flashcard(front=c.get("front", ""), back=c.get("back", "")).model_dump()
                    for c in cards]
    return ref


# --- gen_pdf: quiz / summary / report ---------------------------------------

def gen_pdf(
    title: str, blocks: Sequence[Mapping[str, str]], *,
    user_id: str | None = None, conversation_id: str | None = None,
    config: Config | None = None,
) -> dict[str, Any]:
    """A PDF from ``blocks`` (each ``{type: heading|text, text}``). For a quiz, summary or
    report. Deterministic (reportlab)."""
    config = config or Config.load()
    flat: list[tuple[str, str]] = [("h", title)]
    for b in blocks:
        flat.append(("h" if b.get("type") == "heading" else "p", b.get("text", "")))
    return _persist(_pdf_bytes(flat, config), "pdf",
                    user_id=user_id, conversation_id=conversation_id, config=config)


def _pdf_bytes(blocks: Sequence[tuple[str, str]], config) -> bytes:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4)
    styles = getSampleStyleSheet()
    flow = []
    for kind, text in blocks:
        style = styles["Heading2"] if kind == "h" else styles["BodyText"]
        flow.append(Paragraph(_esc(text or ""), style))
        flow.append(Spacer(1, 8))
    doc.build(flow)
    return buf.getvalue()


# --- gen_image: illustrative pictures (image model) -------------------------

def gen_image(
    prompt: str, *,
    user_id: str | None = None, conversation_id: str | None = None,
    config: Config | None = None,
) -> dict[str, Any]:
    """An illustrative picture from a text prompt via the configured image provider
    (``image_gen`` in config; a stub placeholder when no key). NOT for precise math
    figures — use ``gen_math_visual`` for those."""
    config = config or Config.load()
    data = get_image_generator(config).generate(prompt)
    return _persist(data, "image",
                    user_id=user_id, conversation_id=conversation_id, config=config)


# --- powerpoint_points: a slide OUTLINE as text (not a file) ----------------

def powerpoint_points(
    content: str, llm, *, language: str = "en", instructions: str = "", max_slides: int = 10,
) -> str:
    """A slide-by-slide outline (title + bullet points per slide) as free text, for the
    user to paste into their own slides. NOT a .pptx file. Needs an LLM; returns text.
    ``max_slides`` caps the outline so it stays digestible (config render.slides.max_slides)."""
    if llm is None:
        return content
    prompt_text = POWERPOINT_POINTS_PROMPT.render(
        language=language, instructions=instructions,
        content=content, max_slides=max_slides)
    return llm.invoke([HumanMessage(content=prompt_text)]).content


def _esc(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
