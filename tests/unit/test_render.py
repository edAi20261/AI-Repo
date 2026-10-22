"""Render-tool tests — each tool produces the right file bytes + an artifact_ref. Upload
is faked (no S3/filesystem) and the DB record is faked; no matplotlib window (Agg)."""

from __future__ import annotations

import pytest

from agent import render
from agent.config import get_config


class _MemUploader:
    """Captures the last upload instead of writing anywhere."""
    def __init__(self):
        self.last = None

    def upload(self, data, key, content_type="image/png"):
        self.last = (data, key, content_type)
        return f"mem://{key}"


@pytest.fixture()
def mem(monkeypatch):
    up = _MemUploader()
    monkeypatch.setattr(render, "get_uploader", lambda config: up)
    return up


def test_gen_chart_png(mem):
    ref = render.gen_chart(
        {"type": "bar", "title": "S", "labels": ["A", "B"],
         "series": [{"name": "me", "values": [3, 5]}]}, config=get_config())
    assert ref["kind"] == "chart" and ref["s3_uri"].startswith("mem://")
    assert mem.last[0][:4] == b"\x89PNG"                 # PNG magic
    assert mem.last[2] == "image/png"


def test_gen_mindmap_svg(mem):
    ref = render.gen_mindmap(
        {"label": "Root", "children": [
            {"label": "A", "children": [{"label": "A1"}]}, {"label": "B"}]},
        config=get_config())
    assert ref["kind"] == "mindmap"
    assert mem.last[0].startswith(b"<svg") and mem.last[2] == "image/svg+xml"


def test_gen_math_visual_plot_and_geometry(mem):
    plot = render.gen_math_visual(
        {"series": [{"name": "y=x", "x": [0, 1, 2], "y": [0, 1, 2]}]}, config=get_config())
    assert plot["kind"] == "chart" and mem.last[0][:4] == b"\x89PNG"
    geo = render.gen_math_visual(
        {"geometry": {"points": {"A": [0, 0], "B": [1, 0]},
                      "segments": [[[0, 0], [1, 0]]]}}, config=get_config())
    assert geo["kind"] == "chart" and mem.last[0][:4] == b"\x89PNG"     # PNG figure


def test_gen_pdf_and_flashcards(mem):
    pdf = render.gen_pdf("Quiz", [{"type": "heading", "text": "Q1"},
                                  {"type": "text", "text": "a"}], config=get_config())
    assert pdf["kind"] == "pdf" and mem.last[0][:4] == b"%PDF"
    fc = render.gen_flashcards([{"front": "2+2", "back": "4"}], config=get_config())
    assert fc["kind"] == "flashcards" and fc["cards"] == [{"front": "2+2", "back": "4"}]
    assert mem.last[0][:4] == b"%PDF"


def test_gen_image_stub_png(mem):
    # No provider key in test config -> deterministic stub PNG.
    ref = render.gen_image("a diagram of a cell", config=get_config())
    assert ref["kind"] == "image" and mem.last[0][:4] == b"\x89PNG"


def test_records_artifact_when_user(monkeypatch, mem):
    calls = {}

    class _Cur:
        def execute(self, sql, args): calls["sql"] = sql; calls["args"] = args
        def fetchone(self): return ("art-1",)
        def __enter__(self): return self
        def __exit__(self, *a): return False

    class _Conn:
        def cursor(self): return _Cur()
        def commit(self): calls["committed"] = True
        def __enter__(self): return self
        def __exit__(self, *a): return False

    monkeypatch.setattr(render, "connect", lambda config=None: _Conn())
    ref = render.gen_chart({"type": "bar", "labels": ["A"], "series": [{"values": [1]}]},
                           user_id="u1", conversation_id="c1", config=get_config())
    assert ref["artifact_id"] == "art-1"
    assert "INSERT INTO ai.artifacts" in calls["sql"]
    assert calls["args"] == ("u1", "chart", ref["s3_uri"], "c1")
    assert calls["committed"]


def test_powerpoint_points_uses_llm():
    class _LLM:
        def invoke(self, msgs):
            class _R: content = "Slide 1: T\n- a\n- b"
            return _R()
    out = render.powerpoint_points("photosynthesis", _LLM(), language="en")
    assert out.startswith("Slide 1")
    # No LLM -> echoes the content (never fabricates).
    assert render.powerpoint_points("x", None) == "x"


def test_artifact_keys_are_logical(mem):
    render.gen_chart(
        {"type": "bar", "labels": ["A"], "series": [{"values": [1]}]}, config=get_config()
    )

    assert mem.last[1].startswith("artifacts/anon/chart/")
    assert mem.last[1].endswith(".png")


_SVG = "{http://www.w3.org/2000/svg}"


def _mindmap_boxes(mem, outline):
    import xml.etree.ElementTree as ET
    render.gen_mindmap(outline, config=get_config())
    root = ET.fromstring(mem.last[0].decode("utf-8"))
    rects = root.findall(f"{_SVG}rect")[1:]
    texts = root.findall(f"{_SVG}text")
    return root, list(zip(rects, texts, strict=True))


def _label(text_el) -> str:
    return " ".join(span.text or "" for span in text_el.findall(f"{_SVG}tspan"))


def test_a_long_mindmap_label_is_shown_whole_across_lines(mem):
    """Labels were cut at 22 characters, so six-word labels lost their ending."""
    long = "The water cycle moves water around Earth"
    _, boxes = _mindmap_boxes(mem, {"label": "Root", "children": [{"label": long}]})
    text = next(t for _, t in boxes if "water" in _label(t))
    assert len(text.findall(f"{_SVG}tspan")) > 1
    assert _label(text) == long


def test_a_mindmap_label_with_an_ampersand_is_valid_xml(mem):
    """Escaping before the 22-character cut split '&amp;' and broke the SVG."""
    label = "Rocks & minerals of the earth"
    _, boxes = _mindmap_boxes(mem, {"label": "Root & core", "children": [{"label": label}]})
    assert _label(boxes[0][1]) == label


def test_a_mindmap_label_never_splits_an_emoji(mem):
    """A character cut could fall inside a multi-codepoint emoji and leave half of it."""
    family = "\U0001F468‍\U0001F469‍\U0001F467"
    label = "a" * 17 + family + "b" * 5
    _, boxes = _mindmap_boxes(mem, {"label": "Root", "children": [{"label": label}]})
    spans = [s.text for s in boxes[0][1].findall(f"{_SVG}tspan")]
    assert any(family in s for s in spans)
    assert "".join(spans) == label


def test_an_absurdly_long_mindmap_label_is_capped_with_an_ellipsis(mem):
    cap = get_config().get("render.mindmap.max_label_chars")
    _, boxes = _mindmap_boxes(mem, {"label": "Root", "children": [{"label": "word " * 100}]})
    shown = "".join(s.text for s in boxes[0][1].findall(f"{_SVG}tspan")).replace(" ", "")
    assert shown.endswith("…") and len(shown) <= cap


def test_mindmap_boxes_never_overlap_with_many_leaves(mem):
    """Rows were the canvas height split by leaf count, so 30 leaves stacked boxes on each other."""
    outline = {"label": "Root", "children": [
        {"label": f"Branch {b}", "children": [
            {"label": f"Leaf {b}-{i} with a label long enough to wrap"} for i in range(6)]}
        for b in range(5)]}
    root, boxes = _mindmap_boxes(mem, outline)
    rects = [(float(r.get("x")), float(r.get("y")), float(r.get("width")), float(r.get("height")))
             for r, _ in boxes]
    assert len(rects) == 36
    for i, (x1, y1, w1, h1) in enumerate(rects):
        assert 0 <= y1 and y1 + h1 <= float(root.get("height"))
        assert 0 <= x1 and x1 + w1 <= float(root.get("width"))
        for x2, y2, w2, h2 in rects[i + 1:]:
            assert x1 + w1 <= x2 or x2 + w2 <= x1 or y1 + h1 <= y2 or y2 + h2 <= y1


def test_an_rtl_mindmap_label_is_drawn_right_to_left(mem):
    """Arabic labels were anchored left-to-right, so the box clipped their start."""
    label = "دورة الماء في الطبيعة"
    _, boxes = _mindmap_boxes(mem, {"label": "Root", "children": [{"label": label}]})
    by_label = {_label(t): t for _, t in boxes}
    assert by_label[label].get("direction") == "rtl"
    assert by_label[label].get("text-anchor") == "end"
    assert by_label["Root"].get("direction") is None
