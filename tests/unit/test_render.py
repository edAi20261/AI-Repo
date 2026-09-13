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
