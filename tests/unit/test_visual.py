"""visual_sg tests — collect (dedupe + cap), get_figures-first reuse, kind routing, and
dispatch wiring. Render tools are monkeypatched (no matplotlib/upload)."""

from __future__ import annotations

import pytest

from agent import render as rmod
from agent.config import get_config
from agent.graph import nodes as nmod
from agent.graph.subgraphs import visual_sg as V
from agent.graph.subgraphs.visual_sg import _collect, build_visual_sg


@pytest.fixture(autouse=True)
def _hermetic_llm(monkeypatch):
    """Keep builder tests hermetic: _resolve_llm returns exactly the override passed in
    (None stays None), never resolving a real client from config/env — no accidental
    network calls, deterministic regardless of whether a key is present."""
    monkeypatch.setattr(V, "_resolve_llm", lambda config, task, override: override)


def _fake_renders(monkeypatch):
    """Replace every render tool with a sentinel so routing is observable without drawing."""
    for name in ("gen_chart", "gen_mindmap", "gen_math_visual", "gen_flashcards",
                 "gen_pdf", "gen_image"):
        monkeypatch.setattr(rmod, name,
                            lambda *a, _n=name, **k: {"kind": _n, "s3_uri": f"mem://{_n}"})
    monkeypatch.setattr(rmod, "powerpoint_points",
                        lambda content, llm, **k: "Slide 1: T\n- a")


# --- collect ----------------------------------------------------------------

def test_collect_dedupes_and_caps():
    specs = [
        {"kind": "chart", "params": {"a": 1}},
        {"kind": "chart", "params": {"a": 1}},          # duplicate -> dropped
        {"kind": "mindmap", "params": {"label": "R"}},
    ]
    out = _collect({"specs": specs, "max_per_turn": 1})
    assert len(out["specs"]) == 1                        # capped to 1
    out2 = _collect({"specs": specs, "max_per_turn": 5})
    assert len(out2["specs"]) == 2                       # deduped from 3 to 2


# --- prepare: build render input from the turn's content --------------------

class _JLLM:
    """A fake LLM whose invoke returns a fixed JSON string (the builders extract_json it)."""
    def __init__(self, content):
        self._c = content

    def invoke(self, _messages):
        class _M:
            content = self._c
        return _M()


def test_prepare_builds_mindmap_via_llm():
    llm = _JLLM('{"label": "Photosynthesis", "children": [{"label": "Light"}]}')
    prep = V._build_prepare(get_config(), llm)
    out = prep({"specs": [{"kind": "mindmap", "description": "photosynthesis"}],
                "content": {"explanations": ["light reactions occur in the thylakoid"]},
                "language": "en"})
    p = out["specs"][0]["params"]
    assert p["label"] == "Photosynthesis" and p["children"][0]["label"] == "Light"


def test_prepare_mindmap_skips_without_llm():
    # No LLM -> no branches can be built -> skip, rather than render a meaningless lone box.
    prep = V._build_prepare(get_config(), None)
    out = prep({"specs": [{"kind": "mindmap", "description": "cells"}], "content": {}})
    assert out["specs"] == []


def test_prepare_mindmap_skips_when_no_branches():
    # An LLM reply with a root but no children is degenerate -> skipped.
    llm = _JLLM('{"label": "Cells", "children": []}')
    prep = V._build_prepare(get_config(), llm)
    out = prep({"specs": [{"kind": "mindmap", "description": "cells"}],
                "content": {"explanations": ["cells are the unit of life"]}})
    assert out["specs"] == []


def test_prepare_flashcards_from_quiz_items_no_llm():
    prep = V._build_prepare(get_config(), None)
    content = {"assessment": {"items": [{"question": "Q1", "answer": "A1"},
                                        {"question": "Q2", "explanation": "E2"}]}}
    out = prep({"specs": [{"kind": "flashcards"}], "content": content})
    assert out["specs"][0]["params"]["cards"] == [
        {"front": "Q1", "back": "A1"}, {"front": "Q2", "back": "E2"}]


def test_prepare_pdf_blocks_from_quiz():
    prep = V._build_prepare(get_config(), None)
    content = {"assessment": {"items": [{"question": "Q1", "options": [{"text": "a"}]}]}}
    out = prep({"specs": [{"kind": "pdf", "description": "quiz"}], "content": content})
    blocks = out["specs"][0]["params"]["blocks"]
    assert blocks[0]["type"] == "heading" and "Q1" in blocks[0]["text"]


def test_prepare_chart_from_grade_result():
    prep = V._build_prepare(get_config(), None)
    content = {"grade_result": {"graded": [{"partial_score": 1.0}, {"correct": False}]}}
    out = prep({"specs": [{"kind": "chart"}], "content": content})
    assert out["specs"][0]["params"]["series"][0]["values"] == [1.0, 0.0]


def test_prepare_skips_chart_without_data():
    prep = V._build_prepare(get_config(), None)
    out = prep({"specs": [{"kind": "chart"}], "content": {}})
    assert out["specs"] == []                    # nothing to plot -> skipped, not an empty axis


def test_prepare_grounds_slides_content_from_sibling():
    # Slides content comes from the turn's grounded explanation, not the bare topic.
    prep = V._build_prepare(get_config(), None)
    out = prep({"specs": [{"kind": "slides", "description": "cells"}],
                "content": {"explanations": ["cells are the basic unit of life"]}})
    assert out["specs"][0]["params"]["content"] == "cells are the basic unit of life"


def test_prepare_leaves_prefilled_params_untouched():
    prep = V._build_prepare(get_config(), None)
    spec = {"kind": "mindmap", "params": {"label": "R", "children": []}}
    out = prep({"specs": [spec], "content": {}})
    assert out["specs"][0]["params"] == {"label": "R", "children": []}


# --- math sampling (expr + domain -> points, deterministic) ------------------

def test_sample_math_spec_samples_expression():
    spec = rmod.sample_math_spec(
        {"functions": [{"name": "f", "expr": "x**2", "domain": [-2, 2]}]}, get_config())
    s = spec["series"][0]
    assert len(s["x"]) == 200 and s["x"][0] == -2.0 and s["y"][0] == 4.0


def test_sample_math_spec_geometry_passthrough():
    geo = {"geometry": {"points": {"A": [0, 0]}}, "title": "T"}
    out = rmod.sample_math_spec(geo, get_config())
    assert out["geometry"] == geo["geometry"] and out["title"] == "T"


def test_geometry_png_resolves_label_segments():
    # Regression: a model may name segments by point LABEL ([["A","B"], ...]) rather than
    # raw coord pairs. The renderer must resolve labels via `points`, not index into a string.
    geo = {"points": {"A": [0, 0], "B": [4, 0], "C": [0, 3]},
           "segments": [["A", "B"], ["B", "C"], ["C", "A"]]}
    out = rmod._geometry_png(geo, "3-4-5", get_config())
    assert isinstance(out, bytes) and out[:8].startswith(b"\x89PNG")


def test_geometry_png_skips_malformed_segments():
    # A malformed endpoint (unknown label, wrong arity) is skipped, never raised.
    geo = {"points": {"A": [0, 0]},
           "segments": [["A", "Z"], ["A"], [[1, 1], [2, 2]]]}
    out = rmod._geometry_png(geo, None, get_config())
    assert isinstance(out, bytes) and out[:8].startswith(b"\x89PNG")


def test_explain_sg_emits_math_figure_from_answer(monkeypatch):
    # A math answer that appends a ```json figure block -> a sampled math visual_spec, and the
    # block is stripped from the reader-facing explanation.
    from agent.graph.subgraphs import explain_summarize_sg as E
    from agent.models import RetrievedChunk

    chunk = RetrievedChunk(chunk_id="c1", score=0.9, text="t", lecture_id="lec1",
                           topic_ids=["t1"], page_from=1, page_to=1, heading_path="H",
                           language="en")
    monkeypatch.setattr(E.retrieval, "retrieve", lambda *a, **k: [chunk])
    monkeypatch.setattr(E.retrieval, "resolve_prereqs", lambda *a, **k: ([], []))

    class _LLM:
        def invoke(self, _messages):
            class _M:
                content = ("The parabola opens upward.\n"
                           '```json\n{"functions": [{"name": "f", "expr": "x**2", '
                           '"domain": [-3, 3]}]}\n```')
            return _M()

    sg = E.build_explain_sg(config=get_config(), llm=_LLM())
    out = sg.invoke({"query": "plot x^2", "topic_ids": ["t1"], "namespaces": ["syllabus"],
                     "language": "en"})
    spec = out["visual_spec"]
    assert spec["kind"] == "math" and spec["params"]["series"][0]["y"][0] == 9.0
    assert "```json" not in out["explanation"] and "parabola" in out["explanation"]


# --- reuse (get_figures first) ----------------------------------------------

def test_render_reuses_syllabus_figure(monkeypatch):
    _fake_renders(monkeypatch)
    monkeypatch.setattr(V, "get_figures",
                        lambda topic_ids=None, lecture_id=None, config=None:
                        [{"figure_id": "f1", "s3_uri": "s3://b/f1.png", "caption": "c"}])
    sg = build_visual_sg(config=get_config(), llm=None)
    out = sg.invoke({"specs": [{"kind": "figure", "topic_ids": ["t1"], "description": "cell"}],
                     "max_per_turn": 1})
    v = out["visuals"][0]
    assert v["kind"] == "figure" and v["reused"] and v["s3_uri"] == "s3://b/f1.png"


def test_render_generates_when_no_figure(monkeypatch):
    _fake_renders(monkeypatch)
    monkeypatch.setattr(V, "get_figures", lambda **k: [])
    sg = build_visual_sg(config=get_config(), llm=None)
    out = sg.invoke({"specs": [{"kind": "mindmap", "params": {"label": "R"}}],
                     "max_per_turn": 1})
    assert out["visuals"][0]["kind"] == "gen_mindmap"    # routed to the render tool


def test_figure_reuse_prefers_nearest_page(monkeypatch):
    _fake_renders(monkeypatch)
    figs = [{"figure_id": "f1", "s3_uri": "s3://b/f1", "page": 2},
            {"figure_id": "f2", "s3_uri": "s3://b/f2", "page": 9}]
    monkeypatch.setattr(V, "get_figures", lambda **k: figs)
    sg = build_visual_sg(config=get_config(), llm=None)
    out = sg.invoke({"specs": [{"kind": "figure", "topic_ids": ["t1"], "page": 8}],
                     "max_per_turn": 1})
    assert out["visuals"][0]["figure_id"] == "f2"           # page 9 is nearest page 8


def test_figure_kind_fetches_only_no_generation(monkeypatch):
    _fake_renders(monkeypatch)
    monkeypatch.setattr(V, "get_figures", lambda **k: [])   # no syllabus figure
    sg = build_visual_sg(config=get_config(), llm=None)
    out = sg.invoke({"specs": [{"kind": "figure", "topic_ids": ["t1"]}], "max_per_turn": 1})
    assert out["visuals"] == []                              # figure = fetch-only, never generated


def test_generated_kind_never_substituted_by_figure(monkeypatch):
    # A syllabus figure exists AND the chart spec is grounded — but a chart must still be
    # generated, never replaced by a figure (the bug this guards against).
    _fake_renders(monkeypatch)
    monkeypatch.setattr(V, "get_figures",
                        lambda **k: [{"figure_id": "f1", "s3_uri": "s3://b/f1.png"}])
    sg = build_visual_sg(config=get_config(), llm=None)
    out = sg.invoke({"specs": [{"kind": "chart", "topic_ids": ["t1"],
                                "params": {"labels": ["A"], "series": [{"values": [1]}]}}],
                     "max_per_turn": 1})
    assert out["visuals"][0]["kind"] == "gen_chart"         # generated, not the figure


# --- kind routing -----------------------------------------------------------

def test_dispatch_routes_each_kind(monkeypatch):
    _fake_renders(monkeypatch)
    cases = {"chart": "gen_chart", "mindmap": "gen_mindmap", "math": "gen_math_visual",
             "geometry": "gen_math_visual", "flashcards": "gen_flashcards",
             "pdf": "gen_pdf"}
    for kind, tool in cases.items():
        ref = V._dispatch({"kind": kind, "params": {"cards": []}}, {"config": get_config()},
                          {"language": "en"}, None)
        assert ref["kind"] == tool
    slides = V._dispatch({"kind": "slides", "description": "x"}, {"config": get_config()},
                         {"language": "en"}, object())
    assert slides["kind"] == "slides" and slides["text"].startswith("Slide 1")


def test_unknown_kind_skipped(monkeypatch):
    _fake_renders(monkeypatch)
    monkeypatch.setattr(V, "get_figures", lambda **k: [])
    sg = build_visual_sg(config=get_config(), llm=None)
    out = sg.invoke({"specs": [{"kind": "bogus"}], "max_per_turn": 1})
    assert out["visuals"] == []


# --- subgraph emits a figure spec -------------------------------------------

def test_explain_sg_emits_figure_spec(monkeypatch):
    from agent.graph.subgraphs import explain_summarize_sg as E
    from agent.models import RetrievedChunk

    chunk = RetrievedChunk(chunk_id="c1", score=0.9, text="t", lecture_id="lec9",
                           topic_ids=["t1"], page_from=1, page_to=1, heading_path="H",
                           language="en")
    monkeypatch.setattr(E.retrieval, "retrieve", lambda *a, **k: [chunk])
    monkeypatch.setattr(E.retrieval, "resolve_prereqs", lambda *a, **k: ([], []))
    import copy as _copy
    from agent.config import Config
    base = get_config()
    cfg = Config(_copy.deepcopy(base._data), base.app_env)   # don't mutate the singleton
    cfg._data.setdefault("render", {})["figures_with_explain"] = True   # feature is off by default
    sg = E.build_explain_sg(config=cfg, llm=None)   # stub answer
    out = sg.invoke({"query": "x", "topic_ids": ["t1"], "namespaces": ["syllabus"],
                     "language": "en"})
    spec = out.get("visual_spec")
    assert spec and spec["kind"] == "figure"
    assert spec["topic_ids"] == ["t1"] and spec["lecture_id"] == "lec9"


def test_explain_sg_decline_reports_ungrounded(monkeypatch):
    # A refusal (no inline [n]) must survive the compiled subgraph as grounded=False, so the
    # parent suppresses sources/prereqs. Regression: `grounded` must be in the subgraph schema
    # or LangGraph drops it and the adapter defaults it to True.
    from agent.graph.subgraphs import explain_summarize_sg as E
    from agent.models import RetrievedChunk

    chunk = RetrievedChunk(chunk_id="c1", score=0.5, text="combustion", lecture_id="l",
                           topic_ids=["t1"], page_from=1, page_to=1, heading_path="H",
                           language="en")
    monkeypatch.setattr(E.retrieval, "retrieve", lambda *a, **k: [chunk])
    monkeypatch.setattr(E.retrieval, "resolve_prereqs", lambda *a, **k: ([], [chunk]))

    class _LLM:
        def invoke(self, _m):
            class _M:
                content = "I'm sorry, the context does not cover sourdough."
            return _M()

    sg = E.build_explain_sg(config=get_config(), llm=_LLM())
    out = sg.invoke({"query": "sourdough", "topic_ids": ["t1"], "namespaces": ["syllabus"],
                     "language": "en"})
    assert out.get("grounded") is False
    assert out.get("citations") == [] and "visual_spec" not in out


# --- dispatch wiring --------------------------------------------------------

def test_dispatch_routes_visual():
    sends = nmod.dispatch({
        "tasks": [{"type": "visual", "params": {"query": "a cell",
                   "visual": {"kind": "mindmap"}}, "task_id": "x"}],
        "namespaces": ["syllabus"], "language": "en", "subject_id": "sub1", "user_id": "u1",
    })
    assert len(sends) == 1 and sends[0].node == "visual_adapter"


def _planned_visual(kind: str, message: str):
    from agent.models import PlannedTask, TurnUnderstanding, VisualSpec

    return TurnUnderstanding(
        needs_syllabus=True, student_words=message,
        tasks=[PlannedTask(type="visual", query=message, visual=VisualSpec(kind=kind))])


def test_a_planned_visual_nothing_can_render_is_sent_back_to_the_model():
    """The mind-map starter chip was planned as kind ``image``, which nothing renders,
    so it was skipped silently, the turn produced nothing, and the student saw an
    error instead of a mind map."""
    message = "Draw me a mind map of the main ideas in this topic"
    check = nmod._coherent(["visual"], ("en",), message)
    with pytest.raises(ValueError, match="image"):
        check(_planned_visual("image", message))
    check(_planned_visual("mindmap", message))
    check(_planned_visual("figure", message))
