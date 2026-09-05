"""Dump the compiled runtime graph AND each subgraph as Mermaid (+ PNG when renderable).

    python scripts/draw_graph.py                 # write .mmd (+ best-effort .png) to demos/graphs/
    python scripts/draw_graph.py <out_dir>       # write to a different directory

Runs on the stub path (no API key needed). The parent graph shows subgraphs as single adapter
nodes (they are invoked inside adapters, so not statically expandable); each subgraph is drawn
separately here so the full structure is visible. PNG needs a Mermaid renderer (network /
mermaid.ink) — skipped with a note if unavailable; the .mmd text is always written.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.config import get_config
from agent.graph.graph import build_graph
from agent.graph.subgraphs import (
    build_assessment_sg, build_explain_sg, build_grade_sg, build_report_sg,
    build_summarize_sg, build_visual_sg)


def _graphs():
    cfg = get_config()
    return {
        "runtime_graph": build_graph(),
        "explain_sg": build_explain_sg(config=cfg),
        "summarize_sg": build_summarize_sg(config=cfg),
        "assessment_sg": build_assessment_sg(config=cfg),
        "grade_sg": build_grade_sg(config=cfg),
        "report_sg": build_report_sg(config=cfg),
        "visual_sg": build_visual_sg(config=cfg),
    }


def main() -> None:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("demos/graphs")
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, compiled in _graphs().items():
        g = compiled.get_graph()
        (out_dir / f"{name}.mmd").write_text(g.draw_mermaid(), encoding="utf-8")
        print(f"[wrote {out_dir / f'{name}.mmd'}]")
        try:
            (out_dir / f"{name}.png").write_bytes(g.draw_mermaid_png())
            print(f"[wrote {out_dir / f'{name}.png'}]")
        except Exception as e:
            print(f"[PNG skipped for {name}: {e}]", file=sys.stderr)


if __name__ == "__main__":
    main()
