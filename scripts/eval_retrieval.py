"""Does retrieval find the right lesson, and does it know when it has nothing?

A student asked a maths question in Arabic and was answered from an Arabic
literature lesson. Nothing in the repo could have caught that: there was no way
to tell whether a retrieval change helped or hurt.

Two numbers matter, and the second one is the one that was failing:

  precision@1  — the right lesson comes back first
  separation   — the worst answerable score minus the best unanswerable one

Separation is what makes declining possible. While it is negative, some question
with no answer in the syllabus outranks some question with one, and no threshold
can tell them apart — which is exactly how a maths question gets answered from a
poem.

Usage:
    APP_ENV=staging .venv/bin/python scripts/eval_retrieval.py
    APP_ENV=staging .venv/bin/python scripts/eval_retrieval.py --json before.json
    .venv/bin/python scripts/eval_retrieval.py --compare before.json after.json

Reads the live index, writes nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.config import Config
from agent.retrieval import retrieve
from agent.tools import rag_search

GOLDEN = Path(__file__).with_suffix(".json")


def _probe(query: str, config, k: int) -> list[tuple[float, str]]:
    chunks = rag_search(query, ["syllabus"], k=k, config=config)
    return [(c.score, c.lesson_title or c.lecture_id or "?") for c in chunks]


def _would_answer(query: str, config, k: int) -> bool:
    """Whether the turn grounds in anything at all, i.e. answers rather than
    declines. This is the decision the student actually experiences, and it is
    the floor — not the ranking — that makes it."""
    floor = config.get("agent.min_score", 0.50)
    return bool(retrieve(query, ["syllabus"], k=k, min_score=floor,
                         topic_filter="soft"))


def run(k: int) -> dict:
    golden = json.loads(GOLDEN.read_text())
    config = Config.load()
    report: dict = {"collection": config.vector_store.get("collection"),
                    "answerable": [], "unanswerable": []}

    for case in golden["answerable"]:
        hits = _probe(case["q"], config, k)
        rank = next((i for i, (_, t) in enumerate(hits, 1)
                     if case["expect"].lower() in t.lower()), 0)
        report["answerable"].append({
            "q": case["q"], "lang": case["lang"], "expect": case["expect"],
            "rank": rank,
            "answers": _would_answer(case["q"], config, k),
            "top_score": hits[0][0] if hits else 0.0,
            "top_lesson": hits[0][1] if hits else "(nothing)",
        })

    for case in golden["unanswerable"]:
        hits = _probe(case["q"], config, k)
        report["unanswerable"].append({
            "q": case["q"], "lang": case["lang"],
            "answers": _would_answer(case["q"], config, k),
            "top_score": hits[0][0] if hits else 0.0,
            "top_lesson": hits[0][1] if hits else "(nothing)",
        })
    return _summarise(report)


def _summarise(report: dict) -> dict:
    answerable = report["answerable"]
    hits_at_1 = sum(1 for r in answerable if r["rank"] == 1)
    mrr = sum(1 / r["rank"] for r in answerable if r["rank"]) / max(len(answerable), 1)
    # The lowest score among queries we CAN answer, against the highest among
    # those we cannot. Everything depends on the first being above the second.
    worst_answerable = min((r["top_score"] for r in answerable if r["rank"] == 1),
                           default=0.0)
    best_unanswerable = max((r["top_score"] for r in report["unanswerable"]), default=0.0)
    # What the student experiences: answered when it should be, declined when
    # there is nothing to answer from.
    answered_ok = sum(1 for r in answerable if r["rank"] == 1 and r["answers"])
    declined_ok = sum(1 for r in report["unanswerable"] if not r["answers"])
    report["summary"] = {
        "answered_correctly": f"{answered_ok}/{len(answerable)}",
        "declined_correctly": f"{declined_ok}/{len(report['unanswerable'])}",
        "precision_at_1": round(hits_at_1 / max(len(answerable), 1), 3),
        "hits_at_1": hits_at_1,
        "total": len(answerable),
        "mrr": round(mrr, 3),
        "worst_answerable": round(worst_answerable, 3),
        "best_unanswerable": round(best_unanswerable, 3),
        "separation": round(worst_answerable - best_unanswerable, 3),
    }
    return report


def show(report: dict) -> None:
    print(f"collection: {report['collection']}\n")
    print(f"{'':2} {'lang':4} {'score':>6} {'query':38} lesson returned")
    for r in report["answerable"]:
        mark = "OK" if r["rank"] == 1 and r["answers"] else (
            "!" if r["rank"] == 1 else ("~" + str(r["rank"]) if r["rank"] else "✗"))
        print(f"{mark:2} {r['lang']:4} {r['top_score']:6.3f} {r['q'][:36]:38} {r['top_lesson'][:34]}")
    print(f"\n{'':2} {'lang':4} {'score':>6} no answer exists — must stay low")
    for r in report["unanswerable"]:
        mark = "✗" if r["answers"] else "OK"
        print(f"{mark:2} {r['lang']:4} {r['top_score']:6.3f} {r['q'][:36]:38} "
              f"{'ANSWERED ANYWAY: ' + r['top_lesson'][:22] if r['answers'] else 'declined'}")

    s = report["summary"]
    print(f"\n  answered correctly {s['answered_correctly']}"
          f"      declined correctly {s['declined_correctly']}")
    print(f"  precision@1      {s['hits_at_1']}/{s['total']}  ({s['precision_at_1']})")
    print(f"  MRR              {s['mrr']}")
    print(f"  worst answerable {s['worst_answerable']}")
    print(f"  best unanswerable{s['best_unanswerable']:>7}")
    verdict = ("clean gap — the floor costs nothing" if s["separation"] > 0 else
               "overlap — the floor must lose a right answer to stop a wrong one")
    print(f"  separation       {s['separation']:+.3f}   {verdict}")


def compare(before: Path, after: Path) -> None:
    b, a = json.loads(before.read_text()), json.loads(after.read_text())
    bs, as_ = b["summary"], a["summary"]
    print(f"{'metric':20} {'before':>10} {'after':>10}   change")
    for key in ("answered_correctly", "declined_correctly"):
        print(f"{key:20} {bs.get(key, '-'):>10} {as_.get(key, '-'):>10}")
    for key in ("precision_at_1", "mrr", "worst_answerable",
                "best_unanswerable", "separation"):
        delta = as_[key] - bs[key]
        print(f"{key:20} {bs[key]:>10} {as_[key]:>10}   {delta:+.3f}")

    print(f"\n{'rank':>9}  {'score':>15}  query")
    print(f"{'before':>4} {'after':>4}  {'before':>7} {'after':>7}")
    for rb, ra in zip(b["answerable"], a["answerable"]):
        # Rank 0 means "not in the results at all" — worse than any rank, so it
        # sorts last rather than first.
        old = rb["rank"] or 99
        new = ra["rank"] or 99
        moved = "  better" if new < old else ("  WORSE" if new > old else "")
        print(f"{old if old < 99 else '-':>4} {new if new < 99 else '-':>4}  "
              f"{rb['top_score']:7.3f} {ra['top_score']:7.3f}  {ra['q'][:38]}{moved}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--json", type=Path, help="write the report here")
    parser.add_argument("--compare", nargs=2, type=Path, metavar=("BEFORE", "AFTER"))
    args = parser.parse_args()

    if args.compare:
        compare(*args.compare)
        return 0

    report = run(args.k)
    show(report)
    if args.json:
        args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"\nwritten to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
