"""The four things a RAG system is judged on, measured end to end.

    context recall     did retrieval find the passage that holds the answer
    context precision  how much of what it returned was worth returning
    faithfulness       is the answer supported by the passages it was given
    answer relevance   does the answer address what was asked

Recall and precision are measured against the passage each question was written
from, so they need no judgement. Faithfulness and relevance do, and the judge is
a different model family from the one that writes the answers — a model asked to
mark its own output marks it well.

    .venv/bin/python scripts/eval_rag.py                      # everything
    .venv/bin/python scripts/eval_rag.py --retrieval-only     # free, no model calls
    .venv/bin/python scripts/eval_rag.py --json before.json
    .venv/bin/python scripts/eval_rag.py --compare before.json after.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from agent import retrieval
from shared.config import Config
from shared.llm import get_llm
from shared.postgres import connect
from shared.structured import call_structured

SET = Path(__file__).with_name("eval_rag.json")
K = 5


class Judgement(BaseModel):
    supported: bool = Field(description="True when every factual claim in the answer "
                                        "appears in the passages. Hedged or absent "
                                        "claims do not count against it.")
    addresses: bool = Field(description="True when the answer addresses the question "
                                        "asked, rather than a neighbouring topic.")
    useful_passages: list[int] = Field(
        default_factory=list,
        description="Indices of the passages that carry part of the answer. Passages "
                    "that merely share a word with the question are not useful.")
    note: str = Field(default="", description="One short line on what went wrong, "
                                              "empty when nothing did.")


def _subject_of(config: Config, lecture_id: str) -> str | None:
    with connect(config) as conn, conn.cursor() as cur:
        cur.execute("SELECT subject_id::text FROM content.lessons WHERE id = %s::uuid",
                    (lecture_id,))
        row = cur.fetchone()
    return row[0] if row else None


def _retrieve(case, config, subject_id, lecture_id, floor):
    return retrieval.retrieve(
        case["question"], ["syllabus"], subject_id=subject_id,
        lecture_ids=[lecture_id], k=K, min_score=floor)


def _judge(case, chunks, answer, config, llm):
    passages = "\n\n".join(
        f"[{i}] {c.text[:900]}" for i, c in enumerate(chunks))
    prompt = (f"Question: {case['question']}\n\n"
              f"Passages the tutor was given:\n{passages}\n\n"
              f"The tutor's answer:\n{answer}\n\n"
              "Judge the answer against the passages only.")
    return call_structured(Judgement, [HumanMessage(content=prompt)],
                           task="judge", config=config, llm=llm, label="eval_rag")


def _answer(case, chunks, config, llm):
    passages = "\n\n".join(f"[{i + 1}] {c.text[:900]}" for i, c in enumerate(chunks))
    prompt = ("Answer the student's question using only these passages. If they do "
              "not answer it, say so.\n\n"
              f"Passages:\n{passages}\n\nQuestion: {case['question']}")
    return llm.invoke([HumanMessage(content=prompt)]).content


def run(retrieval_only: bool) -> dict:
    config = Config.load()
    data = json.loads(SET.read_text())
    lecture_id = data["lecture_id"]
    subject_id = _subject_of(config, lecture_id)
    floor = config.get("agent.min_score")

    answer_llm = None if retrieval_only else get_llm(config, "explain")
    judge_llm = None if retrieval_only else get_llm(config, "judge")

    rows = []
    for case in data["cases"]:
        chunks = _retrieve(case, config, subject_id, lecture_id, floor)
        found = [c.chunk_id for c in chunks]
        recall = case["chunk_id"] in found
        row = {"page": case["page"], "question": case["question"],
               "recall": recall, "retrieved": len(chunks),
               "rank": found.index(case["chunk_id"]) + 1 if recall else 0}
        if not retrieval_only and chunks:
            answer = _answer(case, chunks, config, answer_llm)
            verdict = _judge(case, chunks, answer, config, judge_llm)
            useful = sorted(i for i in verdict.useful_passages
                            if 0 <= i < len(chunks))
            row.update(faithful=verdict.supported, relevant=verdict.addresses,
                       useful=len(useful), useful_at=useful,
                       first_useful=(useful[0] + 1) if useful else 0,
                       note=verdict.note)
        rows.append(row)
        mark = "OK " if recall else "✗  "
        print(f"  {mark} p{case['page']:3d} rank={row['rank'] or '-':>2} "
              f"got={row['retrieved']} {case['question'][:52]}")
    return {"floor": floor, "k": K, "rows": rows}


def report(result: dict) -> None:
    """The four metrics, and the two numbers that make one of them readable.

    Raw context precision is bounded by k: when a question is answered by a
    single passage, retrieving five caps it at 0.20 however good retrieval is.
    Most questions here are answered by one passage, so precision@1 and the
    rank of the first useful passage say what the raw figure cannot.
    """
    rows = result["rows"]
    n = len(rows)
    judged = [r for r in rows if "faithful" in r]
    recall = sum(1 for r in rows if r["recall"]) / n

    print(f"\n  cases {n}   k={result['k']}   floor={result['floor']}")
    print(f"  context recall      {recall:.3f}   "
          f"({sum(1 for r in rows if r['recall'])}/{n} found the passage)")
    if judged:
        retrieved = sum(r["retrieved"] for r in judged)
        useful = sum(r["useful"] for r in judged)
        at_one = sum(1 for r in judged if r.get("first_useful") == 1)
        found_any = [r for r in judged if r.get("first_useful")]
        mrr = sum(1 / r["first_useful"] for r in found_any) / len(judged)
        print(f"  context precision   {useful / retrieved:.3f}   "
              f"({useful} of {retrieved} passages carried part of the answer; "
              f"k={result['k']} caps this)")
        print(f"    precision@1       {at_one / len(judged):.3f}   "
              f"({at_one}/{len(judged)} had a useful passage FIRST)")
        print(f"    MRR               {mrr:.3f}   "
              f"(rank of the first useful passage)")
        print(f"    nothing useful    {len(judged) - len(found_any)}  pages "
              f"{[r['page'] for r in judged if not r.get('first_useful')]}")
        print(f"  faithfulness        "
              f"{sum(1 for r in judged if r['faithful'])/len(judged):.3f}")
        print(f"  answer relevance    "
              f"{sum(1 for r in judged if r['relevant'])/len(judged):.3f}")
    nothing = [r for r in rows if r["retrieved"] == 0]
    if nothing:
        print(f"  retrieved nothing   {len(nothing)}  pages "
              f"{[r['page'] for r in nothing]}")


def compare(before: Path, after: Path) -> None:
    b, a = json.loads(before.read_text()), json.loads(after.read_text())
    for label, result in (("before", b), ("after", a)):
        print(f"\n=== {label} ===")
        report(result)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--json", type=Path)
    parser.add_argument("--compare", nargs=2, type=Path)
    args = parser.parse_args()

    if args.compare:
        compare(*args.compare)
        return 0
    result = run(args.retrieval_only)
    report(result)
    if args.json:
        args.json.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"\n  wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
