"""Build a RAG evaluation set from chunks already in the index.

A question written from a known chunk carries its own ground truth: that chunk
is the passage that answers it. That is what makes context recall and context
precision measurable at all — without it, "did retrieval find the right thing"
has no right thing to compare against.

    .venv/bin/python scripts/make_rag_set.py <lecture_id> --out scripts/eval_rag.json

Questions are drawn across the whole document rather than from one section, and
only from chunks long enough to hold an answer.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from shared.config import Config
from shared.llm import get_llm
from shared.postgres import connect
from shared.structured import call_structured

MIN_WORDS = 60


class Question(BaseModel):
    question: str = Field(description="A question a student would type, answerable "
                                      "ONLY from this passage. No 'according to the "
                                      "passage' — write it as a student would.")
    answer: str = Field(description="The answer, in one or two sentences, taken from "
                                    "the passage and nothing else.")
    answered_here: bool = Field(
        description="False when the passage only MENTIONS the subject — a list of "
                    "objectives, a heading, an instruction to read on — rather than "
                    "containing the answer. Be strict: a case whose passage does not "
                    "answer its own question measures nothing.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lecture_id")
    parser.add_argument("--out", default="scripts/eval_rag.json")
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    config = Config.load()
    with connect(config) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT chunk_id::text, page_from, heading_path, text FROM ai.chunks "
            "WHERE lecture_id = %s::uuid ORDER BY order_index", (args.lecture_id,))
        rows = [r for r in cur.fetchall() if len(r[3].split()) >= MIN_WORDS]

    print(f"{len(rows)} chunks long enough to hold an answer")
    # Spread the sample over the document: a set drawn from one section measures
    # one section.
    random.seed(args.seed)
    stride = max(1, len(rows) // args.count)
    picked = [rows[i] for i in range(0, len(rows), stride)][:args.count]

    llm = get_llm(config, "judge")
    cases = []
    for chunk_id, page, heading, text in picked:
        prompt = ("Read this passage from a Grade 6 science textbook and write ONE "
                  "question a student would type into a tutor, which this passage "
                  "answers, plus its answer.\n\nPassage:\n" + text[:2500])
        try:
            q = call_structured(Question, [HumanMessage(content=prompt)],
                                task="judge", config=config, llm=llm, label="make_rag_set")
        except Exception as exc:  # noqa: BLE001 — one bad passage is not the set
            print(f"  p{page}: skipped ({type(exc).__name__})")
            continue
        if not q.answered_here:
            # Two cases in the first set were written from a lesson opener and an
            # objectives box: the question was about wind turbines, the passage only
            # promised to describe them. Retrieval found that passage first and was
            # marked wrong for it.
            print(f"  p{page}: dropped — the passage does not answer its own question")
            continue
        cases.append({"question": q.question, "answer": q.answer,
                      "chunk_id": chunk_id, "page": page, "heading": heading or ""})
        print(f"  p{page}: {q.question[:72]}")

    Path(args.out).write_text(json.dumps(
        {"lecture_id": args.lecture_id, "cases": cases}, indent=2, ensure_ascii=False))
    print(f"\nwrote {len(cases)} cases to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
