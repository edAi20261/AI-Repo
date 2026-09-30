"""Does the tutor sound like the teacher, and does it still say only what is true?

A persona that changes nothing is wasted, and a persona that changes the facts is
harmful, so each case is measured both ways. Every case is a moment held back
from the teacher's uploads: what a student asked, and what the teacher really
answered. The same question goes to ``/ai/chat`` twice, once as the teacher's
student and once with no teacher. Then a judge from another model family reads
the teacher's real answer beside both replies.

It reports how much closer the persona reply sounds to the teacher (judge score
and win rate), and how often the persona reply claims something the plain reply
does not. The second number is the one that must stay at zero.

    python scripts/eval_persona.py --cases runs/persona_cases.json
    python scripts/eval_persona.py --cases ... --base http://127.0.0.1:8011 --save runs/p.json

Cases file:

    [{"id": "mona-ratio-1", "teacher_id": "uuid", "persona_version": 2,
      "subject_id": "uuid", "question": "why is 2:3 not the same as 3:2",
      "teacher_answer": "Think of it like a recipe..."}]

``persona_version`` is optional. Without it the active version is used, and the
turn is sent as a student.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import urllib.request
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from langchain_core.messages import HumanMessage  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from shared.config import Config  # noqa: E402
from shared.llm import get_llm  # noqa: E402
from shared.structured import call_structured, schema_hint  # noqa: E402

JUDGE_PROMPT = """You compare two tutor replies with how a real teacher answered the same
student. Judge manner, not length: the way of explaining, the kind of example, the
questions asked back, the warmth, the humour, the phrasing.

The student asked:
{question}

The teacher really answered:
{teacher}

Reply A:
{a}

Reply B:
{b}

Return JSON only. Fields:
{schema}"""


class Verdict(BaseModel):
    a_sounds_like_teacher: int = Field(description="1 (nothing like them) to 5 (could be them).")
    b_sounds_like_teacher: int = Field(description="1 (nothing like them) to 5 (could be them).")
    a_claims_more: bool = Field(description=(
        "True if reply A states a fact, number or rule that reply B does not, other than "
        "an example or comparison offered as illustration."))
    b_claims_more: bool = Field(description=(
        "True if reply B states a fact, number or rule that reply A does not, other than "
        "an example or comparison offered as illustration."))


def ask(base: str, case: dict, *, with_persona: bool, timeout: float) -> str:
    previewing = with_persona and case.get("persona_version")
    body: dict[str, Any] = {
        "user_id": case["teacher_id"] if previewing else "00000000-0000-0000-0000-00000000e7a1",
        "user_type": "teacher" if previewing else "student",
        "message": case["question"],
        "subject_id": case.get("subject_id"),
    }
    if with_persona:
        body["teacher_id"] = case["teacher_id"]
        if previewing:
            body["persona_version"] = case["persona_version"]
    request = urllib.request.Request(
        f"{base}/ai/chat", data=json.dumps({k: v for k, v in body.items() if v}).encode(),
        headers={"content-type": "application/json",
                 "x-service-token": os.environ.get("AI_WEBHOOK_SHARED_SECRET", "")})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())["response"]


def judge(config: Config, llm: Any, case: dict, persona: str, plain: str) -> dict:
    """The verdict with the persona reply as A, whichever side the judge saw it on:
    a judge that prefers the first reply would otherwise be measured as a persona
    that works."""
    swapped = random.Random(case["id"]).random() < 0.5
    a, b = (plain, persona) if swapped else (persona, plain)
    prompt = JUDGE_PROMPT.format(question=case["question"], teacher=case["teacher_answer"],
                                 a=a, b=b, schema=schema_hint(Verdict))
    verdict = call_structured(Verdict, [HumanMessage(content=prompt)], task="judge",
                              config=config, llm=llm, label="persona_judge")
    scores = verdict.model_dump()
    if swapped:
        scores = {"a_sounds_like_teacher": verdict.b_sounds_like_teacher,
                  "b_sounds_like_teacher": verdict.a_sounds_like_teacher,
                  "a_claims_more": verdict.b_claims_more,
                  "b_claims_more": verdict.a_claims_more}
    return {**scores, "swapped": swapped}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cases", required=True)
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--save")
    args = parser.parse_args()

    config = Config.load()
    llm = get_llm(config, "judge")
    rows = []
    for case in json.loads(Path(args.cases).read_text()):
        persona = ask(args.base, case, with_persona=True, timeout=args.timeout)
        plain = ask(args.base, case, with_persona=False, timeout=args.timeout)
        verdict = judge(config, llm, case, persona, plain)
        rows.append({"id": case["id"], "persona": persona, "plain": plain, **verdict})
        print(f"{case['id']:<28} persona {verdict['a_sounds_like_teacher']}  "
              f"plain {verdict['b_sounds_like_teacher']}"
              f"{'  CLAIMS MORE' if verdict['a_claims_more'] else ''}")

    if not rows:
        print("no cases")
        return 1
    wins = sum(r["a_sounds_like_teacher"] > r["b_sounds_like_teacher"] for r in rows)
    summary = {
        "cases": len(rows),
        "persona_style": round(mean(r["a_sounds_like_teacher"] for r in rows), 2),
        "plain_style": round(mean(r["b_sounds_like_teacher"] for r in rows), 2),
        "persona_win_rate": round(wins / len(rows), 2),
        "persona_claims_more": sum(r["a_claims_more"] for r in rows),
    }
    print(json.dumps(summary, indent=2))
    if args.save:
        Path(args.save).write_text(json.dumps({"summary": summary, "rows": rows},
                                              ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
