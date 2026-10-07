"""End-to-end answer quality, measured against a running service.

``eval_retrieval.py`` asks whether the right chunk comes back. This asks the
question a client asks: is the answer any good? It drives ``/ai/chat`` exactly
as the platform does, then checks each reply two ways.

Deterministically, from the response itself: the script the reply is written in,
whether it cited anything, whether the citations resolve to a real lesson,
whether brackets in the text point at sources that were sent, whether the
prerequisite labels are in the same language as the answer, how long the turn
took.

And semantically, with a judge model from a different family than the one that
wrote the answer — Gemini writes, Claude marks — given the question, the answer,
and the passages the answer cited. Marking your own homework is not a measurement;
neither is checking the language with the same detector the service used to
choose it.

    python scripts/eval_answers.py                       # localhost:8000
    python scripts/eval_answers.py --base http://127.0.0.1:8011
    python scripts/eval_answers.py --only ar             # cases whose id starts with
    python scripts/eval_answers.py --save runs/before.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pydantic import BaseModel, Field  # noqa: E402

from shared.config import Config  # noqa: E402
from shared.language import written_in  # noqa: E402
from shared.llm import get_llm  # noqa: E402
from shared.structured import call_structured, schema_hint  # noqa: E402

CASES = ROOT / "scripts" / "eval_answers.json"


# --- the judge --------------------------------------------------------------

class Judgement(BaseModel):
    """What a reader who knows both languages would say about one reply."""

    language: str = Field(
        description="ISO code of the language the reply is written in: ar, en, fr, de, "
                    "or other. Judge the prose, not quoted source terms.")
    mixed_languages: bool = Field(
        description="True if part of the reply — a heading, a label, a list item — "
                    "is in a different language from the rest.")
    answers_the_question: bool = Field(
        description="True if a student who asked this would consider it answered.")
    supported_by_sources: bool = Field(
        description="True if every factual claim is supported by the passages given. "
                    "A part the reply itself says goes beyond the lesson is not held "
                    "to the passages, but it must be correct and about the same "
                    "subject. True when no passages were given AND the reply makes no "
                    "syllabus claims (a greeting, or a refusal to answer).")
    declines: bool = Field(
        description="True if the reply says it cannot answer from the syllabus.")
    invented_detail: Optional[str] = Field(
        default=None,
        description="A specific claim in the reply that the passages do not support, "
                    "or null.")
    quality: int = Field(
        description="1-5. 5: a teacher would send this to a student unchanged. "
                    "3: correct but thin or awkward. 1: wrong, confusing or empty.")
    why: str = Field(
        description="One sentence IN ENGLISH, whatever language the reply is in. "
                    "The reason for the score.")


JUDGE_PROMPT = """\
You are marking one reply from a school tutoring service for Egyptian Grade 7
students. Be strict and literal: you are the last check before this is shown to
a client.

The service is only allowed to answer from the passages it retrieved from the
student's own syllabus. Declining to answer something outside the syllabus is
CORRECT behaviour, not a failure.

The student asked:
{question}

The service replied:
{answer}

The passages the reply cited{passages_note}:
{passages}

Return JSON only, no prose:
{schema}
"""


def judge(config, llm, question: str, answer: str, passages: list[str]) -> Judgement:
    listing = "\n\n".join(f"[{i}] {p}" for i, p in enumerate(passages, 1)) or "(none)"
    note = "" if passages else " — it cited nothing"
    return call_structured(
        Judgement,
        JUDGE_PROMPT.format(question=question, answer=answer, passages=listing,
                            passages_note=note, schema=schema_hint(Judgement)),
        task="judge", config=config, llm=llm, label="judge",
    )


# --- the service ------------------------------------------------------------

def ask(base: str, case: dict, timeout: float) -> tuple[dict, float]:
    body = {
        "user_id": os.environ.get(
            "EVAL_USER_ID", "00000000-0000-0000-0000-000000000001"),
        "user_type": case.get("user_type", "student"),
        "message": case["message"],
    }
    for key in ("language", "subject_id", "session_id"):
        if case.get(key):
            body[key] = case[key]
    request = urllib.request.Request(
        f"{base}/ai/chat", data=json.dumps(body).encode(),
        headers={"content-type": "application/json",
                 "x-service-token": os.environ.get("AI_WEBHOOK_SHARED_SECRET", "")},
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read())
    return payload, time.perf_counter() - started


# --- checks that need no model ---------------------------------------------

def structural_failures(case: dict, result: dict) -> list[str]:
    """What the response says about itself, checked against what was asked for."""
    expect = case.get("expect", {})
    reply = result.get("response") or ""
    citations = result.get("citations") or []
    failures: list[str] = []

    if not reply.strip():
        failures.append("empty reply")

    wanted = expect.get("language")
    if wanted and not written_in(reply, wanted):
        failures.append(f"reply is not in the script of {wanted!r}")

    if expect.get("cites") is True and not citations:
        failures.append("cited nothing")
    if expect.get("cites") is False and citations:
        failures.append(f"cited {len(citations)} source(s) and should not have")

    if "tasks" in expect and result.get("tasks") != expect["tasks"]:
        failures.append(f"tasks={result.get('tasks')}, expected {expect['tasks']}")

    lesson = expect.get("lesson")
    if lesson and not any(lesson.lower() in (c.get("lesson_title") or "").lower()
                          for c in citations):
        got = [c.get("lesson_title") for c in citations] or "nothing"
        failures.append(f"cited {got}, expected a citation to {lesson!r}")

    # A source the reply never points at is noise in the UI; a marker pointing at
    # a source that was not sent renders as a dead badge.
    import re
    marked = {int(n) for group in re.findall(r"\[\s*(\d+(?:\s*,\s*\d+)*)\s*\]", reply)
              for n in group.split(",")}
    sent = {c.get("n") for c in citations}
    if marked - sent:
        failures.append(f"reply marks {sorted(marked - sent)} but no such source was sent")
    if citations and sent - marked:
        failures.append(f"sent sources {sorted(sent - marked)} the reply never cites")

    # The prerequisite block is generated from stored labels, so it is the one
    # part of a reply whose language the model does not control.
    for label in _prereq_labels(result):
        if wanted and not written_in(label, wanted):
            failures.append(f"prerequisite label {label!r} is not in {wanted!r}")
            break

    return failures


def _prereq_labels(result: dict) -> list[str]:
    topics = result.get("prereq_topics") or []
    return [t.get("label", "") for t in topics if isinstance(t, dict)]


def _passages(config, question: str, limit: int = 6) -> list[str]:
    """The syllabus material this question can be answered from.

    The published citation carries a lesson title and a page range, not the text
    — a client renders a badge, it does not re-read the passage. So the judge is
    given what retrieval offers for the same question, which is what the answer
    was written from.
    """
    from agent.tools import rag_search

    chunks = rag_search(question, ["syllabus"], k=limit, config=config)
    return [f"{c.lesson_title or c.lecture_id}: {c.text}" for c in chunks]


# --- run --------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://localhost:8000")
    parser.add_argument("--only", default=None, help="run cases whose id starts with this")
    parser.add_argument("--save", default=None, help="write the full run to this path")
    parser.add_argument("--timeout", type=float, default=240)
    parser.add_argument("--no-judge", action="store_true")
    args = parser.parse_args()

    cases = json.loads(CASES.read_text())["cases"]
    if args.only:
        cases = [c for c in cases if c["id"].startswith(args.only)]

    config = Config.load()
    llm = None if args.no_judge else get_llm(config, "judge")

    print(f"→ {args.base}   {len(cases)} case(s)\n")
    runs: list[dict] = []
    for case in cases:
        try:
            result, elapsed = ask(args.base, case, args.timeout)
        except urllib.error.HTTPError as error:
            detail = error.read()[:300].decode(errors="replace")
            runs.append({"id": case["id"], "message": case["message"],
                         "failures": [f"HTTP {error.code}: {detail}"], "elapsed": None})
            print(f"✗ {case['id']:28} HTTP {error.code}  {detail[:120]}")
            continue
        except Exception as error:      # noqa: BLE001 — report and keep going
            runs.append({"id": case["id"], "message": case["message"],
                         "failures": [f"{type(error).__name__}: {error}"], "elapsed": None})
            print(f"✗ {case['id']:28} {type(error).__name__}: {error}")
            continue

        failures = structural_failures(case, result)
        verdict: dict[str, Any] = {}
        if llm is not None:
            try:
                marked = judge(config, llm, case["message"],
                               result.get("response") or "",
                               _passages(config, case["message"]))
                verdict = marked.model_dump()
                expect = case.get("expect", {})
                if expect.get("language") and marked.language != expect["language"]:
                    failures.append(
                        f"judge read the reply as {marked.language!r}, "
                        f"expected {expect['language']!r}")
                if marked.mixed_languages:
                    failures.append("judge saw more than one language in the reply")
                if expect.get("declines") is True and not marked.declines:
                    failures.append("judge says it answered something it should decline")
                if expect.get("declines") is False and marked.declines:
                    failures.append("judge says it declined something it should answer")
                if expect.get("declines") is not True and not marked.answers_the_question:
                    failures.append("judge says the question was not answered")
                if not marked.supported_by_sources:
                    failures.append(
                        f"judge says unsupported: {marked.invented_detail or marked.why}")
                if marked.quality <= 2:
                    failures.append(f"judge scored {marked.quality}/5: {marked.why}")
            except Exception as error:  # noqa: BLE001
                failures.append(f"judge unavailable: {type(error).__name__}: {error}")

        runs.append({
            "id": case["id"], "message": case["message"], "expect": case.get("expect", {}),
            "language": result.get("language"), "tasks": result.get("tasks"),
            "citations": [c.get("lesson_title") for c in (result.get("citations") or [])],
            "prereqs": _prereq_labels(result),
            "response": result.get("response"),
            "judge": verdict, "failures": failures, "elapsed": elapsed,
        })
        mark = "✓" if not failures else "✗"
        score = f" q{verdict['quality']}" if verdict else ""
        print(f"{mark} {case['id']:28} {elapsed:5.1f}s  {result.get('language')}"
              f"{score}  {len(result.get('citations') or [])} src")
        for failure in failures:
            print(f"    - {failure}")

    ok = [r for r in runs if not r["failures"]]
    timed = [r["elapsed"] for r in runs if r["elapsed"]]
    scores = [r["judge"]["quality"] for r in runs if r.get("judge")]
    print(f"\n  passed         {len(ok)}/{len(runs)}")
    if scores:
        print(f"  mean quality   {sum(scores) / len(scores):.2f}/5  "
              f"(min {min(scores)})")
    if timed:
        ordered = sorted(timed)
        print(f"  latency        median {ordered[len(ordered) // 2]:.1f}s   "
              f"p90 {ordered[int(len(ordered) * 0.9)]:.1f}s   max {ordered[-1]:.1f}s")

    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save).write_text(json.dumps(runs, ensure_ascii=False, indent=2))
        print(f"  saved          {args.save}")
    return 0 if len(ok) == len(runs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
