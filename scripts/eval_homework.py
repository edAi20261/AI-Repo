"""How well a photographed page is read and marked, against a teacher's marks.

Three numbers decide whether marking a photo can be trusted, and each has its
own failure. Reading: was the student's answer copied as they wrote it, or did a
smudged 3 become an 8. Marking: does the grade agree with the teacher's. Review:
of the grades the model was unsure of, how many were actually wrong, and how
many wrong ones it was sure of.

``grade.review_below`` decides how many grades wait for a teacher. Too low and
wrong marks reach mastery unseen; too high and a teacher re-marks the whole
class. The sweep printed at the end is what it is set from.

Each page in ``eval_homework.json`` is a photo of real homework with what the
teacher read and decided for every question:

    {"id": "ar-math-1", "source": "s3://bucket/eval/homework/ar-math-1.jpg",
     "subject_id": "…", "lecture_ids": ["…"], "language": "ar",
     "answers": [{"number": "1", "student_answer": "29", "correct": true},
                 {"number": "2", "student_answer": null, "correct": null}]}

``source`` is anything ``shared.source.materialise`` reads: s3, https or a path.
Photos of children's work stay out of git.

Grading runs with no student, so no mastery is written. Mistake labels are
filed as tags, as every grade's are.

    .venv/bin/python scripts/eval_homework.py
    .venv/bin/python scripts/eval_homework.py --only ar --json run.json
    scripts/eval_on_aws.sh homework
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.graph.grade import run_grade  # noqa: E402
from agent.homework import CapturedQuestion, HomeworkCapture, capture  # noqa: E402
from agent.models import GradedItem  # noqa: E402
from shared.config import Config  # noqa: E402
from shared.matching import fold_for_match  # noqa: E402
from shared.source import materialise  # noqa: E402

SET = Path(__file__).with_name("eval_homework.json")
SWEEP = (0.5, 0.6, 0.7, 0.8, 0.9)


@dataclass
class Outcome:
    """One question on one page: what the teacher saw and what the service did."""

    page: str
    number: str
    found: bool
    read_right: Optional[bool]
    unclear: bool
    teacher_correct: Optional[bool]
    graded: bool
    model_correct: Optional[bool]
    confidence: Optional[float]


def _key(number: Any) -> str:
    return fold_for_match(str(number or "")).strip(" .:)-").lower()


def _same(read: Optional[str], expected: Optional[str]) -> bool:
    return fold_for_match(read or "").lower() == fold_for_match(expected or "").lower()


def score_page(page_id: str, expected: list[dict[str, Any]], page: HomeworkCapture,
               grades: dict[int, GradedItem]) -> list[Outcome]:
    """Pair the teacher's answers with what was read and graded, by printed number."""
    by_number: dict[str, tuple[int, CapturedQuestion]] = {
        _key(q.number): (i, q) for i, q in enumerate(page.questions) if q.number}
    outcomes = []
    for want in expected:
        hit = by_number.get(_key(want["number"]))
        if hit is None:
            outcomes.append(Outcome(page_id, str(want["number"]), False, None, False,
                                    want.get("correct"), False, None, None))
            continue
        index, question = hit
        grade = grades.get(index)
        outcomes.append(Outcome(
            page=page_id, number=str(want["number"]), found=True,
            read_right=_same(question.student_answer, want.get("student_answer")),
            unclear=question.unclear, teacher_correct=want.get("correct"),
            graded=bool(grade and grade.graded),
            model_correct=grade.correct if grade and grade.graded else None,
            confidence=grade.confidence if grade else None))
    return outcomes


def _share(part: int, whole: int) -> str:
    return f"{part}/{whole}" + (f" ({part / whole:.0%})" if whole else "")


def summarise(outcomes: list[Outcome], unreadable: int, pages: int) -> dict[str, Any]:
    found = [o for o in outcomes if o.found]
    written = [o for o in found if o.read_right is not None]
    misread = [o for o in written if not o.read_right]
    judged = [o for o in found if o.graded and o.teacher_correct is not None]
    agree = [o for o in judged if o.model_correct == o.teacher_correct]
    sweep = []
    for floor in SWEEP:
        held = [o for o in judged
                if o.unclear or o.confidence is None or o.confidence < floor]
        passed = [o for o in judged if o not in held]
        sweep.append({
            "review_below": floor,
            "held": _share(len(held), len(judged)),
            "wrong_let_through": _share(
                sum(1 for o in passed if o.model_correct != o.teacher_correct), len(passed)),
            "wrong_caught": _share(
                sum(1 for o in held if o.model_correct != o.teacher_correct),
                sum(1 for o in judged if o.model_correct != o.teacher_correct)),
        })
    return {
        "pages_unreadable": _share(unreadable, pages),
        "questions_found": _share(len(found), len(outcomes)),
        "answers_read_exactly": _share(len(written) - len(misread), len(written)),
        "misreads_flagged_unclear": _share(sum(1 for o in misread if o.unclear), len(misread)),
        "graded": _share(len(judged), sum(1 for o in found if o.teacher_correct is not None)),
        "grades_agreeing_with_teacher": _share(len(agree), len(judged)),
        "sweep": sweep,
    }


def run_page(case: dict[str, Any], config: Config) -> tuple[list[Outcome], bool]:
    with materialise(case["source"], config) as path:
        image = Path(path).read_bytes()
    page = capture(image, config=config, language_hint=case.get("language"))
    if page.unreadable:
        return [], True
    answered = [(i, q) for i, q in enumerate(page.questions) if q.student_answer]
    result = run_grade({
        "submissions": [{"question": q.question, "student_answer": q.student_answer,
                         "unclear_reading": q.unclear} for _, q in answered],
        "user_id": None,
        "subject_id": case.get("subject_id"),
        "lecture_ids": case.get("lecture_ids"),
        "language": case.get("language") or "en",
        "source": "uploaded",
    }, config=config)
    grades = {i: GradedItem.model_validate(g)
              for (i, _), g in zip(answered, result.get("graded") or [], strict=True)}
    return score_page(case["id"], case["answers"], page, grades), False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--only", help="run the pages whose id starts with this")
    parser.add_argument("--json", help="write every outcome and the summary here")
    args = parser.parse_args()

    pages = [p for p in json.loads(SET.read_text())["pages"]
             if not args.only or p["id"].startswith(args.only)]
    if not pages:
        print(f"no pages to mark in {SET.name}: add photographed pages with the "
              "teacher's marks (see this script's docstring)")
        return 1

    config = Config.load()
    outcomes: list[Outcome] = []
    unreadable = 0
    for case in pages:
        scored, lost = run_page(case, config)
        unreadable += lost
        outcomes += scored
        print(f"{case['id']}: " + ("unreadable" if lost else
              f"{sum(o.read_right is True for o in scored)}/{len(scored)} read, "
              f"{sum(o.model_correct == o.teacher_correct for o in scored if o.graded)}"
              f"/{sum(o.graded for o in scored)} marks agree"))

    summary = summarise(outcomes, unreadable, len(pages))
    print()
    for name, value in summary.items():
        if name != "sweep":
            print(f"{name:32} {value}")
    print(f"\n{'review_below':>12}  {'held':>14}  {'wrong let through':>18}  "
          f"{'wrong caught':>14}")
    for row in summary["sweep"]:
        print(f"{row['review_below']:>12}  {row['held']:>14}  "
              f"{row['wrong_let_through']:>18}  {row['wrong_caught']:>14}")
    if args.json:
        Path(args.json).write_text(json.dumps(
            {"summary": summary, "outcomes": [asdict(o) for o in outcomes]},
            ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
