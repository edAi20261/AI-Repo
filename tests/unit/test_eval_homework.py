"""The homework eval's arithmetic.

``grade.review_below`` is to be set from this script's sweep, so a pairing or
counting slip here would set it wrong with nothing to show for it.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from agent.homework import HomeworkCapture
from agent.models import GradedItem

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("eval_homework", ROOT / "scripts" / "eval_homework.py")
eval_homework = importlib.util.module_from_spec(_spec)
sys.modules["eval_homework"] = eval_homework
_spec.loader.exec_module(eval_homework)


def _page(*questions) -> HomeworkCapture:
    return HomeworkCapture.model_validate({"questions": list(questions)})


def _grade(correct: bool, confidence: float) -> GradedItem:
    return GradedItem(question="q", correct=correct, confidence=confidence)


def test_answers_are_paired_with_the_teacher_s_by_printed_number_not_position():
    """A page read with question 2 first would otherwise score each answer against
    the other's mark."""
    page = _page({"number": "2.", "question": "b", "student_answer": "8"},
                 {"number": "Q1", "question": "a", "student_answer": "29"})
    outcomes = eval_homework.score_page(
        "p", [{"number": "q1", "student_answer": "29", "correct": True},
              {"number": "2", "student_answer": "3", "correct": False}],
        page, {0: _grade(False, 0.9), 1: _grade(True, 0.9)})

    first, second = outcomes
    assert first.read_right is True and first.model_correct is True
    assert second.read_right is False and second.model_correct is False


def test_a_question_the_capture_missed_is_counted_as_not_found():
    outcomes = eval_homework.score_page(
        "p", [{"number": "5", "student_answer": "x", "correct": True}], _page(), {})
    assert outcomes[0].found is False


def test_the_sweep_counts_wrong_grades_let_through_and_caught():
    outcomes = [
        eval_homework.Outcome("p", "1", True, True, False, True, True, True, 0.95),
        eval_homework.Outcome("p", "2", True, True, False, True, True, False, 0.9),
        eval_homework.Outcome("p", "3", True, True, False, False, True, True, 0.55),
        eval_homework.Outcome("p", "4", True, False, True, True, True, False, 0.99),
    ]
    sweep = {row["review_below"]: row
             for row in eval_homework.summarise(outcomes, 0, 1)["sweep"]}

    assert sweep[0.5]["held"] == "1/4 (25%)"
    assert sweep[0.5]["wrong_let_through"] == "2/3 (67%)"
    assert sweep[0.6]["wrong_caught"] == "2/3 (67%)"
    assert sweep[0.6]["wrong_let_through"] == "1/2 (50%)"


def test_a_misread_answer_flagged_unclear_is_counted():
    outcomes = [
        eval_homework.Outcome("p", "1", True, False, True, None, False, None, None),
        eval_homework.Outcome("p", "2", True, False, False, None, False, None, None),
    ]
    assert eval_homework.summarise(outcomes, 0, 1)["misreads_flagged_unclear"] == "1/2 (50%)"
