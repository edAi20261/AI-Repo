"""Reading a student's homework off a photo.

A student who is stuck rarely types the question out — they photograph the page.
What arrives is an image holding several numbered questions, sometimes with
their own handwritten answers beside them, and often taken in poor light.

The failure that matters is not a blurred photo. It is a blurred photo answered
anyway: a misread question produces a confident answer to something the student
never asked, and they have no way to tell.
"""

from __future__ import annotations

import pytest

from agent.homework import CapturedQuestion, HomeworkCapture, capture
from shared.config import Config


def _capture(payload, monkeypatch) -> HomeworkCapture:
    """Run capture with the vision transport replaced by a canned reply."""
    import agent.homework as homework

    monkeypatch.setattr(
        homework, "read_image",
        lambda *a, **k: HomeworkCapture.model_validate(payload) if payload else None)
    return capture(b"png", config=Config.load())


# --- what came off the page -------------------------------------------------

def test_questions_and_the_student_s_own_answers_are_kept_apart(monkeypatch):
    result = _capture({"questions": [
        {"number": "1", "question": "ما هو السجع؟", "student_answer": "تشابه الحروف"},
        {"number": "2", "question": "اذكر مثالاً", "student_answer": None},
    ]}, monkeypatch)

    assert [q.number for q in result.questions] == ["1", "2"]
    assert result.questions[0].student_answer == "تشابه الحروف"
    assert result.questions[1].student_answer is None
    assert [q.number for q in result.answered] == ["1"]


def test_a_wrong_answer_is_copied_not_corrected(monkeypatch):
    """The point of reading the page is to mark what they wrote. A capture that
    silently fixes the answer marks work the student never did."""
    result = _capture({"questions": [
        {"question": "What is 81 - 52?", "student_answer": "29"},
    ]}, monkeypatch)
    assert result.questions[0].student_answer == "29"


def test_a_blank_answer_is_absent_rather_than_empty(monkeypatch):
    """"" and null both mean unanswered, and two spellings of one state is how a
    grader ends up marking an empty string wrong."""
    result = _capture({"questions": [
        {"question": "q", "student_answer": "   "}, {"question": "q2", "number": ""},
    ]}, monkeypatch)
    assert result.questions[0].student_answer is None
    assert result.questions[1].number is None


def test_an_entry_with_no_question_is_not_a_question(monkeypatch):
    result = _capture({"questions": [
        {"question": "   ", "student_answer": "x"}, {"question": "real one"},
    ]}, monkeypatch)
    assert [q.question for q in result.questions] == ["real one"]


def test_a_whole_booklet_is_capped(monkeypatch):
    from agent.homework import MAX_QUESTIONS

    result = _capture(
        {"questions": [{"question": f"q{i}"} for i in range(MAX_QUESTIONS + 5)]},
        monkeypatch)
    assert len(result.questions) == MAX_QUESTIONS


# --- the photo we could not read --------------------------------------------

def test_an_unreadable_photo_says_so_rather_than_returning_no_questions(monkeypatch):
    """A blurred page and a blank page are the same empty list otherwise, and
    the student is told their homework has no questions on it."""
    result = _capture({"questions": [], "unreadable": True}, monkeypatch)
    assert result.unreadable
    assert result.questions == []


def test_a_vision_failure_is_unreadable_not_empty(monkeypatch):
    """Every model failing is not evidence about the page."""
    result = _capture(None, monkeypatch)
    assert result.unreadable is True


def test_a_page_with_no_questions_is_readable_and_empty(monkeypatch):
    """Not every photo is homework. An empty list with unreadable False is a
    different answer from a photo we could not read, and the tutor should say
    something different."""
    result = _capture({"questions": []}, monkeypatch)
    assert result.unreadable is False
    assert result.questions == []


# --- the contract -----------------------------------------------------------

def test_the_capture_prompt_is_transcription_not_translation():
    """The questions go back to the student as they wrote them, and an answer
    being marked must be the answer they actually gave."""
    from shared.prompting import LanguageRule

    from agent.prompts import HOMEWORK_CAPTURE_PROMPT

    assert HOMEWORK_CAPTURE_PROMPT.language_rule is LanguageRule.TRANSCRIBE
    with pytest.raises(ValueError):
        HOMEWORK_CAPTURE_PROMPT.render(language="ar", language_hint="", schema="{}")


def test_marks_cannot_be_negative():
    with pytest.raises(ValueError):
        CapturedQuestion(question="q", marks=-1)
