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


# --- marking the page -------------------------------------------------------

def _mark(page, monkeypatch, graded=None, **kwargs):
    """Mark a page whose capture and grading are both canned; returns the result and
    what was sent to the grader."""
    import agent.homework as homework
    from agent.homework import mark

    monkeypatch.setattr(
        homework, "read_image",
        lambda *a, **k: HomeworkCapture.model_validate(page) if page else None)
    sent: dict = {}

    def run_grade(payload, config):
        sent.update(payload)
        grades = graded if graded is not None else [
            {"question": s["question"], "graded": True, "correct": True, "partial_score": 1.0,
             "confidence": 0.95} for s in payload["submissions"]]
        return {"graded": grades, "summary": {"count": len(grades)}, "mastery_events": 1,
                "topics_updated": 1, "mastery_recorded": True, "degraded": False}

    monkeypatch.setattr(homework, "run_grade", run_grade)
    return mark(b"png", config=Config.load(), user_id="s-1", **kwargs), sent


def test_every_answer_on_the_page_is_marked_and_each_blank_is_reported(monkeypatch):
    result, sent = _mark({"questions": [
        {"number": "1", "question": "What is 81 - 52?", "student_answer": "29"},
        {"number": "2", "question": "What is 7 x 8?", "student_answer": None},
        {"number": "3", "question": "Name a prime number", "student_answer": "9"},
    ]}, monkeypatch)

    assert [q.status for q in result.questions] == ["marked", "blank", "marked"]
    assert [s["question"] for s in sent["submissions"]] == [
        "What is 81 - 52?", "Name a prime number"]
    assert result.questions[1].grade is None
    assert result.summary["blank"] == 1


def test_a_blank_answer_is_never_sent_to_be_scored_as_wrong(monkeypatch):
    """A page photographed half-done is not a page answered badly; a blank marked
    zero would become a weak topic the student never attempted."""
    _, sent = _mark({"questions": [{"question": "q", "student_answer": None}]}, monkeypatch)
    assert sent["submissions"] == []


def test_the_page_is_marked_as_an_upload_for_the_student_and_lessons_given(monkeypatch):
    _, sent = _mark({"questions": [{"question": "q", "student_answer": "a", "marks": 2}]},
                    monkeypatch, subject_id="math", lecture_ids=["l1"])

    assert sent["source"] == "uploaded"
    assert sent["user_id"] == "s-1" and sent["subject_id"] == "math"
    assert sent["lecture_ids"] == ["l1"]
    assert sent["submissions"][0]["points"] == 2


def test_an_answer_read_unclearly_reaches_the_grader_as_unclear(monkeypatch):
    """The grader holds a grade on a half-read answer for the teacher; it can only do
    that if the reading's doubt travels with the answer."""
    _, sent = _mark({"questions": [
        {"question": "q1", "student_answer": "a", "unclear": True},
        {"question": "q2", "student_answer": "b"},
    ]}, monkeypatch)
    assert [s["unclear_reading"] for s in sent["submissions"]] == [True, False]


def test_an_unreadable_photo_marks_nothing(monkeypatch):
    import agent.homework as homework

    monkeypatch.setattr(homework, "run_grade",
                        lambda *a, **k: pytest.fail("an unreadable page was graded"))
    monkeypatch.setattr(homework, "read_image", lambda *a, **k: None)
    from agent.homework import mark

    result = mark(b"png", config=Config.load(), user_id="s-1")
    assert result.unreadable is True and result.questions == []


def test_an_answer_the_syllabus_cannot_settle_is_reported_as_not_marked(monkeypatch):
    result, _ = _mark(
        {"questions": [{"question": "q", "student_answer": "a"}]}, monkeypatch,
        graded=[{"question": "q", "graded": False, "correct": False}])
    assert result.questions[0].status == "not_marked"


def test_feedback_is_written_in_the_language_of_the_page(monkeypatch):
    """The language a client sends is often the app's, not the homework's. A page
    written in Arabic is marked in Arabic whatever the interface is set to."""
    _, sent = _mark({"questions": [
        {"question": "ما هو السجع في البلاغة العربية؟", "student_answer": "تشابه أواخر الكلمات"},
    ]}, monkeypatch, language="en")
    assert sent["language"] == "ar"


def test_the_caller_s_language_is_used_when_the_page_cannot_tell(monkeypatch):
    _, sent = _mark({"questions": [{"question": "2+2=", "student_answer": "4"}]},
                    monkeypatch, language="fr")
    assert sent["language"] == "fr"


def test_grades_that_do_not_line_up_with_the_answers_are_refused(monkeypatch):
    """Pairing by position after one went missing would put a grade beside the wrong
    question on the student's page."""
    with pytest.raises(RuntimeError):
        _mark({"questions": [{"question": "q1", "student_answer": "a"},
                             {"question": "q2", "student_answer": "b"}]},
              monkeypatch, graded=[{"question": "q1", "graded": True}])


def test_a_question_says_when_its_reading_was_a_guess(monkeypatch):
    result = _capture({"questions": [
        {"question": "q", "student_answer": "2?", "unclear": True},
        {"question": "q2", "unclear": None},
    ]}, monkeypatch)
    assert [q.unclear for q in result.questions] == [True, False]
