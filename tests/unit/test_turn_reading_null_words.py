"""The model writes JSON null as the word "null" often enough to fail turns.

Reading a quiz request, Gemini answered ``"assessment": {"types": "null"}`` and
``"grade": "null"``. Validation rejected the whole reading, the retry did the same,
and the student got a 503. Under thirty concurrent students, 37 first readings
failed this way and 4 turns failed outright.
"""

from __future__ import annotations

from agent.models import TurnUnderstanding


def test_the_word_null_is_read_as_an_absent_field_at_every_level():
    reading = TurnUnderstanding.model_validate({
        "student_words": "Quiz me with 3 questions on this.",
        "needs_syllabus": True,
        "tasks": [{"type": "assessment", "query": "null", "level": "null",
                   "assessment": {"count": 3, "types": "null", "difficulty": "None",
                                  "focus": "null"},
                   "grade": "null", "visual": "null"}],
        "message_language": "null",
        "language_request": {"action": "none", "code": "null"},
        "preference": "null",
    })
    task = reading.tasks[0]
    assert task.assessment.count == 3 and task.assessment.types is None
    assert task.grade is None and task.visual is None and task.query is None
    assert task.level == "intermediate"
    assert reading.message_language is None and reading.language_request.code is None


def test_a_real_value_that_merely_contains_null_is_kept():
    reading = TurnUnderstanding.model_validate({
        "needs_syllabus": True,
        "tasks": [{"type": "explain", "query": "what is a null hypothesis"}],
    })
    assert reading.tasks[0].query == "what is a null hypothesis"
