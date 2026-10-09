"""The planner is shown the words a quiz request may use.

"Give me 3 hard multiple choice questions" came back as an MCQ, a true/false and
a short answer. The requested types were a free list of strings, so the model
wrote them in its own words; the validator dropped every one it did not know,
silently, and the default mix took their place.
"""

from __future__ import annotations

import logging

from agent.models import AssessmentSpec


def test_the_planner_sees_which_question_types_it_may_ask_for():
    schema = AssessmentSpec.model_json_schema()
    items = schema["properties"]["types"]["anyOf"][0]["items"]
    assert items["enum"] == ["mcq", "true_false", "short_answer", "essay"]


def test_the_planner_sees_which_difficulties_it_may_ask_for():
    schema = AssessmentSpec.model_json_schema()
    assert schema["properties"]["difficulty"]["anyOf"][0]["enum"] == ["easy", "medium", "hard"]


def test_a_type_outside_the_vocabulary_is_dropped_and_said(caplog):
    with caplog.at_level(logging.WARNING, logger="agent.models"):
        spec = AssessmentSpec(types=["multiple_choice", "mcq"], difficulty="tricky")
    assert spec.types == ["mcq"] and spec.difficulty is None
    dropped = {r.field: r.values for r in caplog.records if hasattr(r, "field")}
    assert dropped == {"types": ["multiple_choice"], "difficulty": ["tricky"]}
