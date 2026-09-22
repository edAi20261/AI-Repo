"""What the student is inside, and what that forbids.

A lesson id has always been able to narrow retrieval. What it could not do is
change the answer when nothing is found: a student sitting in "Photosynthesis"
who asked about the Second World War was told "there isn't enough syllabus
material", which reads as a gap in the product rather than as the boundary
working. The tutor knew it had been scoped and never said so.

Scope is therefore a declared thing, not a filter argument. It travels with the
turn, so retrieval, the refusal, and quiz generation all obey the same
selection, and the refusal can name it.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class StudyScope(BaseModel):
    """The classroom/subject/lesson selection a turn is bounded by.

    ``lecture_id`` is what makes a scope *narrow*: with it the turn answers from
    one lesson, without it from the whole subject.
    """

    subject_id: Optional[str] = None
    classroom_id: Optional[str] = None
    lecture_id: Optional[str] = None
    lesson_title: Optional[str] = Field(
        default=None,
        description="Shown to the student when a question falls outside the lesson.",
    )

    @property
    def is_narrow(self) -> bool:
        """Whether the turn is confined to a single lesson."""
        return bool(self.lecture_id)

    def label(self) -> Optional[str]:
        """How the lesson is named back to the student, if it can be named."""
        return (self.lesson_title or "").strip() or None


def scope_of(state: dict[str, Any]) -> StudyScope:
    """The turn's scope.

    ``lecture_ids`` predates this field and callers still send it, so a single
    id there is read as a lesson selection rather than requiring every caller to
    move at once.
    """
    declared = state.get("study_scope")
    if isinstance(declared, StudyScope):
        return declared
    if isinstance(declared, dict):
        return StudyScope.model_validate(declared)

    lectures = state.get("lecture_ids") or []
    return StudyScope(
        subject_id=state.get("subject_id"),
        lecture_id=lectures[0] if len(lectures) == 1 else None,
    )


def lecture_filter(state: dict[str, Any]) -> Optional[list[str]]:
    """The lessons retrieval may draw on.

    A declared scope wins. Without one, the legacy ``lecture_ids`` list is used
    as it always was, so a caller narrowing to several lessons keeps working.
    """
    scope = scope_of(state)
    if scope.lecture_id:
        return [scope.lecture_id]
    return state.get("lecture_ids") or None
