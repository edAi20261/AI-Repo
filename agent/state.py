"""MainState — the parent runtime graph's shared state.

Subgraphs do NOT inherit this; each has its own private state and is wrapped by
an adapter node. Fields written by fan-out subgraphs use ``operator.add`` so
several dispatched tasks merge their results instead of overwriting.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Optional, TypedDict

from .models import Citation, PrereqTopic, Style


class Task(TypedDict, total=False):
    type: str                       # explain | summarize | assessment | grade | visual
    params: dict[str, Any]          # query, topic_ids, level, length, ...
    task_id: str


class MainState(TypedDict, total=False):
    # -- request (classify inputs) --
    user_id: str
    user_type: str                  # student | teacher
    session_id: Optional[str]       # ephemeral upload namespace, if any
    message: str
    subject_id: Optional[str]       # the subject this turn is about (scopes retrieval)
    lecture_ids: Optional[list[str]]  # optional lecture scope; None -> whole subject
    study_scope: Optional[dict[str, Any]]  # the UI selection this turn is bounded by
    grade: Optional[str]            # academic grade/year level (NOT a grading result — see grade_result)
    grade_ref: Optional[dict[str, Any]]  # structured {assessment_id, kind} from the frontend for a grade turn

    # -- classify --
    pathway: str                    # student | teacher
    intent: str                     # greeting | thanks | meta | chitchat | question
    language: str                   # ar | en | fr | de — the reply language for the turn
    language_decision: dict[str, Any]   # which rule chose it (shared.language.LanguageDecision)

    # -- load_context --
    namespaces: list[str]
    style: Style

    # -- plan --
    tasks: list[Task]

    # -- subgraph fan-in (reducer-merged) --
    answers: Annotated[list[dict[str, Any]], operator.add]
    citations: Annotated[list[Citation], operator.add]
    prereq_topics: Annotated[list[PrereqTopic], operator.add]

    # -- assessment_sg (one ephemeral assessment per turn; surfaced as /ai/chat `assessment`) --
    assessment: dict[str, Any]

    # -- grade_sg (a chat 'grade me this' turn; grades + moves mastery, AI-final) --
    grade_result: dict[str, Any]

    # -- visual_sg: subgraphs emit visual_specs (reducer-merged); visual_sg renders them --
    visual_specs: Annotated[list[dict[str, Any]], operator.add]
    visuals: list[dict[str, Any]]
    visual_error: bool              # a visual_spec RAISED during render (vs produced nothing)

    # -- session memory: the standing language instruction, and any change to it --
    session_language: Optional[str]          # in force this turn, or None
    session_language_write: Optional[dict]   # {"op": "set"|"clear", ...} for persist

    # -- compose --
    response: str
    # The citations the reply actually carries, renumbered globally and capped. Kept
    # apart from ``citations`` because that one merges by addition: compose replaces
    # the subgraphs' local numbering rather than adding to it.
    published_citations: list[Citation]
    language_repair: str            # "" | "repaired" | "failed" — set by verify_language
