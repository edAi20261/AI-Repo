"""Shared data models for the runtime agent.

The pydantic models and lightweight dataclasses that cross module boundaries live
here, so there is one place to look for a shape. Graph state (``TypedDict``) stays
in ``state.py`` and the subgraphs, which own their private state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional, get_args

from pydantic import BaseModel, Field, computed_field, field_validator

# The closed vocabularies, as Literal types (one source of truth). In a pydantic model a
# Literal becomes an enum in the JSON schema, so structured output is constrained to valid
# values. The tuples are the same values for runtime iteration.
TaskType = Literal["explain", "summarize", "assessment", "grade", "visual"]
QuestionType = Literal["mcq", "true_false", "short_answer", "essay"]
Difficulty = Literal["easy", "medium", "hard"]
GradeKind = Literal["quiz", "homework"]
# The visual kinds route to render tools (chart/mindmap/math/geometry/flashcards/pdf/image/
# slides) or reuse a syllabus figure (figure).
VisualKind = Literal["chart", "mindmap", "math", "geometry", "flashcards", "pdf",
                     "image", "slides", "figure"]
VISUAL_KINDS = get_args(VisualKind)

QUESTION_TYPES = get_args(QuestionType)   # design Q2 request/question types
DIFFICULTIES = get_args(Difficulty)


@dataclass
class RetrievedChunk:
    chunk_id: str
    score: float
    text: str
    lecture_id: str
    topic_ids: list[str]
    page_from: Optional[int]
    page_to: Optional[int]
    heading_path: Optional[str]
    language: Optional[str]
    # Readable citation labels, carried in the Qdrant payload (may be absent for
    # chunks ingested before this was added — the citation renderer degrades).
    subject_name: Optional[str] = None
    lesson_title: Optional[str] = None
    lesson_order: Optional[int] = None


@dataclass
class ResolvedTopic:
    topic_id: str
    label: str
    score: float
    subject_id: str = ""


@dataclass
class PrereqTopic:
    topic_id: str
    label: str
    strength: str
    depth: int


class SubjectFacts(BaseModel):
    """A subject's facts read from the backend (``get_subject``), or defaults when it is
    unavailable. Only the fields the agent consumes are modelled; unknown backend fields are
    ignored. TODO: extend as more consumed fields are confirmed."""
    language: Optional[str] = None
    is_math: bool = False
    name: Optional[str] = None


class Style(BaseModel):
    """Voice/style descriptor for a turn (from ``resolve_style``)."""
    tone: str = "neutral"
    voice: str = "neutral"


class AdaptiveProfile(BaseModel):
    """Per-user learned preferences, applied to personalize a turn. One per user across every
    persona (student/teacher/parent), adapted per user. NOT yet wired — see the adaptive-profile
    TODO. A teacher's profile also informs how the agent answers a student's question in that
    teacher's subject (the teacher's teaching style)."""
    user_id: Optional[str] = None
    pathway: Optional[str] = None            # student | teacher | parent
    style: Optional[Style] = None            # preferred tone/voice
    teaching_notes: Optional[str] = None     # how a teacher likes material taught (teacher persona)


class Citation(BaseModel):
    """One source reference for an answer — a retrieved chunk, numbered ``[n]`` inline and in
    the sources block. Rendered INTO the response text (never returned as JSON). ``task_id`` is
    stamped by the adapter so compose can renumber ``[n]`` globally across a multi-task turn."""
    n: int
    lecture_id: Optional[str] = None
    heading_path: Optional[str] = None
    page_from: Optional[int] = None
    page_to: Optional[int] = None
    subject_name: Optional[str] = None
    lesson_title: Optional[str] = None
    lesson_order: Optional[int] = None
    task_id: Optional[str] = None


class AssessmentSpec(BaseModel):
    """A requested assessment shape parsed by the planner (design Q2). All fields are
    optional — only what the user explicitly asked for. Invalid values are coerced to
    None (ignored) so a bad LLM extraction falls back to config defaults. The upper
    bound on ``count`` is a per-role cap applied in the plan node (config
    ``assessment.max_count``), not here — a validator can't see config."""
    count: Optional[int] = None
    types: Optional[list[str]] = None
    difficulty: Optional[str] = None

    @field_validator("count")
    @classmethod
    def _count_positive(cls, v: Optional[int]) -> Optional[int]:
        return v if isinstance(v, int) and v >= 1 else None

    @field_validator("types")
    @classmethod
    def _types_subset(cls, v: Optional[list[str]]) -> Optional[list[str]]:
        kept = [t for t in (v or []) if t in QUESTION_TYPES]
        return kept or None

    @field_validator("difficulty")
    @classmethod
    def _difficulty_allowed(cls, v: Optional[str]) -> Optional[str]:
        return v if v in DIFFICULTIES else None


class AssessmentOption(BaseModel):
    """One MCQ option -> ``quiz_question_options`` (``option_text``, ``is_correct``)."""
    text: str
    is_correct: bool = False


class AssessmentItem(BaseModel):
    """One generated question — maps 1:1 to ``quiz_questions`` / ``homework_questions``
    (and its ``options`` to the *_options table). ``topic_id`` + ``lecture_id`` are
    required (invariant 12): an item cannot be constructed without them, and grading /
    mastery read them straight off the item."""
    type: QuestionType                          # -> question_type
    question: str = Field(min_length=1)         # -> question_text
    topic_id: str = Field(min_length=1)         # -> topic_ids[]
    lecture_id: str = Field(min_length=1)       # -> lesson_id
    answer: Optional[str | list[str]] = None    # -> correct_answer_text (list = multi-correct)
    options: Optional[list[AssessmentOption]] = None
    explanation: Optional[str] = None           # -> explanation
    difficulty: Optional[str] = None            # -> difficulty
    bloom: Optional[str] = None                 # -> bloom
    points: Optional[float] = None              # -> points

    @computed_field  # UI hint: render checkboxes (not radio) when several options are correct.
    @property
    def multiple_correct(self) -> bool:
        return bool(self.options) and sum(o.is_correct for o in self.options) > 1


# --- LLM structured-output schemas ------------------------------------------
# The shapes the planner and the assessment generator ask the model to return via
# `with_structured_output`. Validated by the provider; extract_json is the fallback.

class GradeSpec(BaseModel):
    """Grade params the planner parses from a chat 'grade me this' turn. A DB reference
    (fetch the rubric via ``fetch_rubric``) vs inline free-text (rubric/answers in the
    message) — both optional; absent ``assessment_id`` means grade the inline text."""
    assessment_id: Optional[str] = None
    kind: Optional[str] = None

    @field_validator("kind")
    @classmethod
    def _kind_allowed(cls, v: Optional[str]) -> Optional[str]:
        return v if v in get_args(GradeKind) else None


class VisualSpec(BaseModel):
    """A request for one visual. ``kind`` decides the source: ``figure`` fetches a syllabus
    figure and never generates; ``image`` tries a syllabus figure first, else generates;
    every other kind always generates via its render tool (a figure never substitutes for a
    requested chart). ``params`` is the tool payload (chart spec, mindmap outline, …);
    ``description`` feeds an image prompt or the slide-outline content; ``topic_ids`` /
    ``lecture_id`` / ``page`` scope figure reuse (the nearest page wins)."""
    kind: str
    params: Optional[dict[str, Any]] = None
    description: Optional[str] = None
    topic_ids: Optional[list[str]] = None
    lecture_id: Optional[str] = None
    page: Optional[int] = None          # reference page (figure-reuse prefers the nearest)
    item_id: Optional[str] = None

    @field_validator("kind")
    @classmethod
    def _kind_known(cls, v: str) -> str:
        return v if v in VISUAL_KINDS else "image"


class PlannedTask(BaseModel):
    """One task the planner emits."""
    type: TaskType
    query: Optional[str] = None
    level: str = "intermediate"
    length: str = "medium"
    assessment: Optional[AssessmentSpec] = None   # only on an assessment task (Q2)
    grade: Optional[GradeSpec] = None             # only on a grade task
    visual: Optional[VisualSpec] = None           # only on a visual task


class PlannerOutput(BaseModel):
    """The planner's whole reply: the tasks, plus an explicit answer-language request."""
    output_language: Optional[str] = None
    tasks: list[PlannedTask] = Field(default_factory=list)


class ItemDraft(BaseModel):
    """One raw question the generator returns — the LLM's wire shape, pre-validation. It
    has no topic_id/lecture_id (those are system-stamped from the bound slot/chunk);
    ``_build_item`` stamps them + converts options and produces the final
    ``AssessmentItem``."""
    question: Optional[str] = None
    type: Optional[QuestionType] = None
    options: Optional[list[str]] = None
    answer: Optional[str | list[str]] = None
    explanation: Optional[str] = None


class DraftBatch(BaseModel):
    """A whole assessment drafted in one pass — the structured-output schema."""
    items: list[ItemDraft] = Field(default_factory=list)


# --- grading shapes ---------------------------------------------------------

class SubmissionItem(BaseModel):
    """One answer to grade — the normalized grading input. The student's answer arrives
    in the request, never read from the DB (invariant 4); ``correct_answer`` comes inline
    or from ``fetch_rubric``. ``topic_id`` / ``lecture_id`` may be absent for inline
    free-text and are filled in ``prepare`` (invariant 8) so the mastery event can carry
    them (invariant 12)."""
    question: str = Field(min_length=1)
    student_answer: str = ""                     # blank = unanswered -> incorrect
    correct_answer: Optional[str] = None         # answer key / rubric text
    context: Optional[str] = None                # syllabus grounding when no key (transient, not persisted)
    type: Optional[QuestionType] = None
    topic_id: Optional[str] = None
    lecture_id: Optional[str] = None
    question_id: Optional[str] = None            # -> mastery_events.item_id (null for a live chat quiz)
    difficulty: Optional[str] = None
    bloom: Optional[str] = None
    points: Optional[float] = None


class GradedItem(BaseModel):
    """One graded answer -> a mastery_event + a line in the grade response. The
    invariant-12 spine (``topic_id`` / ``lecture_id``) and ``question_id`` are carried
    from the ``SubmissionItem``; ``mistake_tag_ids`` are filled after ``create_mistake_tag``
    canonicalizes ``mistake_labels``."""
    question: str
    graded: bool = True                          # False -> declined (no key + no syllabus evidence)
    correct: bool = False
    partial_score: Optional[float] = None        # [0,1]; None -> derive from ``correct``
    feedback: Optional[str] = None
    explanation: Optional[str] = None
    correct_answer: Optional[str] = None         # reference answer/rubric, surfaced on a wrong answer
    mistake_labels: list[str] = Field(default_factory=list)   # free-text, wrong answers only
    mistake_tag_ids: list[str] = Field(default_factory=list)  # canonical ids (post-lookup)
    confidence: Optional[float] = None
    topic_id: Optional[str] = None
    lecture_id: Optional[str] = None
    question_id: Optional[str] = None
    difficulty: Optional[str] = None
    bloom: Optional[str] = None
    points: Optional[float] = None


class GradeDraft(BaseModel):
    """One graded answer as the LLM returns it (wire shape, pre-canonicalization).

    ``id`` ties the grade back to its block: answers used to be matched by
    position, so a reply that skipped or reordered one graded a student against
    somebody else's question. Scores are bounded here rather than downstream —
    mastery clamped them, but the score shown to the student did not.
    """
    id: str = Field(default="", description="The block id being graded, copied verbatim.")
    correct: bool = Field(default=False, description="True only if fully correct.")
    partial_score: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="Credit earned: 1 = fully correct, 0 = wrong or blank.")
    feedback: Optional[str] = Field(
        default=None, description="One or two sentences to the student.")
    explanation: Optional[str] = Field(
        default=None, description="The correct reasoning, grounded in the block.")
    mistakes: list[str] = Field(
        default_factory=list,
        description="1-3 short, reusable error labels for a wrong answer; empty when correct.")
    confidence: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Confidence in this grade.")


class GradeBatch(BaseModel):
    """A whole submission graded in one pass — the structured-output schema."""
    items: list[GradeDraft] = Field(default_factory=list)


# --- render artifacts -------------------------------------------------------

class Flashcard(BaseModel):
    """One flashcard — the flip-UI shape ``gen_flashcards`` echoes back."""
    front: str
    back: str = ""


class ArtifactRef(BaseModel):
    """A rendered visual handed to the UI (the ``/ai/chat`` ``visuals`` list). ``s3_uri``
    for an image/chart/mindmap/pdf; ``cards`` for a flashcard flip UI; ``text`` for a slide
    outline (no file). ``figure_id`` / ``caption`` / ``reused`` appear on a reused syllabus
    figure; ``item_id`` links it to the assessment item it illustrates. Fields default to
    ``None`` and are dropped on dump, so each kind carries only what it has."""
    kind: str
    artifact_id: Optional[str] = None
    s3_uri: Optional[str] = None
    cards: Optional[list[Flashcard]] = None
    text: Optional[str] = None
    figure_id: Optional[str] = None
    caption: Optional[str] = None
    reused: Optional[bool] = None
    item_id: Optional[str] = None
