"""grade_sg — grade a submission and move mastery (AI-final).

    prepare -> fetch_rubric -> grade -> write_mastery

Private state (``GradeState``); the parent adapter (chat) or ``run_grade`` (endpoint)
hands it a normalized ``submissions`` list. The student's answers arrive in the request
and are never read from the DB (invariant 4); ``fetch_rubric`` reads only the *correct*
answers. Every mastery event carries ``topic_id`` + ``lecture_id`` (invariant 12);
``resolve_topics`` fills a missing topic (invariant 8). AI-final: mastery moves on grade
with no teacher-approval gate — a later teacher edit is a correction appended as a new
event. ``grade`` is ONE LLM call (score + feedback + mistake labels together);
mistake labels are canonicalized to tag ids deterministically.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, TypedDict

from langchain_core.messages import HumanMessage

from ...mastery import Mastery, MasteryParams, observation, update_aggregate
from ...models import GradeBatch, GradedItem, SubmissionItem
from ...prompts import GRADE_BATCH_PROMPT
from ...tools import (
    create_mistake_tag,
    fetch_rubric,
    get_mastery,
    get_topic_lectures,
    rag_search,
    resolve_topics,
    update_mastery,
    write_mastery_events,
)
from shared.language import language_directive, translate

from ...utils import degraded_message, extract_json, parse_structured

logger = logging.getLogger(__name__)

# Rubric fields (from fetch_rubric) merged onto a submission by position. Both
# `correct_answer_text` and `rubric` populate the answer key — a model answer wins, but an
# open-ended question that ships only a rubric is still gradable against it. `lecture_id`
# also maps (some backends send `lesson_id`).
_RUBRIC_MAP = {
    "correct_answer_text": "correct_answer", "rubric": "correct_answer",
    "question_text": "question", "question_type": "type",
    "lecture_id": "lecture_id", "lesson_id": "lecture_id", "difficulty": "difficulty",
    "bloom": "bloom", "points": "points",
}


class GradeState(TypedDict, total=False):
    # inputs
    submissions: list[dict[str, Any]]   # normalized: question/student_answer (+ optional key)
    subject_id: Optional[str]
    lecture_ids: Optional[list[str]]    # optional lecture scope for grounding; None -> whole subject
    user_id: Optional[str]
    language: str
    instructions: str                   # subject-specific prompt addendum (math), or ""
    kind: Optional[str]                 # quiz | homework (for fetch_rubric)
    assessment_id: Optional[str]        # DB reference -> fetch the answer key
    source: str                         # mastery_event source: practice | quiz | homework
    pass_threshold: float
    namespaces: list[str]               # syllabus scope for grounding a keyless answer
    min_score: float                    # RAG floor applied to grounding retrieval
    # produced
    items: list[SubmissionItem]
    graded: list[GradedItem]
    degraded: bool                      # true when the grade LLM call failed (no mastery moved)
    result: dict[str, Any]


# --- pure helpers (unit-tested directly) ------------------------------------

def _to_items(submissions: list[dict]) -> list[SubmissionItem]:
    """Coerce raw submission dicts into typed ``SubmissionItem``s, dropping any with no
    question."""
    out: list[SubmissionItem] = []
    for s in submissions or []:
        if not (s or {}).get("question"):
            continue
        try:
            out.append(SubmissionItem(
                **{k: s[k] for k in SubmissionItem.model_fields if k in s}))
        except Exception:
            continue
    return out


def _merge_rubric(items: list[SubmissionItem], rubric: list[dict]) -> list[SubmissionItem]:
    """Fill each item's answer key + invariant-12 spine from the rubric row at the same
    position (the submission order mirrors the assessment order). Values already set on
    the item win — an inline correct answer is not overwritten."""
    for i, item in enumerate(items):
        if i >= len(rubric):
            break
        row = rubric[i] or {}
        for src, dst in _RUBRIC_MAP.items():
            if row.get(src) is not None and getattr(item, dst, None) in (None, ""):
                setattr(item, dst, row[src])
    return items


def _resolve_missing_topics(
    items: list[SubmissionItem], subject_id: Optional[str], config,
    lecture_ids: Optional[list[str]] = None,
) -> None:
    """Stamp a topic_id on items that lack one (invariant 8), matching the question text, then
    backfill lecture_id from that topic (a topic belongs to a lecture). This lets a graded
    answer with a resolvable topic move mastery (invariant 12) even when the caller supplied no
    lecture — e.g. a student's free-text practice in chat. Only a question that resolves to NO
    topic stays unattributed (feedback only, no mastery)."""
    for item in items:
        if item.topic_id:
            continue
        try:
            hits = resolve_topics(item.question, subject_id, lecture_ids=lecture_ids,
                                  config=config)
        except Exception:
            logger.warning("resolve_topics failed for a submission", exc_info=True)
            continue
        if hits:
            item.topic_id = hits[0].topic_id
    # Backfill lecture_id from the resolved topic for any item still missing it.
    need = list({it.topic_id for it in items if it.topic_id and not it.lecture_id})
    if need:
        try:
            lectures = get_topic_lectures(need, config=config)
            for it in items:
                if it.topic_id and not it.lecture_id:
                    it.lecture_id = lectures.get(it.topic_id)
        except Exception:
            logger.warning("lecture_id backfill failed", exc_info=True)


def _ground_answers(
    items: list[SubmissionItem], state: GradeState, config
) -> None:
    """Retrieve syllabus context for answers that need grounding: any KEYLESS answer, plus
    keyed OPEN-ENDED answers (essay/short_answer, config-gated) whose model answer is not
    exhaustive — the context lets a valid alternative earn credit. Exact types (mcq/true_false)
    with a key are graded strictly against it, no context. Invariant 1/9; min_score floor. No
    namespaces (e.g. the endpoint path) -> no grounding."""
    namespaces = state.get("namespaces")
    if not namespaces:
        return
    for item in items:
        # A key/rubric wins: when the item already has a reference answer (model answer OR
        # rubric, via _merge_rubric) it is graded against THAT — RAG is only the fallback for a
        # KEYLESS answer. (Was: open-ended got RAG even with a key.)
        if item.context or item.correct_answer:
            continue
        floor = state.get("min_score", 0.0)

        def _grounded(topic_ids):
            chunks = rag_search(
                item.question, namespaces, subject_id=state.get("subject_id"),
                topic_ids=topic_ids, lecture_ids=state.get("lecture_ids") or None,
                config=config)
            return [c.text for c in chunks if c.score >= floor]

        try:
            grounded = _grounded([item.topic_id] if item.topic_id else None)
            # Soft: a topic-filtered miss retries unscoped, so a slightly-off topic can't
            # wrongly decline an answer whose material IS in the subject (same fix as retrieve).
            if not grounded and item.topic_id:
                grounded = _grounded(None)
        except Exception:
            logger.warning("grade grounding RAG failed", exc_info=True)
            continue
        if grounded:
            item.context = "\n\n".join(grounded)


def _grade_blocks(items: list[SubmissionItem]) -> str:
    """One block per answer for the batch prompt (order preserved). Context is shown whenever
    grounded (keyless, or a keyed open-ended answer) so the model grades against the material."""
    blocks = []
    for i, it in enumerate(items, 1):
        ref = it.correct_answer or "(none given)"
        ctx = f"\nSyllabus context: {it.context}" if it.context else ""
        blocks.append(
            f"Block {i}: type={it.type or 'short_answer'}\n"
            f"Question: {it.question}\n"
            f"Reference answer: {ref}{ctx}\n"
            f"Student answer: {it.student_answer or '(blank)'}"
        )
    return "\n\n".join(blocks)


def _correct(draft, pass_threshold: float) -> bool:
    """Prefer partial_score against the pass threshold; fall back to the model's flag."""
    if draft.partial_score is not None:
        return float(draft.partial_score) >= pass_threshold
    return bool(draft.correct)


def _build_graded(item: SubmissionItem, draft, pass_threshold: float) -> GradedItem:
    """Combine a submission's spine (invariant 12 + item_id/difficulty) with the model's
    grade draft into a typed ``GradedItem``."""
    correct = _correct(draft, pass_threshold)
    return GradedItem(
        question=item.question,
        correct=correct,
        partial_score=draft.partial_score,
        feedback=draft.feedback,
        explanation=draft.explanation,
        correct_answer=item.correct_answer,
        mistake_labels=[] if correct else list(draft.mistakes or []),
        confidence=draft.confidence,
        topic_id=item.topic_id,
        lecture_id=item.lecture_id,
        question_id=item.question_id,
        difficulty=item.difficulty,
        bloom=item.bloom,
        points=item.points,
    )


def _gradable(item: SubmissionItem) -> bool:
    """An answer can be graded only against a reference key or retrieved syllabus context —
    never the model's own knowledge."""
    return bool(item.correct_answer or item.context)


# Decline message when there is no key and no syllabus evidence (en is the fallback).



def _not_graded(item: SubmissionItem, feedback: str) -> GradedItem:
    """A not-graded result: no score, no mastery, a clear message (declined or degraded)."""
    return GradedItem(
        question=item.question, graded=False, correct=False, partial_score=None,
        feedback=feedback,
        topic_id=item.topic_id, lecture_id=item.lecture_id, question_id=item.question_id,
        difficulty=item.difficulty, bloom=item.bloom, points=item.points,
    )


def _declined(item: SubmissionItem, language: str) -> GradedItem:
    """Declined: no reference key and no syllabus context — decline rather than grade on the
    model's own knowledge."""
    return _not_graded(item, translate("grade.no_evidence", language))


def _stub_draft(item: SubmissionItem):
    """Deterministic grade with no LLM/key: exact (case-insensitive) match of the student
    answer to the reference answer. Off-key items are marked incorrect."""
    from ...models import GradeDraft
    key = (item.correct_answer or "").strip().lower()
    ans = (item.student_answer or "").strip().lower()
    ok = bool(key) and ans == key
    return GradeDraft(
        correct=ok, partial_score=1.0 if ok else 0.0,
        feedback="graded" if ok else "does not match the reference answer",
        explanation=item.correct_answer, mistakes=[] if ok else ["incorrect answer"],
        confidence=1.0 if key else 0.5,
    )


def _canonicalize(graded: list[GradedItem], subject_id: Optional[str], config) -> None:
    """Map each wrong answer's free-text mistake labels to canonical tag ids
    (``create_mistake_tag`` reuses on exact label). Failures are non-fatal — a missing
    tag just means an unlabelled mistake."""
    cache: dict[str, str] = {}
    for g in graded:
        for label in g.mistake_labels:
            key = (label or "").strip().lower()
            if not key:
                continue
            if key not in cache:
                try:
                    cache[key] = create_mistake_tag(
                        label, subject_id=subject_id, config=config)
                except Exception:
                    logger.warning("create_mistake_tag failed for %r", label, exc_info=True)
                    continue
            g.mistake_tag_ids.append(cache[key])


def _event_from(g: GradedItem, user_id: str, source: str) -> Optional[dict]:
    """A mastery_event dict for one graded answer, or None when it lacks the invariant-12
    spine (topic_id + lecture_id) and so cannot be recorded. ``original_score`` = the AI's
    score now, preserved for a later teacher correction."""
    if not g.graded or not g.topic_id or not g.lecture_id:
        return None
    score = g.partial_score if g.partial_score is not None else (1.0 if g.correct else 0.0)
    return {
        "user_id": user_id, "item_id": g.question_id, "topic_id": g.topic_id,
        "lecture_id": g.lecture_id, "correct": g.correct, "partial_score": score,
        "original_score": score, "difficulty": g.difficulty, "bloom_level": g.bloom,
        "mistake_tag_ids": g.mistake_tag_ids or None, "source": source,
    }


def _rollup_mastery(events: list[dict], user_id: str, config) -> int:
    """Roll one assessment's events into mastery: a topic's answers are AVERAGED into a
    single observation and applied as ONE EMA step (attempts counts assessments, not
    questions, and question order inside the assessment doesn't matter). Returns the
    number of topics updated."""
    params = MasteryParams.from_config(config)
    topic_ids = list({e["topic_id"] for e in events})
    try:
        priors = {r["topic_id"]: r for r in get_mastery(user_id, topic_ids, config=config)}
    except Exception:
        logger.warning("get_mastery failed; starting mastery from zero", exc_info=True)
        priors = {}
    by_topic: dict[str, list[dict]] = {}
    for e in events:
        by_topic.setdefault(e["topic_id"], []).append(e)
    for tid, evs in by_topic.items():
        obs = sum(observation(e) for e in evs) / len(evs)
        # Per-event weight uses that event's own outcome (asymmetric map), then averaged.
        weight = sum(params.weight_for(e.get("difficulty"), observation(e))
                     for e in evs) / len(evs)
        p = priors.get(tid)
        prior = (Mastery(score=p["score"], attempts=p["attempts"],
                         trend=p.get("trend"), confidence=p.get("confidence", 0.0))
                 if p else Mastery())
        m = update_aggregate(prior, obs, params, difficulty_weight=weight)
        update_mastery(user_id, tid, m, config=config)
    return len(by_topic)


def _summary(graded: list[GradedItem]) -> dict:
    """Roll the graded items into an overall score. Declined items (no key + no syllabus
    evidence) are excluded from the score and reported separately, never counted as wrong."""
    scored = [g for g in graded if g.graded]
    n = len(scored)
    earned = sum((g.partial_score if g.partial_score is not None
                  else (1.0 if g.correct else 0.0)) for g in scored)
    return {"count": n, "score": round(earned, 3), "max_score": float(n),
            "correct": sum(1 for g in scored if g.correct),
            "declined": sum(1 for g in graded if not g.graded)}


# --- nodes ------------------------------------------------------------------

def _prepare(state: GradeState) -> dict:
    items = _to_items(state.get("submissions") or [])
    return {"items": items}


def _build_fetch_rubric(config):
    def _node(state: GradeState) -> dict:
        items = state.get("items", [])
        assessment_id = state.get("assessment_id")
        if assessment_id and items:
            try:
                rubric = fetch_rubric(state.get("kind") or "quiz", assessment_id, config=config)
                _merge_rubric(items, rubric)
            except Exception:
                logger.warning("fetch_rubric failed; grading on inline answers only",
                               exc_info=True)
        _resolve_missing_topics(items, state.get("subject_id"), config,
                                lecture_ids=state.get("lecture_ids") or None)
        _ground_answers(items, state, config)
        logger.info("fetch_rubric", extra={
            "items": len(items),
            "grounded": sum(1 for it in items if it.topic_id and it.lecture_id),
            "context_grounded": sum(1 for it in items if it.context)})
        return {"items": items}

    return _node


def _build_grade(llm, config):
    def _node(state: GradeState) -> dict:
        items = state.get("items", [])
        if not items:
            logger.info("grade", extra={"items": 0})
            return {"graded": []}
        threshold = state.get("pass_threshold", 0.5)
        language = state.get("language", "en")
        # Only answers WITH a reference key or syllabus context are graded — never on the model's
        # own knowledge. An answer with neither is declined (no score, no mastery, clear message).
        gradable = [it for it in items if _gradable(it)]
        drafts: list = []
        degraded = False
        if gradable:
            if llm is None:                     # dev/test: no key -> deterministic stub
                drafts = _stub_drafts(gradable)
            else:
                try:
                    drafts = _batch_grade(llm, gradable, state, config)
                except Exception:
                    # Runtime failure: degrade rather than stub — no mastery moves on a grade
                    # we couldn't actually compute.
                    logger.error("grade failed; degrading (no mastery)", exc_info=True)
                    degraded = True
        graded, gi = [], 0
        for it in items:
            if not _gradable(it):
                graded.append(_declined(it, language))          # no key + no context
            elif degraded:
                graded.append(_not_graded(it, translate("common.degraded", language)))
            else:
                graded.append(_build_graded(it, drafts[gi], threshold))
                gi += 1
        _canonicalize(graded, state.get("subject_id"), config)
        logger.info("grade", extra={
            "graded": sum(1 for g in graded if g.graded),
            "declined": sum(1 for g in graded if not g.graded),
            "wrong": sum(1 for g in graded if g.graded and not g.correct),
            "degraded": degraded, "calls": 0 if llm is None else 1})
        return {"graded": graded, "degraded": degraded}

    return _node


def _stub_drafts(items: list[SubmissionItem]) -> list:
    return [_stub_draft(it) for it in items]


def _batch_grade(llm, items, state: GradeState, config=None) -> list:
    """Grade the whole submission in ONE LLM call (score + feedback + mistake labels). A
    validated ``GradeBatch`` via structured output (JSON-parse fallback); drafts map back to
    items by position. A real call failure propagates so the grade node degrades (no stub,
    no mastery moved)."""
    prompt_text = GRADE_BATCH_PROMPT.format(
        language=language_directive(state.get("language", "en")),
        instructions=state.get("instructions", ""),
        n=len(items),
        items_block=_grade_blocks(items),
    )
    messages = [HumanMessage(content=prompt_text)]
    data = parse_structured(llm, GradeBatch, messages, config, "grade")
    raw = data.get("items") if isinstance(data, dict) else None

    raw = raw or []
    drafts = []
    for i, item in enumerate(items):
        r = raw[i] if i < len(raw) else None
        if isinstance(r, dict):
            from ...models import GradeDraft
            try:
                drafts.append(GradeDraft(
                    **{k: r[k] for k in GradeDraft.model_fields if k in r and r[k] is not None}))
                continue
            except Exception:
                pass
        drafts.append(_stub_draft(item))
    return drafts


def _build_write_mastery(config):
    def _node(state: GradeState) -> dict:
        graded = state.get("graded", [])
        user_id = state.get("user_id")
        source = state.get("source", "practice")
        events, topics = [], 0
        if user_id:
            events = [e for g in graded if (e := _event_from(g, user_id, source))]
            if events:
                try:
                    write_mastery_events(events, config=config)
                    topics = _rollup_mastery(events, user_id, config)
                except Exception:
                    logger.error("mastery write failed", exc_info=True)
        else:
            logger.warning("grade with no user_id; feedback only, no mastery write")
        result = {
            "graded": [g.model_dump() for g in graded],
            "summary": _summary(graded),
            "mastery_events": len(events),
            "topics_updated": topics,
            "source": source,
            "degraded": state.get("degraded", False),   # LLM failure -> friendly note in compose
        }
        logger.info("write_mastery", extra={
            "events": len(events), "topics": topics, "user": bool(user_id)})
        return {"result": result}

    return _node


# --- build ------------------------------------------------------------------

def build_grade_sg(*, config, llm=None):
    from langgraph.graph import END, START, StateGraph

    graph = StateGraph(GradeState)
    graph.add_node("prepare", _prepare)
    graph.add_node("fetch_rubric", _build_fetch_rubric(config))
    graph.add_node("grade", _build_grade(llm, config))
    graph.add_node("write_mastery", _build_write_mastery(config))

    graph.add_edge(START, "prepare")
    graph.add_edge("prepare", "fetch_rubric")
    graph.add_edge("fetch_rubric", "grade")
    graph.add_edge("grade", "write_mastery")
    graph.add_edge("write_mastery", END)
    return graph.compile()
