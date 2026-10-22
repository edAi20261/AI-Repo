"""grade_sg — grade a submission and move mastery (AI-final).

    prepare -> fetch_rubric -> grade -> write_mastery
            -> walkthrough          (a review of a saved assessment with no answers sent)

Private state (``GradeState``); the parent adapter (chat) or ``run_grade`` (endpoint)
hands it a normalized ``submissions`` list. The student's answers arrive in the request
and are never read from the DB (invariant 4); ``fetch_rubric`` reads only the *correct*
answers. Every mastery event carries ``topic_id`` + ``lecture_id`` (invariant 12);
``resolve_topics`` fills a missing topic (invariant 8). A grade the model is sure of
moves mastery at once. One it is unsure of, or one marked on an answer that could not be
read clearly, is held for a teacher: it is returned with its ``review`` reasons and moves
nothing until ``record_reviewed`` writes the teacher's decision. A wrong mark the model
doubted would otherwise become a student's weak topic with nobody having looked at it.
``grade`` is ONE LLM call (score + feedback + mistake labels together); mistake labels
are canonicalized to tag ids deterministically.
"""

from __future__ import annotations

import logging
from typing import Any, TypedDict

from langchain_core.messages import HumanMessage

from shared.structured import align_by_id, call_structured, schema_hint

from ...mastery import Mastery, MasteryParams, observation, update_aggregate
from ...models import (
    GradeBatch,
    GradedItem,
    ReviewedGrade,
    ReviewReason,
    SubmissionItem,
    Walkthrough,
    WalkthroughStep,
)
from ...prompts import GRADE_BATCH_PROMPT, GRADE_WALKTHROUGH_PROMPT
from ...retrieval import DEFAULT_MIN_SCORE
from ...tools import (
    create_mistake_tag,
    fetch_rubric,
    get_item,
    get_mastery,
    get_topic_lectures,
    rag_search,
    record_graded_evidence,
    resolve_topics,
)

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
    review: bool                        # checking marked work: answers explained, none recorded
    subject_id: str | None
    lecture_ids: list[str] | None    # optional lecture scope for grounding; None -> whole subject
    user_id: str | None
    language: str
    instructions: str                   # subject-specific prompt addendum (math), or ""
    kind: str | None                 # quiz | homework (for fetch_rubric)
    assessment_id: str | None        # DB reference -> fetch the answer key
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
    items: list[SubmissionItem], subject_id: str | None, config,
    lecture_ids: list[str] | None = None,
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
        if item.context or item.correct_answer or item.known_score is not None:
            continue
        floor = state.get("min_score", DEFAULT_MIN_SCORE)

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


def _block_id(index: int) -> str:
    """The id a block is labelled with, and the grade must come back carrying."""
    return f"b{index}"


def _grade_blocks(items: list[SubmissionItem]) -> str:
    """One block per answer for the batch prompt (order preserved). Context is shown whenever
    grounded (keyless, or a keyed open-ended answer) so the model grades against the material."""
    blocks = []
    for i, it in enumerate(items, 1):
        ref = it.correct_answer or "(none given)"
        ctx = f"\nSyllabus context: {it.context}" if it.context else ""
        blocks.append(
            f"Block id={_block_id(i)}: type={it.type or 'short_answer'}\n"
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


def _scored(item: SubmissionItem, pass_threshold: float) -> GradedItem:
    """An answer the platform already scored: its score is the evidence, as given."""
    return GradedItem(
        question=item.question, correct=item.known_score >= pass_threshold,
        partial_score=item.known_score, correct_answer=item.correct_answer,
        topic_id=item.topic_id, lecture_id=item.lecture_id, question_id=item.question_id,
        difficulty=item.difficulty, bloom=item.bloom, points=item.points,
    )


def _review(item: SubmissionItem, confidence: float | None,
            review_below: float) -> list[ReviewReason]:
    """Why a model's grade needs a teacher before it counts. A missing confidence is
    no evidence of certainty, so it is held like a low one."""
    reasons: list[ReviewReason] = []
    if confidence is None or confidence < review_below:
        reasons.append("low_confidence")
    if item.unclear_reading:
        reasons.append("unclear_reading")
    return reasons


def _gradable(item: SubmissionItem) -> bool:
    """An answer can be graded only against a reference key or retrieved syllabus context —
    never the model's own knowledge."""
    return bool(item.correct_answer or item.context)


def _not_graded(item: SubmissionItem) -> GradedItem:
    """A not-graded result: no score, no mastery. ``graded=False`` is the message; the
    caller says it in the reader's language."""
    return GradedItem(
        question=item.question, graded=False, correct=False, partial_score=None,
        feedback="",
        topic_id=item.topic_id, lecture_id=item.lecture_id, question_id=item.question_id,
        difficulty=item.difficulty, bloom=item.bloom, points=item.points,
    )


def _declined(item: SubmissionItem) -> GradedItem:
    """Declined: no reference key and no syllabus context — decline rather than grade on the
    model's own knowledge."""
    return _not_graded(item)


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


def _canonicalize(graded: list[GradedItem], subject_id: str | None, config) -> None:
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


def _event_from(g: GradedItem, user_id: str, source: str) -> dict | None:
    """A mastery_event dict for one graded answer, or None when it lacks the invariant-12
    spine (topic_id + lecture_id) and so cannot be recorded. ``original_score`` = the AI's
    score now, preserved for a later teacher correction."""
    if not g.graded or g.needs_review or not g.topic_id or not g.lecture_id:
        return None
    score = g.partial_score if g.partial_score is not None else (1.0 if g.correct else 0.0)
    return {
        "user_id": user_id, "item_id": g.question_id, "topic_id": g.topic_id,
        "lecture_id": g.lecture_id, "correct": g.correct, "partial_score": score,
        "original_score": score, "difficulty": g.difficulty, "bloom_level": g.bloom,
        "mistake_tag_ids": g.mistake_tag_ids or None, "source": source,
    }


def _next_mastery(events: list[dict], user_id: str, config) -> dict[str, Mastery]:
    """The mastery each topic moves to after one assessment: a topic's answers are
    AVERAGED into a single observation and applied as ONE EMA step (attempts counts
    assessments, not questions, and question order inside the assessment doesn't matter).

    Raises when the prior cannot be read. Starting from zero instead would upsert a
    fresh score over the student's real history.
    """
    params = MasteryParams.from_config(config)
    topic_ids = list({e["topic_id"] for e in events})
    priors = {r["topic_id"]: r for r in get_mastery(user_id, topic_ids, config=config)}
    by_topic: dict[str, list[dict]] = {}
    for e in events:
        by_topic.setdefault(e["topic_id"], []).append(e)
    nxt: dict[str, Mastery] = {}
    for tid, evs in by_topic.items():
        obs = sum(observation(e) for e in evs) / len(evs)
        # Per-event weight uses that event's own outcome (asymmetric map), then averaged.
        weight = sum(params.weight_for(e.get("difficulty"), observation(e))
                     for e in evs) / len(evs)
        p = priors.get(tid)
        prior = (Mastery(score=p["score"], attempts=p["attempts"],
                         trend=p.get("trend"), confidence=p.get("confidence", 0.0))
                 if p else Mastery())
        nxt[tid] = update_aggregate(prior, obs, params, difficulty_weight=weight)
    return nxt


def _summary(graded: list[GradedItem]) -> dict:
    """Roll the graded items into an overall score. Declined items (no key + no syllabus
    evidence) are excluded from the score and reported separately, never counted as wrong."""
    scored = [g for g in graded if g.graded]
    n = len(scored)
    earned = sum((g.partial_score if g.partial_score is not None
                  else (1.0 if g.correct else 0.0)) for g in scored)
    return {"count": n, "score": round(earned, 3), "max_score": float(n),
            "correct": sum(1 for g in scored if g.correct),
            "declined": sum(1 for g in graded if not g.graded),
            "needs_review": sum(1 for g in scored if g.needs_review)}


def _question_text(question: dict) -> str:
    return (question.get("question_text") or question.get("question") or "").strip()


def _option_text(option: Any) -> str:
    if isinstance(option, dict):
        return str(option.get("option_text") or option.get("text") or "").strip()
    return str(option or "").strip()


def _answer_key_of(question: dict) -> str:
    """A stored question's correct answer, whichever way the platform keeps it: model
    answer text, the options marked correct, or a rubric."""
    if question.get("correct_answer_text"):
        return str(question["correct_answer_text"]).strip()
    correct = [_option_text(o) for o in question.get("options") or []
               if isinstance(o, dict) and o.get("is_correct")]
    if any(correct):
        return "; ".join(c for c in correct if c)
    return str(question.get("rubric") or "").strip()


def _walkthrough_blocks(questions: list[dict]) -> str:
    blocks = []
    for i, q in enumerate(questions, 1):
        options = [t for t in (_option_text(o) for o in q.get("options") or []) if t]
        listed = f"\nOptions: {'; '.join(options)}" if options else ""
        blocks.append(
            f"Block id={_block_id(i)}: type={q.get('question_type') or 'short_answer'}\n"
            f"Question: {_question_text(q)}{listed}\n"
            f"Correct answer: {_answer_key_of(q) or '(none given)'}\n"
            f"Explanation: {q.get('explanation') or '(none given)'}"
        )
    return "\n\n".join(blocks)


def _walk(llm, questions: list[dict], state: GradeState, config) -> Walkthrough:
    """Explain every question in ONE call; a reply that skips one is sent back."""
    if llm is None:
        return Walkthrough.model_construct(note="", items=[
            WalkthroughStep.model_construct(id=_block_id(i), explanation=q.get("explanation") or "")
            for i, q in enumerate(questions, 1)])
    prompt_text = GRADE_WALKTHROUGH_PROMPT.render(
        schema=schema_hint(Walkthrough),
        language=state.get("language", "en"),
        instructions=state.get("instructions", ""),
        n=len(questions),
        items_block=_walkthrough_blocks(questions),
    )
    expected = [_block_id(i) for i in range(1, len(questions) + 1)]
    walkthrough = call_structured(
        Walkthrough,
        [HumanMessage(content=prompt_text)],
        task="grade",
        config=config,
        llm=llm,
        label="grade_walkthrough",
        validate=lambda w: align_by_id(expected, w.items, what="walkthrough"),
    )
    walkthrough.items = align_by_id(expected, walkthrough.items, what="walkthrough")
    return walkthrough


def _walkthrough_result(questions: list[dict], walkthrough: Walkthrough | None,
                        source: str) -> dict:
    result = {
        "graded": [], "summary": _summary([]), "mastery_events": 0, "topics_updated": 0,
        "mastery_recorded": True, "source": source, "degraded": walkthrough is None,
    }
    if walkthrough is not None:
        result["walkthrough"] = {"note": walkthrough.note, "items": [
            {"question": _question_text(q), "correct_answer": _answer_key_of(q) or None,
             "explanation": step.explanation or None, "question_id": q.get("id")}
            for q, step in zip(questions, walkthrough.items)]}
    return result


# --- nodes ------------------------------------------------------------------

def _prepare(state: GradeState) -> dict:
    items = _to_items(state.get("submissions") or [])
    return {"items": items}


def _after_prepare(state: GradeState) -> str:
    """A review that arrived without the student's answers has nothing to mark. Marking
    the chat message against the first question graded a request as an answer."""
    if state.get("review") and not state.get("items") and state.get("assessment_id"):
        return "walkthrough"
    return "fetch_rubric"


def _build_walkthrough(llm, config):
    def _node(state: GradeState) -> dict:
        kind = state.get("kind") or "quiz"
        source = state.get("source", "practice")
        try:
            questions = [q for q in get_item(kind, state["assessment_id"], config=config)
                         if _question_text(q)]
        except Exception:
            logger.error("walkthrough: questions not loaded", exc_info=True,
                         extra={"kind": kind})
            return {"result": _walkthrough_result([], None, source)}
        if not questions:
            logger.error("walkthrough: the assessment has no questions", extra={"kind": kind})
            return {"result": _walkthrough_result([], None, source)}
        try:
            walkthrough = _walk(llm, questions, state, config)
        except Exception:
            logger.error("walkthrough failed; degrading", exc_info=True,
                         extra={"questions": len(questions)})
            walkthrough = None
        logger.info("walkthrough", extra={
            "questions": len(questions), "degraded": walkthrough is None,
            "calls": 0 if llm is None else 1})
        return {"result": _walkthrough_result(questions, walkthrough, source)}

    return _node


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
    review_below = config.get("grade.review_below")

    def _node(state: GradeState) -> dict:
        items = state.get("items", [])
        if not items:
            logger.info("grade", extra={"items": 0})
            return {"graded": []}
        threshold = state.get("pass_threshold", 0.5)
        # Only answers WITH a reference key or syllabus context are graded — never on the model's
        # own knowledge. An answer with neither is declined (no score, no mastery, clear message).
        gradable = [it for it in items if it.known_score is None and _gradable(it)]
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
            if it.known_score is not None:
                graded.append(_scored(it, threshold))
            elif not _gradable(it):
                graded.append(_declined(it))          # no key + no context
            elif degraded:
                graded.append(_not_graded(it))
            else:
                g = _build_graded(it, drafts[gi], threshold)
                g.review = _review(it, g.confidence, review_below)
                graded.append(g)
                gi += 1
        _canonicalize(graded, state.get("subject_id"), config)
        logger.info("grade", extra={
            "graded": sum(1 for g in graded if g.graded),
            "declined": sum(1 for g in graded if not g.graded),
            "wrong": sum(1 for g in graded if g.graded and not g.correct),
            "needs_review": sum(1 for g in graded if g.needs_review),
            "scored_upstream": sum(1 for it in items if it.known_score is not None),
            "degraded": degraded, "calls": 0 if llm is None or not gradable else 1})
        return {"graded": graded, "degraded": degraded}

    return _node


def _stub_drafts(items: list[SubmissionItem]) -> list:
    return [_stub_draft(it) for it in items]


def _batch_grade(llm, items, state: GradeState, config=None) -> list:
    """Grade the whole submission in ONE LLM call (score + feedback + mistake labels).

    Grades come back carrying their block id and are matched on it, so a reply
    that drops or reorders an answer is corrected rather than silently attributed
    to the wrong question. Any failure propagates: the node degrades, and no
    mastery moves on a grade we could not actually compute.
    """
    prompt_text = GRADE_BATCH_PROMPT.render(
        schema=schema_hint(GradeBatch),
        language=state.get("language", "en"),
        instructions=state.get("instructions", ""),
        n=len(items),
        items_block=_grade_blocks(items),
    )
    expected = [_block_id(i) for i in range(1, len(items) + 1)]
    batch = call_structured(
        GradeBatch,
        [HumanMessage(content=prompt_text)],
        task="grade",
        config=config,
        llm=llm,
        label="grade",
        validate=lambda b: align_by_id(expected, b.items, what="grade"),
    )
    return align_by_id(expected, batch.items, what="grade")


def _build_write_mastery(config):
    def _node(state: GradeState) -> dict:
        graded = state.get("graded", [])
        user_id = state.get("user_id")
        source = state.get("source", "practice")
        events: list[dict] = []
        written, topics, recorded = 0, 0, True
        if user_id:
            events = [e for g in graded if (e := _event_from(g, user_id, source))]
            if events:
                try:
                    aggregates = _next_mastery(events, user_id, config)
                    written = len(record_graded_evidence(user_id, events, aggregates,
                                                         config=config))
                    topics = len(aggregates)
                except Exception:
                    recorded = False
                    logger.error("mastery not recorded", exc_info=True,
                                 extra={"events": len(events), "source": source})
        else:
            logger.warning("grade with no user_id; feedback only, no mastery write")
        result = {
            "graded": [g.model_dump() for g in graded],
            "summary": _summary(graded),
            "mastery_events": written,
            "topics_updated": topics,
            "mastery_recorded": recorded,
            "source": source,
            "degraded": state.get("degraded", False),   # LLM failure -> friendly note in compose
        }
        logger.info("write_mastery", extra={
            "events": written, "topics": topics, "recorded": recorded, "user": bool(user_id)})
        return {"result": result}

    return _node


def record_reviewed(user_id: str, reviewed: list[ReviewedGrade], *, source: str,
                    pass_threshold: float, config) -> dict:
    """Move mastery on grades a teacher has decided.

    Each event keeps the model's score as ``original_score`` beside the teacher's, so
    how often teachers overturn held grades stays readable from the evidence."""
    events = [{
        "user_id": user_id, "item_id": r.question_id, "topic_id": r.topic_id,
        "lecture_id": r.lecture_id, "correct": r.score >= pass_threshold,
        "partial_score": r.score, "original_score": r.ai_score,
        "difficulty": r.difficulty, "bloom_level": r.bloom,
        "mistake_tag_ids": r.mistake_tag_ids or None, "source": source,
    } for r in reviewed]
    try:
        aggregates = _next_mastery(events, user_id, config)
        written = record_graded_evidence(user_id, events, aggregates, config=config)
    except Exception:
        logger.error("reviewed grades not recorded", exc_info=True,
                     extra={"events": len(events), "source": source})
        return {"mastery_events": 0, "topics_updated": 0, "mastery_recorded": False}
    overturned = sum(1 for r in reviewed if r.score != r.ai_score)
    logger.info("reviewed grades recorded", extra={
        "events": len(written), "topics": len(aggregates), "overturned": overturned})
    return {"mastery_events": len(written), "topics_updated": len(aggregates),
            "mastery_recorded": True}


# --- build ------------------------------------------------------------------

def build_grade_sg(*, config, llm=None):
    from langgraph.graph import END, START, StateGraph

    graph = StateGraph(GradeState)
    graph.add_node("prepare", _prepare)
    graph.add_node("fetch_rubric", _build_fetch_rubric(config))
    graph.add_node("grade", _build_grade(llm, config))
    graph.add_node("write_mastery", _build_write_mastery(config))
    graph.add_node("walkthrough", _build_walkthrough(llm, config))

    graph.add_edge(START, "prepare")
    graph.add_conditional_edges("prepare", _after_prepare, ["fetch_rubric", "walkthrough"])
    graph.add_edge("walkthrough", END)
    graph.add_edge("fetch_rubric", "grade")
    graph.add_edge("grade", "write_mastery")
    graph.add_edge("write_mastery", END)
    return graph.compile()
