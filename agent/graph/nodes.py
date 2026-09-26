"""Parent graph nodes: classify · load_context · understand · converse · plan · dispatch ·
compose · verify_language · persist.

Node callables that need config, an LLM or a store are built by ``build_*`` factories so
the dependency is captured once at compile time, not looked up per turn.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Optional

from langchain_core.messages import HumanMessage

from shared.errors import UnprocessableError
from shared.language import LanguageDecision, LanguagePolicy, name_of
from shared.language.verify import (
    DEFAULT_MIN_ANSWER_CHARS,
    DEFAULT_MIN_CONFIDENCE,
    verify_answer,
)
from shared.logging import clip
from shared.structured import call_structured

from ..citations import cited_numbers, renumber, strip_markers
from ..errors import TutorUnavailable
from ..conversation.store import REMEMBERING_TYPES
from ..conversation.transcript import recover
from ..conversation import (
    Conversation,
    ConversationStore,
    HistoryWindow,
    PendingAssessment,
    PendingQuestion,
    SessionState,
    Turn,
    fold_if_due,
)
from ..models import (
    AssessmentSpec,
    Citation,
    GradeSpec,
    PlannedTask,
    PrereqTopic,
    Style,
    TurnUnderstanding,
    VisualSpec,
)
from ..prompts import (
    COMPOSE_PROMPT,
    CONVERSE_PROMPT,
    FORMATTING_CONTRACT,
    LANGUAGE_REPAIR_PROMPT,
    NO_MATERIAL_PROMPT,
    TASK_DESCRIPTIONS,
    UNDERSTAND_PROMPT,
)
from ..retrieval import DEFAULT_MIN_SCORE
from ..scope import lecture_filter
from ..subjects import SubjectDirectory
from ..learner import standing
from ..profile import Carryover, carried_over, record
from ..tools import rag_search, resolve_topics
from .pathways import build_namespaces, resolve_style

logger = logging.getLogger(__name__)


# --- classify ---------------------------------------------------------------

def build_classify(config, store: ConversationStore, window: HistoryWindow,
                   subjects: Optional[SubjectDirectory] = None):
    known_pathways = set((config.get("agent.pathways", {}) or {}).keys())

    def classify(state) -> dict:
        user_type = state.get("user_type")
        pathway = user_type if user_type in known_pathways else None
        if pathway is None:
            logger.warning("classify: unknown user_type %r", user_type)
            raise UnprocessableError(
                f"user_type must be one of {', '.join(sorted(known_pathways))}.",
                details={"user_type": user_type})
        user_id, session_id = state.get("user_id"), state.get("session_id")
        conversation = store.load(user_id, session_id)
        # A thread the student can still see, that this service can no longer
        # remember: what we keep expires at ttl_hours, the transcript never
        # does. Read it back rather than meet them as a stranger inside a
        # conversation they are looking at.
        if conversation.is_new and store.remembers(user_id, pathway, session_id):
            recovered, recovered_state = recover(session_id, turns=window.recent_turns,
                                                 config=config)
            if recovered and store.restore(user_id, pathway, session_id,
                                           recovered, recovered_state):
                conversation = store.load(user_id, session_id)
        # Only when there is nothing better to go on. A conversation with turns
        # in it has the student in front of it; what they did last week is two
        # queries for something the last six turns already say better.
        carried = (carried_over(user_id, subjects, config=config)
                   if conversation.is_new and pathway == "student" else Carryover())
        logger.info("classify", extra={
            "pathway": pathway, "remembered_turns": len(conversation.turns),
            "summarised": bool(conversation.summary), **carried.log_fields()})
        return {
            "pathway": pathway,
            "api_language": state.get("language"),
            "conversation": conversation.model_dump(),
            "history": window.render(conversation, carried_over=carried.render()),
        }

    return classify


# --- load_context -----------------------------------------------------------

def build_load_context(config):
    def load_context(state) -> dict:
        pathway = state["pathway"]
        namespaces = build_namespaces(session_id=state.get("session_id"))
        style = resolve_style(pathway)
        logger.info("load_context", extra={"pathway": pathway, "namespaces": namespaces})
        return {
            "namespaces": namespaces,
            "style": style,
            "subject_id": state.get("subject_id"),
            "grade": state.get("grade"),
        }

    return load_context


# --- understand ---------------------------------------------------------------

def _allowed_descriptions(allowed: list[str]) -> str:
    """The task-description bullets for just the allowed tasks (one source of truth in
    prompts.TASK_DESCRIPTIONS), so the model only ever sees runnable tasks."""
    return "\n".join(f"- {TASK_DESCRIPTIONS[t]}" for t in allowed if t in TASK_DESCRIPTIONS)


def _allowed_tasks(config, pathway: str) -> list[str]:
    default_tasks = config.get("agent.default_tasks", ["explain", "summarize"])
    return config.get(f"agent.pathways.{pathway}.tasks") or default_tasks


def _session_facts(subject, requested: Optional[str], policy: LanguagePolicy,
                   pending: Optional[PendingAssessment] = None) -> str:
    lines = []
    if subject.name or subject.language:
        taught = (f", taught in {name_of(subject.language)}"
                  if policy.supports(subject.language) else "")
        lines.append(f"- Subject: {subject.name or 'unnamed'}{taught}.")
    else:
        lines.append("- Subject: not settled yet.")
    if requested:
        lines.append(f"- The student asked to be answered in {name_of(requested)}.")
    if pending and pending.items:
        lines.append("- These questions are open. The student has been asked them and "
                     "has not been marked yet:\n" + pending.numbered())
        lines.append("- If the new message answers them, even partly and even without "
                     "asking to be marked, that is a grade task. If it asks something "
                     "else, answer that and leave the questions open.")
    return "\n".join(lines)


def _coherent(allowed: list[str], supported: tuple[str, ...], message: str):
    """The checks a schema cannot express, fed back to the model on a retry."""
    def check(reading: TurnUnderstanding) -> None:
        if reading.needs_syllabus and not reading.tasks:
            raise ValueError("needs_syllabus is true, so tasks must hold at least one task")
        for task in reading.tasks:
            if task.type not in allowed:
                raise ValueError(f"task type {task.type!r} is not available here; "
                                 f"use only {', '.join(allowed)}")
            if not (task.query or "").strip():
                raise ValueError(f"the {task.type} task needs a query that names what it is about")
        words = " ".join(reading.student_words.split())
        if not words or words not in " ".join(message.split()):
            raise ValueError("student_words must be the student's own words, copied exactly "
                             "from the message")
        request = reading.language_request
        if request.action == "set" and request.code not in supported:
            raise ValueError(f"language_request.code must be one of {', '.join(supported)}")
    return check


def _read_turn(state, llm, config, *, allowed, subject, requested, policy,
               pending: Optional[PendingAssessment] = None) -> TurnUnderstanding:
    """The model's reading of the turn, or a single explain task when there is no model."""
    message = state.get("message", "")
    if llm is None:
        fallback = "explain" if "explain" in allowed else allowed[0]
        return TurnUnderstanding(
            student_words=message,
            needs_syllabus=True,
            tasks=[PlannedTask(type=fallback, query=message)],
            message_language=policy.written_in(message),
        )
    material = (name_of(subject.language) if policy.supports(subject.language)
                else "the language the student writes in")
    prompt = UNDERSTAND_PROMPT.render(
        role=state["pathway"],
        task_descriptions=_allowed_descriptions(allowed),
        allowed=", ".join(allowed),
        session=_session_facts(subject, requested, policy, pending),
        history=state.get("history") or "",
        message=message,
        material_language=material,
        languages=", ".join(policy.supported),
    )
    return call_structured(
        TurnUnderstanding,
        [HumanMessage(content=prompt)],
        task="understand", config=config, llm=llm, label="understand",
        validate=_coherent(allowed, policy.supported, message),
    )


def _standing_language(current: Optional[str], reading: TurnUnderstanding,
                       policy: LanguagePolicy) -> Optional[str]:
    request = reading.language_request
    if request.action == "set" and policy.supports(request.code):
        return request.code
    if request.action == "clear":
        return None
    return current if policy.supports(current) else None


def _planned(reading: TurnUnderstanding) -> list[dict]:
    """The model's tasks as plain dicts, each sub-spec kept only on its own task type."""
    tasks = []
    for task in reading.tasks:
        entry = {"type": task.type, "query": task.query.strip(),
                 "level": task.level, "length": task.length}
        if task.type == "assessment" and task.assessment:
            entry["assessment"] = task.assessment.model_dump(exclude_none=True)
        if task.type == "grade" and task.grade:
            entry["grade"] = task.grade.model_dump(exclude_none=True)
        if task.type == "visual" and task.visual:
            entry["visual"] = task.visual.model_dump(exclude_none=True)
        tasks.append(entry)
    return _merge_same_type(tasks)


def build_understand(config, llm, subjects: SubjectDirectory):
    """Read the turn against the conversation: what is asked, in what language.

    One model call replaces what four hand-written layers used to guess from the
    words of a single message: whether it was small talk, whether it followed
    the last answer, whether it named a language, and which task it wanted.
    """
    policy = LanguagePolicy.from_config(config)

    def understand(state) -> dict:
        pathway = state["pathway"]
        allowed = _allowed_tasks(config, pathway)
        conversation = Conversation.model_validate(state.get("conversation") or {})
        subject_id = state.get("subject_id") or conversation.state.subject_id
        subject = subjects.facts(subject_id)
        current = conversation.state.language
        try:
            reading = _read_turn(state, llm, config, allowed=allowed, subject=subject,
                                 requested=current, policy=policy,
                                 pending=conversation.state.pending)
        except Exception as error:
            logger.error("reading the turn failed", exc_info=True)
            raise TutorUnavailable() from error
        words = reading.student_words.strip() or state.get("message", "")
        requested = _standing_language(current, reading, policy)
        message_language = (reading.message_language
                            if policy.supports(reading.message_language)
                            else policy.written_in(words))
        decision = policy.reply(
            requested=requested, subject_language=subject.language,
            message_language=message_language, api_language=state.get("api_language"))
        kind = "study" if reading.needs_syllabus else "conversation"
        logger.info("understand", extra={
            "turn_kind": kind, "tasks": [t.type for t in reading.tasks],
            "continues_previous": reading.continues_previous,
            "changes_subject": reading.changes_subject,
            "language_request": reading.language_request.action,
            **decision.log_fields()})
        return {
            "raw_message": state.get("message", ""),
            "message": words,
            "understanding": reading.model_dump(),
            "turn_kind": kind,
            "planned_tasks": _planned(reading) if reading.needs_syllabus else [],
            "requested_language": requested,
            "message_language": message_language,
            "language": decision.code,
            "language_decision": decision.model_dump(),
        }

    return understand


def _after_understand(state) -> str:
    return "converse" if state.get("turn_kind") == "conversation" else "plan"


# --- converse -----------------------------------------------------------------

def build_converse(config, llm):
    """A reply to a turn that needs no course material, written by the model."""
    def converse(state) -> dict:
        language = state.get("language", "en")
        if llm is None:
            return {"response": NOTHING_PRODUCED, "reply_body": NOTHING_PRODUCED}
        allowed = _allowed_tasks(config, state["pathway"])
        prompt = CONVERSE_PROMPT.render(
            language=language,
            role=state["pathway"],
            tone=(state.get("style") or Style()).tone,
            capabilities="\n" + _allowed_descriptions(allowed),
            history=state.get("history") or "",
            message=state.get("message", ""),
        )
        try:
            reply = (llm.invoke([HumanMessage(content=prompt)]).content or "").strip()
        except Exception as error:
            raise TutorUnavailable() from error
        if not reply:
            raise TutorUnavailable()
        return {"response": reply, "reply_body": reply}

    return converse


# --- plan -------------------------------------------------------------------

# Types whose answer is prose that genuinely combines; the planner should emit one each,
# and this is the safety net if it over-splits anyway. grade/assessment/visual carry
# distinct per-task specs (answers to mark, question shapes, visual kinds) — never merged.
_MERGEABLE = {"explain", "summarize"}


def _merge_same_type(tasks: list[dict]) -> list[dict]:
    """Collapse multiple explain/summarize tasks into one each: 'explain X and Y' is a
    single grounded answer, not two disjoint retrievals. Order preserved; the first task of
    a type keeps its level/length and absorbs the others' queries."""
    first: dict[str, dict] = {}
    out: list[dict] = []
    for t in tasks:
        typ = t.get("type")
        if typ in _MERGEABLE and typ in first:
            base = first[typ]
            q = t.get("query")
            if q and q not in base["query"]:
                base["query"] = f'{base["query"]}; {q}'
            continue
        out.append(t)
        if typ in _MERGEABLE:
            first[typ] = t
    return out


def _infer_subject(query: str, namespaces: list[str], config,
                   enrolled_ids: set | None = None,
                   language: Optional[str] = None) -> str | None:
    """The subject whose material answers ``query`` best, found by searching for it.

    Matching the question against topic labels guessed a subject from a few words
    of a label; searching the material itself asks the question the answer will be
    held to. The nearest passages by meaning above the similarity floor vote for
    their subject with their score; the lexical reordering is left out, because
    a question's function words match every lesson written in its language. A
    student's vote is confined to the subjects they are enrolled in.
    """
    k = config.get("agent.infer_subject_k", 5)
    floor = config.get("agent.min_score", DEFAULT_MIN_SCORE)
    pool = config.get("agent.hybrid.candidates", k)
    try:
        hits = rag_search(query, namespaces or ["syllabus"], k=max(k, pool), config=config)
    except Exception:
        logger.warning("subject inference search failed", exc_info=True)
        return None
    nearest = sorted(hits, key=lambda chunk: chunk.score, reverse=True)[:k]
    # The same term is taught in Arabic and in French, and a question about
    # static electricity matches both — the embedder was chosen for exactly
    # that. Where the student's own language is among the answers, the rest are
    # another class's material, however well they score.
    if language:
        in_language = [chunk for chunk in nearest if chunk.language == language]
        if in_language:
            nearest = in_language
    votes: dict[str, float] = {}
    for chunk in nearest:
        if chunk.score < floor or not chunk.subject_id:
            continue
        if enrolled_ids and chunk.subject_id not in enrolled_ids:
            continue
        votes[chunk.subject_id] = votes.get(chunk.subject_id, 0.0) + chunk.score
    return max(votes, key=votes.get) if votes else None


def _validate_assessment_spec(raw: Any) -> AssessmentSpec | None:
    """An AssessmentSpec from raw planner output, or None when absent/empty/invalid."""
    if not isinstance(raw, dict):
        return None
    try:
        spec = AssessmentSpec(**raw)
    except Exception:
        return None
    return spec if (spec.count is not None or spec.types or spec.difficulty) else None


def _validate_grade_spec(raw: Any) -> GradeSpec | None:
    """A GradeSpec (assessment_id/kind) from raw planner output, or None when absent."""
    if not isinstance(raw, dict):
        return None
    try:
        spec = GradeSpec(**raw)
    except Exception:
        return None
    return spec if (spec.assessment_id or spec.kind) else None


def _validate_visual_spec(raw: Any) -> VisualSpec | None:
    """A VisualSpec (kind/params/description) from raw planner output, or None when absent."""
    if not isinstance(raw, dict):
        return None
    try:
        return VisualSpec(**raw)
    except Exception:
        return None


def _assessment_type_mix(spec: AssessmentSpec, default: dict) -> dict:
    """Turn an AssessmentSpec into a concrete ``type_mix``: distribute the requested
    count across the requested types, filling gaps from the config default."""
    default = {k: int(v) for k, v in (default or {}).items() if int(v) > 0} or {"mcq": 5}
    types = spec.types or list(default.keys())
    total = spec.count if spec.count else sum(default.values())
    base, extra = divmod(total, len(types))
    mix = {t: base + (1 if i < extra else 0) for i, t in enumerate(types)}
    mix = {t: n for t, n in mix.items() if n > 0}
    return mix or dict(default)


def _enrolled_subjects(client, state) -> set | None:
    if not (client and state.get("pathway") == "student" and state.get("user_id")):
        return None
    try:
        subjects = client.get_student_subjects(state["user_id"]) or []
    except Exception:
        logger.warning("enrolled-subjects lookup failed; inferring unconstrained",
                       exc_info=True)
        return None
    return {s.get("id") for s in subjects if s.get("id")} or None


def _resolve_subject(state, reading: TurnUnderstanding, conversation: Conversation,
                     raw_tasks: list[dict], config, client,
                     subjects: Optional[SubjectDirectory] = None) -> tuple[Optional[str], str]:
    """The subject this turn is scoped to, and how it was decided.

    What the caller sends wins; then the subject the conversation settled on,
    unless the student has moved to another or is no longer writing in its
    language; then the subject whose material answers the question; the
    configured default only as a last resort.

    The language clause is the one a two-language corpus needs. One English
    question settled a conversation on the French science subject, and every
    Arabic question after it was answered from the French book, because a
    settled subject was kept whatever the student then wrote in.
    """
    message_language = state.get("message_language")
    if state.get("subject_id"):
        return state["subject_id"], "request"
    settled = conversation.state.subject_id
    if settled and not reading.changes_subject and not _wrong_language(
            settled, message_language, subjects):
        return settled, "conversation"
    query = next((t["query"] for t in raw_tasks if t.get("query")), state.get("message", ""))
    inferred = _infer_subject(query, state.get("namespaces") or [], config,
                              _enrolled_subjects(client, state),
                              language=message_language)
    if inferred:
        return inferred, "inferred"
    fallback = config.get("agent.default_subject_id")
    if fallback:
        return fallback, "default"
    return settled, "conversation" if settled else "none"


def _wrong_language(subject_id: str, message_language: Optional[str],
                    subjects: Optional[SubjectDirectory]) -> bool:
    """Whether the settled subject is taught in a language the student left behind."""
    if not (message_language and subjects):
        return False
    taught_in = subjects.facts(subject_id).language
    return bool(taught_in) and taught_in != message_language


def _previous_evidence(reading: TurnUnderstanding, conversation: Conversation,
                       subject_id: Optional[str]) -> list[str]:
    """The passages the last answer rested on, when this turn carries it on."""
    if not reading.continues_previous:
        return []
    last = conversation.last_study_turn
    if last is None or (subject_id and last.subject_id and last.subject_id != subject_id):
        return []
    return list(last.cited_chunk_ids)


def _planned_topics(tasks: list[dict]) -> list[str]:
    """Every topic this turn resolved, deduplicated, order kept.

    Narrows the learner model to what the turn is about. A turn that resolved
    no topic gets the whole subject instead — which is the honest answer to
    "what does this student find hard here?" when the question named nothing.
    """
    seen: dict[str, None] = {}
    for task in tasks:
        for topic_id in task.get("params", {}).get("topic_ids") or []:
            seen.setdefault(topic_id, None)
    return list(seen)


def build_plan(config, subjects: SubjectDirectory, client=None):
    """Turn the model's reading into runnable tasks: subject, topics, language, specs."""
    policy = LanguagePolicy.from_config(config)

    def plan(state) -> dict:
        pathway = state["pathway"]
        reading = TurnUnderstanding.model_validate(state.get("understanding") or {
            "needs_syllabus": True})
        conversation = Conversation.model_validate(state.get("conversation") or {})
        raw_tasks = state.get("planned_tasks") or []
        subject_id, subject_source = _resolve_subject(
            state, reading, conversation, raw_tasks, config, client, subjects)
        subject = subjects.facts(subject_id)
        decision = policy.reply(
            requested=state.get("requested_language"),
            subject_language=subject.language,
            message_language=state.get("message_language"),
            api_language=state.get("api_language"),
            subject_chosen=subject_source == "request")

        lecture_ids = lecture_filter(state)
        tasks: list[dict] = []
        for t in raw_tasks:
            resolved = resolve_topics(t["query"], subject_id, lecture_ids=lecture_ids,
                                      config=config)
            params: dict[str, Any] = {
                "query": t["query"],
                "level": t["level"],
                "length": t["length"],
                "topic_ids": [r.topic_id for r in resolved],
                "topic_labels": [r.label for r in resolved],
            }
            # Honour a user-requested count/types on an assessment; else assessment_sg
            # uses the config default type_mix. A requested count is capped to the
            # per-role limit (config assessment.max_count) — a teacher's exam may be large.
            if t["type"] == "assessment":
                spec = _validate_assessment_spec(t.get("assessment"))
                if spec is not None:
                    if spec.count is not None or spec.types:
                        if spec.count is not None:
                            cap = config.get(f"assessment.max_count.{pathway}") \
                                or config.get("assessment.max_count.student", 20)
                            spec.count = min(spec.count, int(cap))
                        params["type_mix"] = _assessment_type_mix(
                            spec, config.get("assessment.type_mix", {}))
                    if spec.difficulty:
                        params["difficulty"] = spec.difficulty
            # A chat grade task: carry any DB reference (assessment_id/kind); the answer
            # text is `query`. The adapter normalizes it into a submission.
            if t["type"] == "grade":
                gspec = _validate_grade_spec(t.get("grade"))
                grade = gspec.model_dump() if gspec else {}
                # A structured grade_ref on the request (the frontend already knows the
                # saved quiz/homework id) wins over the NL-parsed reference.
                ref = _validate_grade_spec(state.get("grade_ref"))
                if ref is not None:
                    grade.update(ref.model_dump(exclude_none=True))
                params["grade"] = grade
            # A chat visual task: carry the visual spec + the resolved topic_ids (so
            # visual_sg can try get_figures first).
            if t["type"] == "visual":
                vspec = _validate_visual_spec(t.get("visual"))
                params["visual"] = vspec.model_dump() if vspec else {}
                params["visual"].setdefault("topic_ids", params.get("topic_ids") or None)
            tasks.append({
                "type": t["type"], "params": params, "task_id": uuid.uuid4().hex[:8],
            })

        # Read once, here, rather than per task: the subject is resolved by
        # this point and the tasks of one turn are all about the same student.
        learner = standing(state.get("user_id"), subject_id,
                           topic_ids=_planned_topics(tasks), config=config)

        out: dict[str, Any] = {
            "tasks": tasks,
            "previous_chunk_ids": _previous_evidence(reading, conversation, subject_id),
            "learner": learner.render(),
            "language": decision.code,
            "language_decision": decision.model_dump(),
        }
        if subject_id and subject_id != state.get("subject_id"):
            out["subject_id"] = subject_id
        logger.info("plan", extra={
            "pathway": pathway, "tasks": len(tasks), "subject_id": subject_id,
            "subject_source": subject_source,
            "reused_passages": len(out["previous_chunk_ids"]),
            **learner.log_fields(), **decision.log_fields()})
        return out

    return plan


# --- dispatch (routing) -----------------------------------------------------

_ADAPTER = {"explain": "explain_adapter", "summarize": "summarize_adapter",
            "assessment": "assessment_adapter", "grade": "grade_adapter",
            "report": "report_adapter", "visual": "visual_adapter"}


def dispatch(state) -> list:
    """Fan out one Send per task to its subgraph adapter (explain, summarize, assessment,
    grade, visual); unknown types are dropped (the allowlist is enforced in plan). ``user_id``
    is forwarded for assessment adaptivity (get_mastery)."""
    from langgraph.types import Send

    # No tasks -> the planner degraded (a friendly response is already set); skip to the end.
    if not state.get("tasks"):
        return "persist"
    # Shared context every task needs; each Send carries it + that one task's payload, so
    # the adapters run in parallel without sharing MainState. A Send per task = the fan-out.
    # `message` is what the student typed; `params.query` is what the planner
    # made of it for searching. A turn is classified on the first — the rewrite
    # always names a subject, because naming one is what a rewrite is for.
    ctx = {k: state.get(k) for k in
           ("namespaces", "style", "language", "subject_id", "lecture_ids", "grade",
            "user_id", "session_id", "pathway", "study_scope", "message",
            "history", "learner", "previous_chunk_ids", "conversation")}
    sends = []
    for task in state.get("tasks", []):
        node = _ADAPTER.get(task.get("type"))
        if node:
            sends.append(Send(node, {"task": task, **ctx}))
    return sends or "persist"


# --- compose ----------------------------------------------------------------

def _dedupe_prereqs(prereqs: list[PrereqTopic]) -> list[PrereqTopic]:
    seen, out = set(), []
    for p in prereqs:
        if p.topic_id and p.topic_id not in seen and p.label:
            seen.add(p.topic_id)
            out.append(p)
    return out


# What follows an answer is marked with symbols rather than words. A heading
# written in the service is a sentence in some language: it had to be kept for
# every language the service answers in and was wrong the day one more was
# added. The text around the marks is the model's, in the reply's language.
SOURCES_MARK = "📚"
PREREQUISITES_MARK = "🧩"
CORRECT_ANSWER_MARK = "→"
REASON_MARK = "💡"
REUSED_MARK = "📖"
VISUAL_MARKS = {
    "mindmap": "🧠", "chart": "📊", "math": "📈", "geometry": "📐",
    "flashcards": "🗂️", "pdf": "📄", "image": "🖼️", "figure": "🖼️",
}

# Trailing punctuation stripped off a heading before appending ", p.N".
_HEAD_TRAIL = " \t،,;:."


def _citation_where(c: Citation) -> str:
    """Readable citation label: ``Subject · N: Title``, degrading to the heading path
    and finally the lecture id."""
    subject = (c.subject_name or "").strip()
    title = (c.lesson_title or "").strip()
    order = c.lesson_order
    if title and order is not None:
        lesson = f"{order}: {title}"
    else:
        lesson = title or (f"#{order}" if order is not None else "")
    label = " · ".join(p for p in (subject, lesson) if p)
    return label or (c.heading_path or c.lecture_id or "").rstrip(_HEAD_TRAIL)


def _prereq_line(prereqs: list[PrereqTopic]) -> str:
    """What to know first, from the topic graph of the passages the answer cited."""
    labels = [p.label for p in prereqs if p.label]
    return f"{PREREQUISITES_MARK} {', '.join(labels)}" if labels else ""


def _assessment_block(assessment: dict) -> str:
    """The questions themselves, numbered, without their answers.

    A line announcing a quiz was a sentence in some language, and the questions
    it announced were not on the screen: the platform does not render the
    structured assessment. The questions are already in the student's language.
    """
    lines = []
    for index, item in enumerate(assessment.get("items") or [], 1):
        lines.append(f"{index}. {item.get('question', '')}")
        for position, option in enumerate(item.get("options") or []):
            lines.append(f"   {chr(ord('a') + position)}) {option.get('text', '')}")
    return "\n".join(lines)


def _visual_line(visuals: list) -> str:
    """A link to each rendered visual; a ``slides`` visual is text, so it is inlined."""
    lines = []
    for v in visuals:
        if v.get("kind") == "slides" and v.get("text"):
            lines.append(v["text"])
        elif v.get("s3_uri"):
            mark = VISUAL_MARKS.get(v.get("kind"), "🔗")
            reused = f" {REUSED_MARK}" if v.get("reused") else ""
            lines.append(f"[{mark}{reused}]({v['s3_uri']})")
    return "\n".join(lines)


def _grade_line(grade_result: dict) -> str:
    """The score, then per answer the feedback and — for a wrong answer — the correct
    answer and why, so the student learns what was right and not only that they missed it."""
    summary = grade_result.get("summary") or {}
    score = summary.get("score", 0)
    total = summary.get("max_score", summary.get("count", 0))
    lines = [f"{score:g}/{total:g}" if isinstance(total, (int, float)) else f"{score}/{total}"]
    for i, g in enumerate(grade_result.get("graded") or [], 1):
        if g.get("graded") is False:
            lines.append(f"∅ {i}.")
            continue
        fb = g.get("feedback")
        if not fb:
            continue
        mark = "✓" if g.get("correct") else "✗"
        lines.append(f"{mark} {i}. {fb}")
        if not g.get("correct"):
            if g.get("correct_answer"):
                lines.append(f"   {CORRECT_ANSWER_MARK} {g['correct_answer']}")
            if g.get("explanation"):
                lines.append(f"   {REASON_MARK} {g['explanation']}")
    return "\n".join(lines)


DEFAULT_MAX_CITATIONS = 6


def _renumber_citations(
    answers: list[dict], citations: list[Citation], limit: int = DEFAULT_MAX_CITATIONS
) -> tuple[list[dict], list[Citation]]:
    """Give one global [n] numbering to a multi-task turn.

    Each subgraph numbers its own citations from [1], so an explain+summarize turn
    collides. Numbering follows the order the answers cite their sources, not the
    order retrieval returned them, and only sources a marker actually points at are
    numbered — an uncited chunk in the sources list is a lesson the student was
    never sent to.

    Sources beyond ``limit`` are dropped along with their markers: eight bracketed
    numbers on one paragraph is noise, and a marker outliving its source is worse
    than no marker.
    """
    by_task: dict[Any, dict[int, Citation]] = {}
    for c in citations:
        by_task.setdefault(c.task_id, {}).setdefault(c.n, c)

    global_n: dict[tuple, int] = {}      # source key -> global number
    ordered: list[Citation] = []         # deduped citations, in global-number order
    remaps: dict[Any, dict[int, int]] = {}
    for a in answers:
        tid = a.get("task_id")
        remap = remaps.setdefault(tid, {})
        available = by_task.get(tid, {})
        for n in cited_numbers(a.get("text", "")):
            c = available.get(n)
            if c is None:
                continue
            key = (c.lecture_id, c.heading_path, c.page_from)
            if key not in global_n:
                if len(global_n) >= limit:
                    continue
                global_n[key] = len(global_n) + 1
                ordered.append(c.model_copy(update={"n": global_n[key]}))
            remap[c.n] = global_n[key]

    new_answers = [
        {**a, "text": renumber(a.get("text", ""), remaps.get(a.get("task_id"), {}))}
        for a in answers
    ]
    return new_answers, ordered


def _sources_block(citations: list[Citation]) -> str:
    seen, lines = set(), []
    for c in citations:
        key = (c.lecture_id, c.heading_path, c.page_from)
        if key in seen:
            continue
        seen.add(key)
        pages = ""
        if c.page_from:
            pages = f", p.{c.page_from}" + (
                f"–{c.page_to}" if c.page_to and c.page_to != c.page_from else "")
        lines.append(f"[{c.n}] {_citation_where(c)}{pages}")
    return f"{SOURCES_MARK}\n" + "\n".join(lines) if lines else ""


def _compose_llm(llm, answers, prereqs, language, tone) -> str:
    joined = "\n\n".join(a.get("text", "") for a in answers)
    prereq_str = ", ".join(p.label for p in prereqs) or "(none)"
    prompt_text = COMPOSE_PROMPT.render(language=language, tone=tone, answers=joined,
        formatting=FORMATTING_CONTRACT, prereq_topics=prereq_str)
    return llm.invoke([HumanMessage(content=prompt_text)]).content


NOTHING_PRODUCED = "∅"


def _no_material(state, llm, language: str) -> str:
    """A turn whose tasks produced nothing, explained by the tutor rather than a fixed line."""
    if llm is None:
        return NOTHING_PRODUCED
    prompt = NO_MATERIAL_PROMPT.render(
        language=language, style=(state.get("style") or Style()).tone, scope="",
        history=state.get("history") or "", message=state.get("message", ""))
    try:
        text = (llm.invoke([HumanMessage(content=prompt)]).content or "").strip()
    except Exception as error:
        raise TutorUnavailable() from error
    if not text:
        raise TutorUnavailable()
    return text


def build_compose(config, llm):
    def compose(state) -> dict:
        out = _compose(state)
        out.setdefault("reply_body", out.get("response", ""))
        return out

    def _compose(state) -> dict:
        answers = state.get("answers", [])
        language = state.get("language", "en")
        tone = (state.get("style") or Style()).tone
        prereqs = _dedupe_prereqs(state.get("prereq_topics", []))
        assessment = state.get("assessment") or {}
        grade_result = state.get("grade_result") or {}
        visuals = state.get("visuals") or []
        if grade_result.get("degraded") or assessment.get("degraded"):
            logger.error("a requested task could not be completed",
                         extra={"assessment_degraded": bool(assessment.get("degraded")),
                                "grade_degraded": bool(grade_result.get("degraded"))})
            raise TutorUnavailable()
        if state.get("visual_error"):
            logger.error("a requested visual could not be rendered")
            if not (answers or visuals or grade_result.get("graded")
                    or assessment.get("items")):
                raise TutorUnavailable()

        if not answers:
            parts = []
            if grade_result.get("graded"):
                parts.append(_grade_line(grade_result))
            if assessment.get("items"):
                parts.append(_assessment_block(assessment))
            if visuals:
                parts.append(_visual_line(visuals))
            response = "\n\n".join(p for p in parts if p)
            return {"response": response or _no_material(state, llm, language)}

        # One global [n] numbering across tasks before merging (each subgraph numbered from [1]).
        answers, citations = _renumber_citations(
            answers, state.get("citations", []),
            config.get("agent.max_citations", DEFAULT_MAX_CITATIONS))

        # Blend multiple task answers into one reply via the LLM; a single answer (or no LLM,
        # or an LLM failure) is just its text concatenated — real content, never a stub.
        if llm is not None and len(answers) > 1:
            try:
                # body = _compose_llm(llm, answers, prereqs, language, tone)
                body = _compose_llm(llm, answers, [], language, tone)
            except Exception:
                logger.error("compose LLM call failed; concatenating", exc_info=True)
                body = "\n\n".join(a.get("text", "") for a in answers)
        else:
            body = "\n\n".join(a.get("text", "") for a in answers)
        body = strip_markers(body)

        # A declined answer (grounded=False, no inline citation) suppresses prereqs — a refusal
        # must not trail prerequisites as if it had answered.
        # any_grounded = any(a.get("grounded", True) for a in answers)
        parts = [body]
        # if prereqs and any_grounded and config.get("agent.show_prereqs", True):
        #     parts.append(_prereq_line(prereqs))
        # parts.append(_sources_block(citations))
        if assessment.get("items"):
            parts.append(_assessment_block(assessment))
        if grade_result.get("graded"):
            parts.append(_grade_line(grade_result))
        if visuals:
            parts.append(_visual_line(visuals))
        logger.info("compose", extra={
            "answers": len(answers), "prereqs": len(prereqs),
            "citations": len(citations)})
        return {"response": "\n\n".join(p for p in parts if p),
                "reply_body": body,
                "published_citations": citations}

    return compose


# --- persist ----------------------------------------------------------------

def _turn_record(state) -> Turn:
    cited = [c.chunk_id for c in (state.get("published_citations") or []) if c.chunk_id]
    tasks = state.get("tasks") or []
    query = next((t.get("params", {}).get("query") for t in tasks
                  if t.get("params", {}).get("query")), "")
    return Turn(
        message=state.get("message", ""),
        reply=state.get("reply_body") or state.get("response") or "",
        kind="study" if state.get("turn_kind") == "study" else "conversation",
        language=state.get("language"),
        subject_id=state.get("subject_id") if cited else None,
        tasks=[t.get("type") for t in tasks],
        query=query,
        cited_chunk_ids=cited,
    )


def _issued_questions(state) -> Optional[PendingAssessment]:
    """The quiz this turn put in front of the student, with its answer key."""
    items = (state.get("assessment") or {}).get("items") or []
    questions = []
    for index, item in enumerate(items, 1):
        item = item if isinstance(item, dict) else item.model_dump()
        question = (item.get("question") or "").strip()
        if not question:
            continue
        questions.append(PendingQuestion(
            id=item.get("id") or f"q{index}",
            question=question,
            type=item.get("type") or "short_answer",
            options=[o.get("text", "") if isinstance(o, dict) else str(o)
                     for o in (item.get("options") or [])],
            answer=_answer_key(item),
            topic_id=item.get("topic_id"),
            lecture_id=item.get("lecture_id"),
        ))
    return PendingAssessment(items=questions) if questions else None


def _answer_key(item: dict) -> str:
    answer = item.get("answer")
    if isinstance(answer, list):
        return ", ".join(str(a) for a in answer)
    if answer:
        return str(answer)
    correct = [o.get("text", "") for o in (item.get("options") or [])
               if isinstance(o, dict) and o.get("is_correct")]
    return ", ".join(correct)


def _still_open(state, previous: SessionState, turn: Turn) -> Optional[PendingAssessment]:
    """A quiz stays open until it is marked, or a new one replaces it.

    A student who breaks off to ask something else comes back to the same
    questions: re-asking them over the student's own question is how the tutor
    would talk past them.
    """
    issued = _issued_questions(state)
    if issued:
        return issued
    if "grade" in turn.tasks and (state.get("grade_result") or {}).get("graded"):
        return None
    return previous.pending


def _next_state(state, previous: SessionState, turn: Turn) -> SessionState:
    subject_id = turn.subject_id if turn.cited_chunk_ids else previous.subject_id
    return SessionState(subject_id=subject_id, language=state.get("requested_language"),
                        pending=_still_open(state, previous, turn))


def _say_why_not_remembered(user_id, pathway, session_id) -> None:
    """Name what was missing when an answered turn was not kept.

    ``remembered: false`` on the turn line says it happened; it does not say
    whether the caller stopped sending a session id, whether the role is one
    this layer keeps anything for, or whether the store is down — and those
    have three different people fixing them. The store logs its own failures,
    so silence from it plus a line from here means the turn never reached it.
    """
    missing = [name for name, value in
               (("user_id", user_id), ("session_id", session_id)) if not value]
    if missing:
        logger.warning("turn not remembered: the request carried no %s",
                       " and no ".join(missing))
    elif pathway not in REMEMBERING_TYPES:
        logger.info("turn not remembered: nothing is kept for a %s", pathway)


def build_persist(config, store: ConversationStore, window: HistoryWindow, summary_llm=None):
    """Remember the turn, then fold old turns into the summary when the window is full."""
    def persist(state) -> dict:
        user_id, session_id = state.get("user_id"), state.get("session_id")
        pathway = state.get("pathway")
        remembered = False
        if pathway and (state.get("reply_body") or state.get("response")):
            conversation = Conversation.model_validate(state.get("conversation") or {})
            turn = _turn_record(state)
            remembered = store.append(user_id, pathway, session_id, turn)
            if not remembered:
                _say_why_not_remembered(user_id, pathway, session_id)
            following = _next_state(state, conversation.state, turn)
            if following != conversation.state:
                store.save_state(user_id, pathway, session_id, following)
            # The summary is quoted at the head of every later prompt, so it is
            # written in the language this conversation is being held in — not
            # whichever one the model reaches for.
            fold_if_due(store, window, summary_llm, user_id=user_id,
                        user_type=pathway, session_id=session_id,
                        language=turn.language or state.get("language") or "en")
        # Outlives the conversation, so it is written whether or not this one is
        # remembered: a preference stated in a session nobody keeps is still a
        # preference the student stated.
        if pathway == "student":
            stated = TurnUnderstanding.model_validate(
                state.get("understanding") or {"needs_syllabus": True}).preference
            if stated:
                record(user_id, stated, config=config)
        # One line per turn, carrying everything needed to explain the answer
        # without re-reading five others: what was asked, in what language and
        # why, what it matched, and what came back. Correlated by request_id.
        decision = state.get("language_decision") or {}
        answers = state.get("answers") or []
        logger.info("turn", extra={
            "pathway": pathway,
            "turn_kind": state.get("turn_kind"),
            "question": clip(state.get("message", "")),
            "language": state.get("language"),
            "language_source": decision.get("source"),
            "subject_id": state.get("subject_id"),
            "tasks": [t.get("type") for t in (state.get("tasks") or [])],
            "grounded": any(a.get("grounded", False) for a in answers),
            "citations": len(state.get("published_citations") or []),
            "cited": [
                f"{c.lesson_title or c.lecture_id}" for c in
                (state.get("published_citations") or [])[:5]
            ],
            "answer_chars": len(state.get("response") or ""),
            "remembered": remembered,
        })
        # Must stay empty: the streaming path merges every node's delta into the
        # response, so anything returned here would show on one endpoint only.
        return {}

    return persist


# --- verify_language --------------------------------------------------------

def build_verify_language(config, llm=None):
    """Check the finished answer is in the language the turn resolved to.

    A directive in the prompt is an instruction, not a guarantee: grounded
    material in another language pulls the model across, and the student is the
    one who finds out. Detection on the answer costs microseconds; a repair costs
    one call and only runs when the mismatch is confident and the language was
    something we actually knew rather than guessed.
    """
    enabled = config.get("language.verify.enabled", True)
    languages = tuple(config.get("language.supported", ["ar", "en", "fr", "de"]))
    min_chars = config.get("language.verify.answer_min_chars", DEFAULT_MIN_ANSWER_CHARS)
    min_confidence = config.get("language.verify.min_confidence", DEFAULT_MIN_CONFIDENCE)

    def verify_language(state) -> dict:
        answer = state.get("response") or ""
        raw_decision = state.get("language_decision")
        if not enabled or not answer or not raw_decision:
            return {}

        decision = LanguageDecision(**raw_decision)
        check = verify_answer(
            answer, decision, languages=languages,
            min_chars=min_chars, min_confidence=min_confidence,
        )
        if not check.needs_repair:
            return {}

        if llm is None:
            logger.warning("language mismatch, no model to repair with",
                           extra={"expected": decision.code, "detected": check.detected})
            return {"language_repair": "failed"}

        try:
            rewritten = (llm.invoke([HumanMessage(content=LANGUAGE_REPAIR_PROMPT.render(language=decision.code, context_may_differ=False,
                answer=answer))]).content or "").strip()
        except Exception:
            logger.exception("language repair call failed")
            return {"language_repair": "failed"}

        # A rewrite that is still wrong, or that came back empty, is worse than the
        # original: the student gets a coherent answer in the wrong language rather
        # than a mangled one in no language at all.
        if not rewritten:
            return {"language_repair": "failed"}
        recheck = verify_answer(
            rewritten, decision, languages=languages,
            min_chars=min_chars, min_confidence=min_confidence,
        )
        if recheck.needs_repair:
            logger.warning("language repair did not take",
                           extra={"expected": decision.code, "detected": recheck.detected})
            return {"language_repair": "failed"}

        logger.info("language repaired", extra={"expected": decision.code,
                                                "was": check.detected})
        return {"response": rewritten, "reply_body": rewritten,
                "language_repair": "repaired"}

    return verify_language
