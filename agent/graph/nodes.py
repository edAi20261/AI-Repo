"""Parent graph nodes: classify · load_context · plan · dispatch · compose · persist.

Node callables that need config or an LLM are built by ``build_*`` factories so the
dependency is captured once at compile time, not looked up per turn. ``classify``
and ``load_context`` need only config.
"""

from __future__ import annotations

import logging
import re
import uuid
from typing import Any, Optional

from langchain_core.messages import HumanMessage

from ..models import (
    AssessmentSpec, Citation, GradeSpec, PlannerOutput, PrereqTopic, Style, VisualSpec)
from ..prompts import COMPOSE_PROMPT, PLAN_PROMPT, TASK_DESCRIPTIONS
from ..tools import resolve_topics
from ..utils import degraded_message, detect_language, extract_json, parse_structured
from .pathways import build_namespaces, resolve_style

logger = logging.getLogger(__name__)


# --- classify ---------------------------------------------------------------

# Shown (localized) when the request has no valid role — the graph halts rather than
# guessing a persona (en is the fallback for any unlisted language).
_SIGN_IN = {
    "ar": "عذرًا، يرجى تسجيل الدخول بحساب صالح (طالب أو معلّم أو وليّ أمر) للمتابعة.",
    "en": "Sorry — please sign in with a valid account (student, teacher, or parent) to continue.",
    "fr": "Désolé — veuillez vous connecter avec un compte valide (élève, enseignant ou parent).",
    "de": "Entschuldigung — bitte melden Sie sich mit einem gültigen Konto an (Schüler, Lehrer oder Elternteil).",
}


def build_classify(config):
    default_lang = config.get("language.default", "en")
    supported = tuple(config.get("language.supported", ["ar", "en", "fr", "de"]))
    arabic_ratio = config.get("language.arabic_ratio", 0.30)
    known_pathways = set((config.get("agent.pathways", {}) or {}).keys())

    def classify(state) -> dict:
        language = detect_language(
            state.get("message", ""), default_lang, supported=supported,
            arabic_ratio=arabic_ratio)
        user_type = state.get("user_type")
        pathway = user_type if user_type in known_pathways else None
        if pathway is None:
            # Unknown/missing role: don't run the graph — ask the user to sign in.
            logger.warning("classify: unknown user_type %r", user_type)
            return {"pathway": None, "language": language,
                    "response": _SIGN_IN.get(language, _SIGN_IN["en"])}
        logger.info("classify", extra={"pathway": pathway, "language": language})
        return {"pathway": pathway, "language": language}

    return classify


def _after_classify(state) -> str:
    """Halt on an unknown role (classify already set a sign-in response); else continue."""
    return "load_context" if state.get("pathway") else "persist"


# --- load_context -----------------------------------------------------------

def build_load_context(config):
    def load_context(state) -> dict:
        pathway = state["pathway"]
        namespaces = build_namespaces(session_id=state.get("session_id"))
        style = resolve_style(pathway)
        # No default here: subject resolution is explicit -> infer -> default, and
        # inference happens in plan. Applying the default now would preempt it. Per-turn
        # retrieval is scoped to this subject by the subject_id payload filter in
        # rag_search. TODO (re-ingest): switch to the general/subject namespace union
        # (see build_namespaces).
        subject_id = state.get("subject_id")
        logger.info("load_context", extra={"pathway": pathway, "namespaces": namespaces})
        return {
            "namespaces": namespaces,
            "style": style,
            "subject_id": subject_id,
            "grade": state.get("grade"),
        }

    return load_context


# --- plan -------------------------------------------------------------------

def _allowed_descriptions(allowed: list[str]) -> str:
    """The task-description bullets for just the allowed tasks (one source of truth in
    prompts.TASK_DESCRIPTIONS), so the planner only ever sees runnable tasks."""
    return "\n".join(f"- {TASK_DESCRIPTIONS[t]}" for t in allowed if t in TASK_DESCRIPTIONS)


def _plan_tasks(
    message: str, allowed: list[str], llm, *, role: str = "user",
    languages: str = "ar, en, fr, de", config=None,
) -> tuple[list[dict], str | None]:
    """LLM plan when a chat model is available; deterministic single task otherwise.
    Returns ``(tasks, output_language)`` — ``output_language`` is set only when the user
    explicitly asked for the answer in a specific language. A real LLM failure PROPAGATES
    (the plan node degrades with a friendly message); an unusable/empty reply is a soft case
    that falls back to a single explain task."""
    if llm is not None:
        prompt_text = PLAN_PROMPT.format(
            allowed=", ".join(allowed), task_descriptions=_allowed_descriptions(allowed),
            role=role, languages=languages, message=message)
        messages = [HumanMessage(content=prompt_text)]
        # structured output when enabled; else one plain call (a real call failure raises here).
        data = parse_structured(llm, PlannerOutput, messages, config, "planner")
        raw = data.get("tasks") if isinstance(data, dict) else None
        cleaned = []
        for t in (raw or []):
            if not (isinstance(t, dict) and t.get("type") in allowed):
                continue
            entry = {
                "type": t.get("type"),
                "query": t.get("query") or message,
                "level": t.get("level", "intermediate"),
                "length": t.get("length", "medium"),
            }
            # Optional user-requested assessment shape, only on assessment tasks.
            if t.get("type") == "assessment" and isinstance(t.get("assessment"), dict):
                entry["assessment"] = t["assessment"]
            # Optional grade reference (assessment_id/kind), only on grade tasks.
            if t.get("type") == "grade" and isinstance(t.get("grade"), dict):
                entry["grade"] = t["grade"]
            # Optional visual spec (kind/params/description), only on visual tasks.
            if t.get("type") == "visual" and isinstance(t.get("visual"), dict):
                entry["visual"] = t["visual"]
            cleaned.append(entry)
        if cleaned:
            return _merge_same_type(cleaned), data.get("output_language")
    # No LLM (dev/test), or the model returned nothing usable (not a failure) -> a safe default:
    # explain the message. A real LLM failure raised above and the plan node degrades instead.
    typ = "explain" if "explain" in allowed else allowed[0]
    return [{"type": typ, "query": message, "level": "intermediate", "length": "medium"}], None


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


def _infer_subject(message: str, config, enrolled_ids: set | None = None) -> str | None:
    """Pick the subject for a request that arrived without one (subject_id is an optional
    contract field): the best-matching topic's subject. When ``enrolled_ids`` is given
    (a student's own subjects), the inference is CONSTRAINED to those — the best hit that
    is actually enrolled, never a subject the student can't access. Returns None when
    nothing clears the resolve threshold (or nothing enrolled matches) — the caller then
    searches the syllabus unscoped."""
    k = config.get("agent.infer_subject_k", 5) if enrolled_ids else 1
    hits = resolve_topics(message, None, k=k, config=config)
    if enrolled_ids:
        hits = [h for h in hits if h.subject_id in enrolled_ids]
    return (hits[0].subject_id or None) if hits else None


def _validate_assessment_spec(raw: Any) -> Optional[AssessmentSpec]:
    """An AssessmentSpec from raw planner output, or None when absent/empty/invalid."""
    if not isinstance(raw, dict):
        return None
    try:
        spec = AssessmentSpec(**raw)
    except Exception:
        return None
    return spec if (spec.count is not None or spec.types or spec.difficulty) else None


def _validate_grade_spec(raw: Any) -> Optional[GradeSpec]:
    """A GradeSpec (assessment_id/kind) from raw planner output, or None when absent."""
    if not isinstance(raw, dict):
        return None
    try:
        spec = GradeSpec(**raw)
    except Exception:
        return None
    return spec if (spec.assessment_id or spec.kind) else None


def _validate_visual_spec(raw: Any) -> Optional[VisualSpec]:
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


def build_plan(config, llm, client=None):
    default_tasks = config.get("agent.default_tasks", ["explain", "summarize"])
    supported = set(config.get("language.supported", ["ar", "en", "fr", "de"]))

    languages = ", ".join(config.get("language.supported", ["ar", "en", "fr", "de"]))

    def plan(state) -> dict:
        # (1) route the message to task(s), scoped to what this pathway may run.
        pathway = state["pathway"]
        allowed = config.get(f"agent.pathways.{pathway}.tasks") or default_tasks
        try:
            raw_tasks, req_lang = _plan_tasks(
                state.get("message", ""), allowed, llm, role=pathway, languages=languages,
                config=config)
        except Exception:
            # Planning itself failed (LLM call error): degrade — a friendly message, no tasks.
            # dispatch sees no tasks and routes straight to the end (never a silently-picked task).
            logger.error("planning failed; degrading", exc_info=True)
            return {"tasks": [], "response": degraded_message(state.get("language", "en"))}
        # TODO(sprint3-clarify): a clarify node that asks for missing assessment params (count/
        # types) before generating — the model is variance-prone at copying an explicit count
        # (e.g. "4 MCQs" sometimes yields 3). A regex over the message is not the fix.
        # TODO(sprint3-router): report-intent chat turns ("how am I doing", "my progress") still
        # misroute (to visual/assessment) despite the plan-prompt routing hints — the /ai/reports
        # endpoint is unaffected. Add a deterministic report-intent router (or fold into clarify).

        # (2) resolve the subject: explicit -> infer -> default (the default is the last
        # resort only, never allowed to preempt inference; unscoped search contaminates).
        # so retrieval stays scoped to one subject (unscoped search contaminates). The
        # default is the last resort only, never allowed to preempt inference.
        subject_id = state.get("subject_id")
        inferred = None
        if not subject_id:
            # A student's inference is constrained to their enrolled subjects (never infer a
            # subject they can't access); teacher/parent inference stays unconstrained.
            enrolled_ids = None
            if client and pathway == "student" and state.get("user_id"):
                try:
                    subs = client.get_student_subjects(state["user_id"]) or []
                    enrolled_ids = {s.get("id") for s in subs if s.get("id")} or None
                except Exception:
                    logger.warning("enrolled-subjects lookup failed; inferring unconstrained",
                                   exc_info=True)
            subject_id = inferred = _infer_subject(
                state.get("message", ""), config, enrolled_ids)
        used_default = False
        if not subject_id:
            subject_id = config.get("agent.default_subject_id")
            used_default = bool(subject_id)

        # (3) build each task's params: resolve topic_ids (invariant 8) + carry any
        # task-specific spec (assessment shape / grade reference / visual spec).
        lecture_ids = state.get("lecture_ids") or None
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
        out: dict[str, Any] = {"tasks": tasks}
        # Propagate a subject resolved here (inferred or default) so dispatch/retrieve
        # scope to it; an explicit subject is already in state.
        if subject_id and subject_id != state.get("subject_id"):
            out["subject_id"] = subject_id
        # An explicit "answer in <language>" request overrides the detected message
        # language for the whole turn.
        if req_lang in supported:
            out["language"] = req_lang
        logger.info("plan", extra={
            "pathway": pathway, "tasks": len(tasks),
            "subject_id": subject_id, "inferred_subject": bool(inferred),
            "used_default_subject": used_default, "output_language": req_lang})
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
    ctx = {k: state.get(k) for k in
           ("namespaces", "style", "language", "subject_id", "lecture_ids", "grade",
            "user_id", "session_id", "pathway")}
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


# Localised section headers (en is the fallback for any unlisted language).
_PREREQ_HEAD = {"ar": "مواضيع تمهيدية", "en": "Prerequisite topics",
                "fr": "Sujets prérequis", "de": "Voraussetzungen"}
_SOURCES_HEAD = {"ar": "المصادر", "en": "Sources", "fr": "Sources", "de": "Quellen"}
_LESSON_WORD = {"ar": "الدرس", "en": "Lesson", "fr": "Leçon", "de": "Lektion"}
# Trailing punctuation stripped off a heading before appending ", p.N".
_HEAD_TRAIL = " \t،,;:."


def _citation_where(c: Citation, language: str) -> str:
    """Readable citation label: ``Subject · Lesson N: Title``, degrading to the
    heading path and finally the lecture id. ``lesson_order`` is rendered raw (the
    numbering base is confirmed before the re-ingest). Pre-metadata chunks
    have none of these -> heading_path / lecture_id fallback."""
    subject = (c.subject_name or "").strip()
    title = (c.lesson_title or "").strip()
    order = c.lesson_order
    lesson = ""
    if order is not None:
        lesson = f"{_LESSON_WORD.get(language, _LESSON_WORD['en'])} {order}"
    if title:
        lesson = f"{lesson}: {title}" if lesson else title
    label = " · ".join(p for p in (subject, lesson) if p)
    return label or (c.heading_path or c.lecture_id or "").rstrip(_HEAD_TRAIL)


def _prereq_line(prereqs: list[PrereqTopic], language: str) -> str:
    labels = ", ".join(p.label for p in prereqs)
    head = _PREREQ_HEAD.get(language, _PREREQ_HEAD["en"])
    return f"{head}: {labels}"


def _assessment_line(assessment: dict, language: str) -> str:
    # A neutral acknowledgment only — the questions + answer keys are in the structured
    # `assessment` field. No UI/save framing (the frontend decides what to do with it).
    n = assessment.get("count", len(assessment.get("items", [])))
    if language == "ar":
        return f"إليك اختبارًا من {n} سؤالًا."
    return f"Here's a {n}-question assessment."


def _visual_line(visuals: list, language: str) -> str:
    """A short chat line pointing at the rendered visual(s). A ``slides`` visual is text
    (inlined); everything else is an artifact the frontend renders from its uri."""
    lines = []
    for v in visuals:
        if v.get("kind") == "slides" and v.get("text"):
            lines.append(v["text"])
        elif v.get("s3_uri"):
            label = "شكل" if language == "ar" else v.get("kind", "visual").capitalize()
            reused = " (from your syllabus)" if v.get("reused") else ""
            lines.append(f"[{label}{reused}]({v['s3_uri']})")
    return "\n".join(lines)


# A short lead-in for a standalone visual turn (no explanatory prose accompanies it),
# by kind and language. Slides carry their own text, so they get no caption.
_VISUAL_CAPTION = {"ar": "إليك {noun}:", "en": "Here's your {noun}:",
                   "fr": "Voici votre {noun} :", "de": "Hier ist Ihre {noun}:"}
_VISUAL_NOUN = {
    "mindmap": {"ar": "خريطة ذهنية", "en": "mind map", "fr": "carte mentale", "de": "Mindmap"},
    "flashcards": {"ar": "بطاقات تعليمية", "en": "flashcards", "fr": "cartes mémoire",
                   "de": "Lernkarten"},
    "pdf": {"ar": "ملف PDF", "en": "PDF", "fr": "PDF", "de": "PDF"},
    "chart": {"ar": "رسم بياني", "en": "chart", "fr": "graphique", "de": "Diagramm"},
    "image": {"ar": "رسم توضيحي", "en": "illustration", "fr": "illustration",
              "de": "Illustration"},
    "figure": {"ar": "شكل", "en": "figure", "fr": "figure", "de": "Abbildung"},
}


def _visual_caption(visuals: list, language: str) -> str:
    """A one-line caption for the first non-slides visual, e.g. 'Here's your mind map:'.
    Empty when there is nothing to caption (slides-only, or an unknown kind)."""
    for v in visuals:
        kind = v.get("kind")
        if kind == "slides":
            continue
        noun_map = _VISUAL_NOUN.get(kind)
        if not noun_map:
            continue
        noun = noun_map.get(language) or noun_map["en"]
        return _VISUAL_CAPTION.get(language, _VISUAL_CAPTION["en"]).format(noun=noun)
    return ""


# Shown when a visual_spec RAISED during render — an honest error, never the off-syllabus decline.
_RENDER_ERROR = {
    "ar": "حدث خطأ أثناء إنشاء العنصر المرئي. يرجى المحاولة مرة أخرى لاحقًا.",
    "en": "Something went wrong creating the visual. Please try again later.",
    "fr": "Une erreur s'est produite lors de la création du visuel. Veuillez réessayer plus tard.",
    "de": "Beim Erstellen des Visuals ist ein Fehler aufgetreten. Bitte versuchen Sie es später erneut.",
}


# Localised inline labels for the correct answer / explanation shown on a wrong answer.
_ANSWER_LABEL = {"ar": "الإجابة الصحيحة", "en": "Correct answer",
                 "fr": "Bonne réponse", "de": "Richtige Antwort"}
_WHY_LABEL = {"ar": "الشرح", "en": "Why", "fr": "Explication", "de": "Erklärung"}


def _grade_line(grade_result: dict, language: str) -> str:
    """A chat summary of a grade turn: the score, then per answer the feedback and — for a
    wrong (or partial) answer — the correct answer and the explanation, so the student learns
    what the right answer was and why (not just that they missed it)."""
    summary = grade_result.get("summary") or {}
    score = summary.get("score", 0)
    total = summary.get("max_score", summary.get("count", 0))
    graded = grade_result.get("graded") or []
    head = (f"صحّحتُ إجابتك: {score} من {total}." if language == "ar"
            else f"I've graded your work: {score} / {total}.")
    ans_label = _ANSWER_LABEL.get(language, _ANSWER_LABEL["en"])
    why_label = _WHY_LABEL.get(language, _WHY_LABEL["en"])
    lines = [head]
    for i, g in enumerate(graded, 1):
        fb = g.get("feedback")
        if not fb:
            continue
        mark = "✓" if g.get("correct") else "✗"
        lines.append(f"{mark} {i}. {fb}")
        # On a wrong/declined answer, surface the correct answer + reasoning (the key is only
        # meaningful when the student got it wrong; a correct answer needs no restatement).
        if not g.get("correct"):
            if g.get("correct_answer"):
                lines.append(f"   {ans_label}: {g['correct_answer']}")
            if g.get("explanation"):
                lines.append(f"   {why_label}: {g['explanation']}")
    return "\n".join(lines)


_CITE_MARKER = re.compile(r"\[(\d+)\]")


def _renumber_citations(
    answers: list[dict], citations: list[Citation]
) -> tuple[list[dict], list[Citation]]:
    """Give one global [n] numbering to a multi-task turn.

    Each subgraph numbers its own citations from [1], so an explain+summarize turn
    collides. Walk answers in order, dedupe sources across tasks into a single
    numbering, remap each task's inline [n] markers in its artifact text, and return
    the rewritten answers plus the deduped, globally-numbered citations.
    """
    by_task: dict[Any, list[Citation]] = {}
    for c in citations:
        by_task.setdefault(c.task_id, []).append(c)

    global_n: dict[tuple, int] = {}      # source key -> global number
    ordered: list[Citation] = []         # deduped citations, in global-number order
    remaps: dict[Any, dict[int, int]] = {}
    for a in answers:
        tid = a.get("task_id")
        remap = remaps.setdefault(tid, {})
        for c in sorted(by_task.get(tid, []), key=lambda x: x.n or 0):
            key = (c.lecture_id, c.heading_path, c.page_from)
            if key not in global_n:
                global_n[key] = len(global_n) + 1
                ordered.append(c.model_copy(update={"n": global_n[key]}))
            remap[c.n] = global_n[key]

    def _rewrite(text: str, remap: dict[int, int]) -> str:
        if not remap:
            return text
        return _CITE_MARKER.sub(
            lambda m: f"[{remap.get(int(m.group(1)), m.group(1))}]", text)

    new_answers = [
        {**a, "text": _rewrite(a.get("text", ""), remaps.get(a.get("task_id"), {}))}
        for a in answers
    ]
    return new_answers, ordered


def _sources_block(citations: list[Citation], language: str) -> str:
    seen, lines = set(), []
    for c in citations:
        key = (c.lecture_id, c.heading_path, c.page_from)
        if key in seen:
            continue
        seen.add(key)
        # Prefer Subject · Lesson N: Title; fall back to a punctuation-stripped
        # heading and finally the lecture id.
        where = _citation_where(c, language)
        pages = ""
        if c.page_from:
            pages = f", p.{c.page_from}" + (
                f"–{c.page_to}" if c.page_to and c.page_to != c.page_from else "")
        lines.append(f"[{c.n}] {where}{pages}")
    if not lines:
        return ""
    head = _SOURCES_HEAD.get(language, _SOURCES_HEAD["en"])
    return head + ":\n" + "\n".join(lines)


def _compose_llm(llm, answers, prereqs, language, tone) -> str:
    joined = "\n\n".join(a.get("text", "") for a in answers)
    prereq_str = ", ".join(p.label for p in prereqs) or "(none)"
    prompt_text = COMPOSE_PROMPT.format(
        language=language, tone=tone, answers=joined, prereq_topics=prereq_str)
    return llm.invoke([HumanMessage(content=prompt_text)]).content


def build_compose(config, llm):
    def compose(state) -> dict:
        answers = state.get("answers", [])
        language = state.get("language", "en")
        tone = (state.get("style") or Style()).tone
        prereqs = _dedupe_prereqs(state.get("prereq_topics", []))
        assessment = state.get("assessment") or {}
        grade_result = state.get("grade_result") or {}
        visuals = state.get("visuals") or []
        # No explain/summarize text -> a grade/assessment/visual-only turn: the response IS
        # that task's own short line (degraded note first — an LLM failure never shows a stub).
        if not answers:
            if grade_result.get("degraded") or assessment.get("degraded"):
                return {"response": degraded_message(language)}
            if grade_result.get("graded"):
                return {"response": _grade_line(grade_result, language)}
            if visuals:
                caption = _visual_caption(visuals, language)
                line = _visual_line(visuals, language)
                return {"response": "\n\n".join(p for p in (caption, line) if p)}
            if assessment.get("items"):
                return {"response": _assessment_line(assessment, language)}
            # A visual that RAISED (not merely off-syllabus) gets an honest error, not a decline.
            if state.get("visual_error"):
                return {"response": _RENDER_ERROR.get(language, _RENDER_ERROR["en"])}
            msg = ("لا توجد مادة كافية في المنهج للإجابة." if language == "ar"
                   else "There isn't enough syllabus material to answer this yet.")
            return {"response": msg}

        # One global [n] numbering across tasks before merging (each subgraph numbered from [1]).
        answers, citations = _renumber_citations(answers, state.get("citations", []))

        # Blend multiple task answers into one reply via the LLM; a single answer (or no LLM,
        # or an LLM failure) is just its text concatenated — real content, never a stub.
        if llm is not None and len(answers) > 1:
            try:
                body = _compose_llm(llm, answers, prereqs, language, tone)
            except Exception:
                logger.error("compose LLM call failed; concatenating", exc_info=True)
                body = "\n\n".join(a.get("text", "") for a in answers)
        else:
            body = "\n\n".join(a.get("text", "") for a in answers)

        # Assemble the reply: body, then prerequisite pointer, sources, and any task lines.
        # A declined answer (grounded=False, no inline citation) suppresses prereqs — a refusal
        # must not trail "Prerequisite topics: …" as if it had answered.
        any_grounded = any(a.get("grounded", True) for a in answers)
        parts = [body]
        if prereqs and any_grounded:
            parts.append(_prereq_line(prereqs, language))
        sources = _sources_block(citations, language)
        if sources:
            parts.append(sources)
        if assessment.get("degraded"):
            parts.append(degraded_message(language))
        elif assessment.get("items"):
            parts.append(_assessment_line(assessment, language))
        if grade_result.get("degraded"):
            parts.append(degraded_message(language))
        elif grade_result.get("graded"):
            parts.append(_grade_line(grade_result, language))
        if visuals:
            parts.append(_visual_line(visuals, language))
        elif state.get("visual_error"):
            parts.append(_RENDER_ERROR.get(language, _RENDER_ERROR["en"]))
        logger.info("compose", extra={"answers": len(answers), "prereqs": len(prereqs)})
        return {"response": "\n\n".join(p for p in parts if p)}

    return compose


# --- persist ----------------------------------------------------------------

def build_persist(config):
    def persist(state) -> dict:
        # TODO (deferred): the checkpointer holds session memory for now; adaptive_profile /
        # dashboard writes and the ephemeral-namespace drop are not implemented yet. Short-term
        # memory (a messages list via add_messages, or Redis) also lands here.
        logger.info("persist", extra={"pathway": state.get("pathway")})
        return {}

    return persist
