"""Sprint 2 demo — a full tour of everything the agent does, with per-step timings.

Runs a curated suite of requests through the runtime graph (and the grade/report
endpoints) and writes a readable Markdown transcript. Each request reports what the
system produced AND a per-step time breakdown, so latency is visible section by section.

Covered: assessments (default / requested count+type+difficulty / subject-language /
general / AR / FR / off-syllabus decline), explain + summarize (grounded, cited,
multi-concept merge, decline), visuals (mind map / flashcards / PDF / math / geometry /
slides — standalone AND riding on another task), grading (free-text right/wrong/
decline + a structured referenced grade), the mastery loop (quiz -> grade -> mastery moves
-> next quiz adapts), and reports (endpoint + chat).

HOW TO RUN (PowerShell — copy one line):

  1) Full guided tour of every feature:

        $env:APP_ENV="local"; .venv\\Scripts\\python.exe demos\\sprint2\\python\\demo.py

  2) Ask your OWN question(s) instead (each in quotes, space-separated):

        $env:APP_ENV="local"; .venv\\Scripts\\python.exe demos\\sprint2\\python\\demo.py "Explain the fire triangle" "Quiz me on oxidation"
        # optional — pin a subject:  $env:SUBJECT="science-en"   (science-en | science-fr | math-en | social-ar)

WHERE IT IS SAVED:
  Every run writes a NEW, time-stamped transcript to
      demos/sprint2/outputs/sprint2-run-<YYYYMMDD-HHMMSS>.md
  (override the name with  $env:DEMO_OUT="myfile.md"). Runs never overwrite each other.
  All demos/sprint2/outputs/*.md are gitignored — nothing from a run is committed. To share a
  specific transcript, copy it to a file OUTSIDE that folder.

WHAT A REVIEWER SEES:
  The console stays quiet (errors only) and the transcript carries no technical/debug traces —
  just the questions, the answers, and a per-step time breakdown so latency is visible.

Needs: the ingested stores up + OPENROUTER_API_KEY (both from .env). Visuals with no S3
configured are written under .ingestion_artifacts/ and referenced by a file:// link.

--- maintainer only (not for reviewers) -----------------------------------------------------
  $env:DEMO_DEV="1"    verbose console + a "⚠️ Warnings (dev only)" section in the transcript
  $env:DEMO_RESET="1"  clears the demo student's mastery before running (off by default)
"""

from __future__ import annotations

import datetime
import logging
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]        # -> AI-Workflows/
sys.path.insert(0, str(ROOT))

from agent.config import get_config
from agent.graph import build_graph
from agent.graph.grade import run_grade
from agent.graph.report import run_report
from agent.render import gen_chart
from agent.tools import get_mastery, get_mastery_rollup, get_topic_labels
from shared.logging import setup_logging
from shared.metering import start_meter

# Offline BACKEND-API stand-in (subject facts, enrolled subjects, a saved quiz's rubric).
# Distinct from data/samples/fixtures.yaml, which is the INGESTION manifest (lessons->paths).
FIXTURE_PATH = ROOT / "demos" / "sprint2" / "backend_fixtures.yaml"

# Subject ids (from the ingested sample lessons) — scope retrieval to one subject.
SCIENCE_EN = "2dfb991c-45d8-5e66-93da-a764df58ccfe"
SCIENCE_FR = "917f7aed-4a03-576b-91e9-1d5cc7beeca9"
MATH_EN = "920d019b-afa8-5bf1-bd73-58487bb3ffab"
SOCIAL_AR = "04c913e1-1c75-5f62-b793-a34ee3aeacf2"
SUBJECTS = {SCIENCE_EN: "Science EN", SCIENCE_FR: "Science FR",
            MATH_EN: "Math EN", SOCIAL_AR: "Social Studies AR"}
SUBJECT_KEYS = {"science-en": SCIENCE_EN, "science-fr": SCIENCE_FR,
                "math-en": MATH_EN, "social-ar": SOCIAL_AR}

# Demo identities (match demos/sprint2/fixtures.yaml — the offline backend stand-in).
DEMO_STUDENT = "11111111-1111-4111-8111-111111111111"   # identity.students
DEMO_TEACHER = "22222222-2222-4222-8222-222222222222"   # identity.teachers
SCIENCE_EN_LECTURE = "f7d65f67-87ea-5f90-bd61-c5f343ddb35f"   # a real ingested Science EN lecture

# MainState fields whose channel reducer is operator.add (accumulated across nodes).
_ADD_FIELDS = {"answers", "citations", "prereq_topics", "visual_specs"}

# Friendly lecture names (from the ingested sample lessons) so the transcript shows names,
# not raw uuids. Keys are the real lecture_ids.
LECTURES = {
    "f7d65f67-87ea-5f90-bd61-c5f343ddb35f": "Science EN · Unit 1 Lesson 3",
    "d2380fc3-0624-5cd4-ab28-eb91188c814a": "Science FR · Unit 3 Lesson 1",
    "24120522-c283-58d1-93a6-1b7388fd21ad": "Math EN · Unit 2 Lesson 2",
    "07aef7b6-3e50-5598-a544-614dafca12ac": "Social AR · Unit 3 Lesson 3",
    "73e4c47d-8bd1-5652-9596-b8bb10b79c8f": "Social AR · Unit 2 Lesson 2",
}
_LABELS: dict[str, str] = {}   # topic_id -> label cache, filled on demand


def _lecture_name(lid) -> str:
    if not lid:
        return "—"
    return LECTURES.get(lid, f"lecture {str(lid)[:8]}…")


def _topic_names(tids) -> dict:
    """{topic_id: label} for the given ids, batching DB lookups and caching."""
    tids = [t for t in tids if t]
    missing = [t for t in tids if t not in _LABELS]
    if missing:
        try:
            _LABELS.update(get_topic_labels(missing))
        except Exception:
            pass
    return {t: (_LABELS.get(t) or f"topic {str(t)[:8]}…") for t in tids}


def _topic_name(tid) -> str:
    return _topic_names([tid]).get(tid, "—") if tid else "—"


# ===========================================================================
# Timed invoke — stream the graph so each node's wall-clock time is captured
# ===========================================================================

def _invoke_timed(graph, payload: dict) -> tuple[dict, list[tuple[str, float]]]:
    """Invoke via ``stream(stream_mode='updates')`` so every node reports separately;
    time between updates ≈ that node's duration. Also rebuilds the final MainState from
    the per-node deltas (add-reducer fields are concatenated, others overwrite)."""
    state = dict(payload)
    steps: list[tuple[str, float]] = []
    last = time.perf_counter()
    for update in graph.stream(payload, stream_mode="updates"):
        now = time.perf_counter()
        for node, partial in (update or {}).items():
            steps.append((node, now - last))
            if isinstance(partial, dict):
                for k, v in partial.items():
                    if k in _ADD_FIELDS and isinstance(v, list):
                        state[k] = (state.get(k) or []) + v
                    else:
                        state[k] = v
        last = now
    return state, steps


def _steps_line(steps: list[tuple[str, float]]) -> str:
    """A compact per-step timing breakdown (slowest first for the eye)."""
    if not steps:
        return "_(no step timings)_"
    parts = [f"`{node}` {secs:.2f}s" for node, secs in steps]
    slow = max(steps, key=lambda s: s[1])
    return (" → ".join(parts)
            + f"  ·  **slowest:** `{slow[0]}` {slow[1]:.2f}s")


# ===========================================================================
# Renderers — turn a result into Markdown
# ===========================================================================

def _assessment_block(a: dict) -> list[str]:
    if not a or not a.get("items"):
        return ["", "_No assessment — the request did not clear the grounding floor "
                "(off-syllabus), so the system declined instead of inventing one._"]
    items = a["items"]
    actual: dict[str, int] = {}
    for it in items:
        actual[it.get("type")] = actual.get(it.get("type"), 0) + 1
    lines = ["", f"**Assessment** — {a.get('count', len(items))} items (source: {a.get('source')})",
             f"- Type mix — requested `{a.get('type_mix')}` · actual `{actual}`",
             f"- Difficulty spread — {[it.get('difficulty') for it in items]}", ""]
    for i, it in enumerate(items, 1):
        lines.append(f"{i}. _({it.get('type')} · {it.get('difficulty')})_ {it.get('question')}")
        for j, opt in enumerate(it.get("options") or []):
            letter = chr(ord('a') + j)          # a) b) c) … (display only)
            lines.append(f"   {letter}) {opt.get('text')}{' ✓' if opt.get('is_correct') else ''}")
        ans = it.get("answer")
        if ans:
            lines.append(f"   **Answer:** {', '.join(ans) if isinstance(ans, list) else ans}")
        expl = it.get("explanation")
        if expl:
            lines.append(f"   _Why:_ {expl}")
        lines.append(f"   _topic:_ {_topic_name(it.get('topic_id'))} · "
                     f"_lecture:_ {_lecture_name(it.get('lecture_id'))}")
    return lines


def _grade_block(g: dict) -> list[str]:
    if not g:
        return ["", "_No grade result._"]
    s = g.get("summary") or {}
    lines = ["", f"**Grade** — score {s.get('score')}/{s.get('max_score')} "
             f"· correct {s.get('correct')}/{s.get('count')} · declined {s.get('declined', 0)} "
             f"· mastery events {g.get('mastery_events', 0)} · topics updated {g.get('topics_updated', 0)}"]
    for it in g.get("graded") or []:
        verdict = ("declined" if not it.get("graded")
                   else "correct" if it.get("correct") else "wrong")
        lines.append(f"- [{verdict}] {it.get('question', '')} — {it.get('feedback', '')}")
        # On a wrong answer, show the correct answer + reasoning (as the chat reply now does).
        if it.get("graded") and not it.get("correct"):
            if it.get("correct_answer"):
                lines.append(f"  correct answer: {it['correct_answer']}")
            if it.get("explanation"):
                lines.append(f"  why: {it['explanation']}")
        if it.get("mistake_tags"):
            lines.append(f"  mistakes: {', '.join(it['mistake_tags'])}")
    return lines


def _visuals_block(visuals: list[dict]) -> list[str]:
    if not visuals:
        return ["", "_No visual produced (a data-tied kind with no data, or the build was "
                "skipped rather than render a wrong one)._"]
    lines = ["", f"**Visuals** — {len(visuals)}"]
    for v in visuals:
        kind = v.get("kind")
        loc = v.get("s3_uri") or v.get("artifact_id") or "(text, no file)"
        lines.append(f"- `{kind}` → {loc}")
        # Slides / powerpoint: the deliverable IS the outline text — show it in full.
        if v.get("text"):
            lines.append("")
            lines += ["  " + ln for ln in str(v["text"]).splitlines()]
            lines.append("")
        # Flashcards: list every card (front / back).
        for i, card in enumerate(v.get("cards") or [], 1):
            lines.append(f"  {i}. **{card.get('front', '')}** — {card.get('back', '')}")
    return lines


def _report_block(r: dict) -> list[str]:
    if not r:
        return ["", "_No report._"]
    if not r.get("available", True):
        return ["", f"_Report scope `{r.get('scope')}` is not available yet (Sprint 3)._"]
    rc = r.get("root_cause")
    lines = ["", f"**Report** — scope `{r.get('scope')}` · trend `{r.get('trend', {}).get('direction')}`",
             f"- Weak topics: {[w.get('label') or w.get('topic_id') for w in r.get('weak_topics') or []] or '—'}",
             f"- Root cause (deepest weak prerequisite): "
             f"{rc.get('label') if rc else '— (weak topics are themselves the root)'}",
             "", r.get("narrative", "")]
    if r.get("visual_spec"):
        params = (r["visual_spec"].get("params") or {})
        vals = params.get("series", [{}])[0].get("values")
        lines.append(f"- Trend chart data: {vals}")
        # Render the trend chart to an image so it is viewable, not just the raw data.
        try:
            ref = gen_chart(params, user_id=DEMO_STUDENT, config=get_config())
            if ref.get("s3_uri"):
                lines.append(f"- Trend chart: [chart]({ref['s3_uri']})")
        except Exception as e:                       # noqa: BLE001 — demo resilience
            lines.append(f"- _(trend chart render failed: {str(e)[:60]})_")
    return lines


def _mastery_block(rows: list[dict], title: str) -> list[str]:
    if not rows:
        return [f"- {title}: _(no mastery on file for this student yet)_"]
    names = _topic_names([r.get("topic_id") for r in rows])
    shown = [f"{names.get(r.get('topic_id'), '?')} = {round(r.get('score') or 0, 2)} ({r.get('status')})"
             for r in rows[:8]]
    return [f"- {title}:"] + [f"    - {s}" for s in shown]


# ===========================================================================
# One chat request
# ===========================================================================

def _run_chat(graph, question: str, subject_id, note: str, *, user_type="student",
              lecture_ids=None, user_id=DEMO_STUDENT) -> tuple[str, float]:
    meter = start_meter()
    started = time.perf_counter()
    payload = {"message": question, "user_type": user_type,
               "subject_id": subject_id, "user_id": user_id}
    if lecture_ids:
        payload["lecture_ids"] = lecture_ids
    state, steps = _invoke_timed(graph, payload)
    elapsed = time.perf_counter() - started
    cost = meter.as_dict()

    tasks = state.get("tasks", [])
    topics: list[str] = []
    for t in tasks:
        for label in t.get("params", {}).get("topic_labels", []):
            if label not in topics:
                topics.append(label)
    used_subject = SUBJECTS.get(state.get("subject_id"), state.get("subject_id")) or "(unscoped)"

    out = [
        f"### {note}",
        f"**Q:** {question}",
        f"- Subject: {used_subject} · Tasks: `{', '.join(t.get('type') for t in tasks) or '—'}`"
        f" · Topics: {', '.join(topics) if topics else '—'}",
        f"- Cost: ${cost['cost_usd']:.4f} ({cost['calls']} calls, "
        f"{cost['prompt_tokens']}+{cost['completion_tokens']} tok) · **Time: {elapsed:.2f}s**",
        f"- Steps: {_steps_line(steps)}",
        "",
        state.get("response", "(no response)"),
    ]
    if state.get("assessment"):
        out += _assessment_block(state["assessment"])
    if state.get("grade_result"):
        out += _grade_block(state["grade_result"])
    if state.get("visuals"):
        out += _visuals_block(state["visuals"])
    out.append("\n---\n")
    return "\n".join(out), elapsed


# ===========================================================================
# Curated suite
# ===========================================================================

# (question, subject_id, note) — mixed subjects AND topics, not one lesson.
ASSESSMENTS = [
    ("Make me an assessment on oxidation and reduction.", SCIENCE_EN, "Assessment · Science EN · default mix"),
    ("Give me 4 hard MCQs on the mean proportional.", MATH_EN, "Assessment · Math EN · count+type+difficulty"),
    ("اصنع لي اختبارًا عن الأكسدة والاختزال.", SCIENCE_EN, "Assessment · Arabic request on EN subject → subject language"),
    ("اختبرني في العوامل التي تؤدي إلى قيام الدول وسقوطها.", SOCIAL_AR, "Assessment · Social Studies AR"),
    ("Fais-moi un test sur la loi de ségrégation de Mendel.", SCIENCE_FR, "Assessment · Science FR"),
    ("Create a quiz on solving real-world proportion problems.", MATH_EN, "Assessment · Math EN · word problems"),
    ("Make me a quiz on the best chocolate cake recipe.", MATH_EN, "Assessment · off-syllabus → decline"),
]

EXPLAIN = [
    ("Explain the difference between oxidizing and reducing agents.", SCIENCE_EN,
     "Explain · Science EN · grounded + citations"),
    ("Explain how to find the mean proportional, and draw an example on a graph.", MATH_EN,
     "Explain + Visual · Math EN (explain task + math figure in ONE turn; is_math → LaTeX)"),
    ("Résume la loi de ségrégation de Mendel.", SCIENCE_FR, "Summarize · Science FR (subject language)"),
    ("لخّص العوامل التي تؤدي إلى قيام الدول وسقوطها.", SOCIAL_AR, "Summarize · Social Studies AR"),
    ("Explain the fire triangle and the methods of extinguishing fires.", SCIENCE_EN,
     "Explain · two concepts → ONE merged task (2E B3)"),
    ("Explain how to bake the perfect sourdough.", SCIENCE_EN, "Explain · off-syllabus → decline"),
]

VISUALS = [
    ("اصنع خريطة ذهنية عن العوامل التي تؤدي إلى قيام الدول وسقوطها.", SOCIAL_AR,
     "Visual · mind map (Arabic, independent task)"),
    ("Explain oxidizing and reducing agents and show me a mind map of it.", SCIENCE_EN,
     "Visual · mind map RIDING on an explain task (two tasks in one turn)"),
    ("Make flashcards for solving real-world proportion problems.", MATH_EN, "Visual · flashcards (Math EN)"),
    ("Make a PDF summary of Mendel's laws.", SCIENCE_FR, "Visual · PDF (Science FR)"),
    ("Draw the graph of y = x^2 for x from -3 to 3.", MATH_EN, "Visual · math figure (Math EN)"),
    ("Show me a right triangle with sides 3, 4 and 5.", MATH_EN, "Visual · geometry (Math EN)"),
    ("Make slides about oxidation and reduction.", SCIENCE_EN, "Visual · slides (outline text)"),
    # NOTE: illustration/image generation is deferred — the visual kinds are the deterministic
    # ones (mind map, flashcards, PDF, math graph, geometry, slides).
    # NOTE: charts are DATA-TIED — a bare "make a chart" has no data to plot, so it's skipped.
    # A real chart is demonstrated in the Reports section (the mastery trend line).
]

GRADING_CHAT = [
    ("Grade my answer. Question: In terms of electrons, what is oxidation? "
     "My answer: oxidation is the loss of electrons.", SCIENCE_EN, "Grade · Science EN free text, correct"),
    ("Grade my answer. Question: What is the mean proportional between 4 and 9? "
     "My answer: 5.", MATH_EN, "Grade · Math EN free text, wrong (correct is 6)"),
    ("صحّح إجابتي. السؤال: اذكر عاملين يؤديان إلى سقوط الدول. إجابتي: الفساد والانقسام الداخلي.",
     SOCIAL_AR, "Grade · Social Studies AR free text"),
    ("Grade my answer: the capital of France is Paris.", SCIENCE_EN,
     "Grade · no key + off-syllabus → DECLINE (never guesses from the model's own knowledge)"),
]


TEACHER = [
    ("Create a 12-question exam covering oxidation, reduction and combustion.", SCIENCE_EN,
     "Teacher · large exam"),
    ("Explain continued proportion for my lesson plan.", MATH_EN,
     "Teacher · Math explain (professional tone; is_math)"),
    ("Fais-moi un quiz sur la dominance complète.", SCIENCE_FR, "Teacher · assessment (Science FR)"),
]


def _special_cases(fh, graph) -> float:
    """Lesson-level scoping and subject inference — the two things a flat query list can't show."""
    started = time.perf_counter()
    # Lecture-scoped: same subject, but pinned to ONE lesson via lecture_ids (payload filter).
    social_lesson = "73e4c47d-8bd1-5652-9596-b8bb10b79c8f"    # Social AR · unit2 lesson2
    block, _ = _run_chat(
        graph, "اختبرني في هذا الدرس فقط.", SOCIAL_AR,
        "Lesson-scoped · assessment pinned to ONE lecture via lecture_ids",
        lecture_ids=[social_lesson])
    fh.write(block + "\n")
    # Subject inference: NO subject_id given — inferred, constrained to the student's enrolled
    # subjects (2E B4). Should land on Science from the fixture enrolment.
    block, _ = _run_chat(
        graph, "Explain how to find the mean proportional between two numbers.", None,
        "Inference · no subject given → inferred (should land on Math) from enrolled subjects (2E B4)")
    fh.write(block + "\n")
    fh.flush()
    return time.perf_counter() - started


def _grade_from_db(fh, graph) -> float:
    """Grade against a SAVED quiz's rubric (fixture) — the correct answers come from the
    'DB' via assessment_id, not inline. This is the teacher/assigned-work path (source=quiz)."""
    started = time.perf_counter()
    # Same ORDER as demo-quiz-1 (rubric merges by position): mcq · true_false · short · essay.
    # lecture_id rides on the submission (as the backend sends it) so mastery can move —
    # invariant 12; topic_id is resolved from the question text. The rubric supplies the key.
    lec = SCIENCE_EN_LECTURE
    submissions = [
        {"question": "Which of these is NOT part of the fire triangle?",
         "student_answer": "Water", "type": "mcq", "lecture_id": lec},        # correct MCQ
        {"question": "True or False: removing any one side of the fire triangle puts the fire out.",
         "student_answer": "False", "type": "true_false", "lecture_id": lec}, # wrong T/F
        {"question": "What are the three components of the fire triangle?",
         "student_answer": "heat, fuel and oxygen", "type": "short_answer", "lecture_id": lec},
        {"question": "Explain why removing oxygen extinguishes a fire.",
         "student_answer": "Fire needs oxygen to burn, so taking the oxygen away stops the "
         "combustion reaction and the fire goes out.", "type": "essay", "lecture_id": lec},
    ]
    result = run_grade({"user_id": DEMO_STUDENT, "subject_id": SCIENCE_EN,
                        "kind": "quiz", "assessment_id": "demo-quiz-1",
                        "submissions": submissions})
    fh.write("### Grade from DB — /ai/grade with assessment_id (rubric fetched, not inline)\n")
    fh.write("\n".join(_grade_block(result)) + "\n\n---\n")
    fh.flush()
    return time.perf_counter() - started


def _section(fh, title: str, blurb: str) -> None:
    fh.write(f"\n\n## {title}\n\n_{blurb}_\n\n")
    fh.flush()
    print(f"\n=== {title} ===")


def _run_list(fh, graph, items, section_total: dict, *, user_type="student",
              user_id=DEMO_STUDENT) -> None:
    for q, subj, note in items:
        block, elapsed = _run_chat(graph, q, subj, note, user_type=user_type, user_id=user_id)
        fh.write(block + "\n")
        fh.flush()
        section_total["t"] += elapsed
        print(f"  [{elapsed:5.1f}s] {note}")


# ===========================================================================
# The mastery loop — quiz -> grade (wrong) -> mastery moves -> next quiz adapts
# ===========================================================================

def _mastery_loop(fh, graph) -> float:
    """Generate a quiz, grade its items WRONG via the endpoint (moves mastery), then read
    mastery back and generate a second quiz to show the weak topic surface first."""
    started = time.perf_counter()
    fh.write("### Step 1 — generate a quiz (capture its items)\n")
    meter = start_meter()
    state, _ = _invoke_timed(graph, {
        "message": "Make me a 3-question assessment on oxidation and reduction.",
        "user_type": "student", "subject_id": SCIENCE_EN, "user_id": DEMO_STUDENT})
    items = (state.get("assessment") or {}).get("items") or []
    fh.write(f"- generated {len(items)} items · ${meter.as_dict()['cost_usd']:.4f}\n")

    if not items:
        fh.write("\n_No items generated — skipping the loop (needs the ingested store)._\n\n---\n")
        return time.perf_counter() - started

    topic_ids = [it.get("topic_id") for it in items if it.get("topic_id")]
    fh.write("\n".join(["", "### Step 2 — mastery BEFORE grading"]
                       + _mastery_block(_safe_mastery(topic_ids), "before")) + "\n")

    # Answer every item WRONG so mastery on those topics drops.
    submissions = [{
        "question": it.get("question"), "student_answer": "(intentionally wrong)",
        "correct_answer": (", ".join(it["answer"]) if isinstance(it.get("answer"), list)
                           else it.get("answer")),
        "topic_id": it.get("topic_id"), "lecture_id": it.get("lecture_id"),
        "difficulty": it.get("difficulty"), "type": it.get("type"),
    } for it in items]

    # Grade the SAME items wrong THREE times: one attempt reads as 'unknown' (min_attempts=2),
    # so it takes >=2 to cross into 'weak' and show up in the report / drive adaptivity.
    rounds = 3
    fh.write(f"\n### Step 3 — grade the answers (all wrong) {rounds}× via /ai/grade endpoint\n")
    fh.write("_(one attempt = 'unknown'; it takes 2+ to read as 'weak')_\n")
    for r in range(1, rounds + 1):
        g_started = time.perf_counter()
        result = run_grade({"user_id": DEMO_STUDENT, "subject_id": SCIENCE_EN,
                            "kind": "quiz", "submissions": submissions})
        fh.write(f"- round {r}: graded in {time.perf_counter() - g_started:.2f}s · "
                 f"mastery events {result.get('mastery_events', 0)} · "
                 f"topics updated {result.get('topics_updated', 0)}\n")
    fh.write("\n".join(_grade_block(result)) + "\n")

    fh.write("\n".join(["", f"### Step 4 — mastery AFTER {rounds} attempts (should read 'weak')"]
                       + _mastery_block(_safe_mastery(topic_ids), "after")) + "\n")

    fh.write("\n### Step 5 — next quiz should now put the weak topic first\n")
    # The weak topics (from the mastery just written) that adaptivity should surface first.
    weak_names = {_topic_name(r["topic_id"]) for r in _safe_mastery(topic_ids)
                  if r.get("status") == "weak"}
    fh.write(f"- weak topics now: {sorted(weak_names) or '—'}\n")
    state2, _ = _invoke_timed(graph, {
        "message": "Quiz me again on this subject.", "user_type": "student",
        "subject_id": SCIENCE_EN, "user_id": DEMO_STUDENT})
    items2 = (state2.get("assessment") or {}).get("items") or []
    order = [_topic_name(it.get("topic_id")) for it in items2]
    # Annotate each covered topic so the weak-first ordering is visible at a glance.
    annotated = [f"{n} (weak)" if n in weak_names else n for n in order]
    fh.write(f"- next quiz topic order: {annotated}\n")
    first_is_weak = bool(order) and order[0] in weak_names
    fh.write(f"- weak topic surfaced first: {'yes ✓' if first_is_weak else 'no'}\n\n---\n")
    fh.flush()
    return time.perf_counter() - started


def _safe_mastery(topic_ids) -> list[dict]:
    try:
        return get_mastery(DEMO_STUDENT, topic_ids)
    except Exception as e:                       # noqa: BLE001 — demo resilience
        return [{"topic_id": "error", "score": 0, "status": str(e)[:40]}]


# ===========================================================================
# Reports
# ===========================================================================

def _reports(fh, graph) -> float:
    started = time.perf_counter()
    fh.write("### Endpoint — GET /ai/reports (run_report, student_progress)\n")
    r_started = time.perf_counter()
    report = run_report({"user_id": DEMO_STUDENT, "subject_id": SCIENCE_EN,
                         "scope": "student_progress"})
    fh.write(f"- built in {time.perf_counter() - r_started:.2f}s\n")
    fh.write("\n".join(_report_block(report)) + "\n\n---\n")

    # Subject/lesson-level mastery rollup (attempts-weighted mean) — the teacher/parent view.
    try:
        by_lecture = get_mastery_rollup(DEMO_STUDENT, "lecture")
        fh.write("\n### Mastery rollup (per lecture)\n")
        for r in by_lecture[:6]:
            fh.write(f"- {_lecture_name(r.get('lecture_id'))} → score {round(r.get('score') or 0, 2)} "
                     f"({r.get('status')}) · strong {r.get('strong')}/weak {r.get('weak')}\n")
        fh.write("\n---\n")
    except Exception as e:                       # noqa: BLE001 — demo resilience
        fh.write(f"\n_(rollup unavailable: {str(e)[:60]})_\n\n---\n")

    # NOTE: the chat 'how am I doing' turn is intentionally omitted — routing report intent from
    # free chat is deferred (see notes/sprints.md, TODO(sprint3-router)). The GET /ai/reports
    # endpoint above already demonstrates the full report + trend chart.
    fh.flush()
    return time.perf_counter() - started


# ===========================================================================
# main
# ===========================================================================

class _WarnCapture(logging.Handler):
    """Collect WARNING+ log records so a DEV run can list what degraded. Off the console —
    the client never sees these; they only surface in the transcript when DEMO_DEV is set."""
    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.records: list[str] = []

    def emit(self, record):
        self.records.append(f"[{record.levelname}] {record.name}: {record.getMessage()}")


def _configure_logging(config, dev: bool) -> _WarnCapture:
    """Client run = quiet console (errors only), no warnings block. DEV run (DEMO_DEV=1) =
    verbose console + a captured-warnings section at the end."""
    setup_logging(config)
    root = logging.getLogger()
    root.setLevel(logging.INFO if dev else logging.WARNING)
    for h in root.handlers:                       # the stdout handler from setup_logging
        h.setLevel(logging.INFO if dev else logging.ERROR)
    cap = _WarnCapture()
    root.addHandler(cap)
    return cap


def _reset_demo_student(config) -> None:
    """Clear the demo student's mastery so a run starts clean/deterministic. OFF by default;
    only runs when DEMO_RESET=1. Touches ONLY the demo student's rows."""
    from agent.tools import connect
    with connect(config) as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM ai.mastery_events WHERE user_id = %s", (DEMO_STUDENT,))
        cur.execute("DELETE FROM ai.mastery WHERE user_id = %s", (DEMO_STUDENT,))
        conn.commit()
    print(f"[reset] cleared mastery for demo student {DEMO_STUDENT[:8]}…")


def _enable_fixtures(config) -> bool:
    """Point the backend client at the demo fixtures so subject facts, enrolled subjects,
    and the saved-quiz rubric resolve offline. Retrieval + mastery still use real Postgres.
    The fixtures file is gitignored (mirrors backend data) — when absent the demo still runs,
    just without enrolled-inference and grade-from-DB. Returns whether fixtures are active."""
    if not FIXTURE_PATH.exists():
        print(f"[note] {FIXTURE_PATH.name} not found — running without backend fixtures "
              "(no enrolled inference / grade-from-DB).")
        return False
    backend = config._data.setdefault("agent", {}).setdefault("backend", {})   # noqa: SLF001 — demo
    backend["fixtures_path"] = str(FIXTURE_PATH)
    backend["prefer_fixtures"] = True
    return True


def main() -> int:
    dev = bool(os.environ.get("DEMO_DEV"))        # DEMO_DEV=1 -> verbose + a warnings section (Mahy only)
    # Save GENERATED visuals (charts, mind maps, flashcards, PDFs, images) under the demo's
    # outputs folder — committed, so a reviewer can open them next to the transcript — instead of
    # the gitignored .ingestion_artifacts. Only affects the local uploader (ignored when S3 is set).
    os.environ.setdefault("LOCAL_ARTIFACT_DIR",
                          str(ROOT / "demos" / "sprint2" / "outputs"))
    config = get_config()
    fixtures_on = _enable_fixtures(config)
    warnings = _configure_logging(config, dev)
    if os.environ.get("DEMO_RESET"):              # off by default; clears the demo student's mastery
        _reset_demo_student(config)
    graph = build_graph(config)

    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    # A FRESH, date-stamped file per run — so every run (yours or a non-dev reviewer's) is its own
    # clean transcript, never piling into one file. All sprint2 outputs are gitignored (local only).
    # Override the name with $env:DEMO_OUT.
    file_stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_name = Path(os.environ.get("DEMO_OUT", f"sprint2-run-{file_stamp}.md")).name
    dest = ROOT / "demos" / "sprint2" / "outputs" / out_name
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Custom mode: run the user's own question(s) as a student, then stop.
    if len(sys.argv) > 1:
        key = os.environ.get("SUBJECT", "").strip().lower()
        subj = SUBJECT_KEYS.get(key, key or None)
        with dest.open("a", encoding="utf-8") as fh:
            fh.write(f"\n\n{'='*70}\n# Custom run {stamp}\n{'='*70}\n\n")
            for q in sys.argv[1:]:
                block, elapsed = _run_chat(graph, q, subj, "custom request")
                fh.write(block + "\n")
                fh.flush()
                print(f"[{elapsed:.1f}s] {q}")
        print(f"\nSaved to {dest}")
        return 0

    # Full tour.
    section_t: dict[str, float] = {}
    total_started = time.perf_counter()
    with dest.open("a", encoding="utf-8") as fh:
        fh.write(f"\n\n{'='*70}\n# Sprint 2 full tour — {stamp}\n{'='*70}\n")

        for name, blurb, items in [
            ("1. Assessments", "Different kinds, languages, requested shapes, and an off-syllabus decline.", ASSESSMENTS),
            ("2. Explain & Summarize", "Grounded, cited, multi-concept merge, and a decline.", EXPLAIN),
            ("3. Visuals", "Every kind — standalone and riding on another task.", VISUALS),
            ("4. Grading (chat)", "Free-text right / wrong / decline.", GRADING_CHAT),
        ]:
            _section(fh, name, blurb)
            t = {"t": 0.0}
            _run_list(fh, graph, items, t)
            section_t[name] = t["t"]

        _section(fh, "5. Grade from DB (fixture rubric)", "Grade a saved quiz by assessment_id — key from the DB, not inline. Covers MCQ · true/false · short answer · essay (open-ended is RAG-grounded).")
        if fixtures_on:
            section_t["5. Grade from DB"] = _grade_from_db(fh, graph)
        else:
            fh.write("_Skipped — needs demos/sprint2/backend_fixtures.yaml (gitignored)._\n\n---\n")

        _section(fh, "6. Teacher pathway", "Teacher-only behaviours: large exams, professional tone.")
        t = {"t": 0.0}
        _run_list(fh, graph, TEACHER, t, user_type="teacher", user_id=DEMO_TEACHER)
        section_t["6. Teacher pathway"] = t["t"]

        _section(fh, "7. Lesson scoping & subject inference", "One-lesson scoping and enrolled-subject inference.")
        section_t["7. Lesson scoping & inference"] = _special_cases(fh, graph)

        _section(fh, "8. Mastery loop", "Quiz → grade wrong → mastery moves → next quiz adapts.")
        section_t["8. Mastery loop"] = _mastery_loop(fh, graph)

        _section(fh, "9. Reports", "Endpoint report (root cause + trend chart) and the subject/lesson mastery rollup.")
        section_t["9. Reports"] = _reports(fh, graph)

        total = time.perf_counter() - total_started
        fh.write("\n\n## ⏱ Timing summary\n\n")
        for name, secs in section_t.items():
            fh.write(f"- {name}: {secs:.1f}s\n")
        fh.write(f"\n**Total: {total:.1f}s**\n")
        # DEV-only: what degraded this run. Written ONLY when DEMO_DEV=1, so a client run
        # (and the committed demo) never shows it.
        if dev and warnings.records:
            fh.write("\n\n## ⚠️ Warnings (dev only)\n\n")
            for line in warnings.records:
                fh.write(f"- {line}\n")
        fh.flush()

    print("\n--- timing summary ---")
    for name, secs in section_t.items():
        print(f"{name}: {secs:.1f}s")
    print(f"TOTAL: {time.perf_counter() - total_started:.1f}s")
    print(f"\nTranscript written to {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
