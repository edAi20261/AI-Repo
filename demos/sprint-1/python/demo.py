"""Demo — grounded answers over the ingested lectures.

Runs a set of questions (English, Arabic, French) through the runtime graph and, for
each, reports what the system did: detected language, matched topics, citations
(grounding), prerequisite topics, and figures on file for the cited lecture. A UTF-8
transcript is written to ../outputs/answers.md (the console garbles Arabic).

    python demos/sprint-1/python/demo.py                 # runs DEMO_QUESTIONS below
    python demos/sprint-1/python/demo.py "your question" # runs your own instead

Needs the ingested stores up and OPENROUTER_API_KEY set (both from .env).
"""

from __future__ import annotations

import datetime
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]        # -> AI-Workflows/
sys.path.insert(0, str(ROOT))

from agent.config import get_config               
from agent.graph import build_graph               
from shared.logging import setup_logging          
from shared.metering import start_meter         

# Subject ids (from the fixtures). They scope retrieval so a question only searches
# its own subject.
SCIENCE_EN = "2dfb991c-45d8-5e66-93da-a764df58ccfe"
SCIENCE_FR = "917f7aed-4a03-576b-91e9-1d5cc7beeca9"
MATH_EN = "920d019b-afa8-5bf1-bd73-58487bb3ffab"
SOCIAL_AR = "04c913e1-1c75-5f62-b793-a34ee3aeacf2"

# ---------------------------------------------------------------------------
# Questions the demo runs by default. Add or remove entries freely.
# Each entry is: (question, subject, note)
#   - question: English, Arabic, or French
#   - subject : one of the constants above, or None to let the AI infer it
#   - note    : a short label shown above the answer
# ---------------------------------------------------------------------------
DEMO_QUESTIONS = [
    ("Explain the fire triangle and its three components.", SCIENCE_EN,
     "Science EN · explain"),
    ("Summarize the different methods used to extinguish fires.", SCIENCE_EN,
     "Science EN · summarize"),
    ("ما العوامل التي تؤدي إلى قيام الدول التاريخية وسقوطها؟", SOCIAL_AR,
     "Social Studies AR · explain"),
    ("Expliquez les lois de Mendel sur l'hérédité.", SCIENCE_FR,
     "Science FR · explain"),
    ("How do you find the mean proportional between two numbers?", MATH_EN,
     "Math EN · explain"),
    ("Explain the fire triangle. Please answer in Arabic.", SCIENCE_EN,
     "English question, answer forced to Arabic"),
    ("What is the best recipe for a chocolate cake?", MATH_EN,
     "Off-syllabus · should decline"),
]

SUBJECTS = {
    SCIENCE_EN: "Science EN", SCIENCE_FR: "Science FR",
    MATH_EN: "Math EN", SOCIAL_AR: "Social Studies AR",
}

# Friendly keys for scoping custom questions via the SUBJECT env var.
SUBJECT_KEYS = {
    "science-en": SCIENCE_EN, "science-fr": SCIENCE_FR,
    "math-en": MATH_EN, "social-ar": SOCIAL_AR,
}

# Short friendly names for the five lectures (for the figures line).
LECTURES = {
    "07aef7b6-3e50-5598-a544-614dafca12ac": "Social Studies AR (unit3)",
    "73e4c47d-8bd1-5652-9596-b8bb10b79c8f": "Social Studies AR (unit2)",
    "f7d65f67-87ea-5f90-bd61-c5f343ddb35f": "Science EN (combustion)",
    "24120522-c283-58d1-93a6-1b7388fd21ad": "Math EN (proportions)",
    "d2380fc3-0624-5cd4-ab28-eb91188c814a": "Science FR (genetics)",
}


def _figure_counts(lecture_ids, config) -> dict[str, int]:
    """Figures on file per lecture — evidence the figure pipeline ran end to end."""
    ids = [x for x in {lid for lid in lecture_ids if lid}]
    if not ids:
        return {}
    raw = config.secret("DATABASE_URL")
    if not raw:  # demo degrades instead of crashing when no DB is configured (O1)
        return {}
    import psycopg

    url = raw.replace("+asyncpg", "").replace("+psycopg", "")
    conn = psycopg.connect(url)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT lecture_id::text, count(*) FROM ai.figures "
            "WHERE lecture_id = ANY(%s) GROUP BY lecture_id", (ids,))
        return {r[0]: r[1] for r in cur.fetchall()}
    finally:
        conn.close()


def _run_one(graph, config, question: str, subject_id, note: str) -> tuple[str, float]:
    meter = start_meter()  # capture this query's LLM cost/tokens
    started = time.perf_counter()
    result = graph.invoke({
        "message": question, "user_type": "student", "subject_id": subject_id})
    elapsed = time.perf_counter() - started
    cost = meter.as_dict()

    tasks = result.get("tasks", [])
    task_types = [t.get("type") for t in tasks]
    topics: list[str] = []
    for t in tasks:  # union across tasks (a both-query has two)
        for label in t.get("params", {}).get("topic_labels", []):
            if label not in topics:
                topics.append(label)
    citations = result.get("citations", [])
    prereqs, seen = [], set()
    for p in result.get("prereq_topics", []):
        label = p.get("label")
        if label and label not in seen:
            seen.add(label)
            prereqs.append(label)

    cited_lectures = {c.get("lecture_id") for c in citations}
    figs = _figure_counts(cited_lectures, config)
    fig_line = ", ".join(
        f"{LECTURES.get(lid, lid[:8])}: {n}" for lid, n in figs.items()) or "—"

    used_subject = result.get("subject_id")
    subject_line = (SUBJECTS.get(used_subject, used_subject) if used_subject
                    else "(unscoped — nothing inferred)")

    out = [
        f"## {note}",
        f"**Q:** {question}",
        "",
        f"- **Subject used:** {subject_line}"
        + ("  _(inferred — none was supplied)_" if subject_id is None else ""),
        f"- **Answer language:** `{result.get('language')}` · **Tasks:** "
        f"`{', '.join(task_types) if task_types else '—'}`",
        f"- **Topics matched:** {', '.join(topics) if topics else '—'}",
        f"- **Citations (grounding):** {len(citations)} · "
        f"**Prerequisite topics:** {', '.join(prereqs) if prereqs else '—'}",
        f"- **Figures on file (cited lecture):** {fig_line}",
        f"- **Query cost:** ${cost['cost_usd']:.4f} "
        f"({cost['calls']} LLM calls, {cost['prompt_tokens']}+{cost['completion_tokens']} tokens)",
        f"- **Response time:** {elapsed:.2f}s",
        "",
        result.get("response", "(no response)"),
    ]
    out += _quiz_block(result.get("quiz"))
    out.append("\n---\n")
    return "\n".join(out), elapsed


def _quiz_block(quiz) -> list[str]:
    """Render a generated quiz (items + model answers) when the turn produced one."""
    if not quiz or not quiz.get("items"):
        return []
    lines = ["", f"**Generated quiz** — {quiz.get('count')} items "
             f"(source: {quiz.get('source')}, mix: {quiz.get('type_mix')}):", ""]
    for i, item in enumerate(quiz["items"], 1):
        lines.append(f"{i}. _({item.get('type')})_ {item.get('question')}")
        if item.get("options"):
            lines += [f"   - {opt}" for opt in item["options"]]
        lines.append(f"   **Answer:** {item.get('answer')}")
    return lines


def main() -> int:
    config = get_config()
    setup_logging(config)
    graph = build_graph(config)

    if len(sys.argv) > 1:
        # Custom questions. Optionally scope with SUBJECT (a friendly key or a raw
        # subject uuid); unset -> the agent infers the subject from the question.
        key = os.environ.get("SUBJECT", "").strip().lower()
        subj = SUBJECT_KEYS.get(key, key or None)
        items = [(q, subj, "custom question") for q in sys.argv[1:]]
    else:
        items = DEMO_QUESTIONS

    # Output file is overridable via DEMO_OUT (basename only) so a personal run can
    # write to its own file; defaults to answers.md.
    out_name = Path(os.environ.get("DEMO_OUT", "answers.md")).name
    dest = ROOT / "demos" / "sprint-1" / "outputs" / out_name
    dest.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    divider = (
        "\n\n" + "-" * 70 + "\n"
        f"# Run {stamp} · {len(items)} question(s)\n"
        + "-" * 70 + "\n\n"
    )
    # Append each run (divider first) and flush per answer, so runs accumulate and a
    # crash mid-run still leaves the answers done so far.
    with dest.open("a", encoding="utf-8") as fh:
        fh.write(divider)
        fh.flush()
        for i, (question, subject_id, note) in enumerate(items, 1):
            block, elapsed = _run_one(graph, config, question, subject_id, note)
            fh.write(block + "\n")
            fh.flush()
            # Console-safe: never print raw Arabic/French (cp1252 crashes); status only.
            print(f"[{i}/{len(items)} done] {note} — {elapsed:.2f}s")

    print(f"\nTranscript ({len(items)} questions) written to {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
