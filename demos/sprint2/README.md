# Sprint 2 Demo — Assessments, grading, mastery, visuals, and reports

Sprint 1 delivered a study assistant that answers questions from the course material.
Sprint 2 builds the teaching loop on top of it: it **generates assessments**, **grades
answers**, **tracks each student's mastery skill by skill**, **produces study visuals**,
and **reports on progress** — every output grounded in the same syllabus material.

> **Scope of this demo.** Results are produced on a **small sample of ingested lessons**
> across four subjects and languages — Science (English: combustion; French: genetics),
> Mathematics (English: proportions), and Social Studies (Arabic: history). It demonstrates
> the features working across subjects and languages, not the full syllabus.

## What was delivered

- **Assessment generation.** Quizzes, homework, exams, and practice sets, built from the
  syllabus. The number of questions, the question types (multiple-choice, true/false, short
  answer, essay), and the difficulty follow the request; when they are not specified, the
  system applies sensible defaults. Every question carries an answer key and an explanation,
  and each is tied to a specific topic and lecture. An off-syllabus request is declined
  rather than answered from outside the material.
- **Grading.** Free-text answers and saved quizzes are graded. A reference answer or rubric
  is used first; where none exists, the answer is graded against the syllabus material, and
  when neither is available the system declines rather than guessing. Each result explains
  what was right or wrong and gives the correct answer.
- **Per-skill mastery.** Grading updates a difficulty-weighted mastery score for each topic.
  Repeated weak performance marks a topic as needing work, which then feeds the next
  assessment (weak topics are surfaced first) and the progress report.
- **Study visuals.** Mind maps, flashcards, PDFs, mathematical graphs and geometry figures,
  and slide outlines — produced on their own or alongside an explanation. All are drawn
  deterministically for precise, reliable output.
- **Progress reports.** A student-facing progress note with the trend over recent work, the
  weak topics to focus on, and the deepest prerequisite behind them (the likely root cause),
  accompanied by a trend chart.
- **Teacher pathway.** Teacher requests are supported with a professional tone and larger
  assessments.
- **Lesson scoping and subject inference.** A request can be pinned to one lecture, and when
  no subject is given it is inferred from the enrolled subjects.

## How it works, in brief

Each request flows through one runtime graph: the language and role are detected, the
request is turned into one or more tasks (explain, summarise, assessment, grade, visual, or
report), the relevant syllabus material is retrieved, each task is handled by its own
component, and the results are merged into a single reply in the language of the request.
Grading is the only step that updates mastery, and it does so only from a reference answer
or the syllabus material — never from outside knowledge.

## Technology

- **Language models (via OpenRouter):** Gemini 2.5 Flash for content generation (assessment,
  grading, explanations, reports) and Gemini 2.5 Flash-Lite for the lighter planning and
  composition steps; Qwen3-VL reads scanned pages and figures during ingestion.
- **Search:** BGE-M3 multilingual embeddings in a **Qdrant** vector index — one index for all
  languages, so a question in one language can retrieve material written in another.
- **Data:** **PostgreSQL** for structured content and mastery; **Amazon S3** for generated
  artifacts and figures (a local folder is used when S3 is not configured).
- **Orchestration:** **LangGraph** (Python).

## What the demo shows

The demo runs a guided tour across four subjects and three languages and records, for each
request, the detected language, the tasks, the topics matched, a per-step time breakdown,
and the full result. It covers:

1. **Assessments** — different kinds, languages, requested shapes, and an off-syllabus decline.
2. **Explain & summarise** — grounded and cited, a multi-concept merge, and a decline.
3. **Visuals** — every kind, standalone and alongside another task.
4. **Grading (chat)** — correct, wrong with the correct answer and explanation, and a decline.
5. **Grading from a saved quiz** — graded against the stored rubric (multiple-choice,
   true/false, short answer, essay).
6. **Teacher pathway** — larger exams and a professional tone.
7. **Lesson scoping and subject inference.**
8. **The mastery loop** — a quiz is graded, mastery moves, and the next quiz surfaces the
   weak topic first.
9. **Reports** — the progress report with trend and root cause, and a per-lecture rollup.

## How to run

The system must be set up first — dependencies installed, `.env` configured, and the sample
lessons ingested — as described in the [main README](../../README.md). A few items are needed
for the full demo, and it degrades gracefully without the last two:

1. **Use a dedicated Qdrant collection.** Before ingesting the sample lessons, rename the local
   collection in `config.yaml`, under `environments: local: vector_store:`, so it does not
   collide with any existing collection:

   ```yaml
       vector_store: { collection: eduai_chunks_yourname }
   ```

   Ingest the lessons into this collection; the demo then reads from it under `APP_ENV=local`.
2. **Backend fixtures.** Place the demo's offline backend file at
   `demos/sprint2/backend_fixtures.yaml`. It stands in for the backend API (subject facts,
   enrolled subjects, and a saved quiz's rubric) so the demo runs without a live backend.
   Without it, sections 5 (grading from a saved quiz) and 7 (enrolled-subject inference) are
   skipped and the rest of the tour runs normally.
3. **Database alignment (local Postgres only).** If running against a local database that
   contains only the `ai.*` tables, apply the alignment script once so the demo can record
   generated artifacts and mastery:

   ```bash
   psql "$DATABASE_URL" -f scripts/demo_db_align.sql
   ```

   It is non-destructive and safe to re-run. It is not needed against the full backend
   database, which already has these columns.

**Run the guided tour** (PowerShell, from the repository root):

```bash
$env:APP_ENV="local"; .venv\Scripts\python.exe demos\sprint2\python\demo.py
```

**Ask your own question(s) instead** — pass each in quotes, in any supported language:

```bash
$env:APP_ENV="local"; .venv\Scripts\python.exe demos\sprint2\python\demo.py "Explain the fire triangle" "Quiz me on oxidation"
```

Pin a subject with `$env:SUBJECT="science-en"` (`science-en` | `science-fr` | `math-en` |
`social-ar`) before the command; clear it afterwards with `Remove-Item Env:SUBJECT`.

## Demo output

Each run writes a new, time-stamped transcript to
`demos/sprint2/outputs/sprint2-run-<YYYYMMDD-HHMMSS>.md`, and any generated visuals are saved
under `demos/sprint2/outputs/artifacts/` and linked from the transcript. The console stays
quiet (errors only) and the transcript carries no technical traces — only the questions, the
results, and the per-step timing.
