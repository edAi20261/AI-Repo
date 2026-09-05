# Sprint 1 Demo — Bilingual study assistant

This sprint built the foundation of the study assistant: it reads a course's own
material and answers students' questions **using only that material**, in Arabic,
English, or French.

> **Scope of this demo.** These results were produced on a **small sample of 5 ingested
> lessons** spanning different subjects and languages — Science (English: combustion;
> French: genetics), Mathematics (English: proportions), and Social Studies (Arabic:
> history). It demonstrates the system working across subjects and languages, not the
> full syllabus; the complete content is ingested in a later sprint.
>
> **Known limitation (being refined).** Re-processing a lesson currently reuses most of its
> existing topics but can occasionally create a duplicate topic when the wording of a skill
> shifts between runs. Existing topics are never lost. A later sprint replaces the wording-based
> match with a content-based one so re-processing reliably reuses topics.

## What was delivered

- **Reads the course material.** Uploaded lecture files — including scanned pages
  and diagrams — are processed once and turned into a searchable knowledge base.
  Each lecture is broken into passages, its key topics are identified, and the
  order topics build on each other (prerequisites) is worked out automatically.
- **Understands the question in either language.** A student can ask in Arabic or
  English; the assistant detects the language and replies in the same one.
- **Finds the right material, even across languages.** A question in English can
  surface the relevant Arabic passage, and vice versa — the course is indexed once,
  not once per language.
- **Answers from the material, with sources.** The reply is grounded in the
  retrieved passages and cites where each point came from, so it can be checked
  against the textbook. If the material doesn't cover the question, it says so
  rather than inventing an answer.
- **Points to prerequisites.** When a topic depends on earlier ones, the answer
  ends with the prerequisite topics the student may need to review first.

Re-processing a lecture later (for example after an edit) keeps each topic's
identity stable, so any progress data tied to it is preserved.

## How it works, in brief

1. **Once per lecture (offline):** each uploaded lesson file is read — including scanned
   pages and diagrams — split into passages, and its key skills (topics) and the order they
   build on each other (prerequisites) are worked out. Everything is turned into a
   searchable form.
2. **For each question (live):** the assistant detects the language, finds the most relevant
   passages **within the student's subject** (choosing the subject automatically from the
   question when the student hasn't opened one), and writes an answer grounded in them — with
   the **sources** it used and any **prerequisite topics** — in the language the student asked
   in (or a specific language if they request one).

**In:** a student's question (and their subject). **Out:** a grounded answer, the sources it
used, and prerequisite topics to review. The right lecture is found automatically by meaning,
not chosen in advance, and a question in one language can find material written in another.

## Technology

- **Language models (via OpenRouter):** Qwen3-VL reads scanned pages and figures; Kimi-K2.5
  identifies topics and prerequisites and writes the answers.
- **Search:** BGE-M3 multilingual embeddings in a **Qdrant** vector index — one index for all
  languages, so a question in one language can retrieve material in another.
- **Data:** **PostgreSQL** for the structured content; **Amazon S3** for figures.
- **Orchestration:** **LangGraph** (Python).

## What the demo shows

The demo asks a set of real questions in **English, Arabic, and French**, across
**Science, Mathematics, and Social Studies**, and prints for each: the language it
detected, the answer grounded in the course material, the **sources** it used, and
the **prerequisite topics** it surfaced. It includes:

- **Same-language** questions (e.g. a French question answered from the French lesson).
- **Cross-language** questions — an English question that finds and answers from the
  Arabic lesson, and vice versa — since the material is indexed once, not per language.
- **Ask-and-summarise** questions that both explain a concept *and* summarise a related
  one in a single reply.

## How to run

The knowledge base must be populated first (the course material processed). Run these
from the project root.

**Run the prepared demo questions:**

```bash
AI-Workflows/.venv/Scripts/python.exe AI-Workflows/demos/sprint-1/python/demo.py
```

**Ask your own question(s)** — pass them as arguments, in any supported language; the
assistant works out the subject automatically:

```bash
AI-Workflows/.venv/Scripts/python.exe AI-Workflows/demos/sprint-1/python/demo.py "Explain the fire triangle."
```

Several at once (English, Arabic, French — quote each one):

```bash
AI-Workflows/.venv/Scripts/python.exe AI-Workflows/demos/sprint-1/python/demo.py "Explain the fire triangle." "اشرح مثلث الحريق" "Expliquez les lois de Mendel."
```

To answer in a specific language, just ask for it in the question, e.g.
`"Explain the fire triangle. Please answer in Arabic."`

**Pin the subject (optional).** By default the assistant infers the subject from your
question. To force one, set the `SUBJECT` variable to a key — `science-en`, `science-fr`,
`math-en`, or `social-ar` — before the command (PowerShell):

```bash
$env:SUBJECT="science-fr"; AI-Workflows/.venv/Scripts/python.exe AI-Workflows/demos/sprint-1/python/demo.py "Explain heredity."
```

Clear it afterwards with `Remove-Item Env:SUBJECT`.

Progress is printed to the terminal; the full answers (with Arabic and French intact) are
appended to [`outputs/graph_demo.md`](outputs/graph_demo.md).

## Demo output

The full transcript of each run is saved to
[`outputs/graph_demo.md`](outputs/graph_demo.md). Runs are appended with a dated divider,
so several accumulate in one file. For every question it records the **subject used**, the
**detected language**, the **task(s)**, the **topics matched**, the number of **sources**
(citations), any **prerequisite topics**, **figures on file** for the cited lecture, the
**per-question cost**, and the full **answer**.

What the runs demonstrate:

- **Grounded answers with sources** — the fire-triangle question returns a clear explanation
  citing the exact syllabus pages, ending with a prerequisite topic to review.
- **Cross-language retrieval** — an English question over the *French* genetics lesson is
  answered in English from the French material; an Arabic question over the *English* science
  lesson is answered in Arabic. The course is indexed once, not once per language.
- **Answer language on request** — "Explain the fire triangle. Please answer in Arabic." is
  answered in Arabic although the question is in English.
- **Explain and summarise in one turn** — a single question can both explain one concept and
  summarise another, merged into one reply.
- **Off-syllabus questions** are politely declined rather than answered from thin air.
- **No subject needed** — when the student doesn't specify a subject, the assistant infers it
  from the question (shown as `Subject used: … (inferred)`).

Each answer also shows its **cost** (a few tenths of a US cent per question with the current
models), so the running cost of a session is visible.
