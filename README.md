# EduAI — AI Layer

The AI service behind the EduAI learning platform. It answers students from their own
syllabus, holds a conversation across turns, generates and grades practice questions,
tracks per-skill mastery, draws study visuals and writes progress reports — in Arabic,
English, French and German, with each subject taught in its own language.

It is built on LangGraph and split into three independently deployable parts:

- **`ingestion/`** — an offline pipeline that runs once per uploaded lesson. All the
  expensive work happens here: reading pages (including scans, with a vision model),
  chunking, embedding, extracting skills and deriving prerequisites.
- **`agent/`** — the runtime graph a chat turn runs through. It searches the index and
  reads the catalogue; the only things it writes are its own records: conversation
  memory, mastery, mistake tags and rendered artifacts.
- **`system/`** — the HTTP service the platform backend calls. Transport only: it
  validates, authenticates and delegates to the two above.

```
frontend  ->  platform backend  ->  AI service (system/)  ->  agent / ingestion
```

---

## Contents

- [How a turn is answered](#how-a-turn-is-answered)
- [Retrieval and grounding](#retrieval-and-grounding)
- [Ingestion](#ingestion)
- [Where data lives](#where-data-lives)
- [Repository layout](#repository-layout)
- [Getting started](#getting-started)
- [Configuration](#configuration)
- [Running](#running)
- [HTTP API](#http-api)
- [Evaluation and tests](#evaluation-and-tests)
- [Known limitations](#known-limitations)

---

## How a turn is answered

```
START
  → classify          role check; load the conversation (recent turns, summary, state)
  → load_context      retrieval namespaces and the pathway's voice
  → understand        one structured model call reads the message against the conversation
      ├── converse    a social turn: a model-written reply, no retrieval
      └── plan        subject, topics, reply language, the last answer's passages
            → dispatch       one parallel branch per task
                ├── explain_sg / summarize_sg   retrieve → answer → prerequisites
                ├── assessment_sg               questions generated from retrieved passages
                ├── grade_sg                    marks answers; the only writer to mastery
                ├── report_sg                   mastery, trend and root-cause prerequisite
                └── visual_sg                   charts, mind maps, maths figures, flashcards, PDFs, slides
            → render_visuals → compose
  → verify_language   the reply is in the language decided, or is repaired once
  → persist           remember the turn; fold old turns into a running summary
END
```

**1. The conversation is loaded.** Every turn is stored in `ai.user_memory` with the
passages its answer cited. A turn sees the last six turns verbatim; older turns are
folded into a running summary (goals, what was covered, preferences, open questions)
written by a small model. Nothing is re-derived from one message alone.

**2. One model call reads the turn.** `understand` returns a structured reading of the
new message in the light of the conversation:

- the student's own words, separated from any instruction the app put in front of them
  (the frontend sends study modes as a prefix such as `Explain this:`);
- whether the syllabus is needed at all — a greeting, thanks or small talk goes to
  `converse` and is answered without a search;
- the tasks (`explain`, `summarize`, `assessment`, `grade`, `visual`, `report`, filtered to
  the pathway's allowlist), each with a self-contained search query that names what an
  "example" or "that" refers to;
- whether the turn carries the last answer on, or moves to another subject;
- whether the student asked to be answered in a particular language.

There are no keyword lists, seed phrases or phrase rules deciding what a message is.
A reading that contradicts itself is sent back to the model with the reason.

**3. The subject and language are settled.** The subject is the one on the request, else
the one the conversation settled on, else the subject whose material answers the query
best (found by searching the index, confined to a student's enrolled subjects). The reply
language is, in order:

1. a language the student asked for in this conversation — it holds until they change it;
2. the subject's language — an Arabic-speaking student in an English science lesson is
   answered in English;
3. the language the student wrote in;
4. the request's `language` field, then the configured default.

`language.reply_follows: message` swaps 2 and 3 for a deployment that answers in the
student's language.

**4. The tasks run, and the reply is composed.** Every sentence a student reads is written
by a model — explanations, refusals, small talk, progress notes. The service writes no
fixed sentences in any language; the only fixed marks are symbols (✓ ✗ → 💡 ∅) and
labels. A model call that fails fails the turn with a `503`, and the client shows its own
message.

**5. The turn is remembered**, along with the settled subject and any standing language
request; the summary is refolded once the window fills.

### Pathways

A pathway changes the task allowlist and the voice. Parents are answered but not
remembered: `ai.user_memory` holds student and teacher rows only.

| Pathway | Allowed tasks | Voice |
|---|---|---|
| `student` | explain, summarize, assessment, grade, visual, report | friendly |
| `teacher` | explain, summarize, assessment, grade, visual | professional |
| `parent`  | explain, summarize, visual | friendly |

### Grading and mastery

Grading is AI-final. An answer is graded against the answer key or rubric when there is
one, against retrieved syllabus passages otherwise, and declined when there is neither —
never on the model's own knowledge. Each graded answer is appended to
`ai.mastery_events`; a topic's answers in one assessment are averaged into one
difficulty-weighted step of its `ai.mastery` score. A teacher correction is a new event,
never an edit.

---

## Retrieval and grounding

One multilingual index serves every language: a question in Arabic finds a passage in
English and the reverse. For an explain or summarise task:

1. **Search** — BGE-M3 dense and sparse vectors over `eduai_chunks_v2`, filtered to the
   namespace, subject and (when selected) lesson, fused with a BM25 rank. Contents and
   index pages are excluded, and lessons not marked `ready` are never searched.
2. **Similarity floor** — passages below `agent.min_score` (0.47, set by a sweep recorded
   in `config.yaml`) are dropped, so an off-syllabus question grounds in nothing and is
   declined.
3. **Lesson scope** — inside one selected lesson, a question another lesson answers
   better (scoped best / unscoped best < 0.90) is refused by naming the lesson.
4. **Read the whole section** — the three closest matches by meaning are read with the two
   chunks either side of them, in lesson order, because a section split at its
   sub-headings is otherwise found one fragment at a time.
5. **Continue the last answer** — a turn that carries the previous one on starts from the
   passages that answer cited.
6. **Answer** — the model answers only from the numbered passages, citing them inline. It
   leads with the lesson's own examples and says when an example is its own.
7. **Grounding check** — an answer that cites nothing is treated as a refusal: no sources,
   no prerequisites, not remembered as evidence.
8. **Prerequisites** — walked through the topic graph from the passages the answer cited.

Citations, sources and prerequisites are returned as structured fields (`citations`,
`prereq_topics`) and are not printed in the reply text.

---

## Ingestion

One PDF (or a lesson's body text) is one lesson:

```
extract       each page as typed blocks in reading order. The PDF's own text layer is
              used wherever it has usable text; a page with none (a scan, or glyph noise
              from subset fonts) is read by a vision model (Qwen3-VL, Claude fallback),
              and what comes back is checked before it is believed. Figures are cropped
              to S3 with alt text. A lesson with more than 5% unreadable pages is held
              back as needs_review and never searched.
chunk         blocks grouped under their heading; running heads and folios dropped,
              contents and index pages marked so search never returns them.
embed         BGE-M3 dense + sparse, each chunk prefixed with its subject, lesson,
              section and page.
topics        3–8 skills named in the lesson's language; a re-ingest reuses existing
              topic ids so mastery history is never orphaned.
dependencies  prerequisite edges between topics, cleaned (no self-edges, capped
              out-degree, cycles broken) and rewritten on re-ingest.
index         ai.chunks, ai.figures, the ai.vector_index manifest, Qdrant points and an
              ai.lecture_ingestion status row. Re-ingesting replaces a lesson in place.
```

The platform triggers ingestion when a lesson is published (`POST /ai/ingest/lesson`),
and it can be run from the command line (see [Running](#running)).

---

## Where data lives

| Store | What | Written by |
|---|---|---|
| `ai.chunks`, `ai.figures`, `ai.vector_index`, `ai.topics`, `ai.topic_dependencies`, `ai.lecture_ingestion`, `ai.page_transcriptions` | the indexed syllabus | ingestion |
| `ai.user_memory` | conversation turns, running summary, session state (subject, language) | agent |
| `ai.mastery`, `ai.mastery_events`, `ai.mistake_tags` | per-skill mastery and its history | agent (grading) |
| `ai.artifacts` | rendered visuals | agent |
| Qdrant `eduai_chunks_v2` / `eduai_topics` | chunk vectors (dense + sparse) and topic-label vectors | ingestion |
| S3 | figure crops and rendered artifacts | both |

Everything about users, classes, lessons and chat transcripts belongs to the platform
backend, which the service reads through `/api/ai-integration`.

---

## Repository layout

```
.
├── agent/                   the runtime graph
│   ├── conversation/        turn memory: records, store (ai.user_memory), history window
│   ├── graph/
│   │   ├── graph.py         compiles the graph; one adapter node per subgraph
│   │   ├── nodes.py         classify · load_context · understand · converse · plan ·
│   │   │                    dispatch · compose · verify_language · persist
│   │   ├── pathways.py      namespaces and voice per pathway
│   │   ├── grade.py         the /ai/grade entrypoint
│   │   ├── report.py        the /ai/reports entrypoint
│   │   └── subgraphs/       explain_summarize · assessment · grade · report · visual
│   ├── tools.py             search, topic resolution, neighbours, rubric, mastery reads/writes
│   ├── retrieval.py         retrieval composition: floor, scope, neighbours, prerequisites
│   ├── citations.py         reading and renumbering [n] markers
│   ├── prompts.py           every prompt, each declaring its reader and language rule
│   ├── models.py            typed contracts (turn reading, tasks, items, citations)
│   ├── subjects.py          subject facts from the backend, cached
│   ├── scope.py             the lesson a turn is bounded by
│   ├── mastery.py           the mastery update rule
│   ├── render.py            deterministic renderers
│   ├── homework.py          reading questions off a photographed page
│   └── errors.py            failures a turn reports
├── ingestion/               the offline pipeline (python -m ingestion)
│   ├── layout/              page → typed blocks → chunks (tier router, text layer, vision)
│   ├── stages/              extract · chunk · embed · topics · dependencies · index
│   ├── prompts/
│   └── stores.py            Postgres and Qdrant writers (dry-run fallback)
├── shared/                  imported by both halves
│   ├── language/            the reply-language policy, detection, directives, verification
│   ├── llm.py · structured.py · prompting.py   model access, schema-checked output, prompts
│   ├── embeddings.py · lexical.py · matching.py · vectors.py · vector_store.py
│   ├── postgres.py · storage.py · backend.py · vision.py
│   └── config.py · logging.py · metering.py · errors.py
├── system/                  FastAPI service: routers, schemas, runtime, ingestion jobs
├── scripts/                 bootstrap, ingest verification, re-embedding, evaluations
├── tests/unit/
├── config.yaml              every behavioural setting
└── .env.example             secrets and connection strings
```

Layering: `shared/` depends on nothing above it; `agent/` and `ingestion/` never import
each other; `system/` sits on top. Read [`CLAUDE.md`](CLAUDE.md) before changing code — it
lists the rules this codebase holds itself to and the failure behind each one.

---

## Getting started

Requirements: Python 3.11+, PostgreSQL 15+, Qdrant 1.12+, and an OpenRouter API key. The
embedding model (BGE-M3, about 2 GB) runs locally and downloads on first use.

```bash
git clone git@github.com:edAi20261/AI-Repo.git
cd AI-Repo
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt  # or requirements.txt for runtime only
cp .env.example .env                 # then fill it in
```

For a local database and Qdrant with nothing in them yet, `python scripts/bootstrap.py`
creates the `ai.*` tables ingestion writes and the Qdrant collection. It is safe to re-run.

---

## Configuration

| File | Holds | Committed |
|---|---|---|
| `.env` | secrets and connection strings | no |
| `config.yaml` | everything that changes behaviour | yes |

The model is a config change: `llm.provider` and `llm.model` move the whole system, and
`llm.overrides` sets a model per task. Today every task runs on Gemini 2.5 through
OpenRouter (Flash for reading turns and writing answers, Flash-Lite for composition,
conversation summaries and visual builders), vision on Qwen3-VL, and the evaluation
judge on GPT-4.1.

`APP_ENV` selects an overlay from `environments:` in `config.yaml`:

| `APP_ENV` | Index it searches | Notes |
|---|---|---|
| `local` | `eduai_chunks_dev` | debug logging; keeps a dev index apart from real data |
| `staging` | `eduai_chunks_v2` | the base configuration |
| `production` | `eduai_chunks_v2` | no `/docs`, warning-level logs, more model retries |

The settings most often tuned: `agent.min_score`, `agent.k`, `agent.expand`,
`agent.scope.relevance_ratio`, `agent.conversation` (window, folding, TTL),
`language.reply_follows`, `assessment.*` and `mastery.*`. Each threshold carries the
measurement that set it.

---

## Running

**The service:**

```bash
uvicorn system.asgi:app --host 0.0.0.0 --port 8000
curl localhost:8000/ready        # {"status":"ready", "graph_ready":true, ...}
```

Set `LOG_FILE=logs/ai-service.log` to also write the log to a size-bounded JSON file.
Each turn logs one `turn` line with its kind, language and why, subject, tasks and
citations. Outside a dev environment the service refuses to start without a model key,
the service token or its stores.

**Ingestion from the command line:**

```bash
# a local file
python -m ingestion path/to/lesson.pdf --lecture-id <uuid> --subject-id <uuid> \
    --lesson-title "The Water Cycle" --lesson-order 4 [--grade g8] [--subject-is-math]

# a lesson resolved through the backend (or a fixtures manifest)
python -m ingestion --resolve <lesson_id> [--fixtures manifest.yaml]
python -m ingestion --backfill [--fixtures manifest.yaml]

python scripts/verify_ingest.py [--lecture-id <uuid>]   # check the stored invariants
```

The direct mode reads the subject's name and language from the backend; without
`--lesson-title` citations from the lesson are labelled by section headings only. With
neither `DATABASE_URL` nor `QDRANT_URL` set, ingestion writes to in-memory dry-run stores.

**The graph, in process:**

```python
from agent.config import get_config
from agent.graph import build_graph

graph = build_graph(get_config())
turn = graph.invoke({
    "user_id": "<uuid>", "user_type": "student", "session_id": "<uuid>",
    "message": "Can you explain evaporation?",
})
print(turn["response"])            # the reply
print(turn["published_citations"]) # what it cited
```

---

## HTTP API

Every route except `/health` and `/ready` requires the `x-service-token` header
(`AI_WEBHOOK_SHARED_SECRET`). The platform backend authenticates users; this service is
not public.

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/ai/chat` | One message → one reply, with any assessment, grade, visual, citations and prerequisites |
| `POST` | `/ai/chat/stream` | The same turn as server-sent events: one `progress` per stage, then `result` |
| `POST` | `/ai/grade` | Grade submitted answers and move mastery |
| `GET` | `/ai/reports/{user_id}` | A progress report with its trend chart |
| `POST` | `/ai/homework/capture` | Read the questions off a photographed homework page |
| `POST` | `/ai/ingest` | Queue a lesson from a source URI; returns `202` and a job id |
| `POST` | `/ai/ingest/lesson` | Queue a lesson by platform id (fired on publish) |
| `GET` | `/ai/ingest`, `/ai/ingest/{job_id}`, `/ai/ingest/lectures/{lecture_id}` | Recent jobs, one job, a lecture's latest job |
| `GET` | `/health`, `/ready` | Liveness and readiness |

Every failure returns one envelope, and every response carries `x-request-id` (an upstream
id is honoured):

```json
{"error": {"code": "service_unavailable", "message": "…", "request_id": "…"}}
```

The backend contract is in [`docs/BACKEND_INTEGRATION.md`](docs/BACKEND_INTEGRATION.md) and
[`docs/openapi.json`](docs/openapi.json).

---

## Evaluation and tests

A behaviour change is measured before and after. The evaluations run against the real
index and models:

| Script | Measures |
|---|---|
| `scripts/eval_retrieval.py` | whether retrieval finds the right lesson, and separates answerable from unanswerable questions |
| `scripts/eval_rag.py` | context precision, context recall, faithfulness and answer relevance |
| `scripts/eval_answers.py` | answer quality, judged by a model from another family |
| `scripts/eval_conversations.py` | multi-turn behaviour: follow-ups, small talk, refusals, language — against a running service |

```bash
pytest                 # the unit suite: no keys, no services
ruff check .           # must not regress
```

---

## Known limitations

- **Chunking is per heading and per page.** A section split at its sub-headings is
  recovered at answer time by reading neighbouring chunks, but chunks never cross a page,
  do not carry their parent heading, and on magazine-style pages can still hold
  fill-in answer keys and stray letters from decorative titles. Fixing it in the chunker
  needs a re-ingest.
- **Topics are coarse on long documents.** A lesson gets 3–8 topics, so a whole book
  ingested as one lesson maps most of its pages to a handful of skills, and prerequisite
  edges are derived by a model without review.
- **The platform does not yet pass structured citations through** to the frontend, and the
  frontend sends the study mode inside the message rather than as a field.
- **Parents have no conversation memory**, and teacher and parent report scopes, classroom
  scoping and session uploads are not built yet.
