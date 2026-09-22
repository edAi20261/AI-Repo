# EduAI — AI Workflows

Multilingual (Arabic, English, French, German) educational AI system built on LangGraph.
It grounds every answer in the uploaded syllabus, generates assessments and grades them,
tracks what each student has actually mastered at the level of individual skills, produces
study visuals and progress reports, and gives teachers drafted grades they review before
anything reaches a student.

The system has three independently deployable parts:

- **`ingestion/`** — an offline pipeline that runs once per uploaded lecture. All the
  expensive LLM work (topic extraction, prerequisite derivation, embedding) happens here.
- **`agent/`** — the runtime graph. It only reads: a vector search and table lookups.
  This is why a chat turn stays fast and predictable.
- **`system/`** — the HTTP service the platform backend calls. Transport only: it
  validates, authenticates, and delegates to the two above.

---

## Table of contents

- [Architecture](#architecture)
- [Repository layout](#repository-layout)
- [Getting started](#getting-started)
- [Configuration](#configuration)
- [Running the system](#running-the-system)
- [Testing](#testing)

---

## Architecture

### Runtime graph

```
START
  → classify            the role, and the conversation so far (turns, summary, state)
  → load_context        retrieval namespaces and the pathway's voice
  → understand          LLM, structured output. Reads the new message against the
                        conversation: does it need the syllabus, which tasks, a
                        self-contained query for each, is it carrying the last answer
                        on, did the student ask to be answered in a language
      ├── converse      social turns: a model reply, no retrieval, no canned text
      └── plan          subject (request → conversation → found by search), topics,
                        reply language, the last answer's passages when continuing
            → dispatch  conditional fan-out
                ├── explain_sg / summarize_sg   retrieve → answer → prerequisites
                ├── assessment_sg               batch item generation over the chunk index
                ├── grade_sg                    the only writer to mastery
                ├── report_sg                   walks prerequisites to find root causes
                └── visual_sg                   deterministic renderers
            → render_visuals → compose
  → verify_language     the answer is in the language decided, or is repaired once
  → persist             remembers the turn; folds old turns into a running summary
END
```

**A conversation is remembered.** Each turn is stored with the passages its answer
cited, and the next turn is read with the recent turns in view and the older ones
summarised. There are no keyword lists, seed phrases or canned replies deciding
what a message is: one model reads it in context.

**A subject is taught in its own language.** The reply language is, in order: a
language the student asked for in this conversation (it holds until they change
it), the subject's language, the language they wrote in. `language.reply_follows`
swaps the last two for a deployment that answers in the student's language.

Session uploads, export (PDF / DOCX), and the open-ended Q&A loop land in later sprints.

Each subgraph is a **separately compiled `StateGraph` with its own private state**. It does
not inherit the parent state and has no idea pathways exist — the parent wraps each one in
an adapter node.

### Pathways

A pathway changes only three things: the retrieval namespace, the task allowlist, and the
voice.

| Pathway | Allowed tasks | Voice |
|---|---|---|
| `student` | explain, summarise, assessment, grade, visual, report | encouraging tutor |
| `teacher` | explain, summarise, assessment, grade, visual | professional |
| `parent`  | explain, summarise, visual | plain, supportive |

Persona and classroom scopes (and their additional namespaces) land in a later sprint; a
student who has joined no classroom still gets the full tutor, which is the normal starting
state, not a degraded one.

### Offline ingestion

```
lecture uploaded
  → extract & normalise   PyMuPDF text + NFKC. Scanned and maths pages go to vision
                          (Qwen3-VL). Born-digital pages whose text is glyph-noise
                          (subset fonts with no ToUnicode) are detected and sent to
                          vision for the text too — their embedded figures are still
                          read directly. Figures come from the embedded rasters where
                          present (deduped) and from vision-detected boxes on scans;
                          each is cropped to S3 with vision alt-text. A page the model
                          cannot read is flagged needs_review and never indexed.
  → chunk                 structure-aware, ~500–800 tokens, keeps headings + pages
  → embed                 BGE-M3, one multilingual index (AR + EN + FR)
  → topics                extract 3–8 skills; on re-ingest, match each against the
                          lecture's own existing topics and reuse its topic_id (so a
                          student's mastery history never fragments), tag chunks by topic
  → dependencies          derive topic → prerequisite edges from the subject catalog,
                          then clean them (drop self-edges, cap out-degree, break cycles);
                          the lecture's edges are rewritten, not accreted, on re-ingest
  → index                 Qdrant points + ai.chunks + ai.vector_index manifest +
                          a per-lecture ai.lecture_ingestion status row, namespace as
                          filter metadata. Re-ingesting a lecture replaces its rows,
                          points and figure objects in place, never duplicating them.
lecture is live

  pending: review gate (derived edges currently ship unreviewed)
```

Order is not optional: **lectures → assessments → teaching card.**

---

## Repository layout

```
.
├── ingestion/              offline pipeline, own entrypoint (python -m ingestion)
│   ├── config.py           config.yaml + .env loader, per-task LLM overrides
│   ├── state.py            IngestionState + domain dataclasses (map onto ai.*)
│   ├── schemas.py          Pydantic contracts for the LLM / vision JSON outputs
│   ├── llm_provider.py     vision client factory (re-exports the shared LLM/embedder seams)
│   ├── extractors.py       topic extraction + dependency derivation over the LLM seam
│   ├── backend.py          lesson-id → source URI + subject resolver (HTTP or fixtures)
│   ├── stores.py           Postgres + Qdrant writers behind interfaces (dry-run fallback)
│   ├── context.py          per-run dependencies handed to every stage
│   ├── graph.py            LangGraph wiring (sequential fallback if langgraph absent)
│   ├── stages/             extract · chunk · embed · topics · dependencies · index
│   ├── prompts/
│   └── utils/              normalise · pdf · tokens · s3
├── shared/                 cross-cutting, imported by both halves
│   ├── config.py           config.yaml + .env loader, per-task LLM overrides
│   ├── embeddings.py       the bilingual index embedder (BGE-M3), cached per process
│   ├── llm.py              provider-agnostic chat-model factory
│   ├── logging.py          structured logging setup (secret / PII-safe)
│   └── metering.py         per-run LLM cost + token metering
├── agent/                  runtime graph, read-only
│   ├── config.py           re-exports the shared Config (agent.* settings)
│   ├── state.py            MainState (TypedDict)
│   ├── models.py           typed contracts (assessment items, graded items, citations…)
│   ├── prompts.py          plan · explain · summarize · assessment · grade · report · visuals · compose
│   ├── tools.py            rag_search · resolve_topics · fetch_rubric · mastery reads/writes · figures
│   ├── retrieval.py        shared RAG (namespace-filtered search + prerequisite walk)
│   ├── render.py           deterministic visual renderers (charts, mind maps, PDFs, geometry, slides)
│   ├── mastery.py          difficulty-weighted per-skill mastery rollup
│   ├── graph/
│   │   ├── nodes.py        classify, load_context, plan, dispatch, compose, persist
│   │   ├── pathways.py     build_namespaces + resolve_style (the only pathway-aware fns)
│   │   ├── grade.py        the /ai/grade entrypoint over grade_sg
│   │   ├── report.py       the /ai/reports entrypoint over report_sg
│   │   ├── graph.py        compiles the MainState graph
│   │   └── subgraphs/      explain_summarize · assessment · grade · report · visual
│   └── utils.py            language detection · JSON extraction · structured-output helper
├── system/                 HTTP service (FastAPI) over the graph + pipeline
│   ├── main.py             app factory; compiles the graph once at startup
│   ├── asgi.py             uvicorn entrypoint (system.asgi:app)
│   ├── settings.py         api.* config + service secrets
│   ├── security.py         service-token auth + startup preflight
│   ├── runtime.py          agent facade; offloads the sync graph to worker threads
│   ├── ingestion_jobs.py   background pipeline runs, polled by job id
│   ├── notifier.py         completion callback to the platform
│   ├── errors.py           error taxonomy -> one JSON envelope
│   ├── middleware.py       request id + access logging
│   ├── routers/            chat · grading · reports · ingestion · health
│   └── schemas/            request/response contracts
├── config.yaml             behavioural configuration, committed
├── scripts/                dev/test bootstrap + maintenance
│   ├── dev_ai_schema.sql   creates the ai.* tables for a local Postgres
│   ├── bootstrap.py        applies the SQL + creates the Qdrant collection
│   ├── verify_ingest.py    checks the live-store invariants per lecture
│   ├── reembed_chunks.py   re-embeds the indexed chunks in place
│   └── eval_*.py           retrieval, RAG, answer and conversation evaluations
├── tests/
└── data/                   syllabus material — not committed
```

---

## Getting started

### Prerequisites

- Python 3.11+
- PostgreSQL 15+
- Qdrant 1.12+
- Redis 7+
- An OpenRouter API key — vision transcription (scanned/maths pages, figure
  captions) and topic extraction. Both fall back to stubs when it is absent, so
  the pipeline runs without it, but produces no real transcription or topics.
- Tesseract OCR — optional; only if `ingestion.ocr.engine` is switched from the
  default `vision` to `tesseract` for cheap prose-only scanned pages.

### Setup

```bash
git clone https://github.com/edAi20261/AI-Workflows.git
cd AI-Workflows
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

```bash
source .venv/Scripts/activate
```

On macOS or Linux use `source .venv/bin/activate`.

Install dependencies:

```bash
pip install -r requirements.txt
```

For development work, install the dev extras instead:

```bash
pip install -r requirements-dev.txt
```

Create your environment file and fill it in:

```bash
cp .env.example .env
```

---

## Configuration

Configuration is split in two, deliberately:

| File | Contents | Committed |
|---|---|---|
| `.env` | secrets and connection strings only | no |
| `config.yaml` | everything that changes behaviour | yes |

If changing a value changes how the system behaves, it belongs in `config.yaml` — where
it is reviewable and its history lives in git. If it would leak something, it belongs in
`.env`. Nothing behavioural is hard-coded.

The system is **model-agnostic by design**. `shared/llm.py` builds the client from
`llm.provider` and `llm.model`, so moving the whole system to a different model means
editing two lines of `config.yaml` and nothing else. Per-task overrides sit under
`llm.overrides`, so deterministic work such as grading or dependency derivation can run
at temperature 0 without touching the global default.

Embeddings are configured separately from the chat model, since the bilingual index has
different requirements. `environments:` at the bottom of `config.yaml` provides overlays
merged over the base config, selected by `APP_ENV`.

Behavioural thresholds — retrieval depth, chunk sizes, the topic dedupe cutoff, the
weak-score threshold, the question source strategy, the loop ceiling — all live there.

---

## Running the system

### Local stores (first-time setup for a real run)

In production the backend owns the database schema and the pipeline auto-creates
the Qdrant collection. To run real writes locally, provision both once:

```bash
python scripts/bootstrap.py
```

This applies `scripts/dev_ai_schema.sql` to `DATABASE_URL` (creating the six
`ai.*` tables ingestion writes — chunks, figures, vector_index, topics,
topic_dependencies, and the per-lecture lecture_ingestion tracking row) and
creates the Qdrant collection sized to the embedder. It skips whichever store isn't configured, and is safe to re-run. With
neither `DATABASE_URL` nor `QDRANT_URL` set, ingestion uses in-memory dry-run
stores and no bootstrap is needed.

### Ingestion

Ingestion runs as a standalone entrypoint over one already-split lesson file
(one PDF = one lesson):

```bash
python -m ingestion path/to/lesson.pdf --lecture-id <uuid> --subject-id <uuid> --grade g8
```

Add `--subject-is-math` to force every page down the vision route (maths notation
flattens under plain text extraction). Two further modes resolve a lesson through
the backend — or a local fixtures manifest — before running:

```bash
python -m ingestion --resolve <lesson_id> [--fixtures manifest.yaml]
python -m ingestion --backfill [--fixtures manifest.yaml]
```

It runs `extract → chunk → embed → topics → dependencies → index` and prints the
per-stage stats.

**Re-ingesting is safe and needs no wipe.** Running a lecture again replaces its
chunks, points, figures and dependency edges in place (delete-before-write) and
**reuses** its `topic_id`s — so nothing duplicates or orphans. Wipe + `bootstrap.py`
are only for a deliberate clean-slate rebuild, not for a normal re-run.

Behaviour is driven entirely by `config.yaml` + `.env`:

- With `DATABASE_URL` and `QDRANT_URL` set, it writes `ai.chunks` /
  `ai.figures` / `ai.vector_index` and upserts Qdrant points for real.
- With neither set, it runs against in-memory dry-run stores — a full pipeline
  pass needs no external services.
- The vision model (scanned/maths transcription, figure captions) and the
  topic-extraction model are read from config; each falls back to a safe stub
  when its key is absent, so the pipeline always completes.

The embedding model (BGE-M3) runs locally and downloads on first use (~2 GB).
Set `QDRANT_URL=:memory:` for a real vector-store pass without a Qdrant server.

After a run, `python scripts/verify_ingest.py [--lecture-id <uuid>]` reports the
per-lecture invariants against the live stores: chunk / manifest / point parity,
a topic on every chunk, namespace and filter fields on every point, NFKC-clean
text, and no re-ingest duplicates.

### Runtime graph

The read-time graph turns a question into a grounded, cited answer. It compiles
`classify → load_context → plan → dispatch → {explain_sg | summarize_sg} → compose
→ persist`; the planner resolves the message to `topic_id`s, retrieval pulls the
relevant chunks plus prerequisite-topic chunks, and the answer is written in the
question's language with inline citations and a prerequisite-topics list.

```python
from agent.graph import build_graph
from agent.config import get_config

graph = build_graph(get_config())
result = graph.invoke({
    "message": "Explain the fire triangle.",
    "user_type": "student",
    "subject_id": "<subject-uuid>",   # scopes retrieval to the subject
})
print(result["response"])       # grounded answer + Sources + Prerequisite topics
```

**How a question is answered:**

1. `classify` — pick the pathway and load the conversation. No LLM.
2. `load_context` — resolve the retrieval namespaces and style for the pathway.
3. `understand` — an LLM reads the message against the conversation and returns the tasks
   (explain, summarise, assessment, grade, visual, report), a self-contained query for each,
   whether the turn carries the last answer on, and any language request. A purely social
   turn goes to `converse` instead.
4. `plan` — settle the subject: the request's, else the conversation's, else the subject whose
   material answers the query best (found by searching it, confined to a student's enrolled
   subjects). `resolve_topics` maps each query to `topic_id`s **within that subject**, and the
   reply language is decided.
5. `dispatch` — fan each task out to its subgraph.
6. the task subgraphs run in parallel — `explain_sg` / `summarize_sg` answer from retrieved
   context with inline citations; `assessment_sg` generates items from the chunk index;
   `grade_sg` grades against a rubric or the syllabus and updates mastery; `report_sg` builds
   a progress note; `visual_sg` renders any requested visuals.
7. `render_visuals` — render every collected visual spec (deduped, capped per turn).
8. `compose` — merge the tasks, apply the voice, append the prerequisites of the cited passages.
9. `verify_language` — check the reply's language, repair it once if it drifted.
10. `persist` — remember the turn and its cited passages; fold old turns into the summary.

The **subject scopes** the search — supplied on the request (optional) or **inferred from
the question** when absent; the **lecture is discovered by similarity**, not chosen up
front. The answer comes out in the user's language unless they explicitly requested another.

Retrieval is one multilingual index: a question in one language retrieves the
relevant passage regardless of the language it was written in, and answers in the
question's language (`ar` / `en` / `fr` / `de`). The building blocks are read-only,
deterministic tools — `rag_search` (namespace-filtered vector search), `resolve_topics`,
`get_topic_deps`, `get_topic_chunks` — every query namespace-scoped, non-`ready`
lectures excluded.

Beyond retrieval, the runtime also **generates assessments**, **grades** answers (rubric
first, syllabus material otherwise, declining when neither supports a grade), tracks
**per-skill mastery** on every graded turn, renders **study visuals**, and produces
**progress reports** with a trend and root-cause prerequisite. Grading is the only writer to
mastery, and only a published grade moves it. Persona and classroom scopes, session uploads,
export, and the open-ended Q&A loop land in later sprints; the HTTP service is in
`system/` (below).


### HTTP service

The service is the transport layer over the runtime graph and the ingestion
pipeline. It holds no teaching logic: it validates a request, authenticates the
caller, and delegates.

```bash
uvicorn system.asgi:app --host 0.0.0.0 --port 8000
```

The graph is compiled once at startup, so the first request is not slower than
the rest. Startup refuses to continue outside a dev environment when the service
token or the stores are unconfigured, rather than serving a broken service.

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/ai/chat` | A message becomes a grounded, cited reply — plus any assessment, grade or visual the planner produced. |
| `POST` | `/ai/chat/stream` | The same turn as SSE: a `progress` event per graph stage, then one `result`. |
| `POST` | `/ai/grade` | Grade submitted answers and move mastery. |
| `GET` | `/ai/reports/{user_id}` | A progress report with its trend chart rendered. |
| `POST` | `/ai/ingest` | Queue a lesson. Returns `202` with a job id; ingestion never blocks a request. |
| `GET` | `/ai/ingest/{job_id}` | Poll one job. `/ai/ingest/lectures/{lecture_id}` reads the latest for a lecture. |
| `GET` | `/health`, `/ready` | Liveness and readiness. Unauthenticated — they expose no data. |

The integration contract for the platform backend is written up in
[`docs/BACKEND_INTEGRATION.md`](docs/BACKEND_INTEGRATION.md), with the generated spec
alongside it in [`docs/openapi.json`](docs/openapi.json).

**This service is not public.** The platform backend authenticates the end user
and calls it with the shared `x-service-token` secret
(`AI_WEBHOOK_SHARED_SECRET`), the same one the platform accepts in the other
direction. Login and per-user authorisation stay in the platform.

```
frontend  ->  platform backend  ->  ai service (system/)  ->  agent / ingestion
```

Every failure returns one envelope, and every response carries `x-request-id`
(an upstream id is honoured, so a request is traceable across both services):

```json
{"error": {"code": "validation_error", "message": "…", "request_id": "…"}}
```

Behaviour is configured under `api:` in `config.yaml` — concurrency, ingestion
workers, whether `/docs` is served (off in production), and the completion
callback. Interactive API docs are at `/docs`.

---

## Testing

```bash
pip install -r requirements-dev.txt
pytest
```

The ingestion unit suite (`tests/unit/test_ingestion.py`) covers the pieces that
must not silently regress — Arabic normalisation, structure-aware chunking,
config overlay + per-task LLM overrides, topic extraction and dedupe, dependency
derivation and edge post-process, the Pydantic output contracts, the vision JSON
parsing and client factory, the backend resolver, the born-digital legibility
gate, figure box scaling and duplicate dedupe, re-ingest idempotency and the
per-lecture tracking record — plus an end-to-end fixture-to-pipeline pass into
dry-run stores. No keys or services required.

The agent suite covers the runtime graph without any keys or services — language
detection, the pathway functions, the planner (with a stub model and a deterministic
fallback), the compose helpers (grade summary, visual captions, decline suppression,
render-error handling), and an end-to-end graph run with the tools monkeypatched
(`test_agent.py`); assessment generation and shape (`test_assessment.py`); grading
authority order, decline, and mastery movement (`test_grade.py`); the mastery rollup
(`test_mastery.py`); the progress report and root-cause walk (`test_report.py`); and the
visual renderers and subgraph, including deterministic geometry (`test_visual.py`).
`test_logging.py` and `test_metering.py` cover the logging-safety guards (secret masking,
URI redaction, third-party clamp) and the cost/token meter.

`test_api.py` covers the HTTP layer against a stubbed runtime — service-token auth, request
validation and the rejection of unknown fields, the response shapes, error-envelope mapping,
request-id propagation, and the startup preflight that refuses an unauthenticated or store-less
production service. No keys, no services, no graph compile.

`rag_search` is additionally exercised end to end against the live stores with
same-language and cross-lingual queries and a namespace-isolation check, confirming
a question in one language retrieves the relevant passage in another.
