# EduAI — AI Layer

The AI service of **EduAI**, a learning platform for schools. It turns each school's own
syllabus into a tutor that students can talk to: it explains lessons, answers follow-up
questions, sets and marks practice, tracks what every student has mastered, and reports
progress — in Arabic, English, French and German, typed or spoken. Teachers get class
analytics and a tutor that can teach in their own style; the platform gets whole books
read, split into lessons and indexed.

---

## Contents

- [What EduAI is](#what-eduai-is)
- [What this repository does](#what-this-repository-does)
- [Quickstart](#quickstart)
- [How a turn is answered](#how-a-turn-is-answered)
- [Retrieval and grounding](#retrieval-and-grounding)
- [Student and class analytics](#student-and-class-analytics)
- [Credits](#credits)
- [Ingestion](#ingestion)
- [Where data lives](#where-data-lives)
- [Repository layout](#repository-layout)
- [Configuration](#configuration)
- [Running](#running)
- [HTTP API](#http-api)
- [Evaluation and tests](#evaluation-and-tests)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [Known limitations](#known-limitations)

---

## What EduAI is

EduAI is a multi-school learning platform for **students, teachers and parents**. A school
offers one or more curricula (a national curriculum, IGCSE, SAT…) across grades 1 to 12.
Each grade has its subjects — science, Arabic, English, mathematics, French and so on —
either official or created by a teacher, and each subject has lessons. When a lesson is
published, its material becomes something students can learn from and ask about.

Students study their lessons, do homework and quizzes, and follow a study plan. Teachers
publish lessons, set work and follow their classes. Parents follow their children.

What we are building is a tutor that behaves like a good teacher who has read exactly the
books the student is studying:

- **It answers from the syllabus, not from the internet.** Every answer is grounded in the
  student's own lessons and cites them; a question the lessons do not cover is declined
  honestly instead of answered from general knowledge.
- **It holds a conversation.** "Explain that more simply", "give me an example" and "what
  about tornadoes?" are understood in the light of what was said before.
- **Each subject is taught in its own language.** An English science lesson is answered
  in English and an Arabic lesson in Arabic, unless the student asks otherwise — and then
  that choice is kept.
- **It knows where each student stands.** Every graded answer moves a per-skill mastery
  score, which drives which topics are practised next and what the progress report says.
- **It teaches the way the student needs.** "I didn't get that", "simpler" or "from the
  very beginning, step by step" get a different explanation, not the same one again — and
  a basic step the book assumes is taught, and marked as going beyond the lesson.

---

## What this repository does

EduAI is three codebases. This one is the AI layer; the other two call it.

```
 student / teacher / parent
            │
            ▼
   frontend  (React)          the web app people use
            │
            ▼
   platform backend (NestJS)  accounts, schools, classes, lessons, homework, quizzes,
            │                 chat transcripts — and the only caller of this service
            ▼
   AI service  (this repo)    tutoring, retrieval, grading, mastery, reports, visuals,
                              lesson ingestion
```

The AI layer provides:

| For | What it does |
|---|---|
| Students | a tutor chat grounded in their lessons, typed or by voice, with replies read aloud; follow-ups re-taught another way; practice quizzes aimed at their weak topics; marking with feedback and the correct answer; a homepage summary of strengths, weak spots, what to practise next and what is being forgotten; progress notes with the root cause of a weakness; mind maps, charts, maths and geometry figures, flashcards, PDFs, slides and illustrations; reading homework questions off a photo |
| Teachers | the same tutor in a professional voice; a persona that lets the tutor teach their students in their own style; quizzes, homework and exams generated from a lesson; marking submitted answers; class analytics (topic standings, students at risk and why, common mistakes, most-missed questions, groups to reteach) and a written class summary; topic suggestions for questions they write themselves |
| Parents | explanations, summaries and visuals of what their child is studying; a progress note written to them |
| The platform | ingesting every published lesson, or a whole book split into its lessons — reading scanned pages, chunking, embedding, extracting skills and their prerequisites — so it can be searched; a credit wallet per user that meters every model call |

It deliberately does **not** handle login, users, schools, classes or chat history: those
belong to the backend, which authenticates every user and calls this service with a
shared secret. The service is never exposed to the public internet.

Inside, it is three deployable parts:

- **`ingestion/`** — an offline pipeline that runs once per published lesson; all the
  expensive reading, embedding and extraction happens here.
- **`agent/`** — the LangGraph graph a chat turn runs through. It searches the index and
  reads the catalogue, and writes only its own records: conversation memory, mastery,
  mistake tags, rendered visuals, reports, personas and credits.
- **`system/`** — the FastAPI service the backend calls. Transport only: it validates,
  authenticates and delegates.

---

## Quickstart

### 1. Install

Requirements: **Python 3.11+**, access to a **PostgreSQL** database and a **Qdrant**
instance, and an **OpenRouter API key**. The embedding model (BGE-M3, about 2 GB) runs
locally and downloads the first time it is needed.

```bash
git clone git@github.com:edAi20261/AI-Repo.git
cd AI-Repo
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
cp .env.example .env
```

### 2. Fill in `.env`

`.env` holds secrets and connection strings only; every behavioural setting is in
`config.yaml`.

| Variable | Needed for | Notes |
|---|---|---|
| `OPENROUTER_API_KEY` | every model call | one key reaches every model the service uses |
| `DATABASE_URL` | retrieval, memory, mastery | the platform's Postgres (the `ai.*` schema) |
| `QDRANT_URL`, `QDRANT_API_KEY` | retrieval | where the lesson vectors live |
| `AI_WEBHOOK_SHARED_SECRET` | every request | the `x-service-token` the backend sends |
| `BACKEND_API_BASE_URL` | subject facts, enrolments, answer keys, ingestion | the backend's `…/api/ai-integration` prefix |
| `APP_ENV` | choosing the config overlay | `local`, `staging` or `production` — see below |
| `S3_BUCKET`, `S3_REGION`, AWS keys | figures and rendered visuals | without them files go to a local folder |
| `ELEVENLABS_API_KEY` | voice | speech-to-text and reading replies aloud |
| `OPENAI_API_KEY` | illustrations | without it the `image` visual falls back to a stub, with a warning |
| `GOOGLE_APPLICATION_CREDENTIALS` | ingestion | a service-account file for Vertex, which reads the pages (`.env.ingest.example`) |

**Which `APP_ENV`?** `local` searches a separate, usually empty, development index
(`eduai_chunks_dev`), so every question will be declined until you ingest something into
it. To try the tutor against lessons that are already ingested, use `staging`, which
searches `eduai_chunks_v2`. If your `DATABASE_URL` and Qdrant point at shared data, note
that a chat turn writes conversation memory and a graded answer writes mastery.

### 3. Start the service

```bash
uvicorn system.asgi:app --host 0.0.0.0 --port 8000
curl localhost:8000/ready        # {"status":"ready", "graph_ready":true, ...}
```

Interactive API docs are at `http://localhost:8000/docs` (outside production). Add
`LOG_FILE=logs/ai-service.log` to keep a copy of the log in a file.

### 4. Ask it something

```bash
curl -s localhost:8000/ai/chat \
  -H "content-type: application/json" \
  -H "x-service-token: <AI_WEBHOOK_SHARED_SECRET from .env>" \
  -d '{"user_id": "<student uuid>", "user_type": "student",
       "session_id": "<chat session uuid>", "message": "What is evaporation?"}'
```

The reply comes back in `response`, with what it cited in `citations`. Conversation
memory is stored against a real platform user and chat session (the rows are keyed to
them), so use ids from the platform — or chat through the web app — to see follow-ups
work. For a sandbox with no platform data, set `agent.conversation.store: memory` in
`config.yaml` and memory is kept in the process instead.

### 5. Run it with the rest of the platform

Point the backend's `AI_SERVICE_BASE_URL` at `http://localhost:8000`, give it the same
`AI_WEBHOOK_SHARED_SECRET`, and start the backend and frontend from their repositories.
A message sent in the app's AI tutor then travels frontend → backend → this service.

### 6. Run the tests

```bash
.venv/bin/python scripts/verify.py   # lint and the unit suite, exactly as CI runs them
```

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
                ├── explain_sg / summarize_sg   retrieve → answer (or teach beyond the lesson) → prerequisites
                ├── assessment_sg               questions generated from retrieved passages
                ├── grade_sg                    marks answers; the only writer to mastery
                ├── report_sg                   mastery, trend and root-cause prerequisite
                └── visual_sg                   charts, mind maps, maths figures, flashcards, PDFs, slides, illustrations
            → render_visuals → compose
  → verify_language   the reply is in the language decided, or is repaired once
  → persist           remember the turn; fold old turns into a running summary
END
```

**1. The conversation is loaded.** Every turn is stored in `ai.user_memory` with the
passages its answer cited. A turn sees the last twelve turns verbatim, within a token
budget; older turns are folded into a running summary (goals, what was covered,
preferences, open questions) written by a small model, in the language the conversation
is being held in. Nothing is re-derived from one message alone.

Folding does not destroy what it covers. A folded turn changes its `kind`, so while the
whole conversation still fits `rebuild_tokens` the next summary is rewritten from what
was said rather than from the summary before it — a summary of a summary loses detail
every time one is taken, and nothing downstream can tell.

Three things survive the conversation:

- **The transcript**, which belongs to the platform. When what this service remembers has
  expired and `engagement.ai_chat_messages` has not, the thread is read back from it
  (`agent/conversation/transcript.py`) rather than meeting the student as a stranger
  inside a conversation they are looking at.
- **What the student has shown**, from `ai.mastery` — the topics graded evidence says are
  weak or secure, scoped to the subject and narrowed to this turn's topics
  (`agent/learner.py`). It steers depth and choice of example, and is never said back to
  the student.
- **Who the student is**, across chats (`agent/profile.py`). Languages and recent subjects
  are derived on read, so they cannot go stale; preferences are stored in
  `ai.adaptive_profile` and only when the student stated one in so many words.

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
- how the student wants it taught — `simpler`, `deeper`, `example`, `step_by_step` or
  `another_way` — each with one teaching strategy in `prompts.TEACHING_APPROACHES`, so a
  student who is lost gets a different explanation rather than the same one again;
- how the question fits the syllabus — on the `topic`, a `foundation` the book assumes,
  or `outside` the subject;
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

**4. The tasks run, and the reply is composed.** A student who asks to "practise my weak
spots" without naming a topic gets a quiz built from their weakest topics in the subject,
and any practice quiz gives weak topics more of its questions. Answers close with one
offer of a next step the student can actually take. Every sentence a student reads is written
by a model — explanations, refusals, small talk, progress notes. The service writes no
fixed sentences in any language; the only fixed marks are symbols (✓ ✗ → 💡 ∅) and
labels. A model call that fails fails the turn with a `503`, and the client shows its own
message.

**5. The turn is remembered**, along with the settled subject and any standing language
request; the summary is refolded once the window fills. A turn that produced no reply is
not remembered, and a turn that could not be remembered says which part of the request
was missing.

`GET /ai/memory/{user_id}` returns what this service holds about a student — the
summaries and their stated preferences, never the turns, which belong to the platform and
are already shown to them. `DELETE /ai/memory/{user_id}` destroys it, optionally for one
conversation. Who may ask is the platform's decision, as it is for every other endpoint
here.

### Pathways

A pathway changes the task allowlist and the voice. Parents are answered but not
remembered: `ai.user_memory` holds student and teacher rows only.

| Pathway | Allowed tasks | Voice |
|---|---|---|
| `student` | explain, summarize, assessment, grade, visual, report | friendly |
| `teacher` | explain, summarize, assessment, grade, visual | professional |
| `parent`  | explain, summarize, visual | friendly |

### Teacher personas

A teacher can give the tutor their own way of teaching. They upload material in their
own words (chat exports, written lesson transcripts, written explanations) and
can answer a short questionnaire. `agent/persona/` reads that material into passages of
the teacher's words, with students' names and contact details removed. A model drafts a
**card** describing how the teacher explains, checks understanding and reacts to
mistakes, and picks up to 40 passages as **examples**. The result is stored as a
draft. It reaches students only after the teacher reviews it and approves it.

On a student's turn, `teacher_id` selects the teacher's approved persona. The card and
the three examples nearest the question are added to the prompts that answer the
student: explain, summarize, conversation, no-material and compose. They sit below the
grounding and language rules, so the persona changes how the tutor explains and never
what it says is true. Grading and quiz generation stay neutral. The contract is in
`docs/BACKEND_INTEGRATION.md` §6, and the tables the backend must create are in
`docs/persona-schema-requests.md`. `scripts/eval_persona.py` measures whether the
replies sound like the teacher, and whether they add claims the plain tutor does not make.

### Voice

A student can ask by voice and hear the reply (`agent/voice/`, `shared/speech.py`).
`POST /ai/voice/transcribe` turns a push-to-talk recording into text with
ElevenLabs Scribe, which detects Arabic and English, including both mixed in one
sentence. The text then goes through `/ai/chat` like anything typed.
`POST /ai/voice/speak` reads a reply aloud with ElevenLabs, which uses the same
key, in the reply's language. It is off by default (`voice.speak.enabled`), and
`GET /ai/voice/capabilities` tells the frontend which buttons to show. Markdown, citations and the sources block are
removed first, and LaTeX is read as words in the reply's language. The contract is in
`docs/BACKEND_INTEGRATION.md` §7.

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
6. **Answer** — the model answers from the numbered passages, citing them inline. It
   leads with the lesson's own examples and says when an example is its own.
7. **Beyond the lesson** — one router in `explain_sg` picks the path. Passages found: the
   answer comes from them, and a step they skip is explained and marked as going beyond
   the lesson. Nothing found but inside the subject: the tutor teaches it, says so, and
   cites nothing. Outside the subject: declined. A `foundation` question searches the
   whole subject, since the basics are usually in an earlier lesson. Summaries never
   leave the material, and inside a selected lesson only its foundations go beyond it.
   `agent.beyond_lessons: false` turns this off per deployment.
8. **Grounding check** — an answer that cites nothing is never remembered as evidence and
   gets no sources or prerequisites. `answer_basis` on the chat response tells a client
   or an evaluation whether it was taught from the lessons, taught beyond them, or
   declined.
9. **Prerequisites** — walked through the topic graph from the passages the answer cited.

Citations, sources and prerequisites are returned as structured fields (`citations`,
`prereq_topics`) and are not printed in the reply text.

### Visuals

`visual_sg` renders up to three visuals a turn (`render.max_per_turn`). Charts, mind maps,
maths plots, geometry figures, flashcards, PDFs and slide decks are built by a small model
and drawn deterministically (`agent/render.py`), because image models cannot be trusted
with exact figures. Only the `image` kind — an illustration with nothing to get wrong —
goes to an image model (`shared/image_gen.py`, `gpt-image-1` by default). Rendered files
are stored in `ai.artifacts` and S3.

### Homework from a photo

`POST /ai/homework/capture` reads up to twenty numbered questions off a photographed page,
with the student's own handwritten answer beside each when there is one, so each can be
explained or marked. A blurred or unreadable page is reported as unreadable rather than
guessed at (`agent/homework.py`).

---

## Student and class analytics

The analytics are read-only and model-free, so a homepage can ask on every load. The
service never decides who may see whom: the backend passes only the students the signed-in
user may see.

- **Student summary** — `GET /ai/learners/{user_id}/summary` (`agent/analytics.py`). Per
  subject, how many topics are strong, developing, weak, unknown or not practised; the
  student's strengths and weak spots with their latest answers; their commonest mistake
  categories; graded answers per week and the trend; and a **review queue** of topics they
  have probably started to forget, with an estimated retention that grows with how often
  and how well each was assessed.
- **What to practise next** — `agent/recommend.py`, a pure policy over the evidence, so the
  homepage, the tutor and a teacher's view can never disagree. Weak topics come first,
  each preceded by any weak prerequisite it rests on; then developing topics; then
  untouched topics in the current lessons; then strong topics whose retention has faded.
  Reasons are returned as codes, never sentences, so the client words them in its reader's
  language.
- **Progress note** — `GET /ai/reports/{user_id}` writes a note (one model call) for the
  student or, with `scope=parent`, for their parent: a narrative whose every sentence cites
  its evidence, the weak topics, strengths, the root-cause prerequisite, the trend and a
  chart. It is stored in `ai.reports`; `.../latest` serves the stored one with no model
  call and says when it is stale.
- **Class analytics** — `POST /ai/classes/analytics` (`agent/class_analytics.py`), for a
  roster of up to 500 students: per-topic standings, students at risk with the facts that
  flagged them (many weak topics, declining, inactive), students not started, common
  mistakes, most-missed questions, score distribution, engagement, and groups of students
  weak on the same topic to reteach together. `POST /ai/classes/report` writes a summary
  for the teacher that never names a student; `GET /ai/classes/{class_id}/report/latest`
  serves the stored one.
- **Topic suggestions** — `POST /ai/topics/suggest` (`agent/tagging.py`). A question a
  teacher writes carries no topic, and an answer with no topic moves no mastery, so the
  nearest syllabus topics are suggested once, when the question is saved.

---

## Credits

Every call that pays a model is charged to one account before any model runs, and
refunded if it fails, times out or comes back degraded (`agent/credits/`,
`system/billing.py`). Each user has a wallet: a monthly allowance from their plan
(`free` 50, `basic` 500, `pro` 2000), spent first and reset each period, plus granted
extra credits that never expire. Every change is an append-only ledger entry that
records what the model gateway actually billed, which is the evidence prices are re-set
from.

| Action | Price (`credits.prices`) |
|---|---|
| chat message | 1 |
| grading | 1 per 10 answers |
| student progress note | 1 |
| class summary | 2 |
| homework capture | 1 |
| persona build | 5 |
| voice transcription | 1 |
| reading a reply aloud | 1 |

An account that cannot cover a call gets `402 insufficient_credits` with its balance and
renewal date, and nothing runs; a stream is refused before it opens. Credits are off
(`credits.enabled: false`) until the backend creates `ai.credit_wallets` and
`ai.credit_ledger`; while off, nothing is charged and the credits endpoints answer `503`.

---

## Ingestion

A PDF given with a lesson id (or a lesson's body text) is one lesson. A PDF given with
only its subject is a **book**, and is split into the lessons it teaches before anything
is indexed (`ingestion/books.py`): a model reads the book's opening pages and outline and
returns its units and the lessons in each, every lesson becomes its own row in
`content.lessons`, and every chunk, figure and topic is filed under the lesson whose pages
it came from. Two lessons taught on the same pages each get those pages, and lessons a
book teaches as a group share the group's pages. Every id is derived from the book, so
uploading the same book again replaces it instead of landing beside it.

```
extract       each page as typed blocks in reading order, chosen page by page. The PDF's
              own text layer is used wherever it can be trusted; a magazine-style page
              can go to a document layout model; a page with no usable text (a scan, or
              glyph noise from subset fonts) is read by a vision model (Gemini 2.5 Flash
              on Vertex, Gemini 2.5 Pro fallback), and what comes back is checked before
              it is believed. Every page read is cached in ai.page_transcriptions, so a
              re-run never pays for it twice. Figures are cropped to S3 with alt text. A
              lesson with more than 5% unreadable pages is held back as needs_review.
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

Before the pipeline walks a book, every page is rendered once, on every core, and both of
its model calls are made from that one image, many pages in flight
(`ingestion/read_ahead.py`).

The platform triggers ingestion when a lesson is published (`POST /ai/ingest/lesson`), and
reports each run's status back to it (`PATCH /lessons/{id}/ingestion`). It can also be
run three other ways:

- **From the command line**, on one machine — see [Running](#running) and
  [`docs/running-ingestion-on-your-own-machine.md`](docs/running-ingestion-on-your-own-machine.md).
- **On GitHub's machines** — the `Ingest a book` workflow (`.github/workflows/ingest-book.yml`)
  takes only a link to the book and its subject id, and splits the work: by default six
  machines read the pages and eight embed the chunks, then one builds the index
  (`ingestion/distributed.py`). A `preview` run lists the lessons it found and writes
  nothing. `scripts/ingest_on_github.sh books.txt` sends a whole shelf. See
  [`docs/ingesting-on-github-actions.md`](docs/ingesting-on-github-actions.md).
- **On AWS** — `scripts/ingest_on_aws.sh`, with `Dockerfile.ingest` and
  `infra/ingest-task-definition.json`.

---

## Where data lives

| Store | What | Written by |
|---|---|---|
| `ai.chunks`, `ai.figures`, `ai.vector_index`, `ai.topics`, `ai.topic_dependencies`, `ai.lecture_ingestion`, `ai.page_transcriptions` | the indexed syllabus | ingestion |
| `ai.user_memory` | conversation turns (quoted and folded), running summary, session state (subject, language) | agent |
| `ai.adaptive_profile` | preferences a student stated, carried across conversations | agent |
| `ai.mastery`, `ai.mastery_events`, `ai.mistake_tags` | per-skill mastery and its history | agent (grading) |
| `ai.reports` | stored progress notes and class summaries | agent |
| `ai.teacher_personas`, `ai.teacher_persona_exemplars` | persona versions, their cards and chosen examples | agent |
| `ai.credit_wallets`, `ai.credit_ledger` | each user's plan, balance and append-only credit history | service |
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
│   ├── conversation/        turn memory: records, store (ai.user_memory), history window,
│   │                        transcript recovery when what we kept has expired
│   ├── learner.py           what graded evidence says this student can do
│   ├── profile.py           what carries across conversations, and what a student may erase
│   ├── analytics.py         the student summary: standings, strengths, weak spots, review queue
│   ├── class_analytics.py   a teacher's view of a class
│   ├── recommend.py         what to practise next, and why
│   ├── tagging.py           topic suggestions for questions written outside ingestion
│   ├── credits/             plans, prices, wallets and the append-only ledger
│   ├── persona/             teacher personas: material, build, store, voice in prompts
│   ├── voice/               transcription and reading replies aloud
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
│   ├── layout/              page → typed blocks → chunks (tier router, text layer,
│   │                        layout model, vision, page cache)
│   ├── stages/              extract · chunk · embed · topics · dependencies · index
│   ├── books.py             a book split into its lessons
│   ├── lesson.py            one lesson, as every ingestion command describes it
│   ├── read_ahead.py        every page read once, on every core, before the pipeline
│   ├── distributed.py       one book across several machines: plan · read · prepare ·
│   │                        embed · finish
│   ├── prompts/
│   └── stores.py            Postgres and Qdrant writers (dry-run fallback)
├── shared/                  imported by both halves
│   ├── language/            the reply-language policy, detection, directives, verification
│   ├── llm.py · structured.py · prompting.py   model access, schema-checked output, prompts
│   ├── embeddings.py · lexical.py · matching.py · vectors.py · vector_store.py
│   ├── postgres.py · storage.py · backend.py · vision.py · gcp.py
│   ├── speech.py · image_gen.py                voice provider, illustrations
│   └── config.py · logging.py · metering.py · errors.py
├── system/                  FastAPI service: routers, schemas, runtime, ingestion jobs,
│                            billing, status reports to the platform
├── scripts/                 bootstrap, verify, ingestion (local, GitHub, AWS), evaluations
├── tests/unit/
├── docs/                    backend contract, OpenAPI, schema requests, ingestion guides
├── sdlc/ · .claude/         the developer and QA agents' change workflow
├── .github/workflows/       deploy on push to main; ingest a book
├── config.yaml              every behavioural setting
└── .env.example             secrets and connection strings
```

Layering: `shared/` depends on nothing above it; `agent/` and `ingestion/` never import
each other; `system/` sits on top. Read [`CLAUDE.md`](CLAUDE.md) before changing code — it
lists the rules this codebase holds itself to and the failure behind each one.

---

## Configuration

| File | Holds | Committed |
|---|---|---|
| `.env` | secrets and connection strings | no |
| `config.yaml` | everything that changes behaviour | yes |

The model is a config change: `llm.provider` and `llm.model` move the whole system, and
`llm.overrides` sets a model per task. Today everything that answers a user runs on
Gemini 2.5 through OpenRouter (Flash for reading turns, writing answers, grading, reports
and personas; Flash-Lite for composition, conversation summaries, speech scripts and
visual builders). Ingestion bills to one account on Vertex: page reading on Gemini 2.5
Flash with a Pro fallback, topics and prerequisites on Flash, and splitting a book into
lessons on Gemini 2.5 Pro. Voice uses ElevenLabs (Scribe v2 to listen, Flash v2.5 to
speak), illustrations `gpt-image-1`, and the evaluation judge GPT-4.1 — a different
family from the model it marks.

Features that need backend tables ship switched off and are turned on in `config.yaml`
once the tables exist: `credits.enabled`, `persona.enabled` and `voice.speak.enabled`.

`APP_ENV` selects an overlay from `environments:` in `config.yaml`:

| `APP_ENV` | Index it searches | Notes |
|---|---|---|
| `local` | `eduai_chunks_dev` | debug logging; keeps a dev index apart from real data |
| `staging` | `eduai_chunks_v2` | the base configuration |
| `production` | `eduai_chunks_v2` | no `/docs`, warning-level logs, more model retries |

The settings most often tuned: `agent.min_score`, `agent.k`, `agent.expand`,
`agent.scope.relevance_ratio`, `agent.conversation` (window, folding, TTL),
`agent.beyond_lessons`, `language.reply_follows`, `assessment.*`, `mastery.*`,
`recommend.*`, `class_analytics.*` and `credits.*`. Each threshold carries the
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
# a whole book: split into its lessons, each created under the subject
python -m ingestion path/to/book.pdf --subject-id <uuid> [--language en]

# one lesson
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
| `GET` | `/ai/learners/{user_id}/summary` | A student's standing, strengths, weak spots, review queue and what to practise next |
| `GET` | `/ai/reports/{user_id}` | Write and store a progress note (for the student or their parent) with its trend chart |
| `GET` | `/ai/reports/{user_id}/latest` | The newest stored note, without writing a new one |
| `POST` | `/ai/classes/analytics` | Analyse a class roster |
| `POST` | `/ai/classes/report` | Write and store a teacher's summary of a class |
| `GET` | `/ai/classes/{class_id}/report/latest` | The newest stored class summary |
| `POST` | `/ai/topics/suggest` | Suggest syllabus topics for questions a teacher wrote |
| `GET` | `/ai/memory/{user_id}` | What the tutor remembers about a student |
| `DELETE` | `/ai/memory/{user_id}` | Destroy it, optionally for one conversation |
| `POST` | `/ai/homework/capture` | Read the questions off a photographed homework page |
| `GET` | `/ai/voice/capabilities` | Which voice features are on, for showing the mic and speaker buttons |
| `POST` | `/ai/voice/transcribe` | Turn a recorded question into text |
| `POST` | `/ai/voice/speak` | Read a tutor reply aloud |
| `POST` | `/ai/personas/build` | Draft a new persona version from a teacher's material |
| `GET` | `/ai/personas/{teacher_id}` | Every persona version a teacher has, and which one students hear |
| `GET` | `/ai/personas/{teacher_id}/versions/{version}` | One version, with the examples chosen from the teacher's words |
| `PUT` | `/ai/personas/{teacher_id}/versions/{version}/card` | Replace a version's card with the teacher's corrections |
| `POST` | `/ai/personas/{teacher_id}/versions/{version}/activate` | Approve a version for the teacher's students |
| `POST` | `/ai/personas/{teacher_id}/deactivate` | Go back to the default tutor |
| `DELETE` | `/ai/personas/{teacher_id}` | Destroy every persona version and example held for a teacher |
| `GET` | `/ai/credits/plans` | Plans and what each action costs |
| `GET` | `/ai/credits/{user_id}` | A user's credit balance |
| `GET` | `/ai/credits/{user_id}/ledger` | A user's credit history, newest first |
| `POST` | `/ai/credits/{user_id}/grants` | Give a user extra credits that never expire |
| `PUT` | `/ai/credits/{user_id}/plan` | Move a user to another plan |
| `POST` | `/ai/ingest` | Queue a lesson from a source URI; returns `202` and a job id |
| `POST` | `/ai/ingest/lesson` | Queue a lesson by platform id (fired on publish) |
| `GET` | `/ai/ingest`, `/ai/ingest/{job_id}`, `/ai/ingest/lectures/{lecture_id}` | Recent jobs, one job, a lecture's latest job |
| `GET` | `/health`, `/ready` | Liveness and readiness |

Every failure returns one envelope, and every response carries `x-request-id` (an upstream
id is honoured):

```json
{"error": {"code": "service_unavailable", "message": "…", "request_id": "…"}}
```

A paid call that the account cannot cover answers `402 insufficient_credits`. Paid
responses carry `usage` — what the model gateway billed for the call.

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
| `scripts/eval_conversations.py` | multi-turn behaviour: follow-ups, re-teaching, teaching beyond the lesson, small talk, refusals, language — against a running service |
| `scripts/eval_persona.py` | whether persona replies sound like the teacher, and whether they add claims the plain tutor does not make |
| `scripts/e2e_features.py` | the real routes end to end against a deployed stack, once and with many students at once |

```bash
.venv/bin/python scripts/verify.py   # the same lint and suite CI runs; this is "done"
```

`scripts/verify.py` keeps the unit run away from production. A bare `pytest` inherits
`.env`, whose `DATABASE_URL` may point at real data.

---

## Deployment

A push to `main` runs the lint and tests, builds the image (`Dockerfile`), pushes it to
ECR and rolls the ECS service in `eu-central-1`
(`.github/workflows/deploy.yml`). `/ready` answers `503` until the graph is compiled, so
it is the deployment gate. Ingestion runs separately: on GitHub's machines through the
`Ingest a book` workflow, or as an ECS task from `Dockerfile.ingest`.

---

## Contributing

- Read [`CLAUDE.md`](CLAUDE.md) first. It lists the rules this codebase holds itself to —
  fix the producer rather than filter at display, one decision in one place, measure
  before and after — each with the failure that made it a rule.
- Work on a branch and open a pull request into **`staging`**. One concern per PR, and the
  description says what was broken, not what the diff does.
- `staging` is released to **`main`** by a pull request. A push to `main` runs the tests
  and deploys the service (`.github/workflows/deploy.yml`).
- Tests are named for the behaviour they protect and must fail before the fix. `ruff check`
  must not gain findings; never run `ruff --fix` across the whole repository.
- A behaviour change comes with a before-and-after number from one of the evaluations.

---

## Known limitations

- **Chunking is per heading and per page.** A section split at its sub-headings is
  recovered at answer time by reading neighbouring chunks, but chunks never cross a page,
  do not carry their parent heading, and on magazine-style pages can still hold
  fill-in answer keys and stray letters from decorative titles. Fixing it in the chunker
  needs a re-ingest.
- **Topics are coarse on long lessons.** A lesson gets 3–8 topics, and prerequisite edges
  are derived by a model without review.
- **The platform does not yet pass structured citations through** to the frontend, and the
  frontend sends the study mode inside the message rather than as a field.
- **Credits, personas and reading aloud are off** until the backend creates their tables
  and the frontend shows them.
- **Parents have no conversation memory**, and session uploads are not built yet.
- **`needs_review` is reported to the platform as `ready`**, because its status enum has
  no equivalent; what was indexed is usable.
