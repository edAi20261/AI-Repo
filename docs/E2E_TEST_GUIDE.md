# End-to-end feature tests

Unit tests prove a function works. They do not prove that the platform reaches
this service, that a reply survives the backend, or that thirty students at once
get an answer. This guide lists every AI feature, how it reaches a user, and how
to test it against the deployed stack with real models. It also records what the
last run found.

The tests are in `scripts/e2e_features.py`. `scripts/e2e_in_vpc.sh` runs them from
inside the VPC.

---

## 1. How a request reaches the AI service

```
browser ──► frontend (React, CloudFront)
              │  /api/ai-chat/*   (JWT)
              ▼
           backend (NestJS, App Runner, public)
              │  x-service-token
              ▼
           internal ALB ──► AI service (FastAPI, ECS, 1 task, 2 vCPU / 4 GB)
                                │
                                ├── Postgres  ai.*   (RDS db.t4g.micro, 79 connections)
                                ├── Qdrant    eduai_chunks_v2
                                ├── OpenRouter (Gemini 2.5 Flash / Flash-Lite, Qwen3-VL)
                                └── S3        rendered visuals, figures
           backend ◄── /api/ai-integration/*  (the AI service reads subjects, lessons, keys)
```

The AI service is behind an **internal** load balancer. Only the backend's VPC
connector security group (`sg-02f9099b3fe1c7b36`) can reach it. From a laptop you
can test only what the backend proxies. For everything else, run the tests in
the VPC (section 5).

## 2. Every feature, and whether a user can reach it today

Every feature below was built in this repository by the AI engineer. The only
outside commits are the Dockerfiles and the CI workflow.

"Deployed" means it was on `main`, which is what ECS runs, during the test rounds
(image `2f07305`, 2026-09-30). `8a94ba1` replaced it at 15:16 UTC on the day of the run;
see 6.3. "Wired" means the backend calls it and the frontend shows the result.

| # | Feature | AI endpoint | Deployed | Backend route | Frontend |
|---|---|---|---|---|---|
| 1 | Tutor chat, grounded and cited (student) | `POST /ai/chat` | yes | `POST /api/ai-chat/sessions/:id/messages` | AI tutor page: text and citations |
| 2 | Conversation memory and follow-ups | `POST /ai/chat` (`session_id`) | yes | same, session id = chat session | yes |
| 3 | Arabic / English / French / German replies | `POST /ai/chat` | yes | same | yes |
| 4 | Off-syllabus refusal | `POST /ai/chat` | yes | same | yes |
| 5 | Lesson scoping | `POST /ai/chat` (`lecture_ids`) | yes | `lectureIds` | not sent by the UI |
| 6 | Practice quiz in chat | `POST /ai/chat` → `assessment` | yes | passed through | questions show as text, `assessment` ignored |
| 7 | Grading a quiz answered in chat | `POST /ai/chat` → `grade_result` | yes | passed through | `gradeResult` ignored |
| 8 | Visuals: mind map, chart, flashcards, figures | `POST /ai/chat` → `visuals` | yes | passed through | **not rendered; link is `s3://`** |
| 9 | Prerequisites | `POST /ai/chat` → `prereq_topics` | yes | passed through | not rendered |
| 10 | Teacher and parent tutor voice | `POST /ai/chat` (`user_type`) | yes | **student only** (`@Roles(student)`) | no |
| 11 | Streamed chat (stage progress) | `POST /ai/chat/stream` | yes | **none** | no |
| 12 | Grading and mastery | `POST /ai/grade` | yes | **none** | no |
| 13 | Student homepage analysis | `GET /ai/learners/{id}/summary` | yes | **none** | no |
| 14 | Progress report, student and parent | `GET /ai/reports/{id}`, `/latest` | yes | **none** | no |
| 15 | Class analytics (teacher) | `POST /ai/classes/analytics` | yes | **none** | no |
| 16 | Class report (teacher) | `POST /ai/classes/report`, `GET …/latest` | yes | **none** | no |
| 17 | Topic suggestions for teacher questions | `POST /ai/topics/suggest` | yes | **none** | no |
| 18 | Homework photo → questions | `POST /ai/homework/capture` | yes | **none** | no |
| 19 | Memory read / erase | `GET`, `DELETE /ai/memory/{id}` | yes | **none** | no |
| 20 | Credits and plans | `/ai/credits/*` | yes, **disabled** (`credits.enabled: false` → 503) | none | no |
| 21 | Lesson ingestion on publish | `POST /ai/ingest/lesson` | yes | fired by `LessonsService` on publish | lesson status |
| 22 | Ingestion jobs | `POST /ai/ingest`, `GET /ai/ingest…` | yes | none (operator) | no |
| 23 | Teacher personas | `/ai/personas/*` | since `8a94ba1`, answers 503 (tables missing) | none | no |
| 24 | Voice: speech to text, reply to speech | `/ai/voice/*` | since `8a94ba1`, keys not in the task | none | no |
| 25 | Health and readiness | `GET /health`, `/ready` | yes | ALB health check | — |

The backend connects to 2 of the AI service's 34 routes (`/ai/chat` and
`/ai/ingest/lesson`). Features 10 to 20 work in the AI service, but no user can
reach them until the backend adds a route and the frontend adds a screen.

## 3. Test data

| What | Value |
|---|---|
| English subject | Science, Grade 7 — `25e488c1-9582-40f8-9656-ccbd229a8656`, lesson `9d3f1b20-0002-4a00-9000-000000000006` |
| Arabic subject | العلوم, Grade 7 — `7b1e5c40-0a6e-4d1a-9f3c-5d2a91c00007`, lesson `9d3f1b20-0002-4a00-9000-000000000014` |
| Topics used for grading | atomic structure, elements/compounds/mixtures, valency, isotopes (ids in the script) |
| Test students | `e2e-<tag>-NN@test.eduai`, Grade 7, National curriculum, made through `/api/auth/register` |

Everything is production: it is the only deployed environment. The tests write
chat sessions, tutor memory, mastery events and stored reports for the test
students only. `cleanup` erases their tutor memory. Mastery, reports and the
accounts stay, because there is no API to remove them.

## 4. The checks

### 4.1 Through the backend, as the frontend calls it (`backend` mode)

| Check | Request | Passes when |
|---|---|---|
| Backend liveness | `GET /api/health` | `status: ok` |
| Register students | `POST /api/auth/register` ×N | 201 (or log in if they exist) |
| Open a session | `POST /api/ai-chat/sessions` with `subjectId` | an id comes back |
| Explains from the syllabus | "What are the main parts of an atom?" | a reply longer than 20 characters, `language: en`, `tasks: [explain]` |
| Citations reach the client | "Where is the mass of an atom concentrated?" | `citations` is not empty |
| Follow-up | "Explain that more simply, like I'm 10." | a reply that continues the topic |
| Quiz | "Give me a 3 question quiz on atomic structure." | `assessment.items` is present |
| Quiz is marked | "My answers: 1) … 2) … 3) …" | `gradeResult` is present |
| Visual | "Draw me a mind map of the structure of the atom." | `visuals` is not empty |
| Off-syllabus refusal | "Who won the 2022 football World Cup final?" | no answer from outside the syllabus |
| Lesson scope | "What is valency?" with `lectureIds` | a reply |
| Arabic | "ما هي مكونات الذرة؟" in the Arabic session | `language: ar` and an Arabic reply |
| History | `GET …/messages` | every turn is stored |
| Ownership | another student reads the session | 403 |
| Authentication | no JWT | 401 |

### 4.2 Inside the VPC, every route the backend does not proxy (`direct` mode)

| Check | Request | Passes when |
|---|---|---|
| Liveness, readiness | `GET /health`, `GET /ready` | `graph_ready: true` |
| Service token | `POST /ai/chat` without `x-service-token` | 401 |
| Strict bodies | an unknown field | 422 error envelope |
| Chat contract | `POST /ai/chat` | `citations` and `usage` are present |
| Stream | `POST /ai/chat/stream` | `progress` events, then `result` |
| Teacher and parent voice | `user_type: teacher` / `parent` | `pathway` matches |
| Grading | 3 answers, one with `known_score` | `mastery_recorded: true`, events ≥ 1, not degraded |
| Homepage | `GET /ai/learners/{id}/summary` | `available`, not degraded |
| Report | `GET /ai/reports/{id}?billed_user_id=` | a non-empty `narrative`, `persisted: true` |
| Stored report | `GET /ai/reports/{id}/latest` | `report_id` |
| Parent report | `?scope=parent` | a non-empty `narrative` |
| Class analytics | `POST /ai/classes/analytics` with the 30 students | not degraded |
| Class report | `POST /ai/classes/report`, then `GET …/latest` | `persisted: true`, then read back |
| Topic suggestions | `POST /ai/topics/suggest` | topics for each question |
| Homework capture | a PNG of 4 printed questions | at least 3 questions read |
| Homework validation | a `.txt` upload | 422 |
| Memory | `GET /ai/memory/{id}` | at least one conversation |
| Credits | `GET /ai/credits/plans` | 200, or 503 while disabled |
| Ingestion | `POST /ai/ingest/lesson` with an unknown lesson | 202, then the job `failed` with the reason |
| Not deployed yet | `/ai/personas/…`, `/ai/voice/speak` | anything but 404 |

### 4.3 Load (`stress-backend`, `stress-direct`)

N students act at the same moment. Each wave starts only after the previous one
has finished. While the in-VPC waves run, `/health` is called every second to
see whether the service stays responsive.

| Wave | What runs at once |
|---|---|
| chat: first question | every student asks a different syllabus question through the backend |
| chat: follow-up | every student asks a follow-up in the same session |
| chat: quiz | every student asks for a quiz |
| chat history | every student reloads their history |
| grade | every student submits 3 answers |
| homepage | every student opens the learner summary |
| chat stream | every student streams a question |
| report | every student builds a progress report |
| memory | every memory read |
| topic suggest | every teacher tags a question |
| homework capture | every student uploads a photo |
| mixed | 10 chats, 10 gradings, 5 summaries and 5 reports together |
| class analytics | the whole roster in one call |

## 5. Running it

```bash
# 1. Through the backend. Run it from anywhere; it registers the students.
python scripts/e2e_features.py backend \
    --api https://pwqfiqua63.eu-central-1.awsapprunner.com/api \
    --students 30 --tag r1 --students-out e2e_students.json --out r1_backend.json

# 2. Inside the VPC. Needs AWS credentials that can run ECS tasks.
scripts/e2e_in_vpc.sh direct e2e_students.json

# 3. Load: 30 students at once.
python scripts/e2e_features.py stress-backend \
    --api https://pwqfiqua63.eu-central-1.awsapprunner.com/api --students 30 --tag r1
scripts/e2e_in_vpc.sh stress-direct e2e_students.json --students 30

# 4. Erase the test students' tutor memory.
scripts/e2e_in_vpc.sh cleanup e2e_students.json
```

During a load run, also watch ECS `CPUUtilization` and `MemoryUtilization` for
`edu-ai-ai/edu-ai-ai-service` and RDS `DatabaseConnections` for `edu-ai-db`. Search
the `/ecs/edu-ai-ai-service` log group for `level = ERROR`.

---

## 6. Results — 2026-10-04, production, image `2f07305`

### Round 1 — one student

**Through the backend: 16 of 16 passed.**

| Check | Time | What came back |
|---|---|---|
| Explains (EN) | 7.3 s | `en`, `[explain]`, 1 prerequisite |
| Citations | 5.1 s | 1 citation reached the client |
| Follow-up | 4.1 s | "Imagine an atom is like a tiny, tiny solar system…" |
| Quiz | 6.1 s | `assessment` with `items`, `count`, `type_mix` |
| Quiz marked | 7.1 s | `gradeResult` with `mastery_recorded` |
| Mind map | 5.0 s | `visuals[0].kind = mindmap`, but at an `s3://` URI (see bug 4) |
| Off-syllabus | 1.9 s | "I can only help you with questions about the course material…" |
| Arabic | 3.6 s | `ar`: "تتكون **الذرة** من جزأين رئيسيين…" |
| History, 403, 401 | <0.3 s | as expected |

**Inside the VPC: 21 of 26 passed.**

| Check | Result |
|---|---|
| Health, ready, token, 422 envelope | pass |
| Chat contract | pass: 1 citation, `usage` $0.0023, 2 calls |
| Stream | pass: 10 events, then `result` |
| Teacher and parent voice | pass: `pathway` teacher / parent |
| Grading | pass: 3 events, 3 topics, not degraded, 1.4 s |
| Homepage summary | pass: available, 1 weak topic, 5 recommendations, 0.0 s |
| **Student report** | **FAIL**: `available: true`, `narrative: ""`, `persisted: false` (bug 1) |
| **Stored report** | **FAIL**: 404, because nothing had been stored (bug 1) |
| **Parent report** | **FAIL**: empty narrative (bug 1) |
| Class analytics and class report | pass: stored, then read back |
| Topic suggestions | pass: "Determine element valency", "Differentiate elements, compounds, mixtures" |
| Homework capture | pass: 4 of 4 questions read, 4.0 s |
| Memory, credits (503 disabled), ingestion (fails cleanly on an unknown lesson) | pass |
| Personas, voice | 404: on `staging`, not deployed |

### Round 2 — 30 students at once

**Through the backend (App Runner → AI):**

| Wave | OK | p50 | p95 | Failures |
|---|---|---|---|---|
| First question | 29/30 | 14.5 s | 25.0 s | 1 × 500 `Already borrowed` (bug 2) |
| Follow-up | 30/30 | 18.2 s | 30.1 s | — |
| Quiz | 25/30 | 21.7 s | 34.0 s | 4 × 503 (bug 3), 1 × 500 (bug 2) |
| History | 30/30 | 0.3 s | 0.5 s | — |

**Inside the VPC:**

| Wave | OK | p50 | p95 |
|---|---|---|---|
| Grade | 30/30 | 3.6 s | 6.1 s |
| Homepage summary | 30/30 | 0.1 s | 0.1 s |
| Chat stream | 30/30 | 16.5 s | 27.0 s |
| Report | 30/30 HTTP 200, **every narrative empty** (bug 1) | 2.0 s | 2.7 s |
| Memory read | 30/30 | 0.7 s | 1.0 s |
| Topic suggest | 30/30 | 5.0 s | 9.4 s |
| Homework capture | 30/30 | 8.2 s | 19.7 s |
| Mixed | 30/30 | 7.4 s | 10.3 s |
| Class analytics (30 students) | 1/1 | 0.0 s | — |
| `/health` during the load | 81 samples | 0.0 s | 0.0 s (max 0.1 s) |

### 6.3 Re-check on the image deployed during the run (`8a94ba1`)

Staging was merged to `main` and deployed at 15:16 UTC, after both rounds had
finished. The in-VPC functional round was re-run against it: **20 of 26 passed.**

| Check | `2f07305` | `8a94ba1` |
|---|---|---|
| Report, stored report, parent report | fail (bug 1) | fail (bug 1) |
| **Homework capture** | pass, 4 of 4 questions | **500** (bug 5) |
| Homework capture, non-image | 422 | 502 from the ALB, on the connection after the 500 |
| Personas | 404 | 503: tables missing, as documented |
| Voice | 404 | 422 on an empty body: the route is live |
| Quiz turns failing on `"null"` | — | 8 more "reading the turn failed" with no load (bug 3) |

**Infrastructure under load:** ECS CPU peaked at **86 %** during the chat waves.
Memory stayed at 37 %. RDS used 19 of 79 connections at 6 % CPU. The database is
not the limit. The single AI task is: chat turns queue behind
`api.max_concurrency: 8`, which is why p95 rises from about 7 s alone to 25–35 s
with thirty students. ECS has **no autoscaling** configured, so the service is
always one task.

## 7. Bugs found, and their fixes

| # | Bug | Cause | Fix |
|---|---|---|---|
| 1 | Every student and parent report has an empty narrative and is never stored, so `/latest` returns 404 | The router passes `language=None` when the query has none, and `payload.get("language", "en")` keeps the `None`. The prompt then refuses to render (`REPORT_STUDENT_PROMPT: needs the language…`). The tests always passed `"en"`, so this never showed. | `fix/report-language-unset`: fall back to `language.default`, read in one place |
| 2 | 1 in 30 concurrent chat turns returns 500 | The BGE-M3 fast tokenizer is shared by the worker threads, and Rust refuses a second borrow (`RuntimeError: Already borrowed`) | `fix/embedder-tokenizer-thread-safety`: a lock inside the embedder |
| 3 | A quiz request intermittently returns 503 "the tutor could not answer" | The model writes `"grade": "null"` and `"types": "null"` as strings. Validation fails twice. 37 first readings failed in one load run. | `fix/planner-null-string-specs`: the planner's models read a null word as an absent field |
| 5 | Homework capture returns 500 since `8a94ba1`; lesson ingestion on publish fails at the topics stage | `vision.provider: vertex` is global, and `extract_topics` / `derive_deps` now go to Vertex. The ECS service task has no `GOOGLE_APPLICATION_CREDENTIALS`, and neither does `infra/ingest-task-definition.json`. Homework capture is a student feature, but it reads its model from the same `vision` block as ingestion. | Open: needs a decision (section 8, item 9) |
| 6 | A provider that cannot be reached is a bare 500 | `VisionUnavailable` is a `RuntimeError` that no handler maps, so the platform gets `internal_error` instead of `service_unavailable` with the reason | Open, together with 5 |
| 4 | Visuals cannot be opened in the browser | The service returns only `s3://` URIs, and the reply text is `[🧠](s3://…)`. The frontend ignores `visuals`. | Open: needs a contract decision (section 8) |

Each fix comes with a regression test that reproduces the production error and
fails without the fix. Full suite: 1375 tests pass before, and after with the
new ones. `ruff check` stays clean.

## 8. Not bugs in this repository, but they block users

1. **The backend calls 2 of 35 routes.** Grading, the homepage, reports, class
   analytics, topic tagging, homework capture, memory and streaming have no
   backend route and no screen. Teachers and parents cannot use the tutor,
   because `/api/ai-chat` is `@Roles(student)`.
2. **Visual delivery.** The backend should turn `visuals[].s3_uri` into signed
   `https` URLs, as `media/:id/url` already does for lessons. The frontend
   should render `visuals`, `assessment` (as a quiz) and `prereqTopics`.
3. **Deployed backend ≠ `main`.** Citations reach the client in production, but
   `ai-chat.service.ts` on `main` does not pass them through. The running
   backend was built from code that is not pushed.
4. **The backend logs the shared secret** (`ai-client.service.ts`,
   `console.log('AI_WEBHOOK_SHARED_SECRET:', …)`). It is in App Runner's logs.
   Remove the log line and rotate the secret.
5. **Capacity.** One ECS task at 86 % CPU with 30 students. Add ECS service
   autoscaling on CPU (minimum 2 tasks also removes the single point of
   failure). `max_concurrency` per task is the other lever.
6. **Deployed but not usable.** Personas and voice are live since `8a94ba1`.
   Personas need `ai.teacher_personas` and `ai.teacher_persona_exemplars`, and
   voice needs `ELEVENLABS_API_KEY` in the task. Credits are off until `ai.credit_wallets`
   and `ai.credit_ledger` exist.
7. **Licensing.** `[AI TEST] Science Fusion G6 Module F`
   (`aaaaaaaa-0000-4000-8000-00000000f001`) is `ready` in `ai.lecture_ingestion`,
   with 545 rows in `ai.chunks`. `CLAUDE.md` says this book must not be in the
   index. It needs removing from `ai.chunks`, `ai.figures`, `ai.topics` and
   Qdrant (a production write, so confirm first).
8. **Schema drift.** `ai.mistake_tags.subject_id` is missing in production, so
   mistake tags are created unscoped (logged on every graded answer).
9. **Vertex in production.** Either give the service and ingest tasks the
   service-account JSON (an SSM secret mapped to `GOOGLE_APPLICATION_CREDENTIALS`,
   or the JSON in a variable as `69a4240` allows), or keep student-facing vision
   (homework capture) on OpenRouter with its own provider setting. Until one of
   them is done, every published lesson stays un-ingested.
