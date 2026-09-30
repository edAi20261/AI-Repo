# Calling the AI service from the platform backend

For the NestJS backend. Everything here is live in `system/` on `staging`;
`docs/openapi.json` is the generated spec, and running the service gives you
Swagger at `/docs`.

- **Base URL** — `AI_SERVICE_BASE_URL`
- **Auth** — header `x-service-token: <AI_WEBHOOK_SHARED_SECRET>`, the same
  secret your `AiServiceGuard` already checks in the other direction
- **Bodies** — JSON, `snake_case`

The one rule that matters: **take `user_id` and `user_type` from the verified
JWT (`sub` and `role`), never from the request body.** This service does not
authenticate users — it trusts what you send. `user_type` decides what the
caller is allowed to do, so a wrong role grants the wrong capabilities.

---

## 1. Chat — the route you need to add

`POST /ai/chat`

```json
{
  "user_id":     "uuid",
  "user_type":   "student | teacher | parent",
  "message":     "Explain the fire triangle.",
  "subject_id":  "uuid",
  "lecture_ids": ["uuid"],
  "session_id":  "uuid",
  "grade":       "g8",
  "language":    "ar | en | fr | de"
}
```

| Field | Required | Notes |
|---|---|---|
| `user_id` | yes | from `sub` |
| `user_type` | yes | from `role`. Anything else is rejected |
| `message` | yes | 1–8000 chars |
| `subject_id` | no | scopes the search. Inferred from the message if omitted |
| `lecture_ids` | no | narrows to specific lessons |
| `session_id` | no | Conversation id. **Now carries the standing answer language**: stable for a conversation, new for a new one |
| `grade` | no | academic year, e.g. `g8` |
| `language` | no | Fallback only — the message decides. Never send the UI language |

Unknown fields are rejected with `422` — deliberate, so a typo surfaces instead
of being silently ignored.

**Response**

```json
{
  "request_id":    "hex",
  "response":      "Combustion needs fuel, oxygen and heat [1].",
  "language":      "en",
  "pathway":       "student",
  "tasks":         ["explain"],
  "visuals":       [{ "kind": "chart", "s3_uri": "s3://..." }],
  "assessment":    {},
  "grade_result":  {},
  "prereq_topics": [{ "label": "Oxygen", "strength": "required", "depth": 1 }],
  "usage":         { "cost_usd": 0.0042, "prompt_tokens": 3100, "completion_tokens": 450, "calls": 2 }
}
```

For a normal turn, pass `response` and `visuals` through to the frontend. The
rest is populated only when the planner produced it — `assessment` when the
student asked for a quiz, `grade_result` when it graded something.

`usage` is always present: what the model gateway billed for every call the
turn made, retries included. It is for metering, never for the student — store
it beside the credit charge so the price table can be checked against the bill.
The streamed `result` event carries it too.

**Set the timeout to at least 120s.** A turn makes about three model calls. The
10s you use for the ingestion trigger will cut off every real answer.

### Streaming (optional)

`POST /ai/chat/stream` — same body, `text/event-stream` back. A `progress` event
per stage, then one `result` event carrying the body above. Useful if you want
the UI to show activity; the plain endpoint is simpler to start with.

---

## 2. Grading

`POST /ai/grade`

```json
{
  "user_id": "uuid",
  "billed_user_id": "uuid",
  "submissions": [
    { "question": "...", "student_answer": "...", "correct_answer": "...",
      "topic_id": "uuid", "difficulty": "easy | medium | hard" }
  ],
  "kind": "quiz | homework",
  "assessment_id": "uuid",
  "subject_id": "uuid"
}
```

Only `user_id` and `submissions` are required (1–200 items). `assessment_id`
makes the service fetch the stored rubric — it requires `kind` alongside it.
Answers arrive in the payload; this service never reads a submission from the
database.

Returns `graded[]`, `summary`, `mastery_events`, `topics_updated`,
`mastery_recorded`, `source`. Grading is the only thing that moves a student's
mastery.

**Answers you have already scored.** Send `known_score` (0–1, the fraction of
full marks) on a submission and the model does not grade it again: the score is
recorded as given. Use it for choice and true/false questions you auto-score,
and for any answer a teacher has marked. Send `question_id`, `topic_id` and
`lecture_id` with it — an answer without a topic and lesson is graded but cannot
move mastery.

**`mastery_recorded: false`** means the answers were graded but the evidence
could not be stored. `mastery_events` is then 0. Retry the same request; nothing
was written.

---

## 3. Student analysis and reports

Everything here is read-only except the report builders, and nothing but the
report builders calls a model. The service never decides who may see whom: pass
only students the signed-in user is allowed to see.

### Student homepage — `GET /ai/learners/{user_id}/summary`

Optional query: `subject_ids` (repeatable: the subjects the student is enrolled
in, counted even when untouched; exactly one scopes everything to that subject)
and `lecture_ids` (the lessons they are working through now).

Returns, with no model call:

- `subjects[]` — per subject, how many topics are `strong`, `developing`, `weak`,
  `unknown` (too little evidence yet) and `not_practised`. Show "3 of 12
  mastered", not an average.
- `strengths[]`, `weaknesses[]`, `not_practised[]` — topics with label, lesson,
  score and the latest answers.
- `recommendations[]` — what to practise next, in order. `action` is
  `review_prerequisite | practise | strengthen | start | refresh`; `reasons` are
  facts with a `code` (`recent_misses`, `low_mastery`, `partial_mastery`,
  `little_evidence`, `not_practised_for`, `weak_prerequisite`, `prerequisite_of`,
  `in_current_lesson`, `mastered`). Word them in the reader's language on your
  side; the service returns no sentences here.
- `mistakes[]` — the student's commonest mistake categories.
- `activity[]` — graded answers per week; `trend` — improving / flat / declining.
- `available: false` means no topic has enough graded work yet. `degraded: true`
  means a read failed and what is shown is incomplete.

### Progress note — `GET /ai/reports/{user_id}` and `.../latest`

`GET /ai/reports/{user_id}` builds a note (one model call) and stores it. Query:
`billed_user_id` (required: the student, or the parent asking), `subject_id`,
`lecture_ids`, `language`, and `scope` = `student_progress`
(to the student) or `parent` (to their parent). Returns `narrative`, `claims[]`
(each sentence with the evidence it cites), `weak_topics[]`, `strengths[]`,
`root_cause`, `trend`, `visual`, `report_id`, `persisted`.

`GET /ai/reports/{user_id}/latest?scope=...` returns the newest stored note with
no model call, plus `stale: true` when graded answers arrived after it was
written. Serve it, and rebuild in the background when it is stale. `404` means
no note has been built yet. A degraded note is never stored.

### Teacher — `POST /ai/classes/analytics`

Body: `{ "user_ids": [...roster], "subject_ids": [...], "class_id": "..." }`
(up to 500 students). Returns, with no model call: `topics[]` (per topic, how
many students are strong / developing / weak / unknown / not started),
`at_risk[]` (each student with `reasons`: `many_weak_topics`, `declining`,
`inactive`), `not_started[]`, `mistakes[]`, `most_missed[]` (question ids to join
with your question text) and `activity[]`.

`POST /ai/classes/report` with the same body plus a required `class_id`, a
required `billed_user_id` (the teacher) and an optional `language` writes and stores a summary for the teacher; it never names a
student. `GET /ai/classes/{class_id}/report/latest` serves the stored one.

### Question tagging — `POST /ai/topics/suggest`

Body: `{ "subject_id": "...", "lecture_ids": [...], "questions": [{ "text": "...",
"question_id": "..." }] }` (up to 200). Returns, per question, the nearest
syllabus topics with their lesson. Show them when a teacher saves a question and
store the chosen `topic_ids` and `lessonId`: an answer to an untagged question
cannot move mastery. `resolved: false` means the lookup failed, not that nothing
matched.

### Practice on weak spots

A student who asks the tutor to "practise my weak spots" without naming a topic
now gets a quiz built from their weakest topics in the subject. Any practice quiz
also gives weak topics more of its questions.

---

## 4. Ingestion — already working

`POST /ai/ingest/lesson` with `{ "lesson_id": "uuid" }` → `202` and a job
handle. This is what `AiClientService.triggerLessonIngestion` already calls; no
change needed.

Status comes back to you as `PATCH /lessons/{id}/ingestion` with
`ingestion_status` from `content.ingestion_status`, reported when a run starts
and when it ends.

**One open question:** our pipeline has a `needs_review` state — some pages
unreadable, the rest indexed fine. Your enum has no equivalent, so it currently
reports `ready`, since what was indexed is usable and `failed` would keep a
mostly-good lesson offline. Worth deciding whether you want the extra value.

---

## 5. Credits

Every call that pays a model is charged to one account before any model runs,
and refunded if the call fails, times out, or comes back degraded (a grade that
fell back to string matching, an unreadable homework photo).

| Call | Charged to | Price (`config.yaml`, `credits.prices`) |
|---|---|---|
| `POST /ai/chat`, `/ai/chat/stream` | `user_id` | 1 per message |
| `POST /ai/grade` | `billed_user_id` | 1 per 10 answers |
| `GET /ai/reports/{user_id}` | `billed_user_id` | 1 |
| `POST /ai/classes/report` | `billed_user_id` | 2 |
| `POST /ai/homework/capture` | `billed_user_id` (query) | 1 |

`billed_user_id` is required on those four; send the account of whoever asked.
Paid responses carry `usage` (what the model gateway billed), the same object as
chat's.

When the account cannot cover the call, nothing runs and the answer is `402`:

```json
{ "error": { "code": "insufficient_credits", "message": "...",
             "details": { "required": 1, "balance": 0, "renews_at": "2026-10-29T10:00:00+00:00" } } }
```

`/ai/chat/stream` refuses with a real `402` before the stream opens, never as an
SSE error event.

Each user has a wallet: a monthly allowance from their plan, spent first and
reset when the period ends, plus extra credits from grants, which never expire.
A user seen for the first time gets the default plan. Plans live in
`credits.plans`.

| Endpoint | Does |
|---|---|
| `GET /ai/credits/plans` | plans, the default plan, and the price of every action |
| `GET /ai/credits/{user_id}` | `balance`, `allowance {remaining, total}`, `extra`, `plan`, `renews_at` |
| `GET /ai/credits/{user_id}/ledger?page=&page_size=` | every change, newest first; `reference_id` is the `request_id` that caused it |
| `POST /ai/credits/{user_id}/grants` | `{ "amount": 100, "granted_by": "uuid", "note": "..." }` |
| `PUT /ai/credits/{user_id}/plan` | `{ "plan": "basic", "changed_by": "uuid" }`; the allowance restarts at the new plan |

**Credits are off** (`credits.enabled: false`): nothing is charged and the
credits endpoints answer `503`. They go on once these two tables exist, which
belong in your Prisma schema like every other `ai.*` table:

`ai.credit_wallets` — one row per user
`user_id uuid PK → identity.users`, `plan varchar(50)`, `allowance int CHECK >= 0`,
`extra int CHECK >= 0`, `period_started_at timestamptz`, `period_ends_at timestamptz`,
`updated_at timestamptz default now()`.

`ai.credit_ledger` — append-only
`id uuid PK default gen_random_uuid()`, `user_id uuid → identity.users`,
`kind varchar(30)` (`allowance_renewal | grant | usage | refund`), `action varchar(50)`,
`allowance_delta int`, `extra_delta int`, `balance_after int`,
`reverses_id uuid UNIQUE → ai.credit_ledger` (one refund per charge),
`reference_id varchar(64)`, `actor_id uuid`, `note varchar(500)`,
`cost_usd numeric(12,6)`, `created_at timestamptz default now()`,
index `(user_id, created_at DESC)`.

If this service cannot reach them while credits are on, paid calls answer `503`
rather than run free.

---

## Errors

Every failure returns the same envelope:

```json
{ "error": { "code": "validation_error", "message": "...", "request_id": "..." } }
```

| Status | `code` | Meaning |
|---|---|---|
| `401` | `unauthorized` | missing or wrong `x-service-token` |
| `403` | `forbidden` | that role may not do that task |
| `404` | `not_found` | unknown job |
| `402` | `insufficient_credits` | the paying account cannot cover the call; see Credits |
| `409` | `conflict` | that lecture is already being ingested |
| `422` | `validation_error` | bad body; `details.fields` names them |
| `504` | `timeout` | the turn exceeded the service's own limit |
| `503` | `service_unavailable` | out of capacity |

Every response carries `x-request-id`. **If you send one, we use yours** — worth
forwarding your own so a request can be traced across both services.

---

## Health

`GET /health` liveness, `GET /ready` readiness. Unauthenticated, no data
exposed. `/ready` returns `503` until the graph is compiled, so use it as the
deployment gate.
