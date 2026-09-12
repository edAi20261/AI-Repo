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
| `session_id` | no | conversation id |
| `grade` | no | academic year, e.g. `g8` |
| `language` | no | overrides the language detected from the message |

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
  "prereq_topics": [{ "label": "Oxygen", "strength": "required", "depth": 1 }]
}
```

For a normal turn, pass `response` and `visuals` through to the frontend. The
rest is populated only when the planner produced it — `assessment` when the
student asked for a quiz, `grade_result` when it graded something.

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

Returns `graded[]`, `summary`, `mastery_events`, `topics_updated`, `source`.
Grading is the only thing that moves a student's mastery.

---

## 3. Progress report

`GET /ai/reports/{user_id}` — optional query: `subject_id`, `lecture_ids`, `language`.

Returns `narrative`, `weak_topics[]`, `root_cause`, `trend`, and `visual` (a
rendered trend chart). `available: false` means there isn't enough graded work
yet — show that as a message, not an error.

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
