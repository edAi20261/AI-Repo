# Schema changes the memory layer needs

**These are requests to the backend, not work this repo can do.**

`ai.user_memory` and `ai.adaptive_profile` are declared in
`backend/prisma/schema.prisma` and owned by the backend's Prisma migrations.
Applying raw SQL to them here would work until the next `prisma migrate`, which
compares the database against `schema.prisma` and generates a migration to undo
anything it does not know about. A change that works in testing and is silently
reverted weeks later is worse than a change that was never made, so each item
below is written as the Prisma edit plus the SQL it produces.

None of the three is required for correctness today. All three are required
before this holds a thousand students.

Measured against production on 2026-09-26: `ai.user_memory` holds 119 rows
across 16 conversations for 1 student, in 256 kB. Every number below is what
that becomes at scale, not what it costs now.

---

## 1. The index the hot read actually uses

Every turn runs one query filtered by `(user_id, source_conversation_id, kind,
expires_at)`. The only index is `idx_user_memory_user` on `user_id` alone, so
the filter narrows to one student and then reads every row that student has
ever written, in every conversation they have ever had, to find the dozen
belonging to this one.

At 119 rows that is free. At a student with a term of chat history it is the
whole term, on every turn, inside a 500 ms statement timeout — and when that
timeout is hit the conversation reads as new. So the symptom is not an error
anybody sees. It is a tutor that forgets.

```prisma
model UserMemory {
  // ...
  @@index([userId], map: "idx_user_memory_user")
  @@index([userId, sourceConversationId, kind, createdAt], map: "idx_user_memory_conversation")
}
```

```sql
CREATE INDEX "idx_user_memory_conversation"
    ON "ai"."user_memory" ("user_id", "source_conversation_id", "kind", "created_at");
```

On a live database this is worth running as `CREATE INDEX CONCURRENTLY` first
and letting Prisma adopt it, so the build takes no exclusive lock.

## 2. Somewhere for expired rows to go

`expires_at` is set on every row written (720 hours by default) and honoured on
read, so an expired row is already invisible. Nothing deletes it. Five expired
rows are in the table today. At a thousand students writing twenty turns a day,
thirty days of expired memory is six hundred thousand rows nobody can read and
everybody pays to skip.

```prisma
model UserMemory {
  // ...
  @@index([expiresAt], map: "idx_user_memory_expires")
}
```

Then, on whatever runs the platform's housekeeping — bounded per run so it can
never hold a lock long enough to matter:

```sql
DELETE FROM "ai"."user_memory"
WHERE "memory_id" IN (
    SELECT "memory_id" FROM "ai"."user_memory"
    WHERE "expires_at" IS NOT NULL AND "expires_at" < now()
    LIMIT 10000
);
```

## 3. Let a student say no

`prefers_examples` and `prefers_visual` are `Boolean @default(false)`, so false
means both "never mentioned it" and "asked not to have them". A student who
says *please stop giving me examples* is stored correctly and read back as a
student who never said anything, so the tutor carries on giving them examples.

`agent/profile.py` logs a warning every time this happens
(`_warn_about_unstorable`), which is the only reason anyone would know.

```prisma
model AdaptiveProfile {
  prefersVisual     Boolean? @map("prefers_visual")
  prefersExamples   Boolean? @map("prefers_examples")
  // ...
}
```

```sql
ALTER TABLE "ai"."adaptive_profile" ALTER COLUMN "prefers_examples" DROP NOT NULL;
ALTER TABLE "ai"."adaptive_profile" ALTER COLUMN "prefers_visual"   DROP NOT NULL;
ALTER TABLE "ai"."adaptive_profile" ALTER COLUMN "prefers_examples" DROP DEFAULT;
ALTER TABLE "ai"."adaptive_profile" ALTER COLUMN "prefers_visual"   DROP DEFAULT;
```

Existing rows keep their `false`. Any reader that treats the column as a plain
boolean has to handle null — today nothing outside this repo reads it, and the
table has no rows.

Once it lands, `agent/profile.py` should stop coalescing the two columns to
false on insert and stop reading false as "never said": `_warn_about_unstorable`
and the `examples=True if examples else None` branch in `_stated_from_row` both
exist only because of this, and both should go.


---

## 4. Let this service read the transcript (urgent, and not a schema change)

`agent/conversation/transcript.py` recovers a conversation older than our own
memory by reading the platform's messages. In production it cannot:

```
InsufficientPrivilege: permission denied for schema engagement
```

The role this service connects as is not granted the `engagement` schema, so a
student reopening an old thread starts cold. The code degrades to that and logs
it, and after the first refusal it stops asking — a permission error is not
transient, and retrying it cost 477 ms on the turn that logged it.

What it needs, on the role this service uses:

```sql
GRANT USAGE ON SCHEMA engagement TO <ai_service_role>;
GRANT SELECT ON engagement.ai_chat_messages, engagement.ai_chat_sessions
    TO <ai_service_role>;
```

Read-only, two tables. Nothing here writes to that schema.

---

## 5. Two ingestion paths, not one (fixed, recorded here so it is not re-learned)

`python -m ingestion` as an ECS task is not the only way a lesson is ingested.
`system/ingestion_jobs.py` imports `run_ingestion` and calls it **inside the
service process**, which is what happens when a teacher publishes a lesson.

So a credential the ingestion needs has to reach *both* the ingest task and the
service task. When vision and topic extraction moved to Vertex, only the ingest
task got it: homework capture returned 500, and a published lesson would have
had every page read as blank and then failed at the topics stage.

The ECS execution role is the other half of it. `edu-ai-ecs-task-execution-ssm`
lists its parameters one by one rather than by prefix, so a new secret is
unreadable until it is added there too — and the task fails to start with
`ResourceInitializationError`, before any application log exists to explain it.

Both are done. The checklist for the next secret:

1. `ssm put-parameter` under `/edu-ai/ai-service/`
2. add the ARN to the `edu-ai-ecs-task-execution-ssm` role policy
3. add it to `infra/ingest-task-definition.json` **and** the service task definition
4. deploy the service onto the new revision — registering it is not deploying it
