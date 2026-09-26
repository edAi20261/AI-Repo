-- Schema changes the memory layer needs. NOT APPLIED BY THIS REPO.
--
-- Production owns the schema and this service never runs DDL (same rule as
-- scripts/dev_ai_schema.sql). Everything here is written to be applied by hand,
-- by whoever owns the production database, after reading it. Each statement is
-- idempotent and each says what happens if it is never run — none of them is
-- required for correctness today, and all three are required before this holds
-- a thousand students.
--
-- Apply:  psql "$DATABASE_URL" -f scripts/memory_schema_changes.sql
--
-- Measured on 2026-09-26 against production: ai.user_memory holds 119 rows
-- across 16 conversations for 1 student, in 256 kB. Every number below is what
-- that becomes at scale, not what it costs now.

-- ---------------------------------------------------------------------------
-- 1. The index the hot read actually uses.
--
-- Every turn runs one query filtered by (user_id, source_conversation_id, kind,
-- expires_at). The only index is idx_user_memory_user on user_id alone, so the
-- filter narrows to one student and then reads every row that student owns, in
-- every conversation they have ever had, to find the dozen belonging to this
-- one. At 119 rows that is free. At a student with a term of chat history it is
-- the whole term, on every turn, inside a 500 ms statement timeout — and when
-- that timeout is hit the conversation reads as new, so the failure is a tutor
-- that forgets rather than an error anybody sees.
--
-- CONCURRENTLY: no exclusive lock, so it can be run against a live database.
-- It cannot run inside a transaction block; run this statement on its own.
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_user_memory_conversation
    ON ai.user_memory (user_id, source_conversation_id, kind, created_at);

-- ---------------------------------------------------------------------------
-- 2. Somewhere for expired rows to go.
--
-- expires_at is set on every row written (ttl_hours, 720 by default) and is
-- honoured on read, so an expired row is already invisible. Nothing deletes it.
-- Five expired rows are in the table today. At a thousand students writing
-- twenty turns a day, thirty days of expired memory is six hundred thousand
-- rows nobody can read and everybody pays to skip.
--
-- The index is what makes the delete cheap; without it this is a sequential
-- scan of the whole table and should not be run on a schedule.
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_user_memory_expires
    ON ai.user_memory (expires_at) WHERE expires_at IS NOT NULL;

-- Then, on a schedule (pg_cron, or whatever runs the platform's housekeeping).
-- Bounded per run so it can never hold a lock long enough to matter:
--
--   DELETE FROM ai.user_memory
--   WHERE memory_id IN (
--       SELECT memory_id FROM ai.user_memory
--       WHERE expires_at IS NOT NULL AND expires_at < now()
--       LIMIT 10000
--   );

-- ---------------------------------------------------------------------------
-- 3. Let a student say no.
--
-- ai.adaptive_profile.prefers_examples and prefers_visual are NOT NULL DEFAULT
-- false, so false means both "never mentioned it" and "asked not to have them".
-- A student who says "please stop giving me examples" is stored correctly and
-- read back as a student who never said anything, so the tutor keeps giving
-- them examples. agent/profile.py logs a warning every time this happens, which
-- is the only reason anyone would know.
--
-- Dropping NOT NULL only widens what the column accepts: existing rows keep
-- their false, and nothing that writes today can break. What must change with
-- it is any reader that treats the column as a plain boolean.
ALTER TABLE ai.adaptive_profile ALTER COLUMN prefers_examples DROP NOT NULL;
ALTER TABLE ai.adaptive_profile ALTER COLUMN prefers_visual   DROP NOT NULL;
ALTER TABLE ai.adaptive_profile ALTER COLUMN prefers_examples DROP DEFAULT;
ALTER TABLE ai.adaptive_profile ALTER COLUMN prefers_visual   DROP DEFAULT;

-- After this, agent/profile.py should stop coalescing the two columns to false
-- on insert and stop reading false as "never said": see _warn_about_unstorable
-- and _stated_from_row, both of which exist only because of this.
