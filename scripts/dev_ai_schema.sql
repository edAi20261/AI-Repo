-- Local/dev bootstrap for the five ai.* tables ingestion writes
-- (chunks, figures, vector_index, topics, topic_dependencies). Production owns
-- the schema; ingestion never runs DDL. Idempotent.
--
-- Two dev-only deviations from the production schema:
--   1. vector_index.grade is varchar (no cross-schema enum dependency).
--   2. vector_index gets UNIQUE(chunk_id, namespace) — the manifest upsert's
--      ON CONFLICT target; the production schema needs it too.
--
-- Apply:  psql "$DATABASE_URL" -f scripts/dev_ai_schema.sql
--    or:  python scripts/bootstrap.py --postgres

CREATE EXTENSION IF NOT EXISTS pgcrypto;   -- gen_random_uuid()

CREATE SCHEMA IF NOT EXISTS ai;

-- dependency_strength is AI-owned (we define the values). CREATE TYPE has no
-- IF NOT EXISTS, so guard it.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_type t
        JOIN pg_namespace n ON n.oid = t.typnamespace
        WHERE t.typname = 'dependency_strength' AND n.nspname = 'ai'
    ) THEN
        CREATE TYPE ai.dependency_strength AS ENUM ('required', 'related');
    END IF;
END$$;

CREATE TABLE IF NOT EXISTS ai.topics (
    topic_id        uuid NOT NULL DEFAULT gen_random_uuid(),
    label           varchar(255) NOT NULL,
    description     text,
    lecture_id      uuid,
    subject_id      uuid NOT NULL,
    bloom_level     varchar(30),
    label_embedding jsonb,
    created_at      timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT topics_pkey PRIMARY KEY (topic_id)
);

CREATE TABLE IF NOT EXISTS ai.topic_dependencies (
    topic_id        uuid NOT NULL,
    prereq_topic_id uuid NOT NULL,
    strength        ai.dependency_strength NOT NULL DEFAULT 'required',
    rationale       text,
    CONSTRAINT topic_dependencies_pkey PRIMARY KEY (topic_id, prereq_topic_id)
);

CREATE TABLE IF NOT EXISTS ai.chunks (
    chunk_id     uuid NOT NULL DEFAULT gen_random_uuid(),
    lecture_id   uuid NOT NULL,
    topic_ids    uuid[],
    text         text NOT NULL,
    heading_path varchar(500),
    page_from    integer,
    page_to      integer,
    language     varchar(20),
    order_index  integer NOT NULL DEFAULT 0,
    version      integer NOT NULL DEFAULT 1,
    created_at   timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT chunks_pkey PRIMARY KEY (chunk_id)
);

CREATE TABLE IF NOT EXISTS ai.figures (
    figure_id  uuid NOT NULL DEFAULT gen_random_uuid(),
    lecture_id uuid NOT NULL,
    chunk_id   uuid,
    topic_ids  uuid[],
    s3_uri     varchar(1024) NOT NULL,
    caption    text,
    alt_text   text,
    page       integer,
    CONSTRAINT figures_pkey PRIMARY KEY (figure_id)
);

CREATE TABLE IF NOT EXISTS ai.vector_index (
    id         uuid NOT NULL DEFAULT gen_random_uuid(),
    chunk_id   uuid NOT NULL,
    lecture_id uuid NOT NULL,
    topic_ids  uuid[],
    subject_id uuid NOT NULL,
    grade      varchar(30),          -- prod: academics.grade_level (see header)
    namespace  varchar(150) NOT NULL,
    language   varchar(20),
    version    integer NOT NULL DEFAULT 1,
    CONSTRAINT vector_index_pkey PRIMARY KEY (id),
    CONSTRAINT vector_index_chunk_namespace_uniq UNIQUE (chunk_id, namespace)
);

-- One row per lecture: the current ingestion state (status, counts, when).
-- Upserted on every ingest so re-ingestion overwrites rather than accretes.
CREATE TABLE IF NOT EXISTS ai.lecture_ingestion (
    lecture_id          uuid NOT NULL,
    subject_id          uuid,
    version             integer NOT NULL DEFAULT 1,
    status              varchar(30) NOT NULL,   -- ready | needs_review
    chunk_count         integer NOT NULL DEFAULT 0,
    figure_count        integer NOT NULL DEFAULT 0,
    topic_count         integer NOT NULL DEFAULT 0,
    needs_review_reason text,
    ingested_at         timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT lecture_ingestion_pkey PRIMARY KEY (lecture_id)
);

-- Read-path helper indexes (retrieval fetches chunks by topic / lecture).
CREATE INDEX IF NOT EXISTS chunks_lecture_idx ON ai.chunks (lecture_id);
CREATE INDEX IF NOT EXISTS chunks_topic_ids_idx ON ai.chunks USING gin (topic_ids);
CREATE INDEX IF NOT EXISTS topics_subject_idx ON ai.topics (subject_id);
CREATE INDEX IF NOT EXISTS vector_index_namespace_idx ON ai.vector_index (namespace);

-- ---------------------------------------------------------------------------
-- Mastery (agent-owned). Additive: CREATE ... IF NOT EXISTS only,
-- so applying this file never touches ingestion data. mastery_events is
-- append-only; mastery is the recency-weighted rollup (see agent/mastery.py).
-- ---------------------------------------------------------------------------

-- mastery_event_source is AI-owned. CREATE TYPE has no IF NOT EXISTS, so guard it.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_type t
        JOIN pg_namespace n ON n.oid = t.typnamespace
        WHERE t.typname = 'mastery_event_source' AND n.nspname = 'ai'
    ) THEN
        CREATE TYPE ai.mastery_event_source AS ENUM
            ('practice', 'quiz', 'homework', 'uploaded');
    END IF;
END$$;

CREATE TABLE IF NOT EXISTS ai.mastery (
    user_id    uuid NOT NULL,
    topic_id   uuid NOT NULL,
    score      double precision NOT NULL DEFAULT 0,
    attempts   integer NOT NULL DEFAULT 0,
    last_seen  timestamptz,
    trend      double precision,
    confidence double precision,
    CONSTRAINT mastery_pkey PRIMARY KEY (user_id, topic_id)
);

CREATE TABLE IF NOT EXISTS ai.mastery_events (
    event_id        uuid NOT NULL DEFAULT gen_random_uuid(),
    user_id         uuid NOT NULL,
    item_id         uuid,                 -- null when the quiz was generated live in chat
    topic_id        uuid NOT NULL,
    lecture_id      uuid,
    correct         boolean,
    partial_score   double precision,
    original_score  double precision,     -- retained on a teacher-correction event
    difficulty      varchar(20),
    bloom_level     varchar(30),
    mistake_tag_ids uuid[],
    source          ai.mastery_event_source NOT NULL,
    created_at      timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT mastery_events_pkey PRIMARY KEY (event_id)
);

CREATE TABLE IF NOT EXISTS ai.mistake_tags (
    tag_id      uuid NOT NULL DEFAULT gen_random_uuid(),
    label       varchar(150) NOT NULL,
    description text,
    created_by  varchar(100),
    created_at  timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT mistake_tags_pkey PRIMARY KEY (tag_id)
);

-- Read paths: rollup lookup by (user, topic); trend query orders events by recency.
CREATE INDEX IF NOT EXISTS mastery_events_user_topic_idx
    ON ai.mastery_events (user_id, topic_id);
CREATE INDEX IF NOT EXISTS mastery_events_created_idx
    ON ai.mastery_events (created_at);
