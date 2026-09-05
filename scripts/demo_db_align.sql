-- Align a partial ai.*-only dev DB with what the agent expects, so the demo runs as if the
-- full backend/DB were present. Non-destructive (IF NOT EXISTS). Safe to re-run.
--   - ai.artifacts: the render index (kind stored as text here to avoid the ai.artifact_kind
--     enum dependency; the real backend uses the enum).
--   - ai.mistake_tags.subject_id: a pending schema add the agent already writes.

CREATE TABLE IF NOT EXISTS ai.artifacts (
    artifact_id     uuid NOT NULL DEFAULT gen_random_uuid(),
    conversation_id uuid,
    user_id         uuid NOT NULL,
    kind            varchar(50) NOT NULL,
    s3_uri          varchar(1024) NOT NULL,
    created_at      timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT artifacts_pkey PRIMARY KEY (artifact_id)
);

ALTER TABLE ai.mistake_tags ADD COLUMN IF NOT EXISTS subject_id uuid;
