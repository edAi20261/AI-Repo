#!/usr/bin/env bash
# Ingest one book on Cloud Run: read, prepare, embed, finish, each as a job
# execution that waits for the one before. The jobs come from
# infra/cloud-run/setup.sh.
#
#   scripts/ingest_on_cloud_run.sh SOURCE SUBJECT_ID [--preview]
#
# --preview stops after prepare and prints the lessons the book was split into;
# nothing is written. A run reuses every page an earlier run already read.
set -euo pipefail

source_uri="${1:?usage: ingest_on_cloud_run.sh SOURCE SUBJECT_ID [--preview]}"
subject="${2:?usage: ingest_on_cloud_run.sh SOURCE SUBJECT_ID [--preview]}"
preview="${3:-}"
project="${PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
region="${REGION:-europe-west3}"
run_id="$(date -u +%Y%m%dT%H%M%SZ)-$RANDOM"
overrides="^@^SOURCE=$source_uri@SUBJECT_ID=$subject@RUN_DIR=/work/runs/$run_id"

execute() {
  local step="$1" tasks="$2"
  echo "$(date -u +%H:%M:%S) $step: $tasks task(s)"
  gcloud run jobs execute "ingest-$step" --project "$project" --region "$region" \
    --tasks "$tasks" --update-env-vars "$overrides" --wait \
    --format "value(metadata.name)"
}

execute read "${READ_TASKS:-6}"
prepared="$(execute prepare 1 | tail -1)"

if [ "$preview" = "--preview" ]; then
  gcloud logging read --project "$project" --freshness 2h --limit 1 --format "value(textPayload)" \
    "resource.type=cloud_run_job AND resource.labels.job_name=ingest-prepare AND labels.\"run.googleapis.com/execution_name\"=$prepared AND textPayload:\"{\\\"lessons\\\": [\""
  exit 0
fi

execute embed "${EMBED_TASKS:-8}"
execute finish 1
echo "$(date -u +%H:%M:%S) done: run $run_id"
