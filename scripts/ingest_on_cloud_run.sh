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

gc() {
  local attempt
  for attempt in 1 2 3 4 5 6 7 8 9 10; do
    if "$@"; then return 0; fi
    sleep $((attempt * 15))
  done
  return 1
}

execute() {
  local step="$1" tasks="$2" name state
  echo "$(date -u +%H:%M:%S) $step: $tasks task(s)" >&2
  name="$(gc gcloud run jobs execute "ingest-$step" --project "$project" --region "$region" \
    --tasks "$tasks" --update-env-vars "$overrides" --async --format "value(metadata.name)")"
  while true; do
    sleep 30
    state="$(gc gcloud run jobs executions describe "$name" --project "$project" \
      --region "$region" --format "csv[no-heading](status.completionTime,status.failedCount,status.succeededCount)")"
    IFS=, read -r done_at failed succeeded <<<"$state"
    if [ -n "${done_at:-}" ]; then
      if [ "${failed:-0}" != "0" ] && [ -n "${failed:-}" ]; then
        echo "$(date -u +%H:%M:%S) $step failed: $name ($failed of $tasks tasks)" >&2
        return 1
      fi
      echo "$(date -u +%H:%M:%S) $step done: $name (${succeeded:-0} tasks)" >&2
      echo "$name"
      return 0
    fi
  done
}

execute read "${READ_TASKS:-6}" >/dev/null
prepared="$(execute prepare 1)"

if [ "$preview" = "--preview" ]; then
  gcloud logging read --project "$project" --freshness 2h --limit 1 --format "value(textPayload)" \
    "resource.type=cloud_run_job AND resource.labels.job_name=ingest-prepare AND labels.\"run.googleapis.com/execution_name\"=$prepared AND textPayload:\"{\\\"lessons\\\": [\""
  exit 0
fi

execute embed "${EMBED_TASKS:-8}" >/dev/null
execute finish 1 >/dev/null
echo "$(date -u +%H:%M:%S) done: run $run_id"
