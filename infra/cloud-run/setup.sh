#!/usr/bin/env bash
# Everything a book's ingestion needs on Cloud Run, created or brought up to
# date. Safe to run again: every step updates what is already there.
#
#   infra/cloud-run/setup.sh path/to/.env
#
# The .env supplies the database, S3 and Qdrant values, which go into Secret
# Manager. Nothing is printed. Gemini needs no secret: the jobs run as
# SERVICE_ACCOUNT, and Vertex is billed to this project.
#
#   PROJECT          default: gcloud's configured project
#   REGION           default: europe-west3, beside the database in Frankfurt
#   SERVICE_ACCOUNT  default: ingestion-deployer@PROJECT.iam.gserviceaccount.com
#   IMAGE_TAG        default: the current commit
set -euo pipefail

env_file="${1:?usage: setup.sh path/to/.env}"
project="${PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
region="${REGION:-europe-west3}"
account="${SERVICE_ACCOUNT:-ingestion-deployer@$project.iam.gserviceaccount.com}"
tag="${IMAGE_TAG:-$(git rev-parse --short HEAD)}"
image="$region-docker.pkg.dev/$project/ingest/ingest:$tag"
bucket="$project-ingest-work"
secrets=(DATABASE_URL QDRANT_URL QDRANT_API_KEY S3_BUCKET S3_REGION AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY)

gcloud config set project "$project" >/dev/null

if ! gcloud artifacts repositories describe ingest --location "$region" >/dev/null 2>&1; then
  gcloud artifacts repositories create ingest --repository-format docker --location "$region"
fi

if ! gcloud storage buckets describe "gs://$bucket" >/dev/null 2>&1; then
  gcloud storage buckets create "gs://$bucket" --location "$region" --uniform-bucket-level-access
  printf '{"rule":[{"action":{"type":"Delete"},"condition":{"age":7}}]}' > /tmp/ingest-lifecycle.json
  gcloud storage buckets update "gs://$bucket" --lifecycle-file /tmp/ingest-lifecycle.json
fi

secret_flags=()
for name in "${secrets[@]}"; do
  value="$(grep -E "^$name=" "$env_file" | head -1 | cut -d= -f2- | sed -E 's/^["'\'']//; s/["'\'']$//')"
  [ -n "$value" ] || { echo "$name is missing from $env_file" >&2; exit 1; }
  secret="ingest-$(echo "$name" | tr '[:upper:]_' '[:lower:]-')"
  if ! gcloud secrets describe "$secret" >/dev/null 2>&1; then
    gcloud secrets create "$secret" --replication-policy automatic >/dev/null
  fi
  printf '%s' "$value" | gcloud secrets versions add "$secret" --data-file=- >/dev/null
  secret_flags+=("$name=$secret:latest")
done
secret_list="$(IFS=,; echo "${secret_flags[*]}")"

gcloud builds submit --config infra/cloud-run/cloudbuild.yaml \
  --substitutions "_IMAGE=$image" .

deploy() {
  local step="$1" tasks="$2" cpu="$3" memory="$4" timeout="$5"
  gcloud run jobs deploy "ingest-$step" --region "$region" --image "$image" \
    --service-account "$account" \
    --command bash --args "scripts/cloud_run_step.sh,$step" \
    --tasks "$tasks" --parallelism "$tasks" --max-retries 0 \
    --cpu "$cpu" --memory "$memory" --task-timeout "$timeout" \
    --set-secrets "$secret_list" \
    --set-env-vars "APP_ENV=production,PYTHONUNBUFFERED=1,TOKENIZERS_PARALLELISM=false,RUN_DIR=/work/runs/unset" \
    --add-volume "name=work,type=cloud-storage,bucket=$bucket" \
    --add-volume-mount "volume=work,mount-path=/work" \
    --execution-environment gen2 --quiet
}

deploy read 6 2 4Gi 3600s
deploy prepare 1 4 16Gi 3600s
deploy embed 8 2 8Gi 3600s
deploy finish 1 2 8Gi 7200s

echo "ready: $image in $region, work bucket gs://$bucket"
