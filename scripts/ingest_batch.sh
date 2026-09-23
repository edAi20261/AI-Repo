#!/usr/bin/env bash
# Ingest many lessons at once, one AWS task each.
#
# A lesson is its own PDF and its own job, and the jobs do not queue behind one
# another: twenty lessons take as long as the slowest lesson, not the sum. The
# work inside a lesson is a model call per page, so the wall clock of one book
# is what it is — the throughput of a shelf is how many run at once.
#
#   scripts/ingest_batch.sh lessons.tsv [extra ingestion arguments]
#
# lessons.tsv is one lesson per line, tab separated, comments and blanks ignored:
#
#   s3://bucket/unit1.pdf<TAB>LECTURE_ID<TAB>SUBJECT_ID<TAB>Lesson title
#
# Extra arguments are passed to every job, e.g. --layout-model.
set -euo pipefail

CLUSTER="${INGEST_CLUSTER:-edu-ai-ai}"
TASK_DEF="${INGEST_TASK_DEF:-edu-ai-ingest}"
REGION="${AWS_REGION:-eu-central-1}"
SUBNETS="${INGEST_SUBNETS:-subnet-0e31fd5aae3a50f31,subnet-00e22697af9ba1ec5}"
SECURITY_GROUPS="${INGEST_SECURITY_GROUPS:-sg-0211526ec39da041c}"
# Fargate's per-account task limit and OpenRouter's rate limit are both real;
# more than this in flight buys queueing at one of them rather than throughput.
MAX_IN_FLIGHT="${INGEST_MAX_IN_FLIGHT:-10}"

manifest="${1:-}"
if [ -z "$manifest" ] || [ ! -f "$manifest" ]; then
  echo "usage: $0 <lessons.tsv> [extra ingestion arguments]" >&2
  exit 2
fi
shift
extra=("$@")

network="awsvpcConfiguration={subnets=[${SUBNETS}],securityGroups=[${SECURITY_GROUPS}],assignPublicIp=DISABLED}"
declare -a arns=() names=()

while IFS=$'\t' read -r source lecture subject title || [ -n "${source:-}" ]; do
  case "${source:-}" in ""|\#*) continue;; esac

  while [ "$(printf '%s\n' "${arns[@]:-}" | grep -c . || true)" -ge "$MAX_IN_FLIGHT" ]; do
    sleep 20
    remaining=()
    for arn in "${arns[@]}"; do
      state=$(aws ecs describe-tasks --cluster "$CLUSTER" --tasks "$arn" --region "$REGION" \
        --query 'tasks[0].lastStatus' --output text 2>/dev/null || echo STOPPED)
      [ "$state" != "STOPPED" ] && remaining+=("$arn")
    done
    arns=("${remaining[@]:-}")
  done

  command=$(python3 - "$source" "$lecture" "$subject" "${title:-}" "${extra[@]:-}" <<'PY'
import json, sys
source, lecture, subject, title, *extra = sys.argv[1:]
argv = ["-m", "ingestion", source, "--lecture-id", lecture, "--subject-id", subject]
if title:
    argv += ["--lesson-title", title]
print(json.dumps(argv + [a for a in extra if a]))
PY
)
  overrides=$(python3 -c 'import json,sys; print(json.dumps({"containerOverrides":[{"name":"ingest","command":json.loads(sys.argv[1])}]}))' "$command")

  arn=$(aws ecs run-task --cluster "$CLUSTER" --task-definition "$TASK_DEF" \
    --launch-type FARGATE --network-configuration "$network" \
    --overrides "$overrides" --region "$REGION" \
    --query 'tasks[0].taskArn' --output text)
  arns+=("$arn")
  names+=("${title:-$source}|$arn")
  echo "started  ${title:-$source}  ${arn##*/}"
done < "$manifest"

echo
echo "waiting for ${#names[@]} lesson(s)…"
failed=0
for entry in "${names[@]}"; do
  title="${entry%%|*}"; arn="${entry##*|}"
  aws ecs wait tasks-stopped --cluster "$CLUSTER" --tasks "$arn" --region "$REGION" 2>/dev/null || true
  code=$(aws ecs describe-tasks --cluster "$CLUSTER" --tasks "$arn" --region "$REGION" \
    --query 'tasks[0].containers[0].exitCode' --output text 2>/dev/null || echo "?")
  if [ "$code" = "0" ]; then
    echo "ingested  ${title}"
  else
    echo "FAILED    ${title}  (exit ${code}, task ${arn##*/})" >&2
    failed=$((failed + 1))
  fi
done

echo
echo "${#names[@]} lesson(s), ${failed} failed"
exit $(( failed > 0 ))
