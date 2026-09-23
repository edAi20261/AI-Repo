#!/usr/bin/env bash
# Ingest a book on AWS, not on the machine you are sitting at.
#
# The layout models are gigabytes and a book is hundreds of pages: reading one
# locally takes a laptop out of service for as long as it runs. This starts the
# ingestion task instead, streams its log, and returns when the book is in.
#
#   scripts/ingest_on_aws.sh s3://edu-ai-media-649631967545/books/unit3.pdf \
#       --lecture-id 8e6366c8-... --subject-id b9ec8535-... --layout-model
#
# Everything after the source is passed to `python -m ingestion` unchanged, so
# --lesson-title, --version and --no-layout-model all work as they do locally.
set -euo pipefail

CLUSTER="${INGEST_CLUSTER:-edu-ai-ai}"
TASK_DEF="${INGEST_TASK_DEF:-edu-ai-ingest}"
REGION="${AWS_REGION:-eu-central-1}"
SUBNETS="${INGEST_SUBNETS:-subnet-0e31fd5aae3a50f31,subnet-00e22697af9ba1ec5}"
SECURITY_GROUPS="${INGEST_SECURITY_GROUPS:-sg-0211526ec39da041c}"
LOG_GROUP="/ecs/edu-ai-ingest"

if [ $# -lt 1 ]; then
  echo "usage: $0 <s3://… | https://…> [ingestion arguments]" >&2
  exit 2
fi

command=$(python3 - "$@" <<'PY'
import json, sys
print(json.dumps(["-m", "ingestion", *sys.argv[1:]]))
PY
)

overrides=$(python3 - "$command" <<'PY'
import json, sys
print(json.dumps({"containerOverrides": [{"name": "ingest", "command": json.loads(sys.argv[1])}]}))
PY
)

network="awsvpcConfiguration={subnets=[${SUBNETS}],securityGroups=[${SECURITY_GROUPS}],assignPublicIp=DISABLED}"

echo "starting ${TASK_DEF} on ${CLUSTER}…"
task_arn=$(aws ecs run-task \
  --cluster "$CLUSTER" \
  --task-definition "$TASK_DEF" \
  --launch-type FARGATE \
  --network-configuration "$network" \
  --overrides "$overrides" \
  --region "$REGION" \
  --query 'tasks[0].taskArn' --output text)

task_id="${task_arn##*/}"
echo "task ${task_id}"
echo "logs: aws logs tail ${LOG_GROUP} --follow --region ${REGION}"

aws ecs wait tasks-running --cluster "$CLUSTER" --tasks "$task_arn" --region "$REGION" 2>/dev/null || true
aws logs tail "$LOG_GROUP" --follow --since 5m --region "$REGION" \
  --log-stream-name-prefix "ingest/ingest/${task_id}" &
tail_pid=$!

aws ecs wait tasks-stopped --cluster "$CLUSTER" --tasks "$task_arn" --region "$REGION"
kill "$tail_pid" 2>/dev/null || true

# The exit code of the container, not of the wait: a task that stopped is not a
# book that was ingested, and a run whose failure reads as success is worse
# than no run at all.
code=$(aws ecs describe-tasks --cluster "$CLUSTER" --tasks "$task_arn" --region "$REGION" \
  --query 'tasks[0].containers[0].exitCode' --output text)
reason=$(aws ecs describe-tasks --cluster "$CLUSTER" --tasks "$task_arn" --region "$REGION" \
  --query 'tasks[0].stoppedReason' --output text)

if [ "$code" = "0" ]; then
  echo "ingested."
else
  echo "ingestion failed (exit ${code}): ${reason}" >&2
fi
exit "${code:-1}"
