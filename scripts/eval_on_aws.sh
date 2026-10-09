#!/usr/bin/env bash
# Run an eval against the deployed service, from inside the VPC.
#
# The load balancer is internal, so nothing on a laptop can reach the running
# service — and an eval that can only be run against a local copy measures a
# local copy. This starts a one-off task on the same cluster, in the same
# subnets, from the same image CI built for the service, and points it at the
# internal load balancer.
#
# The service image rather than the ingest one, which CI also builds: the point
# is to measure what students are talking to, and the ingest image can be a
# deploy behind it without anything saying so.
#
#   scripts/eval_on_aws.sh conversations             # memory, language, greeting
#   scripts/eval_on_aws.sh conversations --repeat 3  # and how often each passes
#   scripts/eval_on_aws.sh retrieval                 # precision@1 and refusals
#   scripts/eval_on_aws.sh homework                  # reading, marking, review sweep
#
# Everything after the eval name is passed to the script unchanged. Only the
# conversation eval speaks HTTP, so only it is pointed at the load balancer;
# the others read the database and the index directly, which is its own reason
# to run them in here. The run's output is its log — --json inside the task
# would write to a filesystem that goes away with it.
set -euo pipefail

CLUSTER="${EVAL_CLUSTER:-edu-ai-ai}"
SERVICE="${EVAL_SERVICE:-edu-ai-ai-service}"
REGION="${AWS_REGION:-eu-central-1}"
LOG_GROUP="${EVAL_LOG_GROUP:-/ecs/edu-ai-ai-service}"

name="${1:-}"
case "$name" in
  conversations|retrieval|rag|answers|homework) shift;;
  *) echo "usage: $0 <conversations|retrieval|rag|answers|homework> [arguments]" >&2; exit 2;;
esac

# The service's own task definition and network, so the eval runs with the same
# database, the same secrets and the same reachability as the thing it measures.
read -r task_def subnets groups <<EOF
$(aws ecs describe-services --cluster "$CLUSTER" --services "$SERVICE" --region "$REGION" \
  --query 'services[0].[taskDefinition,
            join(`,`, networkConfiguration.awsvpcConfiguration.subnets),
            join(`,`, networkConfiguration.awsvpcConfiguration.securityGroups)]' \
  --output text)
EOF

read -r endpoint alb_group <<EOF
$(aws elbv2 describe-load-balancers --region "$REGION" --names "$SERVICE" \
  --query 'LoadBalancers[0].[DNSName, SecurityGroups[0]]' --output text)
EOF

# The service's own security group is the one the load balancer sends *to*, not
# one it accepts *from* — a task wearing only that cannot reach it, and the
# request times out rather than being refused, which is an hour to diagnose.
# Whatever the load balancer already admits on port 80 is what a client needs.
client_group=$(aws ec2 describe-security-groups --group-ids "$alb_group" --region "$REGION" \
  --query 'SecurityGroups[0].IpPermissions[?ToPort==`80`].UserIdGroupPairs[0].GroupId | [0]' \
  --output text)
if [ -n "$client_group" ] && [ "$client_group" != "None" ]; then
  groups="${groups},${client_group}"
fi

echo "eval:     ${name}"
echo "image:    ${task_def##*/}"
if [ "$name" = "conversations" ]; then
  echo "against:  http://${endpoint}"
else
  echo "against:  the database and the index directly"
fi

command=$(python3 - "$name" "$endpoint" "$@" <<'PY'
import json, sys
name, endpoint, *rest = sys.argv[1:]
argv = ["python", f"scripts/eval_{name}.py"]
if name == "conversations":
    argv += ["--url", f"http://{endpoint}/ai/chat"]
print(json.dumps(argv + rest))
PY
)
overrides=$(python3 -c 'import json,sys; print(json.dumps({"containerOverrides":[{"name":"ai-service","command":json.loads(sys.argv[1])}]}))' "$command")

task_arn=$(aws ecs run-task --cluster "$CLUSTER" --task-definition "$task_def" \
  --launch-type FARGATE --region "$REGION" \
  --network-configuration "awsvpcConfiguration={subnets=[${subnets}],securityGroups=[${groups}],assignPublicIp=DISABLED}" \
  --overrides "$overrides" --query 'tasks[0].taskArn' --output text)

task_id="${task_arn##*/}"
echo "as:       ${groups}"
echo "task:     ${task_id}"
echo

aws ecs wait tasks-running --cluster "$CLUSTER" --tasks "$task_arn" --region "$REGION" 2>/dev/null || true
aws logs tail "$LOG_GROUP" --follow --since 5m --region "$REGION" \
  --log-stream-name-prefix "ai-service/ai-service/${task_id}" &
tail_pid=$!

aws ecs wait tasks-stopped --cluster "$CLUSTER" --tasks "$task_arn" --region "$REGION"
kill "$tail_pid" 2>/dev/null || true

# The container's exit code, not the wait's: an eval that stopped is not an eval
# that passed, and a failing run reported as success is worse than no run.
code=$(aws ecs describe-tasks --cluster "$CLUSTER" --tasks "$task_arn" --region "$REGION" \
  --query 'tasks[0].containers[0].exitCode' --output text)
echo
[ "$code" = "0" ] && echo "eval finished." || echo "eval failed (exit ${code})" >&2
exit "${code:-1}"
