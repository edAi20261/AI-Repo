#!/usr/bin/env bash
# Run scripts/e2e_features.py from inside the VPC, where the AI service's
# internal load balancer is reachable, as a one-off task on the service's own
# image. The running service is not touched.
#
#   scripts/e2e_in_vpc.sh direct        e2e_students.json
#   scripts/e2e_in_vpc.sh stress-direct e2e_students.json --students 30
#   scripts/e2e_in_vpc.sh cleanup       e2e_students.json
#
# The task runs with the backend's VPC-connector security group, the only one
# the load balancer admits. A container command override is capped at 8 KB, so
# the script travels through S3 and is removed again afterwards.
set -euo pipefail

MODE=${1:?mode: direct | stress-direct | cleanup}
USERS=${2:?path to the students file written by the backend mode}
shift 2

REGION=${AWS_REGION:-eu-central-1}
CLUSTER=edu-ai-ai
TASK_DEF=${E2E_TASK_DEF:-edu-ai-ai-service}
BUCKET=${E2E_BUCKET:-edu-ai-media-649631967545}
PREFIX=e2e-tests/$(date +%s)
AI_URL=${E2E_AI:-http://internal-edu-ai-ai-service-251565707.eu-central-1.elb.amazonaws.com}
SUBNETS=${E2E_SUBNETS:-subnet-0e31fd5aae3a50f31,subnet-00e22697af9ba1ec5}
SECURITY_GROUP=${E2E_SECURITY_GROUP:-sg-02f9099b3fe1c7b36}
export AWS_PAGER=""

cleanup() { aws s3 rm "s3://$BUCKET/$PREFIX/" --recursive --region "$REGION" --only-show-errors; }
trap cleanup EXIT

aws s3 cp "$(dirname "$0")/e2e_features.py" "s3://$BUCKET/$PREFIX/e2e.py" --region "$REGION" --only-show-errors
aws s3 cp "$USERS" "s3://$BUCKET/$PREFIX/users.json" --region "$REGION" --only-show-errors
SCRIPT_URL=$(aws s3 presign "s3://$BUCKET/$PREFIX/e2e.py" --expires-in 3600 --region "$REGION")
USERS_URL=$(aws s3 presign "s3://$BUCKET/$PREFIX/users.json" --expires-in 3600 --region "$REGION")

FETCH="import httpx,sys;open('/tmp/e.py','w').write(httpx.get(sys.argv[1]).text);open('/tmp/u.json','w').write(httpx.get(sys.argv[2]).text)"
COMMAND="python -c \"$FETCH\" '$SCRIPT_URL' '$USERS_URL' && python /tmp/e.py $MODE --ai $AI_URL --users-file /tmp/u.json $*"
OVERRIDES=$(python3 -c 'import json,sys; print(json.dumps({"containerOverrides":[{"name":"ai-service","command":["sh","-c",sys.argv[1]]}]}))' "$COMMAND")

TASK=$(aws ecs run-task --cluster "$CLUSTER" --task-definition "$TASK_DEF" --launch-type FARGATE \
  --region "$REGION" --started-by e2e-feature-tests --overrides "$OVERRIDES" \
  --network-configuration "awsvpcConfiguration={subnets=[$SUBNETS],securityGroups=[$SECURITY_GROUP],assignPublicIp=DISABLED}" \
  --query 'tasks[0].taskArn' --output text)
echo "task: $TASK"
aws ecs wait tasks-stopped --cluster "$CLUSTER" --tasks "$TASK" --region "$REGION"

aws logs get-log-events --log-group-name /ecs/edu-ai-ai-service --region "$REGION" \
  --log-stream-name "ai-service/ai-service/${TASK##*/}" --query 'events[].message' --output text \
  | tr '\t' '\n' | grep -v '^{"ts"'
