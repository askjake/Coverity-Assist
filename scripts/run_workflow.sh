#!/usr/bin/env bash
set -euo pipefail

GW="${1:-http://127.0.0.1:5000}"
CHAT_URL="${2:-http://coverity-assist.dishtv.technology/chat}"
TOKEN="${3:-REDACTED_BEARER}"
PROFILE="${4:-arn:aws:bedrock:us-west-2:233532778289:application-inference-profile/xchtiicwcai3}"

curl -sS -X POST "${GW}/trigger-workflow" \
  -H "Content-Type: application/json" \
  -d @- <<JSON
{
  "original_request": "CLI requested workflow run",
  "task_description": "Kick off the default pipeline.",
  "chat_url": "${CHAT_URL}",
  "token": "${TOKEN}",
  "inference_profile_arn": "${PROFILE}"
}
JSON
echo
