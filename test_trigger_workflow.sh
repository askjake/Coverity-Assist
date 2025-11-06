
#!/usr/bin/env bash
set -euo pipefail
curl -sS -X POST "http://127.0.0.1:5000/trigger-workflow" \
  -H "Content-Type: application/json" \
  -d '{
        "original_request": "Build a quick status report for STB health",
        "task_description": "Collect system info (uname -a), list docker images (docker images), and fetch https://example.com. Output CSV lines.",
        "inference_profile_arn": ""
      }' | jq .
