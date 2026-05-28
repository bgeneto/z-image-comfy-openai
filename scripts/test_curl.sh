#!/usr/bin/env bash
set -euo pipefail

API_URL=${API_URL:-http://localhost:8000/v1/images/generations}
API_KEY=${API_KEY:-changeme-local-token}

curl -sS "$API_URL" \
  -H "Authorization: Bearer $API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"z-image-turbo-aio-fp8",
    "prompt":"Create a wide-shot image of a high-school student smiling next to a group of friends and holding their books as they walk through the hallway in warm, soft lighting.",
    "size":"1024x1024",
    "n":1,
    "response_format":"b64_json",
    "seed":42
  }' | python -m json.tool
