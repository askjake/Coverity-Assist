#!/usr/bin/env python3
import os, json, sys, requests

COVERITY_ASSIST_URL = os.environ.get("COVERITY_ASSIST_URL", "http://coverity-assist.dishtv.technology/chat").rstrip("/")
TOKEN = os.environ.get("COVERITY_ASSIST_TOKEN", "")
INFERENCE_PROFILE_ARN = os.environ.get("BEDROCK_APPLICATION_INFERENCE_PROFILE_ARN", "")  # optional

def _bearer_header(tok: str) -> dict:
    return {"Authorization": f"Bearer {tok}", "Content-Type": "application/json"}

def _build_chat_payload(user_text: str, system_text: str = None, max_tokens: int = 300):
    payload = {
        "messages": [{"role": "user", "content": user_text}],
        "max_tokens": max_tokens
    }
    if system_text:
        payload["system"] = system_text
    if INFERENCE_PROFILE_ARN:
        payload["inference_profile_arn"] = INFERENCE_PROFILE_ARN
    return payload

def ping():
    # light sanity check that the endpoint, token, and (optional) profile all flow through
    system = "Return STRICT JSON only. No prose."
    user = 'Respond with {"ok": true, "who":"validator"}'
    r = requests.post(COVERITY_ASSIST_URL, headers=_bearer_header(TOKEN),
                      json=_build_chat_payload(user, system, 60), timeout=60)
    r.raise_for_status()
    data = r.json()
    text = data.get("content") or data.get("response") or data.get("text") or json.dumps(data)
    j = json.loads(text)
    assert isinstance(j, dict) and j.get("ok") is True, f"Unexpected ping JSON: {text}"
    return j

def validate_summary(original_request: str, summary_text: str):
    system = "Return STRICT JSON only. No prose."
    user = (
        "Original request:\n---\n" + original_request + "\n---\n\n"
        "Did the summary below fully satisfy the request? If not, list concrete next actions "
        "(bash commands or URLs) we should run/fetch next.\n\n"
        "Summary:\n---\n" + summary_text + "\n---\n\n"
        'Respond JSON with keys: "complete": true|false, "next_actions": [ {"cmd": "."}, {"url": "."} ]'
    )
    r = requests.post(COVERITY_ASSIST_URL, headers=_bearer_header(TOKEN),
                      json=_build_chat_payload(user, system, 300), timeout=120)
    r.raise_for_status()
    data = r.json()
    text = data.get("content") or data.get("response") or data.get("text") or json.dumps(data)
    verdict = json.loads(text)

    # normalize contract
    if not isinstance(verdict, dict):
        verdict = {}
    verdict.setdefault("complete", False)
    verdict.setdefault("next_actions", [])
    return verdict

if __name__ == "__main__":
    if not TOKEN:
        print("COVERITY_ASSIST_TOKEN is required", file=sys.stderr)
        sys.exit(2)

    mode = (sys.argv[1] if len(sys.argv) > 1 else "ping").lower()
    if mode == "ping":
        print(json.dumps(ping(), indent=2))
    else:
        # example usage: python validate_coverity.py validate "request..." "summary..."
        if len(sys.argv) < 4:
            print("usage: validate_coverity.py validate '<request>' '<summary>'", file=sys.stderr)
            sys.exit(2)
        print(json.dumps(validate_summary(sys.argv[2], sys.argv[3]), indent=2))
