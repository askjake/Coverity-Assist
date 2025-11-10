from textwrap import dedent
from pathlib import Path


import os
import json
import zipfile
import subprocess
import requests
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

# ---- Paths & Config (env-overridable) ----
JAMBOT_BASE_URL = os.environ.get("JAMBOT_BASE_URL", "http://127.0.0.1:5000").rstrip("/")
COVERITY_ASSIST_URL = os.environ.get("COVERITY_ASSIST_URL", "http://coverity-assist.dishtv.technology/chat").rstrip("/")
COVERITY_ASSIST_TOKEN = os.environ.get("COVERITY_ASSIST_TOKEN", "")
JOURNALS_DIR = Path(os.environ.get("JOURNALS_DIR", str(Path.cwd() / "journals")))
INSTRUCTIONS_DIR = Path(os.environ.get("INSTRUCTIONS_DIR", str(Path.cwd() / "instructions")))

for d in (JOURNALS_DIR, INSTRUCTIONS_DIR):
    d.mkdir(parents=True, exist_ok=True)

session = requests.Session()

def _bearer_headers(token: str) -> Dict[str, str]:
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

def coverity_chat(
    chat_url: Optional[str],
    token: Optional[str],
    user_text: str,
    system_text: Optional[str] = None,
    max_tokens: int = 800,
    inference_profile_arn: Optional[str] = None,
) -> str:
    """
    Minimal helper to call the Coverity Assist chat gateway.
    Falls back to env COVERITY_ASSIST_URL/TOKEN when args are None.
    """
    url = (chat_url or COVERITY_ASSIST_URL).rstrip("/")
    tok = token or COVERITY_ASSIST_TOKEN
    payload: Dict[str, Any] = {"messages": [{"role": "user", "content": user_text}], "max_tokens": max_tokens}
    if system_text:
        payload["system"] = system_text  # send as top-level system prompt when supported
    if inference_profile_arn:
        payload["inference_profile_arn"] = inference_profile_arn

    r = session.post(url, headers=_bearer_headers(tok), json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    return data.get("content") or data.get("response") or data.get("text") or json.dumps(data)

def write_instructions(task_text: str) -> Path:
    p = INSTRUCTIONS_DIR / "workflow.instruct"
    p.write_text(task_text or "", encoding="utf-8")
    return p

def generate_resource_list(
    task_text: str,
    chat_url: Optional[str] = None,
    token: Optional[str] = None,
    inference_profile_arn: Optional[str] = None,
) -> Path:
    """
    Ask the model for a CSV-like list of resources/commands to execute.
    Columns: Tool/Resource, Specific info needed, Required bash command (if any), URL (if any)
    Arguments chat_url/token are optional; env fallbacks are used if omitted.
    """
    system = "You plan technical workflows. Produce precise, safe, reproducible steps."
    user = (
        "Task:\n"
        f"{task_text}\n\n"
        "Return ONLY CSV rows, no prose, columns:\n"
        "Tool/Resource,Specific info needed,Required bash command (if any),URL (if any)\n"
        "If a column is N/A, put '-'. Keep commands short and safe.\n"
    )
    text = coverity_chat(chat_url, token, user, system_text=system, max_tokens=700, inference_profile_arn=inference_profile_arn)
    out = INSTRUCTIONS_DIR / "workflow.resources"
    out.write_text(text or "", encoding="utf-8")
    return out

def run_commands(resources_path: Path) -> Path:
    """
    For each CSV row, execute BashCommand (if not '-') or fetch URL.
    Writes a stitched 'workflow.data' file with command output or URL content.
    """
    data_path = INSTRUCTIONS_DIR / "workflow.data"
    lines = [ln.strip() for ln in resources_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    with data_path.open("w", encoding="utf-8") as out:
        for ln in lines:
            cols = [c.strip() for c in ln.split(",")]
            cmd = cols[2] if len(cols) > 2 else "-"
            url = cols[3] if len(cols) > 3 else "-"

            if url and url not in ("-", "N/A", "NA", "None"):
                try:
                    r = session.get(url, timeout=30, verify=True)
                    body = r.text[:20000]
                    out.write(f"URL: {url}\n{body}\n\n")
                except Exception as e:
                    out.write(f"URL: {url}\nERROR: {e}\n\n")

            if cmd and cmd not in ("-", "N/A", "NA", "None"):
                try:
                    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=180)
                    out.write(f"CMD: {cmd}\nRET={proc.returncode}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}\n\n")
                except Exception as e:
                    out.write(f"CMD: {cmd}\nERROR: {e}\n\n")
    return data_path

def process_data_for_embedding(data_path: Path, original_request: str) -> None:
    """
    Store a local copy, try to push into JAMbot's /embed-content (best effort),
    and append a journal note.
    """
    text = data_path.read_text(encoding="utf-8")
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    local_copy = INSTRUCTIONS_DIR / f"workflow-{ts}.txt"
    local_copy.write_text(text, encoding="utf-8")

    # Best-effort push to JAMbot for embedding
    try:
        url = f"{JAMBOT_BASE_URL}/embed-content"
        r = session.post(
            url,
            data={"title": f"workflow-{ts}", "model": "Alex"},
            files={"file": ("workflow.txt", text.encode("utf-8"), "text/plain")},
            timeout=60,
        )
        _ = r.status_code  # ignore failure
    except Exception:
        pass

    # Journal append
    try:
        JOURNALS_DIR.mkdir(parents=True, exist_ok=True)
        jf = JOURNALS_DIR / "gabriel.journal"
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with jf.open("a", encoding="utf-8") as f:
            f.write(f"[{stamp}] Workflow note: {original_request}\nSaved: {local_copy.name}\n\n")
    except Exception:
        pass

def summarize_to_text(
    data_path: Path,
    chat_url: Optional[str] = None,
    token: Optional[str] = None,
    inference_profile_arn: Optional[str] = None,
) -> str:
    system = "You are a senior engineer. Summarize tersely and concretely for an exec update."
    user = "Summarize the following findings into 6-10 bullets with concrete metrics and next steps:\n\n" + data_path.read_text(encoding="utf-8")[:28000]
    try:
        return coverity_chat(chat_url, token, user, system_text=system, max_tokens=600, inference_profile_arn=inference_profile_arn)
    except Exception as e:
        return f"(summary failed: {e})"

def bundle_workflow_files() -> Path:
    zip_path = INSTRUCTIONS_DIR / "workflow_bundle.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for name in ("workflow.instruct", "workflow.resources", "workflow.data"):
            p = INSTRUCTIONS_DIR / name
            if p.exists():
                z.write(p, arcname=name)
    return zip_path

def get_workflow_files() -> Dict[str, str]:
    out: Dict[str, str] = {}
    for nm in ("workflow.instruct", "workflow.resources", "workflow.data"):
        p = INSTRUCTIONS_DIR / nm
        if p.exists():
            try:
                out[nm] = p.read_text(encoding="utf-8")
            except Exception as e:
                out[nm] = f"(error reading: {e})"
    zp = INSTRUCTIONS_DIR / "workflow_bundle.zip"
    if zp.exists():
        out["workflow_bundle.zip"] = str(zp)
    return out