#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAMbot API (Flask, Python 3.8 compatible)
- Matches the endpoints expected by coverity_assist_streamlit_app.py
- Gracefully degrades when optional modules (web_search, log tools) are absent
- Avoids PEP 604 typing; uses typing.Optional/Dict/List/Tuple
"""

import os
import re
import json
import time
import platform
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from textwrap import dedent

import requests
from flask import Flask, request, jsonify, Response
from flask_cors import CORS

# Optional deps: imported lazily
def try_import(name: str):
    try:
        return __import__(name)
    except Exception:
        return None

web_search_mod = try_import("web_search")       # expects do_web_search(query: str) -> str
combine_logs_mod = try_import("combine_logs")   # expects combine_logs(src_dir, dst_dir)
journaler_mod = try_import("journaler")         # optional helpers
bert2_mod = try_import("bert2")                 # optional BERTopic pipeline
log_rag_mod = try_import("log_rag")             # optional embed/RAG helpers

from requests.adapters import HTTPAdapter
try:
    # urllib3 v1 & v2 compat
    from urllib3.util.retry import Retry  # type: ignore
except Exception:
    Retry = None  # Fallback if not present

try:
    from bs4 import BeautifulSoup
except Exception:
    BeautifulSoup = None  # graceful fallback for /embed-content

# Prefer local workflow_manager helpers if available
try:
    from workflow_manager import (
        write_instructions,
        generate_resource_list,
        run_commands,
        process_data_for_embedding,
        summarize_to_text,
        bundle_workflow_files,
        get_workflow_files,
    )
    HAVE_WFM = True
except Exception:
    HAVE_WFM = False
# ---------------- Configuration ----------------

APP_PORT = int(os.environ.get("JAMBOT_PORT", "5001"))
APP_HOST = os.environ.get("JAMBOT_HOST", "0.0.0.0")

JAMBOT_ROOT = Path(os.environ.get("JAMBOT_ROOT", Path.cwd()))
BASH_DIRECTORY = JAMBOT_ROOT / "bash"
JOURNALS_DIR = Path(os.environ.get("JOURNALS_DIR", JAMBOT_ROOT / "journals"))
INSTRUCTIONS_DIR = Path(os.environ.get("INSTRUCTIONS_DIR", JAMBOT_ROOT / "instructions"))
EMBED_DIR = Path(os.environ.get("EMBED_DIR", JAMBOT_ROOT / "embedded"))
COVERITY_ASSIST_URL   = os.environ.get("COVERITY_ASSIST_URL", "")
COVERITY_ASSIST_TOKEN = os.environ.get("COVERITY_ASSIST_TOKEN", "")

for d in (BASH_DIRECTORY, JOURNALS_DIR, INSTRUCTIONS_DIR, EMBED_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Ollama model host map (tweak to your infra)
DSGPU3090 = os.environ.get("DSGPU3090_IP", "10.79.85.35")
DSGPU3080 = os.environ.get("DSGPU3080_IP", "10.79.85.47")
DEFAULT_OLLAMA_IP = os.environ.get("OLLAMA_DEFAULT_IP", DSGPU3090)

MODEL_HOST_MAPPING: Dict[str, str] = {
    "ai1": DSGPU3090,
    "aqua": DSGPU3090,
    "gabriel": DSGPU3090,
    "alex": DSGPU3090,
    "billwestfall": DSGPU3090,
    "ai2": DSGPU3080,
    "gemma": DSGPU3080,
    "avery": DSGPU3080,
}

# Logging
LOG_PATH = JAMBOT_ROOT / "logJAM.txt"
logging.basicConfig(
    filename=str(LOG_PATH),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.getLogger("werkzeug").setLevel(logging.WARNING)

# Requests session with retries
session = requests.Session()
if Retry is not None:
    retry = Retry(total=3, backoff_factor=0.3, status_forcelist=[500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

app = Flask(__name__)
CORS(app)

# ---------------- Utilities ----------------
@app.route("/__routes", methods=["GET"])
def __routes():
    out = []
    for r in app.url_map.iter_rules():
        out.append({
            "rule": str(r),
            "methods": sorted(m for m in r.methods if m not in {"HEAD", "OPTIONS"}),
            "endpoint": r.endpoint,
        })
    return jsonify(out), 200

def http_ok(r: requests.Response) -> bool:
    try:
        r.raise_for_status()
        return True
    except requests.HTTPError:
        return False

def model_host_ip(model: str) -> str:
    return MODEL_HOST_MAPPING.get(model.lower(), DEFAULT_OLLAMA_IP)

def ensure_text(b: bytes) -> str:
    try:
        return b.decode("utf-8")
    except Exception:
        return b.decode("latin-1", errors="replace")

# ---------------- Health ----------------

@app.route("/health", methods=["GET"])
def health() -> Response:
    return jsonify({"status": "OK", "time": datetime.utcnow().isoformat() + "Z"}), 200

# ---------------- Journals ----------------

@app.route("/get-journal-files", methods=["GET"])
def get_journal_files() -> Response:
    out: Dict[str, str] = {}
    for p in sorted(JOURNALS_DIR.glob("*.journal")):
        try:
            out[p.name] = p.read_text(encoding="utf-8")
        except Exception as e:
            out[p.name] = "Error reading: %s" % e
    if not out:
        # seed files if none exist
        for seed in ("gabriel.journal", "gemma.journal", "alex.journal"):
            f = JOURNALS_DIR / seed
            if not f.exists():
                f.write_text("", encoding="utf-8")
            out[seed] = ""
    return jsonify(out), 200

@app.route("/journal", methods=["POST"])
def append_journal() -> Response:
    """
    Appends a timestamped entry to gabriel.journal (default),
    or to ?name=<persona>.journal if provided.
    """
    name = request.args.get("name", "gabriel")
    fpath = JOURNALS_DIR / f"{name.lower()}.journal"
    body = request.get_json(silent=True) or {}
    entry = body.get("entry") or f"Auto-entry at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}."
    try:
        with fpath.open("a", encoding="utf-8") as f:
            f.write(entry.strip() + "\n\n")
        return jsonify({"status": "ok", "file": str(fpath)}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------------- Web Search ----------------

@app.route("/web-search", methods=["POST"])
def web_search() -> Response:
    data = request.get_json(silent=True) or {}
    query = data.get("query")
    if not query:
        return jsonify({"error": "Query is required."}), 400

    # Prefer local module function
    if web_search_mod and hasattr(web_search_mod, "do_web_search"):
        try:
            res = web_search_mod.do_web_search(query)  # type: ignore[attr-defined]
            return jsonify({"response": res}), 200
        except Exception as e:
            logging.exception("Local web_search.do_web_search failed: %s", e)

    # Extremely lightweight fallback via DuckDuckGo html (no dependency)
    try:
        r = session.get("https://duckduckgo.com/html", params={"q": query}, timeout=15)
        if r.status_code != 200:
            return jsonify({"response": "Search HTTP %s" % r.status_code}), 200
        # naive scrape
        m = re.search(r'class="result__snippet">(.*?)</a>', r.text, re.S | re.I)
        snippet = m.group(1) if m else "No results found."
        return jsonify({"response": snippet}), 200
    except Exception as e:
        return jsonify({"response": "Fallback search error: %s" % e}), 200

# ---------------- Ollama Chat Bridge ----------------

def ollama_generate(host_ip: str, model: str, prompt: str,
                    num_predict: int = -2, temperature: float = 1.0) -> str:
    url = "http://%s:11434/api/generate" % host_ip
    payload = {
        "model": model,
        "prompt": prompt,
        "options": {
            "num_keep": 250,
            "num_predict": num_predict,
            "repeat_last_n": 33,
            "temperature": temperature,
            "repeat_penalty": 1.2,
            "presence_penalty": 1.5,
            "frequency_penalty": 1.0,
            "num_ctx": 32768,
        },
        "stream": False,
    }
    try:
        r = session.post(url, json=payload, timeout=240)
        if not http_ok(r):
            return "Error: [%s] %s" % (r.status_code, r.text)
        data = r.json()
        return data.get("response", "") or "No response."
    except Exception as e:
        return "Error contacting Ollama: %s" % e

@app.route("/ollama", methods=["POST"])
def handle_conversation() -> Response:
    data = request.get_json(silent=True) or {}
    prompt = data.get("prompt", "")
    model = data.get("model", "Alex")
    options = data.get("options", {}) or {}
    num_predict = int(options.get("num_predict", -2))
    temperature = float(options.get("temperature", 1))

    if not prompt:
        return jsonify({"error": "prompt required"}), 400

    host_ip = model_host_ip(model)
    # TODO: Add embedding/RAG lookups here if desired. For now, pass-through.
    text = ollama_generate(host_ip, model, prompt, num_predict=num_predict, temperature=temperature)
    return jsonify({"response": text}), 200

# ---------------- Embed Content ----------------

@app.route("/embed-content", methods=["POST"])
def embed_content() -> Response:
    """
    Minimal, file/URL/Confluence handler.
    If log_rag module exposes a compatible function, we call it.
    Otherwise we store plaintext in EMBED_DIR (timestamped) and return a short summary.
    """
    # Multipart form fields
    file = request.files.get("file")
    generic_url = request.form.get("generic_url")
    confluence_url = request.form.get("confluence")
    space_key = request.form.get("space_key")
    title = request.form.get("title")
    model = request.form.get("model", "Alex")

    captured_text = None
    source_label = None

    try:
        if file:
            raw = file.read()
            captured_text = ensure_text(raw)
            source_label = "upload:%s" % file.filename

        elif generic_url:
            r = session.get(generic_url, timeout=60, verify=False)
            if not http_ok(r):
                return jsonify({"error": "[%s] %s" % (r.status_code, r.text), "needs_credentials": False}), 400
            text = r.text
            if BeautifulSoup is not None:
                soup = BeautifulSoup(text, "html.parser")
                captured_text = soup.get_text(separator="\n")
            else:
                captured_text = text
            source_label = "url:%s" % generic_url

        elif confluence_url and space_key and title:
            r = session.get(confluence_url, timeout=60, verify=False)
            if not http_ok(r):
                return jsonify({"error": "[%s] %s" % (r.status_code, r.text)}), 400
            text = r.text
            if BeautifulSoup is not None:
                soup = BeautifulSoup(text, "html.parser")
                captured_text = soup.get_text(separator="\n")
            else:
                captured_text = text
            source_label = "confluence:%s/%s" % (space_key, title)
        else:
            return jsonify({"error": "No valid input provided."}), 400

        # If a specialized embed/store is available, defer to it
        if log_rag_mod and hasattr(log_rag_mod, "embed_and_store_content"):
            try:
                # signature may vary; attempt a tolerant call
                result = log_rag_mod.embed_and_store_content(captured_text,  # type: ignore[attr-defined]
                                                             metadata={"source": source_label,
                                                                       "timestamp": datetime.now().isoformat()},
                                                             host_ip=model_host_ip(model),
                                                             table_name="embedded_content",
                                                             model=model,
                                                             max_length=29000)
                # Expect either boolean or dict
                if isinstance(result, dict):
                    return jsonify({"result": "Content embedded (log_rag).",
                                    "status": result.get("status"),
                                    "summaries": result.get("summaries")}), 200
                return jsonify({"result": "Content embedded (log_rag)."}), 200
            except Exception as e:
                logging.exception("log_rag.embed_and_store_content failed: %s", e)

        # Fallback: write to disk and summarize with Ollama
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_file = EMBED_DIR / f"{ts}.txt"
        out_file.write_text(captured_text or "", encoding="utf-8")

        # quick summary via Ollama if reachable
        host_ip = model_host_ip(model)
        summary_prompt = "Summarize the following content in 8-15 bullet points, preserving key technical details:\n\n%s" % (captured_text[:28000])
        summary = ollama_generate(host_ip, model, summary_prompt, num_predict=800, temperature=0.7)

        return jsonify({
            "result": "Content stored locally.",
            "path": str(out_file),
            "summaries": [{"model": model, "file": out_file.name, "part": "summary", "summary": summary}]
        }), 200

    except Exception as e:
        logging.exception("embed-content error: %s", e)
        return jsonify({"error": str(e)}), 500

# ---------------- Tools ----------------

@app.route("/tools/allowed", methods=["GET"])
def tools_allowed() -> Response:
    default_allowed = ["ls", "cat", "head", "tail", "grep", "awk", "sed", "wc", "df", "du",
                       "uname", "whoami", "uptime", "python3", "pip", "bash"]
    txt = (JAMBOT_ROOT / "tools" / "bash.txt")
    try:
        if txt.exists():
            return Response(txt.read_text(encoding="utf-8"), mimetype="text/plain")
        else:
            return Response("\n".join(default_allowed), mimetype="text/plain")
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/tools/<tool>", methods=["POST"])
def run_tool(tool: str) -> Response:
    data = request.get_json(silent=True) or {}
    argument = data.get("argument", "")

    if tool == "bash":
        try:
            # Run within BASH_DIRECTORY
            shell = "powershell" if "windows" in platform.system().lower() else "bash"
            proc = subprocess.run(argument, shell=True, cwd=str(BASH_DIRECTORY),
                                  capture_output=True, text=True, timeout=600)
            out = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
            return jsonify({"result": out.strip()}), 200
        except Exception as e:
            return jsonify({"result": "Error: %s" % e}), 200

    elif tool == "ping":
        try:
            target = argument.strip()
            if not target:
                return jsonify({"result": "No target"}), 200
            cmd = ["ping", "-n", "4", target] if platform.system().lower().startswith("win") else ["ping", "-c", "4", target]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            out = proc.stdout if proc.returncode == 0 else proc.stderr
            return jsonify({"result": out.strip()}), 200
        except Exception as e:
            return jsonify({"result": "Error: %s" % e}), 200

    return jsonify({"error": "Invalid tool requested"}), 400

# ---------------- Workflow ----------------

_workflow_repeats = 0
_max_repeats = int(os.environ.get("WORKFLOW_MAX_REPEATS", "100"))

def _write_instructions(text: str) -> Path:
    p = INSTRUCTIONS_DIR / "workflow.instruct"
    p.write_text(text or "", encoding="utf-8")
    return p

def _generate_resource_list(task_text: str) -> Path:
    """Ask Ollama(Alex) to produce CSV-like resources list."""
    model = "Alex"
    prompt = dedent("""
    The following task requires various tools or resources to complete.
    Extract each tool or resource needed.

    Respond as CSV lines:
    Tool/Resource, Specific info needed, Required bash command (if any), URL (if any)

    Task:
    %s
    """).strip() % task_text

    res = ollama_generate(model_host_ip(model), model, prompt, num_predict=1200, temperature=0.7)
    p = INSTRUCTIONS_DIR / "workflow.resources"
    p.write_text(res or "", encoding="utf-8")
    return p

def _run_commands(resources_path: Path) -> Path:
    data_path = INSTRUCTIONS_DIR / "workflow.data"
    lines = [ln.strip() for ln in resources_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    with data_path.open("w", encoding="utf-8") as out:
        for ln in lines:
            cols = [c.strip() for c in ln.split(",")]
            if not cols:
                continue
            target = cols[-1]
            if target.startswith("http"):
                try:
                    r = session.get(target, timeout=30)
                    text = r.text
                except Exception as e:
                    text = "URL fetch error: %s" % e
            else:
                try:
                    text = subprocess.getoutput(target)
                except Exception as e:
                    text = "Command error: %s" % e
            out.write("Target: %s\nResult:\n%s\n\n" % (target, text))
    return data_path

def _process_data_for_embedding(data_path: Path, original_request: str) -> None:
    data = data_path.read_text(encoding="utf-8")
    model = "Alex"
    host_ip = model_host_ip(model)
    # Store locally
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    (EMBED_DIR / f"workflow-{ts}.txt").write_text(data, encoding="utf-8")

    # Ask the LLM to push forward the task
    prompt = dedent("""
    You are executing a multi-step workflow.

    Original Task:
    %s

    Below is the data gathered so far. Process and complete as much as possible.
    Identify missing pieces and propose the next concrete actions.

    Data:
    %s
    """).strip() % (original_request, data)
    _ = ollama_generate(host_ip, model, prompt, num_predict=1000, temperature=0.7)


@app.route("/trigger-workflow", methods=["POST"])
def trigger_workflow() -> Response:
    global _workflow_repeats
    _workflow_repeats += 1
    if _workflow_repeats > _max_repeats:
        return jsonify({"error": "Maximum workflow repeats reached."}), 400

    data = request.get_json(silent=True) or {}
    task_description       = (data.get("task_description") or "").strip()
    original_request       = (data.get("original_request") or "").strip()
    chat_url               = data.get("chat_url")
    token                  = data.get("token")
    inference_profile_arn  = data.get("inference_profile_arn")

    if not task_description or not original_request:
        return jsonify({"error": "task_description and original_request are required"}), 400

    try:
        # ---------- Preferred path: use workflow_manager helpers ----------
        if HAVE_WFM:
            ins_path  = write_instructions(task_description)
            res_path  = generate_resource_list(
                task_description,
                chat_url=chat_url,
                token=token,
                inference_profile_arn=inference_profile_arn,
            )
            data_path = run_commands(res_path)
            process_data_for_embedding(data_path, original_request)
            summary   = summarize_to_text(
                data_path,
                chat_url=chat_url,
                token=token,
                inference_profile_arn=inference_profile_arn,
            )
            bundle_path = str(bundle_workflow_files())

        # ---------- Fallback path: internal simple pipeline ----------
        else:
            ins_path  = _write_instructions(task_description)
            res_path  = _generate_resource_list(ins_path.read_text(encoding="utf-8"))
            data_path = _run_commands(res_path)
            _process_data_for_embedding(data_path, original_request)

            # Summarize with Ollama if workflow_manager not present
            model   = "Alex"
            host_ip = model_host_ip(model)
            summ_prompt = (
                "Summarize these workflow findings into 6–10 bullets with concrete next steps:\n\n"
                + data_path.read_text(encoding="utf-8")[:28000]
            )
            summary = ollama_generate(host_ip, model, summ_prompt, num_predict=600, temperature=0.7)

            # ZIP bundle
            import zipfile
            zf = INSTRUCTIONS_DIR / "workflow_bundle.zip"
            with zipfile.ZipFile(zf, "w", zipfile.ZIP_DEFLATED) as z:
                for name in ("workflow.instruct", "workflow.resources", "workflow.data"):
                    p = INSTRUCTIONS_DIR / name
                    if p.exists():
                        z.write(p, arcname=name)
            bundle_path = str(zf)

        payload = {
            "status": "ok",
            "workflow_content":  ins_path.read_text(encoding="utf-8"),
            "resources_preview": res_path.read_text(encoding="utf-8")[:2000],
            "data_preview":      data_path.read_text(encoding="utf-8")[:2000],
            "bundle_path":       bundle_path,
            "summary":           summary,
            "inference_profile_arn": inference_profile_arn,
            "repeats": _workflow_repeats,
        }
        return jsonify(payload), 200

    except Exception as e:
        logging.exception("trigger-workflow failed: %s", e)
        return jsonify({"error": str(e)}), 500

@app.route("/get-workflow-step", methods=["GET"])
def get_workflow_step() -> Response:
    return jsonify({"workflow_repeats": _workflow_repeats, "max": _max_repeats}), 200

# ---------------- Code Cleaner (no-op placeholder) ----------------
@app.route("/code-cleaner", methods=["GET"])
def code_cleaner() -> Response:
    """
    Placeholder that simply reports the number of files in bash/.
    (Your original heavy formatter/black/shfmt/etc. can be wired back as needed.)
    """
    try:
        n = len(list(BASH_DIRECTORY.glob("*")))
        return jsonify({"success": True, "message": "bash/ contains %d files." % n})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

# ---------------- Main ----------------

if __name__ == "__main__":
    app.run(host=APP_HOST, port=APP_PORT, debug=True, use_reloader=False)
