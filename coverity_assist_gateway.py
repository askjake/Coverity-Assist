#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Coverity Assist Gateway (v3)
- Stable server: no auto-reloader by default (to avoid restarts when instructions/ZIP change)
- Toggle local vs. WWW search, with domain allowlist/blocklist
- Optional credential handling per-domain: cookies/basic/headers and proxy config
- Iterative workflow that validates with Coverity Assist and appends next actions

Security notes:
- If the 'keyring' library is available, secret material is stored in the OS keyring.
- Otherwise, secrets are kept in-memory only for the process lifetime (NOT persisted).
- The JSON config on disk never stores secrets; only non-sensitive settings are persisted.
"""

import os
import re
import json
import zipfile
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
from urllib.parse import urlparse

import requests
from flask import Flask, request, jsonify

# ---------- Paths & Config ----------

COVERITY_ASSIST_URL = os.environ.get("COVERITY_ASSIST_URL", "http://coverity-assist.dishtv.technology/chat").rstrip("/")
COVERITY_ASSIST_TOKEN = os.environ.get("COVERITY_ASSIST_TOKEN", "")
STATE_DIR = os.environ.get("STATE_DIRECTORY")

ROOT = Path(os.environ.get("JAMBOT_ROOT", Path.cwd()))
INSTRUCTIONS_DIR = Path(os.environ.get("INSTRUCTIONS_DIR", str((Path(STATE_DIR) if STATE_DIR else ROOT) / "instructions")))
EMBED_DIR        = Path(os.environ.get("EMBED_DIR",        str((Path(STATE_DIR) if STATE_DIR else ROOT) / "embedded")))
JOURNALS_DIR     = Path(os.environ.get("JOURNALS_DIR",     str((Path(STATE_DIR) if STATE_DIR else ROOT) / "journals")))
CONFIG_PATH = Path(os.environ.get("GATEWAY_CONFIG", str(ROOT / "coverity_config.json")))

for d in (INSTRUCTIONS_DIR, EMBED_DIR, JOURNALS_DIR):
    d.mkdir(parents=True, exist_ok=True)

def _try_import(name: str):
    try:
        return __import__(name)
    except Exception:
        return None

web_search_mod = _try_import("web_search")  # optional local helper

# Optional secure storage
keyring = _try_import("keyring")

# In-memory secrets fallback (not persisted)
_MEMORY_SECRETS: Dict[str, Dict[str, Any]] = {
    "basic": {},   # domain -> {"username":..., "password":...}
    "cookies": {}, # domain -> {"Cookie": "a=b; ..."}
    "headers": {}, # domain -> {"Header-Name": "Value"}
    "proxy": {},   # {"http": "...", "https": "..."}
}

# Public (non-secret) config persisted on disk
_DEFAULT_CONFIG = {
    "search_mode": "both",  # "local" | "www" | "both"
    "allowlist": [],        # if non-empty, only these domains are used
    "blocklist": ["linkedin.com", "zoominfo.com", "x.com", "twitter.com", "facebook.com", "instagram.com", "tiktok.com"],
    "use_credentials_by_default": False,
    "max_results": 6,
}

def _load_config() -> Dict[str, Any]:
    if CONFIG_PATH.exists():
        try:
            return {**_DEFAULT_CONFIG, **json.loads(CONFIG_PATH.read_text(encoding="utf-8"))}
        except Exception:
            return dict(_DEFAULT_CONFIG)
    return dict(_DEFAULT_CONFIG)

def _save_config(cfg: Dict[str, Any]) -> None:
    # never write secrets to disk
    safe = {k: v for k, v in cfg.items() if k in _DEFAULT_CONFIG}
    CONFIG_PATH.write_text(json.dumps(safe, indent=2), encoding="utf-8")

CONFIG = _load_config()

# ---------- Credentials helpers ----------

def _kr_key(domain: str, kind: str) -> str:
    return f"coverity_gateway::{kind}::{domain}"

def _get_secret(domain: str, kind: str) -> Optional[Dict[str, Any]]:
    if keyring is not None:
        try:
            raw = keyring.get_password("coverity_gateway", _kr_key(domain, kind))
            if raw:
                return json.loads(raw)
        except Exception:
            pass
    # fallback in-memory
    return _MEMORY_SECRETS.get(kind, {}).get(domain)

def _set_secret(domain: str, kind: str, data: Dict[str, Any], persist: bool) -> None:
    if persist and keyring is not None:
        try:
            keyring.set_password("coverity_gateway", _kr_key(domain, kind), json.dumps(data))
            return
        except Exception:
            pass
    # fallback
    _MEMORY_SECRETS.setdefault(kind, {})[domain] = data

def _get_proxies() -> Dict[str, str]:
    # merge env proxies with configured proxy secrets
    proxies = {}
    for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        if os.environ.get(k):
            # requests expects lowercase keys 'http'/'https'
            proxies[k.split("_")[0].lower()] = os.environ.get(k)  # type: ignore
    # explicit proxy set via credentials
    p = _MEMORY_SECRETS.get("proxy") or {}
    proxies.update({k: v for k, v in p.items() if v})
    return proxies

def _domain_of(url: str) -> str:
    try:
        return urlparse(url).hostname or ""
    except Exception:
        return ""

# ---------- HTTP util ----------

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118 Safari/537.36"
BASE_HEADERS = {"User-Agent": UA, "Accept-Language": "en-US,en;q=0.9"}

def _http_ok(r: requests.Response) -> bool:
    try:
        r.raise_for_status()
        return True
    except requests.HTTPError:
        return False

session = requests.Session()

# ---------- Flask ----------

app = Flask(__name__)

@app.get("/health")
def health():
    return jsonify({"status": "OK", "gateway": True, "search_mode": CONFIG.get("search_mode")}), 200

# ----- Config endpoints -----

@app.get("/config/search")
def get_search_config():
    return jsonify(CONFIG), 200

@app.post("/config/search")
def set_search_config():
    data = request.get_json(silent=True) or {}
    for k in ("search_mode", "allowlist", "blocklist", "use_credentials_by_default", "max_results"):
        if k in data:
            CONFIG[k] = data[k]
    _save_config(CONFIG)
    return jsonify({"status": "ok", "config": CONFIG}), 200

@app.post("/credentials")
def set_credentials():
    """
    Body:
      {
        "domain": "example.com" | "*" (for proxy),
        "kind": "basic" | "cookies" | "headers" | "proxy",
        "data": {...},
        "persist": true|false
      }
    For kind=proxy: data can be {"http":"http://user:pass@proxy:8080","https":"http://user:pass@proxy:8443"}
    """
    data = request.get_json(silent=True) or {}
    domain = data.get("domain") or ""
    kind = data.get("kind")
    payload = data.get("data") or {}
    persist = bool(data.get("persist", False))

    if kind not in {"basic", "cookies", "headers", "proxy"}:
        return jsonify({"error": "invalid kind"}), 400

    if kind == "proxy":
        # store globally (no domain)
        _MEMORY_SECRETS["proxy"] = {k: v for k, v in payload.items() if k in ("http", "https")}
        return jsonify({"status": "ok", "note": "proxy configured for this process"}), 200

    if not domain:
        return jsonify({"error": "domain required"}), 400

    _set_secret(domain, kind, payload, persist)
    return jsonify({"status": "ok"}), 200

# ----- Journals -----

@app.get("/get-journal-files")
def get_journal_files():
    out = {}
    for p in sorted(JOURNALS_DIR.glob("*.journal")):
        try:
            out[p.name] = p.read_text(encoding="utf-8")
        except Exception as e:
            out[p.name] = f"Error reading: {e}"
    return jsonify(out), 200

@app.post("/journal")
def append_journal():
    try:
        jf = JOURNALS_DIR / "gabriel.journal"
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        body = (request.get_json(silent=True) or {}).get("entry") or f"Auto-entry at {stamp}."
        with jf.open("a", encoding="utf-8") as f:
            f.write(body.strip() + "\\n\\n")
        return jsonify({"status": "ok", "appended": body}), 200
    except Exception as e:
        return jsonify({"error": f"Journal write failed: {e}"}), 500

# ----- Search helpers -----

def _ddg_lite(query: str, max_results: int = 6, timeout: int = 15) -> List[Dict[str, str]]:
    url = "https://lite.duckduckgo.com/lite/"
    r = session.get(url, params={"q": query}, headers=BASE_HEADERS, timeout=timeout)
    r.raise_for_status()
    html_text = r.text
    rows = re.findall(
        r"<tr[^>]*>\\s*<td[^>]*>\\s*<a[^>]+class=['\\\"]result-link['\\\"][^>]+href=['\\\"](.*?)['\\\"][^>]*>(.*?)</a>.*?<td[^>]*>(.*?)</td>",
        html_text, flags=re.S | re.I
    )
    import html as _html
    out = []
    for href, title_html, snippet_html in rows[:max_results]:
        title = _html.unescape(re.sub("<.*?>", "", title_html)).strip() or "(no title)"
        snippet = _html.unescape(re.sub("<.*?>", "", snippet_html)).strip()
        out.append({"title": title, "url": href, "snippet": snippet})
    return out

def _fetch_text(url: str, domain: str, use_credentials: bool, max_chars: int = 6000) -> str:
    headers = dict(BASE_HEADERS)
    auth = None
    cookies = None
    proxies = _get_proxies()

    if use_credentials:
        b = _get_secret(domain, "basic")
        if b and isinstance(b, dict) and b.get("username") and b.get("password"):
            from requests.auth import HTTPBasicAuth
            auth = HTTPBasicAuth(b["username"], b["password"])
        c = _get_secret(domain, "cookies")
        if c and isinstance(c, dict) and c.get("Cookie"):
            headers["Cookie"] = c["Cookie"]
        h = _get_secret(domain, "headers")
        if h and isinstance(h, dict):
            headers.update({k:str(v) for k,v in h.items()})

    try:
        with session.get(url, headers=headers, auth=auth, timeout=15, stream=True, proxies=proxies, verify=False) as r:
            r.raise_for_status()
            enc = r.encoding or "utf-8"
            size = 0
            buf_chunks = []
            for chunk in r.iter_content(chunk_size=4096):
                if not chunk:
                    break
                buf_chunks.append(chunk)
                size += len(chunk)
                if size >= 200_000:
                    break
        html_text = b"".join(buf_chunks).decode(enc, errors="replace")
        # Strip to plain text quickly
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html_text, "html.parser")
            for t in soup(["script","style","noscript"]):
                t.decompose()
            text = soup.get_text(separator="\\n")
        except Exception:
            text = re.sub("<[^>]+>", " ", html_text)
        text = re.sub(r"\\n{3,}", "\\n\\n", text)
        text = re.sub(r"[ \\t]{2,}", " ", text)
        return text.strip()[:max_chars]
    except Exception as e:
        return f"(fetch error for {domain}: {e})"

def _filter_results(results: List[Dict[str, str]], allowlist: List[str], blocklist: List[str]) -> List[Dict[str, str]]:
    out = []
    for r in results:
        d = _domain_of(r.get("url",""))
        if not d:
            continue
        if allowlist and all(a not in d for a in allowlist):
            continue
        if any(b in d for b in blocklist):
            # Keep but mark skipped? For now, drop it; workflow can't fetch these.
            continue
        r2 = dict(r)
        r2["domain"] = d
        out.append(r2)
    return out

@app.post("/web-search")
def web_search():
    try:
        data = request.get_json(force=True) or {}
        q = (data.get("query") or "").strip()
        if not q:
            return jsonify({"response": "Query is required."}), 200

        cfg = _read_json(SEARCH_CFG, {
            "mode": "both", "use_credentials": False, "fetch_pages": True,
            "allowlist": [], "blocklist": []
        })
        creds = _read_json(CREDS_FILE, {"cookie":"", "basic_user":"", "basic_pass":"", "proxy":"", "headers":{}})

        mode = data.get("mode", cfg["mode"])
        fetch_pages = bool(data.get("fetch_pages", cfg["fetch_pages"]))
        use_creds = bool(data.get("use_credentials", cfg["use_credentials"]))
        allowlist = set(map(str.lower, data.get("allowlist", cfg["allowlist"])))
        blocklist = set(map(str.lower, data.get("blocklist", cfg["blocklist"])))

        # Build a stateless, low-bandwidth session
        s = requests.Session()
        headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/115 Safari/537.36",
                   "Accept-Language": "en-US,en;q=0.9"}
        if use_creds:
            if creds.get("cookie"):
                headers["Cookie"] = creds["cookie"]
            if isinstance(creds.get("headers"), dict):
                headers.update({k:str(v) for k,v in creds["headers"].items()})
        s.headers.update(headers)
        auth = (creds["basic_user"], creds["basic_pass"]) if (use_creds and creds.get("basic_user")) else None
        proxies = {"http": creds["proxy"], "https": creds["proxy"]} if (use_creds and creds.get("proxy")) else None

        def _allowed(url: str) -> bool:
            import urllib.parse as up
            try:
                host = (up.urlparse(url).hostname or "").lower()
            except Exception:
                return False
            if any(host.endswith(b) for b in blocklist if b): return False
            if allowlist and not any(host.endswith(a) for a in allowlist if a): return False
            return True

        results = {"query": q, "fetched_at": int(datetime.now().timestamp()), "results": [], "note": "hybrid search"}

        # Local helper first (if requested and available)
        if mode in ("local", "both") and web_search_mod and hasattr(web_search_mod, "do_web_search"):
            try:
                ro = web_search_mod.do_web_search(q, max_results=6, fetch_pages=fetch_pages)
                for r in ro.get("results", []):
                    if r.get("url") and not _allowed(r["url"]): continue
                    results["results"].append(r)
            except Exception:
                pass

        # Lightweight WWW search (DDG lite) if needed
        if mode in ("www", "both") and len(results["results"]) < 3:
            import re, html as _html
            r = s.get("https://lite.duckduckgo.com/lite/", params={"q": q}, proxies=proxies, auth=auth, timeout=15)
            if r.status_code == 200:
                rows = re.findall(
                    r"<tr[^>]*>\s*<td[^>]*>\s*<a[^>]+class=['\"]result-link['\"][^>]+href=['\"](.*?)['\"][^>]*>(.*?)</a>.*?<td[^>]*>(.*?)</td>",
                    r.text, flags=re.S | re.I
                )
                for href, title_html, snippet_html in rows[:6]:
                    if not _allowed(href): continue
                    title = _html.unescape(re.sub("<.*?>", "", title_html)).strip() or "(no title)"
                    snippet = _html.unescape(re.sub("<.*?>", "", snippet_html)).strip()
                    text = ""
                    if fetch_pages:
                        try:
                            with s.get(href, proxies=proxies, auth=auth, timeout=12, stream=True) as rr:
                                rr.raise_for_status()
                                buf, size = [], 0
                                for chunk in rr.iter_content(4096):
                                    if not chunk: break
                                    buf.append(chunk); size += len(chunk)
                                    if size >= 200_000: break
                            try:
                                from bs4 import BeautifulSoup
                                soup = BeautifulSoup(b"".join(buf).decode(rr.encoding or "utf-8", errors="replace"), "html.parser")
                                for tag in soup(["script","style","noscript"]): tag.decompose()
                                text = soup.get_text("\n")[:6000]
                            except Exception:
                                text = ""
                        except Exception:
                            text = ""
                    results["results"].append({"title": title, "url": href, "snippet": snippet, "text": text})

        return jsonify({"response": results}), 200
    except Exception as e:
        return jsonify({"response": f"Search error: {e}"}), 200

@app.post("/embed-content")
def embed_content():
    try:
        file = request.files.get("file")
        text = ""
        if file:
            text = file.read().decode("utf-8", errors="replace")
        else:
            generic_url = request.form.get("generic_url")
            if generic_url:
                r = session.get(generic_url, timeout=30, verify=False)
                r.raise_for_status()
                text = r.text[:200000]
        if not text:
            return jsonify({"error": "no content"}), 400

        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        out = EMBED_DIR / f"{ts}.txt"
        out.write_text(text, encoding="utf-8")

        return jsonify({"result": "stored", "path": str(out)}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ----- Workflow (iterative with validation) -----

def _write(path: Path, text: str) -> Path:
    path.write_text(text or "", encoding="utf-8")
    return path

def _append_journal(name: str, line: str) -> None:
    f = JOURNALS_DIR / f"{name}.journal"
    with f.open("a", encoding="utf-8") as fp:
        fp.write(line.strip() + "\\n\\n")

def _coverity_chat(user_text: str, system_text: Optional[str] = None,
                   max_tokens: int = 800, inference_profile_arn: Optional[str] = None,
                   url: Optional[str] = None, token: Optional[str] = None) -> str:
    u = (url or COVERITY_ASSIST_URL).rstrip("/")
    tok = token or COVERITY_ASSIST_TOKEN
    payload: Dict[str, Any] = {"messages": [{"role": "user", "content": user_text}], "max_tokens": max_tokens}
    if system_text:
        payload["system"] = system_text
    if inference_profile_arn:
        payload["inference_profile_arn"] = inference_profile_arn
    r = session.post(u, headers={"Authorization": f"Bearer {tok}", "Content-Type": "application/json"}, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    return data.get("content") or data.get("response") or data.get("text") or json.dumps(data)

def _filter_csv_lines(text: str) -> str:
    lines = [ln for ln in (text or "").splitlines() if ln.strip()]
    filtered = []
    for ln in lines:
        lower = ln.lower()
        if "tool/resource" in lower and "url" in lower:
            continue
        if lower.startswith(("tool,", "resource,", "tool/resource,")):
            continue
        filtered.append(ln)
    return "\\n".join(filtered)

def _generate_resources(task_text: str, chat_url: Optional[str], token: Optional[str],
                        inference_profile_arn: Optional[str]) -> Path:
    system = "You plan technical workflows. Produce precise, safe, reproducible steps."
    user = "Task:\\n" + task_text + "\\n\\nReturn ONLY CSV rows, no prose, columns:\\n" + \
           "Tool/Resource,Specific info needed,Required bash command (if any),URL (if any)\\n" + \
           "If a column is N/A, put '-'."
    text = _coverity_chat(user, system_text=system, max_tokens=700,
                          inference_profile_arn=inference_profile_arn,
                          url=chat_url, token=token)
    text = _filter_csv_lines(text)
    p = INSTRUCTIONS_DIR / "workflow.resources"
    _write(p, text)
    return p

def _run_commands(resources_path: Path) -> Path:
    data_path = INSTRUCTIONS_DIR / "workflow.data"
    lines = [ln.strip() for ln in resources_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    with data_path.open("w", encoding="utf-8") as out:
        for ln in lines:
            cols = [c.strip() for c in ln.split(",")]
            while len(cols) < 4:
                cols.append("-")
            cmd, url = cols[2], cols[3]

            if url and url not in ("-", "N/A", "NA", "None", "URL (if any)"):
                try:
                    r = session.get(url, timeout=30, verify=False, headers=BASE_HEADERS, proxies=_get_proxies())
                    body = r.text[:20000]
                    out.write(f"URL: {url}\\n{body}\\n\\n")
                except Exception as e:
                    out.write(f"URL: {url}\\nERROR: {e}\\n\\n")

            if cmd and cmd not in ("-", "N/A", "NA", "None", "Required bash command (if any)"):
                try:
                    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=180)
                    out.write(f"CMD: {cmd}\\nRET={proc.returncode}\\nSTDOUT:\\n{proc.stdout}\\nSTDERR:\\n{proc.stderr}\\n\\n")
                except Exception as e:
                    out.write(f"CMD: {cmd}\\nERROR: {e}\\n\\n")
    return data_path

def _summarize(data_path: Path, chat_url: Optional[str], token: Optional[str],
               inference_profile_arn: Optional[str]) -> str:
    system = "You are a senior engineer. Summarize tersely with metrics and concrete next steps."
    user = "Summarize the following findings into 6–10 bullets with concrete metrics and next steps:\\n\\n" + data_path.read_text(encoding="utf-8")[:28000]
    try:
        return _coverity_chat(user, system_text=system, max_tokens=600,
                              inference_profile_arn=inference_profile_arn,
                              url=chat_url, token=token)
    except Exception as e:
        return f"(summary failed: {e})"

def _validate_with_coverity(original_request: str, summary_text: str,
                            chat_url: Optional[str], token: Optional[str],
                            inference_profile_arn: Optional[str]):
    system = "Return STRICT JSON only. No prose."
    # Avoid f-string brace issues by using .format on a template with numbered placeholders.
    template = (
        "Original request:\\n---\\n{0}\\n---\\n\\n"
        "Did the summary below fully satisfy the request? If not, list concrete next actions (bash commands or URLs) we should run/fetch next.\\n\\n"
        "Summary:\\n---\\n{1}\\n---\\n\\n"
        "Respond JSON with keys: \"complete\": true|false, "
        "\"next_actions\": [ {{\"cmd\": \"...\"}}, {{\"url\": \"...\"}} ]"
    )
    user = (
            "Original request:\n---\n" + original_request + "\n---\n\n"
                                                            "Did the summary below fully satisfy the request? If not, list concrete next actions "
                                                            "(bash commands or URLs) we should run/fetch next.\n\n"
                                                            "Summary:\n---\n" + summary_text + "\n---\n\n"
                                                                                               'Respond JSON with keys: "complete": true|false, '
                                                                                               '"next_actions": [ {"cmd": "..."}, {"url": "..."} ]'
    )

    raw = _coverity_chat(user, system_text=system, max_tokens=300,
                         inference_profile_arn=inference_profile_arn,
                         url=chat_url, token=token)
    try:
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("not an object")
        if "complete" not in data:
            data["complete"] = False
        if "next_actions" not in data or not isinstance(data["next_actions"], list):
            data["next_actions"] = []
        return data
    except Exception:
        return {"complete": False, "next_actions": []}

def _append_actions_to_resources(actions, resources_path: Path) -> None:
    rows = []
    for a in actions or []:
        if not isinstance(a, dict):
            continue
        if a.get("cmd"):
            rows.append(f"Shell,-,{a['cmd']},-")
        elif a.get("url"):
            rows.append(f"Web,-,-,{a['url']}")
    if rows:
        with resources_path.open("a", encoding="utf-8") as f:
            for r in rows:
                f.write(r + "\\n")

def _bundle_files() -> str:
    zf = INSTRUCTIONS_DIR / "workflow_bundle.zip"
    with zipfile.ZipFile(zf, "w", zipfile.ZIP_DEFLATED) as z:
        for name in ("workflow.instruct", "workflow.resources", "workflow.data"):
            p = INSTRUCTIONS_DIR / name
            if p.exists():
                z.write(p, arcname=name)
    return str(zf)

@app.post("/trigger-workflow")
def trigger_workflow():
    data = request.get_json(silent=True) or {}
    original_request = (data.get("original_request") or "").strip()
    task_description  = (data.get("task_description")  or "").strip()
    chat_url          = data.get("chat_url")
    token             = data.get("token")
    inference_profile_arn = data.get("inference_profile_arn")
    auto_iterate      = bool(data.get("auto_iterate", True))
    max_iters         = int(data.get("max_iters", 2))

    if not original_request or not task_description:
        return jsonify({"error": "original_request and task_description are required"}), 400

    instruct_path = INSTRUCTIONS_DIR / "workflow.instruct"
    _write(instruct_path, task_description)

    iterations = []
    final_summary = ""

    for i in range(1, max_iters + 1):
        res_path  = _generate_resources(task_description, chat_url, token, inference_profile_arn)
        data_path = _run_commands(res_path)
        final_summary = _summarize(data_path, chat_url, token, inference_profile_arn)
        verdict = _validate_with_coverity(original_request, final_summary, chat_url, token, inference_profile_arn)

        iterations.append({
            "iteration": i,
            "resources_preview": res_path.read_text(encoding="utf-8")[:2000],
            "data_preview": data_path.read_text(encoding="utf-8")[:2000],
            "summary": final_summary,
            "validation": verdict,
        })

        _append_journal("gabriel", f"[WF] Iter {i}: complete={verdict.get('complete')} next={len(verdict.get('next_actions', []))}")

        if not auto_iterate or verdict.get("complete") is True:
            break

        # Prepare next loop by appending actions
        _append_actions_to_resources(verdict.get("next_actions"), res_path)
        task_description = task_description + "\\n\\nFollow-up actions:\\n" + "\\n".join(
            f"- {a.get('cmd') or a.get('url')}" for a in (verdict.get("next_actions") or [])
        )

    bundle_path = _bundle_files()
    return jsonify({"status": "ok", "bundle_path": bundle_path, "iterations": iterations, "final_summary": final_summary}), 200

# ---------- Main ----------

if __name__ == "__main__":
    # IMPORTANT: debug=False and use_reloader=False to prevent restarts on non-code files
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)

