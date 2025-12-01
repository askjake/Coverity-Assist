#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Coverity Assist Gateway (v3c, 2025-11-06)
- Python 3.7+ compatible (no .removeprefix)
- Stable endpoints (health, config/search, credentials, journal, web-search, embed-content, trigger-workflow)
- Robust DuckDuckGo fallback (lite + html variants)
- Accepts JSON OR form bodies where sensible (prevents 500s from strict form expectations)
- Persists search config in STATE_DIRECTORY/state/search_config.json (or ./state/search_config.json)
- Optional keyring-backed secret storage
"""
from __future__ import annotations
import os
VERIFY_SSL = os.environ.get("GATEWAY_VERIFY_SSL", "0").lower() not in ("0", "false", "no", "off")

if not VERIFY_SSL:
    try:
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    except Exception:
        pass


import re
import json
import zipfile
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from urllib.parse import urlparse
import certifi
import os
os.environ['SSL_CERT_FILE'] = certifi.where()

import requests
from flask import Flask, request, jsonify
from werkzeug.exceptions import BadRequest

VERSION = "v3c-2025-11-06"

# ---------------- Paths & static config ----------------

COVERITY_ASSIST_URL   = os.environ.get("COVERITY_ASSIST_URL", "http://coverity-assist.dishtv.technology/chat").rstrip("/")
COVERITY_ASSIST_TOKEN = os.environ.get("COVERITY_ASSIST_TOKEN", "")
STATE_DIR             = os.environ.get("STATE_DIRECTORY")

ROOT             = Path(os.environ.get("JAMBOT_ROOT") or Path.cwd())
INSTRUCTIONS_DIR = Path(os.environ.get("INSTRUCTIONS_DIR") or str((Path(STATE_DIR) if STATE_DIR else ROOT) / "instructions"))
EMBED_DIR        = Path(os.environ.get("EMBED_DIR")        or str((Path(STATE_DIR) if STATE_DIR else ROOT) / "embedded"))
JOURNALS_DIR     = Path(os.environ.get("JOURNALS_DIR")     or str((Path(STATE_DIR) if STATE_DIR else ROOT) / "journals"))

CONFIG_DIR = Path(STATE_DIR) if STATE_DIR else (ROOT / "state")
for d in (CONFIG_DIR, INSTRUCTIONS_DIR, EMBED_DIR, JOURNALS_DIR):
    d.mkdir(parents=True, exist_ok=True)
SEARCH_CFG  = CONFIG_DIR / "search_config.json"

# ---------------- Helpers ----------------
def _extract_error_chunks(data_path: Path, max_chars: int = 4000) -> str:
    """
    Pull out only the command/URL blocks that look like errors.

    A block starts with 'CMD: ' or 'URL: ' and continues until the next such line.
    A block is considered an error if:
      - It contains a line starting with 'ERROR: ', OR
      - It has a 'RET=' line with a non-zero exit code.
    """
    try:
        data_text = data_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""

    blocks: List[str] = []
    current: List[str] = []

    for line in data_text.splitlines():
        if line.startswith("CMD: ") or line.startswith("URL: "):
            if current:
                blocks.append("\n".join(current))
            current = [line]
        else:
            if current:
                current.append(line)
    if current:
        blocks.append("\n".join(current))

    error_blocks: List[str] = []
    for block in blocks:
        lower = block.lower()
        has_error_line = "error:" in lower  # covers both URL + CMD ERROR lines
        nonzero_ret = False
        for ln in block.splitlines():
            if ln.startswith("RET="):
                try:
                    code = int(ln.split("=", 1)[1].strip())
                    if code != 0:
                        nonzero_ret = True
                except Exception:
                    pass
        if has_error_line or nonzero_ret:
            error_blocks.append(block)

    if not error_blocks:
        return ""

    snippet = "\n\n---\n\n".join(error_blocks)

    # Keep last N chars; most recent attempts are usually at the end
    if len(snippet) > max_chars:
        snippet = snippet[-max_chars:]

    return snippet

def _strip_www(host: str) -> str:
    host = (host or "").lower().lstrip(".")
    return host[4:] if host.startswith("www.") else host

def _norm_host(s: str) -> str:
    try:
        if "://" in s:
            return _strip_www(urlparse(s).netloc)
    except Exception:
        pass
    return _strip_www(s or "")

def _read_json(p: Path, default: dict) -> dict:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return dict(default)

def _write_json(p: Path, data: dict) -> None:
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")
    try:
        os.chmod(p, 0o600)
    except Exception:
        pass

def _domain_of(url: str) -> str:
    try:
        return _strip_www(urlparse(url).netloc)
    except Exception:
        return ""

def _try_import(name: str):
    try:
        return __import__(name)
    except Exception:
        return None

# Optional secure storage
keyring = _try_import("keyring")

# In-memory secrets fallback (not persisted)
_MEMORY_SECRETS: Dict[str, Dict[str, Any]] = {
    "basic": {},   # domain -> {"username":..., "password":...}
    "cookies": {}, # domain -> {"Cookie":"a=b; ..."} or {"cname":"cval"}
    "headers": {}, # domain -> {"Header":"Value"}
    "proxy":  {},  # {"http":"...", "https":"..."}
}

def _kr_key(domain: str, kind: str) -> str:
    return f"coverity_gateway::{kind}::{domain}"

def _set_secret(domain: str, kind: str, data: Dict[str, Any], persist: bool) -> None:
    dom = _norm_host(domain)
    if persist and keyring is not None:
        try:
            keyring.set_password("coverity_gateway", _kr_key(dom, kind), json.dumps(data))
            return
        except Exception:
            pass
    _MEMORY_SECRETS.setdefault(kind, {})[dom] = data

def _get_secret(domain: str, kind: str) -> Optional[Dict[str, Any]]:
    dom = _norm_host(domain)
    if keyring is not None:
        try:
            raw = keyring.get_password("coverity_gateway", _kr_key(dom, kind))
            if raw:
                return json.loads(raw)
        except Exception:
            pass
    return _MEMORY_SECRETS.get(kind, {}).get(dom)

def _get_proxies() -> Dict[str, str]:
    # env wins
    proxies = {}
    for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        if os.environ.get(k):
            proxies[k.split("_")[0].lower()] = os.environ.get(k)  # type: ignore
    # explicit
    p = _MEMORY_SECRETS.get("proxy") or {}
    proxies.update({k: v for k, v in p.items() if v})
    return proxies

# ---------------- Search config (single source) ----------------

DEFAULT_SEARCH_CFG = {
    "mode": "both",                 # "local" | "www" | "both"
    "allowlist": [],                # hostnames only
    "blocklist": ["linkedin.com","zoominfo.com","x.com","twitter.com"],
    "use_credentials": False,
    "fetch_pages": True,
    "max_results": 6,
}

def load_search_cfg() -> dict:
    cfg = _read_json(SEARCH_CFG, DEFAULT_SEARCH_CFG)
    # Normalize lists
    cfg["allowlist"] = [_norm_host(x) for x in (cfg.get("allowlist") or []) if x]
    cfg["blocklist"] = [_norm_host(x) for x in (cfg.get("blocklist") or []) if x]
    # Clamp keys
    out = {**DEFAULT_SEARCH_CFG}
    for k in out:
        if k in cfg:
            out[k] = cfg[k]
    return out

def save_search_cfg(cfg: dict) -> dict:
    merged = {**load_search_cfg(), **cfg}
    merged["allowlist"] = [_norm_host(x) for x in (merged.get("allowlist") or []) if x]
    merged["blocklist"] = [_norm_host(x) for x in (merged.get("blocklist") or []) if x]
    _write_json(SEARCH_CFG, {k: merged[k] for k in DEFAULT_SEARCH_CFG})
    return load_search_cfg()

# ---------------- HTTP/session ----------------

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118 Safari/537.36"
BASE_HEADERS = {"User-Agent": UA, "Accept-Language": "en-US,en;q=0.9"}
session = requests.Session()

# ---------------- Flask app ----------------

app = Flask(__name__)

@app.errorhandler(BadRequest)
def _bad_json(e: BadRequest):
    # Always return JSON, even on parser errors
    return jsonify({"error": "bad request", "detail": str(e)}), 400

@app.get("/health")
def health():
    cfg = load_search_cfg()
    return jsonify({"status": "OK", "gateway": True, "mode": cfg.get("mode"), "version": VERSION}), 200

# ---- Config: search (GET/POST) ----

@app.get("/config/search")
def get_search_config():
    return jsonify(load_search_cfg()), 200

@app.post("/config/search")
def set_search_config():
    body = request.get_json(silent=True) or {}
    # Only accept known keys
    payload = {k: body[k] for k in DEFAULT_SEARCH_CFG if k in body}
    stored = save_search_cfg(payload)
    return jsonify({"status": "ok", "config": stored}), 200

# ---- Config: credentials (POST) ----

@app.post("/credentials")
def set_credentials():
    """
    Body:
      {
        "domain": "example.com" | "*" (proxy only),
        "kind": "cookies" | "basic" | "headers" | "proxy",
        "data": {...},     # cookies: {"Cookie":"a=b; ..."} or cookie dict
                           # basic:   {"username":"...", "password":"..."}
                           # headers: {"X-...":"..."}
                           # proxy:   {"http":"http://user:pass@proxy:8080","https":"http://user:pass@proxy:8443"}
        "persist": true|false   # keyring if available
      }
    """
    data = request.get_json(silent=True) or {}
    kind = (data.get("kind") or "").lower()
    obj  = data.get("data") or {}
    persist = bool(data.get("persist", False))

    if kind not in {"cookies","basic","headers","proxy"}:
        return jsonify({"error": "kind must be cookies|basic|headers|proxy"}), 400

    if kind == "proxy":
        _MEMORY_SECRETS["proxy"] = {k: obj.get(k) for k in ("http","https") if obj.get(k)}
        return jsonify({"ok": True, "note": "proxy configured"}), 200

    domain = (data.get("domain") or "").strip()
    if not domain:
        return jsonify({"error": "domain required for non-proxy credentials"}), 400

    _set_secret(domain, kind, obj, persist)
    return jsonify({"ok": True}), 200

# ---------------- Journals ----------------

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
    jf = JOURNALS_DIR / "gabriel.journal"
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Accept JSON OR form
    if request.is_json:
        body = (request.get_json(silent=True) or {}).get("entry") or f"Auto-entry at {stamp}."
    else:
        body = (request.form.get("entry") or f"Auto-entry at {stamp}.")
    with jf.open("a", encoding="utf-8") as f:
        f.write(str(body).strip() + "\n\n")
    return jsonify({"status": "ok", "appended": body}), 200

# ---------------- Web Search ----------------

def _host_allowed(url: str, allowlist: List[str], blocklist: List[str]) -> bool:
    try:
        host = _strip_www(urlparse(url).netloc)
    except Exception:
        return False
    if blocklist and any(host.endswith(b) or host == b for b in blocklist):
        return False
    if allowlist and not any(host.endswith(a) or host == a for a in allowlist):
        return False
    return True

def _build_request_kwargs(url: str, use_credentials: bool) -> Dict[str, Any]:
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)", "Accept-Language": "en-US,en;q=0.9"}
    auth = None
    proxies = _get_proxies()
    if use_credentials:
        host = _strip_www(urlparse(url).netloc)
        b = _get_secret(host, "basic")
        if b and isinstance(b, dict) and b.get("username") and b.get("password"):
            from requests.auth import HTTPBasicAuth
            auth = HTTPBasicAuth(b["username"], b["password"])
        c = _get_secret(host, "cookies")
        if c and isinstance(c, dict) and c.get("Cookie"):
            headers["Cookie"] = c["Cookie"]
        h = _get_secret(host, "headers")
        if h and isinstance(h, dict):
            headers.update({k: str(v) for k, v in h.items()})
    return {"headers": headers, "auth": auth, "proxies": proxies, "timeout": 20}

def _ddg_lite(query: str, max_results: int = 6, timeout: int = 15):
    url = "https://lite.duckduckgo.com/lite/"
    r = session.get(url, params={"q": query}, headers=BASE_HEADERS, timeout=timeout, verify=VERIFY_SSL)
    r.raise_for_status()
    html_text = r.text
    rows = re.findall(
        r"<tr[^>]*>\s*<td[^>]*>\s*<a[^>]+class=['\"]result-link['\"][^>]+href=['\"](.*?)['\"][^>]*>(.*?)</a>.*?<td[^>]*>(.*?)</td>",
        html_text, flags=re.S | re.I
    )
    import html as _html
    out = []
    for href, title_html, snippet_html in rows[:max_results]:
        title = _html.unescape(re.sub("<.*?>", "", title_html)).strip() or "(no title)"
        snippet = _html.unescape(re.sub("<.*?>", "", snippet_html)).strip()
        out.append({"title": title, "url": href, "snippet": snippet})
    return ("ddg-lite", out)

def _ddg_html(query: str, max_results: int = 6, timeout: int = 15):
    url = "https://html.duckduckgo.com/html/"
    r = session.get(url, params={"q": query}, headers=BASE_HEADERS, timeout=timeout, verify=VERIFY_SSL)
    r.raise_for_status()
    html_text = r.text
    import html as _html
    anchors = re.findall(r"<a[^>]+class=['\"]result__a['\"][^>]+href=['\"](.*?)['\"][^>]*>(.*?)</a>", html_text, flags=re.S|re.I)
    snippets = re.findall(r"class=['\"]result__snippet['\"][^>]*>(.*?)</", html_text, flags=re.S|re.I)
    out = []
    for idx, (href, title_html) in enumerate(anchors[:max_results]):
        title = _html.unescape(re.sub("<.*?>", "", title_html)).strip() or "(no title)"
        snippet_html = snippets[idx] if idx < len(snippets) else ""
        snippet = _html.unescape(re.sub("<.*?>", "", snippet_html)).strip()
        out.append({"title": title, "url": href, "snippet": snippet})
    return ("ddg-html", out)

@app.post("/web-search")
def web_search():
    try:
        try:
            data = request.get_json(force=True) or {}
        except BadRequest:
            data = {}  # tolerate empty/invalid JSON
        cfg   = load_search_cfg()

        q            = (data.get("query") or "").strip()
        mode         = (data.get("mode") or cfg.get("mode","both")).lower()
        allow        = [_norm_host(x) for x in data.get("allowlist", cfg.get("allowlist", []))]
        block        = [_norm_host(x) for x in data.get("blocklist", cfg.get("blocklist", []))]
        fetch_pages  = bool(data.get("fetch_pages", cfg.get("fetch_pages", True)))
        max_results  = int(data.get("max_results", cfg.get("max_results", 6)))
        use_creds    = bool(data.get("use_credentials", cfg.get("use_credentials", False)))

        if not q:
            return jsonify({"response": {"query": q, "results": [], "note": "empty query"}}), 200

        # 1) local helper module (optional, if present in working dir)
        results: List[Dict[str, str]] = []
        source = ""
        local_mod = _try_import("web_search")
        if mode in ("local","both") and local_mod and hasattr(local_mod, "do_web_search"):
            try:
                local = local_mod.do_web_search(q, max_results=max_results, fetch_pages=fetch_pages)
                if isinstance(local, dict) and "results" in local:
                    results.extend(local["results"])
                    source = "local"
            except Exception:
                pass

        # 2) WWW via DDG-lite; fallback to html variant
        if (mode in ("www","both")) and len(results) < max_results:
            try:
                source, ddg = _ddg_lite(q, max_results=max_results)
            except Exception:
                source, ddg = ("ddg-html", [])
            if not ddg:
                try:
                    source, ddg = _ddg_html(q, max_results=max_results)
                except Exception:
                    ddg = []
            results.extend(ddg)

        # 3) Filter
        if block:
            results = [r for r in results if r.get("url") and _host_allowed(r["url"], [], block)]
        if allow:
            results = [r for r in results if r.get("url") and _host_allowed(r["url"], allow, [])]

        # 4) Optional fetch of page text
        enriched: List[Dict[str, Any]] = []
        for r0 in results[:max_results]:
            r1 = dict(r0)
            if fetch_pages and r1.get("url") and _host_allowed(r1["url"], allow, block):
                try:
                    kwargs = _build_request_kwargs(r1["url"], use_credentials=use_creds)
                    with session.get(r1["url"], stream=True, **kwargs) as resp:
                        resp.raise_for_status()
                        buf, size = [], 0
                        enc = resp.encoding or "utf-8"
                        for chunk in resp.iter_content(chunk_size=4096):
                            if not chunk: break
                            buf.append(chunk); size += len(chunk)
                            if size >= 200_000: break
                    html = b"".join(buf).decode(enc, errors="replace")
                    try:
                        from bs4 import BeautifulSoup
                        soup = BeautifulSoup(html, "html.parser")
                        for t in soup(["script","style","noscript"]): t.decompose()
                        text = soup.get_text(separator="\n")
                    except Exception:
                        import re as _re
                        text = _re.sub("<[^>]+>", " ", html)
                    text = re.sub(r"\n{3,}", "\n\n", text)
                    text = re.sub(r"[ \t]{2,}", " ", text).strip()
                    r1["text"] = text[:6000]
                except Exception as e:
                    r1["text"] = f"(fetch error: {e})"
            enriched.append(r1)

        note = ("local+" if source=="local" else "") + ("ddg-lite" if source=="ddg-lite" else "ddg-html" if source=="ddg-html" else "unknown")
        return jsonify({"response": {"query": q, "results": enriched, "note": note}}), 200

    except Exception as e:
        # never 500; always JSON
        return jsonify({"response": f"Search error: {e}"}), 200

# ---------------- Embed simple fetch ----------------

@app.post("/embed-content")
def embed_content():
    try:
        text = ""
        filename = None

        if request.is_json:
            data = request.get_json(silent=True) or {}
            if data.get("text"):
                text = str(data["text"])
            elif data.get("generic_url"):
                r = session.get(str(data["generic_url"]), timeout=30, verify=False)
                r.raise_for_status()
                text = r.text[:200000]
            filename = data.get("filename")
        else:
            file = request.files.get("file")
            if file and hasattr(file, "read"):
                text = file.read().decode("utf-8", errors="replace")
                filename = getattr(file, "filename", None)
            else:
                generic_url = request.form.get("generic_url")
                if generic_url:
                    r = session.get(generic_url, timeout=30, verify=False)
                    r.raise_for_status()
                    text = r.text[:200000]

        if not text:
            return jsonify({"error": "no content"}), 400

        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        base = filename or f"{ts}.txt"
        out = EMBED_DIR / base
        # Ensure unique
        if out.exists():
            out = EMBED_DIR / f"{ts}-{base}"
        out.write_text(text, encoding="utf-8")
        return jsonify({"result": "stored", "path": str(out)}), 200
    except Exception as e:
        return jsonify({"error": f"embed failed: {e}"}), 500

# ---------------- Workflow support ----------------

def _write(path: Path, text: str) -> Path:
    path.write_text(text or "", encoding="utf-8")
    return path

def _append_journal(name: str, line: str) -> None:
    f = JOURNALS_DIR / f"{name}.journal"
    with f.open("a", encoding="utf-8") as fp:
        fp.write(line.strip() + "\n\n")

def _coverity_chat(user_text: str, system_text: Optional[str] = None,
                   max_tokens: int = 4000, inference_profile_arn: Optional[str] = None,
                   url: Optional[str] = None, token: Optional[str] = None) -> str:
    u = (url or COVERITY_ASSIST_URL).rstrip("/")
    tok = token or COVERITY_ASSIST_TOKEN
    payload: Dict[str, Any] = {"messages": [{"role": "user", "content": user_text}], "max_tokens": max_tokens}
    if system_text:
        payload["system"] = system_text
    if inference_profile_arn:
        payload["inference_profile_arn"] = inference_profile_arn
    r = session.post(u, headers={"Authorization": f"Bearer {tok}", "Content-Type": "application/json"}, json=payload, timeout=599, verify=VERIFY_SSL)
    r.raise_for_status()
    data = r.json()
    return data.get("content") or data.get("response") or data.get("text") or json.dumps(data)

def _filter_csv_lines(text: str) -> str:
    lines = [ln for ln in (text or "").splitlines() if ln.strip()]
    filtered = []
    for ln in lines:
        lower = ln.lower()
        if "tool/resource" in lower and "url" in lower: continue
        if lower.startswith(("tool,", "resource,", "tool/resource,")): continue
        filtered.append(ln)
    return "\n".join(filtered)

def _generate_resources(task_text: str, chat_url: Optional[str], token: Optional[str],
                        inference_profile_arn: Optional[str]) -> Path:
    system = "You plan technical workflows. Produce precise, safe, reproducible steps."
    user = "Task:\n" + task_text + "\n\nReturn ONLY CSV rows, no prose, columns:\n" +            "Tool/Resource,Specific info needed,Required bash command (if any),URL (if any)\n" +            "If a column is N/A, put '-'."
    text = _coverity_chat(user, system_text=system, max_tokens=1000,
                          inference_profile_arn=inference_profile_arn,
                          url=chat_url, token=token)
    text = _filter_csv_lines(text)
    p = INSTRUCTIONS_DIR / "workflow.resources"
    _write(p, text)
    return p


def _summarize(data_path: Path, chat_url: Optional[str], token: Optional[str],
               inference_profile_arn: Optional[str]) -> str:
    system = "You are a senior engineer. Summarize tersely with metrics and concrete next steps."
    user = "Summarize the following findings into 6–10 bullets with concrete metrics and next steps:\n\n" + data_path.read_text(encoding="utf-8")[:28000]
    try:
        return _coverity_chat(user, system_text=system, max_tokens=2000,
                              inference_profile_arn=inference_profile_arn,
                              url=chat_url, token=token)
    except Exception as e:
        return f"(summary failed: {e})"

def _validate_with_coverity(
    original_request: str,
    summary_text: str,
    data_path: Path,
    chat_url: Optional[str],
    token: Optional[str],
    inference_profile_arn: Optional[str],
) -> Dict:
    """
    Ask Coverity-Assist if we're done, and if not, what to try next.

    This version:
      * Includes a distilled view of only the failing commands/URLs
      * Asks the model to propose corrected follow-up actions
      * Returns a JSON-like dict with 'complete', 'next_actions', and 'error_snippet'
    """
    error_snippet = _extract_error_chunks(data_path)

    system = (
        "You are validating an automated multi-step technical workflow. "
        "You see the user's original request, a summary of what has been done so far, "
        "and (optionally) a list of commands/URLs that failed. "
        "Return STRICT JSON only. No prose. "
        "If some commands or URLs failed, propose concrete corrected follow-up actions."
    )

    parts = [
        "Original request:\n---\n",
        original_request,
        "\n---\n\nSummary of data gathered so far:\n---\n",
        summary_text,
        "\n---\n\n",
    ]

    if error_snippet:
        parts.extend(
            [
                "The following commands/URLs appear to have errors "
                "(non-zero exit codes, exceptions, or HTTP failures).\n",
                "Use them to suggest corrected follow-up actions. "
                "Avoid blindly repeating obviously broken commands.\n\n---\n",
                error_snippet,
                "\n---\n\n",
            ]
        )

    parts.append(
        'Respond with JSON of the form:\n'
        '{"complete": true|false, '
        ' "next_actions": [ {"cmd": "..."} | {"url": "..."} | {"cmd": "...", "notes": "..."} ] }\n'
        "Only include keys 'complete' and 'next_actions' at the top level."
    )

    user = "".join(parts)

    raw = _coverity_chat(
        user,
        system_text=system,
        max_tokens=750,
        inference_profile_arn=inference_profile_arn,
        url=chat_url,
        token=token,
    )

    try:
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("not an object")

        complete = bool(data.get("complete", False))
        next_actions = data.get("next_actions") or []
        if not isinstance(next_actions, list):
            next_actions = []

        # Attach the error snippet so the UI can show it if you want
        return {
            "complete": complete,
            "next_actions": next_actions,
            "error_snippet": error_snippet,
        }
    except Exception:
        # Fallback: treat as "not complete, nothing new to try"
        return {
            "complete": False,
            "next_actions": [],
            "error_snippet": error_snippet,
        }

def _append_actions_to_resources(actions, resources_path: Path) -> None:
    """
    Append new Shell/Web actions to the workflow.resources CSV
    so the next iteration will run them.

    Expected shapes inside 'actions':
      - {"cmd": "..."} for shell
      - {"url": "..."} for HTTP
    """
    rows = []
    for a in actions or []:
        if not isinstance(a, dict):
            continue
        if a.get("cmd") and isinstance(a.get("cmd"), str):
            rows.append(f"Shell,-,{a['cmd']},-")
        elif a.get("url") and isinstance(a.get("url"), str):
            rows.append(f"Web,-,-,{a['url']}")
    if rows:
        with resources_path.open("a", encoding="utf-8") as f:
            for r in rows:
                f.write(r + "\n")


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
    original_request   = (data.get("original_request") or "").strip()
    task_description   = (data.get("task_description")  or "").strip()
    chat_url           = data.get("chat_url")
    token              = data.get("token")
    inference_profile_arn = data.get("inference_profile_arn")
    auto_iterate       = bool(data.get("auto_iterate", True))
    max_iters          = int(data.get("max_iters", 2))

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
        verdict = _validate_with_coverity(
            original_request,
            final_summary,
            data_path,  # <-- NEW
            chat_url,
            token,
            inference_profile_arn,
        )

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

        _append_actions_to_resources(verdict.get("next_actions"), res_path)
        task_description = task_description + "\n\nFollow-up actions:\n" + "\n".join(
            f"- {a.get('cmd') or a.get('url')}" for a in (verdict.get("next_actions") or [])
        )

    bundle_path = _bundle_files()
    return jsonify({"status": "ok", "bundle_path": bundle_path, "iterations": iterations, "final_summary": final_summary}), 200
def _run_commands(resources_path: Path) -> Path:
    data_path = INSTRUCTIONS_DIR / "workflow.data"
    lines = [ln.strip() for ln in resources_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    with data_path.open("w", encoding="utf-8") as out:
        for ln in lines:
            cols = [c.strip() for c in ln.split(",")]
            while len(cols) < 4: cols.append("-")
            cmd, url = cols[2], cols[3]

            if url and url not in ("-", "N/A", "NA", "None", "URL (if any)"):
                try:
                    r = session.get(url, timeout=30, verify=False, headers=BASE_HEADERS, proxies=_get_proxies())
                    body = r.text[:20000]
                    out.write(f"URL: {url}\n{body}\n\n")
                except Exception as e:
                    out.write(f"URL: {url}\nERROR: {e}\n\n")

            if cmd and cmd not in ("-", "N/A", "NA", "None", "Required bash command (if any)"):
                try:
                    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=180)
                    out.write(f"CMD: {cmd}\nRET={proc.returncode}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}\n\n")
                except Exception as e:
                    out.write(f"CMD: {cmd}\nERROR: {e}\n\n")
    return data_path

# ---------------- Main ----------------

def _parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Coverity Assist Gateway")
    parser.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "5000")))
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    # IMPORTANT: use_reloader=False to prevent restarts when instructions/ZIP change
    app.run(host=args.host, port=args.port, debug=args.debug, use_reloader=False)
