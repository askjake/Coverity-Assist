#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Coverity Assist Gateway (patched currentUser→accountId, TLS verify) (v3g) — Python 3.8+
Adds:
- OAuth/Bearer (PAT) auth for Jira Cloud/DC
- Cookie-free Jira requests to avoid XSRF 403s
- /jira/search/debug endpoint
- Minimal HTML form at /auth to paste email+API token OR Bearer token (PAT/OAuth)
- /jira/auth-check endpoint to confirm auth works and show which method is in use

Notes:
- For Atlassian Cloud: "API token" + email uses Basic auth (recommended).
- For Jira Data Center PATs or OAuth access tokens: use Bearer.
- We never forward cookies to Jira; we also set allow_redirects=False for Jira requests.
"""

import os
import re
import html
import json
import zipfile
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from urllib.parse import urlparse

import requests
from flask import Flask, request, jsonify, Response

# ----------------------- Settings & Paths -----------------------

PORT = int(os.environ.get("PORT", "5002"))
STATE_DIR = os.environ.get("STATE_DIRECTORY") or str(Path.cwd() / "state")
ROOT = Path.cwd()

COVERITY_ASSIST_URL = (os.environ.get("COVERITY_ASSIST_URL") or "http://coverity-assist.dishtv.technology/chat").rstrip("/")
COVERITY_ASSIST_TOKEN = os.environ.get("COVERITY_ASSIST_TOKEN", "")

VERIFY_SSL = os.environ.get("GATEWAY_VERIFY_SSL", "true").lower() not in ("0", "false", "no")

INSTRUCTIONS_DIR = Path(os.environ.get("INSTRUCTIONS_DIR") or (Path(STATE_DIR) / "instructions"))
EMBED_DIR        = Path(os.environ.get("EMBED_DIR")        or (Path(STATE_DIR) / "embedded"))
JOURNALS_DIR     = Path(os.environ.get("JOURNALS_DIR")     or (Path(STATE_DIR) / "journals"))
CFG_DIR          = Path(STATE_DIR)
for d in (INSTRUCTIONS_DIR, EMBED_DIR, JOURNALS_DIR, CFG_DIR):
    d.mkdir(parents=True, exist_ok=True)

SEARCH_CFG = CFG_DIR / "search_config.json"

# ----------------------- Utilities -----------------------
def _get_account_id(domain, kwargs):
    """Return Jira Cloud accountId (or None)."""
    base = f"https://{domain}".rstrip("/")
    r = _jira_http("GET", f"{base}/rest/api/3/myself", **kwargs)
    if r.ok and "application/json" in (r.headers.get("Content-Type", "")):
        try:
            return r.json().get("accountId")
        except Exception:
            return None
    return None

def _rewrite_current_user(jql, account_id):
    """Replace any currentUser() occurrences with the concrete accountId."""
    if not account_id or not isinstance(jql, str) or "currentUser" not in jql:
        return jql
    # Covers currentUser() with/without spaces
    return re.sub(r"currentUser\s*\(\s*\)", account_id, jql)

def _read_json(p: Path, default: dict) -> dict:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return dict(default)

def _write_json(p: Path, data: dict) -> None:
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")
    try:
        os.chmod(str(p), 0o600)
    except Exception:
        pass

def _strip_www(host: str) -> str:
    host = (host or "").lower().lstrip(".")
    if host.startswith("www."):
        return host[4:]
    return host

def _norm_host(s: str) -> str:
    try:
        if "://" in (s or ""):
            return _strip_www(urlparse(s).netloc)
    except Exception:
        pass
    return _strip_www(s or "")

def _domain_of(url: str) -> str:
    try:
        return _strip_www(urlparse(url).hostname or "")
    except Exception:
        return ""

def _looks_like_url(s: str) -> bool:
    return isinstance(s, str) and s.lower().startswith(("http://", "https://"))

def _try_import(name: str):
    try:
        return __import__(name)
    except Exception:
        return None

# Optional secure storage
keyring = _try_import("keyring")

# ----------------------- Global Config -----------------------

DEFAULT_SEARCH_CFG = {
    "mode": "www",  # "local" | "www" | "both"
    "allowlist": [],  # hostnames only
    "blocklist": ["linkedin.com", "zoominfo.com", "x.com", "twitter.com"],
    "use_credentials": True,
    "fetch_pages": True,
    "max_results": 6,
}

def load_search_cfg() -> dict:
    cfg = _read_json(SEARCH_CFG, DEFAULT_SEARCH_CFG)
    out = dict(DEFAULT_SEARCH_CFG)
    for k in out:
        if k in cfg:
            out[k] = cfg[k]
    out["allowlist"] = [_norm_host(x) for x in (out.get("allowlist") or []) if x]
    out["blocklist"] = [_norm_host(x) for x in (out.get("blocklist") or []) if x]
    return out

def save_search_cfg(cfg: dict) -> dict:
    merged = load_search_cfg()
    for k in DEFAULT_SEARCH_CFG:
        if k in cfg:
            merged[k] = cfg[k]
    merged["allowlist"] = [_norm_host(x) for x in (merged.get("allowlist") or []) if x]
    merged["blocklist"] = [_norm_host(x) for x in (merged.get("blocklist") or []) if x]
    _write_json(SEARCH_CFG, {k: merged[k] for k in DEFAULT_SEARCH_CFG})
    return load_search_cfg()

# ----------------------- Credentials (with keyring) -----------------------

_MEMORY_SECRETS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "basic": {},     # domain -> {"username":..., "password":...}
    "cookies": {},   # domain -> {"Cookie":"..."} or cookie dict
    "headers": {},   # domain -> {"X-Header":"Value"}
    "oauth": {},     # domain -> {"access_token": "...", "token_type": "Bearer", "expires_at": "..."}  # Jira DC PATs or OAuth access tokens
    "bearer": {},    # alias for oauth (accepts {"token": "..."} or {"access_token": "..."})
    "proxy": {"*": {}},  # {"*": {"http": "...", "https": "..."}}
}

def _kr_key(domain: str, kind: str) -> str:
    return f"coverity_gateway::{kind}::{_norm_host(domain)}"

def _set_secret(domain: str, kind: str, data: Dict[str, Any], persist: bool) -> None:
    domain = _norm_host(domain)
    if kind == "proxy":
        if persist and keyring is not None:
            try:
                keyring.set_password("coverity_gateway", _kr_key("*", "proxy"), json.dumps(data))
                return
            except Exception:
                pass
        _MEMORY_SECRETS["proxy"]["*"] = data
        return

    if persist and keyring is not None:
        try:
            keyring.set_password("coverity_gateway", _kr_key(domain, kind), json.dumps(data))
            return
        except Exception:
            pass
    _MEMORY_SECRETS.setdefault(kind, {})[domain] = data

def _get_secret(domain: str, kind: str) -> Optional[Dict[str, Any]]:
    domain = _norm_host(domain)
    if kind == "proxy":
        if keyring is not None:
            try:
                raw = keyring.get_password("coverity_gateway", _kr_key("*", "proxy"))
                if raw:
                    return json.loads(raw)
            except Exception:
                pass
        return _MEMORY_SECRETS.get("proxy", {}).get("*")

    if keyring is not None:
        try:
            raw = keyring.get_password("coverity_gateway", _kr_key(domain, kind))
            if raw:
                return json.loads(raw)
        except Exception:
            pass
    return _MEMORY_SECRETS.get(kind, {}).get(domain)

def _get_proxies() -> Dict[str, str]:
    proxies = {}
    for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        if os.environ.get(k):
            key = "http" if k.lower().startswith("http_") else "https"
            proxies[key] = os.environ.get(k)  # type: ignore
    p = _get_secret("*", "proxy") or {}
    for k in ("http", "https"):
        if p.get(k):
            proxies[k] = p[k]
    return proxies

# ----------------------- HTTP session -----------------------

session = requests.Session()
session.verify = VERIFY_SSL

BASE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}

def _build_request_kwargs(url: str, use_credentials: bool) -> Dict[str, Any]:
    headers = dict(BASE_HEADERS)
    auth = None
    proxies = _get_proxies()

    if use_credentials:
        host = _norm_host(url)
        # BASIC
        b = _get_secret(host, "basic")
        if b and isinstance(b, dict) and b.get("username") and b.get("password"):
            from requests.auth import HTTPBasicAuth
            auth = HTTPBasicAuth(b["username"], b["password"])
        # CUSTOM HEADERS
        h = _get_secret(host, "headers")
        if h and isinstance(h, dict):
            for k, v in h.items():
                headers[str(k)] = str(v)

    return {"headers": headers, "auth": auth, "proxies": proxies, "timeout": 25}

# ----------------------- Flask app -----------------------

app = Flask(__name__)

@app.get("/health")
def health():
    cfg = load_search_cfg()
    return jsonify({"status": "OK", "gateway": True, "mode": cfg.get("mode"), "verify_ssl": session.verify}), 200

# ---- Config endpoints ----

@app.get("/config/search")
def get_search_config():
    return jsonify(load_search_cfg()), 200

@app.post("/config/search")
def set_search_config():
    body = request.get_json(silent=True) or {}
    payload = {k: body[k] for k in DEFAULT_SEARCH_CFG if k in body}
    stored = save_search_cfg(payload)
    return jsonify({"status": "ok", "config": stored}), 200

@app.post("/credentials")
def set_credentials():
    data = request.get_json(silent=True) or {}
    kind = str(data.get("kind", "")).lower()
    if kind not in {"basic", "cookies", "headers", "proxy", "oauth", "bearer"}:
        return jsonify({"error": "kind must be basic|cookies|headers|proxy|oauth|bearer"}), 400
    domain = data.get("domain", "")
    persist = bool(data.get("persist", False))
    obj = data.get("data") or {}
    if kind != "proxy" and not domain:
        return jsonify({"error": "domain required for non-proxy creds"}), 400
    # normalize bearer payloads
    if kind in {"oauth", "bearer"}:
        tok = obj.get("access_token") or obj.get("token")
        if not tok:
            return jsonify({"error": "oauth/bearer requires 'access_token' or 'token'"}), 400
        obj = {"access_token": tok, "token_type": obj.get("token_type") or "Bearer", "expires_at": obj.get("expires_at")}
        kind = "oauth"
    _set_secret(domain or "*", kind, obj, persist)
    return jsonify({"ok": True}), 200

# ---- Simple credentials form ----

def _mask(s: Optional[str], keep: int = 3) -> str:
    if not s: return ""
    if len(s) <= keep: return "*"*len(s)
    return s[:keep] + "…" + "*" * max(0, len(s)-keep-1)

FORM_HTML = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Coverity Gateway Auth</title>
  <style>
    body{font-family:system-ui,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:2rem;max-width:860px}
    h1{margin:0 0 1rem 0}
    fieldset{margin:1rem 0;padding:1rem;border-radius:12px;border:1px solid #ccc}
    label{display:block;margin:.5rem 0 .25rem;font-weight:600}
    input[type=text],input[type=password]{width:100%;padding:.55rem;border-radius:8px;border:1px solid #bbb;font-size:14px}
    .row{display:grid;grid-template-columns:1fr 1fr;gap:1rem}
    .actions{margin-top:1rem}
    button{padding:.6rem 1rem;border-radius:10px;border:0;background:#0b63ce;color:#fff;font-weight:700;cursor:pointer}
    .note{font-size:12px;color:#555}
    .ok{background:#e8f7ec;border-left:4px solid #2ca24d;padding:.5rem 1rem;margin:1rem 0}
    .err{background:#fde8ea;border-left:4px solid #d33;padding:.5rem 1rem;margin:1rem 0}
    .card{background:#f8f9fb;border:1px solid #e4e7ee;border-radius:12px;padding:1rem}
    code{background:#eef2f7;border-radius:6px;padding:.1rem .3rem}
  </style>
</head>
<body>
  <h1>Coverity Gateway — Jira Auth</h1>
  <p class="note">Paste credentials locally on this box. We store them in memory or your OS keyring if available. No cookies are sent to Atlassian.</p>

  <div class="card">
    <h3>Current Auth Snapshot</h3>
    <div id="snapshot">loading…</div>
  </div>

  <form method="post" action="/auth">
    <fieldset>
      <legend>Basic (Email + API token)</legend>
      <div class="row">
        <div>
          <label>Jira Domain</label>
          <input name="domain_basic" type="text" placeholder="dishtech-dishtv.atlassian.net" required>
        </div>
        <div>
          <label>Email</label>
          <input name="email" type="text" placeholder="your.name@company.com">
        </div>
      </div>
      <label>API Token</label>
      <input name="api_token" type="password" placeholder="Paste Atlassian API token">
      <div class="actions"><button name="action" value="save_basic">Save Basic</button></div>
    </fieldset>

    <fieldset>
      <legend>Bearer (OAuth access token or Data Center PAT)</legend>
      <div class="row">
        <div>
          <label>Jira Domain</label>
          <input name="domain_bearer" type="text" placeholder="dishtech-dishtv.atlassian.net">
        </div>
        <div>
          <label>Access Token</label>
          <input name="bearer_token" type="password" placeholder="Paste OAuth/PAT token">
        </div>
      </div>
      <div class="actions"><button name="action" value="save_bearer">Save Bearer</button></div>
    </fieldset>
  </form>

  <script>
  async function snap(){
    try {
      const r = await fetch('/jira/auth-check?domain=' + encodeURIComponent(document.querySelector('input[name=domain_basic]').value || 'dishtech-dishtv.atlassian.net'));
      const j = await r.json();
      const hint = j.auth_hint ? ('<b>kind:</b> '+j.auth_hint.auth_kind+'; <b>has:</b> '+j.auth_hint.has_auth+'; <b>user:</b> '+(j.auth_hint.user_hint||'')) : 'n/a';
      document.getElementById('snapshot').innerHTML =
        '<div><b>HTTP:</b> '+j.status+'; <b>Content-Type:</b> '+(j.content_type||'')+'</div>' +
        '<div><b>Auth:</b> '+hint+'</div>' +
        '<div><b>Body sample:</b> <code>'+ (j.body_sample||'').toString().slice(0,180) +'</code></div>';
    } catch(e) {
      document.getElementById('snapshot').innerHTML = '<div class="err">'+e+'</div>';
    }
  }
  snap();
  </script>
</body>
</html>
"""

@app.get("/auth")
def auth_form():
    return Response(FORM_HTML, mimetype="text/html")

@app.post("/auth")
def auth_save():
    act = (request.form.get("action") or "").lower()
    if act == "save_basic":
        domain = (request.form.get("domain_basic") or "").strip()
        email  = (request.form.get("email") or "").strip()
        token  = (request.form.get("api_token") or "").strip()
        if not (domain and email and token):
            return Response("<p class='err'>Missing domain/email/token.</p>"+FORM_HTML, mimetype="text/html")
        _set_secret(domain, "basic", {"username": email, "password": token}, persist=True)
        msg = f"Saved Basic for {html.escape(domain)} as {html.escape(_mask(email))}"
    elif act == "save_bearer":
        domain = (request.form.get("domain_bearer") or "").strip()
        tok    = (request.form.get("bearer_token") or "").strip()
        if not (domain and tok):
            return Response("<p class='err'>Missing domain/token.</p>"+FORM_HTML, mimetype="text/html")
        _set_secret(domain, "oauth", {"access_token": tok, "token_type": "Bearer"}, persist=True)
        msg = f"Saved Bearer for {html.escape(domain)}"
    else:
        msg = "No action"
    body = f"<div class='ok'>{html.escape(msg)}</div>" + FORM_HTML
    return Response(body, mimetype="text/html")



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
    body = (request.get_json(silent=True) or {}).get("entry") or f"Auto-entry at {stamp}."
    with jf.open("a", encoding="utf-8") as f:
        f.write(body.strip() + "\n\n")
    return jsonify({"status": "ok", "appended": body}), 200

# ----------------------- Web Search (unchanged) -----------------------

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

@app.post("/web-search")
def web_search():
    try:
        data = request.get_json(force=True) or {}
        cfg = load_search_cfg()

        q = (data.get("query") or "").strip()
        mode = (data.get("mode") or cfg.get("mode", "www")).lower()
        allow = [_norm_host(x) for x in (data.get("allowlist") or cfg.get("allowlist") or [])]
        block = [_norm_host(x) for x in (data.get("blocklist") or cfg.get("blocklist") or [])]
        fetch_pages = bool(data.get("fetch_pages", cfg.get("fetch_pages", True)))
        max_results = int(data.get("max_results", cfg.get("max_results", 6)))
        use_creds = bool(data.get("use_credentials", cfg.get("use_credentials", True)))

        # Direct URL fetch
        if _looks_like_url(q):
            url = q
            note = "Direct URL fetch"
            try:
                kw = _build_request_kwargs(url, use_creds)
                resp = session.get(url, stream=True, verify=session.verify, **kw)
                status = resp.status_code
                enc = resp.encoding or "utf-8"
                body = resp.text if not fetch_pages else resp.text[:200000]
                try:
                    from bs4 import BeautifulSoup
                    soup = BeautifulSoup(body, "html.parser")
                    for t in soup(["script", "style", "noscript"]):
                        t.decompose()
                    text = soup.get_text(separator="\\n")
                except Exception:
                    import re as _re
                    text = _re.sub("<[^>]+>", " ", body)
                text = re.sub(r"\\n{3,}", "\\n\\n", text)
                text = re.sub(r"[ \\t]{2,}", " ", text).strip()
                result = {"title": "(direct)", "url": url, "snippet": "", "text": text[:6000], "status": status}
                return jsonify({"response": {"note": f"{note} (status={status})", "query": q, "results": [result]}}), 200
            except Exception as e:
                return jsonify({"response": {"note": f"{note} error: {e}", "query": q, "results": []}}), 200

        results: List[Dict[str, str]] = []
        if mode in ("local", "both"):
            local_mod = _try_import("web_search")
            if local_mod and hasattr(local_mod, "do_web_search"):
                try:
                    local = local_mod.do_web_search(q, max_results=max_results, fetch_pages=fetch_pages)
                    if isinstance(local, dict) and "results" in local:
                        results.extend(local["results"])
                except Exception:
                    pass

        if mode in ("www", "both") and len(results) < max_results:
            results.extend(_ddg_lite(q, max_results=max_results))

        filtered = []
        for r in results:
            u = r.get("url", "")
            if u and _host_allowed(u, allow, block):
                filtered.append(r)

        enriched: List[Dict[str, Any]] = []
        for r0 in filtered[:max_results]:
            r1 = dict(r0)
            if fetch_pages and r1.get("url"):
                try:
                    kw = _build_request_kwargs(r1["url"], use_creds)
                    resp = session.get(r1["url"], stream=True, verify=session.verify, **kw)
                    resp.raise_for_status()
                    enc = resp.encoding or "utf-8"
                    html = resp.text[:200000]
                    try:
                        from bs4 import BeautifulSoup
                        soup = BeautifulSoup(html, "html.parser")
                        for t in soup(["script", "style", "noscript"]): t.decompose()
                        text = soup.get_text(separator="\\n")
                    except Exception:
                        import re as _re
                        text = _re.sub("<[^>]+>", " ", html)
                    text = re.sub(r"\\n{3,}", "\\n\\n", text)
                    text = re.sub(r"[ \\t]{2,}", " ", text).strip()
                    r1["text"] = text[:6000]
                except Exception as e:
                    r1["text"] = f"(fetch error: {e})"
            enriched.append(r1)

        return jsonify({"response": {"note": "local+DDG-lite", "query": q, "results": enriched}}), 200

    except Exception as e:
        return jsonify({"response": f"Search error: {e}"}), 200

# ----------------------- Jira Search + Auth -----------------------

def _jira_auth_for(domain: str) -> Tuple[str, Optional[requests.auth.AuthBase], Dict[str,str]]:
    """
    Returns: (auth_kind, auth_object_or_None, extra_headers)
      auth_kind: "oauth" or "basic" or "none"
      auth_object: requests auth object if Basic, else None
      extra_headers: e.g., {"Authorization": "Bearer ..."}
    """
    domain = _norm_host(domain)
    # Prefer OAuth/Bearer if present
    o = _get_secret(domain, "oauth") or _get_secret(domain, "bearer")
    if isinstance(o, dict):
        tok = o.get("access_token") or o.get("token")
        if tok:
            return ("oauth", None, {"Authorization": f"Bearer {tok}"})
    # Fallback to Basic
    b = _get_secret(domain, "basic") or {}
    if b.get("username") and b.get("password"):
        from requests.auth import HTTPBasicAuth
        return ("basic", HTTPBasicAuth(b["username"], b["password"]), {})
    return ("none", None, {})

def _jira_request_kwargs(domain: str) -> dict:
    """Build requests kwargs (headers/auth/proxies/timeout/verify) for Jira.
       Always cookie-free and redirect-free to avoid XSRF 403s.
    """
    domain = _norm_host(domain)
    auth_kind, basic_auth, extra = _jira_auth_for(domain)

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "X-Atlassian-Token": "no-check"
    }
    headers.update(BASE_HEADERS)
    headers.update(extra)   # Authorization: Bearer ... (when oauth)

    kwargs = {
        "headers": headers,
        "auth": basic_auth,         # only set when using Basic
        "proxies": _get_proxies(),
        "timeout": 30,
        "verify": True,
        "cookies": {},              # block cookie jar
        "allow_redirects": False,   # avoid login HTML
    }
    return kwargs

def _jira_http(method: str, url: str, **kwargs):
    # Replace any accidental cookies
    headers = dict(kwargs.pop("headers", {}))
    headers["Cookie"] = ""     # hard override; prevents jar cookies
    kwargs["headers"] = headers
    kwargs.setdefault("cookies", {})
    kwargs.setdefault("allow_redirects", False)
    with requests.Session() as s:
        s.verify = True
        return s.request(method, url, **kwargs)

@app.get("/jira/auth-check")
def jira_auth_check():
    domain = request.args.get("domain","").strip()
    if not domain:
        return jsonify({"error":"domain required"}), 400
    auth_kind, basic_auth, extra = _jira_auth_for(domain)
    user_hint = None
    if auth_kind == "basic":
        b = _get_secret(domain, "basic") or {}
        u = b.get("username") or ""
        if u:
            user_hint = (u[:3] + "…" + u[-3:]) if len(u) > 6 else (u[:1] + "…" if u else "")
    elif auth_kind == "oauth":
        user_hint = "Bearer *"

    base  = f"https://{_norm_host(domain)}"
    r = _jira_http("GET", f"{base}/rest/api/3/myself", **_jira_request_kwargs(domain))
    body_sample = ""
    try:
        body_sample = (r.text or "")[:200]
    except Exception:
        pass
    return jsonify({
        "ok": (r.status_code==200),
        "status": r.status_code,
        "content_type": r.headers.get("Content-Type",""),
        "body_sample": body_sample,
        "auth_hint": {"auth_kind": auth_kind, "has_auth": auth_kind!="none", "user_hint": user_hint}
    }), 200

@app.get("/jira/myself")
def jira_myself():
    domain = request.args.get("domain","").strip()
    if not domain:
        return jsonify({"error":"domain required"}), 400
    base  = f"https://{_norm_host(domain)}"
    r = _jira_http("GET", f"{base}/rest/api/3/myself", **_jira_request_kwargs(domain))
    ct = r.headers.get("Content-Type","")
    if r.ok and "application/json" in ct:
        return jsonify(r.json()), 200
    return jsonify({"status": r.status_code, "content_type": ct, "body": r.text[:800]}), 200

def _ok_json(r: requests.Response) -> bool:
    return r.ok and "application/json" in r.headers.get("Content-Type","")

def _has_nonempty_issues(obj: Any) -> bool:
    return isinstance(obj, dict) and isinstance(obj.get("issues"), list) and len(obj["issues"]) > 0

@app.post("/jira/search")
def jira_search():
    data = request.get_json(silent=True) or {}
    domain = (data.get("domain") or "").strip()
    jql    = (data.get("jql") or data.get("query") or "").strip()
    if not domain or not jql:
        return jsonify({"error": "domain and jql/query are required"}), 400

    fields      = data.get("fields")
    max_results = int(data.get("maxResults", 50))
    start_at    = int(data.get("startAt", 0))
    expand      = data.get("expand")
    force       = (data.get("force_method") or "").lower()

    kwargs = _jira_request_kwargs(domain)
    base  = f"https://{domain}".rstrip("/")

    def ok_json(r):
        return r.ok and "application/json" in r.headers.get("Content-Type", "")

    def shape_has_issues(obj):
        return isinstance(obj, dict) and isinstance(obj.get("issues"), list)

    # Rewrite currentUser() → accountId
    original_jql = jql
    try:
        acct = _get_account_id(domain, kwargs)
        jql  = _rewrite_current_user(jql, acct)
    except Exception:
        pass

    # POST payload / GET params
    post_payload = {"jql": jql, "startAt": start_at, "maxResults": max_results}
    if fields is not None: post_payload["fields"] = fields
    if expand:             post_payload["expand"] = expand

    get_params = {"jql": jql, "startAt": start_at, "maxResults": max_results}
    if fields: get_params["fields"] = ",".join(fields)
    if expand: get_params["expand"] = ",".join(expand) if isinstance(expand, list) else str(expand)

    attempts = []

    def try_post_new():
        url = f"{base}/rest/api/3/search/jql"
        r = _jira_http("POST", url, json=post_payload, **kwargs)
        attempts.append({"method": "POST", "url": url, "status": r.status_code, "ct": r.headers.get("Content-Type", "")})
        return r.json() if ok_json(r) else {"status": r.status_code, "body": r.text[:1000], "ct": r.headers.get("Content-Type", "")}

    def try_get_new():
        url = f"{base}/rest/api/3/search/jql"
        r = _jira_http("GET", url, params=get_params, **kwargs)
        attempts.append({"method": "GET", "url": f"{url}?…", "status": r.status_code, "ct": r.headers.get("Content-Type", "")})
        return r.json() if ok_json(r) else {"status": r.status_code, "body": r.text[:1000], "ct": r.headers.get("Content-Type", "")}

    def try_post_legacy():
        url = f"{base}/rest/api/3/search"
        r = _jira_http("POST", url, json=post_payload, **kwargs)
        attempts.append({"method": "POST", "url": url, "status": r.status_code, "ct": r.headers.get("Content-Type", "")})
        return r.json() if ok_json(r) else {"status": r.status_code, "body": r.text[:1000], "ct": r.headers.get("Content-Type", "")}

    # forced path for debugging
    if force == "post":   return jsonify(try_post_new()), 200
    if force == "get":    return jsonify(try_get_new()),  200
    if force == "legacy": return jsonify(try_post_legacy()), 200

    out = try_post_new()
    if shape_has_issues(out):  return jsonify(out), 200

    out2 = try_get_new()
    if shape_has_issues(out2): return jsonify(out2), 200

    out3 = try_post_legacy()
    if shape_has_issues(out3): return jsonify(out3), 200

    # On total failure, return debug info + original/effective JQL
    return jsonify({
        "note": "no JSON issues from any variant",
        "original_jql": original_jql,
        "effective_jql": jql,
        "attempts": attempts,
        "post_new": out,
        "get_new": out2,
        "legacy": out3
    }), 200

@app.post("/jira/search/debug")
def jira_search_debug():
    data = request.get_json(silent=True) or {}
    domain = _norm_host((data.get("domain") or "").strip())
    jql    = (data.get("jql") or data.get("query") or "").strip()
    if not domain or not jql:
        return jsonify({"error": "domain and jql/query are required"}), 400

    fields      = data.get("fields")
    max_results = int(data.get("maxResults", 50))
    start_at    = int(data.get("startAt", 0))
    expand      = data.get("expand")
    force       = (data.get("force_method") or "").lower()

    base  = f"https://{domain}"
    kwargs = _jira_request_kwargs(domain)

    post_payload = {"jql": jql, "startAt": start_at, "maxResults": max_results}
    if fields is not None: post_payload["fields"] = fields
    if expand:             post_payload["expand"] = expand

    get_params = {"jql": jql, "startAt": start_at, "maxResults": max_results}
    if fields: get_params["fields"] = ",".join(fields)
    if expand: get_params["expand"] = ",".join(expand) if isinstance(expand, list) else str(expand)

    attempts = []

    def capture(r: requests.Response, url: str, method: str):
        attempts.append({
            "url": url if method=="GET" else url,
            "method": method,
            "status": r.status_code,
            "content_type": r.headers.get("Content-Type",""),
            "content_len": r.headers.get("Content-Length",""),
            "set_cookie": "<present>" if r.headers.get("Set-Cookie") else "",
            "x_arequestid": r.headers.get("X-ARequestId",""),
            "location": r.headers.get("Location",""),
            "body_sample": (r.text or "")[:100]
        })

    def do_post_new():
        url = f"{base}/rest/api/3/search/jql"
        r = _jira_http("POST", url, json=post_payload, **kwargs)
        capture(r, url, "POST")
        return r

    def do_get_new():
        url = f"{base}/rest/api/3/search/jql"
        r = _jira_http("GET", url, params=get_params, **kwargs)
        capture(r, f"{url}?…", "GET")
        return r

    def do_post_legacy():
        url = f"{base}/rest/api/3/search"
        r = _jira_http("POST", url, json=post_payload, **kwargs)
        capture(r, url, "POST")
        return r

    results = {}
    if force in ("", "post"):
        r1 = do_post_new()
        if _ok_json(r1): results["post_new"] = r1.json()
    if force in ("get", ""):
        r2 = do_get_new()
        if _ok_json(r2): results["get_new"] = r2.json()
    if force in ("legacy", ""):
        r3 = do_post_legacy()
        if _ok_json(r3): results["legacy"] = r3.json()

    sample_keys = []
    issues_len = None
    try:
        any_issues = (results.get("post_new") or results.get("get_new") or results.get("legacy") or {})
        if isinstance(any_issues, dict) and isinstance(any_issues.get("issues"), list):
            issues_len = len(any_issues["issues"])
            sample_keys = [i.get("key") for i in any_issues["issues"][:5] if isinstance(i, dict)]
    except Exception:
        pass

    return jsonify({
        "attempts": attempts,
        "result_summary": {"issues_len": issues_len, "keys_sample": sample_keys},
        "results": results
    }), 200

@app.post("/debug/clear-session")
def clear_session():
    session.cookies.clear()
    return jsonify({"ok": True, "cookies": list(session.cookies.items())}), 200

# ----------------------- Workflow stubs (unchanged) -----------------------

def _write(path: Path, text: str) -> Path:
    path.write_text(text or "", encoding="utf-8")
    return path

@app.post("/embed-content")
def embed_content():
    file = request.files.get("file")
    text = ""
    if file:
        text = file.read().decode("utf-8", errors="replace")
    else:
        generic_url = request.form.get("generic_url")
        if generic_url:
            kw = _build_request_kwargs(generic_url, use_credentials=True)
            r = session.get(generic_url, timeout=30, verify=session.verify, **kw)
            r.raise_for_status()
            text = r.text[:200000]
    if not text:
        return jsonify({"error": "no content"}), 400
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = EMBED_DIR / f"{ts}.txt"
    out.write_text(text, encoding="utf-8")
    return jsonify({"result": "stored", "path": str(out)}), 200


@app.post("/trigger-workflow")
def trigger_workflow():
    data = request.get_json(silent=True) or {}
    original_request = (data.get("original_request") or "").strip()
    task_description = (data.get("task_description") or "").strip()
    chat_url = data.get("chat_url")
    token = data.get("token")
    inference_profile_arn = data.get("inference_profile_arn")
    auto_iterate = bool(data.get("auto_iterate", True))
    max_iters = int(data.get("max_iters", 2))

    if not original_request or not task_description:
        return jsonify({"error": "original_request and task_description are required"}), 400

    instruct_path = INSTRUCTIONS_DIR / "workflow.instruct"
    _write(instruct_path, task_description)

    iterations = []
    final_summary = ""

    for i in range(1, max_iters + 1):
        res_path = _generate_resources(task_description, chat_url, token, inference_profile_arn)
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

        _append_journal("gabriel",
                        f"[WF] Iter {i}: complete={verdict.get('complete')} next={len(verdict.get('next_actions', []))}")

        if not auto_iterate or verdict.get("complete") is True:
            break

        _append_actions_to_resources(verdict.get("next_actions"), res_path)
        task_description = task_description + "\n\nFollow-up actions:\n" + "\n".join(
            f"- {a.get('cmd') or a.get('url')}" for a in (verdict.get("next_actions") or [])
        )

    bundle_path = _bundle_files()
    return jsonify(
        {"status": "ok", "bundle_path": bundle_path, "iterations": iterations, "final_summary": final_summary}), 200


def _write(path: Path, text: str) -> Path:
    path.write_text(text or "", encoding="utf-8")
    return path

def _append_journal(name: str, line: str) -> None:
    f = JOURNALS_DIR / f"{name}.journal"
    with f.open("a", encoding="utf-8") as fp:
        fp.write(line.strip() + "\n\n")

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
        if "tool/resource" in lower and "url" in lower: continue
        if lower.startswith(("tool,", "resource,", "tool/resource,")): continue
        filtered.append(ln)
    return "\\n".join(filtered)

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
            while len(cols) < 4: cols.append("-")
            cmd, url = cols[2], cols[3]

            if url and url not in ("-", "N/A", "NA", "None", "URL (if any)"):
                try:
                    r = session.get(url, timeout=30, verify=True, headers=BASE_HEADERS, proxies=_get_proxies())
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
    user = (
        "Original request:\\n---\\n" + original_request + "\\n---\\n\\n"
        "Did the summary below fully satisfy the request? If not, list concrete next actions "
        "(bash commands or URLs) we should run/fetch next.\\n\\n"
        "Summary:\\n---\\n" + summary_text + "\\n---\\n\\n"
        'Respond JSON with keys: "complete": true|false, "next_actions": [ {"cmd": "..."}, {"url": "..."} ]'
    )
    raw = _coverity_chat(user, system_text=system, max_tokens=300,
                         inference_profile_arn=inference_profile_arn,
                         url=chat_url, token=token)
    try:
        data = json.loads(raw)
        if not isinstance(data, dict): raise ValueError("not an object")
        if "complete" not in data: data["complete"] = False
        if "next_actions" not in data or not isinstance(data["next_actions"], list):
            data["next_actions"] = []
        return data
    except Exception:
        return {"complete": False, "next_actions": []}

def _append_actions_to_resources(actions, resources_path: Path) -> None:
    rows = []
    for a in actions or []:
        if not isinstance(a, dict): continue
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

# ----------------------- Main -----------------------
def _parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Coverity Assist Gateway (patched currentUser→accountId, TLS verify)")
    parser.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "5000")))
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    app.run(host=args.host, port=args.port, debug=args.debug, use_reloader=False)
