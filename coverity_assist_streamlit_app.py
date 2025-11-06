import os
import json
import time
import requests
import subprocess
from urllib.parse import urlparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, Any, List, Dict

import streamlit as st

# ---------- Config ----------

DEFAULT_CHAT_URL = os.environ.get("COVERITY_ASSIST_URL", "http://coverity-assist.dishtv.technology/chat")
DEFAULT_TOKEN    = os.environ.get("COVERITY_ASSIST_TOKEN", "")
PROMPTS_PATH     = os.environ.get("COVERITY_ASSIST_PROMPTS", str(Path.cwd() / ".coverity_assist_prompts.json"))
SCRIPTS_DIR      = os.environ.get("JAMBOT_SCRIPTS_DIR", str(Path.cwd() / "scripts"))
JAMBOT_BASE_URL  = os.environ.get("JAMBOT_BASE_URL", "http://127.0.0.1:5000")  # Flask from gateway
JOURNALS_DIR     = os.environ.get("JOURNALS_DIR", str(Path.cwd() / "journals"))
DEFAULT_INFERENCE_PROFILE_ARN = os.environ.get("BEDROCK_APPLICATION_INFERENCE_PROFILE_ARN", "")

os.makedirs(SCRIPTS_DIR, exist_ok=True)
os.makedirs(JOURNALS_DIR, exist_ok=True)

# Attempt to load optional local helper modules from the working dir if present.
def try_import(name: str):
    try:
        return __import__(name)
    except Exception:
        return None

web_search_mod = try_import("web_search")
combine_logs_mod = try_import("combine_logs")
bert2_mod = try_import("bert2")
journaler_mod = try_import("journaler")  # may expose utility helpers

# ---------- Utilities ----------

def http_ok(response: requests.Response) -> bool:
    try:
        response.raise_for_status()
        return True
    except requests.HTTPError:
        return False

def bearer_header(token: str) -> Dict[str, str]:
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

def read_prompts() -> Dict[str, str]:
    if Path(PROMPTS_PATH).exists():
        try:
            with open(PROMPTS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            st.warning("Failed to read prompts file; using in-memory defaults.")
    # minimal seed if empty
    return {
        "Aqua": "You are Aqua — energetic, optimistic, asks a clarifying question when unsure.",
        "Alex": "You are Alex — precise network engineer; start replies by addressing the recipient by name.",
        "Avery": "You are Avery — visual analyst who extracts structured data from images.",
        "Gabriel": "You are Gabriel — empathetic observer who suggests prompt updates when useful.",
        "Gemma": "You are Gemma — thoughtful facilitator who nudges deeper clarity.",
        "Claud": "You are Claud — analyzes VAR logs and reports issues per CHANGE_CONTENT windows."
    }

def write_prompts(d: Dict[str, str]):
    Path(PROMPTS_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(PROMPTS_PATH, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2)

def build_chat_payload(user_text: str,
                       system_text: Optional[str],
                       max_tokens: int = 1000,
                       use_top_level_system: bool = False,
                       inference_profile_arn: Optional[str] = None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "messages": [{"role": "user", "content": user_text}],
        "max_tokens": max_tokens
    }
    if system_text:
        if use_top_level_system:
            payload["system"] = system_text
        else:
            payload["messages"][0]["content"] = f"{system_text}\n\n{user_text}"
    if inference_profile_arn:
        payload["inference_profile_arn"] = inference_profile_arn
    return payload

def coverity_chat(url: str,
                  token: str,
                  user_text: str,
                  system_text: Optional[str],
                  max_tokens: int = 1000,
                  use_top_level_system: bool = False,
                  inference_profile_arn: Optional[str] = None) -> Tuple[bool, str]:
    try:
        payload = build_chat_payload(user_text, system_text, max_tokens, use_top_level_system, inference_profile_arn)
        r = requests.post(url, headers=bearer_header(token), json=payload, timeout=120)
        if not http_ok(r):
            return False, f"[{r.status_code}] {r.text}"
        data = r.json()
        out = data.get("content") or data.get("response") or data.get("text") or json.dumps(data)
        return True, out
    except Exception as e:
        return False, f"Client error: {e}"

def jambot_get(url_path: str) -> Tuple[bool, Any]:
    try:
        r = requests.get(f"{JAMBOT_BASE_URL}{url_path}", timeout=30)
        if not http_ok(r):
            return False, f"[{r.status_code}] {r.text}"
        try:
            return True, r.json()
        except Exception:
            return True, r.text
    except Exception as e:
        return False, str(e)

def jambot_post_json(url_path: str, json_body: Optional[Dict[str, Any]] = None) -> Tuple[bool, Any]:
    try:
        r = requests.post(f"{JAMBOT_BASE_URL}{url_path}", json=json_body, timeout=120)
        if not http_ok(r):
            return False, f"[{r.status_code}] {r.text}"
        try:
            return True, r.json()
        except Exception:
            return True, r.text
    except Exception as e:
        return False, str(e)

def jambot_post_form(url_path: str, data: Optional[Dict[str, Any]] = None, files: Optional[Dict[str, Any]] = None) -> Tuple[bool, Any]:
    try:
        r = requests.post(f"{JAMBOT_BASE_URL}{url_path}", data=data, files=files, timeout=300)
        if not http_ok(r):
            return False, f"[{r.status_code}] {r.text}"
        try:
            return True, r.json()
        except Exception:
            return True, r.text
    except Exception as e:
        return False, str(e)

def run_local_script(path: Path, args: List[str]) -> Tuple[bool, str]:
    try:
        proc = subprocess.run(
            ["bash", "-lc", f"chmod +x '{path}' || true && '{path}' " + " ".join(map(str, args))],
            capture_output=True, text=True, timeout=1800
        )
        ok = proc.returncode == 0
        out = proc.stdout + ("\n" + proc.stderr if proc.stderr else "")
        return ok, out
    except Exception as e:
        return False, f"Exec error: {e}"

def parse_domains(text: str) -> List[str]:
    raw = [t.strip() for t in (text or "").replace("\n", ",").split(",")]
    return [x for x in raw if x]

def domain_of(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""

def filter_results(results: List[Dict[str, Any]], allowlist: List[str], blocklist: List[str]) -> List[Dict[str, Any]]:
    if not isinstance(results, list):
        return results
    outs = []
    for r in results:
        d = domain_of(r.get("url",""))
        if allowlist and d and not any(d.endswith(a) or d == a for a in allowlist):
            continue
        if blocklist and d and any(d.endswith(b) or d == b for b in blocklist):
            continue
        outs.append(r)
    return outs

# Remember auto-iterate settings across tabs
if "auto_iter" not in st.session_state:
    st.session_state.auto_iter = True
if "max_iters" not in st.session_state:
    st.session_state.max_iters = 2

def trigger_workflow_from_search(query: str, results_blob: str,
                                 chat_url: str, token: str, inference_profile_arn: str,
                                 auto_iter: bool, max_iters: int) -> Tuple[bool, Any]:
    task = (
        f"Research and summarize the topic:\n"
        f"  {query}\n\n"
        f"Use the following search results (may be truncated) as context. Extract sources and verify.\n"
        f"RESULTS:\n{results_blob[:8000]}"
    )
    return jambot_post_json(
        "/trigger-workflow",
        json_body={
            "original_request": f'Web search: "{query}"',
            "task_description": task,
            "chat_url": chat_url,
            "token": token,
            "inference_profile_arn": inference_profile_arn,
            "auto_iterate": auto_iter,
            "max_iters": int(max_iters),
        },
    )

# ---------- UI ----------

st.set_page_config(page_title="Coverity Assist + JAMbot Console", layout="wide")

with st.sidebar:
    st.subheader("Search config")
    search_mode = st.selectbox(
        "Search mode", ["local", "www", "both"], index=2, key="sb_search_mode"
    )
    use_creds   = st.checkbox("Use credentials", value=False, key="sb_use_creds")
    fetch_pages = st.checkbox("Fetch pages", value=True, key="sb_fetch_pages")
    allowlist_txt = st.text_area("Domain allowlist (one per line)", "", key="sb_allow")
    blocklist_txt = st.text_area("Domain blocklist (one per line)", "", key="sb_block")
    if st.button("Save search config", key="sb_save_search_cfg"):
        cfg = {
            "mode": search_mode,                           # <- was "search_mode"
            "use_credentials": use_creds,
            "fetch_pages": fetch_pages,
            "allowlist": [ln.strip() for ln in allowlist_txt.splitlines() if ln.strip()],
            "blocklist": [ln.strip() for ln in blocklist_txt.splitlines() if ln.strip()],
        }
        ok, resp = jambot_post_json("/config/search", json_body=cfg)
        st.success("Saved.") if ok else st.error(resp)


    st.header("Credentials")
    cookie_hdr = st.text_area("Cookie header")
    basic_user = st.text_input("Basic user")
    basic_pass = st.text_input("Basic password", type="password")
    proxy_url = st.text_input("Proxy URL (http://user:pass@host:port)")
    extra_hdrs = st.text_area("Extra headers (JSON)", "{}")

    if st.button("Save credentials"):
        try:
            hdrs = json.loads(extra_hdrs) if extra_hdrs.strip() else {}
            payload = {
                "cookie": cookie_hdr,
                "basic_user": basic_user,
                "basic_pass": basic_pass,
                "proxy": proxy_url,
                "headers": hdrs,
            }
            ok, resp = jambot_post_json("/credentials", json_body=payload)
            st.success("Saved.") if ok else st.error(resp)
        except Exception as e:
            st.error(f"Bad JSON in Extra headers: {e}")

    st.header("Settings")
    chat_url = st.text_input("Coverity Assist Chat URL", value=DEFAULT_CHAT_URL)
    token = st.text_input("Bearer Token", value=DEFAULT_TOKEN, type="password")
    use_top_level_system = st.checkbox("Send persona as top-level system", value=False)
    st.caption("If unchecked, persona prompt is prefixed to the user message.")

    st.divider()
    st.subheader("JAMbot / Gateway")
    jambot_base = st.text_input("Base URL", value=JAMBOT_BASE_URL, help="Flask server from your gateway")
    if jambot_base != JAMBOT_BASE_URL and jambot_base:
        JAMBOT_BASE_URL = jambot_base

    st.divider()
    st.subheader("Persona + Bedrock")
    inference_profile_arn = st.text_input("Inference Profile ARN", value=DEFAULT_INFERENCE_PROFILE_ARN,
        help="Application Inference Profile ARN (tagged for Cost Explorer)")

    prompts = read_prompts()
    persona = st.selectbox("Persona", options=sorted(prompts.keys()),
                           index=sorted(prompts.keys()).index("Alex") if "Alex" in prompts else 0,
                           key="persona_select")
    persona_text = st.text_area("System Prompt (current persona)", value=prompts.get(persona, ""), height=220)
    if st.button("Save Persona"):
        prompts[persona] = persona_text
        write_prompts(prompts)
        st.success(f"Saved prompt for {persona}.")

st.title("Coverity Assist + JAMbot")
st.caption("Requests include the Application Inference Profile ARN you set in the sidebar. Costs are attributed via the profile's tags.")
tabs = st.tabs(["Chat", "Journal", "Web Search + Embed", "Log Analysis", "Scripts & Workflow", "DB Tools"])

# ---------- Chat Tab ----------

with tabs[0]:
    st.subheader("Chat")
    max_tokens = st.number_input("Max Tokens", min_value=1, max_value=4000, value=1000, step=50)
    user_text = st.text_area("Message", height=150, placeholder="Ask a question, paste logs, or draft an email…")
    if st.button("Send", use_container_width=True):
        if not token:
            st.error("Please provide your Bearer token in the sidebar.")
        else:
            ok, out = coverity_chat(chat_url, token, user_text, prompts.get(persona), max_tokens, use_top_level_system, inference_profile_arn)
            st.write(out if ok else f"❌ {out}")

# ---------- Journal Tab ----------

with tabs[1]:
    st.subheader("Journals")
    colA, colB = st.columns(2)

    with colA:
        st.write("**Local journals directory**")
        st.code(JOURNALS_DIR)
        for fname in sorted(Path(JOURNALS_DIR).glob("*.journal")):
            with st.expander(fname.name):
                try:
                    st.text(fname.read_text(encoding="utf-8"))
                except Exception as e:
                    st.warning(f"Can't read {fname}: {e}")

    with colB:
        st.write("**JAMbot journals (via API)**")
        ok, data = jambot_get("/get-journal-files")
        if ok and isinstance(data, dict):
            for name, content in data.items():
                with st.expander(name):
                    st.text(content if isinstance(content, str) else json.dumps(content, indent=2))
        else:
            st.caption("JAMbot API not reachable or no journals available.")

    st.divider()
    st.write("**Write a new journal entry (Gabriel)**")
    if st.button("Append Fresh Entry via /journal"):
        ok, resp = jambot_post_json("/journal", json_body={})
        st.write(resp if ok else f"❌ {resp}")

# ---------- Web Search + Embed Tab ----------
# Session state to persist the last search so reruns don't lose it
if "last_search" not in st.session_state:
    st.session_state.last_search = {"query": "", "results": None, "raw": ""}

# Config session defaults
# Session config defaults
if "search_config" not in st.session_state:
    st.session_state.search_config = {
        "mode": "both",                    # <- was "search_mode"
        "allowlist": [],
        "blocklist": ["linkedin.com","zoominfo.com","x.com","twitter.com"],
        "use_credentials_by_default": False,
        "fetch_pages": True,
        "max_results": 6
    }


# ---- Web search helpers (fallbacks) ----
def now_in_tz(tz_name: str) -> datetime:
    try:
        from zoneinfo import ZoneInfo  # py>=3.9
        return datetime.now(ZoneInfo(tz_name))
    except Exception:
        try:
            import pytz  # py>=3.8 fallback
            return datetime.now(pytz.timezone(tz_name))  # type: ignore[attr-defined]
        except Exception:
            return datetime.now()

def ddg_lite_search(query: str) -> str:
    """Very light HTML fallback; returns first snippet if available."""
    import re, html
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept-Language": "en-US,en;q=0.9",
    }
    r = requests.get("https://lite.duckduckgo.com/lite/", params={"q": query}, headers=headers, timeout=15)
    if r.status_code != 200:
        return f"Fallback search HTTP {r.status_code}"
    m = re.search(r"<a[^>]+class=['\"]result-link['\"][^>]*>(.*?)</a>.*?<td[^>]*>(.*?)</td>", r.text, re.S | re.I)
    if m:
        title = html.unescape(re.sub("<.*?>", "", m.group(1)))
        snippet = html.unescape(re.sub("<.*?>", "", m.group(2))).strip()
        return f"**{title}**\n\n{snippet}"
    return "No results found."

with tabs[2]:
    st.subheader("Web Search")

    # ----- Search Configuration (Gateway-backed) -----
    with st.expander("Search Configuration (saved on Gateway)", expanded=False):
        cfg = st.session_state.search_config
        col1, col2, col3 = st.columns(3)
        with col1:
            mode = st.selectbox(
                "Search mode",
                options=["local", "www", "both"],
                index=["local", "www", "both"].index(cfg.get("mode", "both")),  # <- was search_mode
                key="exp_search_mode"
            )
            use_creds = st.checkbox(
                "Use credentials by default",
                value=cfg.get("use_credentials_by_default", False),
                key="exp_use_creds_default"
            )
        with col2:
            fetch_pages = st.checkbox(
                "Fetch result pages",
                value=cfg.get("fetch_pages", True),
                key="exp_fetch_pages"
            )
            max_results = st.number_input(
                "Max results", min_value=1, max_value=20,
                value=int(cfg.get("max_results", 6)), key="exp_max_results"
            )
        with col3:
            allowlist_text = st.text_area(
                "Allowlist domains (comma/newline)",
                value="\n".join(cfg.get("allowlist", [])), height=90, key="exp_allow"
            )
            blocklist_text = st.text_area(
                "Blocklist domains (comma/newline)",
                value="\n".join(cfg.get("blocklist", [])), height=90, key="exp_block"
            )

        bcol1, bcol2, bcol3 = st.columns(3)
        with bcol1:
            if st.button("Load current from Gateway", key="exp_load"):
                ok, resp = jambot_get("/config/search")
                if ok:
                    try:
                        resp = resp if isinstance(resp, dict) else json.loads(resp)
                        # normalize to our in-memory schema
                        st.session_state.search_config = {
                            "mode": resp.get("mode", "both"),
                            "allowlist": resp.get("allowlist", []),
                            "blocklist": resp.get("blocklist", []),
                            "use_credentials_by_default": resp.get("use_credentials", False),
                            "fetch_pages": resp.get("fetch_pages", True),
                            "max_results": resp.get("max_results", 6),
                        }
                        st.success("Loaded.")
                    except Exception as e:
                        st.error(f"Bad JSON from gateway: {e}")
        with bcol2:
            if st.button("Save to Gateway", key="exp_save"):
                payload = {
                    "mode": mode,  # <- send 'mode' for gateway compatibility
                    "allowlist": parse_domains(allowlist_text),
                    "blocklist": parse_domains(blocklist_text),
                    "use_credentials": bool(use_creds),
                    "fetch_pages": bool(fetch_pages),
                    "max_results": int(max_results),
                }
                ok, resp = jambot_post_json("/config/search", json_body=payload)
                if ok:
                    st.session_state.search_config.update(payload)
                    st.success("Saved.")
                else:
                    st.error(f"Save failed: {resp}")

        with bcol3:
            st.caption("These settings influence /web-search and the result fetcher on the gateway.")

    # ----- Credentials helper forms -----
    with st.expander("Credentials & Proxy (stored in keyring if available)", expanded=False):
        st.caption("Provide only what you are authorized to supply. Cookies may contain secrets; prefer scoped cookies.")
        cc1, cc2 = st.columns(2)
        with cc1:
            st.write("**Cookies (paste raw Cookie header)**")
            c_domain = st.text_input("Domain (e.g., about.dish.com)", key="c_dom")
            c_cookie = st.text_area("Cookie:", height=80, key="c_cook")
            c_persist = st.checkbox("Persist in keyring (if available)", value=True, key="c_persist")
            if st.button("Save Cookies"):
                payload = {"domain": c_domain.strip(), "kind": "cookies", "data": {"Cookie": c_cookie.strip()}, "persist": c_persist}
                ok, resp = jambot_post_json("/credentials", json_body=payload)
                st.write(resp if ok else f"❌ {resp}")
        with cc2:
            st.write("**Basic Auth**")
            b_domain = st.text_input("Domain", key="b_dom")
            b_user = st.text_input("Username", key="b_user")
            b_pass = st.text_input("Password", type="password", key="b_pass")
            b_persist = st.checkbox("Persist in keyring (if available)", value=True, key="b_persist")
            if st.button("Save Basic Auth"):
                payload = {"domain": b_domain.strip(), "kind": "basic", "data": {"username": b_user, "password": b_pass}, "persist": b_persist}
                ok, resp = jambot_post_json("/credentials", json_body=payload)
                st.write(resp if ok else f"❌ {resp}")

        hc1, hc2 = st.columns(2)
        with hc1:
            st.write("**Extra Headers (JSON)**")
            h_domain = st.text_input("Domain", key="h_dom")
            h_json = st.text_area("Headers JSON", value='{"X-Example":"value"}', key="h_json")
            h_persist = st.checkbox("Persist in keyring (if available)", value=True, key="h_persist")
            if st.button("Save Headers"):
                try:
                    headers_obj = json.loads(h_json)
                except Exception as e:
                    st.error(f"Bad JSON: {e}")
                else:
                    payload = {"domain": h_domain.strip(), "kind": "headers", "data": headers_obj, "persist": h_persist}
                    ok, resp = jambot_post_json("/credentials", json_body=payload)
                    st.write(resp if ok else f"❌ {resp}")
        with hc2:
            st.write("**Proxy (global or domain-scoped)**")
            p_domain = st.text_input("Domain (use * for global)", key="p_dom", value="*")
            p_http   = st.text_input("HTTP proxy",  key="p_http",  placeholder="http://user:pass@proxy:8080")
            p_https  = st.text_input("HTTPS proxy", key="p_https", placeholder="http://user:pass@proxy:8443")
            if st.button("Save Proxy"):
                payload = {"domain": p_domain.strip() or "*", "kind": "proxy", "data": {"http": p_http, "https": p_https}}
                ok, resp = jambot_post_json("/credentials", json_body=payload)
                st.write(resp if ok else f"❌ {resp}")

    # ----- Actual search -----
    with st.form("websearch_form", clear_on_submit=False):
        query = st.text_input(
            "Query (natural language)",
            value=st.session_state.last_search["query"],
            placeholder="find Hopper3 HDMI handshake troubleshooting",
        )
        c1, c2, c3 = st.columns(3)
        with c1:
            search_mode = st.selectbox(
                "Mode (local/www/both)",
                options=["local", "www", "both"],
                index=["local", "www", "both"].index(st.session_state.search_config.get("mode", "both")),
                key="form_search_mode"
            )
        with c2:
            use_credentials = st.checkbox(
                "Use credentials",
                value=st.session_state.search_config.get("use_credentials_by_default", False),
                key="form_use_creds"
            )
            fetch_pages = st.checkbox(
                "Fetch pages",
                value=st.session_state.search_config.get("fetch_pages", True),
                key="form_fetch_pages"
            )
        with c3:
            allowlist_text = st.text_area("Allowlist (for THIS search)", value="\n".join(st.session_state.search_config.get("allowlist", [])), height=80)
            blocklist_text = st.text_area("Blocklist (for THIS search)", value="\n".join(st.session_state.search_config.get("blocklist", [])), height=80)

        c4, c5 = st.columns(2)
        with c4:
            auto_iter = st.checkbox("Auto-iterate", value=st.session_state.auto_iter, help="Let workflow re-run using validation hints")
        with c5:
            max_iters = st.slider("Max iterations", 1, 5, value=int(st.session_state.max_iters))

        do_search   = st.form_submit_button("Search")
        do_kickoff  = st.form_submit_button("Kick off workflow with current results")

    # persist auto-iterate settings
    st.session_state.auto_iter = auto_iter
    st.session_state.max_iters = int(max_iters)

    result_obj = None
    error_text = None

    allowlist = parse_domains(allowlist_text)
    blocklist = parse_domains(blocklist_text)
    max_results = int(st.session_state.search_config.get("max_results", 6))

    if do_search:
        # trivial helpers
        q_lower = (query or "").strip().lower()
        if q_lower in {"what day is it", "what day is it?", "what's the date", "what's the date?"}:
            dt = now_in_tz("America/Denver")
            result_obj = {"results": [{"title": "Current date/time", "url": "", "snippet": dt.strftime("%A, %B %d, %Y (%I:%M %p %Z)")}]}
        else:
            # 1) local web_search.py if chosen or included in 'both'
            if search_mode in ("local", "both") and web_search_mod and hasattr(web_search_mod, "do_web_search"):
                try:
                    local_res = web_search_mod.do_web_search(query, max_results=max_results, fetch_pages=fetch_pages)
                    if isinstance(local_res, dict) and "results" in local_res:
                        local_items = local_res["results"]
                        local_items = filter_results(local_items, allowlist, blocklist)
                        result_obj = {"results": local_items}
                    else:
                        result_obj = local_res
                except Exception as e:
                    error_text = f"Local search error: {e}"

            # 2) gateway /web-search
            if (result_obj is None) and (search_mode in ("www", "both")):
                payload = {
                    "query": query,
                    "mode": search_mode,
                    "allowlist": allowlist,
                    "blocklist": blocklist,
                    "fetch_pages": bool(fetch_pages),
                    "use_credentials": bool(use_credentials),
                    "max_results": max_results,
                }
                ok, jambot_resp = jambot_post_json("/web-search", json_body=payload)
                if ok:
                    if isinstance(jambot_resp, dict) and "response" in jambot_resp:
                        resp = jambot_resp["response"]
                        if isinstance(resp, dict) and "results" in resp:
                            result_obj = {"results": resp["results"]}
                        else:
                            result_obj = resp
                    else:
                        # gateway may return raw results dict already
                        result_obj = jambot_resp
                else:
                    error_text = f"JAMbot /web-search error: {jambot_resp}"

            # 3) DDG lite fallback
            if result_obj is None:
                result_obj = ddg_lite_search(query)

        # Persist to session state
        st.session_state.last_search = {"query": query, "results": result_obj, "raw": result_obj}

    # Render results (from latest run or persisted)
    data = st.session_state.last_search
    if error_text:
        st.info(error_text)
    if data["results"]:
        st.write("### Results")
        ro = data["results"]
        # Prefer structured dict {"results":[{title,url,snippet},...]}
        if isinstance(ro, dict) and "results" in ro and isinstance(ro["results"], list):
            for i, r in enumerate(ro["results"][:10], 1):
                title   = r.get("title", f"Result {i}")
                url     = r.get("url", "")
                snippet = r.get("snippet", "")
                with st.expander(f"{i}. {title}", expanded=(i <= 3)):
                    if url:
                        st.code(url)
                    st.write(snippet or "(no snippet)")
                    if "text" in r and r.get("text"):
                        st.text(r["text"][:1500] + ("..." if len(r["text"])>1500 else ""))
        else:
            # String or other shape
            st.write(ro)

    st.caption("This tab tries: local web_search.py → JAMbot /web-search → DuckDuckGo lite fallback.")

    st.divider()
    st.subheader("Embed Content into DB (via JAMbot)")
    with st.form("embed_content_form"):
        generic_url = st.text_input("Generic URL (http/https)")
        confluence  = st.text_input("Confluence URL")
        space_key   = st.text_input("Confluence Space Key")
        title       = st.text_input("Confluence Page Title")
        table_name  = st.text_input("Target table", value="embedded_logs")
        model_name  = st.text_input("Model", value="Alex")
        up2         = st.file_uploader("Or upload a file", type=None, accept_multiple_files=False)
        submitted   = st.form_submit_button("Embed")

    if submitted:
        files = None
        data_form = {
            "generic_url": generic_url or "",
            "confluence":  confluence  or "",
            "space_key":   space_key   or "",
            "title":       title       or "",
            "table_name":  table_name,
            "model":       model_name,
        }
        if up2:
            files = {"file": (up2.name, up2, "application/octet-stream")}
        ok, resp = jambot_post_form("/embed-content", data=data_form, files=files)
        st.write(resp if ok else f"❌ {resp}")

    # ---- Kick off workflow using the *persisted* results ----
    if do_kickoff:
        last = st.session_state.last_search
        if not last["results"]:
            st.warning("Run a search first.")
        else:
            # Build an 8K text blob from structured results if possible
            if isinstance(last["results"], dict) and "results" in last["results"]:
                lines = []
                for r in last["results"]["results"][:15]:
                    lines.append(f"- {r.get('title','(no title)')} :: {r.get('url','')} :: {r.get('snippet','')}")
                blob = "\n".join(lines)
            else:
                blob = str(last["results"])
            blob = blob[:8000]  # keep payload reasonable

            ok, resp = trigger_workflow_from_search(
                last["query"], blob, chat_url, token, inference_profile_arn,
                auto_iter=st.session_state.auto_iter, max_iters=int(st.session_state.max_iters)
            )
            st.write(resp if ok else f"❌ {resp}")
            if ok and isinstance(resp, dict):
                if resp.get("bundle_path"):
                    st.success(f"Bundle created: {resp['bundle_path']}")
                if resp.get("final_summary"):
                    st.subheader("Final Summary")
                    st.write(resp["final_summary"])
                if resp.get("iterations"):
                    st.subheader("Iterations")
                    for it in resp["iterations"]:
                        with st.expander(f"Iteration {it.get('iteration')}"):
                            st.code(it.get("resources_preview",""))
                            st.code(it.get("data_preview",""))
                            st.write(it.get("summary",""))
                            st.json(it.get("validation",{}))

            # Best-effort journal note
            _ = jambot_post_json("/journal", json_body={"entry": f"Workflow kicked from web search [{last['query']}] @ {datetime.now().isoformat()}"})


# ---------- Log Analysis Tab ----------

with tabs[3]:
    st.subheader("Analyze Logs")
    up = st.file_uploader("Upload one or more log files", type=None, accept_multiple_files=True)
    run_bertopic = st.checkbox("Run BERTopic clustering (if bert2.py present)")

    if st.button("Analyze", use_container_width=True, key="analyze_btn"):
        texts: List[str] = []
        if up:
            for f in up:
                try:
                    texts.append(f.read().decode("utf-8", errors="replace"))
                except Exception:
                    texts.append(f.read().decode("latin-1", errors="replace"))
        if not texts:
            st.warning("Please upload at least one log file.")
        else:
            corpus = "\n".join(texts)
            st.write(f"Total characters: {len(corpus):,}")

            token_env = os.environ.get("COVERITY_ASSIST_TOKEN", "")
            use_token = token or token_env
            if use_token:
                ok, summary = coverity_chat(
                    chat_url,
                    use_token,
                    "Analyze the following logs thoroughly and return a structured summary with timelines, counts, and suspected root causes. "
                    "Keep it concise but explicit. Logs:\n\n" + corpus[:150000],
                    read_prompts().get("Claud"),
                    max_tokens=1200,
                    use_top_level_system=False,
                    inference_profile_arn=inference_profile_arn,
                )
                st.subheader("LLM Summary (Claud)")
                st.write(summary if ok else f"❌ {summary}")
            else:
                st.info("Provide Bearer token to get an LLM-assisted summary with Claud.")

            if run_bertopic and bert2_mod is not None:
                try:
                    from bertopic import BERTopic
                    docs = [ln for ln in corpus.splitlines() if ln.strip()][:3000]
                    topic_model = BERTopic(min_topic_size=15, verbose=False)
                    topics, _ = topic_model.fit_transform(docs)
                    st.subheader("BERTopic — Topical Overview")
                    st.write(topic_model.get_topic_info().head(20))
                except Exception as e:
                    st.warning(f"BERTopic unavailable or failed: {e}")

    st.divider()
    st.subheader("Combine AI Response Logs")
    src = st.text_input("Responses directory (e.g., ai_responses)", value=str(Path.cwd() / "ai_responses"))
    dst = st.text_input("Combined output directory", value=str(Path(src) / "combined_logs"))
    if st.button("Combine Logs", key="combine_btn"):
        if combine_logs_mod is None:
            st.error("combine_logs.py not found in the working dir.")
        else:
            try:
                combine_logs_mod.combine_logs(src, dst)  # type: ignore[attr-defined]
                st.success("Combined logs written.")
            except Exception as e:
                st.error(f"Combine failed: {e}")

# ---------- Scripts & Workflow Tab ----------

with tabs[4]:
    st.subheader("Scripts & Workflow")

    st.caption(f"Scripts dir: {SCRIPTS_DIR}")
    files = sorted(Path(SCRIPTS_DIR).glob("*"))
    sel = st.selectbox("Select a script", options=["(new)"] + [f.name for f in files])

    editor_key = "script_editor_buf"
    if sel == "(new)":
        new_name = st.text_input("New filename (e.g., demo.sh)", value="demo.sh")
        content = st.text_area("Contents", height=240, key=editor_key, value="#!/usr/bin/env bash\n\necho Hello from JAMbot script\n")
        if st.button("Save New"):
            p = Path(SCRIPTS_DIR) / new_name
            p.write_text(content, encoding="utf-8")
            st.success(f"Saved {p}")
    else:
        p = Path(SCRIPTS_DIR) / sel
        try:
            buf = p.read_text(encoding="utf-8")
        except Exception as e:
            buf = f"# Error reading file: {e}"
        content = st.text_area("Contents", height=240, key=editor_key, value=buf)
        colx, coly = st.columns(2)
        with colx:
            if st.button("Save Changes"):
                p.write_text(content, encoding="utf-8")
                st.success(f"Saved {p}")
        with coly:
            args = st.text_input("Arguments", value="")
            if st.button("Run Script"):
                ok, out = run_local_script(p, args.split())
                st.code(out)
                st.success("OK" if ok else "Failed")

    st.divider()
    st.write("**Trigger JAMbot Workflow (/trigger-workflow)**")
    if st.button("Trigger Workflow Now"):
        ok, resp = jambot_post_json("/trigger-workflow", json_body={
            "original_request": "Streamlit requested workflow run",
            "task_description": "Kick off the default pipeline.",
            "chat_url": chat_url,
            "token": token,
            "inference_profile_arn": inference_profile_arn,
            "auto_iterate": st.session_state.auto_iter,
            "max_iters": int(st.session_state.max_iters),
        })
        st.write(resp if ok else f"❌ {resp}")

# ---------- DB Tools Tab ----------

with tabs[5]:
    st.subheader("DB Tools (via JAMbot /tools)")
    st.caption("This executes allowed commands in JAMbot's bash sandbox (see /tools/allowed on server).")

    colA, colB = st.columns(2)
    with colA:
        if st.button("Show Allowed Commands"):
            r = requests.get(f"{JAMBOT_BASE_URL}/tools/allowed", timeout=15)
            st.code(r.text if r.status_code == 200 else f"[{r.status_code}] {r.text}")

    with colB:
        cmd = st.text_input("Command to run (must be in allowed list)", value="echo hello")
        if st.button("Run (bash)"):
            ok, resp = jambot_post_json("/tools/bash", json_body={"argument": cmd})
            st.write(resp if ok else f"❌ {resp}")

st.caption("Tip: Point 'Base URL' to your JAMbot gateway. Search config & credentials are saved on the gateway when exposed.")
