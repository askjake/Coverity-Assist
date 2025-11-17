
from __future__ import annotations
# --- metrics singletons (avoid duplicate registration on re-import) ---
import os
try:
    from prometheus_client import REGISTRY
    from prometheus_client import Counter as _PCounter
    from prometheus_client import Histogram as _PHistogram
    from prometheus_client import Summary as _PSummary
    from prometheus_client import Gauge as _PGauge
except Exception:
    REGISTRY = None
    _PCounter = _PHistogram = _PSummary = _PGauge = None  # testing / no prometheus

_DISABLE_METRICS = os.getenv("CA_DISABLE_METRICS", "").lower() in {"1","true","yes","on"}

class _Dummy:
    def labels(self, *a, **k): return self
    def inc(self, *a, **k): return None
    def observe(self, *a, **k): return None
    def set(self, *a, **k): return None

def _probe(name: str):
    if REGISTRY is None:
        return None
    try:
        for collector, names in getattr(REGISTRY, "_collector_to_names", {}).items():
            if name in names:
                return collector
    except Exception:
        pass
    return None

def _get_or_create(MetricCls, name, documentation, labelnames=(), **kwargs):
    if _DISABLE_METRICS or MetricCls is None:
        return _Dummy()
    existing = _probe(name)
    return existing if existing else MetricCls(name, documentation, labelnames=labelnames, **kwargs)

def _get_or_create_counter(name, documentation, labelnames=(), **kwargs):
    return _get_or_create(_PCounter, name, documentation, labelnames, **kwargs)

def _get_or_create_histogram(name, documentation, labelnames=(), **kwargs):
    return _get_or_create(_PHistogram, name, documentation, labelnames, **kwargs)

def _get_or_create_summary(name, documentation, labelnames=(), **kwargs):
    return _get_or_create(_PSummary, name, documentation, labelnames, **kwargs)

def _get_or_create_gauge(name, documentation, labelnames=(), **kwargs):
    return _get_or_create(_PGauge, name, documentation, labelnames, **kwargs)
# --- end metrics singletons ---

import json
import os
from typing import Any, Dict, List, Optional

import requests
from fastapi import Body, FastAPI, HTTPException
from pydantic import BaseModel

# NOTE: absolute import so it works whether we're a package or just /app on PYTHONPATH
from middleware import RequestTelemetryMiddleware, set_token_usage


def _first_env(*names: str) -> str:
    for n in names:
        v = os.getenv(n)
        if v:
            return v
    return ""


def get_inference_profile_arn() -> str:
    return _first_env(
        "INFERENCE_PROFILE_ARN",
        "APPLICATION_INFERENCE_PROFILE_ARN",
        "BEDROCK_APPLICATION_INFERENCE_PROFILE_ARN",
        "BEDROCK_INFERENCE_PROFILE_ARN",
        "APPLICATION_PROFILE_ARN",
        "PROFILE_ARN",
    )


def get_token_len() -> int:
    return len(os.getenv("COVERITY_ASSIST_TOKEN", ""))


COVERITY_ASSIST_URL = os.getenv(
    "COVERITY_ASSIST_URL", "http://coverity-assist.dishtv.technology/chat"
).rstrip("/")
TOKEN = os.getenv("COVERITY_ASSIST_TOKEN", "")

app = FastAPI(title="coverity-assist-proxy", version="1.0")

app.add_middleware(
    RequestTelemetryMiddleware,
    infer_arn_getter=get_inference_profile_arn,
    token_len_getter=get_token_len,
)

def _bearer_header() -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {TOKEN}",
        "Content-Type": "application/json",
    }


class ChatIn(BaseModel):
    messages: List[Dict[str, Any]]
    max_tokens: int = 800
    system: Optional[str] = None
    inference_profile_arn: Optional[str] = None


@app.get("/health")
def health():
    arn = get_inference_profile_arn()
    return {
        "status": "OK",
        "has_token": get_token_len() > 0,
        "inference_profile_arn": arn or None,
    }


@app.post("/chat")
def chat(body: ChatIn = Body(...)):
    if not TOKEN:
        raise HTTPException(status_code=401, detail="Missing COVERITY_ASSIST_TOKEN")

    payload: Dict[str, Any] = {
        "messages": body.messages,
        "max_tokens": body.max_tokens,
    }
    if body.system:
        payload["system"] = body.system

    req_arn = body.inference_profile_arn or get_inference_profile_arn()
    if req_arn:
        payload["inference_profile_arn"] = req_arn

    try:
        r = requests.post(
            COVERITY_ASSIST_URL,
            headers=_bearer_header(),
            json=payload,
            timeout=120,
        )
        r.raise_for_status()
        data = r.json()
        usage = data.get("usage") if isinstance(data, dict) else None
        if usage is not None:
            try:
                set_token_usage(usage)
            except Exception:
                pass
        return data
    except requests.HTTPError as e:
        raise HTTPException(status_code=r.status_code, detail=r.text) from e


class ValidateIn(BaseModel):
    original_request: str
    summary: str
    max_tokens: int = 300


@app.post("/validate")
def validate(body: ValidateIn):
    if not TOKEN:
        raise HTTPException(status_code=401, detail="Missing COVERITY_ASSIST_TOKEN")

    system = "Return STRICT JSON only. No prose."
    user = (
        "Original request:\n---\n"
        + body.original_request
        + "\n---\n\n"
        "Did the summary below fully satisfy the request? If not, list concrete next actions "
        "(bash commands or URLs) we should run/fetch next.\n\n"
        "Summary:\n---\n"
        + body.summary
        + "\n---\n\n"
        'Respond JSON with keys: "complete": true|false, '
        '"next_actions": [ {"cmd": "."}, {"url": "."} ]'
    )

    payload: Dict[str, Any] = {
        "messages": [{"role": "user", "content": user}],
        "max_tokens": body.max_tokens,
        "system": system,
    }

    arn = get_inference_profile_arn()
    if arn:
        payload["inference_profile_arn"] = arn

    try:
        r = requests.post(
            COVERITY_ASSIST_URL,
            headers=_bearer_header(),
            json=payload,
            timeout=120,
        )
        r.raise_for_status()
        try:
            resp_data = r.json()
        except ValueError:
            resp_data = {}

        textish = (
            (resp_data.get("content") if isinstance(resp_data, dict) else None)
            or (resp_data.get("response") if isinstance(resp_data, dict) else None)
            or (resp_data.get("text") if isinstance(resp_data, dict) else None)
            or r.text
        )

        try:
            data = json.loads(textish)
        except Exception:
            data = {"complete": False, "next_actions": []}

        if not isinstance(data, dict):
            data = {"complete": False, "next_actions": []}

        data.setdefault("complete", False)
        data.setdefault("next_actions", [])
        return data
    except requests.HTTPError as e:
        raise HTTPException(status_code=r.status_code, detail=r.text) from e


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
# --- debug endpoint to see which pod handled the request ---
from fastapi import Request
import os, socket

@app.get("/whoami")
def whoami(request: Request):
    return {
        "pod": os.getenv("HOSTNAME",""),
        "namespace": os.getenv("POD_NAMESPACE",""),
        "inference_profile_arn": os.getenv("INFERENCE_PROFILE_ARN",""),
        "server": socket.gethostname(),
        "x-canary": request.headers.get("x-canary", ""),
    }
# --- metrics + request logging (append) ---
import os, time, json, socket
from fastapi import Request
from fastapi.responses import Response, PlainTextResponse

TRACK = os.getenv("TRACK", "stable")

try:
    from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
    PROM_READY = True
except Exception:
    PROM_READY = False

if PROM_READY:
    REQ_COUNT = _get_or_create_counter(
        "ca_http_requests_total",
        "Total HTTP requests",
        ["track", "method", "path", "status"]
    )
    REQ_LAT = _get_or_create_histogram(
        "ca_http_request_duration_seconds",
        "Request latency (seconds)",
        ["track", "path"]
    )
    APP_UP = _get_or_create_gauge("ca_app_up", "1 when app is up", ["track"])
    APP_UP.labels(track=TRACK).set(1)

@app.middleware("http")
async def _metrics_and_log(request: Request, call_next):
    t0 = time.perf_counter()
    resp = await call_next(request)
    if PROM_READY:
        REQ_COUNT.labels(track=TRACK, method=request.method, path=request.url.path, status=str(resp.status_code)).inc()
        REQ_LAT.labels(track=TRACK, path=request.url.path).observe(time.perf_counter() - t0)
    # single JSON line for Loki/Grafana
    try:
        print(json.dumps({
            "log": "ca_request",
            "track": TRACK,
            "method": request.method,
            "path": request.url.path,
            "status": resp.status_code,
            "duration_ms": int((time.perf_counter()-t0)*1000),
            "pod": os.getenv("HOSTNAME",""),
            "ns": os.getenv("POD_NAMESPACE",""),
        }))
    except Exception:
        pass
    return resp

@app.get("/metrics")
def _metrics():
    if not PROM_READY:
        return PlainTextResponse("# prometheus_client not installed\n", media_type="text/plain; version=0.0.4")
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
# --- tiny Prom metrics so dashboards can split stable vs canary ---
import os
from fastapi import Response
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST

CA_PLANE = os.getenv("CA_PLANE", "stable")
CA_REV   = os.getenv("CA_REVISION", "dev")

REQS = _get_or_create_counter(
    "ca_http_requests_total",
    "Total HTTP requests",
    ["endpoint", "plane", "revision"],
)

def _bump(endpoint: str):
    REQS.labels(endpoint=endpoint, plane=CA_PLANE, revision=CA_REV).inc()

@app.get("/health")
def health():
    _bump("health")
    return {"status": "OK", "has_token": bool(os.getenv("COVERITY_ASSIST_TOKEN")), "inference_profile_arn": os.getenv("INFERENCE_PROFILE_ARN","")}

@app.get("/whoami")
def whoami(request: Request):
    _bump("whoami")
    return {
        "pod": os.getenv("HOSTNAME",""),
        "namespace": os.getenv("POD_NAMESPACE",""),
        "inference_profile_arn": os.getenv("INFERENCE_PROFILE_ARN",""),
        "server": socket.gethostname(),
        "x-canary": request.headers.get("x-canary", ""),
    }

@app.get("/metrics")
def metrics():
    _bump("metrics")
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


