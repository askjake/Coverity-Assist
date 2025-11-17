from __future__ import annotations

# =========================
# Metrics singletons (safe on reload)
# =========================
import os
from typing import Any, Dict, List, Optional

try:
    from prometheus_client import REGISTRY
    from prometheus_client import Counter as _PCounter
    from prometheus_client import Histogram as _PHistogram
    from prometheus_client import Summary as _PSummary
    from prometheus_client import Gauge as _PGauge
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    _PROM_AVAILABLE = True
except Exception:
    REGISTRY = None
    _PCounter = _PHistogram = _PSummary = _PGauge = None
    _PROM_AVAILABLE = False

_DISABLE_METRICS = os.getenv("CA_DISABLE_METRICS", "").lower() in {"1", "true", "yes", "on"}


class _Dummy:
    def labels(self, *a, **k): return self
    def inc(self, *a, **k): return None
    def observe(self, *a, **k): return None
    def set(self, *a, **k): return None


def _probe(name: str):
    """Return existing collector if already registered, else None."""
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


# =========================
# App + endpoints
# =========================
import json
import socket
import time
import requests

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import Response, PlainTextResponse
from pydantic import BaseModel

# absolute import so it works whether we're a package or just /app on PYTHONPATH
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


def _get_token() -> str:
    return os.getenv("COVERITY_ASSIST_TOKEN", "")


def get_token_len() -> int:
    return len(_get_token())


COVERITY_ASSIST_URL = os.getenv(
    "COVERITY_ASSIST_URL", "http://coverity-assist.dishtv.technology/chat"
).rstrip("/")

app = FastAPI(title="coverity-assist-proxy", version="1.0")

# Middleware to surface request telemetry + token/profile hints as headers
app.add_middleware(
    RequestTelemetryMiddleware,
    infer_arn_getter=get_inference_profile_arn,
    token_len_getter=get_token_len,
)

# ---------- Prom metrics (idempotent) ----------
TRACK = os.getenv("CA_PLANE", os.getenv("TRACK", "stable"))
REQ_COUNT = _get_or_create_counter(
    "ca_http_requests_total", "Total HTTP requests", ["track", "method", "path", "status"]
)
REQ_LAT = _get_or_create_histogram(
    "ca_http_request_duration_seconds", "Request latency (seconds)", ["track", "path"]
)
APP_UP = _get_or_create_gauge("ca_app_up", "1 when app is up", ["track"])
APP_UP.labels(track=TRACK).set(1)

# Static app metadata for dashboards
APP_INFO = _get_or_create_gauge("ca_app_info", "Runtime version metadata", ["version", "image", "track"])
APP_INFO.labels(
    version=os.getenv("CA_REVISION", "unknown"),
    image=os.getenv("IMG", "unknown"),
    track=os.getenv("CA_PLANE", "stable"),
).set(1)


@app.middleware("http")
async def _metrics_and_log(request: Request, call_next):
    t0 = time.perf_counter()
    resp = await call_next(request)
    # metrics
    try:
        REQ_COUNT.labels(
            track=TRACK, method=request.method, path=request.url.path, status=str(resp.status_code)
        ).inc()
        REQ_LAT.labels(track=TRACK, path=request.url.path).observe(time.perf_counter() - t0)
    except Exception:
        pass
    # single JSON line for Loki/Grafana
    try:
        print(json.dumps({
            "log": "ca_request",
            "track": TRACK,
            "method": request.method,
            "path": request.url.path,
            "status": resp.status_code,
            "duration_ms": int((time.perf_counter() - t0) * 1000),
            "pod": os.getenv("HOSTNAME", ""),
            "ns": os.getenv("POD_NAMESPACE", ""),
        }))
    except Exception:
        pass
    return resp


def _bearer_header() -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {_get_token()}",
        "Content-Type": "application/json",
    }


class ChatIn(BaseModel):
    messages: List[Dict[str, Any]]
    max_tokens: int = 800
    system: Optional[str] = None
    inference_profile_arn: Optional[str] = None


class ValidateIn(BaseModel):
    original_request: str
    summary: str
    max_tokens: int = 300


@app.get("/health")
def health():
    arn = get_inference_profile_arn()
    return {
        "status": "OK",
        "has_token": get_token_len() > 0,
        "inference_profile_arn": arn or None,
    }


@app.get("/whoami")
def whoami(request: Request):
    return {
        "pod": os.getenv("HOSTNAME", ""),
        "namespace": os.getenv("POD_NAMESPACE", ""),
        "inference_profile_arn": os.getenv("INFERENCE_PROFILE_ARN", ""),
        "server": socket.gethostname(),
        "x-canary": request.headers.get("x-canary", ""),
    }


@app.get("/metrics")
def metrics():
    if not _PROM_AVAILABLE:
        return PlainTextResponse("# prometheus_client not installed\n", media_type="text/plain; version=0.0.4")
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/chat")
def chat(body: ChatIn = Body(...)):
    token = _get_token()
    if not token:
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


@app.post("/validate")
def validate(body: ValidateIn):
    token = _get_token()
    if not token:
        raise HTTPException(status_code=401, detail="Missing COVERITY_ASSIST_TOKEN")

    system = "Return STRICT JSON only. No prose."
    user = (
        "Original request:\n---\n" + body.original_request + "\n---\n\n"
        "Did the summary below fully satisfy the request? If not, list concrete next actions "
        "(bash commands or URLs) we should run/fetch next.\n\n"
        "Summary:\n---\n" + body.summary + "\n---\n\n"
        'Respond JSON with keys: "complete": true|false, "next_actions": [ {"cmd": "."}, {"url": "."} ]'
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
        # Try to normalize to dict with required keys
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

