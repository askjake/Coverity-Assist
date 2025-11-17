from __future__ import annotations

import time
import uuid
from typing import Any, Callable, Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.types import ASGIApp

import contextvars

# best-effort token usage propagation
_token_usage: contextvars.ContextVar[Optional[Any]] = contextvars.ContextVar(
    "token_usage", default=None
)


def set_token_usage(value: Any) -> None:
    _token_usage.set(value)


class RequestTelemetryMiddleware(BaseHTTPMiddleware):
    def __init__(
        self,
        app: ASGIApp,
        infer_arn_getter: Callable[[], str],
        token_len_getter: Callable[[], int],
    ) -> None:
        super().__init__(app)
        self._get_arn = infer_arn_getter
        self._get_tok_len = token_len_getter

    async def dispatch(self, request: Request, call_next):
        req_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        t0 = time.perf_counter()

        response = await call_next(request)

        dt_ms = int((time.perf_counter() - t0) * 1000)
        arn = self._get_arn()
        tok_len = self._get_tok_len()

        # headers
        response.headers["x-request-id"] = req_id
        response.headers["x-duration-ms"] = str(dt_ms)
        response.headers["x-token-present"] = "true" if tok_len > 0 else "false"
        if arn:
            response.headers["x-inference-profile"] = arn

        usage = _token_usage.get()
        if usage is not None:
            response.headers["x-token-usage"] = str(usage)

        return response
