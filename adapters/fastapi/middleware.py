"""
FastAPI/Starlette middleware hook for margin.

    from adapters.fastapi.middleware import MarginMiddleware
    app.add_middleware(MarginMiddleware)

    # GET /margin/health → typed health expression
    # GET /margin/health.json → JSON
"""

from __future__ import annotations

import time
import threading
from collections import deque
from datetime import datetime
from typing import Optional


class MarginMiddleware:
    """
    ASGI middleware that tracks request metrics and exposes
    a health endpoint.

    Collects per-request: status code, duration.
    Computes over a sliding window: p50, p99, error rate, rps.
    """

    def __init__(
        self,
        app,
        window_seconds: float = 60.0,
        health_path: str = "/margin/health",
    ):
        self.app = app
        self.window_seconds = window_seconds
        self.health_path = health_path
        self._requests: deque = deque()
        self._lock = threading.Lock()

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope["path"] == self.health_path:
            await self._serve_health(scope, receive, send)
            return

        if scope["type"] == "http" and scope["path"] == self.health_path + ".json":
            await self._serve_health_json(scope, receive, send)
            return

        start = time.monotonic()
        status_code = 500  # default if something goes wrong

        async def send_wrapper(message):
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            if scope["type"] == "http":
                duration_ms = (time.monotonic() - start) * 1000
                with self._lock:
                    self._requests.append({
                        "time": time.monotonic(),
                        "duration_ms": duration_ms,
                        "status": status_code,
                    })

    def _prune(self) -> list[dict]:
        """Remove old entries, return current window."""
        cutoff = time.monotonic() - self.window_seconds
        with self._lock:
            while self._requests and self._requests[0]["time"] < cutoff:
                self._requests.popleft()
            return list(self._requests)

    def _compute_metrics(self) -> dict[str, float]:
        window = self._prune()
        if not window:
            return {}

        durations = sorted(r["duration_ms"] for r in window)
        n = len(durations)
        errors = sum(1 for r in window if r["status"] >= 500)
        successes = sum(1 for r in window if 200 <= r["status"] < 300)
        elapsed = self.window_seconds

        metrics = {
            "p50_ms": durations[n // 2] if n > 0 else 0.0,
            "p99_ms": durations[int(n * 0.99)] if n > 0 else 0.0,
            "error_rate": errors / n if n > 0 else 0.0,
            "success_rate": successes / n if n > 0 else 0.0,
            "rps": n / elapsed if elapsed > 0 else 0.0,
        }
        return metrics

    def get_expression(self):
        from .endpoints import endpoint_expression
        metrics = self._compute_metrics()
        return endpoint_expression(metrics, endpoint="service", measured_at=datetime.now())

    async def _serve_health(self, scope, receive, send):
        expr = self.get_expression()
        body = expr.to_string().encode()
        await send({"type": "http.response.start", "status": 200,
                     "headers": [[b"content-type", b"text/plain"]]})
        await send({"type": "http.response.body", "body": body})

    async def _serve_health_json(self, scope, receive, send):
        expr = self.get_expression()
        body = expr.to_json().encode()
        await send({"type": "http.response.start", "status": 200,
                     "headers": [[b"content-type", b"application/json"]]})
        await send({"type": "http.response.body", "body": body})
