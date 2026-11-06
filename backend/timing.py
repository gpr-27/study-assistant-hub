"""Per-request timing collector -> the ``Server-Timing`` response header.

A request-scoped dict lives in a ContextVar; ``backend.main``'s middleware installs it, and
auth/db code adds named durations with ``timed("auth")``. The dict object itself is shared into
FastAPI's threadpool (contexts are copied, the dict is not), so sync routes can record too.
No-ops outside a request.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from contextvars import ContextVar

_current: ContextVar[dict | None] = ContextVar("request_timings", default=None)


def start() -> dict:
    timings: dict = {}
    _current.set(timings)
    return timings


def add(name: str, ms: float) -> None:
    timings = _current.get()
    if timings is not None:
        timings[name] = timings.get(name, 0.0) + ms


@contextmanager
def timed(name: str):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        add(name, (time.perf_counter() - t0) * 1000.0)


def header(timings: dict, total_ms: float) -> str:
    parts = [f"{k};dur={v:.1f}" for k, v in timings.items()]
    parts.append(f"total;dur={total_ms:.1f}")
    return ", ".join(parts)
