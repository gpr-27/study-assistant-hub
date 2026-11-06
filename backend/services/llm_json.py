"""Tiny helpers shared by the chat-side services: tolerant JSON extraction from a model reply and
parallel fan-out of blocking DB reads. No provider code lives here (that is ``backend.llm_chain``)."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable


def extract_json(text: str | None) -> Any:
    """Parse the first JSON object/array in ``text`` (fences, prose, truncation tolerated).

    Delegates to the shared ``_gen.parse_json``; returns ``None`` when nothing parses.
    """
    if not text:
        return None
    from backend.services import _gen

    return _gen.parse_json(text)


def extract_json_object(text: str | None) -> dict:
    """Like :func:`extract_json` but always returns a dict (``{}`` if the reply is not an object)."""
    data = extract_json(text)
    return data if isinstance(data, dict) else {}


def parallel(calls: list[Callable[[], Any]], max_workers: int = 6) -> list[Any]:
    """Run blocking zero-arg callables concurrently and return their results in order.

    Used to collapse N sequential MongoDB round-trips into ~1. The pool is created and torn down
    per call (no lingering threads). An exception in one call is re-raised after all finished.
    """
    if not calls:
        return []
    if len(calls) == 1:
        return [calls[0]()]
    with ThreadPoolExecutor(max_workers=min(max_workers, len(calls))) as pool:
        futures = [pool.submit(c) for c in calls]
        return [f.result() for f in futures]
