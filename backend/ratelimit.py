"""Cheap per-owner rate limiting for the paid (LLM) endpoints.

Every paid request spends real money, so a single client must not be able to loop on them. This is
a *per-instance, in-memory* token bucket keyed by owner: no Mongo, no network, a few microseconds.
Serverless instances do not share memory, so on its own the real ceiling is "limit x live instances".
When a shared cache is configured (``backend/cache.py``, Upstash-style Redis over HTTPS) a request that
passes the local bucket is also checked against one shared, atomic bucket per owner, so the limit holds
across instances (fails open if the cache is down). Also pair it with Vercel Firewall rate-limit rules
(see DEPLOYMENT.md) which are enforced at the edge for all instances.

Defaults (env-tunable, see ``backend/config.py`` / ``.env.example``):

* signed-in user : 30 requests / minute and 300 / hour
* guest          : 10 requests / minute and 60 / hour

Each limit is a bucket whose capacity is the quota and which refills continuously, so a short burst
up to the quota is fine and a sustained flood is throttled. A limit of 0 disables that window.
Over the limit -> ``429`` JSON with ``Retry-After`` (no ``llm`` block: no model was involved).

Use as a route dependency::

    @router.post("/generate", dependencies=[Depends(rate_limit)])

``RATE_LIMIT_DISABLED=1`` switches it off for load tests, but ONLY when ``VERCEL`` is unset.
"""
from __future__ import annotations

import asyncio
import math
import threading
import time
from collections import OrderedDict

from fastapi import Depends, HTTPException

from backend import cache, config
from backend.auth import current_owner

GUEST_PREFIX = "guest_"
_MAX_OWNERS = 20_000

_lock = threading.Lock()
# owner -> [minute_tokens, hour_tokens, last_monotonic]
_state: "OrderedDict[str, list[float]]" = OrderedDict()


def limits_for(owner: str) -> tuple[int, int]:
    """``(per_minute, per_hour)`` for this owner (guests are tighter than signed-in users)."""
    if owner.startswith(GUEST_PREFIX):
        return config.RATE_LIMIT_GUEST_PER_MIN, config.RATE_LIMIT_GUEST_PER_HOUR
    return config.RATE_LIMIT_USER_PER_MIN, config.RATE_LIMIT_USER_PER_HOUR


def check(owner: str, now: float | None = None) -> float:
    """Consume one request for ``owner``. Returns 0.0 if allowed, else the seconds to wait."""
    per_min, per_hour = limits_for(owner)
    if per_min <= 0 and per_hour <= 0:
        return 0.0
    now = time.monotonic() if now is None else now
    with _lock:
        st = _state.get(owner)
        if st is None:
            st = [float(per_min), float(per_hour), now]
            _state[owner] = st
            while len(_state) > _MAX_OWNERS:
                _state.popitem(last=False)
        else:
            _state.move_to_end(owner)
            dt = max(0.0, now - st[2])
            if per_min > 0:
                st[0] = min(float(per_min), st[0] + dt * per_min / 60.0)
            if per_hour > 0:
                st[1] = min(float(per_hour), st[1] + dt * per_hour / 3600.0)
            st[2] = now
        wait = 0.0
        if per_min > 0 and st[0] < 1.0:
            wait = max(wait, (1.0 - st[0]) * 60.0 / per_min)
        if per_hour > 0 and st[1] < 1.0:
            wait = max(wait, (1.0 - st[1]) * 3600.0 / per_hour)
        if wait > 0:
            return wait
        if per_min > 0:
            st[0] -= 1.0
        if per_hour > 0:
            st[1] -= 1.0
        return 0.0


def check_shared(owner: str) -> float:
    """Second line of defence, only when a shared cache is configured (``backend/cache.py``): the owner's
    bucket lives in Redis, so the limit holds across ALL serverless instances instead of per instance.
    Returns seconds to wait (0.0 = allowed). Fails OPEN: any cache problem keeps the in-process verdict, and
    a request that the shared bucket denies hands its in-process token back."""
    per_min, per_hour = limits_for(owner)
    if (per_min <= 0 and per_hour <= 0) or not cache.enabled():
        return 0.0
    wait = cache.bucket_take(owner, per_min, per_hour)
    if wait is None or wait <= 0:
        return 0.0
    with _lock:
        st = _state.get(owner)
        if st is not None:                       # refund the local token this denied request consumed
            st[0] = min(float(per_min), st[0] + 1.0) if per_min > 0 else st[0]
            st[1] = min(float(per_hour), st[1] + 1.0) if per_hour > 0 else st[1]
    return wait


def reset() -> None:
    with _lock:
        _state.clear()


async def rate_limit(owner: str = Depends(current_owner)) -> None:
    """FastAPI dependency: 429 + ``Retry-After`` when ``owner`` is over its quota."""
    if config.rate_limit_disabled():
        return
    wait = check(owner)
    if wait <= 0 and cache.enabled():
        wait = await asyncio.to_thread(check_shared, owner)       # one bounded (<=150 ms) round trip, off the loop
    if wait > 0:
        retry = max(1, math.ceil(wait))
        raise HTTPException(
            status_code=429,
            detail=f"You're sending requests too quickly. Please wait {retry} second{'s' if retry != 1 else ''} and try again.",
            headers={"Retry-After": str(retry)},
        )
