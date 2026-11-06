"""Shared cache / state layer with a pluggable backend.

Why: the database ceiling on a free MongoDB Atlas M0 is ~100 ops/s and serverless instances do not
share memory. This module gives every instance a *cheap, optional, shared* place to keep small,
derived, expendable state (read caches, rate-limit buckets, "this guest was just touched", the
LLM chain's "this key is out of quota" memory ...). It is NEVER a source of truth: everything stored
here can be lost at any time and is recomputed from MongoDB.

Backends (``CACHE_BACKEND`` = ``auto`` | ``memory`` | ``upstash``; default ``auto``):

* ``memory``  - in-process only. This is the historical behaviour: callers keep using their own
  per-process caches and every shared helper below is a no-op (``enabled()`` is False).
* ``upstash`` - any Redis reachable over the Upstash-style HTTPS REST API
  (``UPSTASH_REDIS_REST_URL`` + ``UPSTASH_REDIS_REST_TOKEN``; the Vercel Marketplace aliases
  ``KV_REST_API_URL`` / ``KV_REST_API_TOKEN`` are accepted too). Plain ``httpx``; no Redis client.
  ``auto`` selects it when both variables are set, otherwise ``memory``.

Safety rules (all enforced here, so call sites stay trivial):

* Never block a request on the cache: each call has a hard ``CACHE_TIMEOUT_MS`` (default 120, max 150)
  budget and any failure returns "miss" / ``None`` so the caller falls back to its in-process path.
* A circuit breaker on the backend itself: 3 consecutive failures open it for 20 s (calls return
  immediately), then one probe is allowed.
* Keys are namespaced (``sah:v1:<ns>:...``) and owner-scoped: the owner id (a guest id is a bearer
  credential) is never stored in clear text, only ``sha256(owner)[:24]``.
* Values are small JSON (<= ``MAX_VALUE_BYTES``); larger values are silently not cached.
* Every value has a TTL; nothing lives forever.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import queue
import threading
import time
from collections import OrderedDict
from typing import Any, Callable

log = logging.getLogger("backend.cache")

PREFIX = "sah:v1"
MAX_VALUE_BYTES = 48 * 1024
_FAIL_THRESHOLD = 3
_OPEN_S = 20.0


def owner_tag(owner: str) -> str:
    """Stable, non-reversible tag for an owner (never put the raw id in a shared key)."""
    return hashlib.sha256((owner or "").encode("utf-8", "ignore")).hexdigest()[:24]


def key(ns: str, owner: str | None = None, *parts: str) -> str:
    """``sah:v1:<ns>[:<owner-tag>][:<part>...]``."""
    bits = [PREFIX, ns]
    if owner is not None:
        bits.append(owner_tag(owner))
    bits.extend(str(p) for p in parts)
    return ":".join(bits)


# ------------------------------------------------------------------------------------------------
# Backends. Interface: execute(commands) -> list of results (None on a per-command error).
# ------------------------------------------------------------------------------------------------
class BackendError(Exception):
    pass


class MemoryBackend:
    """Tiny in-process Redis subset (GET/SET/DEL/INCR/EXPIRE/HSET/HGETALL/EVAL for the bucket).

    Used as the default backend and by the tests as a stand-in for a shared Redis."""

    name = "memory"

    def __init__(self, max_keys: int = 20_000) -> None:
        self._lock = threading.Lock()
        self._d: "OrderedDict[str, list]" = OrderedDict()      # key -> [expires_at | None, value]
        self._max = max_keys
        self.calls = 0

    def _live(self, k: str):
        ent = self._d.get(k)
        if ent is None:
            return None
        if ent[0] is not None and ent[0] <= time.monotonic():
            del self._d[k]
            return None
        return ent

    def execute(self, commands: list[list]) -> list:
        out = []
        with self._lock:
            self.calls += 1
            for cmd in commands:
                out.append(self._one(cmd))
        return out

    def _put(self, k: str, ent: list) -> None:
        self._d[k] = ent
        self._d.move_to_end(k)
        while len(self._d) > self._max:
            self._d.popitem(last=False)

    def _one(self, cmd: list):
        op = str(cmd[0]).upper()
        if op == "GET":
            ent = self._live(cmd[1])
            return None if ent is None else ent[1]
        if op == "SET":
            ttl = None
            if len(cmd) >= 5 and str(cmd[3]).upper() == "EX":
                ttl = float(cmd[4])
            self._put(cmd[1], [None if ttl is None else time.monotonic() + ttl, str(cmd[2])])
            return "OK"
        if op == "DEL":
            return sum(1 for k in cmd[1:] if self._d.pop(k, None) is not None)
        if op == "INCR":
            ent = self._live(cmd[1])
            n = int(ent[1]) + 1 if ent else 1
            self._put(cmd[1], [ent[0] if ent else None, str(n)])
            return n
        if op == "INCRBY":
            ent = self._live(cmd[1])
            n = (int(ent[1]) if ent else 0) + int(cmd[2])
            self._put(cmd[1], [ent[0] if ent else None, str(n)])
            return n
        if op == "MGET":
            return [(lambda e: None if e is None else e[1])(self._live(k)) for k in cmd[1:]]
        if op == "EXPIRE":
            ent = self._live(cmd[1])
            if ent is None:
                return 0
            ent[0] = time.monotonic() + float(cmd[2])
            return 1
        if op == "HSET":
            ent = self._live(cmd[1])
            if ent is None:
                ent = [None, {}]
                self._put(cmd[1], ent)
            ent[1][str(cmd[2])] = str(cmd[3])
            return 1
        if op == "HGETALL":
            ent = self._live(cmd[1])
            flat: list = []
            for f, v in (ent[1].items() if ent else []):
                flat += [f, v]
            return flat
        if op == "EVAL":
            return self._eval_bucket(cmd)
        raise BackendError(f"unsupported command {op}")

    def _eval_bucket(self, cmd: list):
        # Mirrors _BUCKET_LUA. ARGV: now_ms, per_min, per_hour
        k = cmd[3]
        now_ms, per_min, per_hour = float(cmd[4]), float(cmd[5]), float(cmd[6])
        ent = self._live(k)
        st = ent[1] if ent else {"m": per_min, "h": per_hour, "t": now_ms}
        dt = max(0.0, now_ms - float(st["t"])) / 1000.0
        m, h = float(st["m"]), float(st["h"])
        if per_min > 0:
            m = min(per_min, m + dt * per_min / 60.0)
        if per_hour > 0:
            h = min(per_hour, h + dt * per_hour / 3600.0)
        wait = 0.0
        if per_min > 0 and m < 1.0:
            wait = max(wait, (1.0 - m) * 60.0 / per_min)
        if per_hour > 0 and h < 1.0:
            wait = max(wait, (1.0 - h) * 3600.0 / per_hour)
        if wait <= 0:
            if per_min > 0:
                m -= 1.0
            if per_hour > 0:
                h -= 1.0
        self._put(k, [time.monotonic() + 7200, {"m": m, "h": h, "t": now_ms}])
        return int(wait * 1000 + 0.999) if wait > 0 else 0


class UpstashBackend:
    """Redis over the Upstash REST API (``POST /pipeline``), plain httpx, hard per-call timeout."""

    name = "upstash"

    def __init__(self, url: str, token: str, timeout_s: float) -> None:
        import httpx

        self._url = url.rstrip("/")
        self._client = httpx.Client(
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            timeout=httpx.Timeout(timeout_s, connect=timeout_s),
            limits=httpx.Limits(max_connections=16, max_keepalive_connections=8, keepalive_expiry=30),
        )

    def execute(self, commands: list[list]) -> list:
        try:
            if len(commands) == 1:
                resp = self._client.post(self._url, content=json.dumps(commands[0]))
                resp.raise_for_status()
                body = resp.json()
                if "error" in body:
                    raise BackendError(str(body["error"])[:120])
                return [body.get("result")]
            resp = self._client.post(self._url + "/pipeline", content=json.dumps(commands))
            resp.raise_for_status()
            rows = resp.json()
            out = []
            for row in rows:
                if isinstance(row, dict) and "error" in row:
                    raise BackendError(str(row["error"])[:120])
                out.append(row.get("result") if isinstance(row, dict) else None)
            return out
        except BackendError:
            raise
        except Exception as exc:  # noqa: BLE001 - timeouts, DNS, TLS, 4xx/5xx, bad JSON
            raise BackendError(type(exc).__name__) from exc


# ------------------------------------------------------------------------------------------------
# Selection + circuit breaker
# ------------------------------------------------------------------------------------------------
_lock = threading.Lock()
_backend: Any = None
_shared = False                  # True when _backend is genuinely shared between instances
_fails = 0
_open_until = 0.0
_resolved = False
stats = {"ok": 0, "fail": 0, "skipped_open": 0, "opened": 0}


def _timeout_s() -> float:
    try:
        ms = float(os.environ.get("CACHE_TIMEOUT_MS", "120"))
    except ValueError:
        ms = 120.0
    return max(20.0, min(ms, 150.0)) / 1000.0


def _resolve() -> None:
    global _backend, _shared, _resolved
    with _lock:
        if _resolved:
            return
        _resolved = True
        mode = (os.environ.get("CACHE_BACKEND") or "auto").strip().lower()
        url = os.environ.get("UPSTASH_REDIS_REST_URL") or os.environ.get("KV_REST_API_URL") or ""
        token = os.environ.get("UPSTASH_REDIS_REST_TOKEN") or os.environ.get("KV_REST_API_TOKEN") or ""
        if mode in ("upstash", "auto") and url.startswith("https://") and token:
            try:
                _backend = UpstashBackend(url, token, _timeout_s())
                _shared = True
                log.info("Shared cache: upstash REST backend enabled (timeout %.0f ms).", _timeout_s() * 1000)
                return
            except Exception as exc:  # noqa: BLE001
                log.warning("Shared cache backend could not start (%s); using in-process only.", exc)
        _backend, _shared = MemoryBackend(), False


def configure(backend: Any, shared: bool) -> None:
    """Install a backend explicitly (tests). ``shared=True`` makes every helper active."""
    global _backend, _shared, _resolved, _fails, _open_until
    with _lock:
        _backend, _shared, _resolved = backend, shared, True
        _fails, _open_until = 0, 0.0
    for k in stats:
        stats[k] = 0
    _maps.clear()


def reset() -> None:
    """Forget the resolved backend (tests); the next call re-reads the environment."""
    global _backend, _shared, _resolved, _fails, _open_until
    with _lock:
        _backend, _shared, _resolved, _fails, _open_until = None, False, False, 0, 0.0
    _maps.clear()


def enabled() -> bool:
    """True when a genuinely shared backend is configured and its breaker is not open."""
    if not _resolved:
        _resolve()
    return _shared and time.monotonic() >= _open_until


def backend_name() -> str:
    if not _resolved:
        _resolve()
    return getattr(_backend, "name", "none")


def status() -> dict:
    if not _resolved:
        _resolve()
    return {"backend": backend_name(), "shared": _shared, "breaker_open": time.monotonic() < _open_until, **stats}


def _run(commands: list[list]) -> list | None:
    """Execute on the shared backend; None on any failure / open breaker (caller falls back)."""
    global _fails, _open_until
    if not enabled():
        if _shared:
            stats["skipped_open"] += 1
        return None
    try:
        res = _backend.execute(commands)
    except Exception as exc:  # noqa: BLE001
        stats["fail"] += 1
        with _lock:
            _fails += 1
            if _fails >= _FAIL_THRESHOLD:
                if time.monotonic() >= _open_until:
                    stats["opened"] += 1
                    log.warning("Shared cache breaker OPEN for %.0fs (%s).", _OPEN_S, exc)
                _open_until = time.monotonic() + _OPEN_S
                _fails = _FAIL_THRESHOLD - 1          # one failed probe re-opens immediately
        return None
    stats["ok"] += 1
    with _lock:
        _fails = 0
    return res


# ------------------------------------------------------------------------------------------------
# Primitive helpers (all fail soft)
# ------------------------------------------------------------------------------------------------
def get(k: str) -> str | None:
    res = _run([["GET", k]])
    return res[0] if res else None


def set(k: str, value: str, ttl_s: float) -> bool:  # noqa: A001 - mirrors Redis
    if len(value) > MAX_VALUE_BYTES:
        return False
    return _run([["SET", k, value, "EX", max(1, int(ttl_s))]]) is not None


def delete(*keys: str) -> bool:
    return bool(keys) and _run([["DEL", *keys]]) is not None


def get_json(k: str) -> Any | None:
    raw = get(k)
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return None


def set_json(k: str, value: Any, ttl_s: float) -> bool:
    try:
        return set(k, json.dumps(value, separators=(",", ":"), default=str), ttl_s)
    except (TypeError, ValueError):
        return False


# Fire-and-forget writes (best effort, bounded queue, 2 daemon workers) for state whose loss only costs
# an extra recompute: they never delay the response.
_bg: "queue.Queue[Callable[[], None]]" = queue.Queue(maxsize=256)
_bg_started = False


def _bg_worker() -> None:
    while True:
        fn = _bg.get()
        try:
            fn()
        except Exception:  # noqa: BLE001
            pass


def background(fn: Callable[[], None]) -> None:
    global _bg_started
    if not enabled():
        return
    if not _bg_started:
        with _lock:
            if not _bg_started:
                for i in range(2):
                    threading.Thread(target=_bg_worker, daemon=True, name=f"cache-bg-{i}").start()
                _bg_started = True
    try:
        _bg.put_nowait(fn)
    except queue.Full:
        pass


def set_json_async(k: str, value: Any, ttl_s: float) -> None:
    if enabled():
        background(lambda: set_json(k, value, ttl_s))


# ------------------------------------------------------------------------------------------------
# Epoch-validated per-owner read cache (shared L2 behind each process's own L1)
# ------------------------------------------------------------------------------------------------
_EPOCH_TTL_S = 86_400


def _ekey(owner: str) -> str:
    return key("ep", owner)


def bump_epoch(owner: str) -> None:
    """Record that ``owner`` has new data: every instance's L2 entry for them becomes stale."""
    if owner and enabled():
        _run([["INCR", _ekey(owner)], ["EXPIRE", _ekey(owner), _EPOCH_TTL_S]])


def l2_get(ns: str, owner: str, sub: str = "", max_stale_s: float = 0.0) -> tuple[Any | None, int | None]:
    """``(value_or_None, epoch_now)`` in ONE round trip. ``epoch_now`` is None when the cache is
    unavailable (the caller then skips ``l2_put``). A value is returned only if it was computed at
    the owner's current epoch (or, with ``max_stale_s``, less than that many seconds ago)."""
    if not enabled():
        return None, None
    res = _run([["GET", _ekey(owner)], ["GET", key(ns, owner, sub)]])
    if res is None:
        return None, None
    try:
        ep = int(res[0] or 0)
    except (TypeError, ValueError):
        ep = 0
    if not res[1]:
        return None, ep
    try:
        doc = json.loads(res[1])
        if doc.get("e") == ep or (max_stale_s > 0 and time.time() - float(doc.get("t") or 0) < max_stale_s):
            return doc.get("v"), ep
    except (TypeError, ValueError, AttributeError):
        pass
    return None, ep


def l2_put(ns: str, owner: str, sub: str, value: Any, ep: int | None, ttl_s: float) -> None:
    """Store ``value`` computed while the owner's epoch was ``ep`` (read BEFORE computing)."""
    if ep is None or not enabled():
        return
    set_json(key(ns, owner, sub), {"e": ep, "t": round(time.time(), 1), "v": value}, ttl_s)


# ------------------------------------------------------------------------------------------------
# Single flight: concurrent identical misses in one process compute once
# ------------------------------------------------------------------------------------------------
_flights: dict[Any, list] = {}
_flights_guard = threading.Lock()


class single_flight:
    """``with single_flight(k): <re-check cache; compute; store>`` - the first caller computes while the
    others wait on the same lock and then find the value in the cache. Waiters give up (and compute
    themselves) after ``timeout`` so a stuck leader can never stall a request."""

    def __init__(self, k: Any, timeout: float = 15.0) -> None:
        self.k, self.timeout, self.held = k, timeout, False

    def __enter__(self):
        with _flights_guard:
            ent = _flights.setdefault(self.k, [threading.Lock(), 0])
            ent[1] += 1
            self.lock = ent[0]
        self.held = self.lock.acquire(timeout=self.timeout)
        return self

    def __exit__(self, *exc):
        if self.held:
            self.lock.release()
        with _flights_guard:
            ent = _flights.get(self.k)
            if ent is not None:
                ent[1] -= 1
                if ent[1] <= 0:
                    _flights.pop(self.k, None)
        return False


# ------------------------------------------------------------------------------------------------
# Shared small maps with a non-blocking local snapshot (LLM chain quota / dead-key memory)
# ------------------------------------------------------------------------------------------------
class SharedMap:
    """A Redis hash mirrored into a local dict. Reads NEVER touch the network (they read the snapshot, which a
    background thread refreshes at most every ``refresh_s``); writes update the snapshot at once and the hash
    in the background. Values are ``"<wall-clock expiry>[|extra]"`` strings; expired entries are ignored."""

    def __init__(self, name: str, refresh_s: float = 5.0) -> None:
        self.hkey = key("map", None, name)
        self.refresh_s = refresh_s
        self._snap: dict[str, str] = {}
        self._at = 0.0
        self._busy = False
        self._glock = threading.Lock()

    def _refresh(self) -> None:
        res = _run([["HGETALL", self.hkey]])
        with self._glock:
            self._busy = False
            self._at = time.monotonic()
            if res is not None:
                flat = res[0] or []
                remote = {flat[i]: flat[i + 1] for i in range(0, len(flat) - 1, 2)}
                # keep local entries newer than what Redis returned (a write still in flight)
                for f, v in self._snap.items():
                    if f not in remote:
                        remote[f] = v
                self._snap = remote

    def _maybe_refresh(self) -> None:
        if not enabled():
            return
        now = time.monotonic()
        with self._glock:
            if self._busy or now - self._at < self.refresh_s:
                return
            self._busy = True
            first = self._at == 0.0
        if first:
            self._refresh()           # first use on this instance: one bounded (<=150 ms) lookup
        else:
            background(self._refresh)

    def get(self, field: str) -> str | None:
        """The stored value if its wall-clock expiry is in the future, else None."""
        self._maybe_refresh()
        raw = self._snap.get(field)
        if raw is None:
            return None
        head, _, rest = raw.partition("|")
        try:
            if float(head) <= time.time():
                return None
        except ValueError:
            return None
        return rest

    def put(self, field: str, ttl_s: float, extra: str = "") -> None:
        if not enabled():
            return
        val = f"{time.time() + ttl_s:.3f}|{extra}"
        with self._glock:
            self._snap[field] = val
        k = self.hkey
        background(lambda: _run([["HSET", k, field, val], ["EXPIRE", k, 86_400]]))

    def clear_local(self) -> None:
        with self._glock:
            self._snap.clear()
            self._at = 0.0


_maps: dict[str, SharedMap] = {}


def shared_map(name: str, refresh_s: float = 5.0) -> SharedMap:
    m = _maps.get(name)
    if m is None:
        m = _maps.setdefault(name, SharedMap(name, refresh_s))
    return m


# ------------------------------------------------------------------------------------------------
# Shared token bucket (rate limiter). One round trip, atomic.
# ------------------------------------------------------------------------------------------------
_BUCKET_LUA = """
local t = redis.call('HMGET', KEYS[1], 'm', 'h', 't')
local now = tonumber(ARGV[1]); local pm = tonumber(ARGV[2]); local ph = tonumber(ARGV[3])
local m = tonumber(t[1]) or pm; local h = tonumber(t[2]) or ph; local last = tonumber(t[3]) or now
local dt = math.max(0, now - last) / 1000
if pm > 0 then m = math.min(pm, m + dt * pm / 60) end
if ph > 0 then h = math.min(ph, h + dt * ph / 3600) end
local wait = 0
if pm > 0 and m < 1 then wait = math.max(wait, (1 - m) * 60 / pm) end
if ph > 0 and h < 1 then wait = math.max(wait, (1 - h) * 3600 / ph) end
if wait <= 0 then
  if pm > 0 then m = m - 1 end
  if ph > 0 then h = h - 1 end
end
redis.call('HSET', KEYS[1], 'm', m, 'h', h, 't', now)
redis.call('EXPIRE', KEYS[1], 7200)
return math.ceil(wait * 1000)
"""


def bucket_take(owner: str, per_min: int, per_hour: int) -> float | None:
    """Consume one request from the owner's SHARED bucket. Returns seconds to wait (0.0 = allowed) or
    ``None`` when the shared backend is unavailable (caller keeps its in-process decision)."""
    if not enabled():
        return None
    now_ms = int(time.time() * 1000)
    if isinstance(_backend, MemoryBackend):
        cmd = ["EVAL", _BUCKET_LUA, 1, key("rl", owner), now_ms, per_min, per_hour]
    else:
        cmd = ["EVAL", _BUCKET_LUA, 1, key("rl", owner), str(now_ms), str(per_min), str(per_hour)]
    res = _run([cmd])
    if res is None:
        return None
    try:
        return max(0.0, float(res[0]) / 1000.0)
    except (TypeError, ValueError):
        return None
