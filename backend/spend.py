"""Hard, cross-instance SPEND GUARDS for the paid Anthropic step of the provider chain.

Why: per-instance request-rate limits (``ratelimit.py``) do not bound *money*. A runaway user or a
public abuser could send huge prompts to Sonnet all day. This module keeps a token LEDGER for the
paid provider and lets ``llm_chain`` skip step 1 (falling through to the free Groq/Gemini steps)
once a budget is used up, so the app never stops working - only the premium model is rationed.

Budgets (env, see ``backend/config.py`` / ``.env.example``; tokens = prompt + completion; 0 = unlimited)::

    ANTHROPIC_DAILY_TOKEN_BUDGET       global per UTC day            default 0 (unlimited)
    ANTHROPIC_OWNER_DAILY_TOKENS       per signed-in owner per day   default 0 (unlimited)
    ANTHROPIC_GUEST_DAILY_TOKENS       per guest per day             default 0 (unlimited)
    ANTHROPIC_MAX_REQUEST_INPUT_TOKENS per request (estimated)       default 0 (unlimited)

How it stays cheap and shared WITHOUT Redis:

* every instance counts tokens in memory (``_local``, always available -> the fail-open floor);
* usage is flushed to Mongo collection ``usage_daily`` (one doc per ``(UTC day, scope)``; scope is
  ``global`` or ``owner:<sha256(owner)[:24]>``; TTL 3 days) with ONE ``bulk_write`` of ``$inc`` upserts
  every ~20 s, or earlier once 5% of a budget is pending, or at shutdown (best effort);
* the shared totals are read with ONE projected ``find`` per owner per ~30 s (cached per instance);
  the decision uses ``shared_total_at_read + tokens_added_here_since``;
* when a shared Redis backend is enabled (``backend/cache.py``) the same counters live there
  (``INCRBY`` + day-keyed ``EXPIRE`` in one pipeline, ``MGET`` to read) and Mongo is not touched;
* ANY store failure fails OPEN: the app never blocks, but the per-instance counters still enforce the
  limits on that instance.

Owner attribution: ``begin_request()`` (middleware) installs a mutable holder in a ContextVar and
``auth.current_owner`` fills it; ``llm_chain`` reads :func:`current_owner` (or takes an explicit
``owner=``). Unattributed calls only count against the global budget. No owner id is ever logged,
stored in clear text or exposed (only sha256 prefixes in the store, counts in the health endpoint).
"""
from __future__ import annotations

import contextlib
import contextvars
import logging
import threading
import time
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from backend import cache, config

log = logging.getLogger("backend.spend")

COLLECTION = "usage_daily"
GUEST_PREFIX = "guest_"
FLUSH_INTERVAL_S = 20.0
READ_TTL_S = 30.0
FLUSH_FRACTION = 0.05            # flush early once this share of a budget is pending
WARN_PCT = 80.0
TTL_DAYS = 3
MAX_OWNERS = 20_000
CHARS_PER_TOKEN = 3.6
IMAGE_TOKENS = 1600              # rough cost of one attached image for the pre-call estimate

# ---------------------------------------------------------------------------
# Owner attribution (request-scoped, shared into FastAPI's threadpool by reference)
# ---------------------------------------------------------------------------
_holder: contextvars.ContextVar["dict | None"] = contextvars.ContextVar("spend_owner", default=None)


def begin_request() -> None:
    """Called once per API request by the middleware: a fresh, mutable owner holder."""
    _holder.set({})


def set_owner(owner: str | None) -> None:
    """Attribute the rest of this request to ``owner``. A no-op outside a request."""
    h = _holder.get()
    if h is not None and owner:
        h["owner"] = owner


def current_owner() -> str | None:
    h = _holder.get()
    return (h or {}).get("owner")


@contextlib.contextmanager
def owner_scope(owner: str | None):
    """Attribute LLM calls made inside the block to ``owner`` (background paths, scripts, tests)."""
    token = _holder.set({"owner": owner} if owner else {})
    try:
        yield
    finally:
        _holder.reset(token)


def is_guest(owner: str | None) -> bool:
    return bool(owner) and str(owner).startswith(GUEST_PREFIX)


def estimate_tokens(chars: int, images: int = 0) -> int:
    return int(max(0, chars) / CHARS_PER_TOKEN) + 1 + IMAGE_TOKENS * max(0, images)


def utc_day(now: float | None = None) -> str:
    return datetime.fromtimestamp(time.time() if now is None else now, timezone.utc).strftime("%Y%m%d")


def _scope_for(owner: str) -> str:
    return "owner:" + cache.owner_tag(owner)


# ---------------------------------------------------------------------------
# Stores. Interface: incr(day, {scope: tokens}) and read(day, [scopes]) -> {scope: tokens}.
# Both raise on failure; ``Unavailable`` means "no store configured" (silent, not an error).
# ---------------------------------------------------------------------------
class Unavailable(Exception):
    pass


class MongoStore:
    name = "mongo"

    def _coll(self):
        from backend import db

        database = db._get_db()
        if database is None:
            raise Unavailable("persistence disabled")
        return database[COLLECTION]

    def incr(self, day: str, deltas: dict[str, int]) -> None:
        from pymongo import UpdateOne

        y, m, d = int(day[:4]), int(day[4:6]), int(day[6:])
        expires = datetime(y, m, d, tzinfo=timezone.utc) + timedelta(days=TTL_DAYS)
        ops = [UpdateOne({"_id": f"{day}:{scope}"},
                         {"$inc": {"tokens": int(n)},
                          "$setOnInsert": {"day": day, "scope": scope, "expires_at": expires}}, upsert=True)
               for scope, n in deltas.items() if n > 0]
        if ops:
            self._coll().bulk_write(ops, ordered=False)

    def read(self, day: str, scopes: list[str]) -> dict[str, int]:
        ids = [f"{day}:{s}" for s in scopes]
        rows = self._coll().find({"_id": {"$in": ids}}, {"tokens": 1})
        out = {s: 0 for s in scopes}
        for r in rows:
            sc = str(r.get("_id", ""))[len(day) + 1:]
            if sc in out:
                out[sc] = int(r.get("tokens") or 0)
        return out


class RedisStore:
    name = "redis"

    @staticmethod
    def _k(day: str, scope: str) -> str:
        return f"{cache.PREFIX}:spend:{day}:{scope}"

    def incr(self, day: str, deltas: dict[str, int]) -> None:
        cmds: list[list] = []
        for scope, n in deltas.items():
            if n > 0:
                cmds += [["INCRBY", self._k(day, scope), int(n)], ["EXPIRE", self._k(day, scope), TTL_DAYS * 86400]]
        if cmds and cache._run(cmds) is None:
            raise RuntimeError("shared cache unavailable")

    def read(self, day: str, scopes: list[str]) -> dict[str, int]:
        res = cache._run([["MGET", *[self._k(day, s) for s in scopes]]])
        if res is None:
            raise RuntimeError("shared cache unavailable")
        vals = res[0] or []
        return {s: int(vals[i] or 0) if i < len(vals) else 0 for i, s in enumerate(scopes)}


def default_store():
    """Redis when a shared backend is enabled, else Mongo (raises ``Unavailable`` if neither)."""
    return RedisStore() if cache.enabled() else MongoStore()


# ---------------------------------------------------------------------------
# The ledger
# ---------------------------------------------------------------------------
class Ledger:
    def __init__(self, store: Any = None, clock: Callable[[], float] = time.time,
                 background: bool = True) -> None:
        self._store_override = store
        self._clock = clock
        self._background = background
        self._lock = threading.Lock()
        self._flush_lock = threading.Lock()
        self._reset_day(utc_day(clock()))
        self._pending: dict[tuple[str, str], int] = {}      # (day, scope) -> tokens not yet in the store
        self._last_flush = clock()
        self._read_fail_until = 0.0
        self._flushing = False
        self._retry_after = 0.0
        self.stats = {"flush_ok": 0, "flush_fail": 0, "read_ok": 0, "read_fail": 0}

    # -- day handling
    def _reset_day(self, day: str) -> None:
        self._day = day
        self._local: "OrderedDict[str, int]" = OrderedDict()
        # key ("" = unattributed, else owner scope) -> (read_ts, shared_owner_total, owner_base)
        self._reads: dict[str, tuple[float, int, int]] = {}
        # latest shared GLOBAL total seen: (shared_total, local_global_already_in_store_at_read_time)
        self._gread: tuple[int, int] | None = None
        self._skipped = 0
        self._tripped: set[str] = set()
        self._warned = False

    def _roll(self) -> None:
        day = utc_day(self._clock())
        if day != self._day:
            self._reset_day(day)

    def _store(self):
        return self._store_override if self._store_override is not None else default_store()

    # -- budgets (live env)
    @staticmethod
    def global_limit() -> int:
        return config.anthropic_daily_token_budget()

    @staticmethod
    def owner_limit(owner: str | None) -> int:
        return config.anthropic_owner_daily_tokens(is_guest(owner)) if owner else 0

    # -- recording
    def add(self, owner: str | None, tokens: int) -> None:
        """Count ACTUAL tokens spent on the paid provider (never raises)."""
        try:
            tokens = int(tokens)
            if tokens <= 0:
                return
            gl, ol = self.global_limit(), self.owner_limit(owner)
            with self._lock:
                self._roll()
                day = self._day
                self._local["global"] = self._local.get("global", 0) + tokens
                due = False
                self._pending[(day, "global")] = self._pending.get((day, "global"), 0) + tokens
                if gl and self._pending[(day, "global")] >= gl * FLUSH_FRACTION:
                    due = True
                if owner:
                    sc = _scope_for(owner)
                    self._local[sc] = self._local.get(sc, 0) + tokens
                    self._local.move_to_end(sc)
                    while len(self._local) > MAX_OWNERS + 1:
                        for k in self._local:
                            if k != "global":
                                del self._local[k]
                                break
                    if ol:
                        self._pending[(day, sc)] = self._pending.get((day, sc), 0) + tokens
                        if self._pending[(day, sc)] >= ol * FLUSH_FRACTION:
                            due = True
                if not gl and not ol:                      # nothing is rationed: keep counters local only
                    self._pending.pop((day, "global"), None)
                if self._clock() - self._last_flush >= FLUSH_INTERVAL_S:
                    due = True
                due = due and bool(self._pending) and not self._flushing and self._clock() >= self._retry_after
                if due:
                    self._flushing = True
            self._maybe_warn()
            if due:
                self._dispatch_flush()
        except Exception:  # noqa: BLE001 - accounting must never break an answered request
            log.debug("spend.add failed", exc_info=True)

    def _dispatch_flush(self) -> None:
        if not self._background:
            self.flush()
            return
        try:
            threading.Thread(target=self.flush, daemon=True, name="spend-flush").start()
        except Exception:  # noqa: BLE001
            self.flush()

    def flush(self) -> bool:
        """Write pending usage to the shared store (one round trip). Safe to call any time; returns
        True when nothing is left pending. Failures keep the tokens pending for the next attempt."""
        with self._flush_lock:
            with self._lock:
                self._flushing = True
                batch, self._pending = self._pending, {}
                self._last_flush = self._clock()
            ok = True
            if batch:
                by_day: dict[str, dict[str, int]] = {}
                for (day, scope), n in batch.items():
                    by_day.setdefault(day, {})[scope] = n
                try:
                    store = self._store()
                    for day, deltas in by_day.items():
                        store.incr(day, deltas)
                    self.stats["flush_ok"] += 1
                except Unavailable:
                    pass                                    # no store configured: local counters only
                except Exception as exc:  # noqa: BLE001
                    ok = False
                    self.stats["flush_fail"] += 1
                    self._retry_after = self._clock() + FLUSH_INTERVAL_S      # no retry storm on a sick store
                    log.warning("spend ledger flush failed (%s); will retry", type(exc).__name__)
                    with self._lock:
                        for k, n in batch.items():           # keep it for the next attempt (bounded)
                            if len(self._pending) < MAX_OWNERS * 2:
                                self._pending[k] = self._pending.get(k, 0) + n
            with self._lock:
                self._flushing = False
            return ok

    # -- reading
    def needs_read(self, owner: str | None) -> bool:
        """True when ``check`` would have to hit the store (the async caller then hops to a thread)."""
        if not (self.global_limit() or self.owner_limit(owner)):
            return False
        with self._lock:
            self._roll()
            ent = self._reads.get(owner and _scope_for(owner) or "")
            fresh = ent is not None and self._clock() - ent[0] < READ_TTL_S
        return not fresh and self._clock() >= self._read_fail_until

    def _refresh(self, owner: str | None) -> None:
        key = owner and _scope_for(owner) or ""
        scopes = ["global"] + ([key] if key else [])
        with self._lock:
            day = self._day
        try:
            vals = self._store().read(day, scopes)
        except Unavailable:
            with self._lock:
                self._read_fail_until = self._clock() + READ_TTL_S
            return
        except Exception as exc:  # noqa: BLE001
            self.stats["read_fail"] += 1
            log.warning("spend ledger read failed (%s); using per-instance counters", type(exc).__name__)
            with self._lock:
                self._read_fail_until = self._clock() + READ_TTL_S
            return
        self.stats["read_ok"] += 1
        with self._lock:
            if self._day != day:
                return
            # Local tokens still PENDING are not in the store yet; everything else already is, so
            # shared + (local_now - base) = what the fleet has spent as far as we can tell.
            base_g = self._local.get("global", 0) - self._pending.get((day, "global"), 0)
            self._gread = (int(vals.get("global", 0)), base_g)
            base_o = (self._local.get(key, 0) - self._pending.get((day, key), 0)) if key else 0
            self._reads[key] = (self._clock(), int(vals.get(key, 0)) if key else 0, base_o)

    def _used(self, owner: str | None) -> tuple[int, int]:
        """(global_used, owner_used) as best this instance knows: the shared total at the last read
        plus what was added here since, or the per-instance counters when no read has succeeded."""
        key = owner and _scope_for(owner) or ""
        with self._lock:
            self._roll()
            lg, lo = self._local.get("global", 0), (self._local.get(key, 0) if key else 0)
            g = lg if self._gread is None else max(lg, self._gread[0] + lg - self._gread[1])
            ent = self._reads.get(key) if key else None
            o = lo if ent is None else max(lo, ent[1] + lo - ent[2])
            return g, o

    # -- the decision
    def check(self, owner: str | None, est_prompt: int = 0) -> str | None:
        """None = the paid step may run; else a short reason (``global`` | ``owner``). Fails OPEN."""
        try:
            gl, ol = self.global_limit(), self.owner_limit(owner)
            if not gl and not ol:
                return None
            if self.needs_read(owner):
                self._refresh(owner)
            g, o = self._used(owner)
            reason = None
            if gl and g + est_prompt > gl:
                reason = "global"
            elif ol and o + est_prompt > ol:
                reason = "owner"
            if reason:
                self._note_skip(reason)
            return reason
        except Exception:  # noqa: BLE001 - never block the app on an accounting bug
            log.debug("spend.check failed (fail open)", exc_info=True)
            return None

    def _note_skip(self, reason: str) -> None:
        with self._lock:
            self._skipped += 1
            first = reason not in self._tripped
            self._tripped.add(reason)
        if first:
            log.info("anthropic %s budget tripped for %s: premium step skipped, free providers take over",
                     reason, self._day)

    def _maybe_warn(self) -> None:
        gl = self.global_limit()
        if not gl:
            return
        g, _ = self._used(None)
        if g * 100.0 / gl >= WARN_PCT:
            with self._lock:
                first, self._warned = not self._warned, True
            if first:
                log.warning("anthropic daily token budget at %.0f%% (%d/%d)", g * 100.0 / gl, g, gl)

    # -- observability (no owner ids; counts only)
    def status(self, refresh: bool = False) -> dict:
        if refresh and self.global_limit() and self.needs_read(None):
            self._refresh(None)
        gl = self.global_limit()
        g, _ = self._used(None)
        pct = round(g * 100.0 / gl, 1) if gl else 0.0
        with self._lock:
            owners = max(0, len(self._local) - (1 if "global" in self._local else 0))
            skipped = self._skipped
            tripped = sorted(self._tripped)
        try:
            store = self._store().name
        except Exception:  # noqa: BLE001
            store = "none"
        return {"day": self._day, "global_used": g, "global_limit": gl, "pct": pct,
                "warn": bool(gl) and pct >= WARN_PCT, "exhausted": bool(gl) and g >= gl,
                "owners_tracked": owners, "anthropic_skipped_today": skipped, "tripped": tripped,
                "store": store, "owner_limit": config.anthropic_owner_daily_tokens(False),
                "guest_limit": config.anthropic_owner_daily_tokens(True),
                "max_request_input_tokens": config.anthropic_max_request_input_tokens(),
                **{f"io_{k}": v for k, v in self.stats.items()}}


_ledger = Ledger()


def ledger() -> Ledger:
    return _ledger


def reset_ledger(new: Ledger | None = None) -> Ledger:
    """Replace the process-wide ledger (tests)."""
    global _ledger
    _ledger = new or Ledger()
    return _ledger


def add(owner: str | None, tokens: int) -> None:
    _ledger.add(owner, tokens)


def check(owner: str | None, est_prompt: int = 0) -> str | None:
    return _ledger.check(owner, est_prompt)


def status(refresh: bool = False) -> dict:
    return _ledger.status(refresh)


def flush() -> bool:
    return _ledger.flush()


import atexit  # noqa: E402

atexit.register(lambda: _ledger.flush())     # shutdown-ish best effort
