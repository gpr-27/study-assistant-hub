"""Gemini embeddings over plain REST — serverless-friendly, key-rotating, bounded.

Verified against the live API (see the report): ``gemini-embedding-2`` works with
``GEMINI_API_KEY`` and ``GEMINI_API_KEY_2`` (restricted "auth keys", header
``x-goog-api-key``), returns 3072-dim unit vectors and supports Matryoshka
truncation through ``outputDimensionality`` (768 / 1536 come back unit-norm).

Two things the previous client got wrong and this one fixes:

* ``gemini-embedding-2`` IGNORES the ``taskType`` field (a doc vector and a query
  vector for the same text are bit-identical). Retrieval quality instead needs the
  task spelled out IN the text: ``"title: none | text: ..."`` for documents and
  ``"task: search result | query: ..."`` for queries. ``gemini-embedding-001`` does
  honour ``taskType``. :func:`format_for` hides that difference.
* The free tier allows ~100 embedded *contents* per minute per key per model (every
  item of a batch counts). The old code waited out 429s for 60 s+ inside chat
  requests. Here every call has a hard deadline, keys rotate on 429/401/5xx, a
  per-key sliding window avoids 429s proactively when bulk-indexing, and the
  interactive query path never blocks longer than a few seconds.

No heavy imports: ``requests`` and ``numpy`` are imported on first use.
"""
from __future__ import annotations

import logging
import os
import re
import threading
import time
from collections import OrderedDict, deque

log = logging.getLogger(__name__)

_API_ROOT = "https://generativelanguage.googleapis.com/v1beta"


def _api_root() -> str:
    """TEST-ONLY redirect (``LLM_CHAIN_TEST_BASE_URL``, see backend/llm_chain.py); never on Vercel."""
    base = "" if os.environ.get("VERCEL") else os.environ.get("LLM_CHAIN_TEST_BASE_URL", "").strip().rstrip("/")
    return f"{base}/v1beta" if base else _API_ROOT

DEFAULT_MODEL = "gemini-embedding-2"
# Free tier: 100 embed_content requests/min per key per model; stay a bit under.
RPM_PER_KEY = int(os.environ.get("EMBED_RPM_PER_KEY", "90") or 90)
BATCH_SIZE = 50                      # contents per batchEmbedContents call
QUERY_DEADLINE_S = 3.5               # hard cap for an interactive query embedding
_QUERY_CACHE_MAX = 512
_DEAD_KEY_TTL_S = 600.0              # a 401/403 key is skipped this long
_DAILY_COOLDOWN_S = 3600.0
_MAX_COOLDOWN_S = 65.0
_SHORT_RETRY_S = 1.5                 # honour a 429 retryDelay inline only below this


class GeminiEmbeddingError(RuntimeError):
    """Raised when the Gemini embedding API cannot be reached or returns an error."""


class GeminiQuotaError(GeminiEmbeddingError):
    """Raised when every key's embedding quota is exhausted (or cooling down)."""


# ── configuration ────────────────────────────────────────────────────────────
def load_keys() -> list[str]:
    """Return the configured Gemini keys in priority order (never logged)."""
    names = ("GEMINI_API_KEY", "GEMINI_API_KEY_2", "GEMINI_API_KEY_FALLBACK")
    keys: list[str] = []
    for name in names:
        val = (os.environ.get(name) or "").strip()
        if val and val not in keys:
            keys.append(val)
    if not keys:
        try:                                   # backend.config loads .env on import
            from backend import config

            keys = list(config.gemini_api_keys())
        except Exception:  # noqa: BLE001
            pass
    return keys


def have_keys() -> bool:
    return bool(load_keys())


def format_for(model: str, text: str, kind: str, title: str | None = None) -> tuple[str, str | None]:
    """Return ``(text_to_send, taskType_or_None)`` for ``kind`` in {"query","document"}.

    ``gemini-embedding-2`` ignores taskType, so the task is encoded as a text prefix
    (Google's documented format). Older models take the ``taskType`` field.
    """
    m = model.replace("models/", "")
    if m.startswith("gemini-embedding-2"):
        if kind == "query":
            return f"task: search result | query: {text}", None
        return f"title: {title or 'none'} | text: {text}", None
    return text, ("RETRIEVAL_QUERY" if kind == "query" else "RETRIEVAL_DOCUMENT")


# ── per-key runtime state (process-wide, bounded: ≤ 3 keys × few models) ─────
class _KeyState:
    __slots__ = ("cooldown_until", "dead_until", "sent")

    def __init__(self) -> None:
        self.cooldown_until = 0.0
        self.dead_until = 0.0
        self.sent: deque[tuple[float, int]] = deque()   # (monotonic ts, contents)


_state_lock = threading.RLock()
_states: dict[tuple[str, str], _KeyState] = {}


def _state(key: str, model: str) -> _KeyState:
    # Key the dict by a short digest so key material never sits in a dict key we might log.
    import hashlib

    ident = hashlib.sha1(key.encode()).hexdigest()[:10]
    k = (ident, model)
    with _state_lock:                      # two threads must never create two windows for one key
        st = _states.get(k)
        if st is None:
            st = _states[k] = _KeyState()
        return st


def reset_state() -> None:
    """Forget cooldowns/windows/cache (tests)."""
    with _state_lock:
        _states.clear()
    with _qcache_lock:
        _qcache.clear()
        _inflight.clear()
    _breaker["until"] = 0.0
    _stats.update(last_error=None, last_error_at=0.0, last_ok_at=0.0, queries=0, cache_hits=0,
                  embed_calls=0, errors=0)


def _window_wait(st: _KeyState, n: int, now: float) -> float:
    """Seconds until ``n`` more contents fit in this key's 60 s window (0 = now)."""
    while st.sent and now - st.sent[0][0] >= 60.0:
        st.sent.popleft()
    used = sum(c for _, c in st.sent)
    if used + n <= RPM_PER_KEY:
        return 0.0
    need = used + n - RPM_PER_KEY
    freed = 0
    for ts, c in st.sent:
        freed += c
        if freed >= need:
            return max(0.0, ts + 60.0 - now) + 0.05
    return 60.0


def _refund(st: "_KeyState", ticket: tuple[float, int]) -> None:
    try:
        st.sent.remove(ticket)
    except ValueError:
        pass


def _retry_delay(body) -> float | None:
    err = body.get("error", {}) if isinstance(body, dict) else {}
    for detail in err.get("details", []) or []:
        if str(detail.get("@type", "")).endswith("RetryInfo"):
            m = re.match(r"([0-9.]+)s", str(detail.get("retryDelay", "")))
            if m:
                return float(m.group(1))
    m = re.search(r"retry in ([0-9.]+)s", str(err.get("message", "")))
    return float(m.group(1)) if m else None


def _is_daily(body) -> bool:
    txt = str(body)[:2000].lower()
    return "perday" in txt or "per day" in txt or "daily" in txt


_session = None


def _http():
    global _session
    if _session is None:
        import requests
        from requests.adapters import HTTPAdapter

        sess = requests.Session()
        # default pool is 10 connections: under a burst of concurrent queries it would discard and
        # re-open TLS connections on every call
        adapter = HTTPAdapter(pool_connections=4, pool_maxsize=64)
        sess.mount("https://", adapter)
        sess.mount("http://", adapter)
        _session = sess
    return _session


def _redact(text: str, keys: list[str]) -> str:
    for k in keys:
        if k:
            text = text.replace(k, "<key>")
    return text[:200]


# ── the one HTTP primitive: rotate, bound, never hang ────────────────────────
def _post(model: str, endpoint: str, payload: dict, *, n_contents: int, deadline_s: float,
          patient: bool = False) -> dict:
    """POST ``models/<model>:<endpoint>`` rotating across keys until ``deadline_s``.

    ``patient`` (bulk indexing) sleeps until a key's window/cooldown frees up, within
    the deadline; the interactive path (``patient=False``) never sleeps more than a
    short, server-suggested delay and otherwise fails fast so callers can fall back to
    lexical retrieval.
    """
    keys = load_keys()
    if not keys:
        raise GeminiEmbeddingError("No Gemini API key configured (GEMINI_API_KEY / GEMINI_API_KEY_2).")
    model = model.replace("models/", "")
    url = f"{_api_root()}/models/{model}:{endpoint}"
    t_end = time.monotonic() + deadline_s
    last: str = ""
    saw_quota = False
    saw_auth = False
    server_errors = 0
    rr = int(time.monotonic() * 1000) % len(keys)    # spread load across keys

    while True:
        now = time.monotonic()
        remaining = t_end - now
        if remaining <= 0.05 or server_errors >= (2 if patient else 1) * len(keys):
            break
        # choose the usable key with the smallest wait
        best_i, best_wait = None, None
        with _state_lock:
            for off in range(len(keys)):
                i = (rr + off) % len(keys)
                st = _state(keys[i], model)
                if st.dead_until > now:
                    continue
                wait = max(st.cooldown_until - now, _window_wait(st, n_contents, now) if patient or n_contents > 1 else 0.0)
                if best_wait is None or wait < best_wait:
                    best_i, best_wait = i, wait
                    if wait <= 0:
                        break
            ticket = None
            if best_i is not None and not best_wait:
                # Reserve the window atomically with the choice, BEFORE sending: concurrent callers
                # (20 users uploading at once) must see each other's load or they all pass the check.
                ticket = (now, n_contents)
                _state(keys[best_i], model).sent.append(ticket)
        if best_i is None:
            break                                             # every key is dead (auth)
        if best_wait and best_wait > 0:
            if (not patient and best_wait > _SHORT_RETRY_S) or best_wait > remaining:
                saw_quota = True
                break
            time.sleep(best_wait)
            continue
        key = keys[best_i]
        st = _state(key, model)
        rr = best_i + 1
        timeout = (min(2.5, remaining), max(0.5, remaining) if patient else max(0.5, remaining))
        try:
            resp = _http().post(url, json=payload, headers={"x-goog-api-key": key}, timeout=timeout)
        except Exception as exc:  # noqa: BLE001 — requests.RequestException / timeouts
            last = f"network: {type(exc).__name__}"
            with _state_lock:
                _refund(st, ticket)
                st.cooldown_until = time.monotonic() + 1.0
            server_errors += 1
            if len(keys) == 1 and server_errors <= 2 and (t_end - time.monotonic()) > 0.6:
                time.sleep(0.3)
            continue
        code = resp.status_code
        if code == 200:
            try:
                return resp.json()
            except ValueError as exc:
                raise GeminiEmbeddingError("Gemini returned a non-JSON 200 response.") from exc
        try:
            body = resp.json()
        except ValueError:
            body = {}
        last = f"{code}: {_redact(resp.text, keys)}"
        with _state_lock:
            _refund(st, ticket)               # the request did not consume quota (429 re-fills below)
        if code == 429:
            saw_quota = True
            delay = _retry_delay(body)
            cool = _DAILY_COOLDOWN_S if _is_daily(body) else min((delay or 20.0) + 0.5, _MAX_COOLDOWN_S)
            with _state_lock:
                st.cooldown_until = time.monotonic() + cool
                st.sent.append((time.monotonic(), RPM_PER_KEY))   # window is full for this key
            log.info("Gemini embed 429 on a key (cooldown %.0fs).", cool)
            continue
        if code in (401, 403) or (code == 400 and "api key" in resp.text.lower()):
            saw_auth = True
            with _state_lock:
                st.dead_until = time.monotonic() + _DEAD_KEY_TTL_S
            log.warning("Gemini embed auth failure (%s) on a key; skipping it for %.0fs.", code, _DEAD_KEY_TTL_S)
            continue
        if code in (500, 502, 503, 504):
            server_errors += 1
            with _state_lock:
                st.cooldown_until = time.monotonic() + 1.5
            if len(keys) == 1 and (t_end - time.monotonic()) > 1.0:
                time.sleep(0.4)
            continue
        raise GeminiEmbeddingError(f"Gemini embeddings error {last}")

    if saw_quota:
        raise GeminiQuotaError("Gemini embedding quota exhausted on every configured key; try again shortly.")
    if saw_auth and not last.startswith("5"):
        raise GeminiEmbeddingError("Gemini embeddings auth failed on every key (check GEMINI_API_KEY / GEMINI_API_KEY_2).")
    raise GeminiEmbeddingError(f"Gemini embeddings unavailable ({last or 'timeout'}).")


# ── numpy helpers ────────────────────────────────────────────────────────────
def _as_matrix(vectors: list[list[float]], dim: int | None):
    import numpy as np

    mat = np.asarray(vectors, dtype=np.float32)
    if dim and mat.shape[1] != dim:
        if mat.shape[1] > dim:                      # defensive Matryoshka truncation
            mat = mat[:, :dim]
        else:
            raise GeminiEmbeddingError(f"Embedding dimension {mat.shape[1]} != expected {dim}.")
    if not np.isfinite(mat).all():
        raise GeminiEmbeddingError("Gemini returned non-finite embedding values.")
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (mat / norms).astype(np.float32, copy=False)


def _request(model: str, text: str, kind: str, dim: int | None, title: str | None = None) -> dict:
    body, task = format_for(model, text or " ", kind, title)
    req: dict = {"model": f"models/{model.replace('models/', '')}", "content": {"parts": [{"text": body}]}}
    if task:
        req["taskType"] = task
    if dim:
        req["outputDimensionality"] = dim
    return req


# ── public API ───────────────────────────────────────────────────────────────
_qcache: "OrderedDict[tuple, object]" = OrderedDict()
_qcache_lock = threading.Lock()
_inflight: "dict[tuple, list]" = {}  # single-flight: identical concurrent queries share one API call
_breaker = {"until": 0.0}          # after a total failure, skip embedding briefly
# Cheap observability for /api/health/deep and the doctor script (no secrets, no texts).
_stats: dict = {"last_error": None, "last_error_at": 0.0, "last_ok_at": 0.0,
                "queries": 0, "cache_hits": 0, "embed_calls": 0, "errors": 0}


def _note_error(exc: BaseException) -> None:
    _stats["errors"] += 1
    _stats["last_error_at"] = time.time()
    _stats["last_error"] = "quota_exhausted" if isinstance(exc, GeminiQuotaError) else type(exc).__name__


def status() -> dict:
    """Embedding-client health: booleans, counters and error CODES only (never keys or texts)."""
    now_m, now_w = time.monotonic(), time.time()
    brk = max(0.0, _breaker["until"] - now_m)
    q = _stats["queries"]
    return {
        "keys_configured": len(load_keys()),
        "breaker_open": brk > 0, "breaker_remaining_s": round(brk, 1),
        "last_error": _stats["last_error"],
        "last_error_age_s": round(now_w - _stats["last_error_at"]) if _stats["last_error_at"] else None,
        "last_ok_age_s": round(now_w - _stats["last_ok_at"]) if _stats["last_ok_at"] else None,
        "query_cache_size": len(_qcache),
        "query_cache_hit_rate": round(_stats["cache_hits"] / q, 3) if q else None,
        "embed_calls": _stats["embed_calls"], "errors": _stats["errors"],
    }


def _norm_query(text: str) -> str:
    return " ".join((text or "").split())


def embed_query(text: str, *, model: str = DEFAULT_MODEL, dim: int | None = None,
                deadline_s: float = QUERY_DEADLINE_S):
    """Embed one search query -> float32 unit vector (numpy, read-only). LRU-cached.

    Bounded to ``deadline_s`` (default 3.5 s) so chat never stalls; raises
    :class:`GeminiEmbeddingError` (or the quota subclass) when it cannot, and trips a
    short circuit-breaker so the following queries go straight to lexical retrieval.
    Identical concurrent queries share one API call (no thundering herd on a hot question).
    """
    key = (model, dim, _norm_query(text))
    _stats["queries"] += 1
    with _qcache_lock:
        hit = _qcache.get(key)
        if hit is not None:
            _qcache.move_to_end(key)
            _stats["cache_hits"] += 1
            return hit
        waiter = _inflight.get(key)
        leader = waiter is None
        if leader:
            waiter = _inflight[key] = [threading.Event(), None, None]   # event, vec, error
    if not leader:
        waiter[0].wait(timeout=deadline_s + 0.5)
        if waiter[1] is not None:
            _stats["cache_hits"] += 1
            return waiter[1]
        raise waiter[2] or GeminiEmbeddingError("Embedding temporarily skipped (recent failure).")
    try:
        vec = _embed_query_uncached(key, text, model, dim, deadline_s)
        waiter[1] = vec
        return vec
    except GeminiEmbeddingError as exc:
        waiter[2] = exc
        raise
    finally:
        with _qcache_lock:
            _inflight.pop(key, None)
        waiter[0].set()


def _embed_query_uncached(key: tuple, text: str, model: str, dim: int | None, deadline_s: float):
    now = time.monotonic()
    if _breaker["until"] > now:
        raise GeminiEmbeddingError("Embedding temporarily skipped (recent failure).")
    _stats["embed_calls"] += 1
    try:
        data = _post(model, "embedContent", _request(model, text, "query", dim),
                     n_contents=1, deadline_s=deadline_s)
        values = (data.get("embedding") or {}).get("values")
        if not values:
            raise GeminiEmbeddingError("Gemini query embedding returned no vector.")
        vec = _as_matrix([values], dim)[0]
    except GeminiEmbeddingError as exc:
        # quota: stay off the API until the earliest key cooldown ends (cap 30 s); other: 15 s
        _breaker["until"] = time.monotonic() + (_quota_backoff() if isinstance(exc, GeminiQuotaError) else 15.0)
        _note_error(exc)
        raise
    vec.setflags(write=False)
    _stats["last_ok_at"] = time.time()
    with _qcache_lock:
        _qcache[key] = vec
        while len(_qcache) > _QUERY_CACHE_MAX:
            _qcache.popitem(last=False)
    return vec


def _quota_backoff() -> float:
    """Seconds until the soonest key is usable again, clamped to [5, 30]."""
    now = time.monotonic()
    with _state_lock:
        waits = [max(0.0, st.cooldown_until - now) for st in _states.values() if st.dead_until <= now]
    return min(30.0, max(5.0, min(waits) if waits else 15.0))


def embed_documents(texts: list[str], *, model: str = DEFAULT_MODEL, dim: int | None = None,
                    deadline_s: float = 120.0, patient: bool = True, titles: list[str] | None = None):
    """Embed chunk texts (batched, rate-limit aware) -> float32 (n, dim) matrix.

    All-or-nothing per call; callers wanting partial progress should call this per
    batch (see :func:`iter_embed_documents`).
    """
    import numpy as np

    if not texts:
        return np.zeros((0, dim or 0), dtype=np.float32)
    t_end = time.monotonic() + deadline_s
    out = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start:start + BATCH_SIZE]
        reqs = [_request(model, t, "document", dim, (titles[start + i] if titles else None))
                for i, t in enumerate(batch)]
        data = _post(model, "batchEmbedContents", {"requests": reqs}, n_contents=len(batch),
                     deadline_s=max(1.0, t_end - time.monotonic()), patient=patient)
        embs = data.get("embeddings") or []
        if len(embs) != len(batch):
            raise GeminiEmbeddingError(f"Gemini batch returned {len(embs)} vectors for {len(batch)} inputs.")
        out.append(_as_matrix([e["values"] for e in embs], dim))
    return np.vstack(out)


def iter_embed_documents(texts: list[str], *, model: str = DEFAULT_MODEL, dim: int | None = None,
                         budget_s: float = 30.0, patient: bool = True, titles: list[str] | None = None):
    """Yield ``(start_index, matrix)`` batch by batch until ``budget_s`` is spent.

    Stops cleanly (no exception) when the budget runs out or quota is exhausted, so a
    caller can persist the partial progress and resume later.
    """
    t_end = time.monotonic() + budget_s
    for start in range(0, len(texts), BATCH_SIZE):
        remaining = t_end - time.monotonic()
        if remaining < 1.0:
            return
        batch = texts[start:start + BATCH_SIZE]
        try:
            mat = embed_documents(batch, model=model, dim=dim, deadline_s=remaining, patient=patient,
                                  titles=titles[start:start + BATCH_SIZE] if titles else None)
        except GeminiEmbeddingError as exc:
            log.info("Embedding stopped after %d/%d chunks: %s", start, len(texts), exc)
            return
        yield start, mat


def warm(deadline_s: float = 3.0) -> bool:
    """Best-effort connectivity check (1 content). Never raises."""
    try:
        embed_query("warm up", deadline_s=deadline_s)
        return True
    except Exception:  # noqa: BLE001
        return False
