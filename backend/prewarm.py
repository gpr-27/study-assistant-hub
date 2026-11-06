"""Cold-start prewarm: do the first request's one-off work in background daemon threads at import time.

A fresh serverless instance otherwise pays, on its FIRST request: the Mongo SRV lookup + TCP/TLS/auth
handshake (~400-800 ms), creation of the pooled httpx client, the Clerk JWKS fetch (~100 ms) and the
bundled subject indexes' mmap load (6-10 ms each). Starting those here overlaps them with the platform's
own cold-start/handler dispatch; the process stays alive for the invocation, so the threads finish.

* ``start()`` never blocks and never raises; it is idempotent (double-start guarded).
* Skipped under pytest and when ``PREWARM_DISABLED=1``; each step is skipped when its feature is off
  (no ``MONGODB_URI`` -> no Mongo/index work; no Clerk key -> no JWKS).
* Requests that arrive mid-connect wait for the in-flight Mongo connect (``db._get_db``) -- they never
  start a second one.
"""
from __future__ import annotations

import logging
import os
import sys
import threading
import time

# Process-start reference (first import of this stdlib-only module == start of the function's own import work).
T0 = time.perf_counter()
WALL0 = time.time()
INIT_MS: float | None = None          # set by backend.main once the app object is fully built
PHASES: dict[str, float] = {}         # prewarm step name -> ms from T0 at which it FINISHED (also its start-to-end span in SPANS)
SPANS: dict[str, float] = {}          # prewarm step name -> duration ms
_reported = False

log = logging.getLogger("study_assistant.prewarm")
_started = False
_lock = threading.Lock()


def enabled() -> bool:
    if os.environ.get("PREWARM_DISABLED", "").lower() in ("1", "true", "yes"):
        return False
    return "pytest" not in sys.modules


def mark_ready() -> None:
    global INIT_MS
    INIT_MS = (time.perf_counter() - T0) * 1000.0


def instance_age_s() -> float:
    return time.time() - WALL0


def server_timing_parts() -> str:
    """Extra Server-Timing entries for the FIRST response of an instance only (init + prewarm spans)."""
    global _reported
    if _reported or INIT_MS is None:
        return ""
    _reported = True
    parts = [f"init;dur={INIT_MS:.0f}"]
    for k, v in list(SPANS.items()):
        parts.append(f"pw_{k.replace('-', '_')};dur={v:.0f}")
    parts.append(f"pw_done_at;dur={(time.perf_counter() - T0) * 1000.0:.0f}")   # request arrival offset since T0
    return ", ".join(parts)


def _timed(name: str, fn) -> None:
    t0 = time.perf_counter()
    try:
        fn()
    finally:
        SPANS[name] = (time.perf_counter() - t0) * 1000.0


def _step(name: str, fn) -> None:
    try:
        _timed(name, fn)
    except Exception as exc:  # noqa: BLE001 -- prewarm is an optimisation, never a failure
        log.info("prewarm step %s skipped: %s", name, exc)


def _mongo_then_indexes() -> None:
    from backend import config, db

    if not config.PERSISTENCE_ENABLED:
        return
    t0 = time.perf_counter()
    db.prewarm()
    db._prewarm_done.wait(5.0)                      # indexes after the connection is up
    SPANS["mongo"] = (time.perf_counter() - t0) * 1000.0
    from backend import vector_store

    _step("indexes", vector_store.warm_all_subjects)


_jwks_started = threading.Event()
_jwks_done = threading.Event()


def wait_jwks(timeout: float = 3.0) -> None:
    """A request that needs the JWK set while the prewarm fetch is in flight waits for it instead of
    issuing a second HTTPS fetch. No-op when prewarm never started / already finished."""
    if _jwks_started.is_set() and not _jwks_done.is_set():
        _jwks_done.wait(timeout)


def _jwks() -> None:
    """Clerk JWKS fetch in its OWN thread: a signed-in first request needs it, so it must not queue
    behind the (GIL-heavy) module imports below."""
    from backend import auth, config

    if not config.CLERK_PUBLISHABLE_KEY:
        return
    _jwks_started.set()
    try:
        client = auth._jwks_client()
        if client is not None:
            client.get_signing_keys()          # fills the JWK-set cache (one HTTPS fetch)
    finally:
        _jwks_done.set()


def _network_free_clients() -> None:
    from backend import llm_chain

    _step("llm_chain", llm_chain.warm)

    def _lazy_imports() -> None:         # modules the first chat request would otherwise import itself
        from backend import gemini_embeddings
        from backend.rag import graph  # noqa: F401
        from backend.services import chat_tools  # noqa: F401

        gemini_embeddings._http()         # requests.Session (imports requests/netrc/charset_normalizer)
    _step("lazy_imports", _lazy_imports)


def start() -> bool:
    """Kick the prewarm threads. Returns True only on the call that started them."""
    global _started
    if not enabled():
        return False
    with _lock:
        if _started:
            return False
        _started = True
    for name, fn in (("prewarm-mongo", _mongo_then_indexes), ("prewarm-jwks", _jwks),
                     ("prewarm-clients", _network_free_clients)):
        threading.Thread(target=lambda f=fn, n=name: _step(n, f), daemon=True, name=name).start()
    return True
