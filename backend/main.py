"""FastAPI application: the tool APIs (and, in LOCAL dev only, the SPA).

Deployment model
----------------
* **Vercel (production)**: the SPA is static (CDN); every ``/api/*`` request is rewritten to
  ``api/index.py`` which imports ``app`` from here. Serverless rules apply: no long-lived
  background work and no local-disk persistence; the ONLY threads are the one-shot cold-start prewarm
  daemons (``backend.prewarm``: Mongo connect, JWKS, index mmap) that finish within the first invocation.
  Periodic work is a Vercel Cron
  hitting ``/api/cron/janitor`` (protected by ``CRON_SECRET``).
* **Local dev** (``python run.py``): this app also serves ``frontend/`` so one process is enough.

All environment-specific behaviour is read from ``backend.config`` / the environment.
"""
from backend import prewarm as _prewarm   # stdlib-only module: importing it costs nothing

# Cold start: begin the Mongo connect / JWKS / index preload NOW (before fastapi and the routers are even
# imported) so it overlaps the import itself and the platform's request dispatch. Non-blocking; no-op under
# pytest / PREWARM_DISABLED / without Mongo.
_prewarm.start()

import asyncio  # noqa: E402
import hmac
import logging
import os
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.datastructures import MutableHeaders

from backend import config, db, deadline, spend, timing
from backend.bodylimit import BodyLimitMiddleware
from backend.config import CORS_ORIGINS, FRONTEND_DIR, ConfigError
from backend.errors import llm_http_status, llm_unavailable
from backend.llm_chain import LLMChainError
from backend.ratelimit import rate_limit
from backend.routers import (
    bootstrap,
    chat,
    cheatsheet,
    exam,
    decks,
    flashcards,
    grading,
    guest,
    history,
    insights,
    library,
    mindmap,
    planner,
    predictor,
    progress,
    quiz,
    scannotes,
    subjects,
    teachback,
)

log = logging.getLogger("backend.main")

# Vercel sets VERCEL=1 in every deployment (build + runtime). Locally it is unset.
ON_VERCEL: bool = bool(os.environ.get("VERCEL"))

NO_LLM_LABEL = "Model: none (all providers failed)"
CRON_BUDGET_S = 50.0               # whole janitor run; Vercel maxDuration is 60 s

config.configure_logging()  # idempotent; cheap (one handler) -- also covers `uvicorn backend.main:app`


# Worker threads for blocking work (Mongo, pypdf, requests). Two separate pools exist and both default
# to far too little for an I/O-bound API: Starlette's sync routes use anyio's limiter (40) and
# ``asyncio.to_thread`` uses the loop's default executor (min(32, cpus+4): just 5-6 on a 1-2 vCPU
# serverless instance). Threads here mostly WAIT on network I/O, so a larger pool is cheap.
THREADPOOL_SIZE: int = max(8, int(os.environ.get("THREADPOOL_SIZE", "64") or 64))
_tuned = False


def _tune_concurrency() -> None:
    """Idempotent; must run inside the event loop (first request / lifespan)."""
    global _tuned
    if _tuned:
        return
    _tuned = True
    try:
        import anyio.to_thread

        anyio.to_thread.current_default_thread_limiter().total_tokens = THREADPOOL_SIZE
        asyncio.get_running_loop().set_default_executor(
            ThreadPoolExecutor(max_workers=THREADPOOL_SIZE, thread_name_prefix="app"))
    except Exception as exc:  # noqa: BLE001 -- tuning is an optimisation, never a startup failure
        log.warning("Could not tune the thread pools: %s", exc)


@asynccontextmanager
async def lifespan(_app: "FastAPI"):
    _tune_concurrency()
    # LOCAL only: fail fast with the full multi-line list of missing settings. On Vercel the
    # platform env is the source of truth, optional pieces degrade gracefully, and nothing here
    # may block (or fail) a cold start.
    if not ON_VERCEL:
        config.validate_config()
        config.log_startup_diagnostics()
    yield
    spend.flush()           # best effort: do not lose the last <20 s of the token ledger
    db.close()


app = FastAPI(
    title="Study Assistant Hub",
    version="3.0.0",
    lifespan=lifespan,
    docs_url=None if ON_VERCEL else "/docs",   # no public schema browser in production
    redoc_url=None,
    openapi_url=None if ON_VERCEL else "/openapi.json",
)

# --------------------------------------------------------------------------------------
# Middleware (outermost last). Pure-ASGI so streaming/SSE responses are never buffered.
# --------------------------------------------------------------------------------------
_RID_OK = re.compile(r"^[A-Za-z0-9._\-]{1,64}$")

_SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "X-Frame-Options": "SAMEORIGIN",
    "Permissions-Policy": "camera=(self), microphone=(self), geolocation=()",
}


class RequestMetaMiddleware:
    """Request id + ``Server-Timing`` + one access-log line + security headers.

    * ``X-Request-ID``: reuses a sane inbound id, else a fresh one (also stored on the scope
      so error handlers can echo it).
    * ``Server-Timing``: ``auth;dur=..`` / ``db;dur=..`` (recorded by backend.timing) and
      ``total;dur=..`` (time to first byte). Visible in the browser devtools Network tab.
    * ``Cache-Control: no-store`` default for /api responses so the CDN/browser never caches
      per-user data (routes may set their own).
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        if not _tuned:        # lifespan may not run on every host; the first request always does
            _tune_concurrency()
        t0 = time.perf_counter()
        timings = timing.start()
        spend.begin_request()             # owner attribution for the spend ledger (filled by current_owner)
        deadline.begin_request()          # one shared time budget for everything this request does
        raw = Request(scope).headers.get("x-request-id", "")
        rid = raw if _RID_OK.match(raw or "") else uuid.uuid4().hex[:16]
        scope.setdefault("state", {})["request_id"] = rid
        path = scope.get("path", "")
        status_holder = {"status": 0}

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                status_holder["status"] = message["status"]
                headers = MutableHeaders(scope=message)
                headers["X-Request-ID"] = rid
                headers["Server-Timing"] = timing.header(timings, (time.perf_counter() - t0) * 1000.0)
                extra = _prewarm.server_timing_parts()       # first response of this instance only
                if extra:
                    headers["Server-Timing"] += ", " + extra
                    headers["X-Instance-Age"] = f"{_prewarm.instance_age_s():.2f}"
                for k, v in _SECURITY_HEADERS.items():
                    headers.setdefault(k, v)
                if path.startswith("/api/") and "cache-control" not in headers:
                    headers["Cache-Control"] = "no-store"
            elif message["type"] == "http.response.body" and not message.get("more_body"):
                if path != "/api/health":  # keep uptime probes out of the logs
                    log.info(
                        "%s %s -> %s %.0fms rid=%s",
                        scope.get("method"), path, status_holder["status"],
                        (time.perf_counter() - t0) * 1000.0, rid,
                    )
            await send(message)

        await self.app(scope, receive, send_wrapper)


# Inner -> outer. (add_middleware makes the LAST one added the outermost.)
if CORS_ORIGINS:
    log.info("CORS enabled for origins: %s", ", ".join(CORS_ORIGINS))
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-Guest-Id", "X-Request-ID"],
        expose_headers=["Server-Timing", "X-Request-ID"],
        max_age=600,
    )
app.add_middleware(BodyLimitMiddleware)   # JSON 413 for oversize upload bodies (Vercel's own is not JSON)
app.add_middleware(GZipMiddleware, minimum_size=1024, compresslevel=5)  # skips text/event-stream
app.add_middleware(RequestMetaMiddleware)


# --------------------------------------------------------------------------------------
# Error handling. Optional/renamed exception classes are imported defensively so a refactor
# elsewhere (llm_chain, embeddings, documents) can never stop the API from booting.
# --------------------------------------------------------------------------------------
def _optional(module: str, name: str):  # only for modules still being refactored (documents/embeddings)
    try:
        mod = __import__(module, fromlist=[name])
        return getattr(mod, name, None)
    except Exception:  # noqa: BLE001 -- ImportError or a half-migrated module
        return None


def _rid(request: Request) -> str:
    return request.scope.get("state", {}).get("request_id", "")


def _llm_unavailable(_request, exc) -> JSONResponse:
    log.warning("LLM chain exhausted: %s", str(exc)[:300])
    detail, label = llm_unavailable(
        exc, "The AI service is temporarily unavailable — every model provider failed. "
             "Please try again in a moment.", NO_LLM_LABEL)
    return JSONResponse(status_code=llm_http_status(exc), content={"detail": detail, "llm": {"label": label}})


app.add_exception_handler(LLMChainError, _llm_unavailable)

@app.exception_handler(config.SessionSecretMissing)
async def _on_no_session_secret(_request, exc) -> JSONResponse:
    log.error("%s", exc)
    return JSONResponse(status_code=503, content={"detail": str(exc)})


_DocumentError = _optional("backend.documents", "DocumentError")
if _DocumentError is not None:
    @app.exception_handler(_DocumentError)
    async def _on_document_error(_request, exc) -> JSONResponse:
        log.info("Document error: %s", exc)
        return JSONResponse(status_code=404, content={"detail": str(exc)})

_MissingApiKeyError = getattr(config, "MissingApiKeyError", None)
if _MissingApiKeyError is not None:
    @app.exception_handler(_MissingApiKeyError)
    async def _on_missing_key(_request, exc) -> JSONResponse:
        log.warning("LLM call attempted without a configured provider key: %s", exc)
        return JSONResponse(status_code=503, content={"detail": str(exc), "llm": {"label": NO_LLM_LABEL}})

_GeminiEmbeddingError = _optional("backend.gemini_embeddings", "GeminiEmbeddingError")
_GeminiQuotaError = _optional("backend.gemini_embeddings", "GeminiQuotaError")
if _GeminiEmbeddingError is not None:
    @app.exception_handler(_GeminiEmbeddingError)
    async def _on_embedding_error(_request, exc) -> JSONResponse:
        # Quota exhaustion -> 429 (retry-able); anything else -> 502. Never a bare 500.
        detail = str(exc)
        if _GeminiQuotaError is not None and isinstance(exc, _GeminiQuotaError):
            log.warning("Gemini embedding quota reached: %s", detail)
            return JSONResponse(status_code=429, content={"detail": detail})
        log.error("Embedding backend error: %s", detail)            # upstream text stays in the log
        return JSONResponse(status_code=502, content={"detail": "The embedding service failed. Please try again in a moment."})


@app.exception_handler(Exception)
async def _on_unhandled(request: Request, exc: Exception) -> JSONResponse:
    """Last resort: JSON (never an HTML traceback), request id for log correlation.

    MongoDB failures (pymongo is imported lazily, so match by module) become a clean 503.
    """
    rid = _rid(request)
    headers = {"X-Request-ID": rid} if rid else {}
    if any(c.__module__.split(".")[0] in ("pymongo", "bson") for c in type(exc).__mro__):
        log.error("MongoDB error on %s rid=%s: %s", request.url.path, rid, str(exc)[:300])
        return JSONResponse(
            status_code=503, headers=headers,
            content={"detail": "The database is temporarily unavailable. Please retry in a moment."},
        )
    log.exception("Unhandled error on %s rid=%s", request.url.path, rid)
    return JSONResponse(
        status_code=500, headers=headers,
        content={"detail": "Internal server error.", "request_id": rid},
    )


# --------------------------------------------------------------------------------------
# Health, config, cron.
# --------------------------------------------------------------------------------------
@app.get("/api/health")
async def health() -> dict:
    """Liveness only: no DB, no LLM, no thread hop. Safe for uptime pings."""
    return {"status": "ok"}


@app.get("/api/health/init")
async def health_init() -> dict:
    """Cold-start telemetry: import/init time, instance age, prewarm step spans (ms). No secrets."""
    return {"init_ms": round(_prewarm.INIT_MS or 0), "instance_age_s": round(_prewarm.instance_age_s(), 2),
            "prewarm_ms": {k: round(v) for k, v in _prewarm.SPANS.items()}}


@app.get("/api/health/deep")
def health_deep() -> JSONResponse:
    """Readiness: Mongo ping + provider-chain configuration (booleans only, never keys)."""
    checks: dict = {}
    ok = True
    if config.PERSISTENCE_ENABLED:
        t0 = time.perf_counter()
        try:
            db.ping()
            checks["mongo"] = {"ok": True, "ms": round((time.perf_counter() - t0) * 1000, 1)}
        except Exception as exc:  # noqa: BLE001
            ok = False
            checks["mongo"] = {"ok": False, "error": type(exc).__name__}
    else:
        checks["mongo"] = {"ok": None, "note": "persistence disabled"}
    # Fail-closed production checks: a deployment without Clerk / SESSION_SECRET refuses to serve
    # user data, so report it loudly (degraded -> 503) for uptime monitors.
    if config.auth_misconfigured():
        ok = False
        checks["auth"] = {"ok": False, "warning": "CLERK_PUBLISHABLE_KEY is not set: authenticated routes return 503"}
    else:
        checks["auth"] = {"ok": True}
    if config.session_secret_missing():
        ok = False
        checks["session_secret"] = {"ok": False, "warning": "SESSION_SECRET is not set: chat sessions are refused"}
    try:   # informational: a degraded retrieval path (BM25 only) is never an outage -> no 503
        from backend import vector_store

        checks["retrieval"] = _booleans_only(vector_store.retrieval_health())
    except Exception as exc:  # noqa: BLE001
        checks["retrieval"] = {"ok": False, "error": type(exc).__name__}
    try:
        checks["llm_chain"] = _booleans_only(config.chain_status())
    except Exception as exc:  # noqa: BLE001
        ok = False
        checks["llm_chain"] = {"ok": False, "error": type(exc).__name__}
    try:   # informational (never 503): premium-model spend guard; counts and percentages only, no owner ids
        st = spend.status(refresh=True)
        checks["budgets"] = {k: st[k] for k in ("global_used", "global_limit", "pct", "owners_tracked",
                                                "anthropic_skipped_today", "warn", "exhausted", "store")}
    except Exception as exc:  # noqa: BLE001
        checks["budgets"] = {"ok": False, "error": type(exc).__name__}
    return JSONResponse(status_code=200 if ok else 503, content={"status": "ok" if ok else "degraded", "checks": checks})


def _booleans_only(value):
    """Defence in depth: keep only bool/number leaves (+ short strings for step names), so a
    future change to chain_status can never leak key material through the health endpoint."""
    if isinstance(value, dict):
        return {str(k): _booleans_only(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_booleans_only(v) for v in value]
    if isinstance(value, (bool, int, float)) or value is None:
        return value
    if isinstance(value, str) and len(value) <= 40 and not value.lower().startswith(("sk-", "gsk_", "aiza")):
        return value
    return "<redacted>"


@app.get("/api/config")
def client_config(response: Response) -> dict:
    """Public, non-secret configuration the SPA reads at boot. Driven entirely by env.

    Anonymous, identical for everyone and free of secrets (only the PUBLIC Clerk key), so the CDN
    may hold it briefly (``s-maxage``, then serve stale while revalidating); browsers always
    revalidate (``max-age=0``). The SPA keeps its own client-side copy (localStorage) for instant
    boots. Per-user data lives in ``/api/bootstrap`` etc., which are ``private, no-store``.
    """
    response.headers["Cache-Control"] = "public, max-age=0, s-maxage=60, stale-while-revalidate=300"
    return config.public_config()


def _require_cron_auth(request: Request) -> None:
    """Vercel Cron sends ``Authorization: Bearer $CRON_SECRET`` when that env var is set."""
    secret = os.environ.get("CRON_SECRET", "")
    supplied = request.headers.get("authorization", "")
    expected = f"Bearer {secret}"
    if not secret or not hmac.compare_digest(supplied.encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="Unauthorized")


@app.get("/api/cron/janitor")
def cron_janitor(request: Request) -> dict:
    """Daily housekeeping (Vercel Cron): reap artifacts of TTL-expired guests and make sure the
    Mongo indexes exist. Replaces the old in-process janitor thread."""
    _require_cron_auth(request)
    if not config.PERSISTENCE_ENABLED:
        return {"ok": True, "removed": 0, "note": "persistence disabled"}
    t0 = time.perf_counter()
    # ONE shared budget for the whole run (the function limit is 60 s): ensure_indexes, then the
    # artifact sweep, then the uploads sweep each get what is left. Both sweeps delete as they
    # go and stop at their deadline, so an unfinished run is simply resumed by the next one.
    deadline = time.monotonic() + CRON_BUDGET_S
    indexes = db.ensure_indexes()
    partial = False
    removed = 0
    left = deadline - time.monotonic()
    if left > 2:
        stats: dict = {}
        removed = db.purge_orphan_guest_data(budget_s=max(1.0, left * 0.5), stats=stats)
        partial = partial or bool(stats.get("partial"))
    else:
        partial = True
    uploads_removed = 0
    left = deadline - time.monotonic()
    if left > 2:
        try:  # GridFS uploads/index blobs of expired guests (Agent F's sweep); never fail the cron
            from backend.services import subjects as subjects_service

            stats = {}
            uploads_removed = subjects_service.purge_orphan_guest_uploads(budget_s=min(30, left - 1), stats=stats)
            partial = partial or bool(stats.get("partial"))
        except Exception as exc:  # noqa: BLE001
            log.warning("Orphan upload sweep failed: %s", exc)
    else:
        partial = True
    log.info("Janitor: removed %d artifact(s), %d upload file(s), indexes=%s partial=%s",
             removed, uploads_removed, indexes, partial)
    return {"ok": True, "removed": removed, "uploads_removed": uploads_removed, "indexes": indexes,
            "partial": partial, "ms": round((time.perf_counter() - t0) * 1000)}


# Routers (each defines its own /api/<...> prefix).
# Routers whose endpoints spend LLM tokens get the per-owner limiter (backend/ratelimit.py) as a
# router-level dependency; the cheap bookkeeping routers (subjects, progress, history, guest) do not.
_PAID = [Depends(rate_limit)]
app.include_router(bootstrap.router)
app.include_router(subjects.router)
app.include_router(chat.router, dependencies=_PAID)
app.include_router(quiz.router, dependencies=_PAID)
app.include_router(decks.router, dependencies=_PAID)
app.include_router(flashcards.router, dependencies=_PAID)
app.include_router(planner.router, dependencies=_PAID)
app.include_router(mindmap.router, dependencies=_PAID)
app.include_router(exam.router, dependencies=_PAID)
app.include_router(teachback.router, dependencies=_PAID)
app.include_router(predictor.router, dependencies=_PAID)
app.include_router(progress.router)
app.include_router(grading.router)      # POST /api/grade is rate limited per endpoint; attempts reads are cheap
app.include_router(insights.router, dependencies=_PAID)
app.include_router(cheatsheet.router, dependencies=_PAID)
app.include_router(scannotes.router, dependencies=_PAID)
app.include_router(history.router)
app.include_router(library.router)      # read-only overview; lazy-loaded by the SPA (not in /api/bootstrap)
app.include_router(guest.router)


# --------------------------------------------------------------------------------------
# Static SPA -- LOCAL DEV ONLY. On Vercel the CDN serves the built frontend (see vercel.json);
# this function is only reached for /api/*, so these routes are not registered there.
#
# The SPA ships with FIXED asset filenames (no content hashes), so locally we serve the shell and
# assets with "no-cache" (always revalidate -> cheap 304, fresh file right after an edit).
# --------------------------------------------------------------------------------------
class NoCacheStaticFiles(StaticFiles):
    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response


if not ON_VERCEL and (FRONTEND_DIR / "assets").is_dir():
    app.mount("/assets", NoCacheStaticFiles(directory=str(FRONTEND_DIR / "assets")), name="assets")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(str(FRONTEND_DIR / "index.html"), headers={"Cache-Control": "no-cache"})


_prewarm.mark_ready()      # INIT_MS: process start -> app fully built (reported once in Server-Timing)

# Re-export ConfigError so importers can reference it from main if desired.
__all__ = ["app", "ConfigError"]
