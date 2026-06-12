"""FastAPI application: serves the liquid-glass SPA and the tool APIs.

All environment-specific behaviour (which models are offered, the provider, CORS
origins, log level) is read from ``backend.config`` — never hardcoded here.
"""
import logging
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from groq import APIError, RateLimitError

from backend import config
from backend.config import (
    CORS_ORIGINS,
    FRONTEND_DIR,
    ConfigError,
    MissingApiKeyError,
    has_groq_key,
)
from backend import db
from backend.documents import DocumentError
from backend.gemini_embeddings import GeminiEmbeddingError, GeminiQuotaError
from backend.routers import (
    chat,
    cheatsheet,
    exam,
    flashcards,
    guest,
    history,
    insights,
    mindmap,
    planner,
    predictor,
    progress,
    quiz,
    scannotes,
    subjects,
    teachback,
)
from backend.vector_store import warm_embeddings

log = logging.getLogger("backend.main")


@asynccontextmanager
async def lifespan(_app: "FastAPI"):
    # Configure logging and validate configuration before serving. Validation
    # fails fast with a clear, multi-line message if any required value is unset,
    # so the server never runs in a half-configured state. (run.py validates too,
    # for the common entry point; this covers `uvicorn backend.main:app` directly.)
    config.configure_logging()
    config.validate_config()
    config.log_startup_diagnostics()

    # In the background, warm the local embedding model (one-time download) and
    # build any missing/stale per-subject indexes, so the first chat session is
    # instant and later runs just load saved indexes from disk.
    def _warm() -> None:
        log.info("Warming embedding model in the background…")
        # Re-hydrate any user-uploaded subjects from MongoDB BEFORE building indexes, so
        # uploads survive a restart on a host without a persistent disk (e.g. Render's free
        # tier). No-op when persistence is disabled; bundled subjects are unaffected.
        from backend.services.subjects import restore_uploaded_pdfs

        restore_uploaded_pdfs()
        # Warm the embedding model singleton (one-time ONNX load). Subject indexes are
        # loaded lazily on first chat so startup RAM stays well under 512 MB on the free tier.
        warm_embeddings()
        log.info("Embedding model warm.")

    threading.Thread(target=_warm, daemon=True).start()

    # Start the guest janitor: it reaps artifacts orphaned when an inactive guest is
    # removed by MongoDB's TTL index (TTL drops the guest record but can't cascade to its
    # chats/quizzes/etc). No-op when persistence is disabled.
    db.start_guest_janitor()
    yield
    # Shutdown: release the Mongo client if persistence was used.
    db.close()


app = FastAPI(title="Study Assistant Hub", version="2.0.0", lifespan=lifespan)

# CORS is driven entirely by config.CORS_ORIGINS (empty by default). The SPA is
# served by this same server, so no cross-origin access is configured unless the
# operator explicitly lists origins in .env — no origin is ever hardcoded here.
if CORS_ORIGINS:
    log.info("CORS enabled for origins: %s", ", ".join(CORS_ORIGINS))
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
else:
    log.debug("CORS disabled — SPA and API share one origin.")


@app.exception_handler(DocumentError)
async def _on_document_error(_request, exc: DocumentError) -> JSONResponse:
    log.info("Document error: %s", exc)
    return JSONResponse(status_code=404, content={"detail": str(exc)})



@app.exception_handler(MissingApiKeyError)
async def _on_missing_key(_request, exc: MissingApiKeyError) -> JSONResponse:
    log.warning("LLM call attempted without a Groq key: %s", exc)
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(RateLimitError)
async def _on_rate_limit(_request, _exc: RateLimitError) -> JSONResponse:
    # Groq's free tier caps tokens-per-day; surface a clear, actionable message
    # instead of a bare 500 from the unhandled upstream error. The remedy points
    # at config (a lighter model / DEFAULT_MODEL in .env) without naming any model
    # id in source.
    log.warning("Groq rate limit reached.")
    return JSONResponse(
        status_code=429,
        content={"detail": (
            "Groq rate limit reached — the free-tier daily token limit is used up, or "
            "requests are coming in too fast. Wait a few minutes and retry, switch to a "
            "lighter model (set DEFAULT_MODEL — or a per-tool override — to a smaller "
            "model in .env, then restart), or upgrade your tier at "
            "https://console.groq.com/settings/billing."
        )},
    )


@app.exception_handler(GeminiEmbeddingError)
async def _on_embedding_error(_request, exc: GeminiEmbeddingError) -> JSONResponse:
    # Embedding-backend failures (Gemini): quota exhaustion → 429 (retry-able),
    # anything else → 502. Surfaces a clear message instead of a stuck spinner.
    detail = str(exc)
    if isinstance(exc, GeminiQuotaError):
        log.warning("Gemini embedding quota reached: %s", detail)
        return JSONResponse(status_code=429, content={"detail": detail})
    log.error("Embedding backend error: %s", detail)
    return JSONResponse(status_code=502, content={"detail": f"The embedding service failed: {detail}"})


@app.exception_handler(APIError)
async def _on_groq_error(_request, exc: APIError) -> JSONResponse:
    # Any other Groq/upstream API failure (auth, model, transport) — 502, with the
    # provider's own message when available, rather than a generic 500.
    detail = getattr(exc, "message", None) or str(exc)
    log.error("Upstream AI provider error: %s", detail)
    return JSONResponse(status_code=502, content={"detail": f"The AI service returned an error: {detail}"})


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "groq_key": has_groq_key()}


@app.get("/api/config")
def client_config() -> dict:
    """Public, non-secret configuration the SPA reads at boot (models, provider,
    default model, app env). Driven entirely by .env — no values hardcoded."""
    return config.public_config()


@app.get("/runtime-config.js")
def runtime_config_js() -> Response:
    """Tiny script (loaded before config.js) that seeds the API base into the page.

    Same-origin "/api" in single-server mode, or the absolute API_BASE_URL in split-origin
    mode — so the SPA can make its very FIRST request (GET /api/config) to the right place.
    The frontend server serves its own copy of this on FRONTEND_PORT; this route covers the
    SPA when it's opened directly on the backend."""
    js = f'window.__API_BASE__ = "{config.runtime_api_base()}";\n'
    return Response(content=js, media_type="application/javascript")


# Routers (each defines its own /api/<...> prefix).
app.include_router(subjects.router)
app.include_router(chat.router)
app.include_router(quiz.router)
app.include_router(flashcards.router)
app.include_router(planner.router)
app.include_router(mindmap.router)
app.include_router(exam.router)
app.include_router(teachback.router)
app.include_router(predictor.router)
app.include_router(progress.router)
app.include_router(insights.router)
app.include_router(cheatsheet.router)
app.include_router(scannotes.router)
app.include_router(history.router)
app.include_router(guest.router)

# Static SPA. /api/* is registered above, so it always wins over these.
app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIR / "assets")), name="assets")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(str(FRONTEND_DIR / "index.html"))


# Re-export ConfigError so importers can reference it from main if desired.
__all__ = ["app", "ConfigError"]
