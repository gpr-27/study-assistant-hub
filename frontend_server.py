"""Static SPA server for split-origin mode (the frontend on its OWN port).

Single-server mode (the default) is unaffected — backend.main already serves the SPA on the
API port. When FRONTEND_PORT is set, run.py ALSO starts THIS tiny server on that port to
serve the static SPA, while the browser reaches the API cross-origin at config.API_BASE_URL.

It serves three things, nothing more:
  • GET /runtime-config.js  -> seeds window.__API_BASE__ with the absolute API base, so the
                               SPA's first request (GET /config) goes to the API server.
  • /assets/*               -> the static CSS/JS.
  • everything else         -> index.html (SPA entry; deep links / refresh all work).

No CORS is configured here (it only serves same-origin static files to itself); CORS lives on
the API server (backend.main), driven by config.CORS_ORIGINS.
"""
from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from backend import config

log = logging.getLogger("frontend.server")


def build_frontend_app() -> FastAPI:
    """Build the static-SPA FastAPI app (no API routes, no DB, no LLM)."""
    app = FastAPI(title="Study Assistant Hub — SPA", version="2.0.0")

    @app.get("/runtime-config.js")
    def runtime_config_js() -> Response:
        # In split mode this is the absolute API_BASE_URL; mirrors backend.main's copy.
        js = f'window.__API_BASE__ = "{config.runtime_api_base()}";\n'
        return Response(content=js, media_type="application/javascript")

    # Static assets (same path the SPA references: /assets/...).
    app.mount("/assets", StaticFiles(directory=str(config.FRONTEND_DIR / "assets")), name="assets")

    @app.get("/")
    @app.get("/{_full_path:path}")
    def spa(_full_path: str = "") -> FileResponse:
        # SPA fallback: every non-asset path returns index.html so client-side
        # hash routing / refresh / deep links work.
        return FileResponse(str(config.FRONTEND_DIR / "index.html"))

    return app


def serve() -> None:
    """Run the frontend server (blocking). Used by run.py in a background thread."""
    import uvicorn

    host = config.HOST or "127.0.0.1"
    log.info("Serving the SPA on http://%s:%s (API at %s).", host, config.FRONTEND_PORT, config.runtime_api_base())
    uvicorn.run(
        build_frontend_app(),
        host=host,
        port=config.FRONTEND_PORT,
        log_level=config.LOG_LEVEL.lower(),
    )
