"""Launch the Study Assistant Hub (single FastAPI server, serves SPA + APIs).

Host, port, and auto-reload come entirely from configuration (backend.config,
which reads .env) — nothing about the environment is hardcoded here. The server
refuses to start if required configuration is missing.
"""
import os
import sys
from pathlib import Path

# FAISS and the embedding runtime (onnxruntime/torch) each ship their own copy of
# libomp. On macOS, loading both into one process aborts with "OMP: Error #15:
# Initializing libomp.dylib, but found libomp.dylib already initialized" — which
# crashed the server the moment a subject's embeddings were loaded for chat. Allow
# the duplicate runtime (the standard, supported-enough workaround). Set before any
# heavy import and before the venv re-exec so every child process inherits it.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parent
VENV_PYTHON = ROOT / ".venv" / "bin" / "python"


def _reexec_in_venv_if_needed() -> None:
    """Re-launch under the project's .venv interpreter when started with a
    different Python (e.g. a base conda env that lacks the dependencies).

    Without this, running `python run.py` from the wrong shell fails with
    ModuleNotFoundError (langchain_groq, faiss, …) and leaves nothing serving,
    which shows up in the browser as ERR_CONNECTION_REFUSED.
    """
    if not VENV_PYTHON.exists():
        return
    if Path(sys.prefix).resolve() == (ROOT / ".venv").resolve():
        return  # already running inside the project venv
    os.execv(str(VENV_PYTHON), [str(VENV_PYTHON), str(ROOT / "run.py"), *sys.argv[1:]])


_reexec_in_venv_if_needed()

import uvicorn  # noqa: E402 — imported only after the venv re-exec guard

from backend import config  # noqa: E402 — imported after the venv re-exec guard


def main() -> None:
    config.configure_logging()
    try:
        config.validate_config()
    except config.ConfigError as exc:
        # Fail fast with a clear, actionable message instead of crashing deep in
        # uvicorn/Groq later. Nothing is started until configuration is valid.
        print(str(exc), file=sys.stderr)
        sys.exit(1)

    # Split-origin mode: also serve the static SPA on its own port (FRONTEND_PORT). The SPA
    # talks to the API cross-origin at API_BASE_URL (allowed via CORS_ORIGINS). Single-server
    # mode (FRONTEND_PORT unset) skips this entirely — backend.main serves the SPA itself.
    # The frontend server runs in a daemon thread: it serves static files straight from disk
    # (no reload needed), so the API process keeps its auto-reload behaviour unchanged.
    if config.SPLIT_ORIGIN:
        import threading

        from frontend_server import serve as serve_frontend

        threading.Thread(target=serve_frontend, daemon=True, name="frontend-server").start()

    # Auto-reload watches ONLY the source dirs. Building/saving per-subject indexes
    # under faiss_index/ and saving uploaded PDFs under documents/ must never be
    # mistaken for a code change and restart the server mid-request. RELOAD is
    # derived from APP_ENV (development → reload, production → no reload).
    uvicorn.run(
        "backend.main:app",
        host=config.HOST,
        port=config.PORT,
        reload=config.RELOAD,
        reload_dirs=["backend", "frontend"],
        log_level=config.LOG_LEVEL.lower(),
    )


if __name__ == "__main__":
    main()
