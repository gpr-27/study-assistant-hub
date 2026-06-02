"""Gunicorn configuration for the production container.

Run with:  gunicorn backend.main:app -c gunicorn.conf.py

Everything tunable is read from the environment so one image serves any
deployment (see .env.production.example):

  PORT                          TCP port to bind                  (default 8000)
  WEB_CONCURRENCY               number of worker processes        (default 2)
  GUNICORN_TIMEOUT              worker timeout, seconds           (default 120)
  GUNICORN_GRACEFUL_TIMEOUT     graceful shutdown window, seconds (default 30)
  GUNICORN_KEEPALIVE            keep-alive, seconds               (default 5)
  GUNICORN_MAX_REQUESTS         recycle a worker after N requests (default 1000)
  GUNICORN_MAX_REQUESTS_JITTER  +/- jitter on the above           (default 100)
  LOG_LEVEL                     gunicorn error-log level          (default info)

Workers are Uvicorn workers, so the FastAPI app — including its async lifespan,
which validates configuration, warms the embedding model + per-subject indexes,
and starts the guest janitor — runs exactly as in development.

NOTE: each worker loads its own copy of the local embedding model, so resident
memory scales with WEB_CONCURRENCY. For typical study-app traffic 1–2 workers is
plenty; the app is I/O-bound (waiting on Groq / MongoDB) and FastAPI's sync routes
already run in a threadpool within each worker.
"""
import os


def _int_env(name: str, default: int) -> int:
    """Read an int env var, falling back to ``default`` if unset/blank/invalid."""
    raw = (os.environ.get(name) or "").strip()
    try:
        return int(raw)
    except ValueError:
        return default


# Bind to all interfaces inside the container on the configured port.
bind = f"0.0.0.0:{_int_env('PORT', 8000)}"

worker_class = "uvicorn.workers.UvicornWorker"
workers = _int_env("WEB_CONCURRENCY", 2)

# LLM + embedding work can be slow; don't let gunicorn kill a busy worker.
timeout = _int_env("GUNICORN_TIMEOUT", 120)
graceful_timeout = _int_env("GUNICORN_GRACEFUL_TIMEOUT", 30)
keepalive = _int_env("GUNICORN_KEEPALIVE", 5)

# Recycle workers periodically to bound any slow memory growth; jitter avoids a
# synchronized (thundering-herd) restart. Set MAX_REQUESTS=0 to disable.
max_requests = _int_env("GUNICORN_MAX_REQUESTS", 1000)
max_requests_jitter = _int_env("GUNICORN_MAX_REQUESTS_JITTER", 100)

# Stream logs to stdout/stderr so the container runtime collects them.
accesslog = "-"
errorlog = "-"
loglevel = (os.environ.get("LOG_LEVEL") or "info").lower()

# Each Uvicorn worker runs the app's lifespan AFTER fork (warmup + janitor are
# per-worker), so preloading would not share the model and could double-run hooks.
preload_app = False
