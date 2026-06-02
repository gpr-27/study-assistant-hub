"""Vercel serverless entrypoint — exposes the FastAPI app for the @vercel/python runtime.

Vercel's Python runtime serves a file in the ``api/`` directory that exposes an ASGI
``app`` variable. ``vercel.json`` rewrites every path to this function, so the one FastAPI
app serves the SPA, the static assets, and the ``/api/*`` routes — exactly like the
container does, just on one serverless function instead of a long-lived process.

The real application lives in ``backend.main``; this module re-exports it. On Vercel the
app runs in serverless "lite" mode (``backend.config.IS_SERVERLESS``, auto-detected from the
``VERCEL`` env var): local-embeddings/FAISS retrieval is off, background threads are skipped,
and writable state lives under ``/tmp``. Chat still works — it grounds on the document text
directly. The full retrieval stack runs in the container/local deployment (see DEPLOYMENT.md).
"""
from backend.main import app  # noqa: F401  (re-exported for the @vercel/python runtime)
