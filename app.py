"""Vercel entrypoint — exposes the FastAPI app for the @vercel/python runtime.

Vercel's FastAPI framework detects an ``app`` variable in app.py / index.py /
server.py / main.py at the project root. The real application lives in
``backend.main``; this module simply re-exports it.

On Vercel the app runs in serverless "lite" mode (``backend.config.IS_SERVERLESS``,
auto-detected from the ``VERCEL`` env var): local-embeddings/FAISS retrieval is
disabled, background warmup/janitor threads are skipped, and writable state lives
under ``/tmp`` — so the app fits a serverless function and needs no persistent disk.
Chat still works: it grounds answers on the document text directly. The full
retrieval stack runs in the container/local deployment instead (see DEPLOYMENT.md).
"""
from backend.main import app  # noqa: F401  (re-exported for Vercel to discover)
