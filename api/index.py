"""Vercel entrypoint: exposes the FastAPI ``app`` (ASGI) for the serverless function.

All `/api/*` traffic is rewritten here (see vercel.json). The static SPA never reaches this
function — Vercel serves it from the CDN. Keep this file tiny: importing it IS the cold start.
"""
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:  # the repo root holds the `backend` package
    sys.path.insert(0, _ROOT)

from backend import prewarm as _prewarm  # noqa: E402  (stdlib-only; stamps T0 = start of our import work)

_prewarm.start()   # Mongo connect / JWKS / index preload overlap the (slow) fastapi + router imports below

from backend.main import app  # noqa: E402,F401  (Vercel looks up the module-level `app`)
