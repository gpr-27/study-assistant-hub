"""One-shot SPA boot payload: ``GET /api/bootstrap``.

Replaces the boot waterfall (``/api/config`` -> ``/api/subjects`` -> ``/api/progress/summary`` ->
``/api/history/chats``) with a single request. Every part reuses the existing per-owner caches
(``list_subjects`` / ``progress.build_summary`` / ``db.list_items``) and runs concurrently.

* Anonymous callers (no/invalid identity) get ``{"config": ..., "owner": false}`` -- never a 401, so the
  gate can still use it.
* Signed-in / guest callers also get ``subjects``, ``progress`` and ``chats``; a part that fails is
  ``null`` and named in ``errors`` (the SPA then falls back to the individual endpoint).
* ALWAYS ``Cache-Control: private, no-store``. The CDN-cacheable anonymous piece is ``/api/config``
  itself (``s-maxage``), never this per-identity response. The old endpoints are unchanged.
"""
import asyncio
import logging

from fastapi import APIRouter, HTTPException, Request, Response

from backend import config, db, timing
from backend.auth import current_owner
from backend.documents import list_subjects
from backend.services.progress import build_summary

log = logging.getLogger("backend.bootstrap")
router = APIRouter(prefix="/api", tags=["bootstrap"])

RECENT_CHATS = 20


def _subjects(owner: str) -> dict:
    visible = list_subjects(owner)
    bundled = list_subjects(None)
    shipped = {b.casefold() for b in bundled}
    from backend import prefs
    from backend.services import subjects as subjects_service

    return {"subjects": visible, "bundled": bundled, "own": [s for s in visible if s.casefold() not in shipped],
            "hidden": prefs.hidden_standard(owner), "limits": subjects_service.public_limits()}


def _chats(owner: str) -> dict:
    items = db.list_items("chats", owner, limit=RECENT_CHATS)
    for item in items:
        item.pop("recent_artifact", None)      # same trimming as /api/history/{kind}
    return {"items": items, "persistence": db.is_enabled()}


async def _part(name: str, fn, owner: str, errors: list):
    try:
        with timing.timed(name):
            return await asyncio.to_thread(fn, owner)
    except Exception as exc:  # noqa: BLE001 -- one failing part must never block the boot
        log.warning("bootstrap part %s failed: %s", name, type(exc).__name__)
        errors.append(name)
        return None


@router.get("/bootstrap")
async def bootstrap(request: Request, response: Response) -> dict:
    response.headers["Cache-Control"] = "private, no-store"
    response.headers["Vary"] = "Authorization, X-Guest-Id"
    payload: dict = {"config": config.public_config(), "owner": False}
    try:
        with timing.timed("auth"):
            owner = await asyncio.to_thread(current_owner, request)
    except HTTPException:
        return payload                              # anonymous: config only
    errors: list = []
    subjects, progress, chats = await asyncio.gather(
        _part("subjects", _subjects, owner, errors),
        _part("progress", build_summary, owner, errors),
        _part("chats", _chats, owner, errors),
    )
    payload.update(owner=True, subjects=subjects, progress=progress, chats=chats)
    if errors:
        payload["errors"] = errors
    return payload
