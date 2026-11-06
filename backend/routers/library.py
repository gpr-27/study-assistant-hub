"""Library API: a subject-by-subject overview of the caller's files and everything they did with them.

  GET /api/library                                   summary + one card per visible subject (cached ~60 s)
  GET /api/library/{subject}/timeline?kind=&before=&limit=   grouped-by-time activity chain + model chain

Read-only, owner-scoped (``current_owner``), ``private, no-store``. Served from the per-owner snapshot and a
cached build (see ``backend.services.library``): a warm call costs no MongoDB op, a cold one at most one extra.
A small per-owner limiter (60 calls / minute) guards against a client looping on it.
"""
from __future__ import annotations

import threading
import time
from collections import OrderedDict, deque

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Response
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from backend.auth import current_owner
from backend.services import library as library_service

router = APIRouter(prefix="/api/library", tags=["library"])

_PER_MIN = 60
_hits: "OrderedDict[str, deque]" = OrderedDict()
_hits_lock = threading.Lock()


def _throttle(owner: str) -> None:
    now = time.monotonic()
    with _hits_lock:
        q = _hits.setdefault(owner, deque())
        _hits.move_to_end(owner)
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= _PER_MIN:
            raise HTTPException(status_code=429, detail="Too many requests; try again in a moment.",
                                headers={"Retry-After": "10"})
        q.append(now)
        while len(_hits) > 5000:
            _hits.popitem(last=False)


def _owner(owner: str = Depends(current_owner)) -> str:
    _throttle(owner)
    return owner


# ---- response models (local to this router by design) ----
class ModelUse(BaseModel):
    provider: str
    model: str
    step: int = 0
    count: int


class ModelBlock(BaseModel):
    provider: str
    model: str
    step: int = 0
    label: str | None = None
    skipped: list[str] = []


class TimelineItem(BaseModel):
    id: str
    kind: str
    kind_label: str
    title: str
    summary: str
    ts: str
    item_id: str | None = None
    route: str
    llm: ModelBlock | None = None


class ChainReason(BaseModel):
    reason: str
    count: int


class Chain(BaseModel):
    answers: int
    step1: int
    later_steps: int
    reasons: list[ChainReason] = []


class TimelineResponse(BaseModel):
    subject: str
    items: list[TimelineItem]
    next_cursor: str | None = None
    has_more: bool = False
    counts: dict[str, int] = {}
    models: list[ModelUse] = []
    chain: Chain
    week: dict[str, int] = {}
    truncated: bool = False


class LibraryResponse(BaseModel):
    summary: dict
    subjects: list[dict]
    other: dict | None = None
    persistence: bool = True
    generated_at: str | None = Field(default=None)


@router.get("", response_model=LibraryResponse)
async def get_library(response: Response, refresh: bool = Query(False), owner: str = Depends(_owner)) -> LibraryResponse:
    """Summary across all subjects + one card per subject the caller can see."""
    response.headers["Cache-Control"] = "private, no-store"
    return LibraryResponse(**await run_in_threadpool(library_service.overview, owner, refresh=refresh))


@router.get("/{subject}/timeline", response_model=TimelineResponse)
async def get_timeline(
    response: Response,
    subject: str = Path(..., max_length=200),
    kind: str | None = Query(None, max_length=20),
    before: str | None = Query(None, max_length=120, description="Opaque cursor from a previous page's next_cursor"),
    limit: int = Query(library_service.DEFAULT_TIMELINE_LIMIT, ge=1, le=library_service.MAX_TIMELINE_LIMIT),
    owner: str = Depends(_owner),
) -> TimelineResponse:
    """Newest-first activity for one subject, with the model chain used to answer."""
    response.headers["Cache-Control"] = "private, no-store"
    try:
        data = await run_in_threadpool(library_service.timeline, owner, subject, kind, before, limit)
    except library_service.LibraryError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    return TimelineResponse(**data)
