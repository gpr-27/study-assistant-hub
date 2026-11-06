"""Anonymous guest accounts API — create a guest, restore a guest session, and upgrade
a guest into a Clerk account.

  POST   /api/guest          -> create a new anonymous guest ("Continue as guest")
  GET    /api/guest/me       -> restore: the live guest record for the X-Guest-Id header
  POST   /api/guest/migrate  -> upgrade: move all guest data to the signed-in Clerk user

Guests are MongoDB-backed, so every route requires persistence to be enabled; without it
the SPA never offers the guest option (config.GUEST_ENABLED is false). A guest's chats,
quizzes, flashcards, plans and mind maps are stored in the SAME per-owner collections as a
Clerk user's (owner="guest_<uuid>"), so they work identically and migrate by a single
owner re-key; uploaded subjects (GridFS) are re-keyed by ``subjects.migrate_guest_uploads``.

Security model: the ``X-Guest-Id`` header is the guest's bearer credential — a 122-bit random
uuid4 (unguessable); anyone holding it *is* that anonymous guest, which is the intended model
for an account-less session. Ids are format-checked here (``guest_`` + short alphanumerics) so
no other string (a Clerk id, "local", an internal key, a query operator) can pass as a guest.
A guest id can never be a migration *target* (``require_clerk_user``).
"""
import re
import threading
import time
from collections import deque

from fastapi import APIRouter, HTTPException, Request

from backend import config, db
from backend.auth import GUEST_HEADER, require_clerk_user
from backend.schemas import GuestMigrateResponse, GuestResponse
from backend.services import subjects as subjects_service

router = APIRouter(prefix="/api/guest", tags=["guest"])

_GUEST_ID_RE = re.compile(r"^" + re.escape(db.GUEST_PREFIX) + r"[A-Za-z0-9]{1,48}$")

# Per-instance, per-IP creation throttle. Each POST /api/guest inserts a MongoDB document, so an
# unauthenticated loop could bloat the DB until the 30-day TTL reaps it. This is only a cheap
# first line (instances don't share state) — the Vercel Firewall rate limit is the real control.
_CREATE_LIMIT = 15
_CREATE_WINDOW_S = 600.0
_CREATE_TRACK_MAX = 2000
_create_hits: dict[str, deque] = {}
_create_lock = threading.Lock()

# Serialises migrations on this instance (double-click / two tabs): the second sees the guest gone.
_migrate_lock = threading.Lock()


def _client_ip(request: Request) -> str:
    # Vercel overwrites x-forwarded-for / sets x-real-ip, so these are not client-spoofable there.
    fwd = request.headers.get("x-real-ip") or (request.headers.get("x-forwarded-for") or "").split(",")[0]
    return (fwd or (request.client.host if request.client else "") or "unknown").strip()[:64]


def _allow_create(ip: str) -> bool:
    now = time.monotonic()
    with _create_lock:
        if len(_create_hits) >= _CREATE_TRACK_MAX and ip not in _create_hits:
            for k in [k for k, q in _create_hits.items() if not q or now - q[-1] > _CREATE_WINDOW_S]:
                del _create_hits[k]
            if len(_create_hits) >= _CREATE_TRACK_MAX:
                _create_hits.clear()          # bounded: worst case the throttle resets
        q = _create_hits.setdefault(ip, deque())
        while q and now - q[0] > _CREATE_WINDOW_S:
            q.popleft()
        if len(q) >= _CREATE_LIMIT:
            return False
        q.append(now)
        return True


def _reset_create_throttle() -> None:  # test hook
    with _create_lock:
        _create_hits.clear()


def _require_guests_enabled() -> None:
    """Guest mode needs MongoDB; fail clearly (503) when it isn't configured."""
    if not config.GUEST_ENABLED or not db.is_enabled():
        raise HTTPException(
            status_code=503,
            detail="Guest mode is unavailable — set MONGODB_URI to enable saved guest sessions.",
        )


def _guest_id(request: Request) -> str:
    """Read the X-Guest-Id header: 400 if missing, 404 if it isn't shaped like a guest id."""
    guest_id = (request.headers.get(GUEST_HEADER) or "").strip()
    if not guest_id:
        raise HTTPException(status_code=400, detail="Missing guest id.")
    if not _GUEST_ID_RE.match(guest_id):
        raise HTTPException(status_code=404, detail="Guest session not found or expired.")
    return guest_id


@router.post("", response_model=GuestResponse)
@router.post("/", response_model=GuestResponse, include_in_schema=False)
def create_guest(request: Request) -> GuestResponse:
    """Create a new anonymous guest and log them straight in (the SPA stores guestId)."""
    _require_guests_enabled()
    if not _allow_create(_client_ip(request)):
        raise HTTPException(status_code=429, detail="Too many guest sessions started; try again later.")
    guest = db.create_guest()
    if not guest:
        raise HTTPException(status_code=503, detail="Could not start a guest session right now.")
    return GuestResponse(**guest)


@router.get("/me", response_model=GuestResponse)
def restore_guest(request: Request) -> GuestResponse:
    """Restore a guest session from a stored guestId (used on app startup / refresh)."""
    _require_guests_enabled()
    guest_id = _guest_id(request)
    guest = db.get_guest(guest_id)
    if not guest:
        raise HTTPException(status_code=404, detail="Guest session not found or expired.")
    db.touch_guest(guest_id)  # refresh the TTL clock on restore (in-process cached for 60 s)
    return GuestResponse(**guest)


@router.post("/migrate", response_model=GuestMigrateResponse)
def migrate_guest(request: Request) -> GuestMigrateResponse:
    """Upgrade: move all of this guest's data to the signed-in Clerk user, then retire it.

    Requires BOTH a valid Clerk session (the migration target) and an X-Guest-Id (the
    source). Idempotent-ish: an unknown/already-migrated guest returns migrated=false with
    zero counts rather than an error, so a retried upgrade never fails the user.

    Order (all inside ``db.migrate_guest``): atomic claim of the guest, then uploaded subjects
    (re-owned in GridFS), then the artifacts re-key, with the guest record deleted last — an
    interruption at any point leaves everything reachable and a retry resumes.
    """
    if not config.AUTH_ENABLED:
        raise HTTPException(status_code=400, detail="Account upgrade requires Clerk authentication.")
    _require_guests_enabled()
    target_owner = require_clerk_user(request)  # 401 if not signed in
    guest_id = _guest_id(request)
    with _migrate_lock:
        # Uploads are re-owned by db.migrate_guest itself, AFTER its atomic claim of the guest.
        result = db.migrate_guest(
            guest_id, target_owner,
            after_claim=lambda: subjects_service.migrate_guest_uploads(guest_id, target_owner))
    subjects_moved = int(result.pop("subjects_moved", 0) or 0)
    if subjects_moved:
        result = dict(result)
        result["moved"] = {**result.get("moved", {}), "subjects": subjects_moved}
    return GuestMigrateResponse(**result)
