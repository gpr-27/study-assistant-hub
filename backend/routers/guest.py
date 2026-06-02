"""Anonymous guest accounts API — create a guest, restore a guest session, and upgrade
a guest into a Clerk account.

  POST   /api/guest          -> create a new anonymous guest ("Continue as guest")
  GET    /api/guest/me       -> restore: the live guest record for the X-Guest-Id header
  POST   /api/guest/migrate  -> upgrade: move all guest data to the signed-in Clerk user

Guests are MongoDB-backed, so every route requires persistence to be enabled; without it
the SPA never offers the guest option (config.GUEST_ENABLED is false). A guest's chats,
quizzes, flashcards, plans and mind maps are stored in the SAME per-owner collections as a
Clerk user's (owner="guest_<uuid>"), so they work identically and migrate by a single
owner re-key. The X-Guest-Id header is the guest's bearer credential — anyone holding it is
that anonymous guest, which is the intended model for an account-less session.
"""
from fastapi import APIRouter, HTTPException, Request

from backend import config, db
from backend.auth import GUEST_HEADER, require_clerk_user
from backend.schemas import GuestMigrateResponse, GuestResponse

router = APIRouter(prefix="/api/guest", tags=["guest"])


def _require_guests_enabled() -> None:
    """Guest mode needs MongoDB; fail clearly (503) when it isn't configured."""
    if not config.GUEST_ENABLED or not db.is_enabled():
        raise HTTPException(
            status_code=503,
            detail="Guest mode is unavailable — set MONGODB_URI to enable saved guest sessions.",
        )


def _guest_id(request: Request) -> str:
    """Read and require the X-Guest-Id header (400 if missing)."""
    guest_id = (request.headers.get(GUEST_HEADER) or "").strip()
    if not guest_id:
        raise HTTPException(status_code=400, detail="Missing guest id.")
    return guest_id


@router.post("", response_model=GuestResponse)
@router.post("/", response_model=GuestResponse, include_in_schema=False)
def create_guest() -> GuestResponse:
    """Create a new anonymous guest and log them straight in (the SPA stores guestId)."""
    _require_guests_enabled()
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
    db.touch_guest(guest_id)  # refresh the TTL clock on restore
    return GuestResponse(**guest)


@router.post("/migrate", response_model=GuestMigrateResponse)
def migrate_guest(request: Request) -> GuestMigrateResponse:
    """Upgrade: move all of this guest's data to the signed-in Clerk user, then retire it.

    Requires BOTH a valid Clerk session (the migration target) and an X-Guest-Id (the
    source). Idempotent-ish: an unknown/already-migrated guest returns migrated=false with
    zero counts rather than an error, so a retried upgrade never fails the user.
    """
    if not config.AUTH_ENABLED:
        raise HTTPException(status_code=400, detail="Account upgrade requires Clerk authentication.")
    _require_guests_enabled()
    target_owner = require_clerk_user(request)  # 401 if not signed in
    guest_id = _guest_id(request)
    result = db.migrate_guest(guest_id, target_owner)
    return GuestMigrateResponse(**result)
