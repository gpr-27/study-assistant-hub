"""Resolve the current owner (user) used to scope per-user history.

An ``owner`` is an opaque string; every read/write in ``backend.db`` is scoped by it, so
adding a new kind of identity needs nothing but a new owner string. Three identities
resolve here, in priority order:

  1. Clerk user — when Clerk is ON the SPA sends ``Authorization: Bearer <jwt>``. We verify
     it against Clerk's JWKS (RS256) and use the token's ``sub`` (the Clerk user id). A valid
     Clerk session always wins (so a signed-in user who still has a stale guest id is treated
     as the Clerk user, and migration moves the guest's data across).
  2. Anonymous guest — the SPA sends ``X-Guest-Id: guest_<uuid>``. If that id matches a live
     (non-migrated) guest record, it becomes the owner, and the guest's TTL clock is reset.
  3. Local profile — when Clerk is OFF and no guest id is supplied, everyone is the single
     ``"local"`` profile, exactly as before (fully backward compatible). When Clerk is ON and
     no identity is supplied, a history-scoped route returns 401.

``current_owner`` is used as a FastAPI dependency: ``Depends(current_owner)``.
``require_clerk_user`` is a stricter dependency for the migration route (Clerk user or 401).
"""
from __future__ import annotations

import base64
import logging
from functools import lru_cache

from fastapi import HTTPException, Request

from backend import config, db

log = logging.getLogger("backend.auth")

LOCAL_OWNER = "local"
# Header the SPA uses to present an anonymous guest identity (HTTP headers are
# case-insensitive; this is the canonical spelling).
GUEST_HEADER = "X-Guest-Id"


def _frontend_api_from_pk(pk: str) -> str | None:
    """Derive the Clerk Frontend API host from a publishable key.

    A Clerk publishable key is ``pk_<test|live>_<base64>`` where the base64 segment
    decodes to ``<frontend-api-host>$`` (e.g. ``clerk.example.com$``).
    """
    try:
        b64 = pk.split("_", 2)[2]
        decoded = base64.b64decode(b64 + "==").decode("utf-8")
        return decoded.rstrip("$") or None
    except Exception:  # noqa: BLE001 — a malformed key simply yields no host
        return None


@lru_cache(maxsize=1)
def _jwks_client():
    """A cached PyJWKClient pointing at this Clerk instance's JWKS endpoint."""
    import jwt

    host = _frontend_api_from_pk(config.CLERK_PUBLISHABLE_KEY or "")
    if not host:
        return None
    return jwt.PyJWKClient(f"https://{host}/.well-known/jwks.json")


def _verify_clerk_token(token: str) -> str | None:
    """Verify a Clerk session JWT and return its ``sub`` (user id), or None."""
    import jwt

    client = _jwks_client()
    if client is None:
        return None
    try:
        signing_key = client.get_signing_key_from_jwt(token)
        claims = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            options={"verify_aud": False},  # Clerk session tokens have no fixed audience
        )
        return claims.get("sub")
    except Exception as exc:  # noqa: BLE001 — any verification failure is an auth failure
        log.warning("Clerk token verification failed: %s", exc)
        return None


def _bearer_token(request: Request) -> str:
    """Extract the ``Authorization: Bearer <token>`` value (empty string if absent)."""
    header = request.headers.get("Authorization", "")
    return header[7:].strip() if header[:7].lower() == "bearer " else ""


def _clerk_sub(request: Request) -> str | None:
    """The verified Clerk user id for this request, or None. Only meaningful when AUTH_ENABLED."""
    if not config.AUTH_ENABLED:
        return None
    token = _bearer_token(request)
    return _verify_clerk_token(token) if token else None


def _guest_owner(request: Request) -> str | None:
    """A live guest id from the ``X-Guest-Id`` header (and reset its TTL), or None.

    ``db.touch_guest`` both validates the id (must match a non-migrated guest) and refreshes
    ``lastActiveAt`` — one round-trip that doubles as the spec's "update lastActiveAt whenever
    a guest uses the application". A no-op (returns False) when persistence is disabled.
    """
    guest_id = (request.headers.get(GUEST_HEADER) or "").strip()
    if guest_id and db.touch_guest(guest_id):
        return guest_id
    return None


def current_owner(request: Request) -> str:
    """Return the owner id for the request (FastAPI dependency).

    Priority: verified Clerk user → live guest → local. Raises 401 only when Clerk auth is
    enabled and neither a valid Clerk session nor a valid guest id is presented.
    """
    sub = _clerk_sub(request)
    if sub:
        return sub
    guest = _guest_owner(request)
    if guest:
        return guest
    if config.AUTH_ENABLED:
        raise HTTPException(status_code=401, detail="Sign in or continue as a guest to access your data.")
    return LOCAL_OWNER


def require_clerk_user(request: Request) -> str:
    """Return the verified Clerk user id, or raise 401. Used by the guest-migration route,
    which must move data INTO a real account (a guest id is never an acceptable target)."""
    sub = _clerk_sub(request)
    if not sub:
        raise HTTPException(status_code=401, detail="A signed-in account is required for this action.")
    return sub
