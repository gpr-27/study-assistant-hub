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
import hashlib
import logging
import threading
import time
from functools import lru_cache

from fastapi import HTTPException, Request

from backend import cache, config, db, spend, timing

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


# JWKS handling for serverless: ONE client per instance, keys cached for an hour (and per
# key-id for the process lifetime), a short network timeout, and a negative cache so an
# unreachable Clerk can never cost every request a network wait.
_JWKS_LIFESPAN_S = 3600
_JWKS_TIMEOUT_S = 3
_JWKS_FAIL_BACKOFF_S = 30.0
# A token naming a key id we don't know may mean Clerk rotated keys, so we refetch the JWK set
# -- but at most once per this window, otherwise junk tokens with random ``kid``s would each
# cost a network round-trip to Clerk (amplification / rate-limit risk). Enforced here rather
# than relying on PyJWT's own cooldown, which only exists in newer 2.x releases.
_JWKS_UNKNOWN_KID_COOLDOWN_S = 30.0
_jwks_failed_until = 0.0
_jwks_last_forced_refresh: float | None = None
_jwks_lock = threading.Lock()
# Verification outcomes for recently seen tokens: {token_hash: (expires_monotonic, sub)}.
# Tokens are ~60 s lived and the SPA re-sends the same one on every call, so this removes the
# repeated RSA verify; bounded and never outlives the token's own ``exp``.
_TOKEN_CACHE_MAX = 512
_token_cache: dict[str, tuple[float, str]] = {}


@lru_cache(maxsize=1)
def _jwks_client():
    """A cached PyJWKClient pointing at this Clerk instance's JWKS endpoint."""
    import jwt

    host = _frontend_api_from_pk(config.CLERK_PUBLISHABLE_KEY or "")
    if not host:
        return None
    return jwt.PyJWKClient(
        f"https://{host}/.well-known/jwks.json",
        cache_keys=True,                 # per-kid lru_cache: known keys never trigger a fetch
        cache_jwk_set=True,
        lifespan=_JWKS_LIFESPAN_S,       # JWK-set refetch at most hourly
        timeout=_JWKS_TIMEOUT_S,
    )


class ClerkUnavailable(Exception):
    """Clerk's JWKS could not be fetched, so a presented token can be neither accepted nor
    rejected. Surfaces as a retryable 503 (not a 401, which the SPA reads as "signed out")."""


def _issuer() -> str | None:
    host = _frontend_api_from_pk(config.CLERK_PUBLISHABLE_KEY or "")
    return f"https://{host}" if host else None


def _signing_key(client, token: str):
    """The JWK matching ``token``'s ``kid``. Known keys come from the cached JWK set (no
    network); an unknown ``kid`` forces at most one refetch per cooldown window."""
    global _jwks_last_forced_refresh
    import jwt

    kid = jwt.get_unverified_header(token).get("kid")
    if not kid:
        raise jwt.InvalidTokenError("token has no key id")
    for key in client.get_signing_keys():
        if key.key_id == kid:
            return key
    now = time.monotonic()
    with _jwks_lock:
        last = _jwks_last_forced_refresh
        if last is not None and now - last < _JWKS_UNKNOWN_KID_COOLDOWN_S:
            raise jwt.InvalidTokenError("unknown signing key")
        _jwks_last_forced_refresh = now
    for key in client.get_signing_keys(refresh=True):
        if key.key_id == kid:
            return key
    raise jwt.InvalidTokenError("unknown signing key")


def _verify_clerk_token(token: str) -> str | None:
    """Verify a Clerk session JWT and return its ``sub`` (user id), or None.

    Checks: RS256 signature against this instance's JWKS, ``exp`` (10 s leeway), ``nbf``,
    ``iss`` == ``https://<frontend-api-host>``, a string ``sub``, and an *active* session
    (``sts`` != "pending"). Clerk session tokens carry no audience, so ``aud`` is not checked.
    Raises ``ClerkUnavailable`` when the JWKS is unreachable (and the token isn't cached).
    """
    global _jwks_failed_until
    import jwt

    client = _jwks_client()
    if client is None:
        return None
    key = hashlib.sha256(token.encode("utf-8", "ignore")).hexdigest()
    now_m = time.monotonic()
    hit = _token_cache.get(key)
    if hit is not None:
        if hit[0] > now_m:
            return hit[1]
        _token_cache.pop(key, None)
    if now_m < _jwks_failed_until:
        # Clerk's JWKS was unreachable moments ago. Another instance may have verified this very token
        # (shared cache, only consulted in this degraded state): accept its verdict, else fail fast.
        raw = cache.get(cache.key("tok", None, key)) if cache.enabled() else None
        if raw:
            until, _, sub = raw.partition("|")
            try:
                if float(until) > time.time() and sub:
                    return sub
            except ValueError:
                pass
        raise ClerkUnavailable("Clerk JWKS unreachable (backing off)")
    t0 = time.perf_counter()
    try:
        signing_key = _signing_key(client, token)
        claims = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            leeway=10,                                  # tolerate small clock skew
            issuer=_issuer(),
            options={"verify_aud": False, "require": ["exp", "sub", "iss"]},  # session tokens have no fixed audience
        )
        sub = claims.get("sub")
        if claims.get("sts") == "pending":              # sign-in not finished (e.g. MFA / required task)
            return None
        if isinstance(sub, str) and sub:
            exp = float(claims.get("exp", 0))
            ttl = min(max(exp - time.time(), 0.0), 30.0)  # never cache past exp; 30 s max
            if ttl > 1.0:
                if len(_token_cache) >= _TOKEN_CACHE_MAX:
                    _token_cache.clear()
                _token_cache[key] = (now_m + ttl, sub)
                if cache.enabled():       # share the verdict (token hash -> sub, never past exp) in the background
                    shared_val = f"{time.time() + ttl:.1f}|{sub}"
                    cache.background(lambda: cache.set(cache.key("tok", None, key), shared_val, ttl))
            return sub
        return None
    except jwt.PyJWKClientConnectionError as exc:
        _jwks_failed_until = time.monotonic() + _JWKS_FAIL_BACKOFF_S
        log.warning("Clerk JWKS fetch failed (backing off %ds): %s", _JWKS_FAIL_BACKOFF_S, exc)
        raise ClerkUnavailable("Clerk JWKS unreachable") from exc
    except Exception as exc:  # noqa: BLE001 — any verification failure is an auth failure
        log.warning("Clerk token verification failed: %s", exc)
        return None
    finally:
        timing.add("auth", (time.perf_counter() - t0) * 1000.0)


def _bearer_token(request: Request) -> str:
    """Extract the ``Authorization: Bearer <token>`` value (empty string if absent)."""
    header = request.headers.get("Authorization", "")
    return header[7:].strip() if header[:7].lower() == "bearer " else ""


def _clerk_sub(request: Request) -> str | None:
    """The verified Clerk user id for this request, or None. Only meaningful when AUTH_ENABLED."""
    if not config.AUTH_ENABLED:
        return None
    token = _bearer_token(request)
    if not token:
        return None
    try:
        return _verify_clerk_token(token)
    except ClerkUnavailable as exc:
        # A bearer token WAS presented but Clerk's keys are unreachable: say so (retryable) rather
        # than 401 (SPA treats that as "signed out") or silently downgrading to a stale guest id.
        raise HTTPException(
            status_code=503, detail="Sign-in is temporarily unavailable. Please retry in a moment.",
            headers={"Retry-After": "5"},
        ) from exc


def _guest_owner(request: Request) -> str | None:
    """A live guest id from the ``X-Guest-Id`` header (and reset its TTL), or None.

    ``db.touch_guest`` both validates the id (must match a non-migrated guest) and refreshes
    ``lastActiveAt`` — one round-trip that doubles as the spec's "update lastActiveAt whenever
    a guest uses the application". A no-op (returns False) when persistence is disabled.
    """
    guest_id = (request.headers.get(GUEST_HEADER) or "").strip()
    if not guest_id or len(guest_id) > 64:  # real ids are "guest_" + 32 hex chars
        return None
    with timing.timed("db"):
        return guest_id if db.touch_guest(guest_id) else None


def current_owner(request: Request) -> str:
    """Return the owner id for the request (FastAPI dependency).

    Priority: verified Clerk user → live guest → local. Raises 401 only when Clerk auth is
    enabled and neither a valid Clerk session nor a valid guest id is presented.
    """
    if config.auth_misconfigured():
        # Production without Clerk would collapse every visitor into the shared "local" owner:
        # fail CLOSED with a clear error instead (also flagged by /api/health/deep).
        raise HTTPException(status_code=503, detail=config.AUTH_MISCONFIGURED_MESSAGE)
    sub = _clerk_sub(request)
    if sub:
        spend.set_owner(sub)           # attribute this request's paid-LLM tokens (backend/spend.py)
        return sub
    guest = _guest_owner(request)
    if guest:
        spend.set_owner(guest)
        return guest
    if config.AUTH_ENABLED:
        raise HTTPException(status_code=401, detail="Sign in or continue as a guest to access your data.")
    spend.set_owner(LOCAL_OWNER)
    return LOCAL_OWNER


def require_clerk_user(request: Request) -> str:
    """Return the verified Clerk user id, or raise 401. Used by the guest-migration route,
    which must move data INTO a real account (a guest id is never an acceptable target)."""
    sub = _clerk_sub(request)
    if not sub:
        raise HTTPException(status_code=401, detail="A signed-in account is required for this action.")
    return sub
