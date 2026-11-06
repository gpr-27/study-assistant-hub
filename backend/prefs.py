"""Tiny per-owner preferences: the standard (bundled) subjects an owner chose to hide.

Hiding is a *view* preference. It never deletes the shared bundled files / indexes and never
touches the owner's history (chats, quizzes, ...) about that subject; it only removes the
subject from this owner's lists and makes direct retrieval for it answer "unknown subject".

Storage: one small document per owner in ``owner_prefs`` (``_id`` = owner id, guests included):
``{"_id": owner, "hidden_standard": ["Mathematics"], "updated_at": ...}``. Reads are served from a
60 s per-instance cache plus the shared L2 cache (keyed to the owner epoch), so listing subjects
costs no extra MongoDB round trip on the hot path; every change bumps the owner epoch.
Without MongoDB (local dev / tests) the process-local dict is the store.
"""
from __future__ import annotations

import logging
import threading
import time

from backend import cache, db

log = logging.getLogger(__name__)

COLLECTION = "owner_prefs"
_TTL_S = 60.0
_MAX = 2048
_mem: dict[str, tuple[float, tuple[str, ...]]] = {}
_local: dict[str, tuple[str, ...]] = {}          # authoritative store when MongoDB is off
_lock = threading.Lock()


def _norm(names) -> tuple[str, ...]:
    seen: dict[str, str] = {}
    for n in names or []:
        if isinstance(n, str) and n.strip():
            seen.setdefault(n.strip().casefold(), n.strip())
    return tuple(sorted(seen.values(), key=str.casefold))


def _coll():
    database = db._get_db() if db.is_enabled() else None
    return database[COLLECTION] if database is not None else None


def hidden_standard(owner: str | None) -> list[str]:
    """Names of bundled subjects ``owner`` has hidden (cheap; cached)."""
    if not owner:
        return []
    now = time.monotonic()
    with _lock:
        hit = _mem.get(owner)
        if hit and now - hit[0] < _TTL_S:
            return list(hit[1])
    coll = _coll()
    if coll is None:
        return list(_local.get(owner, ()))
    cached, ep = cache.l2_get("hide", owner)
    if isinstance(cached, list):
        names = _norm(cached)
    else:
        try:
            doc = coll.find_one({"_id": owner}, {"hidden_standard": 1})
            names = _norm((doc or {}).get("hidden_standard"))
        except Exception:  # noqa: BLE001 - never fail a listing because prefs are unreadable
            log.warning("Could not read subject preferences", exc_info=True)
            return list(_local.get(owner, ()))
        cache.l2_put("hide", owner, "", list(names), ep, 300.0)
    with _lock:
        if len(_mem) >= _MAX:
            _mem.clear()
        _mem[owner] = (now, names)
    return list(names)


def is_hidden(owner: str | None, subject: str) -> bool:
    if not owner or not subject:
        return False
    low = subject.casefold()
    return any(h.casefold() == low for h in hidden_standard(owner))


def set_hidden(owner: str, names) -> list[str]:
    """Replace the owner's hidden list; returns the stored (normalised) list."""
    clean = _norm(names)
    coll = _coll()
    if coll is not None:
        try:
            if clean:
                from datetime import datetime, timezone

                coll.update_one({"_id": owner}, {"$set": {"hidden_standard": list(clean),
                                                          "updated_at": datetime.now(timezone.utc)}}, upsert=True)
            else:
                coll.delete_one({"_id": owner})
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not save subject preferences: %s", exc)
            raise
    _local[owner] = clean
    with _lock:
        _mem[owner] = (time.monotonic(), clean)
    cache.bump_epoch(owner)
    return list(clean)


def hide(owner: str, name: str) -> list[str]:
    return set_hidden(owner, [*hidden_standard(owner), name])


def restore(owner: str, name: str | None = None) -> list[str]:
    cur = hidden_standard(owner)
    if name is None:
        return set_hidden(owner, [])
    low = name.casefold()
    return set_hidden(owner, [h for h in cur if h.casefold() != low])


def merge(src: str, dst: str) -> int:
    """Guest -> account: union the hidden lists, then drop the guest's record. Returns names added."""
    if not src or not dst or src == dst:
        return 0
    mine = hidden_standard(src)
    if not mine:
        return 0
    before = hidden_standard(dst)
    merged = set_hidden(dst, [*before, *mine])
    coll = _coll()
    try:
        if coll is not None:
            coll.delete_one({"_id": src})
    except Exception:  # noqa: BLE001
        log.debug("Could not drop the guest's preferences", exc_info=True)
    _local.pop(src, None)
    with _lock:
        _mem.pop(src, None)
    return len(merged) - len(before)


def reset_for_tests() -> None:
    with _lock:
        _mem.clear()
    _local.clear()
