"""Storage accounting + guard for the MongoDB cluster (Atlas M0 = 512 MB, shared by everything).

* :func:`cluster_usage` -- ``dbStats`` (data + index bytes), cached ``STORAGE_CACHE_S`` (5 min) so it
  costs one command per instance per 5 minutes, never one per request.
* :func:`owner_bytes` -- what one owner has stored (uploaded PDFs + index zips, from GridFS ``length``),
  cached for a minute.
* :func:`check_capacity` -- refuse a new upload (HTTP 507 + a clear message) when the cluster is at
  ``STORAGE_REFUSE_PCT`` of ``STORAGE_CAP_MB`` once the new bytes are counted.

Everything fails OPEN: if the numbers cannot be read (no MongoDB, permission denied) uploads proceed.
"""
from __future__ import annotations

import logging
import os
import time

from backend import db

log = logging.getLogger(__name__)

CAP_BYTES = int(float(os.environ.get("STORAGE_CAP_MB", "") or 512) * 1024 * 1024)
REFUSE_PCT = float(os.environ.get("STORAGE_REFUSE_PCT", "") or 90)
CACHE_S = float(os.environ.get("STORAGE_CACHE_S", "") or 300)
_cluster: tuple[float, int | None] = (0.0, None)
_owners: dict[str, tuple[float, int]] = {}


def cluster_usage(*, force: bool = False) -> int | None:
    """Bytes used by the app database (data + indexes), or None when unknown."""
    global _cluster
    now = time.monotonic()
    if not force and _cluster[1] is not None and now - _cluster[0] < CACHE_S:
        return _cluster[1]
    if not force and _cluster[0] and _cluster[1] is None and now - _cluster[0] < 60:
        return None
    database = db._get_db() if db.is_enabled() else None
    used: int | None = None
    if database is not None:
        try:
            st = database.command("dbStats")
            used = int(st.get("storageSize", 0)) + int(st.get("indexSize", 0)) or int(st.get("dataSize", 0))
        except Exception:  # noqa: BLE001
            log.debug("dbStats unavailable", exc_info=True)
    _cluster = (now, used)
    return used


def owner_bytes(owner: str) -> int:
    hit = _owners.get(owner)
    now = time.monotonic()
    if hit and now - hit[0] < 60:
        return hit[1]
    database = db._get_db() if db.is_enabled() else None
    total = 0
    if database is not None and owner:
        for bucket in (db.UPLOADS_BUCKET, db.INDEX_BUCKET):
            try:
                rows = database[f"{bucket}.files"].aggregate(
                    [{"$match": {"owner": owner}}, {"$group": {"_id": None, "n": {"$sum": "$length"}}}])
                total += sum(int(r.get("n", 0)) for r in rows)
            except Exception:  # noqa: BLE001
                log.debug("owner usage unavailable", exc_info=True)
    if len(_owners) > 512:
        _owners.clear()
    _owners[owner] = (now, total)
    return total


def forget_owner(owner: str) -> None:
    _owners.pop(owner, None)


def near_cap() -> bool:
    used = cluster_usage()
    return used is not None and used >= CAP_BYTES * REFUSE_PCT / 100.0


def check_capacity(extra_bytes: int = 0) -> None:
    """Raise ``SubjectError(507)`` when storing ``extra_bytes`` more would push the cluster past the guard."""
    from backend.services.subjects import SubjectError

    used = cluster_usage()
    if used is None:
        return
    if used + max(0, extra_bytes) > CAP_BYTES * REFUSE_PCT / 100.0:
        raise SubjectError(
            "The study library is almost full (the shared database has about "
            f"{max(0, CAP_BYTES - used) // 1048576} MB left), so this upload was not accepted. "
            "Delete files or subjects you no longer need and try again.", 507)


def public_usage(owner: str) -> dict:
    return {"owner_mb": round(owner_bytes(owner) / 1048576, 1), "near_cap": near_cap()}
