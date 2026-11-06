"""Optional MongoDB persistence — per-user history for every generated artifact.

Active only when ``config.MONGODB_URI`` is set; otherwise every function is a
graceful no-op so the app runs identically without a database. EVERY read and write
is scoped by ``owner`` (the signed-in Clerk user id, or "local" when auth is off),
so users never see each other's history.

Artifact kinds: chats, quizzes, flashcards, plans, mindmaps, summaries, exams,
teachbacks, predictions, attempts (progress events), cheatsheets, scans.
"""
from __future__ import annotations

import copy
import logging
import os
import re
import threading
import time
from collections import OrderedDict
from datetime import datetime, timezone
from uuid import uuid4

from backend import cache, config

log = logging.getLogger("backend.db")


def redact_id(value: str | None) -> str:
    """Short, non-reversible tag for logging an owner / guest id.

    A guest id is a bearer credential (whoever presents it IS that guest), so it must never reach
    a log line; this keeps lines correlatable (same id -> same tag) without disclosing it.
    """
    import hashlib

    if not value:
        return "-"
    return "id#" + hashlib.sha256(value.encode("utf-8", "ignore")).hexdigest()[:8]

KINDS: tuple[str, ...] = (
    "chats", "quizzes", "flashcards", "plans", "mindmaps", "summaries",
    "exams", "teachbacks", "predictions", "attempts", "cheatsheets", "scans",
)

# Anonymous guest accounts live in their own collection. A guest's *artifacts* are NOT
# embedded here — they live in the KINDS collections above under owner="guest_<uuid>",
# exactly like a Clerk user's, so upgrading a guest is a single owner re-key (migrate_guest)
# and there are no unbounded arrays inside a guest document.
GUESTS: str = "guests"
GUEST_PREFIX: str = "guest_"

# Big fields excluded from list summaries (the detail endpoint returns them).
_HEAVY_FIELDS = (
    "questions", "cards", "messages", "root", "plan", "summaries",
    "patterns", "predicted_questions", "markdown", "per_question", "topic_breakdown",
    "recent_artifact", "result", "srs",
)
# Text fields searched by the ?q= filter (missing fields simply don't match).
_SEARCH_FIELDS = ("title", "source", "label", "topic", "subject", "messages.content")

_client = None
_db = None
_init_failed = False
# Serializes the lazy client creation (FastAPI sync routes run in a threadpool, so two
# requests can race on a cold instance and would otherwise each build a MongoClient).
_init_lock = threading.Lock()
_indexes_kicked = False

# Serverless-safe client settings. ONE MongoClient per function instance (module global); it is
# created lazily on first use and performs NO network I/O at construction (pymongo connects on
# the first operation), so a cold start that never touches the DB pays nothing. Keep the pool
# small: Atlas free/shared tiers cap total connections and many instances may run at once.
_POOL_KWARGS = dict(
    maxPoolSize=int(os.environ.get("MONGO_MAX_POOL", "10") or 10),
    minPoolSize=int(os.environ.get("MONGO_MIN_POOL", "1") or 0),   # keep ONE live connection warm (see note below)
    maxIdleTimeMS=45_000,          # drop connections that sat idle (stale after instance freeze)
    serverSelectionTimeoutMS=3000, # fail fast -> a clean 503 instead of a hung request
    connectTimeoutMS=3000,
    socketTimeoutMS=20_000,
    waitQueueTimeoutMS=10_000,     # queue behind a busy pool instead of failing a burst with a 503
    retryWrites=True,
    retryReads=True,
    appname="study-assistant-hub",
)
# minPoolSize=1: an instance that already served a request keeps one authenticated connection, so the
# next request after a quiet spell skips the TCP+TLS+auth handshake (~400-800 ms to Atlas). It costs at most
# ONE extra connection per live instance (the pool would hold >=1 idle connection for maxIdleTimeMS anyway),
# i.e. N instances => N (+2 monitor sockets each) of the M0 500-connection cap; maxPoolSize stays 10. A
# frozen instance's socket may be stale on thaw, which retryReads/retryWrites absorb. Set MONGO_MIN_POOL=0
# to disable.
#
# Cold-start prewarm: see ``prewarm()`` below.
_prewarm_thread: "threading.Thread | None" = None
_prewarm_done = threading.Event()
_prewarm_started = False

# Bump when the set of indexes below changes; stored in `_meta` so ensure_indexes() is a single
# cheap read on every later cold start instead of ~15 create_index round-trips.
_INDEX_SCHEMA = "idx-v6"

# Owner-keyed, short-lived collections written by the chat / teach-back services (NOT history
# kinds: they never show up in /api/history). Each has a TTL on `created_at`; they are still
# re-keyed on guest migration and swept with the orphan-guest purge.
EPHEMERAL_TTLS: dict[str, int] = {
    "chat_pdfs": 7 * 86400,          # extracted chunks of an uploaded-PDF chat
    "teachback_sessions": 3 * 3600,  # must equal services/teachback._SESSION_TTL_S
    # Chunked PDF uploads (services/uploads.py). `created_at` is refreshed on every chunk, so the
    # TTL is "1 h after the last activity"; must equal services.uploads.UPLOAD_TTL_S.
    "upload_sessions": 3600,
    "upload_chunks": 3600,
}


def is_enabled() -> bool:
    """True when persistence is configured and the (lazy) client could be built. No I/O."""
    return config.PERSISTENCE_ENABLED and not _init_failed and _get_db() is not None


def ping() -> None:
    """Round-trip to MongoDB (raises on failure). Used only by /api/health/deep."""
    database = _get_db()
    if database is None:
        raise RuntimeError("persistence disabled")
    database.client.admin.command("ping")


def _get_db():
    """Lazily build the pooled Mongo client. Returns None if disabled/failed.

    Cheap and non-blocking: constructing a MongoClient does not touch the network, and
    index creation is handed to a one-shot fire-and-forget thread (see ``_kick_index_ensure``)
    rather than running inline before the first request. Double-checked locking guarantees
    exactly one client per process.
    """
    global _client, _db, _init_failed
    if not config.PERSISTENCE_ENABLED or _init_failed:
        return None
    if _prewarm_started and not _prewarm_done.is_set() and threading.current_thread() is not _prewarm_thread:
        # The cold-start prewarm is mid-connect (SRV + handshake + auth): wait for it instead of racing it
        # with a second connection attempt. Bounded by the server-selection timeout; later calls skip this.
        _tw = time.perf_counter()
        _prewarm_done.wait(_POOL_KWARGS["serverSelectionTimeoutMS"] / 1000.0)
        from backend import timing
        timing.add("mongo_wait", (time.perf_counter() - _tw) * 1000.0)
    if _db is not None:
        return _db
    with _init_lock:
        if _db is not None or _init_failed:
            return _db
        try:
            from pymongo import MongoClient

            _client = MongoClient(config.MONGODB_URI, **_POOL_KWARGS)
            _db = _client[config.MONGODB_DB_NAME]
            log.info("MongoDB client ready (lazy, pool<=%d, db=%s).", _POOL_KWARGS["maxPoolSize"], config.MONGODB_DB_NAME)
        except Exception as exc:  # noqa: BLE001 — a bad URI disables persistence gracefully
            _init_failed = True
            log.error("MongoDB client init failed; persistence disabled for this run: %s", exc)
            return None
    _kick_index_ensure()
    return _db


def prewarm() -> bool:
    """Cold-start: build the Mongo client (pre-resolves the SRV record) and run an authenticated no-op
    (``hello``) in ONE daemon thread, so the first real request finds a connected, authenticated pool
    instead of paying the handshake itself. Never blocks the caller; idempotent (a second call returns
    False); a no-op when persistence is disabled. Requests that arrive mid-connect wait for it
    (see ``_get_db``) rather than opening a second connection. Index ensuring (version-marked) is kicked
    from ``_get_db`` into its own thread, so it never sits on a request path."""
    global _prewarm_thread, _prewarm_started
    if not config.PERSISTENCE_ENABLED:
        return False
    with _init_lock:
        if _prewarm_started:
            return False
        _prewarm_started = True

    def _run() -> None:
        t0 = time.perf_counter()
        try:
            database = _get_db()
            if database is not None:
                database.client.admin.command("hello")
                log.info("Mongo prewarm ready in %d ms.", (time.perf_counter() - t0) * 1000)
        except Exception as exc:  # noqa: BLE001 -- the first real request retries and reports properly
            log.info("Mongo prewarm did not complete: %s", exc)
        finally:
            _prewarm_done.set()

    _prewarm_thread = threading.Thread(target=_run, daemon=True, name="db-prewarm")
    _prewarm_thread.start()
    return True


def _kick_index_ensure() -> None:
    """Start ONE short-lived daemon thread per process that ensures indexes exist.

    Fire-and-forget by design: requests never wait on it, a failure only logs, and the
    daily cron (``ensure_indexes`` via /api/cron/janitor) is the guaranteed backstop.
    """
    global _indexes_kicked
    if _indexes_kicked:
        return
    _indexes_kicked = True

    def _run() -> None:
        try:
            ensure_indexes()
        except Exception as exc:  # noqa: BLE001
            log.warning("Background index ensure failed (cron will retry): %s", exc)

    threading.Thread(target=_run, daemon=True, name="db-ensure-indexes").start()


def ensure_indexes(force: bool = False) -> dict:
    """Create the app's indexes once (idempotent). Returns ``{"skipped"|"created": ...}``.

    A version marker document in ``_meta`` makes every call after the first a single
    ``find_one``. The marker also encodes the guest TTL, so changing GUEST_TTL_DAYS re-runs
    the (collMod-reconciling) guest index step.
    """
    database = _get_db()
    if database is None:
        return {"skipped": "persistence disabled"}
    from pymongo import ASCENDING, DESCENDING

    want = f"{_INDEX_SCHEMA}:ttl{config.GUEST_TTL_SECONDS}"
    meta = database["_meta"]
    if not force:
        marker = meta.find_one({"_id": "indexes"})
        if marker and marker.get("version") == want:
            return {"skipped": "up to date", "version": want}
    for kind in KINDS:
        # owner + created_at(desc): serves list_items (owner filter + newest-first sort),
        # clear_kind and the orphan sweep's `distinct("owner")` prefix scan.
        database[kind].create_index([("owner", ASCENDING), ("created_at", DESCENDING)])
    _ensure_guest_indexes(database, ASCENDING)
    for bucket in (UPLOADS_BUCKET, INDEX_BUCKET):   # list_user_subjects / fetch_uploaded_pdfs filter by owner
        database[bucket + ".files"].create_index([("owner", ASCENDING), ("subject", ASCENDING)], sparse=True)
    database[INDEX_BUCKET + ".files"].create_index([("subject_key", ASCENDING), ("owner", ASCENDING)], sparse=True)
    for coll, ttl in EPHEMERAL_TTLS.items():
        database[coll].create_index("owner")  # orphan sweep / migration filter on owner
        try:
            database[coll].create_index("created_at", expireAfterSeconds=ttl)
        except Exception as exc:  # noqa: BLE001 -- same index with other options (service created it first)
            try:
                database.command("collMod", coll, index={"keyPattern": {"created_at": 1}, "expireAfterSeconds": ttl})
            except Exception:  # noqa: BLE001
                log.warning("Could not create/update the %s TTL index: %s", coll, exc)
    try:   # spend ledger (backend/spend.py): one doc per (UTC day, scope), dropped by Mongo when expires_at passes
        database["usage_daily"].create_index("expires_at", expireAfterSeconds=0)
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not create the usage_daily TTL index: %s", exc)
    database["upload_chunks"].create_index([("upload_id", ASCENDING), ("index", ASCENDING)])
    database["upload_sessions"].create_index([("owner", ASCENDING), ("state", ASCENDING)])
    meta.update_one({"_id": "indexes"}, {"$set": {"version": want, "at": _now()}}, upsert=True)
    log.info("MongoDB indexes ensured (%s).", want)
    return {"created": len(KINDS) + 7 + 2 * len(EPHEMERAL_TTLS), "version": want}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def get_db():
    """Public name for the lazily-created pooled database handle (None when persistence is off)."""
    return _get_db()


def now() -> datetime:
    """Public name for the UTC 'now' used for every stored timestamp."""
    return _now()


def oid(id_str: str):
    """Public name: parse an ObjectId string, None if invalid."""
    return _oid(id_str)


def _oid(id_str: str):
    from bson import ObjectId
    from bson.errors import InvalidId

    try:
        return ObjectId(id_str)
    except (InvalidId, TypeError):
        return None


def _iso_utc(value: datetime) -> str:
    """ISO-format a stored timestamp, ALWAYS with an explicit UTC offset.

    PyMongo returns datetimes naive (tzinfo stripped) even though they are UTC.
    Serializing those naively yields "2026-06-12T18:25:00" — no offset — which
    browsers then parse as LOCAL time, displaying every history entry shifted
    by the user's UTC offset. Tagging UTC first yields "…+00:00", which
    `new Date()` converts correctly to the viewer's local time.
    """
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.isoformat()


def _public(doc: dict | None) -> dict | None:
    """Stringify _id -> id and ISO-format timestamps for JSON responses."""
    if not doc:
        return doc
    doc = dict(doc)
    doc["id"] = str(doc.pop("_id"))
    doc.pop("owner", None)
    for key in ("created_at", "updated_at"):
        if isinstance(doc.get(key), datetime):
            doc[key] = _iso_utc(doc[key])
    return doc


def _ensure_guest_indexes(db, ascending) -> None:
    """Create the guests collection's indexes: unique guestId + a TTL on lastActiveAt.

    The TTL is what reaps inactive guests automatically: Mongo deletes a guest document
    ``config.GUEST_TTL_SECONDS`` after its ``lastActiveAt`` (refreshed on every guest
    request, so active guests never expire). Index creation is isolated in its own
    try/except so a TTL options-conflict (someone changed GUEST_TTL_DAYS after the index
    already existed) can be reconciled with collMod instead of disabling persistence.
    """
    db[GUESTS].create_index([("guestId", ascending)], unique=True, name="guestId_unique")
    try:
        db[GUESTS].create_index(
            "lastActiveAt", expireAfterSeconds=config.GUEST_TTL_SECONDS, name="guest_ttl"
        )
    except Exception as exc:  # noqa: BLE001 — most likely an expiry change on an existing index
        try:
            db.command(
                "collMod", GUESTS,
                index={"name": "guest_ttl", "expireAfterSeconds": config.GUEST_TTL_SECONDS},
            )
            log.info("Updated guest TTL index to %ds.", config.GUEST_TTL_SECONDS)
        except Exception:  # noqa: BLE001 — keep persistence working even if the TTL can't be set
            log.warning("Could not create/update the guest TTL index: %s", exc)


def _public_guest(doc: dict | None) -> dict | None:
    """Shape a guest document for the API: drop _id, ISO-format timestamps."""
    if not doc:
        return doc
    doc = dict(doc)
    doc.pop("_id", None)
    for key in ("createdAt", "lastActiveAt", "migratedAt"):
        if isinstance(doc.get(key), datetime):
            doc[key] = _iso_utc(doc[key])
    return doc


def _guest_active_filter(guest_id: str) -> dict:
    """Match a guest by id that has NOT already been migrated to a Clerk account."""
    return {
        "guestId": guest_id,
        "$or": [{"migratedTo": None}, {"migratedTo": {"$exists": False}}],
    }


# ---- per-owner write epoch (cache invalidation for derived, per-instance caches) ----
# Every write that changes what an owner's history/progress shows bumps that owner's epoch; caches
# built from those reads (services/progress) store the epoch they were computed at and are dropped on
# mismatch. The epoch is local to this instance; when a shared cache is configured (backend/cache.py) it
# is ALSO bumped there, which invalidates every instance's shared (L2) entries for the owner at once.
# Bounded.
_epochs: dict[str, int] = {}
_EPOCH_MAX = 4000
# "Snapshot generation": bumped by writes we cannot patch into the owner snapshot (see below).
_snap_gen: dict[str, int] = {}


def _bump(*owners: str, keep_snapshot: bool = False) -> None:
    for o in owners:
        if not o:
            continue
        if len(_epochs) >= _EPOCH_MAX and o not in _epochs:
            _epochs.clear()           # crude, but any cleared owner just recomputes once
            _snap_gen.clear()
            _epochs["__reset__"] = _epochs.get("__reset__", 0) + 1
        _epochs[o] = _epochs.get(o, 0) + 1
        if not keep_snapshot:
            _snap_gen[o] = _snap_gen.get(o, 0) + 1
        cache.bump_epoch(o)


def epoch(owner: str) -> tuple[int, int]:
    """Opaque token that changes whenever this instance observed a write for ``owner``."""
    return (_epochs.get("__reset__", 0), _epochs.get(owner, 0))


# ---- guest accounts (anonymous, MongoDB-backed) ----
def create_guest() -> dict | None:
    """Create a new anonymous guest record and return its public shape (or None if off)."""
    db = _get_db()
    if db is None:
        return None
    now = _now()
    doc = {
        "guestId": GUEST_PREFIX + uuid4().hex,
        "isGuest": True,
        "displayName": "Guest User",
        "createdAt": now,
        "lastActiveAt": now,
        "migratedTo": None,
        "hasUploads": False,     # lets list_user_subjects skip its GridFS aggregation for the (common) guest with no uploads
    }
    db[GUESTS].insert_one(doc)
    _remember_touch(doc["guestId"], False)      # we just created it: the first request needs no validation round trip
    log.info("Created guest %s.", redact_id(doc["guestId"]))
    return _public_guest(doc)


def get_guest(guest_id: str) -> dict | None:
    """Return a live (non-migrated) guest by id, or None. Used to restore a session."""
    db = _get_db()
    if db is None or not guest_id:
        return None
    return _public_guest(db[GUESTS].find_one(_guest_active_filter(guest_id)))


# Guests validated recently by THIS instance -> skip the per-request Mongo write. The TTL is days,
# so refreshing lastActiveAt at most once per five minutes per guest per instance is indistinguishable
# from refreshing on every request, and removes a DB round-trip from the hot auth path. With a shared
# cache configured the same marker is shared across instances (a cold instance then also skips the write).
_TOUCH_INTERVAL_S = 300.0   # TTL is 30 days, so a 5-minute write granularity is invisible to it
_TOUCH_CACHE_MAX = 2000
_touched: dict[str, float] = {}
# guest id -> hasUploads (False = definitely no uploaded subjects; None/absent = unknown, e.g. legacy guest)
_guest_flags: dict[str, "bool | None"] = {}


def _remember_touch(guest_id: str, has_uploads: "bool | None") -> None:
    if len(_touched) >= _TOUCH_CACHE_MAX:
        _touched.clear()  # crude but bounded; worst case = one extra write per active guest
        _guest_flags.clear()
    _touched[guest_id] = time.monotonic()
    _guest_flags[guest_id] = has_uploads
    if cache.enabled():
        cache.set(cache.key("touch", guest_id), "1" if has_uploads else ("0" if has_uploads is False else "?"),
                  _TOUCH_INTERVAL_S)


def touch_guest(guest_id: str) -> bool:
    """Refresh a live guest's ``lastActiveAt`` (resets its TTL). True if the guest exists.

    Called from the auth layer on every guest-scoped request, so any guest that keeps
    using the app keeps its 30-day clock reset; only idle guests are reaped. A guest that was
    validated in the last ``_TOUCH_INTERVAL_S`` seconds by this instance (or, with a shared cache,
    by ANY instance) is accepted without a DB call (bounded cache; entries are dropped on migration).
    The single ``find_one_and_update`` both validates + touches the guest and returns its
    ``hasUploads`` flag, so the subjects list needs no extra query for a guest without uploads.
    """
    db = _get_db()
    if db is None or not guest_id:
        return False
    now_m = time.monotonic()
    seen = _touched.get(guest_id)
    if seen is not None and now_m - seen < _TOUCH_INTERVAL_S:
        return True
    if cache.enabled():
        mark = cache.get(cache.key("touch", guest_id))
        if mark is not None:
            _touched[guest_id] = now_m
            _guest_flags[guest_id] = {"1": True, "0": False}.get(mark)
            return True
    doc = db[GUESTS].find_one_and_update(
        _guest_active_filter(guest_id), {"$set": {"lastActiveAt": _now()}},
        projection={"hasUploads": 1, "_id": 0})
    if doc is not None:
        _remember_touch(guest_id, doc.get("hasUploads") if isinstance(doc, dict) else None)
        return True
    _touched.pop(guest_id, None)
    _guest_flags.pop(guest_id, None)
    return False


def has_no_uploads(owner: str | None) -> bool:
    """True only when ``owner`` is a guest KNOWN to have uploaded nothing (flag set at creation)."""
    return bool(owner) and owner.startswith(GUEST_PREFIX) and _guest_flags.get(owner, None) is False


def _mark_has_uploads(owner: str | None) -> None:
    """Called BEFORE a guest's first upload is stored, so a crash can only cost an extra aggregation."""
    if not owner or not owner.startswith(GUEST_PREFIX) or _guest_flags.get(owner, None) is True:
        return
    database = _get_db()
    if database is None:
        return
    try:
        database[GUESTS].update_one({"guestId": owner}, {"$set": {"hasUploads": True}})
    except Exception:  # noqa: BLE001 - unknown flag => the aggregation path still finds the uploads
        log.debug("Could not set hasUploads.", exc_info=True)
    _guest_flags[owner] = True
    cache.delete(cache.key("touch", owner))


def migrate_guest(guest_id: str, target_owner: str, *, after_claim=None) -> dict:
    """Move every artifact owned by ``guest_id`` to ``target_owner`` (a Clerk user id).

    Ordering is chosen so an interrupted upgrade can NEVER strand or lose data, and so a
    retry resumes cleanly:

      1. Re-key the artifacts FIRST (owner: guest -> Clerk user). A crash mid-loop leaves the
         already-moved artifacts under the Clerk account (immediately usable) and the rest
         under the still-live guest (still reachable by that guest, and re-keyed by a retry).
      2. Delete the spent guest record LAST. A crash before this just leaves an empty,
         still-live guest that a retry (or, harmlessly, the TTL) cleans up — no data is ever
         owned by a guest that no longer exists.

    ATOMIC CLAIM: before touching anything, the guest record is claimed with a single
    ``find_one_and_update`` that stamps ``migratedTo=target``. The filter only matches an
    unclaimed guest OR one already claimed by the SAME target (a retry), so two different
    accounts presenting the same guest id can never both win and split the data. A claimed guest
    no longer matches ``_guest_active_filter`` (it stops being usable as a guest identity). A guest
    already fully migrated (record gone) returns migrated=false/total=0 -- the settled case.
    """
    db = _get_db()
    result: dict = {"migrated": False, "total": 0, "moved": {k: 0 for k in KINDS}}
    if db is None or not guest_id or not target_owner:
        return result
    claimed = db[GUESTS].find_one_and_update(
        {
            "guestId": guest_id,
            "$or": [{"migratedTo": None}, {"migratedTo": {"$exists": False}}, {"migratedTo": target_owner}],
        },
        {"$set": {"migratedTo": target_owner, "migratedAt": _now()}},
    )
    if claimed is None:
        return result  # unknown / already migrated / claimed by someone else -- nothing to move
    # 0) Uploaded subjects (GridFS) move only AFTER the atomic claim, so two accounts racing with
    # the same guest id can never both re-own the guest's uploads. Idempotent on retry.
    if after_claim is not None:
        try:
            result["subjects_moved"] = int(after_claim() or 0)
        except Exception:  # noqa: BLE001 - leave them with the guest; a retry resumes
            log.warning("Could not migrate a guest's uploaded subjects", exc_info=True)
    # 1) Re-key artifacts first (safe-by-ordering; idempotent on retry).
    total = 0
    for kind in KINDS:
        res = db[kind].update_many(
            {"owner": guest_id}, {"$set": {"owner": target_owner, "updated_at": _now()}}
        )
        result["moved"][kind] = res.modified_count
        total += res.modified_count
    for coll in EPHEMERAL_TTLS:  # transient owner-keyed docs follow the user (not counted as artifacts)
        db[coll].update_many({"owner": guest_id}, {"$set": {"owner": target_owner}})
    result["total"] = total
    result["migrated"] = True
    # 2) Data is safe under the Clerk account — now retire the guest record.
    db[GUESTS].delete_one({"guestId": guest_id})
    _bump(guest_id, target_owner)
    _touched.pop(guest_id, None)
    _guest_flags.pop(guest_id, None)
    cache.delete(cache.key("touch", guest_id))
    log.info("Migrated guest %s -> %s (%d artifacts).", redact_id(guest_id), redact_id(target_owner), total)
    return result


def purge_orphan_guest_data(budget_s: float = 40.0, stats: dict | None = None) -> int:
    """Delete artifacts whose guest owner no longer exists (TTL can't cascade).

    Mongo's TTL index drops an inactive guest *document*, but its artifacts remain. This
    sweep finds the distinct ``guest_*`` owners per collection (an index prefix scan on
    ``owner``), checks which still have a live guest record, and deletes the rest. Memory is
    bounded by the number of distinct guest owners holding data (NOT every live guest, as the
    old ``$nin`` list was) and the sweep stops once ``budget_s`` has elapsed (serverless time
    limit) — whatever is left is picked up by the next run. Returns artifacts removed.
    """
    db = _get_db()
    if db is None:
        return 0
    deadline = time.monotonic() + budget_s
    removed = 0
    for kind in (*KINDS, *EPHEMERAL_TTLS):
        if time.monotonic() > deadline:
            log.warning("Orphan sweep hit its time budget; resuming on the next run.")
            if stats is not None:
                stats["partial"] = True
            break
        owners = db[kind].distinct("owner", {"owner": {"$regex": "^" + GUEST_PREFIX}})
        if not owners:
            continue
        live: set[str] = set()
        for i in range(0, len(owners), 500):
            chunk = owners[i:i + 500]
            live.update(g["guestId"] for g in db[GUESTS].find({"guestId": {"$in": chunk}}, {"guestId": 1, "_id": 0}))
        dead = [o for o in owners if o not in live]
        for i in range(0, len(dead), 500):
            if time.monotonic() > deadline:        # resumable: what is left is found again next run
                if stats is not None:
                    stats["partial"] = True
                return removed
            removed += db[kind].delete_many({"owner": {"$in": dead[i:i + 500]}}).deleted_count
    return removed


def save(kind: str, owner: str, doc: dict) -> str | None:
    """Insert an artifact for ``owner`` and return its id (or None if disabled)."""
    db = _get_db()
    if db is None or kind not in KINDS:
        return None
    payload = dict(doc)
    payload["owner"] = owner
    payload.setdefault("created_at", _now())
    payload["updated_at"] = _now()
    inserted = None
    try:
        inserted = db[kind].insert_one(payload).inserted_id
        return str(inserted)
    except Exception as exc:  # noqa: BLE001
        # A history write must never discard work the user already waited (and the provider billed)
        # for: if Mongo is saturated/unreachable, hand back the result without a history id.
        if type(exc).__module__.split(".")[0] not in ("pymongo", "bson"):
            raise
        log.warning("Could not save %s for %s (returning the result unsaved): %s", kind, redact_id(owner), str(exc)[:160])
        return None
    finally:
        # AFTER the write: a list read racing the insert can't be cached as fresh. A successful insert is
        # patched into this instance's owner snapshot (no re-read); anything else invalidates it.
        if inserted is not None:
            _snap_patch(owner, kind, insert=_snap_row(payload, inserted))
            _bump(owner, keep_snapshot=True)
        else:
            _bump(owner)


# Per-instance cache of NON-default history LISTS (search ``?q=`` / custom limit). Keyed by owner + kind
# + query, so one owner can never be served another's rows; dropped when this instance observes a write
# for the owner (epoch) and after a short TTL (writes on other instances). The default list (what the
# sidebar/drawer and "My quizzes" show) comes from the owner SNAPSHOT below.
_LIST_TTL_S = 15.0
_LIST_MAX = 1000
_list_cache: "dict[tuple, tuple[float, tuple, list]]" = {}
_list_lock = threading.Lock()


def clear_list_cache() -> None:
    with _list_lock:
        _list_cache.clear()
    clear_snapshots()


# ---- owner snapshot: ONE aggregation serves every history list AND the progress dashboard -----------
# A snapshot is the newest ``_SNAP_ROWS`` rows of every artifact kind for one owner, read with a single
# ``$unionWith`` aggregation (one Mongo op instead of one per list + one for the dashboard). It lives in
# this process (``_SNAP_TTL_S``) and, when a shared cache is configured, in the shared cache keyed by the
# owner's shared epoch. Writes made by this instance are PATCHED into it (a new row at the front, a title
# change ...) so a write does not force a re-read; any write we cannot patch invalidates it.
_SNAP_ROWS = 100
_SNAP_TTL_S = 60.0
_SNAP_L2_TTL_S = 120.0
_SNAP_MAX = 256
# fields never kept in a snapshot row (list summaries exclude them; ``topic_breakdown`` is kept because the
# progress dashboard reads it, and is stripped again when a row is served as a list item)
_SNAP_DROP = tuple(f for f in _HEAVY_FIELDS if f != "topic_breakdown") + ("owner",)
_snaps: "OrderedDict[str, dict]" = OrderedDict()
_snap_lock = threading.RLock()


def clear_snapshots() -> None:
    with _snap_lock:
        _snaps.clear()


def _snap_token(owner: str) -> tuple[int, int]:
    return (_epochs.get("__reset__", 0), _snap_gen.get(owner, 0))


def _snap_row(payload: dict, oid_) -> dict:
    row = {k: v for k, v in payload.items() if k not in _SNAP_DROP}
    row["_id"] = oid_
    return copy.deepcopy(row)


def _snap_valid(owner: str) -> dict | None:
    """The owner's live snapshot (``{kind: {"n", "rows"}}``) or None. Caller holds no lock."""
    with _snap_lock:
        ent = _snaps.get(owner)
        if ent is None:
            return None
        if ent["exp"] <= time.monotonic() or ent["tok"] != _snap_token(owner):
            _snaps.pop(owner, None)
            return None
        _snaps.move_to_end(owner)
        return ent["kinds"]


def _snap_store(owner: str, kinds: dict, tok: tuple[int, int]) -> None:
    with _snap_lock:
        _snaps[owner] = {"exp": time.monotonic() + _SNAP_TTL_S, "tok": tok, "kinds": kinds}
        _snaps.move_to_end(owner)
        while len(_snaps) > _SNAP_MAX:
            _snaps.popitem(last=False)


def _snap_patch(owner: str, kind: str, *, insert: dict | None = None, remove_id=None,
                set_fields: dict | None = None, row_id=None, clear: bool = False) -> bool:
    """Apply one write to the owner's live snapshot. Returns False (and drops it) if it cannot be
    applied exactly; True if patched or if there is no snapshot to patch."""
    with _snap_lock:
        ent = _snaps.get(owner)
        if ent is None:
            return True
        if ent["tok"] != _snap_token(owner) or ent["exp"] <= time.monotonic():
            _snaps.pop(owner, None)
            return True
        slot = ent["kinds"].get(kind)
        if slot is None:
            _snaps.pop(owner, None)
            return False
        rows = slot["rows"]
        if clear:
            rows.clear()
        elif insert is not None:
            rows.insert(0, insert)
            del rows[_SNAP_ROWS:]
        elif remove_id is not None:
            i = next((j for j, r in enumerate(rows) if str(r.get("_id")) == str(remove_id)), None)
            if i is None:
                if len(rows) >= _SNAP_ROWS:        # the deleted row might be below the cut: unknowable
                    _snaps.pop(owner, None)
                    return False
            else:
                if len(rows) >= _SNAP_ROWS:        # a hidden row would now belong in the window
                    _snaps.pop(owner, None)
                    return False
                del rows[i]
        elif set_fields is not None:
            for r in rows:
                if str(r.get("_id")) == str(row_id):
                    r.update(copy.deepcopy(set_fields))
        slot["n"] = len(rows)
        return True


def _jsonable_rows(kinds: dict) -> dict:
    """Snapshot -> JSON-safe (ObjectId -> str, datetimes -> ISO with explicit UTC offset)."""
    def conv(v):
        if isinstance(v, datetime):
            return _iso_utc(v)
        if isinstance(v, dict):
            return {k: conv(x) for k, x in v.items()}
        if isinstance(v, list):
            return [conv(x) for x in v]
        if type(v).__name__ == "ObjectId":
            return str(v)
        return v
    return {k: {"n": slot["n"], "rows": [conv(r) for r in slot["rows"]]} for k, slot in kinds.items()}


def warm_owner(owner: str) -> dict | None:
    """The owner's snapshot, loading it if needed (L1 -> shared L2 -> ONE aggregation). None when
    persistence is off or the read failed (callers then use their own query path)."""
    snap = _snap_valid(owner)
    if snap is not None:
        return snap
    database = _get_db()
    if database is None or not owner:
        return None
    with cache.single_flight(("snap", owner)):
        snap = _snap_valid(owner)                       # a concurrent caller may have just loaded it
        if snap is not None:
            return snap
        tok = _snap_token(owner)
        l2, ep = cache.l2_get("snap", owner)
        if isinstance(l2, dict) and set(l2) >= set(KINDS):
            _snap_store(owner, {k: {"n": int(v.get("n", 0)), "rows": list(v.get("rows") or [])} for k, v in l2.items() if k in KINDS}, tok)
            return _snap_valid(owner)

        def branch(kind: str) -> list[dict]:
            return [{"$match": {"owner": owner}}, {"$sort": {"created_at": -1}}, {"$limit": _SNAP_ROWS},
                    {"$addFields": {"_k": kind}}, {"$project": {f: 0 for f in _SNAP_DROP}}]

        pipeline = branch(KINDS[0])
        for kind in KINDS[1:]:
            pipeline.append({"$unionWith": {"coll": kind, "pipeline": branch(kind)}})
        kinds = {k: {"n": 0, "rows": []} for k in KINDS}
        try:
            for row in database[KINDS[0]].aggregate(pipeline, allowDiskUse=False):
                slot = kinds.get(row.pop("_k", None))
                if slot is not None:
                    slot["rows"].append(row)
        except Exception as exc:  # noqa: BLE001
            if type(exc).__module__.split(".")[0] not in ("pymongo", "bson"):
                raise
            log.warning("Could not load the owner snapshot: %s", str(exc)[:160])
            return None
        for slot in kinds.values():
            slot["rows"].sort(key=lambda r: r.get("created_at") or datetime.min, reverse=True)
            slot["n"] = len(slot["rows"])
        _snap_store(owner, kinds, tok)
        if ep is not None:
            cache.l2_put("snap", owner, "", _jsonable_rows(kinds), ep, _SNAP_L2_TTL_S)
        return _snap_valid(owner) or kinds


def _list_from_snapshot(kind: str, snap: dict) -> list[dict]:
    items = []
    for row in snap[kind]["rows"]:
        r = {k: v for k, v in row.items() if k != "topic_breakdown"}
        items.append(_public(r))
    return copy.deepcopy(items)


def list_items(kind: str, owner: str, q: str | None = None, limit: int = 100) -> list[dict]:
    """List an owner's artifacts (summaries, newest first), optionally filtered by ``q``."""
    db = _get_db()
    if db is None or kind not in KINDS:
        return []
    limit = max(1, min(int(limit or 100), 200))  # bounded: never stream an unbounded history
    if not q and limit == _SNAP_ROWS:
        snap = warm_owner(owner)                  # ONE aggregation fills every kind's list + the dashboard
        if snap is not None:
            return _list_from_snapshot(kind, snap)
    key = (owner, kind, q or "", limit)
    ep = epoch(owner)
    now = time.monotonic()
    with _list_lock:
        hit = _list_cache.get(key)
        if hit and hit[0] > now and hit[1] == ep:
            return copy.deepcopy(hit[2])
    with cache.single_flight(("list", key)):
        with _list_lock:
            hit = _list_cache.get(key)
            if hit and hit[0] > time.monotonic() and hit[1] == ep:
                return copy.deepcopy(hit[2])
        query: dict = {"owner": owner}
        if q:
            rx = {"$regex": re.escape(q), "$options": "i"}
            query["$or"] = [{field: rx} for field in _SEARCH_FIELDS]
        projection = {field: 0 for field in _HEAVY_FIELDS}
        projection["owner"] = 0
        cursor = db[kind].find(query, projection).sort("created_at", -1).limit(limit)
        items = [_public(d) for d in cursor]
        with _list_lock:
            if len(_list_cache) >= _LIST_MAX:
                _list_cache.clear()
            _list_cache[key] = (now + _LIST_TTL_S, ep, items)
    return copy.deepcopy(items)


def _pick(row: dict, fields) -> dict:
    """``row`` restricted to ``fields`` (dotted paths become nested dicts, like a Mongo ``$project``)."""
    out: dict = {}
    for f in fields:
        src, dst, cur = row, out, None
        parts = f.split(".")
        ok = True
        for part in parts[:-1]:
            nxt = src.get(part) if isinstance(src, dict) else None
            if not isinstance(nxt, dict):
                ok = False
                break
            src = nxt
            dst = dst.setdefault(part, {})
        if ok and isinstance(src, dict) and parts[-1] in src:
            dst[parts[-1]] = copy.deepcopy(src[parts[-1]])
        elif len(parts) > 1 and not dst and parts[0] in out:
            out.pop(parts[0], None)
    return out


def _recent_from_snapshot(snap: dict, specs: list[tuple]) -> dict | None:
    """Serve ``multi_recent`` specs from a snapshot, or None if any spec needs more than it holds."""
    out: dict = {}
    for kind, limit, fields in specs:
        if kind not in snap or int(limit) > _SNAP_ROWS:
            return None
        if fields is not None and any(f.split(".")[0] in _SNAP_DROP for f in fields):
            return None
        rows = snap[kind]["rows"][:max(1, int(limit))]
        out[kind] = {"n": len(rows), "rows": [] if fields is None else [_pick(r, fields) for r in rows]}
    return out


def multi_recent(owner: str, specs: list[tuple]) -> dict[str, dict]:
    """ONE round trip for the newest items of several kinds (replaces N parallel ``list_items``).

    ``specs`` = ``[(kind, limit, fields)]``; ``fields`` is a list of dotted field names to return
    per row, or ``None`` to only count (rows are then omitted). Result:
    ``{kind: {"n": int, "rows": [dict, ...]}}`` for every requested kind (``n`` = number of
    items among the newest ``limit``, i.e. ``min(limit, total)`` like ``len(list_items(...))``).
    Owner-scoped in every branch; each branch is ``$match owner -> $sort created_at desc ->
    $limit`` so it is served by the (owner, created_at) index. Implemented as one aggregation:
    the first kind is the base collection and the rest are ``$unionWith`` branches. When this instance
    already holds the owner's snapshot (see ``warm_owner``) it is answered from memory with no query.
    """
    out = {k: {"n": 0, "rows": []} for k, _l, _f in specs}
    database = _get_db()
    specs = [sp for sp in specs if sp[0] in KINDS]
    if database is None or not specs:
        return out
    snap = _snap_valid(owner)
    if snap is not None:
        served = _recent_from_snapshot(snap, specs)
        if served is not None:
            out.update(served)
            return out

    def branch(kind: str, limit: int, fields):
        stages: list[dict] = [{"$match": {"owner": owner}}, {"$sort": {"created_at": -1}},
                              {"$limit": max(1, min(int(limit), 200))}]
        if fields is None:
            stages.append({"$group": {"_id": None, "n": {"$sum": 1}}})
            stages.append({"$project": {"_id": 0, "n": 1, "_k": {"$literal": kind}}})
        else:
            proj = {"_id": 0, "_k": {"$literal": kind}}
            proj.update({f: 1 for f in fields})
            stages.append({"$project": proj})
        return stages

    (k0, l0, f0), rest = specs[0], specs[1:]
    pipeline = branch(k0, l0, f0)
    for kind, limit, fields in rest:
        pipeline.append({"$unionWith": {"coll": kind, "pipeline": branch(kind, limit, fields)}})
    for doc in database[k0].aggregate(pipeline, allowDiskUse=False):
        slot = out.get(doc.pop("_k", None))
        if slot is None:
            continue
        if "n" in doc and len(doc) == 1:
            slot["n"] += int(doc["n"])
        else:
            slot["rows"].append(doc)
            slot["n"] += 1
    return out


def get_item(kind: str, owner: str, id_str: str) -> dict | None:
    """Return one full artifact, scoped to ``owner`` (None if not found/forbidden)."""
    db = _get_db()
    oid = _oid(id_str)
    if db is None or kind not in KINDS or oid is None:
        return None
    return _public(db[kind].find_one({"_id": oid, "owner": owner}))


def delete_item(kind: str, owner: str, id_str: str) -> bool:
    """Delete one artifact scoped to ``owner``. Returns True if something was removed."""
    db = _get_db()
    oid = _oid(id_str)
    if db is None or kind not in KINDS or oid is None:
        return False
    ok = False
    try:
        ok = db[kind].delete_one({"_id": oid, "owner": owner}).deleted_count > 0
        return ok
    finally:
        patched = ok and _snap_patch(owner, kind, remove_id=oid)
        _bump(owner, keep_snapshot=patched)


def rename_item(kind: str, owner: str, id_str: str, title: str) -> bool:
    """Rename one artifact (set its title), scoped to ``owner``."""
    db = _get_db()
    oid = _oid(id_str)
    if db is None or kind not in KINDS or oid is None:
        return False
    stamp = _now()
    matched = False
    try:
        res = db[kind].update_one(
            {"_id": oid, "owner": owner},
            {"$set": {"title": title, "updated_at": stamp}},
        )
        matched = res.matched_count > 0
    finally:
        patched = matched and _snap_patch(owner, kind, set_fields={"title": title, "updated_at": stamp}, row_id=oid)
        _bump(owner, keep_snapshot=patched)
    return matched


def clear_kind(kind: str, owner: str) -> int:
    """Delete ALL of an owner's artifacts of a kind. Returns the count removed."""
    db = _get_db()
    if db is None or kind not in KINDS:
        return 0
    done = False
    try:
        n = db[kind].delete_many({"owner": owner}).deleted_count
        done = True
        return n
    finally:
        patched = done and _snap_patch(owner, kind, clear=True)
        _bump(owner, keep_snapshot=patched)


# ---- chat-specific helpers (chats accumulate messages over a session) ----
def new_chat_id() -> str | None:
    """A fresh chat id WITHOUT touching the database (the chats document is created lazily by the first
    turn's upsert, saving one Mongo write per session). None when persistence is off."""
    if _get_db() is None:
        return None
    from bson import ObjectId

    return str(ObjectId())


def create_chat(owner: str, meta: dict) -> str | None:
    """Create a chat document (with an empty messages array) and return its id."""
    return save("chats", owner, {**meta, "messages": []})


def note_chat_write(owner: str, chat_id: str, *, updated_at=None, title: str | None = None,
                    created_doc: dict | None = None) -> None:
    """Record a chat write made by the chat service (turn persisted / retitled) so the owner's cached
    history reflects it without a re-read. ``created_doc`` is given when the write CREATED the chat
    (lazy upsert): it is inserted at the front of the chats list. Falls back to invalidation."""
    oid = _oid(chat_id)
    patched = False
    if oid is not None:
        if created_doc is not None:
            patched = _snap_patch(owner, "chats", insert=_snap_row(created_doc, oid))
        else:
            fields = {}
            if updated_at is not None:
                fields["updated_at"] = updated_at
            if title:
                fields["title"] = title
            patched = _snap_patch(owner, "chats", set_fields=fields, row_id=oid) if fields else True
    _bump(owner, keep_snapshot=patched)


def append_chat_message(owner: str, chat_id: str, role: str, content: str) -> None:
    """Append a message to an owner's chat and bump its updated_at."""
    db = _get_db()
    oid = _oid(chat_id)
    if db is None or oid is None:
        return
    stamp = _now()
    try:
        db["chats"].update_one(
            {"_id": oid, "owner": owner},
            {
                "$push": {"messages": {"role": role, "content": content, "ts": stamp.isoformat()}},
                "$set": {"updated_at": stamp},
            },
        )
    finally:
        note_chat_write(owner, chat_id, updated_at=stamp)


def set_chat_title(owner: str, chat_id: str, title: str) -> None:
    """Set a chat's title only if it is still untitled (first user message wins)."""
    db = _get_db()
    oid = _oid(chat_id)
    if db is None or oid is None:
        return
    try:
        db["chats"].update_one(
            {"_id": oid, "owner": owner, "$or": [{"title": {"$exists": False}}, {"title": ""}, {"title": None}]},
            {"$set": {"title": title}},
        )
    finally:
        _bump(owner)


# ---- uploaded PDFs (binary blobs) — let user-uploaded subjects survive restarts ----
# On a host without a persistent disk (e.g. Render's free tier) the documents/ dir is
# ephemeral, so PDFs added via "Add a Subject" vanish on the next restart/redeploy. When
# MongoDB is configured we ALSO store each uploaded PDF here and re-hydrate documents/ from
# it on startup, so uploads persist at no extra cost. Files are kept in GridFS (chunked), so
# a PDF of any size is handled safely — not capped by Mongo's 16 MB document limit.
#
# Bundled subjects are baked into the image and seeded onto disk by the entrypoint; they are
# NEVER stored here, so there is no duplication and the two sources stay cleanly separated.
UPLOADS_BUCKET: str = "uploaded_pdfs"


def _gridfs():
    """Return a GridFS bucket for uploaded PDFs, or None when persistence is off."""
    db = _get_db()
    if db is None:
        return None
    from gridfs import GridFS

    return GridFS(db, collection=UPLOADS_BUCKET)


def store_uploaded_pdf(subject: str, filename: str, data: bytes, owner: str | None = None) -> bool:
    """Persist one uploaded PDF so it survives a restart. No-op when persistence is off.

    ``owner`` is the Clerk user id (or None for bundled subjects — never used in practice).
    Idempotent: a (owner, subject, filename) already stored is left untouched. Returns True if
    the file is stored (newly or already), False if persistence is unavailable or the write
    fails — callers treat the on-disk copy as authoritative.
    """
    fs = _gridfs()
    if fs is None:
        return False
    _mark_has_uploads(owner)
    try:
        query = {"subject": subject, "filename": filename}
        if owner:
            query["owner"] = owner
        if fs.find_one(query) is not None:
            return True
        meta = {"filename": filename, "subject": subject, "uploaded_at": _now()}
        if owner:
            meta["owner"] = owner
        fs.put(data, **meta)
        log.info("Stored uploaded PDF in MongoDB: %s/%s owner=%s (%d bytes).", subject, filename, redact_id(owner), len(data))
        cache.bump_epoch(owner or "")
        return True
    except Exception as exc:  # noqa: BLE001 — persistence must never break the upload itself
        log.warning("Could not store uploaded PDF %s/%s in MongoDB: %s", subject, filename, exc)
        return False


def store_uploaded_pdf_file(subject: str, filename: str, path, owner: str | None = None) -> bool:
    """Like :func:`store_uploaded_pdf` but streams the file into GridFS (never holds it in memory)."""
    fs = _gridfs()
    if fs is None:
        return False
    _mark_has_uploads(owner)
    try:
        query = {"subject": subject, "filename": filename}
        if owner:
            query["owner"] = owner
        if fs.find_one(query) is not None:
            return True
        meta = {"filename": filename, "subject": subject, "uploaded_at": _now()}
        if owner:
            meta["owner"] = owner
        with open(path, "rb") as fh:
            fs.put(fh, **meta)
        cache.bump_epoch(owner or "")
        return True
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not store uploaded PDF %s/%s in MongoDB: %s", subject, filename, exc)
        return False


def rename_uploaded_subject(owner: str, old: str, new: str) -> int:
    """Re-label this owner's stored PDFs from subject ``old`` to ``new``. Returns files changed."""
    database = _get_db()
    if database is None or not owner:
        return 0
    res = database[f"{UPLOADS_BUCKET}.files"].update_many({"owner": owner, "subject": old},
                                                           {"$set": {"subject": new}})
    cache.bump_epoch(owner)
    return int(getattr(res, "modified_count", 0))


def rename_history_subject(owner: str, old: str, new: str) -> int:
    """Relabel the subject on this owner's saved chats / quizzes / ... (one bounded update_many per
    collection, all served by the owner index). Returns documents changed."""
    database = _get_db()
    if database is None or not owner:
        return 0
    n = 0
    for kind in KINDS:
        try:
            res = database[kind].update_many({"owner": owner, "subject": old}, {"$set": {"subject": new}})
            n += int(getattr(res, "modified_count", 0))
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not relabel %s for a renamed subject: %s", kind, exc)
    if n:
        _bump(owner)
    return n


def iter_uploaded_pdfs():
    """Yield ``(owner, subject, filename, data)`` for every stored upload (empty when off).

    ``owner`` is the Clerk user id stored with the file, or None for pre-migration records
    that were saved without an owner.  Used at startup to re-hydrate documents/; each file is
    restored to its owner-namespaced path so uploads are scoped per user.
    """
    fs = _gridfs()
    if fs is None:
        return
    try:
        for grid_out in fs.find():
            subject = getattr(grid_out, "subject", None)
            filename = grid_out.filename
            owner = getattr(grid_out, "owner", None)
            if not subject or not filename:
                continue
            yield owner, subject, filename, grid_out.read()
    except Exception as exc:  # noqa: BLE001 — a restore failure must not stop startup
        log.warning("Could not read uploaded PDFs from MongoDB: %s", exc)


# ---- retrieval indexes (per user subject) — persist across serverless instances ----
# Bundled subjects ship their index read-only in the deployment (data/index/) and never touch
# MongoDB. A user-uploaded subject's index is ONE zip (manifest + chunks + float16 vectors,
# built by backend.vector_store.pack_index) stored in GridFS under ``subject_key`` =
# ``user__<owner>__<subject>``. Every read/write is also filtered by ``owner`` so one user can
# never load another's chunks, even with a crafted key.
INDEX_BUCKET: str = "subject_index_v2"


def _index_gridfs():
    """GridFS bucket for retrieval indexes, or None when persistence is off."""
    database = _get_db()
    if database is None:
        return None
    from gridfs import GridFS

    return GridFS(database, collection=INDEX_BUCKET)


def store_index_blob(key: str, owner: str | None, subject: str, version: str | None, data: bytes,
                     meta: dict | None = None) -> bool:
    """Replace the stored index zip for ``key``. Writes the new blob first, then deletes
    older ones, so a concurrent reader always finds a complete index. False if off/failed."""
    if not owner:
        return False
    fs = _index_gridfs()
    if fs is None:
        return False
    _mark_has_uploads(owner)
    try:
        new_id = fs.put(data, filename="index.zip", subject_key=key, owner=owner, subject=subject,
                        version=version, saved_at=_now(), **{f"m_{k}": v for k, v in (meta or {}).items()})
        for old in fs.find({"subject_key": key, "owner": owner, "_id": {"$ne": new_id}}):
            fs.delete(old._id)
        log.info("Stored retrieval index in MongoDB: %s (%d bytes).", key, len(data))
        cache.bump_epoch(owner)
        return True
    except Exception as exc:  # noqa: BLE001 — persistence must never break the upload
        log.warning("Could not store retrieval index %s in MongoDB: %s", key, exc)
        return False


def load_index_blob(key: str, owner: str | None) -> bytes | None:
    """Return the newest stored index zip for (key, owner), or None."""
    if not owner:
        return None
    fs = _index_gridfs()
    if fs is None:
        return None
    try:
        doc = fs.find_one({"subject_key": key, "owner": owner}, sort=[("uploadDate", -1)])
        return doc.read() if doc is not None else None
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not load retrieval index %s from MongoDB: %s", key, exc)
        return None


def get_index_version(key: str, owner: str | None) -> str | None:
    """Cheap metadata-only lookup of the stored index version (for cache revalidation)."""
    if not owner:
        return None
    fs = _index_gridfs()
    if fs is None:
        return None
    try:
        doc = fs.find_one({"subject_key": key, "owner": owner}, sort=[("uploadDate", -1)])
        return getattr(doc, "version", None) if doc is not None else None
    except Exception:  # noqa: BLE001
        return None


def index_version_state(key: str, owner: str | None) -> tuple[bool, str | None]:
    """``(reachable, version)``: reachable=False when Mongo is off/errored (state unknown);
    reachable=True with version None means the index is definitively absent (e.g. deleted)."""
    if not owner:
        return False, None
    fs = _index_gridfs()
    if fs is None:
        return False, None
    try:
        doc = fs.find_one({"subject_key": key, "owner": owner}, sort=[("uploadDate", -1)])
    except Exception:  # noqa: BLE001
        return False, None
    return True, (getattr(doc, "version", None) if doc is not None else None)


def delete_index_blob(key: str, owner: str | None) -> int:
    if not owner:
        return 0
    fs = _index_gridfs()
    if fs is None:
        return 0
    n = 0
    try:
        for old in fs.find({"subject_key": key, "owner": owner}):
            fs.delete(old._id)
            n += 1
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not delete retrieval index %s: %s", key, exc)
    if n:
        cache.bump_epoch(owner)          # shared subjects list entries for this owner are stale now
    return n


def list_user_subjects(owner: str | None) -> list[str]:
    """Display names of the subjects this owner has an index OR uploaded PDFs for in MongoDB."""
    if not owner:
        return []
    database = _get_db()
    if database is None:
        return []
    if has_no_uploads(owner):          # a guest flagged at creation as having uploaded nothing: no query
        return []
    cached, ep = cache.l2_get("subj", owner)
    if isinstance(cached, list):
        return [str(x) for x in cached]
    # ONE aggregation over both GridFS file collections (uploads + retrieval indexes), served by the
    # (owner, subject) indexes; returns only distinct subject names, never file metadata.
    branch = [{"$match": {"owner": owner}}, {"$group": {"_id": "$subject"}}]
    try:
        rows = database[UPLOADS_BUCKET + ".files"].aggregate(
            branch + [{"$unionWith": {"coll": INDEX_BUCKET + ".files", "pipeline": branch}}])
        names = sorted({r["_id"] for r in rows if r.get("_id")})
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not list subjects for owner: %s", exc)
        return []
    cache.l2_put("subj", owner, "", names, ep, 120.0)
    return names


_BIG_FETCH_BYTES = 32 * 1024 * 1024


def fetch_uploaded_pdfs(owner: str | None, subject: str) -> list[tuple[str, bytes]]:
    """``[(filename, bytes), …]`` of this owner's stored uploads for ``subject`` (cold-instance
    re-hydration of documents/). Empty when off. Owner-scoped: no owner => nothing."""
    if not owner:
        return []
    fs = _gridfs()
    if fs is None:
        return []
    out: list[tuple[str, bytes]] = []
    try:
        for grid_out in fs.find({"owner": owner, "subject": subject}):
            if int(getattr(grid_out, "length", 0) or 0) > _BIG_FETCH_BYTES:
                import tempfile
                from pathlib import Path

                fd, tmp = tempfile.mkstemp(prefix="sah-fetch-")      # stream big originals to disk
                with os.fdopen(fd, "wb") as fh:
                    while True:
                        block = grid_out.read(1 << 20)
                        if not block:
                            break
                        fh.write(block)
                out.append((grid_out.filename, Path(tmp)))
            else:
                out.append((grid_out.filename, grid_out.read()))
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not fetch uploaded PDFs for a subject: %s", exc)
    return out


def close() -> None:
    """Close the Mongo client (called on app shutdown)."""
    global _client, _db
    if _client is not None:
        _client.close()
    _client = None
    _db = None
