"""Optional MongoDB persistence — per-user history for every generated artifact.

Active only when ``config.MONGODB_URI`` is set; otherwise every function is a
graceful no-op so the app runs identically without a database. EVERY read and write
is scoped by ``owner`` (the signed-in Clerk user id, or "local" when auth is off),
so users never see each other's history.

Artifact kinds: chats, quizzes, flashcards, plans, mindmaps, summaries, exams,
teachbacks, predictions, attempts (progress events), cheatsheets, scans.
"""
from __future__ import annotations

import logging
import re
import threading
import time
from datetime import datetime, timezone
from uuid import uuid4

from backend import config

log = logging.getLogger("backend.db")

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
)
# Text fields searched by the ?q= filter (missing fields simply don't match).
_SEARCH_FIELDS = ("title", "source", "label", "topic", "subject", "messages.content")

_client = None
_db = None
_init_failed = False
# Serializes the lazy client/index initialization. The guest janitor daemon runs its first
# sweep at startup concurrently with the first HTTP request (FastAPI sync routes use a
# threadpool), so two threads can otherwise both build a MongoClient and create indexes.
_init_lock = threading.Lock()


def is_enabled() -> bool:
    """True when persistence is configured and the client initialized OK."""
    return config.PERSISTENCE_ENABLED and not _init_failed and _get_db() is not None


def _get_db():
    """Lazily build the Mongo client + ensure indexes. Returns None if disabled/failed.

    Uses double-checked locking so concurrent callers (e.g. the startup janitor sweep and
    the first request) initialize the client exactly once.
    """
    global _client, _db, _init_failed
    if not config.PERSISTENCE_ENABLED or _init_failed:
        return None
    if _db is not None:
        return _db
    with _init_lock:
        if _db is not None or _init_failed:
            return _db
        try:
            from pymongo import ASCENDING, DESCENDING, MongoClient

            client = MongoClient(
                config.MONGODB_URI,
                serverSelectionTimeoutMS=8000,
                appname="study-assistant-hub",
            )
            client.admin.command("ping")  # fail fast if unreachable
            database = client[config.MONGODB_DB_NAME]
            for kind in KINDS:
                database[kind].create_index([("owner", ASCENDING), ("created_at", DESCENDING)])
            _ensure_guest_indexes(database, ASCENDING)
            _client = client
            _db = database
            log.info("MongoDB persistence ready (db=%s).", config.MONGODB_DB_NAME)
        except Exception as exc:  # noqa: BLE001 — any failure disables persistence gracefully
            _init_failed = True
            log.error("MongoDB init failed; persistence disabled for this run: %s", exc)
            return None
    return _db


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _oid(id_str: str):
    from bson import ObjectId
    from bson.errors import InvalidId

    try:
        return ObjectId(id_str)
    except (InvalidId, TypeError):
        return None


def _public(doc: dict | None) -> dict | None:
    """Stringify _id -> id and ISO-format timestamps for JSON responses."""
    if not doc:
        return doc
    doc = dict(doc)
    doc["id"] = str(doc.pop("_id"))
    doc.pop("owner", None)
    for key in ("created_at", "updated_at"):
        if isinstance(doc.get(key), datetime):
            doc[key] = doc[key].isoformat()
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
            doc[key] = doc[key].isoformat()
    return doc


def _guest_active_filter(guest_id: str) -> dict:
    """Match a guest by id that has NOT already been migrated to a Clerk account."""
    return {
        "guestId": guest_id,
        "$or": [{"migratedTo": None}, {"migratedTo": {"$exists": False}}],
    }


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
    }
    db[GUESTS].insert_one(doc)
    log.info("Created guest %s.", doc["guestId"])
    return _public_guest(doc)


def get_guest(guest_id: str) -> dict | None:
    """Return a live (non-migrated) guest by id, or None. Used to restore a session."""
    db = _get_db()
    if db is None or not guest_id:
        return None
    return _public_guest(db[GUESTS].find_one(_guest_active_filter(guest_id)))


def touch_guest(guest_id: str) -> bool:
    """Refresh a live guest's ``lastActiveAt`` (resets its TTL). True if the guest exists.

    Called from the auth layer on every guest-scoped request, so any guest that keeps
    using the app keeps its 30-day clock reset; only idle guests are reaped.
    """
    db = _get_db()
    if db is None or not guest_id:
        return False
    res = db[GUESTS].update_one(_guest_active_filter(guest_id), {"$set": {"lastActiveAt": _now()}})
    return res.matched_count > 0


def migrate_guest(guest_id: str, target_owner: str) -> dict:
    """Move every artifact owned by ``guest_id`` to ``target_owner`` (a Clerk user id).

    Ordering is chosen so an interrupted upgrade can NEVER strand or lose data, and so a
    retry resumes cleanly:

      1. Re-key the artifacts FIRST (owner: guest -> Clerk user). A crash mid-loop leaves the
         already-moved artifacts under the Clerk account (immediately usable) and the rest
         under the still-live guest (still reachable by that guest, and re-keyed by a retry).
      2. Delete the spent guest record LAST. A crash before this just leaves an empty,
         still-live guest that a retry (or, harmlessly, the TTL) cleans up — no data is ever
         owned by a guest that no longer exists.

    The guest stays "active" throughout, so a retry matches it via ``_guest_active_filter``
    and re-keys only whatever is left (update_many is idempotent here). A guest already fully
    migrated (record gone) returns migrated=false/total=0 — the settled "nothing to do" case.
    """
    db = _get_db()
    result: dict = {"migrated": False, "total": 0, "moved": {k: 0 for k in KINDS}}
    if db is None or not guest_id or not target_owner:
        return result
    if db[GUESTS].find_one(_guest_active_filter(guest_id)) is None:
        return result  # unknown / already fully migrated guest — nothing to move
    # 1) Re-key artifacts first (safe-by-ordering; idempotent on retry).
    total = 0
    for kind in KINDS:
        res = db[kind].update_many(
            {"owner": guest_id}, {"$set": {"owner": target_owner, "updated_at": _now()}}
        )
        result["moved"][kind] = res.modified_count
        total += res.modified_count
    result["total"] = total
    result["migrated"] = True
    # 2) Data is safe under the Clerk account — now retire the guest record.
    db[GUESTS].delete_one({"guestId": guest_id})
    log.info("Migrated guest %s -> %s (%d artifacts).", guest_id, target_owner, total)
    return result


def purge_orphan_guest_data() -> int:
    """Delete artifacts whose guest owner no longer exists (TTL can't cascade).

    Mongo's TTL index drops an inactive guest *document*, but its artifacts remain. This
    sweep removes any artifact owned by a ``guest_*`` id that has no live guest record.
    Returns the number of artifacts removed.
    """
    db = _get_db()
    if db is None:
        return 0
    live = [g["guestId"] for g in db[GUESTS].find({}, {"guestId": 1, "_id": 0})]
    removed = 0
    for kind in KINDS:
        res = db[kind].delete_many({"owner": {"$regex": "^" + GUEST_PREFIX, "$nin": live}})
        removed += res.deleted_count
    return removed


def start_guest_janitor() -> None:
    """Spawn a daemon that purges orphaned guest artifacts now and then periodically.

    Safe to call once at startup; a no-op when persistence is disabled. The first sweep
    runs immediately so a server that restarts after a guest expired cleans up promptly.
    """
    if not config.PERSISTENCE_ENABLED:
        return

    def _run() -> None:
        while True:
            try:
                removed = purge_orphan_guest_data()
                if removed:
                    log.info("Guest janitor removed %d orphaned artifact(s).", removed)
            except Exception as exc:  # noqa: BLE001 — never let the janitor kill its thread
                log.warning("Guest janitor sweep failed: %s", exc)
            time.sleep(max(config.GUEST_PURGE_INTERVAL_SECONDS, 60))

    threading.Thread(target=_run, daemon=True, name="guest-janitor").start()


def save(kind: str, owner: str, doc: dict) -> str | None:
    """Insert an artifact for ``owner`` and return its id (or None if disabled)."""
    db = _get_db()
    if db is None or kind not in KINDS:
        return None
    payload = dict(doc)
    payload["owner"] = owner
    payload.setdefault("created_at", _now())
    payload["updated_at"] = _now()
    return str(db[kind].insert_one(payload).inserted_id)


def list_items(kind: str, owner: str, q: str | None = None, limit: int = 100) -> list[dict]:
    """List an owner's artifacts (summaries, newest first), optionally filtered by ``q``."""
    db = _get_db()
    if db is None or kind not in KINDS:
        return []
    query: dict = {"owner": owner}
    if q:
        rx = {"$regex": re.escape(q), "$options": "i"}
        query["$or"] = [{field: rx} for field in _SEARCH_FIELDS]
    projection = {field: 0 for field in _HEAVY_FIELDS}
    projection["owner"] = 0
    cursor = db[kind].find(query, projection).sort("created_at", -1).limit(limit)
    return [_public(d) for d in cursor]


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
    return db[kind].delete_one({"_id": oid, "owner": owner}).deleted_count > 0


def rename_item(kind: str, owner: str, id_str: str, title: str) -> bool:
    """Rename one artifact (set its title), scoped to ``owner``."""
    db = _get_db()
    oid = _oid(id_str)
    if db is None or kind not in KINDS or oid is None:
        return False
    res = db[kind].update_one(
        {"_id": oid, "owner": owner},
        {"$set": {"title": title, "updated_at": _now()}},
    )
    return res.matched_count > 0


def clear_kind(kind: str, owner: str) -> int:
    """Delete ALL of an owner's artifacts of a kind. Returns the count removed."""
    db = _get_db()
    if db is None or kind not in KINDS:
        return 0
    return db[kind].delete_many({"owner": owner}).deleted_count


# ---- chat-specific helpers (chats accumulate messages over a session) ----
def create_chat(owner: str, meta: dict) -> str | None:
    """Create a chat document (with an empty messages array) and return its id."""
    return save("chats", owner, {**meta, "messages": []})


def append_chat_message(owner: str, chat_id: str, role: str, content: str) -> None:
    """Append a message to an owner's chat and bump its updated_at."""
    db = _get_db()
    oid = _oid(chat_id)
    if db is None or oid is None:
        return
    db["chats"].update_one(
        {"_id": oid, "owner": owner},
        {
            "$push": {"messages": {"role": role, "content": content, "ts": _now().isoformat()}},
            "$set": {"updated_at": _now()},
        },
    )


def set_chat_title(owner: str, chat_id: str, title: str) -> None:
    """Set a chat's title only if it is still untitled (first user message wins)."""
    db = _get_db()
    oid = _oid(chat_id)
    if db is None or oid is None:
        return
    db["chats"].update_one(
        {"_id": oid, "owner": owner, "$or": [{"title": {"$exists": False}}, {"title": ""}, {"title": None}]},
        {"$set": {"title": title}},
    )


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
        log.info("Stored uploaded PDF in MongoDB: %s/%s owner=%s (%d bytes).", subject, filename, owner, len(data))
        return True
    except Exception as exc:  # noqa: BLE001 — persistence must never break the upload itself
        log.warning("Could not store uploaded PDF %s/%s in MongoDB: %s", subject, filename, exc)
        return False


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


# ---- FAISS indexes (per user subject) — persist across restarts without rebuilding ----
# Bundled subjects (Mathematics, Operating System) have their indexes baked into the Docker
# image and never need MongoDB storage. User-uploaded subjects live on ephemeral disk
# (Render free tier) so we persist their 3 index files in GridFS and restore them on startup
# — the user gets their subject back instantly without re-embedding all their PDFs.
INDEX_BUCKET: str = "subject_indexes"
_INDEX_FILES: tuple[str, ...] = ("index.faiss", "index.pkl", "manifest.json")


def _index_gridfs():
    """Return a GridFS bucket for FAISS index files, or None when persistence is off."""
    database = _get_db()
    if database is None:
        return None
    from gridfs import GridFS

    return GridFS(database, collection=INDEX_BUCKET)


def store_subject_index(key: str, index_dir: "Path") -> bool:
    """Persist a user subject's FAISS index files to MongoDB so they survive a restart.

    ``key`` is the internal subject key (e.g. ``user__user_abc123__Notes``). All three files
    (index.faiss, index.pkl, manifest.json) are replaced atomically: old versions are deleted
    first so repeated uploads don't accumulate stale blobs. Returns True if all files were
    stored, False if persistence is unavailable or the write failed.
    """
    fs = _index_gridfs()
    if fs is None:
        return False
    try:
        for fname in _INDEX_FILES:
            path = index_dir / fname
            if not path.is_file():
                continue
            for old in fs.find({"subject_key": key, "filename": fname}):
                fs.delete(old._id)
            fs.put(path.read_bytes(), filename=fname, subject_key=key, saved_at=_now())
        log.info("Stored FAISS index in MongoDB: %s.", key)
        return True
    except Exception as exc:  # noqa: BLE001 — persistence must never break the upload
        log.warning("Could not store FAISS index %s in MongoDB: %s", key, exc)
        return False


def load_subject_index(key: str, index_dir: "Path") -> bool:
    """Restore a user subject's FAISS index files from MongoDB to disk.

    Writes all 3 files to ``index_dir``, creating it if needed. Returns True only when all
    3 files were restored (a partial restore is not useful for FAISS.load_local). A missing
    key or any read failure returns False so the caller can fall back to rebuilding.
    """
    fs = _index_gridfs()
    if fs is None:
        return False
    try:
        restored = 0
        for fname in _INDEX_FILES:
            doc = fs.find_one({"subject_key": key, "filename": fname})
            if doc is None:
                continue
            index_dir.mkdir(parents=True, exist_ok=True)
            (index_dir / fname).write_bytes(doc.read())
            restored += 1
        if restored == len(_INDEX_FILES):
            log.info("Restored FAISS index from MongoDB: %s.", key)
            return True
        return False
    except Exception as exc:  # noqa: BLE001 — a restore failure falls back to rebuild
        log.warning("Could not restore FAISS index %s from MongoDB: %s", key, exc)
        return False


def close() -> None:
    """Close the Mongo client (called on app shutdown)."""
    global _client
    if _client is not None:
        _client.close()
