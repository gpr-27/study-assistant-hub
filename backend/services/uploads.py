"""Resumable chunked PDF upload, assembled server-side.

Why: Vercel Functions reject request bodies above ~4.5 MB (HTTP 413 before reaching us), so a
large PDF cannot travel in one request. The browser instead sends it as small raw chunks
(<= ``CHUNK_SIZE``, well under the limit) and asks us to assemble + index it:

  create    POST   /api/subjects/uploads                         -> upload_id, chunk_size, total_chunks
  chunk     PUT    /api/subjects/uploads/{id}/chunks/{index}     -> {received, total}   (idempotent)
  status    GET    /api/subjects/uploads/{id}                    -> which chunks are stored (resume)
  complete  POST   /api/subjects/uploads/{id}/complete           -> same response as the multipart add
  abort     DELETE /api/subjects/uploads/{id}

Design notes:

* **Storage** -- MongoDB (collections ``upload_sessions`` + ``upload_chunks``, both with a 1 h TTL
  that is refreshed on every chunk, so an abandoned upload vanishes on its own). Without MongoDB
  (local dev) a bounded directory under the OS temp dir is used behind the same interface. On
  Vercel the temp store is refused: instances do not share ``/tmp``.
* **Owner isolation** -- ``upload_id`` is ``secrets.token_urlsafe`` (unguessable), validated against
  a strict regex before it is ever used in a query or a path, and EVERY store call is keyed by
  ``(owner, upload_id)``. Another owner's id is indistinguishable from an unknown one (404).
* **Never trust the client** -- the filename is reduced to a safe basename; chunk bounds/sizes are
  derived from the declared size, which is checked against the same limits as the multipart path;
  the assembled bytes go through the exact same ``subjects.add_subject`` pipeline (PDF magic,
  pypdf parse, page limit, dedupe, indexing budget, partial indexing), so the response shape and
  semantics are identical.
* **Completion** is atomic (a compare-and-set ``open -> completing`` with a stale-lock timeout),
  so concurrent completes cannot both run; a finished upload keeps its small result document
  until the TTL so a retried ``complete`` (lost response) returns the same result.
* **Caps** -- per owner: ``MAX_OPEN_UPLOADS`` concurrent open uploads and ``MAX_PENDING_BYTES``
  declared bytes in flight.
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import secrets
import shutil
import tempfile
import threading
import time
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Protocol

from backend import config, db
from backend.services import subjects as subjects_service
from backend.services.subjects import SubjectError, _env_int

log = logging.getLogger(__name__)

# ---- limits -------------------------------------------------------------------------------
CHUNK_SIZE = _env_int("UPLOAD_CHUNK_BYTES", 3_000_000)        # < 3.5 MB: under Vercel's 4.5 MB body cap
if CHUNK_SIZE > 3_500_000:                                     # never let an env typo exceed the platform cap
    CHUNK_SIZE = 3_500_000
UPLOAD_TTL_S = 3600                                            # abandoned upload lifetime (since last chunk)
MAX_OPEN_UPLOADS = _env_int("UPLOAD_MAX_OPEN", 8)              # per owner (the SPA sends files one at a time)
MAX_PENDING_BYTES = _env_int("UPLOAD_MAX_PENDING_MB", 1024) * 1024 * 1024   # per owner, declared sizes
LOCK_STALE_S = 75.0                                            # a "completing" lock older than this is retaken
TEMP_MAX_BYTES = _env_int("UPLOAD_TEMP_MAX_MB", 256) * 1024 * 1024   # whole temp store (dev fallback)

_ID_RE = re.compile(r"^[A-Za-z0-9_-]{20,64}$")
_SHA_RE = re.compile(r"^[0-9a-fA-F]{64}$")
OPEN, COMPLETING, DONE = "open", "completing", "done"


# ---- helpers ------------------------------------------------------------------------------
def safe_filename(filename: str) -> str:
    """Reduce a client-supplied name to a short, printable basename (never a path)."""
    base = re.sub(r"[\x00-\x1f\x7f]", "", str(filename or "")).replace("\\", "/").rsplit("/", 1)[-1].strip()
    base = base[-150:]
    if base in {"", ".", ".."}:
        base = "document.pdf"
    return base


def total_chunks_for(size: int, chunk_size: int) -> int:
    return max(1, math.ceil(size / chunk_size))


def expected_chunk_len(session: dict, index: int) -> int:
    total, cs, size = session["total_chunks"], session["chunk_size"], session["size"]
    return cs if index < total - 1 else size - cs * (total - 1)


def _valid_id(upload_id: str) -> bool:
    return isinstance(upload_id, str) and bool(_ID_RE.match(upload_id))


def _public(session: dict) -> dict:
    return {"upload_id": session["_id"], "chunk_size": session["chunk_size"],
            "total_chunks": session["total_chunks"]}


# ---- stores -------------------------------------------------------------------------------
class Store(Protocol):
    def owner_load(self, owner: str) -> tuple[int, int]: ...
    def create(self, session: dict) -> None: ...
    def get(self, owner: str, upload_id: str) -> dict | None: ...
    def put_chunk(self, owner: str, upload_id: str, index: int, data: bytes) -> int: ...
    def chunk_lengths(self, owner: str, upload_id: str) -> dict[int, int]: ...
    def read_all(self, owner: str, upload_id: str, total: int) -> bytes: ...
    def assemble(self, owner: str, upload_id: str, total: int, out) -> tuple[int, str]: ...
    def try_lock(self, owner: str, upload_id: str) -> tuple[str, dict | None]: ...
    def unlock(self, owner: str, upload_id: str) -> None: ...
    def finish(self, owner: str, upload_id: str, result: dict) -> None: ...
    def delete(self, owner: str, upload_id: str) -> bool: ...


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _expired(created_at) -> bool:
    if isinstance(created_at, datetime):
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        return created_at < _utcnow() - timedelta(seconds=UPLOAD_TTL_S)
    return False


class MongoStore:
    """Sessions + chunks in MongoDB. Every filter carries ``owner``."""

    def __init__(self, database) -> None:
        self.sessions = database["upload_sessions"]
        self.chunks = database["upload_chunks"]

    def owner_load(self, owner: str) -> tuple[int, int]:
        cutoff = _utcnow() - timedelta(seconds=UPLOAD_TTL_S)
        rows = list(self.sessions.find(
            {"owner": owner, "state": {"$in": [OPEN, COMPLETING]}, "created_at": {"$gt": cutoff}}, {"size": 1}))
        return len(rows), sum(int(r.get("size", 0)) for r in rows)

    def create(self, session: dict) -> None:
        self.sessions.insert_one({**session, "created_at": _utcnow()})

    def get(self, owner: str, upload_id: str) -> dict | None:
        doc = self.sessions.find_one({"_id": upload_id, "owner": owner})
        if doc is None or _expired(doc.get("created_at")):
            return None
        return doc

    def put_chunk(self, owner: str, upload_id: str, index: int, data: bytes) -> int:
        from bson.binary import Binary
        doc = {"_id": f"{upload_id}:{index}", "owner": owner, "upload_id": upload_id, "index": index,
               "length": len(data), "data": Binary(data), "created_at": _utcnow()}
        for attempt in (0, 1):
            try:
                self.chunks.replace_one({"_id": doc["_id"], "owner": owner}, doc, upsert=True)
                break
            except Exception as exc:  # noqa: BLE001 -- two concurrent upserts of one id: retry once
                if attempt or "duplicate key" not in str(exc).lower():
                    raise
        self.sessions.update_one({"_id": upload_id, "owner": owner}, {"$set": {"created_at": _utcnow()}})
        return self.chunks.count_documents({"upload_id": upload_id, "owner": owner})

    def chunk_lengths(self, owner: str, upload_id: str) -> dict[int, int]:
        rows = self.chunks.find({"upload_id": upload_id, "owner": owner}, {"index": 1, "length": 1})
        return {int(r["index"]): int(r.get("length", 0)) for r in rows}

    def read_all(self, owner: str, upload_id: str, total: int) -> bytes:
        parts: dict[int, bytes] = {}
        for r in self.chunks.find({"upload_id": upload_id, "owner": owner}):
            parts[int(r["index"])] = bytes(r["data"])
        return b"".join(parts[i] for i in range(total))

    def assemble(self, owner: str, upload_id: str, total: int, out) -> tuple[int, str]:
        """Append the chunks in order to the open binary file ``out``, one chunk in memory at a time.
        Returns ``(bytes_written, sha256_hex)``."""
        import hashlib
        h, n = hashlib.sha256(), 0
        for i in range(total):
            r = self.chunks.find_one({"_id": f"{upload_id}:{i}", "owner": owner})
            if r is None:
                raise SubjectError(f"Upload is incomplete: missing chunk {i}.", 409)
            data = bytes(r["data"])
            out.write(data)
            h.update(data)
            n += len(data)
            del data, r
        return n, h.hexdigest()

    def try_lock(self, owner: str, upload_id: str) -> tuple[str, dict | None]:
        now = time.time()
        from pymongo import ReturnDocument
        doc = self.sessions.find_one_and_update(
            {"_id": upload_id, "owner": owner,
             "$or": [{"state": OPEN}, {"state": COMPLETING, "lock_at": {"$lt": now - LOCK_STALE_S}}]},
            {"$set": {"state": COMPLETING, "lock_at": now}}, return_document=ReturnDocument.AFTER)
        if doc is not None and not _expired(doc.get("created_at")):
            return "ok", doc
        cur = self.get(owner, upload_id)
        if cur is None:
            return "missing", None
        return ("done", cur) if cur.get("state") == DONE else ("busy", cur)

    def unlock(self, owner: str, upload_id: str) -> None:
        self.sessions.update_one({"_id": upload_id, "owner": owner, "state": COMPLETING},
                                 {"$set": {"state": OPEN}, "$unset": {"lock_at": ""}})

    def finish(self, owner: str, upload_id: str, result: dict) -> None:
        self.sessions.update_one({"_id": upload_id, "owner": owner},
                                 {"$set": {"state": DONE, "result": result, "created_at": _utcnow()}})
        self.chunks.delete_many({"upload_id": upload_id, "owner": owner})

    def delete(self, owner: str, upload_id: str) -> bool:
        res = self.sessions.delete_one({"_id": upload_id, "owner": owner})
        self.chunks.delete_many({"upload_id": upload_id, "owner": owner})
        return bool(getattr(res, "deleted_count", 0))


class TempStore:
    """Bounded directory store under the OS temp dir (local dev without MongoDB).

    Layout: ``<tmp>/sah_uploads/<upload_id>/{meta.json,NNNNNN.part}``. ``upload_id`` is validated
    against ``_ID_RE`` (no separators) before it touches a path, and meta.json's owner is checked
    on every access. Single-process only (a module lock guards meta read-modify-write).
    """

    _lock = threading.RLock()

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root else Path(tempfile.gettempdir()) / "sah_uploads"

    # -- paths / meta
    def _dir(self, upload_id: str) -> Path:
        if not _valid_id(upload_id):
            raise SubjectError("Upload not found.", 404)
        return self.root / upload_id

    def _read_meta(self, d: Path) -> dict | None:
        try:
            meta = json.loads((d / "meta.json").read_text("utf-8"))
        except (OSError, ValueError):
            return None
        meta["created_at"] = (d / "meta.json").stat().st_mtime
        return meta

    def _write_meta(self, d: Path, meta: dict) -> None:
        meta = {k: v for k, v in meta.items() if k != "created_at"}
        tmp = d / "meta.json.tmp"
        tmp.write_text(json.dumps(meta), "utf-8")
        os.replace(tmp, d / "meta.json")

    def _all(self) -> list[tuple[Path, dict]]:
        out: list[tuple[Path, dict]] = []
        if not self.root.is_dir():
            return out
        for d in self.root.iterdir():
            if not d.is_dir() or d.is_symlink() or not _valid_id(d.name):
                continue
            meta = self._read_meta(d)
            if meta is None or time.time() - meta["created_at"] > UPLOAD_TTL_S:
                shutil.rmtree(d, ignore_errors=True)         # abandoned / corrupt -> sweep
                continue
            out.append((d, meta))
        return out

    # -- interface
    def owner_load(self, owner: str) -> tuple[int, int]:
        with self._lock:
            mine = [m for _, m in self._all() if m.get("owner") == owner and m.get("state") in (OPEN, COMPLETING)]
            return len(mine), sum(int(m.get("size", 0)) for m in mine)

    def create(self, session: dict) -> None:
        with self._lock:
            total = sum(int(m.get("size", 0)) for _, m in self._all() if m.get("state") in (OPEN, COMPLETING))
            if total + int(session["size"]) > TEMP_MAX_BYTES:
                raise SubjectError("The server's temporary upload space is full; try again shortly.", 503)
            d = self._dir(session["_id"])
            d.mkdir(parents=True, exist_ok=False)
            self._write_meta(d, session)

    def get(self, owner: str, upload_id: str) -> dict | None:
        d = self._dir(upload_id)
        meta = self._read_meta(d)
        if meta is None or meta.get("owner") != owner or time.time() - meta["created_at"] > UPLOAD_TTL_S:
            return None
        return meta

    def put_chunk(self, owner: str, upload_id: str, index: int, data: bytes) -> int:
        with self._lock:
            d = self._dir(upload_id)
            if self.get(owner, upload_id) is None:
                raise SubjectError("Upload not found.", 404)
            tmp = d / f"{index:06d}.tmp"
            tmp.write_bytes(data)
            os.replace(tmp, d / f"{index:06d}.part")
            self._write_meta(d, self._read_meta(d) or {})    # refresh the activity time (mtime)
            return len(list(d.glob("*.part")))

    def chunk_lengths(self, owner: str, upload_id: str) -> dict[int, int]:
        d = self._dir(upload_id)
        if self.get(owner, upload_id) is None:
            return {}
        return {int(p.stem): p.stat().st_size for p in d.glob("*.part")}

    def read_all(self, owner: str, upload_id: str, total: int) -> bytes:
        d = self._dir(upload_id)
        if self.get(owner, upload_id) is None:
            raise SubjectError("Upload not found.", 404)
        return b"".join((d / f"{i:06d}.part").read_bytes() for i in range(total))

    def assemble(self, owner: str, upload_id: str, total: int, out) -> tuple[int, str]:
        import hashlib
        d = self._dir(upload_id)
        if self.get(owner, upload_id) is None:
            raise SubjectError("Upload not found.", 404)
        h, n = hashlib.sha256(), 0
        for i in range(total):
            with open(d / f"{i:06d}.part", "rb") as fh:
                for block in iter(lambda: fh.read(1 << 20), b""):
                    out.write(block)
                    h.update(block)
                    n += len(block)
        return n, h.hexdigest()

    def try_lock(self, owner: str, upload_id: str) -> tuple[str, dict | None]:
        with self._lock:
            meta = self.get(owner, upload_id)
            if meta is None:
                return "missing", None
            state = meta.get("state")
            if state == DONE:
                return "done", meta
            if state == COMPLETING and time.time() - float(meta.get("lock_at", 0)) < LOCK_STALE_S:
                return "busy", meta
            meta.update(state=COMPLETING, lock_at=time.time())
            self._write_meta(self._dir(upload_id), meta)
            return "ok", meta

    def unlock(self, owner: str, upload_id: str) -> None:
        with self._lock:
            meta = self.get(owner, upload_id)
            if meta is not None and meta.get("state") == COMPLETING:
                meta.update(state=OPEN)
                meta.pop("lock_at", None)
                self._write_meta(self._dir(upload_id), meta)

    def finish(self, owner: str, upload_id: str, result: dict) -> None:
        with self._lock:
            meta = self.get(owner, upload_id)
            if meta is None:
                return
            d = self._dir(upload_id)
            meta.update(state=DONE, result=result)
            self._write_meta(d, meta)
            for p in d.glob("*.part"):
                p.unlink(missing_ok=True)

    def delete(self, owner: str, upload_id: str) -> bool:
        with self._lock:
            if self.get(owner, upload_id) is None:
                return False
            shutil.rmtree(self._dir(upload_id), ignore_errors=True)
            return True


def get_store() -> Store:
    database = db.get_db() if db.is_enabled() else None
    if database is not None:
        return MongoStore(database)
    if config.IS_VERCEL:
        # Serverless instances do not share /tmp: chunks would land on different machines.
        raise SubjectError("Large-file upload needs the database (MONGODB_URI) on this deployment.", 503)
    return TempStore()


# ---- service API (all blocking; routers call them via run_in_threadpool) ---------------------
def create_upload(owner: str, subject: str, filename: str, size: int, sha256: str | None = None) -> dict:
    if not owner:
        raise SubjectError("Sign in or continue as a guest to add a subject.", 401)
    clean = subjects_service.resolve_subject_name(subject, owner)
    name = safe_filename(filename)
    if not name.lower().endswith(".pdf"):
        raise SubjectError(f"{name!r} is not a PDF.", 422)
    if size <= 0:
        raise SubjectError(f"{name!r} is empty.", 422)
    if size > subjects_service.MAX_FILE_BYTES:
        raise SubjectError(f"{name!r} is larger than {subjects_service.MAX_FILE_BYTES // 1048576} MB.", 413)
    if sha256 is not None and not _SHA_RE.match(sha256):
        raise SubjectError("sha256 must be 64 hex characters.", 422)
    from backend.services import storage
    storage.check_capacity(int(size))           # 507 when the shared database is nearly full

    store = get_store()
    n_open, pending = store.owner_load(owner)
    if n_open >= MAX_OPEN_UPLOADS:
        raise SubjectError(f"Too many uploads in progress (max {MAX_OPEN_UPLOADS}); finish or cancel one first.", 429)
    if pending + size > MAX_PENDING_BYTES:
        raise SubjectError("Too much data is already waiting to be processed; finish or cancel an upload first.", 429)

    session = {"_id": secrets.token_urlsafe(24), "owner": owner, "subject": clean, "filename": name,
               "size": int(size), "sha256": sha256.lower() if sha256 else None, "chunk_size": CHUNK_SIZE,
               "total_chunks": total_chunks_for(size, CHUNK_SIZE), "state": OPEN}
    store.create(session)
    return _public(session)


def _require(store: Store, owner: str, upload_id: str) -> dict:
    session = store.get(owner, upload_id) if _valid_id(upload_id) else None
    if session is None:
        raise SubjectError("Upload not found.", 404)       # unknown and other-owner look identical
    return session


def upload_status(owner: str, upload_id: str) -> dict:
    store = get_store()
    session = _require(store, owner, upload_id)
    lengths = store.chunk_lengths(owner, upload_id) if session.get("state") != DONE else {}
    return {**_public(session), "state": session.get("state"), "size": session["size"],
            "received": sorted(lengths)}


def put_chunk(owner: str, upload_id: str, index: int, data: bytes) -> dict:
    store = get_store()
    session = _require(store, owner, upload_id)
    if session.get("state") != OPEN:
        raise SubjectError("This upload is already being completed.", 409)
    if not 0 <= index < session["total_chunks"]:
        raise SubjectError(f"Chunk index out of range (0..{session['total_chunks'] - 1}).", 422)
    want = expected_chunk_len(session, index)
    if len(data) != want:
        raise SubjectError(f"Chunk {index} must be exactly {want} bytes (got {len(data)}).", 422)
    received = store.put_chunk(owner, upload_id, index, data)
    return {"received": received, "total": session["total_chunks"]}


_ASSEMBLY = threading.BoundedSemaphore(_env_int("UPLOAD_MAX_CONCURRENT_ASSEMBLIES", 2))
TMP_MARGIN_BYTES = 64 * 1024 * 1024


def _assemble_and_add(store: Store, owner: str, upload_id: str, session: dict, started: float) -> dict:
    """Stream the chunks into a temp file next to the subject folders (same filesystem, so the final
    step is a rename), verify size + sha256 computed on the way, then index it from disk. Peak memory
    is one chunk plus pypdf's working set, never the whole file."""
    from backend import documents

    root = documents._user_root()
    root.mkdir(parents=True, exist_ok=True)
    size = int(session["size"])
    free = shutil.disk_usage(root).free
    if free < size + TMP_MARGIN_BYTES:
        raise SubjectError(
            f"Not enough temporary space on the server to process a {size // 1048576} MB file right now "
            f"({max(0, free) // 1048576} MB free). Try again in a minute or upload a smaller file.", 507)
    work = root / f".assemble-{secrets.token_hex(8)}"
    work.mkdir()
    try:
        path = work / "upload.pdf"
        with open(path, "wb") as out:
            n, digest = store.assemble(owner, upload_id, session["total_chunks"], out)
        if n != size:
            raise SubjectError(f"Uploaded size ({n} bytes) does not match the declared {size}.", 422)
        if session.get("sha256") and digest != session["sha256"]:
            raise SubjectError("The uploaded file is corrupted (checksum mismatch); please retry.", 422)
        src = subjects_service.FileSource(path, n, digest)
        return subjects_service.add_subject(
            session["subject"], [(session["filename"], src)], owner, started=started)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def complete_upload(owner: str, upload_id: str, *, started: float | None = None) -> dict:
    """Assemble, validate and index. Returns the ``AddSubjectResponse`` fields as a dict."""
    started = time.monotonic() if started is None else started
    store = get_store()
    if not _valid_id(upload_id):
        raise SubjectError("Upload not found.", 404)
    status, session = store.try_lock(owner, upload_id)
    if status == "missing":
        raise SubjectError("Upload not found.", 404)
    if status == "done":
        return dict(session.get("result") or {})           # retried complete: same answer
    if status == "busy":
        raise SubjectError("This upload is already being completed; retry in a few seconds.", 409)

    keep = False                       # True -> the upload stays resumable after a failure
    try:
        have = store.chunk_lengths(owner, upload_id)
        missing = [i for i in range(session["total_chunks"]) if i not in have]
        if missing:
            keep = True
            shown = ", ".join(map(str, missing[:20])) + (" …" if len(missing) > 20 else "")
            raise SubjectError(f"Upload is incomplete: missing chunk(s) {shown}.", 409)
        if not _ASSEMBLY.acquire(blocking=False):      # global per-instance cap: bounded /tmp and RAM
            raise SubjectError("The server is busy assembling other uploads; retry in a few seconds.", 503)
        try:
            result = _assemble_and_add(store, owner, upload_id, session, started)
        finally:
            _ASSEMBLY.release()
    except SubjectError as exc:
        if exc.status_code in (409, 429, 503, 507):
            keep = True
        if keep:
            store.unlock(owner, upload_id)
        else:
            store.delete(owner, upload_id)                  # permanent validation failure: free it now
        raise
    except BaseException:
        store.unlock(owner, upload_id)                      # transient/unknown: allow a retry
        raise
    out = asdict(result)
    store.finish(owner, upload_id, out)
    return out


def abort_upload(owner: str, upload_id: str) -> dict:
    store = get_store()
    session = _require(store, owner, upload_id)
    if session.get("state") == COMPLETING and time.time() - float(session.get("lock_at", 0)) < LOCK_STALE_S:
        raise SubjectError("This upload is being completed and can't be cancelled now.", 409)
    store.delete(owner, upload_id)
    return {"aborted": True}
