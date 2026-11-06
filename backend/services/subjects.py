"""Add-a-subject service: validate + save uploaded PDFs and index them within a time budget.

Serverless-friendly design:

* Everything here is **bounded**: upload size / file count / page count are capped, the
  index build gets a *time budget* (``INDEX_BUDGET_S``, shared with ``vector_store``) and the
  request returns with ``indexing="partial"`` when not every chunk could be embedded in time.
  Unembedded chunks stay BM25-searchable, and :func:`continue_index` embeds the rest later
  (a few seconds of work per call, safe to repeat).
* All functions are synchronous and CPU/IO-bound — routers must call them off the event loop
  (``run_in_threadpool`` / plain ``def`` endpoints).
* Every storage operation is scoped by ``owner``; a user subject's internal key
  (``user__<owner>__<subject>``) embeds the owner, so one user can never address another's.
"""
from __future__ import annotations

import hashlib
import io
import logging
import os
import re
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from backend import db, documents, prefs, vector_store
from backend.documents import DocumentError, restore_uploaded_pdf, save_pdf

log = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(name, "") or default))
    except ValueError:
        return default


def _env_int0(name: str, default: int) -> int:
    """Like :func:`_env_int` but ``0`` is a legal value (meaning "no limit")."""
    try:
        raw = os.environ.get(name, "")
        return max(0, int(raw)) if raw.strip() != "" else default
    except ValueError:
        return default


# ---- limits (env-overridable; Vercel itself caps a request body at ~4.5 MB, so big files travel
# in <= 3 MB pieces - see services/uploads.py). 0 = unlimited where noted. Documented in
# .env.example and DEPLOYMENT.md; the SPA text is generated from ``public_limits()``. ----
MAX_FILES_PER_REQUEST = _env_int("UPLOAD_MAX_FILES", 50)
MAX_FILE_BYTES = _env_int("UPLOAD_MAX_FILE_MB", 200) * 1024 * 1024
MAX_TOTAL_BYTES = _env_int0("UPLOAD_MAX_TOTAL_MB", 0) * 1024 * 1024          # 0 = no per-upload total cap
MAX_PAGES_PER_PDF = _env_int("UPLOAD_MAX_PAGES_PER_PDF", 2000)
MAX_PAGES_PER_REQUEST = _env_int0("UPLOAD_MAX_PAGES_PER_REQUEST", 0)       # 0 = unlimited (extraction is resumable)
MAX_PDFS_PER_SUBJECT = _env_int("UPLOAD_MAX_PDFS_PER_SUBJECT", 100)
MAX_SUBJECTS_PER_OWNER = _env_int("UPLOAD_MAX_SUBJECTS_PER_OWNER", 100)
MAX_SUBJECT_NAME_LEN = 80
# Originals up to this size stay stored (re-readable); bigger PDFs keep only their extracted text
# + chunks + vectors once fully read (MongoDB Atlas M0 is 512 MB in total).
RAW_KEEP_BYTES = _env_int0("RAW_KEEP_MB", 25) * 1024 * 1024
UPLOAD_PARALLEL_CHUNKS = 3                                                   # client-side PUT concurrency


def public_limits() -> dict:
    """The limits the SPA shows and enforces client-side (single source of truth = this module)."""
    from backend import config
    from backend.services import uploads

    mb = 1024 * 1024
    return {
        "max_file_mb": MAX_FILE_BYTES // mb,
        "max_total_mb": MAX_TOTAL_BYTES // mb,                 # 0 = unlimited
        "max_files": MAX_FILES_PER_REQUEST,
        "max_pdfs_per_subject": MAX_PDFS_PER_SUBJECT,
        "max_subjects": MAX_SUBJECTS_PER_OWNER,
        "max_pages_per_pdf": MAX_PAGES_PER_PDF,
        "max_pages_per_upload": MAX_PAGES_PER_REQUEST,         # 0 = unlimited
        "chunk_bytes": uploads.CHUNK_SIZE,
        "single_post_bytes": min(int(config.MAX_BODY_BYTES) - 400_000, 4 * mb),
        "parallel_chunks": UPLOAD_PARALLEL_CHUNKS,
        "raw_keep_mb": RAW_KEEP_BYTES // mb,
        "max_open_uploads": uploads.MAX_OPEN_UPLOADS,
        "max_pending_mb": uploads.MAX_PENDING_BYTES // mb,
    }

REQUEST_DEADLINE_S = 50.0          # whole upload request (function maxDuration is 60 s)
PERSIST_RESERVE_S = 6.0            # time kept back for writing the index to MongoDB
EXTRACT_S_PER_PAGE = 0.05          # coarse pypdf text-extraction estimate used to size the budget
CONTINUE_BUDGET_S = float(os.environ.get("CONTINUE_INDEX_BUDGET_S", "35") or 35)

_NAME_RE = re.compile(r"^[\w .,&'()+#:-]+$")


class FileSource:
    """An uploaded file that lives on disk (assembled from chunks) instead of in memory.

    ``len()`` is its size, so size checks work the same as for ``bytes``; ``sha256`` is computed while
    the chunks are appended. Used by the chunked-upload path so a 200 MB PDF never sits in RAM."""

    def __init__(self, path: Path, size: int, sha256: str) -> None:
        self.path, self.size, self.sha256 = Path(path), int(size), sha256

    def __len__(self) -> int:
        return self.size


def _sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class SubjectError(Exception):
    """Raised when a subject cannot be created / changed. ``status_code`` is the HTTP status."""

    def __init__(self, message: str, status_code: int = 422) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass
class AddSubjectResult:
    subject: str
    saved_pdfs: list[str]
    chunk_count: int
    indexing: str = "complete"             # "complete" | "partial"
    n_embedded: int = 0
    skipped_duplicates: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)      # e.g. scanned-PDF note
    reading: dict = field(default_factory=dict)            # {file: {done, total}} pages still to read


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------
def clean_subject_name(name: str) -> str:
    """Normalise + validate a subject name (create) or path parameter (everything else).

    Rejects separators, control characters, dot-names and the internal ``__`` separator, so a
    name can never be (or smuggle) an internal ``user__<owner>__<subject>`` directory key.
    """
    cleaned = re.sub(r"\s+", " ", (name or "").strip())
    if not cleaned:
        raise SubjectError("Subject name cannot be empty.", 400)
    if len(cleaned) > MAX_SUBJECT_NAME_LEN:
        raise SubjectError(f"Subject name is too long (max {MAX_SUBJECT_NAME_LEN} characters).", 400)
    if cleaned.startswith(".") or cleaned.endswith(".") or "__" in cleaned or not _NAME_RE.match(cleaned):
        raise SubjectError(
            "Subject names may use letters, numbers, spaces and . , & ' ( ) + # : - only "
            "(no slashes, no double underscore, no leading/trailing dot).", 400)
    return cleaned


def _is_bundled(name: str) -> bool:
    try:
        return documents._is_bundled(name)
    except Exception:  # noqa: BLE001
        return False


def _bundled_names() -> set[str]:
    try:
        return {s.casefold() for s in documents.list_subjects(None)}
    except Exception:  # noqa: BLE001
        return set()


def inspect_pdf(filename: str, data: bytes) -> int:
    """Validate one uploaded PDF and return its page count (bounded work, no text extraction).

    Rejects empty / non-PDF / unreadable / encrypted files and PDFs claiming more than
    ``MAX_PAGES_PER_PDF`` pages (a forged page tree can claim millions). pypdf's own
    decompression limits guard against compressed-stream bombs during extraction.
    """
    if not data:
        raise SubjectError(f"Uploaded file {filename!r} is empty.")
    if isinstance(data, FileSource):
        with open(data.path, "rb") as fh:
            head = fh.read(5)
        source = str(data.path)                 # _pdf_reader opens it as a file object (lazy, not slurped)
    else:
        head = data[:5]
        source = io.BytesIO(data)
    if not head.startswith(b"%PDF-"):
        raise SubjectError(f"Uploaded file {filename!r} is not a valid PDF.")
    try:
        reader = documents._pdf_reader(source)
        if reader.is_encrypted:
            raise SubjectError(f"{filename!r} is password-protected; remove the password and retry.")
        pages = len(reader.pages)
    except SubjectError:
        raise
    except Exception as exc:  # noqa: BLE001 — any parse failure means "not a usable PDF"
        raise SubjectError(f"{filename!r} could not be read as a PDF ({type(exc).__name__}).") from exc
    if pages <= 0:
        raise SubjectError(f"{filename!r} has no pages.")
    if pages > MAX_PAGES_PER_PDF:
        raise SubjectError(f"{filename!r} has {pages} pages; the limit is {MAX_PAGES_PER_PDF} per PDF.", 413)
    return pages


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
_active: set[str] = set()
_active_lock = threading.Lock()


def _claim(key: str) -> bool:
    with _active_lock:
        if key in _active:
            return False
        _active.add(key)
        return True


def _release(key: str) -> None:
    with _active_lock:
        _active.discard(key)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _existing_pdfs(name: str, owner: str) -> list[Path]:
    """The subject's current PDFs (hydrates from MongoDB on a cold instance); [] if none yet."""
    try:
        return documents.list_pdf_paths(name, owner)
    except DocumentError:
        return []


def _user_dirs(owner: str, name: str) -> list[Path]:
    key = documents._user_subject_key(owner, name)
    roots = {documents.DOCUMENTS_DIR, documents._user_root()}
    return [r / key for r in roots]


def _rmtree_under(path: Path, roots: set[Path]) -> bool:
    """rmtree ``path`` only if it is a real direct child of one of ``roots`` (defence in depth)."""
    try:
        if path.is_symlink() or not path.is_dir():
            return False
        if path.resolve().parent not in {r.resolve() for r in roots}:
            return False
        shutil.rmtree(path, ignore_errors=True)
        return True
    except OSError:
        return False


def _delete_remote_pdfs(owner: str, name: str, filenames: list[str] | None = None) -> int:
    """Delete this owner's GridFS PDFs for ``name`` (optionally only ``filenames``). Owner-scoped."""
    fs = db._gridfs()
    if fs is None or not owner:
        return 0
    query: dict = {"owner": owner, "subject": name}
    if filenames is not None:
        query["filename"] = {"$in": list(filenames)}
    n = 0
    try:
        for f in list(fs.find(query)):
            fs.delete(f._id)
            n += 1
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not delete stored PDFs for a subject: %s", exc)
    return n


def _rollback(owner: str, name: str, stored: list[str]) -> None:
    """Undo the PDFs saved by a failed add (local + MongoDB); drop the dir if now empty."""
    for d in _user_dirs(owner, name):
        for fn in stored:
            try:
                (d / fn).unlink(missing_ok=True)
            except OSError:
                pass
        try:
            d.rmdir()          # only succeeds when empty
        except OSError:
            pass
    _delete_remote_pdfs(owner, name, stored)
    documents.forget_remote_subjects(owner)


# ---------------------------------------------------------------------------
# add a subject
# ---------------------------------------------------------------------------
def resolve_subject_name(name: str, owner: str) -> str:
    """Validate the target subject name for ``owner`` and return its canonical spelling.

    Rejects unsafe names, built-in names (409) and a new subject past the per-owner cap (409);
    an existing subject of the same name in a different capitalisation is reused. Shared by
    :func:`add_subject` and the chunked-upload ``create`` step so both fail the same way (and
    the chunked path fails *before* any bytes are sent).
    """
    clean = clean_subject_name(name)
    bundled = _bundled_names()
    if clean.casefold() in bundled or _is_bundled(clean):
        raise SubjectError(f"“{clean}” is a built-in subject name; pick a different name.", 409)
    own = [s for s in documents.list_subjects(owner) if s.casefold() not in bundled]
    for existing in own:
        if existing.casefold() == clean.casefold():
            return existing                        # same subject, different capitalisation
    if len(own) >= MAX_SUBJECTS_PER_OWNER:
        raise SubjectError(f"You can have at most {MAX_SUBJECTS_PER_OWNER} subjects; delete one first.", 409)
    return clean


def add_subject(name: str, files: list[tuple[str, bytes]], owner: str | None = None,
                *, started: float | None = None) -> AddSubjectResult:
    """Validate + save uploaded PDFs under a subject and index them within a time budget.

    ``files`` is a list of ``(filename, bytes)``. ``owner`` is REQUIRED (uploads are always
    user-namespaced; the flat bundled path is never written). ``started`` is the request's
    ``time.monotonic()`` start so the embedding budget accounts for time already spent.
    Raises :class:`SubjectError` (with an HTTP ``status_code``) for any user-fixable problem.
    """
    started = time.monotonic() if started is None else started
    if not owner:
        raise SubjectError("Sign in or continue as a guest to add a subject.", 401)
    if not files:
        raise SubjectError("Add at least one PDF to create a subject.", 400)
    if len(files) > MAX_FILES_PER_REQUEST:
        raise SubjectError(f"Too many files (max {MAX_FILES_PER_REQUEST} per upload).", 413)

    clean = resolve_subject_name(name, owner)

    # Validate every file before touching storage.
    total_bytes = sum(len(d) for _, d in files)
    if MAX_TOTAL_BYTES and total_bytes > MAX_TOTAL_BYTES:
        raise SubjectError(f"Upload is too large ({total_bytes // 1048576} MB; max {MAX_TOTAL_BYTES // 1048576} MB).", 413)
    pages_by_file: list[int] = []
    for filename, data in files:
        if len(data) > MAX_FILE_BYTES:
            raise SubjectError(f"{filename!r} is larger than {MAX_FILE_BYTES // 1048576} MB.", 413)
        pages_by_file.append(inspect_pdf(filename, data))
    new_pages = sum(pages_by_file)
    if MAX_PAGES_PER_REQUEST and new_pages > MAX_PAGES_PER_REQUEST:
        raise SubjectError(f"Too many pages in one upload ({new_pages}; max {MAX_PAGES_PER_REQUEST}).", 413)

    key = documents._user_subject_key(owner, clean)
    if not _claim(key):
        raise SubjectError("This subject is already being updated; try again in a moment.", 409)
    stored: list[str] = []
    skipped: list[str] = []
    try:
        # Listing hydrates an existing subject from MongoDB BEFORE we create the dir, so a cold
        # instance never re-indexes only the new PDFs and drops the old ones from the index.
        current = _existing_pdfs(clean, owner)
        by_size: dict[int, list[Path]] = {}
        for p in current:
            try:
                by_size.setdefault(p.stat().st_size, []).append(p)
            except OSError:
                continue
        to_save: list[tuple[str, bytes]] = []
        seen: set[str] = set()
        for filename, data in files:
            digest = data.sha256 if isinstance(data, FileSource) else _sha256(data)
            dup = digest in seen
            for p in ([] if dup else by_size.get(len(data), [])):
                try:
                    if _sha256_path(p) == digest:
                        dup = True
                        break
                except OSError:
                    continue
            if dup:
                skipped.append(filename)
            else:
                seen.add(digest)
                to_save.append((filename, data))
        if len(current) + len(to_save) > MAX_PDFS_PER_SUBJECT:
            raise SubjectError(f"A subject can hold at most {MAX_PDFS_PER_SUBJECT} PDFs.", 413)

        if to_save:
            from backend.services import storage
            storage.check_capacity(sum(len(d) for _, d in to_save))

        if not to_save:                             # everything uploaded was already there
            status = vector_store.index_status(clean, owner)
            return AddSubjectResult(subject=clean, saved_pdfs=[], chunk_count=int(status.get("n_chunks", 0)),
                                    indexing="complete" if status.get("complete") else "partial",
                                    n_embedded=int(status.get("n_embedded", 0)), skipped_duplicates=skipped,
                                    warnings=list(status.get("warnings") or []))

        for filename, data in to_save:
            try:
                if isinstance(data, FileSource):
                    stored_name = documents.save_pdf_file(clean, filename, data.path, owner)
                else:
                    stored_name = save_pdf(clean, filename, data, owner)
            except DocumentError as exc:
                raise SubjectError(str(exc)) from exc
            stored.append(stored_name)
            # Mirror into MongoDB so the PDF survives the (ephemeral) instance.
            if isinstance(data, FileSource):
                saved = db.store_uploaded_pdf_file(clean, stored_name, documents.ensure_subject_dir(clean, owner) / stored_name, owner)
            else:
                saved = db.store_uploaded_pdf(clean, stored_name, data, owner)
            if not saved and db.is_enabled():
                raise SubjectError("Could not save your PDF right now; please retry.", 503)

        documents.forget_remote_subjects(owner)

        total_pages = new_pages + sum(_page_hint(p) for p in current)
        budget = _embed_budget(started, total_pages)
        try:
            chunk_count = vector_store.build_subject_index(
                clean, owner, budget_s=budget, extract_budget_s=_extract_budget(started, budget))
        except vector_store.VectorStoreError as exc:
            raise SubjectError(
                "No readable text was found in the uploaded PDF(s). "
                "If they are scanned documents, they need a text layer (OCR).") from exc
        except SubjectError:
            raise
        except Exception as exc:  # noqa: BLE001
            log.exception("Indexing failed for %s", key)
            raise SubjectError("Could not index the uploaded PDF(s) right now; please retry.", 503) from exc
    except Exception:
        if stored:
            _rollback(owner, clean, stored)
        raise
    finally:
        _release(key)

    indexing, n_embedded, warnings, reading = "partial", 0, [], {}
    _drop_large_raw(clean, owner)
    try:
        status = vector_store.index_status(clean, owner)
        reading = dict(status.get("reading") or {})
        warnings = list(status.get("warnings") or [])
        n_embedded = int(status.get("n_embedded", 0))
        indexing = "complete" if status.get("complete") else "partial"
    except Exception:  # noqa: BLE001 — unknown => let the client call continue-index
        pass
    return AddSubjectResult(subject=clean, saved_pdfs=stored, chunk_count=chunk_count,
                            indexing=indexing, n_embedded=n_embedded, skipped_duplicates=skipped,
                            warnings=warnings, reading=reading)


def _page_hint(path: Path) -> int:
    """Cheap page estimate for an already-stored PDF (~3 KB/page) so we never re-parse it."""
    try:
        return max(1, path.stat().st_size // 3000)
    except OSError:
        return 1


def _embed_budget(started: float, total_pages: int) -> float:
    """Seconds of embedding we can afford: configured budget, capped by what is left of the
    request after text extraction (estimated) and the MongoDB persist reserve."""
    elapsed = time.monotonic() - started
    left = REQUEST_DEADLINE_S - elapsed - PERSIST_RESERVE_S - EXTRACT_S_PER_PAGE * total_pages
    return max(0.0, min(float(vector_store.USER_UPLOAD_EMBED_BUDGET_S), left))


def _extract_budget(started: float, embed_budget: float) -> float:
    """Seconds the request may still spend READING pages (the rest of the deadline minus embedding)."""
    left = REQUEST_DEADLINE_S - (time.monotonic() - started) - PERSIST_RESERVE_S - embed_budget
    return max(3.0, left)


def _drop_large_raw(name: str, owner: str) -> list[str]:
    """Once a big PDF is fully read, keep only its text index: delete the original (local + GridFS).

    Only when MongoDB is on AND the "text only" flag was durably stored first, so the text can never
    be lost; local development (no MongoDB) keeps every original."""
    if not RAW_KEEP_BYTES or not db.is_enabled():
        return []
    try:
        state = vector_store.text_state(name, owner)
        big = []
        for p in _existing_pdfs(name, owner):
            ent = state.get(p.name)
            try:
                size = p.stat().st_size
            except OSError:
                continue
            if ent and ent.get("complete") and ent.get("raw", True) and size > RAW_KEEP_BYTES:
                big.append(p)
        if not big or not vector_store.mark_text_only(name, owner, [p.name for p in big]):
            return []
        names = [p.name for p in big]
        for d in _user_dirs(owner, name):
            for fn in names:
                try:
                    (d / fn).unlink(missing_ok=True)
                except OSError:
                    pass
        _delete_remote_pdfs(owner, name, names)
        documents.forget_remote_subjects(owner)
        return names
    except Exception:  # noqa: BLE001 - dropping the original is an optimisation, never an error
        log.warning("Could not drop large original(s) for %s", name, exc_info=True)
        return []


# ---------------------------------------------------------------------------
# continue indexing
# ---------------------------------------------------------------------------
def continue_index(name: str, owner: str, budget_s: float | None = None) -> dict:
    """Embed more of a partially-indexed user subject within ``budget_s`` (default ~35 s).

    Idempotent and cheap when already complete. Returns
    ``{"subject","indexing","n_chunks","n_embedded","complete","progress","busy"}``.
    ``progress`` is how many chunks this call embedded (0 => quota/keys exhausted, stop polling).
    """
    clean = clean_subject_name(name)
    if _is_bundled(clean):                              # shipped read-only index: nothing to do
        st = vector_store.index_status(clean, None)
        return {"subject": clean, "indexing": "complete" if st.get("complete") else "partial",
                "n_chunks": st.get("n_chunks"), "n_embedded": st.get("n_embedded"),
                "complete": bool(st.get("complete")), "progress": 0, "busy": False}
    budget = max(1.0, min(float(budget_s if budget_s is not None else CONTINUE_BUDGET_S), 45.0))
    key = documents._user_subject_key(owner, clean)
    try:
        before = vector_store.index_status(clean, owner)
    except (vector_store.VectorStoreError, DocumentError) as exc:
        raise SubjectError("Subject not found.", 404) from exc
    busy = False
    after = before
    if not before.get("complete"):
        if _claim(key):                                 # ONE claim per subject: add / continue / delete exclude each other
            try:
                if before.get("reading"):               # still READING pages: resume extraction first
                    t0 = time.monotonic()
                    vector_store.build_subject_index(clean, owner, budget_s=0.0, extract_budget_s=budget * 0.85)
                    budget = max(1.0, budget - (time.monotonic() - t0))
                    _drop_large_raw(clean, owner)
                after = vector_store.continue_subject_index(clean, owner, budget_s=budget)
            except (vector_store.VectorStoreError, DocumentError) as exc:
                raise SubjectError("Subject not found.", 404) from exc
            finally:
                _release(key)
        else:
            busy = True
    return {
        "subject": clean,
        "indexing": "complete" if after.get("complete") else "partial",
        "n_chunks": after.get("n_chunks"), "n_embedded": after.get("n_embedded"),
        "complete": bool(after.get("complete")),
        "progress": (int(after.get("n_embedded", 0)) - int(before.get("n_embedded", 0))
                     + (1 if (before.get("reading") or {}) != (after.get("reading") or {}) else 0)),
        "reading": after.get("reading") or {},
        "busy": busy,
    }


# ---------------------------------------------------------------------------
# list PDFs (page counts cached by file signature)
# ---------------------------------------------------------------------------
_pages_cache: dict[tuple, int] = {}
_PAGES_CACHE_MAX = 1024


def pdf_infos(subject: str, owner: str) -> list[tuple[str, int]]:
    """``[(name, pages)]`` for a subject; page counts are cached per (file, mtime, size)."""
    clean = clean_subject_name(subject)
    internal = documents.subject_internal_key(clean, owner)
    out: list[tuple[str, int]] = []
    for pdf in documents.list_pdf_names(clean, owner):
        try:
            ck = (internal, pdf, documents.pdf_signature(clean, pdf, owner))
            pages = _pages_cache.get(ck)
            if pages is None:
                pages = documents.page_count(clean, pdf, owner)
                if len(_pages_cache) >= _PAGES_CACHE_MAX:
                    _pages_cache.clear()
                _pages_cache[ck] = pages
        except Exception:  # noqa: BLE001 — a broken PDF is listed with 0 pages
            pages = 0
        out.append((pdf, pages))
    return out


# ---------------------------------------------------------------------------
# delete a subject
# ---------------------------------------------------------------------------
def delete_subject(name: str, owner: str) -> dict:
    """Delete one of the caller's own subjects: PDFs (local + MongoDB), index blob, /tmp caches.

    Bundled subjects cannot be deleted. Every step is keyed by ``owner`` so nothing else is
    touched. Raises 404 when the caller has no such subject.
    """
    if not owner:
        raise SubjectError("Sign in or continue as a guest.", 401)
    clean = clean_subject_name(name)
    if _is_bundled(clean):
        raise SubjectError("Built-in subjects cannot be deleted.", 403)
    key = documents._user_subject_key(owner, clean)
    if not _claim(key):
        raise SubjectError("This subject is busy; try again in a moment.", 409)
    try:
        removed_files = _delete_remote_pdfs(owner, clean)
        blobs = db.delete_index_blob(key, owner)
        roots = {documents.DOCUMENTS_DIR, documents._user_root()}
        local = sum(1 for d in _user_dirs(owner, clean) if _rmtree_under(d, roots))
        try:
            vector_store.shutil_rmtree(vector_store._user_cache_dir(key))
            with vector_store._cache_lock:
                vector_store._cache.pop(key, None)
        except Exception:  # noqa: BLE001
            log.debug("Could not clear cached index for %s", key, exc_info=True)
        documents.forget_remote_subjects(owner)
    finally:
        _release(key)
    if not (removed_files or blobs or local):
        raise SubjectError("Subject not found.", 404)
    return {"deleted": clean, "pdfs_removed": removed_files, "index_removed": blobs > 0}


# ---------------------------------------------------------------------------
# inspect / edit one subject: files, remove a file, re-index, rename
# ---------------------------------------------------------------------------
def _own_subject(name: str, owner: str) -> str:
    """Canonical spelling of one of the caller's OWN subjects (bundled -> 403, unknown -> 404)."""
    if not owner:
        raise SubjectError("Sign in or continue as a guest.", 401)
    clean = clean_subject_name(name)
    if clean.casefold() in _bundled_names() or _is_bundled(clean):
        raise SubjectError("Built-in subjects are read-only.", 403)
    for existing in documents.list_subjects(owner):
        if existing.casefold() == clean.casefold():
            return existing
    raise SubjectError("Subject not found.", 404)


def _touch(owner: str) -> None:
    """Bump the owner's write epoch (local + shared) so cached views (Library, subjects) refresh."""
    try:
        db._bump(owner)
        from backend import cache
        cache.bump_epoch(owner)
    except Exception:  # noqa: BLE001
        log.debug("epoch bump failed", exc_info=True)


def subject_files(name: str, owner: str) -> dict:
    """Files of one subject with size / pages / status, plus the index state and the owner's usage."""
    from backend.services import storage

    clean = _own_subject(name, owner)
    try:
        status = vector_store.index_status(clean, owner)
    except (vector_store.VectorStoreError, DocumentError):
        status = {}
    reading = status.get("reading") or {}
    state = {}
    try:
        state = vector_store.text_state(clean, owner)
    except Exception:  # noqa: BLE001
        pass
    files: list[dict] = []
    try:
        infos = pdf_infos(clean, owner)
    except DocumentError:                       # every file is text-only: no PDF folder at all
        infos = []
    for pdf_name, pages in infos:
        try:
            size = next(p.stat().st_size for p in _existing_pdfs(clean, owner) if p.name == pdf_name)
        except (StopIteration, OSError):
            size = 0
        ent = state.get(pdf_name) or {}
        r = reading.get(pdf_name)
        files.append({"name": pdf_name, "size": size, "pages": pages or int(ent.get("total", 0)),
                      "status": "reading" if r else "ready", "text_only": False,
                      "read_pages": r.get("done") if r else None})
    listed = {f["name"] for f in files}
    for fn, ent in state.items():
        if fn not in listed and not ent.get("raw", True):
            files.append({"name": fn, "size": int(ent.get("size", 0)), "pages": int(ent.get("total", 0)),
                          "status": "ready", "text_only": True, "read_pages": None})
    files.sort(key=lambda f: f["name"].casefold())
    return {
        "subject": clean, "files": files,
        "indexing": "complete" if status.get("complete") else "partial",
        "n_chunks": status.get("n_chunks", 0), "n_embedded": status.get("n_embedded", 0),
        "reading": reading, "warnings": status.get("warnings") or [],
        "usage": storage.public_usage(owner),
    }


def _reindex(clean: str, owner: str, started: float) -> dict:
    budget = _embed_budget(started, 0)
    chunk_count = vector_store.build_subject_index(
        clean, owner, budget_s=budget, extract_budget_s=_extract_budget(started, budget))
    _drop_large_raw(clean, owner)
    st = vector_store.index_status(clean, owner)
    return {"subject": clean, "chunk_count": chunk_count, "n_embedded": int(st.get("n_embedded", 0)),
            "indexing": "complete" if st.get("complete") else "partial", "reading": st.get("reading") or {}}


def reindex_subject(name: str, owner: str, *, started: float | None = None) -> dict:
    """Rebuild a subject's index from its files (vectors of unchanged chunks are reused)."""
    started = time.monotonic() if started is None else started
    clean = _own_subject(name, owner)
    key = documents._user_subject_key(owner, clean)
    if not _claim(key):
        raise SubjectError("This subject is busy; try again in a moment.", 409)
    try:
        _existing_pdfs(clean, owner)               # hydrate a cold instance first
        try:
            out = _reindex(clean, owner, started)
            _touch(owner)
            return out
        except vector_store.VectorStoreError as exc:
            raise SubjectError(str(exc)) from exc
    finally:
        _release(key)


def remove_pdf(name: str, owner: str, filename: str, *, started: float | None = None) -> dict:
    """Remove ONE file from a subject and re-index the remainder (unchanged chunks keep their vectors)."""
    started = time.monotonic() if started is None else started
    clean = _own_subject(name, owner)
    fn = (filename or "").strip()
    key = documents._user_subject_key(owner, clean)
    if not _claim(key):
        raise SubjectError("This subject is busy; try again in a moment.", 409)
    try:
        present = {p.name: p for p in _existing_pdfs(clean, owner)}
        state = vector_store.text_state(clean, owner)
        text_only = {n for n, e in state.items() if not e.get("raw", True)}
        if fn not in present and fn not in text_only:
            raise SubjectError("File not found in this subject.", 404)
        if len(set(present) | text_only) <= 1:
            raise SubjectError("A subject needs at least one file; delete the subject instead.", 409)
        if fn in present:
            for d in _user_dirs(owner, clean):
                try:
                    (d / fn).unlink(missing_ok=True)
                except OSError:
                    pass
            _delete_remote_pdfs(owner, clean, [fn])
        vector_store.forget_text(clean, owner, fn)
        documents.forget_remote_subjects(owner)
        _pages_cache.clear()
        try:
            out = _reindex(clean, owner, started)
        except vector_store.VectorStoreError as exc:
            raise SubjectError(str(exc)) from exc
    finally:
        _release(key)
    _touch(owner)
    out["removed"] = fn
    return out


def rename_subject(name: str, new_name: str, owner: str) -> dict:
    """Rename one of the caller's subjects.

    Design: the subject NAME is the key (``user__<owner>__<name>``) of its files, index and caches,
    so a rename MOVES them (no re-embedding: the stored index zip is re-keyed) and relabels the
    ``subject`` field of the owner's saved chats / quizzes / ... with one bounded ``update_many`` per
    collection. This keeps every other module (which addresses subjects by name) unchanged, whereas a
    stable-id scheme would touch every retrieval path. Nothing of another owner is read or written.
    """
    old = _own_subject(name, owner)
    new = clean_subject_name(new_name)
    bundled = _bundled_names()
    if new.casefold() in bundled or _is_bundled(new):
        raise SubjectError(f"“{new}” is a built-in subject name; pick a different name.", 409)
    if new == old:
        return {"subject": old, "renamed": False}
    for existing in documents.list_subjects(owner):
        if existing.casefold() == new.casefold() and existing.casefold() != old.casefold():
            raise SubjectError(f"You already have a subject called “{existing}”.", 409)
    okey, nkey = documents._user_subject_key(owner, old), documents._user_subject_key(owner, new)
    claimed: list[str] = []
    for k in sorted({okey, nkey}):
        if not _claim(k):
            for c in claimed:
                _release(c)
            raise SubjectError("This subject is busy; try again in a moment.", 409)
        claimed.append(k)
    try:
        _existing_pdfs(old, owner)                              # hydrate a cold instance
        try:
            moved_index = vector_store.rename_index(old, new, owner)
        except vector_store.VectorStoreError as exc:
            raise SubjectError(str(exc), 503) from exc
        db.rename_uploaded_subject(owner, old, new)
        roots = {documents.DOCUMENTS_DIR, documents._user_root()}
        for r in roots:
            src, dst = r / okey, r / nkey
            try:
                if src.is_dir() and not src.is_symlink():
                    if dst.exists():
                        shutil.rmtree(dst, ignore_errors=True)
                    src.rename(dst)
            except OSError:
                log.warning("Could not rename the local folder of a subject", exc_info=True)
        relabelled = db.rename_history_subject(owner, old, new)
        documents.forget_remote_subjects(owner)
        _pages_cache.clear()
        from backend.services import storage
        storage.forget_owner(owner)
        _touch(owner)
    finally:
        for c in claimed:
            _release(c)
    return {"subject": new, "renamed": True, "index_moved": bool(moved_index), "history_relabelled": relabelled}


# ---------------------------------------------------------------------------
# standard (bundled) subjects: a per-owner view preference
# ---------------------------------------------------------------------------
def _standard_name(name: str) -> str:
    clean = clean_subject_name(name)
    for b in documents.list_subjects(None):
        if b.casefold() == clean.casefold():
            return b
    raise SubjectError("Standard subject not found.", 404)


def hide_standard(name: str, owner: str) -> dict:
    """Hide a standard subject for THIS owner only (shared files / indexes / history untouched)."""
    if not owner:
        raise SubjectError("Sign in or continue as a guest.", 401)
    canonical = _standard_name(name)
    out = {"hidden": prefs.hide(owner, canonical), "subject": canonical}
    _touch(owner)
    return out


def restore_standard(name: str | None, owner: str) -> dict:
    if not owner:
        raise SubjectError("Sign in or continue as a guest.", 401)
    canonical = _standard_name(name) if name else None
    out = {"hidden": prefs.restore(owner, canonical), "subject": canonical}
    _touch(owner)
    return out


def hidden_standard(owner: str) -> list[str]:
    return prefs.hidden_standard(owner)


# ---------------------------------------------------------------------------
# guest -> account: carry uploaded subjects across (artifacts are re-keyed by db.migrate_guest)
# ---------------------------------------------------------------------------
def migrate_guest_uploads(guest_id: str, target_owner: str) -> int:
    """Re-own a guest's uploaded PDFs + index blobs to ``target_owner``; returns subjects moved.

    A name clash with one of the target's own subjects keeps both ("Name (guest)"). Index
    blobs are re-keyed (``subject_key``/``owner``/``subject``) rather than rebuilt, so the
    moved subject is usable immediately. Idempotent: a second call finds nothing to move.
    """
    if not guest_id or not target_owner or guest_id == target_owner:
        return 0
    try:
        prefs.merge(guest_id, target_owner)          # hidden standard subjects follow the user
    except Exception:  # noqa: BLE001
        log.warning("Could not merge subject preferences", exc_info=True)
    database = db._get_db()
    if database is None:
        return 0
    subjects = sorted(set(db.list_user_subjects(guest_id)))
    if not subjects:
        return 0
    pdf_files = database[f"{db.UPLOADS_BUCKET}.files"]
    idx_files = database[f"{db.INDEX_BUCKET}.files"]
    taken = {s.casefold() for s in db.list_user_subjects(target_owner)}
    moved = 0
    for subject in subjects:
        new_name, n = subject, 1
        while new_name.casefold() in taken:
            n += 1
            new_name = f"{subject} (guest)" if n == 2 else f"{subject} (guest {n - 1})"
        taken.add(new_name.casefold())
        old_key = documents._user_subject_key(guest_id, subject)
        new_key = documents._user_subject_key(target_owner, new_name)
        try:
            pdf_files.update_many({"owner": guest_id, "subject": subject},
                                  {"$set": {"owner": target_owner, "subject": new_name}})
            idx_files.update_many({"owner": guest_id, "subject_key": old_key},
                                  {"$set": {"owner": target_owner, "subject": new_name, "subject_key": new_key}})
            moved += 1
        except Exception:  # noqa: BLE001 — leave it with the guest; a retry resumes
            log.warning("Could not migrate a guest subject upload", exc_info=True)
    documents.forget_remote_subjects(guest_id)
    documents.forget_remote_subjects(target_owner)
    return moved


def purge_orphan_guest_uploads(budget_s: float = 30.0, stats: dict | None = None) -> int:
    """Delete uploads/index blobs of guests whose record has expired (the TTL index only drops
    the guest document). Call from the daily cron. Returns GridFS files removed."""
    database = db._get_db()
    if database is None:
        return 0
    from gridfs import GridFS

    deadline = time.monotonic() + budget_s
    removed = 0
    for bucket in (db.UPLOADS_BUCKET, db.INDEX_BUCKET):
        if time.monotonic() > deadline:
            if stats is not None:
                stats["partial"] = True
            break
        files = database[f"{bucket}.files"]
        owners = files.distinct("owner", {"owner": {"$regex": "^" + db.GUEST_PREFIX}})
        if not owners:
            continue
        live: set[str] = set()
        for i in range(0, len(owners), 500):
            chunk = owners[i:i + 500]
            live.update(g["guestId"] for g in database[db.GUESTS].find({"guestId": {"$in": chunk}}, {"guestId": 1, "_id": 0}))
        fs = GridFS(database, collection=bucket)
        for owner in (o for o in owners if o not in live):
            for f in list(files.find({"owner": owner}, {"_id": 1})):
                if time.monotonic() > deadline:
                    if stats is not None:
                        stats["partial"] = True
                    return removed
                try:
                    fs.delete(f["_id"])
                    removed += 1
                except Exception:  # noqa: BLE001
                    log.warning("Could not delete an orphan guest upload", exc_info=True)
    return removed


# ---------------------------------------------------------------------------
# startup restore (legacy long-lived hosts only; not used on serverless)
# ---------------------------------------------------------------------------
def restore_uploaded_pdfs() -> int:
    """Re-hydrate the user documents root from MongoDB-stored uploads. Returns the count restored.

    Only meaningful on a long-lived host. Each PDF is restored to its owner-namespaced path
    (``user__{owner}__{subject}/``) so uploads remain scoped per user; owner-less legacy
    records are skipped (restoring them to the flat path would publish them as bundled).
    """
    restored = 0
    skipped_ownerless = 0
    for owner, subject, filename, data in db.iter_uploaded_pdfs():
        if not owner:
            skipped_ownerless += 1
            continue
        try:
            if restore_uploaded_pdf(subject, filename, data, owner):
                restored += 1
        except OSError as exc:
            log.warning("Could not restore uploaded PDF %s/%s: %s", subject, filename, exc)
    if skipped_ownerless:
        log.warning("Skipped %d owner-less uploaded PDF(s) from MongoDB (legacy records).", skipped_ownerless)
    if restored:
        log.info("Restored %d uploaded PDF(s) from MongoDB.", restored)
    return restored
