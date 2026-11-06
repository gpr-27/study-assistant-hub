"""Shared document access for every tool: subjects, PDFs, and text extraction.

Serverless-friendly:
  * No heavy imports at module import time (``pypdf`` is loaded on first PDF read;
    the old ``tiktoken`` dependency is gone — ``clip_to_tokens`` uses a cheap
    approximation).
  * Bundled subjects live in the read-only ``documents/`` tree shipped with the
    deployment. User-uploaded subjects are written to a WRITABLE root: ``documents/``
    itself when it is writable (dev / Docker), otherwise ``$TMPDIR/study-documents``
    (Vercel: only ``/tmp`` is writable). They are re-hydrated on demand from MongoDB
    GridFS when a cold instance has no local copy.

Read access (subjects, PDF listing, text extraction) is used by every tool;
the write helpers (`ensure_subject_dir`, `save_pdf`) back the "Add a Subject" feature.
"""
import hashlib
import io
import os
import re
import shutil
import tempfile
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path

from backend import tmpcache
from backend.config import DOCUMENTS_DIR

_PDF_MAGIC: bytes = b"%PDF-"


_CMAP_MEMO_MAX = 256
_cmap_memo_state: dict = {"installed": False}


def _install_cmap_memo() -> None:
    """Memoise pypdf's ToUnicode CMap parsing (once per process, bounded).

    pypdf re-parses a font's ToUnicode CMap for every page that uses the font, which dominated
    upload CPU (about half of the time to index three 4-page lectures: ``parse_bfrange`` runs
    again for each page). The parse is a pure function of the CMap stream and the font subtype,
    so the result is cached under a SHA-256 of those. Callers get copies, never the cached objects.
    Fails open: if pypdf's internals differ, the stock function is used.
    """
    if _cmap_memo_state["installed"]:
        return
    _cmap_memo_state["installed"] = True
    try:
        from pypdf import _cmap
    except Exception:  # noqa: BLE001 — unknown pypdf layout: keep the stock behaviour
        return
    orig = getattr(_cmap, "_parse_to_unicode", None)
    if orig is None or getattr(orig, "_sah_memo", False):
        return
    memo: "OrderedDict[bytes, tuple[dict, list]]" = OrderedDict()
    lock = threading.Lock()

    def _parse_to_unicode(ft):
        tu = ft.get("/ToUnicode") if hasattr(ft, "get") else None
        if tu is None:
            return orig(ft)
        try:
            data = tu.get_object().get_data()
        except Exception:  # noqa: BLE001 — let pypdf decide how to handle a broken stream
            return orig(ft)
        key = hashlib.sha256(data + str(ft.get("/Subtype", "")).encode()).digest()
        with lock:
            hit = memo.get(key)
            if hit is not None:
                memo.move_to_end(key)
        if hit is not None:
            return dict(hit[0]), list(hit[1])
        res = orig(ft)
        with lock:
            memo[key] = (dict(res[0]), list(res[1]))
            while len(memo) > _CMAP_MEMO_MAX:
                memo.popitem(last=False)
        return res

    _parse_to_unicode._sah_memo = True          # type: ignore[attr-defined]
    _parse_to_unicode._sah_orig = orig          # type: ignore[attr-defined]  (tests compare against it)
    _cmap._parse_to_unicode = _parse_to_unicode  # get_encoding() looks this up at call time


def _pdf_reader(source):
    """Lazily import pypdf (≈100 ms) so importing this module stays near-free."""
    import logging

    from pypdf import PdfReader

    logging.getLogger("pypdf").setLevel(logging.ERROR)   # silence per-font warnings
    _install_cmap_memo()
    if isinstance(source, (str, os.PathLike)):
        # Hand pypdf an open file instead of a path: given a path it slurps the WHOLE file into a BytesIO
        # (200 MB PDF -> +200 MB RSS); a file object is read lazily, range by range.
        fh = open(source, "rb")
        try:
            reader = PdfReader(fh)
        except BaseException:
            fh.close()
            raise
        reader._sah_fh = fh                # closed when the reader is garbage-collected
        return reader
    return PdfReader(source)


# ---- token budgeting (no tiktoken) ----
# cl100k averages ~4 characters per token for English prose; symbols/numbers are
# roughly one token each. We walk word/punctuation pieces and charge ceil(len/4) per
# word and 1 per punctuation char — slightly conservative, so slices stay inside the
# model's per-minute token budget. Whitespace is free (it attaches to the next token).
_PIECE_RE = re.compile(r"\w+|[^\w\s]")


def estimate_tokens(text: str) -> int:
    """Cheap token-count approximation (≈ cl100k, a little conservative)."""
    if not text:
        return 0
    total = 0
    for m in _PIECE_RE.finditer(text):
        n = m.end() - m.start()
        total += 1 if n <= 4 else -(-n // 4)
    return total


def clip_to_tokens(text: str, max_tokens: int) -> str:
    """Return at most (about) the first ``max_tokens`` tokens of ``text``.

    The generation tools (quiz, flashcards, planner, mind map) feed ONE bounded
    slice of a document to the LLM per call instead of splitting the whole thing
    into many calls — this keeps every request under the model's per-minute
    token limit. Token counts are approximated (see ``estimate_tokens``); every piece
    costs >= 1 token and >= 1 character, so text no longer than ``max_tokens``
    characters can never exceed the budget and returns immediately.
    """
    if not text or max_tokens <= 0:
        return "" if max_tokens <= 0 else text
    if len(text) <= max_tokens:
        return text
    total = 0
    for m in _PIECE_RE.finditer(text):
        n = m.end() - m.start()
        total += 1 if n <= 4 else -(-n // 4)
        if total > max_tokens:
            return text[:m.start()].rstrip()
    return text


class DocumentError(Exception):
    """Raised when a requested subject or PDF cannot be found or read, or when a
    subject name / uploaded file is invalid."""


def _safe_component(value: str, *, kind: str) -> str:
    """Return a single, traversal-safe path component or raise DocumentError.

    Subjects and PDF filenames arrive as untrusted request input and are then
    turned into filesystem paths. They MUST be a single path component — never
    contain separators, never be '.'/'..', never be absolute — so a crafted
    value like '../../etc' or '/etc/passwd' can't escape documents/. This guards
    the read path (every tool calls read_pdf with request-supplied names); the
    write path has its own stricter validation in _validate_subject_name.
    """
    cleaned = (value or "").strip()
    if not cleaned:
        raise DocumentError(f"{kind} cannot be empty.")
    if (
        cleaned in {".", ".."}
        or "/" in cleaned
        or "\\" in cleaned
        or "\x00" in cleaned
        or Path(cleaned).name != cleaned        # absolute path or has separators
    ):
        raise DocumentError(f"Invalid {kind}: {value!r}")
    return cleaned


# ---- user-subject namespacing ----
# Uploaded subjects are stored as  documents/user__{owner}__{subject}/
# Bundled subjects have NO prefix:  documents/Mathematics/
# Clerk user ids look like user_abc123 (single underscores only), so the double-
# underscore separator is unambiguous and safe in a directory name.
_USER_PREFIX = "user__"


def _user_subject_key(owner: str, subject: str) -> str:
    """Internal directory name for a user-owned subject."""
    return f"{_USER_PREFIX}{owner}__{subject}"


def _parse_user_key(dirname: str) -> tuple[str, str] | None:
    """Parse (owner, display_name) from a user-subject dir name, or None."""
    if not dirname.startswith(_USER_PREFIX):
        return None
    rest = dirname[len(_USER_PREFIX):]
    owner, sep, subj = rest.partition("__")
    if not sep or not owner or not subj:
        return None
    return owner, subj


def _user_root() -> Path:
    """Writable root for user-uploaded subjects.

    ``documents/`` when it is writable (dev, Docker); otherwise a per-instance
    directory under the system temp dir (Vercel's only writable location). Resolved
    on every call so tests that patch ``DOCUMENTS_DIR`` are honoured.
    """
    root = DOCUMENTS_DIR
    try:
        if root.exists() and os.access(root, os.W_OK):
            return root
        if not root.exists() and os.access(root.parent, os.W_OK):
            return root
    except OSError:
        pass
    return Path(tempfile.gettempdir()) / "study-documents"


def _public_subject(subject: str) -> str:
    """Validate a CALLER-SUPPLIED subject display name (request input).

    Internal keys (``user__<owner>__<name>``) and anything containing the reserved ``__``
    separator are rejected, so a caller can never address another user's namespace by
    passing an internal key. Trusted internal code uses ``_subject_dir_internal``.
    """
    safe = _safe_component(subject, kind="subject")
    if safe.startswith(_USER_PREFIX) or "__" in safe:
        raise DocumentError(f"Invalid subject: {subject!r}")
    return safe


def _is_bundled(safe: str) -> bool:
    return (DOCUMENTS_DIR / safe).is_dir()


def _hidden_for(owner: str | None, safe: str) -> bool:
    """True when ``owner`` has hidden the standard subject ``safe`` (view preference only)."""
    if not owner:
        return False
    try:
        from backend import prefs

        return prefs.is_hidden(owner, safe)
    except Exception:  # noqa: BLE001 - preferences must never break retrieval
        return False


def subject_internal_key(subject: str, owner: str | None = None) -> str:
    """Return the internal directory/cache/index key for a subject.

    Bundled subjects resolve to their own name; user-uploaded subjects resolve to
    ``user__{owner}__{subject}`` — ALWAYS, whenever an owner is supplied and the name
    is not a bundled subject, even if no local copy exists yet (a cold serverless
    instance has no local files; the index/PDFs live in MongoDB). This keeps one
    user's key from ever collapsing to a bare name another user could also hit.
    """
    safe = _public_subject(subject)
    if _is_bundled(safe):
        if _hidden_for(owner, safe):
            raise DocumentError(f"Unknown subject: {subject}")
        return safe
    if owner:
        return _user_subject_key(owner, safe)
    return safe


# Short TTL cache of "which subjects does MongoDB know for this owner" so listing
# subjects never costs a DB round-trip per request.
_REMOTE_TTL_S = 30.0
_remote_cache: dict[str, tuple[float, list[str]]] = {}


def _remote_subjects(owner: str) -> list[str]:
    hit = _remote_cache.get(owner)
    now = time.monotonic()
    if hit and now - hit[0] < _REMOTE_TTL_S:
        return hit[1]
    names: list[str] = []
    try:
        from backend import db

        names = list(db.list_user_subjects(owner))
    except Exception:  # noqa: BLE001 — listing must never fail because Mongo is down
        names = []
    if len(_remote_cache) > 256:
        _remote_cache.clear()
    _remote_cache[owner] = (now, names)
    return names


def forget_remote_subjects(owner: str | None = None) -> None:
    """Drop the remote-subject listing cache (call after adding/removing a subject)."""
    if owner is None:
        _remote_cache.clear()
    else:
        _remote_cache.pop(owner, None)


def list_subjects(owner: str | None = None) -> list[str]:
    """Return subject display names visible to owner.

    Always includes bundled subjects (no prefix).  When owner is provided, also
    includes that owner's uploaded subjects (local copies plus those known only to
    MongoDB), displayed without the internal prefix.
    """
    subjects: set[str] = set()
    if DOCUMENTS_DIR.exists():
        for entry in DOCUMENTS_DIR.iterdir():
            if not entry.is_dir() or entry.name.startswith("."):
                continue
            parsed = _parse_user_key(entry.name)
            if parsed is None:
                subjects.add(entry.name)
            elif owner is not None and parsed[0] == owner:
                subjects.add(parsed[1])
    if owner is not None:
        root = _user_root()
        if root != DOCUMENTS_DIR and root.exists():
            for entry in root.iterdir():
                parsed = _parse_user_key(entry.name) if entry.is_dir() else None
                if parsed and parsed[0] == owner:
                    subjects.add(parsed[1])
        subjects.update(_remote_subjects(owner))
        try:
            from backend import prefs

            hidden = {h.casefold() for h in prefs.hidden_standard(owner)}
        except Exception:  # noqa: BLE001
            hidden = set()
        if hidden:
            shipped = {e.name.casefold() for e in DOCUMENTS_DIR.iterdir()
                       if e.is_dir() and not e.name.startswith(".") and _parse_user_key(e.name) is None} \
                if DOCUMENTS_DIR.exists() else set()
            subjects = {s for s in subjects if not (s.casefold() in hidden and s.casefold() in shipped)}
    return sorted(subjects)


# Hydration is single-flight per target directory and ATOMIC: the PDFs are written into a hidden
# sibling temp directory and renamed into place with ``os.replace``, so a concurrent request can
# never list/read/index a half-written subject (the directory either does not exist yet or is
# complete). Losers of the race wait on the lock, then simply find the finished directory.
_hydrate_guard = threading.Lock()
_hydrate_locks: dict[str, list] = {}          # target path -> [lock, refcount]


class _HydrateLock:
    def __init__(self, key: str) -> None:
        self.key = key

    def __enter__(self):
        with _hydrate_guard:
            ent = _hydrate_locks.setdefault(self.key, [threading.Lock(), 0])
            ent[1] += 1
            self._ent = ent
        ent[0].acquire()
        return self

    def __exit__(self, *exc) -> None:
        self._ent[0].release()
        with _hydrate_guard:
            self._ent[1] -= 1
            if self._ent[1] <= 0 and _hydrate_locks.get(self.key) is self._ent:
                del _hydrate_locks[self.key]


def _hydrate_user_subject(owner: str, safe: str, target: Path) -> bool:
    """Pull a user's PDFs for ``safe`` from MongoDB into ``target`` (cold instance).

    Atomic and single-flight (see above). Returns True when ``target`` is a complete directory
    afterwards (written by this call or by a concurrent one).
    """
    with _HydrateLock(str(target)):
        if target.is_dir():                       # another request finished while we waited
            return True
        try:
            from backend import db

            files = db.fetch_uploaded_pdfs(owner, safe)
        except Exception:  # noqa: BLE001
            return False
        tmp = target.parent / f".hydrate-{uuid.uuid4().hex[:12]}"
        wrote = False
        try:
            for filename, data in files or []:
                try:
                    name = _safe_component(filename, kind="PDF name")
                except DocumentError:
                    continue
                tmp.mkdir(parents=True, exist_ok=True)
                if isinstance(data, Path):                # a big original streamed to a temp file by db.fetch_*
                    shutil.move(str(data), str(tmp / name))
                else:
                    (tmp / name).write_bytes(data)
                wrote = True
            if not wrote:
                return False
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.replace(tmp, target)           # atomic: target does not exist (checked above)
            except OSError:
                # target appeared meanwhile (e.g. an upload created it): keep theirs, drop ours
                return target.is_dir()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        tmpcache.touch(target)
        _enforce_tmp_caps()
        return True


def _enforce_tmp_caps() -> None:
    try:
        from backend import vector_store

        vector_store.enforce_tmp_caps()
    except Exception:  # noqa: BLE001 - housekeeping only
        pass


def _subject_dir(subject: str, owner: str | None = None) -> Path:
    """Resolve a caller-supplied subject display name to its directory.

    Checks bundled subjects first; then the owner's uploaded subjects (local copy,
    else re-hydrated from MongoDB). Internal keys are rejected (see ``_public_subject``).
    """
    safe = _public_subject(subject)
    direct = DOCUMENTS_DIR / safe
    if direct.is_dir():
        if _hidden_for(owner, safe):
            raise DocumentError(f"Unknown subject: {subject}")
        return direct
    if owner:
        user_key = _user_subject_key(owner, safe)
        for root in (DOCUMENTS_DIR, _user_root()):
            user_path = root / user_key
            if user_path.is_dir():
                tmpcache.touch(user_path)
                return user_path
        target = _user_root() / user_key
        if _hydrate_user_subject(owner, safe, target):
            return target
    raise DocumentError(f"Unknown subject: {subject}")


def _subject_dir_internal(key: str) -> Path:
    """TRUSTED internal use only: resolve an already-derived internal key (never request input)."""
    safe = _safe_component(key, kind="subject")
    for root in (DOCUMENTS_DIR, _user_root()):
        if (root / safe).is_dir():
            tmpcache.touch(root / safe)
            return root / safe
    raise DocumentError(f"Unknown subject: {key}")


def _pdf_path(subject: str, filename: str, owner: str | None = None) -> Path:
    path = _subject_dir(subject, owner) / _safe_component(filename, kind="PDF name")
    if not path.is_file():
        raise DocumentError(f"PDF not found: {subject}/{filename}")
    return path


def list_pdf_names(subject: str, owner: str | None = None) -> list[str]:
    """Return the .pdf filenames available for a subject."""
    return sorted(
        f.name
        for f in _subject_dir(subject, owner).iterdir()
        if f.is_file() and f.suffix.lower() == ".pdf"
    )


def list_pdf_paths(subject: str, owner: str | None = None) -> list[Path]:
    """Return absolute paths to every PDF for a subject."""
    return [_pdf_path(subject, name, owner) for name in list_pdf_names(subject, owner)]


def page_count(subject: str, filename: str, owner: str | None = None) -> int:
    """Return the number of pages in a PDF."""
    return len(_pdf_reader(str(_pdf_path(subject, filename, owner))).pages)


def pdf_signature(subject: str, filename: str, owner: str | None = None) -> tuple[float, int]:
    """Return a (mtime, size) fingerprint for a subject PDF.

    Used by callers that cache work derived from a PDF (e.g. planner summaries):
    when the file is replaced, the signature changes and the cache misses, so
    stale results are never served.
    """
    st = _pdf_path(subject, filename, owner).stat()
    return (st.st_mtime, st.st_size)


# Extracted text is CPU-heavy (pypdf: ~350 ms for a 1.5 MB lecture PDF, GIL-bound) and every study
# tool re-reads the same few PDFs, so parsed text is memoised per instance, keyed by file identity
# (path, mtime_ns, size): a replaced file can never serve stale text. Bounded in entries AND total
# characters; concurrent first reads of one file parse it once (per-key lock, no stampede).
_TEXT_CACHE_MAX_ENTRIES = 64
_TEXT_CACHE_MAX_CHARS = 6_000_000
_text_cache: dict = {}                      # insertion-ordered: oldest first
_text_cache_chars = 0
_text_cache_lock = threading.Lock()
_text_key_locks: dict = {}


def _text_cache_get(key: tuple) -> str | None:
    with _text_cache_lock:
        text = _text_cache.get(key)
        if text is not None:
            _text_cache[key] = _text_cache.pop(key)       # mark most-recently used
        return text


def _text_cache_put(key: tuple, text: str) -> None:
    global _text_cache_chars
    with _text_cache_lock:
        old = _text_cache.pop(key, None)
        if old is not None:
            _text_cache_chars -= len(old)
        _text_cache[key] = text
        _text_cache_chars += len(text)
        while len(_text_cache) > 1 and (len(_text_cache) > _TEXT_CACHE_MAX_ENTRIES
                                        or _text_cache_chars > _TEXT_CACHE_MAX_CHARS):
            _text_cache_chars -= len(_text_cache.pop(next(iter(_text_cache))))


def clear_text_cache() -> None:
    global _text_cache_chars
    with _text_cache_lock:
        _text_cache.clear()
        _text_key_locks.clear()
        _text_cache_chars = 0


def read_pdf(subject: str, filename: str, owner: str | None = None) -> str:
    """Extract and return the full text of a PDF (empty pages tolerated). Memoised per file version."""
    path = _pdf_path(subject, filename, owner)
    st = path.stat()
    key = (str(path), st.st_mtime_ns, st.st_size)
    hit = _text_cache_get(key)
    if hit is not None:
        return hit
    with _text_cache_lock:
        if len(_text_key_locks) > 256:
            _text_key_locks.clear()
        lock = _text_key_locks.setdefault(key, threading.Lock())
    with lock:                                    # single-flight: one parse, the rest wait and reuse it
        hit = _text_cache_get(key)
        if hit is not None:
            return hit
        reader = _pdf_reader(str(path))
        text = "\n".join((page.extract_text() or "") for page in reader.pages).strip()
        _text_cache_put(key, text)
        return text


def extract_pdf_pages(source) -> list[str]:
    """Return the text of each page of a PDF (path or bytes), '' for empty/unreadable
    pages. Used by the indexer, which needs page numbers for citations."""
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    else:
        source = str(source)
    reader = _pdf_reader(source)
    pages: list[str] = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:  # noqa: BLE001 — one broken page must not sink the document
            pages.append("")
    return pages


def extract_pdf_text(data: bytes) -> str:
    """Extract text from in-memory PDF bytes (for one-off uploads, no save)."""
    reader = _pdf_reader(io.BytesIO(data))
    text = "\n".join((page.extract_text() or "") for page in reader.pages)
    return text.strip()


# ---- write helpers (back the "Add a Subject" feature) ----
def _validate_subject_name(name: str) -> str:
    """Return a cleaned subject name, or raise on anything unsafe.

    Subjects are folder names under documents/, so the name must not contain
    path separators, traversal, or the internal ``__`` separator used for user
    subject namespacing.
    """
    cleaned = name.strip()
    if not cleaned:
        raise DocumentError("Subject name cannot be empty.")
    if cleaned.startswith("."):
        raise DocumentError("Subject name cannot start with a dot.")
    if "/" in cleaned or "\\" in cleaned or cleaned in {".", ".."}:
        raise DocumentError(f"Subject name contains invalid characters: {name!r}")
    if "__" in cleaned:
        raise DocumentError("Subject name cannot contain '__' (reserved for internal use).")
    return cleaned


def ensure_subject_dir(subject: str, owner: str | None = None) -> Path:
    """Validate a subject name and create its documents/ folder if needed.

    When owner is provided the subject is stored under the user-namespaced path
    (``<user root>/user__{owner}__{subject}/``); otherwise it goes to the flat
    bundled path (``documents/{subject}/``).
    """
    safe = _validate_subject_name(subject)
    if owner:
        path = _user_root() / _user_subject_key(owner, safe)
    else:
        path = DOCUMENTS_DIR / safe
    path.mkdir(parents=True, exist_ok=True)
    if owner:
        forget_remote_subjects(owner)
    return path


def _sanitize_pdf_filename(filename: str) -> str:
    """Reduce an uploaded filename to a safe basename ending in .pdf."""
    base = Path(filename).name.strip()
    base = re.sub(r"[^A-Za-z0-9 ._()-]", "_", base)
    if not base or base in {".", ".."}:
        base = "document.pdf"
    if not base.lower().endswith(".pdf"):
        base += ".pdf"
    return base


def _unique_pdf_name(subject_dir: Path, filename: str) -> str:
    """Return a filename that does not collide with an existing PDF, adding a
    numeric suffix (e.g. 'Lec-1 (2).pdf') rather than overwriting."""
    candidate = filename
    stem, suffix = Path(candidate).stem, Path(candidate).suffix
    counter = 2
    while (subject_dir / candidate).exists():
        candidate = f"{stem} ({counter}){suffix}"
        counter += 1
    return candidate


def _write_atomic(path: Path, data: bytes) -> None:
    """Write ``data`` so a concurrent reader sees either nothing or the whole file (the temp
    name does not end in .pdf, so listings never pick it up)."""
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}.part")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass


def save_pdf(subject: str, filename: str, data: bytes, owner: str | None = None) -> str:
    """Save uploaded PDF bytes and return the stored filename.

    When owner is provided the PDF is saved under the user-namespaced subject
    directory (``documents/user__{owner}__{subject}/``).
    Raises DocumentError for empty or non-PDF content.
    """
    if not data:
        raise DocumentError(f"Uploaded file {filename!r} is empty.")
    if not data.startswith(_PDF_MAGIC):
        raise DocumentError(f"Uploaded file {filename!r} is not a valid PDF.")

    subject_dir = ensure_subject_dir(subject, owner)
    stored_name = _unique_pdf_name(subject_dir, _sanitize_pdf_filename(filename))
    _write_atomic(subject_dir / stored_name, data)
    tmpcache.touch(subject_dir)
    return stored_name


def save_pdf_file(subject: str, filename: str, src: Path, owner: str | None = None) -> str:
    """Like :func:`save_pdf` but MOVES an already-assembled temp file into the subject directory
    (no bytes in memory; an atomic rename when on the same filesystem, else a streamed copy)."""
    src = Path(src)
    size = src.stat().st_size
    if not size:
        raise DocumentError(f"Uploaded file {filename!r} is empty.")
    with open(src, "rb") as fh:
        if not fh.read(len(_PDF_MAGIC)).startswith(_PDF_MAGIC):
            raise DocumentError(f"Uploaded file {filename!r} is not a valid PDF.")
    subject_dir = ensure_subject_dir(subject, owner)
    stored_name = _unique_pdf_name(subject_dir, _sanitize_pdf_filename(filename))
    dest = subject_dir / stored_name
    try:
        os.replace(src, dest)
    except OSError:
        tmp = dest.with_name(dest.name + f".{uuid.uuid4().hex[:8]}.tmp")
        shutil.copyfile(src, tmp)
        os.replace(tmp, dest)
    tmpcache.touch(subject_dir)
    return stored_name


def restore_uploaded_pdf(subject: str, filename: str, data: bytes, owner: str | None = None) -> bool:
    """Write a previously-stored uploaded PDF back to its subject directory.

    Used at startup to re-hydrate an ephemeral documents/ dir (e.g. Render's free tier)
    from MongoDB.  When owner is provided the PDF is written to the user-namespaced path.
    Idempotent (skips if the file is already present). Returns True if a file was written,
    False if skipped or the names are unsafe.
    """
    try:
        safe_subject = _validate_subject_name(subject)
        safe_name = _safe_component(filename, kind="PDF name")
    except DocumentError:
        return False
    if owner:
        subject_dir = _user_root() / _user_subject_key(owner, safe_subject)
    else:
        subject_dir = DOCUMENTS_DIR / safe_subject
    target = subject_dir / safe_name
    if target.exists():
        return False
    subject_dir.mkdir(parents=True, exist_ok=True)
    _write_atomic(target, data)
    return True
