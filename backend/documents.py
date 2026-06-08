"""Shared document access for every tool: subjects, PDFs, and text extraction.

Replaces the byte-for-byte duplicated quiz/flashcards `document_utils.py`,
and uses `pypdf` (not the abandoned PyPDF2).

Read access (subjects, PDF listing, text extraction) is used by every tool;
the write helpers (`ensure_subject_dir`, `save_pdf`) back the "Add a Subject"
feature, which saves uploaded PDFs into `documents/<subject>/`.
"""
import io
import re
from pathlib import Path

import tiktoken
from pypdf import PdfReader

from backend.config import DOCUMENTS_DIR

_PDF_MAGIC: bytes = b"%PDF-"
_ENCODING = None


def _token_encoding() -> "tiktoken.Encoding":
    """Lazily load the cl100k_base tokenizer (local, no network or API key)."""
    global _ENCODING
    if _ENCODING is None:
        _ENCODING = tiktoken.get_encoding("cl100k_base")
    return _ENCODING


def clip_to_tokens(text: str, max_tokens: int) -> str:
    """Return at most the first ``max_tokens`` tokens of ``text``.

    The generation tools (quiz, flashcards, planner, mind map) feed ONE bounded
    slice of a document to the LLM per call instead of splitting the whole thing
    into many calls — this keeps every request under the model's per-minute
    token limit. Measured with the cl100k_base tokenizer.
    """
    if not text:
        return text
    enc = _token_encoding()
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return enc.decode(tokens[:max_tokens])


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


def subject_internal_key(subject: str, owner: str | None = None) -> str:
    """Return the internal directory/cache key for a subject.

    Bundled subjects resolve to their own name; user-uploaded subjects resolve to
    ``user__{owner}__{subject}``.  The result is used as the FAISS index dir name
    and the in-process LRU cache key so bundled and user indexes never collide.
    """
    if not DOCUMENTS_DIR.exists():
        return subject
    safe = _safe_component(subject, kind="subject")
    if (DOCUMENTS_DIR / safe).is_dir():
        return safe
    if owner:
        user_key = _user_subject_key(owner, safe)
        if (DOCUMENTS_DIR / user_key).is_dir():
            return user_key
    return safe


def list_subjects(owner: str | None = None) -> list[str]:
    """Return subject display names visible to owner.

    Always includes bundled subjects (no prefix).  When owner is provided, also
    includes that owner's uploaded subjects (displayed without the internal prefix).
    """
    if not DOCUMENTS_DIR.exists():
        return []
    subjects: list[str] = []
    for entry in DOCUMENTS_DIR.iterdir():
        if not entry.is_dir() or entry.name.startswith("."):
            continue
        parsed = _parse_user_key(entry.name)
        if parsed is None:
            subjects.append(entry.name)
        elif owner is not None and parsed[0] == owner:
            subjects.append(parsed[1])
    return sorted(subjects)


def _subject_dir(subject: str, owner: str | None = None) -> Path:
    """Resolve a subject display name to its directory.

    Checks bundled subjects first; then user-owned subjects if owner is given.
    Also accepts internal keys (``user__{owner}__{subject}``) passed directly.
    """
    safe = _safe_component(subject, kind="subject")
    direct = DOCUMENTS_DIR / safe
    if direct.is_dir():
        return direct
    if owner:
        user_path = DOCUMENTS_DIR / _user_subject_key(owner, safe)
        if user_path.is_dir():
            return user_path
    raise DocumentError(f"Unknown subject: {subject}")


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
    return len(PdfReader(str(_pdf_path(subject, filename, owner))).pages)


def pdf_signature(subject: str, filename: str, owner: str | None = None) -> tuple[float, int]:
    """Return a (mtime, size) fingerprint for a subject PDF.

    Used by callers that cache work derived from a PDF (e.g. planner summaries):
    when the file is replaced, the signature changes and the cache misses, so
    stale results are never served.
    """
    st = _pdf_path(subject, filename, owner).stat()
    return (st.st_mtime, st.st_size)


def read_pdf(subject: str, filename: str, owner: str | None = None) -> str:
    """Extract and return the full text of a PDF (empty pages tolerated)."""
    reader = PdfReader(str(_pdf_path(subject, filename, owner)))
    text = "\n".join((page.extract_text() or "") for page in reader.pages)
    return text.strip()


def extract_pdf_text(data: bytes) -> str:
    """Extract text from in-memory PDF bytes (for one-off uploads, no save)."""
    reader = PdfReader(io.BytesIO(data))
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
    (``documents/user__{owner}__{subject}/``); otherwise it goes to the flat
    bundled path (``documents/{subject}/``).
    """
    safe = _validate_subject_name(subject)
    if owner:
        path = DOCUMENTS_DIR / _user_subject_key(owner, safe)
    else:
        path = DOCUMENTS_DIR / safe
    path.mkdir(parents=True, exist_ok=True)
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
    (subject_dir / stored_name).write_bytes(data)
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
        subject_dir = DOCUMENTS_DIR / _user_subject_key(owner, safe_subject)
    else:
        subject_dir = DOCUMENTS_DIR / safe_subject
    target = subject_dir / safe_name
    if target.exists():
        return False
    subject_dir.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return True
