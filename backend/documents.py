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


def list_subjects() -> list[str]:
    """Return subject folder names under documents/ (hidden folders skipped)."""
    if not DOCUMENTS_DIR.exists():
        return []
    return sorted(
        entry.name
        for entry in DOCUMENTS_DIR.iterdir()
        if entry.is_dir() and not entry.name.startswith(".")
    )


def _subject_dir(subject: str) -> Path:
    path = DOCUMENTS_DIR / _safe_component(subject, kind="subject")
    if not path.is_dir():
        raise DocumentError(f"Unknown subject: {subject}")
    return path


def _pdf_path(subject: str, filename: str) -> Path:
    path = _subject_dir(subject) / _safe_component(filename, kind="PDF name")
    if not path.is_file():
        raise DocumentError(f"PDF not found: {subject}/{filename}")
    return path


def list_pdf_names(subject: str) -> list[str]:
    """Return the .pdf filenames available for a subject."""
    return sorted(
        f.name
        for f in _subject_dir(subject).iterdir()
        if f.is_file() and f.suffix.lower() == ".pdf"
    )


def list_pdf_paths(subject: str) -> list[Path]:
    """Return absolute paths to every PDF for a subject."""
    return [_pdf_path(subject, name) for name in list_pdf_names(subject)]


def page_count(subject: str, filename: str) -> int:
    """Return the number of pages in a PDF."""
    return len(PdfReader(str(_pdf_path(subject, filename))).pages)


def pdf_signature(subject: str, filename: str) -> tuple[float, int]:
    """Return a (mtime, size) fingerprint for a subject PDF.

    Used by callers that cache work derived from a PDF (e.g. planner summaries):
    when the file is replaced, the signature changes and the cache misses, so
    stale results are never served.
    """
    st = _pdf_path(subject, filename).stat()
    return (st.st_mtime, st.st_size)


def read_pdf(subject: str, filename: str) -> str:
    """Extract and return the full text of a PDF (empty pages tolerated)."""
    reader = PdfReader(str(_pdf_path(subject, filename)))
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
    path separators or traversal, must not be hidden, and must be non-empty.
    """
    cleaned = name.strip()
    if not cleaned:
        raise DocumentError("Subject name cannot be empty.")
    if cleaned.startswith("."):
        raise DocumentError("Subject name cannot start with a dot.")
    if "/" in cleaned or "\\" in cleaned or cleaned in {".", ".."}:
        raise DocumentError(f"Subject name contains invalid characters: {name!r}")
    return cleaned


def ensure_subject_dir(subject: str) -> Path:
    """Validate a subject name and create its documents/ folder if needed."""
    path = DOCUMENTS_DIR / _validate_subject_name(subject)
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


def save_pdf(subject: str, filename: str, data: bytes) -> str:
    """Save uploaded PDF bytes under documents/<subject>/ and return the stored
    filename. Raises DocumentError for empty or non-PDF content."""
    if not data:
        raise DocumentError(f"Uploaded file {filename!r} is empty.")
    if not data.startswith(_PDF_MAGIC):
        raise DocumentError(f"Uploaded file {filename!r} is not a valid PDF.")

    subject_dir = ensure_subject_dir(subject)
    stored_name = _unique_pdf_name(subject_dir, _sanitize_pdf_filename(filename))
    (subject_dir / stored_name).write_bytes(data)
    return stored_name
