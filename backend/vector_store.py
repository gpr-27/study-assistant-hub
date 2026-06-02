"""Persistent, per-subject FAISS vector store.

Each subject's lecture PDFs are embedded once with the local (keyless) fastembed
model and saved under ``faiss_index/<subject>/``. On later runs the index is
loaded from disk instead of being rebuilt, so chat is fast and embeddings are
recomputed only when a subject's PDFs (or the embedding model) actually change.

A small ``manifest.json`` next to each index records a fingerprint of the source
PDFs plus the embedding/chunk settings; if the current files no longer match the
fingerprint, the index is rebuilt automatically.

The on-disk indexes are gitignored (``faiss_index/``): they are rebuildable,
machine/model-specific binaries that persist locally across runs but are not
committed to the repository.

Every embedding/index object is created lazily inside a function so that
importing this module never downloads a model and never needs a Groq key.
"""
import contextlib
import json
import logging
import threading
from pathlib import Path

try:
    import fcntl  # POSIX advisory file locks (Linux container, macOS dev).
except ImportError:  # pragma: no cover — non-POSIX (e.g. Windows): no cross-process lock.
    fcntl = None

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.config import EMBEDDING_MODEL, FAISS_INDEX_DIR
from backend.documents import list_pdf_paths

log = logging.getLogger(__name__)

_CHUNK_SIZE: int = 1500
_CHUNK_OVERLAP: int = 150
_MANIFEST_NAME: str = "manifest.json"

# In-process cache so repeat subject sessions within one run never touch disk.
_cache: dict[str, FAISS] = {}

# Serialises index builds so the startup warm thread and a chat request can't
# build the same subject twice at once.
_build_lock = threading.Lock()


class VectorStoreError(Exception):
    """Raised when an index cannot be built (e.g. no readable text in the PDFs)."""


def _embeddings() -> FastEmbedEmbeddings:
    """Build the local (keyless) embedding model used for FAISS retrieval."""
    return FastEmbedEmbeddings(model_name=EMBEDDING_MODEL)


def warm_embeddings() -> None:
    """Pre-load the local embedding model (downloads ~once) so the first build
    isn't a cold wait. Safe to call in a background thread; errors are swallowed
    because this is best-effort warmup."""
    try:
        _embeddings().embed_query("warm up")
    except Exception:
        log.debug("Embedding warmup failed (non-fatal).", exc_info=True)


def build_index_from_paths(paths: list[Path]) -> FAISS:
    """Load every PDF, split into chunks, and build an in-memory FAISS store.

    Used both to persist a subject index and for one-off uploaded-PDF chat.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=_CHUNK_SIZE,
        chunk_overlap=_CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = []
    for path in paths:
        docs = PyPDFLoader(str(path)).load()
        for doc in docs:
            doc.metadata["source"] = str(path)
            doc.metadata["document_type"] = "content"
        chunks.extend(splitter.split_documents(docs))

    if not chunks:
        raise VectorStoreError("No readable text was found in the selected PDF(s).")

    return FAISS.from_documents(chunks, _embeddings())


def _subject_index_dir(subject: str) -> Path:
    return FAISS_INDEX_DIR / subject


def _manifest_path(subject: str) -> Path:
    return _subject_index_dir(subject) / _MANIFEST_NAME


def _fingerprint(subject: str) -> dict:
    """Describe a subject's current PDFs + index settings.

    A change in any PDF (added, removed, resized, re-saved) or in the embedding
    model / chunk settings produces a different fingerprint, which triggers a
    rebuild on the next load.
    """
    files = []
    for path in list_pdf_paths(subject):
        st = path.stat()
        files.append({"name": path.name, "size": st.st_size, "mtime": st.st_mtime})
    return {
        "model": EMBEDDING_MODEL,
        "chunk_size": _CHUNK_SIZE,
        "chunk_overlap": _CHUNK_OVERLAP,
        "files": files,
    }


def _read_manifest(subject: str) -> dict | None:
    path = _manifest_path(subject)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _write_manifest(subject: str, fingerprint: dict) -> None:
    _manifest_path(subject).write_text(
        json.dumps(fingerprint, indent=2, sort_keys=True), encoding="utf-8"
    )


def _index_is_fresh(subject: str) -> bool:
    """True if a saved index exists and its manifest matches the current PDFs."""
    index_dir = _subject_index_dir(subject)
    if not (index_dir / "index.faiss").is_file():
        return False
    return _read_manifest(subject) == _fingerprint(subject)


def _load_from_disk(subject: str) -> FAISS:
    """Load a previously saved FAISS index for a subject.

    ``allow_dangerous_deserialization`` is safe here: these files are written by
    this app on the local machine, never fetched from an untrusted source.
    """
    return FAISS.load_local(
        str(_subject_index_dir(subject)),
        _embeddings(),
        allow_dangerous_deserialization=True,
    )


def _build_and_save(subject: str) -> tuple[FAISS, int]:
    """Build a subject index from its PDFs, persist it, and write the manifest.

    Caller must hold ``_build_lock``.
    """
    index_dir = _subject_index_dir(subject)
    index_dir.mkdir(parents=True, exist_ok=True)
    log.info("Building embeddings for subject '%s'…", subject)
    db = build_index_from_paths(list_pdf_paths(subject))
    db.save_local(str(index_dir))
    _write_manifest(subject, _fingerprint(subject))
    chunk_count = db.index.ntotal
    log.info("Built embeddings for subject '%s' (%d chunks).", subject, chunk_count)
    return db, chunk_count


def build_subject_index(subject: str) -> int:
    """Rebuild a subject's index from scratch, persist it, and cache it.

    Called when a subject's PDFs change (e.g. a new subject is added). Returns
    the number of indexed chunks.
    """
    with _build_lock:
        db, chunk_count = _build_and_save(subject)
        _cache[subject] = db
        return chunk_count


def load_or_build_subject_index(subject: str) -> FAISS:
    """Return a subject's FAISS store, building it once if needed.

    Resolution order: in-process cache → fresh on-disk index → (re)build.
    """
    cached = _cache.get(subject)
    if cached is not None:
        return cached

    with _build_lock:
        cached = _cache.get(subject)
        if cached is not None:
            return cached
        if _index_is_fresh(subject):
            log.info("Loading saved embeddings for subject '%s'.", subject)
            db = _load_from_disk(subject)
        else:
            db, _ = _build_and_save(subject)
        _cache[subject] = db
        return db


def get_subject_retriever(subject: str, k: int):
    """Return a retriever over a subject's persistent index."""
    return load_or_build_subject_index(subject).as_retriever(search_kwargs={"k": k})


def warm_all_subjects() -> None:
    """Build or load every subject's index once (run in a background thread at
    startup). Best-effort: one subject failing must not stop the others."""
    from backend.documents import list_subjects

    for subject in list_subjects():
        try:
            load_or_build_subject_index(subject)
        except Exception:
            log.warning("Could not warm embeddings for subject '%s'.", subject, exc_info=True)
