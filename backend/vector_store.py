"""Persistent, per-subject FAISS vector store.

Each subject's lecture PDFs are embedded once — via the hosted Gemini embeddings
API by default (``EMBEDDING_PROVIDER=gemini``), or the local keyless fastembed
model as a fallback (``EMBEDDING_PROVIDER=local``) — and saved under
``faiss_index/<subject>/``. On later runs the index is loaded from disk instead
of being rebuilt, so chat is fast and embeddings are recomputed only when a
subject's PDFs (or the embedding model) actually change.

A small ``manifest.json`` next to each index records a fingerprint of the source
PDFs plus the embedding/chunk settings; if the current files no longer match the
fingerprint, the index is rebuilt automatically.

Bundled subjects' indexes are committed (pre-baked by ``scripts/prebake_all.py``)
so the server merely LOADS them — no embedding call, no key needed for them at
runtime. User subjects' indexes (``faiss_index/user__*/``, gitignored) are
persisted to MongoDB GridFS and restored on startup instead.

Every embedding/index object is created lazily inside a function so that
importing this module never creates a client and never needs an API key.
"""
import contextlib
import hashlib
import json
import logging
import threading
from collections import OrderedDict
from pathlib import Path

try:
    import fcntl  # POSIX advisory file locks (Linux container, macOS dev).
except ImportError:  # pragma: no cover — non-POSIX (e.g. Windows): no cross-process lock.
    fcntl = None

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend import config
from backend.config import EMBEDDING_DIM, EMBEDDING_MODEL, EMBEDDING_PROVIDER, FAISS_INDEX_DIR
from backend.documents import _USER_PREFIX, list_pdf_paths, subject_internal_key

log = logging.getLogger(__name__)

_CHUNK_SIZE: int = 1500
_CHUNK_OVERLAP: int = 150
_MANIFEST_NAME: str = "manifest.json"

# LRU cap: keep at most this many subject indexes in RAM at once.
# Each FAISS index + the shared embedding model ~ 150–250 MB on the free tier (512 MB).
# 2 concurrent subjects leaves ~100 MB for the Python/FastAPI stack.
_CACHE_MAX: int = 2

# LRU in-process cache (OrderedDict used as an ordered map; most-recently-used at end).
_cache: OrderedDict[str, FAISS] = OrderedDict()

# Serialises index builds so the startup warm thread and a chat request can't
# build the same subject twice at once.
_build_lock = threading.Lock()

# Singleton embedding backend — built once per process, shared across all subjects.
_embeddings_instance: Embeddings | None = None
_embeddings_lock = threading.Lock()


class VectorStoreError(Exception):
    """Raised when an index cannot be built (e.g. no readable text in the PDFs)."""


def _build_embeddings() -> Embeddings:
    """Construct the embedding backend selected by EMBEDDING_PROVIDER.

    Imports are deferred so that the unused backend's heavy dependency (fastembed/
    onnxruntime for local; nothing extra for gemini) is never loaded.
    """
    if EMBEDDING_PROVIDER == "gemini":
        from backend.gemini_embeddings import GeminiEmbeddings

        return GeminiEmbeddings(
            model=EMBEDDING_MODEL,
            api_key=config.GEMINI_API_KEY,
            output_dimensionality=EMBEDDING_DIM,
            fallback_api_key=config.GEMINI_API_KEY_FALLBACK,
        )
    # Legacy local fallback (EMBEDDING_PROVIDER=local): keyless on-device model.
    from langchain_community.embeddings import FastEmbedEmbeddings

    return FastEmbedEmbeddings(model_name=EMBEDDING_MODEL)


def _embeddings() -> Embeddings:
    """Return the process-wide embedding backend, building it once on first call."""
    global _embeddings_instance
    if _embeddings_instance is not None:
        return _embeddings_instance
    with _embeddings_lock:
        if _embeddings_instance is None:
            _embeddings_instance = _build_embeddings()
    return _embeddings_instance


def warm_embeddings() -> None:
    """Warm the embedding backend so the first real request isn't a cold wait.

    For gemini this validates the API key + connectivity with one tiny call; for
    local it triggers the one-time model download. Best-effort: errors swallowed."""
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


def _file_sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of a file's bytes (read in 1 MiB blocks)."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _fingerprint(subject: str) -> dict:
    """Describe a subject's current PDFs + index settings.

    Keyed on each PDF's *content* (size + SHA-256), deliberately NOT its mtime.
    A redeploy, container rebuild, `cp` of the seed documents, or MongoDB
    re-hydration changes file timestamps while leaving the bytes identical; an
    mtime-based fingerprint would then treat the unchanged PDFs as "stale" and
    force a full re-embed on every cold start — and rebuilding an index spikes
    resident memory far past Render's 512 MB free-tier cap, OOM-killing the
    worker (whereas loading a pre-baked index costs almost nothing). A content
    hash is invariant under all of those, so a pre-baked index is recognized as
    fresh and merely loaded. A genuine PDF edit (added, removed, or any byte
    change — even one that keeps the same size) still changes the hash and
    correctly triggers a rebuild.
    """
    files = []
    for path in list_pdf_paths(subject):
        files.append({
            "name": path.name,
            "size": path.stat().st_size,
            "sha256": _file_sha256(path),
        })
    return {
        "provider": EMBEDDING_PROVIDER,
        "model": EMBEDDING_MODEL,
        "dim": EMBEDDING_DIM,
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


def _cache_put(subject: str, db: FAISS) -> None:
    """Insert/refresh subject in the LRU cache, evicting the oldest if over cap."""
    _cache.pop(subject, None)
    _cache[subject] = db
    while len(_cache) > _CACHE_MAX:
        evicted, _ = _cache.popitem(last=False)
        log.info("Evicted FAISS index for subject '%s' from cache (cap=%d).", evicted, _CACHE_MAX)


def _is_user_subject(key: str) -> bool:
    return key.startswith(_USER_PREFIX)


def _save_index_to_mongo(key: str) -> None:
    """Best-effort: persist a user subject's FAISS index to MongoDB."""
    if not _is_user_subject(key):
        return
    from backend import db as mongo_db
    mongo_db.store_subject_index(key, _subject_index_dir(key))


def _restore_index_from_mongo(key: str) -> bool:
    """Try to restore a user subject's FAISS index from MongoDB to disk.

    Returns True when all 3 index files were written and the manifest
    still matches the current PDFs (so the index is fresh).
    """
    if not _is_user_subject(key):
        return False
    from backend import db as mongo_db
    if not mongo_db.load_subject_index(key, _subject_index_dir(key)):
        return False
    return _index_is_fresh(key)


def build_subject_index(subject: str, owner: str | None = None) -> int:
    """Rebuild a subject's index from scratch, persist it, and cache it.

    After building, the index is also saved to MongoDB so user-uploaded
    subjects survive a restart without re-embedding their PDFs.
    """
    key = subject_internal_key(subject, owner)
    with _build_lock:
        db, chunk_count = _build_and_save(key)
        _cache_put(key, db)
    _save_index_to_mongo(key)
    return chunk_count


def load_or_build_subject_index(subject: str, owner: str | None = None) -> FAISS:
    """Return a subject's FAISS store, building it once if needed.

    Resolution order:
      1. In-process LRU cache  (instant)
      2. Fresh on-disk index   (disk → load, no embedding work)
      3. MongoDB restore       (disk was wiped; write files + load, no embedding work)
      4. Rebuild from PDFs     (fallback: embed all PDFs via fastembed)
    """
    key = subject_internal_key(subject, owner)
    cached = _cache.get(key)
    if cached is not None:
        _cache.move_to_end(key)
        return cached

    with _build_lock:
        cached = _cache.get(key)
        if cached is not None:
            _cache.move_to_end(key)
            return cached
        if _index_is_fresh(key):
            log.info("Loading saved embeddings for subject '%s'.", key)
            db = _load_from_disk(key)
        elif _restore_index_from_mongo(key):
            log.info("Restored FAISS index from MongoDB for '%s'.", key)
            db = _load_from_disk(key)
        else:
            log.info("Rebuilding FAISS index for '%s'…", key)
            db, _ = _build_and_save(key)
            _save_index_to_mongo(key)
        _cache_put(key, db)
        return db


def get_subject_retriever(subject: str, k: int, owner: str | None = None):
    """Return a retriever over a subject's persistent index."""
    return load_or_build_subject_index(subject, owner).as_retriever(search_kwargs={"k": k})


def warm_all_subjects() -> None:
    """Build or load every bundled subject's index once (called at Docker build time).
    Best-effort: one subject failing must not stop the others."""
    from backend.documents import list_subjects

    for subject in list_subjects():  # no owner = bundled subjects only
        try:
            load_or_build_subject_index(subject)
        except Exception:
            log.warning("Could not warm embeddings for subject '%s'.", subject, exc_info=True)
