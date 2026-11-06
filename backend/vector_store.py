"""Lightweight, serverless-friendly retrieval: numpy vectors + BM25, no FAISS/langchain.

Layout of one index (a directory, identical for bundled and user subjects)::

    chunks.json    [{"text":…, "source": "Lec-1.pdf", "page": 0}, …]
    vectors.npy    float32 (n_chunks, dim) unit vectors; all-zero row = not embedded yet
    manifest.json  model, dim, chunker settings, counts, source-file fingerprint, version

* **Bundled subjects** (Mathematics, Operating System) are pre-built OFFLINE by
  ``scripts/build_index.py`` into ``data/index/<subject>/`` and shipped read-only.
  At runtime they are memory-mapped lazily; nothing is ever embedded for them in a
  request (a missing bundled index degrades to lexical retrieval built from the PDFs).
* **User subjects** are indexed when uploaded (bounded by a time budget; chunks that
  did not fit are still searchable by BM25 and are finished later by
  :func:`continue_subject_index`), persisted to MongoDB (one GridFS zip per subject key,
  scoped by owner) and cached under ``$TMPDIR`` per instance.
* **Search** = dense cosine (query embedded through ``gemini_embeddings``, bounded to a
  few seconds, LRU-cached) fused with BM25 by reciprocal-rank fusion. If the embedding
  call is unavailable, BM25 alone answers — retrieval never hard-fails.

Importing this module is cheap: numpy / pypdf / requests load on first use.
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
import math
import os
import re
import tempfile
import threading
import time
import uuid
import zipfile
from collections import Counter, OrderedDict
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

from backend import config, tmpcache
from backend.documents import _USER_PREFIX, _parse_user_key, subject_internal_key

log = logging.getLogger(__name__)

# ── tunables ─────────────────────────────────────────────────────────────────
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 150
MIN_CHUNK_ALNUM = 20
INDEX_VERSION = 2

DEFAULT_MODEL = "gemini-embedding-2"
DEFAULT_DIM = 768            # Matryoshka-truncated (unit-norm from the API): 4x smaller than 3072

# Where shipped (read-only) bundled indexes live, and the per-instance writable cache.
BUNDLED_INDEX_DIR: Path = config.ROOT_DIR / "data" / "index"


def _cache_root() -> Path:
    return Path(tempfile.gettempdir()) / "study-index"


USER_UPLOAD_EMBED_BUDGET_S = float(os.environ.get("INDEX_BUDGET_S", "35") or 35)
ONEOFF_EMBED_BUDGET_S = 12.0
QUERY_DEADLINE_S = 3.5
_REVALIDATE_S = 120.0

_CACHE_MAX = 4                       # subjects held in RAM per instance
_CACHE_MAX_BYTES = 128 * 1024 * 1024

# Hybrid fusion weights (reciprocal-rank fusion).
_RRF_K = 60
_W_DENSE = 1.0
_W_BM25 = 0.15      # eval (scripts/eval_retrieval.py): dense alone ~ best; BM25 as a light re-ranker/rescuer
_CANDIDATES = 30
_W_GRAPH = 0.25     # third ranker (concept-graph PPR) weight; dev-tuned by scripts/eval_retrieval.py --graph (see ARCHITECTURE 3a)
_BM25_K1 = 1.5
_BM25_B = 0.75


class VectorStoreError(Exception):
    """Raised when an index cannot be built (e.g. no readable text in the PDFs)."""


# ── Document ─────────────────────────────────────────────────────────────────
class Document:
    """Minimal stand-in for langchain's Document: ``page_content`` + ``metadata``."""

    __slots__ = ("page_content", "metadata")

    def __init__(self, page_content: str, metadata: dict | None = None) -> None:
        self.page_content = page_content
        self.metadata = metadata or {}

    def __repr__(self) -> str:  # pragma: no cover
        return f"Document({self.metadata.get('source')!r} p{self.metadata.get('page')}, {len(self.page_content)} chars)"


# ── chunking (character based; no tokenizer dependency) ──────────────────────
_SEPARATORS = ("\n\n", "\n", ". ", " ")


def _split_piece(text: str, size: int, level: int = 0) -> list[str]:
    if len(text) <= size:
        return [text]
    if level >= len(_SEPARATORS):
        return [text[i:i + size] for i in range(0, len(text), size)]
    sep = _SEPARATORS[level]
    parts = text.split(sep)
    if len(parts) == 1:
        return _split_piece(text, size, level + 1)
    out: list[str] = []
    for i, p in enumerate(parts):
        piece = p + (sep if i < len(parts) - 1 else "")
        out.extend(_split_piece(piece, size, level + 1) if len(piece) > size else [piece])
    return out


def split_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split ``text`` into chunks of ≤ ``size`` chars on natural boundaries, with overlap."""
    text = (text or "").strip()
    if not text:
        return []
    pieces = _split_piece(text, size)
    chunks: list[str] = []
    cur = ""
    for piece in pieces:
        if cur and len(cur) + len(piece) > size:
            chunks.append(cur.strip())
            tail = cur[-overlap:] if overlap else ""
            if tail:                                   # start the overlap on a word boundary
                sp = tail.find(" ")
                tail = tail[sp + 1:] if 0 <= sp < len(tail) - 1 else tail
            cur = (tail + piece) if len(tail) + len(piece) <= size else piece
        else:
            cur += piece
    if cur.strip():
        chunks.append(cur.strip())
    return chunks


def _alnum(s: str) -> int:
    return sum(1 for c in s if c.isalnum())


_MIN_STANDALONE = 350        # a page shorter than this is merged with the next one
_DIGITS_RE = re.compile(r"\d+")


def _boilerplate_lines(pages: list[str]) -> set[str]:
    """Normalised lines that repeat on many pages (slide footers, running headers)."""
    if len(pages) < 5:
        return set()
    seen: Counter = Counter()
    for raw in pages:
        for ln in {_DIGITS_RE.sub("#", ln.strip()) for ln in raw.splitlines() if len(ln.strip()) >= 12}:
            seen[ln] += 1
    cutoff = max(4, int(0.3 * len(pages)))
    return {ln for ln, n in seen.items() if n >= cutoff}


def chunk_pages(pages_by_source: list[tuple[str, list[str]]], size: int = CHUNK_SIZE,
                overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """Chunk ``[(source_name, [page_text, …]), …]``.

    Pages stay separate (precise citations) unless a page is tiny, in which case it is merged
    with the following page(s) up to ``size``. Lines repeated on many pages (footers/headers)
    are dropped. Each chunk is ``{"text", "source", "page"}``; ``page`` is the 0-based first
    page (callers render ``page + 1``).
    """
    chunks: list[dict] = []

    def _emit(source: str, text: str, first: int, last: int) -> None:
        c = {"text": text, "source": source, "page": first}
        if last != first:
            c["page_end"] = last                     # merged tiny pages: cite the whole range
        chunks.append(c)

    for source, pages in pages_by_source:
        boiler = _boilerplate_lines(pages)
        buf, buf_page, buf_last = "", 0, 0
        for page_no, raw in enumerate(pages):
            raw = (raw or "").replace("\x00", "")
            if boiler:
                raw = "\n".join(ln for ln in raw.splitlines() if _DIGITS_RE.sub("#", ln.strip()) not in boiler)
            text = re.sub(r"[ \t]+\n", "\n", raw).strip()
            if not text:
                continue
            for part in split_text(text, size, overlap):
                if buf and (len(buf) >= _MIN_STANDALONE or len(buf) + 2 + len(part) > size):
                    _emit(source, buf, buf_page, buf_last)
                    buf = ""
                if not buf:
                    buf_page = buf_last = page_no
                    buf = part
                else:
                    buf = buf + "\n\n" + part
                    buf_last = page_no
        if buf:
            _emit(source, buf, buf_page, buf_last)
    return [c for c in chunks if _alnum(c["text"]) >= MIN_CHUNK_ALNUM]


# ── BM25 (pure python + numpy) ───────────────────────────────────────────────
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "a an and are as at be but by can do does for from has have how i if in into is it its of on or "
    "so that the their then there these this to was we were what when where which who why will with "
    "you your about explain describe define give me tell between using use used".split()
)


@lru_cache(maxsize=65536)
def _stem(w: str) -> str:
    n = len(w)
    if n > 5 and w.endswith("ing"):
        return w[:-3]
    if n > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if n > 4 and w.endswith("es"):
        return w[:-2]
    if n > 4 and w.endswith("ed"):
        return w[:-2]
    if n > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def tokenize(text: str) -> list[str]:
    return [_stem(w) for w in _TOKEN_RE.findall(text.lower()) if w not in _STOP]


class _BM25:
    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        import numpy as np

        self.n = len(texts)
        postings: dict[str, list[tuple[int, int]]] = {}
        lengths = np.zeros(self.n, dtype=np.float32)
        for i, t in enumerate(texts):
            toks = tokenize(t)
            lengths[i] = len(toks)
            for term, tf in Counter(toks).items():
                postings.setdefault(term, []).append((i, tf))
        self.avgdl = float(lengths.mean()) if self.n else 1.0
        self.k1, self.b = k1, b
        self.norm = (k1 * (1 - b + b * lengths / max(self.avgdl, 1e-6))).astype(np.float32)
        self.index: dict[str, tuple] = {}
        for term, plist in postings.items():
            ids = np.fromiter((p[0] for p in plist), dtype=np.int32, count=len(plist))
            tfs = np.fromiter((p[1] for p in plist), dtype=np.float32, count=len(plist))
            df = len(plist)
            idf = math.log(1 + (self.n - df + 0.5) / (df + 0.5))
            self.index[term] = (ids, tfs, np.float32(idf))

    def scores(self, query: str):
        import numpy as np

        out = np.zeros(self.n, dtype=np.float32)
        for term in set(tokenize(query)):
            hit = self.index.get(term)
            if hit is None:
                continue
            ids, tfs, idf = hit
            out[ids] += idf * (tfs * (self.k1 + 1)) / (tfs + self.norm[ids])
        return out


def graph_enabled() -> bool:
    """Fuse the concept graph as a THIRD ranker inside :meth:`VectorIndex.search` (env ``GRAPH_RAG_RANK``).

    OFF by default: on the bundled eval sets (135 chunks) hybrid already recalls ~90 % of multi-hop gold
    and the graph ranker never improved it (scripts/eval_retrieval.py --graph). The graph is still used for
    global questions, related-concept chips and sub-query coverage; flip this on for large subjects."""
    return os.environ.get("GRAPH_RAG_RANK", "0").strip().lower() in {"1", "true", "yes", "on"}


# ── the index ────────────────────────────────────────────────────────────────
class VectorIndex:
    """Immutable-ish in-memory index: chunks + optional dense vectors + lazy BM25."""

    def __init__(self, key: str, chunks: list[dict], vectors=None, manifest: dict | None = None) -> None:
        import numpy as np

        self.key = key
        self.chunks = chunks
        self.manifest = manifest or {}
        self.model = self.manifest.get("model") or DEFAULT_MODEL
        self.dim = int(self.manifest.get("dim") or 0) or None
        self.vectors = None
        self._has = None
        self.problem: str | None = None        # why dense retrieval is disabled (lexical still works)
        self.n_embedded = 0
        if vectors is not None and len(chunks):
            self.problem = self._validate(vectors, chunks, self.manifest)
            if self.problem is None:
                if vectors.dtype != np.float32:
                    vectors = vectors.astype(np.float32)
                if not vectors.flags["C_CONTIGUOUS"]:
                    vectors = np.ascontiguousarray(vectors)
                self.vectors = vectors
                self.dim = vectors.shape[1]
                if self.manifest.get("complete") is True:
                    self._has = None                       # every row valid
                    self.n_embedded = len(chunks)
                else:
                    self._has = np.asarray(vectors.any(axis=1))
                    self.n_embedded = int(self._has.sum())
            else:
                log.warning("Index '%s': dense retrieval disabled (%s); BM25 only.", key, self.problem)
        self._bm25: _BM25 | None = None
        self._bm25_lock = threading.Lock()
        self._graph = None                    # rag.graph.ConceptGraph, loaded/built lazily (see graph())
        self._graph_state = "unloaded"        # unloaded | stored | built | none
        self._graph_dir: Path | None = None   # directory holding a stored graph.json
        self._graph_lock = threading.Lock()

    @staticmethod
    def _validate(vectors, chunks: list[dict], manifest: dict) -> str | None:
        """None when ``vectors`` can serve dense queries, else a short reason."""
        import numpy as np

        if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
            return f"vector rows {getattr(vectors, 'shape', None)} != {len(chunks)} chunks"
        want = manifest.get("dim")
        if want and int(want) != vectors.shape[1]:
            return f"vector dim {vectors.shape[1]} != manifest dim {want}"
        fmt = manifest.get("format")
        if isinstance(fmt, int) and fmt > INDEX_VERSION:
            return f"index format {fmt} is newer than supported {INDEX_VERSION}"
        if not np.isfinite(vectors).all():
            return "non-finite values in vectors"
        return None

    def __len__(self) -> int:
        return len(self.chunks)

    @property
    def nbytes(self) -> int:
        v = 0 if self.vectors is None else int(self.vectors.nbytes)
        return v + sum(len(c["text"]) for c in self.chunks)

    # -- scoring ------------------------------------------------------------
    def _bm25_scores(self, query: str):
        if self._bm25 is None:
            with self._bm25_lock:
                if self._bm25 is None:
                    self._bm25 = _BM25([c["text"] for c in self.chunks], k1=_BM25_K1, b=_BM25_B)
        return self._bm25.scores(query)

    def _dense_scores(self, qvec):
        import numpy as np

        s = self.vectors @ qvec
        if self._has is not None:
            s = np.where(self._has, s, -np.inf)
        return s

    def graph(self):
        """The concept graph (``rag.graph.ConceptGraph``) of this index, or ``None``.

        Loaded from ``graph.json`` when present AND built from exactly these chunks (signature check);
        otherwise built in memory from the chunk texts (CPU only, no API). Memoised per index object,
        so a re-indexed subject (new object) can never serve a stale graph."""
        if self._graph_state != "unloaded":
            return self._graph
        with self._graph_lock:
            if self._graph_state != "unloaded":
                return self._graph
            from backend.rag import graph as rg

            g = None
            if self._graph_dir is not None:
                try:
                    g = rg.load(json.loads((self._graph_dir / rg.GRAPH_FILE).read_text(encoding="utf-8")), self.chunks)
                except (OSError, ValueError):
                    g = None
                if g is not None:
                    self._graph_state = "stored"
            if g is None and rg.build_enabled() and len(self.chunks) >= 2:
                data = rg.get_or_build(self.chunks, sensitive=bool(self.manifest.get("owner")))
                g = rg.load(data, self.chunks) if data else None
                self._graph_state = "built" if g is not None else "none"
            elif g is None:
                self._graph_state = "none"
            self._graph = g
            return g

    def _graph_ranked(self, query: str, n: int) -> list[tuple[int, float]]:
        try:
            g = self.graph()
            return g.rank_chunks(query, n) if g is not None else []
        except Exception:  # noqa: BLE001 - the graph is an extra ranker; never fail retrieval over it
            log.debug("Graph ranking failed.", exc_info=True)
            return []

    def _doc(self, i: int, score: float, mode: str) -> Document:
        c = self.chunks[i]
        meta = {"source": c["source"], "page": c["page"],
                "document_type": "content", "score": round(float(score), 4), "retrieval": mode}
        if c.get("page_end", c["page"]) != c["page"]:
            meta["page_end"] = c["page_end"]
        return Document(c["text"], meta)

    @staticmethod
    def _top(scores, n: int):
        """Indices of the ``n`` best scores, best first (argpartition: O(N), not a full sort)."""
        import numpy as np

        n = min(n, len(scores))
        if n <= 0:
            return np.empty(0, dtype=np.int64)
        if n >= len(scores):
            part = np.arange(len(scores))
        else:
            part = np.argpartition(-scores, n - 1)[:n]
        return part[np.argsort(-scores[part], kind="stable")]

    def rank(self, query: str, qvec, k: int, mode: str = "hybrid") -> list[tuple[int, float, str]]:
        """Rank chunks for ``query`` -> ``[(chunk_index, score, retrieval_mode), …]``.

        ``mode``: ``"hybrid"`` (dense + BM25 by weighted reciprocal-rank fusion, the default),
        ``"dense"``, ``"bm25"``, ``"graph"`` (concept-graph PPR only) or ``"hybrid+graph"`` (the
        concept graph as a THIRD ranker in the same RRF). ``qvec`` may be ``None`` (no embedding
        available), in which case BM25 (+ graph) answers. Pure and side-effect free apart from lazily
        building BM25 / the graph - the evaluation script drives it directly with cached query vectors.
        """
        import numpy as np

        k = max(1, min(int(k), len(self.chunks)))
        use_graph = mode in ("graph", "hybrid+graph")
        if mode == "hybrid+graph":
            mode = "hybrid"
        n_c = min(_CANDIDATES, len(self.chunks))
        graph_rank = self._graph_ranked(query, n_c) if use_graph else []
        if mode == "graph":
            if not graph_rank:
                return [(i, 0.0, "lexical") for i in range(k)]
            return [(i, sc, "graph") for i, sc in graph_rank[:k]]
        want_dense = mode in ("hybrid", "dense") and qvec is not None and self.vectors is not None
        want_lex = mode in ("hybrid", "bm25") or not want_dense
        lex = None
        if want_lex:
            try:
                lex = self._bm25_scores(query)
            except Exception:  # noqa: BLE001
                log.exception("BM25 failed")
        lex_hit = lex is not None and float(lex.max()) > 0.0

        if not want_dense:
            if not lex_hit and not graph_rank:
                return [(i, 0.0, "lexical") for i in range(k)]   # arbitrary-but-stable context
            if not graph_rank:
                return [(int(i), float(lex[i]), "lexical") for i in self._top(lex, k) if lex[i] > 0]
            fused: dict[int, float] = {}
            if lex_hit:
                for rank, i in enumerate(self._top(lex, n_c)):
                    if lex[i] > 0:
                        fused[int(i)] = fused.get(int(i), 0.0) + 1.0 / (_RRF_K + rank + 1)
            for rank, (i, _sc) in enumerate(graph_rank):
                fused[i] = fused.get(i, 0.0) + _W_GRAPH / (_RRF_K + rank + 1)
            order = sorted(fused, key=lambda i: -fused[i])[:k]
            return [(i, float(lex[i]) if lex is not None else 0.0, "lexical") for i in order]

        dense = self._dense_scores(qvec)
        fused = {}
        for rank, i in enumerate(self._top(dense, n_c)):
            if np.isfinite(dense[i]):
                fused[int(i)] = fused.get(int(i), 0.0) + _W_DENSE / (_RRF_K + rank + 1)
        if mode == "hybrid" and lex_hit:
            for rank, i in enumerate(self._top(lex, n_c)):
                if lex[i] > 0:
                    # a row with no vector yet can only be found lexically -> give it full weight
                    w = _W_BM25 if (self._has is None or self._has[i]) else 2.0 * _W_DENSE
                    fused[int(i)] = fused.get(int(i), 0.0) + w / (_RRF_K + rank + 1)
        for rank, (i, _sc) in enumerate(graph_rank):
            fused[i] = fused.get(i, 0.0) + _W_GRAPH / (_RRF_K + rank + 1)
        order = sorted(fused, key=lambda i: -fused[i])[:k]
        label = "hybrid" if (mode == "hybrid" and (lex_hit or graph_rank)) else "dense"
        return [(i, float(dense[i]) if np.isfinite(dense[i]) else 0.0, label) for i in order]

    def search(self, query: str, k: int = 4, *, use_embeddings: bool = True,
               timeout_s: float | None = None) -> list[Document]:
        """Top-``k`` chunks for ``query``. Never raises for retrieval-quality reasons."""
        if not self.chunks or not (query or "").strip():
            return []
        k = max(1, min(int(k), len(self.chunks)))
        qvec = None
        if use_embeddings and self.vectors is not None and self.n_embedded > 0:
            try:
                from backend import gemini_embeddings as ge

                qvec = ge.embed_query(query, model=self.model, dim=self.dim, deadline_s=min(QUERY_DEADLINE_S, timeout_s) if timeout_s else QUERY_DEADLINE_S)
            except Exception as exc:  # noqa: BLE001 — quota/auth/network → lexical fallback
                log.info("Query embedding unavailable (%s); using BM25 only.", type(exc).__name__)
        return [self._doc(i, sc, m) for i, sc, m in self.rank(query, qvec, k, "hybrid+graph" if graph_enabled() else "hybrid")]

    # langchain-compatible spellings used around the codebase
    def similarity_search(self, query: str, k: int = 4) -> list[Document]:
        return self.search(query, k)

    def as_retriever(self, search_kwargs: dict | None = None) -> "Retriever":
        return Retriever(index=self, k=int((search_kwargs or {}).get("k", 4)))


class Retriever:
    """Callable like a langchain retriever: ``.invoke(query)`` / ``.get_relevant_documents``.

    Holds only ``(key, owner, k)`` for subject retrievers and resolves the index through the
    bounded LRU cache on every call, so a long-lived chat session never pins a big index in
    memory and always sees a re-uploaded subject.
    """

    def __init__(self, key: str | None = None, owner: str | None = None, k: int = 4,
                 index: VectorIndex | None = None, subject: str | None = None) -> None:
        self._key, self._owner, self.k = key, owner, k
        self._index = index
        self._subject = subject

    def _resolve(self) -> VectorIndex | None:
        if self._index is not None:
            return self._index
        try:
            return _get_index(self._key, self._subject or self._key, self._owner)
        except Exception:  # noqa: BLE001 — retrieval must never hard-fail a chat turn
            log.warning("Could not load index for '%s'.", self._key, exc_info=True)
            return None

    def invoke(self, query, *args, timeout_s: float | None = None, **kwargs) -> list[Document]:
        """Hybrid search. ``timeout_s`` bounds the query-embedding step (falls back to BM25)."""
        if not isinstance(query, str):
            query = str(getattr(query, "content", query))
        idx = self._resolve()
        return idx.search(query, self.k, timeout_s=timeout_s) if idx is not None else []

    def invoke_lexical(self, query, *args, **kwargs) -> list[Document]:
        """BM25-only search: never calls the embedding API."""
        if not isinstance(query, str):
            query = str(getattr(query, "content", query))
        idx = self._resolve()
        return idx.search(query, self.k, use_embeddings=False) if idx is not None else []

    def get_relevant_documents(self, query, *args, **kwargs) -> list[Document]:
        return self.invoke(query)

    __call__ = invoke


# ── on-disk format ───────────────────────────────────────────────────────────
def _atomic_write(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + f".{uuid.uuid4().hex[:8]}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def save_index_dir(directory: Path, chunks: list[dict], vectors, manifest: dict) -> None:
    import numpy as np

    directory.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    np.save(buf, np.ascontiguousarray(vectors, dtype=np.float32), allow_pickle=False)
    _atomic_write(directory / "vectors.npy", buf.getvalue())
    _atomic_write(directory / "chunks.json",
                  json.dumps(chunks, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    manifest = _write_graph(directory, chunks, manifest)
    _atomic_write(directory / "manifest.json",
                  json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8"))   # manifest last = commit


def _write_graph(directory: Path, chunks: list[dict], manifest: dict) -> dict:
    """Write ``graph.json`` (concept graph, same pass as the index, no API calls) and stamp the manifest.

    Written BEFORE the manifest (which is the commit point), so a reader never sees a manifest that
    promises a graph that is not there. Failure just means "no stored graph": it is rebuilt lazily."""
    from backend.rag import graph as rg

    data = rg.get_or_build(chunks, sensitive=bool(manifest.get("owner")))
    if data is None:
        manifest.pop("graph", None)
        try:
            (directory / rg.GRAPH_FILE).unlink()
        except OSError:
            pass
        return manifest
    try:
        _atomic_write(directory / rg.GRAPH_FILE, rg.dumps(data))
        manifest["graph"] = rg.stamp(data)
    except OSError:
        log.warning("Could not write the concept graph for '%s'.", manifest.get("key"), exc_info=True)
        manifest.pop("graph", None)
    return manifest


def ensure_graph_in_dir(directory: Path) -> bool:
    """Attach (or refresh) ``graph.json`` for an existing index dir WITHOUT touching vectors or the manifest
    ``version``; no API calls. Returns True when a current graph is present afterwards."""
    manifest = read_manifest(directory)
    if manifest is None or not (directory / "chunks.json").is_file():
        return False
    from backend.rag import graph as rg

    chunks = json.loads((directory / "chunks.json").read_text(encoding="utf-8"))
    st = manifest.get("graph") or {}
    if st.get("sig") == rg.chunks_sig(chunks) and (directory / rg.GRAPH_FILE).is_file():
        return True
    manifest = _write_graph(directory, chunks, manifest)
    _atomic_write(directory / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8"))
    return "graph" in manifest


def read_manifest(directory: Path) -> dict | None:
    try:
        return json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def load_index_dir(directory: Path, key: str, mmap: bool = True) -> VectorIndex:
    import numpy as np

    manifest = read_manifest(directory)
    if manifest is None:
        raise VectorStoreError(f"No index manifest in {directory}")
    chunks = json.loads((directory / "chunks.json").read_text(encoding="utf-8"))
    vec_path = directory / "vectors.npy"
    vectors = np.load(vec_path, mmap_mode="r" if mmap else None, allow_pickle=False) if vec_path.is_file() else None
    if vectors is not None and vectors.dtype != np.float32:
        vectors = vectors.astype(np.float32)
    idx = VectorIndex(key, chunks, vectors, manifest)
    if (directory / "graph.json").is_file():
        idx._graph_dir = directory
    return idx


def pack_index(directory: Path) -> bytes:
    """Zip an index dir for MongoDB; vectors are stored as float16 (half the size)."""
    import numpy as np

    vec = np.load(directory / "vectors.npy", allow_pickle=False)
    vb = io.BytesIO()
    np.save(vb, vec.astype(np.float16), allow_pickle=False)
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        z.writestr("manifest.json", (directory / "manifest.json").read_bytes())
        z.writestr("chunks.json", (directory / "chunks.json").read_bytes())
        z.writestr("vectors.npy", vb.getvalue())
        if (directory / "graph.json").is_file():          # optional member: old blobs simply lack it
            z.writestr("graph.json", (directory / "graph.json").read_bytes())
        if (directory / TEXTS_NAME).is_file():            # optional: resumable-extraction / text-only state
            z.writestr(TEXTS_NAME, (directory / TEXTS_NAME).read_bytes())
    return out.getvalue()


def unpack_index(blob: bytes, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        names = set(z.namelist())
        if not {"manifest.json", "chunks.json", "vectors.npy"} <= names:
            raise VectorStoreError("Corrupt stored index (missing members).")
        for name in ("vectors.npy", "chunks.json", "graph.json", TEXTS_NAME, "manifest.json"):     # manifest last = commit
            if name in names:
                _atomic_write(directory / name, z.read(name))
        if TEXTS_NAME not in names:
            try:
                (directory / TEXTS_NAME).unlink()
            except OSError:
                pass
        if "graph.json" not in names:
            try:
                (directory / "graph.json").unlink()            # never keep a graph from an older blob
            except OSError:
                pass


# ── keys & access control ────────────────────────────────────────────────────
def _is_user_key(key: str) -> bool:
    return key.startswith(_USER_PREFIX)


def _check_access(key: str, owner: str | None) -> None:
    """One user must never read another user's index."""
    if not _is_user_key(key):
        return
    parsed = _parse_user_key(key)
    if parsed is None or not owner or parsed[0] != owner:
        raise VectorStoreError("Subject not available.")


def _safe_dirname(key: str) -> str:
    return re.sub(r"[^A-Za-z0-9._ -]", "_", key)[:120] + "-" + hashlib.sha1(key.encode()).hexdigest()[:8]


def _user_cache_dir(key: str) -> Path:
    return _cache_root() / _safe_dirname(key)


def _bundled_dir(key: str) -> Path:
    return BUNDLED_INDEX_DIR / key


# ── model selection ──────────────────────────────────────────────────────────
def embedding_model() -> str:
    m = (getattr(config, "EMBEDDING_MODEL", None) or "").strip()
    return m if m.startswith("gemini-embedding") or m.startswith("models/gemini-embedding") else DEFAULT_MODEL


def embedding_dim() -> int:
    return int(getattr(config, "EMBEDDING_DIM", None) or DEFAULT_DIM)


# ── fingerprints ─────────────────────────────────────────────────────────────
def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fingerprint_paths(paths: list[Path]) -> list[dict]:
    return [{"name": p.name, "size": p.stat().st_size, "sha256": _sha256_file(p)} for p in paths]


def _chunk_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


# ── building ─────────────────────────────────────────────────────────────────
LOW_TEXT_CHARS_PER_PAGE = 40          # below this a PDF is almost certainly scanned/handwritten
SCANNED_NOTE = "This PDF looks scanned; OCR it via Scan Notes."
_scan_local = threading.local()


def last_scan_report() -> list[dict]:
    """Per-source text stats from the most recent :func:`chunks_from_paths` on this thread."""
    return list(getattr(_scan_local, "report", []))


# ---- resumable text extraction (big PDFs) -------------------------------------------------
# A PDF with more than ``RESUME_MIN_PAGES`` pages is read page-range by page-range across calls:
# the pages read so far are kept in ``texts.json.gz`` next to the index (and travel inside the
# stored index zip), so an interrupted or time-boxed call loses nothing and the next
# ``continue-index`` resumes at the first unread page. Once a file was fully read it is
# "text-only": its original PDF may be dropped (RAW_KEEP_MB) and the index is rebuilt from
# this stored text. Smaller PDFs are read in one go and need no state.
TEXTS_NAME = "texts.json.gz"
RESUME_MIN_PAGES = int(os.environ.get("EXTRACT_RESUME_MIN_PAGES", "150") or 150)


def load_texts(directory: Path | None) -> dict:
    """``{"files": {name: {size, total, pages[], complete, raw}}}`` for an index dir (empty if none)."""
    if directory is not None:
        try:
            import gzip

            data = json.loads(gzip.decompress((directory / TEXTS_NAME).read_bytes()).decode("utf-8"))
            if isinstance(data, dict) and isinstance(data.get("files"), dict):
                return data
        except (OSError, ValueError):
            pass
    return {"files": {}}


def save_texts(directory: Path, state: dict) -> None:
    import gzip

    directory.mkdir(parents=True, exist_ok=True)
    if not state.get("files"):
        try:
            (directory / TEXTS_NAME).unlink()
        except OSError:
            pass
        return
    _atomic_write(directory / TEXTS_NAME,
                  gzip.compress(json.dumps(state, ensure_ascii=False, separators=(",", ":")).encode("utf-8"), 6))


def _extract_range(path: Path, ent: dict, deadline: float | None) -> None:
    """Append page texts to ``ent['pages']`` until the PDF ends or ``deadline`` (monotonic) passes."""
    from backend.documents import _pdf_reader

    reader = _pdf_reader(str(path))
    pages_obj = reader.pages
    total = len(pages_obj)
    ent["total"] = total
    pages = ent["pages"]
    i = len(pages)
    first = i
    while i < total:
        if deadline is not None and i > first and time.monotonic() >= deadline:   # always make >= 1 page of progress
            break
        try:
            pages.append(pages_obj[i].extract_text() or "")
        except Exception:  # noqa: BLE001 - one broken page must not sink the document
            pages.append("")
        i += 1
    ent["complete"] = i >= total


def chunks_from_paths(paths: list[Path], *, directory: Path | None = None,
                      extract_budget_s: float | None = None) -> list[dict]:
    """Chunk the PDFs. With ``directory`` (the index dir) big PDFs are read resumably within
    ``extract_budget_s``; ``last_extract_report()`` then says how far each got."""
    from backend.documents import extract_pdf_pages

    if directory is None:
        directory, ctx_budget = getattr(_scan_local, "ctx", (None, None))
        if extract_budget_s is None:
            extract_budget_s = ctx_budget
    state = load_texts(directory)
    files = state["files"]
    deadline = None if extract_budget_s is None else time.monotonic() + max(0.0, extract_budget_s)
    pages_by_source: list[tuple[str, list[str]]] = []
    report: list[dict] = []
    pending: dict[str, dict] = {}
    state_dirty = False
    seen: set[str] = set()
    for path in paths:
        seen.add(path.name)
        try:
            size = path.stat().st_size if directory is not None else 0
            ent = files.get(path.name)
            if ent is not None and ent.get("size") != size:
                ent = None                                  # replaced by a different file
            if ent is not None and ent.get("complete"):
                pages = ent["pages"]
            elif directory is not None and (ent is not None or size > 2_000_000):
                # possibly big: resumable path (cheap page count decides whether to keep state)
                if ent is None:
                    ent = {"size": size, "total": 0, "pages": [], "complete": False, "raw": True}
                _extract_range(path, ent, deadline)
                if ent["total"] > RESUME_MIN_PAGES:
                    files[path.name] = ent
                    state_dirty = True
                elif not ent["complete"]:                   # small file but out of time: keep going next call
                    files[path.name] = ent
                    state_dirty = True
                else:
                    files.pop(path.name, None)
                pages = ent["pages"]
                if not ent["complete"]:
                    pending[path.name] = {"done": len(pages), "total": ent["total"]}
            else:
                pages = extract_pdf_pages(path)
        except Exception:  # noqa: BLE001 — one unreadable PDF must not sink the subject
            log.warning("Could not read PDF %s", path.name, exc_info=True)
            report.append({"name": path.name, "pages": 0, "chars_per_page": 0.0, "unreadable": True})
            continue
        pages_by_source.append((path.name, pages))
        n_alnum = sum(_alnum(p or "") for p in pages)
        report.append({"name": path.name, "pages": len(pages),
                       "chars_per_page": round(n_alnum / max(1, len(pages)), 1)})
    # text-only files: the original PDF is gone, the stored text is all that is left
    for name, ent in list(files.items()):
        if name in seen or ent.get("raw", True) or not ent.get("complete"):
            continue
        pages_by_source.append((name, ent["pages"]))
        n_alnum = sum(_alnum(p or "") for p in ent["pages"])
        report.append({"name": name, "pages": len(ent["pages"]),
                       "chars_per_page": round(n_alnum / max(1, len(ent["pages"])), 1)})
    if directory is not None and (state_dirty or files):
        try:
            save_texts(directory, state)
        except OSError:
            log.warning("Could not save extracted text state", exc_info=True)
    _scan_local.report = report
    _scan_local.pending = pending
    return chunk_pages(pages_by_source)


def last_extract_pending() -> dict:
    """``{name: {done, total}}`` of PDFs only partly read by the latest :func:`chunks_from_paths`."""
    return dict(getattr(_scan_local, "pending", {}))


def low_text_sources(report: list[dict]) -> list[dict]:
    """Sources whose text layer is (almost) empty: scanned pages, photos, handwriting."""
    return [r for r in report if r.get("pages", 0) > 0 and r["chars_per_page"] < LOW_TEXT_CHARS_PER_PAGE]


def scan_warnings(low: list[dict]) -> list[str]:
    if not low:
        return []
    names = ", ".join(r["name"] for r in low[:3]) + (" and more" if len(low) > 3 else "")
    return [f"{SCANNED_NOTE} Little or no text was found in: {names}."]


def embed_missing(chunks: list[dict], vectors, *, model: str, dim: int, budget_s: float,
                  patient: bool = True, on_progress=None):
    """Fill the all-zero rows of ``vectors`` (n, dim) by embedding their chunks.

    Returns ``(vectors, n_new)``. Stops cleanly when ``budget_s`` runs out or quota is spent.
    ``on_progress(vectors, done, total)`` is called after each batch (checkpointing).
    """
    import numpy as np

    from backend import gemini_embeddings as ge

    missing = [i for i in range(len(chunks)) if not vectors[i].any()]
    if not missing or budget_s <= 0 or not ge.have_keys():
        return vectors, 0
    texts = [chunks[i]["text"] for i in missing]
    done = 0
    for start, mat in ge.iter_embed_documents(texts, model=model, dim=dim, budget_s=budget_s,
                                              patient=patient):
        idx = missing[start:start + len(mat)]
        vectors[idx] = mat
        done += len(mat)
        if on_progress:
            on_progress(vectors, done, len(missing))
    return vectors, done


def _make_manifest(key: str, chunks: list[dict], vectors, *, model: str, dim: int, files: list[dict],
                   owner: str | None = None, subject: str | None = None,
                   low_text: list[dict] | None = None, extract: dict | None = None) -> dict:
    n_emb = int(vectors.any(axis=1).sum()) if len(chunks) else 0
    return {
        "low_text": list(low_text or []),
        "extract": dict(extract or {}),
        "format": INDEX_VERSION, "key": key, "subject": subject or key, "owner": owner,
        "model": model, "dim": dim,
        "doc_format": "g2-prefix" if model.replace("models/", "").startswith("gemini-embedding-2") else "taskType",
        "chunk_size": CHUNK_SIZE, "chunk_overlap": CHUNK_OVERLAP,
        "n_chunks": len(chunks), "n_embedded": n_emb, "complete": bool(len(chunks)) and n_emb == len(chunks) and not extract,
        "files": files, "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "version": uuid.uuid4().hex,
    }


def build_index_to_dir(key: str, paths: list[Path], directory: Path, *, owner: str | None = None,
                       subject: str | None = None, budget_s: float = 0.0, patient: bool = True,
                       reuse: VectorIndex | None = None, checkpoint_every: int = 1,
                       log_fn=None, extract_budget_s: float | None = None) -> dict:
    """Chunk ``paths``, embed what the budget allows (reusing vectors for unchanged
    chunks), and write the index to ``directory``. Returns the manifest."""
    import numpy as np

    _scan_local.report = []
    _scan_local.ctx = (directory if owner else None, extract_budget_s)
    try:
        chunks = chunks_from_paths(paths)
    finally:
        _scan_local.ctx = (None, None)
    low = low_text_sources(last_scan_report())
    pending = last_extract_pending()
    if not chunks and not pending:
        raise VectorStoreError("No readable text was found in the selected PDF(s). " + SCANNED_NOTE)
    if not chunks:                       # nothing read yet (first call ran out of time): keep a placeholder
        chunks = [{"text": "(still reading this document…)", "source": paths[0].name if paths else "", "page": 1}]
    model, dim = embedding_model(), embedding_dim()
    vectors = np.zeros((len(chunks), dim), dtype=np.float32)
    if pending:
        budget_s = 0.0                    # chunk boundaries still move while pages are being read: embed afterwards

    # Reuse vectors from an existing compatible index (same model/dim, same chunk text).
    if reuse is not None and reuse.vectors is not None and reuse.model == model and reuse.dim == dim:
        old = {_chunk_hash(c["text"]): i for i, c in enumerate(reuse.chunks)
               if reuse._has is None or reuse._has[i]}
        for i, c in enumerate(chunks):
            j = old.get(_chunk_hash(c["text"]))
            if j is not None:
                vectors[i] = reuse.vectors[j]

    files = fingerprint_paths(paths)
    progress = {"n": 0}

    def _checkpoint(vecs, done, total):
        progress["n"] += 1
        if log_fn:
            log_fn(done, total)
        if checkpoint_every and progress["n"] % checkpoint_every == 0:
            save_index_dir(directory, chunks, vecs,
                           _make_manifest(key, chunks, vecs, model=model, dim=dim, files=files,
                                          owner=owner, subject=subject, low_text=low, extract=pending))

    vectors, _ = embed_missing(chunks, vectors, model=model, dim=dim, budget_s=budget_s,
                               patient=patient, on_progress=_checkpoint if checkpoint_every else None)
    manifest = _make_manifest(key, chunks, vectors, model=model, dim=dim, files=files,
                              owner=owner, subject=subject, low_text=low, extract=pending)
    save_index_dir(directory, chunks, vectors, manifest)
    return manifest


# ── cache ────────────────────────────────────────────────────────────────────
class _Entry:
    __slots__ = ("index", "checked", "version")

    def __init__(self, index: VectorIndex) -> None:
        self.index = index
        self.checked = time.monotonic()
        self.version = index.manifest.get("version")


_cache: "OrderedDict[str, _Entry]" = OrderedDict()
_cache_lock = threading.RLock()

# Per-subject-key re-entrant locks (a cold build/embed of one subject must not block every other
# subject, which a single global lock did). Entries are refcounted and dropped when idle.
_locks_guard = threading.Lock()
_key_locks: dict[str, list] = {}


@contextmanager
def _key_lock(key: str):
    with _locks_guard:
        ent = _key_locks.get(key)
        if ent is None:
            ent = _key_locks[key] = [threading.RLock(), 0]
        ent[1] += 1
    try:
        with ent[0]:
            yield
    finally:
        with _locks_guard:
            ent[1] -= 1
            if ent[1] <= 0 and _key_locks.get(key) is ent:
                del _key_locks[key]


# When the /tmp copy of a user index was last confirmed against MongoDB's version (monotonic).
_tmp_checked: dict[str, float] = {}


def enforce_tmp_caps() -> list[Path]:
    """Keep ``study-documents`` + ``study-index`` under ``TMP_CACHE_MAX_MB`` (LRU, in-use safe).

    Only acts when MongoDB persistence is on (otherwise /tmp holds the only copy of a user's
    PDFs) and only on the per-instance temp roots - never on the shipped ``documents/`` tree.
    """
    from backend import documents

    if not config.PERSISTENCE_ENABLED:
        return []
    roots = [_cache_root()]
    doc_root = documents._user_root()
    if doc_root != documents.DOCUMENTS_DIR:
        roots.append(doc_root)

    def _protect(path: Path) -> bool:
        try:
            from backend.services import subjects as subjects_service

            with subjects_service._active_lock:
                active = set(subjects_service._active)
        except Exception:  # noqa: BLE001
            active = set()
        if path.name in active:
            return True
        with _cache_lock:
            resident = {_user_cache_dir(k).name for k in _cache}
        return path.name in resident or any(_safe_dirname(k) == path.name for k in active)

    return tmpcache.enforce(roots, protect=_protect)


def _cache_put(key: str, index: VectorIndex) -> None:
    with _cache_lock:
        _cache.pop(key, None)
        _cache[key] = _Entry(index)
        while len(_cache) > _CACHE_MAX or (
                len(_cache) > 1 and sum(e.index.nbytes for e in _cache.values()) > _CACHE_MAX_BYTES):
            evicted, _ = _cache.popitem(last=False)
            log.info("Evicted index '%s' from cache.", evicted)


def clear_cache() -> None:
    with _cache_lock:
        _cache.clear()


def _mongo_version(key: str, owner: str | None) -> tuple[bool, str | None]:
    """``(reachable, version)``; reachable=False means "unknown" (never evict on unknown)."""
    try:
        from backend import db

        return db.index_version_state(key, owner)
    except Exception:  # noqa: BLE001
        return False, None


def _load_user_from_mongo(key: str, owner: str | None) -> VectorIndex | None:
    try:
        from backend import db

        blob = db.load_index_blob(key, owner)
    except Exception:  # noqa: BLE001
        log.warning("MongoDB index load failed for '%s'.", key, exc_info=True)
        return None
    if not blob:
        return None
    directory = _user_cache_dir(key)
    try:
        unpack_index(blob, directory)
        return load_index_dir(directory, key, mmap=False)
    except Exception:  # noqa: BLE001
        log.warning("Stored index for '%s' is unreadable.", key, exc_info=True)
        return None


def _load_index(key: str, subject: str, owner: str | None) -> VectorIndex:
    """Cold-load an index: shipped dir → /tmp cache → MongoDB → local PDFs (bounded)."""
    from backend.documents import DocumentError, list_pdf_paths

    if not _is_user_key(key):
        d = _bundled_dir(key)
        if (d / "manifest.json").is_file():
            return load_index_dir(d, key, mmap=True)
        tmp_d = _user_cache_dir(key)
        if (tmp_d / "manifest.json").is_file():
            return load_index_dir(tmp_d, key, mmap=True)
        # No shipped index: lexical-only index from the bundled PDFs. NEVER embeds in a request.
        paths = list_pdf_paths(subject, None)
        build_index_to_dir(key, paths, tmp_d, subject=subject, budget_s=0.0, checkpoint_every=0)
        return load_index_dir(tmp_d, key, mmap=True)

    tmp_d = _user_cache_dir(key)
    if (tmp_d / "manifest.json").is_file():
        # A /tmp copy may be stale (another instance re-indexed, or the subject was deleted):
        # trust it only if it was confirmed against MongoDB within the revalidation window.
        # A cold instance has no record, so its first use of a leftover /tmp copy is checked.
        if time.monotonic() - _tmp_checked.get(key, -1e9) > _REVALIDATE_S:
            local = read_manifest(tmp_d) or {}
            reachable, remote = _mongo_version(key, owner)
            if reachable and remote != local.get("version"):
                log.info("Cached /tmp index '%s' is out of date - discarding it.", key)
                shutil_rmtree(tmp_d)
            else:
                _tmp_checked[key] = time.monotonic()
        if (tmp_d / "manifest.json").is_file():
            tmpcache.touch(tmp_d)
            try:
                cached = load_index_dir(tmp_d, key, mmap=False)
            except Exception:  # noqa: BLE001 — half-written / corrupt /tmp copy: fall through
                log.warning("Unreadable /tmp index '%s'; discarding.", key, exc_info=True)
                cached = None
            if cached is not None and cached.problem is None:
                return cached
            shutil_rmtree(tmp_d)                       # corrupt or incompatible: refetch / rebuild
            _tmp_checked.pop(key, None)
    idx = _load_user_from_mongo(key, owner)
    if idx is not None and idx.problem is None:
        _tmp_checked[key] = time.monotonic()
        enforce_tmp_caps()
        return idx
    if idx is not None:
        log.warning("Stored index '%s' is unusable (%s); rebuilding from the PDFs.", key, idx.problem)
        shutil_rmtree(tmp_d)
    try:
        paths = list_pdf_paths(subject, owner)
    except DocumentError:
        raise
    if not paths:
        raise DocumentError(f"Unknown subject: {subject}")
    log.info("No stored index for '%s' — building now (bounded).", key)
    build_index_to_dir(key, paths, tmp_d, owner=owner, subject=subject,
                       budget_s=USER_UPLOAD_EMBED_BUDGET_S, patient=False, checkpoint_every=0)
    _persist_user_index(key, owner, subject, tmp_d)
    _tmp_checked[key] = time.monotonic()
    tmpcache.touch(tmp_d)
    enforce_tmp_caps()
    return load_index_dir(tmp_d, key, mmap=False)


def _persist_user_index(key: str, owner: str | None, subject: str, directory: Path) -> bool:
    if not _is_user_key(key):
        return False
    try:
        from backend import db

        manifest = read_manifest(directory) or {}
        return db.store_index_blob(key, owner, subject, manifest.get("version"), pack_index(directory),
                                   meta={k: manifest.get(k) for k in ("n_chunks", "n_embedded", "complete", "model", "dim")})
    except Exception:  # noqa: BLE001 — persistence must never break an upload
        log.warning("Could not persist index '%s' to MongoDB.", key, exc_info=True)
        return False


def _get_index(key: str, subject: str, owner: str | None) -> VectorIndex:
    _check_access(key, owner)
    with _cache_lock:
        ent = _cache.get(key)
        if ent is not None:
            _cache.move_to_end(key)
    if ent is not None:
        # Another instance may have re-indexed this user subject; cheap version check (TTL'd).
        if _is_user_key(key) and time.monotonic() - ent.checked > _REVALIDATE_S:
            ent.checked = time.monotonic()
            reachable, remote = _mongo_version(key, owner)
            if reachable and remote != ent.version:
                # Re-indexed elsewhere (new version) OR deleted (remote is None): drop RAM + /tmp copy.
                log.info("Index '%s' %s remotely — dropping cached copy.", key,
                         "deleted" if remote is None else "changed")
                with _cache_lock:
                    _cache.pop(key, None)
                shutil_rmtree(_user_cache_dir(key))
                _tmp_checked.pop(key, None)
                ent = None
            elif reachable:
                _tmp_checked[key] = time.monotonic()
        if ent is not None:
            return ent.index
    with _key_lock(key):                       # per-subject: a cold build never blocks other subjects
        with _cache_lock:
            ent = _cache.get(key)
        if ent is not None:
            return ent.index
        index = _load_index(key, subject, owner)
        _cache_put(key, index)
        return index


def shutil_rmtree(path: Path) -> None:
    import shutil

    shutil.rmtree(path, ignore_errors=True)


# ── public API (names preserved for existing importers) ──────────────────────
def warm_embeddings() -> None:
    """Best-effort connectivity check (1 content, ≤3 s). Safe to skip on serverless."""
    from backend import gemini_embeddings as ge

    ge.warm()


def build_index_from_paths(paths: list[Path]) -> VectorIndex:
    """In-memory index over ``paths`` (one-off uploaded-PDF chat). Bounded embedding time;
    anything not embedded in time is still searchable by BM25."""
    import numpy as np

    _scan_local.report = []
    chunks = chunks_from_paths([Path(p) for p in paths])
    low = low_text_sources(last_scan_report())
    if not chunks:
        raise VectorStoreError("No readable text was found in the selected PDF(s). " + SCANNED_NOTE)
    model, dim = embedding_model(), embedding_dim()
    vectors = np.zeros((len(chunks), dim), dtype=np.float32)
    try:
        vectors, _ = embed_missing(chunks, vectors, model=model, dim=dim,
                                   budget_s=ONEOFF_EMBED_BUDGET_S, patient=False)
    except Exception:  # noqa: BLE001
        log.info("One-off PDF embedding failed; lexical retrieval only.", exc_info=True)
    manifest = _make_manifest(f"oneoff-{uuid.uuid4().hex[:8]}", chunks, vectors, model=model, dim=dim, files=[],
                              low_text=low)
    return VectorIndex(manifest["key"], chunks, vectors, manifest)


def build_subject_index(subject: str, owner: str | None = None, budget_s: float | None = None,
                        extract_budget_s: float | None = None) -> int:
    """(Re)index a subject from its PDFs, persist it, and cache it. Returns the chunk count.

    For user subjects this is the upload-time path: bounded by ``budget_s`` (default
    ``USER_UPLOAD_EMBED_BUDGET_S``); chunks that did not fit stay BM25-searchable and are
    completed later by :func:`continue_subject_index`. Vectors for unchanged chunks of a
    previous index are reused, so adding a PDF to a subject only embeds the new chunks.
    """
    from backend.documents import list_pdf_paths

    key = subject_internal_key(subject, owner)
    _check_access(key, owner)
    try:
        paths = list_pdf_paths(subject, owner)
    except Exception:  # noqa: BLE001 - a subject whose files are all text-only has no PDF dir
        paths = []
    if _is_user_key(key) and not (_user_cache_dir(key) / "manifest.json").is_file():
        _load_user_from_mongo(key, owner)          # cold instance: bring back the stored text state
    if not paths and not any(not e.get("raw", True) for e in load_texts(_user_cache_dir(key)).get("files", {}).values()):
        raise VectorStoreError("No PDFs found for this subject.")
    if budget_s is None:
        budget_s = USER_UPLOAD_EMBED_BUDGET_S if _is_user_key(key) else 0.0
    with _key_lock(key):
        old = None
        with _cache_lock:
            ent = _cache.get(key)
            old = ent.index if ent else None
        directory = _user_cache_dir(key)
        if old is None and (directory / "manifest.json").is_file():
            try:
                old = load_index_dir(directory, key, mmap=False)
            except Exception:  # noqa: BLE001
                old = None
        manifest = build_index_to_dir(key, paths, directory, owner=owner, subject=subject,
                                      budget_s=budget_s, patient=False, reuse=old, checkpoint_every=0,
                                      extract_budget_s=extract_budget_s)
        index = load_index_dir(directory, key, mmap=False)
        _cache_put(key, index)
        _tmp_checked[key] = time.monotonic()
        tmpcache.touch(directory)
        # Persist under the same lock so MongoDB sees writes in the order the disk did.
        _persist_user_index(key, owner, subject, directory)
    enforce_tmp_caps()
    return int(manifest["n_chunks"])


def continue_subject_index(subject: str, owner: str | None = None, budget_s: float = 30.0) -> dict:
    """Embed more of a partially-embedded user subject within ``budget_s`` and persist it.

    Safe to call repeatedly (e.g. from a cron route or right after an upload response).
    Returns ``{"n_chunks", "n_embedded", "complete"}``.

    The embedding step runs WITHOUT the subject lock (it can take tens of seconds), so a PDF may
    be added/removed meanwhile. Before saving we therefore re-check the on-disk manifest
    ``version`` under the lock: if it moved, the freshly embedded vectors are MERGED (by chunk
    text) into whatever index is current - never written over it - and if the subject was
    deleted or re-indexed by another instance nothing is written at all.
    """
    import numpy as np

    key = subject_internal_key(subject, owner)
    _check_access(key, owner)
    index = _get_index(key, subject, owner)
    if index.vectors is None or index.manifest.get("complete"):
        return index_status_of(index)
    base_version = index.manifest.get("version")
    vectors = np.array(index.vectors, dtype=np.float32, copy=True)
    vectors, _ = embed_missing(index.chunks, vectors, model=index.model, dim=index.dim or embedding_dim(),
                               budget_s=budget_s, patient=False)
    directory = _user_cache_dir(key)
    with _key_lock(key):
        current = read_manifest(directory)
        if current is None:                              # deleted while we were embedding
            log.info("Subject '%s' disappeared during continue-indexing; discarding.", key)
            return index_status_of(index)
        target_chunks, target_vectors = index.chunks, vectors
        if current.get("version") != base_version:       # an add/rebuild landed meanwhile -> merge
            fresh = _get_index(key, subject, owner)
            if fresh.vectors is None or fresh.model != index.model or fresh.dim != index.dim:
                return index_status_of(fresh)
            by_hash = {_chunk_hash(c["text"]): vectors[i] for i, c in enumerate(index.chunks) if vectors[i].any()}
            target_chunks = fresh.chunks
            target_vectors = np.array(fresh.vectors, dtype=np.float32, copy=True)
            for i, c in enumerate(target_chunks):
                if not target_vectors[i].any():
                    v = by_hash.get(_chunk_hash(c["text"]))
                    if v is not None:
                        target_vectors[i] = v
            base_manifest = fresh.manifest
        else:
            base_manifest = index.manifest
        reachable, remote = _mongo_version(key, owner)
        if reachable and remote is not None and remote not in (base_version, current.get("version")):
            log.info("Subject '%s' was re-indexed by another instance; not overwriting.", key)
            return index_status_of(index)
        manifest = dict(base_manifest)
        manifest.update(_make_manifest(key, target_chunks, target_vectors, model=index.model,
                                       dim=index.dim or embedding_dim(),
                                       files=base_manifest.get("files", []), owner=owner, subject=subject,
                                       low_text=base_manifest.get("low_text"),
                                       extract=base_manifest.get("extract")))
        save_index_dir(directory, target_chunks, target_vectors, manifest)
        new = load_index_dir(directory, key, mmap=False)
        _cache_put(key, new)
        _tmp_checked[key] = time.monotonic()
        tmpcache.touch(directory)
        _persist_user_index(key, owner, subject, directory)
    return index_status_of(new)


def rename_index(old_subject: str, new_subject: str, owner: str) -> bool:
    """Move a user subject's stored index to a new subject name WITHOUT re-embedding: the zip is
    re-keyed (manifest ``key``/``subject`` + a fresh ``version``), stored under the new key, and
    the old blob / caches are removed. Returns False when there was no index to move."""
    import shutil

    from backend import db

    okey = subject_internal_key(old_subject, owner)
    nkey = subject_internal_key(new_subject, owner)
    if not (_is_user_key(okey) and _is_user_key(nkey)) or okey == nkey:
        return False
    odir, ndir = _user_cache_dir(okey), _user_cache_dir(nkey)
    first, second = sorted((okey, nkey))
    with _key_lock(first), _key_lock(second):
        if not (odir / "manifest.json").is_file():
            blob = db.load_index_blob(okey, owner)
            if not blob:
                return False
            unpack_index(blob, odir)
        shutil_rmtree(ndir)
        ndir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(odir, ndir)
        manifest = read_manifest(ndir) or {}
        manifest.update(key=nkey, subject=new_subject, version=uuid.uuid4().hex)
        _atomic_write(ndir / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8"))
        if db.is_enabled() and not _persist_user_index(nkey, owner, new_subject, ndir):
            shutil_rmtree(ndir)
            raise VectorStoreError("Could not store the renamed subject; nothing was changed.")
        db.delete_index_blob(okey, owner)
        shutil_rmtree(odir)
        with _cache_lock:
            _cache.pop(okey, None)
            _cache.pop(nkey, None)
        _tmp_checked.pop(okey, None)
    return True


def text_state(subject: str, owner: str | None) -> dict:
    """The stored per-file text state (``{name: {size,total,pages,complete,raw}}``) of a user subject."""
    key = subject_internal_key(subject, owner)
    if not _is_user_key(key):
        return {}
    directory = _user_cache_dir(key)
    if not (directory / "manifest.json").is_file():
        try:
            _get_index(key, subject, owner)
        except Exception:  # noqa: BLE001
            return {}
    return load_texts(directory).get("files", {})


def mark_text_only(subject: str, owner: str, names: list[str]) -> bool:
    """Flag fully-read files as "text only" (their PDF may now be dropped) and persist the index.
    Returns True when the flag was durably stored (MongoDB), i.e. the PDF is safe to delete."""
    key = subject_internal_key(subject, owner)
    if not _is_user_key(key):
        return False
    directory = _user_cache_dir(key)
    with _key_lock(key):
        state = load_texts(directory)
        changed = False
        for n in names:
            ent = state["files"].get(n)
            if ent and ent.get("complete") and ent.get("raw", True):
                ent["raw"] = False
                changed = True
        if not changed:
            return False
        save_texts(directory, state)
        return _persist_user_index(key, owner, subject, directory)


def forget_text(subject: str, owner: str, name: str) -> None:
    """Drop one file's stored text state (the file was removed from the subject)."""
    key = subject_internal_key(subject, owner)
    if not _is_user_key(key):
        return
    directory = _user_cache_dir(key)
    with _key_lock(key):
        state = load_texts(directory)
        if state["files"].pop(name, None) is not None:
            save_texts(directory, state)


def index_status_of(index: VectorIndex) -> dict:
    low = list(index.manifest.get("low_text") or [])
    return {"n_chunks": len(index.chunks), "n_embedded": index.n_embedded,
            "complete": bool(index.manifest.get("complete")) and index.problem is None,
            "model": index.model, "dim": index.dim,
            "warnings": scan_warnings(low), "low_text": [r["name"] for r in low],
            "problem": index.problem,
            "reading": dict(index.manifest.get("extract") or {}),
            "stale_model": bool(index.manifest.get("model") and (
                index.manifest.get("model") != embedding_model() or
                int(index.manifest.get("dim") or 0) not in (0, embedding_dim())))}


def index_status(subject: str, owner: str | None = None) -> dict:
    key = subject_internal_key(subject, owner)
    return index_status_of(_get_index(key, subject, owner))


def load_or_build_subject_index(subject: str, owner: str | None = None) -> VectorIndex:
    """Return a subject's index (cache → shipped/`/tmp`/MongoDB → bounded build)."""
    key = subject_internal_key(subject, owner)
    return _get_index(key, subject, owner)


def get_subject_retriever(subject: str, k: int, owner: str | None = None) -> Retriever:
    """Return a retriever over a subject's index (embedding-optional; never hard-fails)."""
    key = subject_internal_key(subject, owner)
    _get_index(key, subject, owner)          # surfaces unknown/forbidden subjects now; warms the cache
    return Retriever(key=key, owner=owner, k=k, subject=subject)


def warm_all_subjects() -> None:
    """Load every bundled subject's index into the cache (cheap: shipped + mmap)."""
    from backend.documents import list_subjects

    for subject in list_subjects():
        try:
            load_or_build_subject_index(subject)
        except Exception:  # noqa: BLE001
            log.warning("Could not warm index for subject '%s'.", subject, exc_info=True)


# ── health ───────────────────────────────────────────────────────────────────
_manifest_memo: dict[str, tuple[int, dict | None]] = {}


def _read_manifest_memo(directory: Path) -> dict | None:
    """Manifest read cached by mtime (health checks poll often; never re-parse an unchanged file)."""
    f = directory / "manifest.json"
    try:
        mt = f.stat().st_mtime_ns
    except OSError:
        return None
    hit = _manifest_memo.get(str(f))
    if hit and hit[0] == mt:
        return hit[1]
    m = read_manifest(directory)
    _manifest_memo[str(f)] = (mt, m)
    return m


def retrieval_health() -> dict:
    """Retrieval readiness for ``/api/health/deep`` and the doctor script. No secrets, no network.

    ``ok`` is False when a bundled index is missing/incomplete or does not match the configured
    model/dimension (retrieval then degrades to BM25 for that subject - never an outage).
    """
    from backend import gemini_embeddings as ge
    from backend.documents import list_subjects

    model, dim = embedding_model(), embedding_dim()
    bundled: dict[str, dict] = {}
    ok = True
    try:
        names = list_subjects()
    except Exception:  # noqa: BLE001
        names = []
    for name in names:
        d = _bundled_dir(name)
        m = _read_manifest_memo(d)
        if m is None:
            bundled[name] = {"present": False}
            ok = False
            continue
        good = bool(m.get("complete")) and m.get("model") == model and int(m.get("dim") or 0) == dim \
            and (d / "vectors.npy").is_file() and (d / "chunks.json").is_file()
        ok = ok and good
        bundled[name] = {"present": True, "ok": good, "n_chunks": m.get("n_chunks"),
                         "n_embedded": m.get("n_embedded"), "dim": m.get("dim"), "model": m.get("model"),
                         "format": m.get("format")}
    with _cache_lock:
        resident = len(_cache)
        nbytes = sum(e.index.nbytes for e in _cache.values())
    return {"ok": ok, "model": model, "dim": dim, "bundled": bundled,
            "cache": {"subjects": resident, "max_subjects": _CACHE_MAX, "bytes": nbytes,
                      "max_bytes": _CACHE_MAX_BYTES},
            "embedding": ge.status()}
