"""Hybrid multi-query retrieval with fusion, cheap reranking, MMR, per-PDF balance and neighbour
expansion. No LLM, no extra model: everything here is CPU work over the (already small) candidate
pool, a few milliseconds per request.

Flow (``gather_lists`` -> ``build``):

  queries ──▶ per-query hybrid search (dense + BM25 RRF inside the index, pool > final k)
  keywords ─▶ one free BM25-only pass (no embedding call)
        │
        ▼  weighted RRF across all result lists
  candidates ─▶ rerank (RRF + keyword coverage/phrase match + dense cosine + position prior + file hint)
        ─▶ MMR (drop near-duplicate chunks) ─▶ per-PDF balance ─▶ neighbour expansion/merge
        ─▶ ``Retrieval`` (blocks in relevance order + confidence)

A "searcher" wraps whatever the caller holds: a ``vector_store.Retriever`` (direct index access →
scored pool, chunk neighbours), or any object with ``invoke(query)`` (tests/fakes → plain top-k, no
neighbours). Several searchers = cross-subject search, fused by RRF; each candidate remembers its
searcher so owner isolation is simply "only searchers built for this owner are ever passed in".
"""
from __future__ import annotations

import inspect
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

RRF_K = 60
DENSE_LO, DENSE_HI = 0.50, 0.80          # cosine range mapped to 0..1 (Gemini embeddings; tuned by scripts/eval_rag.py)
MMR_LAMBDA = 0.75
MMR_DUP_SIM = 0.80                         # token-set Jaccard above which two chunks are near-duplicates
RELEVANCE_FLOOR = 0.30                     # keep candidates scoring >= 30 % of the best one (min 3 kept)
EXPAND_TOP = 3                             # neighbour-expand the best N chunks
LEXICAL_WEIGHT = 0.7


def _env_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(name, "") or default))
    except ValueError:
        return default


def pool_size() -> int:
    return _env_int("AGENTIC_RAG_POOL", 24)


def final_k() -> int:
    return _env_int("AGENTIC_RAG_FINAL_K", 8)


def _tok(text: str) -> list[str]:
    from backend import vector_store

    return vector_store.tokenize(text)


def _document(text: str, meta: dict):
    from backend import vector_store

    return vector_store.Document(text, meta)


# --------------------------------------------------------------------------------------
# Searchers
# --------------------------------------------------------------------------------------
def _accepts(fn, name: str) -> bool:
    try:
        return name in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


class Searcher:
    """Uniform search + neighbour access over one retriever (see module docstring)."""

    def __init__(self, retriever, label: str | None = None) -> None:
        self.retriever = retriever
        self.label = label
        self._idx = None
        self._resolved = False

    def index(self):
        if not self._resolved:
            r = self.retriever
            resolve = getattr(r, "_resolve", None)
            idx = resolve() if callable(resolve) else getattr(r, "_index", None)
            self._idx = idx if (idx is not None and hasattr(idx, "chunks") and hasattr(idx, "search")) else None
            self._resolved = True
            self._direct = self._idx is not None or callable(resolve)
        return self._idx

    @property
    def direct(self) -> bool:
        self.index()
        return self._direct

    def search(self, query: str, k: int, *, lexical: bool = False, timeout_s: float | None = None) -> list:
        idx = self.index()
        if idx is not None:
            kwargs = {"use_embeddings": not lexical}
            if timeout_s:
                kwargs["timeout_s"] = timeout_s
            return list(idx.search(query, k, **kwargs))
        if self._direct:                      # index could not be loaded
            return []
        r = self.retriever
        if lexical:
            fn = getattr(r, "invoke_lexical", None)
            return list(fn(query)) if fn is not None else []
        if timeout_s and _accepts(r.invoke, "timeout_s"):
            return list(r.invoke(query, timeout_s=timeout_s))
        return list(r.invoke(query))

    def positions(self) -> dict | None:
        idx = self.index()
        if idx is None:
            return None
        pos = getattr(idx, "_rag_pos", None)
        if pos is None:
            pos = {(c["source"], c["page"], hash(c["text"])): i for i, c in enumerate(idx.chunks)}
            try:
                idx._rag_pos = pos
            except Exception:  # noqa: BLE001
                pass
        return pos

    def locate(self, doc) -> int | None:
        pos = self.positions()
        if pos is None:
            return None
        m = getattr(doc, "metadata", {}) or {}
        return pos.get((m.get("source"), m.get("page"), hash(getattr(doc, "page_content", "") or "")))

    def chunk(self, i: int) -> dict | None:
        idx = self.index()
        return idx.chunks[i] if idx is not None and 0 <= i < len(idx.chunks) else None


def searchers_for(retriever) -> list[Searcher]:
    """``retriever`` may already be a list of searchers (cross-subject) or a single retriever."""
    if isinstance(retriever, Searcher):
        return [retriever]
    if isinstance(retriever, (list, tuple)):
        return [r if isinstance(r, Searcher) else Searcher(r) for r in retriever]
    return [Searcher(retriever)]


# --------------------------------------------------------------------------------------
# Result lists + fusion
# --------------------------------------------------------------------------------------
@dataclass
class RList:
    searcher: Searcher
    query: str
    docs: list
    weight: float
    lexical: bool = False


def gather_lists(searchers: list[Searcher], queries: list[str], lexical_query: str = "", *,
                 pool: int | None = None, lexical_only: bool = False, timeout_s: float | None = None,
                 lexical_extra: bool = True) -> list[RList]:
    """Run every query against every searcher. Hybrid queries run in parallel threads (each may
    wait on one embedding call); the keyword BM25 pass is free so it runs inline."""
    pool = pool or pool_size()
    jobs: list[tuple[Searcher, str, bool, float]] = []
    for s in searchers:
        for q in queries:
            jobs.append((s, q, lexical_only, 1.0 if not lexical_only else LEXICAL_WEIGHT))
        if lexical_extra and lexical_query and lexical_query not in queries:
            jobs.append((s, lexical_query, True, LEXICAL_WEIGHT))

    def run(job):
        s, q, lex, w = job
        try:
            return RList(s, q, s.search(q, pool, lexical=lex, timeout_s=timeout_s), w, lex)
        except Exception:  # noqa: BLE001 — one bad query/subject must not abort the turn
            log.debug("Retrieval failed for %r.", q, exc_info=True)
            return RList(s, q, [], w, lex)

    heavy = [j for j in jobs if not j[2]]
    if len(heavy) > 1:
        with ThreadPoolExecutor(max_workers=len(heavy)) as ex:
            futs = {id(j): ex.submit(run, j) for j in heavy}
            return [futs[id(j)].result() if id(j) in futs else run(j) for j in jobs]
    return [run(j) for j in jobs]


@dataclass
class Cand:
    searcher: Searcher
    source: str
    page: int | None
    text: str
    doc: object
    idx: int | None = None
    rrf: float = 0.0
    dense: float | None = None
    retrieval: str = ""
    cov: float = 0.0
    final: float = 0.0
    toks: frozenset = frozenset()
    hint: bool = False


def _fuse(lists: list[RList]) -> list[Cand]:
    cands: dict[tuple, Cand] = {}
    for lst in lists:
        for rank, doc in enumerate(lst.docs):
            text = (getattr(doc, "page_content", "") or "").strip()
            if not text:
                continue
            m = getattr(doc, "metadata", {}) or {}
            key = (id(lst.searcher), m.get("source"), m.get("page"), hash(text))
            c = cands.get(key)
            if c is None:
                c = cands[key] = Cand(lst.searcher, str(m.get("source") or ""), m.get("page") if isinstance(m.get("page"), int) else None,
                                      text, doc)
            c.rrf += lst.weight / (RRF_K + rank + 1)
            mode = m.get("retrieval") or ""
            sc = m.get("score")
            if mode in ("hybrid", "dense") and isinstance(sc, (int, float)) and sc > 0:
                c.dense = sc if c.dense is None else max(c.dense, sc)
                c.retrieval = mode
            elif not c.retrieval:
                c.retrieval = mode
    return list(cands.values())


# --------------------------------------------------------------------------------------
# Rerank / MMR / balance
# --------------------------------------------------------------------------------------
def _norm_dense(d: float | None) -> float | None:
    if d is None:
        return None
    return max(0.0, min(1.0, (d - DENSE_LO) / (DENSE_HI - DENSE_LO)))


def rerank(cands: list[Cand], keywords: list[str], ordered_terms: list[str] | None = None,
           hints: list[str] | None = None) -> list[Cand]:
    """Score + sort candidates. ``keywords`` = stemmed query terms; ``ordered_terms`` (stems in
    question order) enable the adjacent-pair phrase bonus."""
    if not cands:
        return []
    kw = set(keywords)
    pairs = set(zip(ordered_terms, ordered_terms[1:])) if ordered_terms and len(ordered_terms) > 1 else set()
    top_rrf = max((c.rrf for c in cands), default=0.0) or 1.0
    for c in cands:
        seq = _tok(c.text)
        c.toks = frozenset(seq)
        cov = (len(kw & c.toks) / len(kw)) if kw else 0.5
        if pairs:
            have = set(zip(seq, seq[1:]))
            cov = min(1.0, cov + 0.15 * len(pairs & have) / len(pairs))
        c.cov = cov
        dn = _norm_dense(c.dense)
        rrf_n = c.rrf / top_rrf
        prior = 0.05 / (1.0 + (c.page or 0) / 10.0)
        c.hint = bool(hints) and any(c.source.endswith(h) or c.source == h for h in hints)
        if dn is not None:
            c.final = 0.40 * rrf_n + 0.30 * cov + 0.25 * dn + prior
        else:
            c.final = 0.50 * rrf_n + 0.45 * cov + prior
        if c.hint:
            c.final += 0.15
    return sorted(cands, key=lambda c: -c.final)


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def mmr_select(ranked: list[Cand], k: int, lam: float = MMR_LAMBDA) -> list[Cand]:
    """Greedy maximal-marginal-relevance pick of ``k`` candidates; near-duplicates are skipped."""
    if not ranked:
        return []
    floor = ranked[0].final * RELEVANCE_FLOOR
    pool = [c for i, c in enumerate(ranked) if i < 3 or c.final >= floor]
    chosen: list[Cand] = []
    while pool and len(chosen) < k:
        best, best_val = None, -1e9
        for c in pool:
            sim = max((_jaccard(c.toks, s.toks) for s in chosen), default=0.0)
            if sim >= MMR_DUP_SIM:
                continue
            val = lam * c.final - (1 - lam) * sim
            if val > best_val:
                best, best_val = c, val
        if best is None:
            break
        chosen.append(best)
        pool.remove(best)
    return chosen


def balance_sources(chosen: list[Cand], ranked: list[Cand], k: int) -> list[Cand]:
    """When a subject has several PDFs, make sure each strongly-matching one is represented."""
    if not chosen or len({c.source for c in ranked}) < 2:
        return chosen
    out = list(chosen)
    top = ranked[0].final
    present = {(id(c.searcher), c.source) for c in out}
    for c in ranked:
        key = (id(c.searcher), c.source)
        if key in present or c.final < 0.6 * top:
            continue
        counts: dict[tuple, int] = {}
        for o in out:
            counts[(id(o.searcher), o.source)] = counts.get((id(o.searcher), o.source), 0) + 1
        if len(out) < k:
            out.append(c)
        else:
            victims = [o for o in out if counts[(id(o.searcher), o.source)] > 1]
            if not victims:
                continue
            out.remove(min(victims, key=lambda o: o.final))
            out.append(c)
        present.add(key)
    return sorted(out, key=lambda c: -c.final)


def _cand_key(lst: "RList", doc) -> tuple | None:
    text = (getattr(doc, "page_content", "") or "").strip()
    if not text:
        return None
    m = getattr(doc, "metadata", {}) or {}
    return (id(lst.searcher), m.get("source"), m.get("page"), hash(text))


def balance_parts(chosen: list[Cand], ranked: list[Cand], parts: list["RList"], k: int) -> list[Cand]:
    """Multi-part / comparison questions: make sure the best passage of EVERY part is in the final set
    (a plain top-k can be filled by the passages of the part that happens to match the wording best)."""
    if not chosen or not parts:
        return chosen
    by_key = {(id(c.searcher), c.source, c.page, hash(c.text)): c for c in ranked}
    out = list(chosen)
    protected: set[int] = set()
    for lst in parts:
        want = [by_key.get(_cand_key(lst, d)) for d in lst.docs[:3]]
        want = [c for c in want if c is not None]
        if not want:
            continue
        present = [c for c in want if c in out]
        if present:
            protected.add(id(present[0]))
            continue
        best = want[0]
        if len(out) < k:
            out.append(best)
        else:
            victims = [o for o in out if id(o) not in protected]
            if not victims:
                continue
            out.remove(min(victims, key=lambda o: o.final))
            out.append(best)
        protected.add(id(best))
    return sorted(out, key=lambda c: -c.final)


# --------------------------------------------------------------------------------------
# Blocks (neighbour expansion + contiguous merge)
# --------------------------------------------------------------------------------------
@dataclass
class Block:
    source: str
    page: int | None            # 0-based first page
    page_end: int | None
    text: str
    score: float
    dense: float | None = None
    cov: float = 0.0
    subject: str | None = None
    keys: frozenset = frozenset()          # (searcher id, chunk idx) of merged chunks - dedupe across hops
    expanded: bool = False
    doc: object = None
    n: int = 0                  # 1-based citation number, assigned when the block joins an Evidence

    def page_label(self) -> str:
        if self.page is None:
            return ""
        a, b = self.page + 1, (self.page_end if self.page_end is not None else self.page) + 1
        return f"p.{a}" if a == b else f"p.{a}-{b}"


def _join(a: str, b: str, max_overlap: int = 260) -> str:
    """Concatenate two consecutive chunks, dropping the text overlap the chunker left between them."""
    for n in range(min(len(a), len(b), max_overlap), 24, -1):
        if a.endswith(b[:n]):
            return a + b[n:]
    return a + "\n\n" + b


def _make_blocks(selected: list[Cand], expand_top: int) -> list[Block]:
    # members: (searcher id, idx) -> (cand or None, chunk dict)
    members: dict[tuple, dict] = {}
    singles: list[Cand] = []
    for rank, c in enumerate(selected):
        if c.idx is None:
            c.idx = c.searcher.locate(c.doc)
        if c.idx is None:
            singles.append(c)
            continue
        key = (id(c.searcher), c.idx)
        members[key] = {"c": c, "chunk": {"text": c.text, "source": c.source, "page": c.page}, "score": c.final}
        if rank < expand_top:
            for j in (c.idx - 1, c.idx + 1):
                ch = c.searcher.chunk(j)
                k2 = (id(c.searcher), j)
                if ch is None or k2 in members or ch["source"] != c.source:
                    continue
                if c.page is not None and isinstance(ch["page"], int) and abs(ch["page"] - c.page) > 1:
                    continue
                members[k2] = {"c": None, "chunk": ch, "score": c.final * 0.5, "exp": True}
    blocks: list[Block] = []
    # group consecutive idx runs per searcher
    by_s: dict[int, list[int]] = {}
    for sid, i in members:
        by_s.setdefault(sid, []).append(i)
    for sid, ids in by_s.items():
        ids.sort()
        run: list[int] = []
        for i in ids + [None]:
            if run and (i is None or i != run[-1] + 1 or members[(sid, i)]["chunk"]["source"] != members[(sid, run[0])]["chunk"]["source"]):
                parts = [members[(sid, j)] for j in run]
                text = parts[0]["chunk"]["text"]
                for p in parts[1:]:
                    text = _join(text, p["chunk"]["text"])
                best = max(parts, key=lambda p: p["score"])
                pages = [p["chunk"]["page"] for p in parts if isinstance(p["chunk"]["page"], int)]
                lead = best["c"] or next(p["c"] for p in parts if p["c"] is not None)
                blocks.append(Block(
                    lead.source, min(pages) if pages else None, max(pages) if pages else None, text,
                    max(p["score"] for p in parts), lead.dense, lead.cov, lead.searcher.label,
                    frozenset((sid, j) for j in run), expanded=len(run) > 1, doc=lead.doc))
                run = []
            if i is not None:
                run.append(i)
    for c in singles:
        blocks.append(Block(c.source, c.page, c.page, c.text, c.final, c.dense, c.cov, c.searcher.label, doc=c.doc))
    blocks.sort(key=lambda b: -b.score)
    return blocks


# --------------------------------------------------------------------------------------
# Public entry
# --------------------------------------------------------------------------------------
@dataclass
class Retrieval:
    blocks: list[Block]
    confidence: float
    mode: str                       # hybrid | lexical | none
    n_candidates: int = 0
    lists: list = field(default_factory=list)
    best_cov: float = 0.0
    best_dense: float | None = None
    related: list = field(default_factory=list)       # concept-graph suggestions for follow-up chips
    themes: list = field(default_factory=list)        # global questions: [{label, summary, n_chunks}]


def confidence_of(top: list[Cand]) -> tuple[float, float, float | None]:
    """(confidence 0..1, best keyword coverage, best raw dense cosine) over the top candidates."""
    if not top:
        return 0.0, 0.0, None
    cov = max(c.cov for c in top)
    dense = max((c.dense for c in top if c.dense is not None), default=None)
    dn = _norm_dense(dense)
    conf = cov if dn is None else 0.5 * cov + 0.5 * dn
    return round(max(0.0, min(1.0, conf)), 3), cov, dense


def build(lists: list[RList], keywords: list[str], *, ordered_terms: list[str] | None = None,
          hints: list[str] | None = None, k: int | None = None, expand: bool = True,
          parts: list[RList] | None = None) -> Retrieval:
    """Fuse -> rerank -> MMR (dedupes across ALL lists, so across sub-query results too) -> balance.
    ``parts`` = the per-part result lists of a comparison / multi-part question (see :func:`balance_parts`)."""
    k = k or final_k()
    cands = _fuse(lists)
    ranked = rerank(cands, keywords, ordered_terms, hints)
    if not ranked:
        return Retrieval([], 0.0, "none", 0, lists)
    chosen = balance_sources(mmr_select(ranked, k), ranked, k)
    if parts:
        chosen = balance_parts(chosen, ranked, parts, k)
    blocks = _make_blocks(chosen, EXPAND_TOP if expand else 0)
    conf, cov, dense = confidence_of(ranked[:3])
    mode = "hybrid" if any(c.retrieval in ("hybrid", "dense") for c in cands) else "lexical"
    return Retrieval(blocks, conf, mode, len(cands), lists, cov, dense)


def global_retrieval(searchers: list[Searcher], question: str, k: int | None = None) -> Retrieval | None:
    """Community view for 'summarise the whole subject' questions: chunks that together cover the subject's
    central concepts (``ConceptGraph.global_chunks``), in reading order, plus the extracted themes. No
    embedding call, no LLM. ``None`` when the question is not a global one or no searcher has a graph."""
    from backend.rag import graph as rg

    k = k or final_k()
    usable = []
    for s in searchers:
        idx = s.index()
        g = idx.graph() if idx is not None and hasattr(idx, "graph") else None
        if g is not None and g.communities and rg.is_global_question(
                question, g, (getattr(idx, "manifest", {}) or {}).get("subject") or ""):
            usable.append((s, idx, g))
    if not usable or len(usable) != len(searchers):
        return None                                           # cross-subject mixes stay on the hybrid path
    per = max(2, k // len(usable))
    cands: list[Cand] = []
    themes: list[dict] = []
    for s, idx, g in usable:
        for i, sc in g.global_chunks(per, question):
            c = idx.chunks[i]
            doc = idx._doc(i, sc, "graph")
            cands.append(Cand(s, c["source"], c["page"] if isinstance(c["page"], int) else None, c["text"], doc,
                              idx=i, rrf=sc, retrieval="graph", cov=0.5, final=0.5 + 0.5 * sc))
        themes += [dict(t, subject=s.label) if s.label else t for t in g.themes()]
    if not cands:
        return None
    blocks = _make_blocks(cands, 0)
    blocks.sort(key=lambda b: (b.subject or "", b.source, b.page if b.page is not None else -1))   # reading order
    return Retrieval(blocks, 0.7, "hybrid", len(cands), [], 0.5, None, themes=themes)


def graph_related(searchers: list[Searcher], question: str, n: int = 4) -> list[str]:
    """Concepts the notes connect to the question's concepts (follow-up suggestions). Best-effort, no I/O."""
    out: list[str] = []
    for s in searchers[:3]:
        try:
            idx = s.index()
            g = idx.graph() if idx is not None and hasattr(idx, "graph") else None
            if g is not None:
                out += [x for x in g.related(question, n) if x not in out]
        except Exception:  # noqa: BLE001
            log.debug("Related-concept lookup failed.", exc_info=True)
    return out[:n]


def doc_for(block: Block):
    """A ``Document``-shaped view of a block (page 0-based like the index, merged text)."""
    meta = {"source": block.source, "page": block.page, "score": round(block.score, 4)}
    if block.subject:
        meta["subject"] = block.subject
    return _document(block.text, meta)
