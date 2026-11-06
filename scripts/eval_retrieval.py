#!/usr/bin/env python3
"""Retrieval-quality evaluation on the bundled subjects (dense vs BM25 vs hybrid RRF).

    ./venv/bin/python scripts/eval_retrieval.py                 # report hit@1/3/5/10 + MRR
    ./venv/bin/python scripts/eval_retrieval.py --sweep         # cheap-knob sweep (RRF k, pool, weight, BM25 k1/b)
    ./venv/bin/python scripts/eval_retrieval.py --chunk-sweep   # BM25-only proxy for chunk size / overlap
    ./venv/bin/python scripts/eval_retrieval.py --json out.json

Gold labels live in ``tests/data/retrieval_eval.json``. A retrieved chunk is *relevant* when its
whitespace/case-normalised text contains any of the query's gold anchor strings, so the labels
survive re-chunking. ``multi`` queries have several gold anchors (recall@10 is reported too);
``negative`` queries have none (we report how separable their top-1 score is instead).

Query embeddings are cached in ``tests/data/retrieval_eval_qvecs.json`` (keyed by model|dim|text),
so reruns are free and offline. Missing vectors are embedded live, one at a time with a small
pause (free tier: ~100 contents/min/key), only when the cache lacks them. Keys are never printed.
Small eval sets are noisy: differences below the bootstrap CI printed with ``--sweep`` are noise.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

EVAL_PATH = ROOT / "tests" / "data" / "retrieval_eval.json"
MULTIHOP_PATH = ROOT / "tests" / "data" / "multihop_eval.json"
QVEC_PATH = ROOT / "tests" / "data" / "retrieval_eval_qvecs.json"
KS = (1, 3, 5, 10)
MULTI_TYPES = ("multi", "multihop", "compare", "global")


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).lower()


def load_eval() -> list[dict]:
    return json.loads(EVAL_PATH.read_text(encoding="utf-8"))["queries"]


# ── query vectors (cached) ───────────────────────────────────────────────────
def _qkey(model: str, dim: int, q: str) -> str:
    return f"{model}|{dim}|{q}"


def load_qvecs() -> dict:
    try:
        return json.loads(QVEC_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def ensure_qvecs(queries: list[dict], model: str, dim: int, *, live: bool = True) -> dict:
    """Return ``{key: np.ndarray}`` for every query, embedding (and caching) the missing ones."""
    import numpy as np

    cache = load_qvecs()
    missing = [q["q"] for q in queries if _qkey(model, dim, q["q"]) not in cache]
    if missing and live:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
        from backend import gemini_embeddings as ge

        print(f"embedding {len(missing)} uncached queries live (free tier, paced)…", file=sys.stderr)
        for q in missing:
            for attempt in range(4):
                try:
                    v = ge.embed_query(q, model=model, dim=dim, deadline_s=20.0)
                    cache[_qkey(model, dim, q)] = [round(float(x), 6) for x in v]
                    break
                except ge.GeminiEmbeddingError as exc:
                    print(f"  retry {attempt + 1} for {q[:40]!r}: {type(exc).__name__}", file=sys.stderr)
                    ge._breaker["until"] = 0.0
                    time.sleep(8 * (attempt + 1))
            time.sleep(0.8)
        QVEC_PATH.write_text(json.dumps(cache, separators=(",", ":")), encoding="utf-8")
    return {k: np.asarray(v, dtype=np.float32) for k, v in cache.items()}


# ── metrics ──────────────────────────────────────────────────────────────────
def anchor_hit(anchor: str, text_norm: str) -> bool:
    """An anchor may list alternatives (``"a || b"``): a topic group is covered by ANY of them."""
    return any(norm(x) in text_norm for x in anchor.split(" || "))


def relevant_flags(index, ranked: list[int], gold: list[str]) -> list[bool]:
    return [any(anchor_hit(a, norm(index.chunks[i]["text"])) for a in gold) for i in ranked]


def _rank(idx, query: str, qv, k: int, mode: str):
    """``idx.rank`` plus the pipeline's routing mode ``routed`` = global questions go to the community
    view (``ConceptGraph.global_chunks``), everything else to plain hybrid."""
    if mode == "routed":
        from backend.rag import graph as rg

        g = idx.graph()
        if g is not None and rg.is_global_question(query, g, idx.manifest.get('subject') or idx.key):
            return [(i, sc, "graph") for i, sc in g.global_chunks(k, query)]
        return idx.rank(query, qv, k, "hybrid")
    return idx.rank(query, qv, k, mode)


def evaluate(indexes: dict, queries: list[dict], qvecs: dict, mode: str, *, model: str, dim: int,
             k_eval: int = 10) -> dict:
    """Metrics for one retrieval ``mode`` ('dense' | 'bm25' | 'hybrid') under the current knobs."""
    per_q: list[dict] = []
    for q in queries:
        idx = indexes[q["subject"]]
        qv = qvecs.get(_qkey(model, dim, q["q"]))
        ranked = [i for i, _s, _m in _rank(idx, q["q"], qv, k_eval, mode)]
        rec = {"q": q["q"], "type": q["type"], "subject": q["subject"]}
        if q["type"] == "negative":
            top = _rank(idx, q["q"], qv, 1, mode)
            rec["top1"] = top[0][1] if top else 0.0
        else:
            rel = relevant_flags(idx, ranked, q["gold"])
            rec["first"] = next((r + 1 for r, ok in enumerate(rel) if ok), None)
            covered = {a for a in q["gold"]
                       if any(anchor_hit(a, norm(idx.chunks[i]["text"])) for i in ranked)}
            rec["recall10"] = len(covered) / len(q["gold"])
            for kk in (5, 8):
                cov_k = {a for a in q["gold"] if any(anchor_hit(a, norm(idx.chunks[i]["text"])) for i in ranked[:kk])}
                rec[f"recall{kk}"] = len(cov_k) / len(q["gold"])
        per_q.append(rec)
    pos = [r for r in per_q if "first" in r]
    out = {"n": len(pos), "mode": mode}
    for k in KS:
        out[f"hit@{k}"] = sum(1 for r in pos if r["first"] and r["first"] <= k) / len(pos)
    out["mrr"] = sum(1.0 / r["first"] for r in pos if r["first"]) / len(pos)
    multi = [r for r in pos if r["type"] in MULTI_TYPES]
    out["multi_recall@10"] = sum(r["recall10"] for r in multi) / len(multi) if multi else None
    out["multi_recall@5"] = sum(r["recall5"] for r in multi) / len(multi) if multi else None
    out["multi_recall@8"] = sum(r["recall8"] for r in multi) / len(multi) if multi else None
    out["by_type_hit@5"] = {}
    for t in sorted({r["type"] for r in pos}):
        rs = [r for r in pos if r["type"] == t]
        out["by_type_hit@5"][t] = round(sum(1 for r in rs if r["first"] and r["first"] <= 5) / len(rs), 3)
    neg = [r["top1"] for r in per_q if "top1" in r]
    out["neg_top1_mean"] = sum(neg) / len(neg) if neg else None
    out["pos_top1_mean"] = None
    out["_per_q"] = per_q
    return out


def top1_scores(indexes, queries, qvecs, mode, model, dim) -> tuple[list[float], list[float]]:
    pos, neg = [], []
    for q in queries:
        idx = indexes[q["subject"]]
        top = idx.rank(q["q"], qvecs.get(_qkey(model, dim, q["q"])), 1, mode)
        (neg if q["type"] == "negative" else pos).append(top[0][1] if top else 0.0)
    return pos, neg


def auc(pos: list[float], neg: list[float]) -> float:
    """P(score(pos) > score(neg)): how well the top-1 score separates answerable from unanswerable."""
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def fmt(m: dict) -> str:
    return (f"{m['mode']:<7} hit@1 {m['hit@1']:.3f}  hit@3 {m['hit@3']:.3f}  hit@5 {m['hit@5']:.3f}  "
            f"hit@10 {m['hit@10']:.3f}  MRR {m['mrr']:.3f}  multi-recall@10 "
            f"{(m['multi_recall@10'] if m['multi_recall@10'] is not None else float('nan')):.3f}")


def bootstrap_diff(a: list[dict], b: list[dict], n: int = 2000, seed: int = 7) -> tuple[float, float, float]:
    """Paired bootstrap of MRR(a) - MRR(b) over queries -> (mean, lo95, hi95)."""
    ra = [1.0 / r["first"] if r.get("first") else 0.0 for r in a if "first" in r]
    rb = [1.0 / r["first"] if r.get("first") else 0.0 for r in b if "first" in r]
    rng = random.Random(seed)
    m = len(ra)
    diffs = []
    for _ in range(n):
        idx = [rng.randrange(m) for _ in range(m)]
        diffs.append(sum(ra[i] - rb[i] for i in idx) / m)
    diffs.sort()
    return sum(ra[i] - rb[i] for i in range(m)) / m, diffs[int(0.025 * n)], diffs[int(0.975 * n)]


# ── multi-hop / global evaluation with the concept graph ────────────────────
def fmt_mh(m: dict) -> str:
    r = lambda x: f"{x:.3f}" if x is not None else "  -  "          # noqa: E731
    return (f"{m['mode']:<13} n={m['n']:<3} hit@1 {m['hit@1']:.3f}  hit@3 {m['hit@3']:.3f}  hit@5 {m['hit@5']:.3f}  "
            f"MRR {m['mrr']:.3f}  recall@5 {r(m['multi_recall@5'])}  recall@8 {r(m['multi_recall@8'])}  "
            f"recall@10 {r(m['multi_recall@10'])}")


def graph_eval(offline: bool, json_out: str | None) -> int:
    """dense | bm25 | hybrid | hybrid+graph | graph on the multi-hop set (dev/test halves) and on the
    single-hop set (does the graph hurt?). Graph knobs are tuned on the DEV half only."""
    from backend import vector_store as vs
    from backend.rag import graph as rg

    mh = json.loads(MULTIHOP_PATH.read_text(encoding="utf-8"))["queries"]
    base = load_eval()
    subjects = sorted({q["subject"] for q in mh + base})
    indexes = load_indexes(subjects)
    model, dim = next(iter(indexes.values())).model, next(iter(indexes.values())).dim
    qvecs = ensure_qvecs(base + mh, model, dim, live=not offline)
    if any(_qkey(model, dim, q["q"]) not in qvecs for q in base + mh):
        print("some query vectors are missing (run without --offline once)", file=sys.stderr)
        return 2
    for i in indexes.values():
        g = i.graph()
        print(f"graph[{i.key}]: {len(g.keys)} concepts, {len(g.src) // 2} edges, {len(g.communities)} communities "
              f"({i._graph_state})")
    dev = [q for q in mh if q["split"] == "dev"]
    test = [q for q in mh if q["split"] == "test"]
    modes = ("dense", "bm25", "hybrid", "hybrid+graph", "graph", "routed")
    out: dict = {"knobs_default": {"w_graph": vs._W_GRAPH, "seed_discount": rg.SEED_DISCOUNT, "method": rg.METHOD,
                                   "hops": rg.HOPS}}

    def show(title: str, qs: list[dict], key: str) -> dict:
        print(f"\n{title} ({len(qs)} questions)")
        res = {}
        for mode in modes:
            res[mode] = evaluate(indexes, qs, qvecs, mode, model=model, dim=dim)
            print("  " + fmt_mh(res[mode]))
        out[key] = {m: {k: v for k, v in r.items() if k != "_per_q"} for m, r in res.items()}
        return res

    # 1. tune on dev
    print("\nTuning graph knobs on the DEV half only (objective = MRR + recall@8):")
    grid = []
    for wg in (0.25, 0.5, 1.0, 1.5, 2.5):
        for disc in (0.0, 0.5, 1.0):
            for method, hops in (("ppr", 2), ("ppr", 3), ("bfs", 1), ("bfs", 2)):
                vs._W_GRAPH, rg.SEED_DISCOUNT, rg.METHOD, rg.HOPS = wg, disc, method, hops
                m = evaluate(indexes, dev, qvecs, "hybrid+graph", model=model, dim=dim)
                grid.append((m["mrr"] + (m["multi_recall@8"] or 0), wg, disc, method, hops, m["mrr"], m["multi_recall@8"]))
    grid.sort(key=lambda r: -r[0])
    for r in grid[:5]:
        print(f"  w_graph={r[1]:<4} seed_discount={r[2]:<4} {r[3]}/{r[4]}  dev MRR {r[5]:.3f} recall@8 {r[6]:.3f}")
    out["tuning_top5"] = [{"w_graph": r[1], "seed_discount": r[2], "method": r[3], "hops": r[4],
                           "dev_mrr": r[5], "dev_recall@8": r[6]} for r in grid[:5]]
    if args_apply_best:
        best = grid[0]
        vs._W_GRAPH, rg.SEED_DISCOUNT, rg.METHOD, rg.HOPS = best[1], best[2], best[3], best[4]
    else:
        vs._W_GRAPH, rg.SEED_DISCOUNT, rg.METHOD, rg.HOPS = (out["knobs_default"][k] for k in
                                                              ("w_graph", "seed_discount", "method", "hops"))
    out["knobs_used"] = {"w_graph": vs._W_GRAPH, "seed_discount": rg.SEED_DISCOUNT, "method": rg.METHOD, "hops": rg.HOPS}
    print(f"\nKnobs used for the reports below: {out['knobs_used']}")

    show("DEV half (tuned on this: optimistic)", dev, "dev")
    res_t = show("TEST half (held out: the honest number)", test, "test")
    res_all = show("ALL multi-hop/compare/global", mh, "all")
    for t in ("compare", "multihop", "global"):
        show(f"  type={t}", [q for q in mh if q["type"] == t], f"type_{t}")
    show("Single-hop set (regression check)", base, "single_hop")
    for a, b in (("hybrid+graph", "hybrid"),):
        d, lo, hi = bootstrap_diff(res_all[a]["_per_q"], res_all[b]["_per_q"])
        print(f"\nMRR {a} - {b} on all multi-hop: {d:+.3f} (95% CI {lo:+.3f}..{hi:+.3f}); "
              f"test half only: {bootstrap_diff(res_t[a]['_per_q'], res_t[b]['_per_q'])[0]:+.3f}")
    if json_out:
        Path(json_out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    return 0


args_apply_best = False


# ── driver ───────────────────────────────────────────────────────────────────
def load_indexes(subjects: list[str]) -> dict:
    from backend import vector_store as vs

    return {s: vs.load_index_dir(vs.BUNDLED_INDEX_DIR / s, s, mmap=False) for s in subjects}


def reset_bm25(indexes: dict) -> None:
    for i in indexes.values():
        i._bm25 = None


def sweep(indexes, queries, qvecs, model, dim) -> list[dict]:
    from backend import vector_store as vs

    base = (vs._RRF_K, vs._W_DENSE, vs._W_BM25, vs._CANDIDATES, vs._BM25_K1, vs._BM25_B)
    rows: list[dict] = []

    def run(label: str, **kw):
        vs._RRF_K, vs._W_DENSE, vs._W_BM25, vs._CANDIDATES, vs._BM25_K1, vs._BM25_B = (
            kw.get("rrf", base[0]), kw.get("wd", base[1]), kw.get("wb", base[2]), kw.get("pool", base[3]),
            kw.get("k1", base[4]), kw.get("b", base[5]))
        reset_bm25(indexes)
        m = evaluate(indexes, queries, qvecs, kw.get("mode", "hybrid"), model=model, dim=dim)
        rows.append({"knob": label, "hit@1": m["hit@1"], "hit@5": m["hit@5"], "hit@10": m["hit@10"], "mrr": m["mrr"]})
        print(f"  {label:<28} hit@1 {m['hit@1']:.3f} hit@5 {m['hit@5']:.3f} hit@10 {m['hit@10']:.3f} MRR {m['mrr']:.3f}")

    print("RRF constant:");   [run(f"rrf_k={v}", rrf=v) for v in (5, 10, 20, 30, 60, 100)]
    print("BM25 weight (dense=1):"); [run(f"w_bm25={v}", wb=v) for v in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.5)]
    print("candidate pool per method:"); [run(f"pool={v}", pool=v) for v in (10, 20, 30, 60)]
    print("BM25 k1:");        [run(f"k1={v}", k1=v) for v in (0.6, 0.9, 1.2, 1.5, 2.0)]
    print("BM25 b:");         [run(f"b={v}", b=v) for v in (0.3, 0.5, 0.75, 0.9)]
    vs._RRF_K, vs._W_DENSE, vs._W_BM25, vs._CANDIDATES, vs._BM25_K1, vs._BM25_B = base
    reset_bm25(indexes)
    return rows


def chunk_sweep(queries: list[dict]) -> list[dict]:
    """BM25-only proxy for chunk size/overlap (free: no embeddings). Dense effects not measured."""
    from backend import vector_store as vs
    from backend.documents import extract_pdf_pages, list_pdf_paths

    pages: dict[str, list] = {}
    for s in sorted({q["subject"] for q in queries}):
        pages[s] = [(p.name, extract_pdf_pages(p)) for p in list_pdf_paths(s)]
    rows = []
    for size, overlap in ((600, 60), (800, 100), (1000, 100), (1500, 150), (2000, 200), (3000, 300)):
        idxs = {}
        for s, pb in pages.items():
            chunks = vs.chunk_pages(pb, size, overlap)
            idxs[s] = vs.VectorIndex(s, chunks, None, {})
        m = evaluate(idxs, queries, {}, "bm25", model="-", dim=0)
        rows.append({"size": size, "overlap": overlap, "n_chunks": sum(len(i.chunks) for i in idxs.values()),
                     "hit@1": m["hit@1"], "hit@5": m["hit@5"], "mrr": m["mrr"]})
        print(f"  size={size:<5} overlap={overlap:<4} chunks={rows[-1]['n_chunks']:<4} "
              f"BM25 hit@1 {m['hit@1']:.3f} hit@5 {m['hit@5']:.3f} MRR {m['mrr']:.3f}")
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--chunk-sweep", action="store_true")
    ap.add_argument("--offline", action="store_true", help="never call the embedding API (cache only)")
    ap.add_argument("--graph", action="store_true", help="multi-hop/global set: dense | bm25 | hybrid | hybrid+graph")
    ap.add_argument("--apply-best", action="store_true", help="with --graph: use the dev-tuned graph knobs")
    ap.add_argument("--json", metavar="PATH")
    args = ap.parse_args(argv)

    from backend import vector_store as vs

    global args_apply_best
    args_apply_best = args.apply_best
    if args.graph:
        return graph_eval(args.offline, args.json)
    queries = load_eval()
    subjects = sorted({q["subject"] for q in queries})
    indexes = load_indexes(subjects)
    model, dim = vs.embedding_model(), vs.embedding_dim()
    for s, i in indexes.items():
        model, dim = i.model, i.dim          # query with each index's own model; bundled ones agree
    qvecs = ensure_qvecs(queries, model, dim, live=not args.offline)
    if any(_qkey(model, dim, q["q"]) not in qvecs for q in queries):
        print("some query vectors are missing (run without --offline once)", file=sys.stderr)
        return 2

    npos = sum(1 for q in queries if q["type"] != "negative")
    print(f"{len(queries)} queries ({npos} answerable, {len(queries) - npos} negative) over {', '.join(subjects)}; "
          f"{sum(len(i.chunks) for i in indexes.values())} chunks; model {model} {dim}d")
    print(f"knobs: rrf_k={vs._RRF_K} w_dense={vs._W_DENSE} w_bm25={vs._W_BM25} pool={vs._CANDIDATES} "
          f"bm25 k1={vs._BM25_K1} b={vs._BM25_B} chunk={vs.CHUNK_SIZE}/{vs.CHUNK_OVERLAP}")
    results = {}
    for mode in ("dense", "bm25", "hybrid"):
        results[mode] = evaluate(indexes, queries, qvecs, mode, model=model, dim=dim)
        print(fmt(results[mode]))
        print("        by type hit@5:", results[mode]["by_type_hit@5"])
    for mode in ("dense", "bm25", "hybrid"):
        p, n = top1_scores(indexes, queries, qvecs, mode, model, dim)
        print(f"negatives {mode}: top-1 score answerable mean {sum(p) / len(p):.3f} vs negative mean "
              f"{sum(n) / len(n):.3f}; separability AUC {auc(p, n):.3f}")
    for a, b in (("hybrid", "dense"), ("hybrid", "bm25")):
        d, lo, hi = bootstrap_diff(results[a]["_per_q"], results[b]["_per_q"])
        print(f"MRR {a} - {b}: {d:+.3f}  (95% bootstrap CI {lo:+.3f}..{hi:+.3f}; CI containing 0 = noise)")
    out = {"summary": {m: {k: v for k, v in r.items() if k != "_per_q"} for m, r in results.items()}}
    if args.sweep:
        print("\nSweep (hybrid, one knob at a time):")
        out["sweep"] = sweep(indexes, queries, qvecs, model, dim)
    if args.chunk_sweep:
        print("\nChunk-size proxy (BM25 only):")
        out["chunk_sweep"] = chunk_sweep(queries)
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
