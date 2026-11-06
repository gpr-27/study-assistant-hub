#!/usr/bin/env python3
"""Retrieval load benchmark against the FAKE Gemini provider (no quota, no real keys, no spend).

    ./venv/bin/python scripts/bench_retrieval.py all            # everything below (~3 min)
    ./venv/bin/python scripts/bench_retrieval.py cold           # fresh-process first query
    ./venv/bin/python scripts/bench_retrieval.py concurrent     # 200 concurrent queries, both subjects
    ./venv/bin/python scripts/bench_retrieval.py micro          # numpy/BM25 cost at 135 / 2k / 10k chunks
    ./venv/bin/python scripts/bench_retrieval.py uploads        # 20 users upload different small PDFs
    ./venv/bin/python scripts/bench_retrieval.py quota          # both keys return 429: BM25 answers, breaker

It spawns ``scripts/fake_providers.py`` and redirects ``gemini_embeddings`` to it through the
test-only ``LLM_CHAIN_TEST_BASE_URL`` (ignored on Vercel; see LOADTEST.md). Fake embeddings are
random vectors, so only SPEED / behaviour is measured here, never quality (see eval_retrieval.py).
Retrieval runs on a 40-thread pool, mirroring Starlette's ``run_in_threadpool``.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
PORT = 8897
BASE = f"http://127.0.0.1:{PORT}"


def rss_mb() -> float:
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True, text=True).stdout
    return int(out.strip() or 0) / 1024.0


def pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p / 100.0 * len(xs)))] if xs else float("nan")


def summarize(label: str, xs_ms: list[float]) -> dict:
    d = {"n": len(xs_ms), "p50": round(pct(xs_ms, 50), 1), "p95": round(pct(xs_ms, 95), 1),
         "p99": round(pct(xs_ms, 99), 1), "max": round(max(xs_ms), 1) if xs_ms else None}
    print(f"  {label:<34} n={d['n']:<4} p50 {d['p50']:>8.1f} ms  p95 {d['p95']:>8.1f}  p99 {d['p99']:>8.1f}  max {d['max']}")
    return d


class FakeServer:
    def __init__(self, latency_ms: int = 120) -> None:
        self.latency_ms = latency_ms
        self.proc = None

    def __enter__(self):
        self.proc = subprocess.Popen(
            [sys.executable, str(ROOT / "scripts" / "fake_providers.py"), "--port", str(PORT),
             "--latency-ms", str(self.latency_ms), "--jitter-ms", "30"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        import requests

        for _ in range(100):
            try:
                requests.get(BASE + "/_stats", timeout=0.5)
                break
            except Exception:  # noqa: BLE001
                time.sleep(0.1)
        return self

    def cfg(self, **kw) -> None:
        import requests

        requests.post(BASE + "/_cfg", json=kw, timeout=3)

    def stats(self) -> dict:
        import requests

        return requests.get(BASE + "/_stats", timeout=3).json()

    def __exit__(self, *exc):
        if self.proc:
            self.proc.terminate()
            self.proc.wait(timeout=5)


def fake_env() -> None:
    os.environ["LLM_CHAIN_TEST_BASE_URL"] = BASE
    os.environ.pop("VERCEL", None)
    os.environ["GEMINI_API_KEY"] = "fake-key-one"
    os.environ["GEMINI_API_KEY_2"] = "fake-key-two"


def queries() -> list[tuple[str, str]]:
    ev = json.loads((ROOT / "tests" / "data" / "retrieval_eval.json").read_text(encoding="utf-8"))["queries"]
    return [(q["subject"], q["q"]) for q in ev]


# ── scenarios ────────────────────────────────────────────────────────────────
def cold_child() -> None:
    fake_env()
    t0 = time.perf_counter()
    from backend import vector_store as vs

    t_import = time.perf_counter() - t0
    r0 = rss_mb()
    out: dict = {"import_ms": round(t_import * 1000), "rss_after_import_mb": round(r0, 1)}
    t = time.perf_counter()
    ret = vs.get_subject_retriever("Operating System", 10)
    out["load_index_ms"] = round((time.perf_counter() - t) * 1000, 1)
    for label, q in (("first_query_ms", "How does the operating system handle an interrupt?"),
                     ("second_query_ms", "What is a microkernel?"),
                     ("repeat_query_ms", "How does the operating system handle an interrupt?")):
        t = time.perf_counter()
        docs = ret.invoke(q)
        out[label] = round((time.perf_counter() - t) * 1000, 1)
        out["k"] = len(docs)
    out["rss_after_first_query_mb"] = round(rss_mb(), 1)
    print(json.dumps(out))


def scenario_cold(srv: FakeServer) -> dict:
    print("\n[cold instance] fresh interpreter, first query for 'Operating System' (fake embed latency "
          f"{srv.latency_ms} ms)")
    res = []
    for _ in range(3):
        p = subprocess.run([sys.executable, __file__, "cold-child"], capture_output=True, text=True,
                           cwd=ROOT, timeout=120)
        res.append(json.loads(p.stdout.strip().splitlines()[-1]))
    med = {k: statistics.median(r[k] for r in res) for k in res[0]}
    print("  median of 3 fresh processes:", json.dumps(med))
    return med


def scenario_concurrent(srv: FakeServer, total: int = 200, unique: bool = False) -> dict:
    from backend import gemini_embeddings as ge
    from backend import vector_store as vs

    ge.reset_state()
    vs.clear_cache()
    base = queries()
    rnd = random.Random(3)
    work = []
    for i in range(total):
        s, q = base[rnd.randrange(len(base))]
        work.append((s, f"{q} {i}" if unique else q))
    rets = {s: vs.get_subject_retriever(s, 10) for s in ("Mathematics", "Operating System")}
    rss0 = rss_mb()
    lat_total, lat_service, modes = [], [], {}
    lock = threading.Lock()
    t_sub = time.perf_counter()

    def one(item):
        s, q = item
        t = time.perf_counter()
        docs = rets[s].invoke(q)
        dt = (time.perf_counter() - t) * 1000
        with lock:
            lat_service.append(dt)
            lat_total.append((time.perf_counter() - t_sub) * 1000)
            m = docs[0].metadata["retrieval"] if docs else "none"
            modes[m] = modes.get(m, 0) + 1

    with ThreadPoolExecutor(max_workers=40) as pool:
        list(pool.map(one, work))
    wall = time.perf_counter() - t_sub
    st = ge.status()
    print(f"\n[{total} concurrent queries, {'ALL unique' if unique else 'repeats mixed in'}, 2 subjects, 40 worker threads]")
    out = {"service_ms": summarize("service time (per query)", lat_service),
           "completion_ms": summarize("completion since burst start", lat_total),
           "wall_s": round(wall, 2), "qps": round(total / wall, 1), "modes": modes,
           "rss_mb": round(rss_mb(), 1), "rss_delta_mb": round(rss_mb() - rss0, 1),
           "embed_api_calls": st["embed_calls"], "query_cache_hit_rate": st["query_cache_hit_rate"]}
    print(f"  wall {out['wall_s']} s ({out['qps']} q/s)  modes {modes}  RSS {out['rss_mb']} MB "
          f"(+{out['rss_delta_mb']})  upstream embed calls {st['embed_calls']}  cache hit rate {st['query_cache_hit_rate']}")
    return out


def scenario_micro() -> dict:
    import numpy as np

    from backend import vector_store as vs

    print("\n[micro] per-query CPU on synthetic indexes (no network; vector = random unit 768-d)")
    rng = np.random.default_rng(1)
    vocab = [f"term{i}" for i in range(3000)]
    out = {}
    for n in (135, 2000, 10000):
        chunks = [{"text": " ".join(rng.choice(vocab, 180)), "source": "s.pdf", "page": i % 50} for i in range(n)]
        vecs = rng.standard_normal((n, 768)).astype(np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        idx = vs.VectorIndex("bench", chunks, vecs, {"model": "gemini-embedding-2", "dim": 768, "complete": True})
        t = time.perf_counter()
        idx._bm25_scores("term1 term2")
        build_ms = (time.perf_counter() - t) * 1000
        q = vecs[0]
        ts = []
        for i in range(200):
            t = time.perf_counter()
            idx.rank("term5 term17 term99", q, 10, "hybrid")
            ts.append((time.perf_counter() - t) * 1000)
        out[n] = {"bm25_build_ms": round(build_ms, 1), "rank_p50_ms": round(pct(ts, 50), 2),
                  "rank_p99_ms": round(pct(ts, 99), 2), "mb": round(idx.nbytes / 1e6, 1)}
        print(f"  {n:>6} chunks: BM25 build {build_ms:7.1f} ms (first query only); hybrid rank p50 "
              f"{pct(ts, 50):.2f} ms p99 {pct(ts, 99):.2f} ms; index {idx.nbytes / 1e6:.1f} MB")
    return out


def _small_pdfs(n: int, pages: int, dest: Path) -> list[bytes]:
    from pypdf import PdfReader, PdfWriter

    sources = [PdfReader(str(ROOT / "documents" / s / f)) for s in ("Mathematics", "Operating System")
               for f in ("Lec-1.pdf", "Lec-2.pdf")]
    blobs = []
    for i in range(n):
        r = sources[i % len(sources)]
        start = (i * 5) % max(1, len(r.pages) - pages)
        w = PdfWriter()
        for p in range(start, start + pages):
            w.add_page(r.pages[p])
        import io

        b = io.BytesIO()
        w.write(b)
        blobs.append(b.getvalue())
    return blobs


def scenario_uploads(srv: FakeServer, users: int = 20, pages: int = 8, wait_continue: bool = True) -> dict:
    from backend import documents
    from backend import gemini_embeddings as ge
    from backend import vector_store as vs

    tmp = Path(tempfile.mkdtemp(prefix="bench-up-"))
    documents.DOCUMENTS_DIR = tmp / "documents"
    documents.DOCUMENTS_DIR.mkdir()
    vs._cache_root = lambda: tmp / "index"          # type: ignore[assignment]
    ge.reset_state()
    vs.clear_cache()
    sent: list[tuple[float, int]] = []
    sent_lock = threading.Lock()
    http = ge._http()
    orig_post = http.post

    def counting_post(url, json=None, **kw):
        n = len((json or {}).get("requests", [None])) if "batchEmbed" in url else 1
        r = orig_post(url, json=json, **kw)
        if r.status_code == 200:
            with sent_lock:
                sent.append((time.monotonic(), n))
        return r

    http.post = counting_post                      # type: ignore[assignment]
    blobs = _small_pdfs(users, pages, tmp)
    owners = [f"user_bench{i:02d}" for i in range(users)]
    for o, b in zip(owners, blobs):
        documents.ensure_subject_dir("Notes", o)
        documents.save_pdf("Notes", "slice.pdf", b, o)
    print(f"\n[{users} concurrent users, each uploading a different {pages}-page PDF; budget "
          f"{vs.USER_UPLOAD_EMBED_BUDGET_S:.0f} s, window {ge.RPM_PER_KEY}/min/key x 2 keys]")
    t0 = time.perf_counter()
    res, dur = {}, {}

    def build(o):
        t = time.perf_counter()
        n = vs.build_subject_index("Notes", o)
        st = vs.index_status("Notes", o)
        dur[o] = (time.perf_counter() - t) * 1000
        res[o] = (n, st)

    with ThreadPoolExecutor(max_workers=users) as pool:
        list(pool.map(build, owners))
    wall = time.perf_counter() - t0
    chunks = sum(v[0] for v in res.values())
    complete = sum(1 for v in res.values() if v[1]["complete"])
    emb = sum(v[1]["n_embedded"] for v in res.values())
    print(f"  phase 1 wall {wall:.1f} s: {complete}/{users} subjects complete, {users - complete} partial; "
          f"{emb}/{chunks} chunks embedded; per-upload latency p50 {pct(list(dur.values()), 50):.0f} ms "
          f"p95 {pct(list(dur.values()), 95):.0f} ms max {max(dur.values()):.0f} ms")
    # sliding-window check: max contents sent in any 60 s window (limit 2 keys x RPM)
    mx = 0
    for ts, _ in sent:
        mx = max(mx, sum(n for t, n in sent if ts <= t < ts + 60))
    print(f"  upstream contents sent: {sum(n for _, n in sent)}; max in any 60 s window: {mx} "
          f"(client cap {2 * ge.RPM_PER_KEY})")
    out = {"users": users, "phase1_wall_s": round(wall, 1), "complete": complete, "partial": users - complete,
           "chunks": chunks, "embedded": emb, "upload_ms_p50": round(pct(list(dur.values()), 50)),
           "upload_ms_p95": round(pct(list(dur.values()), 95)), "max_contents_per_60s": mx}
    if complete < users and wait_continue:
        # partial -> continue-index flow: same minute still blocked (progress 0), then it drains
        o = next(o for o in owners if not res[o][1]["complete"])
        t = time.perf_counter()
        r1 = vs.continue_subject_index("Notes", o, budget_s=5.0)
        print(f"  continue-index right away (window still full): {r1['n_embedded']}/{r1['n_chunks']} "
              f"in {(time.perf_counter() - t) * 1000:.0f} ms (no progress expected, returns fast)")
        print("  waiting 62 s for the quota window to slide, then continue-indexing all partials…")
        time.sleep(62)
        left = [o for o in owners if not res[o][1]["complete"]]
        t = time.perf_counter()
        with ThreadPoolExecutor(max_workers=8) as pool:
            fin = list(pool.map(lambda o: vs.continue_subject_index("Notes", o, budget_s=30.0), left))
        print(f"  continue-index of {len(left)} partial subjects: {sum(1 for f in fin if f['complete'])} "
              f"complete after {time.perf_counter() - t:.1f} s")
        out["continue_complete"] = sum(1 for f in fin if f["complete"])
        out["continue_total"] = len(left)
    http.post = orig_post                           # type: ignore[assignment]
    shutil.rmtree(tmp, ignore_errors=True)
    return out


def scenario_quota(srv: FakeServer) -> dict:
    from backend import gemini_embeddings as ge
    from backend import vector_store as vs

    ge.reset_state()
    vs.clear_cache()
    ret = vs.get_subject_retriever("Operating System", 10)
    print("\n[both Gemini keys return 429]")
    srv.cfg(gemini_fail=1.0, gemini_status=429)
    before = srv.stats()["calls"]["gemini"]
    lat, modes = [], {}
    t_all = time.perf_counter()
    qs = [f"explain interrupts variant {i}" for i in range(60)]

    def one(q):
        t = time.perf_counter()
        d = ret.invoke(q)
        lat.append((time.perf_counter() - t) * 1000)
        m = d[0].metadata["retrieval"] if d else "none"
        modes[m] = modes.get(m, 0) + 1

    with ThreadPoolExecutor(max_workers=40) as pool:
        list(pool.map(one, qs))
    calls = srv.stats()["calls"]["gemini"] - before
    st = ge.status()
    print(f"  60 concurrent queries during the outage: modes {modes}, upstream calls {calls}, "
          f"breaker_open={st['breaker_open']} ({st['breaker_remaining_s']} s left), last_error={st['last_error']}")
    summarize("latency during outage", lat)
    t = time.perf_counter()
    for i in range(20):
        ret.invoke(f"another interrupt question {i}")
    print(f"  20 sequential queries while breaker open: {(time.perf_counter() - t) * 1000 / 20:.1f} ms each, "
          f"extra upstream calls {srv.stats()['calls']['gemini'] - before - calls}")
    srv.cfg(gemini_fail=0.0)
    t_rec = time.perf_counter()
    recovered = None
    while time.perf_counter() - t_rec < 80:
        d = ret.invoke(f"recovery probe {time.time()}")
        # any embedding-backed mode counts as recovered: "dense" (no lexical hit for the probe text) is
        # as good as "hybrid" here; only "lexical" means the provider is still being skipped
        if d and d[0].metadata["retrieval"] in ("hybrid", "dense"):
            recovered = time.perf_counter() - t_rec
            break
        time.sleep(1.0)
    print(f"  provider healthy again -> first embedding-backed answer after {recovered:.1f} s "
          f"(breaker/cooldown close)" if recovered is not None else "  did NOT recover within 80 s")
    return {"outage_modes": modes, "upstream_calls_during_outage": calls, "recovery_s": recovered,
            "outage_latency_ms": summarize("latency (again)", lat)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scenario", choices=["all", "cold", "cold-child", "concurrent", "micro", "uploads", "quota"])
    ap.add_argument("--json", metavar="PATH")
    ap.add_argument("--latency-ms", type=int, default=120)
    args = ap.parse_args(argv)
    if args.scenario == "cold-child":
        cold_child()
        return 0
    fake_env()
    results: dict = {}
    with FakeServer(args.latency_ms) as srv:
        want = (lambda n: args.scenario in ("all", n))
        if want("cold"):
            results["cold"] = scenario_cold(srv)
        if want("micro"):
            results["micro"] = scenario_micro()
        if want("concurrent"):
            results["concurrent_mixed"] = scenario_concurrent(srv)
            results["concurrent_unique"] = scenario_concurrent(srv, unique=True)
        if want("quota"):
            results["quota"] = scenario_quota(srv)
        if want("uploads"):
            results["uploads"] = scenario_uploads(srv)
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
