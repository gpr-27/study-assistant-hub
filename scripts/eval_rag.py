"""End-to-end RAG evaluation for the study chat (retrieval, citations, abstention, latency).

    venv/bin/python scripts/eval_rag.py --mode fake                # plumbing: fake provider server
    venv/bin/python scripts/eval_rag.py --mode live --limit 25     # real chain (spends a few tokens)
    venv/bin/python scripts/eval_rag.py --mode live --json OUT.json

Every question runs through the REAL chat turn (``backend.services.chat.stream_turn``) against the
bundled Operating System / Mathematics indexes. A baseline run re-does only the retrieval the chat
used before this pipeline (``heuristic_queries`` + ``retrieve_bounded``) for the before/after columns.

Metrics
  hit@3 / hit@8   a retrieved passage covers a gold (file, page range)            [questions with gold]
  recall@8        share of gold ranges covered by the final passages                [multi-chunk questions]
  citation valid  share of [n] markers the model emitted (before verification) that name a real passage
  abstain acc     'not in your notes' answer exactly for the out-of-scope questions (type oos)
  faithfulness    share of answer sentences whose content words are >= 60 % in the cited passage text
                  (lexical proxy, snippet-limited; not a human judgement)
  pre-gen ms      time from the start of the turn to the ``generate`` step (fast path, no LLM before it)
  aux LLM calls   blocking LLM calls made before generation (0 on the fast path)

Fake mode starts ``scripts/fake_providers.py`` and points the chain at it, so no keys or spend are used:
its answers are canned, so only the plumbing and the retrieval numbers are meaningful there.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DATA = ROOT / "tests" / "data" / "rag_eval.json"
FAKE_PORT = 0            # chosen at run time (a fixed port can belong to another local server)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _setup_env(mode: str) -> None:
    if mode == "fake":
        # Dummy keys + the test-only redirect: load_dotenv never overrides variables already set.
        os.environ["LLM_CHAIN_TEST_BASE_URL"] = f"http://127.0.0.1:{FAKE_PORT}"
        for var in ("GROQ_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"):
            os.environ[var] = "fake-key-for-eval"
    else:
        from dotenv import dotenv_values

        for k, v in dotenv_values(ROOT / ".env").items():     # keys are never printed
            if k.endswith("_API_KEY") or k.endswith("_API_KEY_2"):
                if v and not os.environ.get(k):
                    os.environ[k] = v


def _start_fake_server() -> subprocess.Popen:
    proc = subprocess.Popen([sys.executable, str(ROOT / "scripts" / "fake_providers.py"), "--port", str(FAKE_PORT),
                             "--latency-ms", "150", "--token-ms", "2"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(100):
        try:
            with socket.create_connection(("127.0.0.1", FAKE_PORT), timeout=0.2):
                return proc
        except OSError:
            time.sleep(0.1)
    proc.kill()
    raise SystemExit("fake provider server did not start")


_PAGE_CACHE: dict[str, list[dict]] = {}


def _page_texts(subject: str, pages: list[tuple[str, int, int]]) -> list[str]:
    """Full chunk texts of the bundled index for the given (file, first page, last page) 1-based ranges."""
    if subject not in _PAGE_CACHE:
        path = ROOT / "data" / "index" / subject / "chunks.json"
        _PAGE_CACHE[subject] = json.loads(path.read_text()) if path.is_file() else []
    out: list[str] = []
    for src, a, b in pages:
        for ch in _PAGE_CACHE[subject]:
            if ch["source"] == src and isinstance(ch.get("page"), int) and a <= ch["page"] + 1 <= b:
                out.append(ch["text"])
    return out


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.0f}%"


def _p(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    return values[min(len(values) - 1, int(round(q * (len(values) - 1))))]


async def run_question(item: dict, counter: dict, owner: str) -> dict:
    from backend import agentic_rag, llm_chain, vector_store
    from backend.rag import metrics
    from backend.services import chat

    session = chat.Session(id="", owner=owner, mode="subject", label="eval", subject=item["subject"],
                           topic=None, chat_id=None, nonce=f"eval-{item['id']}")
    hist = item.get("history") or None
    # ---- new pipeline: the real chat turn
    t0 = time.perf_counter()
    gen_ms = calls_at_gen = None
    passages: list[dict] = []
    steps: list[dict] = []
    raw: list[str] = []
    done = error = None
    before = counter["n"]
    async for ev in chat.stream_turn(session, item["question"], client_history=hist):
        kind = ev["type"]
        if kind == "step":
            st = ev["step"]
            steps.append(st)
            if st.get("kind") == "retrieve":
                passages = st.get("passages") or []
            if st.get("kind") == "generate" and gen_ms is None:
                gen_ms = (time.perf_counter() - t0) * 1000
                calls_at_gen = counter["n"] - before
        elif kind == "token":
            raw.append(ev["text"])
        elif kind == "done":
            done = ev
        elif kind == "error":
            error = ev
    answer = (done or {}).get("answer", "")
    cited_n = {c["n"] for c in (done or {}).get("citations", []) if c.get("cited")}

    # ---- baseline: the retrieval the chat used before this pipeline
    retriever, _persona = chat._prepare(session)
    base_queries = agentic_rag.heuristic_queries(item["question"], chat._history_messages(
        chat._clean_client_history(hist))) or [item["question"]]
    b0 = time.perf_counter()
    base_docs, _mode = await chat.retrieve_bounded(retriever, base_queries)
    base_ms = (time.perf_counter() - b0) * 1000
    base_blocks = [(Path(d.metadata.get("source", "")).name, (d.metadata.get("page") or 0) + 1, (d.metadata.get("page") or 0) + 1)
                   for d in base_docs]

    blocks = [(p["source"], p["page"], p.get("page_end") or p["page"]) for p in passages]
    gold = item.get("gold") or []
    cites_raw = [int(x) for x in re.findall(r"\[(\d{1,3})\]", "".join(raw))]
    valid = sum(1 for n in cites_raw if 1 <= n <= len(passages))
    # support = full text of the PAGES the cited passages come from (the snippets shown in the UI are cut short)
    cited_pages = [(p["source"], p["page"], p.get("page_end") or p["page"]) for p in passages if p["n"] in cited_n]
    support = _page_texts(item["subject"], cited_pages)
    faith_ok, faith_total = metrics.faithfulness(answer, support) if answer and support else (0, 0)
    # CPU cost of the pipeline itself (plan + lexical search + rerank/MMR/expansion), no network
    searchers = agentic_rag.rret.searchers_for(retriever)
    c0 = time.perf_counter()
    plan = agentic_rag.plan_query(item["question"], chat._history_messages(chat._clean_client_history(hist)), retriever)
    agentic_rag._retrieve_sync(searchers, plan, plan.queries, lexical_only=True, timeout_s=None)
    cpu_ms = (time.perf_counter() - c0) * 1000
    return {
        "id": item["id"], "type": item["type"], "question": item["question"],
        "answer": answer, "error": (error or {}).get("detail"),
        "gold": gold,
        "hit3": metrics.gold_hit(blocks, gold, 3) if gold else None,
        "hit8": metrics.gold_hit(blocks, gold, 8) if gold else None,
        "recall8": (sum(any(metrics.covers(b, g) for b in blocks[:8]) for g in gold) / len(gold)) if gold else None,
        "base_hit3": metrics.gold_hit(base_blocks, gold, 3) if gold else None,
        "base_hit8": metrics.gold_hit(base_blocks, gold, 8) if gold else None,
        "cite_total": len(cites_raw), "cite_valid": valid,
        "cited": sorted(cited_n), "n_passages": len(passages),
        "abstained": metrics.is_abstain(answer), "expect_abstain": item["type"] == "oos",
        "faith_ok": faith_ok, "faith_total": faith_total,
        "gen_ms": gen_ms, "aux_calls_before_gen": calls_at_gen,
        "adaptive": any("Weak match" in (s.get("title") or "") for s in steps),
        "cpu_ms": cpu_ms, "base_ms": base_ms,
        "confidence": next((s.get("confidence") for s in steps if s.get("kind") == "retrieve"), None),
    }


async def main_async(items: list[dict], owner: str) -> list[dict]:
    from backend import llm_chain

    counter = {"n": 0}
    real_chat = llm_chain.chat

    def counting_chat(*a, **k):
        counter["n"] += 1
        return real_chat(*a, **k)

    llm_chain.chat = counting_chat
    rows = []
    try:
        for i, item in enumerate(items, 1):
            row = await run_question(item, counter, owner)
            rows.append(row)
            print(f"  [{i:2d}/{len(items)}] {item['id']:<16} gen {row['gen_ms'] or 0:6.0f} ms  "
                  f"hit8={row['hit8']}  abstain={row['abstained']}  err={bool(row['error'])}", flush=True)
    finally:
        llm_chain.chat = real_chat
    return rows


def summarise(rows: list[dict]) -> dict:
    g = [r for r in rows if r["gold"]]
    oos = [r for r in rows if r["expect_abstain"]]
    inscope = [r for r in rows if not r["expect_abstain"]]
    cite_total = sum(r["cite_total"] for r in rows)
    cite_valid = sum(r["cite_valid"] for r in rows)
    f_ok = sum(r["faith_ok"] for r in rows)
    f_tot = sum(r["faith_total"] for r in rows)
    gen = [r["gen_ms"] for r in rows if r["gen_ms"] is not None]
    cpu = [r["cpu_ms"] for r in rows]
    return {
        "questions": len(rows),
        "errors": sum(1 for r in rows if r["error"]),
        "hit@3": {"baseline": _rate([r["base_hit3"] for r in g]), "pipeline": _rate([r["hit3"] for r in g])},
        "hit@8": {"baseline": _rate([r["base_hit8"] for r in g]), "pipeline": _rate([r["hit8"] for r in g])},
        "recall@8_pipeline": _rate([r["recall8"] for r in g if len(r["gold"]) > 1] or [r["recall8"] for r in g]),
        "citation_validity": (cite_valid / cite_total) if cite_total else None,
        "cite_markers": cite_total,
        "abstain_accuracy": _rate([r["abstained"] == r["expect_abstain"] for r in rows]),
        "abstain_on_oos": _rate([r["abstained"] for r in oos]),
        "false_abstain_inscope": _rate([r["abstained"] for r in inscope]),
        "faithfulness_proxy": (f_ok / f_tot) if f_tot else None,
        "pre_gen_ms_p50": statistics.median(gen) if gen else None,
        "pre_gen_ms_p90": _p(gen, 0.9),
        "pipeline_cpu_ms_p50": statistics.median(cpu) if cpu else None,
        "pipeline_cpu_ms_p90": _p(cpu, 0.9),
        "aux_llm_calls_before_gen_max": max((r["aux_calls_before_gen"] or 0) for r in rows) if rows else 0,
        "fast_path_questions": sum(1 for r in rows if not r["adaptive"]),
        "adaptive_second_pass": sum(1 for r in rows if r["adaptive"]),
    }


def _rate(vals: list) -> float | None:
    vals = [bool(v) for v in vals if v is not None]
    return (sum(vals) / len(vals)) if vals else None


def print_summary(s: dict) -> None:
    print("\n=== RAG eval summary ===")
    print(f"questions {s['questions']}  errors {s['errors']}  adaptive second pass {s['adaptive_second_pass']}")
    for k in ("hit@3", "hit@8"):
        print(f"{k:<22} baseline {_pct(s[k]['baseline']):>5}   pipeline {_pct(s[k]['pipeline']):>5}")
    print(f"recall@8 (multi-chunk) pipeline {_pct(s['recall@8_pipeline']):>5}")
    print(f"citation validity      {_pct(s['citation_validity']):>5}   ({s['cite_markers']} markers, before verification)")
    print(f"abstain accuracy       {_pct(s['abstain_accuracy']):>5}   out-of-scope caught {_pct(s['abstain_on_oos'])}, "
          f"in-scope wrongly abstained {_pct(s['false_abstain_inscope'])}")
    print(f"faithfulness (proxy)   {_pct(s['faithfulness_proxy']):>5}")
    print(f"pre-generation time    p50 {s['pre_gen_ms_p50'] or 0:.0f} ms  p90 {s['pre_gen_ms_p90'] or 0:.0f} ms  "
          f"(includes query embeddings)")
    print(f"pipeline CPU (no net)  p50 {s['pipeline_cpu_ms_p50'] or 0:.1f} ms  p90 {s['pipeline_cpu_ms_p90'] or 0:.1f} ms")
    print(f"LLM calls before generation: max {s['aux_llm_calls_before_gen_max']} "
          f"(fast path on {s['fast_path_questions']}/{s['questions']} questions)")


# ── retrieval-pipeline benchmark: agentic loop before/after (no LLM, simulated embedding latency) ─────────
PIPELINE_CONFIGS = {
    "baseline (hybrid only)": {"GRAPH_RAG_GLOBAL": "0", "AGENTIC_RAG_PARTS": "0", "GRAPH_RAG_RELATED": "0", "GRAPH_RAG_RANK": "0"},
    "parts lazy + global": {"AGENTIC_RAG_SPECULATE": "0", "GRAPH_RAG_RANK": "0"},
    "parts parallel + global": {"GRAPH_RAG_RANK": "0"},
    "parts parallel + global + graph ranker": {"GRAPH_RAG_RANK": "1"},
}
_ENV_KEYS = ("GRAPH_RAG_GLOBAL", "AGENTIC_RAG_PARTS", "GRAPH_RAG_RELATED", "GRAPH_RAG_RANK", "AGENTIC_RAG_SPECULATE")


def pipeline_eval(embed_ms: float, repeats: int, embed_parts: bool, json_out: str | None) -> int:
    """Run ``agentic_rag.retrieve_events`` (everything before generation) over the multi-hop set with the
    embedding call replaced by cached vectors + a fixed simulated latency, per configuration. Reports the
    pre-generation wall time p50/p95 and how many gold anchors the FINAL blocks (neighbour-merged) contain.
    Sub-query vectors that are not cached are embedded live only with --embed-parts (<= 60 contents)."""
    import numpy as np

    sys.path.insert(0, str(ROOT / "scripts"))
    import eval_retrieval as er
    from backend import agentic_rag, gemini_embeddings as ge, vector_store as vs
    from backend.rag import query as rq

    mh = json.loads(er.MULTIHOP_PATH.read_text(encoding="utf-8"))["queries"]
    idxs = er.load_indexes(sorted({q["subject"] for q in mh}))
    model, dim = next(iter(idxs.values())).model, next(iter(idxs.values())).dim
    subs = sorted({x for q in mh for x in rq.plan_query(q["q"], [], None).subqueries})
    pool = [{"q": t} for t in [q["q"] for q in mh] + subs]
    qv = er.ensure_qvecs(pool, model, dim, live=embed_parts)
    parent = {}
    for q in mh:
        for x in rq.plan_query(q["q"], [], None).subqueries:
            parent.setdefault(x, q["q"])

    def fake_embed(text, *, model=model, dim=dim, deadline_s=None, **kw):
        time.sleep(embed_ms / 1000.0)
        v = qv.get(er._qkey(model, dim, text))
        if v is None:
            v = qv.get(er._qkey(model, dim, parent.get(text, "")))
        if v is None:
            raise ge.GeminiEmbeddingError("no cached vector")
        return v

    real = ge.embed_query
    ge.embed_query = fake_embed
    out: dict = {}
    try:
        for name, env in PIPELINE_CONFIGS.items():
            for k in _ENV_KEYS:
                os.environ.pop(k, None)
            os.environ.update(env)
            rows = []
            for _ in range(repeats):
                for q in mh:
                    retr = vs.get_subject_retriever(q["subject"], 8, None)
                    t0 = time.perf_counter()

                    async def run():
                        async for e in agentic_rag.retrieve_events(retr, q["q"], [], budget_s=1.5):
                            if e["type"] == "_evidence":
                                return e["evidence"]

                    ev = asyncio.run(run())
                    ms = (time.perf_counter() - t0) * 1000
                    texts = [er.norm(b.text) for b in ev.blocks]
                    cov = sum(1 for a in q["gold"] if any(er.anchor_hit(a, t) for t in texts)) / len(q["gold"])
                    rows.append({"type": q["type"], "ms": ms, "recall": cov, "n_blocks": len(ev.blocks),
                                 "kind": ev.plan.kind, "multi": bool(ev.plan.subqueries), "split": q["split"]})
            out[name] = rows
    finally:
        ge.embed_query = real
        for k in _ENV_KEYS:
            os.environ.pop(k, None)

    def agg(rs):
        ms = [r["ms"] for r in rs]
        return (f"recall {np.mean([r['recall'] for r in rs]):.3f}  pre-gen p50 {np.percentile(ms, 50):.0f} ms  "
                f"p95 {np.percentile(ms, 95):.0f} ms  blocks {np.mean([r['n_blocks'] for r in rs]):.1f}")

    print(f"\nPipeline (retrieve_events, no LLM) on {len(mh)} multi-hop questions x{repeats}; simulated embedding latency "
          f"{embed_ms:.0f} ms per call")
    summary = {}
    for name, rows in out.items():
        print(f"\n{name}")
        summary[name] = {}
        for label, sel in (("all", rows), ("compare/multi-part (decomposed)", [r for r in rows if r["multi"]]),
                           ("global", [r for r in rows if r["type"] == "global"]),
                           ("test half", [r for r in rows if r["split"] == "test"]),
                           ("other", [r for r in rows if not r["multi"] and r["type"] != "global"])):
            if sel:
                print(f"  {label:<34} n={len(sel):<3} {agg(sel)}")
                summary[name][label] = {"n": len(sel), "recall": float(np.mean([r["recall"] for r in sel])),
                                        "p50_ms": float(np.percentile([r["ms"] for r in sel], 50)),
                                        "p95_ms": float(np.percentile([r["ms"] for r in sel], 95))}
    if json_out:
        Path(json_out).write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("fake", "live"), default="fake")
    ap.add_argument("--limit", type=int, default=0, help="first N questions only (0 = all)")
    ap.add_argument("--only-live", action="store_true", help="questions flagged live (<= 25 for the live run)")
    ap.add_argument("--json", help="write per-question rows + summary here")
    ap.add_argument("--pipeline", action="store_true",
                    help="agentic-loop benchmark on the multi-hop set (cached vectors, simulated embed latency, no LLM)")
    ap.add_argument("--embed-ms", type=float, default=300.0, help="--pipeline: simulated embedding latency per call")
    ap.add_argument("--repeats", type=int, default=2, help="--pipeline: repetitions for the latency percentiles")
    ap.add_argument("--embed-parts", action="store_true",
                    help="--pipeline: embed uncached sub-queries LIVE once (<= ~60 contents) and cache them")
    args = ap.parse_args()
    if args.pipeline:
        return pipeline_eval(args.embed_ms, args.repeats, args.embed_parts, args.json)
    global FAKE_PORT
    if args.mode == "fake":
        FAKE_PORT = _free_port()
    _setup_env(args.mode)
    fake = _start_fake_server() if args.mode == "fake" else None
    try:
        from backend.auth import LOCAL_OWNER

        items = json.loads(DATA.read_text())["questions"]
        if args.only_live:
            items = [q for q in items if q.get("live")]
        if args.limit:
            items = items[: args.limit]
        print(f"Running {len(items)} questions in {args.mode} mode ...", flush=True)
        rows = asyncio.run(main_async(items, LOCAL_OWNER))
        summary = summarise(rows)
        print_summary(summary)
        if args.json:
            Path(args.json).write_text(json.dumps({"mode": args.mode, "summary": summary, "rows": rows}, indent=1))
            print(f"\nwrote {args.json}")
        return 0
    finally:
        if fake is not None:
            fake.terminate()


if __name__ == "__main__":
    raise SystemExit(main())
