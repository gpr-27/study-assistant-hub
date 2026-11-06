#!/usr/bin/env python3
"""One-command embeddings doctor: prints a PASS/FAIL table and exits non-zero on any FAIL.

    ./venv/bin/python scripts/verify_embeddings.py            # 2 live embedContent calls (free tier)
    ./venv/bin/python scripts/verify_embeddings.py --offline  # no network: keys + index integrity + BM25

Checks: Gemini keys configured; one live embed (dimension == configured dim, unit norm, finite);
every bundled index (files present, manifest/model/dim agree with config, rows == chunks, no empty
chunks, vectors finite + unit norm, source-PDF fingerprints current, fully embedded); a live dense
query lands on the right chunk; BM25-only fallback answers; and a dead embedding API still gives
search results. Keys and chunk texts are never printed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

import numpy as np  # noqa: E402

from backend import gemini_embeddings as ge  # noqa: E402
from backend import vector_store as vs  # noqa: E402
from backend.documents import list_pdf_paths, list_subjects  # noqa: E402

ROWS: list[tuple[str, str, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    ROWS.append(("PASS" if ok else "FAIL", name, detail))
    return ok


def skip(name: str, why: str) -> None:
    ROWS.append(("SKIP", name, why))


def check_keys(required: bool = True) -> None:
    n = len(ge.load_keys())
    detail = f"{n} key(s)" + ("" if n >= 2 else " (a 2nd key doubles the free-tier quota)")
    if required:
        check("Gemini keys configured", n >= 1, detail)
    else:                                  # --offline: informational only
        ROWS.append(("PASS" if n else "SKIP", "Gemini keys configured", detail + " [offline: not required]"))


def check_live() -> None:
    model, dim = vs.embedding_model(), vs.embedding_dim()
    try:
        v = ge.embed_query("embedding doctor probe", model=model, dim=dim, deadline_s=15.0)
    except Exception as exc:  # noqa: BLE001
        check("live embed call", False, f"{type(exc).__name__}: {str(exc)[:100]}")
        return
    check("live embed call", True, f"{model}")
    check("dimension == configured", v.shape == (dim,), f"got {v.shape[0]}, want {dim}")
    check("unit norm and finite", bool(np.isfinite(v).all()) and abs(float(np.linalg.norm(v)) - 1.0) < 1e-3,
          f"|v|={float(np.linalg.norm(v)):.5f}")


def check_bundled() -> dict:
    model, dim = vs.embedding_model(), vs.embedding_dim()
    loaded: dict = {}
    for s in list_subjects():
        d = vs.BUNDLED_INDEX_DIR / s
        files_ok = all((d / f).is_file() for f in ("manifest.json", "chunks.json", "vectors.npy"))
        if not check(f"[{s}] index files", files_ok, str(d.relative_to(ROOT)) if d.exists() else "missing"):
            continue
        m = vs.read_manifest(d) or {}
        try:
            idx = vs.load_index_dir(d, s, mmap=False)
        except Exception as exc:  # noqa: BLE001
            check(f"[{s}] loads", False, type(exc).__name__)
            continue
        loaded[s] = idx
        check(f"[{s}] model/dim match config", m.get("model") == model and m.get("dim") == dim,
              f"index {m.get('model')}/{m.get('dim')} vs config {model}/{dim}")
        check(f"[{s}] vectors usable", idx.problem is None and idx.vectors is not None, idx.problem or "ok")
        if idx.vectors is not None:
            check(f"[{s}] rows == chunks", idx.vectors.shape[0] == len(idx.chunks),
                  f"{idx.vectors.shape[0]} vs {len(idx.chunks)}")
            norms = np.linalg.norm(idx.vectors, axis=1)
            check(f"[{s}] unit-norm rows, none zero", bool(np.all(np.abs(norms - 1.0) < 1e-2)),
                  f"min {norms.min():.4f} max {norms.max():.4f}")
        check(f"[{s}] no empty chunks", all(vs._alnum(c["text"]) >= vs.MIN_CHUNK_ALNUM for c in idx.chunks),
              f"{len(idx.chunks)} chunks")
        check(f"[{s}] fully embedded", bool(m.get("complete")) and m.get("n_embedded") == m.get("n_chunks"),
              f"{m.get('n_embedded')}/{m.get('n_chunks')}")
        try:
            current = m.get("files") == vs.fingerprint_paths(list_pdf_paths(s))
        except Exception:  # noqa: BLE001
            current = False
        check(f"[{s}] matches source PDFs", current, "sha256 fingerprints" if current else "PDFs changed: rebuild")
    return loaded


def check_retrieval(loaded: dict, live: bool) -> None:
    probes = {"Operating System": ("how does the operating system handle an interrupt", "interrupt"),
              "Mathematics": ("state Bayes theorem", "bayes")}
    for s, (q, needle) in probes.items():
        idx = loaded.get(s)
        if idx is None:
            continue
        lex = idx.search(q, 3, use_embeddings=False)
        check(f"[{s}] BM25-only fallback answers", bool(lex) and any(needle in d.page_content.lower() for d in lex),
              f"{len(lex)} hits, mode={lex[0].metadata['retrieval'] if lex else '-'}")
        with mock.patch.object(ge, "embed_query", side_effect=ge.GeminiQuotaError("simulated 429")):
            dead = idx.search(q, 3)
        check(f"[{s}] search survives dead embed API", bool(dead) and dead[0].metadata["retrieval"] == "lexical",
              "falls back to lexical")
        if live and idx.vectors is not None:
            try:
                docs = idx.search(q, 3)
                top = " ".join(d.page_content.lower() for d in docs)
                check(f"[{s}] live hybrid query hits", needle in top and docs[0].metadata["retrieval"] != "lexical",
                      f"mode={docs[0].metadata['retrieval']}")
            except Exception as exc:  # noqa: BLE001
                check(f"[{s}] live hybrid query hits", False, type(exc).__name__)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--offline", action="store_true", help="skip the live Gemini calls")
    args = ap.parse_args(argv)

    check_keys(required=not args.offline)
    live = not args.offline and ge.have_keys()
    if live:
        check_live()
    else:
        skip("live embed call", "offline or no keys")
    loaded = check_bundled()
    check_retrieval(loaded, live and any(r[1] == "live embed call" and r[0] == "PASS" for r in ROWS))
    h = vs.retrieval_health()
    check("health summary (/api/health/deep)", bool(h.get("ok")), f"bundled ok={h.get('ok')}, breaker_open={h['embedding']['breaker_open']}")

    w = max(len(r[1]) for r in ROWS)
    print(f"{'STATUS':<6}  {'CHECK':<{w}}  DETAIL")
    for st, name, detail in ROWS:
        print(f"{st:<6}  {name:<{w}}  {detail}")
    bad = sum(1 for r in ROWS if r[0] == "FAIL")
    print(f"\n{sum(1 for r in ROWS if r[0] == 'PASS')} passed, {bad} failed, "
          f"{sum(1 for r in ROWS if r[0] == 'SKIP')} skipped")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
