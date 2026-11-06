#!/usr/bin/env python3
"""Pre-build the retrieval indexes for the bundled subjects (run OFFLINE, commit the result).

    .venv/bin/python scripts/build_index.py                  # build/resume every bundled subject
    .venv/bin/python scripts/build_index.py -s Mathematics   # one subject
    .venv/bin/python scripts/build_index.py --force          # ignore existing vectors, re-embed
    .venv/bin/python scripts/build_index.py --check          # report status only, no API calls
    .venv/bin/python scripts/build_index.py --verify "what is a deadlock" -s "Operating System"

Output (read-only at runtime, shipped with the deployment)::

    data/index/<subject>/{chunks.json, vectors.npy, graph.json, manifest.json}

Properties:
  * RESUMABLE — a checkpoint is written after every embedding batch, and vectors for chunks
    whose text is unchanged are reused, so an interrupted/rate-limited run just continues.
  * RATE-LIMIT AWARE — the embedding client keeps a per-key sliding window (free tier: ~100
    contents/min/key) across GEMINI_API_KEY and GEMINI_API_KEY_2 and sleeps instead of 429-ing.
  * SKIPS up-to-date subjects (same PDF bytes, model, dim, chunker settings, fully embedded).

Keys come from the environment / .env only and are never printed.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from backend import vector_store as vs  # noqa: E402
from backend.documents import list_pdf_paths, list_subjects  # noqa: E402


def _is_current(directory: Path, paths: list[Path]) -> bool:
    m = vs.read_manifest(directory)
    if not m or not m.get("complete"):
        return False
    return (m.get("model") == vs.embedding_model() and m.get("dim") == vs.embedding_dim()
            and m.get("chunk_size") == vs.CHUNK_SIZE and m.get("chunk_overlap") == vs.CHUNK_OVERLAP
            and m.get("format") == vs.INDEX_VERSION and m.get("files") == vs.fingerprint_paths(paths))


def build_subject(subject: str, *, force: bool = False, out_root: Path | None = None) -> dict:
    out_root = out_root or vs.BUNDLED_INDEX_DIR
    directory = out_root / subject
    paths = list_pdf_paths(subject)
    if not force and _is_current(directory, paths):
        vs.ensure_graph_in_dir(directory)          # concept graph is free (no API): attach it if missing/stale
        return {"subject": subject, "status": "up-to-date", **(vs.read_manifest(directory) or {})}
    reuse = None
    if not force and (directory / "manifest.json").is_file():
        try:
            reuse = vs.load_index_dir(directory, subject, mmap=False)
        except Exception:  # noqa: BLE001
            reuse = None
    t0 = time.time()

    def progress(done: int, total: int) -> None:
        print(f"    {subject}: embedded {done}/{total} new chunks ({time.time() - t0:.0f}s)", flush=True)

    manifest = vs.build_index_to_dir(subject, paths, directory, subject=subject,
                                     budget_s=3600.0, patient=True, reuse=reuse,
                                     checkpoint_every=1, log_fn=progress)
    manifest["status"] = "built" if manifest["complete"] else "PARTIAL (re-run to resume)"
    return {"subject": subject, **manifest}


def verify(subject: str, queries: list[str], k: int = 3) -> None:
    idx = vs.load_index_dir(vs.BUNDLED_INDEX_DIR / subject, subject)
    for q in queries:
        t = time.time()
        docs = idx.search(q, k)
        print(f"\nQ: {q}   ({(time.time() - t) * 1000:.0f} ms, {docs[0].metadata['retrieval'] if docs else '-'})")
        for d in docs:
            snippet = d.page_content.replace("\n", " ")[:110]
            print(f"   [{d.metadata['source']} p{d.metadata['page'] + 1} {d.metadata['score']:.3f}] {snippet}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-s", "--subject", action="append", help="subject(s) to build (default: all bundled)")
    ap.add_argument("--force", action="store_true", help="re-embed everything")
    ap.add_argument("--check", action="store_true", help="show status only")
    ap.add_argument("--graph-only", action="store_true",
                    help="(re)build only the concept graph (graph.json) for existing indexes; no API calls")
    ap.add_argument("--verify", nargs="*", metavar="QUERY", help="run sample queries against the built index")
    args = ap.parse_args(argv)

    subjects = args.subject or list_subjects()
    print(f"Model: {vs.embedding_model()}  dim: {vs.embedding_dim()}  subjects: {', '.join(subjects)}")
    if args.verify is not None:
        for s in subjects:
            verify(s, args.verify or ["summarize the main ideas"])
        return 0
    failed = 0
    if args.graph_only:
        for s in subjects:
            ok = vs.ensure_graph_in_dir(vs.BUNDLED_INDEX_DIR / s)
            g = (vs.read_manifest(vs.BUNDLED_INDEX_DIR / s) or {}).get("graph") or {}
            print(f"  {s}: graph {'ok' if ok else 'MISSING'} - {g.get('n_concepts')} concepts, {g.get('n_edges')} edges, "
                  f"{g.get('n_communities')} communities")
            failed += 0 if ok else 1
        return 1 if failed else 0
    for s in subjects:
        if args.check:
            m = vs.read_manifest(vs.BUNDLED_INDEX_DIR / s) or {}
            print(f"  {s}: chunks={m.get('n_chunks')} embedded={m.get('n_embedded')} complete={m.get('complete')}")
            continue
        try:
            r = build_subject(s, force=args.force)
            print(f"  {s}: {r['status']} — {r.get('n_embedded')}/{r.get('n_chunks')} chunks embedded "
                  f"({r.get('model')}, {r.get('dim')}d)")
            failed += 0 if r.get("complete") else 1
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  {s}: FAILED — {type(exc).__name__}: {str(exc)[:200]}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
