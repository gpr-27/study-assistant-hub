"""Build the static SPA for Vercel: copy ``frontend/`` -> ``dist/`` and version the assets.

Why: the SPA uses fixed, un-hashed filenames (app.js, views/*.js, styles.css). Serving those with
a long cache would pin users to stale code after a deploy. The versioning logic itself lives in
``scripts/stamp_assets.py`` (one content hash over ``frontend/assets``; it stamps every
``/assets/...`` reference in index.html with ``?v=<hash>`` and sets ``<meta name="asset-version">``
so app.js stamps the views it lazy-loads). This script applies it to the COPY in ``dist/`` so the
source ``frontend/index.html`` (``asset-version`` = "dev") is never rewritten by a deploy.

vercel.json serves URLs carrying ``?v=`` as ``immutable`` for a year (new content => new hash =>
new URL) while ``index.html`` is always revalidated; any asset requested WITHOUT ``?v=`` keeps
``max-age=0, must-revalidate`` (cheap CDN 304s).

No dependencies beyond the standard library.   Usage: python scripts/build_frontend.py [--out dist]
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "frontend"
sys.path.insert(0, str(Path(__file__).resolve().parent))
import minify_assets  # noqa: E402  (comment/whitespace stripper; no deps)
import stamp_assets  # noqa: E402  (sibling script; owns the hashing + HTML rewriting)


def minify_dir(assets: Path) -> int:
    """Strip comments / whitespace from every .css and .js under ``assets`` (in the dist COPY only).
    Returns the bytes saved. A file the minifier mangles would fail ``node --check`` in the tests."""
    saved = 0
    for p in sorted(assets.rglob("*")):
        fn = {".js": minify_assets.minify_js, ".css": minify_assets.minify_css}.get(p.suffix)
        if fn is None or not p.is_file():
            continue
        raw = p.read_text(encoding="utf-8")
        mini = fn(raw)
        if len(mini) < len(raw):
            p.write_text(mini, encoding="utf-8")
            saved += len(raw.encode()) - len(mini.encode())
    return saved


def build(out: Path) -> None:
    if not (SRC / "index.html").is_file():
        raise SystemExit(f"frontend/index.html not found under {SRC}")
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(SRC, out, ignore=shutil.ignore_patterns(".DS_Store", "*.map"))
    build_id = stamp_assets.build_hash()      # hashed over the SOURCE assets, before minification
    saved = minify_dir(out / "assets")
    html = stamp_assets.stamp((SRC / "index.html").read_text(encoding="utf-8"), build_id)
    (out / "index.html").write_text(html, encoding="utf-8")
    total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"build_frontend: {out.name}/ ready - asset-version {build_id}, {total / 1024:.0f} KiB (minify saved {saved / 1024:.0f} KiB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dist")
    build(ROOT / ap.parse_args().out)
