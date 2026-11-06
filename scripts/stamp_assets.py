#!/usr/bin/env python3
"""Stamp frontend asset URLs with a content-hash version so long CDN caching is safe.

Usage:  python3 scripts/stamp_assets.py            # rewrite frontend/index.html in place
        python3 scripts/stamp_assets.py --check    # exit 1 if index.html is not up to date
        python3 scripts/stamp_assets.py --print    # just print the build hash

What it does (idempotent; running twice yields identical output):
  * computes BUILD = first 10 hex chars of sha256 over every file under frontend/assets
    (relative path + bytes, sorted) -- so the stamp changes iff any asset changed;
  * rewrites every  href="/assets/..."  /  src="/assets/..."  in frontend/index.html to
    "/assets/...?v=BUILD" (replacing any previous ?v=...);
  * sets <meta name="asset-version" content="BUILD">, which app.js reads to stamp the view
    scripts it injects lazily (and the head script reads for the editorial skin CSS).

Pair it with vercel.json headers:
  /assets/(.*)  ->  Cache-Control: public, max-age=31536000, immutable
  /index.html   ->  Cache-Control: public, max-age=0, must-revalidate
Run it in the build step (e.g. "buildCommand": "python3 scripts/stamp_assets.py").
No dependencies beyond the Python standard library.
"""
import hashlib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend"
INDEX = FRONTEND / "index.html"
ASSETS = FRONTEND / "assets"


def build_hash() -> str:
    h = hashlib.sha256()
    for p in sorted(ASSETS.rglob("*")):
        if p.is_file() and not p.name.startswith("."):
            h.update(p.relative_to(ASSETS).as_posix().encode())
            h.update(b"\0")
            h.update(p.read_bytes())
            h.update(b"\0")
    return h.hexdigest()[:10]


URL_RE = re.compile(r'''((?:href|src)=")(/assets/[^"?#]+)(?:\?v=[^"]*)?(")''')
META_RE = re.compile(r'(<meta name="asset-version" content=")[^"]*(")')


def stamp(html: str, build: str) -> str:
    html = URL_RE.sub(lambda m: f"{m.group(1)}{m.group(2)}?v={build}{m.group(3)}", html)
    html = META_RE.sub(lambda m: f"{m.group(1)}{build}{m.group(2)}", html)
    return html


def main(argv) -> int:
    build = build_hash()
    if "--print" in argv:
        print(build)
        return 0
    src = INDEX.read_text(encoding="utf-8")
    out = stamp(src, build)
    if "--check" in argv:
        if out != src:
            print(f"index.html is stale (expected build {build})", file=sys.stderr)
            return 1
        print(f"ok: {build}")
        return 0
    if out != src:
        INDEX.write_text(out, encoding="utf-8")
        print(f"stamped frontend/index.html with v={build}")
    else:
        print(f"already up to date (v={build})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
