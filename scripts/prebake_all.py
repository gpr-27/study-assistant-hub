#!/usr/bin/env python3
"""Pre-bake the retrieval indexes for every bundled subject.

Thin wrapper kept for muscle memory — the real implementation is ``scripts/build_index.py``
(numpy vectors + chunks written to ``data/index/<subject>/``; no FAISS, no langchain).

    python scripts/prebake_all.py          # == python scripts/build_index.py

Then commit ``data/index/`` so the deployment ships the indexes read-only.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_index import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
