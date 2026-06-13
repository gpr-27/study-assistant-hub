#!/usr/bin/env python3
"""Pre-bake the NotebookLM-style session-start overview for each bundled subject.

Run once locally (needs GROQ_API_KEY in .env):

    python scripts/prebake_overviews.py

For every bundled subject it generates the overview + starter questions with ONE
Groq call and writes them to ``faiss_index/<subject>/overview.json``, which is
committed and baked into the Docker image. At runtime ``backend.services.overview``
loads that file instantly — no PDF read, no Groq call — so the overview card appears
the moment a session opens, instead of waiting a few seconds for generation.

User-uploaded subjects (no committed overview.json) still generate on first open and
cache the result; only the bundled subjects are pre-baked here.

After running:
    git add faiss_index/*/overview.json
    git commit -m 'chore: pre-bake subject overviews'
    git push
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from backend.config import FAISS_INDEX_DIR, validate_config
from backend.documents import list_subjects
from backend.services import overview as overview_service


def main() -> None:
    validate_config()
    subjects = list_subjects()

    print("=" * 60)
    print(" Pre-baking session-start overviews")
    print(f" Subjects: {', '.join(subjects)}")
    print("=" * 60)
    print()

    ok = err = 0
    for subject in subjects:
        try:
            # refresh=True forces a fresh generation (bypasses any cache/prebake).
            card = overview_service.build_overview(subject=subject, owner=None, refresh=True)
            if not card.get("overview") and not card.get("questions"):
                err += 1
                print(f"  ✗  {subject}  →  generation returned empty (rate limited?). Re-run.")
                continue
            out = {
                "overview": card["overview"],
                "questions": card["questions"],
                "sources": card["sources"],
            }
            dest = FAISS_INDEX_DIR / subject / "overview.json"
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
            ok += 1
            print(f"  ✓  {subject}  →  {len(card['questions'])} questions  ({dest.relative_to(ROOT)})")
        except Exception as exc:  # noqa: BLE001
            err += 1
            print(f"  ✗  {subject}  →  {exc}")

    print()
    print("=" * 60)
    print(f"  Overviews: {ok} baked, {err} failed")
    print("=" * 60)
    if ok:
        print("\nNext steps:\n  git add faiss_index/*/overview.json\n  git commit -m 'chore: pre-bake subject overviews'\n  git push")


if __name__ == "__main__":
    main()
