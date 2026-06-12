#!/usr/bin/env python3
"""Pre-bake FAISS indexes for all bundled subjects.

Run once locally (needs GEMINI_API_KEY in .env for hosted embeddings):

    python scripts/prebake_all.py

Each subject's PDFs are embedded and indexed into faiss_index/<subject>/, which is
committed and baked into the Docker image. At runtime the content-hash manifest
matches, so the index is merely LOADED — no re-embedding, no Gemini call, no key
needed on the server for bundled subjects.

After running:
    git add faiss_index/
    git commit -m 'chore: pre-bake FAISS indexes'
    git push    # Render auto-deploys

New subjects added via the UI go through the same pipeline automatically on upload.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from backend.config import validate_config
from backend.documents import list_subjects
from backend.vector_store import build_subject_index


def main() -> None:
    validate_config()
    subjects = list_subjects()

    print("=" * 60)
    print(" Pre-baking FAISS indexes")
    print(f" Subjects: {', '.join(subjects)}")
    print("=" * 60)
    print()

    ok = err = 0
    for subject in subjects:
        try:
            chunks = build_subject_index(subject)
            ok += 1
            print(f"  ✓  {subject}  →  {chunks} chunks indexed")
        except Exception as exc:
            err += 1
            print(f"  ✗  {subject}  →  {exc}")

    print()
    print("=" * 60)
    print(f"  FAISS indexes: {ok} built, {err} failed")
    print("=" * 60)

    if ok:
        print("\nNext steps:")
        print("  git add faiss_index/")
        print("  git commit -m 'chore: pre-bake FAISS indexes'")
        print("  git push")


if __name__ == "__main__":
    main()
