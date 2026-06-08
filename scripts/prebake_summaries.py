#!/usr/bin/env python3
"""Pre-build chunk summaries for all bundled subjects.

Run once locally with a valid GROQ_API_KEY in .env:

    python scripts/prebake_summaries.py

Summaries are saved to:
  - faiss_index/chunk_summaries/<fingerprint>.json  — baked into the Docker image
    via COPY so quiz/flashcard/plan generation never re-reads PDFs or re-calls
    the LLM for bundled subjects.
  - MongoDB chunk_summaries collection (if MONGODB_URI is set) — survives restarts.

After running, commit faiss_index/chunk_summaries/ and redeploy.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from backend.config import validate_config
from backend.documents import list_subjects, list_pdf_names
from backend.chunk_summaries import get_or_build


def main() -> None:
    validate_config()

    subjects = list_subjects()
    print(f"Pre-building chunk summaries for {len(subjects)} subjects…\n")

    total_pdfs = 0
    total_chunks = 0
    errors = 0

    for subject in subjects:
        pdfs = list_pdf_names(subject)
        print(f"  {subject}  ({len(pdfs)} PDF{'s' if len(pdfs) != 1 else ''})")
        for pdf in pdfs:
            try:
                summaries = get_or_build(subject, pdf)
                total_pdfs += 1
                total_chunks += len(summaries)
                print(f"    ✓  {pdf}  →  {len(summaries)} chunks")
            except Exception as exc:
                errors += 1
                print(f"    ✗  {pdf}  →  {exc}")
        print()

    print("─" * 60)
    print(f"Done:  {total_pdfs} PDFs  |  {total_chunks} chunk summaries  |  {errors} errors")
    print()
    print("Next steps:")
    print("  git add faiss_index/chunk_summaries/")
    print("  git commit -m 'chore: pre-bake chunk summaries for bundled subjects'")
    print("  git push   # Render auto-deploys; summaries load from image, not LLM")


if __name__ == "__main__":
    main()
