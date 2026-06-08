#!/usr/bin/env python3
"""Pre-bake FAISS indexes AND chunk summaries for all bundled subjects.

Run once locally (needs GROQ_API_KEY in .env):

    python scripts/prebake_all.py

What it does
────────────
1. FAISS indexes  — built from the local embedding model (keyless, fast).
   Saved to faiss_index/<subject>/ and baked into the Docker image.

2. Chunk summaries — built by calling the LLM once per subject PDF.
   Saved to:
     • faiss_index/chunk_summaries/<fingerprint>.json  (baked into Docker image)
     • MongoDB chunk_summaries collection              (survives restarts)
   Uses SUMMARY_MODEL (default: llama-3.1-8b-instant) — 500k tokens/day free tier.

After running:
    git add faiss_index/
    git commit -m 'chore: pre-bake FAISS indexes and chunk summaries'
    git push    # Render auto-deploys; no LLM cost for bundled subjects at runtime

New subjects added via the UI go through the same pipeline automatically on upload.
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from backend.config import validate_config
from backend.documents import list_subjects, list_pdf_names
from backend.vector_store import build_subject_index
from backend.chunk_summaries import get_or_build

_RETRY_DELAYS = [30, 60, 120]   # seconds to wait between rate-limit retries


def _build_with_retry(subject: str, pdf: str) -> list[str]:
    """Call get_or_build with simple retry on 429 rate-limit errors."""
    from groq import RateLimitError
    last_exc = None
    for attempt, delay in enumerate([0] + _RETRY_DELAYS, start=1):
        if delay:
            print(f"      rate limit — retrying in {delay}s (attempt {attempt})…")
            time.sleep(delay)
        try:
            return get_or_build(subject, pdf)
        except RateLimitError as exc:
            last_exc = exc
        except Exception:
            raise
    raise last_exc


def main() -> None:
    validate_config()
    subjects = list_subjects()

    print("=" * 60)
    print(" Pre-baking FAISS indexes + chunk summaries")
    print(f" Subjects: {', '.join(subjects)}")
    print("=" * 60)

    # ── Phase 1: FAISS indexes (keyless — fast) ───────────────────
    print("\n▶  Phase 1: FAISS indexes (local embedding model)\n")
    faiss_ok = faiss_err = 0
    for subject in subjects:
        try:
            chunks = build_subject_index(subject)
            faiss_ok += 1
            print(f"  ✓  {subject}  →  {chunks} chunks indexed")
        except Exception as exc:
            faiss_err += 1
            print(f"  ✗  {subject}  →  {exc}")

    # ── Phase 2: chunk summaries (LLM — uses SUMMARY_MODEL) ───────
    print("\n▶  Phase 2: Chunk summaries (LLM)\n")
    total_pdfs = summ_ok = summ_err = 0
    for subject in subjects:
        pdfs = list_pdf_names(subject)
        print(f"  {subject}  ({len(pdfs)} PDF{'s' if len(pdfs) != 1 else ''})")
        for pdf in pdfs:
            total_pdfs += 1
            try:
                summaries = _build_with_retry(subject, pdf)
                summ_ok += 1
                print(f"    ✓  {pdf}  →  {len(summaries)} chunk summaries")
            except Exception as exc:
                summ_err += 1
                print(f"    ✗  {pdf}  →  {exc}")
        print()

    # ── Summary ───────────────────────────────────────────────────
    print("=" * 60)
    print(f"  FAISS indexes  : {faiss_ok} built, {faiss_err} failed")
    print(f"  Chunk summaries: {summ_ok}/{total_pdfs} PDFs done, {summ_err} failed")
    print("=" * 60)

    if summ_err:
        print("\n⚠  Some summaries failed (likely rate limits).")
        print("   Re-run this script after the limit resets to fill the gaps.")
        print("   Already-built summaries are cached and won't be rebuilt.\n")

    if faiss_ok + summ_ok > 0:
        print("\nNext steps:")
        print("  git add faiss_index/")
        print("  git commit -m 'chore: pre-bake FAISS indexes and chunk summaries'")
        print("  git push")


if __name__ == "__main__":
    main()
