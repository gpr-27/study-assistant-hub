"""Per-chunk PDF summaries — built once, reused everywhere.

The generation tools (quiz, flashcards, study-planner summaries, plan) used to
``read_pdf()`` the *whole* document on every request and then clip it to the
first ``MAX_CONTENT_TOKENS`` — re-reading the same PDF each time and silently
dropping everything past the clip.

Instead, each subject PDF is split into chunks ONCE and every chunk is given a
short summary. The compact list of chunk summaries covers the ENTIRE document,
so it is what the tools feed to the LLM thereafter — no more re-reading the full
PDF, and no more losing the document's tail.

The summaries are cached so they are built only when a PDF (or the summary model)
actually changes:

  * in-process memory (per worker),
  * on disk under ``faiss_index/<subject>/chunk_summaries/<pdf>.json`` (survives
    within a container's life, shared across workers), and
  * in MongoDB when configured (survives restarts/redeploys even with no disk —
    keyed by a content hash so a PDF re-hydrated from Mongo with a new mtime still
    hits the cache instead of re-billing the LLM).

Every Groq client is created lazily inside a function — importing this module
never needs a key or the network.
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading

from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend import db
from backend.config import FAISS_INDEX_DIR, MAX_CONTENT_TOKENS, model_for, require_groq_key
from backend.documents import list_pdf_names, read_pdf

log = logging.getLogger(__name__)

# Summary chunks are deliberately larger than the 1500-char retrieval chunks: a
# focused summary per ~3k-char block keeps whole-document coverage while bounding
# the number of LLM calls at build time (especially on Groq's free tier).
_CHUNK_SIZE: int = 3000
_CHUNK_OVERLAP: int = 200

# A rough chars-per-token factor used to batch chunks under the model's token
# budget without importing the tokenizer (which would need a network download).
_CHARS_PER_TOKEN: int = 4
_BATCH_CHAR_BUDGET: int = MAX_CONTENT_TOKENS * _CHARS_PER_TOKEN

# In-process cache: fingerprint -> list[str]. The fingerprint already encodes the
# PDF text + model, so it is the whole identity of a summary set.
_cache: dict[str, list[str]] = {}
_build_lock = threading.Lock()


class ChunkSummaryError(Exception):
    """Raised when chunk summaries cannot be generated for a document."""


def _fingerprint(text: str, model: str | None) -> str:
    """Stable identity for a summary set: hash of (model + extracted text).

    Content-based (not mtime-based) so a PDF re-hydrated from MongoDB onto a fresh
    disk — which changes its mtime — still resolves to the same cached summaries.
    """
    h = hashlib.sha256()
    h.update(((model or "") + "\n").encode("utf-8"))
    h.update(text.encode("utf-8"))
    return h.hexdigest()[:32]


def _split_text(text: str) -> list[str]:
    """Split a document's text into summary-sized chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=_CHUNK_SIZE,
        chunk_overlap=_CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )
    return [c for c in splitter.split_text(text) if c.strip()]


def _batch(chunks: list[str]) -> list[list[str]]:
    """Group chunks into batches that fit one LLM call's char budget."""
    batches: list[list[str]] = []
    current: list[str] = []
    size = 0
    for chunk in chunks:
        if current and size + len(chunk) > _BATCH_CHAR_BUDGET:
            batches.append(current)
            current, size = [], 0
        current.append(chunk)
        size += len(chunk)
    if current:
        batches.append(current)
    return batches


def _summarize_batch(chunks: list[str], model: str | None) -> list[str]:
    """Summarize one batch of chunks in a single Groq JSON call.

    Returns a list of concise summaries. Per-chunk alignment is best-effort: the
    summaries are concatenated downstream, so a count mismatch is tolerated (we
    keep whatever non-empty summaries the model returns).
    """
    from groq import APIError, Groq

    numbered = "\n\n".join(f"[Chunk {i + 1}]\n{c}" for i, c in enumerate(chunks))
    prompt = (
        f"You are building study notes from {len(chunks)} content chunks of a lecture "
        "document. For EACH chunk, write a thorough, self-contained set of notes that a "
        "student could use to make exam questions WITHOUT seeing the original text.\n\n"
        "Requirements for every chunk's notes:\n"
        "1. Capture ALL important points — do not omit specifics. Prefer completeness over brevity.\n"
        "2. Preserve concrete facts: definitions, key terms, formulas/equations, algorithms, "
        "steps, classifications, named entities, dates, numbers, and worked examples.\n"
        "3. Keep technical terms verbatim; explain each briefly so it stands alone.\n"
        "4. Stay strictly faithful to the source — never invent facts not present in the chunk.\n"
        "5. Use compact bullet points (newline-separated) within each chunk's note string.\n\n"
        f"{numbered}\n\n"
        'Return ONLY a JSON object of the form {"summaries": ["detailed notes for chunk 1", '
        '"detailed notes for chunk 2", ...]} with exactly one entry per chunk, in order.'
    )
    client = Groq()
    try:
        resp = client.chat.completions.create(
            model=model_for("summary", model),
            messages=[
                {"role": "system", "content": (
                    "You produce faithful, COMPREHENSIVE study notes from lecture content — "
                    "preserving every important fact, definition, and detail so questions can be "
                    "generated from your notes alone. Never add information not in the source. "
                    "Always reply with valid JSON."
                )},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        data = json.loads(resp.choices[0].message.content or "{}")
    except APIError:
        raise  # rate limit / auth / etc. are formatted by the global handlers
    except json.JSONDecodeError as exc:
        raise ChunkSummaryError(f"The model returned invalid JSON while summarizing: {exc}") from exc
    except Exception as exc:  # noqa: BLE001 — wrap any other transport/SDK failure
        raise ChunkSummaryError(f"Chunk summarization failed: {exc}") from exc

    items = data.get("summaries") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return []
    return [s.strip() for s in items if isinstance(s, str) and s.strip()]


def _build(text: str, model: str | None) -> list[str]:
    """Build chunk summaries for a document's text (one or more Groq calls)."""
    require_groq_key()
    chunks = _split_text(text)
    if not chunks:
        raise ChunkSummaryError("No readable text was found to summarize.")
    summaries: list[str] = []
    for group in _batch(chunks):
        summaries.extend(_summarize_batch(group, model))
    if not summaries:
        raise ChunkSummaryError("The model returned no usable chunk summaries.")
    return summaries


# ---- disk cache ----
def _disk_path(subject: str, pdf: str):
    safe = hashlib.sha256(pdf.encode("utf-8")).hexdigest()[:16]
    return FAISS_INDEX_DIR / subject / "chunk_summaries" / f"{safe}.json"


def _read_disk(subject: str, pdf: str, fingerprint: str) -> list[str] | None:
    path = _disk_path(subject, pdf)
    if not path.is_file():
        return None
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if doc.get("fingerprint") != fingerprint:
        return None
    summaries = doc.get("summaries")
    return summaries if isinstance(summaries, list) else None


def _write_disk(subject: str, pdf: str, fingerprint: str, summaries: list[str]) -> None:
    path = _disk_path(subject, pdf)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"fingerprint": fingerprint, "summaries": summaries}, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError as exc:  # a read-only/full disk must not break generation
        log.debug("Could not cache chunk summaries to disk for %s/%s: %s", subject, pdf, exc)


# ---- public API ----
def get_or_build(subject: str, pdf: str, model: str | None = None) -> list[str]:
    """Return the chunk summaries for a subject PDF, building + caching them once.

    Resolution order: memory → disk → MongoDB → build (LLM). The freshly built set
    is written back to every layer so later requests (and restarts) are free.
    """
    text = read_pdf(subject, pdf)
    if not text or not text.strip():
        raise ChunkSummaryError(f"No extractable text in: {subject}/{pdf}")
    fingerprint = _fingerprint(text, model_for("summary", model))

    cached = _cache.get(fingerprint)
    if cached is not None:
        return cached

    with _build_lock:
        cached = _cache.get(fingerprint)
        if cached is not None:
            return cached

        summaries = _read_disk(subject, pdf, fingerprint)
        source = "disk"
        if summaries is None:
            summaries = db.load_chunk_summaries(fingerprint)
            source = "mongo"
        if summaries is None:
            log.info("Building chunk summaries for %s/%s…", subject, pdf)
            summaries = _build(text, model)
            source = "built"

        # Backfill every cache layer so the next lookup (any worker, any restart) is free.
        _cache[fingerprint] = summaries
        if source != "disk":
            _write_disk(subject, pdf, fingerprint, summaries)
        if source != "mongo":
            db.store_chunk_summaries(fingerprint, subject, pdf, summaries)
        if source == "built":
            log.info("Built %d chunk summaries for %s/%s.", len(summaries), subject, pdf)
        return summaries


def summaries_as_content(summaries: list[str]) -> str:
    """Join chunk summaries into a single content block for the generation tools."""
    return "\n\n".join(f"- {s}" for s in summaries)


def content_for(subject: str, pdf: str, model: str | None = None) -> str:
    """Convenience: the joined chunk-summary content for a subject PDF."""
    return summaries_as_content(get_or_build(subject, pdf, model))


def build_for_subject(subject: str, model: str | None = None) -> int:
    """Build (and cache) chunk summaries for every PDF in a subject.

    Called right after a subject is added so its uploads are summarized up front.
    Best-effort: a per-PDF failure is logged and skipped so one bad PDF never fails
    the whole upload. Returns the number of PDFs successfully summarized.
    """
    done = 0
    for pdf in list_pdf_names(subject):
        try:
            get_or_build(subject, pdf, model)
            done += 1
        except Exception:  # noqa: BLE001 — build is best-effort; lazy build covers misses later
            log.warning("Could not pre-build chunk summaries for %s/%s.", subject, pdf, exc_info=True)
    return done
