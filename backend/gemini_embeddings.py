"""Google Gemini embeddings — a lightweight langchain ``Embeddings`` backend.

Replaces the local, keyless fastembed model with Google's hosted
``gemini-embedding-2`` model, reached over the public Generative Language REST
API. This deliberately avoids the heavy ``langchain-google-genai`` /
``google-generativeai`` dependency tree: it is a thin ``requests`` wrapper over
two endpoints, which also lets us set the per-call ``taskType`` correctly
(``RETRIEVAL_DOCUMENT`` when indexing, ``RETRIEVAL_QUERY`` when searching) for
better retrieval quality than a one-size-fits-all wrapper.

Why this matters operationally:
  * No model is loaded into the process, so the ~150–250 MB resident footprint of
    the local embedder disappears — the single biggest pressure on Render's 512 MB
    free tier — and the FAISS/onnxruntime libomp clash that crashed the server is
    gone (onnxruntime is never imported).
  * Embedding becomes a network call. Index *builds* now cost API requests instead
    of CPU; loading a pre-built index still costs nothing.

The default 3072-dim output is L2-normalized by the API, which is what FAISS's
L2 index expects. A smaller ``output_dimensionality`` can be requested to shrink
the index, but reduced vectors are NOT unit-norm, so we re-normalize them here.
"""
from __future__ import annotations

import logging
import re
import time

import requests
from langchain_core.embeddings import Embeddings

log = logging.getLogger(__name__)

_API_ROOT = "https://generativelanguage.googleapis.com/v1beta"
# batchEmbedContents accepts up to 100 requests per call. NOTE: the free-tier quota
# counts each *content* (not each HTTP call), so a batch of N consumes N of the
# 100-requests-per-minute budget — large subjects WILL hit 429 and must wait out
# the per-minute window (see _sleep_for_retry / honoring the API's retryDelay).
_BATCH_SIZE = 100
_TIMEOUT = 60
_MAX_RETRIES = 6
_MAX_BACKOFF = 65          # cap a single wait at ~one quota window + margin
_RETRYABLE_STATUS = {500, 502, 503, 504}


class GeminiEmbeddingError(RuntimeError):
    """Raised when the Gemini embedding API cannot be reached or returns an error."""


class GeminiQuotaError(GeminiEmbeddingError):
    """Raised when the Gemini embedding quota is exhausted and retries are spent."""


def _retry_after_seconds(body: dict) -> float | None:
    """Extract the server-requested retry delay (seconds) from a 429 response."""
    err = body.get("error", {}) if isinstance(body, dict) else {}
    for detail in err.get("details", []) or []:
        if str(detail.get("@type", "")).endswith("RetryInfo"):
            m = re.match(r"([0-9.]+)s", str(detail.get("retryDelay", "")))
            if m:
                return float(m.group(1))
    m = re.search(r"retry in ([0-9.]+)s", err.get("message", ""))
    return float(m.group(1)) if m else None


def _normalize(vec: list[float]) -> list[float]:
    """L2-normalize a vector (no-op-safe on a zero vector)."""
    norm = sum(x * x for x in vec) ** 0.5
    if norm == 0.0:
        return vec
    return [x / norm for x in vec]


class GeminiEmbeddings(Embeddings):
    """langchain ``Embeddings`` backed by Gemini's ``embedContent`` endpoints."""

    def __init__(self, model: str, api_key: str, output_dimensionality: int | None = None,
                 fallback_api_key: str | None = None) -> None:
        if not api_key:
            raise GeminiEmbeddingError("GEMINI_API_KEY is required for Gemini embeddings.")
        # Accept either "gemini-embedding-2" or the fully-qualified "models/...".
        self._model = model if model.startswith("models/") else f"models/{model}"
        self._api_key = api_key
        # Quota fallback: a SECOND key (separate Google project = separate quota
        # bucket) used when the primary key's quota is spent. Same model on both
        # keys, so vectors stay compatible with the existing FAISS indexes — a
        # fallback *model* would NOT be (its vectors can't query this index).
        self._fallback_api_key = (fallback_api_key or "").strip() or None
        self._dim = output_dimensionality
        self._reduced = output_dimensionality is not None  # reduced dims need re-normalizing
        self._session = requests.Session()

    # ── internal request helpers (honor 429 retryDelay + backoff) ───────────
    def _post(self, endpoint: str, payload: dict, *,
              max_retries: int | None = None, max_quota_wait: float | None = None) -> dict:
        """POST with quota fallback: primary key first, then the fallback key.

        The fallback key (optional GEMINI_API_KEY_FALLBACK) is a separate Google
        project, i.e. a separate quota bucket — same model, so vectors remain
        compatible with the existing indexes.
        """
        try:
            return self._post_with_key(self._api_key, endpoint, payload,
                                       max_retries=max_retries, max_quota_wait=max_quota_wait)
        except GeminiQuotaError:
            if not self._fallback_api_key:
                raise
            log.info("Primary Gemini key quota spent — retrying on the fallback key.")
            return self._post_with_key(self._fallback_api_key, endpoint, payload,
                                       max_retries=max_retries, max_quota_wait=max_quota_wait)

    def _post_with_key(self, api_key: str, endpoint: str, payload: dict, *,
                       max_retries: int | None = None, max_quota_wait: float | None = None) -> dict:
        """``max_retries``/``max_quota_wait`` let interactive callers fail fast.

        Index builds want the patient default loop (waiting out a per-minute
        window is fine in a background build). A live chat message does NOT:
        phones/proxies kill the connection at ~60s, so a long quota wait turns
        into a dead "Load failed" instead of a clean, retryable 429.
        """
        retries = max_retries or _MAX_RETRIES
        url = f"{_API_ROOT}/{self._model}:{endpoint}?key={api_key}"
        last_err: Exception | None = None
        quota_hit = False
        for attempt in range(retries):
            try:
                resp = self._session.post(url, json=payload, timeout=_TIMEOUT)
            except requests.RequestException as exc:  # network blip → retry
                last_err = exc
                log.debug("Gemini embed network error (attempt %d): %s", attempt + 1, exc)
                time.sleep(min(2 ** attempt, 8))
                continue

            if resp.status_code == 200:
                return resp.json()
            if resp.status_code in (401, 403):
                raise GeminiEmbeddingError(
                    f"Gemini embeddings auth failed ({resp.status_code}). "
                    f"Check GEMINI_API_KEY. Body: {resp.text[:200]}"
                )
            if resp.status_code == 429:
                # Free-tier per-minute quota exhausted: wait the server-stated delay.
                quota_hit = True
                last_err = GeminiQuotaError(resp.text[:200])
                try:
                    delay = _retry_after_seconds(resp.json())
                except ValueError:
                    delay = None
                wait = min((delay or 2 ** attempt) + 1, max_quota_wait or _MAX_BACKOFF)
                log.info("Gemini embed quota hit; waiting %.0fs (attempt %d/%d).", wait, attempt + 1, retries)
                time.sleep(wait)
                continue
            if resp.status_code in _RETRYABLE_STATUS:
                last_err = GeminiEmbeddingError(f"{resp.status_code}: {resp.text[:200]}")
                log.debug("Gemini embed retryable %s (attempt %d).", resp.status_code, attempt + 1)
                time.sleep(min(2 ** attempt, 8))
                continue
            raise GeminiEmbeddingError(f"Gemini embeddings error {resp.status_code}: {resp.text[:300]}")

        if quota_hit:
            raise GeminiQuotaError(
                "Gemini embedding quota exhausted (free tier: 100 requests/minute). "
                "Wait a minute and try again, or use a key with higher limits."
            )
        raise GeminiEmbeddingError(f"Gemini embeddings failed after {retries} attempts: {last_err}")

    def _content(self, text: str, task_type: str) -> dict:
        req: dict = {
            "model": self._model,
            "content": {"parts": [{"text": text or " "}]},
            "taskType": task_type,
        }
        if self._dim is not None:
            req["outputDimensionality"] = self._dim
        return req

    def _post_normalize(self, values: list[float]) -> list[float]:
        return _normalize(values) if self._reduced else values

    # ── langchain Embeddings interface ──────────────────────────────────────
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of chunk texts (taskType=RETRIEVAL_DOCUMENT), batched."""
        out: list[list[float]] = []
        for start in range(0, len(texts), _BATCH_SIZE):
            batch = texts[start:start + _BATCH_SIZE]
            payload = {"requests": [self._content(t, "RETRIEVAL_DOCUMENT") for t in batch]}
            data = self._post("batchEmbedContents", payload)
            embeddings = data.get("embeddings")
            if not embeddings or len(embeddings) != len(batch):
                raise GeminiEmbeddingError(
                    f"Gemini batch returned {len(embeddings or [])} vectors for {len(batch)} inputs."
                )
            out.extend(self._post_normalize(e["values"]) for e in embeddings)
            log.debug("Embedded %d/%d document chunks via Gemini.", len(out), len(texts))
        return out

    def embed_query(self, text: str) -> list[float]:
        """Embed a single search query (taskType=RETRIEVAL_QUERY)."""
        # Interactive path (chat retrieval): fail FAST on quota. A live chat must
        # not wait out a per-minute window — a fully-throttled message that waited
        # ~16s/model still retrieved 0 docs in production, so a long wait buys
        # nothing. Two short attempts (≤2s each) ride a brief RPM burst, then raise
        # so the retriever can switch to the fallback index (or skip retrieval).
        data = self._post("embedContent", self._content(text, "RETRIEVAL_QUERY"),
                          max_retries=2, max_quota_wait=2)
        values = data.get("embedding", {}).get("values")
        if not values:
            raise GeminiEmbeddingError(f"Gemini query embedding returned no vector: {str(data)[:200]}")
        return self._post_normalize(values)
