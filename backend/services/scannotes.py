"""Scan Handwritten Notes service — Gemini vision OCR over the plain REST API.

Mirrors ``backend/gemini_embeddings.py``: a thin ``requests`` wrapper over the
Generative Language REST endpoint base, reusing the same key, timeout, and
retry conventions (429 retryDelay honored, transient 5xx retried with backoff)
— deliberately avoiding the heavy google-generativeai dependency tree.

Each photo is sent as a base64 ``inline_data`` part to ``generateContent`` and
transcribed faithfully into clean Markdown. The vision model id is read from
the GEMINI_VISION_MODEL env var here (config.py is frozen for this feature);
it defaults to "gemini-2.0-flash".
"""
from __future__ import annotations

import base64
import logging
import os
import re
import time

import requests

from backend import config

log = logging.getLogger("backend.services.scannotes")

_API_ROOT = "https://generativelanguage.googleapis.com/v1beta"
_DEFAULT_VISION_MODEL = "gemini-2.0-flash"
_VISION_MODEL_ENV = "GEMINI_VISION_MODEL"
_TIMEOUT = 60
_MAX_RETRIES = 4
_MAX_BACKOFF = 65          # cap a single wait at ~one quota window + margin
_RETRYABLE_STATUS = {500, 502, 503, 504}
# Conservative completion cap per image — a dense handwritten page transcribes
# to well under this, and capping keeps every call's cost bounded.
_MAX_OUTPUT_TOKENS = 4096

_PROMPT = (
    "Transcribe these handwritten study notes faithfully into clean Markdown.\n"
    "- Use headings (#, ##) where the notes have titles or section breaks.\n"
    "- Use bullet or numbered lists where the notes use them.\n"
    "- Write math as plain LaTeX-style text (e.g. x^2 + y^2 = r^2), not images or code fences.\n"
    "- Mark anything you genuinely cannot read as [illegible].\n"
    "- Do not add commentary, summaries, or content that is not in the image.\n"
    "Return ONLY the Markdown transcription."
)


class ScanNotesError(Exception):
    """Raised when the Gemini vision API cannot be reached or returns an error."""


class MissingGeminiKeyError(ScanNotesError):
    """Raised when a scan is attempted without a Gemini API key configured."""


def vision_model() -> str:
    """The Gemini vision model id (env-overridable; config.py stays untouched)."""
    return (os.getenv(_VISION_MODEL_ENV) or "").strip() or _DEFAULT_VISION_MODEL


def require_gemini_key() -> None:
    """Raise a clear error if the Gemini key is missing (mirrors require_groq_key)."""
    if not config.GEMINI_API_KEY:
        raise MissingGeminiKeyError(
            "GEMINI_API_KEY is not set. Add your Gemini key to .env "
            "(get one at https://aistudio.google.com/apikey) to scan handwritten notes."
        )


def _retry_after_seconds(body: dict) -> float | None:
    """Extract the server-requested retry delay (seconds) from a 429 response."""
    err = body.get("error", {}) if isinstance(body, dict) else {}
    for detail in err.get("details", []) or []:
        if str(detail.get("@type", "")).endswith("RetryInfo"):
            m = re.match(r"([0-9.]+)s", str(detail.get("retryDelay", "")))
            if m:
                return float(m.group(1))
    m = re.search(r"retry in ([0-9.]+)s", str(err.get("message", "")))
    return float(m.group(1)) if m else None


def _post_with_retries(url: str, payload: dict) -> dict:
    """POST to Gemini, honoring 429 retryDelay and retrying transient failures.

    Any terminal failure raises ScanNotesError with a readable message (the
    router maps it to a 502).
    """
    last_err: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            resp = requests.post(url, json=payload, timeout=_TIMEOUT)
        except requests.RequestException as exc:  # network blip → retry
            last_err = exc
            log.debug("Gemini vision network error (attempt %d): %s", attempt + 1, exc)
            time.sleep(min(2 ** attempt, 8))
            continue

        if resp.status_code == 200:
            return resp.json()
        if resp.status_code in (401, 403):
            raise ScanNotesError(
                f"Gemini vision auth failed ({resp.status_code}). "
                f"Check GEMINI_API_KEY. Body: {resp.text[:200]}"
            )
        if resp.status_code == 429:
            # Free-tier per-minute quota exhausted: wait the server-stated delay.
            last_err = ScanNotesError(f"Gemini quota exhausted: {resp.text[:200]}")
            try:
                delay = _retry_after_seconds(resp.json())
            except ValueError:
                delay = None
            wait = min((delay or 2 ** attempt) + 1, _MAX_BACKOFF)
            log.info("Gemini vision quota hit; waiting %.0fs (attempt %d/%d).",
                     wait, attempt + 1, _MAX_RETRIES)
            time.sleep(wait)
            continue
        if resp.status_code in _RETRYABLE_STATUS:
            last_err = ScanNotesError(f"{resp.status_code}: {resp.text[:200]}")
            log.debug("Gemini vision retryable %s (attempt %d).", resp.status_code, attempt + 1)
            time.sleep(min(2 ** attempt, 8))
            continue
        raise ScanNotesError(f"Gemini vision error {resp.status_code}: {resp.text[:300]}")

    raise ScanNotesError(
        f"Gemini vision request failed after {_MAX_RETRIES} attempts: {last_err}"
    )


def _extract_text(data: dict) -> str:
    """Pull the transcription text out of a generateContent response."""
    candidates = data.get("candidates") if isinstance(data, dict) else None
    if not candidates:
        feedback = (data or {}).get("promptFeedback", {}) if isinstance(data, dict) else {}
        reason = feedback.get("blockReason")
        if reason:
            raise ScanNotesError(f"Gemini declined to transcribe this image (blocked: {reason}).")
        raise ScanNotesError("Gemini returned no transcription for this image; try a clearer photo.")
    parts = (candidates[0].get("content") or {}).get("parts") or []
    text = "\n".join(p.get("text", "") for p in parts if isinstance(p, dict)).strip()
    if not text:
        raise ScanNotesError("Gemini returned an empty transcription; try a clearer photo.")
    return text


def transcribe_image(image_bytes: bytes, mime_type: str) -> str:
    """Transcribe ONE photo of handwritten notes into Markdown (one Gemini call)."""
    require_gemini_key()
    model = vision_model()
    qualified = model if model.startswith("models/") else f"models/{model}"
    url = f"{_API_ROOT}/{qualified}:generateContent?key={config.GEMINI_API_KEY}"
    payload = {
        "contents": [{
            "parts": [
                {"inline_data": {
                    "mime_type": mime_type,
                    "data": base64.b64encode(image_bytes).decode("ascii"),
                }},
                {"text": _PROMPT},
            ],
        }],
        "generationConfig": {
            "temperature": 0.1,                      # faithful transcription, not creativity
            "maxOutputTokens": _MAX_OUTPUT_TOKENS,   # explicit, conservative output cap
        },
    }
    return _extract_text(_post_with_retries(url, payload))


def transcribe_images(images: list[tuple[bytes, str]]) -> str:
    """Transcribe every (bytes, mime_type) photo and join them with page headers.

    Pages keep the caller's order; each gets a "## Page N" heading so a
    multi-photo scan reads as one continuous Markdown document.
    """
    if not images:
        raise ScanNotesError("No images were supplied to transcribe.")
    sections: list[str] = []
    for number, (blob, mime) in enumerate(images, start=1):
        text = transcribe_image(blob, mime).strip()
        sections.append(f"## Page {number}\n\n{text}")
        log.debug("Transcribed page %d/%d (%d bytes, %s).", number, len(images), len(blob), mime)
    return "\n\n".join(sections)
