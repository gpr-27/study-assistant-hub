"""Scan Handwritten Notes service — vision OCR through the provider chain.

Each photo is sent (as image input) to ``backend.llm_chain.achat`` and transcribed faithfully into
clean Markdown. Only vision-capable models can serve this (Claude, ``qwen/qwen3.8-27b`` on Groq and
the Gemini models); the chain is responsible for restricting the candidate list when ``images`` is
given. Pages are transcribed with bounded concurrency so a 6-photo scan costs ~2 calls of latency,
not 6, and nothing blocks the event loop.

If the installed ``llm_chain`` has no ``images=`` support yet, the scan fails with a clear 503
(``ScanUnavailableError``) instead of silently talking to a provider directly.
"""
from __future__ import annotations

import asyncio
import inspect
import logging
from dataclasses import dataclass, field

from backend import config, deadline, llm_chain
from backend.llm_chain import LLMChainError

log = logging.getLogger("backend.services.scannotes")

_MAX_OUTPUT_TOKENS = 4096
_CONCURRENCY = 3
_PAGE_DEADLINE_S = 40.0

_SYSTEM = (
    "You transcribe photos of handwritten study notes faithfully into clean Markdown.\n"
    "- Use headings (#, ##) where the notes have titles or section breaks.\n"
    "- Use bullet or numbered lists where the notes use them.\n"
    "- Write math as plain LaTeX-style text (e.g. x^2 + y^2 = r^2), not images or code fences.\n"
    "- Mark anything you genuinely cannot read as [illegible].\n"
    "- Do not add commentary, summaries, or content that is not in the image.\n"
    "Return ONLY the Markdown transcription."
)
_USER = "Transcribe the attached page of handwritten notes."


class ScanNotesError(Exception):
    """Raised when the vision transcription fails (→ 502)."""


class ScanUnavailableError(ScanNotesError):
    """No vision-capable provider is configured / image input isn't enabled (→ 503)."""


MissingGeminiKeyError = ScanUnavailableError      # back-compat name


def supports_images() -> bool:
    try:
        return "images" in inspect.signature(llm_chain.achat).parameters
    except (TypeError, ValueError):
        return False


def require_vision() -> None:
    """Fail fast (503) when no provider key is set or the chain can't take image input."""
    if not llm_chain.has_any_key():
        raise ScanUnavailableError(
            "No AI provider key is configured. Set at least one of the provider API keys to scan notes.")
    if not supports_images():
        raise ScanUnavailableError("Image input is not enabled in the model chain yet; scanning is unavailable.")


# Back-compat alias used by the router/tests.
require_gemini_key = require_vision


async def transcribe_image(image_bytes: bytes, mime_type: str) -> tuple[str, dict | None]:
    """Transcribe ONE photo -> (markdown, public llm). ``LLMChainError`` propagates (→ 503).

    Photos are user-uploaded content: unless ``ALLOW_GEMINI_FOR_USER_CONTENT=1`` the call is
    ``sensitive`` and never reaches the free-tier Gemini models.
    """
    require_vision()
    res = await llm_chain.achat(
        _SYSTEM, [{"role": "user", "content": _USER}],
        max_tokens=_MAX_OUTPUT_TOKENS, temperature=0.1,
        deadline_s=deadline.clamp(_PAGE_DEADLINE_S),            # shares the request's overall budget
        sensitive=config.user_content_sensitive(),
        images=[(image_bytes, mime_type)],
    )
    text = (res.text or "").strip()
    if not text:
        raise ScanNotesError("The model returned an empty transcription; try a clearer photo.")
    return text, res.to_public()


@dataclass
class ScanOutcome:
    """Result of a multi-page scan: the combined Markdown plus which pages failed."""

    markdown: str
    llm: dict | None
    pages: int
    failed_pages: list[int] = field(default_factory=list)

    @property
    def partial(self) -> bool:
        return bool(self.failed_pages)


def _failure_note(exc: BaseException) -> str:
    if isinstance(exc, LLMChainError):
        return "the AI service was unavailable or ran out of time for this page"
    if isinstance(exc, ScanNotesError):
        return str(exc).rstrip(".")
    return "an unexpected error occurred"


async def transcribe_pages(images: list[tuple[bytes, str]]) -> ScanOutcome:
    """Transcribe every photo; keep the pages that succeeded even when others fail.

    Every page already paid for is kept. A failed page becomes a visible marker in the Markdown
    (and is listed in ``failed_pages``) so the user can re-scan just that photo. When EVERY page
    fails the first error is raised (an ``LLMChainError`` wins, so the router returns the usual
    503 with the model label).
    """
    if not images:
        raise ScanNotesError("No images were supplied to transcribe.")
    require_vision()
    sem = asyncio.Semaphore(_CONCURRENCY)

    async def one(blob: bytes, mime: str):
        async with sem:
            return await transcribe_image(blob, mime)

    results = await asyncio.gather(*[one(b, m) for b, m in images], return_exceptions=True)
    for r in results:
        if isinstance(r, asyncio.CancelledError):          # the request itself was cancelled
            raise r
    ok = [r for r in results if not isinstance(r, BaseException)]
    if not ok:
        errors = [r for r in results if isinstance(r, BaseException)]
        raise next((e for e in errors if isinstance(e, LLMChainError)), errors[0])
    sections: list[str] = []
    failed: list[int] = []
    llm = None
    for n, r in enumerate(results, start=1):
        if isinstance(r, BaseException):
            failed.append(n)
            log.warning("Scan page %d failed: %s", n, type(r).__name__)
            sections.append(f"## Page {n}\n\n> [Page {n} could not be transcribed: {_failure_note(r)}. "
                            "Scan this photo again to add it.]")
        else:
            text, page_llm = r
            llm = llm or page_llm
            sections.append(f"## Page {n}\n\n{text}")
    return ScanOutcome(markdown="\n\n".join(sections), llm=llm, pages=len(images), failed_pages=failed)


async def transcribe_images(images: list[tuple[bytes, str]]) -> tuple[str, dict | None]:
    """Transcribe every (bytes, mime_type) photo; join with "## Page N" headers in the caller's order.

    Returns ``(markdown, llm_public)`` where ``llm_public`` is the model of the first page that
    succeeded. Partial failures are marked in the text (see :func:`transcribe_pages`).
    """
    out = await transcribe_pages(images)
    return out.markdown, out.llm
