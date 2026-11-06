"""User-safe error text.

Raw exception strings must never reach a client or a saved document: pymongo errors embed Atlas
shard hostnames, HTTP-client errors embed URLs, and so on. Code that has to turn an unexpected
exception into prose calls :func:`public_message` (details go to the server log instead).
"""
from __future__ import annotations

DB_DOWN_MESSAGE = "The database is temporarily unavailable. Please retry in a moment."
GENERIC_MESSAGE = "Something went wrong on our side. Please try again in a moment."


def is_mongo_error(exc: BaseException) -> bool:
    """True for pymongo / bson errors (matched by module so pymongo stays a lazy import)."""
    return any(c.__module__.split(".")[0] in ("pymongo", "bson", "gridfs") for c in type(exc).__mro__)


def public_message(exc: BaseException) -> str:
    """A fixed, safe sentence for an unexpected exception (never includes ``str(exc)``)."""
    return DB_DOWN_MESSAGE if is_mongo_error(exc) else GENERIC_MESSAGE


# ---- "only Gemini keys are configured but the request is private" -------------------------------
PRIVATE_CONTENT_DETAIL = (
    "This request uses content you uploaded, which is never sent to Google's free-tier Gemini models "
    "(they may train on prompts), and no other AI provider is configured. Ask the site owner to add an "
    "Anthropic or Groq key (or to set ALLOW_GEMINI_FOR_USER_CONTENT=1)."
)
PRIVATE_CONTENT_LABEL = "Model: none (private content - Gemini disabled)"


def is_private_content_unavailable(exc: BaseException) -> bool:
    """True when an ``LLMChainError`` means "sensitive request, no non-Gemini key" (llm_chain.error())."""
    return "sensitive request" in str(exc)


def llm_unavailable(exc: BaseException, default_detail: str, default_label: str) -> tuple[str, str]:
    """``(detail, label)`` for a 503 caused by ``LLMChainError``; explains the private-content case."""
    if getattr(exc, "http_status", None) == 413:
        return getattr(exc, "PUBLIC_DETAIL", default_detail), getattr(exc, "label", default_label)
    if is_private_content_unavailable(exc):
        return PRIVATE_CONTENT_DETAIL, PRIVATE_CONTENT_LABEL
    return default_detail, getattr(exc, "label", None) or default_label


def llm_http_status(exc: BaseException) -> int:
    """HTTP status for an ``LLMChainError``: 413 for "request too large", else 503."""
    return int(getattr(exc, "http_status", 503))
