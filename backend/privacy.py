"""Which requests must keep their content away from the free-tier Gemini models.

Policy (the lead's call, see DEPLOYMENT.md): Gemini's free tier may train on prompts, so anything
built from content a USER uploaded - photos for scan-notes, a user-uploaded subject, a one-off
uploaded PDF, a PDF chat (``chat_pdfs``) - is sent with ``sensitive=True`` and the provider chain
skips the Gemini steps. Bundled subjects (public sample PDFs) and generic topic-only calls are not
sensitive. ``ALLOW_GEMINI_FOR_USER_CONTENT=1`` lifts the restriction (default OFF).

Mechanism: a request-scoped :class:`~contextvars.ContextVar`. A route (or chat session) calls
:func:`mark` once, and every LLM call made afterwards in that request - however deep in the
services - reads :func:`current`. No per-service signature changes, and a forgotten call site fails
safe only if it is also reading :func:`current` (``tests/test_privacy.py`` covers each flow).
"""
from __future__ import annotations

import contextlib
import contextvars

from backend import config

_flag: contextvars.ContextVar[bool] = contextvars.ContextVar("sensitive_content", default=False)


def policy_active() -> bool:
    """True unless the operator allowed Gemini for user content."""
    return config.user_content_sensitive()


def mark(on: bool = True) -> bool:
    """Flag the rest of this request as handling user content. Returns the resulting value.

    Only ever turns the flag ON for the request (never off), and only while the policy is active.
    """
    value = bool(on) and policy_active()
    if value:
        _flag.set(True)
    return _flag.get()


@contextlib.contextmanager
def scope(on: bool = True):
    """Like :func:`mark` but restored on exit - for service code that may be called from a
    long-lived context (scripts, tests) where a leaked flag would outlive the call."""
    token = _flag.set(_flag.get() or (bool(on) and policy_active()))
    try:
        yield _flag.get()
    finally:
        _flag.reset(token)


def current() -> bool:
    """What LLM calls in this request should pass as ``sensitive=``."""
    return _flag.get()


def for_upload() -> bool:
    """An uploaded PDF / photo: always user content."""
    return policy_active()


def is_user_subject(subject: str | None) -> bool:
    """True for a subject that is NOT one of the bundled (public) ones."""
    if not subject:
        return False
    from backend import documents

    try:
        return not documents._is_bundled(documents._public_subject(subject))
    except documents.DocumentError:
        return True            # odd/unknown names: assume it is user content


def for_session(session) -> bool:
    """A chat session: PDF chats (``chat_pdfs``) and chats over a user-uploaded subject."""
    mode = getattr(session, "mode", None)
    if mode == "pdf":
        return policy_active()
    return mode == "subject" and for_subject(getattr(session, "subject", None))


def for_subject(subject: str | None) -> bool:
    """User-uploaded subject -> sensitive (when the policy is active); bundled subject -> not."""
    return policy_active() and is_user_subject(subject)
