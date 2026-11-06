"""One shared wall-clock budget per request.

Vercel kills a function at ``maxDuration`` (60 s). Several flows chain more than one slow step
(chat tool turn = intent detection + generation; photo scan = two rounds of vision calls), so a
per-step timeout is not enough: the *sum* must fit. The request middleware calls
:func:`begin_request` once; every slow step then asks :func:`clamp` how long it may run, and
:func:`clamp` raises :class:`DeadlineExceeded` (an ``LLMChainError``, hence the usual clean 503
with the ``llm`` label) when too little is left to make a call worthwhile.

Off Vercel there is no deadline (``remaining()`` is ``None``), so local dev and the unit tests
behave exactly as before unless a test calls :func:`start` explicitly.
"""
from __future__ import annotations

import contextvars
import time

from backend import config
from backend.llm_chain import LLMChainError

_deadline: contextvars.ContextVar[float | None] = contextvars.ContextVar("request_deadline", default=None)

MIN_USEFUL_S = 4.0          # below this an LLM round-trip is not worth starting


class DeadlineExceeded(LLMChainError):
    """The request's overall time budget is spent."""

    def __init__(self, msg: str = "Request time budget exhausted") -> None:
        super().__init__(msg, status=None, attempts=[])


def start(total_s: float | None) -> None:
    """Begin a budget of ``total_s`` seconds in the current context (``None`` clears it)."""
    _deadline.set(None if total_s is None else time.monotonic() + float(total_s))


def begin_request() -> None:
    """Called once per API request by the middleware: budget on Vercel, none elsewhere."""
    start(config.REQUEST_BUDGET_S if config.is_serverless() else None)


def remaining() -> float | None:
    """Seconds left, or ``None`` when no deadline applies."""
    d = _deadline.get()
    return None if d is None else d - time.monotonic()


def expired() -> bool:
    r = remaining()
    return r is not None and r <= 0


def clamp(want_s: float, *, min_useful_s: float = MIN_USEFUL_S) -> float:
    """``want_s`` limited to what the request budget still allows.

    Raises :class:`DeadlineExceeded` when less than ``min_useful_s`` remains (no point starting
    another model call that cannot finish).
    """
    r = remaining()
    if r is None:
        return want_s
    if r < min_useful_s:
        raise DeadlineExceeded()
    return min(want_s, r)
