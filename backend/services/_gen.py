"""Shared plumbing for the study-tool generators (quiz, flashcards, exam, cheatsheet, mindmap, planner).

* ``parse_json``      — tolerant JSON extraction (fences, <think>, prose around the object, trailing
                        commas, truncated output salvaged by closing open structures).
* ``generate_json``   — one chain call in JSON mode + parse + (at most) one LLM repair + (at most) one
                        re-sample, all under ONE wall-clock budget so a request can never run past it.
* ``generate_text``   — one chain call for markdown/plain output.
* ``TTLCache``        — tiny bounded, thread-safe cache (callers key it by owner + params).
* ``read_upload``     — size-limited, type-checked reading of an uploaded PDF.

Every LLM call goes through ``backend.llm_chain.chat`` (looked up at call time, so tests can patch it).
``LLMChainError`` is NOT caught here: backend.main maps it to a friendly 503.
"""
from __future__ import annotations

import dataclasses
import json
import re
import threading
import time
from collections import OrderedDict
from typing import Any, Callable

from backend import config, deadline, llm_chain, privacy

# Total wall-clock budget for one logical generation (sample + repair + re-sample). Vercel kills a
# function at 60 s, so serverless gets a tighter default; the per-request deadline (backend.deadline)
# further limits it when earlier steps of the same request already used time.
DEFAULT_BUDGET_S: float = 55.0
SERVERLESS_BUDGET_S: float = 40.0


def default_budget() -> float:
    return SERVERLESS_BUDGET_S if config.is_serverless() else DEFAULT_BUDGET_S

# An extra LLM round is only started when at least this much budget remains.
_MIN_RETRY_BUDGET_S: float = 10.0
_MAX_REPAIR_INPUT_CHARS: int = 7000

# Uploads: Vercel caps request bodies at ~4.5 MB; refuse bigger ones early (and everywhere else).
MAX_UPLOAD_BYTES: int = 4 * 1024 * 1024

# Hard caps on user-supplied free text (topic / subject / filename) so prompts stay bounded.
MAX_TOPIC_CHARS: int = 200
MAX_NAME_CHARS: int = 255

_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.S | re.I)
_FENCE_RE = re.compile(r"^\s*```[a-zA-Z0-9_-]*[ \t]*\n?(.*?)\n?\s*```\s*$", re.S)
_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------
def strip_fences(text: str) -> str:
    """Remove <think> blocks and one wrapping ``` fence."""
    text = _THINK_RE.sub("", text or "").strip()
    m = _FENCE_RE.match(text)
    if m:
        return m.group(1).strip()
    if text.startswith("```"):                       # opening fence but the reply was cut off
        text = re.sub(r"^```[a-zA-Z0-9_-]*[ \t]*\n?", "", text).strip()
    return text


def _balanced_slice(s: str, start: int) -> str | None:
    """Return the balanced {...}/[...] slice starting at ``start`` (string/escape aware), else None."""
    stack: list[str] = []
    in_str = esc = False
    for i in range(start, len(s)):
        c = s[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c in "{[":
            stack.append("}" if c == "{" else "]")
        elif c in "}]":
            if not stack or stack[-1] != c:
                return None
            stack.pop()
            if not stack:
                return s[start:i + 1]
    return None


def _closers(s: str) -> str | None:
    """Text that would close every open string/array/object in ``s`` (None if mis-nested)."""
    stack: list[str] = []
    in_str = esc = False
    for c in s:
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c in "{[":
            stack.append("}" if c == "{" else "]")
        elif c in "}]":
            if not stack or stack[-1] != c:
                return None
            stack.pop()
    return ('"' if in_str else "") + "".join(reversed(stack))


def _loads(s: str) -> Any:
    try:
        v = json.loads(s)
    except (ValueError, TypeError):
        try:
            v = json.loads(_TRAILING_COMMA_RE.sub(r"\1", s))
        except (ValueError, TypeError):
            return None
    return v if isinstance(v, (dict, list)) else None


def _salvage_truncated(s: str, max_tries: int = 60) -> Any:
    """Output cut off mid-way: cut back to a closing bracket and close what is still open,
    so an otherwise good quiz/exam keeps its complete items."""
    start = min((i for i in (s.find("{"), s.find("[")) if i >= 0), default=-1)
    if start < 0:
        return None
    s = s[start:]
    tries = 0
    for m in reversed(list(re.finditer(r"[}\]]", s))):
        cut = s[:m.end()]
        close = _closers(cut)
        if close is None:
            continue
        v = _loads(cut + close)
        if v is not None:
            return v
        tries += 1
        if tries >= max_tries:
            break
    return None


def parse_json(text: str) -> Any:
    """Best-effort parse of model output into a dict/list; None if hopeless."""
    s = strip_fences(text)
    if not s:
        return None
    v = _loads(s)
    if v is not None:
        return v
    # JSON embedded in prose: first balanced object/array that parses.
    for m in re.finditer(r"[{\[]", s):
        sl = _balanced_slice(s, m.start())
        if sl is None:             # the outermost value never closes -> truncated; salvage it whole
            break
        v = _loads(sl)
        if v is not None:
            return v
        if m.start() > 2000:       # don't scan megabytes of prose
            break
    return _salvage_truncated(s)


# ---------------------------------------------------------------------------
# LLM calls
# ---------------------------------------------------------------------------
def _merge_skipped(res: "llm_chain.ChatResult", extra: list[str]) -> "llm_chain.ChatResult":
    if not extra:
        return res
    merged = list(res.skipped)
    for e in extra:
        if e not in merged:
            merged.append(e)
    return dataclasses.replace(res, skipped=merged)


def generate_json(system: str, prompt: str, *, shape: Callable[[Any], Any], max_tokens: int,
                  error_cls: type[Exception], what: str, effort: str = "low",
                  temperature: float | None = 0.4, budget_s: float | None = None,
                  max_samples: int = 2, sensitive: bool = False) -> tuple[Any, "llm_chain.ChatResult"]:
    """Run the chain in JSON mode and return ``(shape(parsed), ChatResult)``.

    ``shape`` turns the parsed JSON into the final usable value, or returns a falsy value when
    nothing usable came out (then we re-sample once, budget permitting). A reply that does not
    parse gets one cheap local repair and then one LLM "fix this JSON" call. When every attempt is
    unusable, ``error_cls(...)`` is raised with a friendly message (routers map it to 502).
    ``LLMChainError`` (all providers down, or the request's overall deadline spent) propagates
    untouched. ``sensitive=True`` keeps the prompt away from Gemini (user-uploaded content).
    """
    t0 = time.monotonic()
    sensitive = sensitive or privacy.current()
    if budget_s is None:
        budget_s = default_budget()

    def remaining() -> float:
        return budget_s - (time.monotonic() - t0)

    skipped: list[str] = []
    last: llm_chain.ChatResult | None = None
    for sample in range(max(1, max_samples)):
        rem = remaining()
        if sample and rem < _MIN_RETRY_BUDGET_S:
            break
        res = llm_chain.chat(system, [{"role": "user", "content": prompt}], max_tokens=max_tokens,
                             temperature=temperature, json_mode=True, effort=effort, sensitive=sensitive,
                             deadline_s=deadline.clamp(max(5.0, min(llm_chain.CHAIN_DEADLINE_S, rem))))
        skipped.extend(res.skipped)
        last = res
        data = parse_json(res.text)
        if data is None and remaining() >= _MIN_RETRY_BUDGET_S:
            fix = llm_chain.chat(
                "You repair malformed JSON. Return ONLY the corrected, complete JSON value — "
                "same content, valid syntax, no commentary.",
                [{"role": "user", "content": res.text[:_MAX_REPAIR_INPUT_CHARS]}],
                max_tokens=max_tokens, temperature=0.0, json_mode=True, effort="low", sensitive=sensitive,
                deadline_s=deadline.clamp(max(5.0, min(25.0, remaining()))))
            skipped.extend(fix.skipped)
            data = parse_json(fix.text)
            if data is not None:
                res = fix
                last = fix
        if data is None:
            continue
        shaped = shape(data)
        if shaped:
            return shaped, _merge_skipped(res, skipped)
    raise error_cls(f"The model's {what} couldn't be read — please try again.")


def generate_text(system: str, prompt: str, *, max_tokens: int, error_cls: type[Exception],
                  what: str, effort: str = "low", temperature: float | None = 0.3,
                  deadline_s: float | None = None, sensitive: bool = False) -> tuple[str, "llm_chain.ChatResult"]:
    """One chain call returning non-empty plain/markdown text (outer ``` fence stripped)."""
    sensitive = sensitive or privacy.current()
    want = (SERVERLESS_BUDGET_S + 5.0 if config.is_serverless() else 50.0) if deadline_s is None else deadline_s
    res = llm_chain.chat(system, [{"role": "user", "content": prompt}], max_tokens=max_tokens,
                         temperature=temperature, json_mode=False, effort=effort, sensitive=sensitive,
                         deadline_s=deadline.clamp(want))
    text = strip_fences(res.text) if res.text.lstrip().startswith("```") else _THINK_RE.sub("", res.text).strip()
    if not text:
        raise error_cls(f"The model returned an empty {what}; try again.")
    return text, res


def public(res: "llm_chain.ChatResult | None") -> dict | None:
    return res.to_public() if res is not None else None


# ---------------------------------------------------------------------------
# Bounded TTL cache
# ---------------------------------------------------------------------------
class TTLCache:
    """Thread-safe LRU + TTL cache. Keys MUST include the owner for per-user data."""

    def __init__(self, maxsize: int = 128, ttl_s: float = 1800.0) -> None:
        self._d: OrderedDict[Any, tuple[float, Any]] = OrderedDict()
        self._lock = threading.Lock()
        self.maxsize, self.ttl_s = maxsize, ttl_s

    def get(self, key: Any) -> Any:
        with self._lock:
            ent = self._d.get(key)
            if ent is None:
                return None
            if time.monotonic() >= ent[0]:
                del self._d[key]
                return None
            self._d.move_to_end(key)
            return ent[1]

    def set(self, key: Any, value: Any) -> None:
        with self._lock:
            self._d[key] = (time.monotonic() + self.ttl_s, value)
            self._d.move_to_end(key)
            while len(self._d) > self.maxsize:
                self._d.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._d.clear()

    def __len__(self) -> int:
        return len(self._d)


# ---------------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------------
def clean_topic(topic: str | None) -> str:
    """Whitespace-collapsed, length-capped topic ('' if blank)."""
    return re.sub(r"\s+", " ", topic or "").strip()[:MAX_TOPIC_CHARS]


def dedupe_names(names: list[str], limit: int) -> list[str]:
    """Unique, non-blank, length-capped names in order; at most ``limit``."""
    out: list[str] = []
    for n in names or []:
        if isinstance(n, str):
            n = n.strip()[:MAX_NAME_CHARS]
            if n and n not in out:
                out.append(n)
    return out[:limit]
