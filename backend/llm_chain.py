"""The 5-key LLM provider chain — the ONLY place the app talks to an LLM provider.

Strict priority (steps top -> bottom, models left -> right, first non-empty reply wins):

    1  Anthropic      ANTHROPIC_API_KEY   claude-sonnet-5-5
    2  Groq key 1     GROQ_API_KEY        openai/gpt-oss-120b -> openai/gpt-oss-20b -> qwen/qwen3.8-27b
    3  Groq key 2     GROQ_API_KEY_2      (same three)
    4  Gemini key 1   GEMINI_API_KEY      gemini-3.8-flash -> gemini-3.6-flash -> gemini-3.1-flash-lite
    5  Gemini key 2   GEMINI_API_KEY_2    (same three)

Design (kept deliberately light for serverless cold start: stdlib + ``httpx`` only; no
langchain / groq SDK / anthropic SDK):

* Per-call timeout 25 s, whole-chain deadline 50 s, **no hidden retries** — the chain itself is
  the retry mechanism. The single exception is a per-minute 429 whose ``retry-after`` is
  <= 2 s and fits the deadline: we wait and re-try that (step, model) once.
* 401/402/403 (and Gemini's "API key not valid" 400) => the key is remembered dead for a while and
  the rest of that step is skipped. 429/5xx/529/timeout/empty/refusal/400 => next model, then
  next key.
* Daily-quota 429s are remembered per (step, model) with a TTL so later requests skip them
  instantly; a per-(step, model) circuit breaker (3 consecutive failures -> 60 s) stops a dead
  provider from costing a timeout on every request.
* ``sensitive=True`` never reaches Gemini (its free tier may train on prompts).
* Shared pooled ``httpx`` clients (keep-alive reuse across calls); all state is thread-safe.
* Keys are read from the environment at call time and are never logged, labelled or put in
  errors; prompts are never logged.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Iterator

import httpx

from backend import cache as _cache
from backend import spend as _spend

log = logging.getLogger("backend.llm_chain")

# ---------------------------------------------------------------------------
# Static chain definition
# ---------------------------------------------------------------------------
ANTHROPIC_MODELS: tuple[str, ...] = ("claude-sonnet-5-5",)
GROQ_MODELS: tuple[str, ...] = ("openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b")
GEMINI_MODELS: tuple[str, ...] = ("gemini-3.8-flash", "gemini-3.6-flash", "gemini-3.1-flash-lite")

# The ONLY chat model ids the app may ever send to a provider.
ALLOWED_MODELS: frozenset[str] = frozenset(ANTHROPIC_MODELS + GROQ_MODELS + GEMINI_MODELS)

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
ANTHROPIC_VERSION = "2023-06-01"


def _test_base() -> str:
    """TEST-ONLY: when ``LLM_CHAIN_TEST_BASE_URL`` is set (e.g. ``http://127.0.0.1:8890`` pointing
    at ``scripts/fake_providers.py``) all three providers are reached at that host with their real
    path layouts. Always ignored on Vercel, so a stray variable can never redirect production
    traffic (or keys) to another host."""
    if os.environ.get("VERCEL"):
        return ""
    return os.environ.get("LLM_CHAIN_TEST_BASE_URL", "").strip().rstrip("/")


def _anthropic_url() -> str:
    base = _test_base()
    return f"{base}/v1/messages" if base else ANTHROPIC_URL


def _groq_url() -> str:
    base = _test_base()
    return f"{base}/openai/v1/chat/completions" if base else GROQ_URL


def _gemini_base() -> str:
    base = _test_base()
    return f"{base}/v1beta/models" if base else GEMINI_BASE

CALL_TIMEOUT_S = 25.0          # per provider call
CHAIN_DEADLINE_S = 50.0        # whole chain
CONNECT_TIMEOUT_S = 8.0
RETRY_AFTER_MAX_S = 2.0        # only wait for a per-minute 429 this short
QUOTA_TTL_S = 600.0            # daily-quota memory (self-heals)
DEAD_KEY_TTL_S = 600.0         # 401/403 memory
CIRCUIT_THRESHOLD = 3          # consecutive failures
CIRCUIT_OPEN_S = 60.0
MIN_REMAINING_S = 0.75         # don't start a call with less budget than this
MIN_OUTPUT_TOKENS = 2048       # reasoning models eat the output budget
MAX_IMAGE_BYTES = 5 * 1024 * 1024   # per image sanity cap (ValueError above)
GROQ_VISION_MODEL = "qwen/qwen3.8-27b"
GROQ_MAX_IMAGES = 3
NO_RESULT_LABEL = "Model: none (all providers failed)"


@dataclass(frozen=True)
class _Step:
    n: int
    provider: str          # anthropic | groq | gemini
    env: str
    name: str              # "Anthropic" | "Groq key 1" ...
    short: str             # "anthropic" | "groq#1" ... (used in the skipped list)
    models: tuple[str, ...]


STEPS: tuple[_Step, ...] = (
    _Step(1, "anthropic", "ANTHROPIC_API_KEY", "Anthropic", "anthropic", ANTHROPIC_MODELS),
    _Step(2, "groq", "GROQ_API_KEY", "Groq key 1", "groq#1", GROQ_MODELS),
    _Step(3, "groq", "GROQ_API_KEY_2", "Groq key 2", "groq#2", GROQ_MODELS),
    _Step(4, "gemini", "GEMINI_API_KEY", "Gemini key 1", "gemini#1", GEMINI_MODELS),
    _Step(5, "gemini", "GEMINI_API_KEY_2", "Gemini key 2", "gemini#2", GEMINI_MODELS),
)
KEY_ENV_VARS: tuple[str, ...] = tuple(s.env for s in STEPS)
assert all(m in ALLOWED_MODELS for s in STEPS for m in s.models)

# Gemini models that reject sampling params (temperature/topP/topK).
_NO_SAMPLING_PREFIXES = ("gemini-3.6", "gemini-3.8")


def _key(step: _Step) -> str | None:
    v = os.environ.get(step.env)
    if v is None:
        return None
    v = v.strip().strip('"').strip("'").strip()
    return v or None


def has_any_key() -> bool:
    return any(_key(s) for s in STEPS)


def configured_steps() -> list[int]:
    return [s.n for s in STEPS if _key(s)]


def _short_model(model: str) -> str:
    return model.split("/")[-1]


# ---------------------------------------------------------------------------
# Public result / error types
# ---------------------------------------------------------------------------
@dataclass
class ChatResult:
    text: str
    provider: str
    model: str
    step: int
    step_name: str
    usage: dict = field(default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
    skipped: list = field(default_factory=list)
    latency_ms: int = 0
    trimmed: int = 0          # estimated input tokens trimmed for the answering provider (0 = full input sent)
    finish: str = ""          # normalised stop reason of a streamed answer: "stop" | "length" | "interrupted" | "" (unknown)

    @property
    def label(self) -> str:
        return f"Model: {self.provider} · {self.model} · step {self.step} ({self.step_name})"

    def to_public(self) -> dict:
        return {
            "provider": self.provider,
            "model": self.model,
            "step": self.step,
            "step_name": self.step_name,
            "label": self.label,
            "skipped": list(self.skipped),
        }


@dataclass
class StreamStart:
    """Yielded by ``stream_chat`` / ``astream_chat`` once a provider has accepted the stream (before
    its first text delta), so a UI can reveal WHICH model is answering immediately. Several may be
    yielded when a provider fails before producing text (the last one wins). Never the final item."""
    provider: str
    model: str
    step: int
    step_name: str
    skipped: list = field(default_factory=list)

    @property
    def label(self) -> str:
        return f"Model: {self.provider} · {self.model} · step {self.step} ({self.step_name})"

    def to_public(self) -> dict:
        return {"provider": self.provider, "model": self.model, "step": self.step,
                "step_name": self.step_name, "label": self.label, "skipped": list(self.skipped)}


class LLMChainError(RuntimeError):
    """All providers failed (or no usable key is configured)."""

    label = NO_RESULT_LABEL

    def __init__(self, message: str, *, status: int | None = None, attempts: list[str] | None = None):
        super().__init__(message)
        self.status = status
        self.attempts = list(attempts or [])

    def to_public(self) -> dict:
        return {"label": self.label, "skipped": list(self.attempts)}


class LLMRequestTooLarge(LLMChainError):
    """The prompt is too big for the paid model's per-request cap and no other model may take it."""

    label = "Model: none (request too large)"
    http_status = 413
    PUBLIC_DETAIL = ("This request is too large for the available models; use fewer or smaller files "
                     "and try again.")


# ---------------------------------------------------------------------------
# Time helpers (patchable in tests)
# ---------------------------------------------------------------------------
def _now() -> float:
    return time.monotonic()


def _sleep(s: float) -> None:
    time.sleep(s)


async def _asleep(s: float) -> None:
    await asyncio.sleep(s)


# ---------------------------------------------------------------------------
# Shared mutable state (thread-safe): quota memory, dead keys, circuit breaker
# ---------------------------------------------------------------------------
class _State:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._exhausted: dict[tuple[int, str], float] = {}
        self._dead: dict[int, tuple[float, int]] = {}      # step -> (until, key fingerprint)
        self._fails: dict[tuple[int, str], int] = {}
        self._open_until: dict[tuple[int, str], float] = {}
        self.breaker_opened = 0      # lifetime counters (observability / load tests)
        self.breaker_closed = 0

    def reset(self) -> None:
        with self._lock:
            self._exhausted.clear()
            self._dead.clear()
            self._fails.clear()
            self._open_until.clear()
            self.breaker_opened = self.breaker_closed = 0
        _cache.shared_map("llm-exhausted").clear_local()
        _cache.shared_map("llm-dead").clear_local()

    # -- quota
    def mark_exhausted(self, n: int, model: str, ttl: float) -> None:
        with self._lock:
            self._exhausted[(n, model)] = _now() + ttl
        _cache.shared_map("llm-exhausted").put(f"{n}:{model}", ttl)       # no-op unless a shared cache is configured

    def is_exhausted(self, n: int, model: str) -> bool:
        with self._lock:
            until = self._exhausted.get((n, model))
            if until is not None:
                if _now() >= until:
                    del self._exhausted[(n, model)]
                else:
                    return True
        # Not known locally: another instance may have hit the quota (non-blocking snapshot read).
        return _cache.enabled() and _cache.shared_map("llm-exhausted").get(f"{n}:{model}") is not None

    @staticmethod
    def _stable_fp(key: str) -> str:
        import hashlib

        return hashlib.sha256(key.encode("utf-8", "ignore")).hexdigest()[:12]      # hash() differs per process

    def share_dead(self, n: int, key: str, ttl: float = DEAD_KEY_TTL_S) -> None:
        _cache.shared_map("llm-dead").put(str(n), ttl, self._stable_fp(key))

    def is_dead_shared(self, n: int, key: str) -> bool:
        return _cache.enabled() and _cache.shared_map("llm-dead").get(str(n)) == self._stable_fp(key)

    # -- dead keys
    def mark_dead(self, n: int, fingerprint: int, ttl: float = DEAD_KEY_TTL_S) -> None:
        with self._lock:
            self._dead[n] = (_now() + ttl, fingerprint)

    def is_dead(self, n: int, fingerprint: int) -> bool:
        with self._lock:
            ent = self._dead.get(n)
            if ent is None:
                return False
            until, fp = ent
            if _now() >= until or fp != fingerprint:    # expired, or the key was replaced
                del self._dead[n]
                return False
            return True

    # -- circuit breaker
    def on_failure(self, n: int, model: str) -> None:
        with self._lock:
            k = (n, model)
            c = self._fails.get(k, 0) + 1
            self._fails[k] = c
            if c >= CIRCUIT_THRESHOLD:
                if k not in self._open_until:
                    self.breaker_opened += 1
                self._open_until[k] = _now() + CIRCUIT_OPEN_S

    def on_success(self, n: int, model: str) -> None:
        with self._lock:
            tripped = self._fails.pop((n, model), 0) >= CIRCUIT_THRESHOLD   # a half-open probe that worked
            self._open_until.pop((n, model), None)
            if tripped:
                self.breaker_closed += 1

    def is_open(self, n: int, model: str) -> bool:
        with self._lock:
            until = self._open_until.get((n, model))
            if until is None:
                return False
            if _now() >= until:
                # half-open: allow one probe; the failure counter stays so a failed probe re-opens
                del self._open_until[(n, model)]
                return False
            return True

    def snapshot(self) -> tuple[dict, dict, dict]:
        with self._lock:
            now = _now()
            ex = {k: max(0.0, v - now) for k, v in self._exhausted.items() if v > now}
            dead = {k: max(0.0, v[0] - now) for k, v in self._dead.items() if v[0] > now}
            op = {k: max(0.0, v - now) for k, v in self._open_until.items() if v > now}
            return ex, dead, op


_state = _State()


def reset_state() -> None:
    """Forget quota / dead-key / circuit memory (tests, admin)."""
    _state.reset()


# ---------------------------------------------------------------------------
# Shared pooled HTTP clients
# ---------------------------------------------------------------------------
_LIMITS = httpx.Limits(max_connections=64, max_keepalive_connections=24, keepalive_expiry=45.0)
_clients_lock = threading.Lock()
_sync_client: httpx.Client | None = None
_async_client: httpx.AsyncClient | None = None
_async_loop: Any = None
_sync_transport: Any = None     # test hook
_async_transport: Any = None    # test hook


def _configure_transports(sync: Any = None, async_: Any = None) -> None:
    """Test hook: route all calls through ``httpx.MockTransport`` (or any transport)."""
    global _sync_transport, _async_transport, _sync_client, _async_client, _async_loop
    with _clients_lock:
        _sync_transport, _async_transport = sync, async_ if async_ is not None else sync
        if _sync_client is not None:
            try:
                _sync_client.close()
            except Exception:  # noqa: BLE001
                pass
        _sync_client = _async_client = None
        _async_loop = None


def _get_sync_client() -> httpx.Client:
    global _sync_client
    c = _sync_client
    if c is not None and not c.is_closed:
        return c
    with _clients_lock:
        if _sync_client is None or _sync_client.is_closed:
            _sync_client = httpx.Client(limits=_LIMITS, transport=_sync_transport,
                                        timeout=httpx.Timeout(CALL_TIMEOUT_S, connect=CONNECT_TIMEOUT_S),
                                        follow_redirects=False)
        return _sync_client


def _get_async_client() -> httpx.AsyncClient:
    """One AsyncClient per running event loop (an AsyncClient is bound to its loop)."""
    global _async_client, _async_loop
    loop = asyncio.get_running_loop()
    c = _async_client
    if c is not None and _async_loop is loop and not c.is_closed:
        return c
    with _clients_lock:
        if _async_client is None or _async_loop is not loop or _async_client.is_closed:
            _async_client = httpx.AsyncClient(limits=_LIMITS, transport=_async_transport,
                                              timeout=httpx.Timeout(CALL_TIMEOUT_S, connect=CONNECT_TIMEOUT_S),
                                              follow_redirects=False)
            _async_loop = loop
        return _async_client


def warm() -> None:
    """Create the pooled sync client now (no network I/O: httpx connects lazily). Cheap."""
    _get_sync_client()


def close_clients() -> None:
    """Close pooled sync client (async one is dropped; it belongs to its loop)."""
    _configure_transports(_sync_transport, _async_transport)


# ---------------------------------------------------------------------------
# Message / request construction
# ---------------------------------------------------------------------------
@dataclass
class _Req:
    system: str
    messages: list          # normalized [{"role": "user"|"assistant", "content": str}]
    max_tokens: int
    temperature: float | None
    json_mode: bool
    effort: str
    images: list = field(default_factory=list)    # [(base64 str, mime)] — never logged


# ---------------------------------------------------------------------------
# Per-provider INPUT fitting and OUTPUT allowance (done here, never in callers)
# ---------------------------------------------------------------------------
# Callers pass the FULL content. Groq's free tier (8,000 TPM) needs the prompt shrunk to fit, so
# Groq attempts get a trimmed COPY (derived from the original request each time; the shared
# request is never modified). Anthropic/Gemini get no product-imposed limit: the *_MAX_INPUT_TOKENS
# values below are provider HARD limits used only as a backstop to avoid a 400 (0 disables).
CHARS_PER_TOKEN = 3.6
TRIM_MARKER = "\n\n[...trimmed...]\n\n"


def _env_int(name: str, default: int) -> int:
    try:
        return max(0, int(float(os.environ.get(name, "") or default)))
    except (TypeError, ValueError):
        return default


def input_budget(provider: str) -> int:
    """Max estimated input tokens for a provider (0 = unlimited)."""
    if provider == "groq":
        return _env_int("GROQ_MAX_INPUT_TOKENS", 5500)
    if provider == "anthropic":
        return _env_int("ANTHROPIC_MAX_INPUT_TOKENS", 190000)     # Sonnet context window (hard limit)
    return _env_int("GEMINI_MAX_INPUT_TOKENS", 900000)            # Gemini context window (hard limit)


# The Anthropic Messages API REQUIRES max_tokens, so "no output cap" means: always send the
# model's maximum output allowance and ignore the caller's value for the Anthropic step.
def _max_request_tokens() -> int:
    """Per-request cap (estimated input tokens) for the PAID step; 0 = unlimited."""
    from backend import config
    return config.anthropic_max_request_input_tokens()


def anthropic_max_output() -> int:
    return _env_int("ANTHROPIC_MAX_OUTPUT_TOKENS", 64000) or 64000


def _est(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN) + 1


def _middle_out(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    keep = max(0, max_chars - len(TRIM_MARKER))
    head = int(keep * 0.6)
    tail = keep - head
    return text[:head] + TRIM_MARKER + (text[len(text) - tail:] if tail else "")


def _fit(system: str, msgs: list[dict], budget: int) -> tuple[str, list[dict], int]:
    """Return (system, messages, tokens_trimmed) fitting ``budget`` estimated tokens, shrinking the
    LARGEST message first (middle-out, head+tail kept); the system prompt only as a last resort.
    Inputs are never mutated."""
    msgs = [dict(m) for m in msgs]
    total = _est(system) + sum(_est(m["content"]) for m in msgs)
    if not budget or total <= budget:
        return system, msgs, 0
    start = total

    def shrink(blocks: list, floor: int) -> None:
        nonlocal total
        for _ in range(64):
            excess = total - budget
            if excess <= 0:
                return
            blocks.sort(key=lambda b: -len(b[0]()))
            big = blocks[0]
            cur = big[0]()
            second = len(blocks[1][0]()) if len(blocks) > 1 else 0
            target = max(floor, second, len(cur) - int(excess * CHARS_PER_TOKEN) - 8)
            if target >= len(cur):
                return
            new = _middle_out(cur, target)
            if len(new) >= len(cur):
                return
            total -= _est(cur) - _est(new)
            big[1](new)

    def mk(i):
        return (lambda: msgs[i]["content"], lambda v: msgs[i].__setitem__("content", v))
    shrink([mk(i) for i in range(len(msgs))], 600)
    if total > budget and system:
        box = [system]
        shrink([(lambda: box[0], lambda v: box.__setitem__(0, v))], 2000)
        system = box[0]
    return system, msgs, max(0, start - total)


_JSON_HINT = "Respond ONLY with a single valid JSON value. No prose, no markdown fences."


def _normalize(system: str | None, messages: list[dict]) -> tuple[str, list[dict]]:
    sys_parts = [system.strip()] if system and system.strip() else []
    out: list[dict] = []
    for m in messages or []:
        role = (m.get("role") or "user").lower()
        content = m.get("content")
        content = "" if content is None else (content if isinstance(content, str) else str(content))
        if not content.strip():
            continue
        if role == "system":
            sys_parts.append(content.strip())
            continue
        role = "assistant" if role in ("assistant", "model", "ai") else "user"
        if out and out[-1]["role"] == role:
            out[-1]["content"] += "\n\n" + content
        else:
            out.append({"role": role, "content": content})
    if not out or all(m["role"] != "user" for m in out):
        raise ValueError("messages must contain at least one non-empty user message")
    return "\n\n".join(sys_parts), out


def _encode_images(images) -> list:
    out = []
    for item in images or []:
        try:
            data, mime = item
        except (TypeError, ValueError):
            raise ValueError("images must be a list of (bytes, mime) tuples") from None
        if not isinstance(data, (bytes, bytearray, memoryview)) or not data:
            raise ValueError("image data must be non-empty bytes")
        if len(data) > MAX_IMAGE_BYTES:
            raise ValueError(f"image too large ({len(data)} bytes > {MAX_IMAGE_BYTES})")
        mime = str(mime or "").lower().strip()
        if not mime.startswith("image/"):
            raise ValueError(f"unsupported image mime type: {mime!r}")
        out.append((base64.b64encode(bytes(data)).decode("ascii"), mime))
    return out


def _make_req(system, messages, max_tokens, temperature, json_mode, effort, images=None) -> _Req:
    s, msgs = _normalize(system, messages)
    return _Req(s, msgs, int(max_tokens), temperature, bool(json_mode), (effort or "low").lower(),
                _encode_images(images))


def _with_json_hint(system: str, msgs: list[dict]) -> str:
    blob = (system + " " + " ".join(m["content"] for m in msgs)).lower()
    if "json" in blob:
        return system
    return (system + "\n\n" if system else "") + _JSON_HINT


def _groq_effort(effort: str) -> str:
    return {"minimal": "low", "low": "low", "medium": "medium", "high": "high",
            "xhigh": "high", "max": "high"}.get(effort, "low")


def _gemini_level(model: str, effort: str) -> str:
    # "minimal" is rejected by gemini-3.8-* (400) and not worth the risk elsewhere: never send it.
    return {"minimal": "low", "low": "low", "medium": "medium", "high": "high",
            "xhigh": "high", "max": "high"}.get(effort, "low")


def _anthropic_effort(effort: str) -> str:
    return effort if effort in ("low", "medium", "high", "xhigh", "max") else "low"


def _build(step: _Step, model: str, req: _Req, key: str, stream: bool, run: Any = None):
    """Return (url, headers, json_body) for one provider call. Keys only appear in headers."""
    system, msgs, trimmed = _fit(req.system, req.messages, input_budget(step.provider))
    if trimmed and run is not None:
        run.trimmed += trimmed
    if req.json_mode:
        system = _with_json_hint(system, msgs)

    if step.provider == "anthropic":
        # No prefill: drop any trailing assistant turns. Never send temperature/top_p/top_k.
        am = list(msgs)
        while am and am[-1]["role"] == "assistant":
            am.pop()
        if req.images and am:
            blocks = [{"type": "image", "source": {"type": "base64", "media_type": mt, "data": b64}}
                      for b64, mt in req.images]
            blocks.append({"type": "text", "text": am[-1]["content"]})
            am[-1] = {"role": "user", "content": blocks}
        body: dict[str, Any] = {
            "model": model,
            "max_tokens": anthropic_max_output(),    # caller value ignored (see anthropic_max_output)
            "messages": am,
            "output_config": {"effort": _anthropic_effort(req.effort)},
        }
        if system:
            body["system"] = system
        if stream:
            body["stream"] = True
        headers = {"x-api-key": key, "anthropic-version": ANTHROPIC_VERSION,
                   "content-type": "application/json"}
        return _anthropic_url(), headers, body

    if step.provider == "groq":
        gm = ([{"role": "system", "content": system}] if system else []) + [dict(m) for m in msgs]
        if req.images:
            li = max(i for i, m in enumerate(gm) if m["role"] == "user")
            gm[li] = {"role": "user", "content": [{"type": "text", "text": gm[li]["content"]}] + [
                {"type": "image_url", "image_url": {"url": f"data:{mt};base64,{b64}"}}
                for b64, mt in req.images[:GROQ_MAX_IMAGES]]}
        body = {
            "model": model,
            "messages": gm,
            "max_completion_tokens": min(max(req.max_tokens, MIN_OUTPUT_TOKENS),
                                         16384 if model.startswith("qwen/") else 65536),
        }
        if req.temperature is not None:
            body["temperature"] = req.temperature
        if model.startswith("openai/gpt-oss"):
            body["reasoning_effort"] = _groq_effort(req.effort)
        if req.json_mode and not stream:
            body["response_format"] = {"type": "json_object"}
        if stream:
            body["stream"] = True
        headers = {"Authorization": f"Bearer {key}", "content-type": "application/json"}
        return _groq_url(), headers, body

    # gemini
    contents = [{"role": "model" if m["role"] == "assistant" else "user",
                 "parts": [{"text": m["content"]}]} for m in msgs]
    if req.images:
        li = max(i for i, c in enumerate(contents) if c["role"] == "user")
        contents[li]["parts"] += [{"inlineData": {"mimeType": mt, "data": b64}} for b64, mt in req.images]
    gen: dict[str, Any] = {
        "maxOutputTokens": min(max(req.max_tokens, MIN_OUTPUT_TOKENS), 65536),
        "thinkingConfig": {"thinkingLevel": _gemini_level(model, req.effort)},
    }
    if req.temperature is not None and not model.startswith(_NO_SAMPLING_PREFIXES):
        gen["temperature"] = req.temperature
    if req.json_mode:
        gen["responseMimeType"] = "application/json"
    body = {"contents": contents, "generationConfig": gen}
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    gbase = _gemini_base()
    url = f"{gbase}/{model}:streamGenerateContent?alt=sse" if stream \
        else f"{gbase}/{model}:generateContent"
    headers = {"x-goog-api-key": key, "content-type": "application/json"}
    return url, headers, body


# ---------------------------------------------------------------------------
# Response parsing / classification
# ---------------------------------------------------------------------------
@dataclass
class _Outcome:
    ok: bool = False
    text: str = ""
    usage: dict = field(default_factory=dict)
    status: int | None = None
    token: str = ""             # short code for the skipped list: "429", "timeout", "empty", ...
    message: str = ""           # sanitized, <=200 chars
    retry_after: float | None = None
    daily: bool = False         # daily-quota exhaustion
    dead_key: bool = False      # 401/402/403 etc.
    countable: bool = True      # counts toward the circuit breaker


_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.S | re.I)
_FENCE_RE = re.compile(r"^\s*```[a-zA-Z0-9_-]*\s*\n?(.*?)\n?\s*```\s*$", re.S)
_SECRET_RE = re.compile(r"(AIza[0-9A-Za-z_\-]{16,}|gsk_[A-Za-z0-9]{10,}|sk-ant-[A-Za-z0-9_\-]{10,}|"
                        r"sk-[A-Za-z0-9_\-]{20,}|Bearer\s+\S+)")


def _sanitize(msg: str, limit: int = 200) -> str:
    msg = msg or ""
    for st in STEPS:                      # redact the literal configured keys, whatever their format
        k = _key(st)
        if k and len(k) >= 8:
            msg = msg.replace(k, "***")
    msg = _SECRET_RE.sub("***", msg)
    msg = re.sub(r"\s+", " ", msg).strip()
    return msg[:limit]


def _clean_text(text: str, json_mode: bool) -> str:
    text = _THINK_RE.sub("", text or "").strip()
    if json_mode:
        m = _FENCE_RE.match(text)
        if m:
            text = m.group(1).strip()
    return text


def _usage(prompt: Any, completion: Any, total: Any = None) -> dict:
    def _i(x: Any) -> int:
        try:
            return int(x or 0)
        except (TypeError, ValueError):
            return 0
    p, c = _i(prompt), _i(completion)
    t = _i(total) or (p + c)
    return {"prompt_tokens": p, "completion_tokens": c, "total_tokens": t}


def _parse_retry_after(headers: Any, body_text: str) -> float | None:
    try:
        v = headers.get("retry-after") if headers is not None else None
        if v is not None:
            return max(0.0, float(v))
    except (TypeError, ValueError):
        pass
    m = re.search(r'"retryDelay"\s*:\s*"([0-9.]+)s"', body_text or "")
    if m:
        return float(m.group(1))
    m = re.search(r"try again in\s+([0-9hmsµ. ]+)", body_text or "", re.I)
    if m:
        total, found = 0.0, False
        for num, unit in re.findall(r"(\d+(?:\.\d+)?)\s*(ms|h|m|s)", m.group(1), re.I):
            found = True
            total += float(num) * {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}[unit.lower()]
        if found:
            return total
    return None


def _error_message(body_text: str) -> str:
    try:
        j = json.loads(body_text)
        if isinstance(j, dict):
            err = j.get("error", j)
            if isinstance(err, dict):
                return str(err.get("message") or err.get("status") or err.get("type") or err)
            return str(err)
    except (ValueError, TypeError):
        pass
    return body_text or ""


def _is_daily(provider: str, low: str) -> bool:
    if provider == "anthropic":
        return ("credit balance" in low or "usage limit" in low or "usage limits" in low
                or "spend limit" in low)
    if provider == "groq":
        return ("tokens per day" in low or "requests per day" in low or "(tpd)" in low
                or "(rpd)" in low or "per day" in low)
    # gemini
    if "perday" in low or "per day" in low or "daily" in low:
        return True
    if "perminute" in low or "per minute" in low or '"retrydelay"' in low:
        return False
    return "quota_exceeded" in low or "exceeded your current quota" in low


def _classify_http(provider: str, status: int, headers: Any, body_text: str) -> _Outcome:
    msg = _sanitize(_error_message(body_text))
    low = (body_text or "").lower()
    out = _Outcome(status=status, token=str(status), message=msg,
                   retry_after=_parse_retry_after(headers, body_text))
    if status in (401, 402, 403):
        out.dead_key, out.countable = True, False
    elif status == 400 and provider == "gemini" and ("api_key_invalid" in low or "api key not valid" in low
                                                      or "api key expired" in low):
        out.dead_key, out.countable = True, False
    elif status == 429:
        out.daily = _is_daily(provider, low)
        out.countable = not out.daily
    elif status == 400 and provider == "anthropic" and _is_daily(provider, low):
        out.daily, out.countable = True, False
    elif status == 400:
        out.countable = False         # request-specific, not a provider-health signal
    return out


def _fail(token: str, message: str = "", *, status: int | None = None, countable: bool = True) -> _Outcome:
    return _Outcome(status=status, token=token, message=_sanitize(message), countable=countable)


def _exc_outcome(exc: Exception) -> _Outcome:
    if isinstance(exc, httpx.TimeoutException):
        return _fail("timeout", f"{type(exc).__name__}")
    if isinstance(exc, httpx.HTTPError):
        return _fail("neterr", f"{type(exc).__name__}: {exc}")
    return _fail("error", f"{type(exc).__name__}: {exc}")


def _parse_ok(provider: str, data: Any, json_mode: bool) -> _Outcome:
    """Parse a 200 JSON body into an _Outcome (ok or empty/refusal failure)."""
    if not isinstance(data, dict):
        return _fail("badjson", "non-object response")
    if provider == "anthropic":
        text = "".join(b.get("text", "") for b in (data.get("content") or [])
                       if isinstance(b, dict) and b.get("type") == "text")
        u = data.get("usage") or {}
        usage = _usage(u.get("input_tokens"), u.get("output_tokens"))
        if data.get("stop_reason") == "refusal":
            return _fail("refusal", "model refused", countable=False)
    elif provider == "groq":
        choices = data.get("choices") or []
        ch = choices[0] if choices else {}
        msg = ch.get("message") or {}
        text = msg.get("content") or ""
        u = data.get("usage") or {}
        usage = _usage(u.get("prompt_tokens"), u.get("completion_tokens"), u.get("total_tokens"))
        fr = ch.get("finish_reason")
        if not str(text).strip():
            if msg.get("refusal") or fr == "content_filter":
                return _fail("refusal", "model refused", countable=False)
            return _fail("empty", f"empty reply (finish_reason={fr})", countable=False)
    else:  # gemini
        fb = data.get("promptFeedback") or {}
        cands = data.get("candidates") or []
        if fb.get("blockReason") and not cands:
            return _fail("refusal", f"blocked: {fb.get('blockReason')}", countable=False)
        cand = cands[0] if cands else {}
        parts = ((cand.get("content") or {}).get("parts")) or []
        text = "".join(p.get("text", "") for p in parts if isinstance(p, dict) and not p.get("thought"))
        um = data.get("usageMetadata") or {}
        usage = _usage(um.get("promptTokenCount"),
                       (um.get("candidatesTokenCount") or 0) + (um.get("thoughtsTokenCount") or 0),
                       um.get("totalTokenCount"))
        fr = cand.get("finishReason")
        if not text.strip() and fr in ("SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "RECITATION",
                                       "IMAGE_SAFETY"):
            return _fail("refusal", f"blocked: {fr}", countable=False)
    text = _clean_text(text, json_mode)
    if not text:
        return _fail("empty", "empty reply", countable=False)
    return _Outcome(ok=True, text=text, usage=usage, status=200)


def _parse_response(provider: str, resp: httpx.Response, json_mode: bool) -> _Outcome:
    if resp.status_code == 200:
        try:
            data = resp.json()
        except ValueError:
            return _fail("badjson", "invalid JSON in 200 response")
        return _parse_ok(provider, data, json_mode)
    try:
        body = resp.text
    except Exception:  # noqa: BLE001
        body = ""
    return _classify_http(provider, resp.status_code, resp.headers, body)


# ---------------------------------------------------------------------------
# Streaming SSE parser (one per attempt)
# ---------------------------------------------------------------------------
class _SSEParser:
    def __init__(self, provider: str) -> None:
        self.provider = provider
        self.text_parts: list[str] = []
        self.usage = _usage(0, 0)
        self.finish: str | None = None
        self.refused = False
        self.error: _Outcome | None = None
        self.done = False

    @property
    def text(self) -> str:
        return "".join(self.text_parts)

    def feed(self, line: str) -> list[str]:
        """Consume one SSE line; return the text deltas it carries (usually 0 or 1)."""
        line = line.strip()
        if not line or line.startswith(":") or not line.startswith("data:"):
            return []
        payload = line[5:].strip()
        if payload == "[DONE]":
            self.done = True
            return []
        try:
            d = json.loads(payload)
        except ValueError:
            return []
        if not isinstance(d, dict):
            return []
        deltas: list[str] = []
        p = self.provider
        if p == "anthropic":
            t = d.get("type")
            if t == "content_block_delta":
                dl = d.get("delta") or {}
                if dl.get("type") == "text_delta" and dl.get("text"):
                    deltas.append(dl["text"])
            elif t == "message_start":
                u = (d.get("message") or {}).get("usage") or {}
                self.usage["prompt_tokens"] = int(u.get("input_tokens") or 0)
            elif t == "message_delta":
                u = d.get("usage") or {}
                if u.get("output_tokens") is not None:
                    self.usage["completion_tokens"] = int(u.get("output_tokens") or 0)
                sr = (d.get("delta") or {}).get("stop_reason")
                if sr:
                    self.finish = sr
                    if sr == "refusal":
                        self.refused = True
            elif t == "message_stop":
                self.done = True
            elif t == "error":
                e = d.get("error") or {}
                et = str(e.get("type") or "")
                st = 529 if "overloaded" in et else 429 if "rate_limit" in et else 500
                self.error = _classify_http("anthropic", st, None, json.dumps({"error": e}))
        elif p == "groq":
            if d.get("error"):
                e = d["error"]
                self.error = _classify_http("groq", int(e.get("status_code") or 500) if isinstance(e, dict) else 500,
                                            None, json.dumps({"error": e}))
                return []
            for ch in d.get("choices") or []:
                c = (ch.get("delta") or {}).get("content")
                if c:
                    deltas.append(c)
                if ch.get("finish_reason"):
                    self.finish = ch["finish_reason"]
                    if self.finish == "content_filter":
                        self.refused = True
            u = d.get("usage") or (d.get("x_groq") or {}).get("usage")
            if isinstance(u, dict):
                self.usage = _usage(u.get("prompt_tokens"), u.get("completion_tokens"), u.get("total_tokens"))
        else:  # gemini
            if d.get("error"):
                e = d["error"]
                st = int(e.get("code") or 500) if isinstance(e, dict) else 500
                self.error = _classify_http("gemini", st, None, json.dumps({"error": e}))
                return []
            fb = d.get("promptFeedback") or {}
            if fb.get("blockReason"):
                self.refused = True
            for cand in d.get("candidates") or []:
                for part in ((cand.get("content") or {}).get("parts")) or []:
                    if isinstance(part, dict) and part.get("text") and not part.get("thought"):
                        deltas.append(part["text"])
                fr = cand.get("finishReason")
                if fr:
                    self.finish = fr
                    if fr in ("SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "RECITATION"):
                        self.refused = True
            um = d.get("usageMetadata")
            if isinstance(um, dict):
                self.usage = _usage(um.get("promptTokenCount"),
                                    (um.get("candidatesTokenCount") or 0) + (um.get("thoughtsTokenCount") or 0),
                                    um.get("totalTokenCount"))
        if deltas:
            self.text_parts.extend(deltas)
        return deltas

    def outcome(self) -> _Outcome:
        if self.error is not None and not self.text.strip():
            return self.error
        text = _clean_text(self.text, False)
        if not text:
            if self.refused:
                return _fail("refusal", "model refused", countable=False)
            return _fail("empty", f"empty stream (finish={self.finish})", countable=False)
        usage = dict(self.usage)
        if not usage.get("total_tokens"):
            usage["total_tokens"] = usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
        return _Outcome(ok=True, text=text, usage=usage, status=200)


# ---------------------------------------------------------------------------
# The chain driver (pure bookkeeping — no I/O; shared by sync/async/stream)
# ---------------------------------------------------------------------------
_usage_lock = threading.Lock()
_usage_totals: dict[str, dict[str, int]] = {}


def _log_usage(step: _Step, model: str, usage: dict | None) -> None:
    """INFO-log token usage (no content) and add it to the per-instance counters."""
    u = usage or {}
    p, c = int(u.get("prompt_tokens") or 0), int(u.get("completion_tokens") or 0)
    log.info("llm usage provider=%s model=%s prompt_tokens=%d completion_tokens=%d", step.provider, model, p, c)
    with _usage_lock:
        t = _usage_totals.setdefault(step.provider, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0})
        t["calls"] += 1
        t["prompt_tokens"] += p
        t["completion_tokens"] += c


def usage_totals() -> dict:
    with _usage_lock:
        return {k: dict(v) for k, v in _usage_totals.items()}


def _req_tokens(req: "_Req") -> int:
    chars = len(req.system) + sum(len(m["content"]) for m in req.messages)
    return _spend.estimate_tokens(chars, len(req.images))


class _Run:
    def __init__(self, sensitive: bool, deadline_s: float, has_images: bool = False, *,
                 owner: str | None = None, req: "_Req | None" = None) -> None:
        self.has_images = has_images
        self.owner = owner if owner is not None else _spend.current_owner()
        self.est_prompt = _req_tokens(req) if req is not None else 0
        self.oversize = False
        self.gate_reason: str | None = None
        self._gated = False
        self.start = _now()
        self.deadline = self.start + max(0.0, float(deadline_s))
        self.sensitive = sensitive
        self.trimmed = 0
        self.skipped: list[str] = []
        self.attempts: list[str] = []
        self.last_status: int | None = None
        self.dead_now: set[int] = set()
        self.deadline_hit = False
        self.eligible = [s for s in STEPS if _key(s) and not (sensitive and s.provider == "gemini")]

    # -- budgeting
    def remaining(self) -> float:
        return self.deadline - _now()

    def call_timeout(self) -> float:
        return max(0.1, min(CALL_TIMEOUT_S, self.remaining()))

    def timeout_obj(self) -> httpx.Timeout:
        t = self.call_timeout()
        return httpx.Timeout(t, connect=min(t, CONNECT_TIMEOUT_S))

    # -- spend guards (paid Anthropic step only; see backend/spend.py)
    def _wants_gate(self) -> bool:
        return (not self._gated and any(s.provider == "anthropic" for s in self.eligible)
                and self.est_prompt <= _max_request_tokens())

    def gate(self) -> None:
        """Decide ONCE per run whether the paid step may run (budget check; fails open)."""
        if self._wants_gate():
            self._gated = True
            self.gate_reason = _spend.check(self.owner, self.est_prompt)

    async def agate(self) -> None:
        if self._wants_gate():
            if _spend.ledger().needs_read(self.owner):      # a stale cache means one Mongo read: off the loop
                await asyncio.to_thread(self.gate)
            else:
                self.gate()

    def _paid_blocked(self, step: _Step) -> bool:
        cap = _max_request_tokens()
        if cap and self.est_prompt > cap:
            self.oversize = True
            self.skipped.append(f"{step.short} too-large")
            return True
        if self.gate_reason:
            self.skipped.append(f"{step.short} budget")
            return True
        return False

    def bill(self, step: _Step, usage: dict | None, text: str = "") -> None:
        """Record ACTUAL paid-provider tokens (estimated when the provider sent no usage)."""
        if step.provider != "anthropic":
            return
        u = usage or {}
        tokens = int(u.get("total_tokens") or 0) or int(u.get("prompt_tokens") or 0) + int(u.get("completion_tokens") or 0)
        if tokens <= 0:
            tokens = self.est_prompt + _est(text)
        _spend.add(self.owner, tokens)

    # -- iteration
    def tag(self, step: _Step, model: str, code: str) -> str:
        if len(step.models) == 1:
            return f"{step.short} {code}"
        return f"{step.short} {_short_model(model)} {code}"

    def candidates(self) -> Iterator[tuple[_Step, str, str]]:
        for step in self.eligible:
            key = _key(step)
            if not key:
                continue
            fp = hash(key)
            if _state.is_dead(step.n, fp) or _state.is_dead_shared(step.n, key):
                self.skipped.append(f"{step.short} dead-key(cached)")
                continue
            if step.provider == "anthropic" and self._paid_blocked(step):
                continue
            for model in step.models:
                if step.n in self.dead_now:
                    break
                if self.has_images and step.provider == "groq" and model != GROQ_VISION_MODEL:
                    self.skipped.append(self.tag(step, model, "no-vision"))
                    continue
                if self.remaining() < MIN_REMAINING_S:
                    self.deadline_hit = True
                    return
                if _state.is_exhausted(step.n, model):
                    self.skipped.append(self.tag(step, model, "quota(cached)"))
                    continue
                if _state.is_open(step.n, model):
                    self.skipped.append(self.tag(step, model, "circuit-open"))
                    continue
                yield step, model, key

    def retry_wait(self, out: _Outcome) -> float:
        """Seconds to wait before one re-try of the same (step, model), else 0."""
        if (not out.ok and out.status == 429 and not out.daily and out.retry_after is not None
                and 0 < out.retry_after <= RETRY_AFTER_MAX_S
                and out.retry_after + MIN_REMAINING_S < self.remaining()):
            return out.retry_after
        return 0.0

    # -- recording
    def record(self, step: _Step, model: str, out: _Outcome, key: str) -> None:
        log.info("llm step=%d(%s) model=%s status=%s %s", step.n, step.short, model,
                 "ok" if out.ok else (out.token or out.status), "" if out.ok else out.message)
        if out.ok:
            _state.on_success(step.n, model)
            _log_usage(step, model, out.usage)
            if out.usage:                      # a stream bills from its parsed usage in _final_stream_result
                self.bill(step, out.usage, out.text)
            return
        self.last_status = out.status if out.status is not None else self.last_status
        self.skipped.append(self.tag(step, model, out.token or "error"))
        code = self.tag(step, model, out.token or "error")
        self.attempts.append(f"{code}: {out.message}" if out.message else code)
        if out.dead_key:
            _state.mark_dead(step.n, hash(key))
            _state.share_dead(step.n, key)
            self.dead_now.add(step.n)
        elif out.daily:
            ttl = QUOTA_TTL_S
            if out.retry_after:
                ttl = min(QUOTA_TTL_S, max(60.0, out.retry_after))
            _state.mark_exhausted(step.n, model, ttl)
        elif out.countable:
            _state.on_failure(step.n, model)

    def result(self, step: _Step, model: str, out: _Outcome) -> ChatResult:
        return ChatResult(text=out.text, provider=step.provider, model=model, step=step.n,
                          step_name=step.name, usage=out.usage or _usage(0, 0),
                          skipped=list(self.skipped),
                          latency_ms=int((_now() - self.start) * 1000), trimmed=self.trimmed)

    def error(self) -> LLMChainError:
        if self.oversize and not any(s.provider != "anthropic" for s in self.eligible):
            return LLMRequestTooLarge(LLMRequestTooLarge.PUBLIC_DETAIL, status=413, attempts=list(self.skipped))
        if not self.eligible:
            if self.sensitive and any(_key(s) for s in STEPS):
                msg = "No non-Gemini LLM key is configured for a sensitive request"
            else:
                msg = "No LLM API key is configured (set at least one of: " + ", ".join(KEY_ENV_VARS) + ")"
            return LLMChainError(msg, status=None, attempts=[])
        why = "deadline exceeded; " if self.deadline_hit else ""
        trail = list(self.skipped)
        return LLMChainError(f"All LLM providers failed ({why}{'; '.join(trail) or 'no attempt made'})",
                             status=self.last_status, attempts=trail)


# ---------------------------------------------------------------------------
# One non-streaming attempt
# ---------------------------------------------------------------------------
def _attempt_sync(client: httpx.Client, run: _Run, step: _Step, model: str, key: str, req: _Req) -> _Outcome:
    url, headers, body = _build(step, model, req, key, stream=False, run=run)
    try:
        resp = client.post(url, headers=headers, json=body, timeout=run.timeout_obj())
    except Exception as exc:  # noqa: BLE001
        return _exc_outcome(exc)
    return _parse_response(step.provider, resp, req.json_mode)


async def _attempt_async(client: httpx.AsyncClient, run: _Run, step: _Step, model: str, key: str,
                         req: _Req) -> _Outcome:
    url, headers, body = _build(step, model, req, key, stream=False, run=run)
    try:
        resp = await client.post(url, headers=headers, json=body, timeout=run.timeout_obj())
    except Exception as exc:  # noqa: BLE001
        return _exc_outcome(exc)
    return _parse_response(step.provider, resp, req.json_mode)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def chat(system: str, messages: list[dict], *, max_tokens: int = 2048, temperature: float | None = 0.3,
         json_mode: bool = False, sensitive: bool = False, effort: str = "low",
         deadline_s: float = CHAIN_DEADLINE_S,
         images: list[tuple[bytes, str]] | None = None, owner: str | None = None) -> ChatResult:
    """Run the chain synchronously; return the first non-empty reply or raise ``LLMChainError``.

    ``images`` = [(bytes, mime)] attached to the last user message (Anthropic, Groq qwen only
    (max 3), Gemini); each image must be <= 5 MB or ValueError is raised."""
    req = _make_req(system, messages, max_tokens, temperature, json_mode, effort, images)
    run = _Run(sensitive, deadline_s, bool(req.images), owner=owner, req=req)
    run.gate()
    client = _get_sync_client()
    for step, model, key in run.candidates():
        out = _attempt_sync(client, run, step, model, key, req)
        wait = run.retry_wait(out)
        if wait:
            _sleep(wait)
            out = _attempt_sync(client, run, step, model, key, req)
        run.record(step, model, out, key)
        if out.ok:
            return run.result(step, model, out)
    raise run.error()


async def achat(system: str, messages: list[dict], *, max_tokens: int = 2048, temperature: float | None = 0.3,
                json_mode: bool = False, sensitive: bool = False, effort: str = "low",
                deadline_s: float = CHAIN_DEADLINE_S,
                images: list[tuple[bytes, str]] | None = None, owner: str | None = None) -> ChatResult:
    """Async twin of :func:`chat` (shared ``httpx.AsyncClient`` per event loop)."""
    req = _make_req(system, messages, max_tokens, temperature, json_mode, effort, images)
    run = _Run(sensitive, deadline_s, bool(req.images), owner=owner, req=req)
    await run.agate()
    client = _get_async_client()
    for step, model, key in run.candidates():
        out = await _attempt_async(client, run, step, model, key, req)
        wait = run.retry_wait(out)
        if wait:
            await _asleep(wait)
            out = await _attempt_async(client, run, step, model, key, req)
        run.record(step, model, out, key)
        if out.ok:
            return run.result(step, model, out)
    raise run.error()


def _norm_finish(raw, interrupted) -> str:
    """Provider stop reason -> "length" (cut off by the token limit) / "stop" / "" (unknown)."""
    r = str(raw or "").lower()
    if interrupted is not None:
        return "interrupted"             # the provider stream broke after output began (partial answer)
    if r in ("length", "max_tokens", "max_output_tokens"):
        return "length"
    return "stop" if r else ""


def _final_stream_result(run: _Run, step: _Step, model: str, parser: _SSEParser,
                         interrupted: _Outcome | None) -> ChatResult:
    usage = dict(parser.usage)
    if not usage.get("total_tokens"):
        usage["total_tokens"] = usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
    if interrupted is not None:
        run.skipped.append(run.tag(step, model, f"stream-{interrupted.token or 'error'}"))
    _log_usage(step, model, usage)
    run.bill(step, usage, parser.text)
    return ChatResult(text=_clean_text(parser.text, False), provider=step.provider, model=model, step=step.n,
                      step_name=step.name, usage=usage, skipped=list(run.skipped),
                      latency_ms=int((_now() - run.start) * 1000), trimmed=run.trimmed,
                      finish=_norm_finish(parser.finish, interrupted))


def stream_chat(system: str, messages: list[dict], *, max_tokens: int = 2048,
                temperature: float | None = 0.3, sensitive: bool = False,
                effort: str = "low",
                images: list[tuple[bytes, str]] | None = None,
                owner: str | None = None) -> Iterator[str | StreamStart | ChatResult]:
    """Yield text deltas as they arrive; the LAST item yielded is the final ``ChatResult``.

    Before the first delta of each provider that accepts the stream, a ``StreamStart`` (model
    reveal) is yielded; callers that only care about text should skip non-``str`` items that are
    not a ``ChatResult``.

    Falls through to the next model only BEFORE the first delta has been yielded. Raises
    ``LLMChainError`` if nothing started streaming. If the stream breaks after output began,
    the final ``ChatResult`` carries the partial text and a ``stream-<reason>`` skipped entry.
    """
    req = _make_req(system, messages, max_tokens, temperature, False, effort, images)
    run = _Run(sensitive, CHAIN_DEADLINE_S, bool(req.images), owner=owner, req=req)
    run.gate()
    client = _get_sync_client()
    for step, model, key in run.candidates():
        url, headers, body = _build(step, model, req, key, stream=True, run=run)
        out: _Outcome | None = None
        for attempt in range(2):
            parser = _SSEParser(step.provider)
            started = False
            interrupted: _Outcome | None = None
            out = None
            try:
                with client.stream("POST", url, headers=headers, json=body, timeout=run.timeout_obj()) as resp:
                    if resp.status_code != 200:
                        resp.read()
                        out = _parse_response(step.provider, resp, False)
                    else:
                        yield StreamStart(step.provider, model, step.n, step.name, list(run.skipped))
                        try:
                            for line in resp.iter_lines():
                                for d in parser.feed(line):
                                    started = True
                                    yield d
                                if parser.error is not None or parser.done:
                                    break
                        except Exception as exc:  # noqa: BLE001
                            if not started:
                                out = _exc_outcome(exc)
                            else:
                                interrupted = _exc_outcome(exc)
                        if out is None:
                            if started:
                                interrupted = interrupted or parser.error
                            else:
                                out = parser.outcome()
            except Exception as exc:  # noqa: BLE001
                if started:
                    interrupted = _exc_outcome(exc)
                else:
                    out = _exc_outcome(exc)
            if started:
                run.record(step, model, _Outcome(ok=True), key)
                yield _final_stream_result(run, step, model, parser, interrupted)
                return
            wait = run.retry_wait(out) if (out is not None and attempt == 0) else 0.0
            if wait:
                _sleep(wait)
                continue
            break
        assert out is not None
        run.record(step, model, out, key)
    raise run.error()


async def astream_chat(system: str, messages: list[dict], *, max_tokens: int = 2048,
                       temperature: float | None = 0.3, sensitive: bool = False,
                       effort: str = "low",
                       images: list[tuple[bytes, str]] | None = None,
                       owner: str | None = None) -> AsyncIterator[str | StreamStart | ChatResult]:
    """Async twin of :func:`stream_chat` (async generator)."""
    req = _make_req(system, messages, max_tokens, temperature, False, effort, images)
    run = _Run(sensitive, CHAIN_DEADLINE_S, bool(req.images), owner=owner, req=req)
    await run.agate()
    client = _get_async_client()
    for step, model, key in run.candidates():
        url, headers, body = _build(step, model, req, key, stream=True, run=run)
        out: _Outcome | None = None
        for attempt in range(2):
            parser = _SSEParser(step.provider)
            started = False
            interrupted: _Outcome | None = None
            out = None
            try:
                async with client.stream("POST", url, headers=headers, json=body,
                                         timeout=run.timeout_obj()) as resp:
                    if resp.status_code != 200:
                        await resp.aread()
                        out = _parse_response(step.provider, resp, False)
                    else:
                        yield StreamStart(step.provider, model, step.n, step.name, list(run.skipped))
                        try:
                            async for line in resp.aiter_lines():
                                for d in parser.feed(line):
                                    started = True
                                    yield d
                                if parser.error is not None or parser.done:
                                    break
                        except (GeneratorExit, asyncio.CancelledError):
                            raise
                        except Exception as exc:  # noqa: BLE001
                            if not started:
                                out = _exc_outcome(exc)
                            else:
                                interrupted = _exc_outcome(exc)
                        if out is None:
                            if started:
                                interrupted = interrupted or parser.error
                            else:
                                out = parser.outcome()
            except (GeneratorExit, asyncio.CancelledError):
                raise
            except Exception as exc:  # noqa: BLE001
                if started:
                    interrupted = _exc_outcome(exc)
                else:
                    out = _exc_outcome(exc)
            if started:
                run.record(step, model, _Outcome(ok=True), key)
                yield _final_stream_result(run, step, model, parser, interrupted)
                return
            wait = run.retry_wait(out) if (out is not None and attempt == 0) else 0.0
            if wait:
                await _asleep(wait)
                continue
            break
        assert out is not None
        run.record(step, model, out, key)
    raise run.error()


# ---------------------------------------------------------------------------
# Status (booleans / seconds only — never key material)
# ---------------------------------------------------------------------------
def chain_status() -> dict:
    ex, dead, op = _state.snapshot()
    steps = []
    for s in STEPS:
        key = _key(s)
        steps.append({
            "step": s.n,
            "name": s.name,
            "provider": s.provider,
            "env": s.env,
            "configured": bool(key),
            "dead_key": bool(key) and s.n in dead,
            "models": [{
                "model": m,
                "exhausted": (s.n, m) in ex,
                "exhausted_for_s": int(ex.get((s.n, m), 0)),
                "circuit_open": (s.n, m) in op,
                "circuit_open_for_s": int(op.get((s.n, m), 0)),
            } for m in s.models],
        })
    configured = [x["step"] for x in steps if x["configured"]]
    return {
        "available": bool(configured),
        "configured_steps": configured,
        "sensitive_capable": any(x["configured"] and x["provider"] != "gemini" for x in steps),
        "anthropic": steps[0]["configured"],
        "groq_1": steps[1]["configured"],
        "groq_2": steps[2]["configured"],
        "gemini_1": steps[3]["configured"],
        "gemini_2": steps[4]["configured"],
        "steps": steps,
        "policy": "auto: best available via chain",
        "timeouts": {"call_s": CALL_TIMEOUT_S, "chain_s": CHAIN_DEADLINE_S},
        "usage_totals": usage_totals(),
        "premium_budget": {k: v for k, v in _spend.status().items() if k in ("pct", "warn", "exhausted")},
    }
