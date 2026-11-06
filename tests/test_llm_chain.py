"""Unit tests for backend.llm_chain — every provider call is mocked (httpx.MockTransport)."""
from __future__ import annotations

import asyncio
import json
import threading
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest

from backend import llm_chain as lc

KEYS = {
    "ANTHROPIC_API_KEY": "test-anthropic-key-AAAA1111",
    "GROQ_API_KEY": "test-groq-key-one-BBBB2222",
    "GROQ_API_KEY_2": "test-groq-key-two-CCCC3333",
    "GEMINI_API_KEY": "test-gemini-key-one-DDDD4444",
    "GEMINI_API_KEY_2": "test-gemini-key-two-EEEE5555",
}
STEP_OF_KEY = {v: i + 1 for i, v in enumerate(KEYS.values())}
CLAUDE = "claude-sonnet-5-5"
G_MODELS = list(lc.GROQ_MODELS)
GEM_MODELS = list(lc.GEMINI_MODELS)
MSGS = [{"role": "user", "content": "hello"}]


# ---------------------------------------------------------------------------
# Response builders
# ---------------------------------------------------------------------------
def r_anthropic(text="hi", stop="end_turn", usage=(11, 7)):
    content = [{"type": "text", "text": text}] if text is not None else []
    return httpx.Response(200, json={"content": content, "stop_reason": stop,
                                     "usage": {"input_tokens": usage[0], "output_tokens": usage[1]}})


def r_groq(text="hi", finish="stop", usage=(11, 7)):
    return httpx.Response(200, json={
        "choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": finish}],
        "usage": {"prompt_tokens": usage[0], "completion_tokens": usage[1], "total_tokens": sum(usage)}})


def r_gemini(text="hi", finish="STOP", usage=(11, 7), thought=None):
    parts = []
    if thought:
        parts.append({"text": thought, "thought": True})
    if text is not None:
        parts.append({"text": text})
    return httpx.Response(200, json={
        "candidates": [{"content": {"parts": parts, "role": "model"}, "finishReason": finish}],
        "usageMetadata": {"promptTokenCount": usage[0], "candidatesTokenCount": usage[1],
                          "totalTokenCount": sum(usage)}})


def err(status, message="boom", headers=None, body=None):
    return httpx.Response(status, headers=headers or {},
                          json=body if body is not None else {"error": {"message": message}})


def sse(lines):
    return httpx.Response(200, headers={"content-type": "text/event-stream"},
                          content=("\n".join(lines) + "\n").encode())


def sse_groq(deltas, finish="stop"):
    out = [f"data: {json.dumps({'choices': [{'delta': {'content': d}}]})}" for d in deltas]
    out.append(f"data: {json.dumps({'choices': [{'delta': {}, 'finish_reason': finish}], 'x_groq': {'usage': {'prompt_tokens': 5, 'completion_tokens': 3, 'total_tokens': 8}}})}")
    out.append("data: [DONE]")
    return sse(out)


def sse_anthropic(deltas, stop="end_turn"):
    out = ["event: message_start", f"data: {json.dumps({'type': 'message_start', 'message': {'usage': {'input_tokens': 9}}})}"]
    for d in deltas:
        out += ["event: content_block_delta",
                f"data: {json.dumps({'type': 'content_block_delta', 'delta': {'type': 'text_delta', 'text': d}})}"]
    out += ["event: message_delta",
            f"data: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': stop}, 'usage': {'output_tokens': 4}})}",
            f"data: {json.dumps({'type': 'message_stop'})}"]
    return sse(out)


def sse_gemini(deltas, finish="STOP"):
    out = []
    for i, d in enumerate(deltas):
        cand = {"content": {"parts": [{"text": d}]}}
        if i == len(deltas) - 1:
            cand["finishReason"] = finish
        out.append(f"data: {json.dumps({'candidates': [cand], 'usageMetadata': {'promptTokenCount': 6, 'candidatesTokenCount': 2, 'totalTokenCount': 8}})}")
    if not deltas:
        out.append(f"data: {json.dumps({'candidates': [{'finishReason': finish}]})}")
    return sse(out)


class Truncated(httpx.Response):
    """A 200 SSE response whose body stream dies after the given bytes."""

    def __init__(self, data: bytes):
        def gen():
            yield data
            raise httpx.ReadError("connection reset")
        super().__init__(200, headers={"content-type": "text/event-stream"}, stream=_ErrStream(gen))


class _ErrStream(httpx.SyncByteStream, httpx.AsyncByteStream):
    def __init__(self, gen):
        self._gen = gen

    def __iter__(self):
        yield from self._gen()

    async def __aiter__(self):
        for chunk in self._gen():
            yield chunk


# ---------------------------------------------------------------------------
# Fake network
# ---------------------------------------------------------------------------
class Net:
    """Routes mocked provider calls. rules[(step, model)] = response | callable(req)->response |
    list (consumed in order; the last element repeats). Unrouted calls return a 500 and are recorded."""

    def __init__(self):
        self.rules: dict = {}
        self.calls: list[dict] = []
        self.lock = threading.Lock()

    def on(self, step, model, resp):
        self.rules[(step, model)] = resp

    def on_all(self, resp, steps=(1, 2, 3, 4, 5)):
        for s in lc.STEPS:
            if s.n in steps:
                for m in s.models:
                    self.rules[(s.n, m)] = resp

    def handler(self, request: httpx.Request):
        url = str(request.url)
        body = json.loads(request.content or b"{}")
        if "anthropic" in url:
            key = request.headers["x-api-key"]
            model = body["model"]
        elif "groq" in url:
            key = request.headers["authorization"].split(" ", 1)[1]
            model = body["model"]
        else:
            key = request.headers["x-goog-api-key"]
            model = url.split("/models/")[1].split(":")[0]
        step = STEP_OF_KEY[key]
        with self.lock:
            self.calls.append({"step": step, "model": model, "body": body, "url": url,
                               "headers": dict(request.headers),
                               "timeout": request.extensions.get("timeout")})
            rule = self.rules.get((step, model))
            if isinstance(rule, list):
                resp = rule[0] if len(rule) == 1 else rule.pop(0)
            else:
                resp = rule
        if resp is None:
            return httpx.Response(500, json={"error": {"message": "unrouted"}})
        if callable(resp):
            resp = resp(request)
        if isinstance(resp, Exception):
            raise resp
        return resp

    @property
    def order(self):
        return [(c["step"], c["model"]) for c in self.calls]

    def calls_for(self, step):
        return [c for c in self.calls if c["step"] == step]


@pytest.fixture()
def net(monkeypatch):
    for k, v in KEYS.items():
        monkeypatch.setenv(k, v)
    lc.reset_state()
    n = Net()
    lc._configure_transports(httpx.MockTransport(n.handler))
    sleeps: list[float] = []
    monkeypatch.setattr(lc, "_sleep", lambda s: sleeps.append(s))

    async def fake_asleep(s):
        sleeps.append(s)
    monkeypatch.setattr(lc, "_asleep", fake_asleep)
    n.sleeps = sleeps
    yield n
    lc._configure_transports(None)
    lc.reset_state()


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, s):
        self.t += s


@pytest.fixture()
def clock(monkeypatch):
    c = Clock()
    monkeypatch.setattr(lc, "_now", c)
    return c


def only_keys(monkeypatch, *names):
    for k in KEYS:
        if k in names:
            monkeypatch.setenv(k, KEYS[k])
        else:
            monkeypatch.delenv(k, raising=False)


# ---------------------------------------------------------------------------
# Ordering / success
# ---------------------------------------------------------------------------
def test_anthropic_first_and_stops_there(net):
    net.on(1, CLAUDE, r_anthropic("Hello there"))
    r = lc.chat("sys", MSGS)
    assert net.order == [(1, CLAUDE)]
    assert (r.text, r.provider, r.model, r.step, r.step_name) == ("Hello there", "anthropic", CLAUDE, 1, "Anthropic")
    assert r.usage == {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}
    assert r.skipped == []
    assert r.label == f"Model: anthropic · {CLAUDE} · step 1 (Anthropic)"
    assert r.to_public() == {"provider": "anthropic", "model": CLAUDE, "step": 1, "step_name": "Anthropic",
                             "label": r.label, "skipped": []}
    assert r.latency_ms >= 0


def test_full_order_when_everything_fails(net):
    net.on_all(err(500))
    with pytest.raises(lc.LLMChainError) as ei:
        lc.chat("sys", MSGS)
    expected = [(1, CLAUDE)] + [(s, m) for s in (2, 3) for m in G_MODELS] + [(s, m) for s in (4, 5) for m in GEM_MODELS]
    assert net.order == expected
    e = ei.value
    assert e.label == "Model: none (all providers failed)"
    assert lc.LLMChainError.label == "Model: none (all providers failed)"
    assert e.status == 500
    assert len(e.attempts) == 13
    assert e.attempts[0].startswith("anthropic 500")
    assert any(a.startswith("groq#1 gpt-oss-120b 500") for a in e.attempts)
    assert e.to_public()["label"] == e.label


def test_only_allowed_models_are_ever_sent(net):
    net.on_all(err(500))
    with pytest.raises(lc.LLMChainError):
        lc.chat("s", MSGS)
    sent = {c["model"] for c in net.calls}
    assert sent == set(lc.ALLOWED_MODELS) and len(sent) == 7
    assert not any("opus" in m or "haiku" in m for m in sent)


def test_falls_through_to_groq_then_label_and_skipped(net):
    net.on(1, CLAUDE, err(529, "overloaded"))
    net.on(2, G_MODELS[0], err(429, "slow down"))
    net.on(2, G_MODELS[1], r_groq("from 20b"))
    r = lc.chat("s", MSGS)
    assert net.order == [(1, CLAUDE), (2, G_MODELS[0]), (2, G_MODELS[1])]
    assert r.skipped == ["anthropic 529", "groq#1 gpt-oss-120b 429"]
    assert r.label == "Model: groq · openai/gpt-oss-20b · step 2 (Groq key 1)"
    assert r.usage["total_tokens"] == 18


@pytest.mark.parametrize("step,provider,models,builder,name", [
    (2, "groq", G_MODELS, r_groq, "Groq key 1"),
    (3, "groq", G_MODELS, r_groq, "Groq key 2"),
    (4, "gemini", GEM_MODELS, r_gemini, "Gemini key 1"),
    (5, "gemini", GEM_MODELS, r_gemini, "Gemini key 2"),
])
def test_each_step_can_win(net, step, provider, models, builder, name):
    for s in range(1, step):
        net.on_all(err(503), steps=(s,))
    net.on(step, models[-1], builder("last model wins"))
    r = lc.chat("s", MSGS)
    assert (r.step, r.step_name, r.provider, r.model) == (step, name, provider, models[-1])
    assert r.label == f"Model: {provider} · {models[-1]} · step {step} ({name})"
    assert r.text == "last model wins"


def test_blank_or_missing_keys_skip_silently(net, monkeypatch):
    only_keys(monkeypatch, "GROQ_API_KEY_2", "GEMINI_API_KEY")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "   ")      # blank = missing
    net.on(3, G_MODELS[0], r_groq("ok"))
    r = lc.chat("s", MSGS)
    assert net.order == [(3, G_MODELS[0])]
    assert r.step == 3 and r.skipped == []


def test_no_keys_raises_without_any_call(net, monkeypatch):
    only_keys(monkeypatch)
    with pytest.raises(lc.LLMChainError) as ei:
        lc.chat("s", MSGS)
    assert net.calls == []
    assert ei.value.attempts == [] and ei.value.status is None
    assert "ANTHROPIC_API_KEY" in str(ei.value)
    assert not lc.has_any_key()


# ---------------------------------------------------------------------------
# Failure kinds all fall through
# ---------------------------------------------------------------------------
FAILURES = {
    "429": (lambda: err(429, "rate limited"), "429"),
    "500": (lambda: err(500), "500"),
    "503": (lambda: err(503), "503"),
    "529": (lambda: err(529, "overloaded"), "529"),
    "400": (lambda: err(400, "bad request"), "400"),
    "404": (lambda: err(404, "model not found"), "404"),
    "timeout": (lambda: httpx.ReadTimeout("slow"), "timeout"),
    "connect": (lambda: httpx.ConnectError("down"), "neterr"),
    "empty": (lambda: r_anthropic(""), "empty"),
    "no-content": (lambda: r_anthropic(None), "empty"),
    "refusal": (lambda: r_anthropic("", stop="refusal"), "refusal"),
    "badjson": (lambda: httpx.Response(200, content=b"<html>nope</html>"), "badjson"),
}


@pytest.mark.parametrize("name", list(FAILURES))
def test_failure_kinds_fall_through_to_next_step(net, name):
    make, token = FAILURES[name]
    net.on(1, CLAUDE, make())
    net.on(2, G_MODELS[0], r_groq("rescued"))
    r = lc.chat("s", MSGS)
    assert r.text == "rescued" and r.step == 2
    assert r.skipped == [f"anthropic {token}"]
    assert len(net.calls_for(1)) == 1            # never retried on the same key+model


def test_groq_empty_reasoning_truncation_falls_through(net):
    net.on(1, CLAUDE, err(500))
    net.on(2, G_MODELS[0], r_groq("", finish="length"))
    net.on(2, G_MODELS[1], r_groq("fine"))
    r = lc.chat("s", MSGS)
    assert r.model == G_MODELS[1]
    assert "groq#1 gpt-oss-120b empty" in r.skipped


def test_gemini_safety_block_and_empty_fall_through(net, monkeypatch):
    only_keys(monkeypatch, "GEMINI_API_KEY")
    net.on(4, GEM_MODELS[0], httpx.Response(200, json={"promptFeedback": {"blockReason": "SAFETY"}}))
    net.on(4, GEM_MODELS[1], r_gemini("", finish="SAFETY"))
    net.on(4, GEM_MODELS[2], r_gemini("ok"))
    r = lc.chat("s", MSGS)
    assert r.model == GEM_MODELS[2]
    assert r.skipped == ["gemini#1 gemini-3.8-flash refusal", "gemini#1 gemini-3.6-flash refusal"]


def test_gemini_thought_parts_are_excluded(net, monkeypatch):
    only_keys(monkeypatch, "GEMINI_API_KEY")
    net.on(4, GEM_MODELS[0], r_gemini("answer", thought="secret reasoning"))
    assert lc.chat("s", MSGS).text == "answer"


def test_think_tags_stripped(net, monkeypatch):
    only_keys(monkeypatch, "GROQ_API_KEY")
    net.on(2, G_MODELS[0], r_groq("<think>hmm</think>\nFinal answer"))
    assert lc.chat("s", MSGS).text == "Final answer"


# ---------------------------------------------------------------------------
# 401/403 dead keys
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("status", [401, 402, 403])
def test_auth_failure_skips_rest_of_that_keys_models(net, status):
    net.on(2, G_MODELS[0], err(status, "invalid api key"))
    net.on(3, G_MODELS[0], r_groq("key two"))
    net.on(1, CLAUDE, err(500))
    r = lc.chat("s", MSGS)
    assert (2, G_MODELS[1]) not in net.order and (2, G_MODELS[2]) not in net.order
    assert r.step == 3
    assert r.skipped == ["anthropic 500", f"groq#1 gpt-oss-120b {status}"]


def test_dead_key_is_remembered_then_expires(net, clock):
    net.on(1, CLAUDE, err(401, "bad key"))
    net.on(2, G_MODELS[0], r_groq("ok"))
    r1 = lc.chat("s", MSGS)
    assert r1.skipped == ["anthropic 401"]
    n_calls = len(net.calls_for(1))
    r2 = lc.chat("s", MSGS)
    assert len(net.calls_for(1)) == n_calls          # not even tried
    assert r2.skipped == ["anthropic dead-key(cached)"]
    assert lc.chain_status()["steps"][0]["dead_key"] is True
    clock.advance(lc.DEAD_KEY_TTL_S + 1)
    net.on(1, CLAUDE, r_anthropic("revived"))
    r3 = lc.chat("s", MSGS)
    assert r3.step == 1 and r3.text == "revived"


def test_replacing_a_dead_key_clears_the_memory(net, monkeypatch):
    net.on(1, CLAUDE, err(401, "bad"))
    net.on(2, G_MODELS[0], r_groq("ok"))
    lc.chat("s", MSGS)
    monkeypatch.setenv("ANTHROPIC_API_KEY", KEYS["ANTHROPIC_API_KEY"] + "-rotated")
    STEP_OF_KEY[KEYS["ANTHROPIC_API_KEY"] + "-rotated"] = 1
    net.on(1, CLAUDE, r_anthropic("new key works"))
    assert lc.chat("s", MSGS).text == "new key works"


def test_gemini_api_key_invalid_400_marks_key_dead(net, monkeypatch):
    only_keys(monkeypatch, "GEMINI_API_KEY", "GEMINI_API_KEY_2")
    net.on(4, GEM_MODELS[0], err(400, body={"error": {"code": 400, "message": "API key not valid. Please pass a valid API key.",
                                                      "status": "INVALID_ARGUMENT",
                                                      "details": [{"reason": "API_KEY_INVALID"}]}}))
    net.on(5, GEM_MODELS[0], r_gemini("second key"))
    r = lc.chat("s", MSGS)
    assert [c for c in net.order if c[0] == 4] == [(4, GEM_MODELS[0])]
    assert r.step == 5


# ---------------------------------------------------------------------------
# Quota memory
# ---------------------------------------------------------------------------
GROQ_TPD = {"error": {"message": "Rate limit reached for model `openai/gpt-oss-120b` in organization `org_x` "
                                 "service tier `on_demand` on tokens per day (TPD): Limit 200000, Used 199990, "
                                 "Requested 500. Please try again in 11m31.2s.", "type": "tokens",
                      "code": "rate_limit_exceeded"}}
GROQ_TPM = {"error": {"message": "Rate limit reached for model `openai/gpt-oss-120b` on tokens per minute (TPM): "
                                 "Limit 8000, Used 7900, Requested 500. Please try again in 5s.",
                      "type": "tokens", "code": "rate_limit_exceeded"}}
GEM_DAILY = {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED",
                       "message": "You exceeded your current quota, please check your plan and billing details.",
                       "details": [{"@type": "type.googleapis.com/google.rpc.QuotaFailure",
                                    "violations": [{"quotaMetric": "generativelanguage.googleapis.com/generate_content_free_tier_requests",
                                                    "quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]}]}}
GEM_MINUTE = {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED",
                        "message": "You exceeded your current quota.",
                        "details": [{"violations": [{"quotaId": "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"}]},
                                    {"retryDelay": "23s"}]}}


def test_groq_daily_quota_is_remembered_and_self_heals(net, clock, monkeypatch):
    only_keys(monkeypatch, "GROQ_API_KEY")
    net.on(2, G_MODELS[0], err(429, body=GROQ_TPD, headers={"retry-after": "691"}))
    net.on(2, G_MODELS[1], r_groq("20b"))
    r1 = lc.chat("s", MSGS)
    assert r1.model == G_MODELS[1] and r1.skipped == ["groq#1 gpt-oss-120b 429"]
    assert len(net.calls_for(2)) == 2
    assert lc.chain_status()["steps"][1]["models"][0]["exhausted"] is True
    r2 = lc.chat("s", MSGS)
    assert [c["model"] for c in net.calls_for(2)] == [G_MODELS[0], G_MODELS[1], G_MODELS[1]]   # 120b not retried
    assert r2.skipped == ["groq#1 gpt-oss-120b quota(cached)"]
    clock.advance(lc.QUOTA_TTL_S + 1)
    net.on(2, G_MODELS[0], r_groq("120b is back"))
    r3 = lc.chat("s", MSGS)
    assert r3.model == G_MODELS[0] and r3.text == "120b is back"


def test_quota_memory_is_per_step_and_model(net):
    net.on(2, G_MODELS[0], err(429, body=GROQ_TPD))
    net.on(2, G_MODELS[1], r_groq("ok"))
    net.on(3, G_MODELS[0], r_groq("key 2 still has quota"))
    net.on(1, CLAUDE, err(500))
    lc.chat("s", MSGS)
    net.rules[(2, G_MODELS[1])] = err(500)
    r = lc.chat("s", MSGS)               # key1 120b skipped (cached), 20b 500, qwen 500 -> key 2 120b
    assert r.step == 3 and r.model == G_MODELS[0]


def test_per_minute_429_is_not_remembered(net, monkeypatch):
    only_keys(monkeypatch, "GROQ_API_KEY")
    net.on(2, G_MODELS[0], [err(429, body=GROQ_TPM, headers={"retry-after": "30"}), r_groq("later ok")])
    net.on(2, G_MODELS[1], r_groq("20b"))
    r1 = lc.chat("s", MSGS)
    assert r1.model == G_MODELS[1]
    assert net.sleeps == []                     # 30 s > 2 s: never wait
    r2 = lc.chat("s", MSGS)
    assert r2.model == G_MODELS[0]              # tried again (no daily-quota memory)


def test_gemini_daily_vs_minute_classification(net, monkeypatch):
    only_keys(monkeypatch, "GEMINI_API_KEY")
    net.on(4, GEM_MODELS[0], err(429, body=GEM_DAILY))
    net.on(4, GEM_MODELS[1], err(429, body=GEM_MINUTE))
    net.on(4, GEM_MODELS[2], r_gemini("flash-lite"))
    lc.chat("s", MSGS)
    st = {m["model"]: m for m in lc.chain_status()["steps"][3]["models"]}
    assert st[GEM_MODELS[0]]["exhausted"] is True
    assert st[GEM_MODELS[1]]["exhausted"] is False
    assert st[GEM_MODELS[2]]["exhausted"] is False


def test_short_retry_after_waits_once_then_succeeds(net, monkeypatch):
    only_keys(monkeypatch, "GROQ_API_KEY")
    net.on(2, G_MODELS[0], [err(429, body=GROQ_TPM, headers={"retry-after": "1.5"}), r_groq("after wait")])
    r = lc.chat("s", MSGS)
    assert net.sleeps == [1.5]
    assert r.text == "after wait" and r.model == G_MODELS[0] and r.skipped == []
    assert len(net.calls) == 2


def test_short_retry_after_waits_at_most_once(net, monkeypatch):
    only_keys(monkeypatch, "GROQ_API_KEY")
    net.on(2, G_MODELS[0], err(429, body=GROQ_TPM, headers={"retry-after": "1"}))
    net.on(2, G_MODELS[1], r_groq("20b"))
    r = lc.chat("s", MSGS)
    assert net.sleeps == [1.0]
    assert [c["model"] for c in net.calls] == [G_MODELS[0], G_MODELS[0], G_MODELS[1]]
    assert r.model == G_MODELS[1]


def test_retry_after_not_waited_when_it_does_not_fit_deadline(net, monkeypatch):
    only_keys(monkeypatch, "GROQ_API_KEY")
    net.on(2, G_MODELS[0], err(429, body=GROQ_TPM, headers={"retry-after": "1.5"}))
    net.on(2, G_MODELS[1], r_groq("20b"))
    r = lc.chat("s", MSGS, deadline_s=2.0)
    assert net.sleeps == [] and r.model == G_MODELS[1]


def test_anthropic_credit_balance_counts_as_exhausted(net):
    net.on(1, CLAUDE, err(400, "Your credit balance is too low to access the Anthropic API."))
    net.on(2, G_MODELS[0], r_groq("ok"))
    lc.chat("s", MSGS)
    assert lc.chain_status()["steps"][0]["models"][0]["exhausted"] is True
    lc.chat("s", MSGS)
    assert len(net.calls_for(1)) == 1


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------
def test_circuit_opens_after_three_failures_and_recovers(net, clock, monkeypatch):
    only_keys(monkeypatch, "GROQ_API_KEY")
    net.on(2, G_MODELS[0], err(500))
    net.on(2, G_MODELS[1], r_groq("20b"))
    for _ in range(3):
        lc.chat("s", MSGS)
    assert len(net.calls_for(2)) == 6                 # 120b failed 3x (+20b ok 3x)
    assert lc.chain_status()["steps"][1]["models"][0]["circuit_open"] is True
    r = lc.chat("s", MSGS)
    assert [c["model"] for c in net.calls_for(2)].count(G_MODELS[0]) == 3      # skipped without a request
    assert r.skipped == ["groq#1 gpt-oss-120b circuit-open"]
    clock.advance(lc.CIRCUIT_OPEN_S + 1)
    net.on(2, G_MODELS[0], r_groq("120b recovered"))
    r = lc.chat("s", MSGS)
    assert r.model == G_MODELS[0]
    assert lc.chain_status()["steps"][1]["models"][0]["circuit_open"] is False


def test_failed_probe_reopens_circuit_immediately(net, clock, monkeypatch):
    only_keys(monkeypatch, "GROQ_API_KEY")
    net.on(2, G_MODELS[0], httpx.ReadTimeout("t"))
    net.on(2, G_MODELS[1], r_groq("ok"))
    for _ in range(3):
        lc.chat("s", MSGS)
    clock.advance(lc.CIRCUIT_OPEN_S + 1)
    lc.chat("s", MSGS)                                  # half-open probe fails
    assert lc.chain_status()["steps"][1]["models"][0]["circuit_open"] is True


def test_success_resets_failure_streak(net, monkeypatch):
    only_keys(monkeypatch, "GROQ_API_KEY")
    net.on(2, G_MODELS[1], r_groq("20b"))
    seq = [err(500), err(500), r_groq("ok"), err(500), err(500)]
    net.on(2, G_MODELS[0], seq[:])
    for _ in range(5):
        lc.chat("s", MSGS)
    assert lc.chain_status()["steps"][1]["models"][0]["circuit_open"] is False


def test_400_and_empty_do_not_trip_circuit(net, monkeypatch):
    only_keys(monkeypatch, "GROQ_API_KEY")
    net.on(2, G_MODELS[0], err(400, "too long"))
    net.on(2, G_MODELS[1], r_groq("ok"))
    for _ in range(5):
        lc.chat("s", MSGS)
    assert len([c for c in net.calls_for(2) if c["model"] == G_MODELS[0]]) == 5


# ---------------------------------------------------------------------------
# Sensitive
# ---------------------------------------------------------------------------
def test_sensitive_never_calls_gemini(net):
    net.on_all(err(503))
    with pytest.raises(lc.LLMChainError) as ei:
        lc.chat("s", MSGS, sensitive=True)
    assert {c["step"] for c in net.calls} == {1, 2, 3}
    assert not any("generativelanguage" in c["url"] for c in net.calls)
    assert len(ei.value.attempts) == 7


def test_sensitive_with_only_gemini_keys_makes_no_call(net, monkeypatch):
    only_keys(monkeypatch, "GEMINI_API_KEY", "GEMINI_API_KEY_2")
    with pytest.raises(lc.LLMChainError) as ei:
        lc.chat("s", MSGS, sensitive=True)
    assert net.calls == []
    assert "sensitive" in str(ei.value)


def test_sensitive_still_uses_groq_and_non_sensitive_reaches_gemini(net):
    net.on_all(err(503), steps=(1, 2, 3))
    net.on(4, GEM_MODELS[0], r_gemini("gem"))
    assert lc.chat("s", MSGS).step == 4
    net.calls.clear()
    net.on(3, G_MODELS[2], r_groq("groq saves"))
    assert lc.chat("s", MSGS, sensitive=True).step == 3


# ---------------------------------------------------------------------------
# Request shape / forbidden params
# ---------------------------------------------------------------------------
def test_anthropic_request_shape_and_forbidden_params(net):
    net.on(1, CLAUDE, r_anthropic())
    lc.chat("be nice", [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"},
                        {"role": "user", "content": "q2"}, {"role": "assistant", "content": "prefill"}],
            max_tokens=77, temperature=0.9, effort="medium")
    c = net.calls[0]
    b = c["body"]
    assert c["url"] == "https://api.anthropic.com/v1/messages"
    assert c["headers"]["x-api-key"] == KEYS["ANTHROPIC_API_KEY"]
    assert c["headers"]["anthropic-version"] == "2023-06-01"
    assert b["model"] == CLAUDE and b["max_tokens"] == 64000 and b["system"] == "be nice"
    assert b["output_config"] == {"effort": "medium"}
    for forbidden in ("temperature", "top_p", "top_k", "thinking"):
        assert forbidden not in b
    assert b["messages"][-1] == {"role": "user", "content": "q2"}       # assistant prefill dropped
    assert "stream" not in b


def test_groq_request_shape(net, monkeypatch):
    only_keys(monkeypatch, "GROQ_API_KEY")
    net.on_all(err(500), steps=(2,))
    net.on(2, G_MODELS[2], r_groq())
    lc.chat("sys", MSGS, max_tokens=64, temperature=0.3)
    by_model = {c["model"]: c for c in net.calls}
    for m, c in by_model.items():
        b = c["body"]
        assert c["url"] == "https://api.groq.com/openai/v1/chat/completions"
        assert c["headers"]["authorization"] == f"Bearer {KEYS['GROQ_API_KEY']}"
        assert b["messages"][0] == {"role": "system", "content": "sys"}
        assert b["max_completion_tokens"] >= 2048           # floor even though caller asked for 64
        assert b["temperature"] == 0.3
        for forbidden in ("logprobs", "logit_bias", "n", "max_tokens", "stream"):
            assert forbidden not in b
        assert all("name" not in m_ for m_ in b["messages"])
    assert by_model[G_MODELS[0]]["body"]["reasoning_effort"] == "low"
    assert by_model[G_MODELS[1]]["body"]["reasoning_effort"] == "low"
    assert "reasoning_effort" not in by_model[G_MODELS[2]]["body"]


def test_groq_large_max_tokens_preserved(net, monkeypatch):
    only_keys(monkeypatch, "GROQ_API_KEY")
    net.on(2, G_MODELS[0], r_groq())
    lc.chat("s", MSGS, max_tokens=6000)
    assert net.calls[0]["body"]["max_completion_tokens"] == 6000


def test_gemini_request_shape_and_forbidden_params(net, monkeypatch):
    only_keys(monkeypatch, "GEMINI_API_KEY")
    net.on_all(err(503), steps=(4,))
    net.on(4, GEM_MODELS[2], r_gemini())
    lc.chat("sys", [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"},
                    {"role": "user", "content": "q2"}], max_tokens=64, temperature=0.3, effort="minimal")
    by_model = {c["model"]: c for c in net.calls}
    for m, c in by_model.items():
        assert c["url"] == f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent"
        assert c["headers"]["x-goog-api-key"] == KEYS["GEMINI_API_KEY"]
        b = c["body"]
        assert b["systemInstruction"] == {"parts": [{"text": "sys"}]}
        assert [x["role"] for x in b["contents"]] == ["user", "model", "user"]
        g = b["generationConfig"]
        assert g["maxOutputTokens"] >= 2048
        assert "thinkingConfig" in g
        assert "x-goog-api-key" in c["headers"] and "key=" not in c["url"]    # key only in header
    for m in (GEM_MODELS[0], GEM_MODELS[1]):                       # 3.8 / 3.6
        g = by_model[m]["body"]["generationConfig"]
        assert not ({"temperature", "topP", "topK"} & set(g))
    assert by_model[GEM_MODELS[0]]["body"]["generationConfig"]["thinkingConfig"]["thinkingLevel"] == "low"   # never minimal for 3.8
    assert by_model[GEM_MODELS[2]]["body"]["generationConfig"]["temperature"] == 0.3
    for m, c in by_model.items():
        assert c["body"]["generationConfig"]["thinkingConfig"]["thinkingLevel"] != "minimal"


@pytest.mark.parametrize("effort,expected", [("low", "low"), ("medium", "medium"), ("high", "high"), ("max", "high")])
def test_effort_mapping(net, monkeypatch, effort, expected):
    only_keys(monkeypatch, "GEMINI_API_KEY")
    net.on(4, GEM_MODELS[0], r_gemini())
    lc.chat("s", MSGS, effort=effort)
    assert net.calls[0]["body"]["generationConfig"]["thinkingConfig"]["thinkingLevel"] == expected


def test_temperature_none_is_omitted_everywhere(net, monkeypatch):
    only_keys(monkeypatch, "GROQ_API_KEY", "GEMINI_API_KEY")
    net.on_all(err(500), steps=(2,))
    net.on(4, GEM_MODELS[2], r_gemini())
    net.on(4, GEM_MODELS[0], err(500))
    net.on(4, GEM_MODELS[1], err(500))
    lc.chat("s", MSGS, temperature=None)
    for c in net.calls:
        assert "temperature" not in c["body"] and "temperature" not in c["body"].get("generationConfig", {})


# ---------------------------------------------------------------------------
# JSON mode
# ---------------------------------------------------------------------------
def test_json_mode_all_three_providers(net):
    net.on(1, CLAUDE, err(500))
    net.on_all(err(500), steps=(2, 3, 4))
    net.on(5, GEM_MODELS[0], r_gemini('{"a": 1}'))
    net.on(2, G_MODELS[0], err(500))
    lc.chat("Return data.", [{"role": "user", "content": "give me data"}], json_mode=True)
    anth = net.calls_for(1)[0]["body"]
    assert "response_format" not in anth and "json" in anth["system"].lower()
    groq = net.calls_for(2)[0]["body"]
    assert groq["response_format"] == {"type": "json_object"}
    assert "json" in (groq["messages"][0]["content"] + groq["messages"][1]["content"]).lower()
    gem = net.calls_for(4)[0]["body"]
    assert gem["generationConfig"]["responseMimeType"] == "application/json"


def test_json_mode_not_duplicated_when_prompt_says_json(net, monkeypatch):
    only_keys(monkeypatch, "GROQ_API_KEY")
    net.on(2, G_MODELS[0], r_groq('{"ok": true}'))
    lc.chat("Answer in JSON please.", MSGS, json_mode=True)
    assert net.calls[0]["body"]["messages"][0]["content"] == "Answer in JSON please."


def test_json_mode_strips_code_fences(net, monkeypatch):
    only_keys(monkeypatch, "ANTHROPIC_API_KEY")
    net.on(1, CLAUDE, r_anthropic('```json\n{"x": [1, 2]}\n```'))
    r = lc.chat("s", MSGS, json_mode=True)
    assert json.loads(r.text) == {"x": [1, 2]}


def test_non_json_mode_leaves_fences_alone(net, monkeypatch):
    only_keys(monkeypatch, "ANTHROPIC_API_KEY")
    net.on(1, CLAUDE, r_anthropic("```py\nprint(1)\n```"))
    assert lc.chat("s", MSGS).text.startswith("```py")


# ---------------------------------------------------------------------------
# Message normalisation
# ---------------------------------------------------------------------------
def test_messages_validation_and_merging(net):
    with pytest.raises(ValueError):
        lc.chat("s", [])
    with pytest.raises(ValueError):
        lc.chat("s", [{"role": "user", "content": "   "}])
    net.on(1, CLAUDE, r_anthropic())
    lc.chat("s", [{"role": "system", "content": "extra rule"}, {"role": "user", "content": "a"},
                  {"role": "user", "content": "b"}, {"role": "user", "content": ""}])
    b = net.calls[0]["body"]
    assert b["system"] == "s\n\nextra rule"
    assert b["messages"] == [{"role": "user", "content": "a\n\nb"}]


# ---------------------------------------------------------------------------
# Timeouts / deadline
# ---------------------------------------------------------------------------
def test_per_call_timeout_is_25s(net):
    net.on(1, CLAUDE, r_anthropic())
    lc.chat("s", MSGS)
    t = net.calls[0]["timeout"]
    assert t["read"] == 25.0 and t["write"] == 25.0 and t["connect"] <= 10.0


def test_chain_deadline_stops_further_attempts(net, clock):
    def slow_fail(_req):
        clock.advance(30)
        return httpx.ReadTimeout("slow")
    net.on_all(slow_fail)
    with pytest.raises(lc.LLMChainError) as ei:
        lc.chat("s", MSGS)                # default 50 s deadline
    assert len(net.calls) == 2            # 30 s + 30 s > 50 s
    assert "deadline" in str(ei.value)
    assert ei.value.attempts[0] == "anthropic timeout"


def test_timeout_budget_shrinks_with_remaining_deadline(net, clock):
    def slow(_req):
        clock.advance(40)
        return err(500)
    net.on(1, CLAUDE, slow)
    net.on(2, G_MODELS[0], r_groq("ok"))
    lc.chat("s", MSGS)
    assert net.calls[1]["timeout"]["read"] == pytest.approx(10.0)


def test_custom_deadline_param(net, clock):
    def slow(_req):
        clock.advance(5)
        return err(500)
    net.on_all(slow)
    with pytest.raises(lc.LLMChainError):
        lc.chat("s", MSGS, deadline_s=8)
    assert len(net.calls) == 2


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------
def collect(gen):
    items = list(gen)
    return [i for i in items if isinstance(i, str)], items[-1], items


def starts(items):
    return [i for i in items if isinstance(i, lc.StreamStart)]


def test_stream_anthropic_deltas_then_result(net):
    net.on(1, CLAUDE, sse_anthropic(["Hel", "lo ", "world"]))
    deltas, final, items = collect(lc.stream_chat("s", MSGS))
    assert deltas == ["Hel", "lo ", "world"]
    assert isinstance(final, lc.ChatResult) and final.text == "Hello world"
    assert final.label == f"Model: anthropic · {CLAUDE} · step 1 (Anthropic)"
    assert final.usage["prompt_tokens"] == 9 and final.usage["completion_tokens"] == 4
    body = net.calls[0]["body"]
    assert body["stream"] is True and "temperature" not in body


def test_stream_start_precedes_first_delta_and_is_public(net):
    net.on(1, CLAUDE, sse_anthropic(["Hel", "lo"]))
    _, final, items = collect(lc.stream_chat("s", MSGS))
    (start,) = starts(items)
    assert items.index(start) == 0 and isinstance(items[1], str)
    assert start.label == f"Model: anthropic · {CLAUDE} · step 1 (Anthropic)" == final.label
    pub = start.to_public()
    assert pub["label"] == start.label and pub["step"] == 1 and pub["skipped"] == []
    assert "key" not in " ".join(map(str, pub.keys())).lower()


def test_stream_start_repeats_on_fallthrough_before_first_token(net, monkeypatch):
    only_keys(monkeypatch, "GROQ_API_KEY")
    net.on(2, G_MODELS[0], sse_groq([]))                  # 200 but no text at all -> next model
    net.on(2, G_MODELS[1], sse_groq(["hi"]))
    _, final, items = collect(lc.stream_chat("s", MSGS))
    labels = [x.label for x in starts(items)]
    assert labels[-1] == final.label and final.model == G_MODELS[1]
    assert len(set(labels)) == len(labels) and len(labels) >= 1


def test_stream_groq_openai_style(net, monkeypatch):
    only_keys(monkeypatch, "GROQ_API_KEY")
    net.on(2, G_MODELS[0], sse_groq(["a", "b", "c"]))
    deltas, final, _ = collect(lc.stream_chat("s", MSGS))
    assert deltas == ["a", "b", "c"] and final.text == "abc" and final.step == 2
    assert final.usage["total_tokens"] == 8
    b = net.calls[0]["body"]
    assert b["stream"] is True and "response_format" not in b


def test_stream_gemini_endpoint(net, monkeypatch):
    only_keys(monkeypatch, "GEMINI_API_KEY")
    net.on(4, GEM_MODELS[0], sse_gemini(["x", "y"]))
    deltas, final, _ = collect(lc.stream_chat("s", MSGS))
    assert deltas == ["x", "y"] and final.text == "xy" and final.provider == "gemini"
    assert net.calls[0]["url"] == ("https://generativelanguage.googleapis.com/v1beta/models/"
                                   f"{GEM_MODELS[0]}:streamGenerateContent?alt=sse")


def test_stream_falls_through_before_first_delta(net):
    net.on(1, CLAUDE, err(529, "overloaded"))
    net.on(2, G_MODELS[0], sse_groq([]))                              # empty stream
    net.on(2, G_MODELS[1], err(429, body=GROQ_TPD))
    net.on(2, G_MODELS[2], httpx.ReadTimeout("t"))
    net.on(3, G_MODELS[0], sse_groq(["ok"]))
    deltas, final, _ = collect(lc.stream_chat("s", MSGS))
    assert deltas == ["ok"] and final.step == 3
    assert final.skipped == ["anthropic 529", "groq#1 gpt-oss-120b empty", "groq#1 gpt-oss-20b 429",
                             "groq#1 qwen3.8-27b timeout"]
    assert lc.chain_status()["steps"][1]["models"][1]["exhausted"] is True


def test_stream_anthropic_refusal_and_error_event_fall_through(net):
    net.on(1, CLAUDE, sse_anthropic([], stop="refusal"))
    net.on(2, G_MODELS[0], sse(['data: {"error": {"message": "Overloaded", "status_code": 503}}']))
    net.on(2, G_MODELS[1], sse_groq(["fine"]))
    _, final, _ = collect(lc.stream_chat("s", MSGS))
    assert final.model == G_MODELS[1]
    assert final.skipped == ["anthropic refusal", "groq#1 gpt-oss-120b 503"]


def test_stream_anthropic_error_event(net, monkeypatch):
    only_keys(monkeypatch, "ANTHROPIC_API_KEY", "GROQ_API_KEY")
    net.on(1, CLAUDE, sse(["event: error", 'data: {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}']))
    net.on(2, G_MODELS[0], sse_groq(["x"]))
    _, final, _ = collect(lc.stream_chat("s", MSGS))
    assert final.skipped == ["anthropic 529"] and final.step == 2


def test_stream_no_fallthrough_after_first_delta(net):
    data = ("data: " + json.dumps({"choices": [{"delta": {"content": "partial"}}]}) + "\n").encode()
    net.on(1, CLAUDE, err(500))
    net.on(2, G_MODELS[0], Truncated(data))
    net.on(2, G_MODELS[1], sse_groq(["MUST NOT BE USED"]))
    deltas, final, _ = collect(lc.stream_chat("s", MSGS))
    assert deltas == ["partial"] and final.text == "partial"
    assert final.model == G_MODELS[0]
    assert "groq#1 gpt-oss-120b stream-neterr" in final.skipped
    assert (2, G_MODELS[1]) not in net.order


def test_stream_all_fail_raises(net):
    net.on_all(err(503))
    with pytest.raises(lc.LLMChainError) as ei:
        list(lc.stream_chat("s", MSGS))
    assert ei.value.label == "Model: none (all providers failed)"
    assert len(net.calls) == 13


def test_stream_sensitive_skips_gemini(net):
    net.on_all(err(503))
    with pytest.raises(lc.LLMChainError):
        list(lc.stream_chat("s", MSGS, sensitive=True))
    assert {c["step"] for c in net.calls} == {1, 2, 3}


def test_stream_401_marks_key_dead(net):
    net.on(2, G_MODELS[0], err(401, "nope"))
    net.on(1, CLAUDE, err(500))
    net.on(3, G_MODELS[0], sse_groq(["k2"]))
    _, final, _ = collect(lc.stream_chat("s", MSGS))
    assert final.step == 3 and (2, G_MODELS[1]) not in net.order


def test_stream_short_retry_after(net, monkeypatch):
    only_keys(monkeypatch, "GROQ_API_KEY")
    net.on(2, G_MODELS[0], [err(429, body=GROQ_TPM, headers={"retry-after": "1"}), sse_groq(["ok"])])
    deltas, final, _ = collect(lc.stream_chat("s", MSGS))
    assert net.sleeps == [1.0] and deltas == ["ok"] and final.model == G_MODELS[0]


def test_stream_abandoned_by_consumer_does_not_raise(net, monkeypatch):
    only_keys(monkeypatch, "GROQ_API_KEY")
    net.on(2, G_MODELS[0], sse_groq(["a", "b", "c"]))
    g = lc.stream_chat("s", MSGS)
    assert isinstance(next(g), lc.StreamStart)
    assert next(g) == "a"
    g.close()


# ---------------------------------------------------------------------------
# Async twins
# ---------------------------------------------------------------------------
def test_achat_fallthrough_and_label(net):
    net.on(1, CLAUDE, err(500))
    net.on(2, G_MODELS[0], httpx.ReadTimeout("t"))
    net.on(2, G_MODELS[1], r_groq("async ok"))
    r = asyncio.run(lc.achat("s", MSGS))
    assert r.text == "async ok" and r.skipped == ["anthropic 500", "groq#1 gpt-oss-120b timeout"]
    assert r.label == "Model: groq · openai/gpt-oss-20b · step 2 (Groq key 1)"


def test_achat_all_fail_and_new_event_loops(net):
    net.on_all(err(503))
    for _ in range(2):                    # two separate loops must both work (client re-created per loop)
        with pytest.raises(lc.LLMChainError):
            asyncio.run(lc.achat("s", MSGS))
    net.on(1, CLAUDE, r_anthropic("fine"))
    lc.reset_state()
    assert asyncio.run(lc.achat("s", MSGS, json_mode=True)).text == "fine"


def test_achat_sensitive_and_dead_key(net):
    net.on(1, CLAUDE, err(401))
    net.on_all(err(503), steps=(2, 3))
    with pytest.raises(lc.LLMChainError):
        asyncio.run(lc.achat("s", MSGS, sensitive=True))
    assert {c["step"] for c in net.calls} == {1, 2, 3}


def test_astream_chat(net):
    async def run():
        items = []
        async for i in lc.astream_chat("s", MSGS):
            items.append(i)
        return items
    net.on(1, CLAUDE, err(529))
    net.on(2, G_MODELS[0], sse_groq(["x", "y"]))
    items = asyncio.run(run())
    assert [i for i in items if isinstance(i, str)] == ["x", "y"]
    assert items[-1].text == "xy" and items[-1].skipped == ["anthropic 529"]
    (start,) = starts(items)
    assert items.index(start) == 0 and start.label == items[-1].label


# ---------------------------------------------------------------------------
# Status, secrets, concurrency, pooling
# ---------------------------------------------------------------------------
def test_chain_status_has_booleans_and_no_key_material(net, monkeypatch):
    only_keys(monkeypatch, "ANTHROPIC_API_KEY", "GEMINI_API_KEY_2")
    st = lc.chain_status()
    assert st["available"] is True and st["configured_steps"] == [1, 5]
    assert (st["anthropic"], st["groq_1"], st["groq_2"], st["gemini_1"], st["gemini_2"]) == (True, False, False, False, True)
    assert st["sensitive_capable"] is True
    assert st["policy"] == "auto: best available via chain"
    dumped = json.dumps(st)
    for v in KEYS.values():
        assert v not in dumped
    only_keys(monkeypatch, "GEMINI_API_KEY")
    assert lc.chain_status()["sensitive_capable"] is False
    only_keys(monkeypatch)
    assert lc.chain_status()["available"] is False


def test_keys_never_leak_into_logs_errors_or_labels(net, caplog):
    secret_echo = {"error": {"message": f"Invalid key {KEYS['GROQ_API_KEY']} Bearer {KEYS['GROQ_API_KEY']} AIzaSyABCDEFGHIJKLMNOPQRSTUV"}}
    net.on(2, G_MODELS[0], err(401, body=secret_echo))
    net.on_all(err(500), steps=(1, 3, 4, 5))
    net.on(2, G_MODELS[1], err(500))
    caplog.set_level("DEBUG", logger="backend.llm_chain")
    with pytest.raises(lc.LLMChainError) as ei:
        lc.chat("my secret prompt text", [{"role": "user", "content": "my secret question"}])
    blob = caplog.text + str(ei.value) + " ".join(ei.value.attempts)
    for v in KEYS.values():
        assert v not in blob
    assert "AIzaSyABCDEFGHIJKLMNOPQRSTUV" not in blob
    assert "my secret" not in blob
    assert "step=2" in caplog.text and "status=401" in caplog.text           # one line per attempt


def test_attempt_messages_are_truncated(net, monkeypatch):
    only_keys(monkeypatch, "GROQ_API_KEY")
    net.on_all(err(500, "x" * 5000), steps=(2,))
    with pytest.raises(lc.LLMChainError) as ei:
        lc.chat("s", MSGS)
    assert all(len(a) <= 230 for a in ei.value.attempts)


def test_concurrent_requests_are_safe(net):
    net.on(1, CLAUDE, err(500))
    net.on(2, G_MODELS[0], r_groq("ok"))
    with ThreadPoolExecutor(16) as ex:
        results = list(ex.map(lambda _: lc.chat("s", MSGS), range(48)))
    assert all(r.text == "ok" for r in results)
    assert len(net.calls) >= 48
    # circuit breaker opened for the failing anthropic model under concurrency (state stayed consistent)
    assert lc.chain_status()["steps"][0]["models"][0]["circuit_open"] is True
    assert len(net.calls_for(1)) < 48


def test_pooled_clients_are_shared():
    lc._configure_transports(None)
    try:
        assert lc._get_sync_client() is lc._get_sync_client()

        async def two():
            return lc._get_async_client(), lc._get_async_client()
        a, b = asyncio.run(two())
        assert a is b
    finally:
        lc._configure_transports(None)


def test_real_client_has_pool_limits_and_timeouts():
    lc._configure_transports(None)
    c = lc._get_sync_client()
    assert c.timeout.read == lc.CALL_TIMEOUT_S
    assert lc._LIMITS.max_keepalive_connections >= 10 and lc._LIMITS.keepalive_expiry >= 30
    lc._configure_transports(None)


def test_no_retries_configured_on_transport():
    lc._configure_transports(None)
    c = lc._get_sync_client()
    t = c._transport
    assert getattr(getattr(t, "_pool", None), "_retries", 0) == 0
    lc._configure_transports(None)


# ---------------------------------------------------------------------------
# config.py integration (chain-aware helpers)
# ---------------------------------------------------------------------------
def test_config_env_names_match_chain():
    from backend import config
    assert config.CHAIN_KEY_VARS == lc.KEY_ENV_VARS
    assert [n for n, _, _ in config.CHAIN_KEY_ENV] == [1, 2, 3, 4, 5]


def test_config_valid_with_any_single_key(monkeypatch):
    from backend import config
    for k in KEYS:
        monkeypatch.delenv(k, raising=False)
    assert config.has_groq_key() is False and config.has_llm_key() is False
    with pytest.raises(config.MissingApiKeyError) as ei:
        config.require_groq_key()
    assert "ANTHROPIC_API_KEY" in str(ei.value) and "GEMINI_API_KEY_2" in str(ei.value)
    monkeypatch.setenv("GEMINI_API_KEY_2", "x-test")
    assert config.has_groq_key() is True
    config.require_groq_key()
    assert config.has_gemini_key() is True


def test_config_validate_does_not_require_groq_specifically(monkeypatch):
    from backend import config
    for name, val in {"APP_ENV": "development", "HOST": "127.0.0.1", "PORT": 8000,
                      "LLM_PROVIDER": "chain", "EMBEDDING_PROVIDER": "gemini",
                      "EMBEDDING_MODEL": "gemini-embedding-2", "DEFAULT_MODEL": None,
                      "AVAILABLE_MODELS": [], "_PORT_OK": True, "MONGODB_URI": None,
                      "CLERK_PUBLISHABLE_KEY": None, "SPLIT_ORIGIN": False, "_FRONTEND_PORT_OK": True,
                      "TOOL_MODEL_OVERRIDES": {}}.items():
        monkeypatch.setattr(config, name, val)
    for k in KEYS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.delenv("GEMINI_API_KEY_FALLBACK", raising=False)
    with pytest.raises(config.ConfigError) as ei:
        config.validate_config()
    assert "ANTHROPIC_API_KEY" in str(ei.value)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "only-this-one")
    monkeypatch.setenv("GEMINI_API_KEY", "emb-key")
    config.validate_config()                               # no GROQ_API_KEY needed


def test_public_config_exposes_booleans_and_policy_not_keys(monkeypatch):
    from backend import config
    for k in KEYS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("GROQ_API_KEY", KEYS["GROQ_API_KEY"])
    pc = config.public_config()
    assert pc["llmChain"] == {"anthropic": False, "groq1": True, "groq2": False, "gemini1": False, "gemini2": False}
    assert pc["groqKey"] is True and pc["llmAvailable"] is True
    assert pc["modelPolicy"] == "auto: best available via chain"
    assert KEYS["GROQ_API_KEY"] not in json.dumps(pc)
    assert KEYS["GROQ_API_KEY"] not in "\n".join(config.startup_diagnostics())
    assert config.chain_status()["groq_1"] is True


def test_warm_makes_no_network_calls(net):
    lc._configure_transports(None)
    lc.warm()
    assert lc._sync_client is not None and not lc._sync_client.is_closed
    assert lc._get_sync_client() is lc._sync_client


def test_config_serverless_and_relaxed_validation(monkeypatch):
    from backend import config
    monkeypatch.delenv("VERCEL", raising=False)
    monkeypatch.delenv("VERCEL_ENV", raising=False)
    assert config.is_serverless() is False
    monkeypatch.setenv("VERCEL", "1")
    assert config.is_serverless() is True
    for name, val in {"APP_ENV": "production", "HOST": None, "PORT": None, "_PORT_OK": True,
                      "LLM_PROVIDER": "chain", "EMBEDDING_PROVIDER": "gemini", "EMBEDDING_MODEL": "gemini-embedding-2",
                      "DEFAULT_MODEL": None, "AVAILABLE_MODELS": [], "MONGODB_URI": None,
                      "CLERK_PUBLISHABLE_KEY": None, "SPLIT_ORIGIN": True, "FRONTEND_PORT": 5173,
                      "API_BASE_URL": None, "_FRONTEND_PORT_OK": True, "TOOL_MODEL_OVERRIDES": {}}.items():
        monkeypatch.setattr(config, name, val)
    monkeypatch.setenv("GROQ_API_KEY", "x-test")
    monkeypatch.setenv("GEMINI_API_KEY", "x-test")
    config.validate_config()            # no HOST/PORT/API_BASE_URL needed


def test_config_embedding_defaults():
    from backend import config
    assert config.EMBEDDING_PROVIDER in ("gemini", "local")
    src = open(config.__file__).read()
    assert '"gemini-embedding-2" if EMBEDDING_PROVIDER == "gemini"' in src


# ---------------------------------------------------------------------------
# Image input
# ---------------------------------------------------------------------------
import base64 as _b64
import inspect

IMG1 = (b"\x89PNG\r\n\x1a\nIMG-ONE", "image/png")
IMG2 = (b"\xff\xd8\xffIMG-TWO", "image/jpeg")
IMGS = [IMG1, IMG2, (b"three", "image/webp"), (b"four", "image/png")]
B64 = lambda b: _b64.b64encode(b).decode()  # noqa: E731


def test_signatures_expose_images():
    for fn in (lc.chat, lc.achat, lc.stream_chat, lc.astream_chat):
        assert "images" in inspect.signature(fn).parameters


def test_anthropic_image_payload(net):
    net.on(1, CLAUDE, r_anthropic("sees it"))
    lc.chat("s", [{"role": "user", "content": "earlier"}, {"role": "assistant", "content": "a"},
                  {"role": "user", "content": "what is this?"}], images=[IMG1, IMG2])
    msgs = net.calls[0]["body"]["messages"]
    assert msgs[0] == {"role": "user", "content": "earlier"}            # only the LAST user message changes
    blocks = msgs[-1]["content"]
    assert blocks[0] == {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": B64(IMG1[0])}}
    assert blocks[1]["source"]["media_type"] == "image/jpeg"
    assert blocks[-1] == {"type": "text", "text": "what is this?"}
    assert "temperature" not in net.calls[0]["body"]


def test_groq_images_only_qwen_max_three(net, monkeypatch):
    only_keys(monkeypatch, "GROQ_API_KEY", "GROQ_API_KEY_2")
    net.on(2, G_MODELS[2], err(500))
    net.on(3, G_MODELS[2], r_groq("qwen sees"))
    r = lc.chat("s", MSGS, images=IMGS)
    assert [c["model"] for c in net.calls] == [G_MODELS[2], G_MODELS[2]]       # gpt-oss never called
    body = net.calls[0]["body"]
    content = body["messages"][-1]["content"]
    assert content[0] == {"type": "text", "text": "hello"}
    urls = [c["image_url"]["url"] for c in content[1:]]
    assert len(urls) == 3 and urls[0] == f"data:image/png;base64,{B64(IMG1[0])}"
    assert r.model == G_MODELS[2] and r.step == 3
    assert r.skipped[:2] == ["groq#1 gpt-oss-120b no-vision", "groq#1 gpt-oss-20b no-vision"]
    assert "groq#1 qwen3.8-27b 500" in r.skipped


def test_gemini_inline_data_for_all_models_and_chain_order(net):
    net.on(1, CLAUDE, err(500))
    net.on_all(err(500), steps=(2, 3))
    net.on(4, GEM_MODELS[0], err(503))
    net.on(4, GEM_MODELS[1], err(503))
    net.on(4, GEM_MODELS[2], r_gemini("flash-lite sees"))
    r = lc.chat("s", MSGS, images=[IMG1, IMG2])
    assert [c["model"] for c in net.calls if c["step"] in (2, 3)] == [G_MODELS[2], G_MODELS[2]]
    assert [c["step"] for c in net.calls][0] == 1 and r.step == 4
    for c in net.calls_for(4):
        parts = c["body"]["contents"][-1]["parts"]
        assert parts[0] == {"text": "hello"}
        assert parts[1] == {"inlineData": {"mimeType": "image/png", "data": B64(IMG1[0])}}
        assert parts[2]["inlineData"]["mimeType"] == "image/jpeg"


def test_no_images_leaves_payloads_plain(net, monkeypatch):
    only_keys(monkeypatch, "GROQ_API_KEY")
    net.on(2, G_MODELS[0], r_groq())
    lc.chat("s", MSGS)
    assert net.calls[0]["body"]["messages"][-1] == {"role": "user", "content": "hello"}


def test_image_validation(net):
    with pytest.raises(ValueError):
        lc.chat("s", MSGS, images=[(b"x" * (5 * 1024 * 1024 + 1), "image/png")])
    with pytest.raises(ValueError):
        lc.chat("s", MSGS, images=[(b"abc", "text/plain")])
    with pytest.raises(ValueError):
        lc.chat("s", MSGS, images=[(b"", "image/png")])
    assert net.calls == []
    net.on(1, CLAUDE, r_anthropic())
    lc.chat("s", MSGS, images=[(b"x" * (5 * 1024 * 1024), "image/png")])      # exactly 5 MB is OK


def test_image_bytes_never_logged(net, caplog):
    net.on_all(err(500))
    caplog.set_level("DEBUG", logger="backend.llm_chain")
    with pytest.raises(lc.LLMChainError) as ei:
        lc.chat("s", MSGS, images=[IMG1])
    blob = caplog.text + str(ei.value) + " ".join(ei.value.attempts)
    assert B64(IMG1[0]) not in blob and "IMG-ONE" not in blob


def test_achat_and_streams_take_images(net, monkeypatch):
    net.on(1, CLAUDE, r_anthropic("async sees"))
    assert asyncio.run(lc.achat("s", MSGS, images=[IMG1])).text == "async sees"
    assert net.calls[0]["body"]["messages"][-1]["content"][0]["type"] == "image"
    net.calls.clear()
    net.on(1, CLAUDE, sse_anthropic(["ok"]))
    deltas, final, _ = collect(lc.stream_chat("s", MSGS, images=[IMG1]))
    assert final.text == "ok" and net.calls[0]["body"]["messages"][-1]["content"][0]["type"] == "image"
    net.calls.clear()
    only_keys(monkeypatch, "GEMINI_API_KEY")
    net.on(4, GEM_MODELS[0], sse_gemini(["g"]))

    async def run():
        return [i async for i in lc.astream_chat("s", MSGS, images=[IMG2])]
    items = asyncio.run(run())
    assert items[-1].text == "g" and "inlineData" in net.calls[0]["body"]["contents"][-1]["parts"][1]


def test_public_config_lists_chain_order_without_keys(monkeypatch):
    from backend import config
    for k, v in KEYS.items():
        monkeypatch.setenv(k, v)
    order = config.public_config()["llmChainOrder"]
    assert [(o["step"], o["name"]) for o in order] == [(s.n, s.name) for s in lc.STEPS]
    assert order[0]["models"] == [CLAUDE] and order[1]["models"] == list(G_MODELS)
    blob = json.dumps(order)
    assert not any(v in blob for v in KEYS.values())


# ---------------------------------------------------------------------------
# Provider-aware input fitting / output allowance
# ---------------------------------------------------------------------------
def _big(n_chars, head="HEADMARK ", tail=" TAILMARK"):
    return head + ("lorem ipsum " * (n_chars // 12)) + tail


def test_anthropic_gets_full_input_and_model_max_output_regardless_of_caller(net):
    big = _big(400_000)                         # ~111k est. tokens: far over Groq budget, under the 120k paid-step request cap
    net.on(1, CLAUDE, r_anthropic("ok"))
    r = lc.chat("sys", [{"role": "user", "content": big}], max_tokens=77)
    b = net.calls[0]["body"]
    assert b["messages"][0]["content"] == big   # byte-for-byte, no product trimming
    assert b["max_tokens"] == 64000 and r.trimmed == 0


def test_anthropic_max_output_env_override(net, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_MAX_OUTPUT_TOKENS", "32000")
    net.on(1, CLAUDE, r_anthropic("ok"))
    lc.chat("s", MSGS, max_tokens=10)
    assert net.calls[0]["body"]["max_tokens"] == 32000


def test_fallthrough_anthropic_fail_groq_gets_trimmed_copy_original_untouched(net):
    big = _big(100_000)                          # ~27k tokens
    msgs = [{"role": "user", "content": big}]
    net.on(1, CLAUDE, err(529))
    net.on(2, G_MODELS[0], r_groq("g"))
    r = lc.chat("be nice", msgs, max_tokens=500)
    assert net.calls[0]["body"]["messages"][0]["content"] == big      # Anthropic saw everything
    gm = net.calls[1]["body"]["messages"]
    sent_user = gm[-1]["content"]
    assert len(sent_user) < len(big) and "[...trimmed...]" in sent_user
    assert sent_user.startswith("HEADMARK") and sent_user.endswith("TAILMARK")
    assert gm[0] == {"role": "system", "content": "be nice"}
    total = sum(len(m["content"]) for m in gm) / lc.CHARS_PER_TOKEN
    assert total <= lc.input_budget("groq") + 50
    assert msgs == [{"role": "user", "content": big}]                  # caller's list unmodified
    assert r.trimmed > 0 and net.calls[1]["body"]["max_completion_tokens"] == 2048


def test_groq_trims_largest_block_only_and_keeps_final_question(net, monkeypatch):
    monkeypatch.setenv("GROQ_MAX_INPUT_TOKENS", "1000")
    only_keys(monkeypatch, "GROQ_API_KEY")
    msgs = [{"role": "user", "content": "first q"}, {"role": "assistant", "content": _big(60_000)},
            {"role": "user", "content": "FINAL QUESTION?"}]
    net.on(2, G_MODELS[0], r_groq("g"))
    lc.chat("sys", msgs)
    gm = net.calls[0]["body"]["messages"]
    assert gm[1]["content"] == "first q" and gm[-1]["content"] == "FINAL QUESTION?"
    assert "[...trimmed...]" in gm[2]["content"]


def test_backstops_trim_minimally_and_zero_disables(net, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_MAX_INPUT_TOKENS", "2000")
    big = _big(50_000)
    net.on(1, CLAUDE, r_anthropic("ok"))
    r = lc.chat("s", [{"role": "user", "content": big}])
    sent = net.calls[0]["body"]["messages"][0]["content"]
    assert "[...trimmed...]" in sent and len(sent) / lc.CHARS_PER_TOKEN <= 2000 + 50 and r.trimmed > 0
    monkeypatch.setenv("ANTHROPIC_MAX_INPUT_TOKENS", "0")
    lc.chat("s", [{"role": "user", "content": big}])
    assert net.calls[1]["body"]["messages"][0]["content"] == big


def test_gemini_backstop_and_output_clamp(net, monkeypatch):
    only_keys(monkeypatch, "GEMINI_API_KEY")
    monkeypatch.setenv("GEMINI_MAX_INPUT_TOKENS", "1500")
    net.on(4, GEM_MODELS[0], r_gemini("ok"))
    lc.chat("s", [{"role": "user", "content": _big(40_000)}], max_tokens=500_000)
    b = net.calls[0]["body"]
    assert "[...trimmed...]" in b["contents"][0]["parts"][0]["text"]
    assert b["generationConfig"]["maxOutputTokens"] == 65536


def test_groq_output_clamped_per_model_and_floor(net, monkeypatch):
    only_keys(monkeypatch, "GROQ_API_KEY")
    net.on(2, G_MODELS[0], err(500))
    net.on(2, G_MODELS[1], err(500))
    net.on(2, G_MODELS[2], err(500))
    with pytest.raises(lc.LLMChainError):
        lc.chat("s", MSGS, max_tokens=500_000)
    by = {c["model"]: c["body"]["max_completion_tokens"] for c in net.calls}
    assert by[G_MODELS[0]] == 65536 and by[G_MODELS[2]] == 16384
    net.calls.clear()
    lc.reset_state()
    with pytest.raises(lc.LLMChainError):
        lc.chat("s", MSGS, max_tokens=10)
    assert {c["body"]["max_completion_tokens"] for c in net.calls} == {2048}


def test_images_are_unaffected_by_trimming(net, monkeypatch):
    only_keys(monkeypatch, "GROQ_API_KEY")
    monkeypatch.setenv("GROQ_MAX_INPUT_TOKENS", "500")
    net.on(2, GROQ_VISION := lc.GROQ_VISION_MODEL, r_groq("seen"))
    lc.chat("s", [{"role": "user", "content": _big(30_000)}], images=[(b"\x89PNGdata", "image/png")])
    parts = net.calls[0]["body"]["messages"][-1]["content"]
    assert any(p.get("type") == "image_url" for p in parts)
    assert "[...trimmed...]" in parts[0]["text"]


def test_stream_and_async_paths_also_fit_per_provider(net):
    big = _big(100_000)
    net.on(1, CLAUDE, err(529))
    net.on(2, G_MODELS[0], sse_groq(["a"]))
    items = list(lc.stream_chat("s", [{"role": "user", "content": big}]))
    assert isinstance(items[-1], lc.ChatResult) and items[-1].trimmed > 0
    assert net.calls[0]["body"]["messages"][0]["content"] == big
    assert "[...trimmed...]" in net.calls[1]["body"]["messages"][-1]["content"]
    net.calls.clear()
    net.on(1, CLAUDE, r_anthropic("ok"))
    r = asyncio.run(lc.achat("s", [{"role": "user", "content": big}], max_tokens=5))
    assert net.calls[0]["body"]["messages"][0]["content"] == big and r.trimmed == 0


def test_usage_totals_counted(net):
    before = lc.usage_totals().get("anthropic", {}).get("prompt_tokens", 0)
    net.on(1, CLAUDE, r_anthropic("ok", usage=(100, 5)))
    lc.chat("s", MSGS)
    t = lc.usage_totals()["anthropic"]
    assert t["prompt_tokens"] - before == 100 and "usage_totals" in lc.chain_status()
