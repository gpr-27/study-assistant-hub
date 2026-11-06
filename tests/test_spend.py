"""Spend guards on the paid Anthropic step: ledger, budgets, fall-through, fail-open, ops cost."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest import mock

import pytest
from fastapi import Request
from fastapi.testclient import TestClient

from backend import cache, config, db, llm_chain as lc, spend
from backend.auth import current_owner
from backend.main import app
from backend.services import chat as chat_service
from tests.test_llm_chain import (CLAUDE, G_MODELS, KEYS, MSGS, err, net, only_keys, r_anthropic,  # noqa: F401
                                  r_groq, sse, sse_anthropic, sse_groq)

@pytest.fixture(autouse=True)
def _explicit_test_budgets(monkeypatch):
    """The shipped defaults are 0 (unlimited); these tests exercise the guard logic, so pin explicit limits.
    Individual tests override these with their own monkeypatch/env when they need other values."""
    monkeypatch.setenv("ANTHROPIC_DAILY_TOKEN_BUDGET", "2000000")
    monkeypatch.setenv("ANTHROPIC_OWNER_DAILY_TOKENS", "400000")
    monkeypatch.setenv("ANTHROPIC_GUEST_DAILY_TOKENS", "100000")
    monkeypatch.setenv("ANTHROPIC_MAX_REQUEST_INPUT_TOKENS", "120000")


DAY1 = datetime(2026, 11, 6, 10, 0, tzinfo=timezone.utc).timestamp()
ALICE, BOB = "user_alice", "user_bob"
GUEST = "guest_" + "a" * 32


class Clock:
    def __init__(self, t=DAY1):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, s):
        self.t += s


class FakeStore:
    """One shared 'database' for several Ledger instances."""

    name = "fake"

    def __init__(self):
        self.data: dict[tuple[str, str], int] = {}
        self.reads = self.writes = 0
        self.fail = False

    def incr(self, day, deltas):
        if self.fail:
            raise RuntimeError("store down")
        self.writes += 1
        for sc, n in deltas.items():
            self.data[(day, sc)] = self.data.get((day, sc), 0) + n

    def read(self, day, scopes):
        if self.fail:
            raise RuntimeError("store down")
        self.reads += 1
        return {s: self.data.get((day, s), 0) for s in scopes}


@pytest.fixture()
def clock():
    return Clock()


@pytest.fixture()
def store():
    return FakeStore()


@pytest.fixture()
def led(store, clock):
    old = spend._ledger
    new = spend.reset_ledger(spend.Ledger(store=store, clock=clock, background=False))
    yield new
    spend._ledger = old


@pytest.fixture(autouse=True)
def _budgets(monkeypatch):
    for k in ("ANTHROPIC_DAILY_TOKEN_BUDGET", "ANTHROPIC_OWNER_DAILY_TOKENS", "ANTHROPIC_GUEST_DAILY_TOKENS",
              "ANTHROPIC_MAX_REQUEST_INPUT_TOKENS"):
        monkeypatch.delenv(k, raising=False)


def _scope(owner):
    return "owner:" + cache.owner_tag(owner)


# --------------------------------------------------------------------------------------------------
# Ledger: accumulation, flush, sharing, reset
# --------------------------------------------------------------------------------------------------
def test_ledger_accumulates_and_flushes_inc_upserts_to_mongo(clock, monkeypatch):
    calls = []

    class Coll:
        def bulk_write(self, ops, ordered=True):
            calls.append(ops)

    led = spend.Ledger(clock=clock, background=False)
    with mock.patch.object(db, "_get_db", return_value={spend.COLLECTION: Coll()}):
        led.add(ALICE, 1000)
        led.add(ALICE, 500)
        led.add(None, 200)                       # unattributed: global only
        assert calls == []                       # nothing written yet (20 s / 5% not reached)
        assert led.flush() is True
    assert len(calls) == 1 and len(calls[0]) == 2          # ONE round trip: global + one owner
    docs = {op._filter["_id"]: op._doc for op in calls[0]}
    g = docs["20261106:global"]
    assert g["$inc"] == {"tokens": 1700} and g["$setOnInsert"]["day"] == "20261106"
    assert g["$setOnInsert"]["expires_at"] == datetime(2026, 11, 9, tzinfo=timezone.utc)      # TTL 3 days
    o = docs[f"20261106:{_scope(ALICE)}"]
    assert o["$inc"] == {"tokens": 1500}
    assert all(op._upsert for op in calls[0])
    assert ALICE not in json.dumps([str(d) for d in docs])          # owner id never stored in clear


def test_flush_after_20s_and_at_5pct_of_budget(led, store, clock, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_DAILY_TOKEN_BUDGET", "1000000")
    led.add(ALICE, 1000)
    assert store.writes == 0
    clock.advance(21)
    led.add(ALICE, 1000)                          # >= 20 s since the last flush
    assert store.writes == 1
    led.add(ALICE, 50_000)                        # 5% of 1,000,000 pending -> immediate
    assert store.writes == 2
    assert store.data[(spend.utc_day(clock()), "global")] == 52_000


def test_cross_instance_sharing_over_one_store(store, clock, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_DAILY_TOKEN_BUDGET", "1000")
    a = spend.Ledger(store=store, clock=clock, background=False)
    b = spend.Ledger(store=store, clock=clock, background=False)
    assert b.check(ALICE, 10) is None
    a.add(ALICE, 995)
    a.flush()
    assert b.check(ALICE, 10) is None             # B's view is cached for ~30 s (bounded staleness)
    clock.advance(31)
    assert b.check(BOB, 10) == "global"           # now B sees A's usage (any owner)
    assert store.reads == 2


def test_budget_resets_at_utc_midnight(led, clock, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_DAILY_TOKEN_BUDGET", "1000")
    clock.t = datetime(2026, 11, 6, 23, 59, 30, tzinfo=timezone.utc).timestamp()
    led.add(ALICE, 2000)
    assert led.check(ALICE, 1) == "global"
    clock.advance(60)                              # 00:00:30 next UTC day
    assert led.check(ALICE, 1) is None
    assert led.status()["global_used"] == 0


def test_zero_means_unlimited_and_touches_no_store(led, store, clock, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_DAILY_TOKEN_BUDGET", "0")
    monkeypatch.setenv("ANTHROPIC_OWNER_DAILY_TOKENS", "0")
    monkeypatch.setenv("ANTHROPIC_GUEST_DAILY_TOKENS", "0")
    led.add(ALICE, 10**9)
    clock.advance(60)
    led.add(GUEST, 10**9)
    assert led.check(ALICE, 10**7) is None and led.check(GUEST, 10**7) is None
    assert (store.reads, store.writes) == (0, 0)


def test_guest_and_signed_in_limits_and_owner_isolation(led, monkeypatch):
    assert (led.owner_limit(GUEST), led.owner_limit(ALICE)) == (100_000, 400_000)
    led.add(GUEST, 120_000)
    led.add(ALICE, 120_000)
    assert led.check(GUEST, 1) == "owner"          # guest spent > 100k
    assert led.check(ALICE, 1) is None             # same tokens are fine for a signed-in user
    assert led.check(BOB, 1) is None               # other owners untouched
    led.add(ALICE, 300_000)
    assert led.check(ALICE, 1) == "owner" and led.check(BOB, 1) is None
    monkeypatch.setenv("ANTHROPIC_GUEST_DAILY_TOKENS", "500000")
    assert led.check(GUEST, 1) is None             # env is read live


def test_unattributed_calls_count_only_against_global(led, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_DAILY_TOKEN_BUDGET", "1000")
    led.add(None, 2000)
    assert led.check(None, 1) == "global"
    assert led.status()["owners_tracked"] == 0


def test_check_includes_estimated_prompt_before_the_call(led, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_OWNER_DAILY_TOKENS", "10000")
    led.add(ALICE, 9000)
    assert led.check(ALICE, 500) is None
    assert led.check(ALICE, 1500) == "owner"       # would cross the cap


# --------------------------------------------------------------------------------------------------
# Fail OPEN on store failure, but per-instance counters still enforce
# --------------------------------------------------------------------------------------------------
def test_store_failure_fails_open_but_local_limits_still_apply(led, store, clock, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_DAILY_TOKEN_BUDGET", "10000")
    store.fail = True
    assert led.check(ALICE, 100) is None           # read fails -> allowed, no exception
    led.add(ALICE, 6000)
    clock.advance(21)
    led.add(ALICE, 6000)                           # a flush is attempted, fails; tokens stay pending
    assert led.stats["flush_fail"] == 2           # at most one retry per 20 s (no storm)
    assert led.check(ALICE, 100) == "global"       # per-instance counter alone trips the global budget
    assert led.flush() is False
    store.fail = False
    clock.advance(40)
    assert led.flush() is True                     # retried later: nothing was lost
    assert store.data[(spend.utc_day(clock()), "global")] == 12000


def test_no_store_configured_is_silent(clock):
    led = spend.Ledger(clock=clock, background=False)
    with mock.patch.object(db, "_get_db", return_value=None):
        led.add(ALICE, 500)
        assert led.flush() is True
        assert led.check(ALICE, 1) is None
    assert led.status()["global_used"] == 500


# --------------------------------------------------------------------------------------------------
# Ops cost (Mongo) and the Redis path
# --------------------------------------------------------------------------------------------------
class CountingColl:
    def __init__(self):
        self.finds, self.bulks, self.docs = [], [], {}

    def find(self, flt, proj=None):
        self.finds.append((flt, proj))
        return iter([{"_id": i, "tokens": self.docs[i]} for i in flt["_id"]["$in"] if i in self.docs])

    def bulk_write(self, ops, ordered=True):
        self.bulks.append(len(ops))
        for op in ops:
            i = op._filter["_id"]
            self.docs[i] = self.docs.get(i, 0) + op._doc["$inc"]["tokens"]


def test_mongo_ops_per_session_is_one_projected_read_per_30s_and_one_bulk_write_per_flush(clock):
    coll = CountingColl()
    led = spend.Ledger(clock=clock, background=False)
    with mock.patch.object(db, "_get_db", return_value={spend.COLLECTION: coll}):
        for _ in range(12):                         # a chatty session: 12 paid calls inside 30 s
            assert led.check(ALICE, 500) is None
            led.add(ALICE, 800)
            clock.advance(2)
        assert len(coll.finds) == 1                # ONE read, projected, for global + owner together
        flt, proj = coll.finds[0]
        assert proj == {"tokens": 1} and len(flt["_id"]["$in"]) == 2
        assert len(coll.bulks) == 1 and coll.bulks[0] == 2      # flush at ~20 s: one bulk_write, two scopes
        clock.advance(31)
        led.check(ALICE, 500)
        assert len(coll.finds) == 2                # next read only after the 30 s cache expires


def test_redis_backend_replaces_mongo(clock, monkeypatch):
    be = cache.MemoryBackend()
    cache.configure(be, True)
    try:
        monkeypatch.setenv("ANTHROPIC_DAILY_TOKEN_BUDGET", "1000")
        boom = mock.Mock(side_effect=AssertionError("Mongo must not be used when Redis is enabled"))
        with mock.patch.object(db, "_get_db", boom):
            a = spend.Ledger(clock=clock, background=False)
            b = spend.Ledger(clock=clock, background=False)
            a.add(ALICE, 990)
            assert a.flush() is True
            assert b.check(BOB, 50) == "global"       # shared through INCRBY/MGET
        keys = list(be._d)
        assert any(k.startswith("sah:v1:spend:20261106:") for k in keys)
        assert ALICE not in "".join(keys)
        ent = be._d[f"sah:v1:spend:20261106:global"]
        assert ent[0] is not None                      # day-keyed TTL
    finally:
        cache.reset()


# --------------------------------------------------------------------------------------------------
# llm_chain integration
# --------------------------------------------------------------------------------------------------
def test_exhausted_global_budget_skips_anthropic_and_falls_through(net, led, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_DAILY_TOKEN_BUDGET", "5000")
    led.add(None, 6000)
    net.on(2, G_MODELS[0], r_groq("free answer"))
    r = lc.chat("sys", MSGS, owner=ALICE)
    assert (r.provider, r.step, r.text) == ("groq", 2, "free answer")
    assert "anthropic budget" in r.skipped
    assert not net.calls_for(1)                       # the paid provider was never called
    assert led.status()["anthropic_skipped_today"] == 1


def test_per_owner_budget_is_isolated(net, led, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_OWNER_DAILY_TOKENS", "5000")
    net.on(1, CLAUDE, r_anthropic("premium", usage=(100, 50)))
    net.on(2, G_MODELS[0], r_groq("free"))
    led.add(ALICE, 9000)
    assert lc.chat("s", MSGS, owner=ALICE).provider == "groq"
    r = lc.chat("s", MSGS, owner=BOB)
    assert r.provider == "anthropic" and r.text == "premium"
    assert led.status()["global_used"] == 9000 + 150          # actual usage recorded for BOB
    assert led.check(BOB, 1) is None


def test_owner_comes_from_the_contextvar_and_actual_usage_is_billed(net, led):
    net.on(1, CLAUDE, r_anthropic("hi", usage=(300, 200)))
    with spend.owner_scope(ALICE):
        lc.chat("s", MSGS)
    assert led._local[_scope(ALICE)] == 500 and led._local["global"] == 500
    lc.chat("s", MSGS)                                         # no owner anywhere -> global only
    assert led._local[_scope(ALICE)] == 500 and led._local["global"] == 1000


def test_missing_usage_is_estimated(net, led):
    net.on(1, CLAUDE, httpx_json({"content": [{"type": "text", "text": "x" * 360}], "stop_reason": "end_turn"}))
    lc.chat("s" * 360, MSGS, owner=ALICE)
    assert led._local["global"] >= 100 + 100                  # est(prompt) + est(completion)


def httpx_json(body):
    import httpx
    return httpx.Response(200, json=body)


def test_streaming_usage_is_parsed_from_message_start_and_delta(net, led):
    net.on(1, CLAUDE, sse_anthropic(["he", "llo"]))            # message_start in=9, message_delta out=4
    items = list(lc.stream_chat("s", MSGS, owner=ALICE))
    assert items[-1].usage["prompt_tokens"] == 9 and items[-1].usage["completion_tokens"] == 4
    assert led._local["global"] == 13 and led._local[_scope(ALICE)] == 13


def test_streaming_without_usage_is_estimated(net, led):
    body = sse(['data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"abcdefghij"}}',
                'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"}}',
                'data: {"type":"message_stop"}'])
    net.on(1, CLAUDE, body)
    list(lc.stream_chat("s", MSGS, owner=ALICE))
    assert led._local["global"] > 0


def test_oversize_request_skips_anthropic_and_goes_to_groq_trimmed(net, led, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_MAX_REQUEST_INPUT_TOKENS", "1000")
    net.on(2, G_MODELS[0], r_groq("short"))
    r = lc.chat("s", [{"role": "user", "content": "word " * 5000}], owner=ALICE)
    assert r.provider == "groq" and "anthropic too-large" in r.skipped
    assert not net.calls_for(1)


def test_oversize_with_only_private_models_returns_clear_413(net, led, monkeypatch):
    only_keys(monkeypatch, "ANTHROPIC_API_KEY", "GEMINI_API_KEY")
    monkeypatch.setenv("ANTHROPIC_MAX_REQUEST_INPUT_TOKENS", "1000")
    with pytest.raises(lc.LLMRequestTooLarge) as ei:
        lc.chat("s", [{"role": "user", "content": "word " * 5000}], sensitive=True, owner=ALICE)
    assert ei.value.status == 413 and "too large for the available models" in str(ei.value)
    assert not net.calls
    # non-sensitive content may still use Gemini (1M context)
    net.on(4, lc.GEMINI_MODELS[0], r_groq_like_gemini())
    r = lc.chat("s", [{"role": "user", "content": "word " * 5000}], sensitive=False, owner=ALICE)
    assert r.provider == "gemini"


def r_groq_like_gemini():
    from tests.test_llm_chain import r_gemini
    return r_gemini("from gemini")


def test_request_cap_zero_is_unlimited(net, led, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_MAX_REQUEST_INPUT_TOKENS", "0")
    net.on(1, CLAUDE, r_anthropic("ok"))
    assert lc.chat("s", [{"role": "user", "content": "word " * 60000}], owner=ALICE).provider == "anthropic"


def test_chain_fails_open_when_the_ledger_store_is_down(net, led, store, monkeypatch):
    store.fail = True
    net.on(1, CLAUDE, r_anthropic("still premium"))
    assert lc.chat("s", MSGS, owner=ALICE).text == "still premium"


def test_async_paths_gate_and_bill(net, led, monkeypatch):
    import asyncio
    asyncio.run(_async_paths(net, led, monkeypatch))


async def _async_paths(net, led, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_OWNER_DAILY_TOKENS", "5000")
    net.on(1, CLAUDE, r_anthropic("a", usage=(4000, 2000)))
    net.on(2, G_MODELS[0], r_groq("free"))
    assert (await lc.achat("s", MSGS, owner=ALICE)).provider == "anthropic"
    r = await lc.achat("s", MSGS, owner=ALICE)
    assert r.provider == "groq" and "anthropic budget" in r.skipped


# --------------------------------------------------------------------------------------------------
# Observability + config surface
# --------------------------------------------------------------------------------------------------
def test_status_and_80pct_warning_and_one_trip_log(led, monkeypatch, caplog):
    monkeypatch.setenv("ANTHROPIC_DAILY_TOKEN_BUDGET", "1000")
    led.add(ALICE, 850)
    st = led.status()
    assert st["warn"] is True and st["exhausted"] is False and st["pct"] == 85.0
    led.add(ALICE, 300)
    with caplog.at_level("INFO", logger="backend.spend"):
        led.check(ALICE, 1)
        led.check(BOB, 1)
        led.check(ALICE, 1)
    trips = [r for r in caplog.records if "budget tripped" in r.getMessage()]
    assert len(trips) == 1                                     # one INFO line the first time per day
    assert ALICE not in caplog.text and _scope(ALICE) not in caplog.text
    st = led.status()
    assert st["exhausted"] and st["anthropic_skipped_today"] == 3 and st["owners_tracked"] == 1


def test_health_deep_budgets_has_no_owner_ids(led, monkeypatch):
    monkeypatch.setattr(config, "PERSISTENCE_ENABLED", False)
    monkeypatch.setenv("ANTHROPIC_DAILY_TOKEN_BUDGET", "1000")
    led.add(ALICE, 900)
    body = TestClient(app).get("/api/health/deep").json()
    b = body["checks"]["budgets"]
    assert b["global_used"] == 900 and b["global_limit"] == 1000 and b["pct"] == 90.0
    assert b["owners_tracked"] == 1 and b["anthropic_skipped_today"] == 0
    assert ALICE not in json.dumps(body) and cache.owner_tag(ALICE) not in json.dumps(body)


def test_public_config_exposes_booleans_only(led, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_DAILY_TOKEN_BUDGET", "1000")
    assert config.public_config()["premiumBudget"] == {"nearlyUsed": False, "exhausted": False}
    led.add(None, 850)
    assert config.public_config()["premiumBudget"] == {"nearlyUsed": True, "exhausted": False}
    led.add(None, 500)
    assert config.public_config()["premiumBudget"] == {"nearlyUsed": True, "exhausted": True}


# --------------------------------------------------------------------------------------------------
# End to end: chat route, owner attribution, fall-through with the model label
# --------------------------------------------------------------------------------------------------
def _frames(body):
    return [json.loads(f.strip()[5:]) for f in body.split("\n\n") if f.strip().startswith("data:")]


def _big_usage_stream(text="premium answer", prompt=6000, out=100):
    lines = [f"data: {json.dumps({'type': 'message_start', 'message': {'usage': {'input_tokens': prompt}}})}",
             f"data: {json.dumps({'type': 'content_block_delta', 'delta': {'type': 'text_delta', 'text': text}})}",
             f"data: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn'}, 'usage': {'output_tokens': out}})}",
             f"data: {json.dumps({'type': 'message_stop'})}"]
    return sse(lines)


def test_e2e_chat_route_attributes_owner_then_falls_through_to_groq(net, led, monkeypatch):
    monkeypatch.setattr(config, "PERSISTENCE_ENABLED", False)
    monkeypatch.setenv("ANTHROPIC_OWNER_DAILY_TOKENS", "5000")
    chat_service._LOCAL.clear()

    def _owner(request: Request) -> str:
        o = request.headers.get("x-test-owner", ALICE)
        spend.set_owner(o)                                      # what auth.current_owner does
        return o

    app.dependency_overrides[current_owner] = _owner
    try:
        c = TestClient(app)
        net.on(1, CLAUDE, _big_usage_stream())
        net.on(2, G_MODELS[0], sse_groq(["free ", "answer"]))

        def ask(owner):
            sid = c.post("/api/chat/start", json={"mode": "topic", "topic": "Recursion"},
                         headers={"x-test-owner": owner}).json()["session_id"]
            r = c.post("/api/chat/message/stream", json={"session_id": sid, "message": "what is recursion?"},
                       headers={"x-test-owner": owner})
            return _frames(r.text)[-1]

        d1 = ask(ALICE)
        assert d1["type"] == "done" and d1["llm"]["step"] == 1               # premium while under budget
        assert led._local[_scope(ALICE)] == 6100                              # actual streamed usage, attributed
        d2 = ask(ALICE)                                                       # over her 5,000: free providers
        assert d2["type"] == "done" and d2["llm"]["step"] == 2 and d2["llm"]["label"].startswith("Model: groq")
        assert "anthropic budget" in d2["llm"]["skipped"] and d2["answer"] == "free answer"
        d3 = ask(BOB)                                                         # someone else is unaffected
        assert d3["llm"]["step"] == 1
    finally:
        app.dependency_overrides.pop(current_owner, None)
        chat_service._LOCAL.clear()


def test_e2e_oversize_request_returns_413_json_from_a_route(net, led, monkeypatch):
    only_keys(monkeypatch, "ANTHROPIC_API_KEY")
    monkeypatch.setenv("ANTHROPIC_MAX_REQUEST_INPUT_TOKENS", "500")
    from backend.main import _llm_unavailable
    exc = None
    try:
        lc.chat("s", [{"role": "user", "content": "word " * 5000}])
    except lc.LLMChainError as e:
        exc = e
    resp = _llm_unavailable(None, exc)
    assert resp.status_code == 413
    body = json.loads(resp.body)
    assert "too large for the available models" in body["detail"] and body["llm"]["label"].startswith("Model: none")


def test_shipped_defaults_are_unlimited(monkeypatch):
    """The owner removed all budget caps: with no env set, nothing is limited."""
    for k in ("ANTHROPIC_DAILY_TOKEN_BUDGET", "ANTHROPIC_OWNER_DAILY_TOKENS",
              "ANTHROPIC_GUEST_DAILY_TOKENS", "ANTHROPIC_MAX_REQUEST_INPUT_TOKENS"):
        monkeypatch.delenv(k, raising=False)
    assert config.DEFAULT_ANTHROPIC_DAILY_TOKEN_BUDGET == 0
    assert config.DEFAULT_ANTHROPIC_OWNER_DAILY_TOKENS == 0
    assert config.DEFAULT_ANTHROPIC_GUEST_DAILY_TOKENS == 0
    assert config.DEFAULT_ANTHROPIC_MAX_REQUEST_INPUT_TOKENS == 0
