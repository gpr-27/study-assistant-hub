"""Shared cache layer (backend/cache.py) and everything that uses it, plus the Mongo op budget of a session.

A "shared Redis" is simulated with one MemoryBackend installed via ``cache.configure(.., shared=True)``;
"another serverless instance" is simulated by clearing the in-process (L1) caches.
"""
from __future__ import annotations

import json
import threading
import time
import unittest
from unittest import mock

import httpx

from backend import cache, db, llm_chain, ratelimit
from backend.services import chat as chat_service
from backend.services import progress


def _shared():
    be = cache.MemoryBackend()
    cache.configure(be, shared=True)
    return be


def _new_instance():
    """Forget every per-process cache, as if a different serverless instance served the next request."""
    db.clear_list_cache()
    db._touched.clear()
    db._guest_flags.clear()
    progress.clear_cache()
    with chat_service._local_lock:
        chat_service._STATE.clear()
    llm_chain.reset_state()


class _Base(unittest.TestCase):
    def setUp(self):
        cache.reset()
        self.addCleanup(cache.reset)
        db.clear_list_cache()
        progress.clear_cache()
        self.addCleanup(db.clear_list_cache)


class TestBackendsAndBreaker(_Base):
    def test_default_is_memory_and_helpers_are_noops(self):
        with mock.patch.dict("os.environ", {"CACHE_BACKEND": "auto"}, clear=False):
            for k in ("UPSTASH_REDIS_REST_URL", "UPSTASH_REDIS_REST_TOKEN", "KV_REST_API_URL", "KV_REST_API_TOKEN"):
                mock.patch.dict("os.environ").start().pop(k, None)
            cache.reset()
            self.assertFalse(cache.enabled())
            self.assertEqual(cache.backend_name(), "memory")
            self.assertIsNone(cache.get("x"))
            self.assertIsNone(cache.bucket_take("o", 5, 10))
            self.assertEqual(cache.l2_get("ns", "o"), (None, None))

    def test_upstash_selected_from_env_and_speaks_the_rest_protocol(self):
        seen = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append((request.url.path, request.headers["authorization"], json.loads(request.content)))
            if request.url.path.endswith("/pipeline"):
                return httpx.Response(200, json=[{"result": "5"}, {"result": None}])
            return httpx.Response(200, json={"result": "OK"})

        be = cache.UpstashBackend("https://example.upstash.io", "tok", 0.1)
        be._client = httpx.Client(transport=httpx.MockTransport(handler),
                                  headers={"Authorization": "Bearer tok"})
        cache.configure(be, shared=True)
        self.assertTrue(cache.set("k", "v", 30))
        self.assertEqual(seen[0], ("/", "Bearer tok", ["SET", "k", "v", "EX", 30]))
        self.assertEqual(cache.l2_get("ns", "owner-1"), (None, 5))
        self.assertEqual(seen[1][0], "/pipeline")
        self.assertEqual([c[0] for c in seen[1][2]], ["GET", "GET"])        # epoch + value in ONE request

    def test_auto_picks_upstash_only_with_url_and_token(self):
        env = {"CACHE_BACKEND": "auto", "UPSTASH_REDIS_REST_URL": "https://x.upstash.io", "UPSTASH_REDIS_REST_TOKEN": "t"}
        with mock.patch.dict("os.environ", env):
            cache.reset()
            self.assertEqual(cache.backend_name(), "upstash")
            self.assertTrue(cache.enabled())
        with mock.patch.dict("os.environ", {**env, "CACHE_BACKEND": "memory"}):
            cache.reset()
            self.assertFalse(cache.enabled())

    def test_failures_fall_back_and_the_breaker_opens_then_recovers(self):
        class Dead:
            name = "dead"
            calls = 0

            def execute(self, cmds):
                Dead.calls += 1
                raise cache.BackendError("boom")

        cache.configure(Dead(), shared=True)
        for _ in range(10):
            self.assertIsNone(cache.get("k"))                     # never raises
        self.assertEqual(Dead.calls, 3)                           # breaker opened after 3 failures
        self.assertFalse(cache.enabled())
        self.assertEqual(cache.status()["opened"], 1)
        cache._open_until = time.monotonic() - 1                  # cool-down over -> one probe
        cache.get("k")
        self.assertEqual(Dead.calls, 4)
        self.assertFalse(cache.enabled())                         # failed probe re-opens at once

    def test_timeout_is_capped_at_150ms(self):
        with mock.patch.dict("os.environ", {"CACHE_TIMEOUT_MS": "5000"}):
            self.assertEqual(cache._timeout_s(), 0.15)

    def test_keys_are_namespaced_and_never_contain_the_raw_owner(self):
        k = cache.key("progress", "guest_secretid123")
        self.assertTrue(k.startswith("sah:v1:progress:"))
        self.assertNotIn("secretid", k)
        self.assertNotEqual(cache.key("progress", "a"), cache.key("progress", "b"))

    def test_oversized_values_are_not_cached(self):
        _shared()
        self.assertFalse(cache.set("k", "x" * (cache.MAX_VALUE_BYTES + 1), 10))

    def test_single_flight_computes_once(self):
        calls, store = [], {}

        def work():
            with cache.single_flight("k"):
                if "v" not in store:
                    calls.append(1)
                    time.sleep(0.05)
                    store["v"] = 1

        ts = [threading.Thread(target=work) for _ in range(8)]
        [t.start() for t in ts]
        [t.join() for t in ts]
        self.assertEqual(len(calls), 1)
        self.assertEqual(cache._flights, {})


class TestEpochL2(_Base):
    def test_value_visible_across_instances_until_the_owner_writes(self):
        _shared()
        ep0 = cache.l2_get("progress", "alice")[1]
        cache.l2_put("progress", "alice", "", {"x": 1}, ep0, 60)
        self.assertEqual(cache.l2_get("progress", "alice")[0], {"x": 1})
        self.assertIsNone(cache.l2_get("progress", "bob")[0])                  # owner scoped
        cache.bump_epoch("alice")                                              # a write on ANY instance
        self.assertIsNone(cache.l2_get("progress", "alice")[0])
        self.assertEqual(cache.l2_get("progress", "alice")[1], ep0 + 1)

    def test_a_write_racing_the_compute_is_never_served_as_fresh(self):
        _shared()
        _v, ep = cache.l2_get("progress", "alice")                              # epoch read BEFORE computing
        cache.bump_epoch("alice")                                               # write lands mid-compute
        cache.l2_put("progress", "alice", "", {"stale": True}, ep, 60)
        self.assertIsNone(cache.l2_get("progress", "alice")[0])

    def test_max_stale_window(self):
        _shared()
        ep = cache.l2_get("insights", "a", "d")[1]
        cache.l2_put("insights", "a", "d", {"v": 1}, ep, 60)
        cache.bump_epoch("a")
        self.assertIsNone(cache.l2_get("insights", "a", "d")[0])
        self.assertEqual(cache.l2_get("insights", "a", "d", max_stale_s=90)[0], {"v": 1})

    def test_progress_summary_shared_between_instances(self):
        _shared()
        rows = {"attempts": {"n": 1, "rows": [{"subject": "DBMS", "topic": "B-tree", "correct": 3, "total": 4,
                                               "created_at": "2026-01-01T00:00:00+00:00"}]}}

        def multi(owner, specs):
            return {k: rows.get(k, {"n": 0, "rows": []}) for k, _l, _f in specs}

        with mock.patch.object(progress.db, "multi_recent", side_effect=multi) as mr:
            first = progress.build_summary("alice")
            _new_instance()
            second = progress.build_summary("alice")                            # other instance: shared L2, no read
            self.assertEqual(mr.call_count, 1)
            self.assertEqual(first, second)
            db._bump("alice")                                                   # a write anywhere invalidates it
            _new_instance()
            progress.build_summary("alice")
            self.assertEqual(mr.call_count, 2)


class TestRateLimitShared(_Base):
    def test_limit_holds_across_instances(self):
        _shared()
        ratelimit.reset()
        with mock.patch.object(ratelimit.config, "RATE_LIMIT_GUEST_PER_MIN", 3), \
                mock.patch.object(ratelimit.config, "RATE_LIMIT_GUEST_PER_HOUR", 0):
            allowed = 0
            for _ in range(6):
                ratelimit.reset()                                              # each request hits a "new instance"
                if ratelimit.check("guest_a") == 0.0 and ratelimit.check_shared("guest_a") == 0.0:
                    allowed += 1
            self.assertEqual(allowed, 3)                                       # in-process alone would allow all 6
            ratelimit.reset()
            self.assertGreater(ratelimit.check_shared("guest_a"), 0.0)
            self.assertEqual(ratelimit.check_shared("guest_b"), 0.0)           # other owner unaffected

    def test_fails_open_when_the_cache_is_down(self):
        class Dead:
            name = "dead"

            def execute(self, cmds):
                raise cache.BackendError("down")

        cache.configure(Dead(), shared=True)
        ratelimit.reset()
        self.assertEqual(ratelimit.check_shared("guest_a"), 0.0)


class TestSharedLlmChainMemory(_Base):
    def test_exhausted_quota_and_dead_key_are_visible_to_other_instances(self):
        _shared()
        llm_chain.reset_state()
        st = llm_chain._State()
        with mock.patch.object(cache, "background", lambda fn: fn()):           # run writes inline for the test
            st.mark_exhausted(2, "gpt-x", 600)
            st.share_dead(3, "sk-secret-key")
        other = llm_chain._State()                                              # fresh process memory
        cache.shared_map("llm-exhausted").clear_local()
        cache.shared_map("llm-dead").clear_local()
        self.assertTrue(other.is_exhausted(2, "gpt-x"))
        self.assertFalse(other.is_exhausted(2, "other-model"))
        self.assertTrue(other.is_dead_shared(3, "sk-secret-key"))
        self.assertFalse(other.is_dead_shared(3, "a-different-key"))            # key replaced => not dead

    def test_without_a_shared_cache_behaviour_is_unchanged(self):
        st = llm_chain._State()
        st.mark_exhausted(1, "m", 600)
        self.assertTrue(st.is_exhausted(1, "m"))
        self.assertFalse(st.is_exhausted(1, "n"))
        self.assertFalse(st.is_dead_shared(1, "k"))


class TestGuestTouchShared(_Base):
    def test_cold_instance_skips_the_write_when_another_instance_touched(self):
        _shared()
        coll = mock.MagicMock()
        coll.find_one_and_update.return_value = {"hasUploads": False}
        fake = {db.GUESTS: coll}
        with mock.patch.object(db, "_get_db", return_value=fake):
            self.assertTrue(db.touch_guest("guest_g"))
            self.assertEqual(coll.find_one_and_update.call_count, 1)
            _new_instance()
            self.assertTrue(db.touch_guest("guest_g"))
            self.assertEqual(coll.find_one_and_update.call_count, 1)           # shared marker, no Mongo op
            self.assertTrue(db.has_no_uploads("guest_g"))                      # flag travels with the marker

    def test_migration_clears_the_shared_marker(self):
        be = _shared()
        db._remember_touch("guest_m", False)
        self.assertIsNotNone(cache.get(cache.key("touch", "guest_m")))
        coll = mock.MagicMock()
        coll.find_one_and_update.return_value = {"guestId": "guest_m"}
        coll.update_many.return_value = mock.Mock(modified_count=0)
        with mock.patch.object(db, "_get_db", return_value=mock.MagicMock(__getitem__=lambda s, k: coll)):
            db.migrate_guest("guest_m", "user_1")
        self.assertIsNone(cache.get(cache.key("touch", "guest_m")))
        self.assertIsNotNone(be)


class TestChatStateShared(_Base):
    def test_followup_on_another_instance_needs_no_read(self):
        _shared()
        s = chat_service.Session(id="s", owner="alice", mode="topic", label="l", subject=None, topic="t",
                                 chat_id="507f1f77bcf86cd799439011", nonce="n")
        chat_service._state_put(s, {"history": [("q1", "a1")], "recent_artifact": None})
        time.sleep(0.05)                                                        # background writer
        with chat_service._local_lock:
            chat_service._STATE.clear()
        st = chat_service._state_get(s)
        self.assertEqual(st["history"], [("q1", "a1")])
        other = chat_service.Session(id="s", owner="mallory", mode="topic", label="l", subject=None, topic="t",
                                     chat_id="507f1f77bcf86cd799439011", nonce="n")
        with chat_service._local_lock:
            chat_service._STATE.clear()
        self.assertIsNone(chat_service._state_get(other))                       # owner scoped


# ------------------------------------------------------------------------------------------------
# Mongo op budget: count every command a full session sends (a fake "listener")
# ------------------------------------------------------------------------------------------------
class _Res:
    def __init__(self, **kw):
        self.matched_count = kw.get("matched", 1)
        self.upserted_id = kw.get("upserted")
        self.inserted_id = kw.get("inserted")
        self.deleted_count = 1
        self.modified_count = 1


_created: set = set()


class _Coll:
    def __init__(self, ops, name):
        self.ops, self.name = ops, name

    def _c(self, what):
        self.ops.append(f"{what}:{self.name}")

    def find_one_and_update(self, *a, **k):
        self._c("findAndModify")
        return {"hasUploads": False}

    def aggregate(self, *a, **k):
        self._c("aggregate")
        return iter([])

    def insert_one(self, doc):
        self._c("insert")
        return _Res(inserted="507f1f77bcf86cd7994390aa")

    def update_one(self, *a, **k):
        self._c("update")
        first = k.get("upsert") and a[0].get("_id") not in _created
        _created.add(a[0].get("_id"))
        return _Res(upserted=("x" if first else None))

    def find(self, *a, **k):
        self._c("find")
        cur = mock.MagicMock()
        cur.sort.return_value.limit.return_value = []
        return cur

    def find_one(self, *a, **k):
        self._c("find")
        return None


class _CountingDB:
    def __init__(self):
        self.ops: list[str] = []

    def __getitem__(self, name):
        return _Coll(self.ops, name)


class TestSessionOpBudget(_Base):
    def test_full_session_stays_within_the_op_budget(self):
        """touch + dashboard-ish reads + 3 chat turns + quiz + flashcards + history + progress."""
        _created.clear()
        fake = _CountingDB()
        owner = "guest_" + "b" * 32
        chat_id = "507f1f77bcf86cd799439011"
        sess = chat_service.Session(id="s", owner=owner, mode="subject", label="Subject · Maths", subject="Maths",
                                    topic=None, chat_id=chat_id, nonce="n")
        with mock.patch.object(db, "_get_db", return_value=fake):
            self.assertTrue(db.touch_guest(owner))                              # 1 (validate + touch + flags)
            self.assertEqual(db.list_user_subjects(owner), [])                  # 0: guest known to have no uploads
            db.list_items("chats", owner)                                       # 1: the owner snapshot
            chat_service.prime_state(owner, chat_id)                            # chat/start: no op (lazy)
            for i in range(3):                                                  # 3 turns = 3 writes (the first creates the chat + title)
                chat_service._persist_exchange(sess, f"q{i}", f"a{i}", None, None, i == 0, title="T" if i == 0 else None)
            db.save("quizzes", owner, {"title": "quiz", "questions": [1]})      # 1
            db.save("flashcards", owner, {"title": "deck", "cards": [1]})       # 1
            hist = db.list_items("quizzes", owner)                              # 0: patched snapshot
            summary = progress.build_summary(owner)                             # 0: computed from the snapshot
        self.assertEqual([h["title"] for h in hist], ["quiz"])
        self.assertEqual({a["tool"]: a["count"] for a in summary["activity"]}["quizzes"], 1)
        self.assertEqual({a["tool"]: a["count"] for a in summary["activity"]}["chats"], 1)
        self.assertEqual(len(fake.ops), 7, fake.ops)
        self.assertEqual(sum(o.startswith("aggregate") for o in fake.ops), 1)
        self.assertEqual(sum(o.startswith(("update:chats")) for o in fake.ops), 3)

    def test_repeat_dashboard_polls_cost_nothing(self):
        fake = _CountingDB()
        with mock.patch.object(db, "_get_db", return_value=fake):
            for _ in range(20):
                db.list_items("chats", "u_poll")
                db.list_items("quizzes", "u_poll")
                progress.build_summary("u_poll")
        self.assertEqual(len(fake.ops), 1, fake.ops)


if __name__ == "__main__":
    unittest.main()
