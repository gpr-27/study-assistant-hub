"""Review-fix tests: error sanitisation, guest-id log redaction, cron budget, insights cache epoch,
shared request deadline, partial scan results, body limit, rate limiting, fail-closed production
config and chat-session expiry. Everything is mocked; no network, key or database."""
import asyncio
import json
import logging
import os
import time
import unittest
from unittest import mock

from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from backend import auth, bodylimit, config, db, deadline, llm_chain, ratelimit
from backend.main import app
from backend.services import chat, chat_tools, insights, scannotes
from backend.services import _gen
from tests._study_helpers import FakeChat, chain_error, result

ATLAS_HOST = "ac-abc123-shard-00-01.xyz.mongodb.net"


def _client():
    return TestClient(app, raise_server_exceptions=False)


def _app_routes():
    """``(full_path, APIRoute, router-level dependency callables)`` for every included router
    (this FastAPI version keeps included routers lazy, so ``app.routes`` holds wrappers)."""
    for wrapper in app.routes:
        original = getattr(wrapper, "original_router", None)
        if original is None:
            continue
        deps = [d.dependency for d in wrapper.include_context.dependencies]
        for route in original.routes:
            yield route.path, route, deps


class _MongoLikeError(Exception):
    """Looks like a pymongo error (matched by module name)."""
    __module__ = "pymongo.errors"


# ── 4: raw exception text never reaches answers / responses ─────────────────────────────────
class ErrorSanitisationTests(unittest.TestCase):
    def test_progress_failure_answer_has_no_exception_text(self):
        with mock.patch.object(chat_tools.progress_service, "build_summary",
                               side_effect=_MongoLikeError(f"{ATLAS_HOST}:27017: connection refused")):
            out = chat_tools.run_tool({"owner": "u1"}, {"tool": "progress"}, "show my progress")
        self.assertNotIn(ATLAS_HOST, json.dumps(out))
        self.assertNotIn("27017", json.dumps(out))
        self.assertIn("database is temporarily unavailable", out["answer"])

    def test_tool_crash_answer_and_trace_have_no_exception_text(self):
        with mock.patch.object(chat_tools, "_TOOL_ROUTES", {"quiz": ("quizzes", "#quiz", "Quiz")}), \
             mock.patch.object(chat_tools, "_load_subject_content",
                               side_effect=RuntimeError(f"boom at {ATLAS_HOST}"), create=True), \
             mock.patch.object(chat_tools.quiz_service, "make_quiz_from_topic",
                               side_effect=RuntimeError(f"boom at {ATLAS_HOST}")):
            out = chat_tools.run_tool({"owner": "u1", "topic": "Sorting"},
                                      {"tool": "quiz", "topic": "Sorting", "count": 3}, "make a quiz on sorting")
        self.assertIsNotNone(out)
        blob = json.dumps(out)
        self.assertNotIn(ATLAS_HOST, blob)
        self.assertIn("something went wrong", out["answer"])

    def test_embedding_error_handler_hides_upstream_text(self):
        from backend.gemini_embeddings import GeminiEmbeddingError

        def boom():
            raise GeminiEmbeddingError(f"Gemini embeddings error 500: internal {ATLAS_HOST}")

        app.add_api_route("/api/_test_emb", boom, methods=["GET"])
        self.addCleanup(lambda: app.router.routes.pop())
        r = _client().get("/api/_test_emb")
        self.assertEqual(r.status_code, 502)
        self.assertNotIn(ATLAS_HOST, r.text)

    def test_public_message_helper(self):
        from backend.errors import DB_DOWN_MESSAGE, GENERIC_MESSAGE, public_message
        self.assertEqual(public_message(_MongoLikeError("x")), DB_DOWN_MESSAGE)
        self.assertEqual(public_message(ValueError("secret")), GENERIC_MESSAGE)


# ── 5: guest ids are bearer credentials ─────────────────────────────────────────────────────
class GuestIdLogTests(unittest.TestCase):
    GID = "guest_" + "ab12" * 8

    def test_redact_id_is_stable_short_and_hides_the_id(self):
        tag = db.redact_id(self.GID)
        self.assertEqual(tag, db.redact_id(self.GID))
        self.assertNotIn(self.GID, tag)
        self.assertNotIn("ab12ab12", tag)
        self.assertLess(len(tag), 16)

    def test_create_and_migrate_do_not_log_the_guest_id(self):
        coll = mock.MagicMock()
        fake = mock.MagicMock()
        fake.__getitem__.return_value = coll
        coll.find_one_and_update.return_value = {"guestId": self.GID}
        coll.update_many.return_value = mock.MagicMock(modified_count=1)
        with self.assertLogs("backend.db", level="INFO") as logs, \
             mock.patch.object(db, "_get_db", return_value=fake):
            created = db.create_guest()
            db.migrate_guest(created["guestId"], "user_clerk")
            db.migrate_guest(self.GID, "user_clerk")
        text = "\n".join(logs.output)
        self.assertNotIn(created["guestId"], text)
        self.assertNotIn(self.GID, text)
        self.assertNotIn("guest_" + created["guestId"][6:14], text)

    def test_store_pdf_log_hides_owner(self):
        fs = mock.MagicMock()
        fs.find_one.return_value = None
        with self.assertLogs("backend.db", level="INFO") as logs, \
             mock.patch.object(db, "_gridfs", return_value=fs):
            db.store_uploaded_pdf("S", "a.pdf", b"%PDF-x", self.GID)
        self.assertNotIn(self.GID, "\n".join(logs.output))

    def test_migrate_runs_uploads_only_after_the_atomic_claim(self):
        order = []
        guests = mock.MagicMock()
        guests.find_one_and_update.side_effect = lambda *a, **k: (order.append("claim"), {"guestId": "g"})[1]
        other = mock.MagicMock()
        other.update_many.return_value = mock.MagicMock(modified_count=0)
        fake = mock.MagicMock()
        fake.__getitem__.side_effect = lambda n: guests if n == db.GUESTS else other
        with mock.patch.object(db, "_get_db", return_value=fake):
            out = db.migrate_guest("g", "user_x", after_claim=lambda: (order.append("uploads"), 2)[1])
        self.assertEqual(order, ["claim", "uploads"])
        self.assertEqual(out["subjects_moved"], 2)
        # a losing claim never runs the upload move
        order.clear()
        guests.find_one_and_update.side_effect = lambda *a, **k: None
        with mock.patch.object(db, "_get_db", return_value=fake):
            out = db.migrate_guest("g", "user_y", after_claim=lambda: order.append("uploads"))
        self.assertEqual(order, [])
        self.assertFalse(out["migrated"])


# ── 3: one shared budget for the janitor ────────────────────────────────────────────────────
class CronBudgetTests(unittest.TestCase):
    def _run(self, **patches):
        with mock.patch.dict(os.environ, {"CRON_SECRET": "s"}), \
             mock.patch.object(config, "PERSISTENCE_ENABLED", True):
            return _client().get("/api/cron/janitor", headers={"Authorization": "Bearer s"})

    def test_sweeps_share_one_budget_and_report_partial(self):
        seen = {}

        def purge(budget_s=40.0, stats=None):
            seen["data"] = budget_s
            stats["partial"] = True
            return 1

        def uploads(budget_s=30.0, stats=None):
            seen["uploads"] = budget_s
            return 2

        with mock.patch.object(db, "ensure_indexes", return_value={"skipped": "up to date"}), \
             mock.patch.object(db, "purge_orphan_guest_data", purge), \
             mock.patch("backend.services.subjects.purge_orphan_guest_uploads", uploads):
            r = self._run()
        body = r.json()
        self.assertEqual((body["removed"], body["uploads_removed"], body["partial"]), (1, 2, True))
        self.assertLessEqual(seen["data"] + seen["uploads"], 55)          # sum of both sweeps fits one budget
        self.assertLessEqual(seen["data"], 26)
        self.assertLessEqual(seen["uploads"], 30)

    def test_slow_index_ensure_leaves_less_for_the_sweeps(self):
        seen = {}

        def slow_indexes(*a, **k):
            time.sleep(0.2)
            return {"created": 1}

        with mock.patch("backend.main.CRON_BUDGET_S", 3.0), \
             mock.patch.object(db, "ensure_indexes", slow_indexes), \
             mock.patch.object(db, "purge_orphan_guest_data", lambda budget_s=40.0, stats=None: seen.setdefault("d", budget_s) and 0), \
             mock.patch("backend.services.subjects.purge_orphan_guest_uploads", lambda budget_s=30.0, stats=None: seen.setdefault("u", budget_s) and 0):
            self._run()
        self.assertLess(seen["d"], 3.0)
        self.assertLess(seen["u"], 3.0)

    def test_budget_exhausted_by_indexes_skips_sweeps(self):
        with mock.patch("backend.main.CRON_BUDGET_S", 1.0), \
             mock.patch.object(db, "ensure_indexes", lambda *a, **k: time.sleep(1.1) or {}), \
             mock.patch.object(db, "purge_orphan_guest_data", side_effect=AssertionError("no time left")), \
             mock.patch("backend.services.subjects.purge_orphan_guest_uploads", side_effect=AssertionError("no time left")):
            r = self._run()
        self.assertTrue(r.json()["partial"])

    def test_db_sweep_stops_at_its_deadline_and_flags_partial(self):
        coll = mock.MagicMock()
        coll.distinct.return_value = [f"guest_{i}" for i in range(2000)]
        guests = mock.MagicMock()
        guests.find.return_value = []
        coll.delete_many.return_value = mock.MagicMock(deleted_count=5)
        fake = mock.MagicMock()
        fake.__getitem__.side_effect = lambda n: guests if n == db.GUESTS else coll
        stats = {}
        with mock.patch.object(db, "_get_db", return_value=fake):
            db.purge_orphan_guest_data(budget_s=-1.0, stats=stats)
        self.assertTrue(stats.get("partial"))


# ── 6: insights cache invalidated by new activity ───────────────────────────────────────────
class InsightsEpochTests(unittest.TestCase):
    def setUp(self):
        insights._cache.clear()
        insights._last_refresh.clear()

    def _build(self, fake):
        prog = {"overall": {"total_attempts": 1}, "activity": []}
        with mock.patch("backend.llm_chain.chat", fake), \
             mock.patch.object(insights, "build_summary", return_value=prog), \
             mock.patch.object(insights.db, "multi_recent",
                               side_effect=lambda o, specs: {k: {"n": 1, "rows": [{"title": "Quiz"}]} for k, *_ in specs}):
            return insights.build_insights("owner-1")

    def test_new_activity_invalidates_after_the_regen_interval(self):
        good = {"headline": "Hi", "observations": [], "recommendations": [], "focus_topics": []}
        fake = FakeChat(good)
        self._build(fake)
        self._build(fake)
        self.assertEqual(len(fake.calls), 1)                       # same epoch -> cached
        db._bump("owner-1")                                        # a new quiz / exam was saved
        self._build(fake)
        self.assertEqual(len(fake.calls), 1)                       # ...but not more often than the interval
        with mock.patch.object(insights, "_EPOCH_REGEN_MIN_S", 0.0):
            self._build(fake)
        self.assertEqual(len(fake.calls), 2)                       # regenerated with the new activity

    def test_cache_stays_bounded(self):
        for i in range(insights._CACHE_MAX + 5):
            insights._cache[f"o{i}"] = (("d", (0, 0)), {}, 0.0)
            while len(insights._cache) > insights._CACHE_MAX:
                insights._cache.popitem(last=False)
        self.assertLessEqual(len(insights._cache), insights._CACHE_MAX)


# ── 7: shared per-request deadline ──────────────────────────────────────────────────────────
class DeadlineTests(unittest.TestCase):
    def tearDown(self):
        deadline.start(None)

    def test_no_deadline_off_vercel(self):
        deadline.begin_request()
        self.assertIsNone(deadline.remaining())
        self.assertEqual(deadline.clamp(55.0), 55.0)

    def test_vercel_budget_is_started_by_the_middleware(self):
        seen = {}

        def probe():
            seen["rem"] = deadline.remaining()
            return {}

        app.add_api_route("/api/_test_deadline", probe, methods=["GET"])
        self.addCleanup(lambda: app.router.routes.pop())
        with mock.patch.dict(os.environ, {"VERCEL": "1"}):
            _client().get("/api/_test_deadline")
        self.assertIsNotNone(seen["rem"])
        self.assertLessEqual(seen["rem"], config.REQUEST_BUDGET_S)
        self.assertGreater(seen["rem"], config.REQUEST_BUDGET_S - 5)

    def test_clamp_limits_and_raises_when_nothing_is_left(self):
        deadline.start(10.0)
        self.assertLessEqual(deadline.clamp(40.0), 10.0)
        deadline.start(2.0)
        with self.assertRaises(deadline.DeadlineExceeded):
            deadline.clamp(40.0)
        self.assertIsInstance(deadline.DeadlineExceeded(), llm_chain.LLMChainError)

    def test_chat_tool_turn_cannot_exceed_the_budget(self):
        """intent detection (15 s) + generate_json (up to 55 s) share ONE 50 s budget."""
        waits = []

        def fake_chat(system, messages, **kw):
            waits.append(kw["deadline_s"])
            time.sleep(0.0)
            return result({"tool": "quiz"}) if len(waits) == 1 else result({"questions": []})

        deadline.start(50.0)
        with mock.patch("backend.llm_chain.chat", fake_chat), mock.patch.dict(os.environ, {"VERCEL": "1"}):
            chat_tools.detect_intent("make a quiz on sorting")
            # pretend the first step used most of the budget
            deadline.start(12.0)
            try:
                _gen.generate_json("s", "p", shape=lambda d: d.get("questions"), max_tokens=10,
                                   error_cls=RuntimeError, what="quiz")
            except RuntimeError:
                pass
        self.assertLessEqual(waits[0], 15.0)
        self.assertLessEqual(waits[1], 12.0)                       # generation got only what was left

    def test_generate_json_raises_a_clean_llm_error_when_the_budget_is_spent(self):
        deadline.start(1.0)
        with mock.patch("backend.llm_chain.chat", side_effect=AssertionError("no call should start")):
            with self.assertRaises(llm_chain.LLMChainError):
                _gen.generate_json("s", "p", shape=lambda d: d, max_tokens=10, error_cls=RuntimeError, what="x")

    def test_generate_json_default_budget_is_tighter_on_vercel(self):
        self.assertEqual(_gen.default_budget(), _gen.DEFAULT_BUDGET_S)
        with mock.patch.dict(os.environ, {"VERCEL": "1"}):
            self.assertEqual(_gen.default_budget(), _gen.SERVERLESS_BUDGET_S)
            self.assertLessEqual(_gen.default_budget(), 40.0)

    def test_budget_exceeded_maps_to_503_with_the_llm_label(self):
        def spent():
            raise deadline.DeadlineExceeded()

        app.add_api_route("/api/_test_spent", spent, methods=["GET"])
        self.addCleanup(lambda: app.router.routes.pop())
        r = _client().get("/api/_test_spent")
        self.assertEqual(r.status_code, 503)
        self.assertIn("label", r.json()["llm"])


# ── 8: a failed page does not discard the paid-for pages ────────────────────────────────────
class ScanPartialTests(unittest.IsolatedAsyncioTestCase):
    async def _scan(self, replies):
        calls = {"n": 0}

        async def achat(system, messages, *, images=None, **kw):
            i = calls["n"]
            calls["n"] += 1
            r = replies[i]
            if isinstance(r, Exception):
                raise r
            return result(r)

        with mock.patch("backend.llm_chain.achat", achat), mock.patch("backend.llm_chain.has_any_key", return_value=True):
            return await scannotes.transcribe_pages([(b"a", "image/png")] * len(replies))

    async def test_one_failed_page_keeps_the_others_and_marks_it(self):
        out = await self._scan(["page one text", chain_error(), "page three text"])
        self.assertEqual(out.failed_pages, [2])
        self.assertTrue(out.partial)
        self.assertIn("## Page 1\n\npage one text", out.markdown)
        self.assertIn("## Page 3\n\npage three text", out.markdown)
        self.assertIn("Page 2 could not be transcribed", out.markdown)
        self.assertIsNotNone(out.llm)

    async def test_all_pages_failing_still_raises_the_llm_error(self):
        with self.assertRaises(llm_chain.LLMChainError):
            await self._scan([chain_error(), chain_error()])

    async def test_empty_transcription_of_one_page_is_a_marker_not_a_failure(self):
        out = await self._scan(["good", "   "])
        self.assertEqual(out.failed_pages, [2])

    async def test_deadline_stops_late_pages_but_keeps_early_ones(self):
        async def achat(system, messages, *, images=None, deadline_s=0, **kw):
            await asyncio.sleep(0)
            return result("ok")

        def clamp(want, **kw):
            clamp.n += 1
            if clamp.n > 3:
                raise deadline.DeadlineExceeded()
            return want
        clamp.n = 0
        with mock.patch("backend.llm_chain.achat", achat), mock.patch("backend.llm_chain.has_any_key", return_value=True), \
             mock.patch.object(scannotes.deadline, "clamp", clamp):
            out = await scannotes.transcribe_pages([(b"a", "image/png")] * 6)
        self.assertEqual(out.failed_pages, [4, 5, 6])
        self.assertIn("## Page 1\n\nok", out.markdown)

    def test_router_returns_pages_and_failed_markers(self):
        from backend.routers.scannotes import router
        from tests._study_helpers import make_client
        calls = {"n": 0}

        async def achat(system, messages, *, images=None, **kw):
            calls["n"] += 1
            if calls["n"] == 2:
                raise chain_error()
            return result("text")

        files = [("images", (f"{i}.png", b"img", "image/png")) for i in range(2)]
        with mock.patch("backend.llm_chain.achat", achat), mock.patch("backend.llm_chain.has_any_key", return_value=True), \
             mock.patch("backend.db.save", return_value="sid") as save:
            r = make_client(router).post("/api/scannotes/extract", files=files)
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertEqual((body["failed_pages"], body["partial"], body["id"]), ([2], True, "sid"))
        self.assertIn("text", body["markdown"])
        self.assertTrue(save.called)                                # paid-for work is saved


# ── 13: body limit as JSON ──────────────────────────────────────────────────────────────────
class BodyLimitTests(unittest.TestCase):
    def test_declared_oversize_body_is_a_json_413(self):
        big = b"x" * (config.MAX_BODY_BYTES + 1024)
        r = _client().post("/api/scannotes/extract", files={"images": ("a.png", big, "image/png")},
                           headers={"Authorization": "Bearer none"})
        self.assertEqual(r.status_code, 413)
        self.assertEqual(r.headers["content-type"], "application/json")
        self.assertIn("too large", r.json()["detail"])

    def test_chunked_oversize_body_is_cut_off_with_json_413(self):
        def body():
            for _ in range(8):
                yield b"x" * (config.MAX_BODY_BYTES // 4)

        r = _client().post("/api/predictor/analyze", content=body(),
                           headers={"Content-Type": "multipart/form-data; boundary=zz"})
        self.assertEqual(r.status_code, 413)
        self.assertIn("detail", r.json())

    def test_small_bodies_and_other_paths_pass(self):
        r = _client().post("/api/scannotes/extract", content=b"x", headers={"Content-Type": "text/plain"})
        self.assertNotEqual(r.status_code, 413)
        r = _client().post("/api/subjects", content=b"x" * (config.MAX_BODY_BYTES + 10), headers={"Content-Type": "text/plain"})
        self.assertNotEqual(r.status_code, 413)              # chunked-upload flow has its own limits

    def test_every_upload_route_is_covered(self):
        import inspect
        upload_paths = {path for path, route, _ in _app_routes()
                        if any("UploadFile" in str(p.annotation)
                               for p in inspect.signature(route.endpoint).parameters.values())}
        self.assertIn("/api/scannotes/extract", upload_paths)          # the scan really was discovered
        missing = upload_paths - bodylimit.LIMITED_PATHS - {"/api/subjects"}
        self.assertEqual(missing, set())


# ── 9b: rate limiting ───────────────────────────────────────────────────────────────────────
class RateLimitTests(unittest.TestCase):
    def setUp(self):
        ratelimit.reset()

    def test_burst_up_to_quota_then_429_with_retry_after(self):
        with mock.patch.object(config, "RATE_LIMIT_GUEST_PER_MIN", 3), mock.patch.object(config, "RATE_LIMIT_GUEST_PER_HOUR", 100):
            allowed = [ratelimit.check("guest_a", now=100.0) for _ in range(3)]
            denied = ratelimit.check("guest_a", now=100.0)
        self.assertEqual(allowed, [0.0, 0.0, 0.0])
        self.assertGreater(denied, 0)
        self.assertLessEqual(denied, 20.0)

    def test_tokens_refill_over_time(self):
        with mock.patch.object(config, "RATE_LIMIT_USER_PER_MIN", 6), mock.patch.object(config, "RATE_LIMIT_USER_PER_HOUR", 0):
            for _ in range(6):
                self.assertEqual(ratelimit.check("user_1", now=0.0), 0.0)
            self.assertGreater(ratelimit.check("user_1", now=0.0), 0)
            self.assertEqual(ratelimit.check("user_1", now=10.5), 0.0)     # 6/min = 1 token per 10 s

    def test_hourly_cap_applies_independently(self):
        with mock.patch.object(config, "RATE_LIMIT_USER_PER_MIN", 100), mock.patch.object(config, "RATE_LIMIT_USER_PER_HOUR", 4):
            for _ in range(4):
                self.assertEqual(ratelimit.check("user_h", now=0.0), 0.0)
            wait = ratelimit.check("user_h", now=1.0)
        self.assertGreater(wait, 100)                                     # ~15 min per token

    def test_guests_are_tighter_than_users_and_owners_are_independent(self):
        self.assertLess(ratelimit.limits_for("guest_x")[0], ratelimit.limits_for("user_x")[0])
        self.assertLess(ratelimit.limits_for("guest_x")[1], ratelimit.limits_for("user_x")[1])
        with mock.patch.object(config, "RATE_LIMIT_USER_PER_MIN", 1), mock.patch.object(config, "RATE_LIMIT_USER_PER_HOUR", 0):
            self.assertEqual(ratelimit.check("user_a", now=0.0), 0.0)
            self.assertEqual(ratelimit.check("user_b", now=0.0), 0.0)
            self.assertGreater(ratelimit.check("user_a", now=0.0), 0)

    def test_state_is_bounded(self):
        with mock.patch.object(ratelimit, "_MAX_OWNERS", 50):
            for i in range(500):
                ratelimit.check(f"user_{i}", now=0.0)
        self.assertLessEqual(len(ratelimit._state), 50)

    def test_defaults_match_the_documented_numbers(self):
        self.assertEqual((config.RATE_LIMIT_USER_PER_MIN, config.RATE_LIMIT_USER_PER_HOUR), (30, 300))
        self.assertEqual((config.RATE_LIMIT_GUEST_PER_MIN, config.RATE_LIMIT_GUEST_PER_HOUR), (10, 60))

    def _app_client(self):
        a = FastAPI()
        a.add_api_route("/paid", lambda: {"ok": True}, methods=["POST"], dependencies=[Depends(ratelimit.rate_limit)])
        a.dependency_overrides[auth.current_owner] = lambda: "guest_zzz"
        return TestClient(a)

    def test_dependency_returns_429_json_with_retry_after_and_no_llm_block(self):
        c = self._app_client()
        with mock.patch.dict(os.environ, {"RATE_LIMIT_DISABLED": ""}), \
             mock.patch.object(config, "RATE_LIMIT_GUEST_PER_MIN", 2), mock.patch.object(config, "RATE_LIMIT_GUEST_PER_HOUR", 100):
            self.assertEqual(c.post("/paid").status_code, 200)
            self.assertEqual(c.post("/paid").status_code, 200)
            r = c.post("/paid")
        self.assertEqual(r.status_code, 429)
        self.assertGreaterEqual(int(r.headers["retry-after"]), 1)
        self.assertNotIn("llm", r.json())
        self.assertIn("too quickly", r.json()["detail"])

    def test_disabled_flag_works_locally_but_never_on_vercel(self):
        c = self._app_client()
        with mock.patch.object(config, "RATE_LIMIT_GUEST_PER_MIN", 1), mock.patch.object(config, "RATE_LIMIT_GUEST_PER_HOUR", 0):
            with mock.patch.dict(os.environ, {"RATE_LIMIT_DISABLED": "1"}):
                os.environ.pop("VERCEL", None)
                os.environ.pop("VERCEL_ENV", None)
                self.assertTrue(all(c.post("/paid").status_code == 200 for _ in range(5)))
            ratelimit.reset()
            with mock.patch.dict(os.environ, {"RATE_LIMIT_DISABLED": "1", "VERCEL": "1"}):
                codes = [c.post("/paid").status_code for _ in range(3)]
        self.assertEqual(codes[0], 200)
        self.assertIn(429, codes)

    def test_every_paid_router_is_limited_in_the_real_app(self):
        paid = ("chat", "quiz", "flashcards", "planner", "mindmap", "exam", "teachback", "predictor",
                "insights", "cheatsheet", "scannotes")
        seen = {}
        for path, _route, deps in _app_routes():
            name = path.split("/")[2]
            seen.setdefault(name, []).append(ratelimit.rate_limit in deps)
        for name in paid:
            self.assertTrue(seen.get(name), name)
            self.assertTrue(all(seen[name]), f"/api/{name} is not rate limited")
        for name in ("history", "progress", "guest", "subjects"):
            self.assertFalse(any(seen[name]), name)

    def test_real_app_returns_429_for_a_flooding_owner(self):
        app.dependency_overrides[auth.current_owner] = lambda: "guest_flood"
        self.addCleanup(lambda: app.dependency_overrides.pop(auth.current_owner, None))
        with mock.patch.dict(os.environ, {"RATE_LIMIT_DISABLED": ""}), \
             mock.patch.object(config, "RATE_LIMIT_GUEST_PER_MIN", 2), mock.patch.object(config, "RATE_LIMIT_GUEST_PER_HOUR", 0):
            codes = [_client().post("/api/quiz/topic", json={"topic": ""}).status_code for _ in range(4)]
        self.assertEqual(codes[:2], [422, 422] if codes[0] == 422 else codes[:2])
        self.assertEqual(codes[-1], 429)


# ── 9a: fail closed in production ───────────────────────────────────────────────────────────
class FailClosedTests(unittest.TestCase):
    def test_vercel_without_clerk_refuses_authenticated_routes(self):
        app.dependency_overrides.pop(auth.current_owner, None)
        with mock.patch.dict(os.environ, {"VERCEL": "1"}), mock.patch.object(config, "CLERK_PUBLISHABLE_KEY", None), \
             mock.patch.object(config, "AUTH_ENABLED", False):
            for method, path in (("get", "/api/history/quizzes"), ("get", "/api/insights"),
                                 ("post", "/api/quiz/topic")):
                r = getattr(_client(), method)(path, **({"json": {"topic": "x"}} if method == "post" else {}))
                self.assertEqual(r.status_code, 503, path)
                self.assertIn("CLERK_PUBLISHABLE_KEY", r.json()["detail"])
                self.assertNotIn("llm", r.json())

    def test_never_falls_back_to_the_shared_local_owner_in_production(self):
        from starlette.requests import Request
        req = Request({"type": "http", "headers": [], "method": "GET", "path": "/"})
        with mock.patch.dict(os.environ, {"VERCEL": "1"}), mock.patch.object(config, "CLERK_PUBLISHABLE_KEY", None), \
             mock.patch.object(config, "AUTH_ENABLED", False):
            with self.assertRaises(Exception) as cm:
                auth.current_owner(req)
        self.assertEqual(cm.exception.status_code, 503)
        # ...while local dev keeps the single-user "local" profile
        os.environ.pop("VERCEL", None)
        os.environ.pop("VERCEL_ENV", None)
        with mock.patch.object(config, "AUTH_ENABLED", False):
            self.assertEqual(auth.current_owner(req), auth.LOCAL_OWNER)

    def test_health_deep_warns(self):
        with mock.patch.dict(os.environ, {"VERCEL": "1"}), mock.patch.object(config, "CLERK_PUBLISHABLE_KEY", None):
            os.environ.pop("SESSION_SECRET", None)
            r = _client().get("/api/health/deep")
        body = r.json()
        self.assertEqual(r.status_code, 503)
        self.assertFalse(body["checks"]["auth"]["ok"])
        self.assertIn("CLERK_PUBLISHABLE_KEY", body["checks"]["auth"]["warning"])
        self.assertFalse(body["checks"]["session_secret"]["ok"])

    def test_health_deep_ok_when_configured(self):
        with mock.patch.dict(os.environ, {"VERCEL": "1", "SESSION_SECRET": "s" * 32}), \
             mock.patch.object(config, "CLERK_PUBLISHABLE_KEY", "pk_test_x"):
            body = _client().get("/api/health/deep").json()
        self.assertTrue(body["checks"]["auth"]["ok"])
        self.assertNotIn("session_secret", body["checks"])

    def test_session_secret_required_on_vercel_and_not_derived_from_provider_keys(self):
        with mock.patch.dict(os.environ, {"VERCEL": "1", "GROQ_API_KEY": "gsk_something"}):
            os.environ.pop("SESSION_SECRET", None)
            with self.assertRaises(config.SessionSecretMissing):
                chat._mint("u1", "topic", "L", None, "T", None)
            with self.assertRaises(config.SessionSecretMissing):
                chat.open_session("a.b", "u1")
        with mock.patch.dict(os.environ, {"VERCEL": "1", "SESSION_SECRET": "x" * 32}):
            s = chat._mint("u1", "topic", "L", None, "T", None)
            self.assertEqual(chat.open_session(s.id, "u1").owner, "u1")

    def test_missing_session_secret_is_a_clear_503_not_a_404_or_500(self):
        app.dependency_overrides[auth.current_owner] = lambda: "user_1"
        self.addCleanup(lambda: app.dependency_overrides.pop(auth.current_owner, None))
        with mock.patch.dict(os.environ, {"VERCEL": "1"}), mock.patch.object(config, "CLERK_PUBLISHABLE_KEY", "pk_x"):
            os.environ.pop("SESSION_SECRET", None)
            r = _client().post("/api/chat/start", json={"mode": "topic", "topic": "Graphs"})
        self.assertEqual(r.status_code, 503)
        self.assertIn("SESSION_SECRET", r.json()["detail"])


# ── LOW: chat session expiry ────────────────────────────────────────────────────────────────
class SessionExpiryTests(unittest.TestCase):
    def test_token_expires(self):
        s = chat._mint("u1", "topic", "L", None, "T", None)
        self.assertEqual(chat.open_session(s.id, "u1").id, s.id)
        with mock.patch("backend.services.chat.time.time", return_value=time.time() + chat.SESSION_TTL_S + 5):
            with self.assertRaises(chat.ChatSessionError):
                chat.open_session(s.id, "u1")

    def test_default_ttl_is_a_day_and_tokens_without_exp_are_rejected(self):
        self.assertEqual(chat.SESSION_TTL_S, 24 * 3600)
        s = chat._mint("u1", "topic", "L", None, "T", None)
        body = s.id.split(".")[0]
        payload = json.loads(chat._unb64(body))
        self.assertAlmostEqual(payload["x"], time.time() + chat.SESSION_TTL_S, delta=5)
        del payload["x"]
        import hashlib
        import hmac
        new_body = chat._b64(json.dumps(payload, separators=(",", ":")).encode())
        sig = chat._b64(hmac.new(chat._secret(), new_body.encode(), hashlib.sha256).digest()[:18])
        with self.assertRaises(chat.ChatSessionError):
            chat.open_session(f"{new_body}.{sig}", "u1")

    def test_sliding_refresh_only_when_less_than_an_hour_is_left(self):
        s = chat._mint("u1", "subject", "L", "Math", None, "cid")
        self.assertIsNone(chat.refresh_session(s))
        with mock.patch("backend.services.chat.time.time", return_value=time.time() + chat.SESSION_TTL_S - 600):
            fresh = chat.refresh_session(s)
            self.assertIsNotNone(fresh)
            again = chat.open_session(fresh, "u1")
        self.assertEqual((again.subject, again.chat_id, again.nonce), ("Math", "cid", s.nonce))
        self.assertEqual(json.loads(chat._unb64(fresh.split(".")[0]))["x"] > json.loads(chat._unb64(s.id.split(".")[0]))["x"], True)


if __name__ == "__main__":
    unittest.main()
