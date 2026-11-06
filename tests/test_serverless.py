"""Serverless-readiness tests: health, cron auth, error handlers, middleware headers, JWKS
caching. Hermetic: no Mongo, no Clerk, no network, no LLM."""
import os
import unittest
from unittest import mock

from fastapi.testclient import TestClient

from backend import auth, config, db
from backend.main import app


def _client(**kw):
    return TestClient(app, raise_server_exceptions=False, **kw)


class HealthTests(unittest.TestCase):
    def test_health_is_cheap_and_has_observability_headers(self):
        # It must not touch the DB or LLM at all.
        with mock.patch.object(db, "_get_db", side_effect=AssertionError("DB touched")):
            resp = _client().get("/api/health")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {"status": "ok"})
        self.assertTrue(resp.headers["x-request-id"])
        self.assertIn("total;dur=", resp.headers["server-timing"])
        self.assertEqual(resp.headers["cache-control"], "no-store")
        self.assertEqual(resp.headers["x-content-type-options"], "nosniff")

    def test_request_id_is_echoed_when_sane_and_replaced_when_not(self):
        c = _client()
        self.assertEqual(c.get("/api/health", headers={"X-Request-ID": "abc-123"}).headers["x-request-id"], "abc-123")
        bad = c.get("/api/health", headers={"X-Request-ID": "bad id with spaces"}).headers["x-request-id"]
        self.assertNotEqual(bad, "bad id with spaces")

    def test_deep_health_without_persistence(self):
        resp = _client().get("/api/health/deep")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["status"], "ok")
        self.assertIsNone(body["checks"]["mongo"]["ok"])

    def test_deep_health_reports_mongo_failure_as_503(self):
        with mock.patch.object(config, "PERSISTENCE_ENABLED", True), \
             mock.patch.object(db, "ping", side_effect=RuntimeError("down")):
            resp = _client().get("/api/health/deep")
        self.assertEqual(resp.status_code, 503)
        self.assertFalse(resp.json()["checks"]["mongo"]["ok"])
        self.assertNotIn("down", resp.text)  # exception text is not echoed, only its type

    def test_deep_health_never_leaks_key_like_strings(self):
        fake = {"steps": [{"name": "Groq key 1", "configured": True}], "leak": "gsk_" + "x" * 30}
        with mock.patch.object(config, "chain_status", return_value=fake):
            body = _client().get("/api/health/deep").json()
        self.assertNotIn("gsk_", str(body))


class CronAuthTests(unittest.TestCase):
    def test_rejects_without_secret_configured(self):
        with mock.patch.dict(os.environ, {"CRON_SECRET": ""}):
            self.assertEqual(_client().get("/api/cron/janitor").status_code, 401)
            self.assertEqual(
                _client().get("/api/cron/janitor", headers={"Authorization": "Bearer "}).status_code, 401)

    def test_rejects_wrong_or_missing_token(self):
        with mock.patch.dict(os.environ, {"CRON_SECRET": "s3cret-value"}):
            self.assertEqual(_client().get("/api/cron/janitor").status_code, 401)
            r = _client().get("/api/cron/janitor", headers={"Authorization": "Bearer nope"})
            self.assertEqual(r.status_code, 401)

    def test_accepts_correct_token_and_runs_janitor(self):
        with mock.patch.dict(os.environ, {"CRON_SECRET": "s3cret-value"}), \
             mock.patch.object(config, "PERSISTENCE_ENABLED", True), \
             mock.patch.object(db, "ensure_indexes", return_value={"skipped": "up to date"}) as ei, \
             mock.patch.object(db, "purge_orphan_guest_data", return_value=7) as purge, \
             mock.patch("backend.services.subjects.purge_orphan_guest_uploads", return_value=3) as up:
            r = _client().get("/api/cron/janitor", headers={"Authorization": "Bearer s3cret-value"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["removed"], 7)
        self.assertEqual(r.json()["uploads_removed"], 3)
        self.assertLessEqual(up.call_args.kwargs["budget_s"], 30)
        ei.assert_called_once()
        purge.assert_called_once()

    def test_no_op_when_persistence_disabled(self):
        with mock.patch.dict(os.environ, {"CRON_SECRET": "s3cret-value"}):
            r = _client().get("/api/cron/janitor", headers={"Authorization": "Bearer s3cret-value"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["removed"], 0)


class ErrorHandlerTests(unittest.TestCase):
    """Add throwaway routes that raise, then remove them."""

    def _with_route(self, exc):
        def boom():
            raise exc

        app.add_api_route("/api/_test_boom", boom, methods=["GET"])
        self.addCleanup(lambda: app.router.routes.pop())
        return _client().get("/api/_test_boom")

    def test_llm_chain_error_is_503_with_label(self):
        from backend.llm_chain import LLMChainError
        resp = self._with_route(LLMChainError("all providers failed"))
        self.assertEqual(resp.status_code, 503)
        body = resp.json()
        self.assertEqual(body["llm"]["label"], "Model: none (all providers failed)")
        self.assertIn("detail", body)

    def test_unhandled_exception_is_json_500_with_request_id(self):
        resp = self._with_route(ValueError("kaboom-secret-detail"))
        self.assertEqual(resp.status_code, 500)
        body = resp.json()
        self.assertEqual(body["detail"], "Internal server error.")
        self.assertTrue(body["request_id"])
        self.assertNotIn("kaboom", resp.text)

    def test_pymongo_failure_is_clean_503(self):
        from pymongo.errors import ServerSelectionTimeoutError
        resp = self._with_route(ServerSelectionTimeoutError("mongodb://user:pw@host unreachable"))
        self.assertEqual(resp.status_code, 503)
        self.assertNotIn("pw@", resp.text)

    def test_http_exceptions_still_pass_through(self):
        self.assertEqual(_client().get("/api/definitely-not-a-route").status_code, 404)


class ConfigCachingTests(unittest.TestCase):
    def test_public_config_is_cdn_cacheable_but_browser_revalidated(self):
        # Anonymous + secret-free (public Clerk key only): the CDN may hold it; browsers revalidate.
        r = _client().get("/api/config")
        self.assertEqual(r.status_code, 200)
        cc = r.headers["cache-control"]
        self.assertIn("s-maxage=60", cc)
        self.assertIn("stale-while-revalidate=300", cc)
        self.assertIn("max-age=0", cc)


class SubjectsShapeTests(unittest.TestCase):
    def test_subjects_endpoint_keeps_its_envelope(self):
        r = _client().get("/api/subjects")
        self.assertEqual(r.status_code, 200)
        self.assertIsInstance(r.json()["subjects"], list)


class JwksResilienceTests(unittest.TestCase):
    def setUp(self):
        auth._token_cache.clear()
        auth._jwks_failed_until = 0.0

    def test_jwks_outage_backs_off_instead_of_retrying_every_request(self):
        import jwt

        client = mock.MagicMock()
        client.get_signing_keys.side_effect = jwt.PyJWKClientConnectionError("down")
        with mock.patch.object(auth, "_jwks_client", return_value=client), \
             mock.patch.object(jwt, "get_unverified_header", return_value={"kid": "k"}):
            for tok in ("a.b.c", "d.e.f", "g.h.i"):
                with self.assertRaises(auth.ClerkUnavailable):   # -> retryable 503 at the route
                    auth._verify_clerk_token(tok)
        self.assertEqual(client.get_signing_keys.call_count, 1)  # then fast-fail for 30 s

    def test_jwks_client_is_configured_for_serverless(self):
        auth._jwks_client.cache_clear()
        pk = "pk_test_" + __import__("base64").b64encode(b"clerk.example.com$").decode()
        with mock.patch.object(config, "CLERK_PUBLISHABLE_KEY", pk):
            client = auth._jwks_client()
        auth._jwks_client.cache_clear()
        self.assertEqual(client.timeout, 3)
        self.assertGreaterEqual(client.jwk_set_cache.lifespan, 3600)

    def test_verified_token_is_cached_until_exp(self):
        import time as _t
        client = mock.MagicMock()
        claims = {"sub": "user_1", "exp": _t.time() + 60}
        with mock.patch.object(auth, "_jwks_client", return_value=client), \
             mock.patch.object(auth, "_signing_key", return_value=mock.MagicMock(key="k")), \
             mock.patch("jwt.decode", return_value=claims) as dec:
            self.assertEqual(auth._verify_clerk_token("tok"), "user_1")
            self.assertEqual(auth._verify_clerk_token("tok"), "user_1")
        self.assertEqual(dec.call_count, 1)


class DbClientTests(unittest.TestCase):
    def test_pool_settings_are_serverless_safe(self):
        kw = db._POOL_KWARGS
        self.assertLessEqual(kw["maxPoolSize"], 10)
        self.assertLessEqual(kw["minPoolSize"], 1)      # one warm connection per instance (see db.py note)
        self.assertLessEqual(kw["serverSelectionTimeoutMS"], 3000)
        self.assertTrue(kw["retryWrites"])

    def test_client_is_created_once_without_network(self):
        created = []

        class FakeClient:
            def __init__(self, *a, **k):
                created.append(k)

            def __getitem__(self, name):
                return mock.MagicMock()

        with mock.patch.object(config, "PERSISTENCE_ENABLED", True), \
             mock.patch.object(config, "MONGODB_URI", "mongodb://localhost:1"), \
             mock.patch.object(db, "_db", None), mock.patch.object(db, "_client", None), \
             mock.patch.object(db, "_init_failed", False), \
             mock.patch.object(db, "_indexes_kicked", True), \
             mock.patch("pymongo.MongoClient", FakeClient):
            a = db._get_db()
            b = db._get_db()
        self.assertIs(a, b)
        self.assertEqual(len(created), 1)


if __name__ == "__main__":
    unittest.main()
