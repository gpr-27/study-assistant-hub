"""Gemini embedding client: key rotation on 429/401/5xx, bounded time, LRU cache,
task-format handling. All HTTP is mocked — no network, no real keys."""
import time
import unittest
from unittest import mock

import numpy as np

from backend import gemini_embeddings as ge


def _resp(status, payload=None, text=""):
    r = mock.MagicMock()
    r.status_code = status
    r.json.return_value = payload if payload is not None else {}
    r.text = text or str(payload or "")
    return r


def _vec(n=8, seed=1):
    rng = np.random.default_rng(seed)
    return rng.normal(size=n).tolist()


class _Base(unittest.TestCase):
    keys = ["key-A", "key-B"]

    def setUp(self):
        ge.reset_state()
        p = mock.patch.object(ge, "load_keys", lambda: list(self.keys))
        p.start()
        self.addCleanup(p.stop)
        self.addCleanup(ge.reset_state)
        s = mock.patch.object(ge.time, "sleep")
        s.start()
        self.addCleanup(s.stop)

    def _session(self, handler):
        sess = mock.MagicMock()
        sess.post.side_effect = handler
        p = mock.patch.object(ge, "_http", lambda: sess)
        p.start()
        self.addCleanup(p.stop)
        return sess


class FormatTests(unittest.TestCase):
    def test_embedding_2_encodes_task_in_text_not_taskType(self):
        text, task = ge.format_for("gemini-embedding-2", "hi", "query")
        self.assertEqual(task, None)
        self.assertTrue(text.startswith("task: search result | query:"))
        text, task = ge.format_for("models/gemini-embedding-2", "hi", "document")
        self.assertEqual((text, task), ("title: none | text: hi", None))

    def test_embedding_001_uses_taskType(self):
        self.assertEqual(ge.format_for("gemini-embedding-001", "hi", "query"), ("hi", "RETRIEVAL_QUERY"))
        self.assertEqual(ge.format_for("gemini-embedding-001", "hi", "document"), ("hi", "RETRIEVAL_DOCUMENT"))


class RotationTests(_Base):
    def test_rotates_to_second_key_on_429(self):
        def handler(url, json=None, headers=None, timeout=None):
            if headers["x-goog-api-key"] == "key-A":
                return _resp(429, {"error": {"message": "quota"}})
            return _resp(200, {"embedding": {"values": _vec()}})
        sess = self._session(handler)
        out = ge.embed_query("hello", dim=8)
        self.assertEqual(out.shape, (8,))
        self.assertAlmostEqual(float(np.linalg.norm(out)), 1.0, places=5)
        used = {c.kwargs["headers"]["x-goog-api-key"] for c in sess.post.call_args_list}
        self.assertIn("key-B", used)

    def test_dead_key_401_is_skipped_afterwards(self):
        calls = []

        def handler(url, json=None, headers=None, timeout=None):
            k = headers["x-goog-api-key"]
            calls.append(k)
            return _resp(401, {"error": {"message": "bad"}}) if k == "key-A" else _resp(200, {"embedding": {"values": _vec()}})
        self._session(handler)
        for q in ("one", "two", "three", "four"):
            ge.embed_query(q, dim=8)
        self.assertLessEqual(calls.count("key-A"), 1)      # contacted at most once, then skipped

    def test_5xx_falls_through_to_other_key(self):
        def handler(url, json=None, headers=None, timeout=None):
            return _resp(503, {}) if headers["x-goog-api-key"] == "key-A" else _resp(200, {"embedding": {"values": _vec()}})
        self._session(handler)
        self.assertEqual(ge.embed_query("x", dim=8).shape, (8,))

    def test_all_keys_quota_raises_quota_error_fast(self):
        self._session(lambda *a, **k: _resp(429, {"error": {"message": "quota"}}))
        t = time.monotonic()
        with self.assertRaises(ge.GeminiQuotaError):
            ge.embed_query("x", dim=8)
        self.assertLess(time.monotonic() - t, ge.QUERY_DEADLINE_S + 1)

    def test_no_keys_raises(self):
        with mock.patch.object(ge, "load_keys", lambda: []):
            with self.assertRaises(ge.GeminiEmbeddingError):
                ge.embed_query("x")

    def test_bad_request_is_not_retried(self):
        sess = self._session(lambda *a, **k: _resp(400, {"error": {"message": "bad input"}}))
        with self.assertRaises(ge.GeminiEmbeddingError):
            ge.embed_query("x", dim=8)
        self.assertEqual(sess.post.call_count, 1)

    def test_key_material_never_in_error_text(self):
        self._session(lambda *a, **k: _resp(500, {}, text="oops key-A key-B"))
        with self.assertRaises(ge.GeminiEmbeddingError) as cm:
            ge.embed_query("x", dim=8)
        self.assertNotIn("key-A", str(cm.exception))


class QueryCacheTests(_Base):
    def test_second_identical_query_hits_cache(self):
        sess = self._session(lambda *a, **k: _resp(200, {"embedding": {"values": _vec()}}))
        a = ge.embed_query("same", dim=8)
        b = ge.embed_query("same", dim=8)
        self.assertEqual(sess.post.call_count, 1)
        np.testing.assert_array_equal(a, b)

    def test_failure_trips_breaker_so_next_query_skips_network(self):
        sess = self._session(lambda *a, **k: _resp(500, {}))
        with self.assertRaises(ge.GeminiEmbeddingError):
            ge.embed_query("a", dim=8)
        n = sess.post.call_count
        with self.assertRaises(ge.GeminiEmbeddingError):
            ge.embed_query("b", dim=8)
        self.assertEqual(sess.post.call_count, n)


class DocumentBatchTests(_Base):
    def test_batches_and_normalises(self):
        def handler(url, json=None, headers=None, timeout=None):
            n = len(json["requests"])
            for r in json["requests"]:
                self.assertIn("title: none | text:", r["content"]["parts"][0]["text"])
                self.assertEqual(r["outputDimensionality"], 8)
            return _resp(200, {"embeddings": [{"values": _vec(8, i)} for i in range(n)]})
        sess = self._session(handler)
        mat = ge.embed_documents([f"t{i}" for i in range(ge.BATCH_SIZE + 5)], dim=8)
        self.assertEqual(mat.shape, (ge.BATCH_SIZE + 5, 8))
        self.assertEqual(sess.post.call_count, 2)
        np.testing.assert_allclose(np.linalg.norm(mat, axis=1), 1.0, atol=1e-5)

    def test_iter_stops_cleanly_when_quota_is_spent(self):
        self._session(lambda *a, **k: _resp(429, {"error": {"message": "quota"}}))
        self.assertEqual(list(ge.iter_embed_documents(["a", "b"], dim=8, budget_s=3.0)), [])


if __name__ == "__main__":
    unittest.main()
