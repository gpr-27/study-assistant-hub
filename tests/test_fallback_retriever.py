"""Quota-fallback retrieval: when the primary embedding model's quota is spent,
the SAME query is searched against the fallback-model index (query vector and
index switch together — a cross-model query would silently return garbage)."""
import unittest
from unittest import mock

from backend import vector_store
from backend.gemini_embeddings import GeminiQuotaError


class _Store:
    def __init__(self, docs=None, raise_quota=False):
        self.docs = docs or []
        self.raise_quota = raise_quota
        self.queries = []

    def similarity_search(self, query, k):
        self.queries.append((query, k))
        if self.raise_quota:
            raise GeminiQuotaError("quota")
        return self.docs


class QuotaFallbackRetrieverTests(unittest.TestCase):
    def setUp(self):
        vector_store._primary_quota_until = 0.0   # reset the cooldown between tests

    def test_primary_success_never_loads_fallback(self):
        primary = _Store(docs=["d1"])
        r = vector_store._QuotaFallbackRetriever("Math", primary, k=4)
        with mock.patch.object(vector_store, "_load_fallback_store",
                               side_effect=AssertionError("must not be called")):
            self.assertEqual(r.invoke("q"), ["d1"])
        self.assertEqual(primary.queries, [("q", 4)])

    def test_quota_switches_to_fallback_index(self):
        primary = _Store(raise_quota=True)
        fallback = _Store(docs=["fb-doc"])
        r = vector_store._QuotaFallbackRetriever("Math", primary, k=3)
        with mock.patch.object(vector_store, "_load_fallback_store", return_value=fallback):
            self.assertEqual(r.invoke("q"), ["fb-doc"])
        self.assertEqual(fallback.queries, [("q", 3)])

    def test_cooldown_skips_primary_after_quota_failure(self):
        primary = _Store(raise_quota=True)
        fallback = _Store(docs=["fb"])
        r = vector_store._QuotaFallbackRetriever("Math", primary, k=2)
        with mock.patch.object(vector_store, "_load_fallback_store", return_value=fallback):
            r.invoke("q1")            # primary probed once, quota -> cooldown set
            r.invoke("q2")            # within cooldown: straight to fallback
            r.invoke("q3")
        self.assertEqual(len(primary.queries), 1)    # primary probed exactly once
        self.assertEqual(len(fallback.queries), 3)

    def test_quota_reraised_when_no_fallback_exists(self):
        primary = _Store(raise_quota=True)
        r = vector_store._QuotaFallbackRetriever("UserSubj", primary, k=3)
        with mock.patch.object(vector_store, "_load_fallback_store", return_value=None):
            with self.assertRaises(GeminiQuotaError):
                r.invoke("q")


if __name__ == "__main__":
    unittest.main()
