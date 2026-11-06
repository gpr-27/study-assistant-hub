"""Retrieval must NEVER hard-fail: when embeddings are unavailable (quota, auth, network,
no keys) the retriever answers from BM25 over the stored chunks, quickly, and recovers
once embeddings come back. Mocked HTTP only."""
import time
import unittest
from unittest import mock

import numpy as np

from backend import gemini_embeddings as ge
from backend import vector_store as vs
from tests.test_vector_store import CHUNKS, DIM, _embed_text, _index


def _resp(status, payload=None):
    r = mock.MagicMock()
    r.status_code = status
    r.json.return_value = payload or {}
    r.text = str(payload or "")
    return r


class LexicalFallbackTests(unittest.TestCase):
    def setUp(self):
        ge.reset_state()
        self.addCleanup(ge.reset_state)

    def _patch_http(self, handler):
        sess = mock.MagicMock()
        sess.post.side_effect = handler
        for p in (mock.patch.object(ge, "_http", lambda: sess),
                  mock.patch.object(ge, "load_keys", lambda: ["key-A", "key-B"]),
                  mock.patch.object(ge.time, "sleep")):
            p.start()
            self.addCleanup(p.stop)
        return sess

    def test_quota_on_every_key_still_returns_relevant_chunks_quickly(self):
        self._patch_http(lambda *a, **k: _resp(429, {"error": {"message": "quota"}}))
        r = _index().as_retriever(search_kwargs={"k": 2})
        t = time.monotonic()
        docs = r.invoke("what is a deadlock between processes")
        self.assertLess(time.monotonic() - t, ge.QUERY_DEADLINE_S + 1.0)
        self.assertEqual(docs[0].page_content, CHUNKS[0]["text"])
        self.assertEqual(docs[0].metadata["retrieval"], "lexical")

    def test_auth_failure_and_network_errors_fall_back_too(self):
        for handler in (lambda *a, **k: _resp(401, {"error": {"message": "x"}}),
                        mock.Mock(side_effect=ConnectionError("down"))):
            ge.reset_state()
            sess = mock.MagicMock()
            sess.post.side_effect = handler
            with mock.patch.object(ge, "_http", lambda: sess), \
                 mock.patch.object(ge, "load_keys", lambda: ["key-A"]), \
                 mock.patch.object(ge.time, "sleep"):
                docs = _index().search("bayes theorem conditional probabilities", 1)
            self.assertEqual(docs[0].page_content, CHUNKS[3]["text"])

    def test_no_keys_configured_means_lexical_not_error(self):
        with mock.patch.object(ge, "load_keys", lambda: []):
            docs = _index().search("paging frames memory", 1)
        self.assertEqual(docs[0].page_content, CHUNKS[1]["text"])

    def test_recovers_to_hybrid_when_embeddings_return(self):
        state = {"ok": False}

        def handler(url, json=None, headers=None, timeout=None):
            if not state["ok"]:
                return _resp(500)
            return _resp(200, {"embedding": {"values": _embed_text("matrix rows columns").tolist()}})
        self._patch_http(handler)
        idx = _index()
        self.assertEqual(idx.search("matrix rows columns", 1)[0].metadata["retrieval"], "lexical")
        state["ok"] = True
        ge.reset_state()                       # clear the short circuit-breaker
        self.assertEqual(idx.search("matrix rows columns", 1)[0].metadata["retrieval"], "hybrid")

    def test_retriever_swallows_load_errors(self):
        r = vs.Retriever(key="user__u__Gone", owner="u", k=3, subject="Gone")
        with mock.patch.object(vs, "_get_index", side_effect=RuntimeError("mongo down")):
            self.assertEqual(r.invoke("anything"), [])

    def test_query_embedding_is_cached_between_agentic_rounds(self):
        sess = self._patch_http(lambda *a, **k: _resp(200, {"embedding": {"values": _embed_text("paging").tolist()}}))
        idx = _index()
        idx.search("paging", 1)
        idx.search("paging", 2)
        self.assertEqual(sess.post.call_count, 1)

    def test_invoke_lexical_never_embeds(self):
        r = _index().as_retriever(search_kwargs={"k": 2})
        with mock.patch.object(ge, "embed_query", side_effect=AssertionError("must not embed")):
            docs = r.invoke_lexical("what is a deadlock")
        self.assertEqual(docs[0].page_content, CHUNKS[0]["text"])
        self.assertEqual(docs[0].metadata["retrieval"], "lexical")

    def test_invoke_timeout_bounds_the_embedding_step(self):
        seen = {}

        def spy(text, *, model=None, dim=None, deadline_s=None):
            seen["deadline"] = deadline_s
            raise ge.GeminiEmbeddingError("slow")
        r = _index().as_retriever(search_kwargs={"k": 1})
        with mock.patch.object(ge, "embed_query", spy):
            docs = r.invoke("paging frames memory", timeout_s=0.8)
        self.assertEqual(seen["deadline"], 0.8)
        self.assertEqual(docs[0].metadata["retrieval"], "lexical")


if __name__ == "__main__":
    unittest.main()
