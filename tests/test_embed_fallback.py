"""Gemini embedding key-fallback: when the primary key's quota is spent, the
request transparently retries on GEMINI_API_KEY_FALLBACK (same model — vectors
stay compatible with the existing FAISS indexes)."""
import unittest
from unittest import mock

from backend.gemini_embeddings import GeminiEmbeddings, GeminiQuotaError


def _resp(status, payload=None, text=""):
    r = mock.MagicMock()
    r.status_code = status
    r.json.return_value = payload if payload is not None else {}
    r.text = text
    return r


class KeyFallbackTests(unittest.TestCase):
    def _client(self, fallback="key-B"):
        return GeminiEmbeddings(model="gemini-embedding-2", api_key="key-A",
                                fallback_api_key=fallback)

    def test_falls_back_to_second_key_on_quota(self):
        emb = self._client()
        vec = {"embedding": {"values": [0.1, 0.2]}}
        def post(url, json=None, timeout=None):
            return _resp(200, vec) if "key=key-B" in url else _resp(429, {}, "quota")
        with mock.patch.object(emb._session, "post", side_effect=post) as p, \
             mock.patch("backend.gemini_embeddings.time.sleep"):
            out = emb.embed_query("hello")
        self.assertEqual(out, [0.1, 0.2])
        urls = [c.args[0] for c in p.call_args_list]
        self.assertTrue(any("key=key-A" in u for u in urls))   # primary tried first
        self.assertTrue(any("key=key-B" in u for u in urls))   # fallback used

    def test_quota_error_still_raised_without_fallback(self):
        emb = self._client(fallback=None)
        with mock.patch.object(emb._session, "post", return_value=_resp(429, {}, "quota")), \
             mock.patch("backend.gemini_embeddings.time.sleep"):
            with self.assertRaises(GeminiQuotaError):
                emb.embed_query("hello")

    def test_primary_success_never_touches_fallback(self):
        emb = self._client()
        vec = {"embedding": {"values": [1.0]}}
        with mock.patch.object(emb._session, "post", return_value=_resp(200, vec)) as p:
            emb.embed_query("hello")
        self.assertTrue(all("key=key-A" in c.args[0] for c in p.call_args_list))


if __name__ == "__main__":
    unittest.main()
