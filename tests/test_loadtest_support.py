"""Load-test support: the test-only provider redirect, the fake providers, and the per-instance PDF text cache."""
from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from pypdf import PdfWriter
from starlette.testclient import TestClient

from backend import documents, gemini_embeddings, llm_chain

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import fake_providers  # noqa: E402


def _pdf(tag: int = 0) -> bytes:
    w = PdfWriter()
    w.add_blank_page(width=200 + tag, height=200)
    buf = io.BytesIO()
    w.write(buf)
    return buf.getvalue()


class TestBaseUrlHook(unittest.TestCase):
    def test_redirects_all_three_providers_and_embeddings(self):
        with mock.patch.dict(os.environ, {"LLM_CHAIN_TEST_BASE_URL": "http://127.0.0.1:8890/"}, clear=False):
            os.environ.pop("VERCEL", None)
            self.assertEqual(llm_chain._anthropic_url(), "http://127.0.0.1:8890/v1/messages")
            self.assertEqual(llm_chain._groq_url(), "http://127.0.0.1:8890/openai/v1/chat/completions")
            self.assertEqual(llm_chain._gemini_base(), "http://127.0.0.1:8890/v1beta/models")
            self.assertEqual(gemini_embeddings._api_root(), "http://127.0.0.1:8890/v1beta")

    def test_ignored_on_vercel(self):
        with mock.patch.dict(os.environ, {"LLM_CHAIN_TEST_BASE_URL": "http://evil.example", "VERCEL": "1"}):
            self.assertEqual(llm_chain._anthropic_url(), llm_chain.ANTHROPIC_URL)
            self.assertEqual(llm_chain._groq_url(), llm_chain.GROQ_URL)
            self.assertEqual(llm_chain._gemini_base(), llm_chain.GEMINI_BASE)
            self.assertTrue(gemini_embeddings._api_root().startswith("https://generativelanguage.googleapis.com"))

    def test_default_urls_when_unset(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("LLM_CHAIN_TEST_BASE_URL", None)
            self.assertEqual(llm_chain._anthropic_url(), llm_chain.ANTHROPIC_URL)


class TestFakeProviders(unittest.TestCase):
    def setUp(self):
        fake_providers._reset()
        fake_providers.CFG.update(latency_ms=0.0, jitter_ms=0.0, token_ms=0.0, anthropic_fail=0.0,
                                  groq_fail=0.0, gemini_fail=0.0, timeout_rate=0.0)
        self.client = TestClient(fake_providers.app)

    def test_replies_parse_for_each_feature(self):
        mr = fake_providers.make_reply
        quiz = json.loads(mr("teacher. Reply with strictly valid JSON only.",
                             "Create exactly 3 multiple choice questions from the content below. "
                             "four options keyed A, B, C and D", True))
        self.assertEqual(len(quiz["questions"]), 3)
        cards = json.loads(mr("flashcards JSON", "Create 4 focused flashcards from this content.", True))
        self.assertEqual(len(cards["cards"]), 4)
        self.assertTrue(json.loads(mr("", "Create a mind map about: x. JSON", True))["branches"])
        exam = json.loads(mr("", "Create a mock exam ... exactly 2 multiple choice questions and exactly 1 short-answer", True))
        self.assertEqual([q["type"] for q in exam["questions"]], ["mcq", "mcq", "short"])

    def test_three_wire_formats_through_the_real_chain(self):
        """llm_chain.chat() against the fake over each provider's real path layout."""
        transport = _asgi_transport(fake_providers.app)
        llm_chain.reset_state()
        llm_chain._configure_transports(sync=transport)
        try:
            env = {"LLM_CHAIN_TEST_BASE_URL": "http://fake", "ANTHROPIC_API_KEY": "k1", "GROQ_API_KEY": "k2",
                   "GEMINI_API_KEY": "k3"}
            with mock.patch.dict(os.environ, env):
                os.environ.pop("VERCEL", None)
                r = llm_chain.chat("sys", [{"role": "user", "content": "hello"}])
                self.assertEqual((r.provider, r.step), ("anthropic", 1))
                fake_providers.CFG.update(anthropic_fail=1.0)
                r = llm_chain.chat("sys", [{"role": "user", "content": "hello"}])
                self.assertEqual((r.provider, r.step), ("groq", 2))
                fake_providers.CFG.update(groq_fail=1.0)
                r = llm_chain.chat("sys", [{"role": "user", "content": "hello"}], json_mode=True)
                self.assertEqual(r.provider, "gemini")
                json.loads(r.text)
                self.assertEqual(fake_providers.STATS["injected"]["anthropic"], 2)
        finally:
            llm_chain._configure_transports()
            llm_chain.reset_state()

    def test_streaming_shapes(self):
        transport = _asgi_transport(fake_providers.app)
        llm_chain.reset_state()
        llm_chain._configure_transports(sync=transport)
        try:
            for fail in ({}, {"anthropic_fail": 1.0}, {"anthropic_fail": 1.0, "groq_fail": 1.0}):
                llm_chain.reset_state()
                fake_providers.CFG.update(anthropic_fail=0.0, groq_fail=0.0)
                fake_providers.CFG.update(fail)
                env = {"LLM_CHAIN_TEST_BASE_URL": "http://fake", "ANTHROPIC_API_KEY": "k1", "GROQ_API_KEY": "k2",
                       "GEMINI_API_KEY": "k3"}
                with mock.patch.dict(os.environ, env):
                    os.environ.pop("VERCEL", None)
                    items = list(llm_chain.stream_chat("sys", [{"role": "user", "content": "hi"}]))
                self.assertGreater(len(items), 2, fail)
                self.assertIsInstance(items[-1], llm_chain.ChatResult)
                self.assertEqual("".join(i for i in items[:-1] if isinstance(i, str)).strip(), items[-1].text)
        finally:
            llm_chain._configure_transports()
            llm_chain.reset_state()

    def test_embeddings_endpoint(self):
        r = self.client.post("/v1beta/models/gemini-embedding-2:embedContent",
                             json={"content": {"parts": [{"text": "x"}]}, "outputDimensionality": 16})
        self.assertEqual(len(r.json()["embedding"]["values"]), 16)
        r = self.client.post("/v1beta/models/gemini-embedding-2:batchEmbedContents",
                             json={"requests": [{"content": {"parts": [{"text": "a"}]}}] * 3})
        self.assertEqual(len(r.json()["embeddings"]), 3)


def _asgi_transport(app):
    """Sync httpx transport that calls a Starlette app in-process (via TestClient)."""
    import httpx

    tc = TestClient(app, base_url="http://fake")

    class _T(httpx.BaseTransport):
        def handle_request(self, request: httpx.Request) -> httpx.Response:
            r = tc.request(request.method, str(request.url), content=request.content, headers=dict(request.headers))
            return httpx.Response(r.status_code, headers=r.headers, content=r.content)

    return _T()


class TestPdfTextCache(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.docs = Path(self._tmp.name)
        (self.docs / "Subj").mkdir()
        self.path = self.docs / "Subj" / "a.pdf"
        self.path.write_bytes(_pdf(1))
        p = mock.patch.object(documents, "DOCUMENTS_DIR", self.docs)
        p.start()
        self.addCleanup(p.stop)
        self.addCleanup(self._tmp.cleanup)
        documents.clear_text_cache()
        self.addCleanup(documents.clear_text_cache)

    def test_parsed_once_then_cached(self):
        with mock.patch.object(documents, "_pdf_reader", wraps=documents._pdf_reader) as rd:
            documents.read_pdf("Subj", "a.pdf")
            documents.read_pdf("Subj", "a.pdf")
        self.assertEqual(rd.call_count, 1)

    def test_replaced_file_is_reparsed(self):
        with mock.patch.object(documents, "_pdf_reader", wraps=documents._pdf_reader) as rd:
            documents.read_pdf("Subj", "a.pdf")
            self.path.write_bytes(_pdf(7))
            os.utime(self.path, ns=(time.time_ns() + 10**9, time.time_ns() + 10**9))
            documents.read_pdf("Subj", "a.pdf")
        self.assertEqual(rd.call_count, 2)

    def test_concurrent_first_reads_parse_once(self):
        calls = []
        real = documents._pdf_reader

        def slow(src):
            calls.append(1)
            time.sleep(0.15)
            return real(src)

        with mock.patch.object(documents, "_pdf_reader", slow):
            ts = [threading.Thread(target=documents.read_pdf, args=("Subj", "a.pdf")) for _ in range(8)]
            [t.start() for t in ts]
            [t.join() for t in ts]
        self.assertEqual(len(calls), 1)

    def test_cache_is_bounded(self):
        with mock.patch.object(documents, "_TEXT_CACHE_MAX_ENTRIES", 3):
            for i in range(10):
                documents._text_cache_put(("k", i, 0), "x" * 10)
        self.assertLessEqual(len(documents._text_cache), 3)
        with mock.patch.object(documents, "_TEXT_CACHE_MAX_CHARS", 25):
            documents.clear_text_cache()
            for i in range(10):
                documents._text_cache_put(("k", i, 0), "x" * 10)
        self.assertLessEqual(documents._text_cache_chars, 25)


class TestSaveIsResilient(unittest.TestCase):
    def test_pymongo_failure_returns_none_instead_of_raising(self):
        from pymongo.errors import WaitQueueTimeoutError

        from backend import db

        coll = mock.MagicMock()
        coll.insert_one.side_effect = WaitQueueTimeoutError("pool busy")
        fake_db = {"quizzes": coll}
        with mock.patch.object(db, "_get_db", return_value=fake_db):
            self.assertIsNone(db.save("quizzes", "guest_x", {"title": "t"}))

    def test_non_mongo_errors_still_raise(self):
        from backend import db

        coll = mock.MagicMock()
        coll.insert_one.side_effect = ValueError("boom")
        with mock.patch.object(db, "_get_db", return_value={"quizzes": coll}):
            with self.assertRaises(ValueError):
                db.save("quizzes", "guest_x", {"title": "t"})


if __name__ == "__main__":
    unittest.main()
