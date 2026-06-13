"""Tests for the chat overview service (NotebookLM-style empty-state card).

No network/Groq: ChatGroq is mocked to return a queued reply, and the document
helpers (list_pdf_names / read_pdf) are mocked. Covers parsing {overview,questions}
from the model, clamping questions to 3, per-subject/topic caching (a 2nd call must
NOT re-invoke the LLM), and the best-effort degrade-to-empty contract when the LLM
raises. The endpoint test registers a subject session directly and asserts the
ChatOverviewResponse shape, with build_overview mocked so no model runs.
"""
import unittest
from unittest import mock

from backend.services import overview


class _Reply:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    """Returns one queued reply; records how many times it was invoked."""

    def __init__(self, reply):
        self._reply = reply
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        return _Reply(self._reply)


class BuildOverviewTests(unittest.TestCase):
    def setUp(self):
        # Each test starts from an empty cache so keys don't leak between tests.
        overview._cache.clear()

    def _patch_llm(self, llm):
        """Patch ChatGroq -> our fake and the Groq key gate so no key is needed."""
        return (
            mock.patch.object(overview, "ChatGroq", return_value=llm),
            mock.patch.object(overview.config, "require_groq_key", lambda: None),
        )

    def test_parses_overview_and_questions_from_subject(self):
        llm = FakeLLM(
            '{"overview": "Covers **trees** and **graphs**.", '
            '"questions": ["What is a BST?", "How does DFS work?"]}'
        )
        p_llm, p_key = self._patch_llm(llm)
        with p_llm, p_key, \
             mock.patch.object(overview, "list_pdf_names", return_value=["a.pdf", "b.pdf"]), \
             mock.patch.object(overview, "read_pdf", return_value="some content"):
            card = overview.build_overview(subject="DSA", owner="u1")

        self.assertEqual(card["label"], "DSA")
        self.assertEqual(card["sources"], 2)                 # total PDF count
        self.assertEqual(card["overview"], "Covers **trees** and **graphs**.")
        self.assertEqual(card["questions"], ["What is a BST?", "How does DFS work?"])
        self.assertEqual(len(llm.calls), 1)

    def test_clamps_questions_to_three(self):
        llm = FakeLLM(
            '{"overview": "o", "questions": ["q1", "q2", "q3", "q4", "q5"]}'
        )
        p_llm, p_key = self._patch_llm(llm)
        with p_llm, p_key, \
             mock.patch.object(overview, "list_pdf_names", return_value=["a.pdf"]), \
             mock.patch.object(overview, "read_pdf", return_value="content"):
            card = overview.build_overview(subject="DSA", owner="u1")
        self.assertEqual(card["questions"], ["q1", "q2", "q3"])

    def test_topic_mode_has_zero_sources_and_skips_pdfs(self):
        llm = FakeLLM('{"overview": "About **recursion**.", "questions": ["q1"]}')
        p_llm, p_key = self._patch_llm(llm)
        with p_llm, p_key, \
             mock.patch.object(overview, "list_pdf_names") as names, \
             mock.patch.object(overview, "read_pdf") as read:
            card = overview.build_overview(topic="Recursion", owner="u1")
        self.assertEqual(card["label"], "Recursion")
        self.assertEqual(card["sources"], 0)
        self.assertEqual(card["overview"], "About **recursion**.")
        names.assert_not_called()   # topic mode never reads PDFs
        read.assert_not_called()

    def test_caches_result_second_call_does_not_reinvoke_llm(self):
        llm = FakeLLM('{"overview": "cached", "questions": ["q1"]}')
        p_llm, p_key = self._patch_llm(llm)
        with p_llm, p_key, \
             mock.patch.object(overview, "list_pdf_names", return_value=["a.pdf"]), \
             mock.patch.object(overview, "read_pdf", return_value="content"):
            first = overview.build_overview(subject="DSA", owner="u1")
            second = overview.build_overview(subject="DSA", owner="u1")

        self.assertEqual(first, second)
        self.assertEqual(len(llm.calls), 1)   # 2nd call served from cache, no LLM

    def test_refresh_bypasses_cache(self):
        llm = FakeLLM('{"overview": "o", "questions": ["q1"]}')
        p_llm, p_key = self._patch_llm(llm)
        with p_llm, p_key, \
             mock.patch.object(overview, "list_pdf_names", return_value=["a.pdf"]), \
             mock.patch.object(overview, "read_pdf", return_value="content"):
            overview.build_overview(subject="DSA", owner="u1")
            overview.build_overview(subject="DSA", owner="u1", refresh=True)
        self.assertEqual(len(llm.calls), 2)   # refresh re-invoked the LLM

    def test_degrades_to_empty_when_llm_raises(self):
        boom = FakeLLM("")
        boom.invoke = mock.Mock(side_effect=RuntimeError("rate limited"))
        p_llm, p_key = self._patch_llm(boom)
        with p_llm, p_key, \
             mock.patch.object(overview, "list_pdf_names", return_value=["a.pdf", "b.pdf"]), \
             mock.patch.object(overview, "read_pdf", return_value="content"):
            card = overview.build_overview(subject="DSA", owner="u1")

        # Best-effort contract: never raises; empty overview/questions.
        self.assertEqual(card["overview"], "")
        self.assertEqual(card["questions"], [])
        # sources is still computed from the PDF count before the failure.
        self.assertEqual(card["sources"], 2)

    def test_empty_failure_is_retried_not_cached(self):
        # A transient failure (e.g. rate limit) must NOT be cached — otherwise the
        # card would stay empty forever. The next open retries the generation.
        boom = FakeLLM("")
        boom.invoke = mock.Mock(side_effect=RuntimeError("boom"))
        p_llm, p_key = self._patch_llm(boom)
        with p_llm, p_key, \
             mock.patch.object(overview, "list_pdf_names", return_value=["a.pdf"]), \
             mock.patch.object(overview, "read_pdf", return_value="content"):
            overview.build_overview(subject="DSA", owner="u1")
            overview.build_overview(subject="DSA", owner="u1")
        self.assertEqual(boom.invoke.call_count, 2)   # retried — empty was not cached

    def test_successful_result_is_cached(self):
        llm = FakeLLM('{"overview": "ok", "questions": ["q1"]}')
        p_llm, p_key = self._patch_llm(llm)
        with p_llm, p_key, \
             mock.patch.object(overview, "list_pdf_names", return_value=["a.pdf"]), \
             mock.patch.object(overview, "read_pdf", return_value="content"):
            overview.build_overview(subject="DSA", owner="u1")
            overview.build_overview(subject="DSA", owner="u1")
        self.assertEqual(len(llm.calls), 1)   # second call served from cache


class OverviewEndpointTests(unittest.TestCase):
    """GET /api/chat/overview/{session_id}: shape + unknown-session error."""

    def setUp(self):
        from fastapi.testclient import TestClient
        from backend.main import app

        self.client = TestClient(app)

    def _register_subject_session(self):
        from backend.services import chat
        return chat._register({
            "mode": "subject", "retriever": object(), "llm": object(), "history": [],
            "label": "Subject · DSA", "topic": None, "subject": "DSA",
            "owner": "local", "chat_id": None, "autonomous": False,
        })

    def test_overview_returns_expected_shape(self):
        from backend.services import overview as overview_mod

        sid = self._register_subject_session()
        card = {
            "label": "DSA",
            "sources": 2,
            "overview": "Covers **trees**.",
            "questions": ["What is a BST?", "How does DFS work?"],
        }
        with mock.patch.object(overview_mod, "build_overview", return_value=card) as build:
            resp = self.client.get(f"/api/chat/overview/{sid}")

        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["label"], "DSA")
        self.assertEqual(body["sources"], 2)
        self.assertEqual(body["overview"], "Covers **trees**.")
        self.assertEqual(body["questions"], ["What is a BST?", "How does DFS work?"])
        build.assert_called_once()

    def test_unknown_session_returns_404(self):
        # Unknown session id -> the same 404 the message endpoint uses.
        resp = self.client.get("/api/chat/overview/does-not-exist")
        self.assertEqual(resp.status_code, 404)


if __name__ == "__main__":
    unittest.main()
