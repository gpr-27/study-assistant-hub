"""Tests for the agentic RAG retrieval loop (rewrite → retrieve → grade → correct).

No network or real Groq: a FakeLLM returns queued JSON/text replies and a FakeRetriever
returns canned documents. Covers JSON extraction, query rewriting (+fallbacks), retrieval
merge/de-dupe, grading, the corrective re-retrieve loop, round bounding, rate-limit
propagation, and the chat service choosing agentic vs classic by config.
"""
import unittest
from unittest import mock

import httpx
from groq import RateLimitError

from backend import agentic_rag, config


_REQUEST = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")


def _rate_limit() -> RateLimitError:
    return RateLimitError("rate limited", response=httpx.Response(429, request=_REQUEST), body=None)


class _Doc:
    def __init__(self, text):
        self.page_content = text


class _Reply:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    """Returns queued replies in order; records the messages it was called with."""

    def __init__(self, replies):
        self._replies = list(replies)
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        if not self._replies:
            return _Reply("")
        nxt = self._replies.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return _Reply(nxt)


class FakeRetriever:
    """Maps a query string -> list[_Doc]; unknown queries return []."""

    def __init__(self, mapping):
        self.mapping = mapping
        self.queries = []

    def invoke(self, query):
        self.queries.append(query)
        return self.mapping.get(query, [])


class JsonExtractionTests(unittest.TestCase):
    def test_plain_json(self):
        self.assertEqual(agentic_rag._extract_json('{"a": 1}'), {"a": 1})

    def test_json_embedded_in_prose_and_fences(self):
        text = "Sure! ```json\n{\"queries\": [\"x\"]}\n``` done"
        self.assertEqual(agentic_rag._extract_json(text), {"queries": ["x"]})

    def test_garbage_returns_empty(self):
        self.assertEqual(agentic_rag._extract_json("no json here"), {})


class RewriteTests(unittest.TestCase):
    def test_parses_and_caps_queries(self):
        llm = FakeLLM(['{"queries": ["a", "b", "c", "d", "e"]}'])
        out = agentic_rag._rewrite(llm, "q", [])
        self.assertEqual(out, ["a", "b", "c"][: config.AGENTIC_RAG_QUERIES])

    def test_falls_back_to_question_on_bad_json(self):
        llm = FakeLLM(["not json"])
        self.assertEqual(agentic_rag._rewrite(llm, "what is X?", []), ["what is X?"])

    def test_rate_limit_propagates(self):
        llm = FakeLLM([_rate_limit()])
        with self.assertRaises(RateLimitError):
            agentic_rag._rewrite(llm, "q", [])


class RetrieveTests(unittest.TestCase):
    def test_merges_and_dedupes(self):
        d1, d2 = _Doc("alpha"), _Doc("beta")
        r = FakeRetriever({"q1": [d1, d2], "q2": [_Doc("alpha"), _Doc("gamma")]})
        docs = agentic_rag._retrieve(r, ["q1", "q2"])
        self.assertEqual([d.page_content for d in docs], ["alpha", "beta", "gamma"])


class GradeTests(unittest.TestCase):
    def test_sufficient_true(self):
        llm = FakeLLM(['{"sufficient": true, "missing": ""}'])
        ok, missing = agentic_rag._grade(llm, "q", "some context")
        self.assertTrue(ok)

    def test_insufficient_with_missing(self):
        llm = FakeLLM(['{"sufficient": false, "missing": "definition of X"}'])
        ok, missing = agentic_rag._grade(llm, "q", "ctx")
        self.assertFalse(ok)
        self.assertEqual(missing, "definition of X")

    def test_empty_context_is_insufficient_without_calling_llm(self):
        llm = FakeLLM([])  # must not be called
        ok, _ = agentic_rag._grade(llm, "q", "   ")
        self.assertFalse(ok)
        self.assertEqual(llm.calls, [])

    def test_unparseable_grade_assumes_sufficient(self):
        llm = FakeLLM(["???"])
        ok, _ = agentic_rag._grade(llm, "q", "ctx")
        self.assertTrue(ok)  # degrade to naive RAG rather than loop forever


class AnswerLoopTests(unittest.TestCase):
    def test_stops_when_first_grade_sufficient(self):
        retriever = FakeRetriever({"q1": [_Doc("relevant ctx")]})
        llm = FakeLLM([
            '{"queries": ["q1"]}',          # rewrite
            '{"sufficient": true}',          # grade -> stop
            "FINAL ANSWER",                  # generate
        ])
        out = agentic_rag.answer(retriever, llm, "question?", [], "persona", max_iters=2)
        self.assertEqual(out, "FINAL ANSWER")
        self.assertEqual(retriever.queries, ["q1"])     # only one retrieval round
        self.assertEqual(len(llm.calls), 3)

    def test_corrective_second_round_then_generate(self):
        retriever = FakeRetriever({"q1": [_Doc("weak")], "q2": [_Doc("strong extra")]})
        llm = FakeLLM([
            '{"queries": ["q1"]}',                       # rewrite #1
            '{"sufficient": false, "missing": "more"}',   # grade #1 -> correct
            '{"queries": ["q2"]}',                       # rewrite #2 (corrective)
            '{"sufficient": true}',                       # grade #2 -> stop
            "ANSWER WITH BOTH",                          # generate
        ])
        out = agentic_rag.answer(retriever, llm, "question?", [], "persona", max_iters=2)
        self.assertEqual(out, "ANSWER WITH BOTH")
        self.assertEqual(retriever.queries, ["q1", "q2"])
        # final generation context should include both retrieved docs
        gen_msgs = llm.calls[-1]
        self.assertIn("weak", gen_msgs[-1].content)
        self.assertIn("strong extra", gen_msgs[-1].content)

    def test_round_cap_respected(self):
        retriever = FakeRetriever({"q": [_Doc("ctx")]})
        # Always insufficient; with max_iters=1 it must retrieve once then generate.
        llm = FakeLLM([
            '{"queries": ["q"]}',
            '{"sufficient": false, "missing": "x"}',
            "ANSWER ANYWAY",
        ])
        out = agentic_rag.answer(retriever, llm, "q?", [], "persona", max_iters=1)
        self.assertEqual(out, "ANSWER ANYWAY")
        self.assertEqual(retriever.queries, ["q"])  # no second round


class ChatServiceWiringTests(unittest.TestCase):
    """chat.ask() routes to agentic vs classic RAG based on config.AGENTIC_RAG."""

    def _session(self, retriever):
        from backend.services import chat
        sid = chat._register({
            "mode": "subject", "retriever": retriever, "llm": object(), "history": [],
            "label": "L", "topic": None, "subject": "Bio", "owner": "local", "chat_id": None,
        })
        return chat, sid

    def test_uses_agentic_when_enabled(self):
        from backend.services import chat
        retriever = FakeRetriever({})
        chat_mod, sid = self._session(retriever)
        with mock.patch.object(chat_mod.config, "AGENTIC_RAG", True), \
             mock.patch.object(chat_mod, "require_groq_key", lambda: None), \
             mock.patch.object(chat_mod.agentic_rag, "answer", return_value="AGENTIC") as ag:
            out = chat_mod.ask(sid, "hello")
        self.assertEqual(out, "AGENTIC")
        ag.assert_called_once()

    def test_uses_classic_when_disabled(self):
        from backend.services import chat
        retriever = FakeRetriever({"hello": [_Doc("ctx")]})
        chat_mod, sid = self._session(retriever)
        # Replace the session llm with a fake so the single-shot path returns text.
        chat_mod._sessions[sid]["llm"] = FakeLLM(["CLASSIC"])
        with mock.patch.object(chat_mod.config, "AGENTIC_RAG", False), \
             mock.patch.object(chat_mod, "require_groq_key", lambda: None), \
             mock.patch.object(chat_mod.agentic_rag, "answer", side_effect=AssertionError("should not be agentic")):
            out = chat_mod.ask(sid, "hello")
        self.assertEqual(out, "CLASSIC")
        self.assertEqual(retriever.queries, ["hello"])  # one direct retrieval


class HttpEndToEndTests(unittest.TestCase):
    """Full path: POST /api/chat/start then /api/chat/message, running the REAL agentic
    loop end-to-end with only the LLM and retriever mocked (no network/Groq)."""

    def setUp(self):
        from fastapi.testclient import TestClient
        from backend.main import app

        self.client = TestClient(app)

    def test_chat_message_runs_agentic_loop(self):
        from backend.services import chat

        retriever = FakeRetriever({"photosynthesis basics": [_Doc("Photosynthesis converts light to sugar.")]})
        llm = FakeLLM([
            '{"queries": ["photosynthesis basics"]}',   # rewrite
            '{"sufficient": true}',                       # grade -> stop
            "Photosynthesis is how plants make food.",   # generate
        ])
        with mock.patch.object(chat.config, "AGENTIC_RAG", True), \
             mock.patch.object(chat, "require_groq_key", lambda: None), \
             mock.patch.object(chat.vector_store, "get_subject_retriever", return_value=retriever), \
             mock.patch.object(chat, "_new_chat_model", return_value=llm):
            start = self.client.post("/api/chat/start", json={"mode": "subject", "subject": "Bio"})
            self.assertEqual(start.status_code, 200, start.text)
            sid = start.json()["session_id"]

            msg = self.client.post("/api/chat/message", json={"session_id": sid, "message": "What is photosynthesis?"})

        self.assertEqual(msg.status_code, 200, msg.text)
        self.assertEqual(msg.json()["answer"], "Photosynthesis is how plants make food.")
        self.assertEqual(retriever.queries, ["photosynthesis basics"])  # retrieval used the rewritten query
        self.assertEqual(len(llm.calls), 3)                              # rewrite + grade + generate


if __name__ == "__main__":
    unittest.main()
