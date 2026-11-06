"""Upstream/internal failures in chat must surface as clean HTTP errors (or an SSE error event),
never a bare 500 or a leaked traceback / exception text. The provider chain is mocked at
``backend.llm_chain`` — no key, network or model.
"""
import json
import unittest
from unittest import mock

from fastapi import Request
from fastapi.testclient import TestClient

from backend import config, llm_chain
from backend.auth import current_owner
from backend.main import app
from backend.services import chat as chat_service


def _owner(request: Request) -> str:
    return "alice"


def _events(body):
    return [json.loads(f.strip()[5:]) for f in body.split("\n\n") if f.strip().startswith("data:")]


class ChatErrorHandlingTests(unittest.TestCase):
    def setUp(self):
        app.dependency_overrides[current_owner] = _owner
        self._p = [mock.patch.object(config, "PERSISTENCE_ENABLED", False),
                   mock.patch.object(llm_chain, "has_any_key", return_value=True)]
        for p in self._p:
            p.start()
        self.client = TestClient(app)
        chat_service._LOCAL.clear()
        self.sid = self.client.post("/api/chat/start", json={"mode": "topic", "topic": "Graphs"}).json()["session_id"]

    def tearDown(self):
        app.dependency_overrides.pop(current_owner, None)
        for p in self._p:
            p.stop()
        chat_service._LOCAL.clear()

    def _failing_stream(self, exc):
        async def gen(*a, **k):
            raise exc
            yield  # pragma: no cover
        return gen

    def test_all_providers_failed_is_503_with_none_label(self):
        exc = llm_chain.LLMChainError("All LLM providers failed", status=503, attempts=["groq#1 429"])
        with mock.patch.object(llm_chain, "astream_chat", self._failing_stream(exc)):
            resp = self.client.post("/api/chat/message", json={"session_id": self.sid, "message": "explain graphs"})
        self.assertEqual(resp.status_code, 503)
        self.assertEqual(resp.json()["llm"]["label"], "Model: none (all providers failed)")
        self.assertIn("unavailable", resp.json()["detail"].lower())

    def test_unexpected_exception_is_502_without_leaking_its_message(self):
        with mock.patch.object(llm_chain, "astream_chat", self._failing_stream(RuntimeError("secret-internal-path /etc/x"))):
            resp = self.client.post("/api/chat/message", json={"session_id": self.sid, "message": "explain graphs"})
        self.assertEqual(resp.status_code, 502)
        self.assertNotIn("secret-internal-path", resp.text)

    def test_stream_error_is_an_event_not_a_dropped_connection(self):
        with mock.patch.object(llm_chain, "astream_chat", self._failing_stream(RuntimeError("boom"))):
            resp = self.client.post("/api/chat/message/stream", json={"session_id": self.sid, "message": "explain graphs"})
        self.assertEqual(resp.status_code, 200)
        last = _events(resp.text)[-1]
        self.assertEqual(last["type"], "error")
        self.assertNotIn("boom", last["detail"])

    def test_database_outage_is_503(self):
        class MongoErr(Exception):
            pass
        MongoErr.__module__ = "pymongo.errors"
        with mock.patch.object(chat_service, "load_state", side_effect=MongoErr("down")):
            resp = self.client.post("/api/chat/message", json={"session_id": self.sid, "message": "explain graphs"})
        self.assertEqual(resp.status_code, 503)
        self.assertIn("database", resp.json()["detail"].lower())

    def test_expired_pdf_session_has_actionable_message(self):
        pdf = chat_service._mint("alice", "pdf", "PDF · a.pdf", None, None, None)
        resp = self.client.post("/api/chat/message", json={"session_id": pdf.id, "message": "what is in it?"})
        self.assertEqual(resp.status_code, 404)
        self.assertIn("upload the pdf again", resp.json()["detail"].lower())

    def test_unreadable_pdf_upload_is_422_not_500(self):
        resp = self.client.post("/api/chat/upload", files={"file": ("a.pdf", b"not a pdf at all", "application/pdf")})
        self.assertEqual(resp.status_code, 422)


if __name__ == "__main__":
    unittest.main()
