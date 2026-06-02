"""Tests that Groq upstream failures surface as clean HTTP errors, not 500s.

The chat/quiz/flashcards/planner endpoints call Groq; when Groq raises (e.g. a
free-tier rate limit), global handlers in backend.main must map it to a readable
429/502 with a `detail` message instead of a bare 500. `chat_service.ask` is
patched so no real model, index, or Groq key is exercised.
"""
import unittest
from unittest import mock

import httpx
from fastapi.testclient import TestClient
from groq import APIError, RateLimitError

from backend.main import app

_REQUEST = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")


class ChatErrorHandlingTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_rate_limit_returns_429_with_actionable_detail(self):
        error = RateLimitError("rate limited", response=httpx.Response(429, request=_REQUEST), body=None)
        with mock.patch("backend.routers.chat.chat_service.ask", side_effect=error):
            resp = self.client.post("/api/chat/message", json={"session_id": "x", "message": "hi"})
        self.assertEqual(resp.status_code, 429)
        detail = resp.json()["detail"].lower()
        self.assertIn("rate limit", detail)
        self.assertIn("groq", detail)

    def test_other_groq_error_returns_502(self):
        error = APIError("upstream boom", request=_REQUEST, body=None)
        with mock.patch("backend.routers.chat.chat_service.ask", side_effect=error):
            resp = self.client.post("/api/chat/message", json={"session_id": "x", "message": "hi"})
        self.assertEqual(resp.status_code, 502)
        self.assertIn("AI service", resp.json()["detail"])


if __name__ == "__main__":
    unittest.main()
