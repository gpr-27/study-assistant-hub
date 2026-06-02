"""The quiz/flashcards services must let Groq errors propagate (not wrap them),
so the global handlers in backend.main can render a clean 429/502. Locks the fix
that previously turned a rate limit into an opaque 502 'generation failed' blob.

The Groq client is fully mocked — no key, network, or real model is used.
"""
import unittest
from unittest import mock

import httpx
from groq import RateLimitError

from backend.services import flashcards, quiz

_REQUEST = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")


def _rate_limit_error() -> RateLimitError:
    return RateLimitError("rate limited", response=httpx.Response(429, request=_REQUEST), body=None)


class GroqErrorPropagationTests(unittest.TestCase):
    def test_quiz_generation_propagates_rate_limit(self):
        with mock.patch.object(quiz, "require_groq_key", lambda: None), \
             mock.patch.object(quiz, "Groq") as mock_groq:
            mock_groq.return_value.chat.completions.create.side_effect = _rate_limit_error()
            with self.assertRaises(RateLimitError):
                quiz.make_quiz("Supply and demand basics.", 5, "easy")

    def test_flashcard_generation_propagates_rate_limit(self):
        with mock.patch.object(flashcards, "require_groq_key", lambda: None), \
             mock.patch.object(flashcards, "Groq") as mock_groq:
            mock_groq.return_value.chat.completions.create.side_effect = _rate_limit_error()
            with self.assertRaises(RateLimitError):
                flashcards.make_flashcards("Supply and demand basics.", 4, "easy")


if __name__ == "__main__":
    unittest.main()
