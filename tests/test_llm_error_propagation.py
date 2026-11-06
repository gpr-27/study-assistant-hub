"""When every provider fails, the quiz/flashcards services must let ``LLMChainError`` propagate
(not wrap it) so backend.main can render a clean 503 with the 'Model: none' label.

The chain is mocked at backend.llm_chain.chat — no key, network, or real model is used.
"""
import unittest
from unittest import mock

from backend import llm_chain
from backend.services import flashcards, quiz
from tests._study_helpers import chain_error, patch_chat


class ChainErrorPropagationTests(unittest.TestCase):
    def test_quiz_generation_propagates_chain_error(self):
        ctx, _ = patch_chat(chain_error())
        with ctx, self.assertRaises(llm_chain.LLMChainError):
            quiz.make_quiz("Supply and demand basics.", 5, "easy")

    def test_flashcard_generation_propagates_chain_error(self):
        ctx, _ = patch_chat(chain_error())
        with ctx, self.assertRaises(llm_chain.LLMChainError):
            flashcards.make_flashcards("Supply and demand basics.", 4, "easy")

    def test_chat_side_services_propagate_chain_error(self):
        """predictor / teachback / insights / overview-generation never swallow or wrap an all-failed chain."""
        from backend.services import insights, overview, predictor, teachback

        ctx, _ = patch_chat(chain_error())
        with ctx, mock.patch.object(predictor, "require_llm_key", lambda: None), \
                self.assertRaises(llm_chain.LLMChainError):
            predictor.predict_questions("Past paper text.")
        with ctx, mock.patch.object(teachback, "require_llm_key", lambda: None), \
                self.assertRaises(llm_chain.LLMChainError):
            teachback.start_session("topic", topic="Graphs")
        with ctx, mock.patch.object(insights, "build_summary", return_value={
                "overall": {"total_attempts": 3}, "subjects": [], "weakest": []}), \
                mock.patch.object(insights.db, "multi_recent", side_effect=lambda o, specs: {k: {"n": 0, "rows": []} for k, _l, _f in specs}), \
                self.assertRaises(llm_chain.LLMChainError):
            insights._cache.clear()
            insights.build_insights("u-err", refresh=True)
        # overview is best-effort BY CONTRACT: it degrades to an empty card instead of raising
        overview._cache.clear()
        with ctx:
            card = overview.build_overview(topic="Graphs", owner="u-err")
        self.assertEqual((card["overview"], card["llm"]), ("", None))

    def test_chat_agentic_helpers_propagate_chain_error(self):
        from backend import agentic_rag

        ctx, _ = patch_chat(chain_error())
        with ctx, self.assertRaises(llm_chain.LLMChainError):
            agentic_rag._rewrite("q", [])
        with ctx, self.assertRaises(llm_chain.LLMChainError):
            agentic_rag._grade("q", "ctx")

    def test_error_carries_the_none_label(self):
        self.assertEqual(chain_error().label, "Model: none (all providers failed)")


if __name__ == "__main__":
    unittest.main()
