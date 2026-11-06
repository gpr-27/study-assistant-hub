"""Tests for the agentic RAG retrieval loop and the chat service's bounded retrieval.

No network and no provider: ``llm_chain.chat`` is replaced by a queued fake and the retriever
returns canned documents. Covers JSON extraction, the zero-LLM heuristic rewrite (the fast
default), the opt-in LLM rewrite/grade/correct loop, round bounding, ``LLMChainError``
propagation, the time-bounded retrieval with its lexical fallback, and history budgeting.
"""
import asyncio
import time
import unittest
from unittest import mock

from backend import agentic_rag, config
from backend.llm_chain import LLMChainError
from tests._study_helpers import FakeChat, chain_error


class _Doc:
    def __init__(self, text, source=""):
        self.page_content = text
        self.metadata = {"source": source} if source else {}


class FakeRetriever:
    """Maps a query string -> list[_Doc]; unknown queries return []."""

    def __init__(self, mapping, delay=0.0, lexical=None):
        self.mapping = mapping
        self.queries = []
        self.delay = delay
        self._lexical = lexical

    def invoke(self, query):
        self.queries.append(query)
        if self.delay:
            time.sleep(self.delay)
        return self.mapping.get(query, [])

    def invoke_lexical(self, query):
        if self._lexical is None:
            raise AttributeError("no lexical")
        return self._lexical.get(query, [])


def _patched(*replies):
    fake = FakeChat(*replies)
    return mock.patch("backend.llm_chain.chat", fake), fake


class JsonExtractionTests(unittest.TestCase):
    def test_plain_json(self):
        self.assertEqual(agentic_rag._extract_json('{"a": 1}'), {"a": 1})

    def test_json_embedded_in_prose_and_fences(self):
        text = "Sure! ```json\n{\"queries\": [\"x\"]}\n``` done"
        self.assertEqual(agentic_rag._extract_json(text), {"queries": ["x"]})

    def test_garbage_returns_empty(self):
        self.assertEqual(agentic_rag._extract_json("no json here"), {})


class HeuristicQueryTests(unittest.TestCase):
    def test_plain_question_is_the_only_query(self):
        self.assertEqual(agentic_rag.heuristic_queries("Explain the CAP theorem in distributed systems", []),
                         ["Explain the CAP theorem in distributed systems"])

    def test_short_followup_pulls_in_previous_question(self):
        hist = [{"role": "user", "content": "What is a B-tree?"}, {"role": "assistant", "content": "A tree..."}]
        qs = agentic_rag.heuristic_queries("why is it balanced?", hist)
        self.assertEqual(qs[0], "why is it balanced?")
        self.assertIn("What is a B-tree?", qs[1])

    def test_bounded_by_config(self):
        with mock.patch.object(config, "AGENTIC_RAG_QUERIES", 1):
            hist = [{"role": "user", "content": "topic"}]
            self.assertEqual(len(agentic_rag.heuristic_queries("and that?", hist)), 1)


class RewriteTests(unittest.TestCase):
    def test_parses_and_caps_queries(self):
        p, fake = _patched('{"queries": ["a", "b", "c", "d", "e"]}')
        with p:
            queries, prompt = agentic_rag._rewrite("q", [])
        self.assertEqual(queries, ["a", "b", "c"][: config.AGENTIC_RAG_QUERIES])
        self.assertIn("search queries", prompt)
        self.assertTrue(fake.calls[0]["json_mode"])          # structured output via json_mode, not native tools

    def test_falls_back_to_question_on_bad_json(self):
        p, _ = _patched("not json")
        with p:
            queries, _prompt = agentic_rag._rewrite("what is X?", [])
        self.assertEqual(queries, ["what is X?"])

    def test_all_providers_failed_propagates(self):
        p, _ = _patched(chain_error())
        with p, self.assertRaises(LLMChainError):
            agentic_rag._rewrite("q", [])


class RetrieveTests(unittest.TestCase):
    def test_merges_and_dedupes(self):
        r = FakeRetriever({"q1": [_Doc("alpha"), _Doc("beta")], "q2": [_Doc("alpha"), _Doc("gamma")]})
        docs = agentic_rag._retrieve(r, ["q1", "q2"])
        self.assertEqual([d.page_content for d in docs], ["alpha", "beta", "gamma"])

    def test_one_failing_query_does_not_abort(self):
        class Boom(FakeRetriever):
            def invoke(self, query):
                if query == "bad":
                    raise RuntimeError("x")
                return super().invoke(query)

        docs = agentic_rag._retrieve(Boom({"ok": [_Doc("fine")]}), ["bad", "ok"])
        self.assertEqual([d.page_content for d in docs], ["fine"])


class GradeTests(unittest.TestCase):
    def test_sufficient_true(self):
        p, _ = _patched('{"sufficient": true, "missing": ""}')
        with p:
            ok, _m, _prompt = agentic_rag._grade("q", "some context")
        self.assertTrue(ok)

    def test_insufficient_with_missing(self):
        p, _ = _patched('{"sufficient": false, "missing": "definition of X"}')
        with p:
            ok, missing, _prompt = agentic_rag._grade("q", "ctx")
        self.assertFalse(ok)
        self.assertEqual(missing, "definition of X")

    def test_empty_context_is_insufficient_without_calling_llm(self):
        p, fake = _patched("{}")
        with p:
            ok, _m, _p = agentic_rag._grade("q", "   ")
        self.assertFalse(ok)
        self.assertEqual(fake.calls, [])

    def test_unparseable_grade_assumes_sufficient(self):
        p, _ = _patched("???")
        with p:
            ok, _m, _p = agentic_rag._grade("q", "ctx")
        self.assertTrue(ok)        # degrade to naive RAG rather than loop forever

    def test_user_relaxed_grading_prompt_is_kept(self):
        self.assertIn("Mark insufficient only when the answer cannot be derived at all", agentic_rag._GRADE_SYSTEM)


class GatherTests(unittest.TestCase):
    def test_default_path_makes_zero_llm_calls(self):
        """Single round + no opt-in => heuristic rewrite, grade skipped: nothing blocks time-to-first-token."""
        retriever = FakeRetriever({"question?": [_Doc("relevant ctx", "/x/a.pdf")]})
        p, fake = _patched("SHOULD NOT BE CALLED")
        with p, mock.patch.object(config, "AGENTIC_RAG_MAX_ITERS", 1), \
                mock.patch.dict("os.environ", {"AGENTIC_RAG_LLM_REWRITE": ""}):
            got = agentic_rag.gather(retriever, "question?", [], persona="P")
        self.assertEqual(fake.calls, [])
        self.assertEqual([s["kind"] for s in got.steps], ["rewrite", "retrieve", "grade", "generate"])
        self.assertEqual(got.steps[2]["verdict"], "final")
        self.assertEqual(got.sources, ["a.pdf"])
        self.assertIn("relevant ctx", got.context)

    def test_llm_loop_stops_when_first_grade_sufficient(self):
        retriever = FakeRetriever({"q1": [_Doc("relevant ctx")]})
        p, fake = _patched('{"queries": ["q1"]}', '{"sufficient": true}')
        with p:
            got = agentic_rag.gather(retriever, "question?", [], max_iters=2)
        self.assertEqual(retriever.queries, ["q1"])
        self.assertEqual(len(fake.calls), 2)
        self.assertEqual([s["kind"] for s in got.steps], ["rewrite", "retrieve", "grade", "generate"])
        self.assertEqual(got.steps[2]["verdict"], "sufficient")

    def test_corrective_second_round(self):
        retriever = FakeRetriever({"q1": [_Doc("weak")], "q2": [_Doc("strong extra")]})
        p, _ = _patched('{"queries": ["q1"]}', '{"sufficient": false, "missing": "more"}', '{"queries": ["q2"]}')
        with p:
            got = agentic_rag.gather(retriever, "question?", [], max_iters=2)
        self.assertEqual(retriever.queries, ["q1", "q2"])
        self.assertIn("weak", got.context)
        self.assertIn("strong extra", got.context)
        self.assertEqual([s["verdict"] for s in got.steps if s["kind"] == "grade"], ["insufficient", "final"])

    def test_on_step_fires_in_order_including_generate(self):
        retriever = FakeRetriever({"question?": [_Doc("ctx")]})
        seen = []
        got = agentic_rag.gather(retriever, "question?", [], max_iters=1, on_step=seen.append)
        self.assertEqual(seen, got.steps)
        self.assertEqual(seen[-1]["kind"], "generate")

    def test_cancel_stops_before_retrieval(self):
        import threading

        ev = threading.Event()
        ev.set()
        retriever = FakeRetriever({"q": [_Doc("ctx")]})
        got = agentic_rag.gather(retriever, "q", [], max_iters=2, cancel=ev)
        self.assertEqual(retriever.queries, [])
        self.assertEqual(got.docs, [])


class AnswerTests(unittest.TestCase):
    def test_answer_generates_once_with_context_in_last_message(self):
        retriever = FakeRetriever({"q?": [_Doc("fact: 42")]})
        p, fake = _patched("FINAL ANSWER")
        with p:
            text, steps, sources, result = agentic_rag.answer(retriever, "q?", [], "persona", max_iters=1)
        self.assertEqual(text, "FINAL ANSWER")
        self.assertIn("fact: 42", fake.calls[-1]["messages"][-1]["content"])
        self.assertEqual(fake.calls[-1]["system"], "persona")
        self.assertEqual(result.model, "openai/gpt-oss-120b")

    def test_chain_failure_propagates(self):
        retriever = FakeRetriever({"q?": [_Doc("x")]})
        p, _ = _patched(chain_error())
        with p, self.assertRaises(LLMChainError):
            agentic_rag.answer(retriever, "q?", [], "persona", max_iters=1)


class BoundedRetrievalTests(unittest.IsolatedAsyncioTestCase):
    async def test_fast_retrieval_is_hybrid(self):
        from backend.services import chat

        r = FakeRetriever({"q": [_Doc("a")]})
        docs, mode = await chat.retrieve_bounded(r, ["q"])
        self.assertEqual((mode, [d.page_content for d in docs]), ("hybrid", ["a"]))

    async def test_slow_embeddings_fall_back_to_lexical_without_waiting(self):
        from backend.services import chat

        r = FakeRetriever({"q": [_Doc("dense")]}, delay=0.6, lexical={"q": [_Doc("bm25 hit")]})
        with mock.patch.object(chat, "_RETRIEVAL_BUDGET_S", 0.05):
            t0 = time.monotonic()
            docs, mode = await chat.retrieve_bounded(r, ["q"])
            elapsed = time.monotonic() - t0
        self.assertEqual(mode, "lexical")
        self.assertEqual(docs[0].page_content, "bm25 hit")
        self.assertLess(elapsed, 0.4)            # did NOT wait for the slow dense call
        await asyncio.sleep(0.7)                 # let the orphaned worker finish quietly

    async def test_no_lexical_support_returns_none_not_error(self):
        from backend.services import chat

        r = FakeRetriever({"q": [_Doc("dense")]}, delay=0.3)
        with mock.patch.object(chat, "_RETRIEVAL_BUDGET_S", 0.02), \
                mock.patch.object(chat, "_lexical", side_effect=RuntimeError("no index")):
            docs, mode = await chat.retrieve_bounded(r, ["q"])
        self.assertEqual((docs, mode), ([], "none"))
        await asyncio.sleep(0.4)


class HistoryBudgetTests(unittest.TestCase):
    def test_history_is_full_conversation_unclipped(self):
        from backend.services import chat

        hist = [(f"question {i} " + "word " * 300, f"answer {i} " + "word " * 600) for i in range(20)]
        out = chat._history_messages(hist)
        self.assertEqual(len(out), 40)                       # every exchange, no turn/token cap
        self.assertEqual(out[0]["content"], hist[0][0])      # nothing clipped
        self.assertEqual(out[-1]["content"], hist[-1][1])
        self.assertEqual(out[-1]["role"], "assistant")
        self.assertEqual(out[0]["role"], "user")


if __name__ == "__main__":
    unittest.main()
