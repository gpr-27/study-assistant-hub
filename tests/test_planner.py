"""Planner service + router with the provider chain mocked at backend.llm_chain.chat."""
import datetime
import os
import threading
import unittest
from unittest import mock

from backend.routers import planner as pl_router
from backend.services import planner
from tests._study_helpers import PDF_BYTES, chain_error, make_client, patch_chat, result

_SUMMARY = "TOPICS:\n- Scheduling - Basic\nSUMMARY:\nShort."
_PLAN = "# Study Plan\n\n## Day 1: 2030-01-01 (2h)\n- stuff"


def _future(days=5) -> str:
    return (datetime.date.today() + datetime.timedelta(days=days)).isoformat()


class ServiceTests(unittest.TestCase):
    def setUp(self):
        planner._summary_cache.clear()

    def test_summaries_parallel_ordered_and_cached_per_owner(self):
        ctx, fake = patch_chat(_SUMMARY)
        with ctx, mock.patch.object(planner, "read_pdf", return_value="text"), \
             mock.patch.object(planner, "pdf_signature", return_value=(1.0, 10)):
            out, llm = planner.summarize_pdfs_ex("S", ["a.pdf", "b.pdf", "a.pdf"], owner="u1")
            self.assertEqual(list(out), ["a.pdf", "b.pdf"])          # deduped, ordered
            self.assertEqual(len(fake.calls), 2)
            planner.summarize_pdfs_ex("S", ["a.pdf", "b.pdf"], owner="u1")
            self.assertEqual(len(fake.calls), 2)                      # served from cache
            planner.summarize_pdfs_ex("S", ["a.pdf"], owner="u2")     # other owner must NOT hit u1's cache
            self.assertGreaterEqual(len(fake.calls), 3)
        self.assertIn("label", llm)

    def test_summary_concurrency_is_capped(self):
        live = {"now": 0, "max": 0}
        lock = threading.Lock()
        gate = threading.Event()

        def slow_chat(system, messages, **kw):
            with lock:
                live["now"] += 1
                live["max"] = max(live["max"], live["now"])
            gate.wait(0.05)
            with lock:
                live["now"] -= 1
            return result(_SUMMARY)

        with mock.patch("backend.llm_chain.chat", slow_chat), \
             mock.patch.object(planner, "read_pdf", return_value="text"), \
             mock.patch.object(planner, "pdf_signature", return_value=(1.0, 1)):
            planner.summarize_pdfs_ex("S", [f"{i}.pdf" for i in range(10)], owner="u")
        self.assertLessEqual(live["max"], planner._MAX_WORKERS)
        self.assertGreater(live["max"], 1)

    def test_summary_input_is_full_text(self):
        ctx, fake = patch_chat(_SUMMARY)
        with ctx, mock.patch.object(planner, "read_pdf", return_value="word " * 100_000), \
             mock.patch.object(planner, "pdf_signature", return_value=(1.0, 1)):
            planner.summarize_pdfs_ex("S", ["a.pdf"], owner="u")
        self.assertGreater(len(fake.calls[0]["messages"][0]["content"]), 500_000)

    def test_plan_from_topic_daily_and_weekly(self):
        ctx, fake = patch_chat(_PLAN)
        with ctx:
            plan, llm = planner.generate_plan_from_topic_ex("Calculus", _future(5))
            planner.generate_plan_from_topic_ex("Calculus", _future(90))
        self.assertEqual(plan, _PLAN)
        self.assertIn("## Day 1", fake.calls[0]["messages"][0]["content"])
        self.assertIn("## Week 1", fake.calls[1]["messages"][0]["content"])
        self.assertGreaterEqual(fake.calls[0]["max_tokens"], 2048)
        self.assertIn("label", llm)

    def test_plan_from_pdfs_returns_plan_call_llm(self):
        ctx, fake = patch_chat(_SUMMARY, _SUMMARY, _PLAN)
        with ctx, mock.patch.object(planner, "read_pdf", return_value="text"), \
             mock.patch.object(planner, "pdf_signature", return_value=(1.0, 1)):
            plan, llm = planner.generate_plan_ex("S", ["a.pdf"], _future(3), owner="u")
        self.assertTrue(plan)
        self.assertIn("PDF Summaries", fake.calls[-1]["messages"][0]["content"])

    def test_date_validation_happens_before_any_llm_call(self):
        ctx, fake = patch_chat(_SUMMARY)
        with ctx:
            with self.assertRaises(planner.PlannerError):
                planner.generate_plan_ex("S", ["a.pdf"], "not a date")
            with self.assertRaises(planner.PlannerError):
                planner.generate_plan_from_topic_ex("T", "9999-01-01")
            with self.assertRaises(planner.PlannerError):
                planner.generate_plan_from_topic_ex("  ", _future())
        self.assertEqual(fake.calls, [])

    def test_past_date_clamps_to_one_day(self):
        _, _, days = planner._days_until("2000-01-01")
        self.assertEqual(days, 1)

    def test_empty_inputs(self):
        with self.assertRaises(planner.PlannerError):
            planner.summarize_pdfs_ex("S", [])
        with self.assertRaises(planner.PlannerError):
            planner.summarize_named_texts_ex([("a.pdf", "  ")])

    def test_chain_error_propagates(self):
        ctx, _ = patch_chat(chain_error())
        with ctx, self.assertRaises(Exception) as cm:
            planner.generate_plan_from_topic_ex("T", _future())
        self.assertEqual(type(cm.exception).__name__, "LLMChainError")


class RouterTests(unittest.TestCase):
    def setUp(self):
        planner._summary_cache.clear()
        self.client = make_client(pl_router.router)

    def test_topic_plan_saves_and_returns_llm(self):
        ctx, _ = patch_chat(_PLAN)
        with ctx, mock.patch.object(pl_router.db, "save", return_value="p1") as save:
            res = self.client.post("/api/planner/topic", json={"topic": "Calculus", "target_date": _future()})
        body = res.json()
        self.assertEqual(res.status_code, 200)
        self.assertEqual((body["plan"], body["id"], body["title"]), (_PLAN, "p1", "Plan · Calculus"))
        self.assertIn("label", body["llm"])
        self.assertEqual(save.call_args.args[2]["llm"], body["llm"])

    def test_subject_plan_and_summaries(self):
        ctx, _ = patch_chat(_SUMMARY, _PLAN)
        with ctx, mock.patch.object(planner, "read_pdf", return_value="text") as rd, \
             mock.patch.object(planner, "pdf_signature", return_value=(1.0, 1)), \
             mock.patch.object(pl_router.db, "save", return_value=None):
            s = self.client.post("/api/planner/summaries", json={"subject": "S", "pdfs": ["a.pdf"]})
            p = self.client.post("/api/planner/generate",
                                 json={"subject": "S", "pdfs": ["a.pdf"], "target_date": _future()})
        self.assertEqual(s.status_code, 200)
        self.assertIn("a.pdf", s.json()["summaries"])
        self.assertIn("label", s.json()["llm"])
        self.assertEqual(p.status_code, 200)
        rd.assert_any_call("S", "a.pdf", "tester")

    def test_validation(self):
        post = self.client.post
        self.assertEqual(post("/api/planner/topic", json={"topic": "x", "target_date": "garbage"}).status_code, 422)
        self.assertEqual(post("/api/planner/topic", json={"topic": "  ", "target_date": _future()}).status_code, 400)
        self.assertEqual(post("/api/planner/summaries", json={"subject": "S", "pdfs": []}).status_code, 422)
        self.assertEqual(post("/api/planner/summaries", json={"subject": "S", "pdfs": ["a"] * 13}).status_code, 422)
        self.assertEqual(post("/api/planner/generate", json={"subject": "S", "pdfs": ["a.pdf"], "target_date": "nope"}).status_code, 422)

    def test_failures(self):
        ctx, _ = patch_chat("  ")
        with ctx:
            res = self.client.post("/api/planner/topic", json={"topic": "x", "target_date": _future()})
        self.assertEqual(res.status_code, 502)
        ctx, _ = patch_chat(chain_error())
        with ctx:
            res = self.client.post("/api/planner/topic", json={"topic": "x", "target_date": _future()})
        self.assertEqual(res.status_code, 503)
        self.assertIn("label", res.json()["llm"])

    def test_upload_multiple_skips_bad_file(self):
        ctx, _ = patch_chat(_SUMMARY, _PLAN)
        with ctx, mock.patch.object(pl_router.db, "save", return_value=None), \
             mock.patch("backend.routers._common.extract_pdf_text", return_value="text"):
            res = self.client.post("/api/planner/upload", data={"target_date": _future()},
                                   files=[("files", ("a.pdf", PDF_BYTES, "application/pdf")),
                                          ("files", ("b.pdf", b"", "application/pdf"))])
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["title"], "Plan · a.pdf")
        self.assertIn("label", res.json()["llm"])

    def test_upload_rejections(self):
        res = self.client.post("/api/planner/upload", data={"target_date": _future()},
                               files=[("files", ("a.pdf", b"", "application/pdf"))])
        self.assertEqual(res.status_code, 400)
        res = self.client.post("/api/planner/upload", data={"target_date": "bad"},
                               files=[("files", ("a.pdf", PDF_BYTES, "application/pdf"))])
        self.assertEqual(res.status_code, 422)


if __name__ == "__main__":
    unittest.main()


class FullPdfToAnthropicTests(unittest.TestCase):
    def test_100k_token_pdf_reaches_anthropic_whole(self):
        import json as _json
        import httpx
        from backend import llm_chain as lc
        text = "word " * 80_000                                  # ~100k tokens (400k chars)
        seen = {}

        def handler(request):
            seen["body"] = _json.loads(request.content)
            return httpx.Response(200, json={"content": [{"type": "text", "text": _SUMMARY}],
                                             "stop_reason": "end_turn", "usage": {"input_tokens": 1, "output_tokens": 1}})
        lc.reset_state()
        lc._configure_transports(httpx.MockTransport(handler))
        try:
            with mock.patch.dict("os.environ", {"ANTHROPIC_API_KEY": "k-anthropic-test-1234"}, clear=False), \
                 mock.patch.object(planner, "read_pdf", return_value=text), \
                 mock.patch.object(planner, "pdf_signature", return_value=(1.0, 1)):
                for k in ("GROQ_API_KEY", "GROQ_API_KEY_2", "GEMINI_API_KEY", "GEMINI_API_KEY_2"):
                    os.environ.pop(k, None)
                planner.summarize_pdfs_ex("S", ["big.pdf"], owner="u-full")
        finally:
            lc._configure_transports(None)
            lc.reset_state()
        assert text in seen["body"]["messages"][0]["content"]
