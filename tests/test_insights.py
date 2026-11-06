"""Tests for the AI insights service + router (cross-tool study analysis).

No network: ``llm_chain.chat`` and db are mocked. Covers the empty-activity fast path (no LLM
call), digest assembly, analysis parsing/validation, the bounded per-day cache (+refresh bypass and
its rate limit), the ``llm`` label on every payload, 503 on total chain failure, and the endpoint.
"""
import importlib.util
import json
import unittest
from pathlib import Path
from unittest import mock

from backend.llm_chain import LLMChainError
from backend.services import insights
from tests._study_helpers import FakeChat, chain_error, result

_GOOD = (
    '{"headline": "Solid start!", "observations": ["You aced DBMS"],'
    ' "recommendations": [{"action": "Take a B-tree quiz", "reason": "weak topic", "tool": "quiz"}],'
    ' "focus_topics": ["B-trees"]}'
)

_EMPTY_PROGRESS = {
    "overall": {"total_attempts": 0, "overall_accuracy": 0.0, "streak_days": 0},
    "subjects": [], "weakest": [],
}
_ACTIVE = dict(_EMPTY_PROGRESS, overall={"total_attempts": 4, "overall_accuracy": 0.5, "streak_days": 2})


def _chat(*replies):
    fake = FakeChat(*replies)
    return mock.patch("backend.llm_chain.chat", fake), fake


def _multi(per_kind):
    """Patch db.multi_recent: ``per_kind`` is a list (same rows for every kind) or a {kind: rows} dict."""
    def fake(owner, specs):
        out = {}
        for kind, _limit, _fields in specs:
            rows = per_kind.get(kind, []) if isinstance(per_kind, dict) else per_kind
            out[kind] = {"n": len(rows), "rows": list(rows)}
        return out
    return mock.patch.object(insights.db, "multi_recent", side_effect=fake)


class BuildInsightsTests(unittest.TestCase):
    def setUp(self):
        insights._cache.clear()
        insights._last_refresh.clear()

    def test_no_activity_returns_empty_without_llm(self):
        ctx, fake = _chat(_GOOD)
        with mock.patch.object(insights, "build_summary", return_value=_EMPTY_PROGRESS), \
             _multi([]), ctx:
            out = insights.build_insights("u1")
        self.assertTrue(out["empty"])
        self.assertEqual(fake.calls, [])

    def test_activity_produces_parsed_insights_with_llm_label(self):
        ctx, fake = _chat(_GOOD)
        with mock.patch.object(insights, "build_summary", return_value=_ACTIVE), \
             _multi([{"title": "Quiz · DBMS"}]), ctx:
            out = insights.build_insights("u1")
        self.assertFalse(out["empty"])
        self.assertEqual(out["insights"]["headline"], "Solid start!")
        self.assertEqual(out["insights"]["recommendations"][0]["tool"], "quiz")
        self.assertIsNotNone(out["generated_at"])
        self.assertTrue(out["llm"]["label"].startswith("Model: groq"))
        self.assertTrue(fake.calls[0]["json_mode"])
        self.assertGreaterEqual(fake.calls[0]["max_tokens"], 2048)

    def test_cached_per_day_and_refresh_bypasses_but_is_rate_limited(self):
        ctx, fake = _chat(_GOOD)
        with mock.patch.object(insights, "build_summary", return_value=_ACTIVE), \
             _multi([{"title": "t"}]), ctx:
            insights.build_insights("u1")
            insights.build_insights("u1")                     # cache hit
            self.assertEqual(len(fake.calls), 1)
            insights.build_insights("u1", refresh=True)       # too soon after the first run → served from cache
            self.assertEqual(len(fake.calls), 1)
            insights._last_refresh["u1"] -= 100               # the cooldown has passed
            insights.build_insights("u1", refresh=True)
            self.assertEqual(len(fake.calls), 2)

    def test_cache_is_per_owner_and_bounded(self):
        ctx, fake = _chat(_GOOD)
        with mock.patch.object(insights, "build_summary", return_value=_ACTIVE), \
             _multi([{"title": "t"}]), ctx, \
             mock.patch.object(insights, "_CACHE_MAX", 3):
            for u in ("a", "b", "c", "d", "e"):
                insights.build_insights(u)
            self.assertEqual(len(fake.calls), 5)              # one generation per distinct owner
            self.assertLessEqual(len(insights._cache), 3)     # no unbounded growth
            self.assertNotIn("a", insights._cache)

    def test_unusable_analysis_raises(self):
        ctx, _ = _chat("not json")
        with mock.patch.object(insights, "build_summary", return_value=_ACTIVE), \
             _multi([]), ctx:
            with self.assertRaises(insights.InsightsError):
                insights.build_insights("u1")

    def test_chain_failure_propagates_for_a_503(self):
        ctx, _ = _chat(chain_error())
        with mock.patch.object(insights, "build_summary", return_value=_ACTIVE), \
             _multi([]), ctx:
            with self.assertRaises(LLMChainError):
                insights.build_insights("u1")

    def test_digest_includes_exam_scores_and_teachback_gaps(self):
        rows = {"exams": [{"title": "Exam · DBMS", "total_score": 7, "max_score": 10}],
                "teachbacks": [{"title": "Teach-back · Joins",
                                "evaluation": {"understanding_score_0_to_10": 6, "gaps": ["outer joins"]}}]}
        with mock.patch.object(insights, "build_summary", return_value=_EMPTY_PROGRESS), \
             _multi(rows):
            digest, activity = insights._activity_digest("u1")
        self.assertEqual(digest["recent_items"]["exams"][0]["score"], "7/10")
        self.assertEqual(digest["recent_items"]["teachbacks"][0]["understanding"], "6/10")
        self.assertEqual(activity, 2)


def _fake_providers():
    """scripts/fake_providers.py loaded by path (scripts/ is not a package); its canned replies stand in for a model."""
    path = Path(__file__).resolve().parent.parent / "scripts" / "fake_providers.py"
    spec = importlib.util.spec_from_file_location("fake_providers_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ParserRealisticReplyTests(unittest.TestCase):
    """The analysis parser must accept what models really send: fenced JSON, prose around it, truncation."""
    def setUp(self):
        insights._cache.clear()
        insights._last_refresh.clear()

    def test_fenced_json_with_surrounding_prose_is_accepted(self):
        reply = ("Sure! Here is your study read-out:\n```json\n" + _GOOD.replace(", ", ",\n  ") +
                 "\n```\nLet me know if you want more detail.")
        ctx, _ = _chat(reply)
        with mock.patch.object(insights, "build_summary", return_value=_ACTIVE), _multi([{"title": "Quiz · DBMS"}]), ctx:
            out = insights.build_insights("u1")
        self.assertEqual(out["insights"]["headline"], "Solid start!")
        self.assertEqual(out["insights"]["focus_topics"], ["B-trees"])

    def test_fake_provider_reply_to_the_real_prompt_is_accepted(self):
        """The dev/load-test fake must answer the insights prompt (digest embeds 'flashcards' etc.) with a valid analysis."""
        fp = _fake_providers()
        seen = {}

        def fake_chat(system, messages, **kwargs):
            seen["system"], seen["user"] = system, messages[-1]["content"]
            return result(fp.make_reply(system, messages[-1]["content"], kwargs.get("json_mode", False)))

        digest_rows = {"flashcards": [{"title": "Review · Bio"}], "quizzes": [{"title": "Quiz · Bio"}]}
        with mock.patch("backend.llm_chain.chat", side_effect=fake_chat), \
             mock.patch.object(insights, "build_summary", return_value=_ACTIVE), _multi(digest_rows):
            out = insights.build_insights("u1")
        self.assertIn('"flashcards"', seen["user"])
        self.assertFalse(out["empty"])
        self.assertTrue(out["insights"]["headline"])
        self.assertEqual(out["insights"]["recommendations"][0]["tool"], "quiz")
        json.dumps(out)


class HttpTests(unittest.TestCase):
    def setUp(self):
        from tests._study_helpers import make_client
        from backend.routers import insights as router
        insights._cache.clear()
        self.client = make_client(router.router, owner="u1")

    def test_endpoint_returns_payload_with_llm(self):
        payload = {"empty": False, "insights": {"headline": "h", "observations": [],
                   "recommendations": [], "focus_topics": []}, "generated_at": "2026-06-12T00:00:00+00:00",
                   "llm": {"label": "Model: groq · x"}}
        with mock.patch("backend.routers.insights.require_llm_key", lambda: None), \
             mock.patch("backend.routers.insights.build_insights", return_value=payload):
            resp = self.client.get("/api/insights")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["insights"]["headline"], "h")
        self.assertEqual(resp.json()["llm"]["label"], "Model: groq · x")

    def test_all_providers_failed_is_503_with_label(self):
        with mock.patch("backend.routers.insights.require_llm_key", lambda: None), \
             mock.patch("backend.routers.insights.build_insights", side_effect=chain_error()):
            resp = self.client.get("/api/insights")
        self.assertEqual(resp.status_code, 503)
        self.assertEqual(resp.json()["llm"]["label"], "Model: none (all providers failed)")

    def test_unusable_analysis_is_502(self):
        with mock.patch("backend.routers.insights.require_llm_key", lambda: None), \
             mock.patch("backend.routers.insights.build_insights", side_effect=insights.InsightsError("bad")):
            self.assertEqual(self.client.get("/api/insights").status_code, 502)


if __name__ == "__main__":
    unittest.main()
