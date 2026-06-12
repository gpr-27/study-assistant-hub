"""Tests for the AI insights service + router (cross-tool study analysis).

No network: Groq and db are mocked. Covers the empty-activity fast path (no LLM
call), digest assembly, analysis parsing/validation, the per-day cache (+refresh
bypass), and the HTTP endpoint.
"""
import unittest
from unittest import mock

from backend.services import insights


def _mk_groq_reply(text):
    """Build a mocked Groq client whose chat.completions.create returns text."""
    client = mock.MagicMock()
    client.chat.completions.create.return_value = mock.MagicMock(
        choices=[mock.MagicMock(message=mock.MagicMock(content=text))]
    )
    return client


_GOOD = (
    '{"headline": "Solid start!", "observations": ["You aced DBMS"],'
    ' "recommendations": [{"action": "Take a B-tree quiz", "reason": "weak topic", "tool": "quiz"}],'
    ' "focus_topics": ["B-trees"]}'
)

_EMPTY_PROGRESS = {
    "overall": {"total_attempts": 0, "overall_accuracy": 0.0, "streak_days": 0},
    "subjects": [], "weakest": [],
}


class BuildInsightsTests(unittest.TestCase):
    def setUp(self):
        insights._cache.clear()

    def test_no_activity_returns_empty_without_llm(self):
        with mock.patch.object(insights, "build_summary", return_value=_EMPTY_PROGRESS), \
             mock.patch.object(insights.db, "list_items", return_value=[]), \
             mock.patch.object(insights, "Groq") as groq:
            out = insights.build_insights("u1")
        self.assertTrue(out["empty"])
        groq.assert_not_called()

    def test_activity_produces_parsed_insights(self):
        progress = dict(_EMPTY_PROGRESS, overall={"total_attempts": 4, "overall_accuracy": 0.5, "streak_days": 2})
        with mock.patch.object(insights, "build_summary", return_value=progress), \
             mock.patch.object(insights.db, "list_items", return_value=[{"title": "Quiz · DBMS"}]), \
             mock.patch.object(insights, "Groq", return_value=_mk_groq_reply(_GOOD)):
            out = insights.build_insights("u1")
        self.assertFalse(out["empty"])
        self.assertEqual(out["insights"]["headline"], "Solid start!")
        self.assertEqual(out["insights"]["recommendations"][0]["tool"], "quiz")
        self.assertIsNotNone(out["generated_at"])

    def test_cached_per_day_and_refresh_bypasses(self):
        progress = dict(_EMPTY_PROGRESS, overall={"total_attempts": 4, "overall_accuracy": 0.5, "streak_days": 2})
        with mock.patch.object(insights, "build_summary", return_value=progress), \
             mock.patch.object(insights.db, "list_items", return_value=[{"title": "t"}]), \
             mock.patch.object(insights, "Groq", return_value=_mk_groq_reply(_GOOD)) as groq:
            insights.build_insights("u1")
            insights.build_insights("u1")           # cache hit
            self.assertEqual(groq.call_count, 1)
            insights.build_insights("u1", refresh=True)  # bypass
            self.assertEqual(groq.call_count, 2)

    def test_unusable_analysis_raises(self):
        progress = dict(_EMPTY_PROGRESS, overall={"total_attempts": 1, "overall_accuracy": 1.0, "streak_days": 1})
        with mock.patch.object(insights, "build_summary", return_value=progress), \
             mock.patch.object(insights.db, "list_items", return_value=[]), \
             mock.patch.object(insights, "Groq", return_value=_mk_groq_reply("not json")):
            with self.assertRaises(insights.InsightsError):
                insights.build_insights("u1")

    def test_digest_includes_exam_scores_and_teachback_gaps(self):
        def fake_list(kind, owner, limit=100):
            if kind == "exams":
                return [{"title": "Exam · DBMS", "total_score": 7, "max_score": 10}]
            if kind == "teachbacks":
                return [{"title": "Teach-back · Joins",
                         "evaluation": {"understanding_score_0_to_10": 6, "gaps": ["outer joins"]}}]
            return []
        with mock.patch.object(insights, "build_summary", return_value=_EMPTY_PROGRESS), \
             mock.patch.object(insights.db, "list_items", side_effect=fake_list):
            digest, activity = insights._activity_digest("u1")
        self.assertEqual(digest["recent_items"]["exams"][0]["score"], "7/10")
        self.assertEqual(digest["recent_items"]["teachbacks"][0]["understanding"], "6/10")
        self.assertEqual(activity, 2)


class HttpTests(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient
        from backend.main import app
        insights._cache.clear()
        self.client = TestClient(app)

    def test_endpoint_returns_payload(self):
        payload = {"empty": False, "insights": {"headline": "h", "observations": [],
                   "recommendations": [], "focus_topics": []}, "generated_at": "2026-06-12T00:00:00+00:00"}
        with mock.patch("backend.routers.insights.require_groq_key", lambda: None), \
             mock.patch("backend.routers.insights.build_insights", return_value=payload):
            resp = self.client.get("/api/insights")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["insights"]["headline"], "h")


if __name__ == "__main__":
    unittest.main()
