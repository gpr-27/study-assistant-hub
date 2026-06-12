"""Tests for the Progress / Weakness Heatmap feature.

Covers the aggregation service (accuracy, streak, weakest topics) and the API
router (record + summary). The database layer is fully mocked — no MongoDB, no
network — and the router is mounted on a throwaway FastAPI app with the
current_owner dependency overridden (it is not wired into main.py here).
"""
import unittest
from datetime import datetime, timedelta, timezone
from unittest import mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.auth import current_owner
from backend.routers.progress import router as progress_router
from backend.services import progress

OWNER = "user_test"


def _iso(days_ago: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()


def _item(correct, total, subject=None, topic=None, days_ago=0):
    return {
        "correct": correct, "total": total,
        "subject": subject, "topic": topic,
        "created_at": _iso(days_ago),
    }


def _app() -> TestClient:
    app = FastAPI()
    app.include_router(progress_router)
    app.dependency_overrides[current_owner] = lambda: OWNER
    return TestClient(app)


class RecordAttemptTests(unittest.TestCase):
    def test_saves_event_with_title_and_clamped_correct(self):
        with mock.patch.object(progress.db, "save", return_value="id123") as save:
            saved = progress.record_attempt(OWNER, "quiz", correct=9, total=5, subject="Maths", topic="Algebra")
        self.assertTrue(saved)
        kind, owner, doc = save.call_args.args
        self.assertEqual((kind, owner), ("attempts", OWNER))
        self.assertEqual(doc["title"], "quiz · Maths")
        self.assertEqual(doc["correct"], 5)  # clamped to total
        self.assertEqual(doc["total"], 5)

    def test_title_falls_back_to_topic_then_general(self):
        with mock.patch.object(progress.db, "save", return_value="x") as save:
            progress.record_attempt(OWNER, "flashcards", correct=1, total=2, topic="Cells")
            progress.record_attempt(OWNER, "flashcards", correct=1, total=2)
        titles = [c.args[2]["title"] for c in save.call_args_list]
        self.assertEqual(titles, ["flashcards · Cells", "flashcards · General"])

    def test_persistence_off_returns_false(self):
        with mock.patch.object(progress.db, "save", return_value=None):
            self.assertFalse(progress.record_attempt(OWNER, "quiz", correct=1, total=2))

    def test_rejects_unknown_tool_and_bad_total(self):
        with self.assertRaises(progress.ProgressError):
            progress.record_attempt(OWNER, "summariser", correct=1, total=2)
        with self.assertRaises(progress.ProgressError):
            progress.record_attempt(OWNER, "quiz", correct=0, total=0)


class BuildSummaryTests(unittest.TestCase):
    def test_empty_history_gives_zeroed_summary(self):
        with mock.patch.object(progress.db, "list_items", return_value=[]):
            data = progress.build_summary(OWNER)
        self.assertEqual(data["overall"], {"total_attempts": 0, "overall_accuracy": 0.0, "streak_days": 0})
        self.assertEqual(data["subjects"], [])
        self.assertEqual(data["weakest"], [])

    def test_aggregates_per_subject_and_topic(self):
        items = [
            _item(4, 5, subject="Maths", topic="Algebra", days_ago=0),
            _item(2, 5, subject="Maths", topic="Algebra", days_ago=1),
            _item(5, 5, subject="Maths", topic="Calculus", days_ago=0),
            _item(1, 4, topic="OS Paging", days_ago=0),
        ]
        with mock.patch.object(progress.db, "list_items", return_value=items):
            data = progress.build_summary(OWNER)
        self.assertEqual(data["overall"]["total_attempts"], 4)
        self.assertAlmostEqual(data["overall"]["overall_accuracy"], 12 / 19, places=3)
        # The no-subject item groups under the "General" subject.
        names = sorted(s["subject"] for s in data["subjects"])
        self.assertEqual(names, ["General", "Maths"])
        maths = next(s for s in data["subjects"] if s["subject"] == "Maths")
        algebra = next(t for t in maths["topics"] if t["topic"] == "Algebra")
        self.assertEqual(algebra["attempts"], 2)
        self.assertAlmostEqual(algebra["accuracy"], 6 / 10, places=3)
        self.assertIsNotNone(algebra["last_seen"])

    def test_weakest_requires_two_attempts_and_sorts_ascending(self):
        items = [
            _item(0, 5, subject="S", topic="Hard"),
            _item(1, 5, subject="S", topic="Hard"),
            _item(5, 5, subject="S", topic="Easy"),
            _item(4, 5, subject="S", topic="Easy"),
            _item(0, 5, subject="S", topic="OneOff"),  # only 1 attempt -> excluded
        ]
        with mock.patch.object(progress.db, "list_items", return_value=items):
            data = progress.build_summary(OWNER)
        self.assertEqual([w["topic"] for w in data["weakest"]], ["Hard", "Easy"])
        self.assertAlmostEqual(data["weakest"][0]["accuracy"], 0.1, places=3)

    def test_streak_counts_consecutive_days(self):
        items = [_item(1, 2, days_ago=d) for d in (0, 1, 2, 5)]  # gap at day 3
        with mock.patch.object(progress.db, "list_items", return_value=items):
            data = progress.build_summary(OWNER)
        self.assertEqual(data["overall"]["streak_days"], 3)

    def test_streak_alive_when_last_attempt_was_yesterday(self):
        items = [_item(1, 2, days_ago=1), _item(1, 2, days_ago=2)]
        with mock.patch.object(progress.db, "list_items", return_value=items):
            data = progress.build_summary(OWNER)
        self.assertEqual(data["overall"]["streak_days"], 2)

    def test_streak_zero_when_stale(self):
        items = [_item(1, 2, days_ago=3)]
        with mock.patch.object(progress.db, "list_items", return_value=items):
            data = progress.build_summary(OWNER)
        self.assertEqual(data["overall"]["streak_days"], 0)

    def test_malformed_items_are_skipped(self):
        items = [
            {"correct": "x", "total": "y", "created_at": _iso(0)},
            {"correct": 1, "total": 0, "created_at": _iso(0)},
            _item(2, 2, topic="Good"),
        ]
        with mock.patch.object(progress.db, "list_items", return_value=items):
            data = progress.build_summary(OWNER)
        self.assertEqual(data["overall"]["total_attempts"], 1)
        self.assertEqual(data["overall"]["overall_accuracy"], 1.0)


class ProgressApiTests(unittest.TestCase):
    def setUp(self):
        self.client = _app()

    def test_record_returns_saved_true(self):
        with mock.patch.object(progress.db, "save", return_value="abc") as save:
            res = self.client.post("/api/progress/record", json={
                "tool": "quiz", "subject": "Maths", "topic": "Algebra", "correct": 3, "total": 5,
            })
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json(), {"saved": True})
        self.assertEqual(save.call_args.args[1], OWNER)

    def test_record_persistence_off_returns_saved_false(self):
        with mock.patch.object(progress.db, "save", return_value=None):
            res = self.client.post("/api/progress/record", json={"tool": "flashcards", "correct": 2, "total": 4})
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json(), {"saved": False})

    def test_record_rejects_unknown_tool(self):
        res = self.client.post("/api/progress/record", json={"tool": "mindmap", "correct": 1, "total": 2})
        self.assertEqual(res.status_code, 422)

    def test_record_validates_total_ge_1(self):
        res = self.client.post("/api/progress/record", json={"tool": "quiz", "correct": 0, "total": 0})
        self.assertEqual(res.status_code, 422)

    def test_summary_shape(self):
        items = [
            _item(4, 5, subject="Maths", topic="Algebra", days_ago=0),
            _item(1, 5, subject="Maths", topic="Algebra", days_ago=0),
        ]
        with mock.patch.object(progress.db, "list_items", return_value=items) as li:
            res = self.client.get("/api/progress/summary")
        self.assertEqual(res.status_code, 200)
        li.assert_called_once_with("attempts", OWNER, limit=100)
        data = res.json()
        self.assertEqual(data["overall"]["total_attempts"], 2)
        self.assertEqual(data["subjects"][0]["subject"], "Maths")
        self.assertEqual(data["weakest"][0]["topic"], "Algebra")
        self.assertAlmostEqual(data["weakest"][0]["accuracy"], 0.5, places=3)


if __name__ == "__main__":
    unittest.main()
