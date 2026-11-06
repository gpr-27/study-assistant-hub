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


def _recent_for(attempts):
    out = {k: {"n": 0, "rows": []} for k, _l, _r in progress._ACTIVITY_TOOLS}
    out["attempts"] = {"n": len(attempts), "rows": list(attempts)}
    return out


def _patch_recent(attempts):
    """Patch the single aggregated read (db.multi_recent) with these attempt rows."""
    return mock.patch.object(progress.db, "multi_recent", side_effect=lambda owner, specs: _recent_for(attempts))


class BuildSummaryTests(unittest.TestCase):
    def setUp(self):
        progress.clear_cache()
        self.addCleanup(progress.clear_cache)

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
        with _patch_recent(items):
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
        with _patch_recent(items):
            data = progress.build_summary(OWNER)
        self.assertEqual([w["topic"] for w in data["weakest"]], ["Hard", "Easy"])
        self.assertAlmostEqual(data["weakest"][0]["accuracy"], 0.1, places=3)

    def test_streak_counts_consecutive_days(self):
        items = [_item(1, 2, days_ago=d) for d in (0, 1, 2, 5)]  # gap at day 3
        with _patch_recent(items):
            data = progress.build_summary(OWNER)
        self.assertEqual(data["overall"]["streak_days"], 3)

    def test_streak_alive_when_last_attempt_was_yesterday(self):
        items = [_item(1, 2, days_ago=1), _item(1, 2, days_ago=2)]
        with _patch_recent(items):
            data = progress.build_summary(OWNER)
        self.assertEqual(data["overall"]["streak_days"], 2)

    def test_streak_zero_when_stale(self):
        items = [_item(1, 2, days_ago=3)]
        with _patch_recent(items):
            data = progress.build_summary(OWNER)
        self.assertEqual(data["overall"]["streak_days"], 0)

    def test_malformed_items_are_skipped(self):
        items = [
            {"correct": "x", "total": "y", "created_at": _iso(0)},
            {"correct": 1, "total": 0, "created_at": _iso(0)},
            _item(2, 2, topic="Good"),
        ]
        with _patch_recent(items):
            data = progress.build_summary(OWNER)
        self.assertEqual(data["overall"]["total_attempts"], 1)
        self.assertEqual(data["overall"]["overall_accuracy"], 1.0)


class ProgressApiTests(unittest.TestCase):
    def setUp(self):
        progress.clear_cache()
        self.addCleanup(progress.clear_cache)
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
        with _patch_recent(items) as li:
            res = self.client.get("/api/progress/summary")
        self.assertEqual(res.status_code, 200)
        # attempts feed the accuracy stats; the unified dashboard ALSO queries
        # every other tool's kind for the all-app activity grid.
        specs = li.call_args.args[1]
        self.assertEqual(specs[0][0], "attempts")
        self.assertEqual(li.call_count, 1)             # ONE Mongo round trip for the whole dashboard
        data = res.json()
        self.assertEqual(data["overall"]["total_attempts"], 2)
        self.assertEqual(data["subjects"][0]["subject"], "Maths")
        self.assertEqual(data["weakest"][0]["topic"], "Algebra")
        self.assertAlmostEqual(data["weakest"][0]["accuracy"], 0.5, places=3)
        self.assertTrue(any(a["tool"] == "exams" for a in data["activity"]))


class HardeningTests(unittest.TestCase):
    def test_total_is_capped_and_labels_are_truncated(self):
        with mock.patch.object(progress.db, "save", return_value="x") as save:
            progress.record_attempt(OWNER, "quiz", correct=10**9, total=10**9, subject="S" * 900, topic="T" * 900)
        doc = save.call_args.args[2]
        self.assertEqual(doc["total"], progress._MAX_TOTAL)
        self.assertEqual(doc["correct"], progress._MAX_TOTAL)
        self.assertLessEqual(len(doc["subject"]), progress._MAX_LABEL)

    def test_non_integer_counts_are_a_progress_error_not_a_500(self):
        with self.assertRaises(progress.ProgressError):
            progress.record_attempt(OWNER, "quiz", correct="abc", total=5)

    def test_router_rejects_absurd_payloads_with_422(self):
        client = _app()
        self.assertEqual(client.post("/api/progress/record", json={"tool": "quiz", "correct": 1, "total": 10**7}).status_code, 422)
        self.assertEqual(client.post("/api/progress/record",
                                     json={"tool": "quiz", "correct": 1, "total": 2, "subject": "x" * 500}).status_code, 422)

    def test_summary_is_one_owner_scoped_read(self):
        with _patch_recent([]) as mr:
            out = progress.build_summary(OWNER)
        self.assertEqual(mr.call_count, 1)
        self.assertEqual(mr.call_args.args[0], OWNER)
        self.assertEqual(len(out["activity"]), len(progress._ACTIVITY_TOOLS))


class SummaryCacheTests(unittest.TestCase):
    def setUp(self):
        progress.clear_cache()
        self.addCleanup(progress.clear_cache)

    def test_second_call_is_served_from_cache(self):
        with _patch_recent([_item(1, 2)]) as mr:
            first = progress.build_summary(OWNER)
            second = progress.build_summary(OWNER)
        self.assertEqual(mr.call_count, 1)
        self.assertEqual(first, second)
        second["overall"]["total_attempts"] = 99                 # callers may mutate their copy
        with _patch_recent([_item(1, 2)]):
            self.assertEqual(progress.build_summary(OWNER)["overall"]["total_attempts"], 1)

    def test_a_write_for_the_owner_invalidates(self):
        with _patch_recent([_item(1, 2)]) as mr:
            progress.build_summary(OWNER)
            with mock.patch.object(progress.db, "_get_db", return_value={"attempts": mock.MagicMock()}):
                progress.db.save("attempts", OWNER, {"correct": 1, "total": 2})
            progress.build_summary(OWNER)
        self.assertEqual(mr.call_count, 2)

    def test_a_write_for_another_owner_does_not_invalidate(self):
        with _patch_recent([_item(1, 2)]) as mr:
            progress.build_summary(OWNER)
            with mock.patch.object(progress.db, "_get_db", return_value={"attempts": mock.MagicMock()}):
                progress.db.save("attempts", "someone_else", {"correct": 1, "total": 2})
            progress.build_summary(OWNER)
        self.assertEqual(mr.call_count, 1)

    def test_ttl_expiry(self):
        with _patch_recent([_item(1, 2)]) as mr, mock.patch.object(progress, "_CACHE_TTL_S", -1.0):
            progress.build_summary(OWNER)
            progress.build_summary(OWNER)
        self.assertEqual(mr.call_count, 2)

    def test_owners_never_share_entries(self):
        def per_owner(owner, specs):
            return _recent_for([_item(5, 5)] if owner == "alice" else [_item(0, 5), _item(0, 5)])

        with mock.patch.object(progress.db, "multi_recent", side_effect=per_owner):
            a = progress.build_summary("alice")
            b = progress.build_summary("bob")
            a2 = progress.build_summary("alice")
        self.assertEqual(a["overall"]["total_attempts"], 1)
        self.assertEqual(b["overall"]["total_attempts"], 2)
        self.assertEqual(a2, a)

    def test_cache_is_bounded(self):
        with _patch_recent([]), mock.patch.object(progress, "_CACHE_MAX", 3):
            for i in range(10):
                progress.build_summary(f"o{i}")
        self.assertLessEqual(len(progress._cache), 3)


if __name__ == "__main__":
    unittest.main()
