"""History timestamps must serialize with an explicit UTC offset.

PyMongo returns stored (UTC) datetimes naive; serializing them without an
offset made browsers parse them as local time, so every history entry showed
shifted by the viewer's UTC offset (regression: "history noted wrong time").
"""
import unittest
from datetime import datetime, timezone

from backend import db


class TimestampSerializationTests(unittest.TestCase):
    def test_public_tags_naive_datetimes_as_utc(self):
        doc = {"_id": "x", "owner": "u", "title": "t",
               "created_at": datetime(2026, 6, 12, 18, 25, 0),          # naive (as PyMongo returns)
               "updated_at": datetime(2026, 6, 12, 18, 30, 0, tzinfo=timezone.utc)}  # already aware
        out = db._public(doc)
        self.assertEqual(out["created_at"], "2026-06-12T18:25:00+00:00")
        self.assertEqual(out["updated_at"], "2026-06-12T18:30:00+00:00")

    def test_public_guest_tags_naive_datetimes_as_utc(self):
        out = db._public_guest({"_id": "x", "guestId": "g",
                                "createdAt": datetime(2026, 6, 12, 1, 2, 3)})
        self.assertTrue(out["createdAt"].endswith("+00:00"))


if __name__ == "__main__":
    unittest.main()
