"""Tests for the anonymous guest account system.

Persistence (MongoDB) and Clerk are OFF in the test environment (see tests/__init__.py),
so these tests mock ``backend.db`` and the Clerk token verifier — no real database or
Clerk instance is touched. They cover: config exposure, guest creation, session
restoration, owner resolution (Clerk > guest > local), the migration/upgrade route, and
the graceful no-op behaviour of every db helper when persistence is disabled.
"""
import unittest
from unittest import mock

from fastapi.testclient import TestClient

from backend import config, db
from backend.main import app


class GuestConfigTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_config_exposes_guest_flag(self):
        body = self.client.get("/api/config").json()
        self.assertIn("guestEnabled", body)
        # Persistence is off in tests, so guests are off too.
        self.assertFalse(body["guestEnabled"])


class GuestCreationTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_create_guest_requires_persistence(self):
        # Persistence off -> the route is unavailable (503), never a silent success.
        resp = self.client.post("/api/guest")
        self.assertEqual(resp.status_code, 503)
        self.assertIn("MONGODB_URI", resp.json()["detail"])

    def test_create_guest_returns_record(self):
        guest = {
            "guestId": "guest_abc123",
            "isGuest": True,
            "displayName": "Guest User",
            "createdAt": "2026-06-02T00:00:00+00:00",
            "lastActiveAt": "2026-06-02T00:00:00+00:00",
        }
        with mock.patch.object(config, "GUEST_ENABLED", True), \
             mock.patch.object(db, "is_enabled", return_value=True), \
             mock.patch.object(db, "create_guest", return_value=guest):
            resp = self.client.post("/api/guest")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["guestId"], "guest_abc123")
        self.assertTrue(body["isGuest"])
        self.assertEqual(body["displayName"], "Guest User")


class GuestRestoreTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def _persistence_on(self):
        return (mock.patch.object(config, "GUEST_ENABLED", True),
                mock.patch.object(db, "is_enabled", return_value=True))

    def test_restore_missing_header(self):
        with mock.patch.object(config, "GUEST_ENABLED", True), \
             mock.patch.object(db, "is_enabled", return_value=True):
            resp = self.client.get("/api/guest/me")
        self.assertEqual(resp.status_code, 400)

    def test_restore_not_found(self):
        with mock.patch.object(config, "GUEST_ENABLED", True), \
             mock.patch.object(db, "is_enabled", return_value=True), \
             mock.patch.object(db, "get_guest", return_value=None):
            resp = self.client.get("/api/guest/me", headers={"X-Guest-Id": "guest_missing"})
        self.assertEqual(resp.status_code, 404)

    def test_restore_ok_and_touches(self):
        guest = {"guestId": "guest_live", "isGuest": True, "displayName": "Guest User"}
        with mock.patch.object(config, "GUEST_ENABLED", True), \
             mock.patch.object(db, "is_enabled", return_value=True), \
             mock.patch.object(db, "get_guest", return_value=guest), \
             mock.patch.object(db, "touch_guest", return_value=True) as touched:
            resp = self.client.get("/api/guest/me", headers={"X-Guest-Id": "guest_live"})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["guestId"], "guest_live")
        touched.assert_called_once_with("guest_live")


class OwnerResolutionTests(unittest.TestCase):
    """current_owner must resolve Clerk > guest > local (via any history-scoped route)."""

    def setUp(self):
        self.client = TestClient(app)

    def _capture_owner(self):
        seen = {}

        def fake_list(kind, owner, q=None, limit=100):
            seen["owner"] = owner
            return []

        return seen, fake_list

    def test_guest_header_becomes_owner(self):
        seen, fake_list = self._capture_owner()
        with mock.patch.object(db, "touch_guest", return_value=True), \
             mock.patch.object(db, "is_enabled", return_value=True), \
             mock.patch.object(db, "list_items", side_effect=fake_list):
            resp = self.client.get("/api/history/chats", headers={"X-Guest-Id": "guest_xyz"})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(seen["owner"], "guest_xyz")

    def test_unknown_guest_falls_back_to_local_when_auth_off(self):
        seen, fake_list = self._capture_owner()
        with mock.patch.object(db, "touch_guest", return_value=False), \
             mock.patch.object(db, "is_enabled", return_value=True), \
             mock.patch.object(db, "list_items", side_effect=fake_list):
            resp = self.client.get("/api/history/chats", headers={"X-Guest-Id": "guest_stale"})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(seen["owner"], "local")

    def test_no_identity_is_local_when_auth_off(self):
        seen, fake_list = self._capture_owner()
        with mock.patch.object(db, "is_enabled", return_value=True), \
             mock.patch.object(db, "list_items", side_effect=fake_list):
            resp = self.client.get("/api/history/chats")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(seen["owner"], "local")

    def test_clerk_session_wins_over_stale_guest_header(self):
        seen, fake_list = self._capture_owner()
        with mock.patch.object(config, "AUTH_ENABLED", True), \
             mock.patch("backend.auth._verify_clerk_token", return_value="clerk_user_1"), \
             mock.patch.object(db, "is_enabled", return_value=True), \
             mock.patch.object(db, "touch_guest", return_value=True) as touched, \
             mock.patch.object(db, "list_items", side_effect=fake_list):
            resp = self.client.get(
                "/api/history/chats",
                headers={"Authorization": "Bearer faketoken", "X-Guest-Id": "guest_old"},
            )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(seen["owner"], "clerk_user_1")
        touched.assert_not_called()  # Clerk wins before the guest path is consulted

    def test_auth_enabled_no_identity_is_401(self):
        with mock.patch.object(config, "AUTH_ENABLED", True), \
             mock.patch("backend.auth._verify_clerk_token", return_value=None):
            resp = self.client.get("/api/history/chats")
        self.assertEqual(resp.status_code, 401)


class GuestMigrationTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_migrate_requires_clerk_when_auth_disabled(self):
        # Auth off -> there is no account to migrate into.
        resp = self.client.post("/api/guest/migrate", headers={"X-Guest-Id": "guest_abc"})
        self.assertEqual(resp.status_code, 400)

    def test_migrate_requires_signed_in_user(self):
        with mock.patch.object(config, "AUTH_ENABLED", True), \
             mock.patch.object(config, "GUEST_ENABLED", True), \
             mock.patch.object(db, "is_enabled", return_value=True), \
             mock.patch("backend.auth._verify_clerk_token", return_value=None):
            resp = self.client.post("/api/guest/migrate", headers={"X-Guest-Id": "guest_abc"})
        self.assertEqual(resp.status_code, 401)

    def test_migrate_requires_guest_id(self):
        with mock.patch.object(config, "AUTH_ENABLED", True), \
             mock.patch.object(config, "GUEST_ENABLED", True), \
             mock.patch.object(db, "is_enabled", return_value=True), \
             mock.patch("backend.auth._verify_clerk_token", return_value="clerk_1"):
            resp = self.client.post("/api/guest/migrate", headers={"Authorization": "Bearer t"})
        self.assertEqual(resp.status_code, 400)

    def test_migrate_moves_data(self):
        result = {"migrated": True, "total": 3,
                  "moved": {"chats": 2, "quizzes": 1, "flashcards": 0, "plans": 0, "mindmaps": 0}}
        with mock.patch.object(config, "AUTH_ENABLED", True), \
             mock.patch.object(config, "GUEST_ENABLED", True), \
             mock.patch.object(db, "is_enabled", return_value=True), \
             mock.patch("backend.auth._verify_clerk_token", return_value="clerk_1"), \
             mock.patch.object(db, "migrate_guest", return_value=result) as migrated:
            resp = self.client.post(
                "/api/guest/migrate",
                headers={"Authorization": "Bearer t", "X-Guest-Id": "guest_abc"},
            )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertTrue(body["migrated"])
        self.assertEqual(body["total"], 3)
        migrated.assert_called_once_with("guest_abc", "clerk_1")


class MigrateGuestOrderingTests(unittest.TestCase):
    """migrate_guest must re-key artifacts BEFORE deleting the guest record (crash-safety),
    and must no-op when the guest is unknown/already migrated. Verified against a fake db so
    no real MongoDB is needed."""

    def _fake_db(self, order, guest_doc):
        guests_col = mock.MagicMock()
        guests_col.find_one.return_value = guest_doc
        guests_col.delete_one.side_effect = lambda *a, **k: (order.append("delete"), mock.MagicMock(deleted_count=1))[1]
        kind_cols = {}

        def getitem(name):
            if name == db.GUESTS:
                return guests_col
            col = kind_cols.get(name)
            if col is None:
                col = mock.MagicMock()
                col.update_many.side_effect = (
                    lambda *a, _n=name, **k: (order.append(("rekey", _n)), mock.MagicMock(modified_count=1))[1]
                )
                kind_cols[name] = col
            return col

        fake = mock.MagicMock()
        fake.__getitem__.side_effect = getitem
        return fake

    def test_rekeys_all_kinds_before_deleting_guest(self):
        order = []
        fake = self._fake_db(order, {"guestId": "guest_x", "migratedTo": None})
        with mock.patch.object(db, "_get_db", return_value=fake):
            result = db.migrate_guest("guest_x", "clerk_1")
        self.assertTrue(result["migrated"])
        self.assertEqual(result["total"], len(db.KINDS))
        # The guest record is deleted LAST, after every artifact re-key.
        self.assertEqual(order[-1], "delete")
        self.assertEqual(len([c for c in order if c != "delete"]), len(db.KINDS))
        self.assertTrue(all(c[0] == "rekey" for c in order[:-1]))

    def test_noop_when_guest_unknown_or_already_migrated(self):
        order = []
        fake = self._fake_db(order, None)  # find_one returns nothing
        with mock.patch.object(db, "_get_db", return_value=fake):
            result = db.migrate_guest("guest_gone", "clerk_1")
        self.assertFalse(result["migrated"])
        self.assertEqual(result["total"], 0)
        self.assertEqual(order, [])  # no re-key, no delete


class DbGracefulNoOpTests(unittest.TestCase):
    """Every guest db helper must be a safe no-op when persistence is disabled."""

    def test_helpers_no_op_without_persistence(self):
        self.assertIsNone(db.create_guest())
        self.assertIsNone(db.get_guest("guest_x"))
        self.assertFalse(db.touch_guest("guest_x"))
        self.assertEqual(db.purge_orphan_guest_data(), 0)
        result = db.migrate_guest("guest_x", "clerk_1")
        self.assertFalse(result["migrated"])
        self.assertEqual(result["total"], 0)

    def test_start_janitor_no_op_without_persistence(self):
        # Should not raise or spawn anything when persistence is off.
        db.start_guest_janitor()

    def test_public_guest_shapes_document(self):
        from datetime import datetime, timezone
        doc = {
            "_id": "deadbeef",
            "guestId": "guest_1",
            "isGuest": True,
            "displayName": "Guest User",
            "createdAt": datetime(2026, 6, 2, tzinfo=timezone.utc),
            "lastActiveAt": datetime(2026, 6, 2, tzinfo=timezone.utc),
        }
        out = db._public_guest(doc)
        self.assertNotIn("_id", out)
        self.assertEqual(out["guestId"], "guest_1")
        self.assertIsInstance(out["createdAt"], str)  # ISO-formatted
        self.assertTrue(out["createdAt"].startswith("2026-06-02"))


if __name__ == "__main__":
    unittest.main()
