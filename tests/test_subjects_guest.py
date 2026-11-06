"""Guest router hardening + guest->account upload migration (hermetic; no Mongo/Clerk)."""
import unittest
from unittest import mock

from fastapi.testclient import TestClient

from backend import config, db
from backend.main import app
from backend.routers import guest as guest_router
from backend.services import subjects as svc

_GUEST = {"guestId": "guest_abc", "isGuest": True, "displayName": "Guest User"}


def _on():
    return (mock.patch.object(config, "GUEST_ENABLED", True),
            mock.patch.object(db, "is_enabled", return_value=True))


class GuestHardeningTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app, raise_server_exceptions=False)
        guest_router._reset_create_throttle()
        self.addCleanup(guest_router._reset_create_throttle)

    def test_malformed_or_foreign_ids_never_reach_the_db(self):
        a, b = _on()
        with a, b, mock.patch.object(db, "get_guest") as get:
            for bad in ("user_2abcDEF", "local", "guest_", "guest_$ne", "guest_" + "a" * 60, "../guest_x", "guest_a b"):
                r = self.client.get("/api/guest/me", headers={"X-Guest-Id": bad})
                self.assertEqual(r.status_code, 404, bad)
        get.assert_not_called()

    def test_create_is_throttled_per_ip(self):
        a, b = _on()
        with a, b, mock.patch.object(db, "create_guest", return_value={**_GUEST}) as create, \
                mock.patch.object(guest_router, "_CREATE_LIMIT", 3):
            codes = [self.client.post("/api/guest", headers={"x-real-ip": "1.2.3.4"}).status_code for _ in range(5)]
            other = self.client.post("/api/guest", headers={"x-real-ip": "5.6.7.8"}).status_code
        self.assertEqual(codes, [200, 200, 200, 429, 429])
        self.assertEqual(other, 200)
        self.assertEqual(create.call_count, 4)

    def test_disabled_guest_mode_does_not_consume_throttle_or_touch_db(self):
        with mock.patch.object(db, "create_guest") as create:
            self.assertEqual(self.client.post("/api/guest").status_code, 503)
        create.assert_not_called()


class MigrationTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app, raise_server_exceptions=False)

    def _post(self, **patches):
        result = {"migrated": True, "total": 1, "moved": {"chats": 1}}
        with mock.patch.object(config, "AUTH_ENABLED", True), mock.patch.object(config, "GUEST_ENABLED", True), \
                mock.patch.object(db, "is_enabled", return_value=True), \
                mock.patch("backend.auth._verify_clerk_token", return_value="user_clerk1"), \
                mock.patch.object(db, "migrate_guest", return_value=result) as mig, \
                mock.patch.object(db, "get_guest", return_value=patches.get("live")), \
                mock.patch.object(svc, "migrate_guest_uploads", return_value=patches.get("moved", 0)) as up:
            r = self.client.post("/api/guest/migrate",
                                 headers={"Authorization": "Bearer t", "X-Guest-Id": "guest_abc"})
        return r, mig, up

    def test_uploads_are_moved_only_after_the_atomic_claim_and_before_the_record_is_retired(self):
        order = []
        result = {"migrated": True, "total": 1, "moved": {"chats": 1}}
        with mock.patch.object(config, "AUTH_ENABLED", True), mock.patch.object(config, "GUEST_ENABLED", True), \
                mock.patch.object(db, "is_enabled", return_value=True), \
                mock.patch("backend.auth._verify_clerk_token", return_value="user_clerk1"), \
                mock.patch.object(db, "get_guest", return_value=_GUEST), \
                mock.patch.object(svc, "migrate_guest_uploads", side_effect=lambda *a: order.append("uploads") or 2), \
                mock.patch.object(db, "migrate_guest",
                                  side_effect=lambda g, t, after_claim=None: (
                                      order.append("claim"),
                                      {**result, "subjects_moved": after_claim()},
                                      order.append("artifacts"))[1]):
            r = self.client.post("/api/guest/migrate",
                                 headers={"Authorization": "Bearer t", "X-Guest-Id": "guest_abc"})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(order, ["claim", "uploads", "artifacts"])
        self.assertEqual(r.json()["moved"]["subjects"], 2)

    def test_already_migrated_guest_does_not_touch_uploads(self):
        r, mig, up = self._post(live=None)
        self.assertEqual(r.status_code, 200)
        up.assert_not_called()
        self.assertEqual(mig.call_args.args, ("guest_abc", "user_clerk1"))

    def test_guest_id_cannot_be_the_target(self):
        with mock.patch.object(config, "AUTH_ENABLED", True), mock.patch.object(config, "GUEST_ENABLED", True), \
                mock.patch.object(db, "is_enabled", return_value=True), \
                mock.patch("backend.auth._verify_clerk_token", return_value=None), \
                mock.patch.object(db, "migrate_guest") as mig:
            r = self.client.post("/api/guest/migrate", headers={"X-Guest-Id": "guest_abc"})
        self.assertEqual(r.status_code, 401)
        mig.assert_not_called()


class MigrateGuestUploadsTests(unittest.TestCase):
    def _fake_db(self):
        cols = {}
        fake = mock.MagicMock()
        fake.__getitem__.side_effect = lambda n: cols.setdefault(n, mock.MagicMock())
        return fake, cols

    def test_rekeys_pdfs_and_index_blobs_and_resolves_name_clashes(self):
        fake, cols = self._fake_db()
        listing = {"guest_abc": ["Algo", "Bio"], "user_c1": ["Algo"]}
        with mock.patch.object(db, "_get_db", return_value=fake), \
                mock.patch.object(db, "list_user_subjects", side_effect=lambda o: listing[o]):
            n = svc.migrate_guest_uploads("guest_abc", "user_c1")
        self.assertEqual(n, 2)
        pdfs = cols[f"{db.UPLOADS_BUCKET}.files"].update_many.call_args_list
        idx = cols[f"{db.INDEX_BUCKET}.files"].update_many.call_args_list
        # every update is filtered by the guest as owner, and rewrites to the target
        for call in pdfs + idx:
            self.assertEqual(call.args[0]["owner"], "guest_abc")
            self.assertEqual(call.args[1]["$set"]["owner"], "user_c1")
        by_old = {c.args[0]["subject"]: c.args[1]["$set"]["subject"] for c in pdfs}
        self.assertEqual(by_old, {"Algo": "Algo (guest)", "Bio": "Bio"})   # clash keeps both
        keys = {c.args[0]["subject_key"]: c.args[1]["$set"]["subject_key"] for c in idx}
        self.assertEqual(keys["user__guest_abc__Algo"], "user__user_c1__Algo (guest)")
        self.assertEqual(keys["user__guest_abc__Bio"], "user__user_c1__Bio")

    def test_noop_without_persistence_or_self_target(self):
        self.assertEqual(svc.migrate_guest_uploads("guest_abc", "user_c1"), 0)    # db off in tests
        self.assertEqual(svc.migrate_guest_uploads("guest_abc", "guest_abc"), 0)


if __name__ == "__main__":
    unittest.main()
