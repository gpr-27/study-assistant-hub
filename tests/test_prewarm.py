"""Cold-start prewarm: one-shot, non-blocking, skipped in tests / without Mongo, no double connect."""
import threading
import time
import unittest
from unittest import mock

from backend import config, db, prewarm


class DbPrewarmTests(unittest.TestCase):
    def setUp(self):
        self._saved = (db._prewarm_started, db._prewarm_thread, db._prewarm_done.is_set())
        db._prewarm_started = False
        db._prewarm_thread = None
        db._prewarm_done.clear()
        self.addCleanup(self._restore)

    def _restore(self):
        db._prewarm_started, db._prewarm_thread = self._saved[0], self._saved[1]
        db._prewarm_done.set() if self._saved[2] else db._prewarm_done.clear()

    def test_noop_when_persistence_disabled(self):
        with mock.patch.object(config, "PERSISTENCE_ENABLED", False):
            self.assertFalse(db.prewarm())
        self.assertFalse(db._prewarm_started)

    def test_connects_once_in_background_and_is_idempotent(self):
        calls = []
        gate = threading.Event()
        fake = mock.MagicMock()
        fake.client.admin.command.side_effect = lambda name: calls.append(name) or gate.wait(2)
        with mock.patch.object(config, "PERSISTENCE_ENABLED", True), \
                mock.patch.object(db, "_get_db", return_value=fake):
            t0 = time.perf_counter()
            self.assertTrue(db.prewarm())
            self.assertLess(time.perf_counter() - t0, 0.5)         # never blocks the caller
            self.assertFalse(db.prewarm())                          # double start guarded
            self.assertFalse(db.prewarm())
            gate.set()
            self.assertTrue(db._prewarm_done.wait(2))
        self.assertEqual(calls, ["hello"])

    def test_failure_still_releases_waiters(self):
        with mock.patch.object(config, "PERSISTENCE_ENABLED", True), \
                mock.patch.object(db, "_get_db", side_effect=RuntimeError("boom")):
            db.prewarm()
            self.assertTrue(db._prewarm_done.wait(2))

    def test_request_waits_for_inflight_connect_instead_of_second_client(self):
        """A request arriving mid-prewarm blocks until it finishes and then reuses the same handle."""
        sentinel = object()
        db._prewarm_started = True
        db._prewarm_thread = threading.Thread()          # some other thread is "the prewarm"
        got = []
        with mock.patch.object(config, "PERSISTENCE_ENABLED", True), \
                mock.patch.object(db, "_db", None), mock.patch.object(db, "_init_failed", False):
            t = threading.Thread(target=lambda: got.append(db._get_db()))
            t.start()
            time.sleep(0.15)
            self.assertTrue(t.is_alive())                 # waiting on the in-flight connect
            db._db = sentinel                             # the prewarm thread finished building the client
            db._prewarm_done.set()
            t.join(2)
            db._db = None
        self.assertEqual(got, [sentinel])


class PrewarmModuleTests(unittest.TestCase):
    def test_skipped_under_pytest_or_when_disabled(self):
        self.assertFalse(prewarm.enabled())                # pytest is in sys.modules
        self.assertFalse(prewarm.start())

    def test_start_is_one_shot(self):
        started = []
        with mock.patch.object(prewarm, "enabled", return_value=True), mock.patch.object(prewarm, "_started", False), \
                mock.patch.object(prewarm.threading, "Thread") as th:
            th.side_effect = lambda **kw: started.append(kw["name"]) or mock.MagicMock()
            self.assertTrue(prewarm.start())
            self.assertFalse(prewarm.start())
        self.assertEqual(sorted(started), ["prewarm-clients", "prewarm-mongo"])

    def test_pool_keeps_one_connection_warm(self):
        self.assertEqual(db._POOL_KWARGS["minPoolSize"], 1)
        self.assertEqual(db._POOL_KWARGS["maxPoolSize"], 10)


if __name__ == "__main__":
    unittest.main()
