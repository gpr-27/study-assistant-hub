"""Regression tests for the load-test fixes (LOADTEST.md section 4b).

* the httpcore event-loop hygiene dependency (sniffio) is declared and makes the async-library probe cheap;
* the guest TTL touch is cached for 5 minutes (fewer MongoDB writes per session);
* the circuit breaker counts opens and closes (reported by the load test);
* the load-test helpers that read Server-Timing and the cleanup scope behave as documented.
"""
from __future__ import annotations

import asyncio
import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock

from backend import db, llm_chain

ROOT = Path(__file__).resolve().parent.parent


class SniffioDependencyTests(unittest.TestCase):
    def test_requirements_declare_sniffio(self):
        reqs = (ROOT / "requirements.txt").read_text(encoding="utf-8")
        self.assertTrue(any(line.strip().startswith("sniffio") for line in reqs.splitlines()))

    def test_async_library_probe_detects_asyncio_without_filesystem_probing(self):
        from httpcore._synchronization import current_async_library

        async def probe():
            return current_async_library()

        self.assertEqual(asyncio.run(probe()), "asyncio")
        # sniffio importable => no failing `import trio` / sys.path walk per call
        import sniffio  # noqa: F401


class GuestTouchCacheTests(unittest.TestCase):
    class _Coll:
        def __init__(self):
            self.calls = 0

        def find_one_and_update(self, _flt, _upd, **_kw):
            self.calls += 1
            return {}

    def setUp(self):
        db._touched.clear()

    def test_touch_writes_at_most_once_per_interval(self):
        coll = self._Coll()
        fake_db = {db.GUESTS: coll}
        clock = [1000.0]
        with mock.patch.object(db, "_get_db", return_value=fake_db), \
                mock.patch.object(db.time, "monotonic", side_effect=lambda: clock[0]):
            self.assertTrue(db.touch_guest("guest_touchcache"))
            clock[0] += db._TOUCH_INTERVAL_S - 1
            self.assertTrue(db.touch_guest("guest_touchcache"))
            self.assertEqual(coll.calls, 1)                 # cached inside the interval
            clock[0] += 2
            self.assertTrue(db.touch_guest("guest_touchcache"))
            self.assertEqual(coll.calls, 2)                 # refreshed after it

    def test_interval_is_five_minutes(self):
        self.assertEqual(db._TOUCH_INTERVAL_S, 300.0)


class BreakerCounterTests(unittest.TestCase):
    def setUp(self):
        llm_chain.reset_state()

    def test_open_then_close_is_counted_once_each(self):
        st = llm_chain._state
        for _ in range(llm_chain.CIRCUIT_THRESHOLD):
            st.on_failure(1, "m")
        for _ in range(llm_chain.CIRCUIT_THRESHOLD):      # further failures while open do not re-count
            st.on_failure(1, "m")
        self.assertEqual(st.breaker_opened, 1)
        self.assertEqual(st.breaker_closed, 0)
        self.assertTrue(st.is_open(1, "m"))
        st.on_success(1, "m")
        self.assertEqual(st.breaker_closed, 1)
        self.assertFalse(st.is_open(1, "m"))
        st.on_success(1, "m")                              # a healthy success does not count as a close
        self.assertEqual(st.breaker_closed, 1)

    def test_single_failure_does_not_open(self):
        st = llm_chain._state
        st.on_failure(2, "m")
        self.assertEqual(st.breaker_opened, 0)


class LoadtestHelperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("loadtest_mod", ROOT / "scripts" / "loadtest.py")
        cls.lt = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.lt)

    def test_server_timing_total_is_parsed(self):
        r = mock.Mock(headers={"server-timing": "db;dur=12.0, total;dur=345.6"})
        self.assertAlmostEqual(self.lt._srv_ms(r), 345.6)
        self.assertIsNone(self.lt._srv_ms(mock.Mock(headers={})))

    def test_upload_blobs_are_distinct_pdfs(self):
        blobs = self.lt._upload_pdf_blobs(3)
        self.assertEqual(len(blobs), 3)
        self.assertEqual(len({b[:2000] for b in blobs}), 3)   # unique metadata per upload
        self.assertTrue(all(b.startswith(b"%PDF") for b in blobs))


if __name__ == "__main__":
    unittest.main()
