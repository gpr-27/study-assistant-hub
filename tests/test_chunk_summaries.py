"""Tests for per-chunk PDF summaries (build-once, reuse-everywhere).

No real Groq calls or network: the build step is mocked, and persistence is OFF in the
test env so the MongoDB helpers exercise their graceful no-op path. Covers the pure
helpers, the cache resolution order (memory/disk/build), the tools' use of summaries
instead of the raw PDF, and the at-upload background build hook.
"""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from backend import chunk_summaries as cs
from backend import config, db


class PureHelperTests(unittest.TestCase):
    def test_split_text_produces_chunks(self):
        chunks = cs._split_text("Mitosis is a type of cell division. " * 300)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(c.strip() for c in chunks))

    def test_split_text_empty(self):
        self.assertEqual(cs._split_text("   "), [])

    def test_batch_respects_char_budget(self):
        big = "x" * (cs._BATCH_CHAR_BUDGET // 2 + 10)
        batches = cs._batch([big, big, big])
        # Each chunk is ~half the budget, so no two fit together -> three batches.
        self.assertEqual(len(batches), 3)

    def test_batch_packs_small_chunks_together(self):
        self.assertEqual(cs._batch(["a", "b", "c"]), [["a", "b", "c"]])

    def test_fingerprint_is_content_and_model_sensitive(self):
        self.assertEqual(cs._fingerprint("same", "m"), cs._fingerprint("same", "m"))
        self.assertNotEqual(cs._fingerprint("a", "m"), cs._fingerprint("b", "m"))
        self.assertNotEqual(cs._fingerprint("same", "m1"), cs._fingerprint("same", "m2"))

    def test_summaries_as_content_joins_as_bullets(self):
        out = cs.summaries_as_content(["First.", "Second."])
        self.assertIn("- First.", out)
        self.assertIn("- Second.", out)


class CacheResolutionTests(unittest.TestCase):
    """get_or_build resolves memory -> disk -> mongo -> build, and backfills."""

    def setUp(self):
        cs._cache.clear()
        self._tmp = tempfile.TemporaryDirectory()
        self.idx = Path(self._tmp.name)
        self._p1 = mock.patch.object(cs, "FAISS_INDEX_DIR", self.idx)
        self._p1.start()

    def tearDown(self):
        self._p1.stop()
        self._tmp.cleanup()
        cs._cache.clear()

    def test_builds_once_then_serves_from_memory(self):
        with mock.patch.object(cs, "read_pdf", return_value="lecture text"), \
             mock.patch.object(cs, "_build", return_value=["s1", "s2"]) as build, \
             mock.patch.object(db, "load_chunk_summaries", return_value=None), \
             mock.patch.object(db, "store_chunk_summaries", return_value=False) as store:
            first = cs.get_or_build("Bio", "Lec-1.pdf")
            second = cs.get_or_build("Bio", "Lec-1.pdf")
        self.assertEqual(first, ["s1", "s2"])
        self.assertEqual(second, ["s1", "s2"])
        build.assert_called_once()          # built once, then cached in memory
        store.assert_called_once()          # mongo backfill attempted on build

    def test_disk_cache_avoids_rebuild(self):
        # First build writes the disk cache; clearing memory then re-reading must hit disk.
        with mock.patch.object(cs, "read_pdf", return_value="lecture text"), \
             mock.patch.object(cs, "_build", return_value=["only"]), \
             mock.patch.object(db, "load_chunk_summaries", return_value=None), \
             mock.patch.object(db, "store_chunk_summaries", return_value=False):
            cs.get_or_build("Bio", "Lec-1.pdf")
        cs._cache.clear()
        with mock.patch.object(cs, "read_pdf", return_value="lecture text"), \
             mock.patch.object(cs, "_build", side_effect=AssertionError("should not rebuild")) as build, \
             mock.patch.object(db, "load_chunk_summaries", return_value=None):
            again = cs.get_or_build("Bio", "Lec-1.pdf")
        self.assertEqual(again, ["only"])
        build.assert_not_called()

    def test_mongo_cache_used_when_disk_missing(self):
        with mock.patch.object(cs, "read_pdf", return_value="lecture text"), \
             mock.patch.object(cs, "_build", side_effect=AssertionError("should not rebuild")), \
             mock.patch.object(db, "load_chunk_summaries", return_value=["from-mongo"]):
            out = cs.get_or_build("Bio", "Lec-1.pdf")
        self.assertEqual(out, ["from-mongo"])

    def test_raises_when_no_text(self):
        with mock.patch.object(cs, "read_pdf", return_value="   "):
            with self.assertRaises(cs.ChunkSummaryError):
                cs.get_or_build("Bio", "Empty.pdf")


class BuildForSubjectTests(unittest.TestCase):
    def test_best_effort_skips_failures(self):
        with mock.patch.object(cs, "list_pdf_names", return_value=["a.pdf", "b.pdf"]), \
             mock.patch.object(cs, "get_or_build", side_effect=[["ok"], cs.ChunkSummaryError("boom")]):
            done = cs.build_for_subject("Bio")
        self.assertEqual(done, 1)  # one succeeded, one failed and was skipped


class DbNoOpTests(unittest.TestCase):
    """Persistence is OFF in tests -> the chunk-summary helpers degrade to no-ops."""

    def test_store_and_load_are_noops(self):
        self.assertFalse(db.store_chunk_summaries("fp", "S", "p.pdf", ["x"]))
        self.assertIsNone(db.load_chunk_summaries("fp"))


class ToolsUseSummariesTests(unittest.TestCase):
    """quiz/flashcards build their content from chunk summaries, not the raw PDF."""

    def setUp(self):
        from fastapi.testclient import TestClient
        from backend.main import app

        self.client = TestClient(app)

    def test_quiz_generate_uses_chunk_summaries(self):
        from backend.routers import quiz as quiz_router

        with mock.patch.object(quiz_router, "content_for", return_value="- key point") as content_for, \
             mock.patch.object(quiz_router, "make_quiz", return_value={"questions": [{
                 "question": "Q?", "options": {"A": "1", "B": "2", "C": "3", "D": "4"},
                 "correct": "A", "explanation": "because",
             }]}) as make_quiz, \
             mock.patch.object(db, "save", return_value=None):
            resp = self.client.post("/api/quiz/generate", json={
                "subject": "Bio", "pdf": "Lec-1.pdf", "num_questions": 1, "difficulty": "easy",
            })
        self.assertEqual(resp.status_code, 200, resp.text)
        content_for.assert_called_once_with("Bio", "Lec-1.pdf", None)
        # make_quiz received the summary content, not raw PDF text.
        self.assertEqual(make_quiz.call_args.kwargs["content"], "- key point")

    def test_flashcards_generate_uses_chunk_summaries(self):
        from backend.routers import flashcards as fc_router

        with mock.patch.object(fc_router, "content_for", return_value="- key point") as content_for, \
             mock.patch.object(fc_router, "make_flashcards", return_value={"cards": [{
                 "front": "F", "back": "B", "topic": "T",
             }]}), \
             mock.patch.object(db, "save", return_value=None):
            resp = self.client.post("/api/flashcards/generate", json={
                "subject": "Bio", "pdf": "Lec-1.pdf", "num_cards": 1, "difficulty": "easy",
            })
        self.assertEqual(resp.status_code, 200, resp.text)
        content_for.assert_called_once_with("Bio", "Lec-1.pdf", None)


if __name__ == "__main__":
    unittest.main()
