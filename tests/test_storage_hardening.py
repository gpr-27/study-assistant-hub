"""Review-fix tests: atomic hydration, single claim key, continue-vs-add merge, per-key locks,
/tmp LRU cap and /tmp index revalidation. Mongo + embeddings are in-memory fakes."""
import os
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from backend import documents, tmpcache
from backend import gemini_embeddings as ge
from backend import vector_store as vs
from backend.services import subjects as svc
from tests.test_vector_store import _PDF, DIM, _Env, _embed_text, fake_iter_embed


def _chunks_for(paths):
    return [{"text": f"content of {p.name} part {i} lorem ipsum dolor sit amet consectetur",
             "source": p.name, "page": i} for p in paths for i in range(2)]


class AtomicHydrationTests(_Env):
    def test_parallel_readers_never_see_a_partial_subject(self):
        names = [f"Lec-{i}.pdf" for i in range(6)]
        calls = {"n": 0}

        def slow_fetch(owner, subject):
            calls["n"] += 1

            def gen():
                for n in names:
                    time.sleep(0.02)              # a slow GridFS read, file by file
                    yield n, _PDF
            return gen()

        results, errors = [], []

        def reader():
            try:
                results.append(documents.list_pdf_names("Algo", "u1"))
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        with mock.patch("backend.db.fetch_uploaded_pdfs", slow_fetch):
            threads = [threading.Thread(target=reader) for _ in range(10)]
            [t.start() for t in threads]
            [t.join() for t in threads]
        self.assertEqual(errors, [])
        self.assertEqual(len(results), 10)
        for r in results:
            self.assertEqual(r, sorted(names))     # complete, every time
        self.assertEqual(calls["n"], 1)            # single-flight
        leftovers = [p.name for p in (self.root / "documents").iterdir() if p.name.startswith(".hydrate")]
        self.assertEqual(leftovers, [])

    def test_failed_hydration_leaves_no_directory(self):
        def boom(owner, subject):
            def gen():
                yield "a.pdf", _PDF
                raise RuntimeError("mongo dropped")
            return gen()
        with mock.patch("backend.db.fetch_uploaded_pdfs", boom):
            with self.assertRaises(RuntimeError):
                documents._hydrate_user_subject("u1", "Algo", self.root / "documents" / "user__u1__Algo")
        d = self.root / "documents"
        self.assertEqual([p.name for p in d.iterdir()] if d.exists() else [], [])

    def test_save_pdf_is_atomic_and_listing_ignores_partials(self):
        documents.save_pdf("Algo", "a.pdf", _PDF, "u1")
        d = self.root / "documents" / "user__u1__Algo"
        (d / ".b.pdf.deadbeef.part").write_bytes(b"%PDF-partial")
        self.assertEqual(documents.list_pdf_names("Algo", "u1"), ["a.pdf"])


class ContinueVsAddTests(_Env):
    def setUp(self):
        super().setUp()
        p = mock.patch.object(vs, "chunks_from_paths", _chunks_for)
        p.start()
        self.addCleanup(p.stop)

    def test_add_during_continue_never_loses_the_added_pdfs_chunks(self):
        documents.save_pdf("Algo", "A.pdf", _PDF, "u1")
        vs.build_subject_index("Algo", "u1", budget_s=0.0)            # chunks, but no vectors yet
        key = documents.subject_internal_key("Algo", "u1")
        self.assertFalse(vs._get_index(key, "Algo", "u1").manifest["complete"])

        def embed_then_add(texts, **kw):
            # while continue-indexing is mid-embed, another request adds B.pdf and rebuilds
            documents.save_pdf("Algo", "B.pdf", _PDF, "u1")
            vs.build_subject_index("Algo", "u1", budget_s=0.0)
            yield from fake_iter_embed(texts, **kw)

        with mock.patch.object(ge, "iter_embed_documents", embed_then_add):
            status = vs.continue_subject_index("Algo", "u1", budget_s=10)

        idx = vs._get_index(key, "Algo", "u1")
        sources = {c["source"] for c in idx.chunks}
        self.assertEqual(sources, {"A.pdf", "B.pdf"})                  # B's chunks survived
        self.assertEqual(status["n_chunks"], 4)
        a_rows = [i for i, c in enumerate(idx.chunks) if c["source"] == "A.pdf"]
        self.assertTrue(all(idx.vectors[i].any() for i in a_rows))     # A's embeddings merged in
        # MongoDB holds the merged index too (what a cold instance would load)
        vs.clear_cache()
        import shutil
        shutil.rmtree(self.root / "cache", ignore_errors=True)
        cold = vs._get_index(key, "Algo", "u1")
        self.assertEqual({c["source"] for c in cold.chunks}, {"A.pdf", "B.pdf"})

    def test_continue_after_delete_writes_nothing(self):
        documents.save_pdf("Algo", "A.pdf", _PDF, "u1")
        vs.build_subject_index("Algo", "u1", budget_s=0.0)
        key = documents.subject_internal_key("Algo", "u1")

        def embed_then_delete(texts, **kw):
            import shutil
            shutil.rmtree(vs._user_cache_dir(key), ignore_errors=True)
            vs.clear_cache()
            self.mongo.blobs.clear()
            yield from fake_iter_embed(texts, **kw)

        before = dict(self.mongo.blobs)
        with mock.patch.object(ge, "iter_embed_documents", embed_then_delete):
            vs.continue_subject_index("Algo", "u1", budget_s=10)
        self.assertEqual(self.mongo.blobs, {})
        self.assertFalse((vs._user_cache_dir(key) / "manifest.json").exists())
        self.assertTrue(before)

    def test_cold_build_of_one_subject_does_not_block_another(self):
        documents.save_pdf("Slow", "A.pdf", _PDF, "u1")
        documents.save_pdf("Fast", "A.pdf", _PDF, "u1")
        vs.build_subject_index("Fast", "u1", budget_s=0.0)
        vs.clear_cache()
        started, release = threading.Event(), threading.Event()
        real = vs._load_index

        def slow_load(key, subject, owner):
            if subject == "Slow":
                started.set()
                release.wait(5)
            return real(key, subject, owner)

        out = {}
        with mock.patch.object(vs, "_load_index", slow_load):
            t = threading.Thread(target=lambda: out.setdefault("slow", vs.index_status("Slow", "u1")))
            t.start()
            self.assertTrue(started.wait(2))
            t0 = time.monotonic()
            vs.index_status("Fast", "u1")                       # must not wait for Slow's build
            self.assertLess(time.monotonic() - t0, 1.0)
            release.set()
            t.join(5)
        self.assertIn("slow", out)


class SingleClaimKeyTests(unittest.TestCase):
    def test_add_continue_and_delete_exclude_each_other(self):
        key = documents._user_subject_key("u9", "Algo")
        self.assertTrue(svc._claim(key))
        self.addCleanup(svc._release, key)
        with mock.patch.object(vs, "index_status", return_value={"n_chunks": 4, "n_embedded": 1, "complete": False}), \
             mock.patch.object(vs, "continue_subject_index") as cont:
            self.assertTrue(svc.continue_index("Algo", "u9")["busy"])
        cont.assert_not_called()
        with self.assertRaises(svc.SubjectError) as cm:
            svc.delete_subject("Algo", "u9")
        self.assertEqual(cm.exception.status_code, 409)

    def test_continue_holds_the_same_key_add_uses(self):
        key = documents._user_subject_key("u9", "Algo")
        seen = {}

        def during(*a, **k):
            seen["claimed_by_add_key"] = not svc._claim(key)
            return {"n_chunks": 4, "n_embedded": 4, "complete": True}

        with mock.patch.object(vs, "index_status", return_value={"n_chunks": 4, "n_embedded": 1, "complete": False}), \
             mock.patch.object(vs, "continue_subject_index", during):
            svc.continue_index("Algo", "u9")
        self.assertTrue(seen["claimed_by_add_key"])


class TmpCacheTests(unittest.TestCase):
    def test_lru_eviction_respects_cap_recency_and_protection(self):
        import tempfile
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            for name in ("old", "mid", "new", "busy"):
                d = root / name
                d.mkdir()
                (d / "f.bin").write_bytes(b"x" * 1000)
                tmpcache.touch(d)
            (root / ".hydrate-tmp").mkdir()
            now = time.monotonic()
            with tmpcache._lock:
                tmpcache._last_used[str(root / "old")] = now - 1000
                tmpcache._last_used[str(root / "busy")] = now - 2000     # oldest, but protected
                tmpcache._last_used[str(root / "mid")] = now - 500
                tmpcache._last_used[str(root / "new")] = now - 1          # inside the grace window
            evicted = tmpcache.enforce([root], protect=lambda p: p.name == "busy", limit_bytes=2500)
            names = sorted(p.name for p in evicted)
            self.assertEqual(names, ["mid", "old"])
            self.assertTrue((root / "busy").exists() and (root / "new").exists())
            self.assertFalse((root / "old").exists())
            self.assertTrue((root / ".hydrate-tmp").exists())

    def test_nothing_evicted_under_the_cap(self):
        import tempfile
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            (root / "a").mkdir()
            (root / "a" / "f").write_bytes(b"x" * 10)
            self.assertEqual(tmpcache.enforce([root], limit_bytes=10_000), [])


class TmpIndexRevalidationTests(_Env):
    def setUp(self):
        super().setUp()
        p = mock.patch.object(vs, "chunks_from_paths", _chunks_for)
        p.start()
        self.addCleanup(p.stop)
        vs._tmp_checked.clear()
        documents.save_pdf("Algo", "A.pdf", _PDF, "u1")
        vs.build_subject_index("Algo", "u1", budget_s=0.0)
        self.key = documents.subject_internal_key("Algo", "u1")

    def test_stale_tmp_copy_is_replaced_after_ram_eviction(self):
        owner, _ver, blob = self.mongo.blobs[self.key]
        self.mongo.blobs[self.key] = (owner, "reindexed-elsewhere", blob)
        vs.clear_cache()
        vs._tmp_checked.clear()                                   # cold instance / old check
        with mock.patch.object(vs, "_load_user_from_mongo", wraps=vs._load_user_from_mongo) as m:
            vs.index_status("Algo", "u1")
        m.assert_called_once()

    def test_recently_validated_tmp_copy_is_trusted(self):
        owner, _ver, blob = self.mongo.blobs[self.key]
        self.mongo.blobs[self.key] = (owner, "reindexed-elsewhere", blob)
        vs.clear_cache()
        vs._tmp_checked[self.key] = time.monotonic()
        with mock.patch.object(vs, "_load_user_from_mongo") as m:
            vs.index_status("Algo", "u1")
        m.assert_not_called()

    def test_mongo_down_keeps_tmp_copy(self):
        vs.clear_cache()
        vs._tmp_checked.clear()
        self.mongo.down = True
        with mock.patch.object(vs, "_load_user_from_mongo") as m:
            self.assertEqual(vs.index_status("Algo", "u1")["n_chunks"], 2)
        m.assert_not_called()


if __name__ == "__main__":
    unittest.main()
