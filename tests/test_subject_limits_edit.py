"""Raised/unlimited upload limits, resumable extraction, storage guard, subject editing and
per-owner hiding of the standard subjects. Hermetic: temp documents dir, no MongoDB, mocked embeddings."""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

from backend import db, documents, prefs, vector_store as vs
from backend.auth import current_owner
from backend.main import app
from backend.services import storage, uploads as up
from backend.services import subjects as svc
from tests.test_subjects import _TmpDocs, make_pdf


class _Owner:
    def __init__(self, owner):
        self.owner = owner

    def __enter__(self):
        app.dependency_overrides[current_owner] = lambda: self.owner
        return TestClient(app, raise_server_exceptions=False)

    def __exit__(self, *a):
        app.dependency_overrides.pop(current_owner, None)


class LimitTests(_TmpDocs):
    def test_defaults_and_unlimited_total(self):
        self.assertEqual(svc.MAX_TOTAL_BYTES, 0)                       # per-upload total cap removed
        self.assertEqual(svc.MAX_FILE_BYTES, 200 * 1048576)
        self.assertEqual((svc.MAX_FILES_PER_REQUEST, svc.MAX_PDFS_PER_SUBJECT, svc.MAX_SUBJECTS_PER_OWNER),
                         (50, 100, 100))
        self.assertEqual(svc.MAX_PAGES_PER_PDF, 2000)
        self.assertEqual(up.MAX_PENDING_BYTES, 1024 * 1048576)
        files = [(f"f{i}.pdf", make_pdf(1, i)) for i in range(5)]
        with mock.patch.object(svc.vector_store, "build_subject_index", return_value=1), \
             mock.patch.object(svc.vector_store, "index_status", return_value={"complete": True}), \
             mock.patch.object(db, "store_uploaded_pdf", return_value=True):
            res = svc.add_subject("Big", files, "u1")              # total >> any old 40 MB-style cap would be too
            self.assertEqual(len(res.saved_pdfs), 5)
            with mock.patch.object(svc, "MAX_TOTAL_BYTES", 10):    # a configured cap still applies
                with self.assertRaises(svc.SubjectError) as cm:
                    svc.add_subject("Big2", files, "u1")
            self.assertEqual(cm.exception.status_code, 413)

    def test_limits_endpoint_and_subjects_payload_carry_the_same_numbers(self):
        with _Owner("u1") as c:
            lim = c.get("/api/subjects/limits").json()
            sub = c.get("/api/subjects").json()
        for k in ("max_file_mb", "max_total_mb", "max_files", "max_pdfs_per_subject", "max_subjects", "raw_keep_mb"):
            self.assertEqual(lim[k], sub["limits"][k])
        self.assertEqual(lim["max_total_mb"], 0)
        self.assertIn("usage", lim)

    def test_100mb_plus_file_goes_through_chunks(self):
        size = 120 * 1048576
        with tempfile.TemporaryDirectory() as t, mock.patch.object(up, "CHUNK_SIZE", 16 * 1048576), \
             mock.patch.object(up, "get_store", return_value=up.TempStore(Path(t))), \
             mock.patch.object(svc, "add_subject", return_value=svc.AddSubjectResult("Huge", ["h.pdf"], 3)) as add:
            info = up.create_upload("u1", "Huge", "h.pdf", size)
            self.assertEqual(info["total_chunks"], 8)
            block = b"%PDF-" + b"\0" * (16 * 1048576 - 5)
            for i in range(8):
                n = up.expected_chunk_len({"total_chunks": 8, "chunk_size": 16 * 1048576, "size": size}, i)
                up.put_chunk("u1", info["upload_id"], i, (block + block)[:n])
            out = up.complete_upload("u1", info["upload_id"])
        self.assertEqual(out["subject"], "Huge")
        self.assertEqual(len(add.call_args.args[1][0][1]), size)
        with self.assertRaises(svc.SubjectError):                   # over the per-file cap is still refused
            up.create_upload("u1", "Huge", "x.pdf", svc.MAX_FILE_BYTES + 1)


class _FakePage:
    def __init__(self, i):
        self.i = i

    def extract_text(self):
        word = "abcdefghijklmnop"[self.i] * 7                       # no digits: repeated-line (boilerplate) filter keeps it
        return f"Lecture about {word} explains the {word} concept with plenty of searchable words here. " * 6


class _FakeReader:
    def __init__(self, n):
        self.pages = [_FakePage(i) for i in range(n)]


class ResumableExtractionTests(unittest.TestCase):
    def test_pages_are_read_across_calls_and_state_persists(self):
        with tempfile.TemporaryDirectory() as t:
            pdf = Path(t) / "big.pdf"
            pdf.write_bytes(b"%PDF-" + b"0" * 2_100_000)
            idx = Path(t) / "idx"
            with mock.patch("backend.documents._pdf_reader", lambda src: _FakeReader(7)), \
                 mock.patch.object(vs, "RESUME_MIN_PAGES", 3):
                seen = []
                for _ in range(20):
                    vs._scan_local.ctx = (idx, 0.0)            # budget 0 => one page per call
                    chunks = vs.chunks_from_paths([pdf])
                    pend = vs.last_extract_pending()
                    seen.append(pend.get("big.pdf", {}).get("done", 7))
                    if not pend:
                        break
                vs._scan_local.ctx = (None, None)
            self.assertEqual(seen, [1, 2, 3, 4, 5, 6, 7])
            self.assertTrue(any("gggggggg" in c["text"] or "ggggggg" in c["text"] for c in chunks))
            self.assertTrue(vs.load_texts(idx)["files"]["big.pdf"]["complete"])

    def test_text_only_files_survive_when_the_pdf_is_gone(self):
        with tempfile.TemporaryDirectory() as t:
            idx = Path(t) / "idx"
            vs.save_texts(idx, {"files": {"gone.pdf": {"size": 9, "total": 2, "complete": True, "raw": False,
                                                       "pages": ["Alpha beta gamma delta epsilon zeta eta theta iota kappa"] * 2}}})
            vs._scan_local.ctx = (idx, None)
            try:
                chunks = vs.chunks_from_paths([])
            finally:
                vs._scan_local.ctx = (None, None)
            self.assertTrue(chunks and chunks[0]["source"] == "gone.pdf")


class StorageGuardTests(_TmpDocs):
    def test_refuses_with_507_when_cluster_is_nearly_full(self):
        with mock.patch.object(storage, "cluster_usage", return_value=int(512 * 1048576 * 0.95)):
            with self.assertRaises(svc.SubjectError) as cm:
                storage.check_capacity(1000)
            self.assertEqual(cm.exception.status_code, 507)
            with _Owner("u1") as c:
                r = c.post("/api/subjects/uploads", json={"subject": "Algo", "filename": "a.pdf", "size": 5000})
            self.assertEqual(r.status_code, 507)
            self.assertIn("almost full", r.json()["detail"])

    def test_fails_open_when_usage_is_unknown(self):
        with mock.patch.object(storage, "cluster_usage", return_value=None):
            storage.check_capacity(10**12)                          # no raise


class EditTests(_TmpDocs):
    def setUp(self):
        super().setUp()
        prefs.reset_for_tests()
        self.addCleanup(prefs.reset_for_tests)
        for p in (mock.patch.object(svc.vector_store, "build_subject_index", return_value=2),
                  mock.patch.object(svc.vector_store, "index_status",
                                    return_value={"complete": True, "n_chunks": 2, "n_embedded": 2, "reading": {}}),
                  mock.patch.object(svc.vector_store, "text_state", return_value={}),
                  mock.patch.object(svc.vector_store, "forget_text"),
                  mock.patch.object(svc.vector_store, "rename_index", return_value=True)):
            p.start()
            self.addCleanup(p.stop)

    def _add(self, name="Algo", n=2, owner="u1"):
        svc.add_subject(name, [(f"f{i}.pdf", make_pdf(1, i)) for i in range(n)], owner)

    def test_files_list_remove_and_last_file_guard(self):
        self._add()
        with _Owner("u1") as c:
            files = c.get("/api/subjects/Algo/files").json()
            self.assertEqual([f["name"] for f in files["files"]], ["f0.pdf", "f1.pdf"])
            self.assertTrue(all(f["size"] > 0 and f["pages"] == 1 for f in files["files"]))
            r = c.delete("/api/subjects/Algo/pdfs/f0.pdf")
            self.assertEqual(r.status_code, 200, r.text)
            self.assertEqual(r.json()["removed"], "f0.pdf")
            self.assertEqual([f["name"] for f in c.get("/api/subjects/Algo/files").json()["files"]], ["f1.pdf"])
            self.assertEqual(c.delete("/api/subjects/Algo/pdfs/f1.pdf").status_code, 409)
            self.assertEqual(c.delete("/api/subjects/Algo/pdfs/nope.pdf").status_code, 404)
            self.assertEqual(c.post("/api/subjects/Algo/reindex").status_code, 200)
        with _Owner("u2") as c:                                       # another owner can't see or touch it
            self.assertEqual(c.get("/api/subjects/Algo/files").status_code, 404)
            self.assertEqual(c.delete("/api/subjects/Algo/pdfs/f1.pdf").status_code, 404)

    def test_rename_moves_files_and_rejects_clashes_and_standard_names(self):
        self._add("Algo")
        self._add("Other", 1)
        with mock.patch.object(db, "rename_history_subject", return_value=3) as hist, \
             mock.patch.object(db, "rename_uploaded_subject", return_value=2):
            with _Owner("u1") as c:
                r = c.patch("/api/subjects/Algo", json={"name": "Algorithms"})
                self.assertEqual(r.status_code, 200, r.text)
                self.assertEqual(r.json()["history_relabelled"], 3)
                hist.assert_called_once_with("u1", "Algo", "Algorithms")
                names = c.get("/api/subjects").json()["own"]
                self.assertIn("Algorithms", names)
                self.assertNotIn("Algo", names)
                self.assertEqual(len(c.get("/api/subjects/Algorithms/files").json()["files"]), 2)
                self.assertEqual(c.patch("/api/subjects/Algorithms", json={"name": "other"}).status_code, 409)
                self.assertEqual(c.patch("/api/subjects/Algorithms", json={"name": "Mathematics"}).status_code, 409)
                self.assertEqual(c.patch("/api/subjects/Mathematics", json={"name": "Maths"}).status_code, 403)
                self.assertEqual(c.patch("/api/subjects/Algorithms", json={"name": "a/b"}).status_code, 400)

    def test_standard_subjects_are_read_only_for_edit_calls(self):
        with _Owner("u1") as c:
            self.assertEqual(c.get("/api/subjects/Mathematics/files").status_code, 403)
            self.assertEqual(c.delete("/api/subjects/Mathematics/pdfs/m.pdf").status_code, 403)
            self.assertEqual(c.delete("/api/subjects/Mathematics").status_code, 403)


class HideStandardTests(_TmpDocs):
    def setUp(self):
        super().setUp()
        prefs.reset_for_tests()
        self.addCleanup(prefs.reset_for_tests)
        (self.docs / "Operating System").mkdir()

    def test_hide_restore_isolation_and_listing(self):
        with _Owner("u1") as c1:
            r = c1.post("/api/subjects/mathematics/hide")             # case-insensitive, canonical in response
            self.assertEqual(r.json(), {"hidden": ["Mathematics"], "subject": "Mathematics"})
            d = c1.get("/api/subjects").json()
            self.assertNotIn("Mathematics", d["subjects"])
            self.assertIn("Mathematics", d["bundled"])                 # still a shipped subject (restorable)
            self.assertEqual(d["hidden"], ["Mathematics"])
            self.assertEqual(d["own"], [])
        with _Owner("u2") as c2:                                       # other owners are unaffected
            d2 = c2.get("/api/subjects").json()
            self.assertIn("Mathematics", d2["subjects"])
            self.assertEqual(d2["hidden"], [])
        self.assertTrue((self.docs / "Mathematics" / "m.pdf").is_file())   # nothing shared was deleted
        with _Owner("u1") as c1:
            self.assertEqual(c1.post("/api/subjects/Nope/hide").status_code, 404)
            c1.post("/api/subjects/Operating System/hide")
            self.assertEqual(c1.post("/api/subjects/Mathematics/restore").json()["hidden"], ["Operating System"])
            self.assertEqual(c1.post("/api/subjects/restore-standard").json()["hidden"], [])
            self.assertIn("Mathematics", c1.get("/api/subjects").json()["subjects"])

    def test_hidden_subject_is_rejected_by_direct_calls_only_for_that_owner(self):
        prefs.hide("u1", "Mathematics")
        with self.assertRaises(documents.DocumentError):
            documents.list_pdf_names("Mathematics", "u1")
        with self.assertRaises(documents.DocumentError):
            documents.subject_internal_key("Mathematics", "u1")
        with self.assertRaises(documents.DocumentError):
            vs.get_subject_retriever("Mathematics", 3, "u1")
        self.assertEqual(documents.list_pdf_names("Mathematics", "u2"), ["m.pdf"])
        self.assertEqual(documents.list_pdf_names("Mathematics", None), ["m.pdf"])
        with _Owner("u1") as c:                                        # HTTP: a clear 404
            r = c.get("/api/subjects/Mathematics/pdfs")
            self.assertEqual(r.status_code, 404)
            self.assertIn("Unknown subject", r.json()["detail"])

    def test_guest_hidden_list_merges_into_the_account(self):
        prefs.hide("guest_a", "Mathematics")
        prefs.hide("user_1", "Operating System")
        with mock.patch.object(db, "_get_db", return_value=None):
            svc.migrate_guest_uploads("guest_a", "user_1")
        self.assertEqual(prefs.hidden_standard("user_1"), ["Mathematics", "Operating System"])
        self.assertEqual(prefs.hidden_standard("guest_a"), [])

    def test_prefs_use_mongo_when_enabled_and_cache_reads(self):
        coll = mock.MagicMock()
        coll.find_one.return_value = {"hidden_standard": ["Mathematics"]}
        with mock.patch.object(prefs, "_coll", return_value=coll):
            self.assertEqual(prefs.hidden_standard("u9"), ["Mathematics"])
            prefs.hidden_standard("u9")
            prefs.hidden_standard("u9")
            self.assertEqual(coll.find_one.call_count, 1)              # hot path: one read, then cached
            prefs.hide("u9", "Operating System")
            coll.update_one.assert_called_once()


if __name__ == "__main__":
    unittest.main()
