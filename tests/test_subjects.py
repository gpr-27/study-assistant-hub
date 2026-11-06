"""Tests for the add / continue-index / delete subject flow (service + router).

Hermetic: DOCUMENTS_DIR is patched to a temp dir, MongoDB/Clerk are off, and the embedding
build is mocked. PDFs are real (blank pages generated with pypdf) so validation is exercised.
"""
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient
from pypdf import PdfWriter

from backend import db, documents, vector_store
from backend.main import app
from backend.services import subjects as svc


def make_pdf(pages: int = 1, tag: int = 0) -> bytes:
    w = PdfWriter()
    for _ in range(pages):
        w.add_blank_page(width=200 + tag, height=200)   # tag makes distinct bytes
    buf = io.BytesIO()
    w.write(buf)
    return buf.getvalue()


class _TmpDocs(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.docs = Path(self._tmp.name)
        (self.docs / "Mathematics").mkdir()
        (self.docs / "Mathematics" / "m.pdf").write_bytes(make_pdf(1, 99))
        p = mock.patch.object(documents, "DOCUMENTS_DIR", self.docs)
        p.start()
        self.addCleanup(p.stop)
        self.addCleanup(self._tmp.cleanup)
        documents.forget_remote_subjects()
        self.addCleanup(documents.forget_remote_subjects)


class NameValidationTests(unittest.TestCase):
    def test_accepts_normal_names(self):
        self.assertEqual(svc.clean_subject_name("  Operating   System "), "Operating System")
        self.assertEqual(svc.clean_subject_name("Calculus II (2026) & Stats"), "Calculus II (2026) & Stats")

    def test_rejects_unsafe_names(self):
        for bad in ("", "   ", "../etc", "a/b", "a\\b", ".hidden", "x.", "user__u1__Foo", "a__b",
                    "nul\x00l", "tab\there?", "x" * 81, "semi;colon", "<script>"):
            with self.assertRaises(svc.SubjectError, msg=repr(bad)):
                svc.clean_subject_name(bad)


class InspectPdfTests(unittest.TestCase):
    def test_valid_pdf_returns_page_count(self):
        self.assertEqual(svc.inspect_pdf("a.pdf", make_pdf(3)), 3)

    def test_rejects_non_pdf_empty_and_garbage(self):
        for data in (b"", b"hello world", b"%PDF-1.4\nthis is not really a pdf"):
            with self.assertRaises(svc.SubjectError):
                svc.inspect_pdf("x.pdf", data)

    def test_page_limit(self):
        with mock.patch.object(svc, "MAX_PAGES_PER_PDF", 2):
            with self.assertRaises(svc.SubjectError) as cm:
                svc.inspect_pdf("big.pdf", make_pdf(3))
        self.assertEqual(cm.exception.status_code, 413)


class AddSubjectTests(_TmpDocs):
    def _add(self, name, files, owner="u1", **kw):
        with mock.patch.object(vector_store, "build_subject_index", return_value=5) as build, \
             mock.patch.object(vector_store, "index_status",
                               return_value={"n_chunks": 5, "n_embedded": kw.pop("emb", 5), "complete": kw.pop("complete", True)}), \
             mock.patch.object(db, "store_uploaded_pdf", return_value=True) as store:
            res = svc.add_subject(name, files, owner)
        return res, build, store

    def test_owner_is_required_and_never_writes_flat_path(self):
        with self.assertRaises(svc.SubjectError) as cm:
            svc.add_subject("Algo", [("a.pdf", make_pdf())], None)
        self.assertEqual(cm.exception.status_code, 401)
        self.assertFalse((self.docs / "Algo").exists())

    def test_saves_under_user_namespace_and_reports_complete(self):
        res, build, store = self._add("Algo", [("Lec 1.pdf", make_pdf())])
        self.assertEqual(res.indexing, "complete")
        self.assertEqual(res.saved_pdfs, ["Lec 1.pdf"])
        self.assertTrue((self.docs / "user__u1__Algo" / "Lec 1.pdf").is_file())
        self.assertFalse((self.docs / "Algo").exists())
        store.assert_called_once()
        self.assertEqual(store.call_args.args[3], "u1")
        self.assertEqual(build.call_args.args[:2], ("Algo", "u1"))
        self.assertIn("budget_s", build.call_args.kwargs)

    def test_partial_indexing_is_reported(self):
        res, _, _ = self._add("Algo", [("a.pdf", make_pdf())], emb=2, complete=False)
        self.assertEqual(res.indexing, "partial")
        self.assertEqual(res.n_embedded, 2)

    def test_embed_budget_shrinks_with_elapsed_time(self):
        import time
        late = time.monotonic() - 49.0                       # 49 s of the 50 s deadline already used
        with mock.patch.object(vector_store, "build_subject_index", return_value=1) as build, \
             mock.patch.object(vector_store, "index_status", return_value={"complete": False}), \
             mock.patch.object(db, "store_uploaded_pdf", return_value=True):
            svc.add_subject("Algo", [("a.pdf", make_pdf())], "u1", started=late)
        self.assertEqual(build.call_args.kwargs["budget_s"], 0.0)

    def test_builtin_name_is_rejected_case_insensitively(self):
        for n in ("Mathematics", "mathematics"):
            with self.assertRaises(svc.SubjectError) as cm:
                self._add(n, [("a.pdf", make_pdf())])
            self.assertEqual(cm.exception.status_code, 409)

    def test_duplicate_files_are_skipped(self):
        pdf = make_pdf(1, 5)
        self._add("Algo", [("a.pdf", pdf)])
        res, build, _ = self._add("Algo", [("renamed.pdf", pdf)])
        self.assertEqual(res.skipped_duplicates, ["renamed.pdf"])
        self.assertEqual(res.saved_pdfs, [])
        build.assert_not_called()                            # nothing new -> no re-index

    def test_same_content_twice_in_one_upload_is_saved_once(self):
        pdf = make_pdf(1, 6)
        res, _, _ = self._add("Algo", [("a.pdf", pdf), ("b.pdf", pdf)])
        self.assertEqual(res.saved_pdfs, ["a.pdf"])
        self.assertEqual(res.skipped_duplicates, ["b.pdf"])

    def test_capitalisation_reuses_existing_subject(self):
        self._add("Algo", [("a.pdf", make_pdf(1, 1))])
        res, _, _ = self._add("algo", [("b.pdf", make_pdf(1, 2))])
        self.assertEqual(res.subject, "Algo")

    def test_failed_index_rolls_back_new_files(self):
        with mock.patch.object(vector_store, "build_subject_index",
                               side_effect=vector_store.VectorStoreError("no text")), \
             mock.patch.object(db, "store_uploaded_pdf", return_value=True), \
             mock.patch.object(svc, "_delete_remote_pdfs") as remote:
            with self.assertRaises(svc.SubjectError):
                svc.add_subject("Scan", [("a.pdf", make_pdf())], "u1")
        self.assertFalse((self.docs / "user__u1__Scan").exists())
        remote.assert_called_once_with("u1", "Scan", ["a.pdf"])

    def test_mongo_write_failure_when_enabled_aborts_and_rolls_back(self):
        with mock.patch.object(db, "store_uploaded_pdf", return_value=False), \
             mock.patch.object(db, "is_enabled", return_value=True):
            with self.assertRaises(svc.SubjectError) as cm:
                svc.add_subject("Algo", [("a.pdf", make_pdf())], "u1")
        self.assertEqual(cm.exception.status_code, 503)
        self.assertFalse((self.docs / "user__u1__Algo").exists())

    def test_users_are_isolated(self):
        self._add("Algo", [("a.pdf", make_pdf(1, 1))], owner="u1")
        self._add("Algo", [("b.pdf", make_pdf(1, 2))], owner="u2")
        self.assertEqual(documents.list_pdf_names("Algo", "u1"), ["a.pdf"])
        self.assertEqual(documents.list_pdf_names("Algo", "u2"), ["b.pdf"])
        self.assertNotIn("Algo", documents.list_subjects("u3"))

    def test_limits(self):
        f = [("a.pdf", make_pdf())]
        with mock.patch.object(svc, "MAX_FILES_PER_REQUEST", 1):
            with self.assertRaises(svc.SubjectError):
                svc.add_subject("X", f * 2, "u1")
        with mock.patch.object(svc, "MAX_FILE_BYTES", 10):
            with self.assertRaises(svc.SubjectError) as cm:
                svc.add_subject("X", f, "u1")
            self.assertEqual(cm.exception.status_code, 413)
        with mock.patch.object(svc, "MAX_PAGES_PER_REQUEST", 2):
            with self.assertRaises(svc.SubjectError):
                svc.add_subject("X", [("a.pdf", make_pdf(3))], "u1")
        with self.assertRaises(svc.SubjectError):
            svc.add_subject("X", [("a.txt", b"hello")], "u1")
        self.assertFalse((self.docs / "user__u1__X").exists())

    def test_cold_instance_hydrates_existing_pdfs_before_reindexing(self):
        """A subject that only exists in MongoDB must be hydrated BEFORE the new PDF is saved,
        otherwise the rebuilt index would contain only the new file."""
        old = make_pdf(1, 7)
        seen = {}

        def build(name, owner, budget_s=None, extract_budget_s=None):
            seen["pdfs"] = sorted(documents.list_pdf_names(name, owner))
            return 3

        with mock.patch.object(db, "fetch_uploaded_pdfs", return_value=[("old.pdf", old)]), \
             mock.patch.object(vector_store, "build_subject_index", side_effect=build), \
             mock.patch.object(vector_store, "index_status", return_value={"complete": True}), \
             mock.patch.object(db, "store_uploaded_pdf", return_value=True):
            svc.add_subject("Algo", [("new.pdf", make_pdf(1, 8))], "u1")
        self.assertEqual(seen["pdfs"], ["new.pdf", "old.pdf"])


class DeleteSubjectTests(_TmpDocs):
    def _make(self, owner, name="Algo"):
        d = self.docs / f"user__{owner}__{name}"
        d.mkdir()
        (d / "a.pdf").write_bytes(make_pdf())
        return d

    def test_deletes_only_the_callers_subject(self):
        mine, theirs = self._make("u1"), self._make("u2")
        with mock.patch.object(db, "delete_index_blob", return_value=1) as blob, \
             mock.patch.object(svc, "_delete_remote_pdfs", return_value=1) as remote:
            out = svc.delete_subject("Algo", "u1")
        self.assertEqual(out["deleted"], "Algo")
        self.assertFalse(mine.exists())
        self.assertTrue(theirs.exists())
        blob.assert_called_once_with("user__u1__Algo", "u1")
        remote.assert_called_once_with("u1", "Algo")

    def test_bundled_cannot_be_deleted(self):
        with self.assertRaises(svc.SubjectError) as cm:
            svc.delete_subject("Mathematics", "u1")
        self.assertEqual(cm.exception.status_code, 403)
        self.assertTrue((self.docs / "Mathematics").exists())

    def test_unknown_subject_is_404(self):
        with self.assertRaises(svc.SubjectError) as cm:
            svc.delete_subject("Nope", "u1")
        self.assertEqual(cm.exception.status_code, 404)

    def test_invalidates_listing_cache_and_cached_index(self):
        self._make("u1")
        documents._remote_cache["u1"] = (1e12, ["Algo"])
        vector_store._cache["user__u1__Algo"] = mock.MagicMock()
        svc.delete_subject("Algo", "u1")
        self.assertNotIn("u1", documents._remote_cache)
        self.assertNotIn("user__u1__Algo", vector_store._cache)


class ContinueIndexTests(_TmpDocs):
    def test_calls_vector_store_with_budget_and_reports_progress(self):
        statuses = [{"n_chunks": 10, "n_embedded": 4, "complete": False}]
        after = {"n_chunks": 10, "n_embedded": 10, "complete": True}
        with mock.patch.object(vector_store, "index_status", side_effect=statuses), \
             mock.patch.object(vector_store, "continue_subject_index", return_value=after) as cont:
            out = svc.continue_index("Algo", "u1", budget_s=20)
        cont.assert_called_once_with("Algo", "u1", budget_s=20)
        self.assertEqual((out["indexing"], out["progress"], out["busy"]), ("complete", 6, False))

    def test_complete_subject_does_no_work(self):
        with mock.patch.object(vector_store, "index_status", return_value={"n_chunks": 3, "n_embedded": 3, "complete": True}), \
             mock.patch.object(vector_store, "continue_subject_index") as cont:
            out = svc.continue_index("Algo", "u1")
        cont.assert_not_called()
        self.assertEqual(out["indexing"], "complete")

    def test_concurrent_call_on_same_subject_is_busy(self):
        st = {"n_chunks": 10, "n_embedded": 4, "complete": False}
        key = documents._user_subject_key("u1", "Algo")
        self.assertTrue(svc._claim(key))
        self.addCleanup(svc._release, key)
        with mock.patch.object(vector_store, "index_status", return_value=st), \
             mock.patch.object(vector_store, "continue_subject_index") as cont:
            out = svc.continue_index("Algo", "u1")
        cont.assert_not_called()
        self.assertTrue(out["busy"])

    def test_unknown_subject_is_404(self):
        with mock.patch.object(vector_store, "index_status", side_effect=documents.DocumentError("x")):
            with self.assertRaises(svc.SubjectError) as cm:
                svc.continue_index("Nope", "u1")
        self.assertEqual(cm.exception.status_code, 404)

    def test_bundled_subject_never_embeds(self):
        with mock.patch.object(vector_store, "index_status", return_value={"n_chunks": 3, "n_embedded": 3, "complete": True}), \
             mock.patch.object(vector_store, "continue_subject_index") as cont:
            out = svc.continue_index("Mathematics", "u1")
        cont.assert_not_called()
        self.assertEqual(out["indexing"], "complete")


class PdfInfoCacheTests(_TmpDocs):
    def test_page_count_is_cached_per_signature(self):
        d = self.docs / "user__u1__Algo"
        d.mkdir()
        (d / "a.pdf").write_bytes(make_pdf(4))
        svc._pages_cache.clear()
        with mock.patch.object(documents, "page_count", wraps=documents.page_count) as pc:
            self.assertEqual(svc.pdf_infos("Algo", "u1"), [("a.pdf", 4)])
            self.assertEqual(svc.pdf_infos("Algo", "u1"), [("a.pdf", 4)])
        self.assertEqual(pc.call_count, 1)


class RouterTests(_TmpDocs):
    def setUp(self):
        super().setUp()
        self.client = TestClient(app, raise_server_exceptions=False)

    def test_list_includes_bundled(self):
        body = self.client.get("/api/subjects").json()
        self.assertIn("Mathematics", body["subjects"])
        self.assertIn("Mathematics", body["bundled"])
        self.assertEqual(body["own"], [])

    def test_list_splits_own_and_bundled(self):
        with mock.patch("backend.routers.subjects.list_subjects",
                        side_effect=lambda o=None: ["Mathematics", "Algo"] if o else ["Mathematics"]):
            body = self.client.get("/api/subjects").json()
        self.assertEqual(body["subjects"], ["Mathematics", "Algo"])
        self.assertEqual(body["own"], ["Algo"])
        self.assertEqual(body["bundled"], ["Mathematics"])

    def test_internal_keys_in_path_are_404_not_a_leak(self):
        other = self.docs / "user__someoneelse__Secret"
        other.mkdir()
        (other / "x.pdf").write_bytes(make_pdf())
        for path in ("user__someoneelse__Secret", "..%2F..%2Fetc"):
            self.assertEqual(self.client.get(f"/api/subjects/{path}/pdfs").status_code, 404)
        self.assertEqual(self.client.post("/api/subjects/user__someoneelse__Secret/continue-index").status_code, 404)
        self.assertEqual(self.client.delete("/api/subjects/user__someoneelse__Secret").status_code, 404)
        self.assertTrue(other.exists())

    def test_upload_returns_indexing_field(self):
        with mock.patch.object(vector_store, "build_subject_index", return_value=9), \
             mock.patch.object(vector_store, "index_status", return_value={"n_chunks": 9, "n_embedded": 3, "complete": False}), \
             mock.patch.object(db, "store_uploaded_pdf", return_value=True):
            r = self.client.post("/api/subjects", data={"name": "Algo"},
                                 files=[("files", ("a.pdf", make_pdf(), "application/pdf"))])
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertEqual(body["indexing"], "partial")
        self.assertEqual(body["chunk_count"], 9)
        self.assertEqual(body["subject"], "Algo")
        pdfs = self.client.get("/api/subjects/Algo/pdfs").json()["pdfs"]
        self.assertEqual(pdfs, [{"name": "a.pdf", "pages": 1}])

    def test_oversize_upload_is_413_without_saving(self):
        with mock.patch.object(svc, "MAX_FILE_BYTES", 100):
            r = self.client.post("/api/subjects", data={"name": "Algo"},
                                 files=[("files", ("a.pdf", make_pdf(), "application/pdf"))])
        self.assertEqual(r.status_code, 413)
        self.assertFalse((self.docs / "user__local__Algo").exists())

    def test_non_pdf_is_rejected_with_422(self):
        r = self.client.post("/api/subjects", data={"name": "Algo"},
                             files=[("files", ("a.pdf", b"not a pdf at all", "application/pdf"))])
        self.assertEqual(r.status_code, 422)
        self.assertIn("not a valid PDF", r.json()["detail"])

    def test_bad_name_is_400(self):
        r = self.client.post("/api/subjects", data={"name": "../evil"},
                             files=[("files", ("a.pdf", make_pdf(), "application/pdf"))])
        self.assertEqual(r.status_code, 400)

    def test_continue_index_and_delete_routes(self):
        d = self.docs / "user__local__Algo"
        d.mkdir()
        (d / "a.pdf").write_bytes(make_pdf())
        with mock.patch.object(vector_store, "index_status", return_value={"n_chunks": 2, "n_embedded": 2, "complete": True}):
            r = self.client.post("/api/subjects/Algo/continue-index")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["indexing"], "complete")
        with mock.patch.object(db, "delete_index_blob", return_value=0):
            r = self.client.delete("/api/subjects/Algo")
        self.assertEqual(r.status_code, 200)
        self.assertFalse(d.exists())
        self.assertEqual(self.client.delete("/api/subjects/Mathematics").status_code, 403)


class PrebakeScriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location(
            "prebake_overviews", Path(__file__).resolve().parent.parent / "scripts" / "prebake_overviews.py")
        cls.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.mod)

    def test_parse_card_is_tolerant(self):
        card = self.mod.parse_card('noise {"overview": " **A** b ", "questions": ["q1", "", "q2", "q3", "q4"]} tail')
        self.assertEqual(card, {"overview": "**A** b", "questions": ["q1", "q2", "q3"]})
        self.assertEqual(self.mod.parse_card("garbage"), {"overview": "", "questions": []})

    def test_writes_next_to_the_bundled_index_using_the_chain(self):
        from backend import llm_chain

        fake = mock.MagicMock(text='{"overview": "O", "questions": ["a", "b", "c"]}')
        fake.label = "Model: test"
        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch.object(self.mod, "INDEX_DIR", Path(tmp)), \
             mock.patch.object(self.mod, "subject_body", return_value=("some text", 2)), \
             mock.patch("backend.documents.list_subjects", return_value=["Mathematics"]), \
             mock.patch.object(llm_chain, "chat", return_value=fake) as chat:
            self.assertEqual(self.mod.main([]), 0)
            data = json.loads((Path(tmp) / "Mathematics" / "overview.json").read_text())
            self.assertEqual(data["questions"], ["a", "b", "c"])
            self.assertEqual(data["sources"], 2)
            self.assertTrue(chat.call_args.kwargs["json_mode"])
            # second run: overview exists -> no further LLM call (quota)
            chat.reset_mock()
            self.assertEqual(self.mod.main([]), 0)
            chat.assert_not_called()


if __name__ == "__main__":
    unittest.main()
