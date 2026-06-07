"""Tests for MongoDB-backed persistence of uploaded PDFs.

On a host without a persistent disk (e.g. Render's free tier) the documents/ dir is
ephemeral, so PDFs added via "Add a Subject" are also stored in MongoDB and re-hydrated
on startup. Persistence is OFF in the test environment (see tests/__init__.py), so:

  * the db helpers are exercised both directly (graceful no-op when persistence is off)
    and with ``backend.db`` mocked (the restore orchestration), and
  * the on-disk restore helper is a pure filesystem test with DOCUMENTS_DIR patched to a
    temp dir — no real database, no network, no embedding model.
"""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from backend import db, documents
from backend.documents import restore_uploaded_pdf
from backend.services import subjects as subjects_service

_PDF_BYTES = b"%PDF-1.4\n%minimal fake pdf body\n"


class StoreUploadedPdfNoPersistenceTests(unittest.TestCase):
    """With persistence off, the store/iter helpers degrade to safe no-ops."""

    def test_store_returns_false_when_persistence_disabled(self):
        # MONGODB_URI is "" in tests, so _gridfs() -> None and the store is a no-op.
        self.assertFalse(db.store_uploaded_pdf("Algorithms", "Lec-1.pdf", _PDF_BYTES))

    def test_iter_yields_nothing_when_persistence_disabled(self):
        self.assertEqual(list(db.iter_uploaded_pdfs()), [])


class RestoreUploadedPdfDiskTests(unittest.TestCase):
    """The on-disk restore helper (re-hydration target) — pure filesystem behaviour."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.docs = Path(self._tmp.name)
        self._patch = mock.patch.object(documents, "DOCUMENTS_DIR", self.docs)
        self._patch.start()

    def tearDown(self):
        self._patch.stop()
        self._tmp.cleanup()

    def test_writes_pdf_into_subject_dir(self):
        wrote = restore_uploaded_pdf("Algorithms", "Lec-1.pdf", _PDF_BYTES)
        self.assertTrue(wrote)
        target = self.docs / "Algorithms" / "Lec-1.pdf"
        self.assertTrue(target.is_file())
        self.assertEqual(target.read_bytes(), _PDF_BYTES)

    def test_is_idempotent_does_not_clobber_existing(self):
        (self.docs / "Algorithms").mkdir(parents=True)
        seeded = self.docs / "Algorithms" / "Lec-1.pdf"
        seeded.write_bytes(b"%PDF-1.4 original bundled\n")
        # A second "restore" of the same path must NOT overwrite the existing (bundled) file.
        wrote = restore_uploaded_pdf("Algorithms", "Lec-1.pdf", _PDF_BYTES)
        self.assertFalse(wrote)
        self.assertEqual(seeded.read_bytes(), b"%PDF-1.4 original bundled\n")

    def test_rejects_unsafe_names_without_writing(self):
        for subject, filename in (("../escape", "x.pdf"), ("ok", "../../etc/passwd")):
            self.assertFalse(restore_uploaded_pdf(subject, filename, _PDF_BYTES))
        # Nothing should have escaped the documents dir.
        self.assertEqual(list(self.docs.rglob("*.pdf")), [])


class RestoreOrchestrationTests(unittest.TestCase):
    """restore_uploaded_pdfs() pulls every stored upload from the DB and writes it to disk."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.docs = Path(self._tmp.name)
        self._patch = mock.patch.object(documents, "DOCUMENTS_DIR", self.docs)
        self._patch.start()

    def tearDown(self):
        self._patch.stop()
        self._tmp.cleanup()

    def test_restores_all_stored_uploads(self):
        stored = [
            ("Algorithms", "Lec-1.pdf", _PDF_BYTES),
            ("Algorithms", "Lec-2.pdf", _PDF_BYTES),
            ("Networks", "Intro.pdf", _PDF_BYTES),
        ]
        with mock.patch.object(db, "iter_uploaded_pdfs", return_value=iter(stored)):
            count = subjects_service.restore_uploaded_pdfs()
        self.assertEqual(count, 3)
        self.assertTrue((self.docs / "Algorithms" / "Lec-1.pdf").is_file())
        self.assertTrue((self.docs / "Algorithms" / "Lec-2.pdf").is_file())
        self.assertTrue((self.docs / "Networks" / "Intro.pdf").is_file())

    def test_no_op_when_nothing_stored(self):
        with mock.patch.object(db, "iter_uploaded_pdfs", return_value=iter([])):
            self.assertEqual(subjects_service.restore_uploaded_pdfs(), 0)


class AddSubjectStoresToDbTests(unittest.TestCase):
    """Adding a subject mirrors each saved PDF into MongoDB (best-effort)."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.docs = Path(self._tmp.name)
        # save_pdf writes via documents.DOCUMENTS_DIR; patch it to a temp dir.
        self._patch = mock.patch.object(documents, "DOCUMENTS_DIR", self.docs)
        self._patch.start()

    def tearDown(self):
        self._patch.stop()
        self._tmp.cleanup()

    def test_add_subject_calls_store_with_stored_name(self):
        # Stub the embedding build (no model/network) and capture the DB writes.
        with mock.patch.object(subjects_service.vector_store, "build_subject_index", return_value=7), \
             mock.patch.object(db, "store_uploaded_pdf", return_value=True) as store:
            result = subjects_service.add_subject("Algorithms", [("Lec-1.pdf", _PDF_BYTES)])
        self.assertEqual(result.chunk_count, 7)
        self.assertEqual(result.saved_pdfs, ["Lec-1.pdf"])
        store.assert_called_once_with("Algorithms", "Lec-1.pdf", _PDF_BYTES)


if __name__ == "__main__":
    unittest.main()
