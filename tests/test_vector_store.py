"""Tests for the per-subject index fingerprint (staleness detection).

`_fingerprint` only stats files and reads settings, so these tests need no
embedding model, network, or Groq key. DOCUMENTS_DIR is patched to a temp dir.
"""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from backend import documents, vector_store

_PDF_BYTES = b"%PDF-1.4\nfake\n"


class FingerprintTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.docs = Path(self._tmp.name)
        self.subject_dir = self.docs / "Networks"
        self.subject_dir.mkdir(parents=True)
        (self.subject_dir / "Lec-1.pdf").write_bytes(_PDF_BYTES)
        (self.subject_dir / "Lec-2.pdf").write_bytes(_PDF_BYTES)

        patcher = mock.patch.object(documents, "DOCUMENTS_DIR", self.docs)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self._tmp.cleanup)

    def _fp(self):
        return vector_store._fingerprint("Networks")

    def test_lists_current_pdfs_with_model_and_chunk_settings(self):
        fp = self._fp()
        self.assertEqual([f["name"] for f in fp["files"]], ["Lec-1.pdf", "Lec-2.pdf"])
        self.assertEqual(fp["model"], vector_store.EMBEDDING_MODEL)
        self.assertIn("chunk_size", fp)
        self.assertIn("chunk_overlap", fp)

    def test_changes_when_a_pdf_is_added(self):
        before = self._fp()
        (self.subject_dir / "Lec-3.pdf").write_bytes(_PDF_BYTES)
        self.assertNotEqual(before, self._fp())

    def test_changes_when_a_pdf_is_removed(self):
        before = self._fp()
        (self.subject_dir / "Lec-2.pdf").unlink()
        self.assertNotEqual(before, self._fp())

    def test_changes_when_a_pdf_content_size_changes(self):
        before = self._fp()
        (self.subject_dir / "Lec-1.pdf").write_bytes(_PDF_BYTES + b"more pages\n")
        self.assertNotEqual(before, self._fp())

    def test_changes_when_pdf_content_edited_but_size_unchanged(self):
        # A same-length edit (byte-for-byte different, identical size) must still
        # invalidate the index — only a content hash catches this, size alone can't.
        before = self._fp()
        (self.subject_dir / "Lec-1.pdf").write_bytes(b"%PDF-1.4\nFAKE\n")  # same len as _PDF_BYTES
        self.assertEqual(len(_PDF_BYTES), len(b"%PDF-1.4\nFAKE\n"))  # guard: sizes match
        self.assertNotEqual(before, self._fp())

    def test_unchanged_when_only_mtime_changes(self):
        # Regression guard for the free-tier OOM: a redeploy / container rebuild /
        # `cp` / MongoDB re-hydration changes file timestamps while leaving the PDF
        # bytes identical. The fingerprint must treat that as FRESH (unchanged) so
        # the pre-baked index is loaded, not re-embedded (which OOMs on 512 MB).
        import os
        before = self._fp()
        for pdf in self.subject_dir.glob("*.pdf"):
            st = pdf.stat()
            os.utime(pdf, (st.st_atime, st.st_mtime + 10_000))
        self.assertEqual(before, self._fp())

    def test_changes_when_embedding_model_changes(self):
        before = self._fp()
        with mock.patch.object(vector_store, "EMBEDDING_MODEL", "some/other-model"):
            self.assertNotEqual(before, self._fp())


if __name__ == "__main__":
    unittest.main()
