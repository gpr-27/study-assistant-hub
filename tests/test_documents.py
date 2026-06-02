"""Tests for the document write helpers backing the Add-a-Subject feature.

These are pure filesystem/string tests — no embedding model, no network, no
Groq key. DOCUMENTS_DIR is patched to a temporary directory per test.
"""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from backend import documents
from backend.documents import (
    DocumentError,
    _safe_component,
    _sanitize_pdf_filename,
    _validate_subject_name,
    read_pdf,
    save_pdf,
)

_PDF_BYTES = b"%PDF-1.4\n%minimal fake pdf body\n"


class ValidateSubjectNameTests(unittest.TestCase):
    def test_returns_trimmed_name_for_valid_input(self):
        self.assertEqual(_validate_subject_name("  Machine Learning  "), "Machine Learning")

    def test_rejects_empty_or_whitespace(self):
        for bad in ("", "   "):
            with self.assertRaises(DocumentError):
                _validate_subject_name(bad)

    def test_rejects_path_separators_and_traversal(self):
        for bad in ("a/b", "a\\b", "..", "."):
            with self.assertRaises(DocumentError):
                _validate_subject_name(bad)

    def test_rejects_hidden_name(self):
        with self.assertRaises(DocumentError):
            _validate_subject_name(".secret")


class SafeComponentTests(unittest.TestCase):
    def test_allows_normal_subject_and_pdf_names(self):
        self.assertEqual(_safe_component("Operating System", kind="subject"), "Operating System")
        self.assertEqual(_safe_component("Lec-1 (2).pdf", kind="PDF name"), "Lec-1 (2).pdf")

    def test_rejects_traversal_and_separators(self):
        for bad in ("..", ".", "../../etc", "a/b", "a\\b", "/etc/passwd", "x\x00y", ""):
            with self.assertRaises(DocumentError):
                _safe_component(bad, kind="subject")


class ReadPathTraversalTests(unittest.TestCase):
    """The read path must not let request-supplied names escape documents/."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        patcher = mock.patch.object(documents, "DOCUMENTS_DIR", Path(self._tmp.name))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self._tmp.cleanup)
        # A secret PDF sitting OUTSIDE the documents dir.
        self._secret = Path(self._tmp.name).parent / "secret.pdf"
        self._secret.write_bytes(_PDF_BYTES)
        self.addCleanup(lambda: self._secret.unlink(missing_ok=True))

    def test_traversal_subject_is_rejected(self):
        with self.assertRaises(DocumentError):
            read_pdf("../..", "secret.pdf")

    def test_absolute_filename_is_rejected(self):
        with self.assertRaises(DocumentError):
            read_pdf("Algorithms", str(self._secret))


class SanitizePdfFilenameTests(unittest.TestCase):
    def test_appends_pdf_extension_when_missing(self):
        self.assertEqual(_sanitize_pdf_filename("report"), "report.pdf")

    def test_strips_directory_components(self):
        self.assertEqual(_sanitize_pdf_filename("/etc/passwd/notes.pdf"), "notes.pdf")

    def test_replaces_unsafe_characters(self):
        self.assertEqual(_sanitize_pdf_filename("weird*name?.pdf"), "weird_name_.pdf")

    def test_falls_back_for_empty_name(self):
        self.assertEqual(_sanitize_pdf_filename(""), "document.pdf")


class SavePdfTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        patcher = mock.patch.object(documents, "DOCUMENTS_DIR", Path(self._tmp.name))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self._tmp.cleanup)

    def test_saves_valid_pdf_and_returns_name(self):
        stored = save_pdf("Algorithms", "Lec-1.pdf", _PDF_BYTES)
        self.assertEqual(stored, "Lec-1.pdf")
        written = Path(self._tmp.name) / "Algorithms" / "Lec-1.pdf"
        self.assertTrue(written.is_file())
        self.assertEqual(written.read_bytes(), _PDF_BYTES)

    def test_dedupes_colliding_filenames(self):
        first = save_pdf("Algorithms", "Lec-1.pdf", _PDF_BYTES)
        second = save_pdf("Algorithms", "Lec-1.pdf", _PDF_BYTES)
        self.assertEqual(first, "Lec-1.pdf")
        self.assertEqual(second, "Lec-1 (2).pdf")

    def test_rejects_non_pdf_bytes(self):
        with self.assertRaises(DocumentError):
            save_pdf("Algorithms", "fake.pdf", b"not a pdf at all")

    def test_rejects_empty_file(self):
        with self.assertRaises(DocumentError):
            save_pdf("Algorithms", "empty.pdf", b"")


if __name__ == "__main__":
    unittest.main()
