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
    clip_to_tokens,
    estimate_tokens,
    list_subjects,
    read_pdf,
    save_pdf,
    subject_internal_key,
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


class ClipToTokensTests(unittest.TestCase):
    """clip_to_tokens is an approximation (no tiktoken) that must stay bounded and cheap."""

    def test_short_text_is_returned_unchanged(self):
        self.assertEqual(clip_to_tokens("hello world", 50), "hello world")
        self.assertEqual(clip_to_tokens("", 10), "")

    def test_long_text_is_cut_near_the_budget_on_a_word_boundary(self):
        text = " ".join(["lorem"] * 5000)
        out = clip_to_tokens(text, 100)
        self.assertLess(len(out), len(text))
        self.assertLessEqual(estimate_tokens(out), 100)
        self.assertGreater(estimate_tokens(out), 80)
        self.assertTrue(out.endswith("lorem"))

    def test_zero_budget_gives_empty(self):
        self.assertEqual(clip_to_tokens("abc def", 0), "")

    def test_estimate_tracks_real_tokenizers_roughly(self):
        # ~4 chars/token prose; symbols cost 1 each.
        self.assertEqual(estimate_tokens("a+b=c"), 5)
        self.assertEqual(estimate_tokens("internationalization"), 5)

    def test_big_input_is_fast(self):
        import time
        text = "word " * 400_000
        t = time.monotonic()
        clip_to_tokens(text, 3000)
        self.assertLess(time.monotonic() - t, 0.5)


class SubjectKeyTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.docs = Path(self._tmp.name) / "documents"
        (self.docs / "Mathematics").mkdir(parents=True)
        p = mock.patch.object(documents, "DOCUMENTS_DIR", self.docs)
        p.start()
        self.addCleanup(p.stop)
        self.addCleanup(self._tmp.cleanup)
        documents.forget_remote_subjects()

    def test_bundled_subject_key_is_its_name_for_everyone(self):
        self.assertEqual(subject_internal_key("Mathematics", "u1"), "Mathematics")
        self.assertEqual(subject_internal_key("Mathematics"), "Mathematics")

    def test_user_subject_key_is_owner_scoped_even_without_a_local_copy(self):
        # Cold serverless instance: nothing on disk, key must still be namespaced.
        self.assertEqual(subject_internal_key("Notes", "u1"), "user__u1__Notes")
        self.assertNotEqual(subject_internal_key("Notes", "u1"), subject_internal_key("Notes", "u2"))

    def test_list_subjects_is_owner_scoped_and_includes_remote(self):
        save_pdf("Mine", "a.pdf", _PDF_BYTES, "u1")
        save_pdf("Theirs", "a.pdf", _PDF_BYTES, "u2")
        with mock.patch("backend.db.list_user_subjects", return_value=["Cloud"]):
            names = list_subjects("u1")
        self.assertEqual(names, ["Cloud", "Mathematics", "Mine"])
        self.assertEqual(list_subjects(), ["Mathematics"])

    def test_missing_user_subject_is_rehydrated_from_mongo_then_read(self):
        with mock.patch("backend.db.fetch_uploaded_pdfs", return_value=[("n.pdf", _PDF_BYTES)]):
            self.assertEqual(documents.list_pdf_names("Remote", "u1"), ["n.pdf"])
        with mock.patch("backend.db.fetch_uploaded_pdfs", return_value=[]):
            with self.assertRaises(DocumentError):
                documents.list_pdf_names("Remote", "u2")

    def test_uploads_go_to_a_writable_root_when_documents_is_read_only(self):
        tmp_root = Path(self._tmp.name) / "tmpdocs"
        with mock.patch.object(documents, "_user_root", lambda: tmp_root):
            save_pdf("Notes", "a.pdf", _PDF_BYTES, "u1")
        self.assertTrue((tmp_root / "user__u1__Notes" / "a.pdf").is_file())
        self.assertFalse((self.docs / "user__u1__Notes").exists())

    def test_importing_documents_does_not_import_heavy_deps(self):
        import subprocess, sys
        code = "import sys, backend.documents; print(any(m in sys.modules for m in ('pypdf','tiktoken','numpy','requests')))"
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                             cwd=str(Path(documents.__file__).resolve().parent.parent)).stdout.strip()
        self.assertEqual(out, "False")


class InternalKeyBoundaryTests(unittest.TestCase):
    """Owner A must not read owner B's files by passing the internal key as the subject."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.docs = Path(self._tmp.name)
        p = mock.patch.object(documents, "DOCUMENTS_DIR", self.docs)
        p.start()
        self.addCleanup(p.stop)
        self.addCleanup(self._tmp.cleanup)
        documents.forget_remote_subjects()
        save_pdf("Secret", "b.pdf", _PDF_BYTES, "ownerB")
        self.key = "user__ownerB__Secret"
        self.assertTrue((self.docs / self.key / "b.pdf").is_file())

    def test_internal_key_rejected_everywhere_a_subject_is_accepted(self):
        for owner in ("ownerA", None, "ownerB"):
            for fn in (lambda o=owner: documents.list_pdf_names(self.key, o),
                       lambda o=owner: documents.list_pdf_paths(self.key, o),
                       lambda o=owner: read_pdf(self.key, "b.pdf", o),
                       lambda o=owner: documents.page_count(self.key, "b.pdf", o),
                       lambda o=owner: documents.pdf_signature(self.key, "b.pdf", o),
                       lambda o=owner: subject_internal_key(self.key, o)):
                with self.assertRaises(DocumentError):
                    fn()

    def test_double_underscore_names_are_rejected(self):
        for bad in ("a__b", "__x", "x__"):
            with self.assertRaises(DocumentError):
                documents.list_pdf_names(bad, "ownerA")

    def test_owner_a_cannot_see_b_subject_by_plain_name(self):
        with mock.patch("backend.db.fetch_uploaded_pdfs", return_value=[]):
            with self.assertRaises(DocumentError):
                documents.list_pdf_names("Secret", "ownerA")
        self.assertEqual(documents.list_pdf_names("Secret", "ownerB"), ["b.pdf"])

    def test_trusted_internal_resolver_still_works(self):
        self.assertEqual(documents._subject_dir_internal(self.key), self.docs / self.key)


if __name__ == "__main__":
    unittest.main()
