"""Peak-memory proof for the streaming chunk assembly (synthetic ~120 MB PDF, small chunks).

The "before" figure re-creates the old behaviour (all chunks joined into one bytes object, pypdf over
BytesIO, bytes written to disk); the "after" figure runs the real chunked path (temp file + sha256 on the
fly + pypdf over a file object + rename into the subject folder + resumable page reading).
Run with ``-s`` to see the numbers."""
import io
import os
import tempfile
import time
import tracemalloc
import unittest
from pathlib import Path
from unittest import mock

from pypdf import PdfWriter

from backend import documents, vector_store as vs
from backend.services import subjects as svc
from backend.services import uploads as up

MB = 1048576
SIZE_MB = int(os.environ.get("SAH_MEM_TEST_MB", "120"))


def _big_pdf(path: Path, mb: int) -> int:
    w = PdfWriter()
    for _ in range(4):
        w.add_blank_page(200, 200)
    w.add_attachment("pad.bin", os.urandom(mb * MB))        # incompressible ballast makes the file ~mb MB
    with open(path, "wb") as fh:
        w.write(fh)
    return path.stat().st_size


def _peak(fn) -> float:
    tracemalloc.start()
    try:
        fn()
        return tracemalloc.get_traced_memory()[1] / MB
    finally:
        tracemalloc.stop()


class StreamingAssemblyMemoryTests(unittest.TestCase):
    def test_peak_memory_of_a_big_upload_is_bounded(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            src = t / "big.pdf"
            size = _big_pdf(src, SIZE_MB)
            self.assertGreater(size, SIZE_MB * MB * 0.9)
            docs = t / "documents"
            chunk = 4 * MB

            def before():
                with open(src, "rb") as fh:                          # old: chunks joined into ONE bytes object
                    data = b"".join(iter(lambda: fh.read(chunk), b""))
                documents._pdf_reader(io.BytesIO(data)).pages[0]
                (t / "old-copy.pdf").write_bytes(data)

            def after():
                with mock.patch.object(documents, "DOCUMENTS_DIR", docs), \
                     mock.patch.object(up, "CHUNK_SIZE", chunk), \
                     mock.patch.object(up, "get_store", return_value=up.TempStore(t / "store")), \
                     mock.patch.object(svc.vector_store, "build_subject_index", return_value=1), \
                     mock.patch.object(svc.vector_store, "index_status", return_value={"complete": True}):
                    documents.forget_remote_subjects()
                    info = up.create_upload("u1", "Huge", "big.pdf", size)
                    with open(src, "rb") as fh:
                        for i in range(info["total_chunks"]):
                            up.put_chunk("u1", info["upload_id"], i, fh.read(chunk))
                    out = up.complete_upload("u1", info["upload_id"])
                    self.assertEqual(out["saved_pdfs"], ["big.pdf"])
                    stored = next(docs.glob("user__u1__Huge/big.pdf"))
                    self.assertEqual(stored.stat().st_size, size)
                    ent = {"size": size, "total": 0, "pages": [], "complete": False, "raw": True}
                    vs._extract_range(stored, ent, time.monotonic() + 30)       # resumable reader, from a file object
                    self.assertTrue(ent["complete"] and ent["total"] == 4)
                    self.assertEqual(list(docs.glob(".assemble-*")), [])           # temp dir removed in `finally`

            warm = PdfWriter(); warm.add_blank_page(10, 10); b = io.BytesIO(); warm.write(b)
            documents._pdf_reader(io.BytesIO(b.getvalue())).pages[0]                # import pypdf outside the window
            peak_before = _peak(before)
            peak_after = _peak(after)
            print(f"\n[upload memory] {size / MB:.0f} MB file: before (bytes in RAM) peak {peak_before:.0f} MB, "
                  f"after (streamed) peak {peak_after:.0f} MB")
            self.assertLess(peak_after, 60)                        # chunk + pypdf working set, far under 300 MB RSS
            self.assertLess(peak_after, peak_before / 2)


class AssemblyGuardTests(unittest.TestCase):
    def _session(self, store, size=1000):
        return {"size": size, "total_chunks": 1, "filename": "a.pdf", "subject": "S", "sha256": None}

    def test_not_enough_tmp_space_is_a_clean_507(self):
        with tempfile.TemporaryDirectory() as t, mock.patch.object(documents, "DOCUMENTS_DIR", Path(t)):
            usage = mock.Mock(free=10 * MB)
            with mock.patch("backend.services.uploads.shutil.disk_usage", return_value=usage):
                with self.assertRaises(svc.SubjectError) as cm:
                    up._assemble_and_add(mock.Mock(), "u1", "x" * 24, self._session(None, 500 * MB), 0.0)
            self.assertEqual(cm.exception.status_code, 507)
            self.assertIn("temporary space", str(cm.exception))

    def test_third_concurrent_assembly_gets_503_busy_and_stays_resumable(self):
        with tempfile.TemporaryDirectory() as t, mock.patch.object(up, "TempStore", up.TempStore):
            store = up.TempStore(Path(t))
            with mock.patch.object(up, "get_store", return_value=store), \
                 mock.patch.object(svc, "resolve_subject_name", return_value="S"):
                info = up.create_upload("u1", "S", "a.pdf", 5)
                up.put_chunk("u1", info["upload_id"], 0, b"%PDF-")
                held = [up._ASSEMBLY.acquire(blocking=False) for _ in range(2)]
                try:
                    self.assertTrue(all(held))
                    with self.assertRaises(svc.SubjectError) as cm:
                        up.complete_upload("u1", info["upload_id"])
                    self.assertEqual(cm.exception.status_code, 503)
                    self.assertEqual(store.get("u1", info["upload_id"])["state"], up.OPEN)   # retryable
                finally:
                    for ok in held:
                        if ok:
                            up._ASSEMBLY.release()


if __name__ == "__main__":
    unittest.main()
