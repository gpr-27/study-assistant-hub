"""Embeddings / retrieval hardening: scanned-PDF detection, page ranges, dimension and
finiteness validation, query single-flight, ranking modes, health payload and the doctor script.

No network: the Gemini REST call is replaced by a counting fake.
"""
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from backend import gemini_embeddings as ge
from backend import vector_store as vs

DIM = 16


def _unit(seed: int) -> np.ndarray:
    v = np.random.default_rng(seed).standard_normal(DIM).astype(np.float32)
    return v / np.linalg.norm(v)


CHUNKS = [
    {"text": "deadlock occurs when processes wait for resources held by others", "source": "os.pdf", "page": 0},
    {"text": "paging divides physical memory into fixed size frames", "source": "os.pdf", "page": 1},
    {"text": "bayes theorem relates conditional probabilities of exclusive events", "source": "m.pdf", "page": 0},
]


class PageRangeAndScanTests(unittest.TestCase):
    def test_merged_tiny_pages_record_the_page_range(self):
        chunks = vs.chunk_pages([("a.pdf", ["Intro text that is quite short indeed.",
                                             "Another short bit of words here ok."])])
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["page"], 0)
        self.assertEqual(chunks[0]["page_end"], 1)

    def test_single_page_chunk_has_no_page_end(self):
        pages = ["word " * 400, "other " * 400]
        chunks = vs.chunk_pages([("a.pdf", pages)])
        self.assertTrue(all("page_end" not in c for c in chunks))

    def test_document_metadata_carries_page_end(self):
        idx = vs.VectorIndex("k", [{"text": "alpha beta gamma delta", "source": "a.pdf", "page": 0,
                                    "page_end": 2}], None, {})
        doc = idx._doc(0, 0.5, "lexical")
        self.assertEqual(doc.metadata["page_end"], 2)

    def test_low_text_pdf_is_flagged_as_scanned(self):
        report = [{"name": "scan.pdf", "pages": 10, "chars_per_page": 2.0},
                  {"name": "text.pdf", "pages": 10, "chars_per_page": 900.0}]
        low = vs.low_text_sources(report)
        self.assertEqual([r["name"] for r in low], ["scan.pdf"])
        warnings = vs.scan_warnings(low)
        self.assertEqual(len(warnings), 1)
        self.assertIn("This PDF looks scanned; OCR it via Scan Notes.", warnings[0])
        self.assertIn("scan.pdf", warnings[0])

    def test_no_warning_for_text_pdfs(self):
        self.assertEqual(vs.scan_warnings(vs.low_text_sources([{"name": "a.pdf", "pages": 3,
                                                               "chars_per_page": 800}])), [])

    def test_chunks_from_paths_reports_per_source_text_density(self):
        fake = {"good.pdf": ["real sentence about operating systems " * 20] * 3, "scan.pdf": ["", "  ", ""]}
        paths = [Path("good.pdf"), Path("scan.pdf")]
        with mock.patch("backend.documents.extract_pdf_pages", side_effect=lambda p: fake[p.name]):
            chunks = vs.chunks_from_paths(paths)
        report = {r["name"]: r for r in vs.last_scan_report()}
        self.assertGreater(report["good.pdf"]["chars_per_page"], vs.LOW_TEXT_CHARS_PER_PAGE)
        self.assertLess(report["scan.pdf"]["chars_per_page"], vs.LOW_TEXT_CHARS_PER_PAGE)
        self.assertTrue(chunks)

    def test_scanned_only_build_fails_with_user_facing_note(self):
        fake = {"scan.pdf": ["", "", ""]}
        with mock.patch("backend.documents.extract_pdf_pages", side_effect=lambda p: fake[p.name]):
            with self.assertRaises(vs.VectorStoreError) as cm:
                vs.build_index_to_dir("k", [Path("scan.pdf")], Path(tempfile_dir()) / "idx", budget_s=0.0)
        self.assertIn("This PDF looks scanned; OCR it via Scan Notes.", str(cm.exception))


def tempfile_dir() -> str:
    import tempfile

    return tempfile.mkdtemp(prefix="emb-test-")


class IndexValidationTests(unittest.TestCase):
    def _manifest(self, **kw):
        m = {"model": "gemini-embedding-2", "dim": DIM, "complete": True, "format": vs.INDEX_VERSION}
        m.update(kw)
        return m

    def test_dimension_mismatch_disables_dense_but_keeps_bm25(self):
        vecs = np.vstack([_unit(i) for i in range(3)])
        vecs = np.hstack([vecs, np.zeros((3, 8), np.float32)])          # 24-d rows, manifest says 16
        idx = vs.VectorIndex("k", [dict(c) for c in CHUNKS], vecs, self._manifest())
        self.assertIsNone(idx.vectors)
        self.assertIn("dim", idx.problem)
        docs = idx.search("deadlock resources", 2, use_embeddings=False)
        self.assertIn("deadlock", docs[0].page_content)
        self.assertEqual(vs.index_status_of(idx)["complete"], False)

    def test_non_finite_vectors_are_rejected(self):
        vecs = np.vstack([_unit(i) for i in range(3)])
        vecs[1, 2] = np.nan
        idx = vs.VectorIndex("k", [dict(c) for c in CHUNKS], vecs, self._manifest())
        self.assertIsNone(idx.vectors)
        self.assertIn("non-finite", idx.problem)

    def test_newer_index_format_is_refused_not_misread(self):
        vecs = np.vstack([_unit(i) for i in range(3)])
        idx = vs.VectorIndex("k", [dict(c) for c in CHUNKS], vecs, self._manifest(format=vs.INDEX_VERSION + 1))
        self.assertIsNone(idx.vectors)
        self.assertIn("newer", idx.problem)

    def test_valid_index_has_no_problem_and_is_contiguous(self):
        vecs = np.asfortranarray(np.vstack([_unit(i) for i in range(3)]))
        idx = vs.VectorIndex("k", [dict(c) for c in CHUNKS], vecs, self._manifest())
        self.assertIsNone(idx.problem)
        self.assertTrue(idx.vectors.flags["C_CONTIGUOUS"])
        self.assertEqual(idx.vectors.dtype, np.float32)

    def test_status_warns_when_model_changed_since_build(self):
        vecs = np.vstack([_unit(i) for i in range(3)])
        idx = vs.VectorIndex("k", [dict(c) for c in CHUNKS], vecs, self._manifest(model="gemini-embedding-001"))
        st = vs.index_status_of(idx)
        self.assertTrue(st["stale_model"])


class RankingTests(unittest.TestCase):
    def setUp(self):
        self.vecs = np.vstack([_unit(i) for i in range(3)])
        self.idx = vs.VectorIndex("k", [dict(c) for c in CHUNKS], self.vecs, {"model": "m", "dim": DIM, "complete": True})

    def test_top_matches_full_sort(self):
        rng = np.random.default_rng(0)
        scores = rng.standard_normal(1000).astype(np.float32)
        for n in (1, 5, 10, 999, 1000, 2000):
            got = list(vs.VectorIndex._top(scores, n))
            want = list(np.argsort(-scores, kind="stable")[:n])
            self.assertEqual(got, want)

    def test_modes_dense_bm25_hybrid(self):
        q = self.vecs[0]
        dense = self.idx.rank("zzz", q, 2, "dense")
        self.assertEqual(dense[0][0], 0)
        self.assertEqual(dense[0][2], "dense")
        lex = self.idx.rank("paging frames memory", None, 2, "hybrid")
        self.assertEqual(lex[0][0], 1)
        self.assertEqual(lex[0][2], "lexical")
        hyb = self.idx.rank("paging frames memory", q, 2, "hybrid")
        self.assertEqual(hyb[0][2], "hybrid")
        self.assertIn(1, [i for i, _, _ in hyb])

    def test_bm25_only_mode_ignores_vectors(self):
        out = self.idx.rank("bayes theorem", self.vecs[0], 1, "bm25")
        self.assertEqual(out[0][0], 2)

    def test_bm25_weight_is_the_tuned_default(self):
        self.assertEqual(vs._W_BM25, 0.15)


class QuerySingleFlightTests(unittest.TestCase):
    def setUp(self):
        ge.reset_state()

    def test_identical_concurrent_queries_share_one_api_call(self):
        calls = {"n": 0}

        def slow_post(model, endpoint, payload, **kw):
            calls["n"] += 1
            time.sleep(0.2)
            return {"embedding": {"values": [1.0] + [0.0] * (DIM - 1)}}

        out = []
        with mock.patch.object(ge, "_post", side_effect=slow_post):
            ts = [threading.Thread(target=lambda: out.append(ge.embed_query("same   question", dim=DIM)))
                  for _ in range(8)]
            for t in ts:
                t.start()
            for t in ts:
                t.join()
        self.assertEqual(calls["n"], 1)
        self.assertEqual(len(out), 8)
        self.assertFalse(out[0].flags["WRITEABLE"])            # cached vectors are read-only

    def test_whitespace_variants_hit_the_same_cache_entry(self):
        with mock.patch.object(ge, "_post", return_value={"embedding": {"values": [0.0, 1.0] + [0.0] * (DIM - 2)}}) as p:
            ge.embed_query("a  b", dim=DIM)
            ge.embed_query("a b", dim=DIM)
        self.assertEqual(p.call_count, 1)

    def test_failure_is_shared_and_reported_by_code(self):
        with mock.patch.object(ge, "_post", side_effect=ge.GeminiQuotaError("quota")):
            with self.assertRaises(ge.GeminiQuotaError):
                ge.embed_query("q1", dim=DIM)
        st = ge.status()
        self.assertEqual(st["last_error"], "quota_exhausted")
        self.assertTrue(st["breaker_open"])
        self.assertNotIn("key", str(st).lower().replace("keys_configured", ""))

    def test_status_contains_no_secrets(self):
        with mock.patch.dict("os.environ", {"GEMINI_API_KEY": "AIzaSECRETVALUE123", "GEMINI_API_KEY_2": "AIzaOTHER456"}):
            text = str(ge.status())
        self.assertNotIn("AIzaSECRET", text)
        self.assertNotIn("AIzaOTHER", text)

    def test_nonfinite_api_values_are_refused(self):
        bad = [float("nan")] + [0.0] * (DIM - 1)
        with mock.patch.object(ge, "_post", return_value={"embedding": {"values": bad}}):
            with self.assertRaises(ge.GeminiEmbeddingError):
                ge.embed_query("nan please", dim=DIM)


class RetrievalHealthTests(unittest.TestCase):
    def test_bundled_indexes_report_ok_without_network(self):
        with mock.patch.object(ge, "_post", side_effect=AssertionError("no network in health")):
            h = vs.retrieval_health()
        self.assertTrue(h["ok"], h)
        self.assertEqual(h["dim"], 768)
        for s in ("Mathematics", "Operating System"):
            self.assertTrue(h["bundled"][s]["ok"])
            self.assertEqual(h["bundled"][s]["dim"], 768)
        self.assertNotIn("GEMINI", str(h))

    def test_missing_bundled_index_marks_retrieval_degraded(self):
        with mock.patch.object(vs, "_bundled_dir", lambda k: Path(tempfile_dir()) / k):
            h = vs.retrieval_health()
        self.assertFalse(h["ok"])


class DoctorScriptTests(unittest.TestCase):
    def test_offline_doctor_passes_on_shipped_indexes(self):
        import importlib.util

        path = Path(__file__).resolve().parent.parent / "scripts" / "verify_embeddings.py"
        spec = importlib.util.spec_from_file_location("verify_embeddings", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        with mock.patch.object(mod.ge, "load_keys", return_value=[]):
            rc = mod.main(["--offline"])
        fails = [r for r in mod.ROWS if r[0] == "FAIL"]
        self.assertEqual(rc, 0, fails)
        self.assertTrue(any("BM25-only fallback answers" in r[1] for r in mod.ROWS))


if __name__ == "__main__":
    unittest.main()
