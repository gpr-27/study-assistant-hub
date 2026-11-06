"""Retrieval store: chunking, BM25, hybrid search, on-disk format, build/resume,
bundled-index loading, MongoDB round-trip and per-owner isolation.

Embeddings are replaced by a deterministic hashed bag-of-words embedder and Mongo by an
in-memory fake, so nothing here touches the network, a key, or a database.
"""
import hashlib
import re
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from backend import documents
from backend import gemini_embeddings as ge
from backend import vector_store as vs

DIM = 64
_PDF = b"%PDF-1.4\nfake\n"


def _embed_text(t: str) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    for w in re.findall(r"[a-z0-9]+", t.lower()):
        v[int(hashlib.md5(w.encode()).hexdigest(), 16) % DIM] += 1.0
    n = np.linalg.norm(v)
    return v / n if n else v


def fake_embed_query(text, *, model=None, dim=None, deadline_s=None):
    return _embed_text(text)


def fake_iter_embed(texts, *, model=None, dim=None, budget_s=30.0, patient=True, titles=None, max_items=None):
    for start in range(0, len(texts), 10):
        if max_items is not None and start >= max_items:
            return
        yield start, np.vstack([_embed_text(t) for t in texts[start:start + 10]])


CHUNKS = [
    {"text": "A deadlock occurs when processes wait on each other for resources held by the others.", "source": "os.pdf", "page": 0},
    {"text": "Paging divides physical memory into fixed size frames and logical memory into pages.", "source": "os.pdf", "page": 1},
    {"text": "A matrix is an ordered rectangular array of numbers arranged in rows and columns.", "source": "math.pdf", "page": 0},
    {"text": "Bayes theorem relates conditional probabilities of mutually exclusive events.", "source": "math.pdf", "page": 1},
]


def _index(with_vectors=True, complete=True) -> vs.VectorIndex:
    vecs = np.vstack([_embed_text(c["text"]) for c in CHUNKS]) if with_vectors else None
    manifest = {"model": "gemini-embedding-2", "dim": DIM, "complete": complete}
    return vs.VectorIndex("k", [dict(c) for c in CHUNKS], vecs, manifest)


class _FakeMongo:
    """In-memory stand-in for the index functions of backend.db (owner-filtered like the real ones)."""

    def __init__(self):
        self.blobs: dict[str, tuple[str, str, bytes]] = {}
        self.down = False

    def store_index_blob(self, key, owner, subject, version, data, meta=None):
        if not owner:
            return False
        self.blobs[key] = (owner, version, data)
        return True

    def load_index_blob(self, key, owner):
        hit = self.blobs.get(key)
        return hit[2] if hit and owner and hit[0] == owner else None

    def get_index_version(self, key, owner):
        hit = self.blobs.get(key)
        return hit[1] if hit and owner and hit[0] == owner else None

    def index_version_state(self, key, owner):
        if self.down:
            return False, None
        return True, self.get_index_version(key, owner)

    def list_user_subjects(self, owner):
        return []

    def fetch_uploaded_pdfs(self, owner, subject):
        return []


class ChunkingTests(unittest.TestCase):
    def test_split_respects_size_and_overlaps(self):
        text = " ".join(f"word{i}" for i in range(1000))
        parts = vs.split_text(text, 500, 50)
        self.assertGreater(len(parts), 5)
        self.assertTrue(all(len(p) <= 500 for p in parts))
        self.assertTrue(set(parts[0].split()[-3:]) & set(parts[1].split()[:12]))   # overlap

    def test_short_text_is_one_chunk_and_empty_is_none(self):
        self.assertEqual(vs.split_text("hello world"), ["hello world"])
        self.assertEqual(vs.split_text("   "), [])

    def test_no_separator_text_is_hard_split(self):
        parts = vs.split_text("x" * 3200, 1500, 100)
        self.assertTrue(all(len(p) <= 1500 for p in parts))

    def test_repeated_footer_lines_are_dropped_and_pages_kept_separate(self):
        names = "alpha bravo charlie delta echo foxtrot golf hotel".split()
        pages = [f"Topic {names[i]} " + f"real content about {names[i]} number words " * 12 + f"\nFooter Company 2018 – {i}" for i in range(8)]
        chunks = vs.chunk_pages([("a.pdf", pages)])
        self.assertEqual(len(chunks), 8)
        self.assertTrue(all("Footer Company" not in c["text"] for c in chunks))
        self.assertEqual([c["page"] for c in chunks], list(range(8)))

    def test_tiny_pages_merge(self):
        chunks = vs.chunk_pages([("a.pdf", ["Intro text that is quite short indeed.", "Another short bit of words here ok."])])
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["page"], 0)


class SearchTests(unittest.TestCase):
    def test_bm25_only_when_no_vectors(self):
        docs = _index(with_vectors=False).search("what is a deadlock", 2)
        self.assertEqual(docs[0].page_content, CHUNKS[0]["text"])
        self.assertEqual(docs[0].metadata["retrieval"], "lexical")
        self.assertEqual(docs[0].metadata["source"], "os.pdf")

    def test_hybrid_when_query_embedding_available(self):
        with mock.patch.object(ge, "embed_query", fake_embed_query):
            docs = _index().search("matrix rows and columns", 2)
        self.assertEqual(docs[0].page_content, CHUNKS[2]["text"])
        self.assertEqual(docs[0].metadata["retrieval"], "hybrid")

    def test_embedding_failure_falls_back_to_bm25_not_error(self):
        with mock.patch.object(ge, "embed_query", side_effect=ge.GeminiQuotaError("quota")):
            docs = _index().search("bayes theorem probabilities", 1)
        self.assertEqual(docs[0].page_content, CHUNKS[3]["text"])
        self.assertEqual(docs[0].metadata["retrieval"], "lexical")

    def test_unembedded_rows_are_never_dense_hits(self):
        idx = _index(complete=False)
        idx.vectors[0] = 0                       # chunk 0 not embedded yet
        idx._has = np.asarray(idx.vectors.any(axis=1))
        idx.n_embedded = int(idx._has.sum())
        with mock.patch.object(ge, "embed_query", fake_embed_query):
            docs = idx.search("deadlock processes wait", 3)
        self.assertEqual(docs[0].page_content, CHUNKS[0]["text"])   # still found via BM25 over all rows

    def test_empty_query_and_empty_index(self):
        self.assertEqual(_index().search("  ", 3), [])
        self.assertEqual(vs.VectorIndex("k", [], None, {}).search("x", 3), [])

    def test_retriever_shape_matches_langchain(self):
        r = _index(with_vectors=False).as_retriever(search_kwargs={"k": 2})
        a, b = r.invoke("paging memory frames"), r.get_relevant_documents("paging memory frames")
        self.assertGreaterEqual(len(a), 1)
        self.assertEqual([d.page_content for d in a], [d.page_content for d in b])
        self.assertTrue(hasattr(a[0], "page_content") and isinstance(a[0].metadata, dict))


class DiskFormatTests(unittest.TestCase):
    def test_save_load_roundtrip_is_mmap_float32(self):
        with tempfile.TemporaryDirectory() as d:
            idx = _index()
            vs.save_index_dir(Path(d), idx.chunks, idx.vectors, {"model": "m", "dim": DIM, "complete": True})
            loaded = vs.load_index_dir(Path(d), "k", mmap=True)
        self.assertEqual(len(loaded), 4)
        self.assertEqual(loaded.vectors.dtype, np.float32)
        self.assertEqual(loaded.n_embedded, 4)

    def test_pack_unpack_float16_roundtrip_keeps_ranking(self):
        with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as e:
            idx = _index()
            vs.save_index_dir(Path(d), idx.chunks, idx.vectors, {"model": "m", "dim": DIM, "complete": True})
            blob = vs.pack_index(Path(d))
            vs.unpack_index(blob, Path(e))
            loaded = vs.load_index_dir(Path(e), "k", mmap=False)
        q = _embed_text("matrix rows columns")
        self.assertEqual(int(np.argmax(loaded.vectors @ q)), 2)

    def test_corrupt_blob_rejected(self):
        import io, zipfile
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr("manifest.json", "{}")
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(vs.VectorStoreError):
                vs.unpack_index(buf.getvalue(), Path(d))


class BundledIndexTests(unittest.TestCase):
    def test_shipped_indexes_load_and_answer(self):
        for subject, query, needle in (("Mathematics", "Bayes theorem", "Bayes"),
                                       ("Operating System", "system call parameter passing", "Parameter")):
            d = vs.BUNDLED_INDEX_DIR / subject
            self.assertTrue((d / "manifest.json").is_file(), f"missing shipped index for {subject}")
            idx = vs.load_index_dir(d, subject)
            m = idx.manifest
            self.assertEqual((m["model"], m["dim"], m["complete"]), ("gemini-embedding-2", 768, True))
            self.assertEqual(idx.vectors.shape, (len(idx), 768))
            with mock.patch.object(ge, "embed_query", side_effect=ge.GeminiEmbeddingError("offline")):
                docs = idx.search(query, 3)
            self.assertTrue(any(needle in d.page_content for d in docs), subject)


class _Env(unittest.TestCase):
    """Temp documents dir, temp /tmp-cache, fake Mongo, fake embeddings."""

    def setUp(self):
        vs.clear_cache()
        self._t = tempfile.TemporaryDirectory()
        self.root = Path(self._t.name)
        self.mongo = _FakeMongo()
        patches = [
            mock.patch.object(documents, "DOCUMENTS_DIR", self.root / "documents"),
            mock.patch.object(vs, "_cache_root", lambda: self.root / "cache"),
            mock.patch.object(vs, "BUNDLED_INDEX_DIR", self.root / "bundled"),
            mock.patch.object(vs, "chunks_from_paths", lambda paths: [dict(c) for c in CHUNKS]),
            mock.patch.object(ge, "iter_embed_documents", fake_iter_embed),
            mock.patch.object(ge, "embed_query", fake_embed_query),
            mock.patch.object(ge, "have_keys", lambda: True),
            mock.patch.object(vs, "embedding_dim", lambda: DIM),
            mock.patch("backend.db.store_index_blob", self.mongo.store_index_blob),
            mock.patch("backend.db.load_index_blob", self.mongo.load_index_blob),
            mock.patch("backend.db.get_index_version", self.mongo.get_index_version),
            mock.patch("backend.db.index_version_state", self.mongo.index_version_state),
            mock.patch("backend.db.list_user_subjects", self.mongo.list_user_subjects),
            mock.patch("backend.db.fetch_uploaded_pdfs", self.mongo.fetch_uploaded_pdfs),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(self._t.cleanup)
        self.addCleanup(vs.clear_cache)
        documents.forget_remote_subjects()

    def _upload(self, subject, owner):
        documents.save_pdf(subject, "Notes.pdf", _PDF, owner)


class BuildAndIsolationTests(_Env):
    def test_user_subject_build_persist_and_reload_on_cold_instance(self):
        self._upload("Notes", "u1")
        n = vs.build_subject_index("Notes", "u1")
        self.assertEqual(n, 4)
        self.assertIn("user__u1__Notes", self.mongo.blobs)
        # Simulate a fresh serverless instance: empty RAM cache, empty /tmp cache, no local PDFs.
        vs.clear_cache()
        import shutil
        shutil.rmtree(self.root / "cache", ignore_errors=True)
        shutil.rmtree(self.root / "documents", ignore_errors=True)
        r = vs.get_subject_retriever("Notes", 2, "u1")
        docs = r.invoke("deadlock processes resources")
        self.assertEqual(docs[0].page_content, CHUNKS[0]["text"])

    def test_cross_owner_isolation(self):
        self._upload("Notes", "u1")
        vs.build_subject_index("Notes", "u1")
        vs.clear_cache()
        # Another user asking for the same subject name gets nothing of u1's.
        with self.assertRaises(documents.DocumentError):
            vs.get_subject_retriever("Notes", 4, "u2")
        # Crafting the internal key does not help either.
        with self.assertRaises((documents.DocumentError, vs.VectorStoreError)):
            vs.get_subject_retriever("user__u1__Notes", 4, "u2")
        with self.assertRaises(vs.VectorStoreError):
            vs._get_index("user__u1__Notes", "Notes", "u2")
        with self.assertRaises(vs.VectorStoreError):
            vs._get_index("user__u1__Notes", "Notes", None)
        # The Mongo layer itself refuses a mismatched owner.
        self.assertIsNone(self.mongo.load_index_blob("user__u1__Notes", "u2"))
        # And an owner-less internal key is derived per owner.
        self.assertNotEqual(documents.subject_internal_key("Notes", "u1"), documents.subject_internal_key("Notes", "u2"))

    def test_same_subject_name_two_owners_stay_separate(self):
        self._upload("Notes", "u1")
        vs.build_subject_index("Notes", "u1")
        other = [{"text": "Completely different content about volcanoes and magma chambers.", "source": "v.pdf", "page": 0}]
        self._upload("Notes", "u2")
        with mock.patch.object(vs, "chunks_from_paths", lambda paths: [dict(c) for c in other]):
            vs.build_subject_index("Notes", "u2")
        d1 = vs.get_subject_retriever("Notes", 1, "u1").invoke("deadlock")
        d2 = vs.get_subject_retriever("Notes", 1, "u2").invoke("volcanoes magma")
        self.assertIn("deadlock", d1[0].page_content)
        self.assertIn("volcanoes", d2[0].page_content)
        self.assertEqual(vs.get_subject_retriever("Notes", 4, "u2").invoke("deadlock processes")[0].metadata["source"], "v.pdf")

    def test_bundled_subject_visible_to_everyone_and_never_embeds_at_request_time(self):
        d = self.root / "documents" / "Physics"
        d.mkdir(parents=True)
        (d / "Lec-1.pdf").write_bytes(_PDF)
        with mock.patch.object(ge, "iter_embed_documents", side_effect=AssertionError("must not embed")):
            r = vs.get_subject_retriever("Physics", 2, "anyone")   # no shipped index -> lexical from PDFs
            docs = r.invoke("deadlock processes")
        self.assertEqual(docs[0].page_content, CHUNKS[0]["text"])
        self.assertEqual(docs[0].metadata["retrieval"], "lexical")

    def test_budgeted_build_is_partial_then_continue_completes_it(self):
        self._upload("Big", "u1")
        many = [{"text": f"unique topic{i} content words about subject {i} " * 3, "source": "b.pdf", "page": i} for i in range(25)]
        calls = {"n": 0}

        def limited(texts, **kw):          # first call embeds one batch of 10, then 'budget runs out'
            calls["n"] += 1
            yield from fake_iter_embed(texts, max_items=10)

        with mock.patch.object(vs, "chunks_from_paths", lambda paths: [dict(c) for c in many]), \
             mock.patch.object(ge, "iter_embed_documents", limited):
            vs.build_subject_index("Big", "u1")
            st = vs.index_status("Big", "u1")
            self.assertEqual((st["n_chunks"], st["n_embedded"], st["complete"]), (25, 10, False))
            # Partially embedded index is still fully searchable (BM25 covers the rest).
            self.assertIn("topic20", vs.get_subject_retriever("Big", 1, "u1").invoke("topic20")[0].page_content)
        with mock.patch.object(ge, "iter_embed_documents", fake_iter_embed):
            st = vs.continue_subject_index("Big", "u1", budget_s=10)
        self.assertEqual((st["n_embedded"], st["complete"]), (25, True))

    def test_adding_a_pdf_reuses_vectors_of_unchanged_chunks(self):
        self._upload("Notes", "u1")
        vs.build_subject_index("Notes", "u1")
        extra = CHUNKS + [{"text": "Brand new chunk about semaphores and mutual exclusion primitives.", "source": "n.pdf", "page": 0}]
        seen = []

        def spy(texts, **kw):
            seen.extend(texts)
            yield from fake_iter_embed(texts)

        with mock.patch.object(vs, "chunks_from_paths", lambda paths: [dict(c) for c in extra]), \
             mock.patch.object(ge, "iter_embed_documents", spy):
            vs.build_subject_index("Notes", "u1")
        self.assertEqual(len(seen), 1)
        self.assertIn("semaphores", seen[0])


    def _expire(self):
        vs._cache["user__u1__Notes"].checked = time.monotonic() - vs._REVALIDATE_S - 1

    def test_deleted_subject_is_dropped_from_warm_cache_and_tmp_on_revalidation(self):
        self._upload("Notes", "u1")
        vs.build_subject_index("Notes", "u1")
        key = "user__u1__Notes"
        self.assertTrue(vs._user_cache_dir(key).exists())
        shutil.rmtree(self.root / "documents")          # subject deleted: PDFs gone too
        self.mongo.blobs.clear()                        # ...and its stored index
        self._expire()
        with self.assertRaises(documents.DocumentError):
            vs._get_index(key, "Notes", "u1")
        self.assertNotIn(key, vs._cache)
        self.assertFalse(vs._user_cache_dir(key).exists())

    def test_unreachable_mongo_never_evicts_a_warm_index(self):
        self._upload("Notes", "u1")
        vs.build_subject_index("Notes", "u1")
        self.mongo.down = True
        self._expire()
        self.assertIsNotNone(vs._get_index("user__u1__Notes", "Notes", "u1"))
        self.assertIn("user__u1__Notes", vs._cache)

    def test_remote_reindex_reloads(self):
        self._upload("Notes", "u1")
        vs.build_subject_index("Notes", "u1")
        owner, _, blob = self.mongo.blobs["user__u1__Notes"]
        self.mongo.blobs["user__u1__Notes"] = (owner, "other-version", blob)
        self._expire()
        idx = vs._get_index("user__u1__Notes", "Notes", "u1")
        self.assertEqual(vs._cache["user__u1__Notes"].version, idx.manifest["version"])

    def test_no_text_raises_vector_store_error(self):
        self._upload("Empty", "u1")
        with mock.patch.object(vs, "chunks_from_paths", lambda paths: []):
            with self.assertRaises(vs.VectorStoreError):
                vs.build_subject_index("Empty", "u1")

    def test_lru_cache_is_bounded(self):
        for i in range(vs._CACHE_MAX + 3):
            vs._cache_put(f"k{i}", _index())
        self.assertLessEqual(len(vs._cache), vs._CACHE_MAX)

    def test_build_index_from_paths_in_memory_with_as_retriever(self):
        p = self.root / "x.pdf"
        p.write_bytes(_PDF)
        store = vs.build_index_from_paths([p])
        docs = store.as_retriever(search_kwargs={"k": 1}).invoke("matrix rows columns")
        self.assertIn("matrix", docs[0].page_content)


if __name__ == "__main__":
    unittest.main()
