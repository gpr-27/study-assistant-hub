"""Chunked (large-PDF) upload: create -> PUT chunks -> complete, on both stores.

Hermetic: DOCUMENTS_DIR is a temp dir, the temp-dir store is rooted in a temp dir, MongoDB is
replaced by a tiny in-memory fake (``FakeMongoDb``), indexing is mocked, and the PDFs are real
(pypdf) so validation is exercised. The happy path uses a PDF larger than Vercel's 4.5 MB cap.
"""
import hashlib
import io
import os
import tempfile
import threading
import time
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient
from pypdf import PdfWriter

from backend import db, documents, vector_store
from backend.auth import current_owner
from backend.main import app
from backend.services import subjects as svc
from backend.services import uploads as up


# ---------------------------------------------------------------------------
# in-memory MongoDB stand-in (only what MongoStore uses)
# ---------------------------------------------------------------------------
class _Res:
    def __init__(self, n):
        self.deleted_count = n


def _match(doc, flt):
    for k, cond in flt.items():
        if k == "$or":
            if not any(_match(doc, sub) for sub in cond):
                return False
            continue
        val = doc.get(k)
        if isinstance(cond, dict):
            for op, arg in cond.items():
                if op == "$in" and val not in arg:
                    return False
                if op == "$gt" and not (val is not None and val > arg):
                    return False
                if op == "$lt" and not (val is not None and val < arg):
                    return False
        elif val != cond:
            return False
    return True


class FakeColl:
    def __init__(self):
        self.docs = {}
        self.lock = threading.Lock()

    def insert_one(self, doc):
        with self.lock:
            if doc["_id"] in self.docs:
                raise RuntimeError("E11000 duplicate key")
            self.docs[doc["_id"]] = dict(doc)

    def find_one(self, flt):
        with self.lock:
            for d in self.docs.values():
                if _match(d, flt):
                    return dict(d)
        return None

    def find(self, flt, projection=None):
        with self.lock:
            rows = [dict(d) for d in self.docs.values() if _match(d, flt)]
        if projection:
            rows = [{k: v for k, v in r.items() if k in projection or k == "_id"} for r in rows]
        return rows

    def count_documents(self, flt):
        return len(self.find(flt))

    def replace_one(self, flt, doc, upsert=False):
        with self.lock:
            for k, d in self.docs.items():
                if _match(d, flt):
                    self.docs[k] = dict(doc)
                    return
            if upsert:
                self.docs[doc["_id"]] = dict(doc)

    def update_one(self, flt, update, upsert=False):
        with self.lock:
            for d in self.docs.values():
                if _match(d, flt):
                    d.update(update.get("$set", {}))
                    for k in update.get("$unset", {}):
                        d.pop(k, None)
                    return

    def find_one_and_update(self, flt, update, return_document=None):
        with self.lock:
            for d in self.docs.values():
                if _match(d, flt):
                    d.update(update.get("$set", {}))
                    return dict(d)
        return None

    def delete_one(self, flt):
        with self.lock:
            for k, d in list(self.docs.items()):
                if _match(d, flt):
                    del self.docs[k]
                    return _Res(1)
        return _Res(0)

    def delete_many(self, flt):
        with self.lock:
            keys = [k for k, d in self.docs.items() if _match(d, flt)]
            for k in keys:
                del self.docs[k]
        return _Res(len(keys))


class FakeMongoDb(dict):
    def __missing__(self, key):
        self[key] = FakeColl()
        return self[key]


# ---------------------------------------------------------------------------
def make_pdf(pages: int = 1, tag: int = 0, pad: int = 0) -> bytes:
    w = PdfWriter()
    for _ in range(pages):
        w.add_blank_page(width=200 + tag, height=200)
    if pad:
        w.add_attachment("pad.bin", os.urandom(pad))      # incompressible payload -> big file
    buf = io.BytesIO()
    w.write(buf)
    return buf.getvalue()


def split(data: bytes, size: int) -> list[bytes]:
    return [data[i:i + size] for i in range(0, len(data), size)] or [b""]


class _Base(unittest.TestCase):
    use_mongo = False

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.docs = Path(tmp.name) / "docs"
        self.docs.mkdir()
        (self.docs / "Mathematics").mkdir()
        (self.docs / "Mathematics" / "m.pdf").write_bytes(make_pdf(1, 99))
        patches = [mock.patch.object(documents, "DOCUMENTS_DIR", self.docs)]
        if self.use_mongo:
            self.fake = FakeMongoDb()
            patches += [mock.patch.object(db, "is_enabled", return_value=True),
                        mock.patch.object(db, "get_db", return_value=self.fake)]
        else:
            self.tmp_store = Path(tmp.name) / "sah_uploads"
            patches += [mock.patch.object(db, "is_enabled", return_value=False),
                        mock.patch.object(up.TempStore, "__init__",
                                          lambda s, root=None: setattr(s, "root", self.tmp_store))]
        patches += [
            mock.patch.object(vector_store, "build_subject_index", return_value=5),
            mock.patch.object(vector_store, "index_status",
                              return_value={"n_chunks": 5, "n_embedded": 5, "complete": True}),
            mock.patch.object(db, "store_uploaded_pdf", return_value=True),
            mock.patch.object(db, "store_uploaded_pdf_file", return_value=True),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        documents.forget_remote_subjects()
        self.addCleanup(documents.forget_remote_subjects)
        self.owner = {"v": "alice"}
        app.dependency_overrides[current_owner] = lambda: self.owner["v"]
        self.addCleanup(app.dependency_overrides.pop, current_owner, None)
        self.client = TestClient(app, raise_server_exceptions=False)

    # -- helpers
    def create(self, data: bytes, subject="Algo", filename="Big Lecture.pdf", sha=True, **kw):
        body = {"subject": subject, "filename": filename, "size": len(data)}
        if sha:
            body["sha256"] = hashlib.sha256(data).hexdigest()
        body.update(kw)
        return self.client.post("/api/subjects/uploads", json=body)

    def put(self, uid, i, chunk):
        return self.client.put(f"/api/subjects/uploads/{uid}/chunks/{i}", content=chunk,
                               headers={"content-type": "application/octet-stream"})

    def complete(self, uid):
        return self.client.post(f"/api/subjects/uploads/{uid}/complete")

    def open_small(self, data=None, chunk=1000, **kw):
        data = data or make_pdf(2, 1, pad=3500)
        with mock.patch.object(up, "CHUNK_SIZE", chunk):
            r = self.create(data, **kw)
        self.assertEqual(r.status_code, 200, r.text)
        return data, r.json()


class _Suite:
    """Behaviour shared by both stores (mixed into a _Base subclass)."""

    def test_happy_path_over_4_5_mb(self):
        data = make_pdf(3, 2, pad=5 * 1024 * 1024)
        self.assertGreater(len(data), 4.5 * 1024 * 1024)
        r = self.create(data)
        self.assertEqual(r.status_code, 200, r.text)
        info = r.json()
        self.assertLessEqual(info["chunk_size"], 3_500_000)
        self.assertEqual(info["total_chunks"], 2)
        self.assertGreaterEqual(len(info["upload_id"]), 30)
        for i, ch in enumerate(split(data, info["chunk_size"])):
            self.assertLess(len(ch), 4.5 * 1024 * 1024)
            pr = self.put(info["upload_id"], i, ch)
            self.assertEqual(pr.status_code, 200, pr.text)
            self.assertEqual(pr.json(), {"received": i + 1, "total": 2})
        done = self.complete(info["upload_id"])
        self.assertEqual(done.status_code, 200, done.text)
        body = done.json()
        self.assertEqual(body["subject"], "Algo")
        self.assertEqual(body["saved_pdfs"], ["Big Lecture.pdf"])
        self.assertEqual(body["indexing"], "complete")
        self.assertEqual(set(body), {"subject", "saved_pdfs", "chunk_count", "indexing", "n_embedded",
                                     "skipped_duplicates", "warnings", "reading"})
        stored = self.docs / "user__alice__Algo" / "Big Lecture.pdf"
        self.assertEqual(stored.read_bytes(), data)

    def test_partial_indexing_is_passed_through(self):
        data, info = self.open_small()
        for i, ch in enumerate(split(data, info["chunk_size"])):
            self.put(info["upload_id"], i, ch)
        with mock.patch.object(vector_store, "index_status",
                               return_value={"n_chunks": 9, "n_embedded": 2, "complete": False}):
            r = self.complete(info["upload_id"])
        self.assertEqual(r.json()["indexing"], "partial")
        self.assertEqual(r.json()["n_embedded"], 2)

    def test_out_of_order_and_duplicate_chunks(self):
        data, info = self.open_small()
        chunks = split(data, info["chunk_size"])
        self.assertGreater(len(chunks), 3)
        uid = info["upload_id"]
        order = list(reversed(range(len(chunks))))
        for i in order:
            self.assertEqual(self.put(uid, i, chunks[i]).status_code, 200)
        again = self.put(uid, 1, chunks[1])                     # duplicate re-PUT is idempotent
        self.assertEqual(again.status_code, 200)
        self.assertEqual(again.json()["received"], len(chunks))
        st = self.client.get(f"/api/subjects/uploads/{uid}").json()
        self.assertEqual(st["received"], list(range(len(chunks))))
        r = self.complete(uid)
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual((self.docs / "user__alice__Algo" / "Big Lecture.pdf").read_bytes(), data)

    def test_missing_chunk_is_409_and_resumable(self):
        data, info = self.open_small()
        chunks = split(data, info["chunk_size"])
        uid = info["upload_id"]
        for i, ch in enumerate(chunks):
            if i != 2:
                self.put(uid, i, ch)
        r = self.complete(uid)
        self.assertEqual(r.status_code, 409)
        self.assertIn("2", r.json()["detail"])
        self.assertEqual(self.put(uid, 2, chunks[2]).status_code, 200)     # still open: resume
        self.assertEqual(self.complete(uid).status_code, 200)

    def test_wrong_chunk_sizes_and_index_bounds(self):
        data, info = self.open_small()
        chunks = split(data, info["chunk_size"])
        uid, n = info["upload_id"], len(chunks)
        self.assertEqual(self.put(uid, 0, chunks[0][:-1]).status_code, 422)     # short
        self.assertEqual(self.put(uid, n - 1, chunks[-1] + b"x").status_code, 422)
        self.assertEqual(self.put(uid, n, chunks[0]).status_code, 422)          # index == total
        self.assertEqual(self.put(uid, -1, chunks[0]).status_code, 422)
        self.assertEqual(self.put(uid, 0, b"").status_code, 422)
        with mock.patch.object(up, "CHUNK_SIZE", 10):                           # body over the cap -> 413
            self.assertEqual(self.put(uid, 0, chunks[0]).status_code, 413)

    def test_wrong_total_size_declared(self):
        data = make_pdf(1, 3, pad=3500)
        with mock.patch.object(up, "CHUNK_SIZE", 1000):
            info = self.client.post("/api/subjects/uploads", json={
                "subject": "Algo", "filename": "a.pdf", "size": len(data) + 5}).json()
        for i, ch in enumerate(split(data, 1000)):
            self.put(info["upload_id"], i, ch)
        # the declared size implies one more byte-range than was sent -> last chunk length differs
        total = info["total_chunks"]
        self.assertEqual(self.put(info["upload_id"], total - 1, split(data, 1000)[-1]).status_code, 422)
        self.assertEqual(self.complete(info["upload_id"]).status_code, 409)

    def test_checksum_mismatch_is_422_and_frees_upload(self):
        data, info = self.open_small(sha=False)
        # re-create with a wrong checksum
        with mock.patch.object(up, "CHUNK_SIZE", 1000):
            r = self.client.post("/api/subjects/uploads", json={
                "subject": "Algo", "filename": "a.pdf", "size": len(data), "sha256": "0" * 64})
        uid = r.json()["upload_id"]
        for i, ch in enumerate(split(data, 1000)):
            self.put(uid, i, ch)
        c = self.complete(uid)
        self.assertEqual(c.status_code, 422)
        self.assertIn("checksum", c.json()["detail"])
        self.assertEqual(self.client.get(f"/api/subjects/uploads/{uid}").status_code, 404)

    def test_non_pdf_bytes_rejected_at_complete(self):
        junk = b"hello this is not a pdf " * 100
        with mock.patch.object(up, "CHUNK_SIZE", 1000):
            info = self.client.post("/api/subjects/uploads", json={
                "subject": "Algo", "filename": "a.pdf", "size": len(junk)}).json()
        for i, ch in enumerate(split(junk, 1000)):
            self.put(info["upload_id"], i, ch)
        r = self.complete(info["upload_id"])
        self.assertEqual(r.status_code, 422)
        self.assertIn("not a valid PDF", r.json()["detail"])
        self.assertFalse((self.docs / "user__alice__Algo").exists())

    def test_page_limit_enforced(self):
        data = make_pdf(3)
        with mock.patch.object(svc, "MAX_PAGES_PER_PDF", 2):
            info = self.create(data).json()
            self.put(info["upload_id"], 0, data)
            self.assertEqual(self.complete(info["upload_id"]).status_code, 413)

    def test_cross_owner_access_is_404(self):
        data, info = self.open_small()
        uid = info["upload_id"]
        chunks = split(data, info["chunk_size"])
        self.put(uid, 0, chunks[0])
        self.owner["v"] = "mallory"
        self.assertEqual(self.put(uid, 1, chunks[1]).status_code, 404)
        self.assertEqual(self.client.get(f"/api/subjects/uploads/{uid}").status_code, 404)
        self.assertEqual(self.complete(uid).status_code, 404)
        self.assertEqual(self.client.delete(f"/api/subjects/uploads/{uid}").status_code, 404)
        self.owner["v"] = "alice"                                  # alice's upload is untouched
        self.assertEqual(self.client.get(f"/api/subjects/uploads/{uid}").json()["received"], [0])

    def test_unguessable_or_malformed_ids(self):
        for bad in ("short", "a" * 70, "..%2F..%2Fetc%2Fpasswd", "x" * 25 + "!", "A" * 24):
            self.assertEqual(self.client.get(f"/api/subjects/uploads/{bad}").status_code, 404, bad)
            self.assertEqual(self.client.post(f"/api/subjects/uploads/{bad}/complete").status_code, 404, bad)
        ids = {self.open_small()[1]["upload_id"] for _ in range(4)}
        self.assertEqual(len(ids), 4)

    def test_open_upload_cap(self):
        for _ in range(up.MAX_OPEN_UPLOADS):
            self.open_small()
        r = self.create(make_pdf(1, 4))
        self.assertEqual(r.status_code, 429)
        self.owner["v"] = "bob"                                     # caps are per owner
        self.assertEqual(self.create(make_pdf(1, 4)).status_code, 200)

    def test_pending_bytes_cap(self):
        with mock.patch.object(up, "MAX_PENDING_BYTES", 10_000):
            self.assertEqual(self.create(b"%PDF-" + b"x" * 5000, sha=False).status_code, 200)
            self.assertEqual(self.create(b"%PDF-" + b"x" * 6000, sha=False).status_code, 429)

    def test_create_validation(self):
        ok = {"subject": "Algo", "filename": "a.pdf", "size": 100}
        cases = [
            ({**ok, "subject": "Mathematics"}, 409),                 # built-in subject
            ({**ok, "subject": "../etc"}, 400),
            ({**ok, "subject": "user__bob__x"}, 400),
            ({**ok, "filename": "notes.txt"}, 422),
            ({**ok, "size": 0}, 422),
            ({**ok, "size": svc.MAX_FILE_BYTES + 1}, 413),
            ({**ok, "sha256": "zz"}, 422),
        ]
        for body, status in cases:
            self.assertEqual(self.client.post("/api/subjects/uploads", json=body).status_code, status, body)

    def test_filename_is_never_trusted(self):
        data = make_pdf(1, 5)
        info = self.create(data, filename="../../etc/passwd/..\\..\\evil.pdf").json()
        self.put(info["upload_id"], 0, data)
        body = self.complete(info["upload_id"]).json()
        self.assertEqual(body["saved_pdfs"], ["evil.pdf"])
        self.assertTrue((self.docs / "user__alice__Algo" / "evil.pdf").is_file())
        self.assertFalse((self.docs.parent / "etc").exists())

    def test_abort(self):
        data, info = self.open_small()
        uid = info["upload_id"]
        self.put(uid, 0, split(data, info["chunk_size"])[0])
        self.assertEqual(self.client.delete(f"/api/subjects/uploads/{uid}").json(), {"aborted": True})
        self.assertEqual(self.put(uid, 0, split(data, info["chunk_size"])[0]).status_code, 404)
        self.assertEqual(self.complete(uid).status_code, 404)
        self.assertEqual(self.client.delete(f"/api/subjects/uploads/{uid}").status_code, 404)
        self.assertEqual(self.create(make_pdf(1, 7)).status_code, 200)   # slot freed

    def test_retry_complete_is_idempotent(self):
        data = make_pdf(2, 6)
        info = self.create(data).json()
        self.put(info["upload_id"], 0, data)
        with mock.patch.object(svc, "add_subject", wraps=svc.add_subject) as add:
            first = self.complete(info["upload_id"])
            second = self.complete(info["upload_id"])
            third = self.complete(info["upload_id"])
        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.json(), first.json())
        self.assertEqual(third.json(), first.json())
        self.assertEqual(add.call_count, 1)                              # indexed once
        self.assertEqual(self.put(info["upload_id"], 0, data).status_code, 409)   # closed for writes

    def test_concurrent_complete_runs_once(self):
        data = make_pdf(2, 8)
        info = self.create(data).json()
        self.put(info["upload_id"], 0, data)
        entered, release = threading.Event(), threading.Event()
        real = svc.add_subject

        def slow(*a, **k):
            entered.set()
            release.wait(5)
            return real(*a, **k)

        results = []
        with mock.patch.object(svc, "add_subject", side_effect=slow) as add:
            t = threading.Thread(target=lambda: results.append(up.complete_upload("alice", info["upload_id"])))
            t.start()
            self.assertTrue(entered.wait(5))
            second = self.complete(info["upload_id"])                   # while the first is running
            self.assertEqual(second.status_code, 409)
            self.assertEqual(self.client.delete(f"/api/subjects/uploads/{info['upload_id']}").status_code, 409)
            release.set()
            t.join(10)
        self.assertEqual(add.call_count, 1)
        self.assertEqual(results[0]["saved_pdfs"], ["Big Lecture.pdf"])

    def test_stale_completing_lock_can_be_retaken(self):
        data = make_pdf(2, 9)
        info = self.create(data).json()
        self.put(info["upload_id"], 0, data)
        store = up.get_store()
        self.assertEqual(store.try_lock("alice", info["upload_id"])[0], "ok")    # a crashed worker
        self.assertEqual(self.complete(info["upload_id"]).status_code, 409)
        with mock.patch.object(up, "LOCK_STALE_S", -1.0):
            self.assertEqual(self.complete(info["upload_id"]).status_code, 200)

    def test_transient_failure_keeps_upload_for_retry(self):
        data = make_pdf(2, 10)
        info = self.create(data).json()
        self.put(info["upload_id"], 0, data)
        with mock.patch.object(svc, "add_subject", side_effect=RuntimeError("boom")):
            self.assertEqual(self.complete(info["upload_id"]).status_code, 500)
        self.assertEqual(self.complete(info["upload_id"]).status_code, 200)

    def test_duplicate_file_reports_skipped_duplicates(self):
        data = make_pdf(2, 11)
        for _ in range(2):
            info = self.create(data, filename="again.pdf").json()
            self.put(info["upload_id"], 0, data)
            body = self.complete(info["upload_id"]).json()
        self.assertEqual(body["skipped_duplicates"], ["again.pdf"])
        self.assertEqual(body["saved_pdfs"], [])

    def test_existing_multipart_path_still_works(self):
        r = self.client.post("/api/subjects", data={"name": "Small"},
                             files=[("files", ("s.pdf", make_pdf(1, 12), "application/pdf"))])
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["saved_pdfs"], ["s.pdf"])


class TempStoreTests(_Suite, _Base):
    use_mongo = False

    def test_ttl_expired_upload_is_gone_and_swept(self):
        data, info = self.open_small()
        uid = info["upload_id"]
        meta = self.tmp_store / uid / "meta.json"
        old = time.time() - up.UPLOAD_TTL_S - 5
        os.utime(meta, (old, old))
        self.assertEqual(self.client.get(f"/api/subjects/uploads/{uid}").status_code, 404)
        self.assertEqual(self.put(uid, 0, split(data, info["chunk_size"])[0]).status_code, 404)
        self.create(make_pdf(1, 13))                                    # next create sweeps the dead dir
        self.assertFalse((self.tmp_store / uid).exists())

    def test_chunk_activity_refreshes_ttl(self):
        data, info = self.open_small()
        uid = info["upload_id"]
        meta = self.tmp_store / uid / "meta.json"
        almost = time.time() - up.UPLOAD_TTL_S + 30
        os.utime(meta, (almost, almost))
        self.assertEqual(self.put(uid, 0, split(data, info["chunk_size"])[0]).status_code, 200)
        self.assertGreater(os.stat(meta).st_mtime, almost + 20)

    def test_global_temp_space_bound(self):
        with mock.patch.object(up, "TEMP_MAX_BYTES", 5000):
            self.assertEqual(self.create(b"%PDF-" + b"x" * 3000, sha=False).status_code, 200)
            self.assertEqual(self.create(b"%PDF-" + b"x" * 3000, sha=False, subject="Other").status_code, 503)

    def test_refused_on_vercel_without_database(self):
        with mock.patch.object(up.config, "IS_VERCEL", True):
            self.assertEqual(self.create(make_pdf(1, 14)).status_code, 503)

    def test_storage_is_under_the_store_root_only(self):
        data, info = self.open_small()
        self.put(info["upload_id"], 0, split(data, info["chunk_size"])[0])
        self.assertEqual([p.name for p in self.tmp_store.iterdir()], [info["upload_id"]])


class MongoStoreTests(_Suite, _Base):
    use_mongo = True

    def test_chunks_live_in_mongo_keyed_by_owner(self):
        data, info = self.open_small()
        uid = info["upload_id"]
        self.put(uid, 0, split(data, info["chunk_size"])[0])
        chunk = self.fake["upload_chunks"].docs[f"{uid}:0"]
        self.assertEqual(chunk["owner"], "alice")
        self.assertEqual(self.fake["upload_sessions"].docs[uid]["owner"], "alice")

    def test_complete_deletes_chunks_but_keeps_result(self):
        data = make_pdf(2, 15)
        info = self.create(data).json()
        self.put(info["upload_id"], 0, data)
        self.complete(info["upload_id"])
        self.assertEqual(len(self.fake["upload_chunks"].docs), 0)
        self.assertEqual(self.fake["upload_sessions"].docs[info["upload_id"]]["state"], "done")

    def test_abort_deletes_session_and_chunks(self):
        data, info = self.open_small()
        self.put(info["upload_id"], 0, split(data, info["chunk_size"])[0])
        self.client.delete(f"/api/subjects/uploads/{info['upload_id']}")
        self.assertEqual(len(self.fake["upload_chunks"].docs), 0)
        self.assertEqual(len(self.fake["upload_sessions"].docs), 0)

    def test_ttl_expired_session_not_usable_before_mongo_sweeps_it(self):
        data, info = self.open_small()
        uid = info["upload_id"]
        self.fake["upload_sessions"].docs[uid]["created_at"] = (
            datetime.now(timezone.utc) - timedelta(seconds=up.UPLOAD_TTL_S + 5))
        self.assertEqual(self.client.get(f"/api/subjects/uploads/{uid}").status_code, 404)
        self.assertEqual(self.put(uid, 0, split(data, info["chunk_size"])[0]).status_code, 404)
        self.assertEqual(self.complete(uid).status_code, 404)
        for _ in range(up.MAX_OPEN_UPLOADS):                       # expired rows don't count against the cap
            self.assertEqual(self.create(make_pdf(1, 16)).status_code, 200)

    def test_chunk_activity_refreshes_session_ttl(self):
        data, info = self.open_small()
        uid = info["upload_id"]
        old = datetime.now(timezone.utc) - timedelta(seconds=up.UPLOAD_TTL_S - 30)
        self.fake["upload_sessions"].docs[uid]["created_at"] = old
        self.put(uid, 0, split(data, info["chunk_size"])[0])
        self.assertGreater(self.fake["upload_sessions"].docs[uid]["created_at"], old + timedelta(seconds=20))

    def test_ttl_indexes_are_declared(self):
        self.assertEqual(db.EPHEMERAL_TTLS["upload_sessions"], up.UPLOAD_TTL_S)
        self.assertEqual(db.EPHEMERAL_TTLS["upload_chunks"], up.UPLOAD_TTL_S)


if __name__ == "__main__":
    unittest.main()
