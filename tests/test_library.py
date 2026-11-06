"""Library overview: shape, owner isolation, pagination, hidden standard subjects, empty state, Mongo op budget."""
from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from unittest import mock

from bson import ObjectId
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend import db, documents
from backend.auth import current_owner
from backend.routers import library as library_router
from backend.services import library, progress

NOW = datetime.now(timezone.utc)


def _row(days=0, **kw):
    return {"_id": ObjectId(), "created_at": NOW - timedelta(days=days), **kw}


def _snap(**rows):
    return {k: {"n": len(rows.get(k, [])), "rows": rows.get(k, [])} for k in db.KINDS}


class FakeColl:
    def __init__(self, rows=None):
        self.rows, self.pipelines = rows or [], []

    def aggregate(self, pipeline, **kw):
        self.pipelines.append(pipeline)
        return iter([dict(r) for r in self.rows])


LLM1 = {"provider": "anthropic", "model": "claude-sonnet-5-5", "step": 1, "label": "Model: anthropic", "skipped": []}
LLM2 = {"provider": "groq", "model": "llama", "step": 3, "label": "Model: groq", "skipped": ["anthropic 429", "groq#1 timeout"]}


class Base(unittest.TestCase):
    def setUp(self):
        library.clear_cache()
        progress.clear_cache()
        self.chat_id = ObjectId()
        self.snap = _snap(
            quizzes=[_row(0, title="Quiz · DBMS / a.pdf", source="DBMS / a.pdf", num_questions=10, llm=LLM1),
                     _row(1, title="Quiz · loose topic", source="loose topic", num_questions=5, llm=LLM1)],
            flashcards=[_row(2, title="Cards", subject="DBMS", num_cards=12, llm=LLM2)],
            exams=[_row(0, title="Exam · DBMS", subject="DBMS", percent=72, llm=LLM1)],
            chats=[{"_id": self.chat_id, "created_at": NOW - timedelta(days=9), "updated_at": NOW - timedelta(days=1),
                    "title": "Chat about keys", "subject": "DBMS", "label": "Subject · DBMS"}],
            cheatsheets=[_row(40, title="Cheat · Mathematics", source="Mathematics", llm=LLM1)],
        )
        self.extra = FakeColl([
            {"_k": "chat", "_id": self.chat_id, "n": 14, "llms": [dict(LLM1), dict(LLM2)]},
            {"_k": "up", "subject": "DBMS", "filename": "a.pdf", "length": 2048, "uploadDate": NOW - timedelta(days=3)},
            {"_k": "ix", "subject": "DBMS", "length": 4096, "uploadDate": NOW, "m_n_chunks": 40, "m_n_embedded": 30, "m_complete": False},
        ])

    def run_lib(self, owner="u1", visible=("DBMS", "Mathematics"), hidden=(), **kw):
        with mock.patch.object(db, "_get_db", return_value={"chats": self.extra}), \
             mock.patch.object(db, "warm_owner", return_value=self.snap), \
             mock.patch.object(documents, "list_subjects", return_value=list(visible)), \
             mock.patch.object(library, "_is_bundled", side_effect=lambda n: n == "Mathematics"), \
             mock.patch.object(library, "_file_entries", side_effect=lambda n, o, b, remote, budget: (
                 [{"name": r["filename"], "pages": 7, "bytes": r["length"], "added": None, "text_only": False} for r in remote])), \
             mock.patch("backend.prefs.hidden_standard", return_value=list(hidden)), \
             mock.patch.object(progress, "build_summary", return_value={"subjects": [{"subject": "DBMS", "attempts": 3, "accuracy": .6, "topics": [
                 {"topic": "Joins", "attempts": 2, "mastery": .3, "questions": 4}, {"topic": "Keys", "attempts": 1, "mastery": .9, "questions": 2},
                 {"topic": "SQL", "attempts": 1, "mastery": .5, "questions": 1}]}]}):
            return library.overview(owner, **kw)


class TestOverview(Base):
    def test_shape_and_counts(self):
        out = self.run_lib()
        cards = {c["name"]: c for c in out["subjects"]}
        d = cards["DBMS"]
        self.assertEqual(d["counts"]["quizzes"], 1)
        self.assertEqual(d["counts"]["flashcard_decks"], 1)
        self.assertEqual(d["counts"]["flashcard_cards"], 12)
        self.assertEqual(d["counts"]["exams"], 1)
        self.assertEqual((d["counts"]["chats"], d["counts"]["chat_messages"]), (1, 14))
        self.assertEqual(d["counts"]["uploads"], 1)
        self.assertTrue(d["own"])
        self.assertEqual(d["totals"]["files"], 1)
        self.assertEqual(d["index"]["state"], "partial")
        self.assertEqual((d["index"]["n_embedded"], d["index"]["n_chunks"]), (30, 40))
        self.assertEqual(d["mastery"]["weak"][0]["topic"], "Joins")
        self.assertEqual(len(d["mastery"]["weak"]), 2)
        self.assertFalse(cards["Mathematics"]["own"])
        self.assertEqual(cards["Mathematics"]["counts"]["cheatsheets"], 1)
        self.assertEqual(out["other"]["counts"]["quizzes"], 1)           # the unattributed topic quiz
        # model usage: 2 quizzes + exam + chat answers (2) + flashcards + cheatsheet
        s = out["summary"]
        self.assertEqual(sum(m["count"] for m in s["models"]), 7)
        self.assertEqual({p["provider"] for p in s["providers"]}, {"anthropic", "groq"})
        self.assertEqual(s["files"], 1)
        self.assertEqual(s["hidden"]["count"], 0)

    def test_hidden_standard_subjects_excluded(self):
        out = self.run_lib(visible=("DBMS",), hidden=("Mathematics",))
        self.assertEqual([c["name"] for c in out["subjects"]], ["DBMS"])
        self.assertEqual(out["summary"]["hidden"], {"count": 1, "names": ["Mathematics"]})
        self.assertNotIn("cheatsheets", out["summary"]["activity_by_kind"])     # hidden subject's work is not shown
        self.assertIsNotNone(out["other"])                                        # but the loose quiz still is

    def test_empty_account(self):
        self.snap = _snap()
        self.extra = FakeColl([])
        out = self.run_lib(visible=())
        self.assertEqual(out["subjects"], [])
        self.assertIsNone(out["other"])
        self.assertEqual(out["summary"]["files"], 0)
        self.assertEqual(out["summary"]["activity_total"], 0)

    def test_persistence_off(self):
        with mock.patch.object(db, "_get_db", return_value=None), mock.patch.object(db, "warm_owner", return_value=None), \
             mock.patch.object(documents, "list_subjects", return_value=["Mathematics"]), \
             mock.patch.object(library, "_file_entries", return_value=[]), mock.patch("backend.prefs.hidden_standard", return_value=[]):
            out = library.overview("u1")
        self.assertEqual(out["subjects"][0]["name"], "Mathematics")


class TestTimeline(Base):
    def tl(self, subject="DBMS", **kw):
        self.run_lib()          # prime the cache (the timeline reads the cached build)
        with mock.patch.object(documents, "list_subjects", return_value=["DBMS", "Mathematics"]), \
             mock.patch("backend.prefs.hidden_standard", return_value=[]):
            return library.timeline("u1", subject, **kw)

    def test_order_labels_and_models(self):
        t = self.tl()
        ts = [i["ts"] for i in t["items"]]
        self.assertEqual(ts, sorted(ts, reverse=True))
        kinds = [i["kind"] for i in t["items"]]
        self.assertEqual(set(kinds), {"quizzes", "flashcards", "exams", "chats", "uploads"})
        quiz = next(i for i in t["items"] if i["kind"] == "quizzes")
        self.assertEqual(quiz["summary"], "Quiz · 10 questions")
        self.assertEqual(quiz["llm"]["provider"], "anthropic")
        self.assertEqual(next(i for i in t["items"] if i["kind"] == "exams")["summary"], "Exam graded · 72%")
        self.assertEqual(next(i for i in t["items"] if i["kind"] == "chats")["summary"], "Chat · 14 messages")
        self.assertEqual({(m["provider"], m["step"]): m["count"] for m in t["models"]}, {("anthropic", 1): 3, ("groq", 3): 2})
        self.assertEqual(t["chain"]["answers"], 5)
        self.assertEqual(t["chain"]["later_steps"], 2)
        self.assertEqual({r["reason"] for r in t["chain"]["reasons"]}, {"429", "timeout"})
        self.assertGreaterEqual(t["week"]["this_week"], 3)

    def test_pagination_and_filter(self):
        first = self.tl(limit=2)
        self.assertEqual(len(first["items"]), 2)
        self.assertTrue(first["has_more"])
        seen = [i["id"] for i in first["items"]]
        cur = first["next_cursor"]
        while cur:
            page = self.tl(limit=2, before=cur)
            seen += [i["id"] for i in page["items"]]
            cur = page["next_cursor"]
        self.assertEqual(len(seen), len(set(seen)))
        self.assertEqual(len(seen), 5)
        only = self.tl(kind="quizzes")
        self.assertEqual([i["kind"] for i in only["items"]], ["quizzes"])
        with self.assertRaises(library.LibraryError):
            self.tl(kind="nope")
        with self.assertRaises(library.LibraryError):
            self.tl(subject="Nope")
        with self.assertRaises(library.LibraryError):
            self.tl(before="garbage")

    def test_other_bucket(self):
        t = self.tl(subject=library.OTHER_KEY)
        self.assertEqual([i["kind"] for i in t["items"]], ["quizzes"])


class TestOpBudget(Base):
    def test_one_extra_op_then_cached(self):
        self.run_lib()
        self.assertEqual(len(self.extra.pipelines), 1)                      # snapshot is mocked: only our aggregation
        owners = [st["$match"]["owner"] for st in self.extra.pipelines[0] if "$match" in st]
        self.assertEqual(owners, ["u1"])
        unions = [st["$unionWith"] for st in self.extra.pipelines[0] if "$unionWith" in st]
        self.assertEqual([u["coll"] for u in unions], [db.UPLOADS_BUCKET + ".files", db.INDEX_BUCKET + ".files"])
        for u in unions:
            self.assertEqual(u["pipeline"][0], {"$match": {"owner": "u1"}})
        self.run_lib()
        self.run_lib()
        self.assertEqual(len(self.extra.pipelines), 1)                      # warm: zero extra ops
        self.run_lib(refresh=True)
        self.assertEqual(len(self.extra.pipelines), 2)

    def test_write_epoch_invalidates(self):
        self.run_lib()
        db._bump("u1", keep_snapshot=True)
        self.run_lib()
        self.assertEqual(len(self.extra.pipelines), 2)


class TestRouter(Base):
    def client(self, owner="u1"):
        app = FastAPI()
        app.include_router(library_router.router)
        app.dependency_overrides[current_owner] = lambda: owner
        return TestClient(app)

    def test_endpoints_headers_and_isolation(self):
        library_router._hits.clear()
        with mock.patch.object(db, "_get_db", return_value={"chats": self.extra}), \
             mock.patch.object(db, "warm_owner", return_value=self.snap), \
             mock.patch.object(documents, "list_subjects", return_value=["DBMS"]), \
             mock.patch.object(library, "_file_entries", return_value=[]), mock.patch("backend.prefs.hidden_standard", return_value=[]):
            r = self.client().get("/api/library")
            self.assertEqual(r.status_code, 200)
            self.assertEqual(r.headers["cache-control"], "private, no-store")
            self.assertEqual(r.json()["subjects"][0]["name"], "DBMS")
            t = self.client().get("/api/library/DBMS/timeline?limit=3")
            self.assertEqual(t.status_code, 200)
            self.assertEqual(len(t.json()["items"]), 3)
            self.assertEqual(self.client().get("/api/library/Nope/timeline").status_code, 404)
            self.assertEqual(self.client().get("/api/library/DBMS/timeline?kind=zz").status_code, 400)
            # another owner never sees the first owner's cached build
            other = _snap()
            with mock.patch.object(db, "warm_owner", return_value=other), mock.patch.object(library, "_load_extra", return_value={"chats": {}, "uploads": {}, "index": {}, "ok": True}):
                o = self.client("u2").get("/api/library").json()
            self.assertEqual(o["summary"]["activity_total"], 0)

    def test_rate_limited(self):
        library_router._hits.clear()
        with mock.patch.object(library_router, "_PER_MIN", 2), \
             mock.patch.object(library, "overview", return_value={"summary": {}, "subjects": []}):
            c = self.client("rl")
            codes = [c.get("/api/library").status_code for _ in range(3)]
        self.assertEqual(codes, [200, 200, 429])


if __name__ == "__main__":
    unittest.main()
