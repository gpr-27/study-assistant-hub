"""Mongo round-trip budget: single-aggregation reads, write-through chat state, owner isolation."""
from __future__ import annotations

import unittest
from unittest import mock

from backend import db
from backend.services import chat as chat_service


def _session(owner="alice", chat_id="507f1f77bcf86cd799439011"):
    return chat_service.Session(id="s", owner=owner, mode="topic", label="l", subject=None, topic="t",
                                chat_id=chat_id, nonce="n1")


class FakeColl:
    def __init__(self, rows=None):
        self.rows, self.pipelines = rows or [], []

    def aggregate(self, pipeline, **kw):
        self.pipelines.append(pipeline)
        return iter(self.rows)


class TestMultiRecent(unittest.TestCase):
    def test_one_aggregation_every_branch_owner_scoped(self):
        coll = FakeColl([{"_k": "attempts", "correct": 1, "total": 2},
                         {"_k": "chats", "n": 3}, {"_k": "exams", "total_score": 5, "max_score": 10}])
        database = {"attempts": coll}
        specs = [("attempts", 100, ["correct", "total"]), ("chats", 100, None), ("exams", 100, ["total_score", "max_score"]),
                 ("quizzes", 100, None)]
        with mock.patch.object(db, "_get_db", return_value=database):
            out = db.multi_recent("guest_a", specs)
        self.assertEqual(len(coll.pipelines), 1)                       # one round trip
        pipe = coll.pipelines[0]
        self.assertEqual(pipe[0], {"$match": {"owner": "guest_a"}})
        unions = [st["$unionWith"] for st in pipe if "$unionWith" in st]
        self.assertEqual([u["coll"] for u in unions], ["chats", "exams", "quizzes"])
        for u in unions:                                                # every branch is owner-scoped + indexed sort
            self.assertEqual(u["pipeline"][0], {"$match": {"owner": "guest_a"}})
            self.assertEqual(u["pipeline"][1], {"$sort": {"created_at": -1}})
        self.assertEqual(out["attempts"]["rows"], [{"correct": 1, "total": 2}])
        self.assertEqual(out["chats"]["n"], 3)
        self.assertEqual(out["exams"]["n"], 1)
        self.assertEqual(out["quizzes"], {"n": 0, "rows": []})

    def test_disabled_db_returns_empty_slots(self):
        with mock.patch.object(db, "_get_db", return_value=None):
            out = db.multi_recent("x", [("chats", 5, None)])
        self.assertEqual(out["chats"], {"n": 0, "rows": []})

    def test_limit_is_clamped(self):
        coll = FakeColl()
        with mock.patch.object(db, "_get_db", return_value={"chats": coll}):
            db.multi_recent("x", [("chats", 10**9, None)])
        self.assertEqual(coll.pipelines[0][2], {"$limit": 200})


class TestListUserSubjects(unittest.TestCase):
    def test_single_aggregate_over_both_buckets(self):
        up = FakeColl([{"_id": "DBMS"}, {"_id": "OS"}, {"_id": None}])
        database = {db.UPLOADS_BUCKET + ".files": up}
        with mock.patch.object(db, "_get_db", return_value=database):
            names = db.list_user_subjects("u1")
        self.assertEqual(names, ["DBMS", "OS"])
        self.assertEqual(len(up.pipelines), 1)
        self.assertEqual(up.pipelines[0][0], {"$match": {"owner": "u1"}})
        self.assertEqual(up.pipelines[0][2]["$unionWith"]["coll"], db.INDEX_BUCKET + ".files")
        self.assertEqual(db.list_user_subjects(None), [])


class TestEpoch(unittest.TestCase):
    def test_save_bumps_only_that_owner(self):
        a0, b0 = db.epoch("ea"), db.epoch("eb")
        with mock.patch.object(db, "_get_db", return_value={"quizzes": mock.MagicMock()}):
            db.save("quizzes", "ea", {"x": 1})
        self.assertNotEqual(db.epoch("ea"), a0)
        self.assertEqual(db.epoch("eb"), b0)


class FakeFind:
    """Collection stand-in: ``aggregate`` serves the owner snapshot (rows tagged with the owner asked for,
    as ``quizzes``), ``find`` serves the non-default (search) list path."""

    def __init__(self):
        self.calls = []           # owners asked for via aggregate (snapshot loads)
        self.finds = []           # owners asked for via find (search variants)

    def aggregate(self, pipeline, **kw):
        owner = pipeline[0]["$match"]["owner"]
        self.calls.append(owner)
        return iter([{"_id": "x", "_k": "quizzes", "title": f"item of {owner}", "created_at": None}])

    def find(self, query, projection=None):
        self.finds.append(query["owner"])
        rows = [{"_id": "x", "title": f"item of {query['owner']}", "created_at": None}]
        cur = mock.MagicMock()
        cur.sort.return_value.limit.return_value = rows
        return cur

    def insert_one(self, doc):
        return mock.Mock(inserted_id="new")


class TestListCache(unittest.TestCase):
    def setUp(self):
        db.clear_list_cache()
        self.addCleanup(db.clear_list_cache)
        self.coll = FakeFind()
        p = mock.patch.object(db, "_get_db", return_value={k: self.coll for k in db.KINDS})
        p.start()
        self.addCleanup(p.stop)

    def test_second_list_is_cached_and_a_copy(self):
        a = db.list_items("quizzes", "lc_alice")
        a[0]["title"] = "mutated"
        b = db.list_items("quizzes", "lc_alice")
        self.assertEqual(self.coll.calls, ["lc_alice"])
        self.assertEqual(b[0]["title"], "item of lc_alice")

    def test_one_aggregation_serves_every_kind(self):
        db.list_items("quizzes", "lc_all")
        db.list_items("chats", "lc_all")
        db.list_items("flashcards", "lc_all")
        self.assertEqual(self.coll.calls, ["lc_all"])
        self.assertEqual(db.list_items("chats", "lc_all"), [])

    def test_owners_isolated(self):
        a = db.list_items("quizzes", "lc_alice")
        b = db.list_items("quizzes", "lc_bob")
        self.assertEqual(a[0]["title"], "item of lc_alice")
        self.assertEqual(b[0]["title"], "item of lc_bob")

    def test_own_save_is_patched_in_without_a_reread(self):
        db.list_items("quizzes", "lc_alice")
        db.save("quizzes", "lc_bob", {"t": 1})                    # another owner's write: untouched
        db.list_items("quizzes", "lc_alice")
        self.assertEqual(self.coll.calls, ["lc_alice"])
        db.save("quizzes", "lc_alice", {"title": "fresh", "questions": [1, 2, 3]})
        got = db.list_items("quizzes", "lc_alice")
        self.assertEqual(self.coll.calls, ["lc_alice"])           # patched, not re-read
        self.assertEqual([i["title"] for i in got], ["fresh", "item of lc_alice"])
        self.assertNotIn("questions", got[0])                     # heavy fields never reach a list row
        self.assertNotIn("owner", got[0])

    def test_unpatchable_write_invalidates_only_that_owner(self):
        db.list_items("quizzes", "lc_alice")
        db.list_items("quizzes", "lc_bob")
        db._bump("lc_alice")                                      # e.g. a bulk change we cannot patch
        db.list_items("quizzes", "lc_alice")
        db.list_items("quizzes", "lc_bob")
        self.assertEqual(self.coll.calls, ["lc_alice", "lc_bob", "lc_alice"])

    def test_delete_rename_clear_are_patched(self):
        coll = self.coll
        coll.delete_one = lambda flt: mock.Mock(deleted_count=1)
        coll.update_one = lambda flt, upd: mock.Mock(matched_count=1)
        coll.delete_many = lambda flt: mock.Mock(deleted_count=1)
        db.list_items("quizzes", "lc_p")
        oid = "507f1f77bcf86cd799439011"
        self.assertTrue(db.rename_item("quizzes", "lc_p", oid, "new name"))
        self.assertEqual(db.list_items("quizzes", "lc_p")[0]["title"], "item of lc_p")   # id "x" differs: unchanged
        self.assertEqual(db.clear_kind("quizzes", "lc_p"), 1)
        self.assertEqual(db.list_items("quizzes", "lc_p"), [])
        self.assertEqual(coll.calls, ["lc_p"])                    # no re-read through any of it

    def test_ttl_and_search_variants(self):
        with mock.patch.object(db, "_SNAP_TTL_S", -1.0):
            db.list_items("quizzes", "lc_t")
            db.list_items("quizzes", "lc_t")
        self.assertEqual(len(self.coll.calls), 2)
        db.clear_list_cache()
        self.coll.calls.clear()
        db.list_items("quizzes", "lc_q")
        db.list_items("quizzes", "lc_q", q="abc")
        self.assertEqual(len(self.coll.calls), 1)
        self.assertEqual(len(self.coll.finds), 1)          # a search is a different query -> its own read

    def test_snapshot_serves_multi_recent_without_a_query(self):
        db.list_items("quizzes", "lc_m")
        out = db.multi_recent("lc_m", [("quizzes", 100, ["title"]), ("chats", 100, None)])
        self.assertEqual(out["quizzes"]["rows"], [{"title": "item of lc_m"}])
        self.assertEqual(out["chats"], {"n": 0, "rows": []})
        self.assertEqual(self.coll.calls, ["lc_m"])


class TestChatStateCache(unittest.TestCase):
    def setUp(self):
        chat_service._STATE.clear()
        self.addCleanup(chat_service._STATE.clear)

    def test_primed_chat_needs_no_read(self):
        s = _session()
        chat_service.prime_state(s.owner, s.chat_id)
        with mock.patch.object(chat_service, "_db_load", side_effect=AssertionError("Mongo read")):
            st = chat_service.load_state(s)
        self.assertEqual(st["history"], [])

    def test_read_once_then_cached(self):
        s = _session()
        with mock.patch.object(chat_service, "_db_load",
                               return_value={"history": [("q", "a")], "recent_artifact": None}) as rd:
            chat_service.load_state(s)
            st = chat_service.load_state(s)
        self.assertEqual(rd.call_count, 1)
        self.assertEqual(st["history"], [("q", "a")])

    def test_persist_is_write_through_and_one_write(self):
        s = _session()
        coll = mock.MagicMock()
        chat_service.prime_state(s.owner, s.chat_id)
        with mock.patch.object(chat_service, "_chats_collection", return_value=coll):
            chat_service._persist_exchange(s, "hello", "hi there", {"label": "m"}, None, True)
        self.assertEqual(coll.update_one.call_count, 1)                # push + title in a single write
        upd = coll.update_one.call_args.args[1]
        self.assertEqual(upd["$set"]["title"], "hello")
        with mock.patch.object(chat_service, "_db_load", side_effect=AssertionError("Mongo read")):
            self.assertEqual(chat_service.load_state(s)["history"], [("hello", "hi there")])

    def test_owners_never_share_state(self):
        a, b = _session("alice"), _session("bob")
        with mock.patch.object(chat_service, "_db_load",
                               side_effect=[{"history": [("qa", "aa")], "recent_artifact": None},
                                            {"history": [("qb", "ab")], "recent_artifact": None}]):
            self.assertEqual(chat_service.load_state(a)["history"], [("qa", "aa")])
            self.assertEqual(chat_service.load_state(b)["history"], [("qb", "ab")])
        self.assertEqual(chat_service.load_state(a)["history"], [("qa", "aa")])

    def test_ttl_expiry_rereads(self):
        s = _session()
        with mock.patch.object(chat_service, "_STATE_TTL_S", -1.0), \
             mock.patch.object(chat_service, "_db_load", return_value={"history": [], "recent_artifact": None}) as rd:
            chat_service.load_state(s)
            chat_service.load_state(s)
        self.assertEqual(rd.call_count, 2)

    def test_bounded(self):
        with mock.patch.object(chat_service, "_STATE_MAX", 3):
            for i in range(10):
                chat_service.prime_state("o", f"c{i}")
        self.assertLessEqual(len(chat_service._STATE), 3)


if __name__ == "__main__":
    unittest.main()
