"""Concept graph (GraphRAG-style retrieval): build, PPR/BFS expansion, communities, fusion, planner
decomposition, owner isolation + version invalidation of stored graphs, embeddings-down fallback.

No network, no keys, no LLM: embeddings are a hashed bag-of-words, Mongo is an in-memory fake, and any
accidental chain call fails the test.
"""
import asyncio
import json
import os
import unittest
from unittest import mock

import numpy as np

from backend import agentic_rag, documents
from backend import gemini_embeddings as ge
from backend import vector_store as vs
from backend.rag import graph as rg
from backend.rag import query as rq
from backend.rag import retrieve as rret
from tests.test_vector_store import DIM, _Env, _embed_text, fake_embed_query

TOPICS = {
    "deadlock": [
        "A deadlock occurs when processes wait on each other forever. Deadlock detection uses a resource allocation graph.",
        "Deadlock prevention removes one of the four necessary conditions; the circular wait condition is broken by ordering resources.",
        "Resource allocation graph edges show which process holds a resource and which process requests a resource.",
        "The banker algorithm avoids deadlock by checking that the system stays in a safe state after each request.",
    ],
    "paging": [
        "Paging divides physical memory into frames and logical memory into pages. The page table maps pages to frames.",
        "A page fault happens when a process touches a page that is not in memory; the page replacement algorithm picks a victim frame.",
        "The translation lookaside buffer caches page table entries so address translation avoids extra memory accesses.",
        "Thrashing occurs when a process spends more time handling page faults than executing; the working set model limits it.",
    ],
    "scheduling": [
        "CPU scheduling selects which process runs next. Round robin scheduling gives each process a time quantum.",
        "Shortest job first scheduling minimises average waiting time but needs a prediction of the next CPU burst.",
        "The ready queue holds processes waiting for the CPU; the dispatcher gives the CPU to the process the scheduler selected.",
        "Priority scheduling can starve a low priority process; aging raises priority over time to prevent starvation.",
    ],
}


def corpus(topics=TOPICS, source="os.pdf", repeat=2):
    chunks, page = [], 0
    for _name, texts in topics.items():
        for t in texts:
            for _ in range(repeat):
                chunks.append({"text": t + " " + t.split(".")[0] + " matters.", "source": source, "page": page})
                page += 1
    return chunks


def index_for(chunks, with_vectors=True, manifest=None):
    vecs = np.vstack([_embed_text(c["text"]) for c in chunks]) if with_vectors else None
    m = {"model": "gemini-embedding-2", "dim": DIM, "complete": True, **(manifest or {})}
    return vs.VectorIndex("k", [dict(c) for c in chunks], vecs, m)


class BuildTests(unittest.TestCase):
    def setUp(self):
        self.chunks = corpus()
        self.data = rg.build_graph(self.chunks)

    def test_concepts_edges_membership(self):
        d = self.data
        self.assertIsNotNone(d)
        names = set(d["names"])
        self.assertTrue({"page table", "deadlock"} & names or any("deadlock" in n for n in names))
        self.assertTrue(any("page table" in n for n in names), sorted(names))
        self.assertGreater(len(d["ea"]), 10)
        self.assertTrue(all(w > 0 for w in d["ew"]))
        self.assertEqual(len(d["mk"]), len(d["mc"]))
        self.assertTrue(set(d["mk"]) <= set(range(len(self.chunks))))
        self.assertEqual(d["n_chunks"], len(self.chunks))
        self.assertEqual(d["sig"], rg.chunks_sig(self.chunks))

    def test_deterministic_and_json_roundtrip(self):
        again = rg.build_graph(self.chunks)
        strip = lambda x: {k: v for k, v in x.items() if k != "built_s"}      # noqa: E731
        self.assertEqual(strip(self.data), strip(again))
        back = json.loads(rg.dumps(self.data).decode("utf-8"))
        self.assertIsNotNone(rg.load(back, self.chunks))

    def test_too_small_corpus_has_no_graph(self):
        self.assertIsNone(rg.build_graph(self.chunks[:1]))
        self.assertIsNone(rg.build_graph([]))

    def test_definition_heading_and_acronym_concepts(self):
        chunks = corpus()
        chunks.append({"text": "The Translation Lookaside Buffer (TLB) is a small cache. TLB hits avoid walks.",
                       "source": "os.pdf", "page": 99})
        chunks.append({"text": "A semaphore is an integer variable used for synchronisation.", "source": "os.pdf", "page": 100})
        d = rg.build_graph(chunks)
        names = [n.lower() for n in d["names"]]
        self.assertIn("tlb", names)
        self.assertIn("semaphore", names)            # df=1 but a definition pattern

    def test_default_path_makes_no_llm_call(self):
        with mock.patch("backend.llm_chain.chat", side_effect=AssertionError("no LLM on the default path")):
            self.assertIsNotNone(rg.build_graph(self.chunks))

    def test_llm_relations_one_batched_call_per_20_chunks_sensitive_for_users(self):
        calls = []

        def fake_chat(system, messages, **kw):
            calls.append(kw)
            return mock.Mock(text=json.dumps({"relations": [{"i": 0, "type": "requires"}, {"i": 1, "type": "bogus"}]}))

        with mock.patch("backend.llm_chain.chat", fake_chat):
            d = rg.build_graph(self.chunks, sensitive=True, llm=True)
        n_calls = -(-len(self.chunks) // rg.CHUNKS_PER_CALL)
        self.assertLessEqual(len(calls), min(n_calls, 5) + 1)        # relations (<= 1 per 20 chunks) + 1 summary call
        self.assertGreaterEqual(len(calls), 1)
        self.assertTrue(all(c["json_mode"] and c["sensitive"] for c in calls))
        self.assertTrue(d["rel"])
        self.assertTrue(all(t in rg.REL_TYPES for _i, t in d["rel"]))
        self.assertIsNotNone(rg.load(d, self.chunks))        # typed edges still load

    def test_llm_failure_keeps_the_graph(self):
        with mock.patch("backend.llm_chain.chat", side_effect=RuntimeError("down")):
            d = rg.build_graph(self.chunks, llm=True)
        self.assertIsNotNone(d)
        self.assertEqual(d["rel"], [])


class ExpansionTests(unittest.TestCase):
    def setUp(self):
        self.chunks = corpus()
        self.g = rg.load(rg.build_graph(self.chunks), self.chunks)

    def text(self, i):
        return self.chunks[i]["text"]

    def test_ppr_reaches_chunks_that_lack_the_query_term(self):
        seeds, mass = self.g.concept_mass("deadlock", method="ppr")
        self.assertTrue(seeds)
        ranked = self.g.rank_chunks("deadlock", 30, seed_discount=0.0)
        self.assertTrue(ranked)
        # expansion-only scoring surfaces passages about deadlock concepts that do not say "deadlock"
        non_literal = [i for i, _ in ranked if "deadlock" not in self.text(i).lower()]
        self.assertTrue(non_literal, "PPR should reach neighbouring concepts' chunks")
        topics = {self.text(i)[:20] for i, _ in ranked[:6]}
        self.assertTrue(all("Paging" not in t for t in topics))

    def test_full_scoring_ranks_the_literal_topic_first(self):
        top = self.g.rank_chunks("page table frames", 5, seed_discount=1.0)
        self.assertTrue(top)
        self.assertIn("page", self.text(top[0][0]).lower())

    def test_bfs_hops_are_bounded(self):
        seeds = self.g.seeds("deadlock")
        m0 = self.g.bfs(seeds, hops=0)
        m1 = self.g.bfs(seeds, hops=1)
        m2 = self.g.bfs(seeds, hops=2)
        n0, n1, n2 = (int((m > 0).sum()) for m in (m0, m1, m2))
        self.assertLessEqual(n0, n1)
        self.assertLessEqual(n1, n2)
        self.assertEqual(n0, len(seeds))
        self.assertGreater(n1, n0)

    def test_unknown_query_has_no_graph_signal(self):
        self.assertEqual(self.g.rank_chunks("zebra cheesecake xylophone", 5), [])
        self.assertEqual(self.g.related("zebra cheesecake xylophone"), [])

    def test_related_concepts_exclude_the_query_itself(self):
        rel = self.g.related("deadlock detection", 4)
        self.assertTrue(rel)
        self.assertFalse(any(r.lower() == "deadlock" for r in rel))

    def test_query_cost_is_tiny(self):
        import time

        self.g.rank_chunks("deadlock", 10)
        t = time.perf_counter()
        for _ in range(50):
            self.g.rank_chunks("deadlock resource allocation", 10)
        self.assertLess((time.perf_counter() - t) / 50, 0.005)        # < 5 ms per query


class CommunityTests(unittest.TestCase):
    def setUp(self):
        self.chunks = corpus()
        self.data = rg.build_graph(self.chunks)
        self.g = rg.load(self.data, self.chunks)

    def test_communities_with_extractive_summaries(self):
        comms = self.g.communities
        self.assertGreaterEqual(len(comms), 2)
        for c in comms:
            self.assertTrue(c["label"])
            self.assertTrue(c["chunks"])
            self.assertTrue(c["summary"], c)
            self.assertTrue(any(c["summary"][:30] in ch["text"] for ch in self.chunks), "summary is extracted, not generated")

    def test_global_chunks_cover_the_topics(self):
        got = [i for i, _ in self.g.global_chunks(6)]
        self.assertEqual(len(got), len(set(got)))
        text = " ".join(self.chunks[i]["text"].lower() for i in got)
        self.assertTrue(all(w in text for w in ("deadlock", "page", "scheduling")), text[:200])

    def test_global_question_detection(self):
        pos = ["Summarise the whole subject", "What are the main themes of my notes?", "Give me an overview of everything",
               "What does this course cover?"]
        neg = ["Summarise the main ideas of deadlocks", "What is a page fault?", "Explain the overview of paging"]
        for q in pos:
            self.assertTrue(rg.is_global_question(q, self.g), q)
        for q in neg:
            self.assertFalse(rg.is_global_question(q, self.g), q)
        self.assertTrue(rg.is_global_question("What are the main topics in the Operating System notes?", self.g,
                                              "Operating System"))

    def test_llm_summaries_are_opt_in(self):
        with mock.patch("backend.llm_chain.chat", side_effect=AssertionError("default path must not call the LLM")):
            rg.build_graph(self.chunks)
        out = json.dumps({"summaries": [f"theme {i}" for i in range(30)]})
        with mock.patch("backend.llm_chain.chat", return_value=mock.Mock(text=out)) as chat:
            d = rg.build_graph(self.chunks, llm=True)
        self.assertTrue(d["comm"][0]["summary"].startswith("theme"))
        self.assertTrue(chat.called)


class FusionTests(unittest.TestCase):
    def setUp(self):
        self.chunks = corpus()
        self.idx = index_for(self.chunks)
        os.environ.pop("GRAPH_RAG_RANK", None)

    def tearDown(self):
        os.environ.pop("GRAPH_RAG_RANK", None)

    def test_graph_is_a_third_ranker_in_the_rrf(self):
        q = "deadlock"
        qv = _embed_text(q)
        base = [i for i, _s, _m in self.idx.rank(q, qv, 12, "hybrid")]
        fused = [i for i, _s, _m in self.idx.rank(q, qv, 12, "hybrid+graph")]
        self.assertNotEqual(base, fused)
        gr = {i for i, _ in self.idx.graph().rank_chunks(q, 30)}
        self.assertTrue(set(fused) & gr)
        self.assertEqual({m for _i, _s, m in self.idx.rank(q, qv, 5, "hybrid+graph")}, {"hybrid"})

    def test_graph_only_mode_and_no_signal_fallback(self):
        self.assertTrue(all(m == "graph" for _i, _s, m in self.idx.rank("deadlock", None, 5, "graph")))
        out = self.idx.rank("zebra", None, 4, "graph")
        self.assertEqual(len(out), 4)                       # stable arbitrary context, never an error

    def test_search_uses_the_graph_only_when_enabled(self):
        with mock.patch.object(ge, "embed_query", fake_embed_query), \
                mock.patch.object(vs.VectorIndex, "rank", wraps=self.idx.rank) as rank:
            self.idx.search("deadlock", 3)
            self.assertEqual(rank.call_args.args[3], "hybrid")
            os.environ["GRAPH_RAG_RANK"] = "1"
            self.idx.search("deadlock", 3)
            self.assertEqual(rank.call_args.args[3], "hybrid+graph")

    def test_embeddings_down_still_fuses_bm25_and_graph(self):
        out = self.idx.rank("deadlock detection", None, 6, "hybrid+graph")
        self.assertTrue(out)
        self.assertTrue(all(m == "lexical" for _i, _s, m in out))
        lex_only = [i for i, _s, _m in self.idx.rank("deadlock detection", None, 6, "hybrid")]
        self.assertIn("deadlock", self.chunks[out[0][0]]["text"].lower())
        self.assertTrue(lex_only)

    def test_graph_failure_never_breaks_retrieval(self):
        with mock.patch.object(rg.ConceptGraph, "rank_chunks", side_effect=RuntimeError("boom")):
            out = self.idx.rank("deadlock", _embed_text("deadlock"), 5, "hybrid+graph")
        self.assertEqual(len(out), 5)


class PlannerTests(unittest.TestCase):
    def test_compare_questions_split_per_item(self):
        k, subs = rq.decompose("Compare monolithic, layered and microkernel operating system structures.")
        self.assertEqual(k, "compare")
        self.assertEqual(len(subs), 3)
        self.assertTrue(all("operating system structures" in s for s in subs))
        k, subs = rq.decompose("What is the difference between symmetric multiprocessing and clustered systems?")
        self.assertEqual((k, subs), ("compare", ["symmetric multiprocessing", "clustered systems"]))
        self.assertEqual(rq.decompose("stack vs heap"), ("compare", ["stack", "heap"]))
        self.assertEqual(rq.decompose("How do virtualization and cloud computing relate to each other?"),
                         ("compare", ["virtualization", "cloud computing"]))

    def test_multi_part_questions(self):
        k, subs = rq.decompose("What is an interrupt? And how does the OS handle it?")
        self.assertEqual(k, "multi")
        self.assertEqual(len(subs), 2)
        k, subs = rq.decompose("Define a matrix and explain how its inverse is found")
        self.assertEqual(k, "multi")

    def test_why_and_single(self):
        k, subs = rq.decompose("Why is dual-mode operation needed?")
        self.assertEqual(k, "why")
        self.assertEqual(len(subs), 2)
        for q in ("What is a process?", "Explain paging", "", "What is a kernel?"):
            self.assertEqual(rq.decompose(q), ("single", []), q)

    def test_subqueries_are_capped_and_plan_carries_them(self):
        p = rq.plan_query("Compare a, b, c, d and e structures of the operating system", [], None)
        self.assertLessEqual(len(p.subqueries), rq.MAX_SUBQUERIES)
        p = rq.plan_query("What is the difference between paging and segmentation?", [], None)
        self.assertEqual((p.kind, p.subqueries), ("compare", ["paging", "segmentation"]))

    def test_part_covered(self):
        blk = mock.Mock(text="Paging divides memory into pages and frames.")
        self.assertTrue(rq.part_covered("paging", [blk]))
        self.assertFalse(rq.part_covered("segmentation", [blk]))


class _Hits:
    """Retriever wrapper around a VectorIndex (exposes ._index like vector_store.Retriever)."""

    def __init__(self, idx):
        self._index = idx


def run_retrieval(idx, message, **kw):
    async def go():
        steps, ev = [], None
        async for e in agentic_rag.retrieve_events(_Hits(idx), message, [], budget_s=1.0, **kw):
            if e["type"] == "step":
                steps.append(e["step"])
            else:
                ev = e["evidence"]
        return steps, ev

    return asyncio.run(go())


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.chunks = corpus()
        self.idx = index_for(self.chunks)
        for k in ("AGENTIC_RAG_SPECULATE", "AGENTIC_RAG_PARTS", "GRAPH_RAG_GLOBAL"):
            os.environ.pop(k, None)
        self.p = mock.patch.object(ge, "embed_query", fake_embed_query)
        self.p.start()
        self.addCleanup(self.p.stop)

    def blocks_text(self, ev):
        return " ".join(b.text.lower() for b in ev.blocks)

    def test_compare_searches_both_parts_in_parallel_and_keeps_both(self):
        steps, ev = run_retrieval(self.idx, "What is the difference between paging and deadlock detection?")
        titles = " ".join(s["title"] for s in steps)
        self.assertIn("parts", titles)
        self.assertEqual(ev.plan.kind, "compare")
        txt = self.blocks_text(ev)
        self.assertIn("page", txt)
        self.assertIn("deadlock", txt)

    def test_lazy_mode_skips_parts_that_are_already_covered(self):
        os.environ["AGENTIC_RAG_SPECULATE"] = "0"
        try:
            with mock.patch.object(agentic_rag, "_gather_parts", wraps=agentic_rag._gather_parts) as g:
                steps, ev = run_retrieval(self.idx, "What is the difference between paging and page table?")
            titles = " ".join(s["title"] for s in steps)
            self.assertTrue("already covered" in titles or g.called)
        finally:
            os.environ.pop("AGENTIC_RAG_SPECULATE", None)

    def test_parts_can_be_switched_off(self):
        os.environ["AGENTIC_RAG_PARTS"] = "0"
        try:
            steps, _ev = run_retrieval(self.idx, "What is the difference between paging and deadlock detection?")
        finally:
            os.environ.pop("AGENTIC_RAG_PARTS", None)
        self.assertFalse(any("parts" in s["title"] for s in steps))

    def test_global_question_uses_the_community_view_without_embeddings(self):
        with mock.patch.object(ge, "embed_query", side_effect=AssertionError("global path needs no embedding")):
            steps, ev = run_retrieval(self.idx, "Give me an overview of everything the notes cover")
        self.assertEqual(ev.plan.kind, "global")
        self.assertTrue(ev.themes)
        self.assertGreaterEqual(len(ev.blocks), 3)
        txt = self.blocks_text(ev)
        self.assertTrue(all(w in txt for w in ("deadlock", "page", "scheduling")))
        self.assertIn("themes", ev.user_message().lower())
        self.assertGreaterEqual(ev.confidence, agentic_rag.weak_conf())

    def test_related_concepts_for_followup_chips(self):
        _steps, ev = run_retrieval(self.idx, "what is deadlock detection")
        self.assertTrue(ev.related)

    def test_embeddings_down_falls_back_to_lexical_and_still_answers(self):
        with mock.patch.object(ge, "embed_query", side_effect=ge.GeminiEmbeddingError("offline")):
            _steps, ev = run_retrieval(self.idx, "What is the difference between paging and deadlock detection?")
        self.assertTrue(ev.blocks)
        txt = self.blocks_text(ev)
        self.assertIn("page", txt)
        self.assertIn("deadlock", txt)

    def test_no_llm_call_before_generation(self):
        with mock.patch("backend.llm_chain.chat", side_effect=AssertionError("no LLM before generation")):
            run_retrieval(self.idx, "Compare paging and scheduling")
            run_retrieval(self.idx, "Summarise the whole subject")

    def test_balance_parts_forces_each_part_into_the_final_set(self):
        s = rret.Searcher(_Hits(self.idx))
        la = rret.RList(s, "a", self.idx.search("paging", 6), 1.0)
        lb = rret.RList(s, "b", self.idx.search("deadlock", 6), 1.0)
        # a first pass dominated by one part...
        ret = rret.build([la, la], ["page"], k=2, expand=False)
        self.assertNotIn("deadlock", " ".join(b.text.lower() for b in ret.blocks))
        # ...is rebalanced once both parts are passed
        ret2 = rret.build([la, la, lb], ["page"], k=2, expand=False, parts=[la, lb])
        txt = " ".join(b.text.lower() for b in ret2.blocks)
        self.assertIn("deadlock", txt)
        self.assertIn("page", txt)


class StorageTests(_Env):
    """Graph persistence inside the index (disk + Mongo zip), owner isolation, version invalidation."""

    def _build(self, owner, chunks):
        self._upload("Notes", owner)
        with mock.patch.object(vs, "chunks_from_paths", lambda paths: [dict(c) for c in chunks]):
            vs.build_subject_index("Notes", owner)
        return documents.subject_internal_key("Notes", owner)

    def test_graph_is_built_in_the_same_pass_and_versioned_in_the_manifest(self):
        key = self._build("u1", corpus())
        d = vs._user_cache_dir(key)
        self.assertTrue((d / rg.GRAPH_FILE).is_file())
        man = vs.read_manifest(d)
        self.assertEqual(man["graph"]["sig"], rg.chunks_sig(corpus()))
        self.assertGreater(man["graph"]["n_concepts"], 10)
        idx = vs._get_index(key, "Notes", "u1")
        self.assertIsNotNone(idx.graph())
        self.assertEqual(idx._graph_state, "stored")

    def test_cold_instance_loads_the_graph_from_the_mongo_blob(self):
        key = self._build("u1", corpus())
        vs.clear_cache()
        import shutil

        shutil.rmtree(self.root / "cache", ignore_errors=True)
        shutil.rmtree(self.root / "documents", ignore_errors=True)
        idx = vs._get_index(key, "Notes", "u1")
        self.assertIsNotNone(idx.graph())
        self.assertEqual(idx._graph_state, "stored")           # came out of the zip, not rebuilt
        import io
        import zipfile

        names = zipfile.ZipFile(io.BytesIO(self.mongo.blobs[key][2])).namelist()
        self.assertIn("graph.json", names)

    def test_old_blob_without_graph_still_loads_and_builds_lazily(self):
        key = self._build("u1", corpus())
        d = vs._user_cache_dir(key)
        (d / rg.GRAPH_FILE).unlink()
        m = vs.read_manifest(d)
        m.pop("graph", None)
        (d / "manifest.json").write_text(json.dumps(m))
        blob = vs.pack_index(d)
        owner, version, _ = self.mongo.blobs[key]
        self.mongo.blobs[key] = (owner, version, blob)
        vs.clear_cache()
        import shutil

        shutil.rmtree(self.root / "cache", ignore_errors=True)
        idx = vs._get_index(key, "Notes", "u1")
        self.assertEqual(idx._graph_state, "unloaded")
        self.assertIsNotNone(idx.graph())
        self.assertEqual(idx._graph_state, "built")

    def test_owner_isolation_of_user_graphs(self):
        k1 = self._build("u1", corpus())
        other = {"volcano": ["Magma rises through the volcanic conduit and erupts at the crater. Volcanic eruption follows."] * 3,
                 "tectonics": ["Tectonic plates collide at a subduction zone. Subduction zone earthquakes release stored strain."] * 3}
        k2 = self._build("u2", corpus(other, source="geo.pdf"))
        self.assertNotEqual(k1, k2)
        g1 = vs._get_index(k1, "Notes", "u1").graph()
        g2 = vs._get_index(k2, "Notes", "u2").graph()
        self.assertTrue(any("deadlock" in n for n in g1.names))
        self.assertFalse(any("deadlock" in n for n in g2.names))
        self.assertTrue(any("subduction" in n for n in g2.names))
        self.assertFalse(any("subduction" in n for n in g1.names))
        with self.assertRaises(vs.VectorStoreError):
            vs._get_index(k1, "Notes", "u2")                  # u2 can never reach u1's index, hence graph
        with self.assertRaises(vs.VectorStoreError):
            vs._get_index(k1, "Notes", None)
        self.assertIsNone(self.mongo.load_index_blob(k1, "u2"))

    def test_stale_graph_file_is_ignored_and_rebuilt_from_the_chunks(self):
        key = self._build("u1", corpus())
        d = vs._user_cache_dir(key)
        wrong = rg.build_graph(corpus({"x": ["Alpha beta gamma delta epsilon. Alpha beta gamma."] * 4}))
        (d / rg.GRAPH_FILE).write_bytes(rg.dumps(wrong))        # graph of ANOTHER chunk set
        vs.clear_cache()
        idx = vs.load_index_dir(d, key, mmap=False)
        g = idx.graph()
        self.assertEqual(idx._graph_state, "built")
        self.assertTrue(any("deadlock" in n for n in g.names))

    def test_reindex_replaces_the_graph_and_never_serves_the_old_one(self):
        key = self._build("u1", corpus())
        first = vs._get_index(key, "Notes", "u1").graph()
        self.assertTrue(any("deadlock" in n for n in first.names))
        new = {"volcano": ["Magma rises through the volcanic conduit and erupts at the crater. Volcanic eruption follows."] * 3,
               "tectonics": ["Tectonic plates collide at a subduction zone. Subduction zone earthquakes release stored strain."] * 3}
        with mock.patch.object(vs, "chunks_from_paths", lambda paths: [dict(c) for c in corpus(new, source="geo.pdf")]):
            vs.build_subject_index("Notes", "u1")
        idx = vs._get_index(key, "Notes", "u1")
        g = idx.graph()
        self.assertFalse(any("deadlock" in n for n in g.names))
        self.assertTrue(any("subduction" in n for n in g.names))
        # and the stored blob carries the NEW graph
        vs.clear_cache()
        import shutil

        shutil.rmtree(self.root / "cache", ignore_errors=True)
        again = vs._get_index(key, "Notes", "u1")
        self.assertTrue(any("subduction" in n for n in again.graph().names))
        self.assertEqual(again._graph_state, "stored")

    def test_continue_indexing_keeps_a_valid_graph(self):
        key = self._build("u1", corpus())
        idx = vs._get_index(key, "Notes", "u1")
        sig = idx.manifest["graph"]["sig"]
        vs.continue_subject_index("Notes", "u1", budget_s=5.0)
        idx2 = vs._get_index(key, "Notes", "u1")
        self.assertEqual(idx2.manifest["graph"]["sig"], sig)

    def test_graph_build_can_be_switched_off_without_breaking_the_index(self):
        os.environ["GRAPH_RAG_BUILD"] = "0"
        rg._memo.clear()
        try:
            key = self._build("u1", corpus())
        finally:
            os.environ.pop("GRAPH_RAG_BUILD", None)
        d = vs._user_cache_dir(key)
        self.assertFalse((d / rg.GRAPH_FILE).exists())
        os.environ["GRAPH_RAG_BUILD"] = "0"
        try:
            self.assertIsNone(vs._get_index(key, "Notes", "u1").graph())
            r = vs.get_subject_retriever("Notes", 3, "u1")
            self.assertTrue(r.invoke("deadlock"))
        finally:
            os.environ.pop("GRAPH_RAG_BUILD", None)

    def test_bundled_indexes_ship_a_valid_graph(self):
        from backend import config

        for subject in ("Mathematics", "Operating System"):
            d = config.ROOT_DIR / "data" / "index" / subject        # the real shipped dir (not the _Env temp one)
            idx = vs.load_index_dir(d, subject)
            g = idx.graph()
            self.assertIsNotNone(g, subject)
            self.assertEqual(idx._graph_state, "stored", f"{subject}: run scripts/build_index.py --graph-only")
            self.assertGreater(len(g.communities), 3)


if __name__ == "__main__":
    unittest.main()
