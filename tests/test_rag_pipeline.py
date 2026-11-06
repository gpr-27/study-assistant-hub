"""Unit + streaming tests for the best-practice RAG pipeline (backend/rag/*, backend/agentic_rag.py).

No network and no provider: the LLM is replaced by ``llm_chain.chat`` / ``astream_chat`` fakes and
retrievers are small in-memory indexes. Covers query condensation, hybrid fusion, reranking, MMR,
per-PDF balance, neighbour expansion, confidence/abstention, citation verification, the adaptive
second pass trigger, owner isolation for cross-subject search, embedding/timeout fallbacks, chain
failure labels and the streaming turn end to end.
"""
import asyncio
import json
import time
import unittest
from unittest import mock

from backend import agentic_rag, config, llm_chain, vector_store
from backend.llm_chain import LLMChainError
from backend.rag import grounding, metrics
from backend.rag import query as rquery
from backend.rag import retrieve as rret
from backend.services import chat
from tests._study_helpers import FakeChat, chain_error, result

CHUNKS = [
    {"text": "Interrupts transfer control to the interrupt service routine. The OS saves the CPU state.",
     "source": "Lec-1.pdf", "page": 13},
    {"text": "The interrupt vector table holds the addresses of each interrupt service routine.",
     "source": "Lec-1.pdf", "page": 13},
    {"text": "Dual-mode operation separates user mode from kernel mode for protection.",
     "source": "Lec-1.pdf", "page": 30},
    {"text": "Microkernels move services out of the kernel and into user space.",
     "source": "Lec-2.pdf", "page": 42},
    {"text": "Bayes theorem gives the posterior probability of an event given evidence.",
     "source": "Lec-2.pdf", "page": 2},
]


class FakeRetriever:
    """Duck-typed retriever: ``invoke`` returns canned docs; optional delay simulates slow embeddings."""

    def __init__(self, mapping=None, delay=0.0, lexical=None):
        self.mapping = mapping or {}
        self.delay = delay
        self.lexical = lexical or {}
        self.queries = []

    def invoke(self, query):
        self.queries.append(query)
        if self.delay:
            time.sleep(self.delay)
        return list(self.mapping.get(query, []))

    def invoke_lexical(self, query):
        return list(self.lexical.get(query, self.mapping.get(query, [])))


class Doc:
    def __init__(self, text, source="Lec-1.pdf", page=0, score=None, retrieval=""):
        self.page_content = text
        self.metadata = {"source": source, "page": page}
        if score is not None:
            self.metadata.update(score=score, retrieval=retrieval)


def real_searcher(chunks=CHUNKS, label=None):
    idx = vector_store.VectorIndex("test-" + str(id(chunks)), [dict(c) for c in chunks])
    return rret.Searcher(vector_store.Retriever(index=idx, k=5), label=label)


def _cand(text, final, source="a.pdf", page=0, toks=None, searcher=None):
    c = rret.Cand(searcher or rret.Searcher(FakeRetriever()), source, page, text, Doc(text))
    c.final = final
    c.toks = frozenset(toks if toks is not None else vector_store.tokenize(text))
    return c


# ---------------------------------------------------------------- query understanding
class QueryTests(unittest.TestCase):
    def test_standalone_question_is_not_condensed(self):
        plan = rquery.plan_query("Explain the CAP theorem in distributed systems", [])
        self.assertFalse(plan.followup)
        self.assertEqual(plan.queries, ["Explain the CAP theorem in distributed systems"])

    def test_followup_is_condensed_with_previous_topic(self):
        hist = [{"role": "user", "content": "What is dual-mode operation?"},
                {"role": "assistant", "content": "It separates user and kernel mode."}]
        plan = rquery.plan_query("Why is it needed?", hist)
        self.assertTrue(plan.followup)
        self.assertIn("dual", plan.standalone)
        self.assertEqual(plan.queries[0], plan.standalone)
        self.assertIn("dual", plan.keywords)

    def test_file_hint_detected_from_question(self):
        plan = rquery.plan_query("What is in Lec-2 about booting?", [], ["Lec-1.pdf", "Lec-2.pdf"])
        self.assertEqual(plan.file_hints, ["Lec-2.pdf"])

    def test_empty_question_is_safe(self):
        plan = rquery.plan_query("", [])
        self.assertEqual(plan.queries, [])
        self.assertEqual(plan.primary, "")


# ---------------------------------------------------------------- fusion, rerank, MMR, balance, neighbours
class FusionTests(unittest.TestCase):
    def test_candidate_found_by_two_lists_outranks_single_hit(self):
        s = rret.Searcher(FakeRetriever())
        both, once = Doc("alpha beta gamma"), Doc("delta epsilon zeta")
        lists = [rret.RList(s, "q1", [both, once], 1.0), rret.RList(s, "q2", [both], 1.0)]
        cands = rret.rerank(rret._fuse(lists), ["alpha", "delta"])
        self.assertEqual(cands[0].text, "alpha beta gamma")

    def test_mmr_drops_near_duplicates(self):
        dup_a = _cand("interrupt service routine saves state", 0.9, toks=["interrupt", "service", "routine", "save", "state"])
        dup_b = _cand("interrupt service routine saves state again", 0.88,
                      toks=["interrupt", "service", "routine", "save", "state", "again"])
        other = _cand("dual mode protects kernel", 0.7, toks=["dual", "mode", "protect", "kernel"])
        picked = rret.mmr_select([dup_a, dup_b, other], k=3)
        texts = [c.text for c in picked]
        self.assertIn(dup_a.text, texts)
        self.assertIn(other.text, texts)
        self.assertNotIn(dup_b.text, texts)

    def test_balance_keeps_second_pdf_when_it_matches_strongly(self):
        shared = rret.Searcher(FakeRetriever())
        a1, a2, a3 = (_cand(f"a{i}", 0.9 - i * 0.01, source="A.pdf", searcher=shared) for i in range(3))
        b1 = _cand("b1", 0.7, source="B.pdf", searcher=shared)
        ranked = [a1, a2, a3, b1]
        out = rret.balance_sources([a1, a2, a3], ranked, 3)
        self.assertIn(b1, out)

    def test_neighbour_expansion_merges_adjacent_chunks(self):
        s = real_searcher()
        lists = rret.gather_lists([s], ["interrupt service routine"], "", lexical_only=True)
        r = rret.build(lists, ["interrupt", "service", "routine"], expand=True)
        merged = [b for b in r.blocks if b.expanded]
        self.assertTrue(merged, "adjacent interrupt chunks should merge into one block")
        self.assertIn("interrupt vector table", merged[0].text)
        self.assertIn("Interrupts transfer control", merged[0].text)
        self.assertEqual(merged[0].page, 13)

    def test_expansion_can_be_disabled(self):
        s = real_searcher()
        lists = rret.gather_lists([s], ["interrupt service routine"], "", pool=1, lexical_only=True)
        on = rret.build(lists, ["interrupt", "service", "routine"], expand=True)
        off = rret.build(lists, ["interrupt", "service", "routine"], expand=False)
        self.assertTrue(any(b.expanded for b in on.blocks), "the neighbour of the top chunk should be added")
        self.assertFalse(any(b.expanded for b in off.blocks))

    def test_join_drops_chunker_overlap(self):
        ov = "nine ten eleven twelve thirteen fourteen fifteen"
        a = "one two three four five six seven eight " + ov
        b = ov + " sixteen seventeen"
        self.assertEqual(rret._join(a, b), a + " sixteen seventeen")


# ---------------------------------------------------------------- confidence / abstention
class ConfidenceTests(unittest.TestCase):
    def _ev(self, question, retriever):
        plan = agentic_rag.plan_query(question, [], retriever)
        return agentic_rag.Evidence(question, plan, [], 0.0, "hybrid", rret.searchers_for(retriever))

    def test_strong_match_is_confident(self):
        r = FakeRetriever({"interrupt service routine": [Doc(CHUNKS[0]["text"])]})
        ret = agentic_rag._retrieve_sync(rret.searchers_for(r), rquery.plan_query("interrupt service routine"),
                                         ["interrupt service routine"], lexical_only=False, timeout_s=None)
        self.assertGreater(ret.confidence, agentic_rag.weak_conf())

    def test_off_topic_question_is_insufficient(self):
        s = real_searcher()
        plan = rquery.plan_query("Who won the football world cup in 2018?")
        ret = agentic_rag._retrieve_sync([s], plan, plan.queries, lexical_only=True, timeout_s=None)
        self.assertLess(ret.confidence, agentic_rag.weak_conf())

    def test_abstention_answer_yields_no_sources(self):
        ev = agentic_rag.Evidence("q", rquery.plan_query("q"), [], 0.0, "none", [])
        v = ev.finalize("I couldn't find this in your notes. From general knowledge: Paris.")
        self.assertTrue(v.abstained)
        self.assertEqual(v.sources, [])

    def test_is_abstain_detects_phrasings(self):
        self.assertTrue(metrics.is_abstain("I couldn't find this in your notes."))
        self.assertTrue(metrics.is_abstain("The notes do not cover that topic."))
        self.assertFalse(metrics.is_abstain("A matrix is a rectangular array [1]."))


# ---------------------------------------------------------------- citation verification
class CitationTests(unittest.TestCase):
    def test_invalid_indices_are_stripped_and_valid_kept(self):
        clean, cited, invalid = grounding.verify_citations("A holds [1] and B holds [7][2].", 3)
        self.assertEqual(clean, "A holds [1] and B holds [2].")
        self.assertEqual(cited, [1, 2])
        self.assertEqual(invalid, [7])

    def test_ranges_and_lists_expand(self):
        _clean, cited, invalid = grounding.verify_citations("See [1-3] and [2, 4].", 4)
        self.assertEqual(cited, [1, 2, 3, 4])
        self.assertEqual(invalid, [])

    def test_citations_inside_code_are_left_alone(self):
        clean, cited, _ = grounding.verify_citations("Use `a[9]` here [1].", 2)
        self.assertIn("`a[9]`", clean)
        self.assertEqual(cited, [1])

    def test_finalize_sources_match_cited_passages(self):
        blocks = [rret.Block("/x/Lec-1.pdf", 13, 13, "alpha", 0.9), rret.Block("/x/Lec-2.pdf", 4, 4, "beta", 0.5)]
        for i, b in enumerate(blocks, 1):
            b.n = i
        ev = agentic_rag.Evidence("q", rquery.plan_query("q"), blocks, 0.9, "hybrid", [])
        v = ev.finalize("Only the first one matters [1] [9].")
        self.assertEqual(v.cited, [1])
        self.assertEqual(v.invalid, [9])
        self.assertEqual(v.sources, ["Lec-1.pdf"])
        self.assertEqual([c["cited"] for c in v.citations], [True, False])
        self.assertEqual(v.citations[0]["page"], 14)

    def _ev(self, question, texts, conf):
        blocks = [rret.Block("/x/Lec-1.pdf", i, i, t, 0.9 - i * 0.1) for i, t in enumerate(texts)]
        for i, b in enumerate(blocks, 1):
            b.n = i
        return agentic_rag.Evidence(question, rquery.plan_query(question), blocks, conf, "hybrid", [])

    MATH = ["A matrix is an ordered rectangular array of numbers.", "A symmetric matrix equals its transpose.",
            "The inverse of a matrix B satisfies BA = I.", "Probability of an event lies between 0 and 1."]

    def test_topic_absent_from_notes_is_insufficient_despite_midrange_confidence(self):
        # 'eigenvalue' never occurs in the notes: dense confidence is ~0.35 (> weak 0.25) but lexical
        # coverage is 0, so the LOW_EVIDENCE prompt must be used (model says "not in your notes").
        ev = self._ev("In one sentence, what is an eigenvalue?", self.MATH, 0.35)
        self.assertEqual(ev.term_coverage, 0.0)
        self.assertTrue(ev.insufficient)
        self.assertIn("LOW", ev.system_suffix())
        self.assertIn("MUST carry at least one", ev.system_suffix())

    def test_covered_topic_stays_sufficient(self):
        ev = self._ev("In one sentence, what is a symmetric matrix?", self.MATH, 0.5)
        self.assertEqual(ev.term_coverage, 1.0)
        self.assertFalse(ev.insufficient)

    def test_general_knowledge_answer_without_markers_attaches_no_sources(self):
        ev = self._ev("what is an eigenvalue?", self.MATH, 0.35)
        v = ev.finalize("An eigenvalue is a scalar lambda with Av = lambda v.")
        self.assertTrue(v.uncited)
        self.assertEqual((v.sources, v.citations), ([], []))

    def test_confident_answer_without_markers_gets_uncited_source_chips(self):
        ev = self._ev("what is a symmetric matrix?", self.MATH, 0.9)
        v = ev.finalize("A symmetric matrix equals its transpose.")
        self.assertTrue(v.uncited)
        self.assertEqual(v.cited, [])
        self.assertEqual(v.sources, ["Lec-1.pdf"])
        self.assertEqual(len(v.citations), 3)
        self.assertFalse(any(c["cited"] for c in v.citations))

    def test_one_sentence_marker_is_kept_not_stripped(self):
        ev = self._ev("what is a symmetric matrix?", self.MATH, 0.9)
        v = ev.finalize("A symmetric matrix equals its transpose [2].")
        self.assertEqual((v.cited, v.invalid, v.uncited), ([2], [], False))
        self.assertIn("[2]", v.answer)

    def test_abstention_keeps_no_sources(self):
        ev = self._ev("what is an eigenvalue?", self.MATH, 0.35)
        v = ev.finalize("I couldn't find this in your notes.")
        self.assertTrue(v.abstained)
        self.assertEqual(v.sources, [])

    def test_unsupported_page_mentions_are_reported(self):
        self.assertEqual(grounding.unsupported_pages("See page 99 and page 14.", {14}), [99])


# ---------------------------------------------------------------- adaptive second pass
class AdaptiveTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Weak retrieval would call the (real) LLM rephrase step: keep every test off the network.
        p = mock.patch.object(agentic_rag, "llm_expand", return_value=[])
        self.expand_calls = p.start()
        self.addCleanup(p.stop)
        chat._LOCAL.clear()          # the in-process chat memory is shared across tests otherwise
        self.addCleanup(chat._LOCAL.clear)

    async def _events(self, retriever, question):
        out = []
        async for e in agentic_rag.retrieve_events(retriever, question, [], budget_s=1.0):
            out.append(e)
        return out

    async def test_strong_first_pass_makes_zero_llm_calls(self):
        r = FakeRetriever({"interrupt service routine": [Doc(CHUNKS[0]["text"], score=0.9, retrieval="hybrid")]})
        with mock.patch("backend.llm_chain.chat", side_effect=AssertionError("no LLM on the fast path")), \
                mock.patch.object(agentic_rag, "llm_expand", side_effect=AssertionError("no adaptive pass")):
            events = await self._events(r, "interrupt service routine")
        kinds = [e["step"]["kind"] for e in events if e["type"] == "step"]
        self.assertEqual(kinds, ["rewrite", "retrieve", "grade", "generate"])

    async def test_weak_first_pass_triggers_one_llm_rephrase(self):
        r = FakeRetriever({"What is the capital of Australia?": [Doc(CHUNKS[3]["text"], "Lec-2.pdf", 41)]})
        calls = []

        def fake_expand(question, standalone, hist, previous):
            calls.append(list(previous))
            return ["capital city of Australia"]

        with mock.patch.object(agentic_rag, "llm_expand", side_effect=fake_expand):
            events = await self._events(r, "What is the capital of Australia?")
        self.assertEqual(len(calls), 1)
        titles = [e["step"]["title"] for e in events if e["type"] == "step"]
        self.assertTrue(any("Weak match" in t for t in titles))

    async def test_adaptive_disabled_by_env(self):
        r = FakeRetriever({"What is the capital of Australia?": [Doc(CHUNKS[3]["text"])]})
        with mock.patch.dict("os.environ", {"AGENTIC_RAG_ADAPTIVE": "0"}), \
                mock.patch.object(agentic_rag, "llm_expand", side_effect=AssertionError("disabled")):
            await self._events(r, "What is the capital of Australia?")


# ---------------------------------------------------------------- owner isolation (cross-subject)
class CrossSubjectIsolationTests(unittest.TestCase):
    def test_owner_a_never_retrieves_owner_b_chunks(self):
        stores = {
            ("Algo", "alice"): [{"text": "ALICE-SECRET quicksort partitions the array around a pivot element.",
                                 "source": "a.pdf", "page": 0}],
            ("Algo", "bob"): [{"text": "BOB-SECRET quicksort uses a pivot too, in bob's notes.",
                               "source": "b.pdf", "page": 0}],
        }

        def fake_get(subject, k, owner=None):
            if (subject, owner) not in stores:
                raise vector_store.VectorStoreError("Subject not available.")   # mirrors _check_access
            idx = vector_store.VectorIndex(f"{owner}-{subject}", [dict(c) for c in stores[(subject, owner)]])
            return vector_store.Retriever(index=idx, k=k, subject=subject)

        def fake_list(owner=None):
            return sorted({s for (s, o) in stores if o == owner})

        with mock.patch("backend.documents.list_subjects", side_effect=fake_list), \
                mock.patch("backend.vector_store.get_subject_retriever", side_effect=fake_get):
            searchers = agentic_rag.all_subjects_searchers("alice")
            lists = rret.gather_lists(searchers, ["quicksort pivot"], "", pool=10)
            texts = " ".join(d.page_content for lst in lists for d in lst.docs)
        self.assertIn("ALICE-SECRET", texts)
        self.assertNotIn("BOB-SECRET", texts)

    def test_all_subjects_skips_unreadable_subject(self):
        def fake_get(subject, k, owner=None):
            raise vector_store.VectorStoreError("boom")

        with mock.patch("backend.documents.list_subjects", return_value=["X"]), \
                mock.patch("backend.vector_store.get_subject_retriever", side_effect=fake_get):
            self.assertEqual(agentic_rag.all_subjects_searchers("alice"), [])


# ---------------------------------------------------------------- failure paths
class FailurePathTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Weak retrieval would call the (real) LLM rephrase step: keep every test off the network.
        p = mock.patch.object(agentic_rag, "llm_expand", return_value=[])
        self.expand_calls = p.start()
        self.addCleanup(p.stop)
        chat._LOCAL.clear()          # the in-process chat memory is shared across tests otherwise
        self.addCleanup(chat._LOCAL.clear)

    async def test_embeddings_down_falls_back_to_bm25(self):
        idx = vector_store.VectorIndex("emb-down", [dict(c) for c in CHUNKS],
                                       vectors=__import__("numpy").eye(len(CHUNKS), 8, dtype="float32"),
                                       manifest={"complete": True, "dim": 8})
        r = vector_store.Retriever(index=idx, k=5)
        with mock.patch("backend.gemini_embeddings.embed_query", side_effect=RuntimeError("quota")):
            events = [e async for e in agentic_rag.retrieve_events(r, "interrupt service routine", [])]
        evidence = [e for e in events if e["type"] == "_evidence"][0]["evidence"]
        self.assertTrue(evidence.blocks)
        self.assertIn("interrupt", evidence.blocks[0].text.lower())

    async def test_slow_dense_retrieval_times_out_to_lexical(self):
        r = FakeRetriever({"q": [Doc("dense hit")]}, delay=0.6, lexical={"q": [Doc("bm25 hit")]})
        t0 = time.monotonic()
        result_ = await agentic_rag._retrieve_hybrid(rret.searchers_for(r), rquery.plan_query("q"), ["q"], 0.05)
        self.assertLess(time.monotonic() - t0, 0.5)
        self.assertEqual(result_.mode, "lexical")
        self.assertEqual(result_.blocks[0].text, "bm25 hit")
        await asyncio.sleep(0.7)

    async def test_chain_down_after_retrieval_gives_labelled_error(self):
        r = FakeRetriever({"interrupt service routine": [Doc(CHUNKS[0]["text"], score=0.9, retrieval="hybrid")]})

        async def dead_stream(system, messages, **kw):
            raise chain_error()
            yield  # pragma: no cover

        session = chat.Session("id", "alice", "subject", "L", "Algo", None, None, "n")
        with mock.patch.object(chat, "_prepare", lambda s: (r, "P")), \
                mock.patch("backend.llm_chain.astream_chat", dead_stream), \
                mock.patch.object(config, "PERSISTENCE_ENABLED", False):
            events = [e async for e in chat.stream_turn(session, "interrupt service routine")]
        self.assertEqual(events[-1]["type"], "error")
        self.assertEqual(events[-1]["llm"], {"label": "Model: none (all providers failed)"})
        self.assertFalse([e for e in events if e["type"] == "done"])


# ---------------------------------------------------------------- streaming integration
class StreamingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Weak retrieval would call the (real) LLM rephrase step: keep every test off the network.
        p = mock.patch.object(agentic_rag, "llm_expand", return_value=[])
        self.expand_calls = p.start()
        self.addCleanup(p.stop)
        chat._LOCAL.clear()          # the in-process chat memory is shared across tests otherwise
        self.addCleanup(chat._LOCAL.clear)

    async def _run(self, retriever, question, reply_text, hist=None):
        async def fake_stream(system, messages, **kw):
            fake_stream.calls.append({"system": system, "messages": messages})
            for piece in [reply_text[:len(reply_text) // 2], reply_text[len(reply_text) // 2:]]:
                yield piece
            yield llm_chain.StreamStart("groq", "m", 2, "Groq key 1")
            yield result(reply_text)

        fake_stream.calls = []
        session = chat.Session("id", "alice", "subject", "L", "Algo", None, None, "n")
        with mock.patch.object(chat, "_prepare", lambda s: (retriever, "PERSONA")), \
                mock.patch("backend.llm_chain.astream_chat", fake_stream), \
                mock.patch.object(config, "PERSISTENCE_ENABLED", False):
            events = [e async for e in chat.stream_turn(session, question, client_history=hist)]
        return events, fake_stream.calls

    async def test_done_carries_cited_sources_and_citation_chips(self):
        r = FakeRetriever({"interrupt service routine": [Doc(CHUNKS[0]["text"], "/d/Lec-1.pdf", 13, 0.9, "hybrid"),
                                                          Doc(CHUNKS[3]["text"], "/d/Lec-2.pdf", 42, 0.2, "hybrid")]})
        events, calls = await self._run(r, "interrupt service routine",
                                        "Interrupts transfer control to the service routine [1] and [9].")
        kinds = [e["step"]["kind"] for e in events if e["type"] == "step"]
        self.assertEqual(kinds, ["rewrite", "retrieve", "grade", "generate"])
        done = events[-1]
        self.assertEqual(done["type"], "done")
        self.assertNotIn("[9]", done["answer"])
        self.assertEqual(done["sources"], ["Lec-1.pdf"])
        self.assertEqual([c["cited"] for c in done["citations"]], [True, False])
        self.assertEqual(done["citations"][0]["file"], "Lec-1.pdf")
        self.assertEqual(done["citations"][0]["page"], 14)
        self.assertIn("[1] Lec-1.pdf p.14", calls[0]["messages"][-1]["content"])
        self.assertIn("GROUNDING RULES", calls[0]["system"])

    async def test_abstention_turn_has_no_sources(self):
        r = FakeRetriever({"who won the world cup": [Doc(CHUNKS[4]["text"], "/d/Lec-2.pdf", 2)]})
        events, _ = await self._run(r, "who won the world cup",
                                    "I couldn't find this in your notes. From general knowledge: France.")
        done = events[-1]
        self.assertEqual(done["sources"], [])
        self.assertEqual([c["cited"] for c in done["citations"]], [False])

    async def test_followup_turn_uses_condensed_query(self):
        r = FakeRetriever({})
        hist = [{"role": "user", "content": "What is dual-mode operation?"},
                {"role": "assistant", "content": "It separates user and kernel mode."}]
        await self._run(r, "Why is it needed?", "Protection [1].", hist=hist)
        self.assertTrue(any("dual" in q for q in r.queries))

    async def test_search_tag_triggers_one_more_hop(self):
        r = FakeRetriever({"interrupt service routine": [Doc(CHUNKS[0]["text"], "/d/Lec-1.pdf", 13, 0.9, "hybrid")],
                           "interrupt vector table": [Doc(CHUNKS[1]["text"], "/d/Lec-1.pdf", 13, 0.8, "hybrid")]})
        ev = None
        async for e in agentic_rag.retrieve_events(r, "interrupt service routine", []):
            if e["type"] == "_evidence":
                ev = e["evidence"]
        ev.hops_left = 1
        seen = []

        async def gen(system, msgs):
            seen.append(len(msgs))
            text = "<search>interrupt vector table</search>" if len(seen) == 1 else "Vectors list the handlers [1][2]."
            yield {"type": "token", "text": text}
            yield {"type": "_final", "result": result("ok")}

        out = [e async for e in agentic_rag.generate_with_hops(ev, gen, lambda e: ("SYS", [{"role": "user", "content": e.user_message()}]))]
        text = "".join(e["text"] for e in out if e["type"] == "token")
        self.assertEqual(text, "Vectors list the handlers [1][2].")
        self.assertEqual(len(seen), 2)
        self.assertTrue(any(e["type"] == "step" and e["step"]["kind"] == "retrieve" for e in out))
        self.assertEqual(len(ev.blocks), 2)


class ToolQueryTests(unittest.TestCase):
    def test_grounding_rule_offers_general_knowledge_marker(self):
        self.assertIn(grounding.GENERAL_MARK, grounding.RULES)

    def test_search_hint_only_while_hops_remain(self):
        self.assertIn("<search>", grounding.system_suffix(insufficient=False, hops_left=1))
        self.assertNotIn("<search>", grounding.system_suffix(insufficient=False, hops_left=0))


if __name__ == "__main__":
    unittest.main()
