"""Tests for the chat overview service (NotebookLM-style empty-state card).

No network/provider: ``llm_chain.chat`` is mocked and the document helpers are mocked. Covers the
request-path rules: pre-baked overviews are pure file reads that NEVER touch the LLM, a bounded
cache serves repeats, generation is single-flight / bounded / throttled, failures are cached only
briefly, and the endpoint is owner-scoped.
"""
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from backend.services import overview
from tests._study_helpers import FakeChat, chain_error


def _chat(*replies):
    fake = FakeChat(*replies)
    return mock.patch("backend.llm_chain.chat", fake), fake


def _patch_docs(names=("a.pdf",), text="content"):
    return (mock.patch.object(overview, "list_pdf_names", return_value=list(names)),
            mock.patch.object(overview, "read_pdf", return_value=text),
            mock.patch.object(overview, "pdf_signature", return_value=(1.0, 2)))


class _Base(unittest.TestCase):
    def setUp(self):
        overview._cache.clear()
        overview._last_gen.clear()
        overview._inflight.clear()


class BuildOverviewTests(_Base):
    def test_parses_overview_and_questions_from_subject(self):
        ctx, fake = _chat('{"overview": "Covers **trees** and **graphs**.", '
                          '"questions": ["What is a BST?", "How does DFS work?"]}')
        p1, p2, p3 = _patch_docs(["a.pdf", "b.pdf"], "some content")
        with ctx, p1, p2, p3:
            card = overview.build_overview(subject="DSA", owner="u1")
        self.assertEqual(card["label"], "DSA")
        self.assertEqual(card["sources"], 2)
        self.assertEqual(card["overview"], "Covers **trees** and **graphs**.")
        self.assertEqual(card["questions"], ["What is a BST?", "How does DFS work?"])
        self.assertTrue(card["llm"]["label"].startswith("Model: groq"))
        self.assertEqual(len(fake.calls), 1)
        self.assertTrue(fake.calls[0]["json_mode"])
        self.assertLessEqual(fake.calls[0]["deadline_s"], 20)           # generation is bounded

    def test_clamps_questions_to_three(self):
        ctx, _ = _chat('{"overview": "o", "questions": ["q1", "q2", "q3", "q4", "q5"]}')
        p1, p2, p3 = _patch_docs()
        with ctx, p1, p2, p3:
            card = overview.build_overview(subject="DSA", owner="u1")
        self.assertEqual(card["questions"], ["q1", "q2", "q3"])

    def test_topic_mode_has_zero_sources_and_skips_pdfs(self):
        ctx, _ = _chat('{"overview": "About **recursion**.", "questions": ["q1"]}')
        with ctx, mock.patch.object(overview, "list_pdf_names") as names, \
                mock.patch.object(overview, "read_pdf") as read:
            card = overview.build_overview(topic="Recursion", owner="u1")
        self.assertEqual((card["label"], card["sources"]), ("Recursion", 0))
        names.assert_not_called()
        read.assert_not_called()

    def test_second_call_is_served_from_cache(self):
        ctx, fake = _chat('{"overview": "cached", "questions": ["q1"]}')
        p1, p2, p3 = _patch_docs()
        with ctx, p1, p2, p3:
            first = overview.build_overview(subject="DSA", owner="u1")
            second = overview.build_overview(subject="DSA", owner="u1")
        self.assertEqual(first, second)
        self.assertEqual(len(fake.calls), 1)

    def test_cache_is_per_owner(self):
        ctx, fake = _chat('{"overview": "o", "questions": ["q1"]}')
        p1, p2, p3 = _patch_docs()
        with ctx, p1, p2, p3:
            overview.build_overview(subject="DSA", owner="alice")
            overview.build_overview(subject="DSA", owner="bob")
        self.assertEqual(len(fake.calls), 2)

    def test_refresh_bypasses_cache_but_is_throttled(self):
        ctx, fake = _chat('{"overview": "o", "questions": ["q1"]}')
        p1, p2, p3 = _patch_docs()
        with ctx, p1, p2, p3:
            overview.build_overview(subject="DSA", owner="u1")
            overview.build_overview(subject="DSA", owner="u1", refresh=True)     # too soon → cached card
            self.assertEqual(len(fake.calls), 1)
            for k in list(overview._last_gen):
                overview._last_gen[k] -= 1000
            overview.build_overview(subject="DSA", owner="u1", refresh=True)
        self.assertEqual(len(fake.calls), 2)

    def test_cache_is_bounded(self):
        ctx, _ = _chat('{"overview": "o", "questions": ["q1"]}')
        with ctx, mock.patch.object(overview, "_MAX_CACHE", 5):
            for i in range(12):
                overview.build_overview(topic=f"t{i}", owner="u1")
        self.assertLessEqual(len(overview._cache), 5)

    def test_degrades_to_empty_when_chain_fails_and_never_raises(self):
        ctx, _ = _chat(chain_error())
        p1, p2, p3 = _patch_docs(["a.pdf", "b.pdf"])
        with ctx, p1, p2, p3:
            card = overview.build_overview(subject="DSA", owner="u1")
        self.assertEqual((card["overview"], card["questions"], card["llm"]), ("", [], None))
        self.assertEqual(card["sources"], 2)

    def test_failure_is_negative_cached_briefly_then_retried(self):
        ctx, fake = _chat(chain_error())
        p1, p2, p3 = _patch_docs()
        with ctx, p1, p2, p3:
            overview.build_overview(subject="DSA", owner="u1")
            overview.build_overview(subject="DSA", owner="u1")       # inside the failure TTL: no new LLM call
            self.assertEqual(len(fake.calls), 1)
            for key, (exp, card) in list(overview._cache.items()):
                overview._cache[key] = (0.0, card)                   # failure TTL elapsed
            overview.build_overview(subject="DSA", owner="u1")
        self.assertEqual(len(fake.calls), 2)

    def test_concurrent_requests_share_one_generation(self):
        gate = threading.Event()
        calls = []

        def slow_chat(system, messages, **kw):
            calls.append(1)
            gate.wait(2)
            from tests._study_helpers import result
            return result('{"overview": "o", "questions": ["q1"]}')

        out = []
        p1, p2, p3 = _patch_docs()
        with mock.patch("backend.llm_chain.chat", slow_chat), p1, p2, p3:
            ts = [threading.Thread(target=lambda: out.append(overview.build_overview(topic="X", owner="u1")))
                  for _ in range(4)]
            for t in ts:
                t.start()
            import time
            time.sleep(0.2)
            gate.set()
            for t in ts:
                t.join(5)
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(out), 4)
        self.assertTrue(all(c["overview"] == "o" for c in out))

    def test_stuck_generation_always_releases_the_inflight_slot(self):
        ctx, _ = _chat(RuntimeError("boom"))
        with ctx:
            overview.build_overview(topic="X", owner="u1")
        self.assertEqual(overview._inflight, {})


class PrebakedTests(_Base):
    def _bundle(self, tmp, name, data):
        d = Path(tmp) / name
        d.mkdir(parents=True)
        (d / "overview.json").write_text(json.dumps(data), encoding="utf-8")

    def test_prebaked_is_served_without_any_llm_call_or_pdf_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._bundle(tmp, "Mathematics",
                         {"overview": "**Calculus** and algebra.", "questions": ["q1", "q2", "q3", "q4"], "sources": 7})
            ctx, fake = _chat("SHOULD NOT BE CALLED")
            with ctx, mock.patch.object(overview, "_bundled_index_dir", return_value=Path(tmp)), \
                    mock.patch.object(overview, "subject_internal_key", return_value="Mathematics"), \
                    mock.patch.object(overview, "read_pdf") as read:
                card = overview.build_overview(subject="Mathematics", owner="u1")
        self.assertEqual(fake.calls, [])
        read.assert_not_called()
        self.assertEqual(card["sources"], 7)
        self.assertEqual(card["questions"], ["q1", "q2", "q3"])
        self.assertEqual(card["llm"], {"label": overview.PREBAKED_UNKNOWN_LABEL})
        self.assertTrue(card["llm"]["label"].startswith("Model:"))

    def test_prebaked_keeps_the_stored_model_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._bundle(tmp, "Mathematics", {"overview": "x", "questions": ["q"], "sources": 1,
                                              "llm": "Model: groq · openai/gpt-oss-120b · step 2 (Groq key 1)"})
            with mock.patch.object(overview, "_bundled_index_dir", return_value=Path(tmp)), \
                    mock.patch.object(overview, "subject_internal_key", return_value="Mathematics"):
                card = overview.build_overview(subject="Mathematics", owner="u1")
        self.assertEqual(card["llm"]["label"], "Model: groq · openai/gpt-oss-120b · step 2 (Groq key 1)")

    def test_user_owned_subject_never_reads_a_bundled_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._bundle(tmp, "Mathematics", {"overview": "bundled", "questions": ["q"], "sources": 1})
            with mock.patch.object(overview, "_bundled_index_dir", return_value=Path(tmp)):
                # real subject_internal_key: a non-bundled name for an owner maps to user__<owner>__<name>
                self.assertIsNone(overview._load_prebaked("SomeoneElsesSubject", "u1"))

    def test_missing_or_malformed_prebake_falls_back_to_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / "Math"
            d.mkdir()
            (d / "overview.json").write_text("{not json", encoding="utf-8")
            ctx, fake = _chat('{"overview": "live", "questions": ["q"]}')
            p1, p2, p3 = _patch_docs()
            with ctx, p1, p2, p3, mock.patch.object(overview, "_bundled_index_dir", return_value=Path(tmp)), \
                    mock.patch.object(overview, "subject_internal_key", return_value="Math"):
                card = overview.build_overview(subject="Math", owner="u1")
        self.assertEqual(card["overview"], "live")
        self.assertEqual(len(fake.calls), 1)


class OverviewEndpointTests(unittest.TestCase):
    """GET /api/chat/overview/{session_id}: shape, owner scoping and unknown-session error."""

    def setUp(self):
        from backend.routers import chat as chat_router
        from tests._study_helpers import make_client
        self.alice = make_client(chat_router.router, owner="alice")
        self.bob = make_client(chat_router.router, owner="bob")

    def _sid(self, owner="alice"):
        from backend.services import chat
        return chat._mint(owner, "subject", "Subject · DSA", "DSA", None, None).id

    def test_overview_returns_expected_shape_with_llm(self):
        card = {"label": "DSA", "sources": 2, "overview": "Covers **trees**.",
                "questions": ["What is a BST?"], "llm": {"label": "Model: groq · x"}}
        with mock.patch.object(overview, "build_overview", return_value=card) as build:
            resp = self.alice.get(f"/api/chat/overview/{self._sid()}")
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual((body["label"], body["sources"]), ("DSA", 2))
        self.assertEqual(body["llm"]["label"], "Model: groq · x")
        build.assert_called_once_with(subject="DSA", topic=None, owner="alice", refresh=False)

    def test_other_users_session_is_404(self):
        with mock.patch.object(overview, "build_overview") as build:
            resp = self.bob.get(f"/api/chat/overview/{self._sid('alice')}")
        self.assertEqual(resp.status_code, 404)
        build.assert_not_called()

    def test_unknown_session_returns_404(self):
        self.assertEqual(self.alice.get("/api/chat/overview/does-not-exist").status_code, 404)


if __name__ == "__main__":
    unittest.main()
