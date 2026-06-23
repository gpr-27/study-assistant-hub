"""Tests for the agentic RAG retrieval loop (rewrite → retrieve → grade → correct).

No network or real Groq: a FakeLLM returns queued JSON/text replies and a FakeRetriever
returns canned documents. Covers JSON extraction, query rewriting (+fallbacks), retrieval
merge/de-dupe, grading, the corrective re-retrieve loop, round bounding, rate-limit
propagation, and the chat service choosing agentic vs classic by config.
"""
import json
import unittest
from unittest import mock

import httpx
from groq import RateLimitError

from backend import agentic_rag, config


_REQUEST = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")


def _rate_limit() -> RateLimitError:
    return RateLimitError("rate limited", response=httpx.Response(429, request=_REQUEST), body=None)


class _Doc:
    def __init__(self, text):
        self.page_content = text


class _Reply:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    """Returns queued replies in order; records the messages it was called with."""

    def __init__(self, replies):
        self._replies = list(replies)
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        if not self._replies:
            return _Reply("")
        nxt = self._replies.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return _Reply(nxt)


class FakeRetriever:
    """Maps a query string -> list[_Doc]; unknown queries return []."""

    def __init__(self, mapping):
        self.mapping = mapping
        self.queries = []

    def invoke(self, query):
        self.queries.append(query)
        return self.mapping.get(query, [])


class JsonExtractionTests(unittest.TestCase):
    def test_plain_json(self):
        self.assertEqual(agentic_rag._extract_json('{"a": 1}'), {"a": 1})

    def test_json_embedded_in_prose_and_fences(self):
        text = "Sure! ```json\n{\"queries\": [\"x\"]}\n``` done"
        self.assertEqual(agentic_rag._extract_json(text), {"queries": ["x"]})

    def test_garbage_returns_empty(self):
        self.assertEqual(agentic_rag._extract_json("no json here"), {})


class RewriteTests(unittest.TestCase):
    def test_parses_and_caps_queries(self):
        llm = FakeLLM(['{"queries": ["a", "b", "c", "d", "e"]}'])
        queries, prompt = agentic_rag._rewrite(llm, "q", [])
        self.assertEqual(queries, ["a", "b", "c"][: config.AGENTIC_RAG_QUERIES])
        self.assertIn("search queries", prompt)

    def test_falls_back_to_question_on_bad_json(self):
        llm = FakeLLM(["not json"])
        queries, _prompt = agentic_rag._rewrite(llm, "what is X?", [])
        self.assertEqual(queries, ["what is X?"])

    def test_rate_limit_propagates(self):
        llm = FakeLLM([_rate_limit()])
        with self.assertRaises(RateLimitError):
            agentic_rag._rewrite(llm, "q", [])


class RetrieveTests(unittest.TestCase):
    def test_merges_and_dedupes(self):
        d1, d2 = _Doc("alpha"), _Doc("beta")
        r = FakeRetriever({"q1": [d1, d2], "q2": [_Doc("alpha"), _Doc("gamma")]})
        docs = agentic_rag._retrieve(r, ["q1", "q2"])
        self.assertEqual([d.page_content for d in docs], ["alpha", "beta", "gamma"])


class GradeTests(unittest.TestCase):
    def test_sufficient_true(self):
        llm = FakeLLM(['{"sufficient": true, "missing": ""}'])
        ok, missing, _prompt = agentic_rag._grade(llm, "q", "some context")
        self.assertTrue(ok)

    def test_insufficient_with_missing(self):
        llm = FakeLLM(['{"sufficient": false, "missing": "definition of X"}'])
        ok, missing, _prompt = agentic_rag._grade(llm, "q", "ctx")
        self.assertFalse(ok)
        self.assertEqual(missing, "definition of X")

    def test_empty_context_is_insufficient_without_calling_llm(self):
        llm = FakeLLM([])  # must not be called
        ok, _missing, _prompt = agentic_rag._grade(llm, "q", "   ")
        self.assertFalse(ok)
        self.assertEqual(llm.calls, [])

    def test_unparseable_grade_assumes_sufficient(self):
        llm = FakeLLM(["???"])
        ok, _missing, _prompt = agentic_rag._grade(llm, "q", "ctx")
        self.assertTrue(ok)  # degrade to naive RAG rather than loop forever


class AnswerLoopTests(unittest.TestCase):
    def test_stops_when_first_grade_sufficient(self):
        retriever = FakeRetriever({"q1": [_Doc("relevant ctx")]})
        llm = FakeLLM([
            '{"queries": ["q1"]}',          # rewrite
            '{"sufficient": true}',          # grade -> stop
            "FINAL ANSWER",                  # generate
        ])
        text, steps, sources = agentic_rag.answer(retriever, llm, "question?", [], "persona", max_iters=2)
        self.assertEqual(text, "FINAL ANSWER")
        self.assertEqual(retriever.queries, ["q1"])     # only one retrieval round
        self.assertEqual(len(llm.calls), 3)
        # trace covers the full path: rewrite → retrieve → grade → generate
        self.assertEqual([s["kind"] for s in steps], ["rewrite", "retrieve", "grade", "generate"])
        self.assertEqual(steps[2]["verdict"], "sufficient")

    def test_corrective_second_round_then_generate(self):
        retriever = FakeRetriever({"q1": [_Doc("weak")], "q2": [_Doc("strong extra")]})
        llm = FakeLLM([
            '{"queries": ["q1"]}',                       # rewrite #1
            '{"sufficient": false, "missing": "more"}',   # grade #1 -> correct
            '{"queries": ["q2"]}',                       # rewrite #2 (corrective)
            "ANSWER WITH BOTH",                          # generate (grade #2 skipped: last round)
        ])
        text, steps, sources = agentic_rag.answer(retriever, llm, "question?", [], "persona", max_iters=2)
        self.assertEqual(text, "ANSWER WITH BOTH")
        self.assertEqual(retriever.queries, ["q1", "q2"])
        # final generation context should include both retrieved docs
        gen_msgs = llm.calls[-1]
        self.assertIn("weak", gen_msgs[-1].content)
        self.assertIn("strong extra", gen_msgs[-1].content)
        # the corrective round is visible in the trace; the last round skips
        # grading (no corrective round can follow) → "final" not "sufficient".
        verdicts = [s["verdict"] for s in steps if s["kind"] == "grade"]
        self.assertEqual(verdicts, ["insufficient", "final"])

    def test_round_cap_respected(self):
        retriever = FakeRetriever({"q": [_Doc("ctx")]})
        # With max_iters=1 the only round is also the last, so grading is skipped
        # (it would be a wasted LLM call): retrieve once, then generate.
        llm = FakeLLM([
            '{"queries": ["q"]}',
            "ANSWER ANYWAY",
        ])
        text, steps, sources = agentic_rag.answer(retriever, llm, "q?", [], "persona", max_iters=1)
        self.assertEqual(text, "ANSWER ANYWAY")
        self.assertEqual(retriever.queries, ["q"])  # no second round
        self.assertEqual([s["verdict"] for s in steps if s["kind"] == "grade"], ["final"])
        self.assertEqual(len(llm.calls), 2)  # rewrite + generate only — no grade call


class OnStepStreamingTests(unittest.TestCase):
    """answer() invokes on_step for EVERY step, in order, as each is appended —
    including the generate step BEFORE the slow final LLM call, so a streaming
    caller can show progress. The callback default (no-op) keeps the return value
    and behaviour identical for non-streaming callers."""

    def test_on_step_fires_for_each_step_in_order(self):
        retriever = FakeRetriever({"q1": [_Doc("relevant ctx")]})
        llm = FakeLLM([
            '{"queries": ["q1"]}',          # rewrite
            '{"sufficient": true}',          # grade -> stop
            "FINAL ANSWER",                  # generate
        ])
        seen: list[dict] = []
        text, steps, _sources = agentic_rag.answer(
            retriever, llm, "question?", [], "persona", max_iters=2, on_step=seen.append
        )
        # The callback saw exactly the same step dicts, in the same order, as the
        # returned trace (rewrite → retrieve → grade → generate).
        self.assertEqual([s["kind"] for s in seen], ["rewrite", "retrieve", "grade", "generate"])
        self.assertEqual(seen, steps)
        self.assertEqual(text, "FINAL ANSWER")

    def test_on_step_for_generate_fires_before_the_final_llm_call(self):
        """The generate step must be emitted BEFORE the (slow) final generation, so
        a streaming UI can show "Answering…" while the answer is being written."""
        retriever = FakeRetriever({"q": [_Doc("ctx")]})
        llm = FakeLLM(['{"queries": ["q"]}', "ANSWER"])  # max_iters=1 skips grading
        order: list[str] = []

        def record(step):
            if step["kind"] == "generate":
                # at this point the model has only been called for rewrite — the
                # final generate invoke has NOT happened yet.
                order.append(f"emit-generate@{len(llm.calls)}")

        agentic_rag.answer(retriever, llm, "q?", [], "persona", max_iters=1, on_step=record)
        # rewrite was call #1; generate was emitted while still at 1 call (before #2).
        self.assertEqual(order, ["emit-generate@1"])
        self.assertEqual(len(llm.calls), 2)  # rewrite + generate

    def test_default_on_step_is_noop_and_behaviour_unchanged(self):
        retriever = FakeRetriever({"q1": [_Doc("relevant ctx")]})
        llm = FakeLLM(['{"queries": ["q1"]}', '{"sufficient": true}', "FINAL ANSWER"])
        # No on_step passed → must not raise and must behave exactly as before.
        text, steps, sources = agentic_rag.answer(retriever, llm, "question?", [], "persona", max_iters=2)
        self.assertEqual(text, "FINAL ANSWER")
        self.assertEqual([s["kind"] for s in steps], ["rewrite", "retrieve", "grade", "generate"])


class ChatServiceWiringTests(unittest.TestCase):
    """chat.ask() routes to agentic vs classic RAG based on config.AGENTIC_RAG."""

    def _session(self, retriever):
        from backend.services import chat
        sid = chat._register({
            "mode": "subject", "retriever": retriever, "llm": object(), "history": [],
            "label": "L", "topic": None, "subject": "Bio", "owner": "local", "chat_id": None,
        })
        return chat, sid

    def test_uses_agentic_when_enabled(self):
        from backend.services import chat
        retriever = FakeRetriever({})
        chat_mod, sid = self._session(retriever)
        with mock.patch.object(chat_mod.config, "AGENTIC_RAG", True), \
             mock.patch.object(chat_mod, "require_groq_key", lambda: None), \
             mock.patch.object(chat_mod.agentic_rag, "answer", return_value=("AGENTIC", [], [])) as ag:
            out = chat_mod.ask(sid, "What is mitosis?")
        self.assertEqual(out["answer"], "AGENTIC")
        ag.assert_called_once()

    def test_uses_classic_when_disabled(self):
        from backend.services import chat
        retriever = FakeRetriever({"What is mitosis?": [_Doc("ctx")]})
        chat_mod, sid = self._session(retriever)
        # Replace the session llm with a fake so the single-shot path returns text.
        chat_mod._sessions[sid]["llm"] = FakeLLM(["CLASSIC"])
        with mock.patch.object(chat_mod.config, "AGENTIC_RAG", False), \
             mock.patch.object(chat_mod, "require_groq_key", lambda: None), \
             mock.patch.object(chat_mod.agentic_rag, "answer", side_effect=AssertionError("should not be agentic")):
            out = chat_mod.ask(sid, "What is mitosis?")
        self.assertEqual(out["answer"], "CLASSIC")
        self.assertEqual(retriever.queries, ["What is mitosis?"])  # one direct retrieval

    def test_smalltalk_skips_retrieval(self):
        """A greeting answers in one LLM call — no embedding, rewrite, or retrieval."""
        from backend.services import chat
        retriever = FakeRetriever({})
        chat_mod, sid = self._session(retriever)
        chat_mod._sessions[sid]["llm"] = FakeLLM(["Hi! Ask me about your notes."])
        with mock.patch.object(chat_mod.config, "AGENTIC_RAG", True), \
             mock.patch.object(chat_mod, "require_groq_key", lambda: None), \
             mock.patch.object(chat_mod.agentic_rag, "answer",
                               side_effect=AssertionError("small talk must not retrieve")):
            out = chat_mod.ask(sid, "hi hlo")
        self.assertEqual(out["answer"], "Hi! Ask me about your notes.")
        self.assertEqual(retriever.queries, [])  # no retrieval for small talk


class RecentArtifactContextTests(unittest.TestCase):
    """A follow-up (non-tool, non-smalltalk) turn must fold a just-created
    artifact's text into the model context so the agent can discuss it."""

    def _session(self, retriever, **over):
        from backend.services import chat
        base = {
            "mode": "subject", "retriever": retriever, "llm": object(), "history": [],
            "label": "L", "topic": None, "subject": "Bio", "owner": "local", "chat_id": None,
            "autonomous": True,
        }
        base.update(over)
        return chat, chat._register(base)

    def test_agentic_turn_injects_artifact_text_into_persona(self):
        from backend.services import chat, chat_tools
        retriever = FakeRetriever({})
        chat_mod, sid = self._session(retriever)
        chat_mod._sessions[sid]["recent_artifact"] = {
            "title": "Quiz · Bio", "kind": "quizzes",
            "text": "Q3. What is mitosis?\n  A. cell division\nCorrect answer: A. cell division",
        }
        captured = {}

        def fake_answer(retr, llm, msg, hist, persona, *a, **k):
            captured["persona"] = persona
            return ("ANSWER", [], [])

        with mock.patch.object(chat_mod.config, "AGENTIC_RAG", True), \
             mock.patch.object(chat_mod, "require_groq_key", lambda: None), \
             mock.patch.object(chat_tools, "detect_intent", return_value=None), \
             mock.patch.object(chat_mod.agentic_rag, "answer", side_effect=fake_answer):
            out = chat_mod.ask(sid, "explain question 3")
        self.assertEqual(out["answer"], "ANSWER")
        self.assertIn("The student just created Quiz · Bio", captured["persona"])
        self.assertIn("What is mitosis?", captured["persona"])
        self.assertIn("Correct answer: A. cell division", captured["persona"])

    def test_topic_turn_injects_artifact_text_into_system_message(self):
        from backend.services import chat, chat_tools
        chat_mod, sid = self._session(None, mode="topic", subject=None, topic="Recursion")
        chat_mod._sessions[sid]["llm"] = FakeLLM(["ANSWER ABOUT THE CARDS"])
        chat_mod._sessions[sid]["recent_artifact"] = {
            "title": "Flashcards · Recursion", "kind": "flashcards",
            "text": "Front: base case\nBack: stops the recursion",
        }
        with mock.patch.object(chat_mod, "require_groq_key", lambda: None), \
             mock.patch.object(chat_tools, "detect_intent", return_value=None):
            out = chat_mod.ask(sid, "summarise what these cards teach me")
        self.assertEqual(out["answer"], "ANSWER ABOUT THE CARDS")
        system_msg = chat_mod._sessions[sid]["llm"].calls[-1][0].content
        self.assertIn("The student just created Flashcards · Recursion", system_msg)
        self.assertIn("stops the recursion", system_msg)


class AskStreamTests(unittest.TestCase):
    """chat.ask_stream() yields a {"type":"step"} event per trace step as it happens,
    then a terminal {"type":"done"} carrying the answer/sources/artifact — for both the
    autonomous TOOL path and the AGENTIC RAG path."""

    def _session(self, retriever, **over):
        from backend.services import chat
        base = {
            "mode": "subject", "retriever": retriever, "llm": object(), "history": [],
            "label": "L", "topic": None, "subject": "Bio", "owner": "local", "chat_id": None,
            "autonomous": True,
        }
        base.update(over)
        return chat, chat._register(base)

    def test_tool_path_streams_tool_step_then_done_with_artifact(self):
        from backend.services import chat, chat_tools
        chat_mod, sid = self._session(FakeRetriever({}))
        art = {"kind": "quizzes", "id": "x", "title": "Quiz · Bio", "route": "#quiz", "tool": "Quiz Generator"}
        tool_result = {
            "answer": "Built your quiz.",
            "steps": [{"kind": "tool", "title": "Generated 5 questions"}],
            "artifact": art,
        }
        with mock.patch.object(chat_mod, "require_groq_key", lambda: None), \
             mock.patch.object(chat_tools, "detect_intent", return_value=[{"tool": "quiz"}]), \
             mock.patch.object(chat_tools, "run_tool", return_value=tool_result):
            events = list(chat_mod.ask_stream(sid, "create a quiz"))

        # First a "Calling <tool>…" step is announced BEFORE the build, then the
        # tool's own result steps, then a terminal done carrying the artifact.
        self.assertEqual(events[0]["type"], "step")
        self.assertEqual(events[0]["step"]["kind"], "tool")
        self.assertEqual(events[0]["step"]["tool"], "quiz")
        self.assertIn("quiz", events[0]["step"]["title"])
        # the tool's own row is tagged kind="tool" and streamed
        self.assertTrue(any(e["type"] == "step" and e["step"].get("title") == "Generated 5 questions"
                            and e["step"]["kind"] == "tool" for e in events))
        done = events[-1]
        self.assertEqual(done["type"], "done")
        self.assertEqual(done["answer"], "Built your quiz.")
        self.assertEqual(done["artifact"], art)
        self.assertEqual(done["artifacts"], [art])   # new multi-artifact field carries every build
        # history was recorded for the turn
        self.assertEqual(chat_mod._sessions[sid]["history"][-1], ("create a quiz", "Built your quiz."))

    def test_agentic_path_streams_steps_then_done(self):
        from backend.services import chat, chat_tools
        retriever = FakeRetriever({"q1": [_Doc("ctx about cells")]})
        chat_mod, sid = self._session(retriever)
        chat_mod._sessions[sid]["llm"] = FakeLLM([
            '{"queries": ["q1"]}',   # rewrite
            '{"sufficient": true}',   # grade -> stop
            "Cells are the unit of life.",   # generate
        ])
        with mock.patch.object(chat_mod, "require_groq_key", lambda: None), \
             mock.patch.object(chat_mod.config, "AGENTIC_RAG", True), \
             mock.patch.object(chat_tools, "detect_intent", return_value=None):
            events = list(chat_mod.ask_stream(sid, "What is a cell?"))

        step_kinds = [e["step"]["kind"] for e in events if e["type"] == "step"]
        # the real agentic loop streamed live: rewrite → retrieve → grade → generate
        self.assertEqual(step_kinds, ["rewrite", "retrieve", "grade", "generate"])
        done = events[-1]
        self.assertEqual(done["type"], "done")
        self.assertEqual(done["answer"], "Cells are the unit of life.")
        self.assertEqual(done["sources"], [])  # _Doc has no source metadata
        self.assertIsNone(done["artifact"])
        # exactly one done event, and it is last
        self.assertEqual(sum(1 for e in events if e["type"] == "done"), 1)

    def test_unknown_session_raises(self):
        from backend.services import chat
        with mock.patch.object(chat, "require_groq_key", lambda: None):
            with self.assertRaises(chat.ChatSessionError):
                list(chat.ask_stream("nope", "hello"))


class StreamEndpointTests(unittest.TestCase):
    """POST /api/chat/message/stream returns text/event-stream with `data:` frames,
    one JSON event per frame, ending in a done event (TestClient, no network)."""

    def setUp(self):
        from fastapi.testclient import TestClient
        from backend.main import app

        self.client = TestClient(app)

    def _parse_sse(self, body: str) -> list[dict]:
        events = []
        for frame in body.split("\n\n"):
            frame = frame.strip()
            if frame.startswith("data:"):
                events.append(json.loads(frame[len("data:"):].strip()))
        return events

    def test_stream_endpoint_emits_sse_frames(self):
        from backend.services import chat

        retriever = FakeRetriever({"photosynthesis basics": [_Doc("Light becomes sugar.")]})
        llm = FakeLLM([
            '{"queries": ["photosynthesis basics"]}',   # rewrite
            '{"sufficient": true}',                       # grade -> stop
            "Plants make food from light.",               # generate
        ])
        with mock.patch.object(chat.config, "AGENTIC_RAG", True), \
             mock.patch.object(chat, "require_groq_key", lambda: None), \
             mock.patch.object(chat.vector_store, "get_subject_retriever", return_value=retriever), \
             mock.patch.object(chat, "_new_chat_model", return_value=llm):
            start = self.client.post("/api/chat/start", json={"mode": "subject", "subject": "Mathematics"})
            self.assertEqual(start.status_code, 200, start.text)
            sid = start.json()["session_id"]

            resp = self.client.post("/api/chat/message/stream",
                                    json={"session_id": sid, "message": "What is photosynthesis?"})

        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertTrue(resp.headers["content-type"].startswith("text/event-stream"))
        events = self._parse_sse(resp.text)
        self.assertTrue(events, "expected at least one SSE data frame")
        self.assertTrue(all(isinstance(e, dict) and "type" in e for e in events))
        self.assertEqual([e["type"] for e in events if e["type"] == "step"][:1], ["step"])
        done = events[-1]
        self.assertEqual(done["type"], "done")
        self.assertEqual(done["answer"], "Plants make food from light.")

    def test_unknown_session_emits_error_frame(self):
        resp = self.client.post("/api/chat/message/stream",
                                json={"session_id": "does-not-exist", "message": "hi"})
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertTrue(resp.headers["content-type"].startswith("text/event-stream"))
        events = self._parse_sse(resp.text)
        self.assertEqual(events[-1]["type"], "error")
        self.assertIn("expired", events[-1]["detail"].lower())


class HttpEndToEndTests(unittest.TestCase):
    """Full path: POST /api/chat/start then /api/chat/message, running the REAL agentic
    loop end-to-end with only the LLM and retriever mocked (no network/Groq)."""

    def setUp(self):
        from fastapi.testclient import TestClient
        from backend.main import app

        self.client = TestClient(app)

    def test_chat_message_runs_agentic_loop(self):
        from backend.services import chat

        retriever = FakeRetriever({"photosynthesis basics": [_Doc("Photosynthesis converts light to sugar.")]})
        llm = FakeLLM([
            '{"queries": ["photosynthesis basics"]}',   # rewrite
            '{"sufficient": true}',                       # grade -> stop
            "Photosynthesis is how plants make food.",   # generate
        ])
        with mock.patch.object(chat.config, "AGENTIC_RAG", True), \
             mock.patch.object(chat, "require_groq_key", lambda: None), \
             mock.patch.object(chat.vector_store, "get_subject_retriever", return_value=retriever), \
             mock.patch.object(chat, "_new_chat_model", return_value=llm):
            # Use a bundled subject that exists on disk — /chat/start validates it.
            start = self.client.post("/api/chat/start", json={"mode": "subject", "subject": "Mathematics"})
            self.assertEqual(start.status_code, 200, start.text)
            sid = start.json()["session_id"]

            msg = self.client.post("/api/chat/message", json={"session_id": sid, "message": "What is photosynthesis?"})

        self.assertEqual(msg.status_code, 200, msg.text)
        self.assertEqual(msg.json()["answer"], "Photosynthesis is how plants make food.")
        self.assertEqual(retriever.queries, ["photosynthesis basics"])  # retrieval used the rewritten query
        self.assertEqual(len(llm.calls), 3)                              # rewrite + grade + generate


if __name__ == "__main__":
    unittest.main()
