"""Privacy policy: flows that send USER-UPLOADED content to a model run ``sensitive=True`` (the
chain skips the Gemini steps); bundled subjects and generic topic calls do not.
``ALLOW_GEMINI_FOR_USER_CONTENT=1`` lifts the restriction. Every provider call is mocked."""
import asyncio
import os
import unittest
from unittest import mock

from backend import config, llm_chain, privacy
from backend.errors import PRIVATE_CONTENT_DETAIL, PRIVATE_CONTENT_LABEL
from backend.routers import cheatsheet, exam, flashcards, mindmap, planner, predictor, quiz, teachback
from backend.services import scannotes
from tests._study_helpers import PDF_BYTES, FakeChat, make_client

BUNDLED = "Mathematics"          # ships in documents/ (public sample PDFs)
USER_SUBJECT = "MyPrivateNotes"  # not bundled -> a user's own upload
OK = {"questions": []}


def _sens(fake: FakeChat) -> list[bool]:
    return [bool(c.get("sensitive")) for c in fake.calls]


class PolicyHelperTests(unittest.TestCase):
    def setUp(self):
        privacy._flag.set(False)

    def test_bundled_vs_user_subject(self):
        self.assertFalse(privacy.for_subject(BUNDLED))
        self.assertTrue(privacy.for_subject(USER_SUBJECT))
        self.assertFalse(privacy.for_subject(None))
        self.assertTrue(privacy.for_upload())

    def test_env_toggle_lifts_the_restriction(self):
        with mock.patch.dict(os.environ, {"ALLOW_GEMINI_FOR_USER_CONTENT": "1"}):
            self.assertFalse(privacy.for_subject(USER_SUBJECT))
            self.assertFalse(privacy.for_upload())
            self.assertFalse(config.user_content_sensitive())

    def test_default_is_off(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ALLOW_GEMINI_FOR_USER_CONTENT", None)
            self.assertTrue(config.user_content_sensitive())

    def test_mark_is_per_context_and_never_cleared(self):
        import contextvars
        ctx = contextvars.copy_context()
        self.assertTrue(ctx.run(lambda: (privacy.mark(True), privacy.mark(False), privacy.current())[-1]))
        self.assertFalse(privacy.current())     # the outer context was untouched


class _Flows(unittest.TestCase):
    def setUp(self):
        privacy._flag.set(False)             # other tests may call services directly in this context
        self.fake = FakeChat(OK)
        for p in (mock.patch("backend.llm_chain.chat", self.fake),
                  mock.patch("backend.routers._common.extract_pdf_text", return_value="paper text"),
                  mock.patch("backend.config.require_llm_key", lambda: None),
                  mock.patch("backend.services.predictor.require_llm_key", lambda: None),
                  mock.patch("backend.services.teachback.require_llm_key", lambda: None),
                  mock.patch("backend.routers.predictor.extract_pdf_text", return_value="paper text"),
                  mock.patch("backend.db.save", return_value=None)):
            p.start()
            self.addCleanup(p.stop)

    def _client(self, router):
        return make_client(router)

    def assertCalledWith(self, expected: bool):
        self.assertTrue(self.fake.calls, "the LLM was never called")
        self.assertEqual(set(_sens(self.fake)), {expected}, _sens(self.fake))


class GenerateFlowTests(_Flows):
    def _post_subject(self, router_mod, path, subject, extra=None):
        body = {"subject": subject, "pdf": "a.pdf", **(extra or {})}
        with mock.patch.object(router_mod, "read_pdf", return_value="lecture text"):
            return self._client(router_mod.router).post(path, json=body)

    def test_quiz_user_subject_sensitive_bundled_not(self):
        self._post_subject(quiz, "/api/quiz/generate", USER_SUBJECT)
        self.assertCalledWith(True)
        self.fake.calls.clear()
        self._post_subject(quiz, "/api/quiz/generate", BUNDLED)
        self.assertCalledWith(False)

    def test_quiz_upload_sensitive_and_topic_not(self):
        self._client(quiz.router).post("/api/quiz/upload", files={"file": ("n.pdf", PDF_BYTES, "application/pdf")})
        self.assertCalledWith(True)
        self.fake.calls.clear()
        self._client(quiz.router).post("/api/quiz/topic", json={"topic": "Algebra"})
        self.assertCalledWith(False)

    def test_flashcards(self):
        self._post_subject(flashcards, "/api/flashcards/generate", USER_SUBJECT)
        self.assertCalledWith(True)
        self.fake.calls.clear()
        self._post_subject(flashcards, "/api/flashcards/generate", BUNDLED)
        self.assertCalledWith(False)
        self.fake.calls.clear()
        self._client(flashcards.router).post("/api/flashcards/upload", files={"file": ("n.pdf", PDF_BYTES, "application/pdf")})
        self.assertCalledWith(True)

    def test_cheatsheet(self):
        self._post_subject(cheatsheet, "/api/cheatsheet/generate", USER_SUBJECT, {"mode": "subject"})
        self.assertCalledWith(True)
        self.fake.calls.clear()
        self._post_subject(cheatsheet, "/api/cheatsheet/generate", BUNDLED, {"mode": "subject"})
        self.assertCalledWith(False)
        self.fake.calls.clear()
        self._client(cheatsheet.router).post("/api/cheatsheet/upload", files={"file": ("n.pdf", PDF_BYTES, "application/pdf")})
        self.assertCalledWith(True)
        self.fake.calls.clear()
        self._client(cheatsheet.router).post("/api/cheatsheet/topic", json={"topic": "Sorting"})
        self.assertCalledWith(False)

    def test_mindmap(self):
        self._post_subject(mindmap, "/api/mindmap/generate", USER_SUBJECT, {"mode": "subject"})
        self.assertCalledWith(True)
        self.fake.calls.clear()
        self._post_subject(mindmap, "/api/mindmap/generate", BUNDLED, {"mode": "subject"})
        self.assertCalledWith(False)
        self.fake.calls.clear()
        self._client(mindmap.router).post("/api/mindmap/upload", files={"file": ("n.pdf", PDF_BYTES, "application/pdf")})
        self.assertCalledWith(True)
        self.fake.calls.clear()
        self._client(mindmap.router).post("/api/mindmap/generate", json={"mode": "topic", "topic": "Graphs"})
        self.assertCalledWith(False)

    def test_exam_generate_upload_grade(self):
        with mock.patch.object(exam, "read_pdf", return_value="lecture text"):
            c = self._client(exam.router)
            c.post("/api/exam/generate", json={"subject": USER_SUBJECT, "pdfs": ["a.pdf"]})
            self.assertCalledWith(True)
            self.fake.calls.clear()
            c.post("/api/exam/generate", json={"subject": BUNDLED, "pdfs": ["a.pdf"]})
            self.assertCalledWith(False)
        self.fake.calls.clear()
        self._client(exam.router).post("/api/exam/upload", files={"file": ("n.pdf", PDF_BYTES, "application/pdf")})
        self.assertCalledWith(True)
        self.fake.calls.clear()
        short = {"id": "s1", "type": "short", "question": "Explain a deadlock.",
                 "expected_points": ["circular wait"], "topic": "Concurrency"}
        self._client(exam.router).post("/api/exam/grade", json={
            "subject": USER_SUBJECT, "questions": [short], "answers": {"s1": "my answer"}})
        self.assertCalledWith(True)
        self.fake.calls.clear()
        self._client(exam.router).post("/api/exam/grade", json={
            "subject": BUNDLED, "questions": [short], "answers": {"s1": "my answer"}})
        self.assertCalledWith(False)

    def test_planner_summaries_generate_upload_topic(self):
        from datetime import date, timedelta
        when = (date.today() + timedelta(days=10)).isoformat()
        with mock.patch("backend.services.planner.read_pdf", return_value="text", create=True), \
             mock.patch("backend.documents.read_pdf", return_value="text"):
            c = self._client(planner.router)
            c.post("/api/planner/summaries", json={"subject": USER_SUBJECT, "pdfs": ["a.pdf"]})
            self.assertCalledWith(True)
            self.fake.calls.clear()
            c.post("/api/planner/summaries", json={"subject": BUNDLED, "pdfs": ["a.pdf"]})
            self.assertCalledWith(False)
            self.fake.calls.clear()
            c.post("/api/planner/generate", json={"subject": USER_SUBJECT, "pdfs": ["a.pdf"], "target_date": when})
            self.assertCalledWith(True)
        self.fake.calls.clear()
        self._client(planner.router).post("/api/planner/upload", data={"target_date": when},
                                          files=[("files", ("a.pdf", PDF_BYTES, "application/pdf"))])
        self.assertCalledWith(True)
        self.fake.calls.clear()
        self._client(planner.router).post("/api/planner/topic", json={"topic": "Calculus", "target_date": when})
        self.assertCalledWith(False)

    def test_predictor_is_always_sensitive(self):
        self._client(predictor.router).post("/api/predictor/analyze",
                                            files=[("files", ("p.pdf", PDF_BYTES, "application/pdf"))])
        self.assertCalledWith(True)

    def test_teachback_start_and_later_turns(self):
        from backend.services import teachback as tb
        start = {"concept": "Trees", "opening": "Explain Trees to me in your own words."}
        reply = {"reply": "Why?", "done": False, "evaluation": None}
        with mock.patch.object(config, "PERSISTENCE_ENABLED", False), \
             mock.patch.object(tb.db, "_get_db", return_value=None):
            c = self._client(teachback.router)
            with mock.patch.object(teachback, "read_pdf", return_value="lecture"):
                self.fake.replies = [start, reply]
                r = c.post("/api/teachback/start", json={"mode": "subject", "subject": USER_SUBJECT, "pdf": "a.pdf"})
                self.assertEqual(r.status_code, 200, r.text)
                sid = r.json()["session_id"]
                self.assertCalledWith(True)
                self.fake.calls.clear()
                c.post("/api/teachback/message", json={"session_id": sid, "message": "It is a tree."})
                self.assertCalledWith(True)             # the session remembers its privacy class
            self.fake.calls.clear()
            self.fake.replies = [start]
            c.post("/api/teachback/start", json={"mode": "topic", "topic": "Algorithms"})
            self.assertCalledWith(False)
            self.fake.calls.clear()
            self.fake.replies = [start]
            with mock.patch.object(teachback, "read_pdf", return_value="lecture"):
                c.post("/api/teachback/start", json={"mode": "subject", "subject": BUNDLED, "pdf": "a.pdf"})
            self.assertCalledWith(False)

    def test_env_toggle_makes_everything_non_sensitive(self):
        with mock.patch.dict(os.environ, {"ALLOW_GEMINI_FOR_USER_CONTENT": "1"}):
            self._post_subject(quiz, "/api/quiz/generate", USER_SUBJECT)
            self.assertCalledWith(False)


class ScanNotesPrivacyTests(unittest.IsolatedAsyncioTestCase):
    async def test_photos_are_sensitive_unless_allowed(self):
        seen = []

        async def fake_achat(system, messages, *, images=None, **kw):
            seen.append(kw.get("sensitive"))
            from tests._study_helpers import result
            return result("text")

        with mock.patch("backend.llm_chain.achat", fake_achat), mock.patch("backend.llm_chain.has_any_key", return_value=True):
            await scannotes.transcribe_image(b"x", "image/png")
            with mock.patch.dict(os.environ, {"ALLOW_GEMINI_FOR_USER_CONTENT": "1"}):
                await scannotes.transcribe_image(b"x", "image/png")
        self.assertEqual(seen, [True, False])


class ChatPrivacyTests(unittest.IsolatedAsyncioTestCase):
    def _session(self, mode, subject=None):
        from backend.services import chat
        return chat.Session("id", "alice", mode, "L", subject, None, None, "n")

    def test_session_classification(self):
        self.assertTrue(privacy.for_session(self._session("pdf")))
        self.assertTrue(privacy.for_session(self._session("subject", USER_SUBJECT)))
        self.assertFalse(privacy.for_session(self._session("subject", BUNDLED)))
        self.assertFalse(privacy.for_session(self._session("topic")))

    async def test_chat_answer_stream_is_sensitive_for_pdf_and_user_subject(self):
        from backend.services import chat
        seen = []

        async def fake_astream(system, messages, **kw):
            seen.append(kw.get("sensitive"))
            from tests._study_helpers import result
            yield "hi"
            yield result("hi")

        class R:
            def invoke(self, q, **k):
                return []

            invoke_lexical = invoke

        for session, expected in ((self._session("pdf"), True),
                                  (self._session("subject", USER_SUBJECT), True),
                                  (self._session("subject", BUNDLED), False),
                                  (self._session("topic"), False)):
            seen.clear()
            with mock.patch("backend.llm_chain.astream_chat", fake_astream), \
                 mock.patch("backend.llm_chain.has_any_key", return_value=True), \
                 mock.patch.object(config, "PERSISTENCE_ENABLED", False), \
                 mock.patch.object(chat, "_prepare", lambda s: (R(), "P")), \
                 mock.patch("backend.agentic_rag.llm_rewrite_enabled", lambda: False), \
                 mock.patch.object(config, "AGENTIC_RAG_MAX_ITERS", 1):
                async def run():           # a fresh task = a fresh context, like one HTTP request
                    return [e async for e in chat.stream_turn(session, "hello there friend")]
                events = await asyncio.create_task(run())
            self.assertTrue(events)
            self.assertTrue(seen, [e for e in events][-1])
            self.assertEqual(set(seen), {expected}, (session.mode, session.subject, seen))


class OnlyGeminiKeysTests(unittest.TestCase):
    def test_sensitive_request_with_only_gemini_keys_gets_a_clear_503(self):
        from fastapi.testclient import TestClient

        from backend.main import app
        exc = llm_chain.LLMChainError("No non-Gemini LLM key is configured for a sensitive request",
                                      status=None, attempts=[])

        def boom():
            raise exc

        app.add_api_route("/api/_test_private", boom, methods=["GET"])
        self.addCleanup(lambda: app.router.routes.pop())
        r = TestClient(app, raise_server_exceptions=False).get("/api/_test_private")
        self.assertEqual(r.status_code, 503)
        self.assertEqual(r.json()["detail"], PRIVATE_CONTENT_DETAIL)
        self.assertEqual(r.json()["llm"]["label"], PRIVATE_CONTENT_LABEL)

    def test_real_chain_error_text_is_recognised(self):
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "g", "ANTHROPIC_API_KEY": "", "GROQ_API_KEY": "",
                                          "GROQ_API_KEY_2": "", "GEMINI_API_KEY_2": ""}):
            run = llm_chain._Run(True, 5.0)
            err = run.error()
        from backend.errors import is_private_content_unavailable
        self.assertTrue(is_private_content_unavailable(err), str(err))


if __name__ == "__main__":
    unittest.main()
