"""Tests for chat tool orchestration (chat creates artifacts in other tools).

No network/Groq: a FakeLLM returns queued replies; tool services and db.save are
mocked. Covers the regex gate (no LLM call for ordinary messages), intent
parsing, quiz/flashcards/plan dispatch + history save shape, readable failure
answers, and ask() returning the artifact through the chat service.
"""
import unittest
from unittest import mock

from backend.services import chat_tools


class _Reply:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    def __init__(self, replies):
        self._replies = list(replies)
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        return _Reply(self._replies.pop(0) if self._replies else "")


class GateTests(unittest.TestCase):
    def test_ordinary_question_never_calls_llm(self):
        llm = FakeLLM([])
        self.assertIsNone(chat_tools.detect_intent(llm, "What is a binary search tree?"))
        self.assertIsNone(chat_tools.detect_intent(llm, "Explain quizzes in psychology research"))
        self.assertEqual(llm.calls, [])  # gate rejected; no LLM cost

    def test_question_about_tool_rejected_by_llm(self):
        llm = FakeLLM(['{"is_request": false, "tool": null, "topic": null, "count": null, "days": null}'])
        self.assertIsNone(chat_tools.detect_intent(llm, "can you make a quiz harder than real exams in general?"))
        self.assertEqual(len(llm.calls), 1)

    def test_creation_request_detected_with_params(self):
        llm = FakeLLM(['{"is_request": true, "tool": "quiz", "topic": "sorting algorithms", "count": 8, "days": null}'])
        intent = chat_tools.detect_intent(llm, "create a quiz with 8 questions on sorting algorithms")
        self.assertEqual(intent, {"tool": "quiz", "topic": "sorting algorithms", "count": 8, "days": None})

    def test_malformed_llm_output_falls_through(self):
        llm = FakeLLM(["not json at all"])
        self.assertIsNone(chat_tools.detect_intent(llm, "make me a mind map of recursion"))

    def test_cheatsheet_request_detected(self):
        llm = FakeLLM(['{"is_request": true, "tool": "cheatsheet", "topic": null, "count": null, "days": null}'])
        intent = chat_tools.detect_intent(llm, "create a cheat sheet for this subject")
        self.assertEqual(intent["tool"], "cheatsheet")
        self.assertEqual(len(llm.calls), 1)  # gate passed, LLM confirmed

    def test_predictor_request_detected(self):
        llm = FakeLLM(['{"is_request": true, "tool": "predictor", "topic": null, "count": null, "days": null}'])
        intent = chat_tools.detect_intent(llm, "predict likely exam questions")
        self.assertEqual(intent["tool"], "predictor")
        self.assertEqual(len(llm.calls), 1)

    def test_new_phrasings_pass_the_regex_gate(self):
        """The gate must let the new cheatsheet/predictor phrasings reach the LLM."""
        for phrase in (
            "create a cheat sheet for this subject",
            "make a cheatsheet",
            "predict likely exam questions",
            "generate exam predictions",
            "generate the likely exam questions for me",
        ):
            llm = FakeLLM(['{"is_request": true, "tool": "cheatsheet", "topic": null, "count": null, "days": null}'])
            chat_tools.detect_intent(llm, phrase)
            self.assertEqual(len(llm.calls), 1, f"gate should pass for: {phrase!r}")

    def test_plain_questions_about_new_tools_skip_the_llm(self):
        """Plain questions (no creation verb) must not trip the gate or cost an LLM call."""
        llm = FakeLLM([])
        for phrase in (
            "what is a cheat sheet?",
            "how do exam predictors work?",
            "why are likely questions useful for revision?",
        ):
            self.assertIsNone(chat_tools.detect_intent(llm, phrase))
        self.assertEqual(llm.calls, [])


def _subject_session(**over):
    base = {"mode": "subject", "subject": "DBMS", "topic": None, "owner": "u1", "llm": FakeLLM([])}
    base.update(over)
    return base


def _topic_session(**over):
    base = {"mode": "topic", "subject": None, "topic": "Recursion", "owner": "u1", "llm": FakeLLM([])}
    base.update(over)
    return base


class RunToolTests(unittest.TestCase):
    def test_quiz_from_subject_saves_router_shape(self):
        qs = [{"question": "Q1", "options": ["a", "b", "c", "d"], "answer_index": 0, "explanation": "e"}]
        with mock.patch.object(chat_tools, "_subject_content", return_value=("CONTENT", ["a.pdf"])), \
             mock.patch.object(chat_tools.quiz_service, "make_quiz", return_value={"questions": qs}) as mk, \
             mock.patch.object(chat_tools.db, "save", return_value="id1") as save:
            out = chat_tools.run_tool(_subject_session(), {"tool": "quiz", "topic": None, "count": 5, "days": None}, "m")
        mk.assert_called_once_with("CONTENT", 5, "medium")
        kind, owner, doc = save.call_args.args
        self.assertEqual((kind, owner), ("quizzes", "u1"))
        self.assertEqual(doc["title"], "Quiz · DBMS")
        self.assertEqual(doc["questions"], qs)
        self.assertEqual(out["artifact"]["route"], "#quiz")
        self.assertEqual(out["artifact"]["id"], "id1")
        self.assertIn("Quiz · DBMS", out["answer"])

    def test_flashcards_from_topic(self):
        cards = [{"front": "f", "back": "b"}]
        with mock.patch.object(chat_tools.flashcards_service, "make_flashcards_from_topic",
                               return_value={"cards": cards}) as mk, \
             mock.patch.object(chat_tools.db, "save", return_value="id2") as save:
            out = chat_tools.run_tool(_topic_session(), {"tool": "flashcards", "topic": None, "count": None, "days": None}, "m")
        mk.assert_called_once_with("Recursion", 10, "medium")
        self.assertEqual(save.call_args.args[0], "flashcards")
        self.assertEqual(out["artifact"]["kind"], "flashcards")

    def test_plan_uses_days_and_saves_target_date(self):
        with mock.patch.object(chat_tools.planner_service, "generate_plan_from_topic", return_value="# plan") as mk, \
             mock.patch.object(chat_tools.db, "save", return_value="id3") as save:
            out = chat_tools.run_tool(_topic_session(), {"tool": "plan", "topic": None, "count": None, "days": 14}, "m")
        target = mk.call_args.args[1]
        self.assertRegex(target, r"^\d{4}-\d{2}-\d{2}$")
        self.assertEqual(save.call_args.args[2]["plan"], "# plan")
        self.assertEqual(out["artifact"]["route"], "#planner")

    def test_summary_in_topic_session_gives_readable_failure(self):
        out = chat_tools.run_tool(_topic_session(), {"tool": "summary", "topic": None, "count": None, "days": None}, "m")
        self.assertIsNone(out["artifact"])
        self.assertIn("documents", out["answer"])

    def test_tool_exception_becomes_friendly_answer(self):
        with mock.patch.object(chat_tools, "_subject_content", side_effect=RuntimeError("boom")):
            out = chat_tools.run_tool(_subject_session(), {"tool": "quiz", "topic": None, "count": None, "days": None}, "m")
        self.assertIsNone(out["artifact"])
        self.assertIn("something went wrong", out["answer"])

    def test_cheatsheet_from_subject_saves_router_shape(self):
        with mock.patch.object(chat_tools, "_subject_content", return_value=("CONTENT", ["a.pdf", "b.pdf"])), \
             mock.patch.object(chat_tools.cheatsheet_service, "make_cheatsheet", return_value="# Cheat") as mk, \
             mock.patch.object(chat_tools.db, "save", return_value="cs1") as save:
            out = chat_tools.run_tool(_subject_session(), {"tool": "cheatsheet", "topic": None, "count": None, "days": None}, "m")
        mk.assert_called_once_with("CONTENT")
        kind, owner, doc = save.call_args.args
        self.assertEqual((kind, owner), ("cheatsheets", "u1"))
        self.assertEqual(doc["title"], "Cheat Sheet · DBMS")
        self.assertEqual(doc["source"], "DBMS")
        self.assertEqual(doc["markdown"], "# Cheat")
        self.assertEqual(out["artifact"]["kind"], "cheatsheets")
        self.assertEqual(out["artifact"]["route"], "#cheatsheet")
        self.assertEqual(out["artifact"]["id"], "cs1")

    def test_cheatsheet_in_topic_session_gives_readable_failure(self):
        out = chat_tools.run_tool(_topic_session(), {"tool": "cheatsheet", "topic": None, "count": None, "days": None}, "m")
        self.assertIsNone(out["artifact"])
        self.assertIn("documents", out["answer"])

    def test_predictor_from_subject_saves_router_shape(self):
        predicted = [{"question": "Explain X", "why": "common"}]
        patterns = ["pattern A"]
        with mock.patch.object(chat_tools, "_subject_papers_corpus", return_value=("CORPUS", ["p1.pdf", "p2.pdf"])), \
             mock.patch.object(chat_tools.predictor_service, "predict_questions",
                               return_value={"predicted_questions": predicted, "patterns": patterns}) as mk, \
             mock.patch.object(chat_tools.db, "save", return_value="pr1") as save:
            out = chat_tools.run_tool(_subject_session(), {"tool": "predictor", "topic": None, "count": None, "days": None}, "m")
        mk.assert_called_once_with("CORPUS", subject_hint="DBMS")
        kind, owner, doc = save.call_args.args
        self.assertEqual((kind, owner), ("predictions", "u1"))
        self.assertEqual(doc["title"], "Predictions · DBMS")
        self.assertEqual(doc["subject"], "DBMS")
        self.assertEqual(doc["num_papers"], 2)
        self.assertEqual(doc["patterns"], patterns)
        self.assertEqual(doc["predicted_questions"], predicted)
        self.assertEqual(out["artifact"]["kind"], "predictions")
        self.assertEqual(out["artifact"]["route"], "#predictor")
        self.assertEqual(out["artifact"]["id"], "pr1")

    def test_predictor_in_topic_session_gives_readable_failure(self):
        out = chat_tools.run_tool(_topic_session(), {"tool": "predictor", "topic": None, "count": None, "days": None}, "m")
        self.assertIsNone(out["artifact"])
        self.assertIn("papers", out["answer"])

    def test_cheatsheet_tolerates_persistence_off(self):
        """db.save returns None when persistence is off — the artifact id is None, no crash."""
        with mock.patch.object(chat_tools, "_subject_content", return_value=("CONTENT", ["a.pdf"])), \
             mock.patch.object(chat_tools.cheatsheet_service, "make_cheatsheet", return_value="# Cheat"), \
             mock.patch.object(chat_tools.db, "save", return_value=None):
            out = chat_tools.run_tool(_subject_session(), {"tool": "cheatsheet", "topic": None, "count": None, "days": None}, "m")
        self.assertIsNone(out["artifact"]["id"])
        self.assertEqual(out["artifact"]["kind"], "cheatsheets")


def _agent_session(**over):
    """A registered chat session for the AUTONOMOUS Agent view (orchestrates tools)."""
    from backend.services import chat
    base = {
        "mode": "subject", "retriever": object(), "llm": FakeLLM([]), "history": [],
        "label": "L", "topic": None, "subject": "DBMS", "owner": "u1", "chat_id": None,
        "autonomous": True,
    }
    base.update(over)
    return chat._register(base)


class AskWiringTests(unittest.TestCase):
    """chat.ask() orchestrates tools ONLY for autonomous sessions; regular chat is pure RAG."""

    def test_autonomous_ask_returns_artifact(self):
        from backend.services import chat
        sid = _agent_session()
        art = {"kind": "quizzes", "id": "x", "title": "Quiz · DBMS", "route": "#quiz", "tool": "Quiz Generator"}
        with mock.patch.object(chat, "require_groq_key", lambda: None), \
             mock.patch.object(chat_tools, "detect_intent", return_value={"tool": "quiz"}) as detect, \
             mock.patch.object(chat_tools, "run_tool",
                               return_value={"answer": "Done", "steps": [], "artifact": art}) as run:
            out = chat.ask(sid, "create a quiz")
        detect.assert_called_once()
        run.assert_called_once()
        self.assertEqual(out["artifact"], art)
        self.assertEqual(out["answer"], "Done")

    def test_regular_chat_never_invokes_tools(self):
        """A non-autonomous (regular) session must NOT call chat_tools — pure RAG only."""
        from backend import agentic_rag
        from backend.services import chat
        sid = _agent_session(autonomous=False)
        with mock.patch.object(chat, "require_groq_key", lambda: None), \
             mock.patch.object(chat.config, "AGENTIC_RAG", True), \
             mock.patch.object(chat_tools, "detect_intent") as detect, \
             mock.patch.object(chat_tools, "run_tool") as run, \
             mock.patch.object(agentic_rag, "answer",
                               return_value=("a normal RAG answer", [], ["a.pdf"])):
            out = chat.ask(sid, "create a quiz")
        detect.assert_not_called()
        run.assert_not_called()
        self.assertIsNone(out["artifact"])
        self.assertEqual(out["answer"], "a normal RAG answer")

    def test_regular_chat_default_when_autonomous_omitted(self):
        """Sessions that never set the autonomous flag default to pure RAG (no tools)."""
        from backend import agentic_rag
        from backend.services import chat
        # Omit the key entirely (session.get("autonomous") is None / falsy).
        sid = _agent_session(autonomous=None)
        del chat._sessions[sid]["autonomous"]
        with mock.patch.object(chat, "require_groq_key", lambda: None), \
             mock.patch.object(chat.config, "AGENTIC_RAG", True), \
             mock.patch.object(chat_tools, "detect_intent") as detect, \
             mock.patch.object(chat_tools, "run_tool") as run, \
             mock.patch.object(agentic_rag, "answer", return_value=("ans", [], [])):
            out = chat.ask(sid, "make a 10-question quiz")
        detect.assert_not_called()
        run.assert_not_called()
        self.assertIsNone(out["artifact"])


if __name__ == "__main__":
    unittest.main()
