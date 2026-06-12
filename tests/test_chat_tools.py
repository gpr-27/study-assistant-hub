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


class AskWiringTests(unittest.TestCase):
    """chat.ask() returns the artifact when orchestration fires, and skips RAG."""

    def test_ask_returns_artifact(self):
        from backend.services import chat
        sid = chat._register({
            "mode": "subject", "retriever": object(), "llm": FakeLLM([]), "history": [],
            "label": "L", "topic": None, "subject": "DBMS", "owner": "u1", "chat_id": None,
        })
        art = {"kind": "quizzes", "id": "x", "title": "Quiz · DBMS", "route": "#quiz", "tool": "Quiz Generator"}
        with mock.patch.object(chat, "require_groq_key", lambda: None), \
             mock.patch.object(chat_tools, "detect_intent", return_value={"tool": "quiz"}), \
             mock.patch.object(chat_tools, "run_tool",
                               return_value={"answer": "Done", "steps": [], "artifact": art}):
            out = chat.ask(sid, "create a quiz")
        self.assertEqual(out["artifact"], art)
        self.assertEqual(out["answer"], "Done")


if __name__ == "__main__":
    unittest.main()
