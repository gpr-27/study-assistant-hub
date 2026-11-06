"""Tests for chat tool orchestration (chat creates artifacts in other tools).

No network/Groq: a FakeLLM returns queued replies; tool services and db.save are
mocked. Covers the regex gate (no LLM call for ordinary messages), intent
parsing, quiz/flashcards/plan dispatch + history save shape, readable failure
answers, and ask() returning the artifact through the chat service.
"""
import unittest
from unittest import mock

from backend import llm_chain
from backend.services import chat_tools


def _chain_result(text):
    return llm_chain.ChatResult(text=text, provider="groq", model="openai/gpt-oss-120b", step=2,
                                step_name="Groq key 1")


class FakeLLM:
    """Queued replies served through a patched ``llm_chain.chat``; records the calls."""

    def __init__(self, replies):
        self._replies = list(replies)
        self.calls = []

    def chat(self, system, messages, **kw):
        self.calls.append((system, messages, kw))
        return _chain_result(self._replies.pop(0) if self._replies else "")

    async def achat(self, system, messages, **kw):
        return self.chat(system, messages, **kw)


def _detect(llm, message):
    with mock.patch.object(llm_chain, "chat", llm.chat):
        return chat_tools.detect_intent(message)


class GateTests(unittest.TestCase):
    def test_ordinary_question_never_calls_llm(self):
        llm = FakeLLM([])
        self.assertIsNone(_detect(llm, "What is a binary search tree?"))
        self.assertIsNone(_detect(llm, "Explain quizzes in psychology research"))
        self.assertEqual(llm.calls, [])  # gate rejected; no LLM cost

    def test_question_about_tool_rejected_by_llm(self):
        llm = FakeLLM(['{"is_request": false, "tool": null, "topic": null, "count": null, "days": null}'])
        self.assertIsNone(_detect(llm, "can you make a quiz harder than real exams in general?"))
        self.assertEqual(len(llm.calls), 1)

    def test_creation_request_detected_with_params(self):
        llm = FakeLLM(['{"is_request": true, "tool": "quiz", "topic": "sorting algorithms", "count": 8, "days": null}'])
        intent = _detect(llm, "create a quiz with 8 questions on sorting algorithms")
        # detect_intent now returns a LIST of intents (one per requested tool); a legacy
        # single-"tool" reply is tolerated and normalized into a one-element list.
        self.assertEqual(intent, [{"tool": "quiz", "topic": "sorting algorithms", "count": 8, "days": None, "difficulty": None}])

    def test_multiple_tools_detected_in_one_message(self):
        llm = FakeLLM(['{"is_request": true, "tools": ['
                       '{"tool": "quiz", "topic": "sorting", "count": null, "days": null}, '
                       '{"tool": "flashcards", "topic": "sorting", "count": null, "days": null}]}'])
        intent = _detect(llm, "make me a quiz and flashcards on sorting")
        self.assertEqual([i["tool"] for i in intent], ["quiz", "flashcards"])

    def test_duplicate_tools_are_de_duped(self):
        llm = FakeLLM(['{"is_request": true, "tools": ['
                       '{"tool": "quiz", "topic": null, "count": null, "days": null}, '
                       '{"tool": "quiz", "topic": null, "count": null, "days": null}]}'])
        intent = _detect(llm, "make a quiz and another quiz")
        self.assertEqual([i["tool"] for i in intent], ["quiz"])

    def test_forced_intents_builds_single_tool_without_llm(self):
        """forced_intents (manual slash pick) skips the gate/LLM and forces the chosen tool."""
        self.assertEqual(chat_tools.forced_intents("quiz"),
                         [{"tool": "quiz", "topic": None, "count": None, "days": None, "difficulty": None}])
        self.assertEqual(chat_tools.forced_intents("teachback")[0]["tool"], "teachback")
        self.assertIsNone(chat_tools.forced_intents("bogus"))
        self.assertIsNone(chat_tools.forced_intents(None))

    def test_forced_intents_parses_count_difficulty_and_days_from_message(self):
        """The slash path still reads explicit params out of the message (no LLM)."""
        q = chat_tools.forced_intents("quiz", "start the quiz with 15 hard questions")[0]
        self.assertEqual(q["count"], 15)
        self.assertEqual(q["difficulty"], "hard")
        f = chat_tools.forced_intents("flashcards", "give me 8 easy cards")[0]
        self.assertEqual((f["count"], f["difficulty"]), (8, "easy"))
        p = chat_tools.forced_intents("plan", "make a plan, exam in 10 days")[0]
        self.assertEqual(p["days"], 10)

    def test_detect_intent_extracts_difficulty(self):
        llm = FakeLLM(['{"is_request": true, "tools": [{"tool": "quiz", "topic": null, "count": 10, "days": null, "difficulty": "hard"}]}'])
        intent = _detect(llm, "make a hard quiz with 10 questions")
        self.assertEqual((intent[0]["count"], intent[0]["difficulty"]), (10, "hard"))

    def test_malformed_llm_output_falls_through(self):
        llm = FakeLLM(["not json at all"])
        self.assertIsNone(_detect(llm, "make me a mind map of recursion"))

    def test_cheatsheet_request_detected(self):
        llm = FakeLLM(['{"is_request": true, "tool": "cheatsheet", "topic": null, "count": null, "days": null}'])
        intent = _detect(llm, "create a cheat sheet for this subject")
        self.assertEqual(intent[0]["tool"], "cheatsheet")
        self.assertEqual(len(llm.calls), 1)  # gate passed, LLM confirmed

    def test_predictor_request_detected(self):
        llm = FakeLLM(['{"is_request": true, "tool": "predictor", "topic": null, "count": null, "days": null}'])
        intent = _detect(llm, "predict likely exam questions")
        self.assertEqual(intent[0]["tool"], "predictor")
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
            _detect(llm, phrase)
            self.assertEqual(len(llm.calls), 1, f"gate should pass for: {phrase!r}")

    def test_mock_exam_phrasings_pass_the_regex_gate(self):
        """The gate must let mock-exam creation phrasings reach the LLM."""
        for phrase in (
            "start a mock exam",
            "create a mock exam",
            "give me a mock exam",
            "let's take an exam",
            "generate a mock exam for this subject",
        ):
            llm = FakeLLM(['{"is_request": true, "tool": "exam", "topic": null, "count": null, "days": null}'])
            _detect(llm, phrase)
            self.assertEqual(len(llm.calls), 1, f"gate should pass for: {phrase!r}")

    def test_exam_request_detected(self):
        llm = FakeLLM(['{"is_request": true, "tool": "exam", "topic": null, "count": null, "days": null}'])
        intent = _detect(llm, "start a mock exam")
        self.assertEqual(intent[0]["tool"], "exam")
        self.assertEqual(len(llm.calls), 1)

    def test_plain_question_about_exam_skips_the_llm(self):
        """A plain question about exams must not trip the gate."""
        llm = FakeLLM([])
        self.assertIsNone(_detect(llm, "what is a mock exam?"))
        self.assertEqual(llm.calls, [])

    def test_plain_questions_about_new_tools_skip_the_llm(self):
        """Plain questions (no creation verb) must not trip the gate or cost an LLM call."""
        llm = FakeLLM([])
        for phrase in (
            "what is a cheat sheet?",
            "how do exam predictors work?",
            "why are likely questions useful for revision?",
        ):
            self.assertIsNone(_detect(llm, phrase))
        self.assertEqual(llm.calls, [])

    def test_teachback_phrasings_pass_the_regex_gate(self):
        """The gate must let teach-back/Feynman creation phrasings reach the LLM."""
        for phrase in (
            "start a teach-back session",
            "let's do a teachback",
            "begin a feynman session on this",
            "give me a teach back",
            "create a teachback for this subject",
        ):
            llm = FakeLLM(['{"is_request": true, "tool": "teachback", "topic": null, "count": null, "days": null}'])
            _detect(llm, phrase)
            self.assertEqual(len(llm.calls), 1, f"gate should pass for: {phrase!r}")

    def test_teachback_request_detected(self):
        llm = FakeLLM(['{"is_request": true, "tool": "teachback", "topic": null, "count": null, "days": null}'])
        intent = _detect(llm, "let's do a teach-back")
        self.assertEqual(intent[0]["tool"], "teachback")
        self.assertEqual(len(llm.calls), 1)

    def test_plain_question_about_teachback_skips_the_llm(self):
        """A plain question about teach-back must not trip the gate."""
        llm = FakeLLM([])
        self.assertIsNone(_detect(llm, "what is the feynman technique?"))
        self.assertEqual(llm.calls, [])

    def test_progress_request_detected_without_creation_verb(self):
        """Progress is read-only: no creation verb, no LLM call — detected by its own gate."""
        for phrase in (
            "show my progress",
            "how am i doing?",
            "my stats",
            "my analytics",
            "my insights",
            "my mastery",
        ):
            llm = FakeLLM([])  # must never reach the LLM
            intent = _detect(llm, phrase)
            self.assertIsNotNone(intent, f"progress not detected for: {phrase!r}")
            self.assertEqual(intent[0]["tool"], "progress", f"wrong tool for: {phrase!r}")
            self.assertEqual(llm.calls, [], f"progress must not cost an LLM call: {phrase!r}")

    def test_ordinary_message_is_not_progress(self):
        """A normal question must not be mistaken for a progress request."""
        llm = FakeLLM([])
        self.assertIsNone(_detect(llm, "what is the progress of a wave?"))
        self.assertEqual(llm.calls, [])


def _subject_session(**over):
    base = {"mode": "subject", "subject": "DBMS", "topic": None, "owner": "u1"}
    base.update(over)
    return base


def _topic_session(**over):
    base = {"mode": "topic", "subject": None, "topic": "Recursion", "owner": "u1"}
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
        # inline-widget payload: data.questions matches the quiz kind
        self.assertEqual(out["artifact"]["data"], {"questions": qs, "subject": "DBMS", "topic": None})
        self.assertTrue(qs[0]["sig"])       # signed so POST /api/grade can verify the attempt

    def test_flashcards_from_topic(self):
        cards = [{"front": "f", "back": "b"}]
        with mock.patch.object(chat_tools.flashcards_service, "make_flashcards_from_topic",
                               return_value={"cards": cards}) as mk, \
             mock.patch.object(chat_tools.db, "save", return_value="id2") as save:
            out = chat_tools.run_tool(_topic_session(), {"tool": "flashcards", "topic": None, "count": None, "days": None}, "m")
        mk.assert_called_once_with("Recursion", 10, "medium")
        self.assertEqual(save.call_args.args[0], "flashcards")
        self.assertEqual(out["artifact"]["kind"], "flashcards")
        self.assertEqual(out["artifact"]["data"], {"cards": cards})

    def test_plan_uses_days_and_saves_target_date(self):
        with mock.patch.object(chat_tools.planner_service, "generate_plan_from_topic_ex", return_value=("# plan", None)) as mk, \
             mock.patch.object(chat_tools.db, "save", return_value="id3") as save:
            out = chat_tools.run_tool(_topic_session(), {"tool": "plan", "topic": None, "count": None, "days": 14}, "m")
        target = mk.call_args.args[1]
        self.assertRegex(target, r"^\d{4}-\d{2}-\d{2}$")
        self.assertEqual(save.call_args.args[2]["plan"], "# plan")
        self.assertEqual(out["artifact"]["route"], "#planner")
        self.assertEqual(out["artifact"]["data"], {"plan": "# plan"})

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
             mock.patch.object(chat_tools.cheatsheet_service, "make_cheatsheet_ex", return_value=("# Cheat", None)) as mk, \
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
        self.assertEqual(out["artifact"]["data"], {"markdown": "# Cheat"})

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
        self.assertEqual(
            out["artifact"]["data"],
            {"patterns": patterns, "predicted_questions": predicted},
        )

    def test_predictor_in_topic_session_gives_readable_failure(self):
        out = chat_tools.run_tool(_topic_session(), {"tool": "predictor", "topic": None, "count": None, "days": None}, "m")
        self.assertIsNone(out["artifact"])
        self.assertIn("papers", out["answer"])

    def test_exam_from_subject_returns_artifact_without_saving(self):
        """A mock exam is generated from the session subject and is NOT saved here
        (it's persisted as a graded attempt only after the user submits)."""
        questions = [
            {"id": "m1", "type": "mcq", "question": "Q1", "options": ["a", "b", "c", "d"],
             "correct": "A", "explanation": "e", "topic": "t"},
            {"id": "s1", "type": "short", "question": "Explain X"},
        ]
        with mock.patch.object(chat_tools, "_subject_content", return_value=("CONTENT", ["a.pdf"])) as content, \
             mock.patch.object(chat_tools.exam_service, "generate_exam",
                               return_value={"questions": questions}) as mk, \
             mock.patch.object(chat_tools.db, "save", return_value="x1") as save:
            out = chat_tools.run_tool(_subject_session(), {"tool": "exam", "topic": None, "count": None, "days": None}, "m")
        content.assert_called_once_with("DBMS", "u1")
        mk.assert_called_once_with("CONTENT", 6, 2, difficulty="medium")  # default 6 mcq + 2 short
        save.assert_not_called()  # NOT saved until the graded attempt is submitted
        art = out["artifact"]
        self.assertEqual(art["kind"], "exams")
        self.assertEqual(art["route"], "#exam")
        self.assertEqual(art["tool"], "Mock Exam")
        self.assertIsNone(art["id"])
        self.assertEqual(art["title"], "Mock Exam · DBMS")
        self.assertEqual(art["data"]["questions"], questions)
        self.assertEqual(art["data"]["subject"], "DBMS")
        self.assertEqual(art["data"]["num_mcq"], 6)
        self.assertEqual(art["data"]["num_short"], 2)

    def test_exam_uses_intent_count_capped_for_mcq(self):
        questions = [{"id": "m1", "type": "mcq", "question": "Q", "options": ["a", "b", "c", "d"], "correct": "A"}]
        with mock.patch.object(chat_tools, "_subject_content", return_value=("CONTENT", ["a.pdf"])), \
             mock.patch.object(chat_tools.exam_service, "generate_exam",
                               return_value={"questions": questions}) as mk, \
             mock.patch.object(chat_tools.db, "save") as save:
            chat_tools.run_tool(_subject_session(), {"tool": "exam", "topic": None, "count": 50, "days": None}, "m")
        mk.assert_called_once_with("CONTENT", 10, 2, difficulty="medium")  # count capped at ~10
        save.assert_not_called()

    def test_exam_in_topic_session_gives_readable_failure(self):
        out = chat_tools.run_tool(_topic_session(), {"tool": "exam", "topic": None, "count": None, "days": None}, "m")
        self.assertIsNone(out["artifact"])
        self.assertIn("documents", out["answer"])

    def test_exam_with_no_subject_gives_readable_failure(self):
        out = chat_tools.run_tool(
            _subject_session(subject=None),
            {"tool": "exam", "topic": None, "count": None, "days": None}, "m",
        )
        self.assertIsNone(out["artifact"])
        self.assertIn("documents", out["answer"])

    def test_teachback_from_subject_starts_a_session(self):
        """A subject teach-back reads the subject content, opens a live session, and
        returns a teachback artifact carrying the session_id/concept/opening — nothing saved."""
        with mock.patch.object(chat_tools, "_subject_content", return_value=("CONTENT", ["a.pdf"])) as content, \
             mock.patch.object(chat_tools.teachback_service, "start_session",
                               return_value=("tb-sess-1", "Normalization", "Explain normalization to me.", None)) as mk, \
             mock.patch.object(chat_tools.db, "save") as save:
            out = chat_tools.run_tool(_subject_session(), {"tool": "teachback", "topic": None, "count": None, "days": None}, "m")
        content.assert_called_once_with("DBMS", "u1")
        mk.assert_called_once_with("subject", content="CONTENT", label="DBMS", owner="u1")
        save.assert_not_called()  # the transcript is saved only when the student finishes
        art = out["artifact"]
        self.assertEqual(art["kind"], "teachbacks")
        self.assertEqual(art["route"], "#teachback")
        self.assertEqual(art["tool"], "Teach-back Tutor")
        self.assertIsNone(art["id"])
        self.assertEqual(art["title"], "Teach-back · DBMS")
        self.assertEqual(art["data"]["session_id"], "tb-sess-1")
        self.assertEqual(art["data"]["concept"], "Normalization")
        self.assertEqual(art["data"]["opening"], "Explain normalization to me.")

    def test_teachback_from_topic_starts_a_session(self):
        with mock.patch.object(chat_tools.teachback_service, "start_session",
                               return_value=("tb-sess-2", "Recursion", "Explain recursion.", None)) as mk, \
             mock.patch.object(chat_tools.db, "save") as save:
            out = chat_tools.run_tool(_topic_session(), {"tool": "teachback", "topic": None, "count": None, "days": None}, "m")
        mk.assert_called_once_with("topic", topic="Recursion", label="Topic · Recursion", owner="u1")
        save.assert_not_called()
        art = out["artifact"]
        self.assertEqual(art["kind"], "teachbacks")
        self.assertEqual(art["title"], "Teach-back · Recursion")
        self.assertEqual(art["data"]["session_id"], "tb-sess-2")

    def test_progress_returns_read_only_summary_without_saving(self):
        """Progress pulls the same summary the dashboard uses and returns it inline; nothing saved."""
        summary = {
            "overall": {"total_attempts": 12, "overall_accuracy": 0.75, "streak_days": 3},
            "subjects": [{"subject": "DBMS", "attempts": 12, "accuracy": 0.75, "topics": []}],
            "weakest": [{"subject": "DBMS", "topic": "Joins", "attempts": 4, "accuracy": 0.4}],
        }
        with mock.patch.object(chat_tools.progress_service, "build_summary", return_value=summary) as mk, \
             mock.patch.object(chat_tools.db, "save") as save:
            out = chat_tools.run_tool(_subject_session(), {"tool": "progress", "topic": None, "count": None, "days": None}, "m")
        mk.assert_called_once_with("u1")
        save.assert_not_called()
        art = out["artifact"]
        self.assertEqual(art["kind"], "progress")
        self.assertEqual(art["route"], "#progress")
        self.assertEqual(art["tool"], "Progress")
        self.assertIsNone(art["id"])
        self.assertEqual(art["data"], summary)

    def test_progress_with_no_activity_gives_friendly_note(self):
        """When there are no attempts, progress still returns the (empty) summary with a friendly answer."""
        empty = {"overall": {"total_attempts": 0}, "subjects": [], "weakest": [], "activity": []}
        with mock.patch.object(chat_tools.progress_service, "build_summary", return_value=empty), \
             mock.patch.object(chat_tools.db, "save") as save:
            out = chat_tools.run_tool(_subject_session(), {"tool": "progress", "topic": None, "count": None, "days": None}, "m")
        save.assert_not_called()
        self.assertEqual(out["artifact"]["kind"], "progress")
        self.assertEqual(out["artifact"]["data"], empty)
        self.assertIn("No activity yet", out["answer"])

    def test_cheatsheet_tolerates_persistence_off(self):
        """db.save returns None when persistence is off — the artifact id is None, no crash."""
        with mock.patch.object(chat_tools, "_subject_content", return_value=("CONTENT", ["a.pdf"])), \
             mock.patch.object(chat_tools.cheatsheet_service, "make_cheatsheet_ex", return_value=("# Cheat", None)), \
             mock.patch.object(chat_tools.db, "save", return_value=None):
            out = chat_tools.run_tool(_subject_session(), {"tool": "cheatsheet", "topic": None, "count": None, "days": None}, "m")
        self.assertIsNone(out["artifact"]["id"])
        self.assertEqual(out["artifact"]["kind"], "cheatsheets")


class RecentArtifactTests(unittest.TestCase):
    """After a successful create, run_tool RETURNS a compact text rendition (the caller persists it) so later (non-tool) chat turns can DISCUSS the artifact."""

    def test_quiz_records_recent_artifact_with_question_and_answer(self):
        qs = [{
            "question": "What does a B-tree index optimize?",
            "options": {"A": "writes", "B": "range scans", "C": "joins", "D": "locks"},
            "correct": "B", "explanation": "B-trees keep keys sorted for range lookups.",
        }]
        session = _subject_session()
        with mock.patch.object(chat_tools, "_subject_content", return_value=("CONTENT", ["a.pdf"])), \
             mock.patch.object(chat_tools.quiz_service, "make_quiz", return_value={"questions": qs}), \
             mock.patch.object(chat_tools.db, "save", return_value="id1"):
            out = chat_tools.run_tool(session, {"tool": "quiz", "topic": None, "count": 5, "days": None}, "m")
        art = out["recent_artifact"]
        self.assertEqual(art["title"], "Quiz · DBMS")
        self.assertEqual(art["kind"], "quizzes")
        self.assertIn("What does a B-tree index optimize?", art["text"])
        self.assertIn("Correct answer: B. range scans", art["text"])
        self.assertIn("B-trees keep keys sorted", art["text"])  # explanation captured

    def test_cheatsheet_records_markdown_rendition(self):
        session = _subject_session()
        with mock.patch.object(chat_tools, "_subject_content", return_value=("CONTENT", ["a.pdf"])), \
             mock.patch.object(chat_tools.cheatsheet_service, "make_cheatsheet_ex",
                               return_value=("# Normalization\n- 1NF\n- 2NF\n- 3NF", None)), \
             mock.patch.object(chat_tools.db, "save", return_value="cs1"):
            out = chat_tools.run_tool(session, {"tool": "cheatsheet", "topic": None, "count": None, "days": None}, "m")
        art = out["recent_artifact"]
        self.assertEqual(art["kind"], "cheatsheets")
        self.assertIn("# Normalization", art["text"])
        self.assertIn("- 2NF", art["text"])

    def test_exam_records_recent_artifact_with_correct_answers(self):
        questions = [
            {"id": "m1", "type": "mcq", "question": "Which is ACID's I?",
             "options": ["Atomicity", "Isolation", "Indexing"], "answer_index": 1,
             "explanation": "Isolation is the I in ACID."},
            {"id": "s1", "type": "short", "question": "Define a deadlock.",
             "expected_points": ["circular wait", "resource hold"]},
        ]
        session = _subject_session()
        with mock.patch.object(chat_tools, "_subject_content", return_value=("CONTENT", ["a.pdf"])), \
             mock.patch.object(chat_tools.exam_service, "generate_exam", return_value={"questions": questions}), \
             mock.patch.object(chat_tools.db, "save", return_value=None):
            out = chat_tools.run_tool(session, {"tool": "exam", "topic": None, "count": None, "days": None}, "m")
        art = out["recent_artifact"]
        self.assertEqual(art["title"], "Mock Exam · DBMS")
        self.assertEqual(art["kind"], "exams")
        self.assertIn("Which is ACID's I?", art["text"])
        self.assertIn("Correct answer: B. Isolation", art["text"])
        self.assertIn("Define a deadlock.", art["text"])
        self.assertIn("circular wait", art["text"])

    def test_failed_create_leaves_no_recent_artifact(self):
        """A readable failure (no artifact) must not stamp a recent_artifact."""
        session = _topic_session()
        out = chat_tools.run_tool(session, {"tool": "summary", "topic": None, "count": None, "days": None}, "m")
        self.assertIsNone(out["artifact"])
        self.assertIsNone(out["recent_artifact"])


class InlineDataTests(unittest.TestCase):
    """Every artifact carries an inline `data` payload matching its kind so the
    chat thread can render a live interactive widget (not just a link card)."""

    def test_summary_artifact_carries_data_summaries(self):
        summaries = {"a.pdf": "summary text"}
        with mock.patch.object(chat_tools, "list_pdf_names", return_value=["a.pdf"]), \
             mock.patch.object(chat_tools.planner_service, "summarize_pdfs_ex", return_value=(summaries, None)), \
             mock.patch.object(chat_tools.db, "save", return_value="sm1"):
            out = chat_tools.run_tool(_subject_session(), {"tool": "summary", "topic": None, "count": None, "days": None}, "m")
        self.assertEqual(out["artifact"]["kind"], "summaries")
        self.assertEqual(out["artifact"]["data"], {"summaries": summaries})

    def test_mindmap_artifact_carries_data_root(self):
        root = {"label": "Recursion", "children": [{"label": "Base case", "children": []}]}
        with mock.patch.object(chat_tools.mindmap_service, "generate_mindmap_ex", return_value=(root, None)), \
             mock.patch.object(chat_tools.db, "save", return_value="mm1"):
            out = chat_tools.run_tool(_topic_session(), {"tool": "mindmap", "topic": None, "count": None, "days": None}, "m")
        self.assertEqual(out["artifact"]["kind"], "mindmaps")
        self.assertEqual(out["artifact"]["data"], {"root": root})

    def test_quiz_data_is_the_same_object_saved_to_history(self):
        """The inline payload IS the renderable content produced when saving — not a re-fetch."""
        qs = [{"question": "Q1", "options": ["a", "b", "c", "d"], "answer_index": 0, "explanation": "e"}]
        with mock.patch.object(chat_tools, "_subject_content", return_value=("CONTENT", ["a.pdf"])), \
             mock.patch.object(chat_tools.quiz_service, "make_quiz", return_value={"questions": qs}), \
             mock.patch.object(chat_tools.db, "save", return_value="id1") as save:
            out = chat_tools.run_tool(_subject_session(), {"tool": "quiz", "topic": None, "count": 5, "days": None}, "m")
        saved_doc = save.call_args.args[2]
        self.assertEqual(out["artifact"]["data"]["questions"], saved_doc["questions"])


class SubjectLockTests(unittest.TestCase):
    """HARD RULE 1: artifacts are generated from the CURRENT SESSION SUBJECT ONLY.
    A Mathematics session must never feed another subject into a tool service."""

    def test_quiz_generates_from_session_subject_only(self):
        """make_quiz is fed the session-subject's own content; no other subject is touched."""
        with mock.patch.object(chat_tools, "_subject_content", return_value=("MATH-CONTENT", ["m.pdf"])) as content, \
             mock.patch.object(chat_tools.quiz_service, "make_quiz",
                               return_value={"questions": [{"question": "q", "options": [], "answer_index": 0, "explanation": ""}]}), \
             mock.patch.object(chat_tools.quiz_service, "make_quiz_from_topic") as from_topic, \
             mock.patch.object(chat_tools.db, "save", return_value="q1"):
            out = chat_tools.run_tool(
                _subject_session(subject="Mathematics"),
                {"tool": "quiz", "topic": None, "count": 5, "days": None}, "m",
            )
        # the content blob was sourced from THIS subject, with THIS owner
        content.assert_called_once_with("Mathematics", "u1")
        # subject sessions never route through the topic generator (no cross-subject path)
        from_topic.assert_not_called()
        self.assertEqual(out["artifact"]["title"], "Quiz · Mathematics")

    def test_cheatsheet_never_uses_a_topic_override_in_a_subject_session(self):
        """Even if the intent carries a stray topic, a subject session condenses the SUBJECT's PDFs."""
        with mock.patch.object(chat_tools, "_subject_content", return_value=("MATH-CONTENT", ["m.pdf"])) as content, \
             mock.patch.object(chat_tools.cheatsheet_service, "make_cheatsheet_ex", return_value=("# Cheat", None)) as mk, \
             mock.patch.object(chat_tools.db, "save", return_value="cs1"):
            out = chat_tools.run_tool(
                _subject_session(subject="Mathematics"),
                {"tool": "cheatsheet", "topic": "Operating Systems", "count": None, "days": None}, "m",
            )
        content.assert_called_once_with("Mathematics", "u1")  # the session subject, not the topic
        mk.assert_called_once_with("MATH-CONTENT")
        self.assertEqual(out["artifact"]["title"], "Cheat Sheet · Mathematics")

    def test_predictor_passes_session_subject_as_hint(self):
        """predict_questions is hinted with the session subject only."""
        with mock.patch.object(chat_tools, "_subject_papers_corpus", return_value=("CORPUS", ["p1.pdf"])) as corpus, \
             mock.patch.object(chat_tools.predictor_service, "predict_questions",
                               return_value={"predicted_questions": [{"question": "q"}], "patterns": ["p"]}) as mk, \
             mock.patch.object(chat_tools.db, "save", return_value="pr1"):
            chat_tools.run_tool(
                _subject_session(subject="Mathematics"),
                {"tool": "predictor", "topic": None, "count": None, "days": None}, "m",
            )
        corpus.assert_called_once_with("Mathematics", "u1")
        mk.assert_called_once_with("CORPUS", subject_hint="Mathematics")


def _session(**over):
    """A signed chat Session (subject mode, owner u1, no persistence)."""
    from backend.services import chat
    return chat._mint(over.get("owner", "u1"), "subject", "L", "DBMS", None, None)


def _run(coro):
    import asyncio
    return asyncio.run(coro)


class AskWiringTests(unittest.TestCase):
    """The chat turn orchestrates tools (every session is autonomous); plain questions are pure RAG."""

    def setUp(self):
        from backend.services import chat
        self.chat = chat
        # No retriever / history DB needed for tool turns; keep persistence local + inert.
        self._p = [
            mock.patch.object(chat, "_prepare", return_value=(None, "persona")),
            mock.patch.object(chat.db, "_get_db", return_value=None),
            mock.patch.object(llm_chain, "has_any_key", return_value=True),
        ]
        for p in self._p:
            p.start()

    def tearDown(self):
        for p in self._p:
            p.stop()

    def test_autonomous_ask_returns_artifact(self):
        art = {"kind": "quizzes", "id": "x", "title": "Quiz · DBMS", "route": "#quiz", "tool": "Quiz Generator"}
        s = _session()
        with mock.patch.object(chat_tools, "adetect_intent", mock.AsyncMock(return_value=[{"tool": "quiz"}])) as detect, \
             mock.patch.object(chat_tools, "run_tool",
                               return_value={"answer": "Done", "steps": [], "artifact": art, "llm": {"label": "L"}}) as run:
            out = _run(self.chat.ask(s, "create a quiz"))
        detect.assert_awaited_once()
        run.assert_called_once()
        self.assertEqual(out["artifact"], art)
        self.assertEqual(out["artifacts"], [art])
        self.assertEqual(out["answer"], "Done")
        self.assertEqual(out["llm"], {"label": "L"})

    def test_autonomous_ask_runs_multiple_tools_in_one_turn(self):
        """A single message asking for two tools builds BOTH artifacts in one turn."""
        art1 = {"kind": "quizzes", "id": "q", "title": "Quiz · DBMS", "route": "#quiz", "tool": "Quiz Generator"}
        art2 = {"kind": "flashcards", "id": "f", "title": "Flashcards · DBMS", "route": "#flashcards", "tool": "Flashcards"}
        results = [
            {"answer": "Quiz done", "steps": [{"kind": "tool", "title": "quiz"}], "artifact": art1},
            {"answer": "Cards done", "steps": [{"kind": "tool", "title": "cards"}], "artifact": art2},
        ]
        with mock.patch.object(chat_tools, "adetect_intent",
                               mock.AsyncMock(return_value=[{"tool": "quiz"}, {"tool": "flashcards"}])), \
             mock.patch.object(chat_tools, "run_tool", side_effect=results) as run:
            out = _run(self.chat.ask(_session(), "make a quiz and flashcards"))
        self.assertEqual(run.call_count, 2)
        self.assertEqual(out["artifacts"], [art1, art2])
        self.assertEqual(out["artifact"], art1)            # legacy single field = first artifact
        self.assertEqual(out["answer"], "Quiz done\n\nCards done")
        self.assertGreaterEqual(len(out["steps"]), 2)      # each tool's trace is collected

    def test_forced_tool_skips_detection_and_runs_that_tool(self):
        """A manual tool pick (tool=...) forces that tool and bypasses intent detection."""
        art = {"kind": "flashcards", "id": "f", "title": "Flashcards · DBMS", "route": "#flashcards", "tool": "Flashcards"}
        with mock.patch.object(chat_tools, "adetect_intent", mock.AsyncMock()) as detect, \
             mock.patch.object(chat_tools, "run_tool",
                               return_value={"answer": "Cards made", "steps": [], "artifact": art}) as run:
            out = _run(self.chat.ask(_session(), "anything", tool="flashcards"))
        detect.assert_not_awaited()                         # forced → detection skipped entirely
        run.assert_called_once()
        self.assertEqual(run.call_args.args[1]["tool"], "flashcards")
        self.assertEqual(out["artifact"], art)

    def test_tool_context_is_a_plain_dict_with_owner_scope(self):
        """run_tool receives only {owner, subject, topic} — no process-global session object."""
        with mock.patch.object(chat_tools, "adetect_intent", mock.AsyncMock(return_value=None)), \
             mock.patch.object(chat_tools, "run_tool") as run, \
             mock.patch.object(llm_chain, "astream_chat", _fake_stream(["Hello ", "there"])):
            out = _run(self.chat.ask(_session(owner="alice"), "explain B-trees to me"))
        run.assert_not_called()
        self.assertEqual(out["answer"], "Hello there")
        with mock.patch.object(chat_tools, "adetect_intent", mock.AsyncMock(return_value=[{"tool": "quiz"}])), \
             mock.patch.object(chat_tools, "run_tool", return_value=None) as run2, \
             mock.patch.object(llm_chain, "astream_chat", _fake_stream(["fallback"])):
            _run(self.chat.ask(_session(owner="alice"), "make a quiz"))
        self.assertEqual(run2.call_args.args[0], {"owner": "alice", "subject": "DBMS", "topic": None})


def _fake_stream(chunks):
    async def gen(system, messages, **kw):
        for c in chunks:
            yield c
        yield llm_chain.ChatResult(text="".join(chunks), provider="groq", model="openai/gpt-oss-120b",
                                   step=2, step_name="Groq key 1")
    return gen


if __name__ == "__main__":
    unittest.main()
