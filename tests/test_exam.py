"""Tests for the Mock Exam feature: service validation/generation/grading and
the router's generate + grade endpoints.

The Groq client and PDF reads are fully mocked — no key, network, or disk PDFs.
The router is mounted on a local FastAPI app (it is not yet wired into main.py)
with the ``current_owner`` dependency overridden.
"""
import json
import unittest
from unittest import mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.auth import current_owner
from backend.routers import exam as exam_router
from backend.services import exam as exam_service

_MCQ = {
    "type": "mcq",
    "question": "What does a scheduler do?",
    "options": ["Allocates CPU", "Paints pixels", "Sends email", "Parses JSON"],
    "answer_index": 0,
    "topic": "Scheduling",
}
_SHORT = {
    "type": "short",
    "question": "Explain a deadlock.",
    "expected_points": ["circular wait", "mutual exclusion"],
    "topic": "Concurrency",
}


def _mock_response(payload):
    content = payload if isinstance(payload, str) else json.dumps(payload)
    return mock.Mock(choices=[mock.Mock(message=mock.Mock(content=content))])


def _patched_groq(payload):
    """Context managers that disable the key check and stub the Groq client."""
    groq_patch = mock.patch.object(exam_service, "Groq")
    key_patch = mock.patch.object(exam_service, "require_groq_key", lambda: None)
    return groq_patch, key_patch, payload


# ---------------------------------------------------------------------------
# Service: validation.
# ---------------------------------------------------------------------------
class ValidateQuestionsTests(unittest.TestCase):
    def test_valid_mcq_and_short_get_ids(self):
        out = exam_service.validate_questions([dict(_MCQ), dict(_SHORT)])
        self.assertEqual([q["id"] for q in out], ["q1", "q2"])
        self.assertEqual(out[0]["type"], "mcq")
        self.assertEqual(out[1]["type"], "short")

    def test_preserves_valid_client_ids(self):
        out = exam_service.validate_questions([{**_MCQ, "id": "abc"}, {**_SHORT, "id": "xyz"}])
        self.assertEqual([q["id"] for q in out], ["abc", "xyz"])

    def test_duplicate_ids_are_deduplicated(self):
        out = exam_service.validate_questions([{**_MCQ, "id": "dup"}, {**_SHORT, "id": "dup"}])
        self.assertEqual(len({q["id"] for q in out}), 2)

    def test_drops_malformed_items(self):
        bad = [
            "nope",
            {"type": "mcq", "question": "x", "options": ["a", "b"], "answer_index": 0},  # 2 options
            {"type": "mcq", **_MCQ, "answer_index": 9},                                  # index out of range
            {"type": "short", "question": "y", "expected_points": []},                   # no points
            {"type": "short", "question": "  ", "expected_points": ["p"]},               # blank question
        ]
        self.assertEqual(exam_service.validate_questions(bad), [])

    def test_tolerates_string_answer_index_and_missing_type(self):
        raw = {k: v for k, v in _MCQ.items() if k != "type"}
        raw["answer_index"] = "2"
        out = exam_service.validate_questions([raw])
        self.assertEqual(out[0]["answer_index"], 2)
        self.assertEqual(out[0]["type"], "mcq")

    def test_missing_topic_defaults_to_general(self):
        raw = {k: v for k, v in _SHORT.items() if k != "topic"}
        out = exam_service.validate_questions([raw])
        self.assertEqual(out[0]["topic"], "General")


# ---------------------------------------------------------------------------
# Service: generation.
# ---------------------------------------------------------------------------
class GenerateExamTests(unittest.TestCase):
    def test_returns_validated_questions(self):
        groq_patch, key_patch, payload = _patched_groq({"questions": [_MCQ, _SHORT, "garbage"]})
        with key_patch, groq_patch as mock_groq:
            mock_groq.return_value.chat.completions.create.return_value = _mock_response(payload)
            data = exam_service.generate_exam("Some lecture text.", num_mcq=2, num_short=2)
        kinds = [q["type"] for q in data["questions"]]
        self.assertEqual(kinds, ["mcq", "short"])

    def test_caps_question_counts_per_type(self):
        many = {"questions": [dict(_MCQ) for _ in range(8)] + [dict(_SHORT) for _ in range(8)]}
        groq_patch, key_patch, payload = _patched_groq(many)
        with key_patch, groq_patch as mock_groq:
            mock_groq.return_value.chat.completions.create.return_value = _mock_response(payload)
            data = exam_service.generate_exam("text", num_mcq=3, num_short=1)
        self.assertEqual(sum(q["type"] == "mcq" for q in data["questions"]), 3)
        self.assertEqual(sum(q["type"] == "short" for q in data["questions"]), 1)

    def test_invalid_json_raises_exam_error(self):
        groq_patch, key_patch, payload = _patched_groq("not json")
        with key_patch, groq_patch as mock_groq:
            mock_groq.return_value.chat.completions.create.return_value = _mock_response(payload)
            with self.assertRaises(exam_service.ExamError):
                exam_service.generate_exam("text")

    def test_empty_content_raises(self):
        with self.assertRaises(exam_service.ExamError):
            exam_service.generate_exam("   ")

    def test_passes_conservative_max_tokens(self):
        groq_patch, key_patch, payload = _patched_groq({"questions": [_MCQ]})
        with key_patch, groq_patch as mock_groq:
            create = mock_groq.return_value.chat.completions.create
            create.return_value = _mock_response(payload)
            exam_service.generate_exam("text", num_mcq=10, num_short=4)
        kwargs = create.call_args.kwargs
        self.assertIn("max_tokens", kwargs)
        self.assertLessEqual(kwargs["max_tokens"], 2048)


# ---------------------------------------------------------------------------
# Service: short-answer grading.
# ---------------------------------------------------------------------------
class GradeShortAnswersTests(unittest.TestCase):
    _ITEMS = [
        {"id": "s1", "question": "Q1", "expected_points": ["a"], "answer": "my answer"},
        {"id": "s2", "question": "Q2", "expected_points": ["b"], "answer": "another"},
    ]

    def test_parses_scores_and_feedback(self):
        grades = {"grades": [
            {"id": "s1", "score_0_to_5": 4, "feedback": "Good."},
            {"id": "s2", "score_0_to_5": "2", "feedback": "Partial."},
        ]}
        groq_patch, key_patch, payload = _patched_groq(grades)
        with key_patch, groq_patch as mock_groq:
            mock_groq.return_value.chat.completions.create.return_value = _mock_response(payload)
            out = exam_service.grade_short_answers(self._ITEMS)
        self.assertEqual(out["s1"], {"score": 4, "feedback": "Good."})
        self.assertEqual(out["s2"]["score"], 2)

    def test_clamps_out_of_range_scores_and_fills_missing_ids(self):
        grades = {"grades": [{"id": "s1", "score_0_to_5": 11, "feedback": "Over."}]}
        groq_patch, key_patch, payload = _patched_groq(grades)
        with key_patch, groq_patch as mock_groq:
            mock_groq.return_value.chat.completions.create.return_value = _mock_response(payload)
            out = exam_service.grade_short_answers(self._ITEMS)
        self.assertEqual(out["s1"]["score"], 5)
        self.assertEqual(out["s2"]["score"], 0)  # missing from model output -> safe default
        self.assertTrue(out["s2"]["feedback"])

    def test_missing_grades_list_raises(self):
        groq_patch, key_patch, payload = _patched_groq({"oops": True})
        with key_patch, groq_patch as mock_groq:
            mock_groq.return_value.chat.completions.create.return_value = _mock_response(payload)
            with self.assertRaises(exam_service.ExamError):
                exam_service.grade_short_answers(self._ITEMS)

    def test_empty_items_short_circuits_without_groq(self):
        with mock.patch.object(exam_service, "Groq") as mock_groq:
            self.assertEqual(exam_service.grade_short_answers([]), {})
            mock_groq.assert_not_called()


# ---------------------------------------------------------------------------
# Router (mounted on a local app; current_owner overridden).
# ---------------------------------------------------------------------------
def _client() -> TestClient:
    app = FastAPI()
    app.include_router(exam_router.router)
    app.dependency_overrides[current_owner] = lambda: "tester"
    return TestClient(app)


class GenerateEndpointTests(unittest.TestCase):
    def setUp(self):
        self.client = _client()

    def test_rejects_non_subject_mode(self):
        res = self.client.post("/api/exam/generate", json={"mode": "upload", "subject": "OS", "pdfs": ["a.pdf"]})
        self.assertEqual(res.status_code, 400)

    def test_rejects_empty_pdfs(self):
        res = self.client.post("/api/exam/generate", json={"subject": "OS", "pdfs": []})
        self.assertEqual(res.status_code, 400)

    def test_rejects_counts_over_caps(self):
        res = self.client.post("/api/exam/generate",
                               json={"subject": "OS", "pdfs": ["a.pdf"], "num_mcq": 11})
        self.assertEqual(res.status_code, 422)  # pydantic Field(le=10)

    def test_generate_happy_path(self):
        questions = exam_service.validate_questions([dict(_MCQ), dict(_SHORT)])
        with mock.patch.object(exam_router, "require_groq_key", lambda: None), \
             mock.patch.object(exam_router, "read_pdf", return_value="lecture text") as read, \
             mock.patch.object(exam_router.exam_service, "generate_exam",
                               return_value={"questions": questions}) as gen:
            res = self.client.post("/api/exam/generate", json={
                "subject": "OS", "pdfs": ["a.pdf", "b.pdf"], "num_mcq": 1, "num_short": 1,
            })
        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertEqual(body["title"], "Exam · OS")
        self.assertEqual(len(body["questions"]), 2)
        self.assertEqual(read.call_count, 2)  # one read per selected PDF
        self.assertEqual(gen.call_args.kwargs["num_mcq"], 1)

    def test_generate_502_when_no_valid_questions(self):
        with mock.patch.object(exam_router, "require_groq_key", lambda: None), \
             mock.patch.object(exam_router, "read_pdf", return_value="lecture text"), \
             mock.patch.object(exam_router.exam_service, "generate_exam",
                               return_value={"questions": []}):
            res = self.client.post("/api/exam/generate", json={"subject": "OS", "pdfs": ["a.pdf"]})
        self.assertEqual(res.status_code, 502)

    def test_generate_502_on_exam_error(self):
        with mock.patch.object(exam_router, "require_groq_key", lambda: None), \
             mock.patch.object(exam_router, "read_pdf", return_value="lecture text"), \
             mock.patch.object(exam_router.exam_service, "generate_exam",
                               side_effect=exam_service.ExamError("model exploded")):
            res = self.client.post("/api/exam/generate", json={"subject": "OS", "pdfs": ["a.pdf"]})
        self.assertEqual(res.status_code, 502)
        self.assertIn("model exploded", res.json()["detail"])


class GradeEndpointTests(unittest.TestCase):
    def setUp(self):
        self.client = _client()
        self.questions = [
            {**_MCQ, "id": "m1"},
            {**_MCQ, "id": "m2", "topic": "Memory", "answer_index": 1},
            {**_SHORT, "id": "s1"},
        ]

    def test_grades_mcqs_in_code_and_shorts_via_service(self):
        with mock.patch.object(exam_router, "require_groq_key", lambda: None), \
             mock.patch.object(exam_router.db, "save", return_value="abc123") as save, \
             mock.patch.object(exam_router.exam_service, "grade_short_answers",
                               return_value={"s1": {"score": 4, "feedback": "Solid."}}) as grader:
            res = self.client.post("/api/exam/grade", json={
                "subject": "OS",
                "questions": self.questions,
                "answers": {"m1": 0, "m2": 0, "s1": "circular wait of processes"},
                "duration_seconds": 333,
            })
        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertEqual(body["total_score"], 1 + 0 + 4)
        self.assertEqual(body["max_score"], 1 + 1 + 5)
        self.assertEqual(body["id"], "abc123")
        self.assertEqual(body["duration_seconds"], 333)
        per = {p["id"]: p for p in body["per_question"]}
        self.assertTrue(per["m1"]["correct"])
        self.assertFalse(per["m2"]["correct"])
        self.assertEqual(per["s1"]["score"], 4)
        self.assertEqual(per["s1"]["feedback"], "Solid.")
        # one batched grading call with only the answered short
        grader.assert_called_once()
        self.assertEqual([it["id"] for it in grader.call_args.args[0]], ["s1"])
        # full report saved under kind "exams" with the contract title
        self.assertEqual(save.call_args.args[0], "exams")
        self.assertEqual(save.call_args.args[1], "tester")
        self.assertEqual(save.call_args.args[2]["title"], "Exam · OS")

    def test_topic_breakdown_aggregates_marks(self):
        with mock.patch.object(exam_router, "require_groq_key", lambda: None), \
             mock.patch.object(exam_router.db, "save", return_value=None), \
             mock.patch.object(exam_router.exam_service, "grade_short_answers",
                               return_value={"s1": {"score": 5, "feedback": "Perfect."}}):
            res = self.client.post("/api/exam/grade", json={
                "subject": "OS",
                "questions": self.questions,
                "answers": {"m1": 0, "s1": "answer"},
            })
        body = res.json()
        self.assertIsNone(body["id"])  # persistence off -> tolerated
        topics = {t["topic"]: t for t in body["topic_breakdown"]}
        self.assertEqual(topics["Scheduling"], {"topic": "Scheduling", "earned": 1, "possible": 1})
        self.assertEqual(topics["Memory"], {"topic": "Memory", "earned": 0, "possible": 1})
        self.assertEqual(topics["Concurrency"], {"topic": "Concurrency", "earned": 5, "possible": 5})

    def test_blank_short_answer_scores_zero_without_llm(self):
        with mock.patch.object(exam_router.db, "save", return_value=None), \
             mock.patch.object(exam_router, "require_groq_key") as key, \
             mock.patch.object(exam_router.exam_service, "grade_short_answers") as grader:
            res = self.client.post("/api/exam/grade", json={
                "subject": "OS",
                "questions": self.questions,
                "answers": {"m1": 0, "m2": 1, "s1": "   "},
            })
        self.assertEqual(res.status_code, 200)
        grader.assert_not_called()
        key.assert_not_called()
        body = res.json()
        per = {p["id"]: p for p in body["per_question"]}
        self.assertEqual(per["s1"]["score"], 0)
        self.assertEqual(body["total_score"], 2)

    def test_grade_400_when_no_valid_questions(self):
        res = self.client.post("/api/exam/grade", json={
            "subject": "OS", "questions": [{"type": "mcq", "question": "x"}], "answers": {},
        })
        self.assertEqual(res.status_code, 400)

    def test_grade_502_on_grader_failure(self):
        with mock.patch.object(exam_router, "require_groq_key", lambda: None), \
             mock.patch.object(exam_router.exam_service, "grade_short_answers",
                               side_effect=exam_service.ExamError("grader down")):
            res = self.client.post("/api/exam/grade", json={
                "subject": "OS", "questions": self.questions, "answers": {"s1": "text"},
            })
        self.assertEqual(res.status_code, 502)


if __name__ == "__main__":
    unittest.main()
