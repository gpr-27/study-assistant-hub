"""Quiz service + router with the provider chain mocked at backend.llm_chain.chat."""
import unittest
from unittest import mock

from backend.routers import quiz as quiz_router
from backend.services import quiz
from tests._study_helpers import PDF_BYTES, chain_error, make_client, patch_chat

_Q = {"question": "What is 2+2?", "options": {"A": "3", "B": "4", "C": "5", "D": "6"},
      "correct": "B", "explanation": "Arithmetic."}


class ServiceTests(unittest.TestCase):
    def test_make_quiz_shape_and_llm(self):
        ctx, fake = patch_chat({"questions": [_Q, {"question": "bad"}]})
        with ctx:
            out = quiz.make_quiz("Some lecture text.", 5, "easy")
        self.assertEqual(len(out["questions"]), 1)
        self.assertEqual(out["questions"][0]["correct"], "B")
        self.assertEqual(out["llm"]["provider"], "groq")
        self.assertIn("label", out["llm"])
        self.assertTrue(fake.calls[0]["json_mode"])
        self.assertGreaterEqual(fake.calls[0]["max_tokens"], 2048)

    def test_tolerant_option_shapes_and_letter_case(self):
        raw = {"question": "q", "options": ["w", "x", "y", "z"], "correct": "c) y", "explanation": "e"}
        ctx, _ = patch_chat({"questions": [raw]})
        with ctx:
            out = quiz.make_quiz("text", 1, "medium")
        self.assertEqual(out["questions"][0]["correct"], "C")

    def test_json_repair_via_fenced_truncated_reply(self):
        trunc = '```json\n{"questions": [' + __import__("json").dumps(_Q) + ', {"question": "cut'
        ctx, fake = patch_chat(trunc)
        with ctx:
            out = quiz.make_quiz("text", 3, "medium")
        self.assertEqual(len(out["questions"]), 1)
        self.assertEqual(len(fake.calls), 1)

    def test_unusable_output_raises_quiz_error(self):
        ctx, _ = patch_chat("nope")
        with ctx, self.assertRaises(quiz.QuizError):
            quiz.make_quiz("text", 3, "medium")

    def test_validation(self):
        with self.assertRaises(quiz.QuizError):
            quiz.make_quiz("   ", 3, "easy")
        with self.assertRaises(quiz.QuizError):
            quiz.make_quiz_from_topic("  ", 3, "easy")

    def test_count_and_difficulty_are_clamped(self):
        ctx, fake = patch_chat({"questions": [_Q] * 40})
        with ctx:
            out = quiz.make_quiz_from_topic("Algebra", 999, "'; DROP")
        self.assertLessEqual(len(out["questions"]), quiz.MAX_QUESTIONS)
        self.assertIn("medium", fake.calls[0]["messages"][0]["content"])
        self.assertNotIn("DROP", fake.calls[0]["messages"][0]["content"])

    def test_chain_error_propagates(self):
        ctx, _ = patch_chat(chain_error())
        with ctx, self.assertRaises(Exception) as cm:
            quiz.make_quiz("text", 3, "easy")
        self.assertEqual(type(cm.exception).__name__, "LLMChainError")


class RouterTests(unittest.TestCase):
    def setUp(self):
        self.client = make_client(quiz_router.router)

    def test_topic_happy_path_saves_llm(self):
        ctx, _ = patch_chat({"questions": [_Q]})
        with ctx, mock.patch.object(quiz_router.db, "save", return_value="id1") as save:
            res = self.client.post("/api/quiz/topic", json={"topic": " Algebra ", "num_questions": 1})
        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertEqual(body["id"], "id1")
        self.assertEqual(body["title"], "Quiz · Algebra")
        self.assertEqual(body["llm"]["model"], "openai/gpt-oss-120b")
        doc = save.call_args.args[2]
        self.assertEqual(doc["llm"], body["llm"])
        self.assertEqual(doc["model"], "openai/gpt-oss-120b")
        self.assertEqual(save.call_args.args[1], "tester")

    def test_generate_from_subject_pdf(self):
        ctx, _ = patch_chat({"questions": [_Q]})
        with ctx, mock.patch.object(quiz_router.db, "save", return_value=None), \
             mock.patch.object(quiz_router, "read_pdf", return_value="lecture text") as rd:
            res = self.client.post("/api/quiz/generate", json={"subject": "OS", "pdf": "a.pdf"})
        self.assertEqual(res.status_code, 200)
        rd.assert_called_once_with("OS", "a.pdf", "tester")      # owner-scoped read
        self.assertIsNone(res.json()["id"])

    def test_blank_pdf_text_is_422(self):
        with mock.patch.object(quiz_router, "read_pdf", return_value="  "):
            res = self.client.post("/api/quiz/generate", json={"subject": "OS", "pdf": "a.pdf"})
        self.assertEqual(res.status_code, 422)

    def test_validation_errors(self):
        self.assertEqual(self.client.post("/api/quiz/topic", json={"topic": ""}).status_code, 422)
        self.assertEqual(self.client.post("/api/quiz/topic", json={"topic": "   "}).status_code, 400)
        self.assertEqual(self.client.post("/api/quiz/topic", json={"topic": "x", "num_questions": 99}).status_code, 422)
        self.assertEqual(self.client.post("/api/quiz/topic", json={"topic": "x", "num_questions": 0}).status_code, 422)

    def test_unusable_model_output_is_502(self):
        ctx, _ = patch_chat("garbage")
        with ctx:
            res = self.client.post("/api/quiz/topic", json={"topic": "x"})
        self.assertEqual(res.status_code, 502)

    def test_chain_failure_is_503_with_label(self):
        ctx, _ = patch_chat(chain_error())
        with ctx:
            res = self.client.post("/api/quiz/topic", json={"topic": "x"})
        self.assertEqual(res.status_code, 503)
        self.assertEqual(res.json()["llm"]["label"], "Model: none (all providers failed)")

    def test_upload_happy_and_limits(self):
        ctx, _ = patch_chat({"questions": [_Q]})
        with ctx, mock.patch.object(quiz_router.db, "save", return_value=None), \
             mock.patch("backend.routers._common.extract_pdf_text", return_value="text"):
            ok = self.client.post("/api/quiz/upload", data={"num_questions": "1"},
                                  files={"file": ("n.pdf", PDF_BYTES, "application/pdf")})
            empty = self.client.post("/api/quiz/upload", files={"file": ("e.pdf", b"", "application/pdf")})
            notpdf = self.client.post("/api/quiz/upload", files={"file": ("e.txt", b"hello", "text/plain")})
            big = self.client.post("/api/quiz/upload", files={"file": ("b.pdf", b"%PDF" + b"0" * (4 * 1024 * 1024 + 1), "application/pdf")})
        self.assertEqual(ok.status_code, 200)
        self.assertEqual(ok.json()["title"], "Quiz · n.pdf")
        self.assertEqual(empty.status_code, 400)
        self.assertEqual(notpdf.status_code, 415)
        self.assertEqual(big.status_code, 413)


if __name__ == "__main__":
    unittest.main()
