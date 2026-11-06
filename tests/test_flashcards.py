"""Flashcards service + router with the provider chain mocked at backend.llm_chain.chat."""
import unittest
from unittest import mock

from backend.routers import flashcards as fc_router
from backend.services import flashcards
from tests._study_helpers import PDF_BYTES, chain_error, make_client, patch_chat

_C = {"front": "Capital of France?", "back": "Paris", "topic": "Geography"}


class ServiceTests(unittest.TestCase):
    def test_shape_validation_cap_and_llm(self):
        ctx, fake = patch_chat({"cards": [_C, {"front": "x", "back": " ", "topic": "t"}, "junk", dict(_C), dict(_C)]})
        with ctx:
            out = flashcards.make_flashcards("text", 2, "easy")
        self.assertEqual(len(out["cards"]), 2)
        self.assertEqual(out["llm"]["step"], 2)
        self.assertTrue(fake.calls[0]["json_mode"])

    def test_bare_list_and_fences_are_accepted(self):
        ctx, _ = patch_chat("```json\n[" + __import__("json").dumps(_C) + "]\n```")
        with ctx:
            out = flashcards.make_flashcards_from_topic("France", 3)
        self.assertEqual(len(out["cards"]), 1)

    def test_unusable_raises_flashcard_error(self):
        ctx, _ = patch_chat({"cards": []})
        with ctx, self.assertRaises(flashcards.FlashcardError):
            flashcards.make_flashcards("text", 3)

    def test_empty_inputs(self):
        with self.assertRaises(flashcards.FlashcardError):
            flashcards.make_flashcards("  ")
        with self.assertRaises(flashcards.FlashcardError):
            flashcards.make_flashcards_from_topic("  ")

    def test_chain_error_propagates(self):
        ctx, _ = patch_chat(chain_error())
        with ctx, self.assertRaises(Exception) as cm:
            flashcards.make_flashcards("text", 3)
        self.assertEqual(type(cm.exception).__name__, "LLMChainError")


class RouterTests(unittest.TestCase):
    def setUp(self):
        self.client = make_client(fc_router.router)

    def test_topic_happy_path(self):
        ctx, _ = patch_chat({"cards": [_C]})
        with ctx, mock.patch.object(fc_router.db, "save", return_value="f1") as save:
            res = self.client.post("/api/flashcards/topic", json={"topic": "France", "num_cards": 1})
        body = res.json()
        self.assertEqual(res.status_code, 200)
        self.assertEqual((body["id"], body["title"]), ("f1", "Flashcards · France"))
        self.assertEqual(body["llm"]["label"], "Model: groq · openai/gpt-oss-120b · step 2 (Groq key 1)")
        self.assertEqual(save.call_args.args[2]["llm"], body["llm"])

    def test_subject_pdf_read_is_owner_scoped(self):
        ctx, _ = patch_chat({"cards": [_C]})
        with ctx, mock.patch.object(fc_router.db, "save", return_value=None), \
             mock.patch.object(fc_router, "read_pdf", return_value="text") as rd:
            res = self.client.post("/api/flashcards/generate", json={"subject": "S", "pdf": "p.pdf"})
        self.assertEqual(res.status_code, 200)
        rd.assert_called_once_with("S", "p.pdf", "tester")

    def test_validation(self):
        post = self.client.post
        self.assertEqual(post("/api/flashcards/topic", json={"topic": "x", "num_cards": 31}).status_code, 422)
        self.assertEqual(post("/api/flashcards/topic", json={"topic": "  "}).status_code, 400)

    def test_failures(self):
        ctx, _ = patch_chat("junk")
        with ctx:
            self.assertEqual(self.client.post("/api/flashcards/topic", json={"topic": "x"}).status_code, 502)
        ctx, _ = patch_chat(chain_error())
        with ctx:
            self.assertEqual(self.client.post("/api/flashcards/topic", json={"topic": "x"}).status_code, 503)

    def test_upload(self):
        ctx, _ = patch_chat({"cards": [_C]})
        with ctx, mock.patch.object(fc_router.db, "save", return_value=None), \
             mock.patch("backend.routers._common.extract_pdf_text", return_value="text"):
            ok = self.client.post("/api/flashcards/upload", files={"file": ("n.pdf", PDF_BYTES, "application/pdf")})
        self.assertEqual(ok.status_code, 200)
        self.assertEqual(ok.json()["title"], "Flashcards · n.pdf")


if __name__ == "__main__":
    unittest.main()
