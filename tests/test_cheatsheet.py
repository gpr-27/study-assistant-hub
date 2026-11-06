"""Cheat-sheet service + router with the provider chain mocked at backend.llm_chain.chat."""
import unittest
from unittest import mock

from backend.routers import cheatsheet as cs_router
from backend.services import cheatsheet as cs
from tests._study_helpers import PDF_BYTES, chain_error, make_client, patch_chat

_SHEET_MD = "## Key definitions\n- Term — meaning\n\n## Pitfalls\n- Forgetting units"


class ServiceTests(unittest.TestCase):
    def test_returns_stripped_markdown_and_llm(self):
        ctx, fake = patch_chat("\n" + _SHEET_MD + "\n\n")
        with ctx:
            md, llm = cs.make_cheatsheet_ex("Some lecture text.")
        self.assertEqual(md, _SHEET_MD)
        self.assertEqual(llm["provider"], "groq")
        self.assertFalse(fake.calls[0]["json_mode"])
        self.assertTrue(2048 <= fake.calls[0]["max_tokens"] <= 6000)

    def test_backcompat_wrappers_return_str(self):
        ctx, _ = patch_chat(_SHEET_MD)
        with ctx:
            self.assertEqual(cs.make_cheatsheet("text"), _SHEET_MD)
            self.assertEqual(cs.make_cheatsheet_from_topic("Topic"), _SHEET_MD)

    def test_outer_markdown_fence_is_stripped(self):
        ctx, _ = patch_chat("```markdown\n" + _SHEET_MD + "\n```")
        with ctx:
            self.assertEqual(cs.make_cheatsheet("text"), _SHEET_MD)

    def test_content_is_sent_in_full(self):
        ctx, fake = patch_chat(_SHEET_MD)
        with ctx:
            cs.make_cheatsheet("word " * 200_000)
        self.assertGreater(len(fake.calls[0]["messages"][0]["content"]), 1_000_000)

    def test_validation_and_blank_output(self):
        with self.assertRaises(cs.CheatsheetError):
            cs.make_cheatsheet("   ")
        with self.assertRaises(cs.CheatsheetError):
            cs.make_cheatsheet_from_topic("   ")

    def test_chain_error_propagates(self):
        ctx, _ = patch_chat(chain_error())
        with ctx, self.assertRaises(Exception) as cm:
            cs.make_cheatsheet("text")
        self.assertEqual(type(cm.exception).__name__, "LLMChainError")


class RouterTests(unittest.TestCase):
    def setUp(self):
        self.client = make_client(cs_router.router)

    def test_generate_subject_mode_saves_llm(self):
        ctx, _ = patch_chat(_SHEET_MD)
        with ctx, mock.patch.object(cs_router, "read_pdf", return_value="text") as rd, \
             mock.patch.object(cs_router.db, "save", return_value="c1") as save:
            res = self.client.post("/api/cheatsheet/generate", json={"subject": "OS", "pdf": "a.pdf"})
        body = res.json()
        self.assertEqual(res.status_code, 200)
        self.assertEqual((body["markdown"], body["id"], body["title"]), (_SHEET_MD, "c1", "Cheat sheet · a.pdf"))
        self.assertIn("label", body["llm"])
        rd.assert_called_once_with("OS", "a.pdf", "tester")
        self.assertEqual(save.call_args.args[:2], ("cheatsheets", "tester"))
        self.assertEqual(save.call_args.args[2]["llm"], body["llm"])

    def test_tolerates_persistence_off(self):
        ctx, _ = patch_chat(_SHEET_MD)
        with ctx, mock.patch.object(cs_router, "read_pdf", return_value="text"), \
             mock.patch.object(cs_router.db, "save", return_value=None):
            res = self.client.post("/api/cheatsheet/generate", json={"subject": "OS", "pdf": "a.pdf"})
        self.assertIsNone(res.json()["id"])

    def test_validation(self):
        post = self.client.post
        self.assertEqual(post("/api/cheatsheet/generate", json={"subject": "OS"}).status_code, 400)
        self.assertEqual(post("/api/cheatsheet/generate", json={"mode": "x", "subject": "a", "pdf": "b"}).status_code, 400)
        self.assertEqual(post("/api/cheatsheet/topic", json={"topic": "  "}).status_code, 400)
        with mock.patch.object(cs_router, "read_pdf", return_value=""):
            self.assertEqual(post("/api/cheatsheet/generate", json={"subject": "a", "pdf": "b"}).status_code, 422)

    def test_failures_502_and_503(self):
        ctx, _ = patch_chat("   ")
        with ctx:
            self.assertEqual(self.client.post("/api/cheatsheet/topic", json={"topic": "x"}).status_code, 502)
        ctx, _ = patch_chat(chain_error())
        with ctx:
            res = self.client.post("/api/cheatsheet/topic", json={"topic": "x"})
        self.assertEqual(res.status_code, 503)

    def test_topic_mode(self):
        ctx, _ = patch_chat(_SHEET_MD)
        with ctx, mock.patch.object(cs_router.db, "save", return_value=None):
            res = self.client.post("/api/cheatsheet/topic", json={"topic": " Thermo "})
        self.assertEqual(res.json()["title"], "Cheat sheet · Thermo")

    def test_upload(self):
        ctx, _ = patch_chat(_SHEET_MD)
        with ctx, mock.patch.object(cs_router.db, "save", return_value=None), \
             mock.patch("backend.routers._common.extract_pdf_text", return_value="text"):
            ok = self.client.post("/api/cheatsheet/upload", files={"file": ("n.pdf", PDF_BYTES, "application/pdf")})
            empty = self.client.post("/api/cheatsheet/upload", files={"file": ("e.pdf", b"", "application/pdf")})
        self.assertEqual(ok.status_code, 200)
        self.assertEqual(ok.json()["title"], "Cheat sheet · n.pdf")
        self.assertEqual(empty.status_code, 400)
        with mock.patch("backend.routers._common.extract_pdf_text", return_value=""):
            res = self.client.post("/api/cheatsheet/upload", files={"file": ("s.pdf", PDF_BYTES, "application/pdf")})
        self.assertEqual(res.status_code, 422)


if __name__ == "__main__":
    unittest.main()
