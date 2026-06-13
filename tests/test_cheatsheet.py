"""Tests for the cheat-sheet feature: service generation + router endpoints.

The Groq client, document reads, and persistence are fully mocked — no key,
network, filesystem documents, or database are used. The router is mounted on
a throwaway FastAPI app (it is not in main.py yet) with current_owner overridden.
"""
import unittest
from unittest import mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.auth import current_owner
from backend.routers import cheatsheet as cheatsheet_router
from backend.services import cheatsheet as cheatsheet_service
from backend.services.cheatsheet import make_cheatsheet_from_topic

_SHEET_MD = "## Key definitions\n- Term — meaning\n\n## Pitfalls\n- Forgetting units"


def _mock_groq_response(content):
    return mock.Mock(choices=[mock.Mock(message=mock.Mock(content=content))])


def _patched_service(content=_SHEET_MD):
    """Patch require_groq_key + Groq inside the service; returns the Groq mock ctx."""
    groq_patch = mock.patch.object(cheatsheet_service, "Groq")
    key_patch = mock.patch.object(cheatsheet_service, "require_groq_key", lambda: None)
    return groq_patch, key_patch, content


class MakeCheatsheetTests(unittest.TestCase):
    def test_returns_stripped_markdown(self):
        groq_patch, key_patch, content = _patched_service("\n" + _SHEET_MD + "\n\n")
        with key_patch, groq_patch as mock_groq:
            mock_groq.return_value.chat.completions.create.return_value = _mock_groq_response(content)
            result = cheatsheet_service.make_cheatsheet("Some lecture text.")
        self.assertEqual(result, _SHEET_MD)

    def test_passes_conservative_max_tokens(self):
        groq_patch, key_patch, _ = _patched_service()
        with key_patch, groq_patch as mock_groq:
            create = mock_groq.return_value.chat.completions.create
            create.return_value = _mock_groq_response(_SHEET_MD)
            cheatsheet_service.make_cheatsheet("Some lecture text.")
        kwargs = create.call_args.kwargs
        self.assertEqual(kwargs["max_tokens"], cheatsheet_service._MAX_OUTPUT_TOKENS)
        self.assertLessEqual(kwargs["max_tokens"], 2048)

    def test_empty_content_raises(self):
        with self.assertRaises(cheatsheet_service.CheatsheetError):
            cheatsheet_service.make_cheatsheet("   ")

    def test_blank_model_output_raises(self):
        groq_patch, key_patch, _ = _patched_service()
        with key_patch, groq_patch as mock_groq:
            mock_groq.return_value.chat.completions.create.return_value = _mock_groq_response("  \n ")
            with self.assertRaises(cheatsheet_service.CheatsheetError):
                cheatsheet_service.make_cheatsheet("Some lecture text.")

    def test_transport_failure_wraps_into_cheatsheet_error(self):
        groq_patch, key_patch, _ = _patched_service()
        with key_patch, groq_patch as mock_groq:
            mock_groq.return_value.chat.completions.create.side_effect = RuntimeError("boom")
            with self.assertRaises(cheatsheet_service.CheatsheetError):
                cheatsheet_service.make_cheatsheet("Some lecture text.")


class MakeCheatsheetFromTopicTests(unittest.TestCase):
    def test_returns_stripped_markdown(self):
        groq_patch, key_patch, content = _patched_service("\n" + _SHEET_MD + "\n\n")
        with key_patch, groq_patch as mock_groq:
            mock_groq.return_value.chat.completions.create.return_value = _mock_groq_response(content)
            result = cheatsheet_service.make_cheatsheet_from_topic("Photosynthesis")
        self.assertEqual(result, _SHEET_MD)

    def test_empty_topic_raises(self):
        with self.assertRaises(cheatsheet_service.CheatsheetError):
            cheatsheet_service.make_cheatsheet_from_topic("   ")

    def test_blank_model_output_raises(self):
        groq_patch, key_patch, _ = _patched_service()
        with key_patch, groq_patch as mock_groq:
            mock_groq.return_value.chat.completions.create.return_value = _mock_groq_response("  \n ")
            with self.assertRaises(cheatsheet_service.CheatsheetError):
                cheatsheet_service.make_cheatsheet_from_topic("Photosynthesis")

    def test_passes_conservative_max_tokens(self):
        groq_patch, key_patch, _ = _patched_service()
        with key_patch, groq_patch as mock_groq:
            create = mock_groq.return_value.chat.completions.create
            create.return_value = _mock_groq_response(_SHEET_MD)
            cheatsheet_service.make_cheatsheet_from_topic("Photosynthesis")
        self.assertEqual(create.call_args.kwargs["max_tokens"], cheatsheet_service._MAX_OUTPUT_TOKENS)

    def test_module_level_export_is_the_same_callable(self):
        self.assertIs(make_cheatsheet_from_topic, cheatsheet_service.make_cheatsheet_from_topic)


class CheatsheetRouterTests(unittest.TestCase):
    def setUp(self):
        app = FastAPI()
        app.include_router(cheatsheet_router.router)
        app.dependency_overrides[current_owner] = lambda: "test-owner"
        self.client = TestClient(app)

    def test_generate_subject_mode(self):
        with mock.patch.object(cheatsheet_router, "require_groq_key", lambda: None), \
             mock.patch.object(cheatsheet_router, "read_pdf", return_value="Lecture text."), \
             mock.patch.object(cheatsheet_router, "make_cheatsheet", return_value=_SHEET_MD) as mk, \
             mock.patch.object(cheatsheet_router.db, "save", return_value="abc123") as save:
            resp = self.client.post(
                "/api/cheatsheet/generate",
                json={"mode": "subject", "subject": "Math", "pdf": "Lec-1.pdf"},
            )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["markdown"], _SHEET_MD)
        self.assertEqual(body["title"], "Cheat sheet · Lec-1.pdf")
        self.assertEqual(body["id"], "abc123")
        mk.assert_called_once()
        save.assert_called_once()
        kind, owner, doc = save.call_args.args
        self.assertEqual((kind, owner), ("cheatsheets", "test-owner"))
        self.assertEqual(doc["title"], "Cheat sheet · Lec-1.pdf")
        self.assertEqual(doc["markdown"], _SHEET_MD)

    def test_generate_tolerates_persistence_off(self):
        with mock.patch.object(cheatsheet_router, "require_groq_key", lambda: None), \
             mock.patch.object(cheatsheet_router, "read_pdf", return_value="Lecture text."), \
             mock.patch.object(cheatsheet_router, "make_cheatsheet", return_value=_SHEET_MD), \
             mock.patch.object(cheatsheet_router.db, "save", return_value=None):
            resp = self.client.post(
                "/api/cheatsheet/generate",
                json={"mode": "subject", "subject": "Math", "pdf": "Lec-1.pdf"},
            )
        self.assertEqual(resp.status_code, 200)
        self.assertIsNone(resp.json()["id"])

    def test_generate_requires_subject_and_pdf(self):
        resp = self.client.post("/api/cheatsheet/generate", json={"mode": "subject", "subject": "Math"})
        self.assertEqual(resp.status_code, 400)

    def test_generate_rejects_unknown_mode(self):
        resp = self.client.post("/api/cheatsheet/generate", json={"mode": "topic", "subject": "x", "pdf": "y"})
        self.assertEqual(resp.status_code, 400)

    def test_service_error_maps_to_502(self):
        with mock.patch.object(cheatsheet_router, "require_groq_key", lambda: None), \
             mock.patch.object(cheatsheet_router, "read_pdf", return_value="Lecture text."), \
             mock.patch.object(
                 cheatsheet_router, "make_cheatsheet",
                 side_effect=cheatsheet_service.CheatsheetError("no good"),
             ):
            resp = self.client.post(
                "/api/cheatsheet/generate",
                json={"mode": "subject", "subject": "Math", "pdf": "Lec-1.pdf"},
            )
        self.assertEqual(resp.status_code, 502)
        self.assertIn("no good", resp.json()["detail"])

    def test_upload_generates_from_pdf_file(self):
        with mock.patch.object(cheatsheet_router, "require_groq_key", lambda: None), \
             mock.patch.object(cheatsheet_router, "extract_pdf_text", return_value="Extracted text."), \
             mock.patch.object(cheatsheet_router, "make_cheatsheet", return_value=_SHEET_MD), \
             mock.patch.object(cheatsheet_router.db, "save", return_value="up1"):
            resp = self.client.post(
                "/api/cheatsheet/upload",
                files={"file": ("notes.pdf", b"%PDF-1.4 fake", "application/pdf")},
            )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["title"], "Cheat sheet · notes.pdf")
        self.assertEqual(body["markdown"], _SHEET_MD)

    def test_upload_rejects_empty_file(self):
        resp = self.client.post(
            "/api/cheatsheet/upload",
            files={"file": ("empty.pdf", b"", "application/pdf")},
        )
        self.assertEqual(resp.status_code, 400)

    def test_upload_rejects_pdf_without_text(self):
        with mock.patch.object(cheatsheet_router, "extract_pdf_text", return_value=""):
            resp = self.client.post(
                "/api/cheatsheet/upload",
                files={"file": ("scan.pdf", b"%PDF-1.4 fake", "application/pdf")},
            )
        self.assertEqual(resp.status_code, 422)

    def test_topic_mode(self):
        with mock.patch.object(cheatsheet_router, "require_groq_key", lambda: None), \
             mock.patch.object(cheatsheet_router, "make_cheatsheet_from_topic", return_value=_SHEET_MD) as mk, \
             mock.patch.object(cheatsheet_router.db, "save", return_value="t1") as save:
            resp = self.client.post("/api/cheatsheet/topic", json={"topic": "Photosynthesis"})
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["markdown"], _SHEET_MD)
        self.assertEqual(body["title"], "Cheat sheet · Photosynthesis")
        self.assertEqual(body["id"], "t1")
        mk.assert_called_once()
        self.assertEqual(mk.call_args.args[0], "Photosynthesis")
        kind, owner, doc = save.call_args.args
        self.assertEqual((kind, owner), ("cheatsheets", "test-owner"))
        self.assertEqual(doc["source"], "Photosynthesis")

    def test_topic_requires_a_topic(self):
        resp = self.client.post("/api/cheatsheet/topic", json={"topic": "   "})
        self.assertEqual(resp.status_code, 400)

    def test_topic_service_error_maps_to_502(self):
        with mock.patch.object(cheatsheet_router, "require_groq_key", lambda: None), \
             mock.patch.object(cheatsheet_router, "make_cheatsheet_from_topic",
                               side_effect=cheatsheet_service.CheatsheetError("topic boom")):
            resp = self.client.post("/api/cheatsheet/topic", json={"topic": "Photosynthesis"})
        self.assertEqual(resp.status_code, 502)
        self.assertIn("topic boom", resp.json()["detail"])


if __name__ == "__main__":
    unittest.main()
