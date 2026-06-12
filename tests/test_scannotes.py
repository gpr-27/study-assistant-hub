"""Tests for Scan Handwritten Notes — router validation + Gemini vision service.

All network calls are mocked (requests.post) — no key, network, or real model.
The router is mounted on a local FastAPI app with current_owner overridden,
since it is not yet included in backend/main.py.
"""
import io
import unittest
from unittest import mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.auth import current_owner
from backend.routers.scannotes import router
from backend.services import scannotes as service

OWNER = "test-owner"


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[current_owner] = lambda: OWNER
    return TestClient(app)


def _gemini_ok(text="# Notes\n- a point"):
    """A mocked 200 generateContent response carrying ``text``."""
    resp = mock.Mock(status_code=200)
    resp.json.return_value = {"candidates": [{"content": {"parts": [{"text": text}]}}]}
    return resp


def _files(n=1, name="notes.png", ctype="image/png", blob=b"fake-image-bytes"):
    return [("images", (name, io.BytesIO(blob), ctype)) for _ in range(n)]


def _with_key():
    return mock.patch.object(service.config, "GEMINI_API_KEY", "test-key")


class ExtractEndpointTests(unittest.TestCase):
    def setUp(self):
        self.client = _client()

    def test_extract_combines_pages_and_saves(self):
        with _with_key(), \
             mock.patch.object(service.requests, "post", return_value=_gemini_ok("hello world")) as post, \
             mock.patch("backend.db.save", return_value="abc123") as save:
            res = self.client.post("/api/scannotes/extract", files=_files(n=2))
        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertEqual(body["pages"], 2)
        self.assertEqual(body["id"], "abc123")
        self.assertEqual(body["title"], "Scan · notes.png")
        self.assertIn("## Page 1", body["markdown"])
        self.assertIn("## Page 2", body["markdown"])
        self.assertIn("hello world", body["markdown"])
        self.assertEqual(post.call_count, 2)  # one Gemini call per image
        # The image is sent as a base64 inline_data part with the right mime type.
        payload = post.call_args.kwargs["json"]
        part = payload["contents"][0]["parts"][0]["inline_data"]
        self.assertEqual(part["mime_type"], "image/png")
        self.assertTrue(part["data"])  # non-empty base64
        # An explicit output cap is always set.
        self.assertEqual(payload["generationConfig"]["maxOutputTokens"], service._MAX_OUTPUT_TOKENS)
        # History stores markdown + page count, never the images.
        kind, owner, doc = save.call_args.args
        self.assertEqual((kind, owner), ("scans", OWNER))
        self.assertEqual(doc["pages"], 2)
        self.assertIn("markdown", doc)
        self.assertNotIn("images", doc)

    def test_persistence_off_returns_null_id(self):
        with _with_key(), \
             mock.patch.object(service.requests, "post", return_value=_gemini_ok()), \
             mock.patch("backend.db.save", return_value=None):
            res = self.client.post("/api/scannotes/extract", files=_files())
        self.assertEqual(res.status_code, 200)
        self.assertIsNone(res.json()["id"])

    def test_missing_gemini_key_is_503(self):
        with mock.patch.object(service.config, "GEMINI_API_KEY", None):
            res = self.client.post("/api/scannotes/extract", files=_files())
        self.assertEqual(res.status_code, 503)
        self.assertIn("GEMINI_API_KEY", res.json()["detail"])

    def test_heic_rejected_with_friendly_422(self):
        with _with_key():
            res = self.client.post(
                "/api/scannotes/extract", files=_files(name="photo.heic", ctype="image/heic")
            )
        self.assertEqual(res.status_code, 422)
        self.assertIn("HEIC", res.json()["detail"])

    def test_heic_extension_rejected_even_with_generic_mime(self):
        with _with_key():
            res = self.client.post(
                "/api/scannotes/extract",
                files=_files(name="photo.HEIC", ctype="application/octet-stream"),
            )
        self.assertEqual(res.status_code, 422)
        self.assertIn("HEIC", res.json()["detail"])

    def test_unsupported_type_rejected_422(self):
        with _with_key():
            res = self.client.post(
                "/api/scannotes/extract", files=_files(name="anim.gif", ctype="image/gif")
            )
        self.assertEqual(res.status_code, 422)
        self.assertIn("JPEG, PNG, or WebP", res.json()["detail"])

    def test_too_many_images_rejected_422(self):
        with _with_key():
            res = self.client.post("/api/scannotes/extract", files=_files(n=7))
        self.assertEqual(res.status_code, 422)
        self.assertIn("up to 6", res.json()["detail"])

    def test_oversize_image_rejected_422(self):
        big = b"x" * (8 * 1024 * 1024 + 1)
        with _with_key():
            res = self.client.post("/api/scannotes/extract", files=_files(blob=big))
        self.assertEqual(res.status_code, 422)
        self.assertIn("8 MB", res.json()["detail"])

    def test_gemini_error_maps_to_502(self):
        bad = mock.Mock(status_code=400, text="bad request payload")
        with _with_key(), mock.patch.object(service.requests, "post", return_value=bad):
            res = self.client.post("/api/scannotes/extract", files=_files())
        self.assertEqual(res.status_code, 502)
        self.assertIn("400", res.json()["detail"])

    def test_empty_candidates_maps_to_502(self):
        empty = mock.Mock(status_code=200)
        empty.json.return_value = {}
        with _with_key(), mock.patch.object(service.requests, "post", return_value=empty):
            res = self.client.post("/api/scannotes/extract", files=_files())
        self.assertEqual(res.status_code, 502)
        self.assertIn("no transcription", res.json()["detail"])


class ServiceTests(unittest.TestCase):
    def test_vision_model_default_and_env_override(self):
        with mock.patch.dict("os.environ", {}, clear=False):
            import os
            os.environ.pop("GEMINI_VISION_MODEL", None)
            self.assertEqual(service.vision_model(), "gemini-2.0-flash")
        with mock.patch.dict("os.environ", {"GEMINI_VISION_MODEL": "gemini-9.9-pro"}):
            self.assertEqual(service.vision_model(), "gemini-9.9-pro")

    def test_transcribe_images_orders_page_headers(self):
        responses = [_gemini_ok("first page"), _gemini_ok("second page")]
        with _with_key(), mock.patch.object(service.requests, "post", side_effect=responses):
            out = service.transcribe_images([(b"a", "image/png"), (b"b", "image/jpeg")])
        self.assertTrue(out.startswith("## Page 1\n\nfirst page"))
        self.assertIn("## Page 2\n\nsecond page", out)

    def test_blocked_response_raises_readable_error(self):
        blocked = mock.Mock(status_code=200)
        blocked.json.return_value = {"promptFeedback": {"blockReason": "SAFETY"}}
        with _with_key(), mock.patch.object(service.requests, "post", return_value=blocked):
            with self.assertRaises(service.ScanNotesError) as ctx:
                service.transcribe_image(b"img", "image/png")
        self.assertIn("SAFETY", str(ctx.exception))

    def test_retryable_500_then_success(self):
        responses = [mock.Mock(status_code=500, text="boom"), _gemini_ok("recovered")]
        with _with_key(), \
             mock.patch.object(service.requests, "post", side_effect=responses), \
             mock.patch.object(service.time, "sleep", lambda *_: None):
            out = service.transcribe_image(b"img", "image/png")
        self.assertEqual(out, "recovered")

    def test_missing_key_raises_before_any_network(self):
        with mock.patch.object(service.config, "GEMINI_API_KEY", None), \
             mock.patch.object(service.requests, "post") as post:
            with self.assertRaises(service.MissingGeminiKeyError):
                service.transcribe_image(b"img", "image/png")
        post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
