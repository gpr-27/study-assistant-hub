"""Tests for Scan Handwritten Notes — router validation + vision OCR through the provider chain.

``llm_chain.achat`` is replaced by a fake (with the ``images=`` keyword the chain must offer) — no
key, network or real model. The router is mounted on a local app with current_owner overridden.
"""
import asyncio
import io
import unittest
from unittest import mock

from backend import llm_chain
from backend.services import scannotes as service
from tests._study_helpers import chain_error, make_client, result

OWNER = "test-owner"


class FakeVision:
    """Async stand-in for ``llm_chain.achat`` that records calls; replies queued per call."""

    def __init__(self, *replies, delay=0.0):
        self.replies = list(replies)
        self.calls = []
        self.active = 0
        self.max_active = 0
        self.delay = delay

    async def __call__(self, system, messages, *, max_tokens=2048, temperature=0.3, json_mode=False,
                       sensitive=False, effort="low", deadline_s=50.0, images=None):
        self.calls.append({"system": system, "messages": messages, "images": images, "max_tokens": max_tokens,
                           "deadline_s": deadline_s})
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            r = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
            if isinstance(r, Exception):
                raise r
            return r if isinstance(r, llm_chain.ChatResult) else result(r)
        finally:
            self.active -= 1


def _env(fake):
    return (mock.patch("backend.llm_chain.achat", fake), mock.patch("backend.llm_chain.has_any_key", return_value=True))


def _client():
    from backend.routers.scannotes import router
    return make_client(router, owner=OWNER)


def _files(n=1, name="notes.png", ctype="image/png", blob=b"fake-image-bytes"):
    return [("images", (name, io.BytesIO(blob), ctype)) for _ in range(n)]


class ExtractEndpointTests(unittest.TestCase):
    def setUp(self):
        self.client = _client()

    def test_extract_combines_pages_sends_images_and_saves_with_llm(self):
        fake = FakeVision("hello world")
        p1, p2 = _env(fake)
        with p1, p2, mock.patch("backend.db.save", return_value="abc123") as save:
            res = self.client.post("/api/scannotes/extract", files=_files(n=2))
        self.assertEqual(res.status_code, 200, res.text)
        body = res.json()
        self.assertEqual((body["pages"], body["id"], body["title"]), (2, "abc123", "Scan · notes.png"))
        self.assertIn("## Page 1", body["markdown"])
        self.assertIn("## Page 2", body["markdown"])
        self.assertTrue(body["llm"]["label"].startswith("Model: groq"))
        self.assertEqual(len(fake.calls), 2)                                # one chain call per image
        self.assertEqual(fake.calls[0]["images"], [(b"fake-image-bytes", "image/png")])
        self.assertGreaterEqual(fake.calls[0]["max_tokens"], 2048)
        kind, owner, doc = save.call_args.args
        self.assertEqual((kind, owner), ("scans", OWNER))
        self.assertEqual(doc["pages"], 2)
        self.assertEqual(doc["llm"]["model"], "openai/gpt-oss-120b")
        self.assertNotIn("images", doc)                                     # never persist the photos

    def test_pages_are_transcribed_concurrently_but_bounded(self):
        fake = FakeVision("x", delay=0.05)
        p1, p2 = _env(fake)
        with p1, p2, mock.patch("backend.db.save", return_value=None):
            res = self.client.post("/api/scannotes/extract", files=_files(n=6))
        self.assertEqual(res.status_code, 200)
        self.assertGreater(fake.max_active, 1)
        self.assertLessEqual(fake.max_active, service._CONCURRENCY)

    def test_persistence_off_returns_null_id(self):
        p1, p2 = _env(FakeVision("ok"))
        with p1, p2, mock.patch("backend.db.save", return_value=None):
            res = self.client.post("/api/scannotes/extract", files=_files())
        self.assertEqual(res.status_code, 200)
        self.assertIsNone(res.json()["id"])

    def test_no_provider_key_is_503(self):
        with mock.patch("backend.llm_chain.has_any_key", return_value=False):
            res = self.client.post("/api/scannotes/extract", files=_files())
        self.assertEqual(res.status_code, 503)
        self.assertIn("key", res.json()["detail"].lower())

    def test_chain_without_image_support_is_a_clear_503(self):
        async def no_images(system, messages, *, max_tokens=2048):    # no ``images`` parameter
            raise AssertionError("must not be called")

        with mock.patch("backend.llm_chain.achat", no_images), mock.patch("backend.llm_chain.has_any_key", return_value=True):
            res = self.client.post("/api/scannotes/extract", files=_files())
        self.assertEqual(res.status_code, 503)
        self.assertIn("image", res.json()["detail"].lower())

    def test_all_providers_failed_is_503_with_label(self):
        p1, p2 = _env(FakeVision(chain_error()))
        with p1, p2, mock.patch("backend.db.save") as save:
            res = self.client.post("/api/scannotes/extract", files=_files())
        self.assertEqual(res.status_code, 503)
        self.assertEqual(res.json()["llm"]["label"], "Model: none (all providers failed)")
        save.assert_not_called()

    def test_heic_rejected_with_friendly_422(self):
        p1, p2 = _env(FakeVision("x"))
        with p1, p2:
            res = self.client.post("/api/scannotes/extract", files=_files(name="photo.heic", ctype="image/heic"))
        self.assertEqual(res.status_code, 422)
        self.assertIn("HEIC", res.json()["detail"])

    def test_heic_extension_rejected_even_with_generic_mime(self):
        p1, p2 = _env(FakeVision("x"))
        with p1, p2:
            res = self.client.post("/api/scannotes/extract",
                                   files=_files(name="photo.HEIC", ctype="application/octet-stream"))
        self.assertEqual(res.status_code, 422)

    def test_unsupported_type_rejected_422(self):
        p1, p2 = _env(FakeVision("x"))
        with p1, p2:
            res = self.client.post("/api/scannotes/extract", files=_files(name="anim.gif", ctype="image/gif"))
        self.assertEqual(res.status_code, 422)
        self.assertIn("JPEG, PNG, or WebP", res.json()["detail"])

    def test_too_many_images_rejected_422(self):
        p1, p2 = _env(FakeVision("x"))
        with p1, p2:
            res = self.client.post("/api/scannotes/extract", files=_files(n=7))
        self.assertEqual(res.status_code, 422)
        self.assertIn("up to 6", res.json()["detail"])

    def test_oversize_image_rejected_422(self):
        big = b"x" * (8 * 1024 * 1024 + 1)
        p1, p2 = _env(FakeVision("x"))
        with p1, p2:
            res = self.client.post("/api/scannotes/extract", files=_files(blob=big))
        self.assertEqual(res.status_code, 422)
        self.assertIn("8 MB", res.json()["detail"])

    def test_total_payload_is_capped(self):
        blob = b"x" * (7 * 1024 * 1024)
        p1, p2 = _env(FakeVision("x"))
        with p1, p2:
            res = self.client.post("/api/scannotes/extract", files=_files(n=5, blob=blob))
        self.assertEqual(res.status_code, 413)

    def test_empty_transcription_maps_to_502(self):
        p1, p2 = _env(FakeVision("   "))
        with p1, p2:
            res = self.client.post("/api/scannotes/extract", files=_files())
        self.assertEqual(res.status_code, 502)
        self.assertIn("empty", res.json()["detail"])


class ServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_transcribe_images_orders_page_headers(self):
        fake = FakeVision("first page", "second page")
        p1, p2 = _env(fake)
        with p1, p2:
            out, llm = await service.transcribe_images([(b"a", "image/png"), (b"b", "image/jpeg")])
        self.assertTrue(out.startswith("## Page 1\n\nfirst page"))
        self.assertIn("## Page 2\n\nsecond page", out)
        self.assertEqual(llm["step"], 2)

    async def test_missing_key_raises_before_any_chain_call(self):
        fake = FakeVision("x")
        with mock.patch("backend.llm_chain.achat", fake), mock.patch("backend.llm_chain.has_any_key", return_value=False):
            with self.assertRaises(service.ScanUnavailableError):
                await service.transcribe_image(b"img", "image/png")
        self.assertEqual(fake.calls, [])

    async def test_no_images_is_an_error(self):
        with self.assertRaises(service.ScanNotesError):
            await service.transcribe_images([])


if __name__ == "__main__":
    unittest.main()
