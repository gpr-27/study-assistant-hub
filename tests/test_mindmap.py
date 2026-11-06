"""Tests for the mind-map service and router (provider chain mocked at backend.llm_chain.chat)."""
import unittest
from unittest import mock

from backend.routers import mindmap as mm_router
from backend.services import mindmap
from tests._study_helpers import PDF_BYTES, chain_error, make_client, patch_chat

_FLAT = {
    "central": "Model Title",
    "branches": [
        {"title": "Branch A", "points": ["p1", "p2"]},
        {"title": "Branch B", "points": ["p3"]},
    ],
}


class TreeFromPayloadTests(unittest.TestCase):
    def test_builds_three_level_tree(self):
        root = mindmap._tree_from_payload(_FLAT, "fallback")
        self.assertEqual(root["label"], "Model Title")
        self.assertEqual([b["label"] for b in root["children"]], ["Branch A", "Branch B"])
        self.assertEqual([leaf["label"] for leaf in root["children"][0]["children"]], ["p1", "p2"])

    def test_uses_fallback_when_central_missing(self):
        root = mindmap._tree_from_payload({"branches": [{"title": "X", "points": []}]}, "Fallback")
        self.assertEqual(root["label"], "Fallback")

    def test_drops_malformed_branches_and_empty_titles(self):
        raw = {"central": "C", "branches": ["nope", {"title": "  ", "points": ["x"]}, {"title": "Good", "points": []}]}
        root = mindmap._tree_from_payload(raw, "f")
        self.assertEqual([b["label"] for b in root["children"]], ["Good"])

    def test_caps_branches_and_points(self):
        raw = {"central": "C", "branches": [{"title": f"b{i}", "points": [f"p{j}" for j in range(20)]} for i in range(20)]}
        root = mindmap._tree_from_payload(raw, "f")
        self.assertLessEqual(len(root["children"]), mindmap._MAX_BRANCHES)
        self.assertLessEqual(len(root["children"][0]["children"]), mindmap._MAX_POINTS)


class GenerateMindmapTests(unittest.TestCase):
    def test_topic_mode_uses_the_users_wording_for_root(self):
        ctx, fake = patch_chat(_FLAT)
        with ctx:
            root, llm = mindmap.generate_mindmap_ex(topic="My Topic")
        self.assertEqual(root["label"], "My Topic")
        self.assertEqual(root["children"][0]["label"], "Branch A")
        self.assertEqual(llm["provider"], "groq")
        self.assertTrue(fake.calls[0]["json_mode"])

    def test_content_mode_keeps_model_title_and_legacy_wrapper(self):
        ctx, _ = patch_chat(_FLAT)
        with ctx:
            root = mindmap.generate_mindmap(content="Some lecture text about cells.")
        self.assertEqual(root["label"], "Model Title")

    def test_fenced_json_is_parsed(self):
        import json
        ctx, fake = patch_chat("```json\n" + json.dumps(_FLAT) + "\n```")
        with ctx:
            root = mindmap.generate_mindmap(topic="x")
        self.assertEqual(len(root["children"]), 2)
        self.assertEqual(len(fake.calls), 1)

    def test_invalid_json_raises_mindmap_error(self):
        ctx, _ = patch_chat("not json")
        with ctx, self.assertRaises(mindmap.MindmapError):
            mindmap.generate_mindmap(topic="x")

    def test_tree_without_branches_raises(self):
        ctx, _ = patch_chat({"central": "Lonely", "branches": []})
        with ctx, self.assertRaises(mindmap.MindmapError):
            mindmap.generate_mindmap(content="text")

    def test_requires_topic_or_content(self):
        with self.assertRaises(mindmap.MindmapError):
            mindmap.generate_mindmap()

    def test_chain_error_propagates(self):
        ctx, _ = patch_chat(chain_error())
        with ctx, self.assertRaises(Exception) as cm:
            mindmap.generate_mindmap(topic="x")
        self.assertEqual(type(cm.exception).__name__, "LLMChainError")


class RouterTests(unittest.TestCase):
    def setUp(self):
        self.client = make_client(mm_router.router)

    def test_topic_mode_returns_and_saves_llm(self):
        ctx, _ = patch_chat(_FLAT)
        with ctx, mock.patch.object(mm_router.db, "save", return_value="m1") as save:
            res = self.client.post("/api/mindmap/generate", json={"mode": "topic", "topic": "Cells"})
        body = res.json()
        self.assertEqual(res.status_code, 200)
        self.assertEqual((body["id"], body["root"]["label"]), ("m1", "Cells"))
        self.assertIn("label", body["llm"])
        self.assertEqual(save.call_args.args[2]["llm"], body["llm"])

    def test_subject_mode_reads_owner_scoped(self):
        ctx, _ = patch_chat(_FLAT)
        with ctx, mock.patch.object(mm_router, "read_pdf", return_value="text") as rd, \
             mock.patch.object(mm_router.db, "save", return_value=None):
            res = self.client.post("/api/mindmap/generate", json={"mode": "subject", "subject": "Bio", "pdf": "c.pdf"})
        self.assertEqual(res.status_code, 200)
        rd.assert_called_once_with("Bio", "c.pdf", "tester")

    def test_validation(self):
        post = self.client.post
        self.assertEqual(post("/api/mindmap/generate", json={"mode": "subject", "subject": "Bio"}).status_code, 400)
        self.assertEqual(post("/api/mindmap/generate", json={"mode": "topic", "topic": "  "}).status_code, 400)
        self.assertEqual(post("/api/mindmap/generate", json={"mode": "weird"}).status_code, 400)

    def test_failures(self):
        ctx, _ = patch_chat({"central": "x", "branches": []})
        with ctx:
            self.assertEqual(self.client.post("/api/mindmap/generate", json={"mode": "topic", "topic": "x"}).status_code, 502)
        ctx, _ = patch_chat(chain_error())
        with ctx:
            self.assertEqual(self.client.post("/api/mindmap/generate", json={"mode": "topic", "topic": "x"}).status_code, 503)

    def test_upload(self):
        ctx, _ = patch_chat(_FLAT)
        with ctx, mock.patch.object(mm_router.db, "save", return_value=None), \
             mock.patch("backend.routers._common.extract_pdf_text", return_value="text"):
            ok = self.client.post("/api/mindmap/upload", files={"file": ("n.pdf", PDF_BYTES, "application/pdf")})
        self.assertEqual(ok.status_code, 200)
        self.assertEqual(ok.json()["root"]["label"], "Model Title")


if __name__ == "__main__":
    unittest.main()
