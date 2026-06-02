"""Tests for the mind-map service: flat-payload → bounded tree + generation.

The Groq client is fully mocked — no key, network, or real model is used.
"""
import json
import unittest
from unittest import mock

from backend.services import mindmap

_FLAT = {
    "central": "Model Title",
    "branches": [
        {"title": "Branch A", "points": ["p1", "p2"]},
        {"title": "Branch B", "points": ["p3"]},
    ],
}


def _mock_response(payload):
    content = payload if isinstance(payload, str) else json.dumps(payload)
    return mock.Mock(choices=[mock.Mock(message=mock.Mock(content=content))])


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
        with mock.patch.object(mindmap, "require_groq_key", lambda: None), \
             mock.patch.object(mindmap, "Groq") as mock_groq:
            mock_groq.return_value.chat.completions.create.return_value = _mock_response(_FLAT)
            root = mindmap.generate_mindmap(topic="My Topic")
        self.assertEqual(root["label"], "My Topic")
        self.assertEqual(root["children"][0]["label"], "Branch A")

    def test_content_mode_keeps_model_title(self):
        with mock.patch.object(mindmap, "require_groq_key", lambda: None), \
             mock.patch.object(mindmap, "Groq") as mock_groq:
            mock_groq.return_value.chat.completions.create.return_value = _mock_response(_FLAT)
            root = mindmap.generate_mindmap(content="Some lecture text about cells.")
        self.assertEqual(root["label"], "Model Title")

    def test_invalid_json_raises_mindmap_error(self):
        with mock.patch.object(mindmap, "require_groq_key", lambda: None), \
             mock.patch.object(mindmap, "Groq") as mock_groq:
            mock_groq.return_value.chat.completions.create.return_value = _mock_response("not json")
            with self.assertRaises(mindmap.MindmapError):
                mindmap.generate_mindmap(topic="x")

    def test_tree_without_branches_raises(self):
        with mock.patch.object(mindmap, "require_groq_key", lambda: None), \
             mock.patch.object(mindmap, "Groq") as mock_groq:
            mock_groq.return_value.chat.completions.create.return_value = _mock_response({"central": "Lonely", "branches": []})
            with self.assertRaises(mindmap.MindmapError):
                mindmap.generate_mindmap(content="text")

    def test_requires_topic_or_content(self):
        with self.assertRaises(mindmap.MindmapError):
            mindmap.generate_mindmap()


if __name__ == "__main__":
    unittest.main()
