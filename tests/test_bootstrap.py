"""GET /api/bootstrap: one boot payload, never CDN-cacheable when it carries owner data."""
import json
import unittest
from pathlib import Path
from unittest import mock

from fastapi import HTTPException
from fastapi.testclient import TestClient

from backend.main import app

ROOT = Path(__file__).resolve().parent.parent
MOD = "backend.routers.bootstrap"


def _client():
    return TestClient(app, raise_server_exceptions=False)


def _deny(_request):
    raise HTTPException(status_code=401, detail="no")


class BootstrapTests(unittest.TestCase):
    def test_anonymous_gets_config_only_and_is_not_cacheable(self):
        with mock.patch(f"{MOD}.current_owner", side_effect=_deny):
            r = _client().get("/api/bootstrap")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["owner"], False)
        self.assertIn("llmChain", body["config"])
        self.assertNotIn("subjects", body)
        self.assertEqual(r.headers["cache-control"], "private, no-store")
        self.assertIn("Authorization", r.headers["vary"])

    def test_owner_gets_everything_in_one_response_private(self):
        with mock.patch(f"{MOD}.current_owner", return_value="user_a"), \
                mock.patch(f"{MOD}.list_subjects", side_effect=lambda o=None: ["Math", "Mine"] if o else ["Math"]), \
                mock.patch(f"{MOD}.build_summary", return_value={"overall": {"total_attempts": 3}}), \
                mock.patch(f"{MOD}.db.list_items", return_value=[{"id": "c1", "recent_artifact": {"x": 1}}]), \
                mock.patch(f"{MOD}.db.is_enabled", return_value=True):
            r = _client().get("/api/bootstrap", headers={"Authorization": "Bearer x"})
        body = r.json()
        self.assertEqual(body["owner"], True)
        subj = body["subjects"]
        self.assertEqual({k: subj[k] for k in ("subjects", "bundled", "own", "hidden")},
                         {"subjects": ["Math", "Mine"], "bundled": ["Math"], "own": ["Mine"], "hidden": []})
        self.assertEqual(subj["limits"]["max_total_mb"], 0)          # no per-upload total cap
        self.assertEqual(body["progress"]["overall"]["total_attempts"], 3)
        self.assertEqual(body["chats"]["items"], [{"id": "c1"}])      # chat-internal memory trimmed
        self.assertEqual(r.headers["cache-control"], "private, no-store")
        self.assertNotIn("public", r.headers["cache-control"])
        self.assertNotIn("s-maxage", r.headers["cache-control"])
        self.assertIn("subjects;dur=", r.headers["server-timing"])

    def test_a_failing_part_is_null_and_named_not_fatal(self):
        with mock.patch(f"{MOD}.current_owner", return_value="user_a"), \
                mock.patch(f"{MOD}.list_subjects", return_value=["Math"]), \
                mock.patch(f"{MOD}.build_summary", side_effect=RuntimeError("boom")), \
                mock.patch(f"{MOD}.db.list_items", return_value=[]):
            r = _client().get("/api/bootstrap")
        body = r.json()
        self.assertEqual(r.status_code, 200)
        self.assertIsNone(body["progress"])
        self.assertEqual(body["errors"], ["progress"])
        self.assertEqual(body["subjects"]["subjects"], ["Math"])

    def test_vercel_json_never_marks_api_responses_cacheable(self):
        cfg = json.loads((ROOT / "vercel.json").read_text())
        for rule in cfg["headers"]:
            if rule["source"].startswith("/api"):
                self.fail("vercel.json must not attach cache headers to /api routes: " + rule["source"])


if __name__ == "__main__":
    unittest.main()
