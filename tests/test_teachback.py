"""Tests for the Socratic Teach-back tutor (service + router).

The Groq client, the Groq-key check, PDF reading, and db.save are all mocked —
no key, network, filesystem, or database is used. The router is mounted on a
local FastAPI app (it is not in main.py yet) with current_owner overridden.
"""
import json
import unittest
from unittest import mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.auth import current_owner
from backend.routers import teachback as teachback_router
from backend.services import teachback as tb

_OWNER = "user_test123"

_START_PAYLOAD = {
    "concept": "Binary Search",
    "opening": "Explain Binary Search to me in your own words, as if I knew nothing about it.",
}
_PROBE_PAYLOAD = {"reply": "Why must the list be sorted first?", "done": False, "evaluation": None}
_EVAL_PAYLOAD = {
    "reply": "Great session — here is my evaluation.",
    "done": True,
    "evaluation": {
        "understanding_score_0_to_10": 7,
        "strengths": ["Clear core idea"],
        "gaps": ["Edge cases unclear"],
        "suggested_review": ["Loop invariants"],
    },
}


def _mock_response(payload):
    content = payload if isinstance(payload, str) else json.dumps(payload)
    return mock.Mock(choices=[mock.Mock(message=mock.Mock(content=content))])


def _groq_returning(*payloads):
    """A mocked Groq class whose completions return the given payloads in order."""
    mock_groq = mock.Mock()
    mock_groq.return_value.chat.completions.create.side_effect = [
        _mock_response(p) for p in payloads
    ]
    return mock_groq


def _make_client() -> TestClient:
    app = FastAPI()
    app.include_router(teachback_router.router)
    app.dependency_overrides[current_owner] = lambda: _OWNER
    return TestClient(app)


class TeachbackApiTests(unittest.TestCase):
    def setUp(self):
        tb._sessions.clear()
        self.client = _make_client()
        # The router checks the Groq key before any LLM work; neutralize it here
        # (the service-level check is patched per-test alongside Groq itself).
        patcher = mock.patch.object(teachback_router, "require_groq_key", lambda: None)
        patcher.start()
        self.addCleanup(patcher.stop)
        save_patcher = mock.patch.object(tb.db, "save", return_value=None)
        self.mock_save = save_patcher.start()
        self.addCleanup(save_patcher.stop)

    def _start_topic(self, groq_mock):
        with mock.patch.object(tb, "require_groq_key", lambda: None), \
             mock.patch.object(tb, "Groq", groq_mock):
            return self.client.post("/api/teachback/start", json={"mode": "topic", "topic": "Algorithms"})

    # ---- /start ----
    def test_start_topic_returns_session_concept_and_opening(self):
        res = self._start_topic(_groq_returning(_START_PAYLOAD))
        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertTrue(body["session_id"])
        self.assertEqual(body["concept"], "Binary Search")
        self.assertIn("Explain Binary Search", body["opening"])
        self.assertIn(body["session_id"], tb._sessions)

    def test_start_topic_requires_a_topic(self):
        res = self.client.post("/api/teachback/start", json={"mode": "topic"})
        self.assertEqual(res.status_code, 400)

    def test_start_subject_reads_and_clips_the_pdf(self):
        with mock.patch.object(teachback_router, "read_pdf", return_value="Lecture text " * 50) as mock_read, \
             mock.patch.object(tb, "require_groq_key", lambda: None), \
             mock.patch.object(tb, "Groq", _groq_returning(_START_PAYLOAD)):
            res = self.client.post(
                "/api/teachback/start",
                json={"mode": "subject", "subject": "Maths", "pdf": "Lec-1.pdf"},
            )
        self.assertEqual(res.status_code, 200)
        mock_read.assert_called_once_with("Maths", "Lec-1.pdf", _OWNER)
        sid = res.json()["session_id"]
        self.assertIn("Lecture text", tb._sessions[sid]["persona"])  # grounding context kept

    def test_start_subject_requires_subject_and_pdf(self):
        res = self.client.post("/api/teachback/start", json={"mode": "subject", "subject": "Maths"})
        self.assertEqual(res.status_code, 400)

    def test_start_with_invalid_model_json_is_502(self):
        res = self._start_topic(_groq_returning("not json"))
        self.assertEqual(res.status_code, 502)

    # ---- /message ----
    def test_message_returns_probing_question_and_keeps_session(self):
        sid = self._start_topic(_groq_returning(_START_PAYLOAD)).json()["session_id"]
        with mock.patch.object(tb, "require_groq_key", lambda: None), \
             mock.patch.object(tb, "Groq", _groq_returning(_PROBE_PAYLOAD)):
            res = self.client.post("/api/teachback/message",
                                   json={"session_id": sid, "message": "It splits the list in half."})
        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertFalse(body["done"])
        self.assertIsNone(body["evaluation"])
        self.assertEqual(body["reply"], _PROBE_PAYLOAD["reply"])
        self.assertIn(sid, tb._sessions)
        self.mock_save.assert_not_called()

    def test_evaluate_message_finishes_saves_and_clears_session(self):
        sid = self._start_topic(_groq_returning(_START_PAYLOAD)).json()["session_id"]
        with mock.patch.object(tb, "require_groq_key", lambda: None), \
             mock.patch.object(tb, "Groq", _groq_returning(_EVAL_PAYLOAD)):
            res = self.client.post("/api/teachback/message",
                                   json={"session_id": sid, "message": "evaluate"})
        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertTrue(body["done"])
        self.assertEqual(body["evaluation"]["understanding_score_0_to_10"], 7)
        self.assertEqual(body["evaluation"]["strengths"], ["Clear core idea"])
        self.assertNotIn(sid, tb._sessions)  # session cleared on done
        kind, owner, doc = self.mock_save.call_args[0]
        self.assertEqual((kind, owner), ("teachbacks", _OWNER))
        self.assertEqual(doc["title"], "Teach-back · Binary Search")
        self.assertEqual(doc["evaluation"]["gaps"], ["Edge cases unclear"])
        # transcript = opening + student message + closing reply
        self.assertEqual([m["role"] for m in doc["messages"]], ["assistant", "user", "assistant"])

    def test_sixth_exchange_forces_the_evaluation(self):
        sid = self._start_topic(_groq_returning(_START_PAYLOAD)).json()["session_id"]
        tb._sessions[sid]["exchanges"] = tb._MAX_EXCHANGES - 1  # next turn must finish
        with mock.patch.object(tb, "require_groq_key", lambda: None), \
             mock.patch.object(tb, "Groq", _groq_returning(_EVAL_PAYLOAD)) as groq:
            res = self.client.post("/api/teachback/message",
                                   json={"session_id": sid, "message": "And that is why it is O(log n)."})
        self.assertEqual(res.status_code, 200)
        self.assertTrue(res.json()["done"])
        sent = groq.return_value.chat.completions.create.call_args.kwargs["messages"]
        self.assertIn("done=true", sent[-1]["content"])  # finish directive appended

    def test_forced_finish_retries_then_502_without_evaluation(self):
        sid = self._start_topic(_groq_returning(_START_PAYLOAD)).json()["session_id"]
        with mock.patch.object(tb, "require_groq_key", lambda: None), \
             mock.patch.object(tb, "Groq", _groq_returning(_PROBE_PAYLOAD, _PROBE_PAYLOAD)):
            res = self.client.post("/api/teachback/message",
                                   json={"session_id": sid, "message": "done"})
        self.assertEqual(res.status_code, 502)

    def test_done_without_evaluation_keeps_probing(self):
        sid = self._start_topic(_groq_returning(_START_PAYLOAD)).json()["session_id"]
        bogus = {"reply": "So?", "done": True, "evaluation": None}  # early done, no eval
        with mock.patch.object(tb, "require_groq_key", lambda: None), \
             mock.patch.object(tb, "Groq", _groq_returning(bogus)):
            res = self.client.post("/api/teachback/message",
                                   json={"session_id": sid, "message": "It is fast."})
        self.assertEqual(res.status_code, 200)
        self.assertFalse(res.json()["done"])
        self.assertIn(sid, tb._sessions)

    def test_unknown_session_is_404(self):
        res = self.client.post("/api/teachback/message",
                               json={"session_id": "nope", "message": "hello"})
        self.assertEqual(res.status_code, 404)

    def test_session_expires_after_50_messages_with_403(self):
        sid = self._start_topic(_groq_returning(_START_PAYLOAD)).json()["session_id"]
        tb._sessions[sid]["messages"] = [{"role": "user", "content": "x"}] * tb._MAX_MESSAGES
        res = self.client.post("/api/teachback/message",
                               json={"session_id": sid, "message": "still here"})
        self.assertEqual(res.status_code, 403)
        self.assertIn("50-message", res.json()["detail"])
        self.assertNotIn(sid, tb._sessions)

    def test_blank_message_is_400(self):
        res = self.client.post("/api/teachback/message",
                               json={"session_id": "any", "message": "   "})
        self.assertEqual(res.status_code, 400)


class StartUploadTests(unittest.TestCase):
    """The /start-upload route extracts the PDF's text and feeds it as subject-mode
    content to start_session (no service change) — mirroring subject mode."""

    def setUp(self):
        tb._sessions.clear()
        self.client = _make_client()
        patcher = mock.patch.object(teachback_router, "require_groq_key", lambda: None)
        patcher.start()
        self.addCleanup(patcher.stop)
        save_patcher = mock.patch.object(tb.db, "save", return_value=None)
        save_patcher.start()
        self.addCleanup(save_patcher.stop)

    def test_upload_starts_session_from_extracted_text(self):
        with mock.patch.object(teachback_router, "extract_pdf_text", return_value="Lecture text " * 50) as extract, \
             mock.patch.object(tb, "require_groq_key", lambda: None), \
             mock.patch.object(tb, "Groq", _groq_returning(_START_PAYLOAD)):
            res = self.client.post(
                "/api/teachback/start-upload",
                files={"file": ("notes.pdf", b"%PDF-1.4 fake", "application/pdf")},
            )
        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertTrue(body["session_id"])
        self.assertEqual(body["concept"], "Binary Search")
        extract.assert_called_once()
        sid = body["session_id"]
        self.assertIn(sid, tb._sessions)
        # uploaded text is grounding context, like subject mode; label is the filename
        self.assertIn("Lecture text", tb._sessions[sid]["persona"])
        self.assertEqual(tb._sessions[sid]["label"], "notes.pdf")

    def test_upload_rejects_empty_file(self):
        res = self.client.post(
            "/api/teachback/start-upload",
            files={"file": ("empty.pdf", b"", "application/pdf")},
        )
        self.assertEqual(res.status_code, 400)

    def test_upload_rejects_pdf_without_text(self):
        with mock.patch.object(teachback_router, "extract_pdf_text", return_value=""):
            res = self.client.post(
                "/api/teachback/start-upload",
                files={"file": ("scan.pdf", b"%PDF-1.4 fake", "application/pdf")},
            )
        self.assertEqual(res.status_code, 422)


class EvaluationNormalisationTests(unittest.TestCase):
    def test_score_is_clamped_and_lists_cleaned(self):
        ev = tb._normalise_evaluation({
            "understanding_score_0_to_10": 99,
            "strengths": ["  ok  ", "", 42],
            "gaps": "not a list",
            "suggested_review": None,
        })
        self.assertEqual(ev["understanding_score_0_to_10"], 10)
        self.assertEqual(ev["strengths"], ["ok", "42"])
        self.assertEqual(ev["gaps"], [])
        self.assertEqual(ev["suggested_review"], [])

    def test_missing_or_non_numeric_score_is_unusable(self):
        self.assertIsNone(tb._normalise_evaluation({"strengths": []}))
        self.assertIsNone(tb._normalise_evaluation({"understanding_score_0_to_10": "great"}))
        self.assertIsNone(tb._normalise_evaluation("nope"))


if __name__ == "__main__":
    unittest.main()
