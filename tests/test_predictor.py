"""Tests for the Past-paper Question Predictor (service + router).

The Groq client, PDF extraction, and db persistence are fully mocked — no key,
network, file parsing, or database is used. The router is mounted on a local
FastAPI app (it is not registered in backend.main yet) with the ``current_owner``
dependency overridden, per the parallel-build test rules.
"""
import json
import unittest
from unittest import mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.auth import current_owner
from backend.routers import predictor as predictor_router
from backend.services import predictor

_GOOD = {
    "patterns": [
        {"topic": "Paging", "frequency_note": "in 3 of 4 papers", "trend": "rising"},
        {"topic": "Deadlocks", "frequency_note": "every paper", "trend": "stable"},
    ],
    "predicted_questions": [
        {"question": "Explain demand paging.", "topic": "Paging", "likelihood": "high", "reasoning": "Appears every year."},
        {"question": "Define deadlock conditions.", "topic": "Deadlocks", "likelihood": "medium", "reasoning": "Recurs often."},
    ],
}


def _mock_response(payload):
    content = payload if isinstance(payload, str) else json.dumps(payload)
    return mock.Mock(choices=[mock.Mock(message=mock.Mock(content=content))])


class PredictServiceTests(unittest.TestCase):
    def _run(self, payload, corpus="paper text", **kwargs):
        with mock.patch.object(predictor, "require_groq_key", lambda: None), \
             mock.patch.object(predictor, "Groq") as mock_groq:
            mock_groq.return_value.chat.completions.create.return_value = _mock_response(payload)
            result = predictor.predict_questions(corpus, **kwargs)
        return result, mock_groq

    def test_valid_payload_passes_through(self):
        result, _ = self._run(_GOOD)
        self.assertEqual(len(result["patterns"]), 2)
        self.assertEqual(result["patterns"][0]["topic"], "Paging")
        self.assertEqual(result["predicted_questions"][0]["likelihood"], "high")

    def test_sets_explicit_max_tokens(self):
        _, mock_groq = self._run(_GOOD)
        kwargs = mock_groq.return_value.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs["max_tokens"], predictor._MAX_OUTPUT_TOKENS)
        self.assertEqual(kwargs["response_format"], {"type": "json_object"})

    def test_subject_hint_lands_in_prompt(self):
        _, mock_groq = self._run(_GOOD, subject_hint="Operating Systems")
        messages = mock_groq.return_value.chat.completions.create.call_args.kwargs["messages"]
        self.assertIn("Operating Systems", messages[1]["content"])

    def test_unknown_likelihood_demotes_to_low(self):
        payload = {
            "patterns": [],
            "predicted_questions": [{"question": "Q?", "likelihood": "certain", "reasoning": "r"}],
        }
        result, _ = self._run(payload)
        self.assertEqual(result["predicted_questions"][0]["likelihood"], "low")

    def test_malformed_items_dropped_and_predictions_capped_at_12(self):
        payload = {
            "patterns": ["nope", {"topic": "  "}, {"topic": "Good"}],
            "predicted_questions": ["bad", {"no_question": 1}]
            + [{"question": f"Q{i}", "likelihood": "high"} for i in range(20)],
        }
        result, _ = self._run(payload)
        self.assertEqual([p["topic"] for p in result["patterns"]], ["Good"])
        self.assertEqual(len(result["predicted_questions"]), 12)

    def test_invalid_json_raises(self):
        with self.assertRaises(predictor.PredictorError):
            self._run("not json")

    def test_no_usable_predictions_raises(self):
        with self.assertRaises(predictor.PredictorError):
            self._run({"patterns": [], "predicted_questions": []})

    def test_empty_corpus_raises_without_calling_groq(self):
        with self.assertRaises(predictor.PredictorError):
            predictor.predict_questions("   ")


class AnalyzeRouteTests(unittest.TestCase):
    def setUp(self):
        app = FastAPI()
        app.include_router(predictor_router.router)
        app.dependency_overrides[current_owner] = lambda: "user_test"
        self.client = TestClient(app)

    def _post(self, n_files=1, subject_hint=None):
        files = [("files", (f"paper{i + 1}.pdf", b"%PDF-1.4 fake", "application/pdf")) for i in range(n_files)]
        data = {}
        if subject_hint is not None:
            data["subject_hint"] = subject_hint
        return self.client.post("/api/predictor/analyze", files=files, data=data)

    def test_analyze_happy_path_saves_with_hint_title(self):
        with mock.patch.object(predictor_router, "extract_pdf_text", return_value="exam text"), \
             mock.patch.object(predictor_router, "predict_questions", return_value=_GOOD) as svc, \
             mock.patch.object(predictor_router.db, "save", return_value="abc123") as save:
            resp = self._post(n_files=2, subject_hint="Operating Systems")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["id"], "abc123")
        self.assertEqual(body["title"], "Predictions · Operating Systems")
        self.assertEqual(len(body["patterns"]), 2)
        self.assertEqual(body["predicted_questions"][0]["likelihood"], "high")
        # the combined corpus contains both labelled papers
        corpus = svc.call_args.args[0]
        self.assertIn("=== PAPER: paper1.pdf ===", corpus)
        self.assertIn("=== PAPER: paper2.pdf ===", corpus)
        # saved under the "predictions" kind, scoped to the overridden owner
        kind, owner, doc = save.call_args.args
        self.assertEqual(kind, "predictions")
        self.assertEqual(owner, "user_test")
        self.assertEqual(doc["num_papers"], 2)

    def test_title_falls_back_to_first_filename(self):
        with mock.patch.object(predictor_router, "extract_pdf_text", return_value="exam text"), \
             mock.patch.object(predictor_router, "predict_questions", return_value=_GOOD), \
             mock.patch.object(predictor_router.db, "save", return_value=None):
            resp = self._post(n_files=1)
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["title"], "Predictions · paper1.pdf")
        self.assertIsNone(body["id"])  # persistence off -> id None is tolerated

    def test_422_when_no_extractable_text(self):
        with mock.patch.object(predictor_router, "extract_pdf_text", return_value=""):
            resp = self._post()
        self.assertEqual(resp.status_code, 422)
        self.assertIn("No extractable text", resp.json()["detail"])

    def test_400_when_too_many_files(self):
        resp = self._post(n_files=6)
        self.assertEqual(resp.status_code, 400)
        self.assertIn("at most 5", resp.json()["detail"])

    def test_predictor_error_maps_to_502(self):
        with mock.patch.object(predictor_router, "extract_pdf_text", return_value="exam text"), \
             mock.patch.object(predictor_router, "predict_questions",
                               side_effect=predictor.PredictorError("model went sideways")):
            resp = self._post()
        self.assertEqual(resp.status_code, 502)
        self.assertIn("model went sideways", resp.json()["detail"])


if __name__ == "__main__":
    unittest.main()
