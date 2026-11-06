"""Tests for the Past-paper Question Predictor (service + router).

``llm_chain.chat``, PDF extraction, and db persistence are fully mocked — no key,
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
from backend.llm_chain import LLMChainError
from backend.services import predictor
from tests._study_helpers import FakeChat, chain_error

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


class PredictServiceTests(unittest.TestCase):
    def _run(self, payload, corpus="paper text", **kwargs):
        fake = FakeChat(payload)
        with mock.patch.object(predictor, "require_llm_key", lambda: None), \
             mock.patch("backend.llm_chain.chat", fake):
            result = predictor.predict_questions(corpus, **kwargs)
        return result, fake

    def test_valid_payload_passes_through_with_llm_label(self):
        result, _ = self._run(_GOOD)
        self.assertEqual(len(result["patterns"]), 2)
        self.assertEqual(result["patterns"][0]["topic"], "Paging")
        self.assertEqual(result["predicted_questions"][0]["likelihood"], "high")
        self.assertTrue(result["llm"]["label"].startswith("Model: groq"))

    def test_uses_json_mode_and_a_generous_output_cap(self):
        _, fake = self._run(_GOOD)
        call = fake.calls[0]
        self.assertTrue(call["json_mode"])
        self.assertEqual(call["max_tokens"], predictor._MAX_OUTPUT_TOKENS)
        self.assertGreaterEqual(call["max_tokens"], 2048)

    def test_subject_hint_lands_in_prompt(self):
        _, fake = self._run(_GOOD, subject_hint="Operating Systems")
        self.assertIn("Operating Systems", fake.calls[0]["messages"][0]["content"])

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

    def test_truncated_json_is_salvaged(self):
        cut = json.dumps(_GOOD)[:-3]          # model hit the token cap mid-object
        result, _ = self._run(cut)
        self.assertGreaterEqual(len(result["predicted_questions"]), 1)

    def test_no_usable_predictions_raises(self):
        with self.assertRaises(predictor.PredictorError):
            self._run({"patterns": [], "predicted_questions": []})

    def test_empty_corpus_raises_without_calling_the_chain(self):
        fake = FakeChat("{}")
        with mock.patch("backend.llm_chain.chat", fake), self.assertRaises(predictor.PredictorError):
            predictor.predict_questions("   ")
        self.assertEqual(fake.calls, [])

    def test_chain_failure_propagates(self):
        with self.assertRaises(LLMChainError):
            self._run(chain_error())


class AnalyzeRouteTests(unittest.TestCase):
    def setUp(self):
        app = FastAPI()
        app.include_router(predictor_router.router)
        app.dependency_overrides[current_owner] = lambda: "user_test"
        self._key = mock.patch.object(predictor_router, "require_llm_key", lambda: None)
        self._key.start()
        self.addCleanup(self._key.stop)
        self.client = TestClient(app)

    def _post(self, n_files=1, subject_hint=None):
        files = [("files", (f"paper{i + 1}.pdf", b"%PDF-1.4 fake", "application/pdf")) for i in range(n_files)]
        data = {}
        if subject_hint is not None:
            data["subject_hint"] = subject_hint
        return self.client.post("/api/predictor/analyze", files=files, data=data)

    def test_analyze_happy_path_saves_with_hint_title(self):
        with mock.patch.object(predictor_router, "extract_pdf_text", return_value="exam text"), \
             mock.patch.object(predictor_router, "predict_questions", return_value={**_GOOD, "llm": {"label": "Model: groq · x"}}) as svc, \
             mock.patch.object(predictor_router.db, "save", return_value="abc123") as save:
            resp = self._post(n_files=2, subject_hint="Operating Systems")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["id"], "abc123")
        self.assertEqual(body["llm"]["label"], "Model: groq · x")
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
        self.assertEqual(doc["llm"]["label"], "Model: groq · x")

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

    def test_all_providers_failed_is_503_not_500(self):
        from tests._study_helpers import make_client
        client = make_client(predictor_router.router, owner="u")
        with mock.patch.object(predictor_router, "extract_pdf_text", return_value="exam text"), \
             mock.patch.object(predictor_router, "predict_questions", side_effect=chain_error()):
            resp = client.post("/api/predictor/analyze",
                               files=[("files", ("p.pdf", b"%PDF-1.4 fake", "application/pdf"))])
        self.assertEqual(resp.status_code, 503)
        self.assertEqual(resp.json()["llm"]["label"], "Model: none (all providers failed)")

    def test_oversized_paper_is_413(self):
        big = b"%PDF-" + b"0" * (predictor_router._MAX_BYTES + 10)
        resp = self.client.post("/api/predictor/analyze", files=[("files", ("big.pdf", big, "application/pdf"))])
        self.assertEqual(resp.status_code, 413)

    def test_corrupt_pdf_is_422_not_500(self):
        with mock.patch.object(predictor_router, "extract_pdf_text", side_effect=ValueError("bad pdf")):
            resp = self._post()
        self.assertEqual(resp.status_code, 422)


if __name__ == "__main__":
    unittest.main()
