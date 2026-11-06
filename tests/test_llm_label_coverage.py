"""Coverage of the "which model produced this?" contract across EVERY AI-generating endpoint.

The REAL provider chain runs; only its HTTP transport is mocked (``tests.test_llm_chain.net``), so each
case exercises route -> service -> llm_chain -> response/persistence together. For each endpoint we assert:

* the response carries a top-level ``llm.label`` that starts with ``Model:`` (and names step 1 / Anthropic,
  the model our fake transport answered with),
* when the result is saved, the SAME label is returned by ``/api/history/{kind}`` (list) and
  ``/api/history/{kind}/{id}`` (reopen),
* when every provider fails, the error body carries ``Model: none (all providers failed)``,
* chat: an early ``model`` SSE event, the final ``done`` event and the ``error`` event all carry labels,
  chat-created artifacts carry their own, and a resumed chat returns each stored assistant label.

No key material is ever printed or asserted on.
"""
from __future__ import annotations

import copy
import datetime
import io
import json

import pytest
from bson import ObjectId
from fastapi.testclient import TestClient

from backend import config, db, llm_chain as lc
from backend.auth import current_owner
from backend.main import app
from backend.routers import predictor as predictor_router
from backend.services import chat as chat_service
from backend.services import insights as insights_service
from backend.services import overview as overview_service
from backend.services import planner as planner_service
from backend.services import teachback as teachback_service
from tests.test_llm_chain import CLAUDE, err, net, r_anthropic, sse_anthropic  # noqa: F401  (net is a fixture)

OWNER = "alice"
LABEL = f"Model: anthropic · {CLAUDE} · step 1 (Anthropic)"
NONE_LABEL = "Model: none (all providers failed)"


# ---------------------------------------------------------------------------------------
# In-memory persistence (db.save / list_items / get_item + the chats collection)
# ---------------------------------------------------------------------------------------
class MemDB:
    def __init__(self):
        self.docs: dict[str, dict[str, dict]] = {}

    def save(self, kind, owner, doc):
        if kind not in db.KINDS:
            return None
        i = str(ObjectId())
        self.docs.setdefault(kind, {})[i] = {**copy.deepcopy(doc), "owner": owner, "id": i,
                                             "created_at": "2030-01-01T00:00:00+00:00"}
        return i

    def list_items(self, kind, owner, q=None, limit=100):
        heavy = set(db._HEAVY_FIELDS)
        out = []
        for d in self.docs.get(kind, {}).values():
            if d["owner"] == owner:
                out.append({k: copy.deepcopy(v) for k, v in d.items() if k not in heavy and k != "owner"})
        return out

    def get_item(self, kind, owner, item_id):
        d = self.docs.get(kind, {}).get(item_id)
        if not d or d["owner"] != owner:
            return None
        return {k: copy.deepcopy(v) for k, v in d.items() if k != "owner"}

    # --- minimal pymongo-ish "chats" collection used by services/chat.py ---
    def chats_collection(self):
        outer = self

        class Coll:
            def find_one(self, flt, projection=None):
                d = outer.docs.get("chats", {}).get(str(flt["_id"]))
                if not d or d["owner"] != flt["owner"]:
                    return None
                return {"messages": copy.deepcopy(d.get("messages", [])), "recent_artifact": d.get("recent_artifact")}

            def update_one(self, flt, update, upsert=False):
                d = outer.docs.get("chats", {}).get(str(flt["_id"]))
                if d is None and upsert:       # the chats document is created lazily by the first turn
                    d = outer.docs.setdefault("chats", {})[str(flt["_id"])] = {
                        "owner": flt["owner"], "id": str(flt["_id"]), "messages": [], **update.get("$setOnInsert", {})}
                if not d or d["owner"] != flt["owner"]:
                    return
                push = update.get("$push", {}).get("messages")
                if push:
                    d.setdefault("messages", []).extend(copy.deepcopy(push["$each"]))
                for k, v in update.get("$set", {}).items():
                    d[k] = v

        return Coll()


@pytest.fixture()
def mem(monkeypatch):
    m = MemDB()
    monkeypatch.setattr(db, "save", m.save)
    monkeypatch.setattr(db, "new_chat_id", lambda: str(ObjectId()))
    monkeypatch.setattr(db, "list_items", m.list_items)
    monkeypatch.setattr(db, "get_item", m.get_item)
    monkeypatch.setattr(db, "is_enabled", lambda: True)
    monkeypatch.setattr(chat_service, "_chats_collection", m.chats_collection)
    chat_service._LOCAL.clear()
    chat_service._STATE.clear()
    return m


@pytest.fixture()
def client(net, mem, monkeypatch):
    monkeypatch.setattr(config, "PERSISTENCE_ENABLED", False)   # no real Mongo anywhere
    app.dependency_overrides[current_owner] = lambda: OWNER
    c = TestClient(app)
    yield c
    app.dependency_overrides.pop(current_owner, None)
    chat_service._LOCAL.clear()
    chat_service._STATE.clear()


def _answer(net, payload):
    """Make the chain's first step (Anthropic) answer every call with ``payload``."""
    text = payload if isinstance(payload, str) else json.dumps(payload)
    net.on(1, CLAUDE, r_anthropic(text))


# ---------------------------------------------------------------------------------------
# Fixtures for each generator's reply
# ---------------------------------------------------------------------------------------
_QUIZ = {"questions": [{"question": "2+2?", "options": {"A": "3", "B": "4", "C": "5", "D": "6"},
                        "correct": "B", "explanation": "Arithmetic."}]}
_CARDS = {"cards": [{"front": "Capital of France?", "back": "Paris", "topic": "Geo"}]}
_SHEET = "## Key terms\n- Term: meaning\n\n## Pitfalls\n- Units"
_MAP = {"central": "Algebra", "branches": [{"title": "Groups", "points": ["closure", "identity"]}]}
_PLAN = "# Study Plan\n\n## Day 1\n- read\n- practise"
_SUMMARY = "TOPICS:\n- Scheduling - Basic\nSUMMARY:\nShort."
_MCQ = {"type": "mcq", "question": "What does a scheduler do?",
        "options": ["Allocates CPU", "Paints pixels", "Sends email", "Parses JSON"], "answer_index": 0,
        "topic": "Scheduling"}
_SHORT = {"type": "short", "question": "Explain a deadlock.", "expected_points": ["circular wait"],
          "topic": "Concurrency"}
_EXAM = {"questions": [_MCQ, _SHORT]}
_GRADES = {"grades": [{"id": "s1", "score_0_to_5": 4, "feedback": "Solid."}]}
_PRED = {"patterns": [{"topic": "Paging", "frequency_note": "3 of 4", "trend": "rising"}],
         "predicted_questions": [{"question": "Explain paging.", "topic": "Paging", "likelihood": "high",
                                  "reasoning": "Every year."}]}
_START = {"concept": "Binary Search", "opening": "Explain binary search in your own words."}
_PROBE = {"reply": "Why must the list be sorted?", "done": False, "evaluation": None}
_EVAL = {"reply": "Great session.", "done": True,
         "evaluation": {"understanding_score_0_to_10": 7, "strengths": ["core idea"], "gaps": ["edge cases"],
                        "suggested_review": ["invariants"]}}
_INSIGHTS = {"headline": "Solid start!", "observations": ["You aced DBMS"],
             "recommendations": [{"action": "Take a B-tree quiz", "reason": "weak topic", "tool": "quiz"}],
             "focus_topics": ["B-trees"]}
_OVERVIEW = {"overview": "**Recursion** is self-reference.", "questions": ["What is a base case?"]}


def _future() -> str:
    return (datetime.date.today() + datetime.timedelta(days=5)).isoformat()


def _pdf(name="n.pdf"):
    return ("file", (name, b"%PDF-1.4 fake", "application/pdf"))


# (case id, (method, url, request kwargs), reply, history kind or None)
JSON_CASES = [
    ("quiz-topic", ("post", "/api/quiz/topic", {"json": {"topic": "Algebra", "num_questions": 1}}), _QUIZ, "quizzes"),
    ("quiz-generate", ("post", "/api/quiz/generate", {"json": {"subject": "OS", "pdf": "a.pdf"}}), _QUIZ, "quizzes"),
    ("quiz-upload", ("post", "/api/quiz/upload", {"files": [_pdf()]}), _QUIZ, "quizzes"),
    ("flashcards-topic", ("post", "/api/flashcards/topic", {"json": {"topic": "France", "num_cards": 1}}), _CARDS,
     "flashcards"),
    ("flashcards-generate", ("post", "/api/flashcards/generate", {"json": {"subject": "S", "pdf": "p.pdf"}}),
     _CARDS, "flashcards"),
    ("flashcards-upload", ("post", "/api/flashcards/upload", {"files": [_pdf()]}), _CARDS, "flashcards"),
    ("cheatsheet-topic", ("post", "/api/cheatsheet/topic", {"json": {"topic": "Sorting"}}), _SHEET, "cheatsheets"),
    ("cheatsheet-generate", ("post", "/api/cheatsheet/generate",
                             {"json": {"mode": "subject", "subject": "S", "pdf": "p.pdf"}}), _SHEET, "cheatsheets"),
    ("cheatsheet-upload", ("post", "/api/cheatsheet/upload", {"files": [_pdf()]}), _SHEET, "cheatsheets"),
    ("mindmap-topic", ("post", "/api/mindmap/generate", {"json": {"mode": "topic", "topic": "Algebra"}}), _MAP,
     "mindmaps"),
    ("mindmap-subject", ("post", "/api/mindmap/generate",
                         {"json": {"mode": "subject", "subject": "S", "pdf": "p.pdf"}}), _MAP, "mindmaps"),
    ("mindmap-upload", ("post", "/api/mindmap/upload", {"files": [_pdf()]}), _MAP, "mindmaps"),
    ("planner-topic", ("post", "/api/planner/topic", {"json": {"topic": "Algebra", "target_date": _future()}}),
     _PLAN, "plans"),
    ("planner-generate", ("post", "/api/planner/generate",
                          {"json": {"subject": "S", "pdfs": ["p.pdf"], "target_date": _future()}}), _PLAN, "plans"),
    ("planner-upload", ("post", "/api/planner/upload",
                        {"files": [("files", ("n.pdf", b"%PDF-1.4 fake", "application/pdf"))],
                         "data": {"target_date": _future()}}), _PLAN, "plans"),
    ("planner-summaries", ("post", "/api/planner/summaries", {"json": {"subject": "S", "pdfs": ["p.pdf"]}}),
     _SUMMARY, "summaries"),
    ("exam-topic", ("post", "/api/exam/topic", {"json": {"topic": "OS", "num_mcq": 1, "num_short": 1}}), _EXAM, None),
    ("exam-generate", ("post", "/api/exam/generate", {"json": {"subject": "OS", "pdfs": ["a.pdf"]}}), _EXAM, None),
    ("exam-upload", ("post", "/api/exam/upload", {"files": [_pdf()], "data": {"num_mcq": "1", "num_short": "1"}}),
     _EXAM, None),
    ("exam-grade", ("post", "/api/exam/grade", {"json": {
        "subject": "OS", "questions": [{**_MCQ, "id": "m1"}, {**_SHORT, "id": "s1"}],
        "answers": {"m1": 0, "s1": "circular wait of processes"}}}), _GRADES, "exams"),
    ("predictor", ("post", "/api/predictor/analyze",
                   {"files": [("files", ("p1.pdf", b"%PDF-1.4 x", "application/pdf"))]}), _PRED, "predictions"),
    ("teachback-start", ("post", "/api/teachback/start", {"json": {"mode": "topic", "topic": "Algorithms"}}), _START,
     None),
    ("teachback-start-upload", ("post", "/api/teachback/start-upload", {"files": [_pdf()]}), _START, None),
    ("scannotes", ("post", "/api/scannotes/extract",
                   {"files": [("images", ("note.png", io.BytesIO(b"fake-image"), "image/png"))]}),
     "Handwritten **notes**", "scans"),
]


@pytest.fixture()
def patched_docs(monkeypatch):
    """PDF text extraction / subject PDF reads are file-system work: stub them (no real documents)."""
    text = "Lecture text about algorithms. " * 20
    import importlib
    for mod in ("quiz", "flashcards", "cheatsheet", "mindmap", "planner", "exam"):
        m = importlib.import_module(f"backend.routers.{mod}")
        if hasattr(m, "read_pdf"):
            monkeypatch.setattr(m, "read_pdf", lambda *a, **k: text)
    monkeypatch.setattr("backend.routers._common.extract_pdf_text", lambda *a, **k: text)
    monkeypatch.setattr(predictor_router, "extract_pdf_text", lambda *a, **k: text)
    monkeypatch.setattr("backend.routers.teachback.extract_pdf_text", lambda *a, **k: text)
    monkeypatch.setattr(planner_service, "read_pdf", lambda *a, **k: text)
    monkeypatch.setattr(planner_service, "pdf_signature", lambda *a, **k: (1.0, 1))
    planner_service._summary_cache.clear()
    teachback_service._local.clear()


def _call(client, spec):
    method, url, kwargs = spec
    return getattr(client, method)(url, **kwargs)


@pytest.mark.parametrize("case", JSON_CASES, ids=[c[0] for c in JSON_CASES])
def test_endpoint_returns_label_and_history_keeps_it(case, client, net, patched_docs, mem):
    _, spec, reply, kind = case
    _answer(net, reply)
    res = _call(client, spec)
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["llm"]["label"] == LABEL, body.get("llm")
    assert body["llm"]["label"].startswith("Model:")
    if kind is None:
        return
    assert body["id"], "result was not saved"
    listed = client.get(f"/api/history/{kind}").json()["items"]
    assert [i["llm"]["label"] for i in listed] == [LABEL], f"{kind} list lost the label"
    full = client.get(f"/api/history/{kind}/{body['id']}").json()
    assert full["llm"]["label"] == LABEL, f"{kind} reopen lost the label"


def test_exam_report_keeps_generator_and_grader_labels(client, net, patched_docs):
    _answer(net, _EXAM)
    paper = client.post("/api/exam/topic", json={"topic": "OS", "num_mcq": 1, "num_short": 1}).json()
    assert paper["llm"]["label"] == LABEL
    _answer(net, _GRADES)
    qs = [{**q, "id": f"q{i}"} for i, q in enumerate(paper["questions"])]
    body = {"subject": "OS", "questions": [{**_MCQ, "id": "m1"}, {**_SHORT, "id": "s1"}],
            "answers": {"m1": 0, "s1": "circular wait"}, "generated_llm": paper["llm"]}
    rep = client.post("/api/exam/grade", json=body).json()
    assert rep["llm"]["label"] == LABEL and rep["generated_llm"]["label"] == LABEL
    full = client.get(f"/api/history/exams/{rep['id']}").json()
    assert full["generated_llm"]["label"] == LABEL
    # a forged / malformed block is dropped, never echoed
    body["generated_llm"] = {"label": "evil <script>", "provider": 5}
    assert client.post("/api/exam/grade", json=body).json()["generated_llm"] is None


@pytest.mark.parametrize("case", JSON_CASES, ids=[c[0] for c in JSON_CASES])
def test_endpoint_total_failure_carries_none_label(case, client, net, patched_docs):
    _, spec, _reply, _kind = case
    net.on_all(err(503, "down"))
    res = _call(client, spec)
    body = res.json()
    if _ == "exam-grade":      # grading degrades instead of failing: 200 + provisional scores + the none label
        assert res.status_code == 200 and body["result"]["provisional"] is True, res.text
    else:
        assert res.status_code in (502, 503), res.text
    assert (body.get("llm") or {}).get("label") == NONE_LABEL, body


# ---------------------------------------------------------------------------------------
# Multi-step flows
# ---------------------------------------------------------------------------------------
def test_teachback_message_label_and_saved_session(client, net, patched_docs, mem):
    _answer(net, _START)
    sid = client.post("/api/teachback/start", json={"mode": "topic", "topic": "Algorithms"}).json()["session_id"]
    _answer(net, _PROBE)
    probe = client.post("/api/teachback/message", json={"session_id": sid, "message": "It halves the list."}).json()
    assert probe["llm"]["label"] == LABEL
    _answer(net, _EVAL)
    done = client.post("/api/teachback/message", json={"session_id": sid, "message": "evaluate"}).json()
    assert done["done"] and done["llm"]["label"] == LABEL
    # the finished session is saved WITH its label; list + reopen return it
    listed = client.get("/api/history/teachbacks").json()["items"]
    assert [i["llm"]["label"] for i in listed] == [LABEL]
    full = client.get(f"/api/history/teachbacks/{listed[0]['id']}").json()
    assert full["llm"]["label"] == LABEL
    ai = [m for m in full["messages"] if m["role"] == "assistant"]
    assert len(ai) == 3 and all(m["llm"]["label"] == LABEL for m in ai)       # opening + probe + evaluation
    assert all("llm" not in m for m in full["messages"] if m["role"] == "user")


def test_teachback_message_failure_has_none_label(client, net, patched_docs):
    _answer(net, _START)
    sid = client.post("/api/teachback/start", json={"mode": "topic", "topic": "Algorithms"}).json()["session_id"]
    net.on_all(err(503))
    res = client.post("/api/teachback/message", json={"session_id": sid, "message": "hi"})
    assert res.status_code == 503 and res.json()["llm"]["label"] == NONE_LABEL


def test_insights_label_fresh_cached_and_failure(client, net, monkeypatch):
    insights_service._cache.clear()
    insights_service._last_refresh.clear()
    active = {"overall": {"total_attempts": 4, "overall_accuracy": 0.5, "streak_days": 2}, "subjects": [],
              "weakest": []}
    monkeypatch.setattr(insights_service, "build_summary", lambda owner: active)
    monkeypatch.setattr(insights_service.db, "multi_recent",
                        lambda owner, specs: {s[0]: {"n": 1, "rows": [{"title": "Quiz"}]} for s in specs})
    _answer(net, _INSIGHTS)
    first = client.get("/api/insights").json()
    assert first["llm"]["label"] == LABEL
    cached = client.get("/api/insights").json()                    # served from the per-day cache: same label
    assert cached["llm"]["label"] == LABEL
    insights_service._cache.clear()
    insights_service._last_refresh.clear()
    net.on_all(err(503))
    res = client.get("/api/insights")
    assert res.status_code == 503 and res.json()["llm"]["label"] == NONE_LABEL


def test_progress_summary_has_no_ai_text(client):
    """/api/progress/summary is pure aggregation (no model call) - nothing to label."""
    body = client.get("/api/progress/summary").json()
    assert "llm" not in body


# ---------------------------------------------------------------------------------------
# Chat: overview, non-stream, SSE (model / done / error), tools, resume
# ---------------------------------------------------------------------------------------
def _frames(body: str):
    out = []
    for frame in body.split("\n\n"):
        frame = frame.strip()
        if frame.startswith("data:"):
            out.append(json.loads(frame[5:].strip()))
    return out


def _start(client, topic="Recursion"):
    r = client.post("/api/chat/start", json={"mode": "topic", "topic": topic})
    assert r.status_code == 200, r.text
    return r.json()["session_id"], r.json()["chat_id"]


def _stream(client, sid, message, **extra):
    return client.post("/api/chat/message/stream", json={"session_id": sid, "message": message, **extra})


def test_chat_stream_model_event_precedes_tokens_and_matches_done(client, net):
    net.on(1, CLAUDE, sse_anthropic(["Recur", "sion is ", "self-reference."]))
    sid, _ = _start(client)
    events = _frames(_stream(client, sid, "What is recursion?").text)
    kinds = [e["type"] for e in events]
    assert "model" in kinds and "token" in kinds
    assert kinds.index("model") < kinds.index("token"), kinds           # label is known BEFORE the first token
    model_ev = next(e for e in events if e["type"] == "model")
    assert model_ev["llm"]["label"] == LABEL and model_ev["llm"]["label"].startswith("Model:")
    done = events[-1]
    assert done["type"] == "done" and done["llm"]["label"] == LABEL


def test_chat_stream_model_event_reflects_fall_through(client, net):
    from tests.test_llm_chain import G_MODELS, sse_groq
    net.on(1, CLAUDE, err(529))
    net.on(2, G_MODELS[0], sse_groq(["hello ", "world"]))
    sid, _ = _start(client)
    events = _frames(_stream(client, sid, "hello there my friend").text)
    models = [e["llm"]["label"] for e in events if e["type"] == "model"]
    assert models and models[-1] == events[-1]["llm"]["label"]
    assert models[-1].startswith("Model: groq") and "step 2" in models[-1]


def test_chat_stream_error_event_has_none_label(client, net):
    net.on_all(err(500, "down"))
    sid, _ = _start(client)
    events = _frames(_stream(client, sid, "hello there my friend").text)
    assert events[-1]["type"] == "error" and events[-1]["llm"]["label"] == NONE_LABEL
    assert not [e for e in events if e["type"] == "model"]                # nothing was ever streaming


def test_chat_pre_stream_503_has_label(client, net, monkeypatch):
    sid, _ = _start(client)
    for k in lc.KEY_ENV_VARS:
        monkeypatch.setenv(k, "")
    r = _stream(client, sid, "hello")
    assert r.status_code == 503 and r.json()["llm"]["label"] == NONE_LABEL


def test_chat_non_stream_label_and_failure(client, net):
    net.on(1, CLAUDE, sse_anthropic(["Fine ", "answer."]))
    sid, _ = _start(client)
    ok = client.post("/api/chat/message", json={"session_id": sid, "message": "explain stacks please"})
    assert ok.status_code == 200 and ok.json()["llm"]["label"] == LABEL
    net.on_all(err(500, "down"))
    bad = client.post("/api/chat/message", json={"session_id": sid, "message": "explain queues please"})
    assert bad.status_code == 503 and bad.json()["llm"]["label"] == NONE_LABEL


def test_chat_overview_label_generated_and_prebaked(client, net, monkeypatch):
    overview_service._cache.clear()
    _answer(net, _OVERVIEW)
    sid, _ = _start(client, topic="Overview coverage topic")
    card = client.get(f"/api/chat/overview/{sid}").json()
    assert card["overview"] and card["llm"]["label"] == LABEL
    # pre-baked (shipped) overviews never call a model, but still say where the text came from
    monkeypatch.setattr(overview_service, "_load_prebaked",
                        lambda s, o: {"overview": "x", "questions": ["q"], "sources": 1,
                                      "llm": overview_service._prebaked_llm(None)})
    monkeypatch.setattr(overview_service, "list_pdf_names", lambda *a, **k: [])
    pre = overview_service.build_overview(subject="Mathematics", owner="alice")
    assert pre["llm"]["label"].startswith("Model:")


# One forced-tool chat turn per generator: the artifact AND the turn carry the generating label.
TOOL_CASES = [
    ("quiz", _QUIZ, "quizzes"), ("flashcards", _CARDS, "flashcards"), ("cheatsheet", _SHEET, "cheatsheets"),
    ("mindmap", _MAP, "mindmaps"), ("plan", _PLAN, "plans"), ("summary", _SUMMARY, "summaries"),
    ("teachback", _START, "teachbacks"), ("exam", _EXAM, "exams"), ("predictor", _PRED, "predictions"),
]


@pytest.mark.parametrize("tool,reply,kind", TOOL_CASES, ids=[t[0] for t in TOOL_CASES])
def test_chat_tool_artifacts_carry_the_label(tool, reply, kind, client, net, patched_docs, mem, monkeypatch):
    from backend.services import chat_tools
    monkeypatch.setattr("backend.vector_store.get_subject_retriever", lambda *a, **k: None)
    monkeypatch.setattr(chat_service, "_subject_persona", lambda subject, owner=None: "PERSONA")
    monkeypatch.setattr(chat_tools, "_subject_content", lambda subject, owner: ("Lecture text. " * 40, ["a.pdf"]))
    monkeypatch.setattr(chat_tools, "_subject_papers_corpus", lambda subject, owner: ("Paper text. " * 40, ["p.pdf"]))
    monkeypatch.setattr(chat_tools, "list_pdf_names", lambda subject, owner: ["a.pdf"])
    _answer(net, reply)
    r = client.post("/api/chat/start", json={"mode": "subject", "subject": "DBMS"})
    assert r.status_code == 200, r.text
    sid = r.json()["session_id"]
    events = _frames(_stream(client, sid, f"make me a {tool}", tool=tool).text)
    done = events[-1]
    assert done["type"] == "done", events[-1]
    assert done["llm"]["label"] == LABEL
    arts = done["artifacts"]
    assert arts, f"{tool}: no artifact produced"
    assert arts[0]["llm"]["label"] == LABEL
    if tool not in ("teachback", "exam"):                                  # those two are saved on completion
        assert arts[0].get("id"), f"{tool}: artifact not saved"
        saved = client.get(f"/api/history/{kind}/{arts[0]['id']}").json()
        assert saved["llm"]["label"] == LABEL


def test_resumed_chat_returns_stored_assistant_labels(client, net, mem):
    net.on(1, CLAUDE, sse_anthropic(["Recursion ", "is self-reference."]))
    sid, chat_id = _start(client)
    events = _frames(_stream(client, sid, "What is recursion?").text)
    assert events[-1]["type"] == "done"
    # saved chat detail (history drawer) keeps the label on the assistant message
    full = client.get(f"/api/history/chats/{chat_id}").json()
    assistants = [m for m in full["messages"] if m["role"] == "assistant"]
    assert assistants and all(m["llm"]["label"] == LABEL for m in assistants)
    # ... and /resume hands it back for rendering
    resumed = client.post("/api/chat/resume", json={"chat_id": chat_id}).json()
    msgs = [m for m in resumed["messages"] if m["role"] == "assistant"]
    assert msgs and all(m["llm"]["label"] == LABEL for m in msgs)
    users = [m for m in resumed["messages"] if m["role"] == "user"]
    assert users and all(m.get("llm") is None for m in users)             # only AI output is labelled
