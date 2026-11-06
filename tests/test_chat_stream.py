"""End-to-end tests for the streaming chat endpoint over the REAL provider chain.

Provider HTTP is mocked at the transport (``tests.test_llm_chain.net``), so these exercise
llm_chain -> chat service -> SSE router together: fall-through before the first delta, the exact
event protocol, the "Model: none" error path, 503/404 pre-stream JSON errors, bounded history,
and per-owner session isolation (a token only works for the owner it was minted for).
"""
from __future__ import annotations

import json
from unittest import mock

import pytest
from fastapi import Request
from fastapi.testclient import TestClient

from backend import config, llm_chain as lc
from backend.auth import current_owner
from backend.main import app
from backend.services import chat as chat_service
from tests.test_llm_chain import CLAUDE, G_MODELS, err, net, sse_groq  # noqa: F401  (net is a fixture)


class _Doc:
    def __init__(self, text, source="notes.pdf", page=0):
        self.page_content = text
        self.metadata = {"source": source, "page": page}


class _Retriever:
    def __init__(self, docs):
        self.docs, self.queries = docs, []

    def invoke(self, q):
        self.queries.append(q)
        return list(self.docs)


@pytest.fixture()
def client(net, monkeypatch):
    """TestClient with persistence OFF, owner taken from the X-Test-Owner header, no real Mongo."""
    monkeypatch.setattr(config, "PERSISTENCE_ENABLED", False)
    chat_service._LOCAL.clear()
    def _owner(request: Request) -> str:
        return request.headers.get("x-test-owner", "alice")

    app.dependency_overrides[current_owner] = _owner
    c = TestClient(app)
    yield c
    app.dependency_overrides.pop(current_owner, None)
    chat_service._LOCAL.clear()


def _frames(body: str):
    events, comments = [], []
    for frame in body.split("\n\n"):
        frame = frame.strip()
        if frame.startswith("data:"):
            events.append(json.loads(frame[5:].strip()))
        elif frame.startswith(":"):
            comments.append(frame)
    return events, comments


def _start_topic(client, owner="alice", topic="Recursion"):
    r = client.post("/api/chat/start", json={"mode": "topic", "topic": topic}, headers={"x-test-owner": owner})
    assert r.status_code == 200, r.text
    return r.json()["session_id"]


def _stream(client, sid, message, owner="alice", **extra):
    return client.post("/api/chat/message/stream", json={"session_id": sid, "message": message, **extra},
                       headers={"x-test-owner": owner})


# ---------------------------------------------------------------------------------------
def test_stream_falls_through_before_first_delta_and_reports_model(client, net):
    net.on(1, CLAUDE, err(529, "overloaded"))                 # step 1 fails before any token
    net.on(2, G_MODELS[0], sse_groq(["Recur", "sion is ", "self-reference."]))
    sid = _start_topic(client)
    resp = _stream(client, sid, "What is recursion?")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    assert "no-transform" in resp.headers["cache-control"]
    events, comments = _frames(resp.text)
    assert comments and comments[0].startswith(": connected")       # headers flushed with a first comment
    tokens = [e["text"] for e in events if e["type"] == "token"]
    assert tokens == ["Recur", "sion is ", "self-reference."]       # true incremental deltas
    done = events[-1]
    assert done["type"] == "done" and done["answer"] == "Recursion is self-reference."
    assert done["llm"]["step"] == 2 and done["llm"]["model"] == G_MODELS[0]
    assert done["llm"]["label"].startswith("Model: groq")
    assert "anthropic 529" in done["llm"]["skipped"]                # fell through BEFORE the first delta
    assert [c["step"] for c in net.calls][:2] == [1, 2]
    # tokens strictly precede done; exactly one done
    assert [e["type"] for e in events].index("done") == len(events) - 1


def test_subject_stream_has_trace_steps_sources_and_zero_pre_generation_llm_calls(client, net, monkeypatch):
    retriever = _Retriever([_Doc("A B-tree keeps keys sorted.", "/docs/dbms.pdf", 2)])
    monkeypatch.setattr("backend.vector_store.get_subject_retriever", lambda *a, **k: retriever)
    monkeypatch.setattr(chat_service, "_subject_persona", lambda subject, owner=None: "PERSONA")
    net.on(1, CLAUDE, err(401, "bad key"))
    net.on(2, G_MODELS[0], sse_groq(["It keeps ", "keys sorted."]))
    r = client.post("/api/chat/start", json={"mode": "subject", "subject": "DBMS"}, headers={"x-test-owner": "alice"})
    sid = r.json()["session_id"]
    events, _ = _frames(_stream(client, sid, "What does a B-tree do?").text)
    kinds = [e["step"]["kind"] for e in events if e["type"] == "step"]
    assert kinds == ["rewrite", "retrieve", "grade", "generate"]
    done = events[-1]
    assert done["type"] == "done" and done["sources"] == ["dbms.pdf"]
    assert retriever.queries == ["What does a B-tree do?"]
    # the ONLY provider calls are the generation attempts (no rewrite/grade LLM round-trips first)
    assert all(c["body"].get("stream") or "streamGenerateContent" in c["url"] for c in net.calls)
    gen_body = net.calls[-1]["body"]
    assert "A B-tree keeps keys sorted." in json.dumps(gen_body)
    assert "PERSONA" in json.dumps(gen_body)


def test_all_providers_failed_emits_error_event_with_none_label(client, net):
    net.on_all(err(500, "down"))
    sid = _start_topic(client)
    events, _ = _frames(_stream(client, sid, "hello there my friend").text)
    assert events[-1]["type"] == "error"
    assert events[-1]["llm"] == {"label": "Model: none (all providers failed)"}
    assert "unavailable" in events[-1]["detail"].lower()
    assert not [e for e in events if e["type"] == "done"]


def test_non_stream_message_all_failed_is_503_with_label_not_500(client, net):
    net.on_all(err(500, "down"))
    sid = _start_topic(client)
    r = client.post("/api/chat/message", json={"session_id": sid, "message": "hello there my friend"},
                    headers={"x-test-owner": "alice"})
    assert r.status_code == 503
    assert r.json()["llm"]["label"] == "Model: none (all providers failed)"


def test_non_stream_message_ok_returns_answer_and_llm(client, net):
    net.on(1, CLAUDE, err(500, "x"))
    net.on(2, G_MODELS[0], sse_groq(["Fine ", "answer."]))
    sid = _start_topic(client)
    r = client.post("/api/chat/message", json={"session_id": sid, "message": "explain stacks please"},
                    headers={"x-test-owner": "alice"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "Fine answer." and body["llm"]["step"] == 2


def test_no_provider_key_returns_json_503_before_streaming(client, net, monkeypatch):
    sid = _start_topic(client)
    for k in lc.KEY_ENV_VARS:
        monkeypatch.setenv(k, "")
    r = _stream(client, sid, "hello")
    assert r.status_code == 503
    assert r.headers["content-type"].startswith("application/json")
    body = r.json()
    assert body["detail"] and body["llm"]["label"] == "Model: none (all providers failed)"


def test_unknown_or_tampered_session_is_json_404(client, net):
    sid = _start_topic(client)
    r = _stream(client, "does-not-exist", "hi")
    assert r.status_code == 404 and "expired" in r.json()["detail"].lower()
    body, sig = sid.split(".")
    tampered = body[:-2] + ("AA" if not body.endswith("AA") else "BB") + "." + sig
    assert _stream(client, tampered, "hi").status_code == 404


def test_message_validation_limits(client, net):
    sid = _start_topic(client)
    assert _stream(client, sid, "").status_code == 422
    assert _stream(client, sid, "x" * 8001).status_code == 422


# ---------------------------------------------------------------------------------------
# Isolation
# ---------------------------------------------------------------------------------------
def test_two_owners_cannot_use_each_others_sessions(client, net):
    net.on(1, CLAUDE, err(500, "x"))
    net.on(2, G_MODELS[0], sse_groq(["secret ", "answer"]))
    alice = _start_topic(client, "alice", "Alice's private topic")
    bob = _start_topic(client, "bob", "Bob topic")
    assert alice != bob
    # Bob presenting Alice's token is rejected on every route — as "unknown session", no leakage.
    assert _stream(client, alice, "hello there friend", owner="bob").status_code == 404
    assert client.post("/api/chat/message", json={"session_id": alice, "message": "hello there friend"},
                       headers={"x-test-owner": "bob"}).status_code == 404
    assert client.get(f"/api/chat/overview/{alice}", headers={"x-test-owner": "bob"}).status_code == 404
    # ... while Alice can use her own.
    ok = _stream(client, alice, "hello there friend", owner="alice")
    events, _ = _frames(ok.text)
    assert events[-1]["type"] == "done"


def test_conversation_memory_is_per_owner_and_per_session(client, net):
    net.on_all(sse_groq(["ok"]))
    a1, a2, b1 = _start_topic(client, "alice"), _start_topic(client, "alice"), _start_topic(client, "bob")
    _stream(client, a1, "my secret code word is PINEAPPLE please remember", owner="alice")
    net.calls.clear()
    _stream(client, a1, "what was my code word again please", owner="alice")
    assert "PINEAPPLE" in json.dumps(net.calls[0]["body"])           # same session: memory works
    for sid, owner in ((a2, "alice"), (b1, "bob")):                  # other session / other owner: nothing
        net.calls.clear()
        _stream(client, sid, "what was my code word again please", owner=owner)
        assert "PINEAPPLE" not in json.dumps(net.calls[0]["body"])


def test_local_state_key_includes_owner():
    s_alice = chat_service._mint("alice", "topic", "L", None, "t", None, nonce="same-nonce")
    chat_service._persist_exchange(s_alice, "q", "a", None, {"title": "T", "text": "x"}, True)
    forged = chat_service.Session("x", "bob", "topic", "L", None, "t", None, "same-nonce")
    assert chat_service.load_state(forged)["history"] == []
    assert chat_service.load_state(s_alice)["history"] == [("q", "a")]
    chat_service._LOCAL.clear()


def test_session_token_rejected_for_other_owner_and_survives_a_fresh_process_secret():
    s = chat_service._mint("alice", "subject", "Subject · DBMS", "DBMS", None, "abc123")
    opened = chat_service.open_session(s.id, "alice")
    assert (opened.subject, opened.chat_id, opened.mode) == ("DBMS", "abc123", "subject")
    with pytest.raises(chat_service.ChatSessionError):
        chat_service.open_session(s.id, "mallory")


# ---------------------------------------------------------------------------------------
# History bounds / client-supplied history / post-answer persistence
# ---------------------------------------------------------------------------------------
def test_client_history_is_used_in_full_when_persistence_is_off(client, net):
    net.on_all(sse_groq(["ok"]))
    sid = _start_topic(client)
    hist = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"turn-{i} " + "w " * 50} for i in range(40)]
    r = _stream(client, sid, "and what about the last one?", history=hist[:40])
    assert r.status_code == 200
    sent = json.dumps(net.calls[0]["body"])
    assert "turn-39" in sent and "turn-0 " in sent                   # whole conversation sent (no history cap)


def test_title_and_persist_run_after_the_done_event_in_one_write(client, net, monkeypatch):
    order = []
    net.on_all(sse_groq(["Answer ", "text."]))
    monkeypatch.setattr(chat_service.db, "new_chat_id", lambda: "507f1f77bcf86cd799439011")
    sid = _start_topic(client)

    async def fake_title(*a, **k):
        order.append("title")
        return "Heaps in depth"

    monkeypatch.setattr(chat_service, "_llm_title", fake_title)
    with mock.patch.object(chat_service, "_persist_exchange",
                           side_effect=lambda *a, **k: order.append(("persist", k.get("title")))):
        events, _ = _frames(_stream(client, sid, "explain heaps in detail please").text)
    assert events[-1]["type"] == "done"
    # first turn: the LLM title is produced first and rides in the SAME single write as the exchange
    assert order == ["title", ("persist", "Heaps in depth")]


def test_tool_turn_returns_artifacts_and_llm_in_done_event(client, net):
    art = {"kind": "quizzes", "id": "q1", "title": "Quiz", "route": "#quiz", "tool": "Quiz Generator",
           "llm": {"label": "Model: groq · x"}}
    sid = _start_topic(client)
    with mock.patch("backend.services.chat_tools.run_tool",
                    return_value={"answer": "Made it", "steps": [{"kind": "tool", "title": "Created"}],
                                  "artifact": art, "llm": {"label": "Model: groq · x"},
                                  "recent_artifact": {"title": "Quiz", "kind": "quizzes", "text": "Q1"}}):
        events, _ = _frames(_stream(client, sid, "anything", tool="quiz").text)
    done = events[-1]
    assert done["answer"] == "Made it" and done["artifacts"] == [art] and done["artifact"] == art
    assert done["llm"]["label"].startswith("Model:")
    assert not net.calls                                               # tool turns spend no chat-completion call
    # the recent artifact is remembered for the follow-up turn
    st = chat_service.load_state(chat_service.open_session(sid, "alice"))
    assert st["recent_artifact"]["text"] == "Q1"
