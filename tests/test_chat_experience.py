"""ChatGPT-style chat experience: Stop (partial persisted + upstream closed), Regenerate (versions), Edit
(branch/truncate), Continue, follow-up chips, citations, feedback, pin/rename - all with a mocked chain."""
from __future__ import annotations

import asyncio
import json

import pytest
from fastapi import Request
from fastapi.testclient import TestClient

from backend import config, db, llm_chain
from backend.auth import current_owner
from backend.main import app
from backend.services import chat as cs
from tests._fakechats import FakeChats
from tests.test_llm_chain import net, sse_groq  # noqa: F401  (net is a fixture)

@pytest.fixture()
def anyio_backend():
    return "asyncio"


FOLLOW = '\n<followups>["What is a base case?", "Show an example", "Why does it overflow?"]</followups>'


class Chain:
    """Scriptable replacement for ``llm_chain.astream_chat`` (records calls and whether the stream was closed)."""

    def __init__(self):
        self.script: list[dict] = []
        self.calls: list[dict] = []
        self.closed = 0
        self.started = asyncio.Event() if False else None

    def push(self, deltas, finish="stop", hang_after=False):
        self.script.append({"deltas": deltas, "finish": finish, "hang": hang_after})

    async def __call__(self, system, messages, **kw):
        self.calls.append({"system": system, "messages": [dict(m) for m in messages]})
        spec = self.script.pop(0)
        yield llm_chain.StreamStart("groq", "m1", 2, "Groq", [])
        text = ""
        try:
            for d in spec["deltas"]:
                text += d
                yield d
                await asyncio.sleep(0)
            if spec["hang"]:
                await asyncio.sleep(30)
        finally:
            self.closed += 1
        clean = text
        yield llm_chain.ChatResult(text=clean, provider="groq", model="m1", step=2, step_name="Groq",
                                   usage={"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
                                   finish=spec["finish"])


@pytest.fixture()
def env(monkeypatch):
    chain = Chain()
    chats = FakeChats()
    monkeypatch.setattr(llm_chain, "astream_chat", chain)
    monkeypatch.setattr(cs, "_chats_collection", lambda: chats)
    monkeypatch.setattr(cs.db, "_bump", lambda *a: None)
    from backend.services import chat_tools

    async def _no_intent(msg):
        return []
    monkeypatch.setattr(chat_tools, "adetect_intent", _no_intent)

    async def _noname(*a, **k):
        return None
    monkeypatch.setattr(cs, "_autoname", _noname)
    cs._STATE.clear()
    cs._LOCAL.clear()
    chat_id = chats.add("alice", title="t")
    session = cs._mint("alice", "topic", "Topic · Recursion", None, "Recursion", chat_id)
    return chain, chats, session, chat_id


async def run(session, msg, **kw):
    out = []
    async for ev in cs.stream_turn(session, msg, **kw):
        out.append(ev)
    return out


def done_of(events):
    return next(e for e in events if e["type"] == "done")


# ----------------------------------------------------------------------------------------- followups
@pytest.mark.parametrize("n", [1, 2, 3, 5, 9, 1000])
def test_followup_filter_is_streaming_safe(n):
    text = "Recursion calls itself." + FOLLOW
    f = cs.FollowupFilter()
    shown = ""
    for i in range(0, len(text), n):
        shown += f.feed(text[i:i + n])
    shown += f.finish()
    assert shown.strip() == "Recursion calls itself."
    assert f.followups() == ["What is a base case?", "Show an example", "Why does it overflow?"]


def test_followup_filter_keeps_lookalike_text_and_drops_bad_json():
    f = cs.FollowupFilter()
    out = f.feed("a < b and x<fo") + f.feed("o") + f.finish()
    assert out == "a < b and x<foo"
    assert cs.strip_followups("Hi <followups>[not json]</followups>") == ("Hi", [])
    assert cs.strip_followups("Hi <followups>[\"a long question?\"") == ("Hi", [])           # unterminated -> no chips
    vis, chips = cs.strip_followups('x<followups>["q one?","q one?","q two?","q three?","q four?",5,""]</followups>')
    assert vis == "x" and chips == ["q one?", "q two?", "q three?"]


@pytest.mark.anyio
async def test_turn_strips_followups_and_returns_them(env):
    chain, chats, session, cid = env
    chain.push(["Recursion ", "is self-reference.", "\n<follow", 'ups>["Why?", "Example please"]</followups>'])
    ev = await run(session, "what is recursion")
    tokens = "".join(e["text"] for e in ev if e["type"] == "token")
    assert "followups" not in tokens and "<" not in tokens
    d = done_of(ev)
    assert d["answer"] == "Recursion is self-reference."
    assert d["followups"] == ["Why?", "Example please"]
    assert d["usage"]["total_tokens"] == 8 and d["finish"] == "stop"
    assert "<followups>" in chain.calls[0]["system"]
    msgs = chats.get(cid)["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[1]["content"] == "Recursion is self-reference." and msgs[1]["id"] == d["message_ids"]["assistant"]
    assert "followups" not in json.dumps(msgs)


@pytest.mark.anyio
async def test_no_followups_when_model_omits_them(env):
    chain, *_ = env
    chain.push(["Just text."])
    assert done_of(await run(env[2], "explain"))["followups"] == []


# ----------------------------------------------------------------------------------------- stop
@pytest.mark.anyio
async def test_stop_cancels_upstream_and_persists_partial(env):
    chain, chats, session, cid = env
    chain.push(["Partial ", "answer so far"], hang_after=True)
    seen = asyncio.Event()

    async def consume():
        async for ev in cs.stream_turn(session, "explain"):
            if ev["type"] == "token" and "so far" in ev["text"]:
                seen.set()

    task = asyncio.ensure_future(consume())
    await asyncio.wait_for(seen.wait(), 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert chain.closed == 1                                  # the provider stream context was closed
    msgs = chats.get(cid)["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[1]["content"] == "Partial answer so far" and msgs[1]["stopped"] is True
    assert msgs[1]["llm"]["model"] == "m1"


@pytest.mark.anyio
async def test_stop_before_any_token_persists_nothing(env):
    chain, chats, session, cid = env
    chain.push([], hang_after=True)
    task = asyncio.ensure_future(run(session, "explain"))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert chats.get(cid)["messages"] == []


@pytest.mark.anyio
async def test_router_stream_cancels_turn_on_disconnect():
    from backend.routers import chat as router
    state = {"cancelled": False}

    async def events():
        try:
            yield {"type": "token", "text": "hi"}
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            state["cancelled"] = True
            raise

    agen = router._sse_stream(events())
    first = await agen.__anext__()
    assert first.startswith(": connected")
    assert '"token"' in await agen.__anext__()
    await agen.aclose()                                        # what Starlette does when the client goes away
    assert state["cancelled"] is True


# ----------------------------------------------------------------------------------------- regenerate
@pytest.mark.anyio
async def test_regenerate_keeps_versions_and_replaces_answer(env):
    chain, chats, session, cid = env
    chain.push(["First answer."])
    d1 = done_of(await run(session, "q1"))
    chain.push(["Second answer."])
    d2 = done_of(await run(session, "q1", action="regenerate", target_id=d1["message_ids"]["assistant"]))
    msgs = chats.get(cid)["messages"]
    assert len(msgs) == 2 and msgs[0]["content"] == "q1"
    assert msgs[0]["id"] == d1["message_ids"]["user"]            # the user message is kept, not duplicated
    assert msgs[1]["content"] == "Second answer."
    assert [v["content"] for v in msgs[1]["versions"]] == ["First answer."]
    assert chain.calls[1]["messages"][-1]["content"].endswith("q1") or chain.calls[1]["messages"][-1]["content"] == "q1"
    assert len(chain.calls[1]["messages"]) == 1                  # the old answer is NOT in the prompt
    chain.push(["Third."])
    await run(session, "q1", action="regenerate", target_id=d2["message_ids"]["assistant"])
    assert [v["content"] for v in chats.get(cid)["messages"][1]["versions"]] == ["First answer.", "Second answer."]


@pytest.mark.anyio
async def test_regenerate_only_latest_answer(env):
    chain, chats, session, cid = env
    chain.push(["A1."]); d1 = done_of(await run(session, "q1"))
    chain.push(["A2."]); await run(session, "q2")
    ev = await run(session, "q1", action="regenerate", target_id=d1["message_ids"]["assistant"])
    assert ev[-1]["type"] == "error" and ev[-1]["status"] == 409
    assert len(chats.get(cid)["messages"]) == 4                  # nothing was touched


@pytest.mark.anyio
async def test_failed_regenerate_keeps_old_answer(env):
    chain, chats, session, cid = env
    chain.push(["Keep me."]); d1 = done_of(await run(session, "q1"))

    async def boom(system, messages, **kw):
        raise llm_chain.LLMChainError("down")
        yield
    llm_chain.astream_chat = boom
    ev = await run(session, "q1", action="regenerate", target_id=d1["message_ids"]["assistant"])
    assert ev[-1]["type"] == "error"
    assert chats.get(cid)["messages"][1]["content"] == "Keep me."


# ----------------------------------------------------------------------------------------- edit
@pytest.mark.anyio
async def test_edit_truncates_everything_after_the_edited_message(env):
    chain, chats, session, cid = env
    chain.push(["A1."]); d1 = done_of(await run(session, "q1"))
    chain.push(["A2."]); await run(session, "q2")
    chain.push(["A3."]); await run(session, "q3")
    chain.push(["Edited answer."])
    users = [m for m in chats.get(cid)["messages"] if m["role"] == "user"]
    d = done_of(await run(session, "q2 edited", action="edit", target_id=users[1]["id"]))
    msgs = chats.get(cid)["messages"]
    assert [m["content"] for m in msgs] == ["q1", "A1.", "q2 edited", "Edited answer."]
    sent = chain.calls[-1]["messages"]
    assert [m["content"] for m in sent[:2]] == ["q1", "A1."] and "q2 edited" in sent[-1]["content"]
    assert "A2." not in json.dumps(sent) and "A3." not in json.dumps(sent)
    assert d["message_ids"]["user"] != users[1]["id"]


@pytest.mark.anyio
async def test_edit_first_message_keeps_user_rename(env):
    chain, chats, session, cid = env
    chain.push(["A1."]); d1 = done_of(await run(session, "q1"))
    chats.get(cid)["title"] = "My renamed chat"
    chain.push(["A1b."])
    await run(session, "new q1", action="edit", target_id=d1["message_ids"]["user"])
    assert chats.get(cid)["title"] == "My renamed chat"
    assert [m["content"] for m in chats.get(cid)["messages"]] == ["new q1", "A1b."]


@pytest.mark.anyio
async def test_edit_unknown_message_is_409(env):
    chain, chats, session, cid = env
    ev = await run(session, "x", action="edit", target_id="nope")
    assert ev[-1]["type"] == "error" and ev[-1]["status"] == 409


@pytest.mark.anyio
async def test_edit_in_stateless_mode_trusts_client_history(monkeypatch):
    chain = Chain()
    monkeypatch.setattr(llm_chain, "astream_chat", chain)
    monkeypatch.setattr(cs, "_chats_collection", lambda: None)
    from backend.services import chat_tools

    async def _no(msg):
        return []
    monkeypatch.setattr(chat_tools, "adetect_intent", _no)
    cs._LOCAL.clear()
    session = cs._mint("bob", "topic", "Topic · X", None, "X", None)
    chain.push(["A1."]); await run(session, "q1")
    chain.push(["A2."]); await run(session, "q2")
    chain.push(["A2b."])
    hist = [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "A1."}]
    await run(session, "q2b", client_history=hist, action="edit", target_id="whatever")
    sent = chain.calls[-1]["messages"]
    assert [m["content"] for m in sent[:2]] == ["q1", "A1."] and "A2." not in json.dumps(sent)
    chain.push(["A3."])
    await run(session, "q3", client_history=hist + [{"role": "user", "content": "q2b"},
                                                      {"role": "assistant", "content": "A2b."}])
    assert "A2." not in json.dumps(chain.calls[-1]["messages"][:-1]).replace("A2b.", "")


# ----------------------------------------------------------------------------------------- continue
@pytest.mark.anyio
async def test_continue_extends_the_stopped_answer(env):
    chain, chats, session, cid = env
    chain.push(["Part one, "], hang_after=True)
    seen = asyncio.Event()

    async def consume():
        async for ev in cs.stream_turn(session, "long q"):
            if ev["type"] == "token":
                seen.set()
    t = asyncio.ensure_future(consume())
    await asyncio.wait_for(seen.wait(), 5)
    t.cancel()
    with pytest.raises(asyncio.CancelledError):
        await t
    last = chats.get(cid)["messages"][-1]
    assert last["stopped"] is True
    chain.push(["part two."])
    d = done_of(await run(session, "long q", action="continue", target_id=last["id"]))
    msgs = chats.get(cid)["messages"]
    assert len(msgs) == 2 and msgs[1]["content"] == "Part one, part two." and d["answer"] == "Part one, part two."
    assert "stopped" not in msgs[1]
    sent = chain.calls[-1]["messages"]
    assert sent[-2] == {"role": "assistant", "content": "Part one,"} or sent[-2]["content"].startswith("Part one")
    assert "Continue your previous answer" in sent[-1]["content"]


@pytest.mark.anyio
async def test_length_finish_is_reported_and_persisted(env):
    chain, chats, session, cid = env
    chain.push(["Cut off mid"], finish="length")
    d = done_of(await run(session, "q"))
    assert d["finish"] == "length"
    assert chats.get(cid)["messages"][1]["finish"] == "length"


def test_chain_reports_finish_reason(net):
    import asyncio as aio
    from tests.test_llm_chain import G_MODELS
    net.on(2, G_MODELS[0], sse_groq(["abc"], finish="length"))
    for s in llm_chain.STEPS:
        pass

    async def go():
        last = None
        async for item in llm_chain.astream_chat("sys", [{"role": "user", "content": "x"}]):
            last = item
        return last
    from tests.test_llm_chain import err
    net.on(1, "claude-sonnet-5-5", err(529, "x"))
    res = aio.run(go())
    assert res.finish == "length"


# ----------------------------------------------------------------------------------------- citations
@pytest.mark.anyio
async def test_done_carries_citations_with_page_and_snippet(env, monkeypatch):
    chain, chats, session, cid = env

    class D:
        def __init__(self, t, p):
            self.page_content, self.metadata = t, {"source": "/x/Notes.pdf", "page": p}

    class R:
        def invoke(self, q):
            return [D("Recursion needs a base case. " * 30, 3)]
    session2 = cs._mint("alice", "topic", "l", None, "Recursion", cid)
    monkeypatch.setattr(cs, "_prepare", lambda s: (R(), "persona"))
    chain.push(["Ans."])
    d = done_of(await run(session2, "what is a base case"))
    assert d["citations"][0]["source"] == "Notes.pdf" and d["citations"][0]["page"] == 4
    assert len(d["citations"][0]["text"]) <= 305
    assert chats.get(cid)["messages"][1]["citations"][0]["page"] == 4


# ----------------------------------------------------------------------------------------- endpoints
@pytest.fixture()
def api(monkeypatch):
    chats = FakeChats()
    monkeypatch.setattr(cs, "_chats_collection", lambda: chats)
    monkeypatch.setattr(cs.db, "_bump", lambda *a: None)
    monkeypatch.setattr(db, "is_enabled", lambda: True)

    def _owner(request: Request) -> str:
        return request.headers.get("x-test-owner", "alice")
    app.dependency_overrides[current_owner] = _owner
    yield TestClient(app), chats
    app.dependency_overrides.pop(current_owner, None)


def test_feedback_is_owner_scoped_and_clearable(api):
    c, chats = api
    cid = chats.add("alice", [{"role": "user", "content": "q", "id": "u1"},
                              {"role": "assistant", "content": "a", "id": "a1"}])
    url = f"/api/chat/{cid}/feedback"
    assert c.post(url, json={"message_id": "a1", "value": "up"}).json() == {"saved": True, "value": "up"}
    assert chats.get(cid)["messages"][1]["feedback"] == "up"
    # another user cannot rate it; neither can a user message be rated
    assert c.post(url, json={"message_id": "a1", "value": "down"}, headers={"x-test-owner": "mallory"}).status_code == 404
    assert c.post(url, json={"message_id": "u1", "value": "down"}).status_code == 404
    assert chats.get(cid)["messages"][1]["feedback"] == "up"
    assert c.post(url, json={"message_id": "a1", "value": None}).status_code == 200
    assert "feedback" not in chats.get(cid)["messages"][1]
    assert c.post(url, json={"message_id": "a1", "value": "meh"}).status_code == 422


def test_feedback_on_legacy_message_without_id(api):
    c, chats = api
    cid = chats.add("alice", [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}])
    assert c.post(f"/api/chat/{cid}/feedback", json={"message_id": "i1", "value": "down"}).status_code == 200
    assert chats.get(cid)["messages"][1]["feedback"] == "down"


def test_pin_and_rename_are_owner_scoped(api):
    c, chats = api
    cid = chats.add("alice", [], title="old")
    assert c.patch(f"/api/chat/{cid}", json={"pinned": True}).status_code == 200
    assert chats.get(cid)["pinned"] is True and chats.get(cid)["title"] == "old"
    assert c.patch(f"/api/chat/{cid}", json={"title": "  New   name "}).json()["title"] == "New name"
    assert c.patch(f"/api/chat/{cid}", json={"title": "x", "pinned": False}, headers={"x-test-owner": "mallory"}).status_code == 404
    assert chats.get(cid)["title"] == "New name" and chats.get(cid)["pinned"] is True
    assert c.patch(f"/api/chat/{cid}", json={"title": "   "}).status_code == 400
    assert c.patch("/api/chat/not-an-id", json={"pinned": True}).status_code == 404


def test_resume_returns_ids_versions_and_flags(api, monkeypatch):
    c, chats = api
    monkeypatch.setattr(db, "get_item", lambda kind, owner, cid: chats.find_one({"_id": __import__("bson").ObjectId(cid), "owner": owner}))
    cid = chats.add("alice", [{"role": "user", "content": "q"},
                              {"role": "assistant", "content": "a", "id": "a1", "stopped": True,
                               "versions": [{"content": "old"}], "feedback": "up"}], mode="topic", topic="T", label="Topic · T")
    r = c.post("/api/chat/resume", json={"chat_id": cid}).json()
    u, a = r["messages"]
    assert u["id"] == "i0" and a["id"] == "a1" and a["stopped"] is True and a["versions"][0]["content"] == "old"
    assert a["feedback"] == "up"
    assert c.post("/api/chat/resume", json={"chat_id": cid}, headers={"x-test-owner": "mallory"}).status_code == 404


def test_message_request_validates_action(api):
    c, _ = api
    r = c.post("/api/chat/message/stream", json={"session_id": "x", "message": "hi", "action": "nuke"})
    assert r.status_code == 422
