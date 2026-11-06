"""TEST-ONLY fake LLM providers for load tests (no real provider, no spend, no keys).

One async Starlette app that speaks the three wire formats the provider chain uses, with the same
path layouts the real hosts expose:

    POST /v1/messages                                   Anthropic  (JSON + SSE when "stream": true)
    POST /openai/v1/chat/completions                    Groq       (JSON + SSE when "stream": true)
    POST /v1beta/models/<model>:generateContent         Gemini     (JSON)
    POST /v1beta/models/<model>:streamGenerateContent   Gemini     (SSE, ?alt=sse)
    POST /v1beta/models/<model>:embedContent|batchEmbedContents   Gemini embeddings (768-d unit vectors)

Point the backend at it with the test-only env var ``LLM_CHAIN_TEST_BASE_URL=http://127.0.0.1:8890``
(``backend/llm_chain.py`` ignores it whenever ``VERCEL`` is set). Any non-empty API key is accepted.

Replies are deterministic and shaped for what each feature parses: the fake sniffs the prompt (quiz,
flashcards, mind map, exam, teach-back, insights, overview, predictor, agentic rewrite/grade, intent
classifier ...) and returns matching JSON in JSON mode, or a few sentences of markdown otherwise.

Knobs (CLI flags or, at run time, ``POST /_cfg`` with a JSON body; ``GET /_stats`` shows counters,
``POST /_reset`` zeroes them):

    --latency-ms N      time to first byte of every reply (default 150)
    --jitter-ms N       +/- uniform jitter on latency (default 50)
    --token-ms N        delay between streamed chunks (default 12)
    --anthropic-fail P  probability (0..1) of failing an Anthropic call, with --anthropic-status (529)
    --groq-fail P       same for Groq, --groq-status (429, TPM-style message so the chain falls through)
    --gemini-fail P     same for Gemini, --gemini-status (503)
    --timeout-rate P    probability that ANY call hangs for --hang-s seconds (default 30) instead
                        of answering (simulates a provider timeout; the chain's call timeout is 25 s)

Usage:  python scripts/fake_providers.py --port 8890 --anthropic-fail 0.5
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import time
from typing import Any

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

CFG: dict[str, Any] = {
    "latency_ms": 150.0, "jitter_ms": 50.0, "token_ms": 12.0,
    "anthropic_fail": 0.0, "anthropic_status": 529,
    "groq_fail": 0.0, "groq_status": 429,
    "gemini_fail": 0.0, "gemini_status": 503,
    "timeout_rate": 0.0, "hang_s": 30.0,
}
STATS: dict[str, Any] = {}


def _reset() -> None:
    STATS.clear()
    STATS.update({"calls": {"anthropic": 0, "groq": 0, "gemini": 0},
                  "injected": {"anthropic": 0, "groq": 0, "gemini": 0, "timeouts": 0},
                  "stream": {"anthropic": 0, "groq": 0, "gemini": 0}, "inflight": 0, "max_inflight": 0})


_reset()


# --------------------------------------------------------------------------------------------
# Deterministic replies.
# --------------------------------------------------------------------------------------------
_WORDS = ("Linear algebra studies vector spaces and the linear maps between them. A matrix represents "
          "a linear transformation, and its eigenvalues describe directions that are only scaled. "
          "Solving systems of equations reduces to row operations on the augmented matrix. ").split()


def _answer_text(n_words: int = 48) -> str:
    return " ".join((_WORDS * 4)[:n_words]) + "."


def _int(pattern: str, text: str, default: int) -> int:
    m = re.search(pattern, text)
    try:
        return max(1, min(30, int(m.group(1)))) if m else default
    except ValueError:
        return default


def make_reply(system: str, user: str, json_mode: bool) -> str:
    """Reply text for one prompt (sniffs the feature from the prompt wording)."""
    t = (system + "\n" + user).lower()
    # Insights first: its prompt embeds the activity digest, whose keys ("flashcards", "quizzes", ...) would
    # otherwise match the generic tool sniffers below and return the wrong reply shape (-> 502 upstream).
    if "study coach" in t or '"focus_topics"' in t:
        return json.dumps({"headline": "You are making steady progress.",
                           "observations": ["You practise regularly."],
                           "recommendations": [{"action": "Take a quiz on weak topics", "reason": "Low recent score",
                                                "tool": "quiz"}],
                           "focus_topics": ["Eigenvalues"]})
    if '{"queries"' in t:
        return json.dumps({"queries": ["eigenvalues definition", "matrix transformation"]})
    if '"sufficient"' in t:
        return json.dumps({"sufficient": True, "missing": ""})
    if '"is_request"' in t:
        return json.dumps({"is_request": False, "tools": []})
    if "conversation in 3 to 6 words" in t:
        return "Linear algebra basics"
    if "marking student answers against a rubric" in t:       # backend/services/grading.py
        ids = re.findall(r'"id":\s*"([^"]+)"', user)
        return json.dumps({"grades": [{"id": i, "covered": [1], "score": 0.8, "verdict": "correct",
                                       "feedback": "Good coverage.", "missing_points": [], "model_answer_hint": ""} for i in ids]})
    if "grading short answers" in t:
        ids = re.findall(r'"id":\s*"([^"]+)"', user)
        return json.dumps({"grades": [{"id": i, "score_0_to_5": 4, "feedback": "Good coverage."} for i in ids]})
    if "mock exam" in t:
        mcq = _int(r"exactly (\d+) multiple choice", t, 2)
        short = _int(r"and exactly (\d+) short", t, 1)
        qs = [{"type": "mcq", "question": f"Which statement about topic {i + 1} is correct?",
               "options": ["Alpha", "Beta", "Gamma", "Delta"], "answer_index": i % 4, "topic": "Basics"}
              for i in range(mcq)]
        qs += [{"type": "short", "question": f"Explain concept {i + 1}.",
                "expected_points": ["definition", "example"], "topic": "Basics"} for i in range(short)]
        return json.dumps({"questions": qs})
    if "multiple choice questions" in t and "keyed a, b, c" in t:
        n = _int(r"exactly (\d+) multiple choice", t, 3)
        return json.dumps({"questions": [
            {"question": f"Question {i + 1}: what does an eigenvalue describe?",
             "options": {"A": "A scale factor", "B": "A rotation", "C": "A translation", "D": "A shear"},
             "correct": "ABCD"[i % 4], "explanation": "Eigenvalues scale eigenvectors."} for i in range(n)]})
    if "flashcards" in t:
        n = _int(r"create (\d+) focused flashcards", t, 3)
        return json.dumps({"cards": [
            {"front": f"Term {i + 1}?", "back": f"Definition {i + 1}.", "topic": "Basics"} for i in range(n)]})
    if "mind map" in t:
        return json.dumps({"central": "Linear Algebra", "branches": [
            {"title": f"Branch {i + 1}", "points": ["point one", "point two", "point three"]} for i in range(5)]})
    if "feynman" in t and "opening" in t:
        return json.dumps({"concept": "Eigenvalues",
                           "opening": "Explain eigenvalues to me in your own words, as if I knew nothing about it."})
    if "teach-back" in t or '"reply"' in t:
        return json.dumps({"reply": "Interesting. Can you give a concrete example?", "done": False,
                           "evaluation": None})
    if "starter questions" in t and '"overview"' in t:
        return json.dumps({"overview": "This material covers **linear algebra** basics.",
                           "questions": ["What is an eigenvalue?", "How do I row-reduce?", "What is rank?"]})
    if "predicted_questions" in t:
        return json.dumps({"patterns": [{"topic": "Eigenvalues", "frequency_note": "every year", "trend": "up"}],
                           "predicted_questions": [{"question": "Define eigenvalue.", "topic": "Eigenvalues",
                                                    "likelihood": "high", "reasoning": "Appears often."}]})
    if json_mode:
        return json.dumps({"ok": True})
    return ("## Summary\n\n" + _answer_text(60) + "\n\n- Key idea one\n- Key idea two\n- Key idea three\n\n"
            + _answer_text(40))


def _chunks(text: str, size: int = 4) -> list[str]:
    words = text.split(" ")
    return [" ".join(words[i:i + size]) + (" " if i + size < len(words) else "")
            for i in range(0, len(words), size)] or [text]


def _tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _embed(req: dict) -> list[float]:
    """Deterministic pseudo-embedding (unit vector) of the request text."""
    dim = int(req.get("outputDimensionality") or 768)
    text = " ".join(p.get("text", "") for p in (req.get("content") or {}).get("parts", []))
    rng = random.Random(hash(text) & 0xFFFFFFFF)
    v = [rng.gauss(0, 1) for _ in range(dim)]
    n = sum(x * x for x in v) ** 0.5 or 1.0
    return [round(x / n, 6) for x in v]


# --------------------------------------------------------------------------------------------
# Provider simulation.
# --------------------------------------------------------------------------------------------
async def _gate(provider: str) -> JSONResponse | None:
    """Count the call, apply latency, and maybe inject a timeout or an HTTP failure."""
    STATS["calls"][provider] += 1
    STATS["inflight"] += 1
    STATS["max_inflight"] = max(STATS["max_inflight"], STATS["inflight"])
    try:
        lat = max(0.0, CFG["latency_ms"] + random.uniform(-CFG["jitter_ms"], CFG["jitter_ms"])) / 1000.0
        if CFG["timeout_rate"] and random.random() < CFG["timeout_rate"]:
            STATS["injected"]["timeouts"] += 1
            await asyncio.sleep(CFG["hang_s"])
        await asyncio.sleep(lat)
        if CFG[f"{provider}_fail"] and random.random() < CFG[f"{provider}_fail"]:
            STATS["injected"][provider] += 1
            st = int(CFG[f"{provider}_status"])
            return JSONResponse(status_code=st, content=_error_body(provider, st))
        return None
    finally:
        STATS["inflight"] -= 1


def _error_body(provider: str, status: int) -> dict:
    if provider == "anthropic":
        kind = {529: "overloaded_error", 429: "rate_limit_error", 500: "api_error"}.get(status, "api_error")
        return {"type": "error", "error": {"type": kind, "message": "Overloaded" if status == 529 else "injected"}}
    if provider == "groq":
        msg = ("Rate limit reached for model `openai/gpt-oss-120b` in organization `org_x` on tokens per minute "
               "(TPM): Limit 8000, Used 7900, Requested 2000. Please try again in 6.5s.") if status == 429 \
            else "injected failure"
        return {"error": {"message": msg, "type": "tokens", "code": "rate_limit_exceeded"}}
    return {"error": {"code": status, "message": "The model is overloaded. Please try again later.",
                      "status": "UNAVAILABLE"}}


def _sse(lines: list[str]) -> str:
    return "".join(lines)


async def _stream(provider: str, frames: list[str]):
    STATS["stream"][provider] += 1
    for f in frames:
        yield f
        await asyncio.sleep(CFG["token_ms"] / 1000.0)


def _last_user(messages: list[dict]) -> str:
    for m in reversed(messages):
        c = m.get("content")
        if m.get("role") == "user":
            return c if isinstance(c, str) else " ".join(b.get("text", "") for b in c if isinstance(b, dict))
    return ""


async def anthropic(request: Request):
    body = await request.json()
    if (err := await _gate("anthropic")) is not None:
        return err
    system = body.get("system") or ""
    system = system if isinstance(system, str) else " ".join(b.get("text", "") for b in system)
    user = _last_user(body.get("messages") or [])
    json_mode = "json" in (system + user).lower()
    text = make_reply(system, user, json_mode)
    model = body.get("model", "claude-sonnet-5-5")
    usage = {"input_tokens": _tokens(system + user), "output_tokens": _tokens(text)}
    if body.get("stream"):
        frames = [f'event: message_start\ndata: {json.dumps({"type": "message_start", "message": {"id": "msg_fake", "model": model, "usage": {"input_tokens": usage["input_tokens"], "output_tokens": 1}}})}\n\n',
                  'event: content_block_start\ndata: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n\n']
        frames += [f'event: content_block_delta\ndata: {json.dumps({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": c}})}\n\n'
                   for c in _chunks(text)]
        frames += ['event: content_block_stop\ndata: {"type":"content_block_stop","index":0}\n\n',
                   f'event: message_delta\ndata: {json.dumps({"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": usage["output_tokens"]}})}\n\n',
                   'event: message_stop\ndata: {"type":"message_stop"}\n\n']
        return StreamingResponse(_stream("anthropic", frames), media_type="text/event-stream")
    return JSONResponse({"id": "msg_fake", "type": "message", "role": "assistant", "model": model,
                         "content": [{"type": "text", "text": text}], "stop_reason": "end_turn", "usage": usage})


async def groq(request: Request):
    body = await request.json()
    if (err := await _gate("groq")) is not None:
        return err
    msgs = body.get("messages") or []
    system = " ".join(m.get("content", "") for m in msgs if m.get("role") == "system" and isinstance(m.get("content"), str))
    user = _last_user(msgs)
    json_mode = bool(body.get("response_format")) or "json" in (system + user).lower()
    text = make_reply(system, user, json_mode)
    model = body.get("model", "openai/gpt-oss-120b")
    u = {"prompt_tokens": _tokens(system + user), "completion_tokens": _tokens(text)}
    u["total_tokens"] = u["prompt_tokens"] + u["completion_tokens"]
    if body.get("stream"):
        frames = [f'data: {json.dumps({"id": "c1", "object": "chat.completion.chunk", "model": model, "choices": [{"index": 0, "delta": {"content": c}, "finish_reason": None}]})}\n\n'
                  for c in _chunks(text)]
        frames.append(f'data: {json.dumps({"id": "c1", "object": "chat.completion.chunk", "model": model, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "usage": u})}\n\n')
        frames.append("data: [DONE]\n\n")
        return StreamingResponse(_stream("groq", frames), media_type="text/event-stream")
    return JSONResponse({"id": "c1", "object": "chat.completion", "model": model,
                         "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                                      "finish_reason": "stop"}], "usage": u})


async def gemini(request: Request):
    model_action = request.path_params["model_action"]
    model, _, action = model_action.partition(":")
    body = await request.json()
    if (err := await _gate("gemini")) is not None:
        return err
    if action in ("embedContent", "batchEmbedContents"):
        reqs = body.get("requests") if action == "batchEmbedContents" else [body]
        vecs = [_embed(r) for r in reqs or []]
        if action == "embedContent":
            return JSONResponse({"embedding": {"values": vecs[0]}})
        return JSONResponse({"embeddings": [{"values": v} for v in vecs]})
    sys_parts = (body.get("systemInstruction") or {}).get("parts") or []
    system = " ".join(p.get("text", "") for p in sys_parts)
    contents = body.get("contents") or []
    user = ""
    for c in reversed(contents):
        if c.get("role") == "user":
            user = " ".join(p.get("text", "") for p in c.get("parts", []))
            break
    json_mode = (body.get("generationConfig") or {}).get("responseMimeType") == "application/json"
    text = make_reply(system, user, json_mode)
    um = {"promptTokenCount": _tokens(system + user), "candidatesTokenCount": _tokens(text)}
    um["totalTokenCount"] = um["promptTokenCount"] + um["candidatesTokenCount"]

    def frame(t: str, final: bool) -> dict:
        d: dict[str, Any] = {"candidates": [{"content": {"role": "model", "parts": [{"text": t}]},
                                             "index": 0, **({"finishReason": "STOP"} if final else {})}],
                             "modelVersion": model}
        if final:
            d["usageMetadata"] = um
        return d

    if action == "streamGenerateContent":
        parts = _chunks(text)
        frames = [f"data: {json.dumps(frame(c, i == len(parts) - 1))}\r\n\r\n" for i, c in enumerate(parts)]
        return StreamingResponse(_stream("gemini", frames), media_type="text/event-stream")
    return JSONResponse(frame(text, True))


# --------------------------------------------------------------------------------------------
# Control plane.
# --------------------------------------------------------------------------------------------
async def cfg(request: Request):
    if request.method == "POST":
        for k, v in (await request.json()).items():
            if k in CFG:
                CFG[k] = type(CFG[k])(v)
    return JSONResponse(CFG)


async def stats(_request: Request):
    return JSONResponse(STATS)


async def reset(_request: Request):
    _reset()
    return JSONResponse({"ok": True})


app = Starlette(routes=[
    Route("/v1/messages", anthropic, methods=["POST"]),
    Route("/openai/v1/chat/completions", groq, methods=["POST"]),
    Route("/v1beta/models/{model_action:path}", gemini, methods=["POST"]),
    Route("/_cfg", cfg, methods=["GET", "POST"]),
    Route("/_stats", stats, methods=["GET"]),
    Route("/_reset", reset, methods=["POST"]),
])


def main() -> None:
    import uvicorn

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8890)
    for k, v in CFG.items():
        ap.add_argument("--" + k.replace("_", "-"), type=type(v), default=v)
    a = ap.parse_args()
    for k in CFG:
        CFG[k] = getattr(a, k)
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
