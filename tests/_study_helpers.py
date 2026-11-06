"""Shared fakes for the study-tool tests (quiz, flashcards, exam, cheatsheet, mindmap, planner).

Everything LLM-related is mocked at ``backend.llm_chain.chat`` — no key, network or provider.
"""
from __future__ import annotations

import json
from unittest import mock

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from backend import llm_chain
from backend.auth import current_owner
from backend.documents import DocumentError


def result(text, *, skipped=None, model="openai/gpt-oss-120b") -> llm_chain.ChatResult:
    return llm_chain.ChatResult(text=text if isinstance(text, str) else json.dumps(text),
                                provider="groq", model=model, step=2, step_name="Groq key 1",
                                usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                                skipped=list(skipped or []), latency_ms=5)


class FakeChat:
    """Stand-in for ``llm_chain.chat``: replies come from a queue (str/dict -> ChatResult, Exception -> raised)."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls: list[dict] = []

    def __call__(self, system, messages, **kwargs):
        self.calls.append({"system": system, "messages": messages, **kwargs})
        r = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(r, Exception):
            raise r
        return r if isinstance(r, llm_chain.ChatResult) else result(r)


def patch_chat(*replies) -> tuple[mock._patch, FakeChat]:
    fake = FakeChat(*replies)
    return mock.patch("backend.llm_chain.chat", fake), fake


def chain_error() -> llm_chain.LLMChainError:
    return llm_chain.LLMChainError("All LLM providers failed", status=503, attempts=["groq#1 429"])


def make_client(*routers, owner="tester") -> TestClient:
    """Throwaway app with the same LLMChainError / DocumentError handlers main.py installs."""
    app = FastAPI()
    for r in routers:
        app.include_router(r)

    @app.exception_handler(llm_chain.LLMChainError)
    async def _llm(_req, exc):
        return JSONResponse(status_code=503, content={"detail": "AI unavailable", "llm": {"label": exc.label}})

    @app.exception_handler(DocumentError)
    async def _doc(_req, exc):
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    app.dependency_overrides[current_owner] = lambda: owner
    return TestClient(app)


PDF_BYTES = b"%PDF-1.4 fake"
