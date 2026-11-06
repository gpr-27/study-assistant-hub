"""Request-body cap for the upload endpoints, answered as JSON.

Vercel rejects any request body above ~4.5 MB with its own non-JSON error page, which the SPA
cannot show nicely. These endpoints are therefore capped a little below that (``MAX_BODY_BYTES``,
default 4.4 MB) and refuse bigger bodies with a clean JSON 413 *before* the multipart body is
parsed. The same limit is documented for every environment so behaviour matches local dev.

Pure ASGI (no buffering): a declared ``Content-Length`` is checked up front; a body without one
(chunked) is counted as it streams and cut off at the limit.
"""
from __future__ import annotations

import json

from backend import config

# Endpoints that take an uploaded PDF / photo in the request body.
LIMITED_PATHS: frozenset[str] = frozenset({
    "/api/scannotes/extract",
    "/api/predictor/analyze",
    "/api/chat/upload",
    "/api/planner/upload",
    "/api/quiz/upload",
    "/api/flashcards/upload",
    "/api/exam/upload",
    "/api/cheatsheet/upload",
    "/api/mindmap/upload",
    "/api/teachback/start-upload",
})


def too_large_message(limit: int) -> str:
    mb = limit / 1_000_000
    return (f"That upload is too large: requests are limited to about {mb:.1f} MB in total. "
            "Compress the files or upload fewer at a time.")


class BodyLimitMiddleware:
    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("method") != "POST" or scope.get("path") not in LIMITED_PATHS:
            await self.app(scope, receive, send)
            return
        limit = int(config.MAX_BODY_BYTES)
        declared = None
        for k, v in scope.get("headers") or []:
            if k == b"content-length":
                try:
                    declared = int(v)
                except ValueError:
                    declared = None
                break
        started = {"v": False}

        async def respond() -> None:
            body = json.dumps({"detail": too_large_message(limit)}).encode()
            await send({"type": "http.response.start", "status": 413,
                        "headers": [(b"content-type", b"application/json"),
                                    (b"content-length", str(len(body)).encode()),
                                    (b"connection", b"close")]})
            await send({"type": "http.response.body", "body": body})

        if declared is not None and declared > limit:
            await respond()
            return

        seen = {"n": 0}
        state = {"cut": False}

        async def counting_receive():
            if state["cut"]:
                return {"type": "http.disconnect"}
            msg = await receive()
            if msg["type"] == "http.request":
                seen["n"] += len(msg.get("body") or b"")
                if seen["n"] > limit and not started["v"]:
                    # Answer NOW (FastAPI would otherwise turn the aborted read into a 400), then
                    # tell the app the client went away; anything it sends afterwards is dropped.
                    state["cut"] = True
                    await respond()
                    return {"type": "http.disconnect"}
            return msg

        async def tracking_send(message):
            if state["cut"]:
                return
            if message["type"] == "http.response.start":
                started["v"] = True
            await send(message)

        await self.app(scope, counting_receive, tracking_send)
