"""Chat (RAG) router: start a session (subject/topic/PDF), exchange messages, and resume a saved chat.

Every route is owner-scoped (``current_owner``) and the ``session_id`` is a signed, stateless
token (see ``services/chat.py``), so any serverless instance can serve any turn and one user can
never drive another user's session.

SSE protocol of ``POST /api/chat/message/stream`` (one JSON object per ``data:`` line):
  {"type":"step","step":{...}}                         agentic-trace row (rewrite/retrieve/grade/generate/tool)
  {"type":"model","llm":{...}}                         the model that is answering (as soon as streaming starts; may repeat
                                                       if the chain falls through before the first token)
  {"type":"token","text":"..."}                        incremental answer delta (true provider streaming)
  {"type":"done","answer","sources","artifact","artifacts","llm":{...},
   "followups":[<=3 strings],"citations":[{source,page,text}],"finish":"stop|length|interrupted",
   "usage":{...},"message_ids":{"user","assistant"}}                       final event
  {"type":"error","detail","llm":{...},"status":N}     terminal failure (llm.label = "Model: none (all providers failed)")
Request ``action`` = send | edit | regenerate | continue (+ ``target_id`` / ``prefix``), see ``ChatMessageRequest``.
STOP: the client aborts the fetch; the server cancels the turn at once (the upstream provider stream is closed)
and saves the partial answer with ``stopped: true``.
Comment lines (": connected", ": ka") are keep-alives and carry no data. If the stream cannot start
(no keys / unknown session) a normal JSON error body ``{"detail","llm"?}`` is returned instead.
"""
import asyncio
import json
import logging

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

from backend import db, llm_chain
from backend.auth import current_owner
from backend.schemas import (
    ChatFeedbackRequest,
    ChatMessage,
    ChatMessageRequest,
    ChatMessageResponse,
    ChatOverviewResponse,
    ChatPatchRequest,
    ChatResumeRequest,
    ChatResumeResponse,
    ChatSessionResponse,
    ChatStartRequest,
)
from backend.services import chat as chat_service

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["chat"])

_KEEPALIVE_S = 8.0
_CANCEL_GRACE_S = 4.0
_SSE_HEADERS = {
    "Cache-Control": "no-cache, no-transform",
    "X-Accel-Buffering": "no",
    "Connection": "keep-alive",
}
_NO_KEY_DETAIL = ("No AI provider is configured — set at least one of the provider API keys "
                  "(see .env.example).")


def _no_llm(detail: str = _NO_KEY_DETAIL, status: int = 503) -> JSONResponse:
    return JSONResponse(status_code=status, content={"detail": detail, "llm": {"label": llm_chain.NO_RESULT_LABEL}})


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


@router.post("/start", response_model=ChatSessionResponse)
def start_session(request: ChatStartRequest, owner: str = Depends(current_owner)) -> ChatSessionResponse:
    """Start a subject- or topic-based chat session."""
    mode = request.mode
    if mode == "subject":
        if not request.subject:
            raise HTTPException(status_code=400, detail="A subject is required for subject mode.")
        session_id, label, chat_id = chat_service.start_subject_session(request.subject, owner=owner)
    elif mode == "topic":
        if not request.topic or not request.topic.strip():
            raise HTTPException(status_code=400, detail="A topic is required for topic mode.")
        session_id, label, chat_id = chat_service.start_topic_session(request.topic.strip(), owner=owner)
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported chat mode: {mode}")
    return ChatSessionResponse(session_id=session_id, mode=mode, label=label, chat_id=chat_id)


@router.post("/upload", response_model=ChatSessionResponse)
async def upload_pdf(
    file: UploadFile = File(...),
    model: str | None = Form(None),
    owner: str = Depends(current_owner),
) -> ChatSessionResponse:
    """Start a chat session backed by a single uploaded PDF."""
    file_bytes = await file.read(chat_service._MAX_PDF_BYTES + 1)
    if not file_bytes:
        raise HTTPException(status_code=400, detail="The uploaded PDF was empty.")
    if len(file_bytes) > chat_service._MAX_PDF_BYTES:
        raise HTTPException(status_code=413, detail="That PDF is too large (limit 20 MB).")
    filename = file.filename or "uploaded.pdf"
    try:
        # PDF parsing + Mongo writes are blocking: keep them off the event loop.
        session_id, label, chat_id = await asyncio.to_thread(
            chat_service.start_pdf_session, file_bytes, filename, None, owner)
    except chat_service.ChatSessionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return ChatSessionResponse(session_id=session_id, mode="pdf", label=label, chat_id=chat_id)


@router.post("/resume", response_model=ChatResumeResponse)
def resume_session(request: ChatResumeRequest, owner: str = Depends(current_owner)) -> ChatResumeResponse:
    """Re-open a saved chat from history as a live session, returning prior messages."""
    try:
        session_id, label, mode, chat_id, messages = chat_service.resume_session(owner, request.chat_id)
    except chat_service.ChatSessionError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return ChatResumeResponse(
        session_id=session_id, mode=mode, label=label, chat_id=chat_id,
        messages=[ChatMessage(**m) for m in messages],
    )


@router.get("/overview/{session_id}", response_model=ChatOverviewResponse)
def session_overview(session_id: str, refresh: bool = False, owner: str = Depends(current_owner)) -> ChatOverviewResponse:
    """NotebookLM-style overview card (summary + starter questions) for a session.

    Best-effort: the service never raises on LLM/parse failure (it returns an empty overview), so
    the only error is an unknown/foreign session → 404. Pre-baked overviews are served instantly.
    """
    try:
        result = chat_service.session_overview(session_id, owner, refresh=refresh)
    except chat_service.ChatSessionError as exc:
        raise HTTPException(status_code=404, detail="Session expired, start a new chat.") from exc
    return ChatOverviewResponse(
        label=result["label"], sources=result["sources"], overview=result["overview"],
        questions=result["questions"], llm=result.get("llm"),
    )


@router.post("/message", response_model=ChatMessageResponse)
async def send_message(request: ChatMessageRequest, owner: str = Depends(current_owner)):
    """Send a message within an existing session and return the assistant reply (non-streaming)."""
    if not llm_chain.has_any_key():
        return _no_llm()
    try:
        session = chat_service.open_session(request.session_id, owner)
    except chat_service.ChatSessionError:
        raise HTTPException(status_code=404, detail="Session expired, start a new chat.")
    try:
        result = await chat_service.ask(session, request.message, tool=request.tool, client_history=request.history,
                                        action=request.action, target_id=request.target_id, prefix=request.prefix,
                                        user_id=request.user_id, assistant_id=request.assistant_id)
    except chat_service.ChatTurnError as exc:
        content = {"detail": exc.detail}
        if exc.llm:
            content["llm"] = exc.llm
        return JSONResponse(status_code=exc.status, content=content)
    return ChatMessageResponse(
        answer=result["answer"], steps=result.get("steps", []), sources=result.get("sources", []),
        artifact=result.get("artifact"), artifacts=result.get("artifacts", []), llm=result.get("llm"),
        followups=result.get("followups", []), citations=result.get("citations", []),
        finish=result.get("finish"), usage=result.get("usage"), message_ids=result.get("message_ids"),
    )


async def _sse_stream(events):
    """Wrap the turn's event generator: flush headers at once, interleave keep-alives, and stop
    the turn when the client goes away. The turn itself shields its post-answer DB writes."""
    yield ": connected" + " " * 1024 + "\n\n"        # flush the response head through any proxy buffer
    queue: asyncio.Queue = asyncio.Queue()

    async def pump() -> None:
        try:
            async for ev in events:
                await queue.put(ev)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            log.exception("Chat stream failed")
            await queue.put({"type": "error", "detail": "Something went wrong while answering. Please try again.",
                             "llm": None, "status": 502})
        finally:
            await queue.put(None)

    task = asyncio.ensure_future(pump())
    try:
        while True:
            try:
                ev = await asyncio.wait_for(queue.get(), _KEEPALIVE_S)
            except asyncio.TimeoutError:
                yield ": ka\n\n"
                continue
            if ev is None:
                break
            yield _sse(ev)
    finally:
        # Client gone (Stop / closed tab): cancel the turn so the provider stream is closed immediately,
        # then give it a moment to save the partial answer before this response ends.
        task.cancel()
        try:
            await asyncio.wait({task}, timeout=_CANCEL_GRACE_S)
        except BaseException:  # noqa: BLE001 — never let cleanup mask the disconnect
            pass


@router.post("/message/stream")
async def stream_message(request: ChatMessageRequest, owner: str = Depends(current_owner)):
    """Stream the assistant's reasoning steps and answer tokens live as Server-Sent Events."""
    if not llm_chain.has_any_key():
        return _no_llm()
    try:
        session = chat_service.open_session(request.session_id, owner)
    except chat_service.ChatSessionError:
        return JSONResponse(status_code=404, content={"detail": "Session expired, start a new chat."})
    events = chat_service.stream_turn(session, request.message, tool=request.tool, client_history=request.history,
                                      action=request.action, target_id=request.target_id, prefix=request.prefix,
                                      user_id=request.user_id, assistant_id=request.assistant_id)
    return StreamingResponse(_sse_stream(events), media_type="text/event-stream", headers=_SSE_HEADERS)


@router.post("/{chat_id}/feedback")
def message_feedback(chat_id: str, body: ChatFeedbackRequest, owner: str = Depends(current_owner)) -> dict:
    """Thumbs up / down on one assistant message (``value`` null clears it). Stored as a field of the chat
    document, owner-scoped: someone else's chat or message id is simply a 404."""
    if not db.is_enabled():
        return {"saved": False}                       # persistence off: the rating lives in the browser only
    if not chat_service.set_feedback(owner, chat_id, body.message_id, body.value):
        raise HTTPException(status_code=404, detail="Message not found.")
    return {"saved": True, "value": body.value}


@router.patch("/{chat_id}")
def patch_chat(chat_id: str, body: ChatPatchRequest, owner: str = Depends(current_owner)) -> dict:
    """Rename and/or pin a saved chat (owner-scoped)."""
    title = None
    if body.title is not None:
        title = " ".join(body.title.split())[:120]
        if not title:
            raise HTTPException(status_code=400, detail="Title cannot be empty.")
    if not chat_service.patch_chat(owner, chat_id, title=title, pinned=body.pinned):
        raise HTTPException(status_code=404, detail="Chat not found.")
    return {"updated": True, "title": title, "pinned": body.pinned}
