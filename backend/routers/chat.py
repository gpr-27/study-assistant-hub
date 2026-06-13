"""Chat (RAG) router: start a session (subject/topic/PDF), exchange messages, and
resume a saved chat. Sessions and messages are persisted per user (when enabled).

DocumentError and MissingApiKeyError bubble up to the global handlers in main.py
(mapped to 404 and 503 respectively). LLM/parse failures surface as 502.
"""
import json

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from backend.auth import current_owner
from backend.schemas import (
    ChatMessage,
    ChatMessageRequest,
    ChatMessageResponse,
    ChatOverviewResponse,
    ChatResumeRequest,
    ChatResumeResponse,
    ChatSessionResponse,
    ChatStartRequest,
)
from backend.services import chat as chat_service

router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("/start", response_model=ChatSessionResponse)
def start_session(request: ChatStartRequest, owner: str = Depends(current_owner)) -> ChatSessionResponse:
    """Start a subject- or topic-based chat session."""
    mode = request.mode
    if mode == "subject":
        if not request.subject:
            raise HTTPException(status_code=400, detail="A subject is required for subject mode.")
        session_id, label, chat_id = chat_service.start_subject_session(
            request.subject, model=request.model, owner=owner
        )
    elif mode == "topic":
        if not request.topic:
            raise HTTPException(status_code=400, detail="A topic is required for topic mode.")
        session_id, label, chat_id = chat_service.start_topic_session(
            request.topic, model=request.model, owner=owner
        )
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
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="The uploaded PDF was empty.")

    filename = file.filename or "uploaded.pdf"
    try:
        session_id, label, chat_id = chat_service.start_pdf_session(file_bytes, filename, model=model, owner=owner)
    except chat_service.ChatSessionError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

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
def session_overview(session_id: str, refresh: bool = False) -> ChatOverviewResponse:
    """Return a NotebookLM-style overview card (summary + starter questions) for a session.

    Best-effort: the service never raises on LLM/parse failure (it returns an empty
    overview), so the only error here is an unknown session — mapped to the same 404
    the message endpoint uses. ``?refresh=1`` regenerates the card, bypassing the cache.
    """
    try:
        result = chat_service.session_overview(session_id, refresh=refresh)
    except chat_service.ChatSessionError as exc:
        raise HTTPException(status_code=404, detail="Session expired, start a new chat.") from exc

    return ChatOverviewResponse(
        label=result["label"],
        sources=result["sources"],
        overview=result["overview"],
        questions=result["questions"],
    )


@router.post("/message", response_model=ChatMessageResponse)
def send_message(request: ChatMessageRequest) -> ChatMessageResponse:
    """Send a message within an existing session and return the assistant reply."""
    try:
        result = chat_service.ask(request.session_id, request.message)
    except chat_service.ChatSessionError as exc:
        raise HTTPException(status_code=404, detail="Session expired, start a new chat.") from exc

    return ChatMessageResponse(
        answer=result["answer"],
        steps=result.get("steps", []),
        sources=result.get("sources", []),
        artifact=result.get("artifact"),
    )


@router.post("/message/stream")
def stream_message(request: ChatMessageRequest) -> StreamingResponse:
    """Stream the assistant's reasoning live as Server-Sent Events.

    Yields one ``data:`` frame per ``ask_stream`` event — a ``{"type":"step",...}``
    for every trace step the moment it happens, then a terminal ``{"type":"done",...}``
    carrying the final answer, sources, and artifact. The non-streaming
    ``POST /api/chat/message`` endpoint stays unchanged for clients that can't stream.

    An unknown/expired session is reported as a single ``{"type":"error"}`` frame (the
    response has already begun, so we can't switch to a 404), matching the 404 the
    non-stream endpoint returns by surfacing the same message for the client to handle.
    """
    def _events():
        try:
            for event in chat_service.ask_stream(request.session_id, request.message):
                yield f"data: {json.dumps(event)}\n\n"
        except chat_service.ChatSessionError:
            err = {"type": "error", "detail": "Session expired, start a new chat."}
            yield f"data: {json.dumps(err)}\n\n"
        except Exception as exc:  # noqa: BLE001 — report mid-stream failures to the client
            err = {"type": "error", "detail": str(exc) or "Something went wrong."}
            yield f"data: {json.dumps(err)}\n\n"

    return StreamingResponse(
        _events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
