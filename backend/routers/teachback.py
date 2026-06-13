"""Teach-back API router: start a Socratic Feynman-technique session (topic or
subject PDF) and exchange messages until the tutor returns a final evaluation.

DocumentError and MissingApiKeyError bubble up to the global handlers in main.py
(mapped to 404 and 503 respectively). LLM/parse failures surface as 502; the
50-message session cap surfaces as a friendly 403. Request/response models are
defined locally (schemas.py is untouched).
"""
from typing import Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from backend.auth import current_owner
from backend.config import MAX_CONTENT_TOKENS, require_groq_key
from backend.documents import clip_to_tokens, extract_pdf_text, read_pdf
from backend.services.teachback import (
    TeachbackError,
    TeachbackSessionError,
    TeachbackSessionExpiredError,
    respond,
    start_session,
)

router = APIRouter(prefix="/api/teachback", tags=["teachback"])


# ---- local request/response contracts ----
class TeachbackStartRequest(BaseModel):
    mode: Literal["topic", "subject"]
    topic: str | None = None      # topic mode
    subject: str | None = None    # subject mode
    pdf: str | None = None        # subject mode
    model: str | None = None


class TeachbackStartResponse(BaseModel):
    session_id: str
    concept: str
    opening: str


class TeachbackMessageRequest(BaseModel):
    session_id: str
    message: str


class TeachbackEvaluation(BaseModel):
    understanding_score_0_to_10: int = Field(ge=0, le=10)
    strengths: list[str]
    gaps: list[str]
    suggested_review: list[str]


class TeachbackMessageResponse(BaseModel):
    reply: str
    done: bool
    evaluation: TeachbackEvaluation | None = None


# ---- endpoints ----
@router.post("/start", response_model=TeachbackStartResponse)
def start_teachback(request: TeachbackStartRequest, owner: str = Depends(current_owner)) -> TeachbackStartResponse:
    """Open a session: the LLM picks the concept and writes the opening prompt."""
    require_groq_key()

    if request.mode == "topic":
        topic = (request.topic or "").strip()
        if not topic:
            raise HTTPException(status_code=400, detail="A topic is required for topic mode.")
        try:
            session_id, concept, opening = start_session(
                "topic", topic=topic, label=f"Topic · {topic}", model=request.model, owner=owner,
            )
        except TeachbackError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
    else:  # mode == "subject" (Literal guarantees one of the two)
        if not request.subject or not request.pdf:
            raise HTTPException(status_code=400, detail="A subject and a PDF are required for subject mode.")
        # DocumentError (unknown subject/PDF) bubbles to the global 404 handler.
        content = clip_to_tokens(read_pdf(request.subject, request.pdf, owner), MAX_CONTENT_TOKENS)
        if not content:
            raise HTTPException(status_code=422, detail="No extractable text found in that PDF.")
        try:
            session_id, concept, opening = start_session(
                "subject", content=content,
                label=f"{request.subject} / {request.pdf}",
                model=request.model, owner=owner,
            )
        except TeachbackError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    return TeachbackStartResponse(session_id=session_id, concept=concept, opening=opening)


@router.post("/start-upload", response_model=TeachbackStartResponse)
async def start_teachback_upload(
    file: UploadFile = File(...),
    model: str | None = Form(None),
    owner: str = Depends(current_owner),
) -> TeachbackStartResponse:
    """Open a session from a single uploaded PDF: extract its text and feed it as
    subject-mode content to the session opener (no service change needed)."""
    require_groq_key()
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="The uploaded PDF was empty.")
    text = extract_pdf_text(data)
    if not text:
        raise HTTPException(status_code=422, detail="No extractable text found in that PDF.")
    fname = file.filename or "uploaded.pdf"
    content = clip_to_tokens(text, MAX_CONTENT_TOKENS)
    try:
        session_id, concept, opening = start_session(
            "subject", content=content, label=fname, model=model, owner=owner,
        )
    except TeachbackError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return TeachbackStartResponse(session_id=session_id, concept=concept, opening=opening)


@router.post("/message", response_model=TeachbackMessageResponse)
def send_teachback_message(request: TeachbackMessageRequest, owner: str = Depends(current_owner)) -> TeachbackMessageResponse:
    """One student turn. done=true carries the final evaluation (session is closed)."""
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Say something — explain the concept in your own words.")
    require_groq_key()
    try:
        result = respond(request.session_id, request.message)
    except TeachbackSessionExpiredError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except TeachbackSessionError as exc:
        raise HTTPException(status_code=404, detail="Session expired — start a new teach-back.") from exc
    except TeachbackError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    evaluation = result.get("evaluation")
    return TeachbackMessageResponse(
        reply=result["reply"],
        done=bool(result["done"]),
        evaluation=TeachbackEvaluation(**evaluation) if evaluation else None,
    )
