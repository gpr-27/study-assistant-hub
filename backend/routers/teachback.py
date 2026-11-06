"""Teach-back API router: start a Socratic Feynman-technique session (topic or
subject PDF) and exchange messages until the tutor returns a final evaluation.

DocumentError and MissingApiKeyError bubble up to the global handlers in main.py
(mapped to 404 and 503 respectively); LLMChainError (every provider failed) is a 503 with the
"no model" llm label via the app-wide handler. Parse failures surface as 502; the
50-message session cap surfaces as a friendly 403. Request/response models are
defined locally (schemas.py is untouched).
"""
import asyncio
from typing import Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from backend import privacy
from backend.auth import current_owner
from backend.config import require_llm_key
from backend.documents import extract_pdf_text, read_pdf
from backend.services.teachback import (
    TeachbackError,
    TeachbackSessionError,
    TeachbackSessionExpiredError,
    respond,
    start_session,
)

router = APIRouter(prefix="/api/teachback", tags=["teachback"])

_MAX_UPLOAD_BYTES = 12 * 1024 * 1024


# ---- local request/response contracts ----
class TeachbackStartRequest(BaseModel):
    mode: Literal["topic", "subject"]
    topic: str | None = Field(default=None, max_length=400)      # topic mode
    subject: str | None = Field(default=None, max_length=255)   # subject mode
    pdf: str | None = Field(default=None, max_length=255)       # subject mode
    model: str | None = None      # accepted for back-compat, ignored (the chain picks the model)


class TeachbackStartResponse(BaseModel):
    session_id: str
    concept: str
    opening: str
    llm: dict | None = None


class TeachbackMessageRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=64)
    message: str = Field(max_length=8000)


class TeachbackEvaluation(BaseModel):
    understanding_score_0_to_10: int = Field(ge=0, le=10)
    strengths: list[str]
    gaps: list[str]
    suggested_review: list[str]


class TeachbackMessageResponse(BaseModel):
    reply: str
    done: bool
    evaluation: TeachbackEvaluation | None = None
    llm: dict | None = None
    result: dict | None = None      # unified graded result (same shape as POST /api/grade) once done


# ---- endpoints ----
@router.post("/start", response_model=TeachbackStartResponse)
def start_teachback(request: TeachbackStartRequest, owner: str = Depends(current_owner)) -> TeachbackStartResponse:
    """Open a session: the LLM picks the concept and writes the opening prompt."""
    require_llm_key()

    if request.mode == "topic":
        topic = (request.topic or "").strip()
        if not topic:
            raise HTTPException(status_code=400, detail="A topic is required for topic mode.")
        try:
            session_id, concept, opening, llm = start_session(
                "topic", topic=topic, label=f"Topic · {topic}", model=request.model, owner=owner,
            )
        except TeachbackError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
    else:  # mode == "subject" (Literal guarantees one of the two)
        if not request.subject or not request.pdf:
            raise HTTPException(status_code=400, detail="A subject and a PDF are required for subject mode.")
        privacy.mark(privacy.for_subject(request.subject))
        # DocumentError (unknown subject/PDF) bubbles to the global 404 handler.
        content = read_pdf(request.subject, request.pdf, owner)
        if not content:
            raise HTTPException(status_code=422, detail="No extractable text found in that PDF.")
        try:
            session_id, concept, opening, llm = start_session(
                "subject", content=content,
                label=f"{request.subject} / {request.pdf}",
                model=request.model, owner=owner,
            )
        except TeachbackError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    return TeachbackStartResponse(session_id=session_id, concept=concept, opening=opening, llm=llm)


@router.post("/start-upload", response_model=TeachbackStartResponse)
async def start_teachback_upload(
    file: UploadFile = File(...),
    model: str | None = Form(None),
    owner: str = Depends(current_owner),
) -> TeachbackStartResponse:
    """Open a session from a single uploaded PDF: extract its text and feed it as
    subject-mode content to the session opener (no service change needed)."""
    require_llm_key()
    data = await file.read(_MAX_UPLOAD_BYTES + 1)
    if not data:
        raise HTTPException(status_code=400, detail="The uploaded PDF was empty.")
    if len(data) > _MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="That PDF is too large (limit 12 MB).")
    try:
        text = await asyncio.to_thread(extract_pdf_text, data)    # pypdf is blocking
    except Exception:  # noqa: BLE001 — corrupt / encrypted PDF
        raise HTTPException(status_code=422, detail="That file could not be read as a PDF.")
    if not text:
        raise HTTPException(status_code=422, detail="No extractable text found in that PDF.")
    fname = file.filename or "uploaded.pdf"
    privacy.mark(privacy.for_upload())
    content = text
    try:
        session_id, concept, opening, llm = await asyncio.to_thread(     # blocking LLM call
            lambda: start_session("subject", content=content, label=fname[:200], model=model, owner=owner))
    except TeachbackError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return TeachbackStartResponse(session_id=session_id, concept=concept, opening=opening, llm=llm)


@router.post("/message", response_model=TeachbackMessageResponse)
def send_teachback_message(request: TeachbackMessageRequest, owner: str = Depends(current_owner)) -> TeachbackMessageResponse:
    """One student turn. done=true carries the final evaluation (session is closed)."""
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Say something — explain the concept in your own words.")
    require_llm_key()
    try:
        result = respond(request.session_id, request.message, owner)
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
        llm=result.get("llm"),
        result=result.get("result"),
    )
