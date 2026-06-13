"""Cheat-sheet router: build a one-page exam crib sheet from a subject PDF or an upload.
Each generated sheet is saved to the signed-in user's history (when persistence is on).

DocumentError (unknown subject/PDF) bubbles to the global 404 handler; Groq errors
bubble to the global 429/502 handlers; CheatsheetError maps to 502 here.
"""
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from backend import config, db
from backend.auth import current_owner
from backend.config import MAX_CONTENT_TOKENS, require_groq_key
from backend.documents import clip_to_tokens, extract_pdf_text, read_pdf
from backend.services.cheatsheet import (
    CheatsheetError,
    make_cheatsheet,
    make_cheatsheet_from_topic,
)

router = APIRouter(prefix="/api/cheatsheet", tags=["cheatsheet"])


# ---- local request/response models (this feature's contract lives here) ----
class CheatsheetRequest(BaseModel):
    mode: str = Field("subject", description="Only 'subject' is supported for JSON requests.")
    subject: str | None = None
    pdf: str | None = None
    model: str | None = None


class CheatsheetTopicRequest(BaseModel):
    topic: str
    model: str | None = None


class CheatsheetResponse(BaseModel):
    markdown: str
    title: str
    id: str | None = None


def _finalize(owner: str, markdown: str, *, title: str, source: str, model: str | None) -> CheatsheetResponse:
    """Save the generated sheet to history (tolerating no persistence) and respond."""
    saved_id = db.save("cheatsheets", owner, {
        "title": title,
        "source": source,
        "model": config.model_for("cheatsheet", model),
        "markdown": markdown,
    })
    return CheatsheetResponse(markdown=markdown, title=title, id=saved_id)


def _build(owner: str, *, title: str, source: str, content: str, model: str | None) -> CheatsheetResponse:
    """Run the single Groq call from document text, save to history, respond."""
    require_groq_key()
    try:
        markdown = make_cheatsheet(content, model=model)
    except CheatsheetError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return _finalize(owner, markdown, title=title, source=source, model=model)


@router.post("/generate", response_model=CheatsheetResponse)
def generate(request: CheatsheetRequest, owner: str = Depends(current_owner)) -> CheatsheetResponse:
    """Build a cheat sheet from one subject PDF's extracted text."""
    if request.mode != "subject":
        raise HTTPException(status_code=400, detail=f"Unsupported cheat-sheet mode: {request.mode}")
    if not request.subject or not request.pdf:
        raise HTTPException(status_code=400, detail="A subject and one PDF are required.")
    content = clip_to_tokens(read_pdf(request.subject, request.pdf, owner), MAX_CONTENT_TOKENS)
    return _build(
        owner,
        title=f"Cheat sheet · {request.pdf}",
        source=f"{request.subject} / {request.pdf}",
        content=content,
        model=request.model,
    )


@router.post("/topic", response_model=CheatsheetResponse)
def topic(request: CheatsheetTopicRequest, owner: str = Depends(current_owner)) -> CheatsheetResponse:
    """Build a cheat sheet about a free topic (no document)."""
    label = (request.topic or "").strip()
    if not label:
        raise HTTPException(status_code=400, detail="A topic is required.")
    require_groq_key()
    try:
        markdown = make_cheatsheet_from_topic(label, model=request.model)
    except CheatsheetError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return _finalize(owner, markdown, title=f"Cheat sheet · {label}", source=label, model=request.model)


@router.post("/upload", response_model=CheatsheetResponse)
async def upload(
    file: UploadFile = File(...),
    model: str | None = Form(None),
    owner: str = Depends(current_owner),
) -> CheatsheetResponse:
    """Build a cheat sheet from a single uploaded PDF."""
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="The uploaded PDF was empty.")
    text = extract_pdf_text(data)
    if not text:
        raise HTTPException(status_code=422, detail="No extractable text found in that PDF.")
    fname = file.filename or "uploaded.pdf"
    content = clip_to_tokens(text, MAX_CONTENT_TOKENS)
    return _build(owner, title=f"Cheat sheet · {fname}", source=fname, content=content, model=model)
