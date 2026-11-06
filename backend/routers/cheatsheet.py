"""Cheat-sheet router: build a one-page exam crib sheet from a subject PDF, a topic or an upload.
Each generated sheet is saved to the signed-in user's history (when persistence is on), including
the `llm` that produced it.

DocumentError (unknown subject/PDF) bubbles to the global 404 handler; LLMChainError to the
global 503 handler; CheatsheetError maps to 502 here.
"""
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from backend import db
from backend import privacy
from backend.auth import current_owner
from backend.documents import read_pdf
from backend.routers._common import model_of, read_pdf_upload, require_text, require_topic
from backend.services.cheatsheet import (
    CheatsheetError,
    make_cheatsheet_ex,
    make_cheatsheet_from_topic_ex,
)

router = APIRouter(prefix="/api/cheatsheet", tags=["cheatsheet"])


# ---- local request/response models (this feature's contract lives here) ----
class CheatsheetRequest(BaseModel):
    mode: str = Field("subject", max_length=20, description="Only 'subject' is supported for JSON requests.")
    subject: str | None = Field(None, max_length=255)
    pdf: str | None = Field(None, max_length=255)
    model: str | None = None


class CheatsheetTopicRequest(BaseModel):
    topic: str = Field(max_length=400)
    model: str | None = None


class CheatsheetResponse(BaseModel):
    markdown: str
    title: str
    id: str | None = None
    llm: dict | None = None


def _finalize(owner: str, markdown: str, llm: dict | None, *, title: str, source: str) -> CheatsheetResponse:
    """Save the generated sheet to history (tolerating no persistence) and respond."""
    saved_id = db.save("cheatsheets", owner, {
        "title": title,
        "source": source,
        "model": model_of(llm),
        "llm": llm,
        "markdown": markdown,
    })
    return CheatsheetResponse(markdown=markdown, title=title, id=saved_id, llm=llm)


def _build(owner: str, *, title: str, source: str, content: str) -> CheatsheetResponse:
    """Run the single chain call from document text, save to history, respond."""
    try:
        markdown, llm = make_cheatsheet_ex(content)
    except CheatsheetError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return _finalize(owner, markdown, llm, title=title, source=source)


@router.post("/generate", response_model=CheatsheetResponse)
def generate(request: CheatsheetRequest, owner: str = Depends(current_owner)) -> CheatsheetResponse:
    """Build a cheat sheet from one subject PDF's extracted text."""
    if request.mode != "subject":
        raise HTTPException(status_code=400, detail=f"Unsupported cheat-sheet mode: {request.mode}")
    if not request.subject or not request.pdf:
        raise HTTPException(status_code=400, detail="A subject and one PDF are required.")
    privacy.mark(privacy.for_subject(request.subject))
    content = require_text(read_pdf(request.subject, request.pdf, owner))
    return _build(owner, title=f"Cheat sheet · {request.pdf}",
                  source=f"{request.subject} / {request.pdf}", content=content)


@router.post("/topic", response_model=CheatsheetResponse)
def topic(request: CheatsheetTopicRequest, owner: str = Depends(current_owner)) -> CheatsheetResponse:
    """Build a cheat sheet about a free topic (no document)."""
    label = require_topic(request.topic)
    try:
        markdown, llm = make_cheatsheet_from_topic_ex(label)
    except CheatsheetError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return _finalize(owner, markdown, llm, title=f"Cheat sheet · {label}", source=label)


@router.post("/upload", response_model=CheatsheetResponse)
async def upload(
    file: UploadFile = File(...),
    model: str | None = Form(None),
    owner: str = Depends(current_owner),
) -> CheatsheetResponse:
    """Build a cheat sheet from a single uploaded PDF."""
    text, fname = await read_pdf_upload(file)
    privacy.mark(privacy.for_upload())
    return await run_in_threadpool(lambda: _build(owner, title=f"Cheat sheet · {fname}", source=fname, content=text))
