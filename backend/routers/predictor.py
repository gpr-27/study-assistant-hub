"""Past-paper Question Predictor router: upload 1–5 past-paper PDFs, get the
recurring patterns across them and the questions most likely to appear in the
next exam. Each analysis is saved to the signed-in user's history (kind
"predictions") when persistence is enabled.

Request/response models are defined locally (this feature does not touch
backend/schemas.py).
"""
import asyncio

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from backend import db
from backend import privacy
from backend.auth import current_owner
from backend.config import require_llm_key
from backend.documents import extract_pdf_text
from backend.services.predictor import PredictorError, predict_questions

router = APIRouter(prefix="/api/predictor", tags=["predictor"])

_MAX_FILES = 5
_MAX_BYTES = 12 * 1024 * 1024        # per paper


# ---- local Pydantic models (per the parallel-build rules; NOT in schemas.py) ----
class Pattern(BaseModel):
    topic: str
    frequency_note: str = ""
    trend: str = ""


class PredictedQuestion(BaseModel):
    question: str
    topic: str = ""
    likelihood: str = Field(default="low", pattern="^(high|medium|low)$")
    reasoning: str = ""


class PredictorResponse(BaseModel):
    patterns: list[Pattern]
    predicted_questions: list[PredictedQuestion]
    id: str | None = None
    title: str
    llm: dict | None = None


@router.post("/analyze", response_model=PredictorResponse)
async def analyze_papers(
    files: list[UploadFile] = File(...),
    subject_hint: str | None = Form(None),
    model: str | None = Form(None),
    owner: str = Depends(current_owner),
) -> PredictorResponse:
    """Analyze 1–5 past-paper PDFs and predict likely next-exam questions."""
    if not files:
        raise HTTPException(status_code=400, detail="Upload at least one past-paper PDF.")
    if len(files) > _MAX_FILES:
        raise HTTPException(status_code=400, detail=f"Upload at most {_MAX_FILES} past-paper PDFs.")
    require_llm_key()
    privacy.mark(privacy.for_upload())          # past papers the user uploaded => no Gemini

    # Extract every paper's text and combine into one labelled corpus, then clip the
    # COMBINED corpus to the shared token budget (one bounded Groq call total).
    chunks: list[str] = []
    first_filename: str | None = None
    for upload in files:
        data = await upload.read(_MAX_BYTES + 1)
        if not data:
            continue
        if len(data) > _MAX_BYTES:
            raise HTTPException(status_code=413, detail=f"{upload.filename or 'paper'!r} is larger than 12 MB.")
        try:
            text = await asyncio.to_thread(extract_pdf_text, data)    # pypdf is blocking
        except Exception:  # noqa: BLE001 — corrupt / encrypted PDF
            raise HTTPException(status_code=422, detail=f"{upload.filename or 'paper'!r} could not be read as a PDF.")
        if not text:
            continue
        fname = (upload.filename or "paper.pdf")[:200]
        if first_filename is None:
            first_filename = fname
        chunks.append(f"=== PAPER: {fname} ===\n{text}")
    if not chunks:
        raise HTTPException(status_code=422, detail="No extractable text found in the uploaded PDF(s).")
    corpus = "\n\n".join(chunks)    # full text; the LLM chain fits it per provider

    hint = (subject_hint or "").strip()[:200] or None
    try:
        data = await asyncio.to_thread(predict_questions, corpus, hint, model)    # blocking LLM call
    except PredictorError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    patterns = [Pattern(**item) for item in data.get("patterns", [])]
    predictions = [PredictedQuestion(**item) for item in data.get("predicted_questions", [])]
    if not predictions:
        raise HTTPException(status_code=502, detail="The model returned no usable predictions; try again.")

    title = f"Predictions · {hint or first_filename or 'past papers'}"
    llm_public = data.get("llm")
    saved_id = await asyncio.to_thread(db.save, "predictions", owner, {
        "title": title,
        "source": hint or first_filename or "past papers",
        "subject": hint or "",
        "llm": llm_public,
        "num_papers": len(chunks),
        "patterns": [p.model_dump() for p in patterns],
        "predicted_questions": [p.model_dump() for p in predictions],
    })
    return PredictorResponse(patterns=patterns, predicted_questions=predictions, id=saved_id, title=title,
                             llm=llm_public)
