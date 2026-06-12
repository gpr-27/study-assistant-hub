"""Past-paper Question Predictor router: upload 1–5 past-paper PDFs, get the
recurring patterns across them and the questions most likely to appear in the
next exam. Each analysis is saved to the signed-in user's history (kind
"predictions") when persistence is enabled.

Request/response models are defined locally (this feature does not touch
backend/schemas.py).
"""
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from backend import config, db
from backend.auth import current_owner
from backend.config import MAX_CONTENT_TOKENS, require_groq_key
from backend.documents import clip_to_tokens, extract_pdf_text
from backend.services.predictor import PredictorError, predict_questions

router = APIRouter(prefix="/api/predictor", tags=["predictor"])

_MAX_FILES = 5


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
    require_groq_key()

    # Extract every paper's text and combine into one labelled corpus, then clip the
    # COMBINED corpus to the shared token budget (one bounded Groq call total).
    chunks: list[str] = []
    first_filename: str | None = None
    for upload in files:
        data = await upload.read()
        if not data:
            continue
        text = extract_pdf_text(data)
        if not text:
            continue
        fname = upload.filename or "paper.pdf"
        if first_filename is None:
            first_filename = fname
        chunks.append(f"=== PAPER: {fname} ===\n{text}")
    if not chunks:
        raise HTTPException(status_code=422, detail="No extractable text found in the uploaded PDF(s).")
    corpus = clip_to_tokens("\n\n".join(chunks), MAX_CONTENT_TOKENS)

    hint = (subject_hint or "").strip() or None
    try:
        data = predict_questions(corpus, subject_hint=hint, model=model)
    except PredictorError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    patterns = [Pattern(**item) for item in data.get("patterns", [])]
    predictions = [PredictedQuestion(**item) for item in data.get("predicted_questions", [])]
    if not predictions:
        raise HTTPException(status_code=502, detail="The model returned no usable predictions; try again.")

    title = f"Predictions · {hint or first_filename or 'past papers'}"
    saved_id = db.save("predictions", owner, {
        "title": title,
        "source": hint or first_filename or "past papers",
        "subject": hint or "",
        "model": config.model_for("predictor", model),
        "num_papers": len(chunks),
        "patterns": [p.model_dump() for p in patterns],
        "predicted_questions": [p.model_dump() for p in predictions],
    })
    return PredictorResponse(patterns=patterns, predicted_questions=predictions, id=saved_id, title=title)
