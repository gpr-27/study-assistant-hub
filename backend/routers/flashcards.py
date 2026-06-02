"""Flashcards API: generate active-recall cards from a subject PDF, a free
topic, or an uploaded PDF — mirroring the Chat tool's three sources. Each generated
deck is saved to the signed-in user's history (when persistence is enabled)."""
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from backend import config, db
from backend.auth import current_owner
from backend.documents import extract_pdf_text, read_pdf
from backend.schemas import (
    Flashcard,
    FlashcardsRequest,
    FlashcardsResponse,
    FlashcardsTopicRequest,
)
from backend.services.flashcards import (
    FlashcardError,
    make_flashcards,
    make_flashcards_from_topic,
)

router = APIRouter(prefix="/api/flashcards", tags=["flashcards"])


def _finalize(result: dict, owner: str, *, title: str, source: str, difficulty: str, model) -> FlashcardsResponse:
    """Validate cards, save the deck to this user's history, and return it with its id."""
    cards = [Flashcard(**card) for card in result.get("cards", [])]
    if not cards:
        raise HTTPException(
            status_code=502,
            detail="The model did not return any usable flashcards. Try again or pick a different source.",
        )
    saved_id = db.save("flashcards", owner, {
        "title": title,
        "source": source,
        "difficulty": difficulty,
        "model": config.model_for("flashcards", model),
        "num_cards": len(cards),
        "cards": [c.model_dump() for c in cards],
    })
    return FlashcardsResponse(cards=cards, id=saved_id, title=title)


@router.post("/generate", response_model=FlashcardsResponse)
def generate_flashcards(request: FlashcardsRequest, owner: str = Depends(current_owner)) -> FlashcardsResponse:
    """Generate flashcards from the requested subject PDF's extracted text.

    Document/key errors bubble to global handlers (404 / 503). Generation or
    parse failures map to 502; an empty result is treated as a 502 too.
    """
    content = read_pdf(request.subject, request.pdf)
    try:
        result = make_flashcards(
            content=content,
            num_cards=request.num_cards,
            difficulty=request.difficulty,
            model=request.model,
        )
    except FlashcardError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return _finalize(
        result, owner,
        title=f"Flashcards · {request.subject} / {request.pdf}",
        source=f"{request.subject} / {request.pdf}",
        difficulty=request.difficulty, model=request.model,
    )


@router.post("/topic", response_model=FlashcardsResponse)
def generate_flashcards_topic(request: FlashcardsTopicRequest, owner: str = Depends(current_owner)) -> FlashcardsResponse:
    """Generate flashcards about a free topic (no document)."""
    try:
        result = make_flashcards_from_topic(
            topic=request.topic,
            num_cards=request.num_cards,
            difficulty=request.difficulty,
            model=request.model,
        )
    except FlashcardError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return _finalize(
        result, owner,
        title=f"Flashcards · {request.topic}", source=request.topic,
        difficulty=request.difficulty, model=request.model,
    )


@router.post("/upload", response_model=FlashcardsResponse)
async def generate_flashcards_upload(
    file: UploadFile = File(...),
    num_cards: int = Form(10),
    difficulty: str = Form("medium"),
    model: str | None = Form(None),
    owner: str = Depends(current_owner),
) -> FlashcardsResponse:
    """Generate flashcards from a single uploaded PDF's extracted text."""
    data_bytes = await file.read()
    if not data_bytes:
        raise HTTPException(status_code=400, detail="The uploaded PDF was empty.")
    content = extract_pdf_text(data_bytes)
    if not content:
        raise HTTPException(status_code=422, detail="No extractable text found in that PDF.")
    try:
        result = make_flashcards(content=content, num_cards=num_cards, difficulty=difficulty, model=model)
    except FlashcardError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    fname = file.filename or "uploaded.pdf"
    return _finalize(result, owner, title=f"Flashcards · {fname}", source=fname, difficulty=difficulty, model=model)
