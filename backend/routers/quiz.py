"""Quiz API router: generate a multiple-choice quiz from a subject PDF, a free
topic, or an uploaded PDF — mirroring the Chat tool's three sources. Each generated
quiz is saved to the signed-in user's history (when persistence is enabled)."""
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from backend import config, db
from backend.auth import current_owner
from backend.config import MAX_CONTENT_TOKENS
from backend.documents import clip_to_tokens, extract_pdf_text, read_pdf
from backend.schemas import QuizQuestion, QuizRequest, QuizResponse, QuizTopicRequest
from backend.services.quiz import QuizError, make_quiz, make_quiz_from_topic

router = APIRouter(prefix="/api/quiz", tags=["quiz"])


def _finalize(data: dict, owner: str, *, title: str, source: str, difficulty: str, model) -> QuizResponse:
    """Validate questions, save the quiz to this user's history, and return it with its id."""
    questions = [QuizQuestion(**item) for item in data.get("questions", [])]
    if not questions:
        raise HTTPException(status_code=502, detail="The model returned no usable questions; try again.")
    saved_id = db.save("quizzes", owner, {
        "title": title,
        "source": source,
        "difficulty": difficulty,
        "model": config.model_for("quiz", model),
        "num_questions": len(questions),
        "questions": [q.model_dump() for q in questions],
    })
    return QuizResponse(questions=questions, id=saved_id, title=title)


@router.post("/generate", response_model=QuizResponse)
def generate_quiz(request: QuizRequest, owner: str = Depends(current_owner)) -> QuizResponse:
    """Build the quiz from the chosen subject PDF's extracted text."""
    content = clip_to_tokens(read_pdf(request.subject, request.pdf, owner), MAX_CONTENT_TOKENS)
    try:
        data = make_quiz(
            content=content,
            num_questions=request.num_questions,
            difficulty=request.difficulty,
            model=request.model,
        )
    except QuizError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return _finalize(
        data, owner,
        title=f"Quiz · {request.subject} / {request.pdf}",
        source=f"{request.subject} / {request.pdf}",
        difficulty=request.difficulty, model=request.model,
    )


@router.post("/topic", response_model=QuizResponse)
def generate_quiz_topic(request: QuizTopicRequest, owner: str = Depends(current_owner)) -> QuizResponse:
    """Generate a quiz about a free topic (no document)."""
    try:
        data = make_quiz_from_topic(
            topic=request.topic,
            num_questions=request.num_questions,
            difficulty=request.difficulty,
            model=request.model,
        )
    except QuizError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return _finalize(
        data, owner,
        title=f"Quiz · {request.topic}", source=request.topic,
        difficulty=request.difficulty, model=request.model,
    )


@router.post("/upload", response_model=QuizResponse)
async def generate_quiz_upload(
    file: UploadFile = File(...),
    num_questions: int = Form(5),
    difficulty: str = Form("medium"),
    model: str | None = Form(None),
    owner: str = Depends(current_owner),
) -> QuizResponse:
    """Generate a quiz from a single uploaded PDF's extracted text."""
    data_bytes = await file.read()
    if not data_bytes:
        raise HTTPException(status_code=400, detail="The uploaded PDF was empty.")
    text = extract_pdf_text(data_bytes)
    if not text:
        raise HTTPException(status_code=422, detail="No extractable text found in that PDF.")
    fname = file.filename or "uploaded.pdf"
    content = clip_to_tokens(text, MAX_CONTENT_TOKENS)
    try:
        data = make_quiz(content=content, num_questions=num_questions, difficulty=difficulty, model=model)
    except QuizError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return _finalize(data, owner, title=f"Quiz · {fname}", source=fname, difficulty=difficulty, model=model)
