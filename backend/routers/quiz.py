"""Quiz API router: generate a multiple-choice quiz from a subject PDF, a free
topic, or an uploaded PDF — mirroring the Chat tool's three sources. Each generated
quiz is saved to the signed-in user's history (when persistence is enabled), including the
`llm` (provider/model label) that produced it.

DocumentError bubbles to the global 404 handler; LLMChainError (every provider failed) to the
global 503 handler; QuizError (unusable model output) maps to 502 here.
"""
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from backend import db
from backend import privacy
from backend.auth import current_owner
from backend.documents import read_pdf
from backend.routers._common import model_of, read_pdf_upload, require_text, require_topic
from backend.services import grading
from backend.schemas import QuizQuestion, QuizRequest, QuizResponse, QuizTopicRequest
from backend.services.quiz import MAX_QUESTIONS, QuizError, make_quiz, make_quiz_from_topic

router = APIRouter(prefix="/api/quiz", tags=["quiz"])


def _finalize(data: dict, owner: str, *, title: str, source: str, difficulty: str) -> QuizResponse:
    """Validate questions, save the quiz to this user's history, and return it with its id."""
    questions = [QuizQuestion(**{**item, "sig": grading.compute_sig(grading.normalise_item(item))})
                 for item in data.get("questions", []) if grading.normalise_item(item) is not None]
    if not questions:
        raise HTTPException(status_code=502, detail="The model returned no usable questions; try again.")
    llm = data.get("llm")
    saved_id = db.save("quizzes", owner, {
        "title": title,
        "source": source,
        "difficulty": difficulty,
        "model": model_of(llm),
        "llm": llm,
        "num_questions": len(questions),
        "questions": [q.model_dump() for q in questions],
    })
    return QuizResponse(questions=questions, id=saved_id, title=title, llm=llm)


def _run(fn, *args) -> dict:
    try:
        return fn(*args)
    except QuizError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/generate", response_model=QuizResponse)
def generate_quiz(request: QuizRequest, owner: str = Depends(current_owner)) -> QuizResponse:
    """Build the quiz from the chosen subject PDF's extracted text."""
    privacy.mark(privacy.for_subject(request.subject))      # user-uploaded subject => no Gemini
    content = require_text(read_pdf(request.subject, request.pdf, owner))
    data = _run(make_quiz, content, request.num_questions, request.difficulty)
    return _finalize(
        data, owner,
        title=f"Quiz · {request.subject} / {request.pdf}",
        source=f"{request.subject} / {request.pdf}",
        difficulty=request.difficulty,
    )


@router.post("/topic", response_model=QuizResponse)
def generate_quiz_topic(request: QuizTopicRequest, owner: str = Depends(current_owner)) -> QuizResponse:
    """Generate a quiz about a free topic (no document)."""
    topic = require_topic(request.topic)
    data = _run(make_quiz_from_topic, topic, request.num_questions, request.difficulty)
    return _finalize(data, owner, title=f"Quiz · {topic}", source=topic, difficulty=request.difficulty)


@router.post("/upload", response_model=QuizResponse)
async def generate_quiz_upload(
    file: UploadFile = File(...),
    num_questions: int = Form(5, ge=1, le=MAX_QUESTIONS),
    difficulty: str = Form("medium", max_length=20),
    model: str | None = Form(None),
    owner: str = Depends(current_owner),
) -> QuizResponse:
    """Generate a quiz from a single uploaded PDF's extracted text."""
    text, fname = await read_pdf_upload(file)
    privacy.mark(privacy.for_upload())                       # uploaded PDF => no Gemini

    def work() -> QuizResponse:
        data = _run(make_quiz, text, num_questions, difficulty)
        return _finalize(data, owner, title=f"Quiz · {fname}", source=fname, difficulty=difficulty)

    return await run_in_threadpool(work)
