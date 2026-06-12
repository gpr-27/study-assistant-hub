"""Mock Exam API router: generate a mixed MCQ + short-answer exam from a
subject's PDFs, then grade a submission — MCQs in code, short answers via one
batched Groq call. Each graded report is saved to the signed-in user's history
(kind "exams") when persistence is enabled.

DocumentError (unknown subject/PDF) bubbles to the global 404 handler; Groq
errors bubble to the global 429/502/503 handlers; ExamError maps to 502 here.
"""
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from backend import config, db
from backend.auth import current_owner
from backend.config import MAX_CONTENT_TOKENS, require_groq_key
from backend.documents import clip_to_tokens, read_pdf
from backend.services import exam as exam_service
from backend.services.exam import (
    DEFAULT_MCQ,
    DEFAULT_SHORT,
    MAX_MCQ,
    MAX_SHORT,
    SHORT_MAX_SCORE,
    ExamError,
)

router = APIRouter(prefix="/api/exam", tags=["exam"])


# ---------------------------------------------------------------------------
# Local request/response models (per the parallel-build contract: NOT in schemas.py).
# ---------------------------------------------------------------------------
class ExamGenerateRequest(BaseModel):
    mode: str = "subject"                      # v1 supports subject mode only
    subject: str
    pdfs: list[str] = Field(default_factory=list)
    num_mcq: int = Field(default=DEFAULT_MCQ, ge=0, le=MAX_MCQ)
    num_short: int = Field(default=DEFAULT_SHORT, ge=0, le=MAX_SHORT)
    model: str | None = None


class ExamGenerateResponse(BaseModel):
    questions: list[dict]
    title: str
    subject: str


class ExamGradeRequest(BaseModel):
    subject: str
    questions: list[dict]
    answers: dict[str, Any] = Field(default_factory=dict)
    duration_seconds: int | None = Field(default=None, ge=0)
    model: str | None = None


class ExamGradeResponse(BaseModel):
    id: str | None = None                      # None when persistence is off
    title: str
    subject: str
    total_score: int
    max_score: int
    duration_seconds: int | None = None
    per_question: list[dict]
    topic_breakdown: list[dict]


# ---------------------------------------------------------------------------
# Endpoints.
# ---------------------------------------------------------------------------
@router.post("/generate", response_model=ExamGenerateResponse)
def generate_exam(request: ExamGenerateRequest, owner: str = Depends(current_owner)) -> ExamGenerateResponse:
    """Build the exam from the chosen subject PDFs' extracted text (one Groq call)."""
    if request.mode != "subject":
        raise HTTPException(status_code=400, detail="Only subject mode is supported for exams (v1).")
    if not request.pdfs:
        raise HTTPException(status_code=400, detail="Select at least one PDF to build the exam from.")
    if request.num_mcq + request.num_short == 0:
        raise HTTPException(status_code=400, detail="An exam needs at least one question.")
    require_groq_key()
    # DocumentError (unknown subject/PDF) bubbles to the global 404 handler.
    texts = [read_pdf(request.subject, name, owner) for name in request.pdfs]
    content = clip_to_tokens("\n\n".join(t for t in texts if t), MAX_CONTENT_TOKENS)
    if not content.strip():
        raise HTTPException(status_code=422, detail="No extractable text found in the selected PDFs.")
    try:
        data = exam_service.generate_exam(
            content=content,
            num_mcq=request.num_mcq,
            num_short=request.num_short,
            model=request.model,
        )
    except ExamError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    questions = data.get("questions") or []
    if not questions:
        raise HTTPException(status_code=502, detail="The model returned no usable questions; try again.")
    return ExamGenerateResponse(
        questions=questions,
        title=f"Exam · {request.subject}",
        subject=request.subject,
    )


def _coerce_choice(value: Any) -> int | None:
    """Best-effort int from an MCQ answer (index), or None for no/invalid answer."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value.strip())
    return None


@router.post("/grade", response_model=ExamGradeResponse)
def grade_exam(request: ExamGradeRequest, owner: str = Depends(current_owner)) -> ExamGradeResponse:
    """Grade a submitted exam: MCQs in code, short answers in ONE batched Groq call.

    The full report is saved to history (kind "exams") when persistence is on.
    """
    questions = exam_service.validate_questions(request.questions)
    if not questions:
        raise HTTPException(status_code=400, detail="No valid questions were supplied to grade.")

    # Short answers with actual text go to the batched grader; blank ones score 0 in code.
    to_grade = []
    for q in questions:
        if q["type"] != "short":
            continue
        answer = request.answers.get(q["id"])
        text = answer.strip() if isinstance(answer, str) else ""
        if text:
            to_grade.append({"id": q["id"], "question": q["question"],
                             "expected_points": q["expected_points"], "answer": text})
    short_grades: dict[str, dict] = {}
    if to_grade:
        require_groq_key()
        try:
            short_grades = exam_service.grade_short_answers(to_grade, model=request.model)
        except ExamError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    per_question: list[dict] = []
    topics: dict[str, dict] = {}
    total = 0
    max_total = 0
    for q in questions:
        topic = q["topic"]
        if q["type"] == "mcq":
            chosen = _coerce_choice(request.answers.get(q["id"]))
            correct = chosen == q["answer_index"]
            earned, possible = (1 if correct else 0), 1
            right = q["options"][q["answer_index"]]
            feedback = "Correct." if correct else (
                f"Not quite — the correct answer is **{right}**." if chosen is not None
                else f"Unanswered — the correct answer is **{right}**."
            )
            entry = {
                "id": q["id"], "type": "mcq", "topic": topic, "question": q["question"],
                "correct": correct, "score": earned, "max_score": possible,
                "answer_index": q["answer_index"], "user_answer": chosen,
                "feedback": feedback,
            }
        else:
            graded = short_grades.get(q["id"])
            if graded is None:
                earned, feedback = 0, "No answer was provided."
            else:
                earned, feedback = graded["score"], graded["feedback"]
            possible = SHORT_MAX_SCORE
            entry = {
                "id": q["id"], "type": "short", "topic": topic, "question": q["question"],
                "score": earned, "max_score": possible, "feedback": feedback,
            }
        per_question.append(entry)
        total += earned
        max_total += possible
        agg = topics.setdefault(topic, {"topic": topic, "earned": 0, "possible": 0})
        agg["earned"] += earned
        agg["possible"] += possible

    topic_breakdown = list(topics.values())
    title = f"Exam · {request.subject}"
    saved_id = db.save("exams", owner, {       # returns None when persistence is off
        "title": title,
        "subject": request.subject,
        "model": config.model_for("exam", request.model),
        "total_score": total,
        "max_score": max_total,
        "duration_seconds": request.duration_seconds,
        "per_question": per_question,
        "topic_breakdown": topic_breakdown,
        "questions": questions,
    })
    return ExamGradeResponse(
        id=saved_id, title=title, subject=request.subject,
        total_score=total, max_score=max_total,
        duration_seconds=request.duration_seconds,
        per_question=per_question, topic_breakdown=topic_breakdown,
    )
