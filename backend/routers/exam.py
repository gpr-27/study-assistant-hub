"""Mock Exam API router: generate a mixed MCQ + short-answer exam from a
subject's PDFs, then grade a submission — MCQs in code, short answers via one
batched chain call. Each graded report is saved to the signed-in user's history
(kind "exams") when persistence is enabled, with the `llm` that graded it.

DocumentError (unknown subject/PDF) bubbles to the global 404 handler; LLMChainError (all
providers failed) to the global 503 handler; ExamError maps to 502 here.
"""
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from backend import db
from backend import privacy
from backend.auth import current_owner
from backend.documents import read_pdf
from backend.routers._common import read_pdf_upload, require_topic
from backend.routers.grading import GradeRequest, run_grade
from backend.services import _gen, grading
from backend.services import exam as exam_service
from backend.services.exam import (
    DEFAULT_MCQ,
    DEFAULT_SHORT,
    MAX_MCQ,
    MAX_SHORT,
    SHORT_MAX_SCORE,
    ExamError,
)

MAX_PDFS = 8
MAX_GRADE_QUESTIONS = 40

router = APIRouter(prefix="/api/exam", tags=["exam"])


# ---------------------------------------------------------------------------
# Local request/response models (per the parallel-build contract: NOT in schemas.py).
# ---------------------------------------------------------------------------
class ExamGenerateRequest(BaseModel):
    mode: str = "subject"                      # v1 supports subject mode only
    subject: str = Field(max_length=255)
    pdfs: list[str] = Field(default_factory=list, max_length=MAX_PDFS)
    num_mcq: int = Field(default=DEFAULT_MCQ, ge=0, le=MAX_MCQ)
    num_short: int = Field(default=DEFAULT_SHORT, ge=0, le=MAX_SHORT)
    difficulty: str = Field(default="medium", max_length=20)
    model: str | None = None


class ExamTopicRequest(BaseModel):
    topic: str = Field(max_length=400)
    difficulty: str = Field(default="medium", max_length=20)
    num_mcq: int = Field(default=DEFAULT_MCQ, ge=0, le=MAX_MCQ)
    num_short: int = Field(default=DEFAULT_SHORT, ge=0, le=MAX_SHORT)
    model: str | None = None


class ExamGenerateResponse(BaseModel):
    questions: list[dict]
    title: str
    subject: str                               # the source label (subject / topic / filename)
    llm: dict | None = None


class ExamGradeRequest(BaseModel):
    subject: str = Field(max_length=255)
    questions: list[dict] = Field(max_length=MAX_GRADE_QUESTIONS)
    answers: dict[str, Any] = Field(default_factory=dict, max_length=100)
    duration_seconds: int | None = Field(default=None, ge=0)
    model: str | None = None
    # `llm` block of the call that WROTE this paper (echoed back by the client from /generate), so the
    # saved report can say who wrote the questions as well as who graded them.
    generated_llm: dict | None = None
    attempt_id: str | None = Field(default=None, max_length=64)    # client-minted: a retried submit is a no-op


class ExamGradeResponse(BaseModel):
    id: str | None = None                      # None when persistence is off
    title: str
    subject: str
    total_score: float
    max_score: float
    duration_seconds: int | None = None
    per_question: list[dict]
    topic_breakdown: list[dict]
    llm: dict | None = None                    # the grading call (None when no short answer needed the LLM)
    generated_llm: dict | None = None          # the call that wrote the questions (client-echoed, sanitized)
    result: dict | None = None                 # the unified result (same shape as POST /api/grade)


# ---------------------------------------------------------------------------
# Endpoints.
# ---------------------------------------------------------------------------
def _finalize_generate(data: dict, *, title: str, source: str) -> ExamGenerateResponse:
    """Validate the generated paper has usable questions and shape the response.

    ``source`` is echoed back as ``subject`` (the grader keys history/breakdown by
    it and the UI displays it) regardless of which source produced the exam.
    """
    questions = data.get("questions") or []
    if not questions:
        raise HTTPException(status_code=502, detail="The model returned no usable questions; try again.")
    grading.sign_items(questions)        # each question carries an HMAC of its answer key (verified at grade time)
    return ExamGenerateResponse(questions=questions, title=title, subject=source, llm=data.get("llm"))


def _run(fn, **kwargs) -> dict:
    try:
        return fn(**kwargs)
    except ExamError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


def _gather_content(subject: str, names: list[str], owner: str) -> str:
    """Concatenate the FULL text of the selected PDFs (llm_chain fits it per provider)."""
    parts: list[str] = []
    for name in names:
        # DocumentError (unknown subject/PDF) bubbles to the global 404 handler.
        text = read_pdf(subject, name, owner)
        if text and text.strip():
            parts.append(text)
    return "\n\n".join(parts)


@router.post("/generate", response_model=ExamGenerateResponse)
def generate_exam(request: ExamGenerateRequest, owner: str = Depends(current_owner)) -> ExamGenerateResponse:
    """Build the exam from the chosen subject PDFs' extracted text (one chain call)."""
    if request.mode != "subject":
        raise HTTPException(status_code=400, detail="Only subject mode is supported on this endpoint.")
    names = _gen.dedupe_names(request.pdfs, MAX_PDFS)
    if not names:
        raise HTTPException(status_code=400, detail="Select at least one PDF to build the exam from.")
    if request.num_mcq + request.num_short == 0:
        raise HTTPException(status_code=400, detail="An exam needs at least one question.")
    privacy.mark(privacy.for_subject(request.subject))
    content = _gather_content(request.subject, names, owner)
    if not content.strip():
        raise HTTPException(status_code=422, detail="No extractable text found in the selected PDFs.")
    data = _run(exam_service.generate_exam, content=content, num_mcq=request.num_mcq,
                num_short=request.num_short, difficulty=request.difficulty)
    return _finalize_generate(data, title=f"Exam · {request.subject}", source=request.subject)


@router.post("/topic", response_model=ExamGenerateResponse)
def generate_exam_topic(request: ExamTopicRequest, owner: str = Depends(current_owner)) -> ExamGenerateResponse:
    """Build an exam about a free topic (no document) in one chain call."""
    topic = require_topic(request.topic)
    if request.num_mcq + request.num_short == 0:
        raise HTTPException(status_code=400, detail="An exam needs at least one question.")
    data = _run(exam_service.generate_exam_from_topic, topic=topic, num_mcq=request.num_mcq,
                num_short=request.num_short, difficulty=request.difficulty)
    return _finalize_generate(data, title=f"Exam · {topic}", source=topic)


@router.post("/upload", response_model=ExamGenerateResponse)
async def generate_exam_upload(
    file: UploadFile = File(...),
    num_mcq: int = Form(DEFAULT_MCQ, ge=0, le=MAX_MCQ),
    num_short: int = Form(DEFAULT_SHORT, ge=0, le=MAX_SHORT),
    difficulty: str = Form("medium", max_length=20),
    model: str | None = Form(None),
    owner: str = Depends(current_owner),
) -> ExamGenerateResponse:
    """Build an exam from a single uploaded PDF's extracted text (one chain call)."""
    if num_mcq + num_short == 0:
        raise HTTPException(status_code=400, detail="An exam needs at least one question.")
    text, fname = await read_pdf_upload(file)
    privacy.mark(privacy.for_upload())
    data = await run_in_threadpool(lambda: _run(exam_service.generate_exam, content=text, num_mcq=num_mcq,
                                                num_short=num_short, difficulty=difficulty))
    return _finalize_generate(data, title=f"Exam · {fname}", source=fname)


@router.post("/grade", response_model=ExamGradeResponse)
def grade_exam(request: ExamGradeRequest, owner: str = Depends(current_owner)) -> ExamGradeResponse:
    """Compatibility wrapper over the unified grader (``POST /api/grade``): same marking, same persistence
    (the report is saved to history as kind "exams"), legacy response fields plus the unified ``result``."""
    result = run_grade(owner, GradeRequest(
        kind="exam", subject=request.subject, title=f"Exam \u00b7 {request.subject}", items=request.questions,
        answers=request.answers, attempt_id=request.attempt_id, duration_seconds=request.duration_seconds,
        generated_llm=request.generated_llm))
    legacy = grading.legacy_exam_fields(result)
    return ExamGradeResponse(
        id=result.get("id"), title=result["title"], subject=request.subject,
        total_score=legacy["total_score"], max_score=legacy["max_score"],
        duration_seconds=request.duration_seconds, per_question=legacy["per_question"],
        topic_breakdown=legacy["topic_breakdown"], llm=result.get("llm"),
        generated_llm=result.get("generated_llm"), result=result)
