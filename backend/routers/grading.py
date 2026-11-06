"""Unified grading API - the ONE place quiz, exam, chat widgets and practice are marked.

POST /api/grade            items + answers -> unified result (idempotent per ``attempt_id``)
GET  /api/attempts         the owner's graded attempts (projected summaries)
GET  /api/attempts/{id}    one attempt's full unified result
POST /api/grade/explain    "explain this answer" (one labelled LLM call)
POST /api/grade/similar    "give me a similar question" (one labelled LLM call; the item comes back signed)

MCQ / true-false / numeric are marked in code; only non-blank short / long answers reach the model, in batches
(see :mod:`backend.services.grading`). If the model chain is down the call still succeeds with ``graded: false`` and
provisional lexical scores. Everything is owner-scoped via ``current_owner``.
"""
from __future__ import annotations

import uuid
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Path, Query
from pydantic import BaseModel, Field

from backend import db, privacy
from backend.auth import current_owner
from backend.ratelimit import rate_limit
from backend.services import attempts, grading, review

router = APIRouter(tags=["grading"])

MAX_REQUEST_ITEMS = 60


class GradeRequest(BaseModel):
    kind: Literal["quiz", "exam", "practice"] = "practice"
    subject: str | None = Field(default=None, max_length=255)
    topic: str | None = Field(default=None, max_length=200)
    title: str | None = Field(default=None, max_length=200)
    items: list[dict] = Field(max_length=MAX_REQUEST_ITEMS)
    answers: dict[str, Any] = Field(default_factory=dict, max_length=200)
    attempt_id: str | None = Field(default=None, max_length=64)
    quiz_id: str | None = Field(default=None, max_length=64)          # saved-quiz fallback for verification
    duration_seconds: int | None = Field(default=None, ge=0, le=7 * 86400)
    generated_llm: dict | None = None
    regrade: bool = False


class ExplainRequest(BaseModel):
    question: str = Field(min_length=1, max_length=grading.MAX_QUESTION_CHARS)
    topic: str = Field(default="", max_length=80)
    options: list[str] = Field(default_factory=list, max_length=8)
    correct_answer: str = Field(default="", max_length=800)
    user_answer: str = Field(default="", max_length=grading.MAX_ANSWER_CHARS)
    explanation: str = Field(default="", max_length=800)


class SimilarRequest(BaseModel):
    item: dict


def verify_items(items: list[dict], owner: str, quiz_id: str | None) -> bool:
    """True when every item's key is authentic (valid HMAC signature, or matches the owner's saved quiz).

    A saved-quiz match REPLACES the client's key with the stored one (in place) so a tampered key can never score.
    """
    if items and all(grading.verify_sig(it) for it in items):
        return True
    if not quiz_id:
        return False
    doc = db.get_item("quizzes", owner, quiz_id)
    stored = {}
    for raw in (doc or {}).get("questions") or []:
        n = grading.normalise_item(raw)
        if n is not None:
            stored[grading.norm_text(n["question"])] = n
    if not stored:
        return False
    ok = True
    for it in items:
        if grading.verify_sig(it):
            continue
        m = stored.get(grading.norm_text(it["question"]))
        if m is None:
            ok = False
            continue
        keep = it["id"]
        it.clear()
        it.update({**m, "id": keep})
    return ok


def run_grade(owner: str, req: GradeRequest) -> dict:
    """Shared by ``POST /api/grade`` and the legacy ``/api/exam/grade`` wrapper."""
    if not req.subject or privacy.for_subject(req.subject):
        privacy.mark(True)       # student text + questions may derive from uploads: keep it off Gemini's free tier
    items = grading.normalise_items(req.items)
    if not items:
        raise HTTPException(status_code=400, detail="No valid questions were supplied to grade.")
    attempt_id = req.attempt_id if attempts.valid_attempt_id(req.attempt_id) else uuid.uuid4().hex
    if req.attempt_id and attempt_id != req.attempt_id:
        raise HTTPException(status_code=422, detail="attempt_id must be 8-64 letters, digits, '-' or '_'.")

    def compute() -> dict:
        verified = verify_items(items, owner, req.quiz_id)
        return grading.grade_submission(
            items, req.answers, kind=req.kind, attempt_id=attempt_id, subject=req.subject, topic=req.topic,
            title=req.title, verified=verified, duration_seconds=req.duration_seconds,
            generated_llm=grading.clean_llm(req.generated_llm))

    result, _ = attempts.run_idempotent(owner, attempt_id, req.kind, compute, regrade=req.regrade)
    return result


@router.post("/api/grade", dependencies=[Depends(rate_limit)])
def grade(req: GradeRequest, owner: str = Depends(current_owner)) -> dict:
    """Mark a submission and record it. Safe to retry: the same ``attempt_id`` returns the stored result."""
    return run_grade(owner, req)


@router.get("/api/attempts")
def list_attempts(
    subject: str | None = Query(None, max_length=255),
    kind: Literal["quiz", "exam", "practice", "teachback", "flashcards"] | None = None,
    limit: int = Query(30, ge=1, le=100),
    owner: str = Depends(current_owner),
) -> dict:
    return {"items": attempts.list_attempts(owner, subject=subject, kind=kind, limit=limit),
            "persistence": db.is_enabled()}


@router.get("/api/attempts/{attempt_id}")
def get_attempt(attempt_id: str = Path(..., max_length=64), owner: str = Depends(current_owner)) -> dict:
    res = attempts.get_attempt(owner, attempt_id)
    if res is None:
        raise HTTPException(status_code=404, detail="Attempt not found.")
    return res


@router.post("/api/grade/explain", dependencies=[Depends(rate_limit)])
def explain(req: ExplainRequest, owner: str = Depends(current_owner)) -> dict:
    privacy.mark(True)
    try:
        text, llm = review.explain_answer(question=req.question, topic=req.topic, options=req.options,
                                          correct_answer=req.correct_answer, user_answer=req.user_answer,
                                          explanation=req.explanation)
    except review.ReviewError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {"explanation": text, "llm": llm}


@router.post("/api/grade/similar", dependencies=[Depends(rate_limit)])
def similar(req: SimilarRequest, owner: str = Depends(current_owner)) -> dict:
    privacy.mark(True)
    try:
        item, llm = review.similar_question(req.item)
    except review.ReviewError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {"item": item, "llm": llm}
