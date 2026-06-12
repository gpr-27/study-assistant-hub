"""Progress API router: record quiz/flashcard attempt events and serve the
aggregated weakness-heatmap summary. NO LLM is involved — recording is one
db.save (kind "attempts") and the summary is plain-Python aggregation, so both
endpoints work (or gracefully no-op) exactly like the rest of the history
system when persistence is off."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from backend.auth import current_owner
from backend.services.progress import ProgressError, build_summary, record_attempt

router = APIRouter(prefix="/api/progress", tags=["progress"])


# ---- request/response models (local to this router by design) ----
class RecordRequest(BaseModel):
    tool: str = Field(..., description='Which tool produced the attempt: "quiz" or "flashcards".')
    subject: str | None = None
    topic: str | None = None
    correct: int = Field(..., ge=0)
    total: int = Field(..., ge=1)
    detail: str | None = None


class RecordResponse(BaseModel):
    saved: bool


class TopicStats(BaseModel):
    topic: str
    attempts: int
    accuracy: float
    last_seen: str | None = None


class SubjectStats(BaseModel):
    subject: str
    attempts: int
    accuracy: float
    topics: list[TopicStats]


class WeakTopic(BaseModel):
    subject: str
    topic: str
    attempts: int
    accuracy: float


class OverallStats(BaseModel):
    total_attempts: int
    overall_accuracy: float
    streak_days: int


class SummaryResponse(BaseModel):
    overall: OverallStats
    subjects: list[SubjectStats]
    weakest: list[WeakTopic]


@router.post("/record", response_model=RecordResponse)
def record(request: RecordRequest, owner: str = Depends(current_owner)) -> RecordResponse:
    """Record one finished quiz/flashcard session as an attempt event.

    Returns {"saved": false} (not an error) when persistence is disabled — the
    SPA calls this fire-and-forget and must never be disturbed by it.
    """
    try:
        saved = record_attempt(
            owner,
            tool=request.tool,
            correct=request.correct,
            total=request.total,
            subject=request.subject,
            topic=request.topic,
            detail=request.detail,
        )
    except ProgressError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return RecordResponse(saved=saved)


@router.get("/summary", response_model=SummaryResponse)
def summary(owner: str = Depends(current_owner)) -> SummaryResponse:
    """The owner's aggregated progress: overall stats, per-subject/topic
    accuracy cells (heatmap), and the weakest topics."""
    try:
        return SummaryResponse(**build_summary(owner))
    except ProgressError as exc:  # defensive — build_summary shouldn't raise today
        raise HTTPException(status_code=502, detail=str(exc)) from exc
