"""Insights API — one personalized, cross-tool study analysis per user.

GET /api/insights           -> cached (per UTC day) AI analysis of all activity
GET /api/insights?refresh=1 -> bypass the cache and re-analyze now
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from backend.auth import current_owner
from backend.config import require_groq_key
from backend.services.insights import InsightsError, build_insights

router = APIRouter(prefix="/api/insights", tags=["insights"])


class InsightsResponse(BaseModel):
    empty: bool
    insights: dict | None = None
    generated_at: str | None = None


@router.get("", response_model=InsightsResponse)
def get_insights(
    refresh: bool = Query(False, description="Bypass the daily cache and re-analyze."),
    model: str | None = Query(None),
    owner: str = Depends(current_owner),
) -> InsightsResponse:
    require_groq_key()
    try:
        return InsightsResponse(**build_insights(owner, model=model, refresh=refresh))
    except InsightsError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
