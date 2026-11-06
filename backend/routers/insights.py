"""Insights API — one personalized, cross-tool study analysis per user.

GET /api/insights           -> cached (per UTC day) AI analysis of all activity
GET /api/insights?refresh=1 -> bypass the cache and re-analyze now (rate limited per user)

All providers failing (LLMChainError) is mapped to a 503 carrying the "no model" llm label by the app-wide handler.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from backend.auth import current_owner
from backend.config import require_llm_key
from backend.services.insights import InsightsError, build_insights

router = APIRouter(prefix="/api/insights", tags=["insights"])


class InsightsResponse(BaseModel):
    empty: bool
    insights: dict | None = None
    generated_at: str | None = None
    llm: dict | None = None      # model that wrote the analysis (llm_chain ChatResult.to_public())


@router.get("", response_model=InsightsResponse)
def get_insights(
    refresh: bool = Query(False, description="Bypass the daily cache and re-analyze."),
    model: str | None = Query(None, max_length=120, description="Ignored (the provider chain picks the model)."),
    owner: str = Depends(current_owner),
) -> InsightsResponse:
    require_llm_key()    # → 503 + "no model" label via the global MissingApiKeyError handler
    try:
        return InsightsResponse(**build_insights(owner, model=model, refresh=refresh))
    except InsightsError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
