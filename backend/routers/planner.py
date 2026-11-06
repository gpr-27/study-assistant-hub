"""Study Planner router: PDF summaries and a markdown study plan. Generated plans and summaries
are saved to the signed-in user's history (when persistence is enabled), including the `llm`
(provider/model label) that produced them.

Three sources mirror the Chat tool: a subject's PDFs, a free topic, or uploaded PDFs.
PlannerError maps to HTTP 502 (bad/unparseable target date -> 422); DocumentError bubbles to the
global 404 handler; LLMChainError (all providers failed) to the global 503 handler.
"""
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from backend import db
from backend import privacy
from backend.auth import current_owner
from backend.routers._common import model_of, read_pdf_upload, require_topic
from backend.schemas import (
    PlanRequest,
    PlanResponse,
    SummariesRequest,
    SummariesResponse,
    TopicPlanRequest,
)
from backend.services.planner import (
    MAX_PDFS,
    PlannerError,
    _days_until,
    generate_plan_ex,
    generate_plan_from_texts_ex,
    generate_plan_from_topic_ex,
    summarize_pdfs_ex,
)

router = APIRouter(prefix="/api/planner", tags=["planner"])

MAX_UPLOAD_FILES = MAX_PDFS


def _check_date(target_date: str) -> None:
    """Reject an unparseable / absurd target date up front (422) — before any LLM spend."""
    try:
        _days_until(target_date)
    except PlannerError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _save_plan(owner: str, plan: str, llm: dict | None, *, title: str, source: str, target_date: str) -> PlanResponse:
    """Persist a generated plan for this user and return it with its id."""
    saved_id = db.save("plans", owner, {
        "title": title,
        "source": source,
        "target_date": target_date,
        "model": model_of(llm),
        "llm": llm,
        "plan": plan,
    })
    return PlanResponse(plan=plan, id=saved_id, title=title, llm=llm)


def _guard(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except PlannerError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/summaries", response_model=SummariesResponse)
def create_summaries(request: SummariesRequest, owner: str = Depends(current_owner)) -> SummariesResponse:
    """Generate a per-PDF summary for each selected document, and save it to the
    user's history (separate from saved plans) when persistence is enabled."""
    privacy.mark(privacy.for_subject(request.subject))
    summaries, llm = _guard(summarize_pdfs_ex, request.subject, request.pdfs, owner=owner)
    title = f"Summary · {request.subject}"
    saved_id = db.save("summaries", owner, {
        "title": title,
        "source": request.subject,
        "subject": request.subject,
        "pdfs": list(summaries),
        "model": model_of(llm),
        "llm": llm,
        "summaries": summaries,
    })
    return SummariesResponse(summaries=summaries, id=saved_id, title=title, llm=llm)


@router.post("/generate", response_model=PlanResponse)
def create_plan(request: PlanRequest, owner: str = Depends(current_owner)) -> PlanResponse:
    """Generate a study plan (day-by-day, or week-by-week for long horizons) from subject PDFs."""
    _check_date(request.target_date)
    privacy.mark(privacy.for_subject(request.subject))
    plan, llm = _guard(generate_plan_ex, request.subject, request.pdfs, request.target_date, owner=owner)
    return _save_plan(owner, plan, llm, title=f"Plan · {request.subject}", source=request.subject,
                      target_date=request.target_date)


@router.post("/topic", response_model=PlanResponse)
def create_plan_topic(request: TopicPlanRequest, owner: str = Depends(current_owner)) -> PlanResponse:
    """Generate a plan to learn a free topic by the target date (no documents)."""
    topic = require_topic(request.topic)
    _check_date(request.target_date)
    plan, llm = _guard(generate_plan_from_topic_ex, topic, request.target_date)
    return _save_plan(owner, plan, llm, title=f"Plan · {topic}", source=topic, target_date=request.target_date)


@router.post("/upload", response_model=PlanResponse)
async def create_plan_upload(
    files: list[UploadFile] = File(...),
    target_date: str = Form(..., max_length=40),
    model: str | None = Form(None),
    owner: str = Depends(current_owner),
) -> PlanResponse:
    """Generate a plan from one or more uploaded PDFs and a target date."""
    if len(files) > MAX_UPLOAD_FILES:
        raise HTTPException(status_code=413, detail=f"Upload at most {MAX_UPLOAD_FILES} PDFs at a time.")
    _check_date(target_date)
    items: list[tuple[str, str]] = []
    for upload in files:
        try:
            items.append(await read_pdf_upload(upload))      # (text, filename)
        except HTTPException as exc:
            if exc.status_code in (400, 415, 422) and len(files) > 1:   # one bad file among several: skip it
                continue
            raise
    if not items:
        raise HTTPException(status_code=422, detail="No extractable text found in the uploaded PDF(s).")
    label = items[0][1] if len(items) == 1 else f"{len(items)} uploaded documents"
    named = [(name, text) for text, name in items]
    privacy.mark(privacy.for_upload())

    def work() -> PlanResponse:
        plan, llm = _guard(generate_plan_from_texts_ex, label, named, target_date)
        return _save_plan(owner, plan, llm, title=f"Plan · {label}", source=label, target_date=target_date)

    return await run_in_threadpool(work)

