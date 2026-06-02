"""Study Planner router: PDF summaries and a markdown study plan. Generated plans
are saved to the signed-in user's history (when persistence is enabled).

Three sources mirror the Chat tool: a subject's PDFs, a free topic, or uploaded
PDFs. Endpoints require a Groq key (MissingApiKeyError -> 503 globally).
PlannerError maps to HTTP 502; DocumentError bubbles to the global 404 handler.
"""
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from backend import config, db
from backend.auth import current_owner
from backend.config import require_groq_key
from backend.documents import extract_pdf_text
from backend.schemas import (
    PlanRequest,
    PlanResponse,
    SummariesRequest,
    SummariesResponse,
    TopicPlanRequest,
)
from backend.services.planner import (
    PlannerError,
    generate_plan,
    generate_plan_from_texts,
    generate_plan_from_topic,
    summarize_pdfs,
)

router = APIRouter(prefix="/api/planner", tags=["planner"])


def _save_plan(owner: str, plan: str, *, title: str, source: str, target_date: str, model) -> PlanResponse:
    """Persist a generated plan for this user and return it with its id."""
    saved_id = db.save("plans", owner, {
        "title": title,
        "source": source,
        "target_date": target_date,
        "model": config.model_for("planner", model),
        "plan": plan,
    })
    return PlanResponse(plan=plan, id=saved_id, title=title)


@router.post("/summaries", response_model=SummariesResponse)
def create_summaries(request: SummariesRequest) -> SummariesResponse:
    """Generate a per-PDF summary for each selected document (subject mode)."""
    require_groq_key()
    try:
        summaries = summarize_pdfs(request.subject, request.pdfs, model=request.model)
    except PlannerError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return SummariesResponse(summaries=summaries)


@router.post("/generate", response_model=PlanResponse)
def create_plan(request: PlanRequest, owner: str = Depends(current_owner)) -> PlanResponse:
    """Generate a day-by-day markdown study plan from subject PDFs."""
    require_groq_key()
    try:
        plan = generate_plan(request.subject, request.pdfs, request.target_date, model=request.model)
    except PlannerError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return _save_plan(owner, plan, title=f"Plan · {request.subject}", source=request.subject,
                      target_date=request.target_date, model=request.model)


@router.post("/topic", response_model=PlanResponse)
def create_plan_topic(request: TopicPlanRequest, owner: str = Depends(current_owner)) -> PlanResponse:
    """Generate a plan to learn a free topic by the target date (no documents)."""
    require_groq_key()
    try:
        plan = generate_plan_from_topic(request.topic, request.target_date, model=request.model)
    except PlannerError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return _save_plan(owner, plan, title=f"Plan · {request.topic}", source=request.topic,
                      target_date=request.target_date, model=request.model)


@router.post("/upload", response_model=PlanResponse)
async def create_plan_upload(
    files: list[UploadFile] = File(...),
    target_date: str = Form(...),
    model: str | None = Form(None),
    owner: str = Depends(current_owner),
) -> PlanResponse:
    """Generate a plan from one or more uploaded PDFs and a target date."""
    require_groq_key()
    items: list[tuple[str, str]] = []
    for upload in files:
        data = await upload.read()
        if not data:
            continue
        text = extract_pdf_text(data)
        if text:
            items.append((upload.filename or "document.pdf", text))
    if not items:
        raise HTTPException(status_code=422, detail="No extractable text found in the uploaded PDF(s).")
    label = items[0][0] if len(items) == 1 else f"{len(items)} uploaded documents"
    try:
        plan = generate_plan_from_texts(label, items, target_date, model=model)
    except PlannerError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return _save_plan(owner, plan, title=f"Plan · {label}", source=label, target_date=target_date, model=model)
