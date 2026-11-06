"""Mind Map router: build a concept map from a subject PDF, a topic, or an upload.
Each generated map is saved to the signed-in user's history (when persistence is on), including
the `llm` that produced it.

DocumentError (unknown subject/PDF) bubbles to the global 404 handler; LLMChainError to the
global 503 handler; MindmapError maps to 502 here.
"""
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from backend import db
from backend import privacy
from backend.auth import current_owner
from backend.documents import read_pdf
from backend.routers._common import model_of, read_pdf_upload, require_text, require_topic
from backend.schemas import MindmapNode, MindmapRequest, MindmapResponse
from backend.services import mindmap as mindmap_service

router = APIRouter(prefix="/api/mindmap", tags=["mindmap"])


def _build(owner: str, *, title: str, source: str, topic=None, content=None) -> MindmapResponse:
    try:
        root, llm = mindmap_service.generate_mindmap_ex(topic=topic, content=content)
    except mindmap_service.MindmapError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    node = MindmapNode(**root)
    label = node.label or title
    saved_id = db.save("mindmaps", owner, {
        "title": label,
        "source": source,
        "model": model_of(llm),
        "llm": llm,
        "root": node.model_dump(),
    })
    return MindmapResponse(root=node, id=saved_id, title=label, llm=llm)


@router.post("/generate", response_model=MindmapResponse)
def generate(request: MindmapRequest, owner: str = Depends(current_owner)) -> MindmapResponse:
    """Build a mind map for a single subject PDF or a typed topic."""
    if request.mode == "subject":
        if not request.subject or not request.pdf:
            raise HTTPException(status_code=400, detail="A subject and one PDF are required.")
        privacy.mark(privacy.for_subject(request.subject))
        content = require_text(read_pdf(request.subject, request.pdf, owner))
        return _build(owner, title=f"Mind map · {request.subject} / {request.pdf}",
                      source=f"{request.subject} / {request.pdf}", content=content)
    if request.mode == "topic":
        if not (request.topic or "").strip():
            raise HTTPException(status_code=400, detail="A topic is required for topic mode.")
        topic = require_topic(request.topic)
        return _build(owner, title=f"Mind map · {topic}", source=topic, topic=topic)
    raise HTTPException(status_code=400, detail=f"Unsupported mind map mode: {request.mode}")


@router.post("/upload", response_model=MindmapResponse)
async def upload(
    file: UploadFile = File(...),
    model: str | None = Form(None),
    owner: str = Depends(current_owner),
) -> MindmapResponse:
    """Build a mind map from a single uploaded PDF."""
    text, fname = await read_pdf_upload(file)
    privacy.mark(privacy.for_upload())
    return await run_in_threadpool(lambda: _build(owner, title=f"Mind map · {fname}", source=fname, content=text))
