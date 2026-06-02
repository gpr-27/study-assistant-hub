"""Mind Map router: build a concept map from a subject PDF, a topic, or an upload.
Each generated map is saved to the signed-in user's history (when persistence is on).

DocumentError (unknown subject/PDF) bubbles to the global 404 handler; Groq
errors bubble to the global 429/502 handlers; MindmapError maps to 502 here.
"""
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from backend import config, db
from backend.auth import current_owner
from backend.documents import extract_pdf_text, read_pdf
from backend.schemas import MindmapNode, MindmapRequest, MindmapResponse
from backend.services import mindmap as mindmap_service

router = APIRouter(prefix="/api/mindmap", tags=["mindmap"])


def _build(owner: str, *, title: str, source: str, topic=None, content=None, model=None) -> MindmapResponse:
    try:
        root = mindmap_service.generate_mindmap(topic=topic, content=content, model=model)
    except mindmap_service.MindmapError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    node = MindmapNode(**root)
    label = node.label or title
    saved_id = db.save("mindmaps", owner, {
        "title": label,
        "source": source,
        "model": config.model_for("mindmap", model),
        "root": node.model_dump(),
    })
    return MindmapResponse(root=node, id=saved_id, title=label)


@router.post("/generate", response_model=MindmapResponse)
def generate(request: MindmapRequest, owner: str = Depends(current_owner)) -> MindmapResponse:
    """Build a mind map for a single subject PDF or a typed topic."""
    if request.mode == "subject":
        if not request.subject or not request.pdf:
            raise HTTPException(status_code=400, detail="A subject and one PDF are required.")
        content = read_pdf(request.subject, request.pdf)
        if not content:
            raise HTTPException(status_code=422, detail="That PDF has no extractable text.")
        return _build(owner, title=f"Mind map · {request.subject} / {request.pdf}",
                      source=f"{request.subject} / {request.pdf}", content=content, model=request.model)
    if request.mode == "topic":
        if not request.topic:
            raise HTTPException(status_code=400, detail="A topic is required for topic mode.")
        return _build(owner, title=f"Mind map · {request.topic}", source=request.topic,
                      topic=request.topic, model=request.model)
    raise HTTPException(status_code=400, detail=f"Unsupported mind map mode: {request.mode}")


@router.post("/upload", response_model=MindmapResponse)
async def upload(
    file: UploadFile = File(...),
    model: str | None = Form(None),
    owner: str = Depends(current_owner),
) -> MindmapResponse:
    """Build a mind map from a single uploaded PDF."""
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="The uploaded PDF was empty.")
    content = extract_pdf_text(data)
    if not content:
        raise HTTPException(status_code=422, detail="No extractable text found in that PDF.")
    fname = file.filename or "uploaded.pdf"
    return _build(owner, title=f"Mind map · {fname}", source=fname, content=content, model=model)
