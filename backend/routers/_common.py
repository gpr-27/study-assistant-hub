"""Helpers shared by the study-tool routers (quiz, flashcards, exam, cheatsheet, mindmap, planner)."""
from __future__ import annotations

from fastapi import HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from backend.documents import extract_pdf_text
from backend.services._gen import MAX_NAME_CHARS, MAX_UPLOAD_BYTES


async def read_pdf_upload(file: UploadFile) -> tuple[str, str]:
    """Read one uploaded PDF (size-limited, type-checked) and return ``(text, filename)``.

    The body is read in a bounded way (never more than MAX_UPLOAD_BYTES + 1 bytes), and the
    blocking PDF text extraction runs in the threadpool so the event loop stays free.
    """
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if not data:
        raise HTTPException(status_code=400, detail="The uploaded PDF was empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413,
                            detail=f"That PDF is too large (limit {MAX_UPLOAD_BYTES // (1024 * 1024)} MB).")
    if b"%PDF" not in data[:1024]:
        raise HTTPException(status_code=415, detail="Please upload a PDF file.")
    try:
        text = await run_in_threadpool(extract_pdf_text, data)
    except Exception as exc:  # noqa: BLE001 — corrupt/encrypted PDF -> a clean 422, never a 500
        raise HTTPException(status_code=422, detail="That PDF could not be read.") from exc
    if not text or not text.strip():
        raise HTTPException(status_code=422, detail="No extractable text found in that PDF.")
    name = (file.filename or "uploaded.pdf").strip()[:MAX_NAME_CHARS] or "uploaded.pdf"
    return text, name


def require_topic(topic: str | None) -> str:
    """Whitespace-collapsed, length-capped topic or a 400."""
    from backend.services._gen import clean_topic
    label = clean_topic(topic)
    if not label:
        raise HTTPException(status_code=400, detail="A topic is required.")
    return label


def require_text(text: str | None, what: str = "the selected PDF") -> str:
    """Non-blank document text or a 422."""
    if not text or not text.strip():
        raise HTTPException(status_code=422, detail=f"No extractable text found in {what}.")
    return text


def model_of(llm: dict | None) -> str | None:
    """The model id that actually served a request (for the history record)."""
    return (llm or {}).get("model")
