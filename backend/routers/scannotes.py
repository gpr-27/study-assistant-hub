"""Scan Handwritten Notes router — photos in, clean Markdown out.

POST /api/scannotes/extract takes 1–6 photos (multipart field ``images``),
runs each through the provider chain's vision models (see services/scannotes.py), joins the
per-image transcriptions with "## Page N" headers, and saves the result to the
user's "scans" history (markdown + page count only — never the images).

Error mapping: no provider key / no vision support, or every provider failing → 503 (with the
"no model" llm label); HEIC / unsupported type / too large / too many → 422 with a friendly
message; unusable model output → 502.
"""
import asyncio

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel

from backend import config, db
from backend.auth import current_owner
from backend.services.scannotes import (
    ScanNotesError,
    ScanUnavailableError,
    require_vision,
    transcribe_pages,
)

router = APIRouter(prefix="/api/scannotes", tags=["scannotes"])

_MAX_IMAGES = 6
_MAX_BYTES = 8 * 1024 * 1024  # ~8 MB per photo
_MAX_TOTAL_BYTES = config.MAX_BODY_BYTES   # ~4.4 MB: Vercel rejects bigger bodies with a non-JSON error
_ALLOWED_MIME = {"image/jpeg", "image/png", "image/webp"}
_HEIC_MIME = {"image/heic", "image/heif", "image/heic-sequence", "image/heif-sequence"}
_HEIC_EXT = (".heic", ".heif")
_EXT_MIME = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}

_HEIC_MESSAGE = (
    "HEIC photos aren't supported yet — convert to JPEG or PNG first "
    "(iPhone: Settings → Camera → Formats → Most Compatible) and try again."
)


# Local request/response models (schemas.py is frozen for this feature).
class ScanResponse(BaseModel):
    """One completed scan: the combined Markdown plus its saved-history handle."""

    markdown: str
    pages: int
    id: str | None = None  # None when persistence is off — tolerated everywhere
    title: str
    llm: dict | None = None  # model that transcribed (llm_chain ChatResult.to_public())
    failed_pages: list[int] = []  # 1-based pages that could not be transcribed (marked in the text)
    partial: bool = False


def _resolve_mime(upload: UploadFile) -> str:
    """Return the validated image MIME type for an upload, or raise 422.

    HEIC gets its own friendly rejection; anything not JPEG/PNG/WebP (by
    declared content type, falling back to the filename extension) is refused.
    """
    name = (upload.filename or "").lower()
    ctype = (upload.content_type or "").lower().split(";")[0].strip()
    if ctype in _HEIC_MIME or name.endswith(_HEIC_EXT):
        raise HTTPException(status_code=422, detail=_HEIC_MESSAGE)
    if ctype in _ALLOWED_MIME:
        return ctype
    for ext, mime in _EXT_MIME.items():
        if name.endswith(ext):
            return mime
    raise HTTPException(
        status_code=422,
        detail=f"Unsupported image type for {upload.filename or 'photo'!r} — use JPEG, PNG, or WebP.",
    )


@router.post("/extract", response_model=ScanResponse)
async def extract_notes(
    images: list[UploadFile] = File(...),
    owner: str = Depends(current_owner),
) -> ScanResponse:
    """Transcribe 1–6 photos of handwritten notes into one Markdown document."""
    if not images:
        raise HTTPException(status_code=422, detail="Attach at least one photo of your notes.")
    if len(images) > _MAX_IMAGES:
        raise HTTPException(
            status_code=422, detail=f"Too many images — up to {_MAX_IMAGES} photos per scan."
        )
    # Fail fast with a clear 503 before reading any bytes (mirrors require_groq_key).
    try:
        require_vision()
    except ScanUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc), headers=None) from exc

    pages: list[tuple[bytes, str]] = []
    total = 0
    for upload in images:
        mime = _resolve_mime(upload)
        blob = await upload.read(_MAX_BYTES + 1)
        if not blob:
            raise HTTPException(
                status_code=422, detail=f"{upload.filename or 'photo'!r} is empty."
            )
        if len(blob) > _MAX_BYTES:
            raise HTTPException(
                status_code=422,
                detail=f"{upload.filename or 'photo'!r} is larger than 8 MB — "
                       "resize or recompress it and try again.",
            )
        total += len(blob)
        if total > _MAX_TOTAL_BYTES:
            raise HTTPException(status_code=413, detail="The photos together are too large (about 4 MB max per request) — scan fewer pages at once.")
        pages.append((blob, mime))

    try:
        outcome = await transcribe_pages(pages)
    except ScanUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ScanNotesError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    markdown, llm_public = outcome.markdown, outcome.llm

    title = f"Scan · {(images[0].filename or 'photo')[:120]}"
    # Markdown + page count only — the images themselves are never persisted.
    saved_id = await asyncio.to_thread(db.save, "scans", owner, {
        "title": title,
        "markdown": markdown,
        "pages": len(pages),
        "llm": llm_public,
    })
    return ScanResponse(markdown=markdown, pages=len(pages), id=saved_id, title=title, llm=llm_public,
                        failed_pages=outcome.failed_pages, partial=outcome.partial)
