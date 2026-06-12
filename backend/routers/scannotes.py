"""Scan Handwritten Notes router — photos in, clean Markdown out.

POST /api/scannotes/extract takes 1–6 photos (multipart field ``images``),
runs each through Gemini vision OCR (see services/scannotes.py), joins the
per-image transcriptions with "## Page N" headers, and saves the result to the
user's "scans" history (markdown + page count only — never the images).

Error mapping: missing GEMINI_API_KEY → 503; HEIC / unsupported type / too
large / too many → 422 with a friendly message; Gemini failures → 502.
"""
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel

from backend import db
from backend.auth import current_owner
from backend.services.scannotes import (
    MissingGeminiKeyError,
    ScanNotesError,
    require_gemini_key,
    transcribe_images,
)

router = APIRouter(prefix="/api/scannotes", tags=["scannotes"])

_MAX_IMAGES = 6
_MAX_BYTES = 8 * 1024 * 1024  # ~8 MB per photo
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
        require_gemini_key()
    except MissingGeminiKeyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    pages: list[tuple[bytes, str]] = []
    for upload in images:
        mime = _resolve_mime(upload)
        blob = await upload.read()
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
        pages.append((blob, mime))

    try:
        markdown = transcribe_images(pages)
    except MissingGeminiKeyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ScanNotesError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    title = f"Scan · {images[0].filename or 'photo'}"
    # Markdown + page count only — the images themselves are never persisted.
    saved_id = db.save("scans", owner, {
        "title": title,
        "markdown": markdown,
        "pages": len(pages),
    })
    return ScanResponse(markdown=markdown, pages=len(pages), id=saved_id, title=title)
