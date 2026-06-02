"""Subjects router: list subjects, list a subject's PDFs, and add a subject.

Owns every `/api/subjects` route. DocumentError bubbles up to the global handler
in main.py (mapped to 404); SubjectError (invalid upload / nothing to embed) is
mapped here to 502.
"""
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from backend.documents import list_pdf_names, list_subjects, page_count
from backend.schemas import AddSubjectResponse, PdfInfo, PdfsResponse, SubjectsResponse
from backend.services import subjects as subjects_service

router = APIRouter(prefix="/api/subjects", tags=["subjects"])


@router.get("", response_model=SubjectsResponse)
def get_subjects() -> SubjectsResponse:
    """List subject folders under documents/."""
    return SubjectsResponse(subjects=list_subjects())


@router.get("/{subject}/pdfs", response_model=PdfsResponse)
def get_pdfs(subject: str) -> PdfsResponse:
    """List the PDFs (with page counts) filed under a subject."""
    infos: list[PdfInfo] = []
    for name in list_pdf_names(subject):
        try:
            pages = page_count(subject, name)
        except Exception:
            pages = 0
        infos.append(PdfInfo(name=name, pages=pages))
    return PdfsResponse(pdfs=infos)


@router.post("", response_model=AddSubjectResponse)
async def add_subject(
    name: str = Form(...),
    files: list[UploadFile] = File(...),
) -> AddSubjectResponse:
    """Create (or extend) a subject from uploaded PDFs and build its index.

    Saves the PDFs under documents/<name>/ and embeds that one subject so it is
    immediately usable by every tool. Reuses an existing subject name to append
    PDFs and rebuild its index.
    """
    uploads: list[tuple[str, bytes]] = []
    for upload in files:
        data = await upload.read()
        uploads.append((upload.filename or "document.pdf", data))

    try:
        result = subjects_service.add_subject(name, uploads)
    except subjects_service.SubjectError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return AddSubjectResponse(
        subject=result.subject,
        saved_pdfs=result.saved_pdfs,
        chunk_count=result.chunk_count,
    )
