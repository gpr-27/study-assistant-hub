"""Subjects router: list subjects, list a subject's PDFs, add / extend / delete a subject, and
finish embedding a partially-indexed subject.

  GET    /api/subjects                          bundled + the caller's own subjects (30 s cached)
  GET    /api/subjects/{name}/pdfs              PDFs (+ page counts) of a subject
  POST   /api/subjects                          upload PDFs; returns fast with indexing=complete|partial
  POST   /api/subjects/uploads                  start a chunked (large-PDF) upload -> upload_id
  PUT    /api/subjects/uploads/{id}/chunks/{i}  one raw chunk (<= 3.5 MB; idempotent)
  GET    /api/subjects/uploads/{id}             which chunks are stored (resume)
  POST   /api/subjects/uploads/{id}/complete    assemble + index; same response as POST /api/subjects
  DELETE /api/subjects/uploads/{id}             abort
  POST   /api/subjects/{name}/continue-index    read more pages / embed more of a partial subject (<= ~35 s per call)
  GET    /api/subjects/limits                   upload limits (the SPA text is generated from these) + usage
  GET    /api/subjects/{name}/files             files of an own subject: size / pages / status + index state
  DELETE /api/subjects/{name}/pdfs/{filename}   remove one file, re-index the rest
  POST   /api/subjects/{name}/reindex           rebuild the index from the stored files
  PATCH  /api/subjects/{name}                   rename an own subject {"name": "..."}
  POST   /api/subjects/{name}/hide              hide a STANDARD subject for this account only
  POST   /api/subjects/{name}/restore           show a hidden standard subject again
  POST   /api/subjects/restore-standard         show all hidden standard subjects again
  DELETE /api/subjects/{name}                   delete one of the caller's own subjects

Serverless notes: PDF parsing / chunking / embedding is blocking, so it runs in the thread pool,
never on the event loop; every call is time-bounded (see ``backend.services.subjects``).
"""
import time

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from backend.auth import current_owner
from backend.documents import list_subjects
from backend import prefs
from backend.schemas import AddSubjectResponse, PdfInfo, PdfsResponse, SubjectsResponse
from backend.services import subjects as subjects_service
from backend.services import uploads as uploads_service
from backend.services.subjects import SubjectError

router = APIRouter(prefix="/api/subjects", tags=["subjects"])

_CHUNK = 1024 * 1024


def _http(exc: SubjectError) -> HTTPException:
    return HTTPException(status_code=exc.status_code, detail=str(exc))


def _param(name: str) -> str:
    """Validate a ``{name}`` path parameter. A name that is not a plain subject name (e.g. an
    internal ``user__<other>__<x>`` key) is reported as 404, exactly like an unknown subject."""
    try:
        return subjects_service.clean_subject_name(name)
    except SubjectError:
        raise HTTPException(status_code=404, detail="Subject not found.") from None


@router.get("", response_model=SubjectsResponse)
def get_subjects(owner: str = Depends(current_owner)) -> SubjectsResponse:
    """List subjects visible to the signed-in user (bundled + their own uploads)."""
    visible = list_subjects(owner)
    bundled = list_subjects(None)
    shipped = {b.casefold() for b in bundled}
    return SubjectsResponse(
        subjects=visible, bundled=bundled, own=[s for s in visible if s.casefold() not in shipped],
        hidden=prefs.hidden_standard(owner), limits=subjects_service.public_limits())


@router.get("/limits")
def get_limits(owner: str = Depends(current_owner)) -> dict:
    """Upload limits (single source of truth for the SPA copy) + this owner's storage usage."""
    from backend.services import storage

    return {**subjects_service.public_limits(), "usage": storage.public_usage(owner)}


@router.get("/{subject}/pdfs", response_model=PdfsResponse)
def get_pdfs(subject: str, owner: str = Depends(current_owner)) -> PdfsResponse:
    """List the PDFs (with page counts) filed under a subject."""
    clean = _param(subject)
    return PdfsResponse(pdfs=[PdfInfo(name=n, pages=p) for n, p in subjects_service.pdf_infos(clean, owner)])


async def _read_limited(upload: UploadFile, limit: int, budget_left: int) -> bytes:
    """Read an upload in chunks, aborting (413) as soon as it exceeds the per-file or remaining
    total limit instead of buffering an unbounded body."""
    cap = min(limit, budget_left) if budget_left else limit        # budget_left == 0: no total cap
    buf = bytearray()
    while True:
        chunk = await upload.read(_CHUNK)
        if not chunk:
            break
        buf += chunk
        if len(buf) > cap:
            raise HTTPException(
                status_code=413,
                detail=f"Upload is too large (max {limit // 1048576} MB per file"
                       + (f", {subjects_service.MAX_TOTAL_BYTES // 1048576} MB total)." if subjects_service.MAX_TOTAL_BYTES else ")."))
    return bytes(buf)


@router.post("", response_model=AddSubjectResponse)
async def add_subject(
    name: str = Form(..., max_length=200),
    files: list[UploadFile] = File(...),
    owner: str = Depends(current_owner),
) -> AddSubjectResponse:
    """Create (or extend) a subject from uploaded PDFs and index it within a time budget.

    The subject is stored under the signed-in user's namespace so it is only visible to them.
    Bundled subjects (Mathematics, Operating System, …) stay visible to all users and cannot be
    overwritten. If embedding did not finish within the budget the response carries
    ``indexing="partial"`` (the subject is already usable via keyword search) — call
    ``POST /api/subjects/{name}/continue-index`` to finish it.
    """
    started = time.monotonic()
    if len(files) > subjects_service.MAX_FILES_PER_REQUEST:
        raise HTTPException(status_code=413, detail=f"Too many files (max {subjects_service.MAX_FILES_PER_REQUEST}).")
    uploads: list[tuple[str, bytes]] = []
    remaining = subjects_service.MAX_TOTAL_BYTES
    for upload in files:
        data = await _read_limited(upload, subjects_service.MAX_FILE_BYTES, remaining)
        if subjects_service.MAX_TOTAL_BYTES:
            remaining = max(1, remaining - len(data))
        uploads.append((upload.filename or "document.pdf", data))

    try:
        result = await run_in_threadpool(subjects_service.add_subject, name, uploads, owner, started=started)
    except SubjectError as exc:
        raise _http(exc) from exc

    return AddSubjectResponse(
        subject=result.subject,
        saved_pdfs=result.saved_pdfs,
        chunk_count=result.chunk_count,
        indexing=result.indexing,
        n_embedded=result.n_embedded,
        skipped_duplicates=result.skipped_duplicates,
        warnings=result.warnings,
        reading=result.reading,
    )


# ---- chunked upload (Vercel rejects request bodies > 4.5 MB, so large PDFs travel in pieces) ----
# NOTE: these literal ``/uploads`` routes are 2-4 path segments deep, so they never collide with
# the single-segment ``DELETE /{subject}`` or the ``/{subject}/pdfs`` / ``/continue-index`` routes.
class UploadCreate(BaseModel):
    subject: str = Field(..., max_length=200)
    filename: str = Field(..., max_length=300)
    size: int = Field(..., ge=1, le=2**40)
    sha256: str | None = Field(default=None, max_length=64)


def _upload_error(exc: SubjectError) -> HTTPException:
    return HTTPException(status_code=exc.status_code, detail=str(exc))


@router.post("/uploads")
async def create_upload(body: UploadCreate, owner: str = Depends(current_owner)) -> dict:
    """Validate the target (name, size, type, caps) and open an upload: returns
    ``{upload_id, chunk_size, total_chunks}``."""
    try:
        return await run_in_threadpool(
            uploads_service.create_upload, owner, body.subject, body.filename, body.size, body.sha256)
    except SubjectError as exc:
        raise _upload_error(exc) from exc


@router.put("/uploads/{upload_id}/chunks/{index}")
async def put_upload_chunk(upload_id: str, index: int, request: Request,
                           owner: str = Depends(current_owner)) -> dict:
    """Store one raw ``application/octet-stream`` chunk (re-sending an index overwrites it)."""
    cap = uploads_service.CHUNK_SIZE
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > cap:
        raise HTTPException(status_code=413, detail=f"Chunk is too large (max {cap} bytes).")
    buf = bytearray()
    async for part in request.stream():
        buf += part
        if len(buf) > cap:
            raise HTTPException(status_code=413, detail=f"Chunk is too large (max {cap} bytes).")
    try:
        return await run_in_threadpool(uploads_service.put_chunk, owner, upload_id, index, bytes(buf))
    except SubjectError as exc:
        raise _upload_error(exc) from exc


@router.get("/uploads/{upload_id}")
def get_upload(upload_id: str, owner: str = Depends(current_owner)) -> dict:
    try:
        return uploads_service.upload_status(owner, upload_id)
    except SubjectError as exc:
        raise _upload_error(exc) from exc


@router.post("/uploads/{upload_id}/complete", response_model=AddSubjectResponse)
async def complete_upload(upload_id: str, owner: str = Depends(current_owner)) -> AddSubjectResponse:
    """Assemble the chunks, validate and index the PDF (same pipeline and response as the multipart
    add). Safe to retry: a finished upload answers with its stored result; a concurrent complete
    gets 409."""
    started = time.monotonic()
    try:
        result = await run_in_threadpool(uploads_service.complete_upload, owner, upload_id, started=started)
    except SubjectError as exc:
        raise _upload_error(exc) from exc
    return AddSubjectResponse(**result)


@router.delete("/uploads/{upload_id}")
def abort_upload(upload_id: str, owner: str = Depends(current_owner)) -> dict:
    try:
        return uploads_service.abort_upload(owner, upload_id)
    except SubjectError as exc:
        raise _upload_error(exc) from exc


@router.post("/{subject}/continue-index")
def continue_index(subject: str, owner: str = Depends(current_owner)) -> dict:
    """Embed more of a partially-indexed subject (bounded to ~35 s). Call repeatedly (sequentially)
    while ``indexing == "partial"`` and ``progress > 0``; ``busy: true`` means another call for the
    same subject is already running on this instance."""
    clean = _param(subject)
    try:
        return subjects_service.continue_index(clean, owner)
    except SubjectError as exc:
        raise _http(exc) from exc


class RenameBody(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)


@router.get("/{subject}/files")
def get_files(subject: str, owner: str = Depends(current_owner)) -> dict:
    try:
        return subjects_service.subject_files(_param(subject), owner)
    except SubjectError as exc:
        raise _http(exc) from exc


@router.delete("/{subject}/pdfs/{filename}")
async def remove_pdf(subject: str, filename: str, owner: str = Depends(current_owner)) -> dict:
    started = time.monotonic()
    try:
        return await run_in_threadpool(subjects_service.remove_pdf, _param(subject), owner, filename, started=started)
    except SubjectError as exc:
        raise _http(exc) from exc


@router.post("/{subject}/reindex")
async def reindex(subject: str, owner: str = Depends(current_owner)) -> dict:
    started = time.monotonic()
    try:
        return await run_in_threadpool(subjects_service.reindex_subject, _param(subject), owner, started=started)
    except SubjectError as exc:
        raise _http(exc) from exc


@router.patch("/{subject}")
async def rename(subject: str, body: RenameBody, owner: str = Depends(current_owner)) -> dict:
    try:
        return await run_in_threadpool(subjects_service.rename_subject, _param(subject), body.name, owner)
    except SubjectError as exc:
        raise _http(exc) from exc


@router.post("/restore-standard")
def restore_all(owner: str = Depends(current_owner)) -> dict:
    try:
        return subjects_service.restore_standard(None, owner)
    except SubjectError as exc:
        raise _http(exc) from exc


@router.post("/{subject}/hide")
def hide(subject: str, owner: str = Depends(current_owner)) -> dict:
    """Hide a standard subject for this account only (never deletes anything shared)."""
    try:
        return subjects_service.hide_standard(_param(subject), owner)
    except SubjectError as exc:
        raise _http(exc) from exc


@router.post("/{subject}/restore")
def restore(subject: str, owner: str = Depends(current_owner)) -> dict:
    try:
        return subjects_service.restore_standard(_param(subject), owner)
    except SubjectError as exc:
        raise _http(exc) from exc


@router.delete("/{subject}")
def delete_subject(subject: str, owner: str = Depends(current_owner)) -> dict:
    """Delete one of the caller's own subjects (PDFs, index, caches). Bundled subjects: 403."""
    clean = _param(subject)
    try:
        return subjects_service.delete_subject(clean, owner)
    except SubjectError as exc:
        raise _http(exc) from exc
