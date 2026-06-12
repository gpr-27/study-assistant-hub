"""Add-a-subject service: save uploaded PDFs and embed them once.

Orchestrates the document-write helpers (`backend.documents`) and the persistent
vector store (`backend.vector_store`): the PDFs are saved under
documents/<subject>/ so every tool can see them, then that one subject's FAISS
index is (re)built and persisted to disk. Existing subjects are left untouched —
only the affected subject is embedded.
"""
import logging
from dataclasses import dataclass

from backend import db, vector_store
from backend.documents import DocumentError, restore_uploaded_pdf, save_pdf

log = logging.getLogger(__name__)


class SubjectError(Exception):
    """Raised when a subject cannot be created from the uploaded files."""


@dataclass
class AddSubjectResult:
    subject: str
    saved_pdfs: list[str]
    chunk_count: int


def add_subject(name: str, files: list[tuple[str, bytes]], owner: str | None = None) -> AddSubjectResult:
    """Save uploaded PDFs under a subject and build its persistent index.

    `files` is a list of (filename, bytes). When owner is provided the subject is
    stored under the user-namespaced path so it is only visible to that user.
    Raises SubjectError if no files are given, none are valid PDFs, or the PDFs
    contain no readable text.
    """
    if not files:
        raise SubjectError("Add at least one PDF to create a subject.")

    saved: list[str] = []
    for filename, data in files:
        try:
            stored_name = save_pdf(name, filename, data, owner)
        except DocumentError as exc:
            raise SubjectError(str(exc)) from exc
        saved.append(stored_name)
        # Mirror the saved PDF into MongoDB so it survives a restart (best-effort).
        db.store_uploaded_pdf(name, stored_name, data, owner)

    if not saved:
        raise SubjectError("None of the uploaded files were valid PDFs.")

    try:
        chunk_count = vector_store.build_subject_index(name, owner)
    except vector_store.VectorStoreError as exc:
        raise SubjectError(
            f"Saved {len(saved)} PDF(s), but no readable text was found to embed. "
            "If these are scanned documents, they need a text layer (OCR)."
        ) from exc

    return AddSubjectResult(subject=name, saved_pdfs=saved, chunk_count=chunk_count)


def restore_uploaded_pdfs() -> int:
    """Re-hydrate documents/ from MongoDB-stored uploads. Returns the count restored.

    Called once at startup so user-uploaded subjects reappear after a restart on a
    host without a persistent disk. Each PDF is restored to its owner-namespaced path
    (``documents/user__{owner}__{subject}/``) so uploads remain scoped per user.
    """
    restored = 0
    for owner, subject, filename, data in db.iter_uploaded_pdfs():
        try:
            if restore_uploaded_pdf(subject, filename, data, owner):
                restored += 1
        except OSError as exc:
            log.warning("Could not restore uploaded PDF %s/%s: %s", subject, filename, exc)
    if restored:
        log.info("Restored %d uploaded PDF(s) from MongoDB into documents/.", restored)
    return restored
