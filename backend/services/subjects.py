"""Add-a-subject service: save uploaded PDFs and embed them once.

Orchestrates the document-write helpers (`backend.documents`) and the persistent
vector store (`backend.vector_store`): the PDFs are saved under
documents/<subject>/ so every tool can see them, then that one subject's FAISS
index is (re)built and persisted to disk. Existing subjects are left untouched —
only the affected subject is embedded.
"""
from dataclasses import dataclass

from backend import config, vector_store
from backend.documents import DocumentError, list_pdf_names, read_pdf, save_pdf


class SubjectError(Exception):
    """Raised when a subject cannot be created from the uploaded files."""


@dataclass
class AddSubjectResult:
    subject: str
    saved_pdfs: list[str]
    chunk_count: int


def add_subject(name: str, files: list[tuple[str, bytes]]) -> AddSubjectResult:
    """Save uploaded PDFs under a subject and build its persistent index.

    `files` is a list of (filename, bytes). Adding to an existing subject name
    appends the PDFs and rebuilds that subject's index. Raises SubjectError if no
    files are given, none are valid PDFs, or the PDFs contain no readable text.
    """
    if not files:
        raise SubjectError("Add at least one PDF to create a subject.")

    saved: list[str] = []
    for filename, data in files:
        try:
            saved.append(save_pdf(name, filename, data))
        except DocumentError as exc:
            raise SubjectError(str(exc)) from exc

    if not saved:
        raise SubjectError("None of the uploaded files were valid PDFs.")

    if config.RAG_ENABLED:
        try:
            chunk_count = vector_store.build_subject_index(name)
        except vector_store.VectorStoreError as exc:
            # The PDFs are saved, but nothing could be embedded (e.g. scanned images
            # with no extractable text). Surface that clearly; the files remain.
            raise SubjectError(
                f"Saved {len(saved)} PDF(s), but no readable text was found to embed. "
                "If these are scanned documents, they need a text layer (OCR)."
            ) from exc
    else:
        # Lite mode (no embeddings): nothing to index, but still reject text-less PDFs so the
        # subject is actually usable by the tools, which read this text directly.
        if not any(read_pdf(name, n).strip() for n in list_pdf_names(name)):
            raise SubjectError(
                f"Saved {len(saved)} PDF(s), but no readable text was found. "
                "If these are scanned documents, they need a text layer (OCR)."
            )
        chunk_count = 0

    return AddSubjectResult(subject=name, saved_pdfs=saved, chunk_count=chunk_count)
