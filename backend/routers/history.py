"""History API — list / search / view / delete saved artifacts, scoped per user.

One uniform surface for every tool's memory:
  GET    /api/history/{kind}?q=<search>   -> summaries (newest first), this user only
  GET    /api/history/{kind}/{id}          -> the full saved artifact
  DELETE /api/history/{kind}/{id}          -> remove it

``kind`` is one of chats | quizzes | flashcards | plans | mindmaps. Every route is
scoped by current_owner, so a user can only ever see or delete their own items. When
persistence is disabled (no MONGODB_URI) the list is simply empty.
"""
from fastapi import APIRouter, Depends, HTTPException, Query

from backend import db
from backend.auth import current_owner
from backend.schemas import RenameRequest

router = APIRouter(prefix="/api/history", tags=["history"])

_KINDS = set(db.KINDS)


def _check_kind(kind: str) -> None:
    if kind not in _KINDS:
        raise HTTPException(status_code=404, detail=f"Unknown history type: {kind}")


@router.get("/{kind}")
def list_history(
    kind: str,
    q: str | None = Query(None, description="Case-insensitive search filter"),
    owner: str = Depends(current_owner),
) -> dict:
    _check_kind(kind)
    return {"items": db.list_items(kind, owner, q=q), "persistence": db.is_enabled()}


@router.get("/{kind}/{item_id}")
def get_history(kind: str, item_id: str, owner: str = Depends(current_owner)) -> dict:
    _check_kind(kind)
    doc = db.get_item(kind, owner, item_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Item not found.")
    return doc


@router.patch("/{kind}/{item_id}")
def rename_history(kind: str, item_id: str, body: RenameRequest, owner: str = Depends(current_owner)) -> dict:
    _check_kind(kind)
    title = (body.title or "").strip()[:120]
    if not title:
        raise HTTPException(status_code=400, detail="Title cannot be empty.")
    if not db.rename_item(kind, owner, item_id, title):
        raise HTTPException(status_code=404, detail="Item not found.")
    return {"renamed": True, "title": title}


@router.delete("/{kind}/{item_id}")
def delete_history(kind: str, item_id: str, owner: str = Depends(current_owner)) -> dict:
    _check_kind(kind)
    if not db.delete_item(kind, owner, item_id):
        raise HTTPException(status_code=404, detail="Item not found.")
    return {"deleted": True}


@router.delete("/{kind}")
def clear_history(kind: str, owner: str = Depends(current_owner)) -> dict:
    """Delete ALL of this user's items of a kind (Clear all)."""
    _check_kind(kind)
    return {"deleted": db.clear_kind(kind, owner)}
