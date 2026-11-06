"""Flashcard deck management: named decks per owner, typed cards, import / export, duplicates, related, typed-answer check.

A deck is ONE document in the ``flashcards`` collection (cards embedded; SRS state in ``srs`` - see services/srs.py), so
every operation is 1-2 Mongo round trips. Edits are read-modify-write guarded by a ``ver`` counter (a concurrent edit
gets a 409 asking the client to reload), which keeps many users / tabs safe without multi-document transactions.
Limits (clear 4xx): <= 200 decks per owner, <= 5,000 cards per deck, <= 1,000 cards per import.
"""
from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Path, Query
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from backend import db, privacy
from backend.auth import current_owner
from backend.services import deckio, grading, srs
from backend.services.deckio import MAX_CARDS_PER_DECK, MAX_DECKS_PER_OWNER, MAX_IMPORT

router = APIRouter(prefix="/api/flashcards/decks", tags=["flashcard-decks"])

DeckId = Path(..., max_length=64)


class DeckCreate(BaseModel):
    title: str = Field(min_length=1, max_length=120)
    subject: str | None = Field(default=None, max_length=255)
    cards: list[dict] = Field(default_factory=list, max_length=MAX_IMPORT)


class DeckPatch(BaseModel):
    title: str = Field(min_length=1, max_length=120)


class CardsAdd(BaseModel):
    cards: list[dict] = Field(min_length=1, max_length=MAX_IMPORT)
    skip_duplicates: bool = False


class CardPatch(BaseModel):
    front: str | None = Field(default=None, max_length=deckio.MAX_FRONT)
    back: str | None = Field(default=None, max_length=deckio.MAX_BACK)
    type: str | None = None
    tags: list[str] | None = Field(default=None, max_length=10)
    hint: str | None = Field(default=None, max_length=500)
    explanation: str | None = Field(default=None, max_length=1500)
    topic: str | None = Field(default=None, max_length=80)
    starred: bool | None = None
    options: list[str] | None = Field(default=None, max_length=6)
    answer_index: int | None = None


class ImportReq(BaseModel):
    text: str = Field(min_length=1, max_length=2_000_000)
    format: Literal["lines", "csv"] = "lines"
    skip_duplicates: bool = True


class CheckReq(BaseModel):
    card_id: str = Field(max_length=40)
    answer: str = Field(max_length=4000)
    attempt_id: str | None = Field(default=None, max_length=64)


def _coll():
    c = db.get_db()
    if c is None:
        raise HTTPException(status_code=503, detail="Decks need history to be enabled (persistence is off).")
    return c["flashcards"]


def _oid(deck_id: str):
    oid = db.oid(deck_id)
    if oid is None:
        raise HTTPException(status_code=404, detail="Deck not found.")
    return oid


def _load(owner: str, deck_id: str) -> dict:
    d = _coll().find_one({"_id": _oid(deck_id), "owner": owner})
    if not d:
        raise HTTPException(status_code=404, detail="Deck not found.")
    return d


def _summary(d: dict, today) -> dict:
    st = d.get("srs") or {}
    n_due = sum(1 for v in st.values() if srs.status(v, today) in ("due", "learning"))
    units = d.get("num_units") or d.get("num_cards") or len(d.get("cards") or [])
    return {"id": str(d["_id"]), "title": d.get("title"), "subject": d.get("subject"),
            "cards": d.get("num_cards") or len(d.get("cards") or []), "due": n_due,
            "new": max(0, units - len(st)), "last_studied": d.get("srs_reviewed_at"),
            "created_at": d.get("created_at"), "source": d.get("source")}


def _iso(v):
    return v.isoformat() if hasattr(v, "isoformat") else v


def _public_deck(d: dict, today) -> dict:
    out = _summary(d, today)
    out["last_studied"], out["created_at"] = _iso(out["last_studied"]), _iso(out["created_at"])
    return out


def _mutate(owner: str, deck_id: str, fn) -> tuple[dict, Any]:
    """Read-modify-write one deck with optimistic concurrency. ``fn(deck) -> (fields_to_set, result)``."""
    coll, oid = _coll(), _oid(deck_id)
    for _ in range(3):
        deck = coll.find_one({"_id": oid, "owner": owner})
        if not deck:
            raise HTTPException(status_code=404, detail="Deck not found.")
        fields, result = fn(deck)
        cards = fields.get("cards", deck.get("cards") or [])
        fields["num_cards"] = len(cards)
        fields["num_units"] = len(srs.units(cards))
        fields["updated_at"] = db.now()
        fields["ver"] = (deck.get("ver") or 0) + 1
        try:
            res = coll.update_one({"_id": oid, "owner": owner, "ver": deck.get("ver")}, {"$set": fields})
        finally:
            db._bump(owner)
        if res.matched_count:
            deck.update(fields)
            return deck, result
    raise HTTPException(status_code=409, detail="This deck was changed elsewhere. Reload it and try again.")


def _prune_srs(deck: dict, cards: list[dict]) -> dict:
    keep = set(srs.unit_ids(cards))
    state = {k: v for k, v in (deck.get("srs") or {}).items() if k in keep}
    due = [v["d"] for v in state.values() if v.get("d")]
    return {"srs": state, "srs_due": min(due) if due else None}


def _normalise_many(raws: list[dict]) -> list[dict]:
    cards = [c for c in (deckio.normalise_card(r) for r in raws) if c]
    if not cards:
        raise HTTPException(status_code=422, detail="No valid cards. Basic/typed/reversed cards need a front and a back; "
                                                    "cloze needs {{c1::...}}; multiple choice needs 2-6 options and an answer.")
    return cards


def _add(owner: str, deck_id: str, raws: list[dict], skip_dupes: bool) -> tuple[dict, dict]:
    cards = _normalise_many(raws)

    def fn(deck):
        existing = deck.get("cards") or []
        dupes = deckio.find_duplicates(cards, existing)
        bad = {d["index"] for d in dupes} if skip_dupes else set()
        fresh = [c for i, c in enumerate(cards) if i not in bad]
        have = {c.get("id") for c in existing}
        fresh = [c for c in fresh if c["id"] not in have]
        if len(existing) + len(fresh) > MAX_CARDS_PER_DECK:
            raise HTTPException(status_code=409, detail=f"A deck holds at most {MAX_CARDS_PER_DECK} cards.")
        return {"cards": existing + fresh}, {"added": [c["id"] for c in fresh], "duplicates": dupes}

    return _mutate(owner, deck_id, fn)


@router.get("")
def list_decks(owner: str = Depends(current_owner)) -> dict:
    from datetime import datetime, timezone
    today = datetime.now(timezone.utc).date()
    coll = db.get_db()
    if coll is None:
        return {"decks": [], "persistence": False}
    rows = coll["flashcards"].find({"owner": owner}, {"title": 1, "subject": 1, "num_cards": 1, "num_units": 1, "srs": 1,
                                                       "srs_reviewed_at": 1, "created_at": 1, "source": 1, "cards": 1}).sort("created_at", -1).limit(MAX_DECKS_PER_OWNER)
    decks = [_public_deck(d, today) for d in rows]
    return {"decks": decks, "persistence": True, "limit": MAX_DECKS_PER_OWNER}


@router.post("")
def create_deck(req: DeckCreate, owner: str = Depends(current_owner)) -> dict:
    coll = _coll()
    if coll.count_documents({"owner": owner}) >= MAX_DECKS_PER_OWNER:
        raise HTTPException(status_code=409, detail=f"You can keep at most {MAX_DECKS_PER_OWNER} decks. Delete one first.")
    cards = [c for c in (deckio.normalise_card(r) for r in req.cards) if c]
    deck_id = db.save("flashcards", owner, {
        "title": req.title.strip(), "subject": req.subject, "source": "manual", "num_cards": len(cards),
        "num_units": len(srs.units(cards)), "cards": cards, "ver": 0})
    if not deck_id:
        raise HTTPException(status_code=503, detail="Could not save the deck. Please retry.")
    return {"id": deck_id, "title": req.title.strip(), "cards": len(cards)}


@router.get("/{deck_id}")
def get_deck(deck_id: str = DeckId, q: str | None = Query(None, max_length=100), tag: str | None = Query(None, max_length=40),
             owner: str = Depends(current_owner)) -> dict:
    from datetime import datetime, timezone
    d = _load(owner, deck_id)
    cards = d.get("cards") or []
    tags = sorted({t for c in cards for t in c.get("tags") or []})
    if q:
        n = grading.norm_text(q)
        cards = [c for c in cards if n in grading.norm_text(c.get("front", "") + " " + c.get("back", "") + " " + " ".join(c.get("tags") or []))]
    if tag:
        cards = [c for c in cards if tag in (c.get("tags") or [])]
    out = _public_deck(d, datetime.now(timezone.utc).date())
    out.update(cards=cards, tags=tags, total_cards=d.get("num_cards") or len(d.get("cards") or []), llm=d.get("llm"))
    return out


@router.patch("/{deck_id}")
def rename_deck(req: DeckPatch, deck_id: str = DeckId, owner: str = Depends(current_owner)) -> dict:
    _mutate(owner, deck_id, lambda d: ({"title": req.title.strip()}, None))
    return {"id": deck_id, "title": req.title.strip()}


@router.delete("/{deck_id}")
def delete_deck(deck_id: str = DeckId, owner: str = Depends(current_owner)) -> dict:
    try:
        n = _coll().delete_one({"_id": _oid(deck_id), "owner": owner}).deleted_count
    finally:
        db._bump(owner)
    if not n:
        raise HTTPException(status_code=404, detail="Deck not found.")
    return {"deleted": True}


@router.post("/{deck_id}/duplicate")
def duplicate_deck(deck_id: str = DeckId, owner: str = Depends(current_owner)) -> dict:
    d = _load(owner, deck_id)
    if _coll().count_documents({"owner": owner}) >= MAX_DECKS_PER_OWNER:
        raise HTTPException(status_code=409, detail=f"You can keep at most {MAX_DECKS_PER_OWNER} decks. Delete one first.")
    cards = [{**c, "id": deckio.new_id()} for c in d.get("cards") or []]       # fresh ids: schedules start from scratch
    new_id = db.save("flashcards", owner, {"title": f"{d.get('title') or 'Deck'} (copy)", "subject": d.get("subject"),
                                           "source": "copy", "num_cards": len(cards), "num_units": len(srs.units(cards)),
                                           "cards": cards, "ver": 0})
    return {"id": new_id, "cards": len(cards)}


@router.post("/{deck_id}/cards")
def add_cards(req: CardsAdd, deck_id: str = DeckId, owner: str = Depends(current_owner)) -> dict:
    deck, res = _add(owner, deck_id, req.cards, req.skip_duplicates)
    return {**res, "total": deck["num_cards"]}


@router.patch("/{deck_id}/cards/{card_id}")
def edit_card(req: CardPatch, deck_id: str = DeckId, card_id: str = Path(..., max_length=40),
              owner: str = Depends(current_owner)) -> dict:
    patch = req.model_dump(exclude_none=True)

    def fn(deck):
        cards = list(deck.get("cards") or [])
        ids = srs.card_ids(cards)
        if card_id not in ids:
            raise HTTPException(status_code=404, detail="Card not found.")
        i = ids.index(card_id)
        new = deckio.normalise_card({**cards[i], **patch, "id": card_id if deckio._ID_RE.match(card_id) else None})
        if new is None:
            raise HTTPException(status_code=422, detail="That edit leaves the card invalid.")
        if not deckio._ID_RE.match(card_id):                 # legacy (hash-id) card: keep its schedule by pinning the old id
            new["id"] = card_id
        cards[i] = new
        return {"cards": cards, **_prune_srs(deck, cards)}, new

    deck, new = _mutate(owner, deck_id, fn)
    return {"card": new}


@router.delete("/{deck_id}/cards/{card_id}")
def delete_card(deck_id: str = DeckId, card_id: str = Path(..., max_length=40), owner: str = Depends(current_owner)) -> dict:
    """Returns the removed card and its schedule so the client can offer Undo (re-POST /cards with the same id)."""
    def fn(deck):
        cards = list(deck.get("cards") or [])
        ids = srs.card_ids(cards)
        if card_id not in ids:
            raise HTTPException(status_code=404, detail="Card not found.")
        i = ids.index(card_id)
        removed = cards.pop(i)
        removed = {**removed, "id": removed.get("id") or card_id}
        return {"cards": cards, **_prune_srs(deck, cards)}, removed

    _, removed = _mutate(owner, deck_id, fn)
    return {"removed": removed}


class SimilarReq(BaseModel):
    num_cards: int = Field(default=5, ge=1, le=20)
    difficulty: str = Field(default="medium", max_length=20)
    card_ids: list[str] = Field(default_factory=list, max_length=30)      # base the new cards on these (default: the deck)


@router.post("/{deck_id}/similar")
def similar_cards(req: SimilarReq, deck_id: str = DeckId, owner: str = Depends(current_owner)) -> dict:
    """AI: more cards like the deck's (or the chosen) cards. Grounded in them, deduped against the deck, labelled."""
    from backend.services.flashcards import FlashcardError, make_similar_cards
    privacy.mark(True)
    cards = _load(owner, deck_id).get("cards") or []
    ids = srs.card_ids(cards)
    seeds = [c for c, i in zip(cards, ids) if not req.card_ids or i in req.card_ids] or cards
    try:
        out = make_similar_cards(seeds[-30:] if not req.card_ids else seeds, req.num_cards, req.difficulty)
    except FlashcardError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    deck, res = _add(owner, deck_id, out["cards"], True)
    return {"added": res["added"], "duplicates_skipped": len(res["duplicates"]), "total": deck["num_cards"], "llm": out["llm"]}


@router.post("/{deck_id}/import")
def import_cards(req: ImportReq, deck_id: str = DeckId, owner: str = Depends(current_owner)) -> dict:
    raws = (deckio.parse_csv if req.format == "csv" else deckio.parse_bulk)(req.text)
    if not raws:
        raise HTTPException(status_code=422, detail="Nothing to import. Use one 'front :: back' per line, or Q:/A: pairs, or CSV.")
    if len(raws) > MAX_IMPORT:
        raise HTTPException(status_code=422, detail=f"Import at most {MAX_IMPORT} cards at a time.")
    deck, res = _add(owner, deck_id, raws, req.skip_duplicates)
    return {**res, "parsed": len(raws), "skipped_invalid": len(raws) - len(res["added"]) - len(res["duplicates"]), "total": deck["num_cards"]}


@router.get("/{deck_id}/export")
def export_deck(deck_id: str = DeckId, format: Literal["csv", "tsv", "json"] = "csv", owner: str = Depends(current_owner)):
    cards = _load(owner, deck_id).get("cards") or []
    body, mime = {"csv": (deckio.export_csv, "text/csv"), "tsv": (deckio.export_anki_tsv, "text/tab-separated-values"),
                  "json": (deckio.export_json, "application/json")}[format]
    return PlainTextResponse(body(cards), media_type=mime + "; charset=utf-8")


@router.get("/{deck_id}/related")
def related_cards(deck_id: str = DeckId, card_id: str = Query(..., max_length=40), owner: str = Depends(current_owner)) -> dict:
    cards = _load(owner, deck_id).get("cards") or []
    ids = srs.card_ids(cards)
    if card_id not in ids:
        raise HTTPException(status_code=404, detail="Card not found.")
    found, how = deckio.related(cards[ids.index(card_id)] | {"id": card_id}, [c | {"id": i} for c, i in zip(cards, ids)])
    return {"related": found, "method": how}


@router.post("/{deck_id}/check")
def check_answer(req: CheckReq, deck_id: str = DeckId, owner: str = Depends(current_owner)) -> dict:
    """Mark a typed answer (or an MCQ choice). Normalised match against the back / its ``/``-separated alternatives
    first (no model); otherwise the rubric grader (labelled ``llm``; provisional if the chain is down)."""
    cards = _load(owner, deck_id).get("cards") or []
    ids = srs.card_ids(cards)
    base = req.card_id.split(":")[0]
    if base not in ids:
        raise HTTPException(status_code=404, detail="Card not found.")
    c = cards[ids.index(base)]
    if c.get("type") == "mcq":
        item = grading.normalise_item({"type": "mcq", "question": c["front"], "options": c["options"], "answer_index": c["answer_index"]})
        row = grading.grade_choice(item, req.answer)
        return {"correct": row["verdict"] == "correct", "score": row["score"], "feedback": row["feedback"], "llm": None, "provisional": False,
                "answer": c.get("back")}
    back = c.get("back", "")
    given = grading.norm_text(req.answer)
    alts = [grading.norm_text(a) for a in str(back).replace(";", "/").split("/") if a.strip()] + [grading.norm_text(back)]
    if given and given in alts:
        return {"correct": True, "score": 1, "feedback": "Correct.", "llm": None, "provisional": False, "answer": back}
    if not given:
        return {"correct": False, "score": 0, "feedback": "No answer given.", "llm": None, "provisional": False, "answer": back}
    privacy.mark(True)
    item = grading.normalise_item({"type": "short", "question": c["front"], "expected_points": [back], "max": 1})
    rows, llm, why = grading.grade_free_text([{"item": {**item, "id": "c1"}, "answer": req.answer.strip()}])
    row = rows["c1"]
    return {"correct": row["verdict"] == "correct", "score": row["score"], "feedback": row["feedback"], "llm": llm,
            "provisional": bool(row.get("provisional")), "answer": back}
