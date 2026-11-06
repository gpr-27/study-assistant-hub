"""Flashcards API: generate active-recall cards from a subject PDF, a free
topic, or an uploaded PDF — mirroring the Chat tool's three sources. Each generated
deck is saved to the signed-in user's history (when persistence is enabled), including the
`llm` (provider/model label) that produced it.

DocumentError bubbles to the global 404 handler; LLMChainError to the global 503 handler;
FlashcardError (unusable model output) maps to 502 here.
"""
from datetime import datetime, timezone
from typing import Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from backend import db
from backend import privacy
from backend.auth import current_owner
from backend.documents import read_pdf
from backend.routers._common import model_of, read_pdf_upload, require_text, require_topic
from backend.schemas import (
    Flashcard,
    FlashcardsRequest,
    FlashcardsResponse,
    FlashcardsTopicRequest,
)
from backend.services import attempts, grading, srs
from backend.services.flashcards import (
    MAX_CARDS,
    FlashcardError,
    make_flashcards,
    make_flashcards_from_topic,
    make_typed_cards,
)

router = APIRouter(prefix="/api/flashcards", tags=["flashcards"])


def _finalize(result: dict, owner: str, *, title: str, source: str, difficulty: str) -> FlashcardsResponse:
    """Validate cards, save the deck to this user's history, and return it with its id."""
    cards = [Flashcard(**card) for card in result.get("cards", [])]
    if not cards:
        raise HTTPException(
            status_code=502,
            detail="The model did not return any usable flashcards. Try again or pick a different source.",
        )
    llm = result.get("llm")
    saved_id = db.save("flashcards", owner, {
        "title": title,
        "source": source,
        "difficulty": difficulty,
        "model": model_of(llm),
        "llm": llm,
        "num_cards": len(cards),
        "cards": [c.model_dump() for c in cards],
    })
    return FlashcardsResponse(cards=cards, id=saved_id, title=title, llm=llm)


def _run(fn, *args) -> dict:
    try:
        return fn(*args)
    except FlashcardError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/generate", response_model=FlashcardsResponse)
def generate_flashcards(request: FlashcardsRequest, owner: str = Depends(current_owner)) -> FlashcardsResponse:
    """Generate flashcards from the requested subject PDF's extracted text."""
    privacy.mark(privacy.for_subject(request.subject))
    content = require_text(read_pdf(request.subject, request.pdf, owner))
    result = _run(make_flashcards, content, request.num_cards, request.difficulty)
    return _finalize(
        result, owner,
        title=f"Flashcards · {request.subject} / {request.pdf}",
        source=f"{request.subject} / {request.pdf}",
        difficulty=request.difficulty,
    )


@router.post("/topic", response_model=FlashcardsResponse)
def generate_flashcards_topic(request: FlashcardsTopicRequest, owner: str = Depends(current_owner)) -> FlashcardsResponse:
    """Generate flashcards about a free topic (no document)."""
    topic = require_topic(request.topic)
    result = _run(make_flashcards_from_topic, topic, request.num_cards, request.difficulty)
    return _finalize(result, owner, title=f"Flashcards · {topic}", source=topic, difficulty=request.difficulty)


@router.post("/upload", response_model=FlashcardsResponse)
async def generate_flashcards_upload(
    file: UploadFile = File(...),
    num_cards: int = Form(10, ge=1, le=MAX_CARDS),
    difficulty: str = Form("medium", max_length=20),
    model: str | None = Form(None),
    owner: str = Depends(current_owner),
) -> FlashcardsResponse:
    """Generate flashcards from a single uploaded PDF's extracted text."""
    text, fname = await read_pdf_upload(file)
    privacy.mark(privacy.for_upload())

    def work() -> FlashcardsResponse:
        result = _run(make_flashcards, text, num_cards, difficulty)
        return _finalize(result, owner, title=f"Flashcards · {fname}", source=fname, difficulty=difficulty)

    return await run_in_threadpool(work)


class TextDeckRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2_000_000)         # pasted notes: sent to the model in full
    title: str | None = Field(default=None, max_length=120)
    num_cards: int = Field(default=10, ge=1, le=MAX_CARDS)
    difficulty: str = Field(default="medium", max_length=20)
    card_type: Literal["basic", "reversed", "cloze", "mcq", "typed"] = "basic"
    topic_only: bool = False                                        # treat ``text`` as a topic name, not material


@router.post("/text", response_model=FlashcardsResponse)
def generate_flashcards_text(request: TextDeckRequest, owner: str = Depends(current_owner)) -> FlashcardsResponse:
    """AI cards of a chosen TYPE from pasted text (or a topic name); saved as a named deck. Labelled with ``llm``."""
    privacy.mark(True)                                              # pasted notes are user content: keep off Gemini
    if db.get_db() is not None and db.get_db()["flashcards"].count_documents({"owner": owner}) >= 200:
        raise HTTPException(status_code=409, detail="You can keep at most 200 decks. Delete one first.")
    result = _run(lambda: make_typed_cards(request.text, request.num_cards, request.difficulty, request.card_type,
                                           is_topic=request.topic_only))
    cards = result.get("cards") or []
    if not cards:
        raise HTTPException(status_code=502, detail="The model did not return any usable flashcards. Try again.")
    title = request.title or ("Flashcards \u00b7 " + (request.text.strip().splitlines()[0][:50] if request.text.strip() else "notes"))
    llm = result.get("llm")
    saved_id = db.save("flashcards", owner, {"title": title, "source": "text", "difficulty": request.difficulty, "model": model_of(llm),
                                             "llm": llm, "num_cards": len(cards), "num_units": len(srs.units(cards)),
                                             "cards": cards, "ver": 0})
    return FlashcardsResponse(cards=[Flashcard(front=c["front"], back=c["back"], topic=c.get("topic") or "General") for c in cards],
                              id=saved_id, title=title, llm=llm)


# ---------------------------------------------------------------------------
# Spaced repetition (SM-2): due queue + review. State lives on the deck document (see services/srs.py).
# ---------------------------------------------------------------------------
class ReviewItem(BaseModel):
    card_id: str = Field(min_length=1, max_length=24)
    rating: Literal["again", "hard", "good", "easy"]


class ReviewRequest(BaseModel):
    deck_id: str = Field(min_length=1, max_length=64)
    batch_id: str = Field(min_length=8, max_length=64)       # client-minted: a retried / double-clicked batch is a no-op
    reviews: list[ReviewItem] = Field(min_length=1, max_length=300)


DAILY_NEW_LIMIT = 20


def _today():
    return datetime.now(timezone.utc).date()


def _load_deck(owner: str, deck_id: str, fields: dict) -> dict:
    """The owner's deck document (projected) or a 404 - another user's deck id is simply unknown."""
    coll = db.get_db()
    oid = db.oid(deck_id)
    doc = coll["flashcards"].find_one({"_id": oid, "owner": owner}, fields) if coll is not None and oid else None
    if not doc:
        raise HTTPException(status_code=404, detail="Deck not found. Open a saved deck to use spaced repetition.")
    return doc


@router.get("/due")
def due_cards(
    deck_id: str | None = Query(None, max_length=64),
    new_limit: int = Query(10, ge=0, le=50),
    limit: int = Query(50, ge=1, le=200),
    mode: Literal["due", "cram", "starred"] = "due",
    tag: str | None = Query(None, max_length=40),
    owner: str = Depends(current_owner),
) -> dict:
    """With ``deck_id``: that deck's study queue (due first, then a few new cards) and counts.
    Without: the decks that have cards due today (counts only) - the "due today" badge."""
    today = _today()
    if deck_id:
        deck = _load_deck(owner, deck_id, {"title": 1, "cards": 1, "srs": 1, "srs_new": 1})
        cards = deck.get("cards") or []
        state = deck.get("srs") or {}
        used = (deck.get("srs_new") or {})
        used = used.get("n", 0) if used.get("d") == today.isoformat() else 0
        new_limit = max(0, min(new_limit, DAILY_NEW_LIMIT - used))        # daily new-card budget
        return {"deck_id": deck_id, "title": deck.get("title"), "today": today.isoformat(),
                "counts": srs.counts(srs.unit_ids(cards), state, today),
                "cards": srs.build_queue(cards, state, today, new_limit=new_limit, limit=limit, mode=mode, tag=tag)}
    coll = db.get_db()
    if coll is None:
        return {"today": today.isoformat(), "decks": [], "due_total": 0, "persistence": False}
    rows = coll["flashcards"].find({"owner": owner, "srs_due": {"$lte": today.isoformat()}},
                                   {"title": 1, "num_cards": 1, "srs": 1}).sort("srs_due", 1).limit(30)
    decks, due_total = [], 0
    for d in rows:
        st = d.get("srs") or {}
        n_due = sum(1 for v in st.values() if srs.status(v, today) in ("due", "learning"))
        if n_due:
            due_total += n_due
            decks.append({"deck_id": str(d["_id"]), "title": d.get("title"), "due": n_due,
                          "total": d.get("num_cards") or len(st)})
    return {"today": today.isoformat(), "decks": decks, "due_total": due_total, "persistence": True}


@router.post("/review")
def review_cards(req: ReviewRequest, owner: str = Depends(current_owner)) -> dict:
    """Apply a batch of ratings to a deck: ONE read + ONE ``$set`` on the deck + one attempt record (kind
    ``flashcards`` in the ``attempts`` collection, so Analytics and weak topics count the study session).
    Idempotent per ``batch_id``: a retried batch replays the stored attempt and never reschedules twice."""
    if not attempts.valid_attempt_id(req.batch_id):
        raise HTTPException(status_code=422, detail="batch_id must be 8-64 letters, digits, '-' or '_'.")
    today = _today()

    def compute() -> dict:
        deck = _load_deck(owner, req.deck_id, {"title": 1, "cards": 1, "srs": 1, "srs_new": 1})
        cards = deck.get("cards") or []
        ids = srs.unit_ids(cards)
        by_id = {u["cid"]: u for u in srs.units(cards)}
        state = {k: dict(v) for k, v in (deck.get("srs") or {}).items()}
        rows: list[dict] = []
        sets: dict = {}
        fresh = 0
        for rv in req.reviews:
            card = by_id.get(rv.card_id)
            if card is None:
                continue                                      # unknown card id (stale client): skipped, not an error
            if state.get(rv.card_id) is None:
                fresh += 1
            new = srs.schedule(state.get(rv.card_id), rv.rating, today)
            state[rv.card_id] = new
            sets[f"srs.{rv.card_id}"] = new
            # Analytics count a card as correct only when it was rated good or easy (hard/again are misses).
            correct = rv.rating in ("good", "easy")
            days = new["i"]
            when = "again today" if rv.rating == "again" else f"in {days} day{'s' if days != 1 else ''}"
            rows.append({
                # Topic: the card's own topic, else the deck title (so weak topics name the deck, not "General").
                "id": rv.card_id, "topic": card.get("topic") or deck.get("title") or "General", "kind": "flashcard",
                "question": card.get("front", ""), "score": 1.0 if correct else 0.0, "max": 1.0,
                "verdict": "correct" if correct else "incorrect",
                "feedback": f"Next review {when}.", "user_answer": rv.rating, "correct_answer": card.get("back", ""),
                "due": new["d"], "interval": days})
        if not rows:
            raise HTTPException(status_code=422, detail="None of those cards belong to this deck.")
        due_dates = [v["d"] for v in state.values() if v.get("d")]
        sets["srs_due"] = min(due_dates)
        prev = deck.get("srs_new") or {}
        sets["srs_new"] = {"d": today.isoformat(), "n": (prev.get("n", 0) if prev.get("d") == today.isoformat() else 0) + fresh}
        sets["srs_reviewed_at"] = db.now()
        sets["updated_at"] = db.now()
        coll = db.get_db()
        try:
            coll["flashcards"].update_one({"_id": db.oid(req.deck_id), "owner": owner}, {"$set": sets})
        finally:
            db._bump(owner)
        result = grading.build_result(items=[], rows=rows, kind="flashcards", attempt_id=req.batch_id,
                                      subject=deck.get("title"), topic=deck.get("title"),
                                      title=f"Review \u00b7 {deck.get('title') or 'flashcards'}", verified=True)
        result["deck_id"] = req.deck_id
        result["srs"] = srs.counts(ids, state, today)
        return result

    result, _ = attempts.run_idempotent(owner, req.batch_id, "flashcards", compute)
    return result
