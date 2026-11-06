"""Spaced repetition for flashcard decks (SM-2 style) - persisted compactly, ONE field on the deck document.

State per card (short keys, a few dozen bytes): ``{"e": ease, "i": interval_days, "n": reps, "l": lapses,
"d": due "YYYY-MM-DD" (UTC), "r": last reviewed "YYYY-MM-DD"}``, stored under ``flashcards.<deck>.srs.<card id>``;
the deck also carries ``srs_due`` (earliest due date) so "what is due today across my decks" is one indexed-by-owner
query. A card with no state is *new*.

Ratings (``again`` / ``hard`` / ``good`` / ``easy``) map to SM-2 quality 1 / 3 / 4 / 5:

* ease update (SM-2): ``EF' = max(1.3, EF + 0.1 - (5-q) * (0.08 + (5-q) * 0.02))`` -> hard -0.14, good +0, easy +0.10;
* ``again`` is a lapse: reps reset to 0, lapses += 1, ease -= 0.20 (floor 1.3), due again TODAY (the card stays in
  the session queue), and the next success restarts the 1 -> 6 -> ``I*EF`` ladder;
* intervals: first success 1 day (easy 3), second 6 days (hard 3, easy 8), afterwards ``round(I * EF)``
  (hard ``I * 1.2``, easy ``I * EF * 1.3``) and always at least +1 day (+2 for easy) so a card never stalls.

Pure functions, no I/O: the router does the Mongo read + the single ``$set`` write.
"""
from __future__ import annotations

import hashlib
import re
from datetime import date, timedelta

RATINGS = ("again", "hard", "good", "easy")
DEFAULT_EASE = 2.5
MIN_EASE = 1.3
MATURE_DAYS = 21
_Q = {"hard": 3, "good": 4, "easy": 5}
_FIRST = {"hard": 1, "good": 1, "easy": 3}
_SECOND = {"hard": 3, "good": 6, "easy": 8}
RATING_SCORE = {"again": 0.0, "hard": 0.5, "good": 1.0, "easy": 1.0}


def card_ids(cards: list[dict]) -> list[str]:
    """Stable ids for a deck's cards: the card's own ``id`` (cards made in the deck editor) else a hash of
    front+back (+ occurrence when a legacy deck repeats a card)."""
    seen: dict[str, int] = {}
    out = []
    for c in cards:
        own = c.get("id")
        if isinstance(own, str) and own:
            out.append(own)
            continue
        base = hashlib.sha1(f"{c.get('front', '')}\x1f{c.get('back', '')}".encode("utf-8")).hexdigest()[:10]
        n = seen.get(base, 0)
        seen[base] = n + 1
        out.append(base if n == 0 else f"{base}{n}")
    return out


_CLOZE = re.compile(r"\{\{c(\d+)::(.*?)(?:::(.*?))?\}\}", re.S)


def cloze_numbers(text: str) -> list[int]:
    return sorted({int(m.group(1)) for m in _CLOZE.finditer(text or "")})


def cloze_render(text: str, n: int, reveal: bool) -> str:
    """Render cloze ``text`` for deletion ``n``: that deletion hidden as ``[...]`` (or ``[hint]``) unless ``reveal``;
    every other deletion shown plainly."""
    def sub(m):
        if int(m.group(1)) != n:
            return m.group(2)
        if reveal:
            return f"**{m.group(2)}**"
        return f"[{m.group(3)}]" if m.group(3) else "[...]"
    return _CLOZE.sub(sub, text or "")


def units(cards: list[dict]) -> list[dict]:
    """Expand cards into independently scheduled study units (one per ``cid``).

    basic / mcq / typed -> 1 unit (cid = card id); reversed -> 2 (``:a`` front->back, ``:b`` back->front);
    cloze -> one per ``{{cN::..}}`` number (``:cN``) so each deletion is scheduled on its own.
    """
    out: list[dict] = []
    for cid, c in zip(card_ids(cards), cards):
        t = c.get("type") or "basic"
        base = {"topic": c.get("topic", ""), "type": t, "tags": c.get("tags") or [], "hint": c.get("hint", ""),
                "explanation": c.get("explanation", ""), "starred": bool(c.get("starred")), "parent": cid}
        front, back = c.get("front", ""), c.get("back", "")
        if t == "reversed":
            out.append({**base, "cid": cid + ":a", "front": front, "back": back})
            out.append({**base, "cid": cid + ":b", "front": back, "back": front})
        elif t == "cloze":
            for n in cloze_numbers(front) or [1]:
                out.append({**base, "cid": f"{cid}:c{n}", "front": cloze_render(front, n, False),
                            "back": cloze_render(front, n, True) + (f"\n\n{back}" if back else "")})
        elif t == "mcq":
            out.append({**base, "cid": cid, "front": front, "back": back, "options": c.get("options") or [],
                        "answer_index": c.get("answer_index")})
        else:
            out.append({**base, "cid": cid, "front": front, "back": back})
    return out


def unit_ids(cards: list[dict]) -> list[str]:
    return [u["cid"] for u in units(cards)]


def _d(value: str | None) -> date | None:
    try:
        return date.fromisoformat(value) if value else None
    except ValueError:
        return None


def schedule(state: dict | None, rating: str, today: date) -> dict:
    """The new card state after ``rating`` on ``today`` (pure; ``state`` None = new card)."""
    if rating not in RATINGS:
        raise ValueError(f"unknown rating {rating!r}")
    st = state or {}
    ease = float(st.get("e", DEFAULT_EASE))
    interval = int(st.get("i", 0))
    reps = int(st.get("n", 0))
    lapses = int(st.get("l", 0))
    if rating == "again":
        ease = max(MIN_EASE, ease - 0.20)
        reps, interval, lapses = 0, 0, lapses + 1
    else:
        q = _Q[rating]
        ease = max(MIN_EASE, ease + 0.1 - (5 - q) * (0.08 + (5 - q) * 0.02))
        if reps == 0:
            interval = _FIRST[rating]
        elif reps == 1:
            interval = _SECOND[rating]
        elif rating == "hard":
            interval = max(interval + 1, round(interval * 1.2))
        elif rating == "good":
            interval = max(interval + 1, round(interval * ease))
        else:
            interval = max(interval + 2, round(interval * ease * 1.3))
        reps += 1
    return {"e": round(ease, 3), "i": interval, "n": reps, "l": lapses,
            "d": (today + timedelta(days=interval)).isoformat(), "r": today.isoformat()}


def status(state: dict | None, today: date) -> str:
    """``new`` | ``due`` (review overdue / due today) | ``learning`` (lapsed, due today) | ``later``."""
    if not state:
        return "new"
    due = _d(state.get("d"))
    if due is None or due > today:
        return "later"
    return "learning" if int(state.get("n", 0)) == 0 else "due"


def counts(card_id_list: list[str], srs: dict, today: date) -> dict:
    out = {"total": len(card_id_list), "new": 0, "learning": 0, "due": 0, "later": 0, "mature": 0}
    for cid in card_id_list:
        st = srs.get(cid)
        out[status(st, today)] += 1
        if st and int(st.get("i", 0)) >= MATURE_DAYS:
            out["mature"] += 1
    nxt = [d for d in (_d((srs.get(c) or {}).get("d")) for c in card_id_list) if d is not None and d > today]
    out["next_due"] = min(nxt).isoformat() if nxt else None
    return out


def build_queue(cards: list[dict], srs: dict, today: date, *, new_limit: int = 10, limit: int = 50,
                mode: str = "due", tag: str | None = None) -> list[dict]:
    """Study queue. ``due`` (default): overdue first (most overdue, then lowest ease), lapsed, then up to
    ``new_limit`` new. ``cram``: every unit, shuffled-stable order. ``starred``: starred cards only (due first).
    ``tag`` restricts any mode to cards carrying that tag."""
    due_rows, new_rows, all_rows = [], [], []
    for u in units(cards):
        if tag and tag not in u["tags"]:
            continue
        if mode == "starred" and not u["starred"]:
            continue
        st = srs.get(u["cid"])
        s = status(st, today)
        row = {**u, "state": s, "due": (st or {}).get("d"), "interval": (st or {}).get("i", 0),
               "lapses": (st or {}).get("l", 0)}
        all_rows.append(row)
        if s in ("due", "learning"):
            overdue = (today - (_d(st.get("d")) or today)).days
            due_rows.append((-overdue, float(st.get("e", DEFAULT_EASE)), row))
        elif s == "new":
            new_rows.append(row)
    if mode == "cram":
        queue = sorted(all_rows, key=lambda r: hashlib.sha1((r["cid"] + today.isoformat()).encode()).hexdigest())
    else:
        due_rows.sort(key=lambda t: (t[0], t[1]))
        queue = [r for _, _, r in due_rows] + new_rows[:max(0, new_limit)]
    return queue[:max(1, limit)]
