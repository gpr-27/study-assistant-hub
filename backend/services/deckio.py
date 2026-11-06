"""Flashcard deck content helpers: card validation (5 card types), bulk / CSV import, CSV / Anki-TSV / JSON export,
duplicate detection and "related cards" (lexical, plus embeddings when allowed). Pure functions - the router does I/O.

Card types: ``basic`` (front/back), ``reversed`` (two-way: scheduled as two units), ``cloze`` (``{{c1::text::hint}}``
in ``front``; one scheduled unit per deletion number), ``mcq`` (``options`` + ``answer_index`` - graded in code) and
``typed`` (the student types the answer; normalised match first, rubric LLM fallback via the grading core).
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import threading
import uuid
from collections import OrderedDict

from backend import config, privacy
from backend.services import grading, srs

CARD_TYPES = ("basic", "reversed", "cloze", "mcq", "typed")
MAX_CARDS_PER_DECK = 5000
MAX_DECKS_PER_OWNER = 200
MAX_IMPORT = 1000
MAX_FRONT = 4000
MAX_BACK = 8000
DUP_JACCARD = 0.8
DUP_COSINE = 0.92
_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{4,24}$")


def _s(v, n) -> str:
    return re.sub(r"[ \t]+", " ", str(v)).strip()[:n] if isinstance(v, (str, int, float)) and not isinstance(v, bool) else ""


def new_id() -> str:
    return uuid.uuid4().hex[:10]


def normalise_card(raw) -> dict | None:
    """A clean card dict (with an ``id``) or None when unusable."""
    if not isinstance(raw, dict):
        return None
    t = str(raw.get("type") or "basic").lower()
    if t not in CARD_TYPES:
        return None
    front, back = _s(raw.get("front"), MAX_FRONT), _s(raw.get("back"), MAX_BACK)
    if not front:
        return None
    card = {"id": raw["id"] if isinstance(raw.get("id"), str) and _ID_RE.match(raw["id"]) else new_id(),
            "type": t, "front": front, "back": back, "topic": _s(raw.get("topic"), 80),
            "tags": sorted({x for x in (_s(g, 30).lower() for g in (raw.get("tags") or []) if isinstance(g, str)) if x})[:10],
            "hint": _s(raw.get("hint"), 500), "explanation": _s(raw.get("explanation"), 1500),
            "starred": bool(raw.get("starred"))}
    if t == "cloze":
        if not srs.cloze_numbers(front):
            return None
    elif t == "mcq":
        opts = [_s(o, 400) for o in (raw.get("options") or []) if _s(o, 400)] if isinstance(raw.get("options"), list) else []
        idx = raw.get("answer_index")
        if not 2 <= len(opts) <= 6 or not isinstance(idx, int) or isinstance(idx, bool) or not 0 <= idx < len(opts):
            return None
        card.update(options=opts, answer_index=idx, back=back or opts[idx])
    elif not back:
        return None                     # basic / reversed / typed need an answer
    return card


# ---------------------------------------------------------------------------------------------
# Import / export
# ---------------------------------------------------------------------------------------------
def parse_bulk(text: str) -> list[dict]:
    """``front :: back`` per line, or ``Q: ...`` / ``A: ...`` blocks. Lines containing ``{{c1::`` become cloze cards."""
    out: list[dict] = []
    q = None
    for line in (text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^(?:q|question)\s*[:.)]\s*(.+)$", line, re.I)
        a = re.match(r"^(?:a|answer)\s*[:.)]\s*(.+)$", line, re.I)
        if m:
            q = m.group(1)
        elif a and q is not None:
            out.append({"front": q, "back": a.group(1)})
            q = None
        elif "{{c" in line:
            out.append({"type": "cloze", "front": line, "back": ""})
        elif "::" in line:
            f, _, b = line.partition("::")
            out.append({"front": f.strip(), "back": b.strip()})
        elif "\t" in line:
            f, _, b = line.partition("\t")
            out.append({"front": f.strip(), "back": b.strip()})
    return out


_HEAD = ("front", "back", "tags", "type", "hint", "explanation", "topic")


def parse_csv(text: str) -> list[dict]:
    """CSV (or Anki TSV) with columns front, back[, tags, type, hint, explanation, topic]; a header row is optional."""
    sample = (text or "")[:2000]
    delim = "\t" if sample.count("\t") > sample.count(",") else ","
    rows = [r for r in csv.reader(io.StringIO(text or ""), delimiter=delim) if any(c.strip() for c in r)]
    if rows and [c.strip().lower() for c in rows[0][:2]] == ["front", "back"]:
        cols = [c.strip().lower() for c in rows[0]]
        rows = rows[1:]
    else:
        cols = list(_HEAD)
    out = []
    for r in rows:
        d = {cols[i]: v for i, v in enumerate(r) if i < len(cols)}
        d["tags"] = re.split(r"[\s,;]+", d.get("tags", "").strip()) if d.get("tags") else []
        if not d.get("type") and "{{c" in d.get("front", ""):
            d["type"] = "cloze"
        out.append(d)
    return out


def export_csv(cards: list[dict]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(_HEAD)
    for c in cards:
        w.writerow([c.get("front", ""), c.get("back", ""), " ".join(c.get("tags") or []), c.get("type", "basic"),
                    c.get("hint", ""), c.get("explanation", ""), c.get("topic", "")])
    return buf.getvalue()


def export_anki_tsv(cards: list[dict]) -> str:
    """Anki "Import file" format: tab separated ``front<TAB>back<TAB>tags`` (cloze uses Anki's own ``{{c1::..}}``)."""
    rows = []
    for c in cards:
        clean = lambda x: str(x).replace("\t", " ").replace("\r", "").replace("\n", "<br>")   # noqa: E731
        back = c.get("back", "")
        if c.get("type") == "mcq":
            back = "<br>".join(f"{chr(65 + i)}. {o}" for i, o in enumerate(c.get("options") or [])) + f"<br><br>{back}"
        rows.append("\t".join([clean(c.get("front", "")), clean(back), " ".join(c.get("tags") or [])]))
    return "\n".join(rows) + ("\n" if rows else "")


def export_json(cards: list[dict]) -> str:
    return json.dumps({"cards": cards}, ensure_ascii=False, indent=1)


# ---------------------------------------------------------------------------------------------
# Duplicates / related
# ---------------------------------------------------------------------------------------------
def _key(card) -> str:
    return grading.norm_text(re.sub(r"\{\{c\d+::(.*?)(?:::.*?)?\}\}", r"\1", card.get("front", "")))


def _jaccard(a: str, b: str) -> float:
    x, y = grading._stems(a), grading._stems(b)
    return len(x & y) / len(x | y) if (x | y) else 0.0


_emb_cache: "OrderedDict[str, list]" = OrderedDict()
_emb_lock = threading.Lock()


def _embeddings(texts: list[str]):
    """Vectors for ``texts`` via ONE batched Gemini call for the cache misses (text-hash cache). None when embeddings
    are unavailable or not allowed for user content (privacy policy) - callers fall back to lexical matching."""
    if privacy.policy_active():
        return None
    try:
        from backend import gemini_embeddings as ge
        if not ge.have_keys():
            return None
        hashes = [hashlib.sha1(t.encode()).hexdigest() for t in texts]
        with _emb_lock:
            miss = [i for i, h in enumerate(hashes) if h not in _emb_cache]
        if miss:
            mat = ge.embed_documents([texts[i] for i in miss], deadline_s=20.0, patient=False)
            with _emb_lock:
                for i, vec in zip(miss, mat):
                    _emb_cache[hashes[i]] = vec
                while len(_emb_cache) > 4000:
                    _emb_cache.popitem(last=False)
        with _emb_lock:
            return [_emb_cache[h] for h in hashes]
    except Exception:  # noqa: BLE001 - semantic features are best effort
        return None


def find_duplicates(new: list[dict], existing: list[dict], *, semantic: bool = True) -> list[dict]:
    """``[{index, match_id, match_front, how}]`` for each new card that duplicates an existing / earlier-new one."""
    pool = list(existing)
    keys = {_key(c): c for c in pool}
    out: list[dict] = []
    vecs = None
    if semantic and new and pool:
        vecs = _embeddings([_key(c) for c in new] + [_key(c) for c in pool[:1000]])
    for i, c in enumerate(new):
        k = _key(c)
        hit, how = keys.get(k), "exact"
        if hit is None:
            for p in pool:
                if _jaccard(k, _key(p)) >= DUP_JACCARD:
                    hit, how = p, "similar"
                    break
        if hit is None and vecs is not None:
            for j, p in enumerate(pool[:1000]):
                v, w = vecs[i], vecs[len(new) + j]
                if float(sum(a * b for a, b in zip(v, w))) >= DUP_COSINE:
                    hit, how = p, "semantic"
                    break
        if hit is not None:
            out.append({"index": i, "match_id": hit.get("id"), "match_front": hit.get("front"), "how": how})
        else:
            pool.append(c)
            keys[k] = c
    return out


def related(card: dict, cards: list[dict], k: int = 5) -> tuple[list[dict], str]:
    """The ``k`` cards most related to ``card`` -> (cards, method). Embeddings when allowed, else lexical overlap."""
    others = [c for c in cards if c.get("id") != card.get("id")][:1000]
    vecs = _embeddings([_key(card)] + [_key(c) for c in others]) if others else None
    if vecs is not None:
        scored = sorted(((float(sum(a * b for a, b in zip(vecs[0], vecs[i + 1]))), c) for i, c in enumerate(others)),
                        key=lambda t: -t[0])
        return [c for s, c in scored[:k] if s > 0.5], "semantic"
    scored = sorted(((_jaccard(_key(card) + " " + card.get("back", ""), _key(c) + " " + c.get("back", "")), c)
                     for c in others), key=lambda t: -t[0])
    return [c for s, c in scored[:k] if s > 0], "lexical"
