"""Library service -- a subject-by-subject overview of everything a user has uploaded and done.

NO LLM is involved. Everything is derived from data the app already keeps:

* **history rows** (chats, quizzes, flashcards, ...): the per-owner snapshot ``db.warm_owner`` (the newest
  ``db._SNAP_ROWS`` rows of every kind, ONE aggregation shared with the history lists and the progress
  dashboard). Rows are attributed to a subject by their ``subject`` / ``source`` / ``label`` / ``title``.
* **files + index health + chat message counts / per-answer models**: ONE extra aggregation over ``chats`` +
  the two GridFS ``.files`` collections (metadata only -- no file bytes are read).
* **mastery / weak topics**: ``services.progress.build_summary`` (cached, same snapshot).
* **bundled (standard) subjects**: the local ``documents/`` and shipped ``data/index`` folders (no Mongo).

Cost: at most 1 Mongo op per cold build (+1 if the owner snapshot itself is cold), 0 when cached. The built
payload is cached per owner for ``_TTL_S`` and dropped on the owner's write epoch (``db.epoch``), a changed
subject list or a changed hidden-standard list. The timeline endpoint slices the same cached events, so a page
of the timeline costs 0 extra ops.  Because the snapshot holds the newest 100 rows per kind, counts and the
timeline cover that window (``truncated`` tells the UI when a kind is at the cap).
"""
from __future__ import annotations

import copy
import logging
import re
import threading
import time
from collections import Counter, OrderedDict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from backend import db, documents, vector_store
from backend.services import progress

log = logging.getLogger(__name__)

OTHER_KEY = "__other__"           # "__" is rejected in real subject names, so this can never collide
_TTL_S = 60.0
_CACHE_MAX = 256
MAX_TIMELINE_LIMIT = 50
DEFAULT_TIMELINE_LIMIT = 20
_PARSE_BUDGET = 24                # local PDFs whose page count may be parsed in one build

# history kind -> (label, SPA route, noun for counts)
KIND_INFO: dict[str, tuple[str, str]] = {
    "chats": ("Chat", "#chat"),
    "quizzes": ("Quiz", "#quiz"),
    "attempts": ("Attempt", "#progress"),
    "flashcards": ("Flashcards", "#flashcards"),
    "exams": ("Exam", "#exam"),
    "cheatsheets": ("Cheat sheet", "#cheatsheet"),
    "mindmaps": ("Mind map", "#mindmap"),
    "plans": ("Study plan", "#planner"),
    "summaries": ("Summaries", "#planner"),
    "predictions": ("Prediction", "#predictor"),
    "teachbacks": ("Teach-back", "#teachback"),
    "scans": ("Scan notes", "#scannotes"),
    "uploads": ("Upload", "#subjects"),
}
EVENT_KINDS: tuple[str, ...] = tuple(KIND_INFO)
# kinds that are opened through db.get_item by the SPA (everything except attempts / uploads)
_OPENABLE = {k for k in KIND_INFO if k not in ("attempts", "uploads")}


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _dt(value) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str) and value:
        try:
            d = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    return None


def _iso(d: datetime | None) -> str | None:
    return d.astimezone(timezone.utc).isoformat().replace("+00:00", "Z") if d else None


def _num(v):
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _llm_pub(llm) -> dict | None:
    """The public, compact model block of a stored ``llm`` (None when no real model answered)."""
    if not isinstance(llm, dict) or not llm.get("provider") or not llm.get("model"):
        return None
    return {"provider": str(llm["provider"]), "model": str(llm["model"]), "step": _num(llm.get("step")) or 0,
            "label": llm.get("label") if isinstance(llm.get("label"), str) else None,
            "skipped": [str(s)[:80] for s in (llm.get("skipped") or [])[:6]] if isinstance(llm.get("skipped"), list) else []}


_PREFIX = re.compile(r"^[^·:/]{1,24}\s*[·:]\s*")


def _text_matches(text, names: list[tuple[str, str]]) -> str | None:
    """The subject whose name ``text`` starts with (``"Maths / Lec-1.pdf"``, ``"Subject · Maths"``), else None.
    ``names`` is ``[(casefolded, display)]`` longest first."""
    if not isinstance(text, str) or not text.strip():
        return None
    t = text.strip()
    for cand in (t, _PREFIX.sub("", t, count=1)):
        low = cand.casefold()
        for cf, disp in names:
            if not low.startswith(cf):
                continue
            rest = low[len(cf):].lstrip()
            if rest == "" or rest[0] in "/·:-,(":
                return disp
    return None


def attribute(row: dict, names: list[tuple[str, str]]) -> str | None:
    """Which known subject a stored history row belongs to (None = not tied to one)."""
    for field in ("subject", "source", "label", "title"):
        hit = _text_matches(row.get(field), names)
        if hit:
            return hit
    return None


# ---------------------------------------------------------------------------
# files (bundled: local disk + shipped manifest; user: GridFS metadata)
# ---------------------------------------------------------------------------
_pages_memo: dict[tuple, int] = {}


def _pages_of(path: Path, budget: list[int]) -> int | None:
    try:
        st = path.stat()
        ck = (str(path), st.st_mtime_ns, st.st_size)
    except OSError:
        return None
    hit = _pages_memo.get(ck)
    if hit is not None:
        return hit
    if budget[0] <= 0:
        return None
    budget[0] -= 1
    try:
        n = len(documents._pdf_reader(str(path)).pages)
    except Exception:  # noqa: BLE001 - a broken PDF just has unknown pages
        return None
    if len(_pages_memo) > 2048:
        _pages_memo.clear()
    _pages_memo[ck] = n
    return n


def _local_dirs(name: str, owner: str, bundled: bool) -> list[Path]:
    """Existing local folders for a subject (never hydrates from Mongo)."""
    if bundled:
        return [documents.DOCUMENTS_DIR / name]
    key = documents._user_subject_key(owner, name)
    return [r / key for r in (documents.DOCUMENTS_DIR, documents._user_root()) if (r / key).is_dir()]


def _is_bundled(name: str) -> bool:
    try:
        return (documents.DOCUMENTS_DIR / documents._public_subject(name)).is_dir()
    except Exception:  # noqa: BLE001
        return False


def _bundled_index(name: str) -> dict | None:
    try:
        return vector_store._read_manifest_memo(vector_store._bundled_dir(name))
    except Exception:  # noqa: BLE001
        return None


def _text_only_files(name: str, owner: str) -> list[dict]:
    """Large PDFs whose original was dropped and only the extracted text index remains (local state only)."""
    try:
        d = vector_store._user_cache_dir(documents._user_subject_key(owner, name))
        files = vector_store.load_texts(d).get("files", {}) if d.is_dir() else {}
    except Exception:  # noqa: BLE001
        return []
    out = []
    for fname, ent in files.items():
        if isinstance(ent, dict) and not ent.get("raw", True):
            pages = ent.get("total") or len(ent.get("pages") or []) or None
            out.append({"name": fname, "pages": pages, "bytes": _num(ent.get("size")), "added": None,
                        "text_only": True})
    return out


def _file_entries(name: str, owner: str, bundled: bool, remote: list[dict], budget: list[int]) -> list[dict]:
    files: dict[str, dict] = {}
    for d in _local_dirs(name, owner, bundled):
        try:
            for p in sorted(d.iterdir()):
                if p.is_file() and p.suffix.lower() == ".pdf":
                    st = p.stat()
                    files[p.name] = {"name": p.name, "pages": _pages_of(p, budget), "bytes": st.st_size,
                                     "added": None if bundled else _iso(datetime.fromtimestamp(st.st_mtime, timezone.utc)),
                                     "text_only": False}
        except OSError:
            continue
    for r in remote:                                   # GridFS metadata is authoritative for user files
        fname = r.get("filename")
        if not fname:
            continue
        cur = files.get(fname, {"name": fname, "pages": None, "text_only": False})
        cur["bytes"] = _num(r.get("length")) if _num(r.get("length")) is not None else cur.get("bytes")
        cur["added"] = _iso(_dt(r.get("uploaded_at") or r.get("uploadDate"))) or cur.get("added")
        files[fname] = cur
    for t in _text_only_files(name, owner) if not bundled else []:
        files.setdefault(t["name"], t)
    return sorted(files.values(), key=lambda f: f["name"].casefold())


def _index_state(name: str, bundled: bool, ix_rows: list[dict]) -> dict:
    if bundled:
        m = _bundled_index(name) or {}
        n_chunks, n_emb, complete, model = m.get("n_chunks"), m.get("n_embedded"), m.get("complete"), m.get("model")
        known = bool(m)
    else:
        newest = max(ix_rows, key=lambda r: _dt(r.get("saved_at") or r.get("uploadDate")) or datetime.min.replace(tzinfo=timezone.utc),
                     default=None)
        known = newest is not None
        n_chunks, n_emb = (newest or {}).get("m_n_chunks"), (newest or {}).get("m_n_embedded")
        complete, model = (newest or {}).get("m_complete"), (newest or {}).get("m_model")
    n_chunks, n_emb = int(n_chunks or 0), int(n_emb or 0)
    state = "unknown" if not known else ("complete" if complete else ("partial" if n_chunks else "none"))
    return {"state": state, "n_chunks": n_chunks, "n_embedded": n_emb if state != "complete" else n_chunks or n_emb,
            "model": model if isinstance(model, str) else None}


# ---------------------------------------------------------------------------
# the one extra Mongo read
# ---------------------------------------------------------------------------
def _extra_pipeline(owner: str, chat_limit: int) -> list[dict]:
    msgs = {"$ifNull": ["$messages", []]}
    return [
        {"$match": {"owner": owner}}, {"$sort": {"created_at": -1}}, {"$limit": chat_limit},
        {"$project": {"_k": {"$literal": "chat"}, "n": {"$size": msgs},
                      "llms": {"$map": {"input": {"$filter": {"input": msgs, "cond": {"$eq": ["$$this.role", "assistant"]}}},
                                        "in": {"provider": "$$this.llm.provider", "model": "$$this.llm.model",
                                               "step": "$$this.llm.step", "skipped": "$$this.llm.skipped"}}}}},
        {"$unionWith": {"coll": db.UPLOADS_BUCKET + ".files", "pipeline": [
            {"$match": {"owner": owner}},
            {"$project": {"_k": {"$literal": "up"}, "filename": 1, "subject": 1, "length": 1, "uploadDate": 1, "uploaded_at": 1}}]}},
        {"$unionWith": {"coll": db.INDEX_BUCKET + ".files", "pipeline": [
            {"$match": {"owner": owner}},
            {"$project": {"_k": {"$literal": "ix"}, "subject": 1, "length": 1, "uploadDate": 1, "saved_at": 1,
                          "m_n_chunks": 1, "m_n_embedded": 1, "m_complete": 1, "m_model": 1}}]}},
    ]


def _load_extra(owner: str) -> dict:
    out = {"chats": {}, "uploads": {}, "index": {}, "ok": False}
    database = db._get_db()
    if database is None:
        return out
    try:
        for row in database["chats"].aggregate(_extra_pipeline(owner, db._SNAP_ROWS), allowDiskUse=False):
            k = row.pop("_k", None)
            if k == "chat":
                out["chats"][str(row.get("_id"))] = row
            elif k == "up" and row.get("subject"):
                out["uploads"].setdefault(row["subject"], []).append(row)
            elif k == "ix" and row.get("subject"):
                out["index"].setdefault(row["subject"], []).append(row)
        out["ok"] = True
    except Exception as exc:  # noqa: BLE001
        if type(exc).__module__.split(".")[0] not in ("pymongo", "bson"):
            raise
        log.warning("Library: could not read file metadata: %s", str(exc)[:160])
    return out


# ---------------------------------------------------------------------------
# events
# ---------------------------------------------------------------------------
def _pct(a, b) -> int | None:
    a, b = _num(a), _num(b)
    return round(100 * a / b) if a is not None and b else None


def _event(kind: str, row: dict, chat_extra: dict | None) -> dict | None:
    rid = str(row.get("_id") or "")
    created = _dt(row.get("created_at"))
    ts = _dt(row.get("updated_at")) if kind == "chats" else created
    ts = ts or created
    if ts is None:
        return None
    label = KIND_INFO[kind][0]
    llm = _llm_pub(row.get("llm"))
    title = str(row.get("title") or "").strip()[:140]
    if kind == "chats":
        n = int((chat_extra or {}).get("n") or 0)
        what = f"Chat · {n} message{'s' if n != 1 else ''}" if n else "Chat"
        llms = [x for x in ((chat_extra or {}).get("llms") or []) if isinstance(x, dict) and x.get("provider") and x.get("model")]
        llm = _llm_pub(llms[-1]) if llms else None
    elif kind == "quizzes":
        n = _num(row.get("num_questions"))
        what = f"Quiz · {int(n)} questions" if n else "Quiz generated"
    elif kind == "attempts":
        pct = row.get("percent") if _num(row.get("percent")) is not None else _pct(row.get("correct"), row.get("total"))
        tool = str(row.get("tool") or row.get("kind") or "quiz")
        who = "Flashcards reviewed" if tool.startswith("flash") else ("Exam attempt" if tool.startswith("exam") else "Quiz attempt")
        c, t = _num(row.get("correct")), _num(row.get("total"))
        what = f"{who} · {int(c)}/{int(t)}" + (f" · {round(pct)}%" if pct is not None else "") if c is not None and t else who
    elif kind == "flashcards":
        n = _num(row.get("num_cards"))
        what = f"Flashcards · {int(n)} cards" if n else "Flashcard deck"
    elif kind == "exams":
        pct = _num(row.get("percent"))
        if pct is None:
            pct = _pct(row.get("total_score"), row.get("max_score"))
        what = f"Exam graded · {round(pct)}%" if pct is not None else "Exam graded"
    elif kind == "teachbacks":
        sc = (row.get("evaluation") or {}).get("understanding_score_0_to_10") if isinstance(row.get("evaluation"), dict) else None
        what = f"Teach-back · {sc}/10" if _num(sc) is not None else "Teach-back session"
    elif kind == "cheatsheets":
        what = "Cheat sheet generated"
    elif kind == "mindmaps":
        what = "Mind map generated"
    elif kind == "plans":
        what = "Study plan created"
    elif kind == "summaries":
        what = "Summaries generated"
    elif kind == "predictions":
        what = "Question prediction"
    elif kind == "scans":
        what = "Notes scanned"
    else:
        what = label
    return {"id": f"{kind}:{rid}", "kind": kind, "kind_label": label, "title": title or label, "summary": what,
            "ts": _iso(ts), "item_id": rid if kind in _OPENABLE and rid else None, "route": KIND_INFO[kind][1],
            "llm": llm, "_dt": ts}


def _upload_event(subject: str, r: dict, pages) -> dict | None:
    ts = _dt(r.get("uploaded_at") or r.get("uploadDate"))
    fname = r.get("filename")
    if ts is None or not fname:
        return None
    return {"id": f"uploads:{subject}:{fname}", "kind": "uploads", "kind_label": "Upload",
            "title": fname, "summary": f"Uploaded {fname}" + (f" · {pages} pages" if pages else ""),
            "ts": _iso(ts), "item_id": None, "route": "#subjects", "llm": None, "_dt": ts}


def _model_rows(pairs: list[dict]) -> list[dict]:
    """``[{provider, model, step, count}]`` (most used first) from model blocks."""
    c: Counter = Counter()
    for m in pairs:
        c[(m["provider"], m["model"], int(m.get("step") or 0))] += 1
    return [{"provider": p, "model": m, "step": s, "count": n}
            for (p, m, s), n in sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))]


def _chain(models: list[dict]) -> dict:
    """Fall-through summary for a list of model blocks (answers, how many came from step >= 2, why)."""
    total = len(models)
    later = sum(1 for m in models if int(m.get("step") or 0) >= 2)
    why: Counter = Counter()
    for m in models:
        for s in m.get("skipped") or []:
            parts = str(s).split()
            why[parts[-1] if parts else "error"] += 1
    return {"answers": total, "step1": total - later, "later_steps": later,
            "reasons": [{"reason": r, "count": n} for r, n in why.most_common(6)]}


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------
def _all_rows(snap: dict | None) -> list[tuple[str, dict]]:
    if not snap:
        return []
    out = []
    for kind in KIND_INFO:
        slot = snap.get(kind)
        if slot:
            out.extend((kind, r) for r in slot.get("rows") or [])
    return out


def _mastery_for(owner: str) -> dict[str, dict]:
    try:
        summ = progress.build_summary(owner)
    except Exception:  # noqa: BLE001
        return {}
    out = {}
    for s in summ.get("subjects") or []:
        topics = s.get("topics") or []
        wsum = sum(max(1, t.get("questions") or 1) for t in topics)
        m = (sum((t.get("mastery") or 0) * max(1, t.get("questions") or 1) for t in topics) / wsum) if wsum else None
        weak = sorted((t for t in topics if (t.get("attempts") or 0) >= 1 and t.get("mastery") is not None),
                      key=lambda t: (t["mastery"], -(t.get("attempts") or 0)))[:2]
        out[str(s.get("subject", "")).casefold()] = {
            "pct": round(m * 100) if m is not None else None, "attempts": int(s.get("attempts") or 0),
            "weak": [{"topic": t["topic"], "mastery": round(t["mastery"] * 100), "attempts": t.get("attempts", 0)} for t in weak
                     if t["mastery"] < 0.75]}
    return out


def _build(owner: str) -> dict:
    visible = documents.list_subjects(owner)
    hidden = []
    try:
        from backend import prefs
        hidden = prefs.hidden_standard(owner)
    except Exception:  # noqa: BLE001
        pass
    snap = None
    try:
        snap = db.warm_owner(owner)
    except Exception:  # noqa: BLE001
        pass
    extra = _load_extra(owner)
    mastery = _mastery_for(owner)
    bundled_names = {s.casefold() for s in visible if _is_bundled(s)}
    names = sorted(((s.casefold(), s) for s in [*visible, *hidden]), key=lambda x: -len(x[0]))
    hidden_cf = {h.casefold() for h in hidden}

    events: dict[str, list[dict]] = {s: [] for s in visible}
    events[OTHER_KEY] = []
    models: dict[str, list[dict]] = {k: [] for k in events}
    kind_counts: dict[str, Counter] = {k: Counter() for k in events}
    truncated = False
    chat_msgs: Counter = Counter()
    card_total: Counter = Counter()
    for kind, row in _all_rows(snap):
        subj = attribute(row, names)
        if subj is not None and subj not in events and subj.casefold() in hidden_cf:
            continue                                  # belongs to a hidden standard subject: not shown
        key = subj if subj in events else OTHER_KEY
        ce = extra["chats"].get(str(row.get("_id"))) if kind == "chats" else None
        ev = _event(kind, row, ce)
        if ev is None:
            continue
        events[key].append(ev)
        kind_counts[key][kind] += 1
        if kind == "chats":
            chat_msgs[key] += int((ce or {}).get("n") or 0)
            for x in (ce or {}).get("llms") or []:
                m = _llm_pub(x)
                if m:
                    models[key].append(m)
        else:
            if ev["llm"]:
                models[key].append(ev["llm"])
            if kind == "flashcards":
                card_total[key] += int(_num(row.get("num_cards")) or 0)
    if snap:
        truncated = any((snap.get(k) or {}).get("n", 0) >= db._SNAP_ROWS for k in KIND_INFO if k in snap)

    cards = []
    total_files = total_pages = total_bytes = total_chunks = 0
    budget = [_PARSE_BUDGET]
    for name in visible:
        bundled = name.casefold() in bundled_names
        remote = extra["uploads"].get(name) or []
        ix_rows = extra["index"].get(name) or []
        files = _file_entries(name, owner, bundled, remote, budget)
        for r in remote:
            f = next((x for x in files if x["name"] == r.get("filename")), None)
            ev = _upload_event(name, r, f.get("pages") if f else None)
            if ev:
                events[name].append(ev)
                kind_counts[name]["uploads"] += 1
        index = _index_state(name, bundled, ix_rows)
        pages = sum(f.get("pages") or 0 for f in files)
        size = sum(f.get("bytes") or 0 for f in files)
        if not bundled:                                # stored bytes = originals + the index blob
            size = sum(int(_num(r.get("length")) or 0) for r in remote + ix_rows) or size
        evs = sorted(events[name], key=lambda e: (e["_dt"], e["id"]), reverse=True)
        events[name] = evs
        c = kind_counts[name]
        card = {
            "name": name, "key": name, "own": not bundled, "files": files,
            "totals": {"files": len(files), "pages": pages, "chunks": index["n_chunks"], "bytes": size},
            "index": index,
            "last_activity": evs[0]["ts"] if evs else None,
            "mastery": mastery.get(name.casefold()) or {"pct": None, "attempts": 0, "weak": []},
            "counts": {"chats": c["chats"], "chat_messages": chat_msgs[name], "quizzes": c["quizzes"],
                       "flashcard_decks": c["flashcards"], "flashcard_cards": card_total[name], "exams": c["exams"],
                       "cheatsheets": c["cheatsheets"], "mindmaps": c["mindmaps"], "plans": c["plans"] + c["summaries"],
                       "teachbacks": c["teachbacks"], "scans": c["scans"], "predictions": c["predictions"],
                       "attempts": c["attempts"], "uploads": c["uploads"]},
            "models": _model_rows(models[name])[:4],
        }
        cards.append(card)
        total_files += len(files)
        total_pages += pages
        total_bytes += size
        total_chunks += index["n_chunks"]
    cards.sort(key=lambda x: x["name"].casefold())
    cards.sort(key=lambda x: x["last_activity"] or "", reverse=True)          # newest activity first, idle subjects last (A-Z)
    other = sorted(events[OTHER_KEY], key=lambda e: (e["_dt"], e["id"]), reverse=True)
    events[OTHER_KEY] = other

    all_models = [m for ms in models.values() for m in ms]
    by_provider = Counter(m["provider"] for m in all_models)
    tot = sum(by_provider.values()) or 1
    by_kind = Counter()
    for kc in kind_counts.values():
        by_kind.update(kc)
    summary = {
        "subjects": len(cards), "files": total_files, "pages": total_pages, "chunks": total_chunks,
        "activity_by_kind": {k: by_kind[k] for k in EVENT_KINDS if by_kind[k]},
        "activity_total": sum(by_kind.values()),
        "models": _model_rows(all_models), "providers": [{"provider": p, "count": n, "pct": round(100 * n / tot)}
                                                          for p, n in by_provider.most_common()],
        "chain": _chain(all_models),
        "storage": {"owner_mb": round(sum(c["totals"]["bytes"] for c in cards if c["own"]) / 1048576, 1)},
        "hidden": {"count": len(hidden), "names": list(hidden)},
        "truncated": truncated,
    }
    other_card = None
    if other:
        oc = kind_counts[OTHER_KEY]
        other_card = {"name": "Not tied to a subject", "key": OTHER_KEY, "last_activity": other[0]["ts"], "counts": dict(oc)}
    return {"cards": cards, "other": other_card, "summary": summary, "events": events, "models": models,
            "persistence": db.is_enabled(), "extra_ok": extra["ok"]}


# ---------------------------------------------------------------------------
# cache + public API
# ---------------------------------------------------------------------------
_cache: "OrderedDict[str, tuple[float, tuple, dict]]" = OrderedDict()
_lock = threading.Lock()


def clear_cache() -> None:
    with _lock:
        _cache.clear()


def _token(owner: str) -> tuple:
    hidden: tuple = ()
    try:
        from backend import prefs
        hidden = tuple(prefs.hidden_standard(owner))
    except Exception:  # noqa: BLE001
        pass
    return (db.epoch(owner), tuple(documents.list_subjects(owner)), hidden)


def get(owner: str, *, refresh: bool = False) -> dict:
    """The owner's built library (cached ``_TTL_S``). Treat as read-only."""
    from backend import cache as shared

    tok = _token(owner)
    now = time.monotonic()
    if not refresh:
        with _lock:
            hit = _cache.get(owner)
            if hit and hit[0] > now and hit[1] == tok:
                _cache.move_to_end(owner)
                return hit[2]
    with shared.single_flight(("library", owner)):
        if not refresh:
            with _lock:
                hit = _cache.get(owner)
                if hit and hit[0] > time.monotonic() and hit[1] == tok:
                    return hit[2]
        data = _build(owner)
        with _lock:
            _cache[owner] = (time.monotonic() + _TTL_S, tok, data)
            _cache.move_to_end(owner)
            while len(_cache) > _CACHE_MAX:
                _cache.popitem(last=False)
    return data


def overview(owner: str, *, refresh: bool = False) -> dict:
    d = get(owner, refresh=refresh)
    return copy.deepcopy({"summary": d["summary"], "subjects": d["cards"], "other": d["other"],
                          "persistence": d["persistence"], "generated_at": _iso(datetime.now(timezone.utc))})


class LibraryError(Exception):
    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


def _cursor(ev: dict) -> str:
    return f"{int(ev['_dt'].timestamp() * 1000)}~{ev['id']}"


def _parse_cursor(before: str) -> tuple[int, str]:
    ms, sep, ident = before.partition("~")
    try:
        return int(ms), ident if sep else ""
    except ValueError:
        raise LibraryError("Invalid cursor.", 400) from None


def timeline(owner: str, subject: str, kind: str | None = None, before: str | None = None,
             limit: int = DEFAULT_TIMELINE_LIMIT, *, refresh: bool = False) -> dict:
    d = get(owner, refresh=refresh)
    key = OTHER_KEY if subject == OTHER_KEY else next((s for s in d["events"] if s != OTHER_KEY and s.casefold() == subject.casefold()), None)
    if key is None:
        raise LibraryError("Subject not found.", 404)
    if kind and kind not in KIND_INFO:
        raise LibraryError(f"Unknown kind: {kind}", 400)
    limit = max(1, min(int(limit or DEFAULT_TIMELINE_LIMIT), MAX_TIMELINE_LIMIT))
    evs = d["events"][key]
    counts = Counter(e["kind"] for e in evs)
    if kind:
        evs = [e for e in evs if e["kind"] == kind]
    if before:
        ms, ident = _parse_cursor(before)
        evs = [e for e in evs if (int(e["_dt"].timestamp() * 1000), e["id"]) < (ms, ident)]
    page = evs[:limit]
    has_more = len(evs) > limit
    items = [{k: v for k, v in e.items() if k != "_dt"} for e in page]
    now = datetime.now(timezone.utc)
    week = {"this_week": 0, "last_week": 0, "this_week_ai": 0, "last_week_ai": 0}
    for e in d["events"][key]:
        age = now - e["_dt"]
        w = "this_week" if age <= timedelta(days=7) else ("last_week" if age <= timedelta(days=14) else None)
        if w:
            week[w] += 1
            if e["llm"]:
                week[w + "_ai"] += 1
    ms = d["models"][key]
    return {"subject": key if key != OTHER_KEY else OTHER_KEY, "items": items,
            "next_cursor": _cursor(page[-1]) if has_more and page else None, "has_more": has_more,
            "counts": dict(counts), "models": _model_rows(ms), "chain": _chain(ms), "week": week,
            "truncated": d["summary"]["truncated"]}
