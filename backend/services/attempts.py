"""Graded attempts: persistence, idempotency and history reads.

Collections: ``attempts`` for quiz / practice / flashcards / teach-back results and ``exams`` for mock-exam reports
(the exam history drawer and the dashboard's exam metrics already read that collection, so an exam is ONE write
there, with the unified fields added next to the legacy report fields).

One document per attempt, written ONCE (one Mongo write per submission; the per-owner TTL caches in
:mod:`backend.services.progress` / :mod:`backend.db` are invalidated by ``db.save``'s epoch bump):

    {owner, attempt_id, kind, subject, topic, title, tool,
     correct, total, percent, grade_band,        # aggregate fields the dashboard reads
     topics: [{topic, earned, possible, questions}], weak_topics, verified, graded, provisional,
     duration_seconds, llm, result: <full unified result>, created_at}

**Idempotent submissions.** The client mints an ``attempt_id`` (uuid) when the paper is opened and re-sends it on
retry / double-click / refresh-and-resubmit. :func:`run_idempotent` serialises concurrent calls for the same
``(owner, attempt_id)`` in this process, replays the stored result instead of grading (and charging the LLM)
again, and a partial unique index ``(owner, attempt_id)`` closes the cross-instance race. Everything is
owner-scoped: another user's attempt id is simply unknown. A *provisional* result (AI grader was down) can be
re-graded in place with ``regrade=True`` - the document is replaced, never duplicated.

With persistence off, results live in a small bounded in-process LRU so double-submits are still harmless in
single-process dev.
"""
from __future__ import annotations

import logging
import re
import threading
from collections import OrderedDict
from typing import Callable

from backend import db

log = logging.getLogger(__name__)

KIND = "attempts"
EXAM_KIND = "exams"
_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{8,64}$")
_MEM_MAX = 300
_mem: "OrderedDict[tuple[str, str], dict]" = OrderedDict()
_mem_lock = threading.Lock()
_locks: dict[tuple[str, str], list] = {}
_locks_guard = threading.Lock()
_index_ready: set[str] = set()


def collection_for(kind: str) -> str:
    return EXAM_KIND if kind == "exam" else KIND


def valid_attempt_id(value: object) -> bool:
    return isinstance(value, str) and bool(_ID_RE.match(value))


# ---------------------------------------------------------------------------------------------
def _coll(kind: str = "quiz"):
    name = collection_for(kind)
    database = db.get_db()
    if database is None:
        return None
    coll = database[name]
    if name not in _index_ready:
        try:
            coll.create_index([("owner", 1), ("attempt_id", 1)], unique=True, background=True,
                              partialFilterExpression={"attempt_id": {"$type": "string"}})
            _index_ready.add(name)
        except Exception:  # noqa: BLE001 - best effort; the in-process lock + pre-check still dedupe
            log.debug("attempts unique index not created.", exc_info=True)
    return coll


def _mem_get(owner: str, attempt_id: str) -> dict | None:
    with _mem_lock:
        hit = _mem.get((owner, attempt_id))
        if hit is not None:
            _mem.move_to_end((owner, attempt_id))
        return hit


def _mem_put(owner: str, attempt_id: str, result: dict) -> None:
    with _mem_lock:
        _mem[(owner, attempt_id)] = result
        _mem.move_to_end((owner, attempt_id))
        while len(_mem) > _MEM_MAX:
            _mem.popitem(last=False)


def clear_memory() -> None:
    with _mem_lock:
        _mem.clear()
    _index_ready.clear()


def _find_db(owner: str, attempt_id: str, kind: str) -> dict | None:
    coll = _coll(kind)
    doc = coll.find_one({"owner": owner, "attempt_id": attempt_id}, {"result": 1}) if coll is not None else None
    res = (doc or {}).get("result")
    return {**res, "id": str(doc["_id"])} if isinstance(res, dict) else None


def find(owner: str, attempt_id: str, kind: str = "quiz") -> dict | None:
    """The stored unified result for this owner's attempt, or None."""
    hit = _mem_get(owner, attempt_id)
    if hit is not None:
        return hit
    res = _find_db(owner, attempt_id, kind)
    if res is not None:
        _mem_put(owner, attempt_id, res)
    return res


def _doc_for(result: dict) -> dict:
    topics = [{"topic": b["topic"], "earned": b["earned"], "possible": b["possible"],
               "questions": b.get("questions", 1)} for b in result.get("topic_breakdown") or []]
    doc = {
        "title": result.get("title"), "tool": result["kind"], "kind": result["kind"],
        "attempt_id": result["attempt_id"], "subject": result.get("subject"), "topic": result.get("topic"),
        "correct": result["total"], "total": result["max"], "percent": result["percent"],
        "grade_band": result["grade_band"], "topics": topics, "weak_topics": result.get("weak_topics") or [],
        "verified": bool(result.get("verified")), "graded": bool(result.get("graded", True)),
        "provisional": bool(result.get("provisional")), "duration_seconds": result.get("duration_seconds"),
        "llm": result.get("llm"), "result": result,
    }
    if result["kind"] == "exam":                      # legacy report fields (history drawer / dashboard metric)
        from backend.services import grading
        doc.update(grading.legacy_exam_fields(result))
    return doc


def save(owner: str, result: dict) -> str | None:
    """Insert the attempt (one write). Returns its id, or None (persistence off / duplicate / DB error)."""
    return db.save(collection_for(result["kind"]), owner, _doc_for(result))      # None when persistence is off


def replace(owner: str, result: dict) -> bool:
    """Overwrite an existing attempt in place (provisional -> final)."""
    coll = _coll(result["kind"])
    if coll is None:
        return False
    try:
        doc = _doc_for(result)
        doc["updated_at"] = db.now()
        res = coll.update_one({"owner": owner, "attempt_id": result["attempt_id"]}, {"$set": doc})
        return res.matched_count > 0
    finally:
        db._bump(owner)


def run_idempotent(owner: str, attempt_id: str, kind: str, compute: Callable[[], dict], *,
                   regrade: bool = False) -> tuple[dict, bool]:
    """Grade-and-store exactly once per ``(owner, attempt_id)``. Returns ``(result, replayed)``.

    ``compute()`` runs at most once per call and only when no stored result exists (or ``regrade`` is set and the
    stored one is provisional). The result gets ``saved`` / ``id`` fields describing persistence.
    """
    key = (owner, attempt_id)
    with _locks_guard:
        slot = _locks.setdefault(key, [threading.Lock(), 0])
        slot[1] += 1
    try:
        with slot[0]:
            prior = find(owner, attempt_id, kind)
            if prior is not None and not (regrade and prior.get("provisional")):
                return {**prior, "replayed": True, "saved": bool(prior.get("id")) or prior.get("saved", False)}, True
            result = compute()
            if prior is not None:                              # re-grade of a provisional result: replace in place
                result["id"] = prior.get("id")
                result["saved"] = replace(owner, result) if prior.get("id") else False
            else:
                new_id = save(owner, result)
                if new_id is None and _coll(kind) is not None:     # lost a cross-instance race / DB hiccup
                    raced = _find_db(owner, attempt_id, kind)
                    if raced is not None:
                        _mem_put(owner, attempt_id, raced)
                        return {**raced, "replayed": True, "saved": True}, True
                result["id"] = new_id
                result["saved"] = new_id is not None
            _mem_put(owner, attempt_id, result)
            return result, False
    finally:
        with _locks_guard:
            slot[1] -= 1
            if slot[1] <= 0:
                _locks.pop(key, None)


# ---------------------------------------------------------------------------------------------
_LIST_FIELDS = {"attempt_id": 1, "kind": 1, "subject": 1, "topic": 1, "title": 1, "correct": 1, "total": 1,
                "percent": 1, "grade_band": 1, "weak_topics": 1, "verified": 1, "provisional": 1,
                "duration_seconds": 1, "llm.label": 1, "created_at": 1}


def list_attempts(owner: str, *, subject: str | None = None, kind: str | None = None, limit: int = 30) -> list[dict]:
    """The owner's newest attempts (projected summaries, never the heavy ``result``), newest first."""
    limit = max(1, min(int(limit or 30), 100))
    colls = ["exam"] if kind == "exam" else ["quiz"] if kind else ["quiz", "exam"]
    rows: list[dict] = []
    for ck in colls:
        coll = _coll(ck)
        if coll is None:
            continue
        query: dict = {"owner": owner, "attempt_id": {"$type": "string"}}
        if subject:
            query["subject"] = subject
        if kind:
            query["kind"] = kind
        cur = coll.find(query, _LIST_FIELDS).sort("created_at", -1).limit(limit)
        rows += [db._public(d) for d in cur]
    rows.sort(key=lambda r: str(r.get("created_at") or ""), reverse=True)
    return rows[:limit]


def get_attempt(owner: str, id_str: str) -> dict | None:
    """One attempt's full unified result, owner-scoped (None when unknown / not yours)."""
    oid = db.oid(id_str)
    if oid is None:
        return None
    for ck in ("quiz", "exam"):
        coll = _coll(ck)
        if coll is None:
            return None
        doc = coll.find_one({"_id": oid, "owner": owner}, {"result": 1})
        res = (doc or {}).get("result")
        if isinstance(res, dict):
            return {**res, "id": str(doc["_id"]), "saved": True}
    return None
