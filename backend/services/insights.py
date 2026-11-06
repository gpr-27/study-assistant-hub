"""AI Insights — one personalized study read-out across every tool.

Aggregates the user's cross-tool activity (quiz/flashcard attempts, exam scores,
teach-back evaluations, chats, plans, scans …) into a compact digest and asks
the LLM for a short, actionable analysis: what's going well, where the weak
spots are, and what to do next — each recommendation pointing at the tool that
does it.

Cost control: the digest is plain Python over history summaries (no LLM; the ~20 MongoDB reads
run concurrently), the analysis is ONE bounded ``llm_chain`` call, and results are cached
in-process per owner per UTC day in a BOUNDED LRU (a Refresh button bypasses the cache, rate
limited per owner). Concurrent requests for the same owner share one generation. With
persistence off there is no activity to analyze, so the endpoint returns an "empty" payload
without ever calling the LLM. Every payload carries ``llm`` (the model that wrote it).
"""
from __future__ import annotations

import json
import logging
import threading
import time
from collections import OrderedDict
from datetime import datetime, timezone

from backend import cache, db, llm_chain
from backend.services.llm_json import extract_json_object
from backend.services.progress import build_summary

log = logging.getLogger(__name__)

_MAX_OUTPUT_TOKENS = 2048   # reasoning models spend output tokens thinking; a small cap = empty reply
_RECENT_TITLES = 6          # how many recent item titles per kind feed the digest
_CACHE_MAX = 512
_REFRESH_MIN_INTERVAL_S = 20.0
# New activity (db.epoch changes) invalidates the day's card, but never more often than this per
# owner: every chat message bumps the epoch, and each regeneration is a paid LLM call.
_EPOCH_REGEN_MIN_S = 90.0
_DEADLINE_S = 30.0

_cache_lock = threading.Lock()
_cache: "OrderedDict[str, tuple[tuple, dict, float]]" = OrderedDict()    # owner -> ((UTC date, db.epoch), payload, made_at)
_last_refresh: "OrderedDict[str, float]" = OrderedDict()
_owner_locks: "OrderedDict[str, threading.Lock]" = OrderedDict()


class InsightsError(Exception):
    """Raised when the insights analysis fails."""


_SYSTEM = (
    "You are a study coach analyzing a student's activity across their study app "
    "(chat questions, quizzes, flashcards, exams, teach-back sessions, plans). "
    "Be specific, encouraging and brutally practical. Ground every claim in the "
    "digest — never invent topics that aren't in it. Always reply with valid JSON only."
)


def _activity_digest(owner: str) -> tuple[dict, int]:
    """Build the cross-tool digest. Returns (digest, activity_count)."""
    kinds = ("chats", "quizzes", "flashcards", "exams", "teachbacks", "plans", "predictions", "scans")
    # build_summary is one cached round trip; the per-kind digest rows are one more (was ~20 reads).
    progress = build_summary(owner)
    recent = db.multi_recent(owner, [
        (k, 50, ["title", "total_score", "max_score", "evaluation.understanding_score_0_to_10",
                 "evaluation.gaps"]) for k in kinds])
    recent_by_kind: dict[str, list] = {}
    counts: dict[str, int] = {}
    for kind in kinds:
        slot = recent.get(kind) or {"n": 0, "rows": []}
        items = slot["rows"]
        counts[kind] = slot["n"]
        slim = []
        for item in items[:_RECENT_TITLES]:
            entry: dict = {"title": item.get("title") or "(untitled)"}
            # Scores travel when present (exams store totals; teach-backs an evaluation).
            if item.get("total_score") is not None and item.get("max_score"):
                entry["score"] = f"{item['total_score']}/{item['max_score']}"
            ev = item.get("evaluation")
            if isinstance(ev, dict) and ev.get("understanding_score_0_to_10") is not None:
                entry["understanding"] = f"{ev['understanding_score_0_to_10']}/10"
                if ev.get("gaps"):
                    entry["gaps"] = ev["gaps"][:3]
            slim.append(entry)
        if slim:
            recent_by_kind[kind] = slim
    recent = recent_by_kind
    activity = progress["overall"]["total_attempts"] + sum(counts.values())
    digest = {
        "performance": progress,
        "tool_usage_counts": counts,
        "recent_items": recent,
    }
    return digest, activity


def _analyze(digest: dict) -> tuple[dict, dict]:
    """One bounded chain call turning the digest into structured insights. Returns (insights, llm)."""
    instruction = (
        "Student activity digest (JSON):\n"
        + json.dumps(digest, default=str)[: 6000 * 3]  # chars ≈ tokens*3; stay well under TPM
        + "\n\nReturn ONLY JSON:\n"
        '{"headline": "<one motivating sentence summarizing where they stand>",\n'
        ' "observations": ["<2-4 specific, evidence-based observations>"],\n'
        ' "recommendations": [{"action": "<imperative, concrete next step>", '
        '"reason": "<why, tied to the digest>", '
        '"tool": "quiz"|"flashcards"|"exam"|"teachback"|"chat"|"planner"|"cheatsheet"|"mindmap"|null}],\n'
        ' "focus_topics": ["<up to 5 topics to prioritize>"]}\n'
        "2-4 recommendations. If activity is thin, say so honestly and recommend how to start."
    )
    result = llm_chain.chat(_SYSTEM, [{"role": "user", "content": instruction}],
                            max_tokens=_MAX_OUTPUT_TOKENS, temperature=0.4, json_mode=True,
                            deadline_s=_DEADLINE_S)
    data = extract_json_object(result.text)
    if not data.get("headline"):
        raise InsightsError("The model returned an unusable analysis; try refreshing.")
    insights = {
        "headline": str(data.get("headline")),
        "observations": [str(o) for o in (data.get("observations") or []) if o][:4],
        "recommendations": [
            {
                "action": str(r.get("action") or ""),
                "reason": str(r.get("reason") or ""),
                "tool": r.get("tool") if isinstance(r.get("tool"), str) else None,
            }
            for r in (data.get("recommendations") or [])
            if isinstance(r, dict) and r.get("action")
        ][:4],
        "focus_topics": [str(t) for t in (data.get("focus_topics") or []) if t][:5],
    }
    return insights, result.to_public()


def _owner_lock(owner: str) -> threading.Lock:
    with _cache_lock:
        lock = _owner_locks.get(owner)
        if lock is None:
            lock = _owner_locks[owner] = threading.Lock()
        _owner_locks.move_to_end(owner)
        while len(_owner_locks) > _CACHE_MAX:
            _owner_locks.popitem(last=False)
        return lock


def _cached(owner: str, today: str) -> dict | None:
    """The cached card for ``owner`` if it is still valid: same UTC day AND no newer activity
    (``db.epoch``) - except that fresh activity may only trigger a regeneration every
    ``_EPOCH_REGEN_MIN_S`` seconds."""
    epoch = db.epoch(owner)
    with _cache_lock:
        hit = _cache.get(owner)
        if hit and hit[0][0] == today and (
                hit[0][1] == epoch or time.monotonic() - hit[2] < _EPOCH_REGEN_MIN_S):
            _cache.move_to_end(owner)
            return hit[1]
    return None


def build_insights(owner: str, model: str | None = None, refresh: bool = False) -> dict:
    """The full insights payload for ``owner`` (cached per owner per UTC day; ``model`` is ignored)."""
    today = datetime.now(timezone.utc).date().isoformat()
    if refresh:
        with _cache_lock:
            last = _last_refresh.get(owner, -1e9)
        if time.monotonic() - last < _REFRESH_MIN_INTERVAL_S:
            refresh = False          # spam-click protection: serve the cached card instead of another LLM call
    if not refresh:
        hit = _cached(owner, today)
        if hit is not None:
            return hit

    with _owner_lock(owner):         # concurrent requests for one owner share a single generation
        if not refresh:
            hit = _cached(owner, today)
            if hit is not None:
                return hit
        epoch = db.epoch(owner)          # read BEFORE building, so activity racing the build re-invalidates
        l2, l2_ep = (None, None) if refresh else cache.l2_get("insights", owner, today, max_stale_s=_EPOCH_REGEN_MIN_S)
        if isinstance(l2, dict):         # another instance already paid for today's card
            with _cache_lock:
                _cache[owner] = ((today, epoch), l2, time.monotonic())
                _cache.move_to_end(owner)
            return l2
        if l2_ep is None:
            l2_ep = cache.l2_get("insights", owner, today)[1]
        digest, activity = _activity_digest(owner)
        if activity == 0:
            return {"empty": True, "insights": None, "generated_at": None, "llm": None}
        with _cache_lock:
            _last_refresh[owner] = time.monotonic()
            while len(_last_refresh) > _CACHE_MAX:
                _last_refresh.popitem(last=False)
        insights, llm_public = _analyze(digest)
        payload = {
            "empty": False,
            "insights": insights,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "llm": llm_public,
        }
        with _cache_lock:
            _cache[owner] = ((today, epoch), payload, time.monotonic())
            _cache.move_to_end(owner)
            while len(_cache) > _CACHE_MAX:
                _cache.popitem(last=False)
        cache.l2_put("insights", owner, today, payload, l2_ep, 86_400)
        return payload
