"""AI Insights — one personalized study read-out across every tool.

Aggregates the user's cross-tool activity (quiz/flashcard attempts, exam scores,
teach-back evaluations, chats, plans, scans …) into a compact digest and asks
the LLM for a short, actionable analysis: what's going well, where the weak
spots are, and what to do next — each recommendation pointing at the tool that
does it.

Cost control: the digest is plain Python over history summaries (no LLM), the
analysis is ONE bounded Groq call, and results are cached in-process per owner
per UTC day (a Refresh button bypasses the cache). With persistence off there
is no activity to analyze, so the endpoint returns an "empty" payload without
ever calling the LLM.
"""
from __future__ import annotations

import json
import logging
import re
import threading
from datetime import datetime, timezone

from groq import Groq

from backend import config, db
from backend.services.progress import build_summary

log = logging.getLogger(__name__)

_MAX_OUTPUT_TOKENS = 700
_RECENT_TITLES = 6          # how many recent item titles per kind feed the digest

_cache_lock = threading.Lock()
_cache: dict[str, tuple[str, dict]] = {}   # owner -> (UTC date iso, payload)


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
    progress = build_summary(owner)

    kinds = ("chats", "quizzes", "flashcards", "exams", "teachbacks", "plans", "predictions", "scans")
    recent: dict[str, list] = {}
    counts: dict[str, int] = {}
    for kind in kinds:
        items = db.list_items(kind, owner, limit=50)
        counts[kind] = len(items)
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
            recent[kind] = slim

    activity = progress["overall"]["total_attempts"] + sum(counts.values())
    digest = {
        "performance": progress,
        "tool_usage_counts": counts,
        "recent_items": recent,
    }
    return digest, activity


def _extract_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not match:
        return {}
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _analyze(digest: dict, model: str | None) -> dict:
    """One bounded Groq call turning the digest into structured insights."""
    client = Groq(api_key=config.GROQ_API_KEY)
    resolved = config.model_for("insights", model)
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
    resp = client.chat.completions.create(
        model=resolved,
        messages=[{"role": "system", "content": _SYSTEM}, {"role": "user", "content": instruction}],
        max_tokens=_MAX_OUTPUT_TOKENS,
        temperature=0.4,
    )
    data = _extract_json(resp.choices[0].message.content or "")
    if not data.get("headline"):
        raise InsightsError("The model returned an unusable analysis; try refreshing.")
    return {
        "headline": str(data.get("headline")),
        "observations": [str(o) for o in (data.get("observations") or [])][:4],
        "recommendations": [
            {
                "action": str(r.get("action") or ""),
                "reason": str(r.get("reason") or ""),
                "tool": r.get("tool") if isinstance(r.get("tool"), str) else None,
            }
            for r in (data.get("recommendations") or [])
            if isinstance(r, dict) and r.get("action")
        ][:4],
        "focus_topics": [str(t) for t in (data.get("focus_topics") or [])][:5],
    }


def build_insights(owner: str, model: str | None = None, refresh: bool = False) -> dict:
    """The full insights payload for ``owner`` (cached per owner per UTC day)."""
    today = datetime.now(timezone.utc).date().isoformat()
    if not refresh:
        with _cache_lock:
            cached = _cache.get(owner)
        if cached and cached[0] == today:
            return cached[1]

    digest, activity = _activity_digest(owner)
    if activity == 0:
        return {"empty": True, "insights": None, "generated_at": None}

    insights = _analyze(digest, model)
    payload = {
        "empty": False,
        "insights": insights,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    with _cache_lock:
        _cache[owner] = (today, payload)
    return payload
