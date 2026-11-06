"""Progress service — record quiz/flashcard attempts and aggregate them.

An "attempt" is one finished quiz run or flashcard review session, stored as an
event in the per-user history (kind "attempts", see backend.db). NO LLM is used
here: recording is a single db.save and the summary is computed in Python from
the most recent events, so the feature works (gracefully degrades) exactly like
the rest of the history system — when persistence is off nothing is saved and
the summary is simply empty.
"""
from __future__ import annotations

import copy
import threading
import time
from collections import OrderedDict
from datetime import datetime, timedelta, timezone

from backend import cache, db

# How many recent attempts feed the summary (matches db.list_items' default cap).
_SUMMARY_LIMIT: int = 100
# A topic needs at least this many attempts before it can be called "weak"
# (one bad roll of the dice shouldn't brand a topic).
_MIN_ATTEMPTS_FOR_WEAKNESS: int = 2
# ... unless one graded sitting already asked this many questions on the topic (a mock exam covers several topics).
_MIN_QUESTIONS_FOR_WEAKNESS: int = 3
# Mastery weights recent attempts more: an attempt's weight halves every this many days.
_HALF_LIFE_DAYS: float = 21.0
_MAX_WEAKEST: int = 5

_MAX_TOTAL: int = 10_000        # sane ceiling for one attempt's question/card count
_MAX_LABEL: int = 200
_VALID_TOOLS: tuple[str, ...] = ("quiz", "flashcards")
_FALLBACK_LABEL: str = "General"


# Per-instance summary cache, keyed by owner (an entry can only ever be returned to the owner whose
# key it is). Dropped when this instance observes a write for that owner (db.epoch) and after a short
# TTL, which bounds staleness when the write happened on another serverless instance.
_CACHE_TTL_S: float = 20.0
_CACHE_MAX: int = 512
_cache: "OrderedDict[str, tuple[float, tuple, dict]]" = OrderedDict()
_cache_lock = threading.Lock()


def clear_cache() -> None:
    with _cache_lock:
        _cache.clear()


class ProgressError(Exception):
    """Raised when an attempt cannot be recorded or summarised."""


def _clean(value: str | None) -> str | None:
    """Strip a free-text label; empty/whitespace-only becomes None."""
    if value is None:
        return None
    value = str(value).strip()[:_MAX_LABEL]
    return value or None


def record_attempt(
    owner: str,
    tool: str,
    correct: int,
    total: int,
    subject: str | None = None,
    topic: str | None = None,
    detail: str | None = None,
) -> bool:
    """Save one attempt event for ``owner``. Returns True when persisted.

    Tolerant by design (it's called fire-and-forget from the SPA): labels are
    trimmed, ``correct`` is clamped into [0, total], and a disabled database
    simply returns False — never an exception.
    """
    tool = (tool or "").strip().lower()
    if tool not in _VALID_TOOLS:
        raise ProgressError(f"Unknown tool {tool!r}; expected one of {', '.join(_VALID_TOOLS)}.")
    try:
        total, correct = int(total), int(correct)
    except (TypeError, ValueError) as exc:
        raise ProgressError("`correct` and `total` must be integers.") from exc
    if total < 1:
        raise ProgressError("`total` must be at least 1.")
    total = min(total, _MAX_TOTAL)
    correct = max(0, min(correct, total))
    subject = _clean(subject)
    topic = _clean(topic)
    label = subject or topic or _FALLBACK_LABEL
    doc = {
        "title": f"{tool} · {label}",
        "tool": tool,
        "subject": subject,
        "topic": topic,
        "correct": correct,
        "total": total,
    }
    detail = _clean(detail)
    if detail:
        doc["detail"] = detail
    saved_id = db.save("attempts", owner, doc)
    return saved_id is not None


def _parse_dt(value) -> datetime | None:
    """Parse a created_at value (ISO string or datetime) into an aware datetime."""
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    return None


def _streak_days(dates: set, today=None) -> int:
    """Consecutive calendar days (UTC) with >=1 attempt, ending today or yesterday.

    A streak is still "alive" if the last attempt was yesterday (the user hasn't
    studied *yet* today), so counting starts at today when present, else yesterday.
    """
    if not dates:
        return 0
    today = today or datetime.now(timezone.utc).date()
    start = today if today in dates else today - timedelta(days=1)
    if start not in dates:
        return 0
    streak = 0
    day = start
    while day in dates:
        streak += 1
        day -= timedelta(days=1)
    return streak


# Every tool that contributes to the unified dashboard: history kind -> display.
_ACTIVITY_TOOLS: tuple[tuple[str, str, str], ...] = (
    ("chats", "Chats", "#chat"),
    ("quizzes", "Quizzes", "#quiz"),
    ("flashcards", "Flashcard decks", "#flashcards"),
    ("exams", "Mock exams", "#exam"),
    ("teachbacks", "Teach-backs", "#teachback"),
    ("plans", "Study plans", "#planner"),
    ("summaries", "Summaries", "#planner"),
    ("mindmaps", "Mind maps", "#mindmap"),
    ("predictions", "Predictions", "#predictor"),
    ("cheatsheets", "Cheat sheets", "#cheatsheet"),
    ("scans", "Note scans", "#scannotes"),
)


def _activity_rows(owner: str, recent: dict) -> list[dict]:
    """Per-tool activity for the whole app: counts plus a performance metric
    where one exists (exam average score, teach-back average understanding).

    Pure aggregation over history summaries — no LLM. Tools with zero items
    are included (count 0) so the dashboard shows the full app surface. ``recent`` is the
    result of the single ``db.multi_recent`` round trip made by :func:`build_summary`.
    """
    activity = []
    for kind, label, route in _ACTIVITY_TOOLS:
        slot = recent.get(kind) or {"n": 0, "rows": []}
        items = slot["rows"]
        metric = None
        if kind == "exams":
            scores = [
                (float(i["total_score"]), float(i["max_score"]))
                for i in items
                if isinstance(i.get("total_score"), (int, float))
                and isinstance(i.get("max_score"), (int, float)) and i["max_score"]
            ]
            if scores:
                avg = sum(s / m for s, m in scores) / len(scores)
                metric = f"avg {round(avg * 100)}%"
        elif kind == "teachbacks":
            scores = [
                float(ev["understanding_score_0_to_10"])
                for i in items
                if isinstance(ev := i.get("evaluation"), dict)
                and isinstance(ev.get("understanding_score_0_to_10"), (int, float))
            ]
            if scores:
                metric = f"avg {round(sum(scores) / len(scores), 1)}/10"
        activity.append({
            "tool": kind, "label": label, "route": route,
            "count": slot["n"], "metric": metric,
        })
    return activity


def build_summary(owner: str) -> dict:
    """Aggregate the owner's recent attempts into a progress dashboard payload.

    Shape:
      {
        "overall": {"total_attempts", "overall_accuracy", "streak_days"},
        "subjects": [{"subject", "attempts", "accuracy",
                      "topics": [{"topic", "attempts", "accuracy", "last_seen"}]}],
        "weakest": [{"subject", "topic", "attempts", "accuracy"}],  # acc asc
      }
    Accuracy values are fractions in [0, 1] (the SPA renders percentages).
    """
    ep = db.epoch(owner)
    now = time.monotonic()
    with _cache_lock:
        hit = _cache.get(owner)
        if hit and hit[0] > now and hit[1] == ep:
            _cache.move_to_end(owner)
            return copy.deepcopy(hit[2])
    # Concurrent identical misses (a dashboard polled by many tabs) compute once; the rest read the L1 entry.
    with cache.single_flight(("progress", owner)):
        with _cache_lock:
            hit = _cache.get(owner)
            if hit and hit[0] > time.monotonic() and hit[1] == ep:
                return copy.deepcopy(hit[2])
        payload, l2_ep = cache.l2_get("progress", owner)      # shared across instances (epoch-validated)
        if not isinstance(payload, dict):
            payload = _compute_summary(owner)
            cache.l2_put("progress", owner, "", payload, l2_ep, 120.0)
        with _cache_lock:
            _cache[owner] = (now + _CACHE_TTL_S, ep, payload)
            _cache.move_to_end(owner)
            while len(_cache) > _CACHE_MAX:
                _cache.popitem(last=False)
    return copy.deepcopy(payload)


def _entries(item: dict) -> list[tuple[str | None, str | None, float, float, int]]:
    """``[(subject, topic, earned, possible, questions)]`` for one stored attempt / exam row.

    Unified rows (``backend.services.attempts``) carry per-topic ``topics`` (exams: ``topic_breakdown``) so ONE
    exam feeds several topic cells. Legacy rows (``/api/progress/record``) are one ``correct/total`` cell.
    Rows flagged ``verified: false`` (answer key could not be verified) are skipped; a missing flag = legacy = trusted.
    """
    if item.get("verified") is False:
        return []
    subject = _clean(item.get("subject"))
    raw = item.get("topics") if isinstance(item.get("topics"), list) else item.get("topic_breakdown")
    out = []
    if isinstance(raw, list) and raw:
        for t in raw:
            if not isinstance(t, dict):
                continue
            try:
                earned, possible = float(t.get("earned")), float(t.get("possible"))
            except (TypeError, ValueError):
                continue
            if possible <= 0:
                continue
            q = t.get("questions")
            out.append((subject, _clean(t.get("topic")), max(0.0, min(earned, possible)), possible,
                        int(q) if isinstance(q, (int, float)) and q >= 1 else 1))
        if out:
            return out
    try:
        correct, total = float(item.get("correct", 0)), float(item.get("total", 0))
    except (TypeError, ValueError):
        return []
    if total < 1 or (item.get("topics") is None and item.get("topic_breakdown") is None and "correct" not in item):
        return []
    return [(subject, _clean(item.get("topic")), max(0.0, min(correct, total)), total, 1)]


def _recency(when: datetime | None, now: datetime) -> float:
    """Weight in (0, 1]: an attempt loses half its weight every ``_HALF_LIFE_DAYS`` days."""
    if when is None:
        return 0.5
    age = max(0.0, (now - when).total_seconds() / 86400.0)
    return 0.5 ** (age / _HALF_LIFE_DAYS)


def _compute_summary(owner: str) -> dict:
    # ONE MongoDB round trip for attempts + every tool's activity (was 12 separate reads), and none at all
    # when the owner snapshot (db.warm_owner, shared with the history lists) is already in memory.
    try:
        db.warm_owner(owner)
    except Exception:  # noqa: BLE001 - the explicit query below is the fallback
        pass
    specs = [("attempts", _SUMMARY_LIMIT, ["subject", "topic", "correct", "total", "created_at", "topics", "verified"])]
    for kind, _l, _r in _ACTIVITY_TOOLS:
        if kind == "exams":
            specs.append((kind, 100, ["total_score", "max_score", "topics", "topic_breakdown", "subject", "verified", "created_at"]))
        elif kind == "teachbacks":
            specs.append((kind, 100, ["evaluation.understanding_score_0_to_10"]))
        else:
            specs.append((kind, 100, None))
    recent = db.multi_recent(owner, specs)
    items = list(recent["attempts"]["rows"])
    # Mock exams are graded attempts too: their per-topic marks feed the same mastery cells.
    items += [r for r in (recent.get("exams") or {}).get("rows", []) if isinstance(r.get("topics"), list) or isinstance(r.get("topic_breakdown"), list)]
    activity_rows = _activity_rows(owner, recent)

    now = datetime.now(timezone.utc)
    total_attempts = 0
    sum_correct = 0.0
    sum_total = 0.0
    dates: set = set()
    # (subject, topic) -> raw + recency-weighted marks, attempts and question counts
    cells: dict[tuple[str, str], dict] = {}

    for item in items:
        entries = _entries(item)
        if not entries:
            continue
        when = _parse_dt(item.get("created_at"))
        weight = _recency(when, now)
        total_attempts += 1
        if when is not None:
            dates.add(when.astimezone(timezone.utc).date())
        for subject, topic, earned, possible, questions in entries:
            sum_correct += earned
            sum_total += possible
            cell = cells.setdefault(
                (subject or _FALLBACK_LABEL, topic or _FALLBACK_LABEL),
                {"attempts": 0, "correct": 0.0, "total": 0.0, "wc": 0.0, "wt": 0.0, "questions": 0, "last_seen": None},
            )
            cell["attempts"] += 1
            cell["correct"] += earned
            cell["total"] += possible
            cell["wc"] += weight * earned
            cell["wt"] += weight * possible
            cell["questions"] += questions
            if when is not None and (cell["last_seen"] is None or when > cell["last_seen"]):
                cell["last_seen"] = when

    def mastery(cell: dict) -> float:
        return cell["wc"] / cell["wt"] if cell["wt"] else 0.0

    # Per-subject groups with per-topic cells.
    by_subject: dict[str, list[dict]] = {}
    for (subject, topic), cell in cells.items():
        accuracy = cell["correct"] / cell["total"] if cell["total"] else 0.0
        by_subject.setdefault(subject, []).append({
            "topic": topic,
            "attempts": cell["attempts"],
            "accuracy": round(accuracy, 4),
            "mastery": round(mastery(cell), 4),
            "questions": cell["questions"],
            "last_seen": cell["last_seen"].isoformat() if cell["last_seen"] else None,
        })

    subjects = []
    for subject, topics in sorted(by_subject.items()):
        topics.sort(key=lambda t: t["topic"])
        s_correct = sum(cells[(subject, t["topic"])]["correct"] for t in topics)
        s_total = sum(cells[(subject, t["topic"])]["total"] for t in topics)
        subjects.append({
            "subject": subject,
            "attempts": sum(t["attempts"] for t in topics),
            "accuracy": round(s_correct / s_total, 4) if s_total else 0.0,
            "topics": topics,
        })

    weakest = sorted(
        (
            {"subject": subject, "topic": topic,
             "attempts": cell["attempts"],
             "accuracy": round(cell["correct"] / cell["total"], 4) if cell["total"] else 0.0,
             "mastery": round(mastery(cell), 4)}
            for (subject, topic), cell in cells.items()
            if cell["attempts"] >= _MIN_ATTEMPTS_FOR_WEAKNESS or cell["questions"] >= _MIN_QUESTIONS_FOR_WEAKNESS
        ),
        key=lambda w: (w["mastery"], -w["attempts"]),
    )[:_MAX_WEAKEST]

    return {
        "overall": {
            "total_attempts": total_attempts,
            "overall_accuracy": round(sum_correct / sum_total, 4) if sum_total else 0.0,
            "streak_days": _streak_days(dates),
        },
        "subjects": subjects,
        "weakest": weakest,
        "activity": activity_rows,
    }
