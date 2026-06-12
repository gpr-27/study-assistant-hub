"""Progress service — record quiz/flashcard attempts and aggregate them.

An "attempt" is one finished quiz run or flashcard review session, stored as an
event in the per-user history (kind "attempts", see backend.db). NO LLM is used
here: recording is a single db.save and the summary is computed in Python from
the most recent events, so the feature works (gracefully degrades) exactly like
the rest of the history system — when persistence is off nothing is saved and
the summary is simply empty.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from backend import db

# How many recent attempts feed the summary (matches db.list_items' default cap).
_SUMMARY_LIMIT: int = 100
# A topic needs at least this many attempts before it can be called "weak"
# (one bad roll of the dice shouldn't brand a topic).
_MIN_ATTEMPTS_FOR_WEAKNESS: int = 2
_MAX_WEAKEST: int = 5

_VALID_TOOLS: tuple[str, ...] = ("quiz", "flashcards")
_FALLBACK_LABEL: str = "General"


class ProgressError(Exception):
    """Raised when an attempt cannot be recorded or summarised."""


def _clean(value: str | None) -> str | None:
    """Strip a free-text label; empty/whitespace-only becomes None."""
    if value is None:
        return None
    value = str(value).strip()
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
    total = int(total)
    if total < 1:
        raise ProgressError("`total` must be at least 1.")
    correct = max(0, min(int(correct), total))
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


def _tool_activity(owner: str) -> list[dict]:
    """Per-tool activity for the whole app: counts plus a performance metric
    where one exists (exam average score, teach-back average understanding).

    Pure aggregation over history summaries — no LLM. Tools with zero items
    are included (count 0) so the dashboard shows the full app surface.
    """
    activity = []
    for kind, label, route in _ACTIVITY_TOOLS:
        items = db.list_items(kind, owner, limit=100)
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
            "count": len(items), "metric": metric,
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
    items = db.list_items("attempts", owner, limit=_SUMMARY_LIMIT)

    total_attempts = 0
    sum_correct = 0
    sum_total = 0
    dates: set = set()
    # (subject, topic) -> {"attempts", "correct", "total", "last_seen": datetime}
    cells: dict[tuple[str, str], dict] = {}

    for item in items:
        try:
            correct = int(item.get("correct", 0))
            total = int(item.get("total", 0))
        except (TypeError, ValueError):
            continue
        if total < 1:
            continue
        correct = max(0, min(correct, total))
        total_attempts += 1
        sum_correct += correct
        sum_total += total

        when = _parse_dt(item.get("created_at"))
        if when is not None:
            dates.add(when.astimezone(timezone.utc).date())

        subject = _clean(item.get("subject")) or _FALLBACK_LABEL
        topic = _clean(item.get("topic")) or _FALLBACK_LABEL
        cell = cells.setdefault(
            (subject, topic), {"attempts": 0, "correct": 0, "total": 0, "last_seen": None}
        )
        cell["attempts"] += 1
        cell["correct"] += correct
        cell["total"] += total
        if when is not None and (cell["last_seen"] is None or when > cell["last_seen"]):
            cell["last_seen"] = when

    # Per-subject groups with per-topic cells.
    by_subject: dict[str, list[dict]] = {}
    for (subject, topic), cell in cells.items():
        accuracy = cell["correct"] / cell["total"] if cell["total"] else 0.0
        by_subject.setdefault(subject, []).append({
            "topic": topic,
            "attempts": cell["attempts"],
            "accuracy": round(accuracy, 4),
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
             "accuracy": round(cell["correct"] / cell["total"], 4) if cell["total"] else 0.0}
            for (subject, topic), cell in cells.items()
            if cell["attempts"] >= _MIN_ATTEMPTS_FOR_WEAKNESS
        ),
        key=lambda w: (w["accuracy"], -w["attempts"]),
    )[:_MAX_WEAKEST]

    return {
        "overall": {
            "total_attempts": total_attempts,
            "overall_accuracy": round(sum_correct / sum_total, 4) if sum_total else 0.0,
            "streak_days": _streak_days(dates),
        },
        "subjects": subjects,
        "weakest": weakest,
        "activity": _tool_activity(owner),
    }
