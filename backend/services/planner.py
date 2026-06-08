"""Study Planner service: per-PDF summaries and a markdown study plan.

Ported from the OpenAI-based ``studyplanner`` package to Groq. Summaries use
SUMMARY_MODEL and follow a TOPICS/SUMMARY layout; the plan is produced as
markdown so the SPA can render it directly (no brittle regex parsing). All LLM
clients are created lazily inside functions — never at import time.
"""
import datetime

from dateutil import parser as date_parser
from groq import APIError

from backend import chunk_summaries
from backend.config import MAX_CONTENT_TOKENS, model_for, require_groq_key
from backend.documents import clip_to_tokens, pdf_signature

_HOURS_PER_DAY: int = 2

# Cache of generated summaries keyed by (subject, pdf, signature) to avoid
# re-billing the same document within a process lifetime. The signature
# (mtime, size) is part of the key so replacing a PDF re-summarizes it instead
# of serving a stale summary.
_summary_cache: dict[tuple[str, str, tuple[float, int], str | None], str] = {}


class PlannerError(Exception):
    """Raised when a summary or study plan cannot be generated."""


def _summary_prompt(pdf_name: str, pdf_text: str) -> str:
    """Build the TOPICS/SUMMARY analysis prompt for a single PDF."""
    content = clip_to_tokens(pdf_text, MAX_CONTENT_TOKENS)
    return (
        "Analyze the following lecture content and respond in plain text "
        "(not JSON) using EXACTLY this format:\n\n"
        "TOPICS:\n"
        "- <Topic 1> - <Basic|Intermediate|Advanced>\n"
        "  - <Key concept>\n"
        "  - <Key concept>\n"
        "- <Topic 2> - <Basic|Intermediate|Advanced>\n"
        "  - <Key concept>\n\n"
        "SUMMARY:\n"
        "<A brief overall summary of the content>\n\n"
        f"PDF: {pdf_name}\n\n"
        "CONTENT:\n"
        f"{content}"
    )


def _summarize_text(name: str, text: str, model: str | None = None) -> str:
    """Summarize a single document's text via Groq (TOPICS/SUMMARY layout).

    Shared by subject PDFs and uploaded PDFs — both reduce to (name, text).
    """
    if not text or not text.strip():
        raise PlannerError(f"No extractable text in: {name}")

    require_groq_key()
    from groq import Groq

    client = Groq()
    try:
        resp = client.chat.completions.create(
            model=model_for("summary", model),
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an educational assistant that summarizes "
                        "lecture content effectively."
                    ),
                },
                {"role": "user", "content": _summary_prompt(name, text)},
            ],
            temperature=0.4,
        )
    except APIError:
        raise  # Groq errors (rate limit, auth, …) are formatted by the global handlers
    except Exception as exc:  # noqa: BLE001 - re-raised as a domain error
        raise PlannerError(f"Groq summary request failed for {name}: {exc}") from exc

    summary = (resp.choices[0].message.content or "").strip()
    if not summary:
        raise PlannerError(f"Empty summary returned for {name}")
    return summary


def _summarize_one(subject: str, pdf: str, model: str | None = None, owner: str | None = None) -> str:
    """Return a cached or freshly generated TOPICS/SUMMARY view for a subject PDF."""
    resolved = model_for("summary", model)
    cache_key = (subject, pdf, pdf_signature(subject, pdf, owner), resolved)
    if cache_key in _summary_cache:
        return _summary_cache[cache_key]
    chunks = chunk_summaries.get_or_build(subject, pdf, model, owner)
    summary = _summarize_text(pdf, chunk_summaries.summaries_as_content(chunks), model=model)
    _summary_cache[cache_key] = summary
    return summary


def summarize_pdfs(subject: str, pdfs: list[str], model: str | None = None, owner: str | None = None) -> dict[str, str]:
    """Summarize each selected subject PDF and return a {pdf_name: summary} map."""
    if not pdfs:
        raise PlannerError("No PDFs selected to summarize.")
    return {pdf: _summarize_one(subject, pdf, model, owner) for pdf in pdfs}


def summarize_named_texts(items: list[tuple[str, str]], model: str | None = None) -> dict[str, str]:
    """Summarize uploaded (name, text) documents.

    Each upload is first chunked and chunk-summarized (cached/saved by content, like a
    subject PDF), then the TOPICS/SUMMARY view is built from those chunk summaries rather
    than the raw text — so a re-upload reuses the saved work instead of re-billing it.
    """
    if not items:
        raise PlannerError("No documents provided to summarize.")
    out: dict[str, str] = {}
    for name, text in items:
        chunks = chunk_summaries.get_or_build_for_text(text, name, model)
        out[name] = _summarize_text(name, chunk_summaries.summaries_as_content(chunks), model=model)
    return out


def _days_until(target_date: str) -> tuple[datetime.date, datetime.date, int]:
    """Parse the target date and return (start, target, days) with days >= 1."""
    try:
        target = date_parser.parse(target_date).date()
    except (ValueError, OverflowError, TypeError) as exc:
        raise PlannerError(f"Could not parse target date: {target_date!r}") from exc

    start = datetime.date.today()
    days = max(1, (target - start).days)
    return start, target, days


def _plan_prompt(
    subject: str,
    summaries: dict[str, str],
    start: datetime.date,
    target: datetime.date,
    days: int,
) -> str:
    """Build the day-by-day study-plan prompt from the PDF summaries."""
    content_for_plan = "\n\n".join(
        f"PDF: {pdf_name}\nSummary: {summary}"
        for pdf_name, summary in summaries.items()
    )
    return (
        f"Create a detailed day-by-day study plan for {days} day(s), running "
        f"from {start.isoformat()} to {target.isoformat()}. The plan covers the "
        f'lecture materials for the subject "{subject}".\n\n'
        "PDF Summaries:\n"
        f"{content_for_plan}\n\n"
        "Guidelines:\n"
        "1. Use the topics and difficulty levels from the summaries to sequence work.\n"
        "2. Start with basic topics and progress to advanced ones.\n"
        "3. Suggest study techniques matched to each topic's difficulty.\n"
        "4. Add short breaks and review sessions.\n"
        f"5. IMPORTANT: each day MUST include at least {_HOURS_PER_DAY} hours of study.\n\n"
        "FORMAT THE PLAN EXACTLY AS MARKDOWN LIKE THIS:\n"
        f"# Study Plan for {subject}\n\n"
        f"Duration: {days} days | Exam Date: {target.isoformat()}\n\n"
        "## Key Objectives\n"
        "- <objective 1>\n"
        "- <objective 2>\n\n"
        f"## Day 1: {start.isoformat()} ({_HOURS_PER_DAY}h)\n"
        "- <topic to study with PDF reference>\n"
        "- <study technique>\n"
        "- <break / review timing>\n\n"
        "## Day 2: <date> (<hours>h)\n"
        "Continue for all days, keeping the plan progressive with regular reviews."
    )


def _topic_plan_prompt(topic: str, start: datetime.date, target: datetime.date, days: int) -> str:
    """Build a day-by-day study-plan prompt for a free topic (no documents)."""
    return (
        f"Create a detailed day-by-day study plan for {days} day(s), running from "
        f'{start.isoformat()} to {target.isoformat()}, to learn the topic "{topic}" '
        "from the basics to a solid working level.\n\n"
        "Guidelines:\n"
        "1. Break the topic into subtopics and sequence them basics-first.\n"
        "2. Suggest study techniques and practice for each subtopic.\n"
        "3. Add short breaks and review sessions.\n"
        f"4. IMPORTANT: each day MUST include at least {_HOURS_PER_DAY} hours of study.\n\n"
        "FORMAT THE PLAN EXACTLY AS MARKDOWN LIKE THIS:\n"
        f"# Study Plan for {topic}\n\n"
        f"Duration: {days} days | Target Date: {target.isoformat()}\n\n"
        "## Key Objectives\n"
        "- <objective 1>\n"
        "- <objective 2>\n\n"
        f"## Day 1: {start.isoformat()} ({_HOURS_PER_DAY}h)\n"
        "- <subtopic to study>\n"
        "- <study technique>\n"
        "- <break / review timing>\n\n"
        "## Day 2: <date> (<hours>h)\n"
        "Continue for all days, keeping the plan progressive with regular reviews."
    )


_PLAN_SYSTEM: str = (
    "You are an educational expert who creates effective, realistic study plans "
    f"with at least {_HOURS_PER_DAY} hours of study time per day. Respond in clean markdown."
)


def _call_plan(prompt: str, model: str | None = None) -> str:
    """Run the study-plan prompt through Groq and return clean markdown."""
    require_groq_key()
    from groq import Groq

    client = Groq()
    try:
        resp = client.chat.completions.create(
            model=model_for("planner", model),
            messages=[
                {"role": "system", "content": _PLAN_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0.5,
        )
    except APIError:
        raise  # Groq errors (rate limit, auth, …) are formatted by the global handlers
    except Exception as exc:  # noqa: BLE001 - re-raised as a domain error
        raise PlannerError(f"Groq study-plan request failed: {exc}") from exc

    plan = (resp.choices[0].message.content or "").strip()
    if not plan:
        raise PlannerError("The model returned an empty study plan.")
    return plan


def _run_plan(label: str, summaries: dict[str, str], target_date: str, model: str | None = None) -> str:
    """Build a plan from already-computed summaries."""
    start, target, days = _days_until(target_date)
    return _call_plan(_plan_prompt(label, summaries, start, target, days), model)


def generate_plan(subject: str, pdfs: list[str], target_date: str, model: str | None = None, owner: str | None = None) -> str:
    """Generate a markdown study plan for selected subject PDFs and a target date."""
    return _run_plan(subject, summarize_pdfs(subject, pdfs, model, owner), target_date, model)


def generate_plan_from_texts(label: str, items: list[tuple[str, str]], target_date: str, model: str | None = None) -> str:
    """Generate a plan from uploaded (name, text) documents and a target date."""
    return _run_plan(label, summarize_named_texts(items, model), target_date, model)


def generate_plan_from_topic(topic: str, target_date: str, model: str | None = None) -> str:
    """Generate a plan to learn a free topic by the target date (no documents)."""
    if not topic or not topic.strip():
        raise PlannerError("Enter a topic to plan for.")
    start, target, days = _days_until(target_date)
    return _call_plan(_topic_plan_prompt(topic.strip(), start, target, days), model)
