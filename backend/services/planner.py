"""Study Planner service: per-PDF summaries and a markdown study plan.

Summaries follow a TOPICS/SUMMARY layout; the plan is produced as markdown so the SPA can render it
directly (no brittle regex parsing). Everything goes through the provider chain. Independent
per-PDF summaries run concurrently (capped to protect the providers' per-minute token budgets), the
content fed to each call is bounded, and finished summaries are kept in a small per-owner TTL cache.
"""
import datetime
from concurrent.futures import ThreadPoolExecutor

from dateutil import parser as date_parser

from backend.documents import pdf_signature, read_pdf
from backend.services import _gen

_HOURS_PER_DAY: int = 2

MAX_PDFS: int = 12                 # per request
_MAX_WORKERS: int = 3              # concurrent summary calls (TPM protection)
_MAX_PLAN_DAYS: int = 366
_DAILY_LIMIT: int = 21             # beyond this many days the plan is week-by-week

# Output sizes are requests: Anthropic ignores them (always its model max), Groq/Gemini clamp them.
_SUMMARY_MAX_OUTPUT_TOKENS: int = 2500
_PLAN_MAX_OUTPUT_TOKENS: int = 8000

# Finished summaries, keyed by (owner, subject, pdf, signature) so one user's cache entry is never
# served to another and a replaced PDF re-summarizes. Bounded + TTL'd; per-instance on serverless.
_summary_cache = _gen.TTLCache(maxsize=128, ttl_s=1800.0)


class PlannerError(Exception):
    """Raised when a summary or study plan cannot be generated."""


def _summary_prompt(pdf_name: str, pdf_text: str) -> str:
    """Build the TOPICS/SUMMARY analysis prompt for a single PDF."""
    content = pdf_text    # full text; llm_chain trims per provider (Groq only)
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


_SUMMARY_SYSTEM = "You are an educational assistant that summarizes lecture content effectively."


def _summarize_text(name: str, text: str, model: str | None = None):
    """Summarize one document's text (TOPICS/SUMMARY layout) -> (summary, llm_public)."""
    if not text or not text.strip():
        raise PlannerError(f"No extractable text in: {name}")
    summary, res = _gen.generate_text(_SUMMARY_SYSTEM, _summary_prompt(name, text),
                                      max_tokens=_SUMMARY_MAX_OUTPUT_TOKENS, error_cls=PlannerError,
                                      what=f"summary for {name}", temperature=0.4)
    return summary, _gen.public(res)


def _summarize_one(subject: str, pdf: str, model: str | None = None, owner: str | None = None):
    """Cached or freshly generated TOPICS/SUMMARY view for a subject PDF -> (summary, llm_public)."""
    try:
        key = (owner or "", subject, pdf, pdf_signature(subject, pdf, owner))
    except Exception:  # noqa: BLE001 — unknown signature: just don't cache
        key = None
    if key is not None:
        hit = _summary_cache.get(key)
        if hit is not None:
            return hit
    out = _summarize_text(pdf, read_pdf(subject, pdf, owner), model=model)
    if key is not None:
        _summary_cache.set(key, out)
    return out


def _run_parallel(jobs: list) -> list:
    """Run zero-arg callables with bounded concurrency, preserving order; first error re-raises."""
    if len(jobs) == 1:
        return [jobs[0]()]
    with ThreadPoolExecutor(max_workers=min(_MAX_WORKERS, len(jobs))) as pool:
        futures = [pool.submit(j) for j in jobs]
        return [f.result() for f in futures]


def _collect(names: list[str], results: list) -> tuple[dict[str, str], dict | None]:
    summaries = {n: r[0] for n, r in zip(names, results)}
    llm = next((r[1] for r in reversed(results) if r[1]), None)
    return summaries, llm


def summarize_pdfs_ex(subject: str, pdfs: list[str], model: str | None = None,
                      owner: str | None = None) -> tuple[dict[str, str], dict | None]:
    """Summarize each selected subject PDF -> ({pdf_name: summary}, llm_public of the last call)."""
    names = _gen.dedupe_names(pdfs, MAX_PDFS)
    if not names:
        raise PlannerError("No PDFs selected to summarize.")
    results = _run_parallel([lambda n=n: _summarize_one(subject, n, model, owner) for n in names])
    return _collect(names, results)


def summarize_pdfs(subject: str, pdfs: list[str], model: str | None = None, owner: str | None = None) -> dict[str, str]:
    """Back-compat: {pdf_name: summary} only (see :func:`summarize_pdfs_ex`)."""
    return summarize_pdfs_ex(subject, pdfs, model, owner)[0]


def summarize_named_texts_ex(items: list[tuple[str, str]], model: str | None = None):
    """Summarize uploaded (name, text) documents -> ({name: summary}, llm_public)."""
    items = [(n, t) for n, t in (items or []) if t and t.strip()][:MAX_PDFS]
    if not items:
        raise PlannerError("No documents provided to summarize.")
    names = [n for n, _ in items]
    results = _run_parallel([lambda n=n, t=t: _summarize_text(n, t, model=model) for n, t in items])
    return _collect(names, results)


def summarize_named_texts(items: list[tuple[str, str]], model: str | None = None) -> dict[str, str]:
    """Back-compat: {name: summary} only."""
    return summarize_named_texts_ex(items, model)[0]


def _days_until(target_date: str) -> tuple[datetime.date, datetime.date, int]:
    """Parse the target date and return (start, target, days) with 1 <= days <= _MAX_PLAN_DAYS."""
    raw = (target_date or "").strip()
    try:
        try:
            target = datetime.date.fromisoformat(raw[:10])
        except ValueError:
            target = date_parser.parse(raw).date()
    except (ValueError, OverflowError, TypeError) as exc:
        raise PlannerError(f"Could not parse target date: {target_date!r}") from exc

    start = datetime.date.today()
    days = max(1, (target - start).days)
    if days > _MAX_PLAN_DAYS:
        raise PlannerError(f"Target date is too far away; pick a date within {_MAX_PLAN_DAYS} days.")
    return start, target, days


def _schedule_format(days: int, start: datetime.date, item_hint: str) -> str:
    """The markdown layout instructions (daily blocks for short plans, weekly blocks for long ones)."""
    if days <= _DAILY_LIMIT:
        return (
            f"## Day 1: {start.isoformat()} ({_HOURS_PER_DAY}h)\n"
            f"- <{item_hint}>\n"
            "- <study technique>\n"
            "- <break / review timing>\n\n"
            "## Day 2: <date> (<hours>h)\n"
            "Continue for all days, keeping the plan progressive with regular reviews."
        )
    return (
        f"## Week 1: {start.isoformat()} to <date> ({_HOURS_PER_DAY}h/day)\n"
        f"- <{item_hint}>\n"
        "- <study technique>\n"
        "- <daily rhythm and review sessions>\n\n"
        "## Week 2: <dates> (<hours>h/day)\n"
        "Continue week by week until the target date (this plan spans many days, so group them by week), "
        "keeping it progressive with regular reviews."
    )


def _plan_prompt(
    subject: str,
    summaries: dict[str, str],
    start: datetime.date,
    target: datetime.date,
    days: int,
) -> str:
    """Build the study-plan prompt from the PDF summaries."""
    content_for_plan = "\n\n".join(
        f"PDF: {pdf_name}\nSummary: {summary}"
        for pdf_name, summary in summaries.items()
    )
    unit = "day-by-day" if days <= _DAILY_LIMIT else "week-by-week"
    return (
        f"Create a detailed {unit} study plan for {days} day(s), running "
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
        + _schedule_format(days, start, "topic to study with PDF reference")
    )


def _topic_plan_prompt(topic: str, start: datetime.date, target: datetime.date, days: int) -> str:
    """Build a study-plan prompt for a free topic (no documents)."""
    unit = "day-by-day" if days <= _DAILY_LIMIT else "week-by-week"
    return (
        f"Create a detailed {unit} study plan for {days} day(s), running from "
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
        + _schedule_format(days, start, "subtopic to study")
    )


_PLAN_SYSTEM: str = (
    "You are an educational expert who creates effective, realistic study plans "
    f"with at least {_HOURS_PER_DAY} hours of study time per day. Respond in clean markdown."
)


def _call_plan(prompt: str, model: str | None = None) -> tuple[str, dict | None]:
    """Run the study-plan prompt through the chain -> (markdown, llm_public)."""
    plan, res = _gen.generate_text(_PLAN_SYSTEM, prompt, max_tokens=_PLAN_MAX_OUTPUT_TOKENS,
                                   error_cls=PlannerError, what="study plan", temperature=0.5)
    return plan, _gen.public(res)


def _run_plan(label: str, summaries: dict[str, str], dates: tuple, model: str | None = None):
    start, target, days = dates
    return _call_plan(_plan_prompt(label, summaries, start, target, days), model)


def generate_plan_ex(subject: str, pdfs: list[str], target_date: str, model: str | None = None,
                     owner: str | None = None) -> tuple[str, dict | None]:
    """Markdown plan for selected subject PDFs -> ``(plan, llm_public of the plan call)``."""
    dates = _days_until(target_date)               # fail fast, before any LLM spend
    summaries, _ = summarize_pdfs_ex(subject, pdfs, model, owner)
    return _run_plan(subject, summaries, dates, model)


def generate_plan_from_texts_ex(label: str, items: list[tuple[str, str]], target_date: str,
                                model: str | None = None) -> tuple[str, dict | None]:
    """Plan from uploaded (name, text) documents -> ``(plan, llm_public)``."""
    dates = _days_until(target_date)
    summaries, _ = summarize_named_texts_ex(items, model)
    return _run_plan(label, summaries, dates, model)


def generate_plan_from_topic_ex(topic: str, target_date: str, model: str | None = None) -> tuple[str, dict | None]:
    """Plan to learn a free topic by the target date (no documents) -> ``(plan, llm_public)``."""
    topic = _gen.clean_topic(topic)
    if not topic:
        raise PlannerError("Enter a topic to plan for.")
    start, target, days = _days_until(target_date)
    return _call_plan(_topic_plan_prompt(topic, start, target, days), model)


def generate_plan(subject: str, pdfs: list[str], target_date: str, model: str | None = None, owner: str | None = None) -> str:
    """Back-compat: plan markdown only (see :func:`generate_plan_ex`)."""
    return generate_plan_ex(subject, pdfs, target_date, model, owner)[0]


def generate_plan_from_texts(label: str, items: list[tuple[str, str]], target_date: str, model: str | None = None) -> str:
    """Back-compat: plan markdown only."""
    return generate_plan_from_texts_ex(label, items, target_date, model)[0]


def generate_plan_from_topic(topic: str, target_date: str, model: str | None = None) -> str:
    """Back-compat: plan markdown only."""
    return generate_plan_from_topic_ex(topic, target_date, model)[0]
