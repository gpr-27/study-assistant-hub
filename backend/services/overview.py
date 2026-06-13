"""Chat overview service: a NotebookLM-style empty-state card for a new session.

When a Chat Assistant session opens on a subject or topic, the SPA shows a short
overview of what the materials cover plus a few tappable "starter question" chips.
This module produces that card from ONE bounded Groq call generated from the PDF
TEXT (never embeddings — the Gemini embedding quota is exhausted) and caches the
result per subject/topic so reopening a session is free.

Design rules:
  * NO embeddings / FAISS. We read the raw PDF text via ``backend.documents`` and
    clip it to a token budget, exactly like the other generation tools, so a single
    request stays within the free-tier per-minute limit.
  * ONE LLM call, bounded output (max_tokens). The model returns a small JSON object
    ``{"overview": "...", "questions": ["...", "...", "..."]}`` which we parse
    tolerantly (regex the first ``{...}``) and clamp to 3 questions.
  * Fully best-effort: this NEVER raises. Any failure (no key, unreadable PDFs,
    rate limit, bad JSON) degrades to ``overview="", questions=[]`` — and that empty
    result is STILL cached, so a broken subject isn't re-tried on every open.
"""
import json
import logging
import re
import threading

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from backend import config
from backend.documents import clip_to_tokens, list_pdf_names, pdf_signature, read_pdf

log = logging.getLogger(__name__)

# Read a few documents and clip to a single bounded slice — a few PDFs are plenty
# for an overview within the free-tier token budget (input + reserved output).
_MAX_PDFS: int = 3
_MAX_CONTENT_TOKENS: int = 4000
_MAX_OUTPUT_TOKENS: int = 500   # 2-3 sentences + 3 short questions is small
_MAX_QUESTIONS: int = 3
# Cap the cache so a guest-heavy stream of distinct subjects/topics can't grow it
# without bound on the 512MB box; oldest entries are evicted first (insertion order).
_MAX_CACHE: int = 256

_SYSTEM: str = (
    "You help a student get oriented before they start studying. Given the study "
    "material (or a topic), you write a short, inviting overview and a few specific "
    "starter questions. You always reply with a single valid JSON object and nothing else."
)

# In-process cache keyed per owner+subject or per topic, guarded by a lock. An empty
# (failed) result is cached too, so a broken subject is not re-generated on every open.
_cache: dict[str, dict] = {}
_cache_lock = threading.Lock()

_EMPTY: dict = {"overview": "", "questions": []}


def _subject_fingerprint(subject: str, owner: str) -> str:
    """A (mtime,size) fingerprint over the subject's PDFs so the cache misses when
    the materials change (a re-upload regenerates the overview, never serves stale)."""
    sig: list = []
    for name in list_pdf_names(subject, owner)[:_MAX_PDFS]:
        try:
            sig.append((name, pdf_signature(subject, name, owner)))
        except Exception:  # noqa: BLE001 — a missing/unreadable file just drops from the print
            continue
    return repr(sig)


def _cache_key(*, subject: str | None, topic: str | None, owner: str) -> str:
    if subject:
        return f"{owner}::subject::{subject}::{_subject_fingerprint(subject, owner)}"
    return f"topic::{topic}"


def _cache_get(key: str) -> dict | None:
    with _cache_lock:
        cached = _cache.get(key)
    # Deep-copy the questions list so a caller mutating it can't corrupt the cache.
    return None if cached is None else {**cached, "questions": list(cached["questions"])}


def _cache_put(key: str, card: dict) -> None:
    with _cache_lock:
        _cache[key] = {**card, "questions": list(card["questions"])}
        while len(_cache) > _MAX_CACHE:
            del _cache[next(iter(_cache))]   # dicts preserve insertion order


def _subject_body(subject: str, owner: str) -> str:
    """Concatenate (and clip) up to a few of the subject's PDFs into one blob."""
    parts: list[str] = []
    for pdf in list_pdf_names(subject, owner)[:_MAX_PDFS]:
        try:
            parts.append(f"## {pdf}\n{read_pdf(subject, pdf, owner)}")
        except Exception:  # noqa: BLE001 — skip unreadable PDFs, keep the rest
            continue
    return clip_to_tokens("\n\n".join(parts), _MAX_CONTENT_TOKENS)


def _prompt(label: str, body: str, is_subject: bool) -> str:
    spec = (
        "Return ONLY a single JSON object of exactly this form:\n"
        '{"overview": "<2-3 sentence markdown summary>", '
        '"questions": ["<question>", "<question>", "<question>"]}\n'
        "The overview is 2-3 sentences that say what the material covers; use "
        "**bold** markdown around 2-4 key terms. Then write exactly 3 specific, "
        "answerable starter questions a student could tap to begin. No extra keys, "
        "no commentary, no markdown fences."
    )
    if is_subject and body:
        return (
            f"Study material for “{label}” is below.\n\n{spec}\n\nMATERIAL:\n{body}"
        )
    return f"The student wants to learn about: {label}.\n\n{spec}"


def _parse(raw: str) -> dict:
    """Tolerantly extract {overview, questions} from the model's reply."""
    match = re.search(r"\{.*\}", raw or "", re.DOTALL)
    data = json.loads(match.group(0)) if match else {}
    if not isinstance(data, dict):
        return dict(_EMPTY)
    overview = str(data.get("overview", "") or "").strip()
    raw_qs = data.get("questions")
    questions: list[str] = []
    if isinstance(raw_qs, list):
        for q in raw_qs:
            text = str(q or "").strip()
            if text:
                questions.append(text)
    return {"overview": overview, "questions": questions[:_MAX_QUESTIONS]}


def _generate(label: str, body: str, is_subject: bool, model: str | None) -> dict:
    """Run the single bounded Groq call and parse it. Raises on any failure."""
    llm = ChatGroq(model=config.model_for("chat", model), temperature=0.3, max_tokens=_MAX_OUTPUT_TOKENS)
    raw = str(llm.invoke([
        SystemMessage(content=_SYSTEM),
        HumanMessage(content=_prompt(label, body, is_subject)),
    ]).content)
    return _parse(raw)


def build_overview(
    *,
    subject: str | None = None,
    topic: str | None = None,
    owner: str = "local",
    model: str | None = None,
    refresh: bool = False,
) -> dict:
    """Return a NotebookLM-style overview card for a subject or topic session.

    Returns ``{"label", "sources", "overview", "questions"}``:
      * subject mode: ``sources`` is the subject's total PDF count; the overview is
        generated from up to a few of those PDFs' text.
      * topic mode: ``sources`` is 0; the overview is generated from the topic alone.

    Best-effort by contract — NEVER raises. On any failure ``overview`` is "" and
    ``questions`` is ``[]``, and that empty result is cached so a broken subject is
    not re-generated on every open. ``refresh=True`` bypasses the cache.
    """
    is_subject = bool(subject)
    label = (subject if is_subject else topic) or ""
    key = _cache_key(subject=subject, topic=topic, owner=owner)

    if not refresh:
        cached = _cache_get(key)
        if cached is not None:
            return cached

    sources = 0
    body = ""
    result = dict(_EMPTY)
    try:
        if is_subject:
            sources = len(list_pdf_names(subject, owner))
            body = _subject_body(subject, owner)
        config.require_groq_key()
        result = _generate(label, body, is_subject, model)
    except Exception:  # noqa: BLE001 — overview is best-effort; degrade to empty
        log.debug("Overview generation failed for %r; serving empty card.", key, exc_info=True)
        result = dict(_EMPTY)

    card = {"label": label, "sources": sources, **result}
    _cache_put(key, card)
    return card
