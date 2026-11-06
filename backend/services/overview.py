"""Chat overview service: a NotebookLM-style empty-state card for a new session.

When a Chat Assistant session opens on a subject or topic, the SPA shows a short overview of
what the materials cover plus a few tappable "starter question" chips.

Request-path rules (this is hit on every session open):
  1. BUNDLED subjects ship a pre-baked ``data/index/<subject>/overview.json`` (written by
     ``scripts/prebake_overviews.py`` at build time). Reading it is a pure file read — no PDF
     parse, no embedding, NO LLM call. That is the normal case.
  2. Otherwise a bounded in-process TTL cache (per owner + subject/topic + PDF fingerprint).
  3. Only when both miss is ONE bounded LLM call made through ``backend.llm_chain`` (JSON mode,
     ≤ ``_GEN_DEADLINE_S``), de-duplicated across concurrent requests for the same key. A failed
     generation is cached briefly (negative cache) so a provider outage doesn't make every
     session-open pay the timeout, and ``refresh`` is rate-limited per key.

The function never raises: any failure degrades to ``overview="", questions=[]``.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from collections import OrderedDict
from pathlib import Path

from backend import deadline, llm_chain, privacy
from backend.config import ROOT_DIR
from backend.documents import list_pdf_names, pdf_signature, read_pdf, subject_internal_key
from backend.services.llm_json import extract_json_object

log = logging.getLogger(__name__)

_PREBAKE_NAME = "overview.json"


def _bundled_index_dir() -> Path:
    try:
        from backend import vector_store

        return Path(vector_store.BUNDLED_INDEX_DIR)
    except Exception:  # noqa: BLE001
        return ROOT_DIR / "data" / "index"


PREBAKED_UNKNOWN_LABEL = "Model: not recorded (pre-generated overview)"


def _prebaked_llm(stored: object) -> dict:
    """Public ``llm`` block for a shipped overview: the label the prebake script recorded, else an
    honest "not recorded" label (older files predate model tracking) — never a made-up model."""
    if isinstance(stored, dict) and str(stored.get("label") or "").startswith("Model:"):
        return {"label": str(stored["label"])}
    if isinstance(stored, str) and stored.startswith("Model:"):
        return {"label": stored}
    return {"label": PREBAKED_UNKNOWN_LABEL}


def _load_prebaked(subject: str, owner: str | None) -> dict | None:
    """``{overview, questions, sources}`` from a shipped overview.json (bundled subjects only)."""
    try:
        key = subject_internal_key(subject, owner)
        if key != subject:          # a user-owned subject: never serve (or probe) a bundled file
            return None
        path = _bundled_index_dir() / key / _PREBAKE_NAME
        data = json.loads(path.read_text(encoding="utf-8"))
        overview = str(data.get("overview", "") or "").strip()
        questions = [str(q).strip() for q in (data.get("questions") or []) if str(q).strip()][:_MAX_QUESTIONS]
        if not overview and not questions:
            return None
        return {"overview": overview, "questions": questions, "sources": int(data.get("sources") or 0),
                "llm": _prebaked_llm(data.get("llm"))}
    except Exception:  # noqa: BLE001 — absent/malformed prebake just falls back to cache/generation
        return None


_MAX_PDFS: int = 3
_MAX_OUTPUT_TOKENS: int = 700
_MAX_QUESTIONS: int = 3
_GEN_DEADLINE_S: float = 18.0
_MAX_CACHE: int = 256
_OK_TTL_S: float = 6 * 3600.0
_FAIL_TTL_S: float = 30.0
_REFRESH_MIN_INTERVAL_S: float = 30.0

_SYSTEM: str = (
    "You help a student get oriented before they start studying. Given the study "
    "material (or a topic), you write a short, inviting overview and a few specific "
    "starter questions. You always reply with a single valid JSON object and nothing else."
)

_EMPTY: dict = {"overview": "", "questions": []}

_cache: "OrderedDict[str, tuple[float, dict]]" = OrderedDict()     # key -> (expires_at, card)
_last_gen: "OrderedDict[str, float]" = OrderedDict()               # key -> last generation start
_inflight: dict[str, threading.Event] = {}
_lock = threading.Lock()


def _subject_fingerprint(subject: str, owner: str) -> str:
    sig: list = []
    for name in list_pdf_names(subject, owner)[:_MAX_PDFS]:
        try:
            sig.append((name, pdf_signature(subject, name, owner)))
        except Exception:  # noqa: BLE001
            continue
    return repr(sig)


def _cache_key(*, subject: str | None, topic: str | None, owner: str) -> str:
    if subject:
        return f"{owner}::subject::{subject}::{_subject_fingerprint(subject, owner)}"
    return f"{owner}::topic::{topic}"


def _cache_get(key: str) -> dict | None:
    with _lock:
        hit = _cache.get(key)
        if hit is None:
            return None
        if time.monotonic() >= hit[0]:
            del _cache[key]
            return None
        _cache.move_to_end(key)
        card = hit[1]
    return {**card, "questions": list(card["questions"])}


def _cache_put(key: str, card: dict, ttl: float) -> None:
    with _lock:
        _cache[key] = (time.monotonic() + ttl, {**card, "questions": list(card["questions"])})
        _cache.move_to_end(key)
        while len(_cache) > _MAX_CACHE:
            _cache.popitem(last=False)


def _subject_body(subject: str, owner: str) -> str:
    parts: list[str] = []
    for pdf in list_pdf_names(subject, owner)[:_MAX_PDFS]:
        try:
            parts.append(f"## {pdf}\n{read_pdf(subject, pdf, owner)}")
        except Exception:  # noqa: BLE001 — skip unreadable PDFs, keep the rest
            continue
    return "\n\n".join(parts)    # full text; llm_chain trims per provider


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
        return f"Study material for “{label}” is below.\n\n{spec}\n\nMATERIAL:\n{body}"
    return f"The student wants to learn about: {label}.\n\n{spec}"


def _parse(raw: str) -> dict:
    """Tolerantly extract {overview, questions} from the model's reply."""
    data = extract_json_object(raw)
    overview = str(data.get("overview", "") or "").strip()
    raw_qs = data.get("questions")
    questions: list[str] = []
    if isinstance(raw_qs, list):
        for q in raw_qs:
            text = str(q or "").strip()
            if text:
                questions.append(text)
    return {"overview": overview, "questions": questions[:_MAX_QUESTIONS]}


def _generate(label: str, body: str, is_subject: bool) -> tuple[dict, dict | None]:
    """The single bounded chain call. Returns (parsed card parts, public llm). Raises on failure."""
    res = llm_chain.chat(_SYSTEM, [{"role": "user", "content": _prompt(label, body, is_subject)}],
                         max_tokens=_MAX_OUTPUT_TOKENS, temperature=0.3, json_mode=True,
                         sensitive=privacy.current(), deadline_s=deadline.clamp(_GEN_DEADLINE_S))
    return _parse(res.text), res.to_public()


def build_overview(
    *,
    subject: str | None = None,
    topic: str | None = None,
    owner: str = "local",
    model: str | None = None,
    refresh: bool = False,
) -> dict:
    """Return ``{"label", "sources", "overview", "questions", "llm"}``. NEVER raises.

    ``llm`` is the public chain label of the generating model (for a pre-baked card: the label stored
    in overview.json, or a "not recorded" label), or ``None`` for an empty/failed card. ``model`` is accepted for back-compat and ignored.
    """
    is_subject = bool(subject)
    label = (subject if is_subject else topic) or ""

    # 1) Pre-baked (pure file read; no LLM) — skipped on an explicit refresh.
    if is_subject and not refresh:
        pre = _load_prebaked(subject, owner)
        if pre is not None:
            try:
                sources = pre["sources"] or len(list_pdf_names(subject, owner))
            except Exception:  # noqa: BLE001
                sources = pre["sources"]
            return {"label": label, "sources": sources, "overview": pre["overview"],
                    "questions": pre["questions"], "llm": pre["llm"]}

    try:
        key = _cache_key(subject=subject, topic=topic, owner=owner)
    except Exception:  # noqa: BLE001 — unknown/forbidden subject → empty card, not an error
        return {"label": label, "sources": 0, **_EMPTY, "llm": None}

    # 2) Cache.
    if not refresh:
        cached = _cache_get(key)
        if cached is not None:
            return cached

    # 3) Generate once (single-flight), bounded.
    now = time.monotonic()
    with _lock:
        if refresh and now - _last_gen.get(key, -1e9) < _REFRESH_MIN_INTERVAL_S:
            refresh = False                     # too soon: serve whatever we have instead of burning the LLM
        waiter = _inflight.get(key)
        if waiter is None:
            _inflight[key] = threading.Event()
            _last_gen[key] = now
            while len(_last_gen) > _MAX_CACHE:
                _last_gen.popitem(last=False)
    if waiter is not None:
        waiter.wait(_GEN_DEADLINE_S + 2)
        cached = _cache_get(key)
        return cached if cached is not None else {"label": label, "sources": 0, **_EMPTY, "llm": None}
    try:
        if not refresh:
            cached = _cache_get(key)        # a refresh-throttled call may still have a card
            if cached is not None:
                return cached
        sources, body, llm_public = 0, "", None
        result = dict(_EMPTY)
        with privacy.scope(privacy.for_subject(subject) if is_subject else False):   # user subject => private
            try:
                if is_subject:
                    sources = len(list_pdf_names(subject, owner))
                    body = _subject_body(subject, owner)
                result, llm_public = _generate(label, body, is_subject)
            except Exception:  # noqa: BLE001 — overview is best-effort; degrade to empty
                log.info("Overview generation failed for %r; serving empty card.", label, exc_info=True)
                result, llm_public = dict(_EMPTY), None
        card = {"label": label, "sources": sources, **result, "llm": llm_public}
        ok = bool(result["overview"] or result["questions"])
        _cache_put(key, card, _OK_TTL_S if ok else _FAIL_TTL_S)      # failures are cached only briefly
        return card
    finally:
        _release(key)


def _release(key: str) -> None:
    with _lock:
        ev = _inflight.pop(key, None)
    if ev is not None:
        ev.set()
