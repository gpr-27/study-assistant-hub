"""Agentic RAG for chat — an LLM-driven retrieval loop.

Classic ("naive") RAG does one fixed step: embed the question, retrieve top-k, stuff
the chunks into the prompt, generate once. This module replaces that with a small
agentic loop that *reasons about its own retrieval*:

    rewrite ─▶ retrieve ─▶ grade ─┬─(sufficient)─▶ generate
       ▲                          │
       └────(insufficient)────────┘   (corrective re-retrieve, bounded)

Steps
  1. REWRITE  — turn the (history-aware) question into one or more focused search
                queries, resolving follow-ups/pronouns and decomposing multi-part
                questions ("compare X and Y" → query X, query Y).
  2. RETRIEVE — run every query through the session retriever; merge + de-dupe.
  3. GRADE    — ask the LLM whether the gathered context can actually answer the
                question; if not, what's missing.
  4. CORRECT  — if insufficient and rounds remain, rewrite again using the gap
                feedback and retrieve more; otherwise stop.
  5. GENERATE — answer grounded ONLY in the gathered context (the persona already
                instructs the model to say it doesn't know when unsupported).

Design rules:
  * Every auxiliary step (rewrite/grade) is best-effort: on malformed output it
    falls back so the loop degrades to — never worse than — classic RAG.
  * Groq rate-limit / API errors are re-raised so the global handlers render a clean
    429/502 instead of silently looping.
  * Self-contained and dependency-light (no graph framework); the session's existing
    ChatGroq instance is reused for every call, so no new model/keys are needed.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from groq import APIError
from langchain_core.messages import HumanMessage, SystemMessage

from backend import config

log = logging.getLogger(__name__)

_REWRITE_SYSTEM = (
    "You rewrite a student's question into focused search queries for retrieving "
    "passages from their course notes. Resolve pronouns/follow-ups using the prior "
    "turns, and split multi-part questions into separate queries. Always reply with "
    "valid JSON."
)
_GRADE_SYSTEM = (
    "You judge whether a set of retrieved context passages contains enough information "
    "to answer the student's question. Be strict: partial or tangential context is not "
    "sufficient. Always reply with valid JSON."
)


def _invoke(llm, messages) -> str:
    """Call the chat model and return its text content."""
    return str(llm.invoke(messages).content)


def _extract_json(text: str) -> dict:
    """Parse the first JSON object out of a model reply (tolerant of stray prose/fences)."""
    if not text:
        return {}
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except (ValueError, TypeError):
            return {}
    return {}


def _prompt_text(system: str, user: str) -> str:
    """Render a system+user message pair as a single readable prompt for the trace UI."""
    return f"SYSTEM:\n{system}\n\nUSER:\n{user}"


def _rewrite(
    llm,
    question: str,
    history_messages: list,
    *,
    previous: list[str] | None = None,
    missing: str | None = None,
) -> tuple[list[str], str]:
    """Produce up to ``AGENTIC_RAG_QUERIES`` search queries for ``question``.

    On the corrective pass, ``previous``/``missing`` steer it toward *different* angles
    and the still-missing information. Returns ``(queries, prompt)`` — the prompt is
    surfaced in the agentic trace. Falls back to ``[question]`` on any failure.
    """
    n = config.AGENTIC_RAG_QUERIES
    instruction = (
        f"Question: {question}\n\n"
        f"Write 1 to {n} concise search queries (keywords/short phrases) that would "
        "retrieve passages needed to answer it."
    )
    if previous:
        instruction += (
            "\n\nThe previous queries below did NOT retrieve enough information"
            + (f" (missing: {missing})." if missing else ".")
            + " Try DIFFERENT wording, synonyms, or sub-questions.\nPrevious queries: "
            + "; ".join(previous)
        )
    instruction += '\n\nReturn ONLY JSON: {"queries": ["...", "..."]}'
    prompt = _prompt_text(_REWRITE_SYSTEM, instruction)

    messages = [SystemMessage(content=_REWRITE_SYSTEM), *history_messages, HumanMessage(content=instruction)]
    try:
        data = _extract_json(_invoke(llm, messages))
    except APIError:
        raise
    except Exception:  # noqa: BLE001 — rewriting is best-effort
        log.debug("Query rewrite failed; using the raw question.", exc_info=True)
        return [question], prompt

    raw = data.get("queries") if isinstance(data, dict) else None
    queries: list[str] = []
    if isinstance(raw, list):
        for q in raw:
            if isinstance(q, str) and q.strip() and q.strip() not in queries:
                queries.append(q.strip())
    return (queries[:n] or [question]), prompt


def _retrieve(retriever, queries: list[str]) -> list:
    """Retrieve for every query and return merged, de-duplicated documents."""
    seen: set[str] = set()
    docs: list = []
    for query in queries:
        try:
            results = retriever.invoke(query)
        except APIError:
            raise
        except Exception:  # noqa: BLE001 — one bad query must not abort the round
            log.debug("Retrieval failed for query %r.", query, exc_info=True)
            continue
        for doc in results:
            content = getattr(doc, "page_content", "") or ""
            key = content.strip()
            if key and key not in seen:
                seen.add(key)
                docs.append(doc)
    return docs


# Hard cap on the assembled retrieval context. The corrective loop can merge
# 5 chunks × 3 queries × 2 rounds ≈ 30+ chunks — far more than Groq's free-tier
# 6,000 tokens-per-MINUTE budget, which counts the WHOLE request (a 7,700-token
# prompt is rejected outright with 413 "Request too large"). 3,000 tokens of
# context + persona + history + reserved output stays comfortably under it.
_MAX_CONTEXT_TOKENS = 3000


def _format_context(docs: list) -> str:
    """Join document contents into a single context block (token-capped)."""
    from backend.documents import clip_to_tokens

    joined = "\n\n".join(
        (getattr(d, "page_content", "") or "").strip()
        for d in docs if getattr(d, "page_content", "").strip()
    )
    return clip_to_tokens(joined, _MAX_CONTEXT_TOKENS)


def passage_view(docs: list, *, limit: int = 12, snippet: int = 320) -> list[dict]:
    """Project retrieved documents into a compact, JSON-safe shape for the trace UI.

    Each entry: ``{source, page, text}`` — the source PDF filename, 1-based page
    number (when known), and a truncated text snippet. Shared with the classic-RAG
    path in the chat service so both render passages identically.
    """
    out: list[dict] = []
    for d in docs[:limit]:
        meta = getattr(d, "metadata", {}) or {}
        src = meta.get("source", "") or ""
        page = meta.get("page")
        text = (getattr(d, "page_content", "") or "").strip()
        out.append({
            "source": Path(src).name if src else "—",
            "page": (page + 1) if isinstance(page, int) else None,
            "text": text[:snippet] + ("…" if len(text) > snippet else ""),
        })
    return out


def _grade(llm, question: str, context: str) -> tuple[bool, str, str]:
    """Return ``(sufficient, missing, prompt)``. On any failure, assume sufficient.

    The prompt is surfaced in the trace. A truncated context is shown there to keep
    the grading prompt readable.
    """
    instruction = (
        f"Question: {question}\n\n"
        f"Context passages:\n{context}\n\n"
        "Can the question be fully answered using ONLY these passages? "
        'Return ONLY JSON: {"sufficient": true|false, "missing": "what is missing, if any"}'
    )
    prompt = _prompt_text(_GRADE_SYSTEM, instruction)
    if not context.strip():
        return False, "no context retrieved", prompt
    messages = [SystemMessage(content=_GRADE_SYSTEM), HumanMessage(content=instruction)]
    try:
        data = _extract_json(_invoke(llm, messages))
    except APIError:
        raise
    except Exception:  # noqa: BLE001 — grading is best-effort
        log.debug("Context grading failed; treating context as sufficient.", exc_info=True)
        return True, "", prompt
    if not isinstance(data, dict) or "sufficient" not in data:
        return True, "", prompt  # unparseable grade → don't loop; answer with what we have
    return bool(data.get("sufficient")), str(data.get("missing") or ""), prompt


def _generate_user(question: str, context: str) -> str:
    """Build the user message for the final generation step."""
    return (
        f"Context:\n{context}\n\nQuestion: {question}"
        if context.strip()
        else f"Question: {question}\n\n(No relevant context was found in the notes.)"
    )


def _generate(llm, persona: str, history_messages: list, question: str, context: str) -> str:
    """Produce the final answer grounded in the gathered context."""
    messages = [SystemMessage(content=persona), *history_messages, HumanMessage(content=_generate_user(question, context))]
    return _invoke(llm, messages)


def answer(
    retriever, llm, question: str, history_messages: list, persona: str, *,
    max_iters: int | None = None, on_step=None, cancel=None,
) -> tuple[str, list[dict], list[str]]:
    """Run the agentic retrieval loop and return ``(answer, steps, sources)``.

    ``steps`` is a list of structured trace events (one dict per step), each with a
    ``kind`` (rewrite | retrieve | grade | generate), a human ``title``, and the real
    internals for that step — the queries written, the passages retrieved (source,
    page, snippet), the grade verdict + what's missing, and the prompt sent to the
    LLM — so the UI can show exactly what happened, not just a summary.

    ``history_messages`` is the already-rendered list of prior Human/AI messages.
    Reuses the session ``llm`` (a ChatGroq) for every step.

    ``on_step`` is an optional callback invoked with each step dict at the moment it
    is appended — including the ``generate`` step, which is emitted BEFORE the slow
    final LLM call so a streaming caller can show progress while the answer is being
    written. It defaults to a no-op, so the return value and behaviour are identical
    for non-streaming callers.
    """
    rounds = max_iters or config.AGENTIC_RAG_MAX_ITERS
    docs: list = []
    seen: set[str] = set()
    steps: list[dict] = []

    def _emit(step: dict) -> None:
        """Record a step and notify the (optional) streaming callback."""
        steps.append(step)
        if on_step is not None:
            on_step(step)

    queries, rewrite_prompt = _rewrite(llm, question, history_messages)
    q_word = "query" if len(queries) == 1 else "queries"
    _emit({
        "kind": "rewrite",
        "title": f"Rewrote your question into {len(queries)} search {q_word}",
        "queries": queries,
        "prompt": rewrite_prompt,
    })

    for attempt in range(rounds):
        if cancel is not None and cancel.is_set():
            break   # client disconnected / pressed Stop → don't burn more rate-limited LLM calls
        round_docs = list(_retrieve(retriever, queries))
        round_new: list = []
        for doc in round_docs:
            key = (getattr(doc, "page_content", "") or "").strip()
            if key and key not in seen:
                seen.add(key)
                docs.append(doc)
                round_new.append(doc)
        n_new = len(round_new)
        _emit({
            "kind": "retrieve",
            "title": (
                f"Retrieved {n_new} new passage{'' if n_new == 1 else 's'}"
                + (f" ({len(docs)} total)" if len(docs) != n_new else "")
            ),
            "round": attempt + 1,
            "queries": list(queries),
            "passages": passage_view(round_new),
        })

        context = _format_context(docs)
        # On the final round there is no corrective round left to act on the
        # verdict, so grading would be a wasted LLM call (we answer with what we
        # have either way). Skip it — saving one rate-limited Groq call per
        # message, which with the default single-round loop is EVERY message.
        if attempt == rounds - 1:
            log.info(
                "Agentic RAG round %d/%d: %d queries, %d docs, final (grade skipped).",
                attempt + 1, rounds, len(queries), len(docs),
            )
            _emit({
                "kind": "grade", "verdict": "final",
                "title": "Answering with the retrieved context",
                "missing": "", "prompt": "",
            })
            break
        sufficient, missing, grade_prompt = _grade(llm, question, context)
        log.info(
            "Agentic RAG round %d/%d: %d queries, %d docs, sufficient=%s.",
            attempt + 1, rounds, len(queries), len(docs), sufficient,
        )
        if sufficient:
            _emit({
                "kind": "grade", "verdict": "sufficient",
                "title": "Context verified — enough to answer",
                "missing": "", "prompt": grade_prompt,
            })
            break
        _emit({
            "kind": "grade", "verdict": "insufficient",
            "title": "Context looks insufficient — searching again",
            "missing": missing, "prompt": grade_prompt,
        })
        queries, rewrite_prompt = _rewrite(llm, question, history_messages, previous=queries, missing=missing)
        _emit({
            "kind": "rewrite",
            "title": f"Reformulated into {len(queries)} new {'query' if len(queries) == 1 else 'queries'}",
            "queries": queries,
            "missing": missing,
            "prompt": rewrite_prompt,
        })
        if not queries:
            break

    final_context = _format_context(docs)
    _emit({
        "kind": "generate",
        "title": "Writing the grounded answer",
        "passages_used": len(docs),
        "context_chars": len(final_context),
        # Show the prompt skeleton without re-dumping the (already-listed) passages.
        "prompt": _prompt_text(
            persona,
            f"Context:\n[{len(docs)} retrieved passage(s), {len(final_context)} chars — shown above]"
            f"\n\nQuestion: {question}",
        ),
    })
    if cancel is not None and cancel.is_set():
        return "", steps, []   # cancelled before the (most expensive) final generation
    final_answer = _generate(llm, persona, history_messages, question, final_context)

    seen_names: set[str] = set()
    sources: list[str] = []
    for doc in docs:
        src = (getattr(doc, "metadata", {}) or {}).get("source", "")
        if src:
            name = Path(src).name
            if name not in seen_names:
                seen_names.add(name)
                sources.append(name)

    return final_answer, steps, sources
