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


def _rewrite(
    llm,
    question: str,
    history_messages: list,
    *,
    previous: list[str] | None = None,
    missing: str | None = None,
) -> list[str]:
    """Produce up to ``AGENTIC_RAG_QUERIES`` search queries for ``question``.

    On the corrective pass, ``previous``/``missing`` steer it toward *different* angles
    and the still-missing information. Falls back to ``[question]`` on any failure.
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

    messages = [SystemMessage(content=_REWRITE_SYSTEM), *history_messages, HumanMessage(content=instruction)]
    try:
        data = _extract_json(_invoke(llm, messages))
    except APIError:
        raise
    except Exception:  # noqa: BLE001 — rewriting is best-effort
        log.debug("Query rewrite failed; using the raw question.", exc_info=True)
        return [question]

    raw = data.get("queries") if isinstance(data, dict) else None
    queries: list[str] = []
    if isinstance(raw, list):
        for q in raw:
            if isinstance(q, str) and q.strip() and q.strip() not in queries:
                queries.append(q.strip())
    return queries[:n] or [question]


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


def _format_context(docs: list) -> str:
    """Join document contents into a single context block."""
    return "\n\n".join((getattr(d, "page_content", "") or "").strip() for d in docs if getattr(d, "page_content", "").strip())


def _grade(llm, question: str, context: str) -> tuple[bool, str]:
    """Return (sufficient, missing). On any failure, assume sufficient (degrade to naive RAG)."""
    if not context.strip():
        return False, "no context retrieved"
    instruction = (
        f"Question: {question}\n\n"
        f"Context passages:\n{context}\n\n"
        "Can the question be fully answered using ONLY these passages? "
        'Return ONLY JSON: {"sufficient": true|false, "missing": "what is missing, if any"}'
    )
    messages = [SystemMessage(content=_GRADE_SYSTEM), HumanMessage(content=instruction)]
    try:
        data = _extract_json(_invoke(llm, messages))
    except APIError:
        raise
    except Exception:  # noqa: BLE001 — grading is best-effort
        log.debug("Context grading failed; treating context as sufficient.", exc_info=True)
        return True, ""
    if not isinstance(data, dict) or "sufficient" not in data:
        return True, ""  # unparseable grade → don't loop; answer with what we have
    return bool(data.get("sufficient")), str(data.get("missing") or "")


def _generate(llm, persona: str, history_messages: list, question: str, context: str) -> str:
    """Produce the final answer grounded in the gathered context."""
    user = (
        f"Context:\n{context}\n\nQuestion: {question}"
        if context.strip()
        else f"Question: {question}\n\n(No relevant context was found in the notes.)"
    )
    messages = [SystemMessage(content=persona), *history_messages, HumanMessage(content=user)]
    return _invoke(llm, messages)


def answer(retriever, llm, question: str, history_messages: list, persona: str, *, max_iters: int | None = None) -> str:
    """Run the agentic retrieval loop and return a grounded answer.

    ``history_messages`` is the already-rendered list of prior Human/AI messages (so this
    module doesn't need to know how the caller stores history). Reuses the session ``llm``
    (a ChatGroq) for every step.
    """
    rounds = max_iters or config.AGENTIC_RAG_MAX_ITERS
    docs: list = []
    seen: set[str] = set()
    queries = _rewrite(llm, question, history_messages)

    for attempt in range(rounds):
        for doc in _retrieve(retriever, queries):
            key = (getattr(doc, "page_content", "") or "").strip()
            if key and key not in seen:
                seen.add(key)
                docs.append(doc)
        context = _format_context(docs)
        sufficient, missing = _grade(llm, question, context)
        log.info(
            "Agentic RAG round %d/%d: %d queries, %d docs, sufficient=%s.",
            attempt + 1, rounds, len(queries), len(docs), sufficient,
        )
        if sufficient or attempt == rounds - 1:
            break
        queries = _rewrite(llm, question, history_messages, previous=queries, missing=missing)
        if not queries:
            break

    return _generate(llm, persona, history_messages, question, _format_context(docs))
