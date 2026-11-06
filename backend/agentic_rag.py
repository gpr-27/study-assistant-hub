"""Agentic RAG for chat — retrieval that reasons about itself, with a fast default path.

    rewrite ─▶ retrieve ─▶ grade ─┬─(sufficient)─▶ generate
       ▲                          │
       └────(insufficient)────────┘   (corrective re-retrieve, bounded)

Steps
  1. REWRITE  — turn the (history-aware) question into focused search queries.
  2. RETRIEVE — run every query through the session retriever; merge + de-dupe.
  3. GRADE    — ask the LLM whether the gathered context can answer the question.
  4. CORRECT  — if insufficient and rounds remain, rewrite with the gap feedback and re-retrieve.
  5. GENERATE — answer grounded ONLY in the gathered context (done by the caller: the chat
                service streams it token by token through ``backend.llm_chain.astream_chat``).

Time-to-first-token is the product metric, so the DEFAULT path (``AGENTIC_RAG_MAX_ITERS == 1``,
no ``AGENTIC_RAG_LLM_REWRITE``) makes ZERO LLM calls before generation: the rewrite step is a
deterministic heuristic (follow-ups/pronouns are resolved against the previous question) and
grading is skipped (with one round it could not change the outcome). Set
``AGENTIC_RAG_MAX_ITERS>1`` or ``AGENTIC_RAG_LLM_REWRITE=1`` to bring back the LLM rewrite/grade
loop; it then goes through ``backend.llm_chain`` (JSON mode) like everything else.

Design rules:
  * Every auxiliary LLM step is best-effort: malformed output degrades to — never worse than —
    classic RAG. ``LLMChainError`` (every provider failed) is re-raised so the caller can answer
    503 instead of looping.
  * History is passed as plain ``[{"role": "user"|"assistant", "content": str}]`` dicts.
"""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import NamedTuple

from backend import config, llm_chain, privacy
from backend.llm_chain import LLMChainError
from backend.services.llm_json import extract_json_object

log = logging.getLogger(__name__)

_REWRITE_SYSTEM = (
    "You rewrite a student's question into focused search queries for retrieving "
    "passages from their course notes. Resolve pronouns/follow-ups using the prior "
    "turns, and split multi-part questions into separate queries. Always reply with "
    "valid JSON."
)
_GRADE_SYSTEM = (
    "You judge whether a set of retrieved context passages contains enough information "
    "to answer the student's question. Mark sufficient when the key facts can be derived "
    "from the passages (definitions, formulas, worked examples), even if wording differs "
    "from the question. Mark insufficient only when the answer cannot be derived at all. "
    "Always reply with valid JSON."
)

# Output cap for the small auxiliary calls (the chain raises it to its own floor where needed).
_AUX_MAX_TOKENS = 600


def llm_rewrite_enabled() -> bool:
    """True when the LLM-driven rewrite should run even in single-round mode (opt-in)."""
    return os.environ.get("AGENTIC_RAG_LLM_REWRITE", "").strip().lower() in {"1", "true", "yes", "on"}


def _extract_json(text: str) -> dict:
    """Parse the first JSON object out of a model reply (tolerant of stray prose/fences)."""
    return extract_json_object(text)


def _prompt_text(system: str, user: str) -> str:
    """Render a system+user message pair as a single readable prompt for the trace UI."""
    return f"SYSTEM:\n{system}\n\nUSER:\n{user}"


# --------------------------------------------------------------------------------------
# Rewrite
# --------------------------------------------------------------------------------------
_PRONOUN_RE = re.compile(
    r"\b(it|its|this|that|these|those|they|them|their|he|she|his|her|above|previous|earlier|"
    r"same|former|latter|one|ones)\b", re.IGNORECASE)


def heuristic_queries(question: str, history_messages: list[dict]) -> list[str]:
    """Deterministic, zero-latency rewrite.

    The raw question is always the first query. For a short or pronoun-heavy follow-up, a second
    query prefixes the previous user question so retrieval still sees the topic.
    """
    q = (question or "").strip()
    queries = [q] if q else []
    prev = next((m["content"] for m in reversed(history_messages or [])
                 if m.get("role") == "user" and (m.get("content") or "").strip()), "")
    if q and prev and (len(q.split()) <= 8 or _PRONOUN_RE.search(q)):
        merged = f"{prev.strip()[:200]} {q}"
        if merged not in queries:
            queries.append(merged)
    return queries[: max(1, config.AGENTIC_RAG_QUERIES)]


def _rewrite(
    question: str,
    history_messages: list[dict],
    *,
    previous: list[str] | None = None,
    missing: str | None = None,
) -> tuple[list[str], str]:
    """LLM rewrite into up to ``AGENTIC_RAG_QUERIES`` queries. Falls back to ``[question]``.

    On the corrective pass, ``previous``/``missing`` steer it toward *different* angles.
    Returns ``(queries, prompt)`` — the prompt is surfaced in the agentic trace.
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

    messages = [*history_messages, {"role": "user", "content": instruction}]
    try:
        result = llm_chain.chat(_REWRITE_SYSTEM, messages, max_tokens=_AUX_MAX_TOKENS,
                                temperature=0.2, json_mode=True, deadline_s=15.0,
                                sensitive=privacy.current())
        data = _extract_json(result.text)
    except LLMChainError:
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


# --------------------------------------------------------------------------------------
# Retrieve / format
# --------------------------------------------------------------------------------------
def merge_docs(batches: list[list], seen: set[str] | None = None) -> list:
    """Merge several retrieval result lists into one de-duplicated list (first occurrence wins)."""
    seen = set() if seen is None else seen
    docs: list = []
    for results in batches:
        for doc in results or []:
            key = (getattr(doc, "page_content", "") or "").strip()
            if key and key not in seen:
                seen.add(key)
                docs.append(doc)
    return docs


def _retrieve(retriever, queries: list[str]) -> list:
    """Retrieve for every query and return merged, de-duplicated documents."""
    batches: list[list] = []
    for query in queries:
        try:
            batches.append(list(retriever.invoke(query)))
        except LLMChainError:
            raise
        except Exception:  # noqa: BLE001 — one bad query must not abort the round
            log.debug("Retrieval failed for query %r.", query, exc_info=True)
    return merge_docs(batches)


def _format_context(docs: list) -> str:
    """Join document contents into a single context block. Not capped here: llm_chain trims
    per provider (Groq only) when the prompt does not fit."""
    joined = "\n\n".join(
        (getattr(d, "page_content", "") or "").strip()
        for d in docs if (getattr(d, "page_content", "") or "").strip()
    )
    return joined


format_context = _format_context


def passage_view(docs: list, *, limit: int = 12, snippet: int = 320) -> list[dict]:
    """Project retrieved documents into a compact, JSON-safe shape for the trace UI.

    Each entry: ``{source, page, text}`` — the source PDF filename, 1-based page
    number (when known), and a truncated text snippet.
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


def source_names(docs: list) -> list[str]:
    """Distinct source filenames of ``docs`` in first-seen order."""
    seen: set[str] = set()
    names: list[str] = []
    for doc in docs:
        src = (getattr(doc, "metadata", {}) or {}).get("source", "")
        if src:
            name = Path(src).name
            if name not in seen:
                seen.add(name)
                names.append(name)
    return names


# --------------------------------------------------------------------------------------
# Grade
# --------------------------------------------------------------------------------------
def _grade(question: str, context: str) -> tuple[bool, str, str]:
    """Return ``(sufficient, missing, prompt)``. On any failure, assume sufficient."""
    instruction = (
        f"Question: {question}\n\n"
        f"Context passages:\n{context}\n\n"
        "Can the question be fully answered using ONLY these passages? "
        'Return ONLY JSON: {"sufficient": true|false, "missing": "what is missing, if any"}'
    )
    prompt = _prompt_text(_GRADE_SYSTEM, instruction)
    if not context.strip():
        return False, "no context retrieved", prompt
    try:
        result = llm_chain.chat(_GRADE_SYSTEM, [{"role": "user", "content": instruction}],
                                max_tokens=_AUX_MAX_TOKENS, temperature=0.0, json_mode=True,
                                deadline_s=15.0, sensitive=privacy.current())
        data = _extract_json(result.text)
    except LLMChainError:
        raise
    except Exception:  # noqa: BLE001 — grading is best-effort
        log.debug("Context grading failed; treating context as sufficient.", exc_info=True)
        return True, "", prompt
    if not isinstance(data, dict) or "sufficient" not in data:
        return True, "", prompt  # unparseable grade → don't loop; answer with what we have
    return bool(data.get("sufficient")), str(data.get("missing") or ""), prompt


# --------------------------------------------------------------------------------------
# Generate (prompt construction; the call itself is the caller's: streamed or not)
# --------------------------------------------------------------------------------------
def _generate_user(question: str, context: str) -> str:
    """Build the user message for the final generation step."""
    return (
        f"Context:\n{context}\n\nQuestion: {question}"
        if context.strip()
        else f"Question: {question}\n\n(No relevant context was found in the notes.)"
    )


generate_user = _generate_user


def generate_step(persona: str, question: str, docs: list, context: str) -> dict:
    """The trace row announcing the final answer (emitted BEFORE generation starts)."""
    return {
        "kind": "generate",
        "title": "Writing the grounded answer",
        "passages_used": len(docs),
        "context_chars": len(context),
        # Show the prompt skeleton without re-dumping the (already-listed) passages.
        "prompt": _prompt_text(
            persona,
            f"Context:\n[{len(docs)} retrieved passage(s), {len(context)} chars — shown above]"
            f"\n\nQuestion: {question}",
        ),
    }


class Gathered(NamedTuple):
    docs: list
    context: str
    steps: list
    sources: list


def gather(
    retriever, question: str, history_messages: list[dict], *,
    max_iters: int | None = None, on_step=None, cancel=None, llm_rewrite: bool | None = None,
    persona: str = "",
) -> Gathered:
    """Run rewrite → retrieve → (grade → correct)* and return the evidence, WITHOUT generating.

    ``on_step`` is called with each trace step dict the moment it is appended (including the
    final ``generate`` row), so a streaming caller can show progress. ``cancel`` is an optional
    ``threading.Event`` checked between rounds.
    """
    rounds = max_iters or config.AGENTIC_RAG_MAX_ITERS
    use_llm = (rounds > 1 or llm_rewrite_enabled()) if llm_rewrite is None else llm_rewrite
    docs: list = []
    seen: set[str] = set()
    steps: list[dict] = []

    def _emit(step: dict) -> None:
        steps.append(step)
        if on_step is not None:
            on_step(step)

    if use_llm:
        queries, rewrite_prompt = _rewrite(question, history_messages)
    else:
        queries, rewrite_prompt = heuristic_queries(question, history_messages) or [question], ""
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
        round_new = merge_docs([_retrieve(retriever, queries)], seen)
        docs.extend(round_new)
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
        # On the final round there is no corrective round left to act on the verdict, so grading
        # would be a wasted LLM call (we answer with what we have either way).
        if attempt == rounds - 1:
            log.info("Agentic RAG round %d/%d: %d queries, %d docs, final (grade skipped).",
                     attempt + 1, rounds, len(queries), len(docs))
            _emit({
                "kind": "grade", "verdict": "final",
                "title": "Answering with the retrieved context",
                "missing": "", "prompt": "",
            })
            break
        sufficient, missing, grade_prompt = _grade(question, context)
        log.info("Agentic RAG round %d/%d: %d queries, %d docs, sufficient=%s.",
                 attempt + 1, rounds, len(queries), len(docs), sufficient)
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
        queries, rewrite_prompt = _rewrite(question, history_messages, previous=queries, missing=missing)
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
    _emit(generate_step(persona, question, docs, final_context))
    return Gathered(docs, final_context, steps, source_names(docs))


def answer(
    retriever, question: str, history_messages: list[dict], persona: str, *,
    max_iters: int | None = None, on_step=None, cancel=None,
) -> tuple[str, list[dict], list[str], "llm_chain.ChatResult | None"]:
    """Blocking convenience: gather evidence, then generate once. Returns
    ``(answer, steps, sources, chat_result)`` (``chat_result`` is None when cancelled).

    The chat service does NOT use this (it streams); it exists for scripts and tests.
    """
    got = gather(retriever, question, history_messages, max_iters=max_iters, on_step=on_step,
                 cancel=cancel, persona=persona)
    if cancel is not None and cancel.is_set():
        return "", got.steps, [], None   # cancelled before the (most expensive) final generation
    messages = [*history_messages, {"role": "user", "content": _generate_user(question, got.context)}]
    result = llm_chain.chat(persona, messages, max_tokens=2048, temperature=0.3,
                            sensitive=privacy.current())
    return result.text, got.steps, got.sources, result


# ======================================================================================
# Best-practice pipeline (default path): plan -> hybrid retrieve -> rerank/MMR/neighbours ->
# confidence -> (adaptive LLM re-query only when weak) -> grounded, cited generation ->
# (tool-style re-search hops) -> citation verification.
#
# The default path adds ZERO LLM calls before generation. Building blocks live in
# ``backend.rag`` (query / retrieve / grounding); this section orchestrates them and yields the
# SSE ``step`` rows. Response shapes (additive to the original ``sources`` list of filenames):
#
#   done.sources    ["Lec-2.pdf", ...]  distinct files of the CITED passages (all retrieved files when
#                   nothing is cited; [] when the answer says it could not find it in the notes)
#   done.citations  [{"n": 1, "source": "Lec-2.pdf", "file": "Lec-2.pdf", "page": 4 | None (1-based),
#                     "text": "<snippet>", "snippet": "<same>", "score": 0.82, "cited": true}, ...]
#                   one entry per retrieved passage, in citation order; ``source``/``page``/``text`` are
#                   the pre-existing keys, the rest is new. ``subject`` is added for cross-subject chats.
#
# Env knobs: AGENTIC_RAG_ADAPTIVE (default on), AGENTIC_RAG_LLM_REWRITE (always LLM multi-query/HyDE),
# AGENTIC_RAG_LLM_RERANK (listwise rerank of the top passages), AGENTIC_RAG_HOPS (default 2),
# AGENTIC_RAG_POOL (24), AGENTIC_RAG_FINAL_K (8), AGENTIC_RAG_LOW_CONF (0.2), AGENTIC_RAG_WEAK_CONF (0.25).
# ======================================================================================
import asyncio  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from typing import AsyncIterator, Callable  # noqa: E402

from backend import deadline  # noqa: E402
from backend.rag import grounding  # noqa: E402
from backend.rag import query as rquery  # noqa: E402
from backend.rag import retrieve as rret  # noqa: E402

ALL_SUBJECTS = "*"              # chat "subject" meaning: search every subject the owner can see
_SNIPPET = 300


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return default


def adaptive_enabled() -> bool:
    return _env_flag("AGENTIC_RAG_ADAPTIVE", True)


def low_conf() -> float:
    return _env_float("AGENTIC_RAG_LOW_CONF", 0.2)


def weak_conf() -> float:
    return _env_float("AGENTIC_RAG_WEAK_CONF", 0.25)


def max_hops() -> int:
    try:
        return max(0, int(os.environ.get("AGENTIC_RAG_HOPS", "") or 2))
    except ValueError:
        return 2


_EXPAND_SYSTEM = (
    "You help search a student's course notes. Given a question, write alternative search queries that "
    "would find the relevant passages using different wording, synonyms, or one query per part when the "
    "question has several parts (a comparison, a list of things, a chain of reasoning), plus a short "
    "hypothetical passage that would answer it (as the notes might phrase it). Always reply with valid JSON."
)


def llm_expand(question: str, standalone: str, history_messages: list[dict], previous: list[str]) -> list[str]:
    """OPTIONAL LLM multi-query + HyDE. Returns extra queries ([] on any failure). Blocking."""
    n = max(2, config.AGENTIC_RAG_QUERIES)
    instruction = (
        f"Question: {standalone or question}\n\nPrevious queries that found weak matches: "
        f"{'; '.join(previous) or '(none)'}\n\nReturn ONLY JSON: "
        f'{{"queries": ["...", "..."], "hypothetical_answer": "2-3 sentences"}} with at most {n} queries.'
    )
    try:
        t = deadline.clamp(10.0)
        result = llm_chain.chat(_EXPAND_SYSTEM, [*history_messages[-6:], {"role": "user", "content": instruction}],
                                max_tokens=_AUX_MAX_TOKENS, temperature=0.3, json_mode=True,
                                deadline_s=t, sensitive=privacy.current())
        data = _extract_json(result.text)
    except Exception:  # noqa: BLE001 - optional step: never fail the turn over it (generation reports a dead chain)
        log.info("LLM query expansion unavailable; keeping the heuristic queries.", exc_info=True)
        return []
    out: list[str] = []
    raw = data.get("queries") if isinstance(data, dict) else None
    for q in (raw if isinstance(raw, list) else []):
        if isinstance(q, str) and q.strip() and q.strip() not in out and q.strip() not in previous:
            out.append(q.strip())
    hyde = data.get("hypothetical_answer") if isinstance(data, dict) else None
    if isinstance(hyde, str) and hyde.strip():
        out.append(" ".join(hyde.split())[:400])
    return out[: n + 1]


_RERANK_SYSTEM = ("You order passages from a student's notes by how directly they answer a question. "
                  "Always reply with valid JSON.")


def llm_rerank(ev: "Evidence") -> bool:
    """Listwise LLM rerank of the retrieved blocks (opt-in). Reorders + renumbers in place."""
    listing = "\n\n".join(f"[{b.n}] {' '.join(b.text.split())[:600]}" for b in ev.blocks)
    instruction = (f"Question: {ev.plan.standalone or ev.question}\n\nPassages:\n{listing}\n\n"
                   'Return ONLY JSON: {"order": [passage numbers, most relevant first]}')
    try:
        t = deadline.clamp(10.0)
        res = llm_chain.chat(_RERANK_SYSTEM, [{"role": "user", "content": instruction}], max_tokens=_AUX_MAX_TOKENS,
                             temperature=0.0, json_mode=True, deadline_s=t, sensitive=privacy.current())
        order = _extract_json(res.text).get("order")
    except Exception:  # noqa: BLE001
        return False
    by_n = {b.n: b for b in ev.blocks}
    seq = [by_n[n] for n in dict.fromkeys(order) if isinstance(n, int) and n in by_n] if isinstance(order, list) else []
    if not seq:
        return False
    seq += [b for b in ev.blocks if b not in seq]
    ev.blocks[:] = _number(seq)
    return True


@dataclass
class Verified:
    answer: str
    cited: list[int]
    invalid: list[int]
    sources: list[str]
    citations: list[dict]
    abstained: bool = False
    bad_pages: list[int] = field(default_factory=list)
    uncited: bool = False      # answer carried no [n] marker; ``sources`` are best-supported chips (cited:false) or empty


_META_STEMS = frozenset({"one", "sentenc", "sentence", "short", "brief", "briefli", "briefly", "simpl", "simpli",
                         "simply", "explain", "defin", "defini", "definit", "definition", "summari", "summary",
                         "quick", "quickli", "tell", "give", "exampl", "example", "mean", "word", "line"})
_UNCITED_TOP = 3


_ABSTAIN_RE = re.compile(r"couldn.t find|could not find|not (?:covered |mentioned |found )?in your notes|"
                         r"notes (?:do not|don.t|doesn.t) (?:contain|cover|mention)", re.I)


@dataclass
class Evidence:
    """Everything the generation step needs, plus the post-hoc citation check."""
    question: str
    plan: "rquery.QueryPlan"
    blocks: list
    confidence: float
    mode: str
    searchers: list
    steps: list = field(default_factory=list)
    lists: list = field(default_factory=list)
    hops_left: int = 0
    adaptive_used: bool = False
    grounded: bool = True
    related: list = field(default_factory=list)       # concept-graph suggestions (follow-up chips)
    themes: list = field(default_factory=list)        # global questions: extracted subject themes

    @property
    def term_coverage(self) -> float | None:
        """Fraction of the question's distinctive keywords found in ANY retrieved passage (None when the
        question has no usable keywords). Dense scores are inflated inside a subject (a maths question
        always looks 'similar' to a maths PDF), so a topic the notes never mention - 'eigenvalue' in a
        matrices/probability PDF - still scores ~0.35 confidence; zero lexical coverage exposes it."""
        kws = [k for k in (self.plan.keywords or []) if k not in _META_STEMS]
        if not kws or not self.blocks:
            return None
        from backend import vector_store

        seen: set[str] = set()
        for b in self.blocks:
            seen.update(vector_store.tokenize(b.text))
        return sum(1 for k in kws if k in seen) / len(kws)

    @property
    def insufficient(self) -> bool:
        if not self.blocks or self.confidence < weak_conf():
            return True
        cov = self.term_coverage
        return cov is not None and cov < 0.5 and self.confidence < _env_float("AGENTIC_RAG_COVERAGE_CONF", 0.6)

    @property
    def docs(self) -> list:
        return [rret.doc_for(b) for b in self.blocks]

    def context(self) -> str:
        return grounding.format_context(self.blocks)

    def user_message(self, question: str | None = None) -> str:
        msg = grounding.user_message(question or self.question, self.blocks)
        if self.themes:
            outline = "; ".join(t["label"] for t in self.themes[:10] if t.get("label"))
            msg += ("\n\n(Themes detected across the notes, to organise a broad overview - cite only the numbered "
                    f"passages: {outline})")
        return msg

    def system_suffix(self) -> str:
        return grounding.system_suffix(insufficient=self.insufficient, hops_left=self.hops_left)

    def citations(self, cited: set[int] | None = None) -> list[dict]:
        cited = cited or set()
        out = []
        for b in self.blocks:
            text = " ".join(b.text.split())
            snip = text[:_SNIPPET] + ("…" if len(text) > _SNIPPET else "")
            name = Path(b.source).name if b.source else "—"
            row = {"n": b.n, "source": name, "file": name, "page": (b.page + 1) if isinstance(b.page, int) else None,
                   "text": snip, "snippet": snip, "score": round(float(b.score), 3), "cited": b.n in cited}
            if b.subject:
                row["subject"] = b.subject
            out.append(row)
        return out

    def finalize(self, answer: str) -> Verified:
        clean, cited, invalid = grounding.verify_citations(answer, len(self.blocks))
        abstained = bool(_ABSTAIN_RE.search(clean[:240])) and not cited
        by_n = {b.n: b for b in self.blocks}
        uncited = False
        if cited:
            pool, rows = [by_n[n] for n in cited], self.citations(set(cited))
        elif abstained or not self.blocks:
            pool, rows = [], self.citations(set())
        elif self.insufficient:
            # No marker AND weak/uncovered evidence: the model answered from general knowledge. Showing
            # unrelated passages as "sources" would mislead, so attach none.
            pool, rows, uncited = [], [], True
        else:
            # Confident retrieval but the model skipped the [n] markers: surface the best-supported
            # passages as "Sources used" (cited:false) instead of silently dropping them.
            pool, uncited = self.blocks[:_UNCITED_TOP], True
            rows = self.citations(set())[:_UNCITED_TOP]
        names = list(dict.fromkeys(Path(b.source).name for b in pool if b.source))
        pages = {b.page + 1 + i for b in self.blocks if isinstance(b.page, int)
                 for i in range((b.page_end if isinstance(b.page_end, int) else b.page) - b.page + 1)}
        return Verified(clean, cited, invalid, names, rows, abstained,
                        grounding.unsupported_pages(clean, pages) if pages else [], uncited)


def plan_query(question: str, history_messages: list[dict], retriever=None) -> "rquery.QueryPlan":
    """Cheap query understanding (no LLM). ``retriever`` only supplies source filenames for scoping."""
    names: list[str] | None = None
    try:
        s = rret.searchers_for(retriever)[0] if retriever is not None else None
        idx = s.index() if s is not None else None
        if idx is not None:
            names = getattr(idx, "_rag_sources", None)
            if names is None:
                names = sorted({c["source"] for c in idx.chunks})
                idx._rag_sources = names
    except Exception:  # noqa: BLE001
        names = None
    return rquery.plan_query(question, history_messages, names)


def rewrite_step(plan: "rquery.QueryPlan", *, prompt: str = "", missing: str = "") -> dict:
    n = len(plan.queries)
    word = "query" if n == 1 else "queries"
    title = (f"Resolved your follow-up into {n} search {word}" if plan.followup
             else f"Rewrote your question into {n} search {word}")
    step = {"kind": "rewrite", "title": title, "queries": list(plan.queries), "prompt": prompt}
    if plan.followup:
        step["standalone"] = plan.standalone
    if missing:
        step["missing"] = missing
    return step


def _passages(blocks: list, limit: int = 12, snippet: int = 320) -> list[dict]:
    out = []
    for b in blocks[:limit]:
        text = " ".join(b.text.split())
        out.append({"n": b.n, "source": Path(b.source).name if b.source else "—",
                    "page": (b.page + 1) if isinstance(b.page, int) else None,
                    "page_end": (b.page_end + 1) if isinstance(b.page_end, int) else None,
                    "text": text[:snippet] + ("…" if len(text) > snippet else "")})
    return out


def retrieve_step(ev: Evidence, *, round_no: int = 1, title: str | None = None, blocks: list | None = None) -> dict:
    shown = ev.blocks if blocks is None else blocks
    n = len(shown)
    extra = " (keyword search - embeddings were slow)" if ev.mode == "lexical" else ""
    return {"kind": "retrieve", "round": round_no, "queries": list(ev.plan.queries),
            "title": title or f"Retrieved {n} passage{'' if n == 1 else 's'}{extra}",
            "passages": _passages(shown), "confidence": ev.confidence, "mode": ev.mode}


def grade_step(ev: Evidence) -> dict:
    pct = f"{round(ev.confidence * 100)}%"
    if ev.insufficient:
        return {"kind": "grade", "verdict": "insufficient", "confidence": ev.confidence,
                "title": f"Notes match weakly ({pct}) - will say so if they don't cover it",
                "missing": "", "prompt": ""}
    return {"kind": "grade", "verdict": "final", "confidence": ev.confidence,
            "title": f"Answering with the retrieved context ({pct} match)", "missing": "", "prompt": ""}


def _number(blocks: list, start: int = 1) -> list:
    for i, b in enumerate(blocks, start):
        b.n = i
    return blocks


def _retrieve_sync(searchers: list, plan: "rquery.QueryPlan", queries: list[str], *, lexical_only: bool,
                   timeout_s: float | None, prior: list | None = None, parts: list | None = None,
                   related: bool = False) -> "rret.Retrieval":
    lists = rret.gather_lists(searchers, queries, plan.lexical, lexical_only=lexical_only, timeout_s=timeout_s)
    lists = (prior or []) + lists
    ret = rret.build(lists, plan.keywords, ordered_terms=rquery._stems(plan.question) or None,
                     hints=plan.file_hints, parts=parts)
    if related:
        ret.related = rret.graph_related(searchers, plan.question)
    return ret


def _sub_plan(plan: "rquery.QueryPlan", sub: str) -> "rquery.QueryPlan":
    p = rquery.plan_query(sub, [], None)
    return rquery.QueryPlan(sub, sub, [sub], p.lexical, p.keywords, False, plan.file_hints)


def _sub_search(searchers: list, plan: "rquery.QueryPlan", sub: str, timeout_s: float | None) -> list:
    """Retrieval lists of ONE part of a question (hybrid + free keyword pass). Blocking: run in a thread."""
    sp = _sub_plan(plan, sub)
    return rret.gather_lists(searchers, [sub], sp.lexical, lexical_only=False, timeout_s=timeout_s)


async def _retrieve_hybrid(searchers: list, plan, queries: list[str], budget_s: float, prior: list | None = None,
                           parts: list | None = None, related: bool = False):
    """Hybrid retrieval inside ``budget_s``; on overrun fall back to BM25-only (no embedding call)."""
    task = asyncio.ensure_future(asyncio.to_thread(
        _retrieve_sync, searchers, plan, queries, lexical_only=False, timeout_s=budget_s, prior=prior,
        parts=parts, related=related))
    task.add_done_callback(lambda t: t.cancelled() or t.exception())
    try:
        return await asyncio.wait_for(asyncio.shield(task), budget_s + 0.25)
    except asyncio.TimeoutError:
        log.info("Hybrid retrieval exceeded %.1fs; falling back to lexical.", budget_s)
    except Exception:  # noqa: BLE001
        log.info("Hybrid retrieval failed; falling back to lexical.", exc_info=True)
    try:
        r = await asyncio.wait_for(asyncio.to_thread(
            _retrieve_sync, searchers, plan, queries, lexical_only=True, timeout_s=None, prior=prior,
            parts=parts, related=related), 2.5)
        r.mode = "lexical" if r.blocks else "none"
        return r
    except Exception:  # noqa: BLE001
        log.info("Lexical fallback unavailable.", exc_info=True)
        return rret.Retrieval([], 0.0, "none")


def graph_global_enabled() -> bool:
    return _env_flag("GRAPH_RAG_GLOBAL", True)


def graph_related_enabled() -> bool:
    return _env_flag("GRAPH_RAG_RELATED", True)


def graph_parts_enabled() -> bool:
    return _env_flag("AGENTIC_RAG_PARTS", True)


def _safe_global(searchers: list, question: str):
    try:
        return rret.global_retrieval(searchers, question)
    except Exception:  # noqa: BLE001 - the community view is an extra; fall back to the normal path
        log.info("Global (community) retrieval failed; using hybrid.", exc_info=True)
        return None


def _parts_of(plan: "rquery.QueryPlan") -> list[str]:
    return plan.subqueries[: max(1, _env_int("AGENTIC_RAG_SUBQUERIES", rquery.MAX_SUBQUERIES))]


async def _gather_parts(searchers: list, plan: "rquery.QueryPlan", subs: list[str], budget_s: float) -> dict:
    """Search every part concurrently (``asyncio.gather``; one worker thread per part, each bounded by the
    retrieval budget). Returns ``{sub-query: [RList, ...]}``; failed/timed-out parts are simply absent."""
    results = await asyncio.gather(*[
        asyncio.wait_for(asyncio.to_thread(_sub_search, searchers, plan, x, budget_s), budget_s + 0.25) for x in subs],
        return_exceptions=True)
    return {x: res for x, res in zip(subs, results) if not isinstance(res, BaseException) and res}


async def _fuse_parts(searchers: list, plan: "rquery.QueryPlan", ret: "rret.Retrieval", subs: list[str],
                      found: dict):
    part_lists = [lists[0] for lists in found.values()]
    extra = [lst for lists in found.values() for lst in lists]
    keywords = list(dict.fromkeys([*plan.keywords, *[w for x in subs for w in rquery._stems(x)]]))
    merged = await asyncio.to_thread(
        lambda: rret.build(list(ret.lists) + extra, keywords, ordered_terms=rquery._stems(plan.question) or None,
                           hints=plan.file_hints, parts=part_lists))
    if merged.blocks:
        merged.related = ret.related
        return merged
    return None


def parts_step(plan: "rquery.QueryPlan", subs: list[str], *, covered: bool = False) -> dict:
    label = "comparison" if plan.kind == "compare" else "question"
    if covered:
        title = f"All {len(subs)} parts of the {label} are already covered - no extra searches"
    else:
        title = f"Searching {len(subs)} part{'' if len(subs) == 1 else 's'} of your {label} in parallel"
    return {"kind": "rewrite", "queries": list(subs), "prompt": "", "title": title}


async def retrieve_events(retriever, message: str, history_messages: list[dict], *, budget_s: float = 1.5,
                          plan: "rquery.QueryPlan | None" = None, subject_mode: bool = True) -> AsyncIterator[dict]:
    """Run the retrieval half of the pipeline. Yields ``{"type": "step", "step": {...}}`` rows as they
    happen, then a final ``{"type": "_evidence", "evidence": Evidence}``."""
    searchers = rret.searchers_for(retriever)
    plan = plan or plan_query(message, history_messages, retriever)
    queries = list(plan.queries)
    steps: list[dict] = []

    def emit(step: dict) -> dict:
        steps.append(step)
        return {"type": "step", "step": step}

    forced = llm_rewrite_enabled()
    if forced and not deadline.expired():
        extra = await asyncio.to_thread(llm_expand, message, plan.standalone, history_messages, queries)
        queries = list(dict.fromkeys([*queries, *extra]))
    shown = rquery.QueryPlan(plan.question, plan.standalone, queries, plan.lexical, plan.keywords, plan.followup,
                             plan.file_hints, plan.kind, plan.subqueries)
    yield emit(rewrite_step(shown))

    glob = None
    if subject_mode and graph_global_enabled() and not forced:
        glob = await asyncio.to_thread(_safe_global, searchers, plan.standalone or message)
    if glob is not None:
        plan.kind = shown.kind = "global"
        ret = glob
        yield emit({"kind": "rewrite", "queries": [], "prompt": "",
                    "title": "Broad question - summarising by the subject's main themes: "
                             + ", ".join(t["label"].split(",")[0] for t in glob.themes[:5] if t.get("label"))})
    else:
        subs = _parts_of(plan) if (subject_mode and plan.subqueries and graph_parts_enabled()) else []
        sub_fut = None
        r0 = deadline.remaining()
        if subs and (r0 is None or r0 >= 6.0):
            if _env_flag("AGENTIC_RAG_SPECULATE", True):
                # Start the part searches NOW so they overlap the first pass (latency = max, not sum).
                yield emit(parts_step(plan, subs))
                sub_fut = asyncio.ensure_future(_gather_parts(searchers, plan, subs, budget_s))
                sub_fut.add_done_callback(lambda t: t.cancelled() or t.exception())
        else:
            subs = []
        ret = await _retrieve_hybrid(searchers, plan, queries, budget_s, related=graph_related_enabled())
        if subs:
            found: dict = {}
            if sub_fut is None:                       # lazy mode: only the parts the first pass leaves uncovered
                todo = [x for x in subs if not rquery.part_covered(x, ret.blocks)]
                if not todo:
                    yield emit(parts_step(plan, subs, covered=True))
                else:
                    yield emit(parts_step(plan, todo))
                    found = await _gather_parts(searchers, plan, todo, budget_s)
            else:                                     # speculative: results are already in flight / done
                found = await sub_fut
                todo = [x for x in subs if not rquery.part_covered(x, ret.blocks)]
                found = {x: v for x, v in found.items() if x in todo}      # early stop: covered parts are dropped
                if not todo:
                    steps[-1] = parts_step(plan, subs, covered=True)
            if found:
                merged = await _fuse_parts(searchers, plan, ret, [x for x in subs if x in found], found)
                if merged is not None:
                    ret = merged
    ev = Evidence(message, plan, _number(ret.blocks), ret.confidence, ret.mode, searchers, steps, ret.lists)
    ev.plan = shown
    ev.related = list(getattr(ret, "related", []) or [])
    ev.themes = list(getattr(ret, "themes", []) or [])

    # Optional listwise LLM rerank (env flag only).
    if _env_flag("AGENTIC_RAG_LLM_RERANK", False) and len(ev.blocks) > 2 and not deadline.expired():
        if await asyncio.to_thread(llm_rerank, ev):
            steps.append({"kind": "rewrite", "title": "Re-ranked the passages with the model", "queries": [], "prompt": ""})
            yield {"type": "step", "step": steps[-1]}

    # Adaptive second pass: only when the first retrieval is weak (never on the fast path).
    reretrieved = False
    if adaptive_enabled() and subject_mode and not forced and ev.confidence < low_conf():
        r = deadline.remaining()
        if r is None or r > 12.0:
            yield emit({"kind": "rewrite", "title": "Weak match - trying other phrasings of your question",
                        "queries": [], "prompt": "", "missing": "low retrieval confidence"})
            extra = await asyncio.to_thread(llm_expand, message, plan.standalone, history_messages, queries)
            if extra:
                steps[-1]["queries"] = extra
                ev.adaptive_used = True
                ret2 = await _retrieve_hybrid(searchers, plan, extra, budget_s, prior=ret.lists)
                if ret2.blocks and ret2.confidence >= ev.confidence:
                    ev = Evidence(message, shown, _number(ret2.blocks), ret2.confidence, ret2.mode, searchers,
                                  steps, ret2.lists, adaptive_used=True, related=ev.related)
                    reretrieved = True
                    yield emit(retrieve_step(ev, round_no=2, title=f"Re-retrieved {len(ev.blocks)} passages "
                                                                  f"after rephrasing ({round(ev.confidence * 100)}% match)"))
    if not reretrieved:
        yield emit(retrieve_step(ev))
    ev.steps = steps
    hops = max_hops() if (subject_mode and ev.confidence < _env_float("AGENTIC_RAG_HOP_CONF", 0.6)) else 0
    r = deadline.remaining()
    if r is not None and r < 20.0:
        hops = 0
    ev.hops_left = max(0, hops - (1 if ev.adaptive_used else 0))
    yield emit(grade_step(ev))
    yield emit(generate_step("", message, ev.docs, ev.context()))
    yield {"type": "_evidence", "evidence": ev}


async def research(ev: Evidence, query: str, budget_s: float = 1.5) -> AsyncIterator[dict]:
    """One extra search hop (the model asked ``<search>query</search>``): fuse the new results into the
    evidence, keeping existing citation numbers stable. Yields ``step`` events."""
    yield {"type": "step", "step": {"kind": "retrieve", "title": f"Searching your notes for “{query[:80]}”…",
                                    "queries": [query], "passages": []}}
    sub = rquery.plan_query(query, [], None)
    plan = rquery.QueryPlan(ev.plan.question, ev.plan.standalone, [query], sub.lexical,
                            list(dict.fromkeys([*ev.plan.keywords, *sub.keywords])), False, ev.plan.file_hints)
    ret = await _retrieve_hybrid(ev.searchers, plan, [query], budget_s, prior=ev.lists)
    have_keys = set().union(*[b.keys for b in ev.blocks]) if ev.blocks else set()
    have_text = {b.text for b in ev.blocks}
    fresh = [b for b in ret.blocks if not (b.keys and b.keys & have_keys) and b.text not in have_text]
    fresh = fresh[: max(2, rret.final_k() // 2)]
    _number(fresh, len(ev.blocks) + 1)
    ev.blocks.extend(fresh)
    ev.lists = ret.lists
    ev.confidence = max(ev.confidence, ret.confidence)
    ev.mode = ret.mode if ret.mode != "none" else ev.mode
    n = len(fresh)
    step = {"kind": "retrieve", "title": f"Found {n} more passage{'' if n == 1 else 's'} for “{query[:60]}”",
            "queries": [query], "passages": _passages(fresh), "confidence": ev.confidence, "mode": ev.mode}
    ev.steps.append(step)
    yield {"type": "step", "step": step}


_SEARCH_RE = re.compile(r"<search>\s*(.*?)\s*(?:</search>|$)", re.I | re.S)


async def generate_with_hops(ev: Evidence, generate: Callable, rebuild: Callable) -> AsyncIterator[dict]:
    """Stream generation; when the model replies with only ``<search>query</search>`` (offered in the
    system prompt while hops remain) run another retrieval hop and generate again.

    ``generate(system, msgs)`` is the chat service's token stream; ``rebuild(ev)`` returns the current
    ``(system, msgs)`` (it reads ``ev.hops_left`` / ``ev.blocks``). Events pass through unchanged except
    the swallowed search request. With no hops left this is a pure pass-through (zero added latency)."""
    while True:
        system, msgs = rebuild(ev)
        if ev.hops_left <= 0:
            async for it in generate(system, msgs):
                yield it
            return
        held, decided, swallow = "", False, False
        async for it in generate(system, msgs):
            kind = it["type"]
            if kind == "token":
                if decided:
                    yield it
                    continue
                held += it["text"]
                probe = held.lstrip().lower()
                if probe.startswith("<search"):
                    swallow = True
                    continue
                if len(probe) < 7 and "<search".startswith(probe):
                    continue                       # could still turn into <search> - hold a moment
                decided = True
                yield {"type": "token", "text": held}
                held = ""
            elif kind == "_final":
                if swallow:
                    continue
                if held and not decided:
                    decided = True
                    yield {"type": "token", "text": held}
                    held = ""
                yield it
            else:
                yield it
        if not swallow:
            return
        m = _SEARCH_RE.search(held)
        q = (m.group(1) if m else "").strip()
        r = deadline.remaining()
        if not q or (r is not None and r < 12.0):
            ev.hops_left = 0                       # cannot search: answer from what we have
            continue
        async for st in research(ev, q):
            yield st
        ev.hops_left -= 1


def all_subjects_searchers(owner: str | None, k: int = 10) -> list:
    """One searcher per subject visible to ``owner`` (bundled + the owner's own, never anyone else's)."""
    from backend import documents, vector_store

    out = []
    for name in documents.list_subjects(owner):
        try:
            out.append(rret.Searcher(vector_store.get_subject_retriever(name, k, owner), label=name))
        except Exception:  # noqa: BLE001 - one unreadable subject must not break the search
            log.info("Skipping subject %r in cross-subject search.", name, exc_info=True)
    return out
