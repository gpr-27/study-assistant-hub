"""Query understanding without an LLM: follow-up condensation, keywords, file scoping.

``plan_query(question, history_messages, source_names)`` returns a :class:`QueryPlan`:

  * ``queries``    embedding/hybrid queries, primary first (the condensed standalone question for a
                   follow-up, then the raw question); at most ``config.AGENTIC_RAG_QUERIES``
  * ``lexical``    keyword string (question + carried-over topic terms) for a free BM25-only pass
  * ``keywords``   stemmed content terms used by the reranker and the confidence score
  * ``file_hints`` source filenames the question explicitly mentions ("in Lec-2")
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from backend import config

_QUESTION_WORDS = frozenset(
    "what whats which who whom whose why how when where is are was were do does did can could would "
    "should shall may might will please tell show give explain describe define list summarize "
    "summarise write state mention compare difference differences between about notes note "
    "lecture lectures pdf pdfs document documents text slides slide".split())

_ANAPHORA = re.compile(
    r"\b(it|its|this|that|these|those|they|them|their|he|she|his|her|above|previous|earlier|"
    r"same|former|latter|ones?)\b", re.I)
_FOLLOWUP_START = re.compile(
    r"^\s*(and|but|so|also|then|why|how come|what about|how about|what if|more|another|again|"
    r"continue|go on|elaborate|expand|example|examples|in detail|can you (?:explain|elaborate|give)|"
    r"tell me more|explain (?:more|further|again|that|it|this)|give (?:me )?(?:an? )?example)\b", re.I)
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9'\-]*|\d+(?:\.\d+)?")


def _stems(text: str) -> list[str]:
    from backend import vector_store

    return vector_store.tokenize(text)


def content_terms(text: str) -> list[str]:
    """Surface content words (lower-cased, no question/stop words), in order, de-duplicated."""
    from backend import vector_store

    seen: set[str] = set()
    out: list[str] = []
    for w in _WORD.findall(text or ""):
        lw = w.lower().strip("'-")
        if len(lw) < 2 or lw in _QUESTION_WORDS or lw in vector_store._STOP or lw in seen:
            continue
        seen.add(lw)
        out.append(lw)
    return out


def is_followup(question: str, n_terms: int | None = None) -> bool:
    """True when ``question`` leans on earlier turns (pronoun, 'and ...?', or almost no content)."""
    n = len(content_terms(question)) if n_terms is None else n_terms
    return bool(n <= 1 or _ANAPHORA.search(question) or _FOLLOWUP_START.match(question))


def _topic_terms(question_terms: set[str], history_messages: list[dict], limit: int = 8) -> list[str]:
    """Content terms of the most recent earlier USER question that carries a topic."""
    looked = 0
    for m in reversed(history_messages or []):
        if m.get("role") != "user":
            continue
        looked += 1
        if looked > 4:
            break
        text = str(m.get("content") or "")
        terms = [t for t in content_terms(text) if t not in question_terms]
        if len(terms) >= 2 or (terms and not is_followup(text)):
            return terms[:limit]
    return []


@dataclass
class QueryPlan:
    question: str
    standalone: str
    queries: list[str]
    lexical: str
    keywords: list[str]
    followup: bool = False
    file_hints: list[str] = field(default_factory=list)
    kind: str = "single"                  # single | compare | multi | why | global (set by the pipeline)
    subqueries: list[str] = field(default_factory=list)   # one per part of a compare / multi-part question

    @property
    def primary(self) -> str:
        return self.queries[0] if self.queries else self.question


def file_hints(question: str, source_names: list[str] | None) -> list[str]:
    """Filenames the question names explicitly (full stem like ``Lec-2``/``lec 2``)."""
    q = re.sub(r"[\s_\-]+", " ", (question or "").lower())
    hits: list[str] = []
    for name in source_names or []:
        stem = re.sub(r"[\s_\-]+", " ", Path(name).stem.lower()).strip()
        if len(stem) >= 3 and re.search(r"(?<![a-z0-9])" + re.escape(stem) + r"(?![a-z0-9])", q):
            hits.append(name)
    return hits


# --------------------------------------------------------------------------------------
# Decomposition (heuristic, no LLM): compare / contrast / relate, multi-part, why
# --------------------------------------------------------------------------------------
MAX_SUBQUERIES = 3
_CMP_TAIL = (r"(?:\s+(?:in|for|on|of|according to|from)\s+(?:the\s+)?(?:notes?|lectures?|pdfs?|documents?))?\s*[?.!]*\s*$")
_COMPARE_RES = [re.compile(p, re.I) for p in (
    r"\b(?:difference|differences|distinctions?|similarit(?:y|ies)(?: and differences?)?|contrast)\s+(?:between|of|among)\s+(?P<x>.+?)" + _CMP_TAIL,
    r"\bcompare(?: and contrast)?\s+(?P<x>.+?)" + _CMP_TAIL,
    r"\bcontrast\s+(?P<x>.+?)" + _CMP_TAIL,
    r"\b(?:relationship|relation|link|connection)\s+(?:between|of)\s+(?P<x>.+?)" + _CMP_TAIL,
    r"\bhow\s+(?:does|do|is|are|can)\s+(?P<x>\S.*?\band\b.+?)\s+(?:relate|differ|compare|interact|connect|work together|fit together)\b.*$",
    r"\bhow\s+(?:does|do|is|are|can)\s+(?P<x>.+?\s(?:differ(?:s)?(?: from)?|compare[sd]? (?:to|with)|relate[sd]? to|"
    r"relate[sd]? with|connect(?:s|ed)? (?:to|with)|depend[s]? on|work(?:s)? with|interact(?:s)? with))\s+(?P<y>.+?)" + _CMP_TAIL,
    r"\b(?P<x>[\w\- ]{2,40}?)\s+(?:vs\.?|versus)\s+(?P<y>[\w\- ]{2,40}?)" + _CMP_TAIL,
)]
_SPLIT_ITEMS = re.compile(r"\s*,\s*(?:and\s+|or\s+)?|\s+(?:and|or|vs\.?|versus|with|to|against|&)\s+", re.I)
_LEAD = re.compile(r"^(?:the|a|an|each|both|these|those|between|among)\s+", re.I)
_MULTI_SPLIT = re.compile(
    r"\?\s+(?=\S)|;\s+|\s+(?:and|then|also|plus)\s+(?:also\s+)?(?=(?:what|why|how|when|where|which|who|explain|describe|"
    r"define|give|list|state|compare|derive|show|prove|solve|find|calculate)\b)|(?:\(\d\)|\b\d[.)])\s+(?=\S)", re.I)
_WHY = re.compile(r"^\s*(?:why|how come|what(?:'s| is) the (?:reason|purpose|point|need|use) )", re.I)
_ITEM_TRAIL = re.compile(r"\b(?:differ(?:s|ent)?|different|compare[sd]?|related?|relate[sd]?)\b.*$", re.I)


_FILLER = frozenset("example examples detail details more briefly simply short quickly step steps".split())


def _content_len(text: str) -> int:
    return len([t for t in content_terms(text) if t not in _FILLER])


def _expand_items(items: list[str]) -> list[str]:
    """['monolithic', 'layered', 'microkernel operating system structures'] -> every item gets the shared tail."""
    items = [i for i in items if i]
    if len(items) < 2:
        return items
    tail_words = items[-1].split()[1:]
    if len(tail_words) >= 2 and all(len(i.split()) == 1 for i in items[:-1]):
        return [f"{i} {' '.join(tail_words)}" for i in items[:-1]] + [items[-1]]
    return items


def decompose(question: str) -> tuple[str, list[str]]:
    """``(kind, subqueries)``: compare/contrast/relate -> one sub-query per compared thing; multi-part -> one
    per part; why -> the topic + its purpose. ``("single", [])`` when the question is one lookup."""
    q = " ".join((question or "").split())
    if not q:
        return "single", []
    q = re.sub(r"\b(?:to |with )?(?:each other|one another)\b", "", q, flags=re.I)
    for rx in _COMPARE_RES:
        m = rx.search(q)
        if not m:
            continue
        x = (m.group("x") or "").strip()
        y = (m.groupdict().get("y") or "").strip()
        raw = f"{_ITEM_TRAIL.sub('', x)} and {y}" if y else x
        items = [_LEAD.sub("", i).strip(" ,.?") for i in _SPLIT_ITEMS.split(raw)]
        items = [i for i in _expand_items(items) if _content_len(i) >= 1]
        items = list(dict.fromkeys(items))
        if len(items) >= 2:
            return "compare", items[:MAX_SUBQUERIES]
    parts = [p.strip(" ,.?;") for p in _MULTI_SPLIT.split(q) if p and p.strip(" ,.?;")]
    parts = [p for p in parts if _content_len(p) >= 2 or (_content_len(p) == 1 and not _ANAPHORA.search(p))]
    if len(parts) >= 2:
        return "multi", list(dict.fromkeys(parts))[:MAX_SUBQUERIES]
    if _WHY.match(q):
        topic = " ".join(content_terms(q)[:6])
        if topic:
            return "why", [f"{topic} purpose reason benefit", topic]
    return "single", []


def part_covered(sub: str, blocks: list, threshold: float = 0.8) -> bool:
    """True when some retrieved block already contains (at least ``threshold`` of) the stems of ``sub``."""
    stems = set(_stems(" ".join(content_terms(sub)))) or set(_stems(sub))
    if not stems:
        return True
    for b in blocks:
        have = set(_stems(b.text))
        if len(stems & have) / len(stems) >= threshold:
            return True
    return False


def plan_query(question: str, history_messages: list[dict] | None = None,
               source_names: list[str] | None = None) -> QueryPlan:
    q = " ".join((question or "").split())
    terms = content_terms(q)
    followup = bool(history_messages) and is_followup(q, len(terms))
    carried = _topic_terms(set(terms), history_messages or []) if followup else []
    if carried:
        standalone = f"{q} (about {' '.join(carried)})"
        queries = [standalone, q]
    else:
        followup = False
        standalone = q
        queries = [q]
    queries = [x for x in queries if x][: max(1, config.AGENTIC_RAG_QUERIES)]
    lexical_terms = terms + [t for t in carried if t not in terms]
    keywords = list(dict.fromkeys(_stems(" ".join(lexical_terms))))
    kind, subs = decompose(q)
    return QueryPlan(q, standalone, queries, " ".join(lexical_terms), keywords, followup,
                     file_hints(q, source_names), kind, subs)
