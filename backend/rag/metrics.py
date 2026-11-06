"""Cheap, LLM-free RAG quality metrics used by ``scripts/eval_rag.py`` (and unit-tested).

  * ``gold_hit``      did a retrieved block cover a gold (source, page range)?
  * ``faithfulness``  share of answer sentences whose content words are supported by the cited text
                      (lexical-overlap proxy; not a substitute for human review)
  * ``is_abstain``    does the answer say it could not find it in the notes?
"""
from __future__ import annotations

import re
from pathlib import Path

_ABSTAIN = re.compile(
    r"couldn.t find|could not find|can.t find|cannot find|not (?:covered |mentioned |found |included )?in (?:your|the) notes|"
    r"(?:notes|passages|material) (?:do not|don.t|doesn.t|does not) (?:contain|cover|mention|include|address)|"
    r"no (?:relevant )?(?:information|mention) (?:about|on|of)", re.I)
_SENT = re.compile(r"(?<=[.!?])\s+|\n+")
_CITE = re.compile(r"\[\d{1,3}(?:\s*[,;\-–]\s*\d{1,3})*\]")


def is_abstain(answer: str) -> bool:
    return bool(_ABSTAIN.search((answer or "")[:400]))


def covers(block: tuple[str, int | None, int | None], gold: dict) -> bool:
    """``block`` = (source, first_page_1based, last_page_1based); ``gold`` = {source, pages: [a, b]}."""
    src, p1, p2 = block
    if Path(src).name != gold["source"] or p1 is None:
        return False
    a, b = gold["pages"]
    return p1 <= b and (p2 if p2 is not None else p1) >= a


def gold_hit(blocks: list[tuple[str, int | None, int | None]], gold: list[dict], k: int | None = None) -> bool:
    return any(covers(b, g) for b in (blocks[:k] if k else blocks) for g in gold)


def first_gold_rank(blocks: list[tuple[str, int | None, int | None]], gold: list[dict]) -> int | None:
    for i, b in enumerate(blocks, 1):
        if any(covers(b, g) for g in gold):
            return i
    return None


def _terms(text: str) -> set[str]:
    from backend.rag.query import content_terms, _stems

    return set(_stems(" ".join(content_terms(text))))


def faithfulness(answer: str, support: list[str], per_sentence: list[list[str]] | None = None,
                 threshold: float = 0.6) -> tuple[int, int]:
    """(supported, total) over answer sentences with >= 3 content words. A sentence is supported when
    >= ``threshold`` of its content stems occur in the supporting text. Sentences carrying a citation
    marker are checked against ``per_sentence`` support when given, else against ``support``."""
    pool = set()
    for t in support:
        pool |= _terms(t)
    ok = total = 0
    for sent in _SENT.split(answer or ""):
        sent = sent.strip()
        if not sent or sent.lower().startswith(("from general knowledge", "i couldn")):
            continue
        terms = _terms(_CITE.sub("", sent))
        if len(terms) < 3:
            continue
        total += 1
        if len(terms & pool) / len(terms) >= threshold:
            ok += 1
    return ok, total
