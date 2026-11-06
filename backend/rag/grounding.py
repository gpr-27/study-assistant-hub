"""Grounded generation helpers: numbered context blocks, prompt rules, citation verification.

Context shape given to the model (one block per retrieved passage, most relevant first)::

    [1] Lec-2.pdf p.4-5
    <passage text, neighbouring chunks merged>

The model cites inline as ``[1][3]``; :func:`verify_citations` (no LLM) then strips any index that
does not exist, so the UI source chips always match the answer.
"""
from __future__ import annotations

import re
from pathlib import Path

NOT_FOUND = "I couldn't find this in your notes."
GENERAL_MARK = "From general knowledge (not from your notes):"

RULES = (
    "\n\nGROUNDING RULES - the numbered passages in the student's message are the ONLY source of "
    "truth about their notes:\n"
    "1. Answer ONLY from those passages. Do not add facts that are not in them.\n"
    "2. Cite the passages you used inline, right after the claim they support, like [1] or [2][4]. "
    "Cite only numbers that exist. Never invent citations, file names or page numbers; mention a "
    "page only exactly as it appears in a passage header. EVERY answer that relies on the passages "
    "- including a one-sentence or one-line answer, a definition or a follow-up reply - MUST carry at least one "
    "[n] marker; only an answer that uses nothing from them may have none.\n"
    f"3. If the passages do not contain the answer, begin with \"{NOT_FOUND}\" Then, if it helps, add a short "
    f"answer introduced by \"{GENERAL_MARK}\" and do not cite anything for it. If the passages cover only "
    "part of the question, answer that part with citations and say which part the notes do not cover.\n"
    "4. Be focused: short paragraphs or bullets, key terms in bold."
)
LOW_EVIDENCE = (
    "\nRetrieval confidence is LOW: the passages are probably unrelated to this question. Unless one of "
    "them clearly answers it, follow rule 3 and do not stretch an unrelated passage into an answer."
)


def search_tool_rule(hops_left: int) -> str:
    return (
        "\nSEARCH: if the passages plainly lack what you need and a differently-worded search of the notes "
        "would likely find it, reply with ONLY <search>short keyword query</search> (no other text) - at most "
        f"{hops_left} more time{'s' if hops_left != 1 else ''}. Otherwise just answer."
    )


def system_suffix(*, insufficient: bool, hops_left: int = 0) -> str:
    return RULES + (LOW_EVIDENCE if insufficient else "") + (search_tool_rule(hops_left) if hops_left > 0 else "")


def block_header(n: int, source: str, page_label: str) -> str:
    name = Path(source).name if source else "notes"
    return f"[{n}] {name}" + (f" {page_label}" if page_label else "")


def format_context(blocks: list) -> str:
    """``blocks``: objects with ``n``, ``source``, ``page_label()``, ``text``."""
    return "\n\n".join(f"{block_header(b.n, b.source, b.page_label())}\n{b.text}" for b in blocks)


def user_message(question: str, blocks: list) -> str:
    if not blocks:
        return (f"Question: {question}\n\n(No passages were retrieved from the student's notes for this "
                "question - follow rule 3.)")
    return f"Notes passages:\n\n{format_context(blocks)}\n\nQuestion: {question}"


# --------------------------------------------------------------------------------------
# Citation verification
# --------------------------------------------------------------------------------------
_CODE_SPLIT = re.compile(r"(```.*?```|`[^`\n]*`)", re.S)
_CITE = re.compile(r"\[(\d{1,3}(?:\s*(?:,|;|-|–|—)\s*\d{1,3})*)\](?!\()")
_PIECE = re.compile(r"(\d{1,3})\s*(,|;|-|–|—)?\s*")


def _expand(spec: str) -> list[int]:
    """'1, 3' -> [1, 3];  '2-4' -> [2, 3, 4]."""
    nums: list[int] = []
    parts = re.split(r"\s*[,;]\s*", spec.strip())
    for part in parts:
        m = re.fullmatch(r"(\d{1,3})\s*[-–—]\s*(\d{1,3})", part)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            nums.extend(range(a, b + 1) if a <= b and b - a < 30 else [a, b])
        elif part.isdigit():
            nums.append(int(part))
    return nums


def verify_citations(answer: str, n_blocks: int) -> tuple[str, list[int], list[int]]:
    """Return ``(clean_answer, cited, invalid)``: indices that exist (sorted, unique) and those removed."""
    cited: set[int] = set()
    invalid: set[int] = set()

    def fix(m: re.Match) -> str:
        nums = _expand(m.group(1))
        good = [x for x in dict.fromkeys(nums) if 1 <= x <= n_blocks]
        bad = [x for x in nums if not 1 <= x <= n_blocks]
        cited.update(good)
        invalid.update(bad)
        if not bad:
            return m.group(0)
        return "".join(f"[{x}]" for x in good)

    parts = _CODE_SPLIT.split(answer or "")
    for i in range(0, len(parts), 2):
        parts[i] = _CITE.sub(fix, parts[i])
    clean = "".join(parts)
    clean = re.sub(r"[ \t]+([.,;:!?])", r"\1", clean) if invalid else clean
    clean = re.sub(r"[ \t]{2,}", " ", clean) if invalid else clean
    return clean, sorted(cited), sorted(invalid)


_PAGE_MENTION = re.compile(r"\b(?:pages?|pp?\.)\s*(\d{1,4})\b", re.I)


def unsupported_pages(answer: str, pages: set[int]) -> list[int]:
    """Page numbers the answer mentions that appear in no retrieved passage (1-based)."""
    return sorted({int(x) for x in _PAGE_MENTION.findall(answer or "")} - set(pages))
