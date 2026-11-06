"""Cheat-sheet service: condense a document's text into a one-page exam crib sheet.

ONE chain call per sheet — plain markdown out (no JSON mode). The content slice is bounded with
NOT clipped here: the full text is passed and llm_chain trims it per provider (Groq only).
"""
from backend.services import _gen

# A dense one-page crib sheet is ~800-1200 tokens of terse bullets/tables; the rest is headroom
# for reasoning models (their thinking tokens count against the completion budget).
_MAX_OUTPUT_TOKENS: int = 3500

_SYSTEM_PROMPT: str = (
    "You are an expert exam coach who writes ultra-dense one-page crib sheets. "
    "You never write prose paragraphs — only terse bullets and compact markdown tables."
)


class CheatsheetError(Exception):
    """Raised when a cheat sheet cannot be generated from the input."""


def _build_prompt(content: str) -> str:
    return (
        "Condense the study material below into an ULTRA-DENSE one-page exam cheat sheet "
        "in markdown. Requirements:\n"
        "- Key definitions: term — crisp definition (one line each).\n"
        "- Formulas / rules: each on its own line, with symbols explained inline.\n"
        "- Contrasts: short 'X vs Y' comparisons (a compact markdown table where it helps).\n"
        "- Common pitfalls: bullets of mistakes students make and the fix.\n"
        "- A mnemonic ONLY if one genuinely fits the material.\n"
        "Style: terse bullet points and tables ONLY — no prose paragraphs, no intro, no outro, "
        "no filler words. Use short ## section headings. Abbreviate aggressively. "
        "Everything must fit on roughly one printed page.\n\n"
        "MATERIAL:\n"
        f"{content}"
    )


def _build_topic_prompt(topic: str) -> str:
    return (
        "Write an ULTRA-DENSE one-page exam cheat sheet in markdown about the topic: "
        f'"{topic}". Use your own knowledge of the topic. Requirements:\n'
        "- Key definitions: term — crisp definition (one line each).\n"
        "- Formulas / rules: each on its own line, with symbols explained inline.\n"
        "- Contrasts: short 'X vs Y' comparisons (a compact markdown table where it helps).\n"
        "- Common pitfalls: bullets of mistakes students make and the fix.\n"
        "- A mnemonic ONLY if one genuinely fits the material.\n"
        "Style: terse bullet points and tables ONLY — no prose paragraphs, no intro, no outro, "
        "no filler words. Use short ## section headings. Abbreviate aggressively. "
        "Everything must fit on roughly one printed page."
    )


def _run(prompt: str) -> tuple[str, dict | None]:
    """One chain call -> (non-empty cheat-sheet markdown, llm_public). Shared by both entry points."""
    markdown, res = _gen.generate_text(_SYSTEM_PROMPT, prompt, max_tokens=_MAX_OUTPUT_TOKENS,
                                       error_cls=CheatsheetError, what="cheat sheet", temperature=0.3)
    return markdown, _gen.public(res)


def make_cheatsheet_ex(content: str, model: str | None = None) -> tuple[str, dict | None]:
    """Cheat sheet from document text -> ``(markdown, llm_public)``. ``model`` is ignored (chain picks)."""
    if not content or not content.strip():
        raise CheatsheetError("No content was supplied to build a cheat sheet from.")
    return _run(_build_prompt(content))


def make_cheatsheet_from_topic_ex(topic: str, model: str | None = None) -> tuple[str, dict | None]:
    """Cheat sheet about a free topic (model's own knowledge) -> ``(markdown, llm_public)``."""
    topic = _gen.clean_topic(topic)
    if not topic:
        raise CheatsheetError("No topic was supplied to build a cheat sheet from.")
    return _run(_build_topic_prompt(topic))


def make_cheatsheet(content: str, model: str | None = None) -> str:
    """Back-compat: cheat-sheet markdown only (see :func:`make_cheatsheet_ex` for the ``llm`` info)."""
    return make_cheatsheet_ex(content, model)[0]


def make_cheatsheet_from_topic(topic: str, model: str | None = None) -> str:
    """Back-compat: markdown only (see :func:`make_cheatsheet_from_topic_ex`)."""
    return make_cheatsheet_from_topic_ex(topic, model)[0]
