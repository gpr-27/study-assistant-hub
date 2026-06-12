"""Cheat-sheet service: condense a document's text into a one-page exam crib sheet.

ONE Groq call per sheet — plain markdown out (no JSON mode), with a conservative
completion cap so input + reserved output stays inside the free tier's per-minute
token budget. The content slice is bounded with clip_to_tokens before the call.
"""
from groq import APIError, Groq

from backend.config import MAX_CONTENT_TOKENS, model_for, require_groq_key
from backend.documents import clip_to_tokens

# Output (completion) cap. A dense one-page crib sheet is ~800-1000 tokens of terse
# bullets/tables; 1200 leaves headroom without letting Groq reserve a huge default.
_MAX_OUTPUT_TOKENS: int = 1200

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


def make_cheatsheet(content: str, model: str | None = None) -> str:
    """Generate the cheat-sheet markdown from document text in a single Groq call.

    Raises CheatsheetError on empty input, transport failures, or an empty/blank
    completion. Groq APIErrors bubble for the global 429/502 handlers.
    """
    if not content or not content.strip():
        raise CheatsheetError("No content was supplied to build a cheat sheet from.")
    snippet = clip_to_tokens(content, MAX_CONTENT_TOKENS)
    require_groq_key()
    client = Groq()
    try:
        response = client.chat.completions.create(
            model=model_for("cheatsheet", model),
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": _build_prompt(snippet)},
            ],
            temperature=0.3,
            max_tokens=_MAX_OUTPUT_TOKENS,
        )
        markdown = response.choices[0].message.content
    except APIError:
        raise  # Groq errors (rate limit, auth, …) are formatted by the global handlers
    except Exception as exc:  # noqa: BLE001 — wrap any other transport/SDK failure
        raise CheatsheetError(f"Cheat-sheet generation failed: {exc}") from exc

    if not markdown or not markdown.strip():
        raise CheatsheetError("The model returned an empty cheat sheet; try again.")
    return markdown.strip()
