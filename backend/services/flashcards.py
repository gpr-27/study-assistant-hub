"""Flashcard generation on Groq (JSON mode).

Ports the old ``FlashcardMaker`` (OpenAI + PyPDF2) to Groq's JSON-mode chat
completion. The document text is sent in a single bounded call (the whole thing
when it fits within config.MAX_CONTENT_TOKENS, otherwise its first slice); cards
are validated (non-empty front/back/topic), malformed items dropped, and the
result capped to the requested count.
"""
import json

from groq import APIError, Groq

from backend.config import MAX_CONTENT_TOKENS, model_for, require_groq_key
from backend.documents import clip_to_tokens

_SYSTEM_PROMPT: str = "You are a helpful teacher creating educational flashcards. Always respond with valid JSON."

# Output (completion) cap. A provider's per-minute token budget counts the RESERVED
# output too, and with no cap Groq reserves the model's full ~8k default completion —
# which alone can exceed the free-tier limit. Size it to the actual deck: a JSON card
# is ~100 tokens; clamp so input + this stays under the model's TPM cap.
_TOKENS_PER_CARD: int = 110
_MIN_OUTPUT_TOKENS: int = 512
_MAX_OUTPUT_TOKENS: int = 2048


def _output_tokens(num_cards: int) -> int:
    """Completion-token cap sized to the requested number of cards."""
    return max(_MIN_OUTPUT_TOKENS, min(_MAX_OUTPUT_TOKENS, num_cards * _TOKENS_PER_CARD + 256))


class FlashcardError(Exception):
    """Raised when flashcards cannot be generated or parsed from the model."""


def _build_prompt(content: str, num_cards: int, difficulty: str) -> str:
    """Build the JSON-mode prompt for a single chunk of content."""
    return f"""
Create {num_cards} focused flashcards from this content.
Difficulty level: {difficulty}

Rules for creating effective flashcards:
1. One concept per card - break complex ideas into multiple cards.
2. The front should require active recall, not just recognition.
3. Questions should be specific and clear.
4. Answers should be concise but complete.
5. Avoid true/false or multiple-choice formats.
6. For {difficulty} difficulty, pick simpler or harder concepts, but keep cards clear.

Examples of good flashcards:
- Front: "What is Newton's First Law of Motion?"
  Back: "An object stays at rest or in uniform motion unless acted upon by an external force."
  Topic: "Physics - Motion"
- Front: "Complete the equation: E = ___"
  Back: "mc^2"
  Topic: "Physics - Energy"

Content:
{content}

Return ONLY valid JSON with this exact structure:
{{
    "cards": [
        {{
            "front": "Question or term goes here",
            "back": "Answer or definition goes here",
            "topic": "Brief topic/category label"
        }}
    ]
}}
""".strip()


def _is_valid_card(card: object) -> bool:
    """Return True if a parsed card has non-empty front, back, and topic."""
    if not isinstance(card, dict):
        return False
    return all(
        isinstance(card.get(key), str) and card[key].strip()
        for key in ("front", "back", "topic")
    )


def _clean_card(card: dict) -> dict:
    """Return a card with only the schema fields, trimmed."""
    return {
        "front": card["front"].strip(),
        "back": card["back"].strip(),
        "topic": card["topic"].strip(),
    }


def _build_topic_prompt(topic: str, num_cards: int, difficulty: str) -> str:
    """Build the JSON-mode prompt for a free topic (no document)."""
    return f"""
Create {num_cards} focused flashcards to help a student learn the topic: "{topic}".
Difficulty level: {difficulty}

Rules for creating effective flashcards:
1. One concept per card - break complex ideas into multiple cards.
2. The front should require active recall, not just recognition.
3. Questions should be specific and clear.
4. Answers should be concise but complete.
5. Avoid true/false or multiple-choice formats.
6. For {difficulty} difficulty, pick simpler or harder concepts, but keep cards clear.

Return ONLY valid JSON with this exact structure:
{{
    "cards": [
        {{
            "front": "Question or term goes here",
            "back": "Answer or definition goes here",
            "topic": "Brief topic/category label"
        }}
    ]
}}
""".strip()


def _generate_from_prompt(prompt: str, model, max_tokens: int) -> list[dict]:
    """Call Groq once with a prepared prompt and return validated cards."""
    client = Groq()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.5,
            max_tokens=max_tokens,
        )
        raw = response.choices[0].message.content
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise FlashcardError(f"Model returned invalid JSON: {exc}") from exc
    except APIError:
        raise  # Groq errors (rate limit, auth, …) are formatted by the global handlers
    except Exception as exc:
        raise FlashcardError(f"Flashcard generation failed: {exc}") from exc

    cards = data.get("cards") if isinstance(data, dict) else None
    if not isinstance(cards, list):
        raise FlashcardError("Model response did not contain a 'cards' list.")
    return [_clean_card(card) for card in cards if _is_valid_card(card)]


def make_flashcards(content: str, num_cards: int = 10, difficulty: str = "medium", model: str | None = None) -> dict:
    """Create flashcards from the document text in a single bounded Groq call.

    The text is sent in full when it fits within MAX_CONTENT_TOKENS; a longer
    document is capped to its first MAX_CONTENT_TOKENS so the request stays under
    the model's per-minute token limit. Returns ``{"cards": [...]}`` capped to
    ``num_cards``.
    """
    require_groq_key()
    if not content or not content.strip():
        raise FlashcardError("Cannot create flashcards from empty content.")
    target = max(1, num_cards)
    snippet = clip_to_tokens(content, MAX_CONTENT_TOKENS)
    cards = _generate_from_prompt(
        _build_prompt(snippet, target, difficulty), model_for("flashcards", model), _output_tokens(target)
    )
    return {"cards": cards[:target]}


def make_flashcards_from_topic(topic: str, num_cards: int = 10, difficulty: str = "medium", model: str | None = None) -> dict:
    """Create flashcards about a free topic using the model's own knowledge.

    Returns ``{"cards": [...]}`` like make_flashcards. One Groq call is enough
    since there is no document to chunk.
    """
    require_groq_key()
    if not topic or not topic.strip():
        raise FlashcardError("Cannot create flashcards without a topic.")
    target = max(1, num_cards)
    cards = _generate_from_prompt(
        _build_topic_prompt(topic.strip(), target, difficulty), model_for("flashcards", model), _output_tokens(target)
    )
    return {"cards": cards[:target]}
