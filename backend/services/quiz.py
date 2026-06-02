"""Quiz generation service — Groq JSON mode, ported from the old QuizMaker.

Takes a single bounded slice of the document text (config.MAX_CONTENT_TOKENS)
and generates the quiz in ONE Groq call, then validates every question against
the QuizQuestion contract. Bounding the input keeps each request under the
model's per-minute token limit instead of fanning out into many chunk calls.
"""
import json

from groq import APIError, Groq

from backend.config import MAX_CONTENT_TOKENS, model_for, require_groq_key
from backend.documents import clip_to_tokens
from backend.schemas import QuizQuestion

_VALID_KEYS: tuple[str, str, str, str] = ("A", "B", "C", "D")
_VALID_DIFFICULTIES: tuple[str, str, str] = ("easy", "medium", "hard")


class QuizError(Exception):
    """Raised when a quiz cannot be generated or parsed from the model output."""


def _normalise_difficulty(difficulty: str) -> str:
    """Return a known difficulty label, defaulting to 'medium'."""
    value = (difficulty or "").strip().lower()
    return value if value in _VALID_DIFFICULTIES else "medium"


def _build_prompt(content: str, num_questions: int, difficulty: str) -> str:
    """Build the user prompt; the word JSON is present so Groq JSON mode is happy."""
    return (
        f"Create exactly {num_questions} multiple choice questions from the content below.\n"
        f"Difficulty level: {difficulty}.\n\n"
        "Content:\n"
        f"{content}\n\n"
        "Make questions that test understanding, not rote memorisation. "
        "For each question provide four options keyed A, B, C and D, mark the single "
        "correct option, and add a brief explanation of why it is correct.\n\n"
        "Return ONLY a JSON object with this exact structure:\n"
        "{\n"
        '  "questions": [\n'
        "    {\n"
        '      "question": "...",\n'
        '      "options": {"A": "...", "B": "...", "C": "...", "D": "..."},\n'
        '      "correct": "A",\n'
        '      "explanation": "..."\n'
        "    }\n"
        "  ]\n"
        "}"
    )


def _normalise_question(raw: object) -> QuizQuestion | None:
    """Coerce a raw model item into a QuizQuestion, or return None if unusable.

    Tolerant of the model's common shape drift: option keys in any case
    ("a".."d"), surrounding whitespace, and a ``correct`` given either as a
    letter ("B", "b", "B) ...") or as the full text of the right option. This
    keeps an otherwise-good question from being dropped over formatting.
    """
    if not isinstance(raw, dict):
        return None
    question = raw.get("question")
    explanation = raw.get("explanation")
    options = raw.get("options")
    correct = raw.get("correct")
    if not isinstance(question, str) or not question.strip():
        return None
    if not isinstance(explanation, str) or not explanation.strip():
        return None
    if not isinstance(options, dict):
        return None

    normalised: dict[str, str] = {}
    for key, value in options.items():
        if not isinstance(key, str) or not isinstance(value, str):
            return None
        upper = key.strip().upper()
        if upper in _VALID_KEYS and value.strip():
            normalised[upper] = value.strip()
    if set(normalised) != set(_VALID_KEYS):
        return None

    letter: str | None = None
    if isinstance(correct, str):
        text = correct.strip()
        if text[:1].upper() in _VALID_KEYS:
            letter = text[:1].upper()
        else:
            for key in _VALID_KEYS:
                if normalised[key].lower() == text.lower():
                    letter = key
                    break
    if letter is None:
        return None

    return QuizQuestion(
        question=question.strip(),
        options=normalised,
        correct=letter,
        explanation=explanation.strip(),
    )


def _build_topic_prompt(topic: str, num_questions: int, difficulty: str) -> str:
    """Build a quiz prompt for a free topic (no document — uses the model's own
    knowledge); the word JSON keeps Groq JSON mode happy."""
    return (
        f"Create exactly {num_questions} multiple choice questions that test understanding "
        f'of the topic: "{topic}".\n'
        f"Difficulty level: {difficulty}.\n\n"
        "Make questions that test understanding, not rote memorisation. "
        "For each question provide four options keyed A, B, C and D, mark the single "
        "correct option, and add a brief explanation of why it is correct.\n\n"
        "Return ONLY a JSON object with this exact structure:\n"
        "{\n"
        '  "questions": [\n'
        "    {\n"
        '      "question": "...",\n'
        '      "options": {"A": "...", "B": "...", "C": "...", "D": "..."},\n'
        '      "correct": "A",\n'
        '      "explanation": "..."\n'
        "    }\n"
        "  ]\n"
        "}"
    )


def _generate_from_prompt(prompt: str, model) -> list[QuizQuestion]:
    """Call Groq once with a prepared prompt and return validated questions."""
    require_groq_key()
    client = Groq()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful teacher creating quiz questions."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.5,
        )
        payload = response.choices[0].message.content
        data = json.loads(payload or "{}")
    except json.JSONDecodeError as exc:
        raise QuizError(f"The model returned invalid JSON: {exc}") from exc
    except APIError:
        raise  # Groq errors (rate limit, auth, …) are formatted by the global handlers
    except Exception as exc:  # noqa: BLE001 — wrap any other transport/SDK failure
        raise QuizError(f"Quiz generation failed: {exc}") from exc

    items = data.get("questions") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return []
    return [q for item in items if (q := _normalise_question(item)) is not None]


def make_quiz(content: str, num_questions: int, difficulty: str, model: str | None = None) -> dict:
    """Create a quiz from the document text in a single bounded Groq call.

    The text is sent in full when it fits within MAX_CONTENT_TOKENS; a longer
    document is capped to its first MAX_CONTENT_TOKENS so the request stays under
    the model's per-minute token limit. Returns {"questions": [...]} matching the
    QuizQuestion schema; malformed items are dropped, never raised on.
    """
    if not content or not content.strip():
        raise QuizError("No content was supplied to build a quiz from.")
    wanted = max(1, int(num_questions))
    level = _normalise_difficulty(difficulty)
    snippet = clip_to_tokens(content, MAX_CONTENT_TOKENS)
    resolved = model_for("quiz", model)
    questions = _generate_from_prompt(_build_prompt(snippet, wanted, level), resolved)
    return {"questions": [q.model_dump() for q in questions[:wanted]]}


def make_quiz_from_topic(topic: str, num_questions: int, difficulty: str, model: str | None = None) -> dict:
    """Create a quiz about a free topic, using the model's own knowledge.

    Returns {"questions": [...]} like make_quiz. One Groq call is enough since
    there is no document to chunk.
    """
    if not topic or not topic.strip():
        raise QuizError("No topic was supplied to build a quiz from.")
    wanted = max(1, int(num_questions))
    level = _normalise_difficulty(difficulty)
    prompt = _build_topic_prompt(topic.strip(), wanted, level)
    resolved = model_for("quiz", model)
    # Single Groq call here, so one malformed roll = total failure. Retry once
    # before giving up (the global handler still surfaces a clear 502 if both fail).
    questions = _generate_from_prompt(prompt, resolved) or _generate_from_prompt(prompt, resolved)
    return {"questions": [q.model_dump() for q in questions[:wanted]]}
