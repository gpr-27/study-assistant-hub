"""Quiz generation service — Groq JSON mode, ported from the old QuizMaker.

Takes a single bounded slice of the document text (config.MAX_CONTENT_TOKENS)
and generates the quiz in ONE Groq call, then validates every question against
the QuizQuestion contract. Bounding the input keeps each request under the
model's per-minute token limit instead of fanning out into many chunk calls.
"""
import json
import logging

from groq import APIError, Groq

from backend.config import MAX_CONTENT_TOKENS, model_for, require_groq_key
from backend.documents import clip_to_tokens
from backend.schemas import QuizQuestion

_VALID_KEYS: tuple[str, str, str, str] = ("A", "B", "C", "D")
_VALID_DIFFICULTIES: tuple[str, str, str] = ("easy", "medium", "hard")

# Output (completion) cap. The model's per-minute token budget counts the reserved
# output too, so cap it to the actual quiz size (~150 tokens for a question with four
# options + explanation in JSON) rather than letting Groq reserve its full default.
_TOKENS_PER_QUESTION: int = 160
_MIN_OUTPUT_TOKENS: int = 512
_MAX_OUTPUT_TOKENS: int = 2048


def _output_tokens(num_questions: int) -> int:
    """Completion-token cap sized to the requested number of questions."""
    return max(_MIN_OUTPUT_TOKENS, min(_MAX_OUTPUT_TOKENS, num_questions * _TOKENS_PER_QUESTION + 256))


log = logging.getLogger(__name__)

_JSON_RETRIES = 3


def _is_json_validate_failed(exc: Exception) -> bool:
    """Groq's JSON mode 400s with code 'json_validate_failed' when the model emits
    malformed JSON. Small models do this intermittently; a fresh sample usually fixes it."""
    return "json_validate_failed" in str(exc).lower()


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


def _generate_from_prompt(prompt: str, model, max_tokens: int) -> list[QuizQuestion]:
    """Call Groq with JSON mode and return validated questions.

    Retries when the model emits malformed JSON: Groq's JSON mode rejects invalid
    output with a 400 ``json_validate_failed`` (and a stray bad token from a small
    model can also fail json.loads). A fresh sample almost always validates, so we
    retry a few times before surfacing a friendly error — never the raw 400.
    """
    require_groq_key()
    client = Groq()
    last_err: Exception | None = None
    for attempt in range(_JSON_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system",
                     "content": "You are a helpful teacher creating quiz questions. Reply with strictly valid JSON only."},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.5,
                max_tokens=max_tokens,
            )
            payload = response.choices[0].message.content
            data = json.loads(payload or "{}")
        except json.JSONDecodeError as exc:
            last_err = exc
            log.warning("Quiz JSON parse failed (attempt %d/%d); retrying.", attempt + 1, _JSON_RETRIES)
            continue
        except APIError as exc:
            # Retry the model's malformed-JSON 400; real Groq errors (rate limit, auth)
            # still propagate to the global handlers.
            if _is_json_validate_failed(exc):
                last_err = exc
                log.warning("Groq json_validate_failed (attempt %d/%d); retrying.", attempt + 1, _JSON_RETRIES)
                continue
            raise
        except Exception as exc:  # noqa: BLE001 — wrap any other transport/SDK failure
            raise QuizError(f"Quiz generation failed: {exc}") from exc

        items = data.get("questions") if isinstance(data, dict) else None
        questions = ([q for item in items if (q := _normalise_question(item)) is not None]
                     if isinstance(items, list) else [])
        if questions:
            return questions
        last_err = QuizError("the model returned no usable questions")

    raise QuizError("Couldn't generate a valid quiz after a few tries — please try again.")


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
    questions = _generate_from_prompt(_build_prompt(snippet, wanted, level), resolved, _output_tokens(wanted))
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
    budget = _output_tokens(wanted)
    # Single Groq call here, so one malformed roll = total failure. Retry once
    # before giving up (the global handler still surfaces a clear 502 if both fail).
    questions = _generate_from_prompt(prompt, resolved, budget) or _generate_from_prompt(prompt, resolved, budget)
    return {"questions": [q.model_dump() for q in questions[:wanted]]}
