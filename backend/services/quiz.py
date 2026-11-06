"""Quiz generation service — one bounded call through the provider chain (JSON mode).

Takes the full document text and generates the quiz in ONE chain call (llm_chain trims per
provider where needed), then validates every question against the QuizQuestion contract. Malformed model JSON
is repaired (locally, then once via the LLM) before a clean QuizError is raised.
"""
import logging

from backend.schemas import QuizQuestion
from backend.services import _gen

_VALID_KEYS: tuple[str, str, str, str] = ("A", "B", "C", "D")
_VALID_DIFFICULTIES: tuple[str, str, str] = ("easy", "medium", "hard")

MAX_QUESTIONS: int = 20
_TOKENS_PER_QUESTION: int = 200
_MIN_OUTPUT_TOKENS: int = 2048
_MAX_OUTPUT_TOKENS: int = 4096

_SYSTEM = "You are a helpful teacher creating quiz questions. Reply with strictly valid JSON only."

log = logging.getLogger(__name__)


def _output_tokens(num_questions: int) -> int:
    """Completion-token cap sized to the requested number of questions."""
    return max(_MIN_OUTPUT_TOKENS, min(_MAX_OUTPUT_TOKENS, num_questions * _TOKENS_PER_QUESTION + 512))


class QuizError(Exception):
    """Raised when a quiz cannot be generated or parsed from the model output."""


def _normalise_difficulty(difficulty: str) -> str:
    """Return a known difficulty label, defaulting to 'medium'."""
    value = (difficulty or "").strip().lower()
    return value if value in _VALID_DIFFICULTIES else "medium"


def _build_prompt(content: str, num_questions: int, difficulty: str) -> str:
    """Build the user prompt; the word JSON is present so provider JSON modes are happy."""
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
    if isinstance(options, list) and len(options) == len(_VALID_KEYS):
        options = dict(zip(_VALID_KEYS, options))      # a bare list of four -> A..D
    if not isinstance(options, dict):
        return None

    normalised: dict[str, str] = {}
    for key, value in options.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            value = str(value)
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
    knowledge); the word JSON keeps provider JSON modes happy."""
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


def _shape(data: object) -> list[QuizQuestion]:
    items = data.get("questions") if isinstance(data, dict) else data if isinstance(data, list) else None
    if not isinstance(items, list):
        return []
    return [q for item in items if (q := _normalise_question(item)) is not None]


def _generate_from_prompt(prompt: str, max_tokens: int):
    """One chain call (JSON mode) -> (validated questions, ChatResult)."""
    return _gen.generate_json(_SYSTEM, prompt, shape=_shape, max_tokens=max_tokens,
                              error_cls=QuizError, what="quiz", temperature=0.5)


def _count(num_questions: int) -> int:
    return max(1, min(MAX_QUESTIONS, int(num_questions)))


def make_quiz(content: str, num_questions: int, difficulty: str, model: str | None = None) -> dict:
    """Create a quiz from the document text in a single bounded chain call.

    Returns {"questions": [...], "llm": {...}} (questions match the QuizQuestion schema; malformed
    items are dropped, never raised on). ``model`` is accepted for back-compat and ignored: the
    provider chain chooses the model.
    """
    if not content or not content.strip():
        raise QuizError("No content was supplied to build a quiz from.")
    wanted = _count(num_questions)
    level = _normalise_difficulty(difficulty)
    snippet = content
    questions, res = _generate_from_prompt(_build_prompt(snippet, wanted, level), _output_tokens(wanted))
    return {"questions": [q.model_dump() for q in questions[:wanted]], "llm": _gen.public(res)}


def make_quiz_from_topic(topic: str, num_questions: int, difficulty: str, model: str | None = None) -> dict:
    """Create a quiz about a free topic, using the model's own knowledge. Same shape as make_quiz."""
    topic = _gen.clean_topic(topic)
    if not topic:
        raise QuizError("No topic was supplied to build a quiz from.")
    wanted = _count(num_questions)
    level = _normalise_difficulty(difficulty)
    questions, res = _generate_from_prompt(_build_topic_prompt(topic, wanted, level), _output_tokens(wanted))
    return {"questions": [q.model_dump() for q in questions[:wanted]], "llm": _gen.public(res)}
