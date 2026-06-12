"""Past-paper question prediction service — Groq JSON mode.

Takes ONE bounded slice of the combined past-paper text (config.MAX_CONTENT_TOKENS,
clipped by the router before calling in) and makes a single Groq call that
identifies recurring topics/patterns across the papers and predicts likely
next-exam questions. The model output is validated against a strict shape:

    {
      "patterns": [{"topic", "frequency_note", "trend"}, ...],
      "predicted_questions": [{"question", "topic", "likelihood", "reasoning"}, ...]
    }

Malformed items are dropped (never raised on); predictions are capped at
``_MAX_PREDICTIONS`` and likelihoods normalised to high|medium|low.
"""
import json

from groq import APIError, Groq

from backend.config import model_for, require_groq_key

_VALID_LIKELIHOODS: tuple[str, str, str] = ("high", "medium", "low")
_MAX_PATTERNS: int = 10
_MAX_PREDICTIONS: int = 12

# Output (completion) cap. The model's per-minute token budget counts the reserved
# output too: ~10 patterns + ~12 predictions with reasoning fit comfortably in JSON
# under this cap, so we never let Groq reserve its much larger default.
_MAX_OUTPUT_TOKENS: int = 2048


class PredictorError(Exception):
    """Raised when predictions cannot be generated or parsed from the model output."""


def _build_prompt(corpus: str, subject_hint: str | None) -> str:
    """Build the user prompt; the word JSON is present so Groq JSON mode is happy."""
    hint = f"The papers are for the subject: {subject_hint.strip()}.\n" if subject_hint and subject_hint.strip() else ""
    return (
        "You are given the combined text of several past exam papers (separated by "
        "'=== PAPER: <filename> ===' markers).\n"
        f"{hint}\n"
        "Past papers:\n"
        f"{corpus}\n\n"
        "Analyze the papers as an exam-pattern expert:\n"
        "1. Identify recurring topics and question patterns across the papers — how often "
        "each topic appears and whether attention to it is rising, stable, or falling.\n"
        "2. Predict the questions most likely to appear in the NEXT exam, based on those "
        "patterns. For each, state the topic it covers, a likelihood of exactly one of "
        '"high", "medium" or "low", and brief reasoning grounded in the observed patterns.\n\n'
        f"Give at most {_MAX_PATTERNS} patterns and at most {_MAX_PREDICTIONS} predicted questions.\n\n"
        "Return ONLY a JSON object with this exact structure:\n"
        "{\n"
        '  "patterns": [\n'
        '    {"topic": "...", "frequency_note": "appeared in 3 of 4 papers", "trend": "rising"}\n'
        "  ],\n"
        '  "predicted_questions": [\n'
        '    {"question": "...", "topic": "...", "likelihood": "high", "reasoning": "..."}\n'
        "  ]\n"
        "}"
    )


def _clean_str(value: object) -> str | None:
    """Return the stripped string, or None when it is not a non-empty string."""
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _normalise_pattern(raw: object) -> dict | None:
    """Coerce a raw model item into a pattern dict, or None if unusable."""
    if not isinstance(raw, dict):
        return None
    topic = _clean_str(raw.get("topic"))
    if topic is None:
        return None
    return {
        "topic": topic,
        "frequency_note": _clean_str(raw.get("frequency_note")) or "",
        "trend": _clean_str(raw.get("trend")) or "",
    }


def _normalise_prediction(raw: object) -> dict | None:
    """Coerce a raw model item into a prediction dict, or None if unusable.

    Tolerant of likelihood case/whitespace drift ("High", " medium ") but an
    unrecognised likelihood demotes to "low" rather than dropping the question.
    """
    if not isinstance(raw, dict):
        return None
    question = _clean_str(raw.get("question"))
    if question is None:
        return None
    likelihood = (_clean_str(raw.get("likelihood")) or "").lower()
    if likelihood not in _VALID_LIKELIHOODS:
        likelihood = "low"
    return {
        "question": question,
        "topic": _clean_str(raw.get("topic")) or "",
        "likelihood": likelihood,
        "reasoning": _clean_str(raw.get("reasoning")) or "",
    }


def predict_questions(corpus: str, subject_hint: str | None = None, model: str | None = None) -> dict:
    """Analyze a combined past-paper corpus in ONE Groq call and return validated
    {"patterns": [...], "predicted_questions": [...]}.

    The caller is responsible for clipping ``corpus`` to MAX_CONTENT_TOKENS so the
    request stays inside the provider's per-minute token budget.
    """
    if not corpus or not corpus.strip():
        raise PredictorError("No past-paper text was supplied to analyze.")
    require_groq_key()
    resolved = model_for("predictor", model)
    client = Groq()
    try:
        response = client.chat.completions.create(
            model=resolved,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert exam analyst who spots patterns in past papers and predicts likely future questions.",
                },
                {"role": "user", "content": _build_prompt(corpus, subject_hint)},
            ],
            response_format={"type": "json_object"},
            temperature=0.4,
            max_tokens=_MAX_OUTPUT_TOKENS,
        )
        payload = response.choices[0].message.content
        data = json.loads(payload or "{}")
    except json.JSONDecodeError as exc:
        raise PredictorError(f"The model returned invalid JSON: {exc}") from exc
    except APIError:
        raise  # Groq errors (rate limit, auth, …) are formatted by the global handlers
    except Exception as exc:  # noqa: BLE001 — wrap any other transport/SDK failure
        raise PredictorError(f"Question prediction failed: {exc}") from exc

    if not isinstance(data, dict):
        raise PredictorError("The model returned an unexpected payload shape.")
    raw_patterns = data.get("patterns")
    raw_predictions = data.get("predicted_questions")
    patterns = [p for item in (raw_patterns if isinstance(raw_patterns, list) else [])
                if (p := _normalise_pattern(item)) is not None][:_MAX_PATTERNS]
    predictions = [p for item in (raw_predictions if isinstance(raw_predictions, list) else [])
                   if (p := _normalise_prediction(item)) is not None][:_MAX_PREDICTIONS]
    if not predictions:
        raise PredictorError("The model returned no usable predicted questions; try again.")
    return {"patterns": patterns, "predicted_questions": predictions}
