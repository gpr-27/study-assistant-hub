"""Mock Exam service — Groq JSON mode, mirroring the quiz service's shape.

Two LLM entry points, each a SINGLE bounded Groq call:

  * ``generate_exam``       — build a mixed MCQ + short-answer paper from one
                              bounded slice of the selected PDFs' text.
  * ``grade_short_answers`` — grade ALL answered short questions in one batched
                              JSON call returning per-question score + feedback.

MCQs are graded in plain code by the router (compare ``answer_index``) — no
model call is spent on them. Every model payload is validated here; malformed
questions are dropped rather than raised on, and an exam with zero usable
questions is the router's 502.
"""
import json

from groq import APIError, Groq

from backend.config import MAX_CONTENT_TOKENS, model_for, require_groq_key
from backend.documents import clip_to_tokens

# Hard caps from the feature contract.
MAX_MCQ: int = 10
MAX_SHORT: int = 4
DEFAULT_MCQ: int = 6
DEFAULT_SHORT: int = 2
MCQ_OPTIONS: int = 4
SHORT_MAX_SCORE: int = 5
_MAX_EXPECTED_POINTS: int = 6

# Output (completion) caps. Groq's free tier counts the RESERVED output against the
# per-minute token budget, so size the cap to the actual JSON we expect back.
_GEN_TOKENS_PER_MCQ: int = 140
_GEN_TOKENS_PER_SHORT: int = 130
_GEN_MIN_OUTPUT: int = 512
_GEN_MAX_OUTPUT: int = 2048
_GRADE_TOKENS_PER_ITEM: int = 140
_GRADE_MIN_OUTPUT: int = 256
_GRADE_MAX_OUTPUT: int = 1024

# Student answers are untrusted free text — clip each one before it is embedded
# in the grading prompt so a pasted essay can't blow the per-minute token budget.
_MAX_ANSWER_TOKENS: int = 400


class ExamError(Exception):
    """Raised when an exam cannot be generated/graded or parsed from model output."""


def _chat_json(prompt: str, *, system: str, model: str | None, max_tokens: int) -> dict:
    """One Groq JSON-mode call -> parsed dict. APIError bubbles (global handlers)."""
    require_groq_key()
    client = Groq()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.4,
            max_tokens=max_tokens,
        )
        payload = response.choices[0].message.content
        data = json.loads(payload or "{}")
    except json.JSONDecodeError as exc:
        raise ExamError(f"The model returned invalid JSON: {exc}") from exc
    except APIError as exc:
        # Groq JSON mode can 400 ("failed to generate JSON" / json_validate_failed) when the model
        # overflows max_tokens or emits non-JSON — a CONTENT failure (retry-able), NOT a rate-limit/
        # auth/transport error. Convert it to a clean ExamError so the chat shows a friendly "try
        # again" instead of dumping the raw provider error (incl. the giant failed_generation).
        _code = (getattr(exc, "code", "") or "").lower()
        _emsg = str(getattr(exc, "message", "") or exc).lower()
        if _code == "json_validate_failed" or "failed to generate json" in _emsg or "failed_generation" in _emsg:
            raise ExamError("The model had trouble formatting the exam — please try again.") from exc
        raise  # rate-limit/auth/transport errors are formatted by the global handlers
    except Exception as exc:  # noqa: BLE001 — wrap any other SDK/transport failure
        raise ExamError(f"The exam call failed: {exc}") from exc
    return data if isinstance(data, dict) else {}


# ---------------------------------------------------------------------------
# Question validation — shared by generation and (re-)validation before grading.
# ---------------------------------------------------------------------------
def _clean_topic(value: object) -> str:
    return value.strip() if isinstance(value, str) and value.strip() else "General"


def _normalise_mcq(raw: dict) -> dict | None:
    """Coerce a raw model/client item into a valid MCQ dict, or None."""
    question = raw.get("question")
    options = raw.get("options")
    answer = raw.get("answer_index")
    if not isinstance(question, str) or not question.strip():
        return None
    if not isinstance(options, list):
        return None
    opts = [o.strip() for o in options if isinstance(o, str) and o.strip()]
    if len(opts) != MCQ_OPTIONS:
        return None
    if isinstance(answer, bool):
        return None
    if isinstance(answer, str) and answer.strip().lstrip("-").isdigit():
        answer = int(answer.strip())
    if isinstance(answer, float) and answer.is_integer():
        answer = int(answer)
    if not isinstance(answer, int) or not 0 <= answer < MCQ_OPTIONS:
        return None
    return {
        "type": "mcq",
        "question": question.strip(),
        "options": opts,
        "answer_index": answer,
        "topic": _clean_topic(raw.get("topic")),
    }


def _normalise_short(raw: dict) -> dict | None:
    """Coerce a raw model/client item into a valid short-answer dict, or None."""
    question = raw.get("question")
    points = raw.get("expected_points")
    if not isinstance(question, str) or not question.strip():
        return None
    if not isinstance(points, list):
        return None
    cleaned = [p.strip() for p in points if isinstance(p, str) and p.strip()]
    if not cleaned:
        return None
    return {
        "type": "short",
        "question": question.strip(),
        "expected_points": cleaned[:_MAX_EXPECTED_POINTS],
        "topic": _clean_topic(raw.get("topic")),
    }


def validate_questions(items: object) -> list[dict]:
    """Validate a raw questions list; drop malformed items; ensure unique ids.

    A valid client-supplied id is preserved (grading keys answers by id);
    anything else gets a fresh sequential ``q<n>`` that never collides.
    """
    out: list[dict] = []
    seen: set[str] = set()
    for raw in items if isinstance(items, list) else []:
        if not isinstance(raw, dict):
            continue
        kind = raw.get("type")
        if kind == "mcq":
            q = _normalise_mcq(raw)
        elif kind == "short":
            q = _normalise_short(raw)
        else:  # no/unknown type tag — accept whichever shape the item satisfies
            q = _normalise_mcq(raw) or _normalise_short(raw)
        if q is None:
            continue
        rid = raw.get("id")
        rid = rid.strip() if isinstance(rid, str) and rid.strip() else ""
        if not rid or rid in seen:
            n = len(out) + 1
            rid = f"q{n}"
            while rid in seen:
                n += 1
                rid = f"q{n}"
        seen.add(rid)
        q["id"] = rid
        out.append(q)
    return out


# ---------------------------------------------------------------------------
# Generation.
# ---------------------------------------------------------------------------
def _build_exam_prompt(content: str, num_mcq: int, num_short: int, difficulty: str = "medium") -> str:
    return (
        f"Create a mock exam from the study content below at a {difficulty} difficulty level: "
        f"exactly {num_mcq} multiple choice questions and exactly {num_short} short-answer questions.\n\n"
        "Content:\n"
        f"{content}\n\n"
        "Rules:\n"
        f"- Produce EXACTLY {num_mcq} MCQ and EXACTLY {num_short} short-answer questions — no more, no fewer.\n"
        f"- Calibrate the questions to a {difficulty} difficulty level.\n"
        "- Questions must test understanding of the content, not rote memorisation.\n"
        "- Tag every question with a short topic label drawn from the content.\n"
        "- Each MCQ has exactly 4 options and one correct option (answer_index 0-3).\n"
        "- Each short-answer question lists 2-4 expected_points a good answer covers.\n"
        "- Output ONLY one minified JSON object — no markdown fences, no commentary, nothing after the final brace.\n\n"
        "Return ONLY a JSON object with this exact structure:\n"
        "{\n"
        '  "questions": [\n'
        '    {"type": "mcq", "question": "...", "options": ["...", "...", "...", "..."],'
        ' "answer_index": 0, "topic": "..."},\n'
        '    {"type": "short", "question": "...", "expected_points": ["...", "..."],'
        ' "topic": "..."}\n'
        "  ]\n"
        "}"
    )


def _build_exam_topic_prompt(topic: str, num_mcq: int, num_short: int) -> str:
    """Build an exam prompt for a free topic (no document — uses the model's own
    knowledge); the word JSON keeps Groq JSON mode happy."""
    return (
        f'Create a mock exam about the topic: "{topic}" — exactly {num_mcq} multiple '
        f"choice questions and exactly {num_short} short-answer questions.\n\n"
        "Rules:\n"
        "- Questions must test understanding of the topic, not rote memorisation.\n"
        "- Tag every question with a short topic label (a sub-area of the topic).\n"
        "- Each MCQ has exactly 4 options and one correct option (answer_index 0-3).\n"
        "- Each short-answer question lists 2-4 expected_points a good answer covers.\n\n"
        "Return ONLY a JSON object with this exact structure:\n"
        "{\n"
        '  "questions": [\n'
        '    {"type": "mcq", "question": "...", "options": ["...", "...", "...", "..."],'
        ' "answer_index": 0, "topic": "..."},\n'
        '    {"type": "short", "question": "...", "expected_points": ["...", "..."],'
        ' "topic": "..."}\n'
        "  ]\n"
        "}"
    )


def _gen_output_tokens(num_mcq: int, num_short: int) -> int:
    budget = num_mcq * _GEN_TOKENS_PER_MCQ + num_short * _GEN_TOKENS_PER_SHORT + 256
    return max(_GEN_MIN_OUTPUT, min(_GEN_MAX_OUTPUT, budget))


def _shape_exam_output(data: dict, mcq: int, short: int) -> dict:
    """Validate a raw model payload and cap each question type to what was asked."""
    questions = validate_questions(data.get("questions"))
    mcqs = [q for q in questions if q["type"] == "mcq"][:mcq]
    shorts = [q for q in questions if q["type"] == "short"][:short]
    return {"questions": mcqs + shorts}


def generate_exam(content: str, num_mcq: int = DEFAULT_MCQ, num_short: int = DEFAULT_SHORT,
                  difficulty: str = "medium", model: str | None = None) -> dict:
    """Build the exam paper from document text in ONE bounded Groq call.

    Returns ``{"questions": [...]}`` with server-validated questions (each given
    a unique id). Malformed items are dropped, never raised on; an entirely
    unusable payload simply yields an empty list (the router's 502).
    """
    if not content or not content.strip():
        raise ExamError("No content was supplied to build an exam from.")
    mcq = max(0, min(MAX_MCQ, int(num_mcq)))
    short = max(0, min(MAX_SHORT, int(num_short)))
    if mcq + short == 0:
        raise ExamError("An exam needs at least one question.")
    snippet = clip_to_tokens(content, MAX_CONTENT_TOKENS)
    data = _chat_json(
        _build_exam_prompt(snippet, mcq, short, difficulty),
        system="You are an examiner writing a fair mock exam from course material.",
        model=model_for("exam", model),
        max_tokens=_gen_output_tokens(mcq, short),
    )
    # Never hand back more than was asked for, per type.
    return _shape_exam_output(data, mcq, short)


def generate_exam_from_topic(topic: str, num_mcq: int = DEFAULT_MCQ, num_short: int = DEFAULT_SHORT,
                             model: str | None = None) -> dict:
    """Build an exam paper about a free topic, using the model's own knowledge.

    Same return shape and validation as ``generate_exam`` ({"questions": [...]}
    with server-validated, uniquely-id'd items). One bounded Groq call is enough
    since there is no document to chunk; malformed items are dropped, never raised
    on, and an entirely unusable payload yields an empty list (the router's 502).
    """
    if not topic or not topic.strip():
        raise ExamError("No topic was supplied to build an exam from.")
    mcq = max(0, min(MAX_MCQ, int(num_mcq)))
    short = max(0, min(MAX_SHORT, int(num_short)))
    if mcq + short == 0:
        raise ExamError("An exam needs at least one question.")
    prompt = _build_exam_topic_prompt(topic.strip(), mcq, short)
    resolved = model_for("exam", model)
    budget = _gen_output_tokens(mcq, short)
    # Single Groq call here, so one malformed roll = total failure. Retry once
    # before giving up (the global handler still surfaces a clear 502 if both fail).
    data = _chat_json(
        prompt,
        system="You are an examiner writing a fair mock exam from your own subject knowledge.",
        model=resolved,
        max_tokens=budget,
    )
    shaped = _shape_exam_output(data, mcq, short)
    if not shaped["questions"]:
        data = _chat_json(
            prompt,
            system="You are an examiner writing a fair mock exam from your own subject knowledge.",
            model=resolved,
            max_tokens=budget,
        )
        shaped = _shape_exam_output(data, mcq, short)
    return shaped


# ---------------------------------------------------------------------------
# Grading (short answers only — MCQs are graded in code by the router).
# ---------------------------------------------------------------------------
def _build_grading_prompt(items: list[dict]) -> str:
    payload = [
        {
            "id": it["id"],
            "question": it["question"],
            "expected_points": it["expected_points"],
            "student_answer": clip_to_tokens(it["answer"], _MAX_ANSWER_TOKENS),
        }
        for it in items
    ]
    return (
        "Grade each student short answer below against its expected points. "
        f"Score each from 0 to {SHORT_MAX_SCORE} ({SHORT_MAX_SCORE} = covers all expected "
        "points accurately; 0 = irrelevant or wrong) and give one or two sentences of "
        "specific, encouraging feedback.\n\n"
        "Answers to grade:\n"
        f"{json.dumps(payload, ensure_ascii=False)}\n\n"
        "Return ONLY a JSON object with this exact structure:\n"
        "{\n"
        '  "grades": [\n'
        '    {"id": "q1", "score_0_to_5": 3, "feedback": "..."}\n'
        "  ]\n"
        "}"
    )


def _coerce_score(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        value = int(value.strip())
    if isinstance(value, float):
        value = round(value)
    if not isinstance(value, int):
        return None
    return max(0, min(SHORT_MAX_SCORE, value))


def grade_short_answers(items: list[dict], model: str | None = None) -> dict[str, dict]:
    """Grade answered short questions in ONE batched Groq JSON call.

    ``items``: ``[{"id", "question", "expected_points", "answer"}, ...]``.
    Returns ``{id: {"score": 0-5, "feedback": str}}`` covering EVERY requested id —
    an id the model skipped or mangled gets a conservative 0 with a clear note,
    so one flaky row never sinks the whole report.
    """
    if not items:
        return {}
    data = _chat_json(
        _build_grading_prompt(items),
        system="You are a fair, consistent examiner grading short answers.",
        model=model_for("exam", model),
        max_tokens=max(_GRADE_MIN_OUTPUT,
                       min(_GRADE_MAX_OUTPUT, len(items) * _GRADE_TOKENS_PER_ITEM + 128)),
    )
    raw = data.get("grades")
    if not isinstance(raw, list):
        raise ExamError("The grader returned no usable grades; try again.")
    graded: dict[str, dict] = {}
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        gid = entry.get("id")
        score = _coerce_score(entry.get("score_0_to_5"))
        if not isinstance(gid, str) or score is None:
            continue
        feedback = entry.get("feedback")
        graded[gid] = {
            "score": score,
            "feedback": feedback.strip() if isinstance(feedback, str) and feedback.strip()
            else "No feedback was returned for this answer.",
        }
    return {
        it["id"]: graded.get(it["id"], {
            "score": 0,
            "feedback": "The grader did not return a result for this answer.",
        })
        for it in items
    }
