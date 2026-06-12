"""Socratic Teach-back tutor service — the student explains a concept, the LLM probes.

Modeled on backend/services/chat.py's in-module session registry (dict + uuid +
eviction cap) and backend/services/quiz.py's Groq JSON-mode call pattern. The tutor
uses the Feynman technique: it never lectures or gives the answer; it asks ONE
pointed follow-up question per turn and, after 4-6 exchanges (or when the student
says "done"/"evaluate"), returns a structured evaluation of the student's
understanding. Finished sessions are persisted to the "teachbacks" history kind.

Every Groq object is created lazily inside a function so importing this module
never requires a key. All calls set an explicit, conservative max_tokens because
the free tier's per-minute budget counts input AND reserved output.
"""
import json
import threading
from uuid import uuid4

from groq import APIError, Groq

from backend import db
from backend.auth import LOCAL_OWNER
from backend.config import model_for, require_groq_key
from backend.documents import clip_to_tokens

# In-memory session store: {session_id: {concept, persona, messages, exchanges,
# owner, mode, label, model}}. Oldest sessions are evicted past the cap (an
# evicted session simply needs restarting), mirroring chat.py.
_sessions: dict[str, dict] = {}
_sessions_lock = threading.Lock()
_MAX_SESSIONS: int = 200

# A session hard-expires once its transcript reaches this many messages.
_MAX_MESSAGES: int = 50
# The tutor is told to wrap up between these exchange counts; at the max we force it.
_MIN_EXCHANGES: int = 4
_MAX_EXCHANGES: int = 6

# Token budgets (Groq free tier = 6000 tokens/min, input + output, so keep every
# request small: clipped grounding context + short history + a capped completion).
_CONTEXT_TOKENS: int = 1200       # grounding context kept in the persona each turn
_MESSAGE_TOKENS: int = 600        # cap on a single student message
_START_OUTPUT_TOKENS: int = 350   # concept + opening prompt
_REPLY_OUTPUT_TOKENS: int = 700   # one question, or the final evaluation lists
_HISTORY_TURNS: int = 8           # exchanges replayed to the model (> _MAX_EXCHANGES)

# Student phrases that end the session immediately (the spec's "done"/"evaluate";
# the UI's "Evaluate me now" button posts the literal message "evaluate").
_FINISH_WORDS: frozenset[str] = frozenset({
    "done", "evaluate", "evaluate me", "evaluate me now", "i am done", "i'm done", "im done",
})


class TeachbackError(Exception):
    """Raised when the tutor's reply cannot be generated or parsed (-> 502)."""


class TeachbackSessionError(Exception):
    """Raised for an unknown/already-finished session id (-> 404)."""


class TeachbackSessionExpiredError(Exception):
    """Raised when a session hits the 50-message cap (-> 403, friendly)."""


def _persona(concept: str, context: str | None) -> str:
    """System persona for every tutoring turn: Socratic, Feynman, JSON-only."""
    grounding = (
        "\n\nGround your questions in this course material (do not quote it back at length):\n"
        f"{context}\n" if context else ""
    )
    return (
        "You are a Socratic tutor using the Feynman technique. The student is teaching YOU "
        f'the concept "{concept}" in their own words to prove they understand it.{grounding}\n'
        "STRICT RULES:\n"
        "- NEVER lecture, explain the concept yourself, or give the answer.\n"
        "- Probe the student's explanation with exactly ONE pointed follow-up question per reply.\n"
        "- If the student says something wrong, do NOT correct them — ask a question that exposes "
        "the contradiction so they discover it themselves.\n"
        "- Keep each reply to 1-3 sentences ending in your single question.\n"
        f"- After {_MIN_EXCHANGES}-{_MAX_EXCHANGES} exchanges, or as soon as the student says "
        '"done" or "evaluate", stop questioning and produce the final evaluation.\n\n'
        "Respond ONLY with a JSON object. While the session continues:\n"
        '{"reply": "<your one probing question>", "done": false, "evaluation": null}\n'
        "When finishing:\n"
        '{"reply": "<a short closing remark>", "done": true, "evaluation": {'
        '"understanding_score_0_to_10": <integer 0-10>, '
        '"strengths": ["..."], "gaps": ["..."], "suggested_review": ["..."]}}'
    )


def _start_prompt(topic: str | None, context: str | None) -> str:
    """One-shot prompt that picks the concept and writes the opening challenge (JSON)."""
    if context:
        head = (
            "From the course material below, pick ONE specific concept a student should be "
            "able to explain from memory.\n\nMaterial:\n"
            f"{context}"
        )
    else:
        head = (
            f'The student wants to practise the topic: "{topic}". '
            "Pick ONE specific, teachable concept within that topic."
        )
    return (
        f"{head}\n\n"
        "You are setting up a Feynman-technique teach-back session: the student will explain "
        "the concept to you in their own words. Write the opening prompt in the form "
        '"Explain <concept> to me in your own words, as if I knew nothing about it." '
        "(you may add one short encouraging sentence).\n\n"
        "Return ONLY a JSON object with this exact structure:\n"
        '{"concept": "...", "opening": "Explain <concept> to me in your own words..."}'
    )


def _call_llm(model_id: str | None, messages: list[dict], max_tokens: int) -> dict:
    """One Groq JSON-mode call -> parsed dict. Groq APIErrors bubble to the global handlers."""
    require_groq_key()
    client = Groq()
    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.4,
            max_tokens=max_tokens,
        )
        payload = response.choices[0].message.content
        data = json.loads(payload or "{}")
    except json.JSONDecodeError as exc:
        raise TeachbackError(f"The model returned invalid JSON: {exc}") from exc
    except APIError:
        raise  # formatted by the global Groq handlers (rate limit, auth, …)
    except Exception as exc:  # noqa: BLE001 — wrap any other transport/SDK failure
        raise TeachbackError(f"The tutor could not respond: {exc}") from exc
    return data if isinstance(data, dict) else {}


def _str_list(value: object) -> list[str]:
    """Coerce a model-supplied list into a clean list of non-empty strings."""
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _normalise_evaluation(raw: object) -> dict | None:
    """Shape the model's evaluation into the response contract, or None if unusable."""
    if not isinstance(raw, dict):
        return None
    try:
        score = int(raw.get("understanding_score_0_to_10"))
    except (TypeError, ValueError):
        return None
    return {
        "understanding_score_0_to_10": max(0, min(10, score)),
        "strengths": _str_list(raw.get("strengths")),
        "gaps": _str_list(raw.get("gaps")),
        "suggested_review": _str_list(raw.get("suggested_review")),
    }


def _register(session: dict) -> str:
    """Store a session under a fresh id, evicting the oldest past the cap."""
    session_id = uuid4().hex
    with _sessions_lock:
        _sessions[session_id] = session
        while len(_sessions) > _MAX_SESSIONS:
            del _sessions[next(iter(_sessions))]  # dicts preserve insertion order
    return session_id


def start_session(
    mode: str,
    *,
    topic: str | None = None,
    content: str | None = None,
    label: str = "",
    model: str | None = None,
    owner: str = LOCAL_OWNER,
) -> tuple[str, str, str]:
    """Pick the concept + opening prompt and open a live session.

    ``content`` (subject mode) is the already-extracted PDF text; it is clipped
    again here to the small per-turn grounding budget. Returns
    (session_id, concept, opening).
    """
    resolved = model_for("teachback", model)
    context = clip_to_tokens(content, _CONTEXT_TOKENS) if content else None
    raw = _call_llm(
        resolved,
        [
            {"role": "system", "content": "You set up Feynman-technique teach-back study sessions."},
            {"role": "user", "content": _start_prompt(topic, context)},
        ],
        _START_OUTPUT_TOKENS,
    )
    concept = str(raw.get("concept") or "").strip() or (topic or label or "this concept").strip()
    opening = str(raw.get("opening") or "").strip() or (
        f"Explain {concept} to me in your own words, as if I knew nothing about it."
    )
    session_id = _register({
        "mode": mode,
        "concept": concept,
        "label": label or concept,
        "model": resolved,
        "owner": owner,
        "persona": _persona(concept, context),
        "messages": [{"role": "assistant", "content": opening}],  # full transcript
        "exchanges": 0,
    })
    return session_id, concept, opening


def _history_messages(transcript: list[dict]) -> list[dict]:
    """The last few transcript entries as plain chat messages for the model."""
    recent = transcript[-(_HISTORY_TURNS * 2 + 1):]
    return [{"role": m["role"], "content": m["content"]} for m in recent]


def _save_and_close(session_id: str, session: dict, evaluation: dict) -> None:
    """Persist the finished session to history (id or None tolerated) and drop it."""
    db.save("teachbacks", session["owner"], {
        "title": f"Teach-back · {session['concept']}",
        "concept": session["concept"],
        "mode": session["mode"],
        "source": session["label"],
        "model": session["model"],
        "messages": list(session["messages"]),
        "evaluation": evaluation,
    })
    with _sessions_lock:
        _sessions.pop(session_id, None)


def respond(session_id: str, message: str) -> dict:
    """Handle one student message; returns {reply, done, evaluation}.

    The session ends (done=True, evaluation attached, transcript saved, session
    cleared) when the student says "done"/"evaluate" or the exchange cap is hit;
    otherwise the tutor returns its single probing question.
    """
    session = _sessions.get(session_id)
    if session is None:
        raise TeachbackSessionError(f"Unknown or expired teach-back session: {session_id}")
    if len(session["messages"]) >= _MAX_MESSAGES:
        with _sessions_lock:
            _sessions.pop(session_id, None)
        raise TeachbackSessionExpiredError(
            "This teach-back session reached its 50-message limit — nice stamina! "
            "Please start a fresh session to keep going."
        )

    text = clip_to_tokens((message or "").strip(), _MESSAGE_TOKENS)
    if not text:
        raise TeachbackError("The message was empty — explain the concept in your own words.")

    wants_eval = text.lower().strip(" .!?,") in _FINISH_WORDS
    must_finish = wants_eval or (session["exchanges"] + 1 >= _MAX_EXCHANGES)

    user_content = text
    if must_finish:
        user_content += (
            "\n\n[The session is over. Do not ask another question — reply with done=true "
            "and the complete final evaluation JSON now.]"
        )

    llm_messages = [{"role": "system", "content": session["persona"]}]
    llm_messages.extend(_history_messages(session["messages"]))
    llm_messages.append({"role": "user", "content": user_content})

    raw = _call_llm(session["model"], llm_messages, _REPLY_OUTPUT_TOKENS)
    evaluation = _normalise_evaluation(raw.get("evaluation"))
    done = bool(raw.get("done"))
    if must_finish and (not done or evaluation is None):
        # One retry before giving up — a forced finish must yield an evaluation.
        raw = _call_llm(session["model"], llm_messages, _REPLY_OUTPUT_TOKENS)
        evaluation = _normalise_evaluation(raw.get("evaluation"))
        done = bool(raw.get("done"))
        if not done or evaluation is None:
            raise TeachbackError("The tutor could not produce a final evaluation; please try again.")
    if done and evaluation is None:
        done = False  # the model claimed done without a usable evaluation — keep probing

    reply = str(raw.get("reply") or "").strip()
    if not reply:
        reply = (
            "Thanks — here is my evaluation of your understanding." if done
            else "Go on — tell me more, in your own words."
        )

    session["messages"].append({"role": "user", "content": text})
    session["messages"].append({"role": "assistant", "content": reply})
    session["exchanges"] += 1

    if done:
        _save_and_close(session_id, session, evaluation)
        return {"reply": reply, "done": True, "evaluation": evaluation}
    return {"reply": reply, "done": False, "evaluation": None}
