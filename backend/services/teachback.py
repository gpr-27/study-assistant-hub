"""Socratic Teach-back tutor service — the student explains a concept, the LLM probes.

The tutor uses the Feynman technique: it never lectures or gives the answer; it asks ONE pointed
follow-up question per turn and, after 4-6 exchanges (or when the student says "done"/"evaluate"),
returns a structured evaluation of the student's understanding. Finished sessions are persisted
to the "teachbacks" history kind.

Stateless-friendly: a live session is NOT kept in a process-global dict. It lives in MongoDB
(``teachback_sessions``, TTL-expired after 3 h, owner-scoped on every access) so any serverless
instance can serve any turn; with persistence off it falls back to a small bounded in-process LRU
(single-process dev). Every LLM call goes through ``backend.llm_chain`` (JSON mode).
"""
import logging
import threading
import time
from collections import OrderedDict
from uuid import uuid4

from backend import db, llm_chain, privacy
from backend.auth import LOCAL_OWNER
from backend.config import require_llm_key
from backend.services.llm_json import extract_json

log = logging.getLogger(__name__)

# In-process fallback (persistence off): {session_id: session}. Bounded LRU + TTL.
_local: "OrderedDict[str, dict]" = OrderedDict()
_local_lock = threading.Lock()
_MAX_SESSIONS: int = 200
_SESSION_TTL_S: int = 3 * 3600
_ttl_index_ready = False

# A session hard-expires once its transcript reaches this many messages.
_MAX_MESSAGES: int = 50
# The tutor is told to wrap up between these exchange counts; at the max we force it.
_MIN_EXCHANGES: int = 4
_MAX_EXCHANGES: int = 6

# Token budgets (free tiers count input AND reserved output per minute, so keep every request
# small: clipped grounding context + short history + a capped completion; the chain raises the
# output cap to its own floor where reasoning models need it).
_START_OUTPUT_TOKENS: int = 350   # concept + opening prompt
_REPLY_OUTPUT_TOKENS: int = 700   # one question, or the final evaluation lists

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


def _call_llm(messages: list[dict], max_tokens: int, system: str, sensitive: bool = False) -> tuple[dict, dict | None]:
    """One chain JSON-mode call -> (parsed dict, public llm). ``LLMChainError`` propagates (→ 503)."""
    require_llm_key()
    # Providers require the conversation to start with a user turn; the transcript starts with the
    # tutor's opening, so a short synthetic user turn goes first.
    if not messages or messages[0]["role"] != "user":
        messages = [{"role": "user", "content": "Begin the teach-back session."}, *messages]
    result = llm_chain.chat(system, messages, max_tokens=max_tokens, temperature=0.4, json_mode=True,
                            sensitive=sensitive or privacy.current())
    data = extract_json(result.text)
    if data is None:
        raise TeachbackError("The model returned invalid JSON; please try again.")
    return (data if isinstance(data, dict) else {}), result.to_public()


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


# --------------------------------------------------------------------------------------
# Session storage (MongoDB, owner-scoped; bounded in-process fallback when persistence is off)
# --------------------------------------------------------------------------------------
def _coll():
    global _ttl_index_ready
    database = db._get_db()
    if database is None:
        return None
    coll = database["teachback_sessions"]
    if not _ttl_index_ready:
        try:
            coll.create_index("created_at", expireAfterSeconds=_SESSION_TTL_S, background=True)
            _ttl_index_ready = True
        except Exception:  # noqa: BLE001 — index creation is best-effort; the session still works
            log.debug("teachback TTL index not created.", exc_info=True)
    return coll


def _store_new(session: dict) -> str:
    session_id = uuid4().hex
    coll = _coll()
    if coll is not None:
        coll.insert_one({"_id": session_id, "created_at": db._now(), **session})
        return session_id
    with _local_lock:
        _local[session_id] = {**session, "_t": time.monotonic()}
        while len(_local) > _MAX_SESSIONS:
            _local.popitem(last=False)
    return session_id


def _store_get(owner: str, session_id: str) -> dict | None:
    """The session if it exists AND belongs to ``owner`` (else None — never leaks another user's)."""
    if not isinstance(session_id, str) or not session_id or len(session_id) > 64:
        return None
    coll = _coll()
    if coll is not None:
        return coll.find_one({"_id": session_id, "owner": owner})
    with _local_lock:
        s = _local.get(session_id)
        if s is None or s.get("owner") != owner or time.monotonic() - s["_t"] > _SESSION_TTL_S:
            return None
        _local.move_to_end(session_id)
        return {**s, "messages": list(s["messages"])}


def _store_append(owner: str, session_id: str, new_messages: list[dict]) -> None:
    coll = _coll()
    if coll is not None:
        coll.update_one({"_id": session_id, "owner": owner},
                        {"$push": {"messages": {"$each": new_messages}}, "$inc": {"exchanges": 1}})
        return
    with _local_lock:
        s = _local.get(session_id)
        if s is not None and s.get("owner") == owner:
            s["messages"].extend(new_messages)
            s["exchanges"] += 1


def _store_drop(owner: str, session_id: str) -> bool:
    """Delete a session; True only for the caller that actually removed it (double-finish guard)."""
    coll = _coll()
    if coll is not None:
        return coll.delete_one({"_id": session_id, "owner": owner}).deleted_count > 0
    with _local_lock:
        s = _local.get(session_id)
        if s is not None and s.get("owner") == owner:
            del _local[session_id]
            return True
    return False


def start_session(
    mode: str,
    *,
    topic: str | None = None,
    content: str | None = None,
    label: str = "",
    model: str | None = None,
    owner: str = LOCAL_OWNER,
) -> tuple[str, str, str, dict | None]:
    """Pick the concept + opening prompt and open a live session.

    ``content`` (subject mode) is the already-extracted PDF text; it is used in full as grounding
    (llm_chain trims per provider). Returns ``(session_id, concept, opening, llm_public)``.
    ``model`` is accepted for back-compat and ignored (the chain picks the model).
    """
    context = content or None
    raw, llm_public = _call_llm(
        [{"role": "user", "content": _start_prompt(topic, context)}],
        _START_OUTPUT_TOKENS, "You set up Feynman-technique teach-back study sessions.",
    )
    concept = str(raw.get("concept") or "").strip()[:200] or (topic or label or "this concept").strip()
    opening = str(raw.get("opening") or "").strip() or (
        f"Explain {concept} to me in your own words, as if I knew nothing about it."
    )
    session_id = _store_new({
        "mode": mode,
        "concept": concept,
        "label": (label or concept)[:200],
        "owner": owner,
        "persona": _persona(concept, context),
        "messages": [{"role": "assistant", "content": opening, "llm": llm_public}],  # full transcript
        "exchanges": 0,
        "sensitive": privacy.current(),          # later turns keep the same privacy class
    })
    return session_id, concept, opening, llm_public


def _history_messages(transcript: list[dict]) -> list[dict]:
    """The whole transcript (already bounded by the 50-message session limit) as chat messages."""
    return [{"role": m["role"], "content": m["content"]} for m in transcript]


def _record_result(owner: str, session_id: str, session: dict, evaluation: dict, llm_public: dict | None) -> dict | None:
    """The unified result for a finished session, stored once as a graded attempt (feeds Analytics). Never raises."""
    try:
        from backend.services import attempts, grading

        subject = session["label"].split(" / ")[0] if session.get("mode") == "subject" else None
        return attempts.run_idempotent(owner, session_id, "teachback", lambda: grading.teachback_result(
            attempt_id=session_id, concept=session["concept"], subject=subject, evaluation=evaluation,
            llm=llm_public))[0]
    except Exception:  # noqa: BLE001 - analytics must never break the evaluation the student is waiting for
        log.warning("Could not record the teach-back result.", exc_info=True)
        return None


def _save_and_close(owner: str, session_id: str, session: dict, evaluation: dict, llm_public: dict | None) -> dict | None:
    """Persist the finished session to history (id or None tolerated) and drop it (once). Returns the unified result."""
    if not _store_drop(owner, session_id):
        return None   # a concurrent finish already saved it
    db.save("teachbacks", owner, {
        "title": f"Teach-back · {session['concept']}",
        "concept": session["concept"],
        "mode": session["mode"],
        "source": session["label"],
        "llm": llm_public,
        "messages": list(session["messages"]),
        "evaluation": evaluation,
    })
    return _record_result(owner, session_id, session, evaluation, llm_public)


def respond(session_id: str, message: str, owner: str = LOCAL_OWNER) -> dict:
    """Handle one student message; returns {reply, done, evaluation, llm}.

    The session ends (done=True, evaluation attached, transcript saved, session cleared) when the
    student says "done"/"evaluate" or the exchange cap is hit; otherwise the tutor returns its
    single probing question. Owner-scoped: a foreign session id is "unknown".
    """
    session = _store_get(owner, session_id)
    if session is None:
        raise TeachbackSessionError(f"Unknown or expired teach-back session: {session_id}")
    if len(session["messages"]) >= _MAX_MESSAGES:
        _store_drop(owner, session_id)
        raise TeachbackSessionExpiredError(
            "This teach-back session reached its 50-message limit — nice stamina! "
            "Please start a fresh session to keep going."
        )

    text = (message or "").strip()
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

    llm_messages = _history_messages(session["messages"])
    llm_messages.append({"role": "user", "content": user_content})

    raw, llm_public = _call_llm(llm_messages, _REPLY_OUTPUT_TOKENS, session["persona"],
                                sensitive=bool(session.get("sensitive")))
    evaluation = _normalise_evaluation(raw.get("evaluation"))
    done = bool(raw.get("done"))
    if must_finish and (not done or evaluation is None):
        # One retry before giving up — a forced finish must yield an evaluation.
        raw, llm_public = _call_llm(llm_messages, _REPLY_OUTPUT_TOKENS, session["persona"],
                                sensitive=bool(session.get("sensitive")))
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

    new_messages = [{"role": "user", "content": text},
                    {"role": "assistant", "content": reply, "llm": llm_public}]    # each AI turn keeps its own model label
    if done:
        session["messages"] = list(session["messages"]) + new_messages
        result = _save_and_close(owner, session_id, session, evaluation, llm_public)
        return {"reply": reply, "done": True, "evaluation": evaluation, "llm": llm_public, "result": result}
    _store_append(owner, session_id, new_messages)
    return {"reply": reply, "done": False, "evaluation": None, "llm": llm_public}
