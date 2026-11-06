"""Chat (RAG) service: subject / topic / uploaded-PDF study assistant — stateless & streaming.

Built on ``backend.llm_chain`` (the 5-key provider chain); no langchain / groq SDK.

Serverless-safe design
  * There is NO process-global session registry. A ``session_id`` is a *signed token* that
    carries everything needed to rebuild the session on ANY instance: the owner binding, mode,
    subject/topic, the persisted ``chat_id`` and a nonce. A token minted for one owner is
    rejected for every other owner (cross-user isolation) and can't be forged (HMAC).
  * Conversation memory is reconstructed per request from MongoDB (last N messages of the
    owner's chat) — or, when persistence is off, from the client (``history`` field) with a small
    bounded per-process fallback for single-process dev. History sent to the model is bounded
    by a token budget.
  * Uploaded-PDF chats store their text chunks in MongoDB (``chat_pdfs``), so any instance can
    rebuild the (lexical) retriever.

Time-to-first-token
  * No LLM call precedes generation on a normal question (see ``agentic_rag``); retrieval is
    bounded (~1.5 s for hybrid, then a lexical fallback) and the answer streams token by token.
  * DB writes, progress bookkeeping and chat auto-naming happen AFTER the final event, inside
    the same request (shielded from client disconnect) — never a daemon thread.

``stream_turn`` yields plain event dicts; ``routers/chat.py`` turns them into SSE:
``step`` {step}, ``model`` {llm}, ``token`` {text}, ``done`` {answer, sources, artifact, artifacts, llm},
``error`` {detail, llm, status}.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import AsyncIterator
from uuid import uuid4

from backend import agentic_rag, cache, config, db, llm_chain, privacy
from backend.auth import LOCAL_OWNER
from backend.documents import list_pdf_names
from backend.errors import llm_http_status, llm_unavailable
from backend.llm_chain import LLMChainError

log = logging.getLogger(__name__)

_RETRIEVE_K: int = 10      # richer context; llm_chain trims it for Groq only
# The model sees the WHOLE conversation; llm_chain trims per provider (Groq only). 200 exchanges is
# the storage bound (the DB keeps at most 400 messages), not a prompt budget.
_HISTORY_TURNS: int = 200
_CLIENT_MESSAGE_CHARS: int = 50000      # abuse bound on one client-sent history message
_MAX_STORED_MESSAGES: int = 400         # chats.messages is capped (Mongo 16 MB doc limit)
_MAX_MESSAGE_CHARS: int = 8000
_ANSWER_MAX_TOKENS: int = 2048
_RETRIEVAL_BUDGET_S: float = 1.5        # hybrid (embedding) retrieval budget before the lexical fallback
_LEXICAL_BUDGET_S: float = 2.5
_MAX_PDF_BYTES: int = 20 * 1024 * 1024
_MAX_PDF_CHARS: int = 600_000           # stored per uploaded-PDF chat
_LOCAL_MAX: int = 200                   # dev/persistence-off fallback store (bounded LRU)
_LOCAL_TURNS: int = 20

NO_LLM_LABEL = llm_chain.NO_RESULT_LABEL
UNAVAILABLE_DETAIL = ("The AI service is temporarily unavailable — every model provider failed. "
                      "Please try again in a moment.")

_SUBJECT_PERSONA_BASE: str = (
    "You are a skilled, thoughtful teaching assistant. Help the student understand "
    "the course material by guiding them through ideas. Answer using ONLY the provided "
    "context; if the answer is not supported by the context, clearly say you don't know. "
    "Keep responses clear, concise, and focused. Break complex concepts into simpler "
    "parts, use analogies or examples when helpful, and encourage critical thinking."
)
_GENERIC_PERSONA: str = (
    "You are a skilled, thoughtful teaching assistant. Help the student learn through "
    "clear explanations and examples. Keep responses clear, concise, and focused."
)
_SMALLTALK_PERSONA: str = (
    "You are a warm, concise study assistant. The student sent a greeting or a quick "
    "acknowledgement — reply in ONE short, friendly sentence and invite them to ask "
    "about their study material. Do not say you don't know; this is just small talk."
)

# Greetings / acknowledgements that need NO retrieval. A message counts as small talk only when
# it is short AND every word is a social token — so "define heap" or "ok now explain X" still
# retrieve, but "hi", "thank you", "ok cool" short-circuit to a single fast reply.
_SMALLTALK_WORD = re.compile(
    r"^(?:h+i+|h+e+l+o+|h+l+o+|h+e+y+|hiya|yo+|sup|wassup|howdy|hola|greetings|"
    r"good|morning|afternoon|evening|night|"
    r"thanks?|thankyou|thx|ty|tysm|"
    r"ok|okay|kk|k|cool|nice|great|awesome|amazing|perfect|fine|alright|"
    r"bye+|goodbye|cya|gn|gm|"
    r"please|pls|you|u|there|a|lot|much|so|mate|bro|buddy|man)$"
)


def _is_smalltalk(message: str) -> bool:
    """True for short greeting/acknowledgement messages that need no retrieval."""
    words = re.findall(r"[a-z']+", (message or "").lower())
    return bool(words) and len(words) <= 5 and all(_SMALLTALK_WORD.match(w) for w in words)


class ChatSessionError(Exception):
    """Raised when a chat session id is unknown, forged, expired or belongs to someone else."""


class ChatTurnError(Exception):
    """A chat turn failed; carries the HTTP status and the public ``llm`` block for the client."""

    def __init__(self, detail: str, *, status: int = 502, llm: dict | None = None):
        super().__init__(detail)
        self.detail = detail
        self.status = status
        self.llm = llm


# --------------------------------------------------------------------------------------
# Signed, stateless session tokens
# --------------------------------------------------------------------------------------
_RANDOM_SECRET = secrets.token_bytes(32)    # last resort when no secret material is configured
SESSION_TTL_S = 24 * 3600                   # a session token is valid for a day (refreshed by re-minting)
SESSION_REFRESH_BELOW_S = 3600              # ...and is re-minted by refresh_session() once < 1 h is left


def _secret() -> bytes:
    """Stable HMAC key shared by every serverless instance (derived from deployment secrets).

    ``SESSION_SECRET`` wins; otherwise a digest of the other configured secrets (Clerk / Mongo /
    provider keys — identical on every instance of one deployment, never exposed). With none
    configured (bare local dev) a per-process random key is used.
    """
    if config.session_secret_missing():
        # Production never derives the signing key from provider keys (rotating one would silently
        # invalidate every session, and a leaked provider key would let anyone forge sessions).
        raise config.SessionSecretMissing(
            "Chat sessions are unavailable: SESSION_SECRET is not configured on this deployment.")
    parts = [os.environ.get(n, "") for n in
             ("SESSION_SECRET", "CLERK_SECRET_KEY", "MONGODB_URI", *config.CHAIN_KEY_VARS)]
    material = "\x00".join(parts)
    if not material.strip("\x00"):
        return _RANDOM_SECRET
    return hashlib.sha256(b"study-hub-session-v1\x00" + material.encode()).digest()


def _owner_tag(owner: str) -> str:
    return hashlib.sha256(("owner\x00" + owner).encode()).hexdigest()[:16]


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode().rstrip("=")


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


@dataclass
class Session:
    id: str
    owner: str
    mode: str                  # "subject" | "topic" | "pdf"
    label: str
    subject: str | None
    topic: str | None
    chat_id: str | None
    nonce: str


def _mint(owner: str, mode: str, label: str, subject: str | None, topic: str | None,
          chat_id: str | None, nonce: str | None = None) -> Session:
    nonce = nonce or uuid4().hex[:16]
    payload = {"v": 1, "o": _owner_tag(owner), "m": mode, "l": label[:160],
               "s": subject, "t": topic, "c": chat_id, "n": nonce,
               "x": int(time.time()) + SESSION_TTL_S}          # expiry (epoch seconds)
    body = _b64(json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode())
    sig = _b64(hmac.new(_secret(), body.encode(), hashlib.sha256).digest()[:18])
    return Session(f"{body}.{sig}", owner, mode, payload["l"], subject, topic, chat_id, nonce)


def refresh_session(session: Session) -> str | None:
    """A fresh token for ``session`` when its current one has < ``SESSION_REFRESH_BELOW_S`` left
    (sliding expiry), else ``None``. The new token is identical apart from ``exp``."""
    try:
        body = session.id.split(".", 1)[0]
        left = int(json.loads(_unb64(body)).get("x") or 0) - time.time()
    except Exception:  # noqa: BLE001
        return None
    if left >= SESSION_REFRESH_BELOW_S:
        return None
    return _mint(session.owner, session.mode, session.label, session.subject, session.topic,
                 session.chat_id, session.nonce).id


def open_session(session_id: str, owner: str) -> Session:
    """Validate a session token for ``owner`` and rebuild its (immutable) description.

    Raises ``ChatSessionError`` for malformed, forged or foreign tokens — a token is only valid
    for the owner it was minted for, so one user can never drive another user's session.
    """
    try:
        body, sig = (session_id or "").split(".", 1)
        expect = _b64(hmac.new(_secret(), body.encode(), hashlib.sha256).digest()[:18])
        if not hmac.compare_digest(sig, expect):
            raise ValueError("bad signature")
        p = json.loads(_unb64(body))
        if p.get("v") != 1 or not hmac.compare_digest(str(p.get("o")), _owner_tag(owner)):
            raise ValueError("owner mismatch")
        if int(p.get("x") or 0) < time.time():
            raise ValueError("expired")
        return Session(session_id, owner, str(p["m"]), str(p.get("l") or "Chat"), p.get("s"), p.get("t"),
                       p.get("c"), str(p.get("n") or ""))
    except config.SessionSecretMissing:
        raise                                  # misconfiguration, not a bad token -> 503 via the app handler
    except Exception as exc:  # noqa: BLE001 — every failure is just "unknown session"
        raise ChatSessionError("Unknown or expired chat session.") from exc


# --------------------------------------------------------------------------------------
# Per-session state: DB first, bounded in-process fallback (persistence off / dev)
# --------------------------------------------------------------------------------------
_LOCAL: "OrderedDict[str, dict]" = OrderedDict()
_local_lock = threading.Lock()


def _local_state(session: Session) -> dict:
    key = f"{session.owner}\x00{session.nonce}"
    with _local_lock:
        st = _LOCAL.get(key)
        if st is None:
            st = {"history": [], "recent_artifact": None, "index": None}
            _LOCAL[key] = st
        _LOCAL.move_to_end(key)
        while len(_LOCAL) > _LOCAL_MAX:
            _LOCAL.popitem(last=False)
        return st


def _pair_messages(stored: list[dict]) -> list[tuple[str, str]]:
    """Rebuild (question, answer) pairs from alternating stored/client messages."""
    pairs: list[tuple[str, str]] = []
    pending: str | None = None
    for msg in stored or []:
        role, content = msg.get("role"), str(msg.get("content") or "")
        if role == "user":
            pending = content
        elif role == "assistant" and pending is not None:
            pairs.append((pending, content))
            pending = None
    return pairs


def _clean_client_history(raw: list | None) -> list[tuple[str, str]]:
    """Validate the optional client-sent history (``[{role, content}]``): bounded, strings only."""
    if not isinstance(raw, list):
        return []
    msgs = [{"role": m.get("role"), "content": str(m.get("content") or "")[:_CLIENT_MESSAGE_CHARS]}
            for m in raw[-2 * _HISTORY_TURNS:] if isinstance(m, dict)]
    return _pair_messages(msgs)


def _chats_collection():
    database = db._get_db()
    return None if database is None else database["chats"]


def _db_load(session: Session) -> dict | None:
    """Last messages + remembered artifact of the owner's chat (one projected read)."""
    coll = _chats_collection()
    oid = db._oid(session.chat_id) if session.chat_id else None
    if coll is None or oid is None:
        return None
    doc = coll.find_one({"_id": oid, "owner": session.owner},
                        {"messages": {"$slice": -(2 * _HISTORY_TURNS + 2)}, "recent_artifact": 1})
    if not doc:
        return {"history": [], "recent_artifact": None}
    return {"history": _pair_messages(doc.get("messages") or []), "recent_artifact": doc.get("recent_artifact")}


# Write-through memory of recent chat state, per instance, keyed by (owner, chat_id): once this instance
# has loaded or written a chat, the next turn on it needs NO Mongo read. The cache can only be stale when
# consecutive turns of one chat hit different serverless instances within the TTL (the follow-up then
# sees slightly older context; the stored transcript is never affected because persistence is a $push).
_STATE_TTL_S: float = 120.0
_STATE_MAX: int = 500
_STATE: "OrderedDict[tuple[str, str], tuple[float, dict]]" = OrderedDict()


def _state_get(session: Session) -> dict | None:
    key = (session.owner, session.chat_id or "")
    with _local_lock:
        hit = _STATE.get(key)
        if hit is not None and hit[0] >= time.monotonic():
            _STATE.move_to_end(key)
            return {"history": list(hit[1]["history"]), "recent_artifact": hit[1]["recent_artifact"]}
        _STATE.pop(key, None)
    # Shared cache (optional): a follow-up turn that lands on another instance still needs no Mongo read.
    if cache.enabled():
        shared = cache.get_json(cache.key("chat", session.owner, session.chat_id or ""))
        if isinstance(shared, dict) and isinstance(shared.get("history"), list):
            state = {"history": [tuple(t) for t in shared["history"] if isinstance(t, list) and len(t) == 2],
                     "recent_artifact": shared.get("recent_artifact")}
            _state_put(session, state, remote=False)
            return state
    return None


def _state_put(session: Session, state: dict, remote: bool = True) -> None:
    key = (session.owner, session.chat_id or "")
    hist = list(state["history"])[-_HISTORY_TURNS:]
    with _local_lock:
        _STATE[key] = (time.monotonic() + _STATE_TTL_S, {"history": hist, "recent_artifact": state["recent_artifact"]})
        _STATE.move_to_end(key)
        while len(_STATE) > _STATE_MAX:
            _STATE.popitem(last=False)
    if remote:
        cache.set_json_async(cache.key("chat", session.owner, session.chat_id or ""),
                             {"history": [list(t) for t in hist], "recent_artifact": state["recent_artifact"]},
                             _STATE_TTL_S)


def prime_state(owner: str, chat_id: str | None) -> None:
    """A chat that was just created on this instance is known to be empty: skip its first read."""
    if chat_id:
        _state_put(Session(id="", owner=owner, mode="", label="", subject=None, topic=None, nonce="", chat_id=chat_id),
                   {"history": [], "recent_artifact": None}, remote=False)


def load_state(session: Session, client_history: list | None = None) -> dict:
    """{"history": [(q, a)…], "recent_artifact": dict|None} for this session, from any instance."""
    state = None
    if session.chat_id:
        state = _state_get(session)
        if state is None:
            try:
                state = _db_load(session)
                if state is not None:
                    _state_put(session, state)
            except Exception:  # noqa: BLE001 — memory is best-effort; never fail the turn over it
                log.warning("Could not load chat history.", exc_info=True)
    if state is None:
        local = _local_state(session)
        state = {"history": list(local["history"]), "recent_artifact": local["recent_artifact"]}
    provided = _clean_client_history(client_history)
    if provided and len(provided) >= len(state["history"]):
        state["history"] = provided
    return state


def _new_id() -> str:
    return uuid4().hex[:12]


def _valid_id(value: str | None) -> str | None:
    return value if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_-]{6,40}", value) else None


def _persist_exchange(session: Session, message: str, answer: str, llm: dict | None,
                      recent_artifact: dict | None, first_turn: bool, *,
                      turn: "_Turn | None" = None, ids: dict | None = None, extra: dict | None = None,
                      title: str | None = None) -> None:
    """Mirror one finished exchange (user + assistant message, with the model label) to the chat.

    ``turn`` (edit / regenerate / continue) makes this a REWRITE of the stored chat: the messages that stay
    (``turn.kept``) plus the new exchange are written in one ``$set``. ``extra`` adds assistant-message fields
    (stopped, finish, usage, citations, versions)."""
    ids = ids or {}
    local = _local_state(session)
    with _local_lock:
        local["history"].append((message, answer))
        del local["history"][:-_LOCAL_TURNS]
        if recent_artifact:
            local["recent_artifact"] = recent_artifact
    coll = _chats_collection()
    oid = db._oid(session.chat_id) if session.chat_id else None
    if coll is None or oid is None:
        return
    now = db._now()
    user_msg = {"role": "user", "content": message, "ts": (turn.user_ts if turn and turn.user_ts else now.isoformat()),
                "id": ids.get("user") or _new_id()}
    assistant = {"role": "assistant", "content": answer, "ts": now.isoformat(), "id": ids.get("assistant") or _new_id()}
    if llm:
        assistant["llm"] = llm
    for k, v in (extra or {}).items():
        if v not in (None, "", [], {}, False):
            assistant[k] = v
    if turn is not None and turn.kept is not None:
        update: dict = {"$set": {"messages": [*turn.kept, user_msg, assistant][-_MAX_STORED_MESSAGES:],
                                 "updated_at": now}}
    else:
        update = {
            "$push": {"messages": {"$each": [user_msg, assistant], "$slice": -_MAX_STORED_MESSAGES}},
            "$set": {"updated_at": now},
        }
    if recent_artifact:
        update["$set"]["recent_artifact"] = recent_artifact
    if first_turn:
        update["$set"]["title"] = title or message[:60]    # the LLM title when we have it, else the first words
    # The chat document is created LAZILY by this write (``chat/start`` costs no Mongo op): upsert, with the
    # session metadata only applied on insert. One write per turn; the first one also carries the title.
    meta = {k: v for k, v in (("mode", session.mode), ("label", session.label), ("subject", session.subject),
                              ("topic", session.topic)) if v}
    update["$setOnInsert"] = {"created_at": now, **meta}
    created = False
    try:
        res = coll.update_one({"_id": oid, "owner": session.owner}, update, upsert=True)
        created = getattr(res, "upserted_id", None) is not None
    finally:
        if created:
            db.note_chat_write(session.owner, session.chat_id, created_doc={
                **meta, "title": update["$set"].get("title") or "", "created_at": now, "updated_at": now})
        else:
            db.note_chat_write(session.owner, session.chat_id, updated_at=now,
                               title=update["$set"].get("title") if first_turn else None)
    prev = _state_get(session) or {"history": [], "recent_artifact": None}
    if turn is not None and turn.kept is not None:
        prev["history"] = [*turn.history, (message, answer)]
    else:
        prev["history"].append((message, answer))
    if recent_artifact:
        prev["recent_artifact"] = recent_artifact
    _state_put(session, prev)


async def _llm_title(message: str, answer: str, deadline_s: float = 4.0) -> str | None:
    """A concise LLM title for a new chat (best-effort, short deadline), or None. Generated BEFORE the first
    turn is persisted so the title rides in that single write instead of costing a second one."""
    prompt = (
        "Create a short, specific title for this conversation in 3 to 6 words. "
        "Reply with ONLY the title — no quotation marks and no trailing punctuation.\n\n"
        f"User: {message[:500]}\nAssistant: {answer[:500]}"
    )
    try:
        res = await llm_chain.achat("You write short conversation titles.", [{"role": "user", "content": prompt}],
                                    max_tokens=2048, temperature=0.3, deadline_s=deadline_s,
                                    sensitive=privacy.current())
        title = " ".join(res.text.strip().strip('"').strip("'").strip().rstrip(".").split())[:80]
        return title or None
    except Exception:  # noqa: BLE001 — the provisional title (first words) is the fallback
        log.debug("Auto-naming failed.", exc_info=True)
        return None


async def _autoname(session: Session, message: str, answer: str) -> None:
    """Give a new chat a concise LLM title (best-effort; runs after the answer was delivered)."""
    if not session.chat_id:
        return
    prompt = (
        "Create a short, specific title for this conversation in 3 to 6 words. "
        "Reply with ONLY the title — no quotation marks and no trailing punctuation.\n\n"
        f"User: {message[:500]}\nAssistant: {answer[:500]}"
    )
    try:
        res = await llm_chain.achat("You write short conversation titles.", [{"role": "user", "content": prompt}],
                                    max_tokens=2048, temperature=0.3, deadline_s=12.0,
                                    sensitive=privacy.current())
        title = " ".join(res.text.strip().strip('"').strip("'").strip().rstrip(".").split())[:80]
        if not title:
            return
        coll = _chats_collection()
        oid = db._oid(session.chat_id)
        if coll is not None and oid is not None:
            # Only replace the PROVISIONAL title: if the user renamed the chat meanwhile, keep theirs.
            try:
                await asyncio.to_thread(coll.update_one,
                                        {"_id": oid, "owner": session.owner, "title": message[:60]},
                                        {"$set": {"title": title}})
            finally:
                db._bump(session.owner)
    except Exception:  # noqa: BLE001 — keep the provisional title if naming fails
        log.debug("Auto-naming failed.", exc_info=True)


# --------------------------------------------------------------------------------------
# Personas / history rendering
# --------------------------------------------------------------------------------------
_PERSONA_CACHE: "OrderedDict[tuple, tuple[float, str]]" = OrderedDict()
_persona_lock = threading.Lock()


def _subject_persona(subject: str, owner: str | None = None) -> str:
    """System persona for a subject session, including the PDF list (60 s bounded cache)."""
    key = (owner or "", subject)
    now = time.monotonic()
    with _persona_lock:
        hit = _PERSONA_CACHE.get(key)
        if hit and now - hit[0] < 60:
            return hit[1]
    names = list_pdf_names(subject, owner)
    if names:
        pdf_list = "\n".join(f"  - {n}" for n in names[:60])
        persona = f"{_SUBJECT_PERSONA_BASE}\n\nThis subject contains {len(names)} PDF(s):\n{pdf_list}"
    else:
        persona = _SUBJECT_PERSONA_BASE
    with _persona_lock:
        _PERSONA_CACHE[key] = (now, persona)
        while len(_PERSONA_CACHE) > 256:
            _PERSONA_CACHE.popitem(last=False)
    return persona


def _topic_persona(topic: str) -> str:
    return (
        f"You are a skilled, thoughtful teaching assistant specializing in {topic}. "
        f"Help the student learn {topic} through clear explanations and examples. "
        f"Keep responses clear, concise, and focused; break complex concepts into "
        f"simpler parts; provide practical examples when relevant; and encourage "
        f"hands-on learning. If a question falls outside {topic}, kindly redirect."
    )


def _recent_artifact_block(recent: dict | None) -> str:
    """Labelled context for an artifact the student just created, or "".

    Lets a normal follow-up turn discuss what was just built ("explain question 3").
    """
    if not isinstance(recent, dict) or not recent.get("text"):
        return ""
    return f"The student just created {recent.get('title') or 'a study artifact'}. Here it is:\n{recent['text']}"


def _history_messages(history: list[tuple[str, str]]) -> list[dict]:
    """The whole stored conversation as chat messages (bounded only by the storage cap).

    Not trimmed here: llm_chain shrinks the largest blocks for Groq's 8k-TPM tier and leaves
    Anthropic/Gemini untouched."""
    out: list[dict] = []
    for question, answer in history[-_HISTORY_TURNS:]:
        out.append({"role": "user", "content": question})
        out.append({"role": "assistant", "content": answer})
    return out


# --------------------------------------------------------------------------------------
# Session creation / resume
# --------------------------------------------------------------------------------------
def start_subject_session(subject: str, model: str | None = None, owner: str = LOCAL_OWNER) -> tuple[str, str, str | None]:
    """Start a session backed by a subject's index (warmed now; unknown/forbidden subjects raise).

    Every chat session is smart ("autonomous"): it answers questions AND orchestrates tools when
    asked to build an artifact (the regex gate inside chat_tools keeps ordinary questions free).
    ``model`` is accepted for API compatibility and ignored — the provider chain picks the model.
    """
    from backend import vector_store

    if subject != agentic_rag.ALL_SUBJECTS:
        vector_store.get_subject_retriever(subject, _RETRIEVE_K, owner)
    label = "All subjects" if subject == agentic_rag.ALL_SUBJECTS else f"Subject · {subject}"
    chat_id = db.new_chat_id()   # document created by the first turn (lazy upsert)
    prime_state(owner, chat_id)
    s = _mint(owner, "subject", label, subject, None, chat_id)
    return s.id, label, chat_id


def start_topic_session(topic: str, model: str | None = None, owner: str = LOCAL_OWNER) -> tuple[str, str, str | None]:
    """Start a document-free session that teaches a specific topic."""
    label = f"Topic · {topic}"
    chat_id = db.new_chat_id()   # document created by the first turn (lazy upsert)
    prime_state(owner, chat_id)
    s = _mint(owner, "topic", label, None, topic, chat_id)
    return s.id, label, chat_id


def _pdf_chunks(file_bytes: bytes, filename: str) -> list[dict]:
    from backend import vector_store
    from backend.documents import extract_pdf_pages

    try:
        pages = extract_pdf_pages(file_bytes)
    except Exception as exc:  # noqa: BLE001 — corrupt/encrypted PDF
        raise ChatSessionError("That file could not be read as a PDF.") from exc
    chunks = vector_store.chunk_pages([(filename, pages)])
    kept: list[dict] = []
    total = 0
    for c in chunks:
        total += len(c["text"])
        if total > _MAX_PDF_CHARS:
            break
        kept.append(c)
    if not kept:
        raise ChatSessionError("No readable text was found in the selected PDF(s).")
    return kept


def _pdf_index(chunks: list[dict], key: str):
    from backend import vector_store

    return vector_store.VectorIndex(f"oneoff-{key}", chunks)       # lexical (BM25) — no embeddings needed


def start_pdf_session(file_bytes: bytes, filename: str, model: str | None = None, owner: str = LOCAL_OWNER) -> tuple[str, str, str | None]:
    """Start a session backed by a single uploaded PDF (BM25 retrieval; chunks kept in MongoDB)."""
    if len(file_bytes) > _MAX_PDF_BYTES:
        raise ChatSessionError("That PDF is too large (limit 20 MB).")
    filename = (filename or "uploaded.pdf")[:120]
    chunks = _pdf_chunks(file_bytes, filename)
    label = f"PDF · {filename}"
    chat_id = db.new_chat_id()
    s = _mint(owner, "pdf", label, None, None, chat_id)
    coll_db = db._get_db()
    if chat_id and coll_db is not None:
        coll_db["chat_pdfs"].replace_one({"_id": chat_id},
                                         {"_id": chat_id, "owner": owner, "chunks": chunks, "created_at": db._now()},
                                         upsert=True)
    else:
        local = _local_state(s)
        local["index"] = _pdf_index(chunks, s.nonce)
    return s.id, label, chat_id


def _pdf_retriever(session: Session):
    local = _local_state(session)
    if local.get("index") is not None:
        return local["index"].as_retriever({"k": _RETRIEVE_K})
    database = db._get_db()
    if database is not None and session.chat_id:
        doc = database["chat_pdfs"].find_one({"_id": session.chat_id, "owner": session.owner})
        if doc and doc.get("chunks"):
            idx = _pdf_index(doc["chunks"], session.nonce)
            with _local_lock:
                local["index"] = idx          # reuse within this instance (bounded by _LOCAL_MAX)
            return idx.as_retriever({"k": _RETRIEVE_K})
    raise ChatSessionError("This PDF chat has expired — upload the PDF again.")


def resume_session(owner: str, chat_id: str) -> tuple[str, str, str, str, list[dict]]:
    """Re-open a saved chat as a live session and return its prior messages.

    Returns (session_id, label, mode, chat_id, messages). Owner-scoped: someone else's chat id
    is simply "not found".
    """
    doc = db.get_item("chats", owner, chat_id)
    if not doc:
        raise ChatSessionError("That chat could not be found.")
    mode = doc.get("mode", "topic")
    label = doc.get("label") or "Chat"
    all_msgs = doc.get("messages") or []
    start = max(0, len(all_msgs) - 200)
    messages = []
    for i, m in enumerate(all_msgs[start:], start):
        row = {"role": m.get("role"), "content": m.get("content", ""), "llm": m.get("llm"),
               "id": m.get("id") or f"i{i}", "ts": m.get("ts")}
        for k in ("stopped", "finish", "versions", "feedback", "citations", "usage"):
            if m.get(k):
                row[k] = m[k]
        messages.append(row)
    s = _mint(owner, mode, label, doc.get("subject"), doc.get("topic"), chat_id)
    return s.id, label, mode, chat_id, messages


# --------------------------------------------------------------------------------------
# Retrieval (bounded)
# --------------------------------------------------------------------------------------
def _prepare(session: Session) -> tuple[object | None, str]:
    """(retriever, persona) for the session — blocking, run in a worker thread."""
    if session.mode == "subject" and session.subject == agentic_rag.ALL_SUBJECTS:
        # Cross-subject search: one retriever per subject THIS owner can see, fused by RRF in the pipeline.
        return (agentic_rag.all_subjects_searchers(session.owner, _RETRIEVE_K),
                _SUBJECT_PERSONA_BASE + "\n\nYou are searching across ALL of the student's subjects; "
                "passage headers show the file each one comes from.")
    if session.mode == "subject" and session.subject:
        from backend import vector_store

        retriever = vector_store.get_subject_retriever(session.subject, _RETRIEVE_K, session.owner)
        return retriever, _subject_persona(session.subject, session.owner)
    if session.mode == "pdf":
        return _pdf_retriever(session), _SUBJECT_PERSONA_BASE
    persona = _topic_persona(session.topic) if session.topic else _GENERIC_PERSONA
    return None, persona


def _safe_invoke(retriever, query: str) -> list:
    try:
        return list(retriever.invoke(query))
    except Exception:  # noqa: BLE001 — one bad query must not abort the turn
        log.debug("Retrieval failed for %r.", query, exc_info=True)
        return []


def _lexical(retriever, queries: list[str]) -> list:
    """BM25-only retrieval (no embedding call) — the fallback when embeddings are slow/down."""
    fn = getattr(retriever, "invoke_lexical", None)
    if fn is None:
        resolve = getattr(retriever, "_resolve", None)
        idx = resolve() if callable(resolve) else getattr(retriever, "_index", None)
        k = getattr(retriever, "k", _RETRIEVE_K)
        fn = (lambda q: idx.search(q, k, use_embeddings=False)) if idx is not None and hasattr(idx, "search") else None
    if fn is None:
        return []
    return agentic_rag.merge_docs([list(fn(q)) for q in queries])


async def retrieve_bounded(retriever, queries: list[str]) -> tuple[list, str]:
    """Retrieve for ``queries`` within the time budget. Returns ``(docs, mode)`` where mode is
    ``"hybrid"`` (normal), ``"lexical"`` (embeddings too slow → BM25 only) or ``"none"``."""

    async def dense() -> list:
        batches = await asyncio.gather(*[asyncio.to_thread(_safe_invoke, retriever, q) for q in queries])
        return agentic_rag.merge_docs(list(batches))

    task = asyncio.ensure_future(dense())
    task.add_done_callback(lambda t: t.cancelled() or t.exception())      # never "exception never retrieved"
    try:
        return await asyncio.wait_for(asyncio.shield(task), _RETRIEVAL_BUDGET_S), "hybrid"
    except asyncio.TimeoutError:
        log.info("Dense retrieval exceeded %.1fs; falling back to lexical.", _RETRIEVAL_BUDGET_S)
    try:
        docs = await asyncio.wait_for(asyncio.to_thread(_lexical, retriever, queries), _LEXICAL_BUDGET_S)
        if docs:
            return docs, "lexical"
    except Exception:  # noqa: BLE001 — timeout or index error → answer without context
        log.info("Lexical fallback unavailable.", exc_info=True)
    return [], "none"


# --------------------------------------------------------------------------------------
# Turn control: edit / regenerate / continue, follow-up chips, per-message feedback, pin
# --------------------------------------------------------------------------------------
_MAX_VERSIONS: int = 5                  # earlier answers kept per message after "Regenerate"
_CITATIONS_MAX: int = 6
_CITATION_CHARS: int = 300

FOLLOWUP_INSTRUCTION: str = (
    "\n\nAfter your complete answer, add ONE final line in exactly this machine-readable form (nothing after it):\n"
    '<followups>["first short follow-up question","second one","third one"]</followups>\n'
    "Give 2-3 short questions (under 70 characters each) the student could ask next. Never mention this line."
)
CONTINUE_INSTRUCTION: str = ("Continue your previous answer exactly where it stopped. Do not repeat anything you "
                             "already wrote and do not add a preamble.")
_FOLLOWUP_OPEN = "<followups"
_FOLLOWUP_RE = re.compile(r"<followups>\s*(\[.*?\])\s*</followups>", re.S | re.I)


class FollowupFilter:
    """Streaming-safe stripper of the trailing ``<followups>[...]</followups>`` line.

    ``feed(text)`` returns the part of ``text`` that is safe to show (it holds back a possible partial
    ``<followups`` opener until the next chunk resolves it); everything from the opener on is swallowed.
    ``finish()`` flushes a held-back tail that turned out not to be the tag. ``followups()`` then returns the
    validated chips (<= 3), or ``[]`` when the line is absent / malformed / unterminated."""

    def __init__(self) -> None:
        self._hold = ""
        self._tail = ""
        self._in_tag = False

    def feed(self, text: str) -> str:
        if self._in_tag:
            self._tail += text
            return ""
        buf = self._hold + text
        self._hold = ""
        idx = buf.lower().find(_FOLLOWUP_OPEN)
        if idx >= 0:
            self._in_tag = True
            self._tail = buf[idx:]
            return buf[:idx]
        keep = 0
        for n in range(min(len(_FOLLOWUP_OPEN) - 1, len(buf)), 0, -1):
            if _FOLLOWUP_OPEN.startswith(buf[-n:].lower()):
                keep = n
                break
        if keep:
            self._hold = buf[-keep:]
            return buf[:-keep]
        return buf

    def finish(self) -> str:
        out, self._hold = ("" if self._in_tag else self._hold), ""
        return out

    def followups(self) -> list[str]:
        return parse_followups(self._tail)


def parse_followups(raw: str) -> list[str]:
    m = _FOLLOWUP_RE.search(raw or "")
    if not m:
        return []
    try:
        data = json.loads(m.group(1))
    except Exception:  # noqa: BLE001
        return []
    out: list[str] = []
    for item in data if isinstance(data, list) else []:
        if isinstance(item, str):
            q = " ".join(item.split()).strip()
            if 3 <= len(q) <= 100 and q.lower() not in {o.lower() for o in out}:
                out.append(q)
    return out[:3]


def strip_followups(text: str) -> tuple[str, list[str]]:
    """Non-streaming twin: ``(visible text, followups)``."""
    f = FollowupFilter()
    visible = f.feed(text or "") + f.finish()
    return visible.strip(), f.followups()


def citations_from(docs: list) -> list[dict]:
    """Compact ``[{source, page, text}]`` of the retrieved passages behind an answer (chips that expand)."""
    try:
        rows = agentic_rag.passage_view(docs, limit=_CITATIONS_MAX, snippet=_CITATION_CHARS)
    except Exception:  # noqa: BLE001
        return []
    return [r for r in rows if r.get("source") and r["source"] != "—"]


@dataclass
class _Turn:
    """How a non-"send" turn rewrites the stored chat (``kept`` = messages that stay; None = client-history mode)."""
    action: str
    message: str
    history: list[tuple[str, str]]
    kept: list[dict] | None = None
    prefix: str = ""
    versions: list[dict] = field(default_factory=list)
    user_ts: str | None = None
    user_id: str | None = None


def _resolve_turn(session: Session, message: str, action: str, target_id: str | None, prefix: str | None,
                  client_history: list | None) -> "_Turn | None":
    """Plan an edit / regenerate / continue turn (blocking: one Mongo read). ``None`` for a normal send."""
    if action not in ("edit", "regenerate", "continue"):
        return None
    coll = _chats_collection()
    oid = db._oid(session.chat_id) if session.chat_id else None
    if coll is None or oid is None:
        # Stateless / persistence-off mode: the client already truncated its own history.
        history = _clean_client_history(client_history)
        local = _local_state(session)
        with _local_lock:
            local["history"][:] = history[-_LOCAL_TURNS:]       # drop what the edit/regenerate removed
        if action == "continue":
            if not (prefix or "").strip():
                raise ChatTurnError("There is no answer to continue.", status=409)
            return _Turn(action, message, history, None, prefix=prefix or "")
        return _Turn(action, message, history, None)
    doc = coll.find_one({"_id": oid, "owner": session.owner}, {"messages": 1})
    if not doc:
        raise ChatTurnError("That chat could not be found.", status=404)
    msgs = [dict(m) for m in (doc.get("messages") or []) if isinstance(m, dict)]
    for i, m in enumerate(msgs):
        m.setdefault("id", f"i{i}")
    if action == "edit":
        idx = next((i for i, m in enumerate(msgs) if m["id"] == target_id and m.get("role") == "user"), None)
        if idx is None:
            raise ChatTurnError("That message is no longer in this chat — reload the chat and try again.", status=409)
        kept = msgs[:idx]
        return _Turn(action, message, _pair_messages(kept)[-_HISTORY_TURNS:], kept)
    # regenerate / continue act on the LATEST answer only (ids are stable, so a stale tab gets a clean 409)
    if (len(msgs) < 2 or msgs[-1].get("role") != "assistant" or msgs[-2].get("role") != "user"
            or (target_id and msgs[-1]["id"] != target_id)):
        raise ChatTurnError(f"Only the latest answer can be {'regenerated' if action == 'regenerate' else 'continued'}.",
                            status=409)
    old, user = msgs[-1], msgs[-2]
    kept = msgs[:-2]
    versions = list(old.get("versions") or [])
    if action == "regenerate":
        versions = (versions + [{k: old[k] for k in ("content", "llm", "ts") if old.get(k)}])[-_MAX_VERSIONS:]
    return _Turn(action, str(user.get("content") or ""), _pair_messages(kept)[-_HISTORY_TURNS:], kept,
                 prefix=str(old.get("content") or "") if action == "continue" else "",
                 versions=versions, user_ts=user.get("ts"), user_id=user["id"])


def set_feedback(owner: str, chat_id: str, message_id: str, value: str | None) -> bool:
    """Rate one assistant message (``up`` / ``down`` / None to clear). Owner-scoped; False if not found."""
    coll = _chats_collection()
    oid = db._oid(chat_id)
    if coll is None or oid is None:
        return False
    m = re.fullmatch(r"i(\d{1,5})", message_id or "")
    if m:                                           # legacy message without a stored id: address by position
        path, flt = f"messages.{int(m.group(1))}.feedback", {"_id": oid, "owner": owner}
    else:
        path = "messages.$.feedback"
        flt = {"_id": oid, "owner": owner, "messages": {"$elemMatch": {"id": message_id, "role": "assistant"}}}
    update = {"$set": {path: value}} if value else {"$unset": {path: ""}}
    return coll.update_one(flt, update).matched_count > 0


def patch_chat(owner: str, chat_id: str, *, title: str | None = None, pinned: bool | None = None) -> bool:
    """Rename and/or pin a chat. Owner-scoped; a rename is final (auto-naming never overwrites it)."""
    coll = _chats_collection()
    oid = db._oid(chat_id)
    if coll is None or oid is None:
        return False
    sets: dict = {}
    if title is not None:
        sets["title"] = title
    if pinned is not None:
        sets["pinned"] = bool(pinned)
    if not sets:
        return True
    try:
        return coll.update_one({"_id": oid, "owner": owner}, {"$set": sets}).matched_count > 0
    finally:
        db._bump(owner)


# --------------------------------------------------------------------------------------
# The turn
# --------------------------------------------------------------------------------------
def _err_event(detail: str, llm: dict | None, status: int) -> dict:
    return {"type": "error", "detail": detail, "llm": llm, "status": status}


def _step_event(step: dict) -> dict:
    return {"type": "step", "step": step}


async def _generate(system: str, messages: list[dict]) -> AsyncIterator[dict]:
    """Stream one answer: a ``model`` event as soon as a provider accepts the stream (again if the chain
    falls through before the first token), ``token`` events, then a private ``_final`` event."""
    async for item in llm_chain.astream_chat(system, messages, max_tokens=_ANSWER_MAX_TOKENS, temperature=0.3,
                                             sensitive=privacy.current()):
        if isinstance(item, str):
            if item:
                yield {"type": "token", "text": item}
        elif isinstance(item, llm_chain.StreamStart):
            yield {"type": "model", "llm": item.to_public()}
        else:
            yield {"type": "_final", "result": item}


async def stream_turn(session: Session, message: str, tool: str | None = None,
                      client_history: list | None = None, *, action: str = "send",
                      target_id: str | None = None, prefix: str | None = None,
                      user_id: str | None = None, assistant_id: str | None = None) -> AsyncIterator[dict]:
    """Run one chat turn, yielding ``step`` / ``token`` / ``done`` / ``error`` event dicts.

    ``action`` = send | edit | regenerate | continue (see ``_resolve_turn``). If the consumer goes away
    (Stop / closed tab) the turn is cancelled: the upstream provider stream is closed at once and the
    partial answer is saved, marked ``stopped``."""
    from backend.services import chat_tools

    message = (message or "").strip()[:_MAX_MESSAGE_CHARS]
    privacy.mark(privacy.for_session(session))     # PDF chats / user-uploaded subjects never reach Gemini
    streamed = ""                                  # visible answer text streamed so far (for a Stop)
    live_llm: dict | None = None
    turn: _Turn | None = None
    ids = {"user": _valid_id(user_id) or _new_id(), "assistant": _valid_id(assistant_id) or _new_id()}
    first_turn = False
    answered = False
    try:
        prep_task = asyncio.ensure_future(asyncio.to_thread(_prepare, session))
        prep_task.add_done_callback(lambda t: t.cancelled() or t.exception())
        turn = await asyncio.to_thread(_resolve_turn, session, message, action, target_id, prefix, client_history)
        if turn is not None:
            message = turn.message[:_MAX_MESSAGE_CHARS] if action != "edit" else message
            ids["user"] = turn.user_id or ids["user"]
            state = {"history": turn.history, "recent_artifact": (_state_get(session) or {}).get("recent_artifact")}
        else:
            state = await asyncio.to_thread(load_state, session, client_history)
        history: list[tuple[str, str]] = state["history"]
        first_turn = not history and turn is None
        flt = FollowupFilter()
        citations: list[dict] = []
        finish = ""
        usage: dict | None = None
        steps: list[dict] = []
        sources: list[str] = []
        recent_artifact: dict | None = state.get("recent_artifact")
        new_artifact: dict | None = None
        artifacts: list[dict] = []
        result: llm_chain.ChatResult | None = None
        llm_public: dict | None = None
        answer = ""

        # --- Autonomous tool gate ------------------------------------------------------
        if action == "continue":
            intents = []
        else:
            intents = chat_tools.forced_intents(tool, message) if tool else await chat_tools.adetect_intent(message)
        if intents:
            ctx = {"owner": session.owner, "subject": session.subject, "topic": session.topic}
            answers: list[str] = []
            for one in intents:
                tr = await asyncio.to_thread(chat_tools.run_tool, ctx, one, message)
                if tr is None:
                    continue    # this tool declined (e.g. PDF session with no topic) → skip, no phantom row
                starting = {"kind": "tool", "tool": one.get("tool"), "title": f"Calling {one.get('tool')}…"}
                steps.append(starting)
                yield _step_event(starting)
                answers.append(tr["answer"])
                if tr.get("artifact"):
                    artifacts.append(tr["artifact"])
                if tr.get("recent_artifact"):
                    new_artifact = tr["recent_artifact"]
                if tr.get("llm"):
                    llm_public = tr["llm"]
                for st in tr["steps"]:
                    row = {**st} if isinstance(st, dict) else {"kind": "tool", "title": str(st)}
                    row.setdefault("kind", "tool")
                    steps.append(row)
                    yield _step_event(row)
            if answers:
                answer = "\n\n".join(answers)
        evidence: agentic_rag.Evidence | None = None            # set by the RAG path below
        if not answer:
            # --- Normal answer: smalltalk / RAG / topic ---------------------------------
            retriever, persona = await prep_task
            evidence: agentic_rag.Evidence | None = None       # set by the best-practice RAG path below
            artifact_block = _recent_artifact_block(recent_artifact)
            hist = _history_messages(history)
            if _is_smalltalk(message):
                step = {"kind": "generate", "title": "Replied directly — no document lookup needed"}
                steps.append(step)
                yield _step_event(step)
                system, msgs = _SMALLTALK_PERSONA, [*hist, {"role": "user", "content": message}]
            elif retriever is not None:
                full_persona = f"{persona}\n\n{artifact_block}" if artifact_block else persona
                queries = agentic_rag.heuristic_queries(message, hist) or [message]
                if config.AGENTIC_RAG_MAX_ITERS > 1:
                    # Opt-in slow path: LLM rewrite / grade / correct loop (blocking → worker thread).
                    got = await asyncio.to_thread(agentic_rag.gather, retriever, message, hist,
                                                  persona=full_persona)
                    for st in got.steps:
                        steps.append(st)
                        yield _step_event(st)
                    docs, context, sources = got.docs, got.context, got.sources
                else:
                    # Default path: plan -> hybrid retrieve -> rerank/MMR/neighbours -> confidence (no LLM call
                    # unless retrieval is weak). Steps stream as they happen. See backend/agentic_rag.py.
                    async for e in agentic_rag.retrieve_events(retriever, message, hist, budget_s=_RETRIEVAL_BUDGET_S):
                        if e["type"] == "step":
                            steps.append(e["step"])
                            yield e
                        else:
                            evidence = e["evidence"]
                    docs, context, sources = evidence.docs, evidence.context(), agentic_rag.source_names(evidence.docs)
                if evidence is not None:
                    system = full_persona + evidence.system_suffix()
                    msgs = [*hist, {"role": "user", "content": evidence.user_message(message)}]
                    citations = evidence.citations()
                else:
                    system = full_persona
                    msgs = [*hist, {"role": "user", "content": agentic_rag.generate_user(message, context)}]
                    citations = citations_from(docs)
            else:
                full_persona = f"{persona}\n\n{artifact_block}" if artifact_block else persona
                step = {"kind": "generate", "title": "Answered from topic knowledge (no document retrieval)"}
                steps.append(step)
                yield _step_event(step)
                system, msgs = full_persona, [*hist, {"role": "user", "content": message}]

            want_followups = system is not _SMALLTALK_PERSONA
            if want_followups:
                system = system + FOLLOWUP_INSTRUCTION
            prefix_text = turn.prefix if (turn is not None and action == "continue") else ""
            if prefix_text:
                msgs = [*msgs, {"role": "assistant", "content": prefix_text},
                        {"role": "user", "content": CONTINUE_INSTRUCTION}]
            gen = _generate(system, msgs)
            if evidence is not None and evidence.hops_left > 0:
                def _rebuild(e, _persona=full_persona):
                    m = [*hist, {"role": "user", "content": e.user_message(message)}]
                    if prefix_text:
                        m += [{"role": "assistant", "content": prefix_text}, {"role": "user", "content": CONTINUE_INSTRUCTION}]
                    return _persona + e.system_suffix() + (FOLLOWUP_INSTRUCTION if want_followups else ""), m

                gen = agentic_rag.generate_with_hops(evidence, _generate, _rebuild)
            async for ev in gen:
                kind = ev["type"]
                if kind == "_final":
                    result = ev["result"]
                elif kind == "token":
                    vis = flt.feed(ev["text"]) if want_followups else ev["text"]
                    if vis:
                        streamed += vis
                        yield {"type": "token", "text": vis}
                else:
                    if kind == "model":
                        live_llm = ev["llm"]
                    yield ev
            if want_followups:
                tail_vis = flt.finish()
                if tail_vis:
                    streamed += tail_vis
                    yield {"type": "token", "text": tail_vis}
            if result is None or not (streamed.strip() or result.text.strip()):
                raise LLMChainError("empty stream")
            answer = (prefix_text + (streamed if want_followups else result.text)).strip()
            if evidence is not None:           # drop citations that do not exist; sources/chips match the answer
                ver = evidence.finalize(answer)
                answer, sources, citations = ver.answer, ver.sources, ver.citations
            llm_public = result.to_public()
            finish, usage = result.finish, result.usage

        # --- Persist AFTER the answer (shielded: survives the client closing the stream) --
        followups = flt.followups() if answer and not artifacts else []
        extra = {"finish": finish if finish in ("length", "interrupted") else None,
                 "usage": usage if usage and usage.get("total_tokens") else None,
                 "citations": citations, "versions": turn.versions if turn else None}

        async def _tail() -> None:
            try:
                title = await _llm_title(message, answer) if (first_turn and session.chat_id) else None
                await asyncio.to_thread(_persist_exchange, session, message, answer, llm_public,
                                        new_artifact, first_turn, turn=turn, ids=ids, extra=extra, title=title)
            except Exception:  # noqa: BLE001 — persistence must never break an answered turn
                log.warning("Could not persist chat exchange.", exc_info=True)

        answered = True
        tail = asyncio.ensure_future(_tail())
        done_event = {
            "type": "done", "answer": answer, "sources": sources,
            "artifact": artifacts[0] if artifacts else None,      # legacy clients
            "artifacts": artifacts, "llm": llm_public,
            "followups": followups, "citations": citations, "finish": finish or None, "usage": usage,
            "related": list(evidence.related) if evidence is not None else [],     # concept-graph suggestions
            "message_ids": {"user": ids["user"], "assistant": ids["assistant"]},
            "action": action,
        }
        renewed = refresh_session(session)           # sliding expiry: only present when < 1 h was left
        if renewed:
            done_event["session_id"] = renewed
        yield done_event
        await asyncio.shield(tail)
    except LLMChainError as exc:
        log.warning("Chat turn: all providers failed: %s", str(exc)[:200])
        _detail, _label = llm_unavailable(exc, UNAVAILABLE_DETAIL, NO_LLM_LABEL)
        yield _err_event(_detail, {"label": _label}, llm_http_status(exc))
    except ChatSessionError as exc:
        yield _err_event(str(exc) if "PDF" in str(exc) else "Session expired, start a new chat.", None, 404)
    except ChatTurnError as exc:
        yield _err_event(exc.detail, exc.llm, exc.status)
    except asyncio.CancelledError:
        # Stop / disconnect: the upstream provider stream was closed by the cancellation itself. Keep what
        # the student already saw — persisted as a ``stopped`` answer (also survives a reload).
        if not answered:
            partial = ((turn.prefix if turn else "") + streamed).lstrip()      # keep trailing space: "Continue" appends
            if streamed.strip() and partial:
                persist = asyncio.ensure_future(asyncio.to_thread(
                    _persist_exchange, session, message, partial, live_llm, None, first_turn,
                    turn=turn, ids=ids, extra={"stopped": True, "versions": turn.versions if turn else None}))
                try:
                    await asyncio.shield(persist)
                except BaseException:  # noqa: BLE001 — a second cancel must not mask the first
                    log.debug("Stopped-answer persist interrupted.", exc_info=True)
        raise
    except Exception as exc:  # noqa: BLE001 — surface a generic message, log the detail
        log.exception("Chat turn failed")
        name = type(exc).__name__
        if any(c.__module__.split(".")[0] in ("pymongo", "bson") for c in type(exc).__mro__):
            yield _err_event("The database is temporarily unavailable. Please retry in a moment.", None, 503)
        elif name in ("DocumentError", "VectorStoreError"):
            yield _err_event(f"Couldn't open this subject's notes: {exc}", None, 404)
        else:
            yield _err_event("Something went wrong while answering. Please try again.", None, 502)


async def ask(session: Session, message: str, tool: str | None = None, client_history: list | None = None,
              **turn_args) -> dict:
    """Non-streaming twin of :func:`stream_turn` (the ``/message`` fallback). Returns the ``done``
    payload plus the collected ``steps``; raises ``ChatTurnError`` if the turn failed."""
    steps: list[dict] = []
    async for ev in stream_turn(session, message, tool, client_history, **turn_args):
        t = ev["type"]
        if t == "step":
            steps.append(ev["step"])
        elif t == "error":
            raise ChatTurnError(ev["detail"], status=ev.get("status", 502), llm=ev.get("llm"))
        elif t == "done":
            return {**{k: v for k, v in ev.items() if k != "type"}, "steps": steps}
    raise ChatTurnError("The assistant produced no answer.", status=502)


def session_overview(session_id: str, owner: str, *, refresh: bool = False) -> dict:
    """NotebookLM-style overview card for a session (owner-checked; see ``overview.build_overview``)."""
    session = open_session(session_id, owner)
    from backend.services import overview

    return overview.build_overview(subject=session.subject, topic=session.topic, owner=session.owner,
                                   refresh=refresh)
