"""Chat (RAG) service: subject / topic / uploaded-PDF study assistant.

Ported from the old Streamlit `chat/` app, but rebuilt to run entirely on Groq:
  * LLM via langchain_groq.ChatGroq (was ChatOpenAI).
  * Embeddings run locally and keyless (was OpenAIEmbeddings) via `backend.vector_store`.
  * Subject retrieval uses the persistent per-subject FAISS index in
    `backend.vector_store`, which is built once and reloaded from disk on later
    runs. A one-off uploaded PDF is embedded into an in-memory store (not saved).
  * Conversation history is managed manually (no deprecated LangChain chains/memory).
  * When persistence is enabled, each session is mirrored to MongoDB (scoped to the
    owner) so chats can be listed, searched, resumed, and deleted later.

Every LLM object is created lazily inside a function so that importing this
module never requires a Groq key and stays fast.
"""
import tempfile
import threading
from pathlib import Path as _Path
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from backend import agentic_rag, config, db, vector_store
from backend.auth import LOCAL_OWNER
from backend.config import model_for, require_groq_key
from backend.documents import list_pdf_names

# In-memory session store. Each value holds: mode, retriever (or None), llm,
# history (list of (question, answer) tuples), label, topic (topic mode only),
# subject (subject mode, for resume), owner, and chat_id (persisted history id).
_sessions: dict[str, dict] = {}
_sessions_lock = threading.Lock()

_RETRIEVE_K: int = 5
_HISTORY_TURNS: int = 6
# Cap the number of live sessions so a long-running server can't grow without
# bound (each "Start session" click stores an LLM + retriever + history). The
# oldest sessions are evicted first; an evicted session simply needs restarting.
_MAX_SESSIONS: int = 200

_SUBJECT_PERSONA_BASE: str = (
    "You are a skilled, thoughtful teaching assistant. Help the student understand "
    "the course material by guiding them through ideas. Answer using ONLY the provided "
    "context; if the answer is not supported by the context, clearly say you don't know. "
    "Keep responses clear, concise, and focused. Break complex concepts into simpler "
    "parts, use analogies or examples when helpful, and encourage critical thinking."
)


def _subject_persona(subject: str, owner: str | None = None) -> str:
    """Build the system persona for a subject session, including the PDF list."""
    names = list_pdf_names(subject, owner)
    if not names:
        return _SUBJECT_PERSONA_BASE
    pdf_list = "\n".join(f"  - {n}" for n in names)
    return (
        f"{_SUBJECT_PERSONA_BASE}\n\n"
        f"This subject contains {len(names)} PDF(s):\n{pdf_list}"
    )
_GENERIC_PERSONA: str = (
    "You are a skilled, thoughtful teaching assistant. Help the student learn through "
    "clear explanations and examples. Keep responses clear, concise, and focused."
)


class ChatSessionError(Exception):
    """Raised when a chat session id is unknown or has expired."""


def _new_chat_model(model: str | None = None) -> ChatGroq:
    """Create the chat LLM lazily (never at import time)."""
    return ChatGroq(model=model_for("chat", model), temperature=0.3)


def _register(session: dict) -> str:
    """Store a session and return its new id, evicting the oldest if over cap."""
    session_id = uuid4().hex
    with _sessions_lock:
        _sessions[session_id] = session
        while len(_sessions) > _MAX_SESSIONS:
            del _sessions[next(iter(_sessions))]   # dicts preserve insertion order
    return session_id


def start_subject_session(subject: str, model: str | None = None, owner: str = LOCAL_OWNER) -> tuple[str, str, str | None]:
    """Start a session backed by a subject's persistent FAISS index.

    The index is built once and reloaded from disk on later runs (see
    backend.vector_store), so reopening a subject is fast.
    """
    retriever = vector_store.get_subject_retriever(subject, _RETRIEVE_K, owner)
    label = f"Subject · {subject}"
    chat_id = db.create_chat(owner, {"mode": "subject", "subject": subject, "label": label,
                                     "model": model_for("chat", model)})
    session_id = _register({
        "mode": "subject",
        "retriever": retriever,
        "llm": _new_chat_model(model),
        "history": [],
        "label": label,
        "topic": None,
        "subject": subject,
        "owner": owner,
        "chat_id": chat_id,
        "persona": _subject_persona(subject, owner),
    })
    return session_id, label, chat_id


def start_topic_session(topic: str, model: str | None = None, owner: str = LOCAL_OWNER) -> tuple[str, str, str | None]:
    """Start a document-free session that teaches a specific topic."""
    label = f"Topic · {topic}"
    chat_id = db.create_chat(owner, {"mode": "topic", "topic": topic, "label": label,
                                     "model": model_for("chat", model)})
    session_id = _register({
        "mode": "topic",
        "retriever": None,
        "llm": _new_chat_model(model),
        "history": [],
        "label": label,
        "topic": topic,
        "subject": None,
        "owner": owner,
        "chat_id": chat_id,
    })
    return session_id, label, chat_id


def start_pdf_session(file_bytes: bytes, filename: str, model: str | None = None, owner: str = LOCAL_OWNER) -> tuple[str, str, str | None]:
    """Start a session backed by a single uploaded PDF (its bytes)."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    try:
        tmp.write(file_bytes)
        tmp.close()
        store = vector_store.build_index_from_paths([_Path(tmp.name)])
    except vector_store.VectorStoreError as exc:
        raise ChatSessionError(str(exc)) from exc
    finally:
        _Path(tmp.name).unlink(missing_ok=True)

    retriever = store.as_retriever(search_kwargs={"k": _RETRIEVE_K})
    label = f"PDF · {filename}"
    chat_id = db.create_chat(owner, {"mode": "pdf", "label": label, "model": model_for("chat", model)})
    session_id = _register({
        "mode": "pdf",
        "retriever": retriever,
        "llm": _new_chat_model(model),
        "history": [],
        "label": label,
        "topic": None,
        "subject": None,
        "owner": owner,
        "chat_id": chat_id,
    })
    return session_id, label, chat_id


def _topic_persona(topic: str) -> str:
    """Build the system persona for a topic-only (document-free) session."""
    return (
        f"You are a skilled, thoughtful teaching assistant specializing in {topic}. "
        f"Help the student learn {topic} through clear explanations and examples. "
        f"Keep responses clear, concise, and focused; break complex concepts into "
        f"simpler parts; provide practical examples when relevant; and encourage "
        f"hands-on learning. If a question falls outside {topic}, kindly redirect."
    )


def _autoname_chat(owner: str, chat_id: str, question: str, answer: str, llm: "ChatGroq") -> None:
    """Generate a concise LLM title for a new chat and save it (best-effort, background).

    Runs in a daemon thread so the first reply is never delayed by the extra call.
    Falls back silently to the provisional title (first message) on any failure.
    """
    try:
        prompt = (
            "Create a short, specific title for this conversation in 3 to 6 words. "
            "Reply with ONLY the title — no quotation marks and no trailing punctuation.\n\n"
            f"User: {question[:500]}\nAssistant: {answer[:500]}"
        )
        raw = str(llm.invoke([HumanMessage(content=prompt)]).content)
        title = " ".join(raw.strip().strip('"').strip("'").strip().rstrip(".").split())
        if title:
            db.rename_item("chats", owner, chat_id, title[:80])
    except Exception:  # noqa: BLE001 — keep the provisional title if naming fails
        pass


def _history_messages(history: list[tuple[str, str]]) -> list:
    """Render the last few turns as alternating Human/AI messages."""
    messages: list = []
    for question, answer in history[-_HISTORY_TURNS:]:
        messages.append(HumanMessage(content=question))
        messages.append(AIMessage(content=answer))
    return messages


def ask(session_id: str, message: str) -> dict:
    """Answer a question within an existing session and record the exchange.

    Returns a dict with ``answer``, ``steps`` (agentic trace), and ``sources``
    (PDF filenames referenced in retrieved passages).
    """
    session = _sessions.get(session_id)
    if session is None:
        raise ChatSessionError(f"Unknown or expired chat session: {session_id}")

    require_groq_key()

    history: list[tuple[str, str]] = session["history"]
    retriever = session["retriever"]
    first_turn = not history
    steps: list[str] = []
    sources: list[str] = []
    artifact = None

    persona = session.get("persona") or _SUBJECT_PERSONA_BASE

    # Tool orchestration: "create a quiz/flashcards/summary/plan/mind map" requests
    # are detected (cheap gate + one small LLM call) and fulfilled by the matching
    # tool, saved to its history. Anything else falls through to the normal answer.
    from backend.services import chat_tools
    intent = chat_tools.detect_intent(session["llm"], message)
    tool_result = chat_tools.run_tool(session, intent, message) if intent else None
    if tool_result is not None:
        answer = tool_result["answer"]
        steps = tool_result["steps"]
        artifact = tool_result.get("artifact")
    elif retriever is not None and config.AGENTIC_RAG:
        # Agentic RAG: an LLM-driven rewrite→retrieve→grade→correct loop.
        answer, steps, sources = agentic_rag.answer(
            retriever, session["llm"], message, _history_messages(history), persona
        )
        answer = str(answer)
    elif retriever is not None:
        # Classic single-shot RAG (AGENTIC_RAG disabled).
        docs = retriever.invoke(message)
        context = "\n\n".join(doc.page_content for doc in docs)
        steps = [
            {
                "kind": "retrieve",
                "title": f"Retrieved {len(docs)} passage{'s' if len(docs) != 1 else ''}",
                "queries": [message],
                "passages": agentic_rag.passage_view(docs),
            },
            {"kind": "generate", "title": "Writing the answer", "passages_used": len(docs)},
        ]
        seen_names: set[str] = set()
        for doc in docs:
            src = (getattr(doc, "metadata", {}) or {}).get("source", "")
            if src:
                name = _Path(src).name
                if name not in seen_names:
                    seen_names.add(name)
                    sources.append(name)
        msgs = [SystemMessage(content=persona)]
        msgs.extend(_history_messages(history))
        msgs.append(HumanMessage(content=f"Context:\n{context}\n\nQuestion: {message}"))
        answer = str(session["llm"].invoke(msgs).content)
    else:
        _persona = _topic_persona(session["topic"]) if session.get("topic") else _GENERIC_PERSONA
        steps = [{
            "kind": "generate",
            "title": "Answered from topic knowledge (no document retrieval)",
        }]
        msgs = [SystemMessage(content=_persona)]
        msgs.extend(_history_messages(history))
        msgs.append(HumanMessage(content=message))
        answer = str(session["llm"].invoke(msgs).content)

    history.append((message, answer))

    # Mirror the exchange to the persisted chat (no-op when persistence is off).
    owner, chat_id = session.get("owner", LOCAL_OWNER), session.get("chat_id")
    if chat_id:
        if first_turn:
            db.set_chat_title(owner, chat_id, message[:60])  # provisional, until the summary lands
        db.append_chat_message(owner, chat_id, "user", message)
        db.append_chat_message(owner, chat_id, "assistant", answer)
        if first_turn:
            # Auto-name the chat with an LLM-generated summary, in the background.
            threading.Thread(
                target=_autoname_chat, args=(owner, chat_id, message, answer, session["llm"]), daemon=True
            ).start()
    return {"answer": answer, "steps": steps, "sources": sources, "artifact": artifact}


def resume_session(owner: str, chat_id: str) -> tuple[str, str, str, str, list[dict]]:
    """Re-open a saved chat as a live session and return its prior messages.

    Subject chats restore their retriever; topic chats restore their persona; PDF
    chats continue without the original document (its embeddings weren't persisted).
    Returns (session_id, label, mode, chat_id, messages).
    """
    doc = db.get_item("chats", owner, chat_id)
    if not doc:
        raise ChatSessionError("That chat could not be found.")

    mode = doc.get("mode", "topic")
    label = doc.get("label") or "Chat"
    subject = doc.get("subject")
    topic = doc.get("topic")
    model = doc.get("model")
    stored = doc.get("messages", []) or []

    retriever = None
    if mode == "subject" and subject:
        retriever = vector_store.get_subject_retriever(subject, _RETRIEVE_K, owner)

    # Rebuild the (question, answer) history from stored alternating messages.
    history: list[tuple[str, str]] = []
    pending_q: str | None = None
    for msg in stored:
        if msg.get("role") == "user":
            pending_q = msg.get("content", "")
        elif msg.get("role") == "assistant" and pending_q is not None:
            history.append((pending_q, msg.get("content", "")))
            pending_q = None

    session_id = _register({
        "mode": mode,
        "retriever": retriever,
        "llm": _new_chat_model(model),
        "history": history,
        "label": label,
        "topic": topic,
        "subject": subject,
        "owner": owner,
        "chat_id": chat_id,
        "persona": _subject_persona(subject, owner) if mode == "subject" and subject else None,
    })
    messages = [{"role": m.get("role"), "content": m.get("content", "")} for m in stored]
    return session_id, label, mode, chat_id, messages
