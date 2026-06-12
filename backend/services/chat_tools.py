"""Chat tool orchestration — the Chat Assistant can CREATE artifacts in the other tools.

When a chat message is a creation request ("make me a quiz on sorting", "create
flashcards", "generate a study plan for next week", "build a mind map of this"),
the chat agent detects the intent, invokes the matching tool's service with the
session's own context (subject PDFs / topic), saves the result to that tool's
per-user history — exactly as if it had been generated inside the tool — and
returns an artifact card for the thread.

Design rules:
  * Detection is two-stage: a cheap regex gate (creation verb + tool noun) so the
    vast majority of messages never pay an extra LLM call, then one small JSON
    LLM call to confirm intent and extract parameters. Any parse failure means
    "not a request" — the message falls through to the normal RAG answer.
  * Tool failures never 502 the chat: the user gets a readable "couldn't create
    it because…" answer instead. Groq rate-limit/API errors still propagate to
    the global handlers.
  * Artifacts are saved with the SAME document shape as each tool's router, so
    they open seamlessly from that tool's history drawer.
"""
from __future__ import annotations

import datetime
import json
import logging
import re

from groq import APIError
from langchain_core.messages import HumanMessage, SystemMessage

from backend import config, db
from backend.config import MAX_CONTENT_TOKENS
from backend.documents import clip_to_tokens, list_pdf_names, read_pdf
from backend.services import flashcards as flashcards_service
from backend.services import mindmap as mindmap_service
from backend.services import planner as planner_service
from backend.services import quiz as quiz_service

log = logging.getLogger(__name__)

# ---- stage 1: regex gate (no LLM cost) ----
_VERBS = r"(?:create|make|generate|build|prepare|design|give\s+me|set\s+up|setup|can\s+you\s+(?:create|make|generate|build))"
_TOOLS = r"(?:quiz(?:zes)?|test|flash\s?cards?|summar(?:y|ies|ize|ise)|study\s?plan|plan(?:ner)?|mind\s?map)"
_GATE = re.compile(rf"\b{_VERBS}\b.{{0,60}}\b{_TOOLS}\b", re.IGNORECASE | re.DOTALL)

_INTENT_SYSTEM = (
    "You classify whether a student's chat message is a request to CREATE a study "
    "artifact (quiz, flashcards, summary, study plan, or mind map), and extract "
    "parameters. Questions ABOUT these things (e.g. 'what is a mind map?') are NOT "
    "creation requests. Always reply with valid JSON only."
)

_TOOL_ROUTES = {
    "quiz": ("quizzes", "#quiz", "Quiz Generator"),
    "flashcards": ("flashcards", "#flashcards", "Flashcards"),
    "summary": ("summaries", "#planner", "Study Planner"),
    "plan": ("plans", "#planner", "Study Planner"),
    "mindmap": ("mindmaps", "#mindmap", "Mind Map"),
}


def detect_intent(llm, message: str) -> dict | None:
    """Return {"tool", "topic", "count", "days"} if the message asks to create one, else None."""
    if not _GATE.search(message or ""):
        return None
    instruction = (
        f"Message: {message}\n\n"
        'Return ONLY JSON: {"is_request": true|false, '
        '"tool": "quiz"|"flashcards"|"summary"|"plan"|"mindmap"|null, '
        '"topic": "<specific topic mentioned, or null>", '
        '"count": <number of questions/cards if mentioned, else null>, '
        '"days": <days until target date if mentioned, else null>}'
    )
    try:
        raw = str(llm.invoke([SystemMessage(content=_INTENT_SYSTEM), HumanMessage(content=instruction)]).content)
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        data = json.loads(match.group(0)) if match else {}
    except APIError:
        raise
    except Exception:  # noqa: BLE001 — detection is best-effort; fall through to normal RAG
        log.debug("Intent detection failed; treating as a normal question.", exc_info=True)
        return None
    if not isinstance(data, dict) or not data.get("is_request"):
        return None
    tool = data.get("tool")
    if tool not in _TOOL_ROUTES:
        return None
    count = data.get("count")
    days = data.get("days")
    return {
        "tool": tool,
        "topic": (data.get("topic") or None),
        "count": int(count) if isinstance(count, (int, float)) and count else None,
        "days": int(days) if isinstance(days, (int, float)) and days else None,
    }


def _subject_content(subject: str, owner: str | None) -> tuple[str, list[str]]:
    """Concatenate (and clip) the subject's PDFs into one content blob."""
    pdfs = list_pdf_names(subject, owner)
    if not pdfs:
        raise _ToolFailure(f"“{subject}” has no PDFs to work from.")
    parts = []
    for pdf in pdfs[:3]:  # a few documents are plenty within the token budget
        try:
            parts.append(f"## {pdf}\n{read_pdf(subject, pdf, owner)}")
        except Exception:  # noqa: BLE001 — skip unreadable PDFs, keep the rest
            continue
    if not parts:
        raise _ToolFailure(f"Couldn't read any PDFs in “{subject}”.")
    return clip_to_tokens("\n\n".join(parts), MAX_CONTENT_TOKENS), pdfs[:3]


class _ToolFailure(Exception):
    """A readable reason the artifact couldn't be created (shown as the answer)."""


def _trace(title: str, detail: str | None = None) -> dict:
    step = {"kind": "tool", "title": title}
    if detail:
        step["missing"] = detail  # reuses the trace-note rendering in the UI
    return step


def run_tool(session: dict, intent: dict, message: str) -> dict | None:
    """Create the requested artifact from the session's context.

    Returns {"answer", "steps", "artifact"} — or None to signal "answer normally"
    (e.g. a PDF-upload session with no topic to fall back on).
    """
    tool = intent["tool"]
    kind, route, tool_name = _TOOL_ROUTES[tool]
    owner = session.get("owner") or "local"
    subject = session.get("subject")
    topic = intent.get("topic") or session.get("topic")
    steps = [_trace(f"Detected a {tool_name.lower()} request", intent.get("topic") and f"topic → {intent['topic']}")]

    try:
        if tool == "quiz":
            n = min(intent.get("count") or 5, 15)
            if subject:
                content, _ = _subject_content(subject, owner)
                data = quiz_service.make_quiz(content, n, "medium")
                source = subject
            elif topic:
                data = quiz_service.make_quiz_from_topic(topic, n, "medium")
                source = topic
            else:
                return None
            questions = data.get("questions") or []
            if not questions:
                raise _ToolFailure("the model returned no usable questions — try again.")
            title = f"Quiz · {source}"
            saved_id = db.save("quizzes", owner, {
                "title": title, "source": source, "difficulty": "medium",
                "model": config.model_for("quiz", None),
                "num_questions": len(questions), "questions": questions,
            })
            detail = f"{len(questions)} questions"

        elif tool == "flashcards":
            n = min(intent.get("count") or 10, 25)
            if subject:
                content, _ = _subject_content(subject, owner)
                data = flashcards_service.make_flashcards(content, n, "medium")
                source = subject
            elif topic:
                data = flashcards_service.make_flashcards_from_topic(topic, n, "medium")
                source = topic
            else:
                return None
            cards = data.get("cards") or []
            if not cards:
                raise _ToolFailure("the model returned no usable cards — try again.")
            cards = [c.model_dump() if hasattr(c, "model_dump") else c for c in cards]
            title = f"Flashcards · {source}"
            saved_id = db.save("flashcards", owner, {
                "title": title, "source": source, "difficulty": "medium",
                "model": config.model_for("flashcards", None),
                "num_cards": len(cards), "cards": cards,
            })
            detail = f"{len(cards)} cards"

        elif tool == "summary":
            if not subject:
                raise _ToolFailure(
                    "summaries are built from documents — start a Subject chat, or use the "
                    "Study Planner's “Generate summary” with your PDFs."
                )
            pdfs = list_pdf_names(subject, owner)[:3]
            if not pdfs:
                raise _ToolFailure(f"“{subject}” has no PDFs to summarize.")
            summaries = planner_service.summarize_pdfs(subject, pdfs, owner=owner)
            title = f"Summary · {subject}"
            saved_id = db.save("summaries", owner, {
                "title": title, "source": subject, "subject": subject, "pdfs": pdfs,
                "model": config.model_for("planner", None), "summaries": summaries,
            })
            detail = f"{len(summaries)} document(s)"

        elif tool == "plan":
            days = min(intent.get("days") or 7, 90)
            target = (datetime.date.today() + datetime.timedelta(days=days)).isoformat()
            if subject:
                pdfs = list_pdf_names(subject, owner)[:3]
                if not pdfs:
                    raise _ToolFailure(f"“{subject}” has no PDFs to plan around.")
                plan = planner_service.generate_plan(subject, pdfs, target, owner=owner)
                source = subject
            elif topic:
                plan = planner_service.generate_plan_from_topic(topic, target)
                source = topic
            else:
                return None
            title = f"Plan · {source}"
            saved_id = db.save("plans", owner, {
                "title": title, "source": source, "target_date": target,
                "model": config.model_for("planner", None), "plan": plan,
            })
            detail = f"through {target}"

        else:  # mindmap
            if subject:
                content, _ = _subject_content(subject, owner)
                root = mindmap_service.generate_mindmap(content=content)
                source = subject
            elif topic:
                root = mindmap_service.generate_mindmap(topic=topic)
                source = topic
            else:
                return None
            if not isinstance(root, dict) or not root.get("label"):
                raise _ToolFailure("the model returned an unusable mind map — try again.")
            title = root.get("label") or f"Mind map · {source}"
            saved_id = db.save("mindmaps", owner, {
                "title": title, "source": source,
                "model": config.model_for("mindmap", None), "root": root,
            })
            detail = None

    except _ToolFailure as exc:
        steps.append(_trace("Couldn't create it", str(exc)))
        return {
            "answer": f"I couldn't create that {tool_name.lower()} — {exc}",
            "steps": steps, "artifact": None,
        }
    except APIError:
        raise
    except Exception as exc:  # noqa: BLE001 — a tool bug must not take down the chat
        log.warning("Chat tool orchestration failed (%s): %s", tool, exc)
        steps.append(_trace("Couldn't create it", str(exc)))
        return {
            "answer": f"I tried to create that {tool_name.lower()} but something went wrong: {exc}",
            "steps": steps, "artifact": None,
        }

    steps.append(_trace(f"Created “{title}”" + (f" — {detail}" if detail else "")))
    saved_note = (
        f"It's saved in your **{tool_name}** history." if saved_id
        else f"Open the **{tool_name}** to use it (history is off, so it wasn't saved)."
    )
    return {
        "answer": f"Done — I created **{title}**" + (f" ({detail})" if detail else "") + f". {saved_note}",
        "steps": steps,
        "artifact": {"kind": kind, "id": saved_id, "title": title, "route": route, "tool": tool_name},
    }
