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
from backend.services import cheatsheet as cheatsheet_service
from backend.services import exam as exam_service
from backend.services import flashcards as flashcards_service
from backend.services import mindmap as mindmap_service
from backend.services import planner as planner_service
from backend.services import predictor as predictor_service
from backend.services import quiz as quiz_service

log = logging.getLogger(__name__)

# ---- stage 1: regex gate (no LLM cost) ----
_VERBS = r"(?:create|make|generate|build|prepare|design|predict|start|begin|take|attempt|give\s+me|set\s+up|setup|let'?s\s+(?:start|do|take|make|create|build|begin|generate)|can\s+you\s+(?:create|make|generate|build|predict|start|do|take|begin))"
_TOOLS = r"(?:quiz(?:zes)?|test|flash\s?cards?|summar(?:y|ies|ize|ise)|study\s?plan|plan(?:ner)?|mind\s?map|cheat\s?sheet|predict(?:ion|or)?s?|likely\s+(?:exam\s+)?questions|(?:mock\s?)?exam)"
_GATE = re.compile(rf"\b{_VERBS}\b.{{0,60}}\b{_TOOLS}\b", re.IGNORECASE | re.DOTALL)

_INTENT_SYSTEM = (
    "You classify whether a student's chat message is a request to CREATE a study "
    "artifact (quiz, flashcards, summary, study plan, mind map, cheat sheet, "
    "exam-question predictions, or a mock exam to take), and extract parameters. "
    "Questions ABOUT these things (e.g. 'what is a mind map?') are NOT creation "
    "requests. Always reply with valid JSON only."
)

_TOOL_ROUTES = {
    "quiz": ("quizzes", "#quiz", "Quiz Generator"),
    "flashcards": ("flashcards", "#flashcards", "Flashcards"),
    "summary": ("summaries", "#planner", "Study Planner"),
    "plan": ("plans", "#planner", "Study Planner"),
    "mindmap": ("mindmaps", "#mindmap", "Mind Map"),
    "cheatsheet": ("cheatsheets", "#cheatsheet", "Cheat Sheet"),
    "predictor": ("predictions", "#predictor", "Question Predictor"),
    "exam": ("exams", "#exam", "Mock Exam"),
}


def detect_intent(llm, message: str) -> dict | None:
    """Return {"tool", "topic", "count", "days"} if the message asks to create one, else None."""
    if not _GATE.search(message or ""):
        return None
    instruction = (
        f"Message: {message}\n\n"
        'Return ONLY JSON: {"is_request": true|false, '
        '"tool": "quiz"|"flashcards"|"summary"|"plan"|"mindmap"|"cheatsheet"|"predictor"|"exam"|null, '
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


def _subject_papers_corpus(subject: str, owner: str | None) -> tuple[str, list[str]]:
    """Build a paper-labelled corpus for the predictor, matching its router.

    The Question Predictor's prompt tells the model the papers are separated by
    ``=== PAPER: <filename> ===`` markers (see routers/predictor.py). Reusing
    ``_subject_content`` would feed ``## <name>`` headers instead, so the model
    is told to key on a separator that isn't there — degrading cross-paper
    pattern detection. This emits the exact markers the router does. Returns
    ``(corpus, used_pdfs)`` where ``used_pdfs`` is only the papers actually read.
    """
    pdfs = list_pdf_names(subject, owner)
    if not pdfs:
        raise _ToolFailure(f"“{subject}” has no past papers to analyze.")
    parts: list[str] = []
    used: list[str] = []
    for pdf in pdfs[:3]:  # a few papers stay within the token budget
        try:
            parts.append(f"=== PAPER: {pdf} ===\n{read_pdf(subject, pdf, owner)}")
            used.append(pdf)
        except Exception:  # noqa: BLE001 — skip unreadable PDFs, keep the rest
            continue
    if not parts:
        raise _ToolFailure(f"Couldn't read any papers in “{subject}”.")
    return clip_to_tokens("\n\n".join(parts), MAX_CONTENT_TOKENS), used


class _ToolFailure(Exception):
    """A readable reason the artifact couldn't be created (shown as the answer)."""


# Roughly the token budget for the stored rendition; keeps follow-up context bounded.
_ARTIFACT_TEXT_TOKENS = 1500


def _outline(node: object, depth: int = 0) -> list[str]:
    """Render a mind-map {label, children} tree as an indented bullet outline."""
    if not isinstance(node, dict):
        return []
    label = str(node.get("label") or "").strip()
    lines: list[str] = []
    if label:
        lines.append(f"{'  ' * depth}- {label}")
    for child in node.get("children") or []:
        lines.extend(_outline(child, depth + 1))
    return lines


def _artifact_text(tool: str, title: str, art_data: dict) -> str:
    """Build a compact text rendition of a created artifact for follow-up chat.

    Captures what the student is looking at so a later, non-tool turn can discuss
    it ("why is C correct", "explain question 3", "summarise this cheat sheet").
    The result is clipped to ``_ARTIFACT_TEXT_TOKENS`` by the caller.
    """
    parts: list[str] = []
    if tool in ("quiz", "exam"):
        for i, q in enumerate(art_data.get("questions") or [], start=1):
            if not isinstance(q, dict):
                continue
            prompt = str(q.get("question") or "").strip()
            if q.get("type") == "short" or (tool == "exam" and not q.get("options")):
                points = q.get("expected_points") or []
                parts.append(f"Q{i} (short answer). {prompt}")
                if points:
                    parts.append("Expected points: " + "; ".join(str(p) for p in points))
                continue
            options = q.get("options")
            if isinstance(options, dict):  # quiz: {"A": "...", ...}, correct is a letter
                opt_lines = [f"  {k}. {v}" for k, v in options.items()]
                correct = str(q.get("correct") or "").strip()
                correct_text = options.get(correct, "")
                parts.append(f"Q{i}. {prompt}\n" + "\n".join(opt_lines))
                parts.append(f"Correct answer: {correct}. {correct_text}".rstrip())
            elif isinstance(options, list):  # exam MCQ: list + answer_index
                opt_lines = [f"  {chr(65 + j)}. {o}" for j, o in enumerate(options)]
                idx = q.get("answer_index")
                correct_text = options[idx] if isinstance(idx, int) and 0 <= idx < len(options) else ""
                letter = chr(65 + idx) if isinstance(idx, int) and 0 <= idx < len(options) else "?"
                parts.append(f"Q{i}. {prompt}\n" + "\n".join(opt_lines))
                parts.append(f"Correct answer: {letter}. {correct_text}".rstrip())
            else:
                parts.append(f"Q{i}. {prompt}")
            explanation = str(q.get("explanation") or "").strip()
            if explanation:
                parts.append(f"Explanation: {explanation}")
    elif tool == "flashcards":
        for i, c in enumerate(art_data.get("cards") or [], start=1):
            if not isinstance(c, dict):
                continue
            front = str(c.get("front") or "").strip()
            back = str(c.get("back") or "").strip()
            if front or back:
                parts.append(f"Card {i}: {front} → {back}")
    elif tool in ("cheatsheet", "plan"):
        parts.append(str(art_data.get("markdown") or art_data.get("plan") or ""))
    elif tool == "summary":
        for pdf, summary in (art_data.get("summaries") or {}).items():
            parts.append(f"## {pdf}\n{summary}")
    elif tool == "mindmap":
        parts.extend(_outline(art_data.get("root")))
    elif tool == "predictor":
        for i, p in enumerate(art_data.get("predicted_questions") or [], start=1):
            if not isinstance(p, dict):
                continue
            question = str(p.get("question") or "").strip()
            topic = str(p.get("topic") or "").strip()
            likelihood = str(p.get("likelihood") or "").strip()
            reasoning = str(p.get("reasoning") or "").strip()
            head = f"{i}. [{likelihood}] {question}" if likelihood else f"{i}. {question}"
            if topic:
                head += f" (topic: {topic})"
            parts.append(head)
            if reasoning:
                parts.append(f"   Why: {reasoning}")
    text = "\n".join(p for p in parts if p).strip()
    return clip_to_tokens(text, _ARTIFACT_TEXT_TOKENS) if text else ""


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
            art_data = {"questions": questions}

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
            art_data = {"cards": cards}

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
            art_data = {"summaries": summaries}

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
            art_data = {"plan": plan}

        elif tool == "mindmap":
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
            art_data = {"root": root}

        elif tool == "cheatsheet":
            if not subject:
                raise _ToolFailure(
                    "cheat sheets are built from documents — start a Subject chat so I have "
                    "PDFs to condense."
                )
            content, _ = _subject_content(subject, owner)
            markdown = cheatsheet_service.make_cheatsheet(content)
            title = f"Cheat Sheet · {subject}"
            saved_id = db.save("cheatsheets", owner, {
                "title": title, "source": subject,
                "model": config.model_for("cheatsheet", None), "markdown": markdown,
            })
            detail = None
            art_data = {"markdown": markdown}

        elif tool == "predictor":
            if not subject:
                raise _ToolFailure(
                    "predictions are built from past papers — start a Subject chat so I have "
                    "papers to analyze."
                )
            corpus, pdfs = _subject_papers_corpus(subject, owner)
            data = predictor_service.predict_questions(corpus, subject_hint=subject)
            predicted = data.get("predicted_questions") or []
            patterns = data.get("patterns") or []
            title = f"Predictions · {subject}"
            saved_id = db.save("predictions", owner, {
                "title": title, "source": subject, "subject": subject,
                "model": config.model_for("predictor", None), "num_papers": len(pdfs),
                "patterns": patterns, "predicted_questions": predicted,
            })
            detail = f"{len(predicted)} likely questions"
            art_data = {"patterns": patterns, "predicted_questions": predicted}

        else:  # exam — subject-locked; generated only from the session subject
            if not subject:
                raise _ToolFailure(
                    "mock exams are built from documents — start a Subject chat so I have "
                    "PDFs to examine you on."
                )
            num_mcq = min(intent.get("count") or 6, 10)
            num_short = 2
            content, _ = _subject_content(subject, owner)
            data = exam_service.generate_exam(content, num_mcq, num_short)
            questions = data.get("questions") or []
            if not questions:
                raise _ToolFailure("the model returned no usable questions — try again.")
            # An exam is persisted as a GRADED ATTEMPT only after the user submits
            # (POST /api/exam/grade), so nothing is saved here.
            saved_id = None
            title = f"Mock Exam · {subject}"
            detail = f"{num_mcq} MCQ + {num_short} short"
            art_data = {
                "questions": questions, "subject": subject,
                "num_mcq": num_mcq, "num_short": num_short,
            }

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

    # Remember a compact rendition so later, non-tool turns can DISCUSS this
    # artifact (e.g. "explain question 3", "summarise this cheat sheet"). This
    # changes nothing shown in the chat — chat.ask() injects it into context.
    try:
        rendition = _artifact_text(tool, title, art_data)
        if rendition:
            session["recent_artifact"] = {"title": title, "kind": kind, "text": rendition}
    except Exception:  # noqa: BLE001 — a rendition bug must never break artifact creation
        log.debug("Could not build recent-artifact rendition (%s).", tool, exc_info=True)

    if tool == "exam":
        # Not saved yet — it's persisted as a graded attempt only after submit.
        return {
            "answer": "Here is your mock exam — take it below, then hit Submit and "
                      "I'll grade it for you.",
            "steps": steps,
            "artifact": {
                "kind": kind, "id": None, "title": title, "route": route,
                "tool": tool_name, "data": art_data,
            },
        }
    saved_note = (
        f"It's saved in your **{tool_name}** history." if saved_id
        else f"Open the **{tool_name}** to use it (history is off, so it wasn't saved)."
    )
    return {
        "answer": f"Done — I created **{title}**" + (f" ({detail})" if detail else "") + f". {saved_note}",
        "steps": steps,
        "artifact": {
            "kind": kind, "id": saved_id, "title": title, "route": route,
            "tool": tool_name, "data": art_data,
        },
    }
