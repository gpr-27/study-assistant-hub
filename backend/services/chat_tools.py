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
    it because…" answer instead. ``LLMChainError`` (every provider failed) still
    propagates so the chat answers 503 / an SSE error carrying the "no model" label.
  * Stateless: ``run_tool`` takes a plain context dict and RETURNS what the session should
    remember (``recent_artifact``) instead of mutating a process-global session, so it works
    across serverless instances.
  * Artifacts are saved with the SAME document shape as each tool's router, so
    they open seamlessly from that tool's history drawer.
"""
from __future__ import annotations

import datetime
import json
import logging
import re

import time
from collections import OrderedDict
from threading import Lock

from backend import config, db, deadline, llm_chain
from backend.errors import public_message
from backend.llm_chain import LLMChainError
from backend.services.llm_json import extract_json_object
from backend.documents import list_pdf_names, read_pdf
from backend.services import cheatsheet as cheatsheet_service
from backend.services import exam as exam_service
from backend.services import grading
from backend.services import flashcards as flashcards_service
from backend.services import mindmap as mindmap_service
from backend.services import planner as planner_service
from backend.services import predictor as predictor_service
from backend.services import progress as progress_service
from backend.services import quiz as quiz_service
from backend.services import teachback as teachback_service

log = logging.getLogger(__name__)

# ---- stage 1: regex gate (no LLM cost) ----
_VERBS = r"(?:create|make|generate|build|prepare|design|predict|start|begin|take|attempt|give\s+me|set\s+up|setup|let'?s\s+(?:start|do|take|make|create|build|begin|generate)|can\s+you\s+(?:create|make|generate|build|predict|start|do|take|begin))"
_TOOLS = r"(?:quiz(?:zes)?|test|flash\s?cards?|summar(?:y|ies|ize|ise)|study\s?plan|plan(?:ner)?|mind\s?map|cheat\s?sheet|predict(?:ion|or)?s?|likely\s+(?:exam\s+)?questions|(?:mock\s?)?exam|teach[\s-]?back|feynman)"
_GATE = re.compile(rf"\b{_VERBS}\b.{{0,60}}\b{_TOOLS}\b", re.IGNORECASE | re.DOTALL)

_INTENT_SYSTEM = (
    "You classify whether a student's chat message is a request to CREATE a study "
    "artifact (quiz, flashcards, summary, study plan, mind map, cheat sheet, "
    "exam-question predictions, a mock exam to take, or a teach-back/Feynman "
    "session where the student explains a concept aloud), and extract parameters. "
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
    "teachback": ("teachbacks", "#teachback", "Teach-back Tutor"),
    "progress": ("progress", "#progress", "Progress"),
}

# Read-only progress/insights request — no creation verb, so it's detected
# separately from the creation gate (see detect_intent).
_PROGRESS_GATE = re.compile(
    r"\bmy\s+progress\b|\bhow\s+am\s+i\s+doing\b|\bmy\s+(?:stats|analytics|insights|mastery)\b"
    r"|\bshow\b.{0,30}\bprogress\b",
    re.IGNORECASE | re.DOTALL,
)


# Difficulty levels the quiz / flashcards / exam tools accept (default "medium").
_DIFFICULTIES = {"easy", "medium", "hard"}


def _norm_difficulty(value: object) -> str | None:
    """Normalize a difficulty value to easy|medium|hard, or None."""
    if isinstance(value, str) and value.strip().lower() in _DIFFICULTIES:
        return value.strip().lower()
    return None


def _difficulty_from_text(text: str) -> str | None:
    """Pull an explicit difficulty out of free text ("go with hard", "make it easy")."""
    t = (text or "").lower()
    if re.search(r"\b(hard|tough|difficult|advanced|challenging)\b", t):
        return "hard"
    if re.search(r"\b(easy|simple|basic|beginner|gentle)\b", t):
        return "easy"
    if re.search(r"\b(medium|moderate|intermediate)\b", t):
        return "medium"
    return None


def _count_from_text(text: str) -> int | None:
    """Pull an explicit item count ("15 questions", "15 hard questions", "10 cards")."""
    m = re.search(
        r"(\d{1,3})\s*(?:(?:easy|medium|hard|simple|tough|difficult|new|more|extra)\s+)*"
        r"(?:questions?|mcqs?|cards?|flash\s?cards?|items?)",
        text or "", re.IGNORECASE,
    )
    return int(m.group(1)) if m else None


def _days_from_text(text: str) -> int | None:
    """Pull an explicit day count ("in 7 days", "2 weeks" not handled) out of free text."""
    m = re.search(r"(\d{1,3})\s*days?", text or "", re.IGNORECASE)
    return int(m.group(1)) if m else None


def _short_count_from_text(text: str) -> int | None:
    """Pull an explicit short-answer count ("3 short", "2 short-answer questions") — exam only."""
    m = re.search(r"(\d{1,2})\s*short(?:[\s-]?answer)?", text or "", re.IGNORECASE)
    return int(m.group(1)) if m else None


def forced_intents(tool: str | None, message: str = "") -> list[dict] | None:
    """Build a one-tool intent list for a MANUAL tool pick (the slash / tool picker), or None.

    Bypasses the LLM ``detect_intent`` (the user explicitly chose the tool), but still reads
    explicit parameters straight out of the message — count ("15 questions" / "10 cards"),
    difficulty ("go with hard"), and days ("in 7 days") — so "/quiz 15 hard questions" honours
    both. Anything unspecified stays None and the tool falls back to its default.
    """
    if tool not in _TOOL_ROUTES:
        return None
    return [{
        "tool": tool,
        "topic": None,
        "count": _count_from_text(message),
        "days": _days_from_text(message),
        "difficulty": _difficulty_from_text(message),
    }]


def _coerce_intent(item: object) -> dict | None:
    """Normalize one raw intent object into {tool, topic, count, days}, or None if invalid."""
    if not isinstance(item, dict):
        return None
    tool = item.get("tool")
    if tool not in _TOOL_ROUTES:
        return None
    count = item.get("count")
    days = item.get("days")
    return {
        "tool": tool,
        "topic": (item.get("topic") or None),
        "count": int(count) if isinstance(count, (int, float)) and count else None,
        "days": int(days) if isinstance(days, (int, float)) and days else None,
        "difficulty": _norm_difficulty(item.get("difficulty")),
    }


def needs_llm_detection(message: str) -> bool:
    """Cheap regex gate: does this message even look like a tool request? (no LLM cost)"""
    return bool(_PROGRESS_GATE.search(message or "") or _GATE.search(message or ""))


def _intent_instruction(message: str) -> str:
    return (
        f"Message: {message}\n\n"
        "The student may ask for SEVERAL artifacts in one message — return one entry per "
        "distinct tool they ask to create (in the order requested).\n"
        'Return ONLY JSON: {"is_request": true|false, "tools": [ '
        '{"tool": "quiz"|"flashcards"|"summary"|"plan"|"mindmap"|"cheatsheet"|"predictor"|"exam"|"teachback", '
        '"topic": "<specific topic mentioned, or null>", '
        '"count": <number of questions/cards if mentioned, else null>, '
        '"days": <days until target date if mentioned, else null>, '
        '"difficulty": "easy"|"medium"|"hard"|null} ] }'
    )


def _parse_intents(raw: str) -> list[dict] | None:
    data = extract_json_object(raw)
    if not data or not data.get("is_request"):
        return None
    # Prefer the new "tools" array; tolerate a legacy single-"tool" object for back-compat.
    raw_items = data.get("tools")
    if not isinstance(raw_items, list):
        raw_items = [data] if data.get("tool") else []
    intents: list[dict] = []
    seen: set[str] = set()
    for item in raw_items:
        coerced = _coerce_intent(item)
        if coerced and coerced["tool"] not in seen:   # de-dupe repeated tools
            seen.add(coerced["tool"])
            intents.append(coerced)
    return intents or None


def detect_intent(message: str) -> list[dict] | None:
    """Return a LIST of {"tool","topic","count","days","difficulty"} the message asks to create.

    One message can request several artifacts ("make a quiz AND flashcards on sorting"),
    so each requested tool becomes its own intent and the chat runs them all in one turn.
    Returns ``None`` (not ``[]``) when nothing is requested, so callers can ``if intents:``.
    Prompt-level structured JSON (``json_mode``) — no provider-native tool APIs, which differ
    across the 5 chain steps. ``LLMChainError`` propagates; any other failure means "not a request".
    """
    if _PROGRESS_GATE.search(message or ""):
        return [{"tool": "progress", "topic": None, "count": None, "days": None}]
    if not _GATE.search(message or ""):
        return None
    try:
        result = llm_chain.chat(_INTENT_SYSTEM, [{"role": "user", "content": _intent_instruction(message)}],
                                max_tokens=700, temperature=0.0, json_mode=True,
                                deadline_s=deadline.clamp(15.0))
        return _parse_intents(result.text)
    except LLMChainError:
        raise
    except Exception:  # noqa: BLE001 — detection is best-effort; fall through to normal RAG
        log.debug("Intent detection failed; treating as a normal question.", exc_info=True)
        return None


async def adetect_intent(message: str) -> list[dict] | None:
    """Async twin of :func:`detect_intent` (never blocks the event loop)."""
    if _PROGRESS_GATE.search(message or ""):
        return [{"tool": "progress", "topic": None, "count": None, "days": None}]
    if not _GATE.search(message or ""):
        return None
    try:
        result = await llm_chain.achat(_INTENT_SYSTEM, [{"role": "user", "content": _intent_instruction(message)}],
                                       max_tokens=700, temperature=0.0, json_mode=True,
                                       deadline_s=deadline.clamp(15.0))
        return _parse_intents(result.text)
    except LLMChainError:
        raise
    except Exception:  # noqa: BLE001
        log.debug("Intent detection failed; treating as a normal question.", exc_info=True)
        return None


# Bounded TTL cache of extracted+clipped PDF text per (owner, subject, kind, file signatures):
# parsing 3 PDFs with pypdf costs seconds, and a chat session builds several artifacts from the
# same material. Keys include the files' (mtime,size) so a re-upload never serves stale text;
# the owner is part of the key so users never share entries. Per-instance, so it only ever
# speeds things up (a miss just re-reads).
_CONTENT_CACHE: "OrderedDict[tuple, tuple[float, tuple]]" = OrderedDict()
_CONTENT_CACHE_MAX = 32
_CONTENT_TTL_S = 600.0
_content_lock = Lock()


def _signature(subject: str, pdfs: list[str], owner: str | None) -> tuple:
    from backend.documents import pdf_signature

    sig = []
    for name in pdfs:
        try:
            sig.append((name, pdf_signature(subject, name, owner)))
        except Exception:  # noqa: BLE001 — unreadable/missing file just drops from the print
            sig.append((name, None))
    return tuple(sig)


def _cached_build(kind: str, subject: str, owner: str | None, pdfs: list[str], build):
    key = (owner or "", subject, kind, _signature(subject, pdfs, owner))
    now = time.monotonic()
    with _content_lock:
        hit = _CONTENT_CACHE.get(key)
        if hit and now - hit[0] < _CONTENT_TTL_S:
            _CONTENT_CACHE.move_to_end(key)
            return hit[1]
    value = build()
    with _content_lock:
        _CONTENT_CACHE[key] = (time.monotonic(), value)
        _CONTENT_CACHE.move_to_end(key)
        while len(_CONTENT_CACHE) > _CONTENT_CACHE_MAX:
            _CONTENT_CACHE.popitem(last=False)
    return value


# Documents fed to one tool call (a sanity bound on work, not a prompt budget; llm_chain fits per provider).
_MAX_TOOL_PDFS = 10


def _subject_content(subject: str, owner: str | None) -> tuple[str, list[str]]:
    """Concatenate the subject's PDFs (full text) into one content blob."""
    pdfs = list_pdf_names(subject, owner)
    if not pdfs:
        raise _ToolFailure(f"“{subject}” has no PDFs to work from.")
    pdfs = pdfs[:_MAX_TOOL_PDFS]

    def build():
        parts = []
        for pdf in pdfs:
            try:
                parts.append(f"## {pdf}\n{read_pdf(subject, pdf, owner)}")
            except Exception:  # noqa: BLE001 — skip unreadable PDFs, keep the rest
                continue
        if not parts:
            raise _ToolFailure(f"Couldn't read any PDFs in “{subject}”.")
        return "\n\n".join(parts), list(pdfs)

    return _cached_build("content", subject, owner, pdfs, build)


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
    pdfs = pdfs[:_MAX_TOOL_PDFS]

    def build():
        parts: list[str] = []
        used: list[str] = []
        for pdf in pdfs:
            try:
                parts.append(f"=== PAPER: {pdf} ===\n{read_pdf(subject, pdf, owner)}")
                used.append(pdf)
            except Exception:  # noqa: BLE001 — skip unreadable PDFs, keep the rest
                continue
        if not parts:
            raise _ToolFailure(f"Couldn't read any papers in “{subject}”.")
        return "\n\n".join(parts), used

    return _cached_build("papers", subject, owner, pdfs, build)


class _ToolFailure(Exception):
    """A readable reason the artifact couldn't be created (shown as the answer)."""


# Roughly the token budget for the stored rendition; keeps follow-up context bounded.



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
    The result is returned in full.
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
    return text


def _trace(title: str, detail: str | None = None) -> dict:
    step = {"kind": "tool", "title": title}
    if detail:
        step["missing"] = detail  # reuses the trace-note rendering in the UI
    return step


def _take_llm(obj: object) -> dict | None:
    """Pop the public ``llm`` block a generation service attached to its dict result, if any."""
    if isinstance(obj, dict):
        llm = obj.pop("llm", None)
        return llm if isinstance(llm, dict) else None
    return None


def run_tool(session: dict, intent: dict, message: str) -> dict | None:
    """Create the requested artifact from the session's context.

    ``session`` is a plain dict ``{"owner", "subject", "topic"}`` (nothing is mutated). Returns
    {"answer", "steps", "artifact", "recent_artifact", "llm"} — or None to signal "answer
    normally" (e.g. a PDF-upload session with no topic to fall back on). ``recent_artifact`` is
    the compact rendition the CALLER should remember so later turns can discuss the artifact.
    BLOCKING (services call the LLM chain synchronously): run it in a worker thread.
    """
    llm_public: dict | None = None
    tool = intent["tool"]
    kind, route, tool_name = _TOOL_ROUTES[tool]
    owner = session.get("owner") or "local"
    subject = session.get("subject")
    topic = intent.get("topic") or session.get("topic")

    # Progress is a READ-ONLY insight view, not a creation: pull the same summary
    # the Progress dashboard uses (pure Python aggregation, no LLM, nothing saved).
    if tool == "progress":
        steps = [_trace("Pulled your progress")]
        try:
            summary = progress_service.build_summary(owner)
        except Exception as exc:  # noqa: BLE001 — never 502 the chat over a read
            log.warning("Chat progress read failed: %s", exc)       # details stay in the server log
            return {
                "answer": f"I couldn't pull your progress right now. {public_message(exc)}",
                "steps": steps, "artifact": None, "recent_artifact": None, "llm": None,
            }
        has_activity = bool(summary.get("overall", {}).get("total_attempts")) or any(
            (a.get("count") or 0) for a in (summary.get("activity") or [])
        )
        answer = (
            "Here's where you stand across your tools."
            if has_activity
            else "No activity yet — take a quiz, review flashcards or sit a mock exam and "
                 "your progress will show up here."
        )
        return {
            "answer": answer,
            "steps": steps,
            "artifact": {
                "kind": kind, "id": None, "title": "Progress", "route": route,
                "tool": tool_name, "data": summary,
            },
            "recent_artifact": None, "llm": None,
        }

    steps = [_trace(f"Detected a {tool_name.lower()} request", intent.get("topic") and f"topic → {intent['topic']}")]

    try:
        if tool == "quiz":
            n = min(intent.get("count") or 5, 15)
            difficulty = intent.get("difficulty") or "medium"
            if subject:
                content, _ = _subject_content(subject, owner)
                data = quiz_service.make_quiz(content, n, difficulty)
                llm_public = _take_llm(data)
                source = subject
            elif topic:
                data = quiz_service.make_quiz_from_topic(topic, n, difficulty)
                llm_public = _take_llm(data)
                source = topic
            else:
                return None
            questions = data.get("questions") or []
            if not questions:
                raise _ToolFailure("the model returned no usable questions — try again.")
            grading.sign_items(questions)         # HMAC of each answer key: POST /api/grade verifies the attempt
            title = f"Quiz · {source}"
            saved_id = db.save("quizzes", owner, {
                "title": title, "source": source, "difficulty": difficulty,
                "llm": llm_public,
                "num_questions": len(questions), "questions": questions,
            })
            detail = f"{len(questions)} questions · {difficulty}"
            art_data = {"questions": questions, "subject": subject, "topic": None if subject else topic}

        elif tool == "flashcards":
            n = min(intent.get("count") or 10, 25)
            difficulty = intent.get("difficulty") or "medium"
            if subject:
                content, _ = _subject_content(subject, owner)
                data = flashcards_service.make_flashcards(content, n, difficulty)
                llm_public = _take_llm(data)
                source = subject
            elif topic:
                data = flashcards_service.make_flashcards_from_topic(topic, n, difficulty)
                llm_public = _take_llm(data)
                source = topic
            else:
                return None
            cards = data.get("cards") or []
            if not cards:
                raise _ToolFailure("the model returned no usable cards — try again.")
            cards = [c.model_dump() if hasattr(c, "model_dump") else c for c in cards]
            title = f"Flashcards · {source}"
            saved_id = db.save("flashcards", owner, {
                "title": title, "source": source, "difficulty": difficulty,
                "llm": llm_public,
                "num_cards": len(cards), "cards": cards,
            })
            detail = f"{len(cards)} cards · {difficulty}"
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
            summaries, llm_public = planner_service.summarize_pdfs_ex(subject, pdfs, owner=owner)
            title = f"Summary · {subject}"
            saved_id = db.save("summaries", owner, {
                "title": title, "source": subject, "subject": subject, "pdfs": pdfs,
                "llm": llm_public, "summaries": summaries,
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
                plan, llm_public = planner_service.generate_plan_ex(subject, pdfs, target, owner=owner)
                source = subject
            elif topic:
                plan, llm_public = planner_service.generate_plan_from_topic_ex(topic, target)
                source = topic
            else:
                return None
            title = f"Plan · {source}"
            saved_id = db.save("plans", owner, {
                "title": title, "source": source, "target_date": target,
                "llm": llm_public, "plan": plan,
            })
            detail = f"through {target}"
            art_data = {"plan": plan}

        elif tool == "mindmap":
            if subject:
                content, _ = _subject_content(subject, owner)
                root, llm_public = mindmap_service.generate_mindmap_ex(content=content)
                source = subject
            elif topic:
                root, llm_public = mindmap_service.generate_mindmap_ex(topic=topic)
                source = topic
            else:
                return None
            if not isinstance(root, dict) or not root.get("label"):
                raise _ToolFailure("the model returned an unusable mind map — try again.")
            title = root.get("label") or f"Mind map · {source}"
            saved_id = db.save("mindmaps", owner, {
                "title": title, "source": source,
                "llm": llm_public, "root": root,
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
            markdown, llm_public = cheatsheet_service.make_cheatsheet_ex(content)
            title = f"Cheat Sheet · {subject}"
            saved_id = db.save("cheatsheets", owner, {
                "title": title, "source": subject,
                "llm": llm_public, "markdown": markdown,
            })
            detail = None
            art_data = {"markdown": markdown}

        elif tool == "teachback":
            # A live Socratic session: pick the concept + opening prompt and open
            # the session. The transcript is saved to "teachbacks" only when the
            # student finishes (POST /api/teachback/message -> done), not here.
            if subject:
                content, _ = _subject_content(subject, owner)
                session_id, concept, opening, llm_public = teachback_service.start_session(
                    "subject", content=content, label=subject, owner=owner,
                )
                source = subject
            elif topic:
                session_id, concept, opening, llm_public = teachback_service.start_session(
                    "topic", topic=topic, label=f"Topic · {topic}", owner=owner,
                )
                source = topic
            else:
                return None
            saved_id = None
            title = f"Teach-back · {source}"
            detail = concept
            art_data = {"session_id": session_id, "concept": concept, "opening": opening}

        elif tool == "predictor":
            if not subject:
                raise _ToolFailure(
                    "predictions are built from past papers — start a Subject chat so I have "
                    "papers to analyze."
                )
            corpus, pdfs = _subject_papers_corpus(subject, owner)
            data = predictor_service.predict_questions(corpus, subject_hint=subject)
            llm_public = _take_llm(data)
            predicted = data.get("predicted_questions") or []
            patterns = data.get("patterns") or []
            title = f"Predictions · {subject}"
            saved_id = db.save("predictions", owner, {
                "title": title, "source": subject, "subject": subject,
                "llm": llm_public, "num_papers": len(pdfs),
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
            num_short = min(_short_count_from_text(message) or 2, 4)   # "with 3 short answers"
            difficulty = intent.get("difficulty") or "medium"
            content, _ = _subject_content(subject, owner)
            data = exam_service.generate_exam(content, num_mcq, num_short, difficulty=difficulty)
            llm_public = _take_llm(data)
            questions = data.get("questions") or []
            if not questions:
                raise _ToolFailure("the model returned no usable questions — try again.")
            grading.sign_items(questions)
            # An exam is persisted as a GRADED ATTEMPT only after the user submits
            # (POST /api/exam/grade), so nothing is saved here.
            saved_id = None
            title = f"Mock Exam · {subject}"
            detail = f"{num_mcq} MCQ + {num_short} short · {difficulty}"
            art_data = {
                "questions": questions, "subject": subject,
                "num_mcq": num_mcq, "num_short": num_short,
            }

    except _ToolFailure as exc:
        steps.append(_trace("Couldn't create it", str(exc)))
        return {
            "answer": f"I couldn't create that {tool_name.lower()} — {exc}",
            "steps": steps, "artifact": None, "recent_artifact": None, "llm": None,
        }
    except LLMChainError:
        raise
    except Exception as exc:  # noqa: BLE001 — a tool bug must not take down the chat
        log.warning("Chat tool orchestration failed (%s): %s", tool, exc, exc_info=True)   # details stay server-side
        safe = public_message(exc)
        steps.append(_trace("Couldn't create it", safe))
        return {
            "answer": f"I tried to create that {tool_name.lower()} but something went wrong. {safe}",
            "steps": steps, "artifact": None, "recent_artifact": None, "llm": None,
        }

    steps.append(_trace(f"Created “{title}”" + (f" — {detail}" if detail else "")))

    # Remember a compact rendition so later, non-tool turns can DISCUSS this
    # artifact (e.g. "explain question 3", "summarise this cheat sheet"). This
    # changes nothing shown in the chat — chat.ask() injects it into context.
    recent_artifact = None
    try:
        rendition = _artifact_text(tool, title, art_data)
        if rendition:
            recent_artifact = {"title": title, "kind": kind, "text": rendition}
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
                "tool": tool_name, "data": art_data, "llm": llm_public,
            },
            "recent_artifact": recent_artifact, "llm": llm_public,
        }
    if tool == "teachback":
        # A live session — saved to history only when the student finishes it.
        return {
            "answer": f"Let's do a teach-back on **{detail}**. Explain it to me below in "
                      "your own words, and I'll probe your understanding.",
            "steps": steps,
            "artifact": {
                "kind": kind, "id": None, "title": title, "route": route,
                "tool": tool_name, "data": art_data, "llm": llm_public,
            },
            "recent_artifact": recent_artifact, "llm": llm_public,
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
            "tool": tool_name, "data": art_data, "llm": llm_public,
        },
        "recent_artifact": recent_artifact, "llm": llm_public,
    }
