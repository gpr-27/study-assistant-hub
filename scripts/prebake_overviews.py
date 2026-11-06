#!/usr/bin/env python3
"""Pre-bake the NotebookLM-style session-start overview for each bundled subject.

Run locally (needs at least one provider key in .env — see backend/llm_chain.py):

    python scripts/prebake_overviews.py                 # only subjects WITHOUT an overview.json
    python scripts/prebake_overviews.py --force         # regenerate everything (costs quota)
    python scripts/prebake_overviews.py --only Mathematics --dry-run

For every bundled subject it makes ONE call through the provider chain (``llm_chain.chat`` with
``json_mode=True``) and writes ``data/index/<subject>/overview.json`` — the same directory that
ships the subject's pre-built retrieval index with the deployment. At runtime the overview
service loads that file instantly (no PDF read, no LLM call), so the overview card appears the
moment a session opens.

User-uploaded subjects have no committed overview and generate on first open instead.

After running, commit ``data/index/*/overview.json`` so it ships with the next deploy.
"""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

INDEX_DIR = ROOT / "data" / "index"
OVERVIEW_NAME = "overview.json"
MAX_PDFS = 3
MAX_CONTENT_TOKENS = 4000
MAX_QUESTIONS = 3

SYSTEM = (
    "You write the opening card of a study assistant. Reply with ONE JSON object only — "
    "no markdown fences, no commentary."
)


def overview_path(subject: str) -> Path:
    return INDEX_DIR / subject / OVERVIEW_NAME


def build_prompt(label: str, body: str) -> str:
    spec = (
        "Respond in JSON: return ONLY a single JSON object of exactly this form:\n"
        '{"overview": "<2-3 sentence markdown summary>", '
        '"questions": ["<question>", "<question>", "<question>"]}\n'
        "The overview is 2-3 sentences that say what the material covers; use **bold** markdown "
        "around 2-4 key terms. Then write exactly 3 specific, answerable starter questions a "
        "student could tap to begin. No extra keys."
    )
    return f"Study material for “{label}” is below.\n\n{spec}\n\nMATERIAL:\n{body}"


def parse_card(raw: str) -> dict:
    """Tolerantly extract ``{overview, questions}`` from the model reply."""
    match = re.search(r"\{.*\}", raw or "", re.DOTALL)
    data = json.loads(match.group(0)) if match else {}
    if not isinstance(data, dict):
        return {"overview": "", "questions": []}
    overview = str(data.get("overview", "") or "").strip()
    questions = [str(q).strip() for q in (data.get("questions") or []) if str(q or "").strip()]
    return {"overview": overview, "questions": questions[:MAX_QUESTIONS]}


def subject_body(subject: str) -> tuple[str, int]:
    """(clipped text of the first few PDFs, total PDF count) for a bundled subject."""
    from backend.documents import clip_to_tokens, list_pdf_names, read_pdf

    names = list_pdf_names(subject, None)
    parts: list[str] = []
    for pdf in names[:MAX_PDFS]:
        try:
            parts.append(f"## {pdf}\n{read_pdf(subject, pdf, None)}")
        except Exception:  # noqa: BLE001 — skip unreadable PDFs, keep the rest
            continue
    return clip_to_tokens("\n\n".join(parts), MAX_CONTENT_TOKENS), len(names)


def generate(subject: str) -> dict:
    """One chain call -> ``{overview, questions, sources, llm}``. Raises on failure/empty."""
    from backend import llm_chain

    body, n_pdfs = subject_body(subject)
    if not body.strip():
        raise RuntimeError("no readable PDF text")
    res = llm_chain.chat(
        SYSTEM, [{"role": "user", "content": build_prompt(subject, body)}],
        max_tokens=2048, temperature=0.3, json_mode=True,
    )
    card = parse_card(res.text)
    if not card["overview"] and not card["questions"]:
        raise RuntimeError("model returned an empty overview")
    return {**card, "sources": n_pdfs, "llm": res.label}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--force", action="store_true", help="regenerate even when overview.json exists")
    ap.add_argument("--only", action="append", default=[], help="limit to this subject (repeatable)")
    ap.add_argument("--dry-run", action="store_true", help="list what would be generated; make no LLM calls")
    args = ap.parse_args(argv)

    from backend.documents import list_subjects

    subjects = [s for s in list_subjects(None) if not args.only or s in args.only]
    todo = [s for s in subjects if args.force or not overview_path(s).is_file()]

    print("=" * 60)
    print(" Pre-baking session-start overviews")
    print(f" Subjects: {', '.join(subjects) or '(none)'}")
    print(f" Missing/selected: {', '.join(todo) or '(nothing to do)'}")
    print("=" * 60)
    if args.dry_run or not todo:
        return 0

    ok = err = 0
    for subject in todo:
        try:
            out = generate(subject)
            dest = overview_path(subject)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
            shown = dest.relative_to(ROOT) if dest.is_relative_to(ROOT) else dest
            ok += 1
            print(f"  ok   {subject}: {len(out['questions'])} questions -> {shown}  [{out['llm']}]")
        except Exception as exc:  # noqa: BLE001
            err += 1
            print(f"  FAIL {subject}: {type(exc).__name__}: {exc}")
    print(f"\nOverviews: {ok} baked, {err} failed")
    if ok:
        print("Next: commit data/index/*/overview.json so it ships with the deployment.")
    return 1 if err else 0


if __name__ == "__main__":
    raise SystemExit(main())
