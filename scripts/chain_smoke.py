"""Live smoke test for backend.llm_chain (spends a few real tokens; never prints keys).

    .venv/bin/python scripts/chain_smoke.py            # full chain + each step forced + per-model probes
    .venv/bin/python scripts/chain_smoke.py --no-probe # skip the per-model probes

Loads ``.env`` via python-dotenv. Each call sends ONE tiny prompt (max_tokens=64 for Anthropic;
the Groq/Gemini floors of 2048 apply but a one-word answer is only a few tokens).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import dotenv_values  # noqa: E402

for _k, _v in dotenv_values(ROOT / ".env").items():
    if _k.endswith("_API_KEY") or _k.endswith("_API_KEY_2"):
        if _v and not os.environ.get(_k):
            os.environ[_k] = _v

from backend import llm_chain as lc  # noqa: E402

SYSTEM = "You are a terse assistant."
MSGS = [{"role": "user", "content": "Reply with exactly one word: pong"}]


def _only_step(n: int) -> dict[str, str | None]:
    """Blank every key except step n's (in-process env override); returns the saved env."""
    saved = {s.env: os.environ.get(s.env) for s in lc.STEPS}
    for s in lc.STEPS:
        if s.n != n and s.env in os.environ:
            del os.environ[s.env]
    return saved


def _restore(saved: dict[str, str | None]) -> None:
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def _trunc(s: str, n: int = 200) -> str:
    return (s or "").replace("\n", " ")[:n]


def run_chain(label: str, **kw):
    lc.reset_state()
    try:
        r = lc.chat(SYSTEM, MSGS, max_tokens=64, **kw)
        print(f"[{label}] ANSWERED text={_trunc(r.text, 40)!r} usage={r.usage} {r.latency_ms}ms skipped={r.skipped}")
        print(f"[{label}] {r.label}")
        return r
    except lc.LLMChainError as e:
        print(f"[{label}] FAILED status={e.status} {_trunc(str(e))}")
        print(f"[{label}] {e.label}")
        return None


def probe(step: lc._Step, model: str) -> None:
    key = lc._key(step)
    if not key:
        print(f"  probe {step.short} {model}: no key")
        return
    req = lc._make_req(SYSTEM, MSGS, 64, 0.3, False, "low")
    run = lc._Run(False, 30)
    out = lc._attempt_sync(lc._get_sync_client(), run, step, model, key, req)
    state = "answered" if out.ok else "failed"
    print(f"  probe {step.short:9s} {model:26s} {state} status={out.status} "
          f"{_trunc(out.text, 30)!r}{_trunc(out.message)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-probe", action="store_true")
    args = ap.parse_args()

    print("configured steps:", lc.configured_steps())
    first = run_chain("full chain")
    for s in lc.STEPS:
        if s.n == 1 and first is not None and first.step == 1:
            print("[forced step 1 (Anthropic)] skipped: the full chain already answered at step 1 (avoids a 2nd paid call)")
            continue
        saved = _only_step(s.n)
        try:
            if lc._key(s):
                run_chain(f"forced step {s.n} ({s.name})")
            else:
                print(f"[forced step {s.n} ({s.name})] no key configured")
        finally:
            _restore(saved)
    if not args.no_probe:
        print("per-model probes (direct, one call each):")
        for s in lc.STEPS:
            if s.n in (2, 4):             # key-2 steps share models; step 1 is paid, covered above
                for m in s.models:
                    probe(s, m)
        # JSON mode sanity on Groq/Gemini
        for s in (lc.STEPS[1], lc.STEPS[3]):
            saved = _only_step(s.n)
            try:
                lc.reset_state()
                r = lc.chat("Return JSON.", [{"role": "user", "content": 'Give {"ok": true} as JSON.'}],
                            json_mode=True, max_tokens=64)
                print(f"json_mode {s.name}: {r.label} -> {_trunc(r.text, 60)!r}")
            except lc.LLMChainError as e:
                print(f"json_mode {s.name}: FAILED {_trunc(str(e))}")
            finally:
                _restore(saved)
        # Streaming sanity (Groq key 1 then Gemini key 1)
        for s in (lc.STEPS[1], lc.STEPS[3]):
            saved = _only_step(s.n)
            try:
                lc.reset_state()
                parts, final = [], None
                for item in lc.stream_chat(SYSTEM, MSGS, max_tokens=64):
                    if isinstance(item, lc.ChatResult):
                        final = item
                    elif isinstance(item, lc.StreamStart):
                        continue
                    else:
                        parts.append(item)
                print(f"stream {s.name}: deltas={len(parts)} {final.label if final else '?'} -> "
                      f"{_trunc(final.text if final else '', 40)!r}")
            except lc.LLMChainError as e:
                print(f"stream {s.name}: FAILED {_trunc(str(e))}")
            finally:
                _restore(saved)


if __name__ == "__main__":
    main()
