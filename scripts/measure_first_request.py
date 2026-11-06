"""Measure what a FRESH process pays on its first requests (the serverless cold-start experience).

For each run: spawn a brand-new ``uvicorn api.index:app`` (real MongoDB from .env, fake LLM providers via
``LLM_CHAIN_TEST_BASE_URL`` -- no paid calls), wait only for the TCP port to open (like Vercel, the first
request arrives right after import), then time, in order:
  guest      POST /api/guest             (first Mongo use)
  start      POST /api/chat/start        (Mathematics)
  connected  POST /api/chat/message/stream -> time to the first ``: connected`` line
  ttft       ... and to the first ``token`` event
Throwaway guests are deleted afterwards (by id). Usage:
  python scripts/measure_first_request.py [--runs 5] [--no-prewarm]
Never prints env values.
"""
from __future__ import annotations

import argparse
import os
import signal
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
API, FAKE = 8931, 8932


def _wait_port(port: int, timeout: float = 60.0) -> None:
    end = time.time() + timeout
    while time.time() < end:
        with socket.socket() as s:
            s.settimeout(0.2)
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.02)
    raise SystemExit(f"port {port} never opened")


def _kill(p: subprocess.Popen) -> None:
    if p.poll() is None:
        os.killpg(p.pid, signal.SIGTERM)
        try:
            p.wait(5)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)


def one_run(no_prewarm: bool) -> tuple[dict, str | None]:
    env = dict(os.environ, RATE_LIMIT_DISABLED="1", LOG_LEVEL="WARNING",
               LLM_CHAIN_TEST_BASE_URL=f"http://127.0.0.1:{FAKE}")
    env.pop("VERCEL", None)
    if no_prewarm:
        env["PREWARM_DISABLED"] = "1"
    devnull = open(os.devnull, "w")
    p = subprocess.Popen([sys.executable, "-m", "uvicorn", "api.index:app", "--port", str(API), "--log-level", "warning",
                          "--no-access-log"], cwd=ROOT, env=env, stdout=devnull, stderr=devnull, start_new_session=True)
    out: dict = {}
    gid = None
    try:
        _wait_port(API)
        base = f"http://127.0.0.1:{API}"
        with httpx.Client(base_url=base, timeout=30) as c:
            t = time.perf_counter()
            r = c.post("/api/guest")
            out["guest"] = (time.perf_counter() - t) * 1000
            gid = r.json().get("guestId") if r.status_code == 200 else None
            if not gid:
                raise SystemExit(f"guest create failed: HTTP {r.status_code}")
            H = {"X-Guest-Id": gid}
            t = time.perf_counter()
            r = c.post("/api/chat/start", headers=H, json={"mode": "subject", "subject": "Mathematics"})
            out["start"] = (time.perf_counter() - t) * 1000
            sid = r.json()["session_id"]
            t = time.perf_counter()
            with c.stream("POST", "/api/chat/message/stream", headers=H,
                          json={"session_id": sid, "message": "What is a matrix?"}) as r:
                for line in r.iter_lines():
                    if line.startswith(": connected") and "connected" not in out:
                        out["connected"] = (time.perf_counter() - t) * 1000
                    elif line.startswith("data:") and '"token"' in line:
                        out["ttft"] = (time.perf_counter() - t) * 1000
                        break
    finally:
        _kill(p)
    return out, gid


def cleanup(ids: list[str]) -> None:
    from backend import db

    d = db.get_db()
    for coll in (*db.KINDS, *db.EPHEMERAL_TTLS):
        d[coll].delete_many({"owner": {"$in": ids}})
    d[db.GUESTS].delete_many({"guestId": {"$in": ids}})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--no-prewarm", action="store_true")
    a = ap.parse_args()
    fake = subprocess.Popen([sys.executable, "scripts/fake_providers.py", "--port", str(FAKE)], cwd=ROOT,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    ids: list[str] = []
    rows: list[dict] = []
    try:
        _wait_port(FAKE)
        for _ in range(a.runs):
            row, gid = one_run(a.no_prewarm)
            rows.append(row)
            if gid:
                ids.append(gid)
            print({k: round(v) for k, v in row.items()})
    finally:
        _kill(fake)
        if ids:
            cleanup(ids)
    print("median ms:", {k: round(statistics.median(r[k] for r in rows if k in r)) for k in rows[0]})


if __name__ == "__main__":
    main()
