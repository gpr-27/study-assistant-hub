"""Cold-start benchmark against a (fresh) deployment: first function hit, signed-in bootstrap, guest create.

    ./venv/bin/python scripts/cold_bench.py https://study-assistant-hub.vercel.app [--idle-note label]

Creates ONE throwaway Clerk DEV user (CLERK_SECRET_KEY from .env), mints a JWT, and ALWAYS deletes the
user + any guest created. Prints wall time and the full Server-Timing header per step. No secrets printed.
Steps run in this order so each shows what a *cold* instance pays: config (first function hit) ->
bootstrap(signed-in, first auth) -> guest create -> guest bootstrap -> bootstrap again (warm reference).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
import clerk_e2e as ce  # noqa: E402


def hit(c: httpx.Client, label: str, method: str, path: str, **kw):
    t0 = time.perf_counter()
    r = c.request(method, path, **kw)
    ms = (time.perf_counter() - t0) * 1000
    st = r.headers.get("server-timing", "-")
    print(f"{label:<22} {r.status_code} wall={ms:7.0f}ms age={r.headers.get('x-instance-age', '-'):>6}  {st}", flush=True)
    return r


def main() -> None:
    base = (sys.argv[1] if len(sys.argv) > 1 else "https://study-assistant-hub.vercel.app").rstrip("/")
    clerk = ce.Clerk()
    user = None
    gid = None
    try:
        user = clerk.create_user("COLD")
        _, tok = clerk.session_token(user["id"])
        with httpx.Client(base_url=base, timeout=30, headers={"Cache-Control": "no-cache"}) as c:
            hit(c, "GET /api/config", "GET", "/api/config")                       # first function hit
            hit(c, "GET /api/bootstrap(auth)", "GET", "/api/bootstrap", headers=ce.hdr(tok))
            r = hit(c, "POST /api/guest", "POST", "/api/guest")
            gid = (r.json() or {}).get("guestId") if r.status_code == 200 else None
            if gid:
                hit(c, "GET /api/bootstrap(guest)", "GET", "/api/bootstrap", headers=ce.hdr(guest=gid))
            hit(c, "GET /api/bootstrap(auth)#2", "GET", "/api/bootstrap", headers=ce.hdr(tok))
            hit(c, "GET /api/health/init", "GET", "/api/health/init")
    finally:
        if user:
            print("clerk user deleted:", clerk.delete_user(user["id"]))
        if gid:
            try:
                print("mongo cleanup:", ce.cleanup_mongo([user["id"]] if user else [], [gid]))
            except Exception as exc:  # noqa: BLE001
                print("mongo cleanup failed:", type(exc).__name__)


if __name__ == "__main__":
    main()
