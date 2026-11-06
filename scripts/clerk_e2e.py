"""End-to-end check of the REAL Clerk sign-in path (backend verifies genuine Clerk session JWTs).

Throwaway-but-kept verification script. It:
  1. creates two clearly-marked Clerk test users (Backend API, CLERK_SECRET_KEY from the env),
  2. creates a session for each and mints real session JWTs,
  3. boots the backend locally on PORT (default 8851) as a child process and exercises auth,
     owner isolation, forged/expired/tampered tokens, guest-header precedence, Server-Timing,
  4. checks JWKS caching / outage behaviour in-process (no secrets involved),
  5. ALWAYS cleans up in ``finally``: deletes both Clerk users, deletes Mongo docs whose
     ``owner`` is one of the two test user ids (or the throwaway guest created here), stops
     ONLY the server processes this script started.

Secrets are never printed (only statuses/lengths). Needs network + .env with a Clerk DEV
instance (pk_test_/sk_test_). One real LLM call is made unless --no-llm is passed.

    ./venv/bin/python scripts/clerk_e2e.py [--port 8851] [--no-llm] [--skip-expiry]
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import secrets
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

CLERK_API = "https://api.clerk.com/v1"
RESULTS: list[tuple[bool, str]] = []


def check(ok: bool, label: str, detail: str = "") -> bool:
    RESULTS.append((ok, label))
    print(f"[{'PASS' if ok else 'FAIL'}] {label}" + (f"  ({detail})" if detail else ""), flush=True)
    return ok


def b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def b64url_json(obj: dict) -> str:
    return b64url(json.dumps(obj, separators=(",", ":")).encode())


def unb64(seg: str) -> bytes:
    return base64.urlsafe_b64decode(seg + "=" * (-len(seg) % 4))


class Clerk:
    def __init__(self) -> None:
        sk = os.environ.get("CLERK_SECRET_KEY") or ""
        if not sk.startswith("sk_test_"):
            raise SystemExit("CLERK_SECRET_KEY must be a development key (sk_test_...)")
        self.http = httpx.Client(base_url=CLERK_API, headers={"Authorization": f"Bearer {sk}"}, timeout=20)

    def create_user(self, tag: str) -> dict:
        rnd = secrets.token_hex(4)
        r = self.http.post("/users", json={
            "email_address": [f"sah-e2e-{tag}-{rnd}@example.com"], "username": f"sah_e2e_{tag.lower()}_{rnd}",
            "first_name": "SAH-E2E-TEST", "last_name": f"DELETE-ME-{tag}",
            "password": secrets.token_urlsafe(24) + "aA1!", "skip_password_checks": True,
            "skip_password_requirement": True,
        })
        r.raise_for_status()
        return r.json()

    def session_token(self, user_id: str, expires_in: int | None = None) -> tuple[str, str]:
        r = self.http.post("/sessions", json={"user_id": user_id})
        r.raise_for_status()
        sid = r.json()["id"]
        body = {"expires_in_seconds": expires_in} if expires_in else {}
        r = self.http.post(f"/sessions/{sid}/tokens", json=body)
        r.raise_for_status()
        return sid, r.json()["jwt"]

    def delete_user(self, user_id: str) -> int:
        return self.http.delete(f"/users/{user_id}").status_code


def start_server(port: int, extra_env: dict | None = None) -> subprocess.Popen:
    env = {**os.environ, "PORT": str(port), "APP_ENV": "production", "LOG_LEVEL": "WARNING"}
    env.update(extra_env or {})
    log = open(f"/tmp/clerk_e2e_server_{port}.log", "w")
    proc = subprocess.Popen([sys.executable, str(ROOT / "run.py")], cwd=ROOT, env=env,
                            stdout=log, stderr=subprocess.STDOUT)
    base = f"http://127.0.0.1:{port}"
    for _ in range(120):
        try:
            if httpx.get(base + "/api/config", timeout=2).status_code == 200:
                return proc
        except httpx.HTTPError:
            pass
        if proc.poll() is not None:
            raise SystemExit(f"server on {port} exited early (see /tmp/clerk_e2e_server_{port}.log)")
        time.sleep(1)
    raise SystemExit("server did not start in time")


def stop_server(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(10)
    except subprocess.TimeoutExpired:
        proc.kill()


def server_timing(resp: httpx.Response, name: str = "auth") -> float | None:
    for part in resp.headers.get("server-timing", "").split(","):
        part = part.strip()
        if part.startswith(name + ";dur="):
            return float(part.split("=", 1)[1])
    return None


def hdr(token: str | None = None, guest: str | None = None) -> dict:
    h = {}
    if token:
        h["Authorization"] = "Bearer " + token
    if guest:
        h["X-Guest-Id"] = guest
    return h


def inprocess_jwks_checks(token: str) -> None:
    """JWKS cache / outage behaviour using backend.auth directly (counts real network fetches)."""
    import jwt

    from backend import auth, config

    auth._jwks_client.cache_clear()
    auth._token_cache.clear()
    auth._jwks_failed_until = 0.0
    fetches = {"n": 0}
    orig = jwt.PyJWKClient.fetch_data

    def counting(self):
        fetches["n"] += 1
        return orig(self)

    jwt.PyJWKClient.fetch_data = counting
    try:
        t = time.perf_counter()
        sub = auth._verify_clerk_token(token)
        cold = (time.perf_counter() - t) * 1000
        n_cold = fetches["n"]
        t = time.perf_counter()
        sub2 = auth._verify_clerk_token(token)
        warm_token_cache = (time.perf_counter() - t) * 1000
        auth._token_cache.clear()  # force a real RSA verify, JWKS key cache still warm
        t = time.perf_counter()
        sub3 = auth._verify_clerk_token(token)
        warm_verify = (time.perf_counter() - t) * 1000
        check(bool(sub) and sub == sub2 == sub3, "in-process verify returns Clerk sub (cold/warm/rsa-only)")
        check(n_cold == 1 and fetches["n"] == 1, "JWKS fetched exactly once; later verifies make no network call",
              f"fetches={fetches['n']}")
        print(f"        auth cold={cold:.1f}ms  warm(token-cache)={warm_token_cache:.3f}ms  "
              f"warm(rsa verify, key cached)={warm_verify:.2f}ms", flush=True)

        # Junk tokens with random kids must not each cost a network fetch to Clerk.
        from cryptography.hazmat.primitives.asymmetric import rsa
        junk_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        auth._jwks_last_forced_refresh = None
        before = fetches["n"]
        res = []
        for i in range(8):
            junk = jwt.encode({"sub": "user_x", "exp": int(time.time()) + 60, "iss": auth._issuer()}, junk_key,
                              "RS256", headers={"kid": f"ins_junk{i}"})
            res.append(auth._verify_clerk_token(junk))
        check(all(x is None for x in res) and fetches["n"] - before <= 1,
              "8 tokens with unknown random kids -> all rejected, at most 1 JWKS refetch", f"refetches={fetches['n']-before}")

        # Outage: point a fresh client at an unreachable host (blackhole IP => exercises the timeout).
        host = "10.255.255.1"
        dead_pk = "pk_test_" + base64.b64encode(f"{host}$".encode()).decode().rstrip("=")
        auth._jwks_client.cache_clear()
        auth._token_cache.clear()
        auth._jwks_failed_until = 0.0
        orig_pk = config.CLERK_PUBLISHABLE_KEY
        config.CLERK_PUBLISHABLE_KEY = dead_pk
        try:
            def attempt():
                try:
                    return auth._verify_clerk_token(token)
                except auth.ClerkUnavailable:
                    return "UNAVAILABLE"
            t = time.perf_counter()
            r1 = attempt()
            d1 = time.perf_counter() - t
            t = time.perf_counter()
            r2 = attempt()
            d2 = time.perf_counter() - t
        finally:
            config.CLERK_PUBLISHABLE_KEY = orig_pk
            auth._jwks_client.cache_clear()
            auth._jwks_failed_until = 0.0
            auth._token_cache.clear()
        check(r1 == "UNAVAILABLE" and r2 == "UNAVAILABLE" and d1 < 6 and d2 < 0.1,
              "JWKS outage -> ClerkUnavailable (503) bounded by timeout, then fail-fast backoff (no hang)",
              f"first={d1:.2f}s second={d2*1000:.1f}ms")
    finally:
        jwt.PyJWKClient.fetch_data = orig


def cleanup_mongo(owner_ids: list[str], guest_ids: list[str]) -> dict:
    """Targeted delete: ONLY docs whose owner is one of our throwaway ids (or those guest records)."""
    from backend import db

    out: dict = {}
    database = db.get_db()
    if database is None:
        return {"persistence": "off"}
    owners = [o for o in owner_ids + guest_ids if o]
    assert all(o.startswith(("user_", "guest_")) for o in owners), "refusing to clean unexpected owner"
    names = list(db.KINDS) + list(db.EPHEMERAL_TTLS) + ["fs.files", "uploads.files", "indexes.files"]
    for coll in names:
        try:
            n = database[coll].delete_many({"owner": {"$in": owners}}).deleted_count
            if n:
                out[coll] = n
        except Exception as exc:  # noqa: BLE001
            out[coll] = f"err {type(exc).__name__}"
    if guest_ids:
        out["guests"] = database[db.GUESTS].delete_many({"guestId": {"$in": guest_ids}}).deleted_count
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8851)
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--skip-expiry", action="store_true")
    args = ap.parse_args()
    base = f"http://127.0.0.1:{args.port}"

    clerk = Clerk()
    users: list[dict] = []
    guest_ids: list[str] = []
    procs: list[subprocess.Popen] = []
    try:
        ua, ub = clerk.create_user("A"), clerk.create_user("B")
        users = [ua, ub]
        check(ua["id"].startswith("user_") and ub["id"] != ua["id"], "created 2 Clerk test users")
        sid_a, tok_a = clerk.session_token(ua["id"])
        sid_b, tok_b = clerk.session_token(ub["id"])
        # a very short-lived token for the expiry test (falls back to waiting if Clerk ignores it)
        try:
            _, tok_short = clerk.session_token(ua["id"], expires_in=30)  # Clerk's minimum custom lifetime
        except httpx.HTTPError:
            tok_short = tok_a
        h, p = (json.loads(unb64(s)) for s in tok_a.split(".")[:2])
        print(f"        token header alg={h.get('alg')} kid={'yes' if h.get('kid') else 'no'}; "
              f"claims={sorted(p)}; lifetime={p['exp']-p['iat']}s; "
              f"iss_host_matches_pk={p.get('iss','').startswith('https://')}", flush=True)
        minted_at = time.time()

        procs.append(start_server(args.port))
        c = httpx.Client(base_url=base, timeout=120)

        # ---- config + auth enabled
        r = c.get("/api/config")
        check(r.status_code == 200 and r.json().get("authEnabled") is True, "GET /api/config 200, authEnabled", f"{r.status_code}")

        # ---- protected routes
        r = c.get("/api/subjects")
        check(r.status_code == 401, "GET /api/subjects without token -> 401", str(r.status_code))
        r = c.get("/api/subjects", headers=hdr(tok_a))
        body = r.json() if r.status_code == 200 else {}
        check(r.status_code == 200, "GET /api/subjects with Clerk JWT -> 200", f"{r.status_code} keys={sorted(body)[:6]}")
        print(f"        Server-Timing auth on the server's FIRST ever token (JWKS fetch + RSA verify): {server_timing(r)}ms", flush=True)
        r = c.get("/api/subjects", headers=hdr(tok_a))
        print(f"        ...same token again (token cache): auth={server_timing(r)} (absent = cache hit, ~0ms)", flush=True)
        r = c.get("/api/progress/summary", headers=hdr(tok_a))
        check(r.status_code == 200, "GET /api/progress/summary with Clerk JWT -> 200", str(r.status_code))
        r = c.get("/api/progress/summary")
        check(r.status_code == 401, "GET /api/progress/summary without token -> 401", str(r.status_code))
        r = c.get("/api/history/chats")
        check(r.status_code == 401, "GET /api/history/chats without token -> 401", str(r.status_code))

        # ---- bad tokens -> 401 (never 500)
        hh, pp, sig = tok_a.split(".")
        forged_payload = dict(json.loads(unb64(pp)), sub=ub["id"])
        tampered = f"{hh}.{b64url_json(forged_payload)}.{sig}"
        alg_none = f"{b64url_json({'alg': 'none', 'typ': 'JWT'})}.{pp}."
        hs = f"{b64url_json({'alg': 'HS256', 'typ': 'JWT', 'kid': h.get('kid')})}.{pp}.{b64url(b'x' * 32)}"
        bad = {
            "tampered payload (sub swapped to user B)": tampered,
            "alg=none": alg_none,
            "HS256 confusion": hs,
            "garbage": "not.a.jwt",
            "truncated signature": f"{hh}.{pp}.{sig[:10]}",
        }
        try:
            from cryptography.hazmat.primitives.asymmetric import rsa
            import jwt
            k = rsa.generate_private_key(public_exponent=65537, key_size=2048)
            claims = dict(json.loads(unb64(pp)))
            claims["iss"] = "https://evil.example.com"
            bad["locally signed (wrong key + issuer), real kid"] = jwt.encode(claims, k, "RS256", headers={"kid": h.get("kid")})
        except Exception as exc:  # noqa: BLE001
            print("        (skipped locally-signed token:", type(exc).__name__, ")")
        for name, t in bad.items():
            r = c.get("/api/subjects", headers=hdr(t))
            check(r.status_code == 401, f"bad token [{name}] -> 401", str(r.status_code))

        # ---- guest header cannot impersonate a Clerk user / escalate
        r = c.get("/api/history/chats", headers=hdr(guest=ua["id"]))
        check(r.status_code == 401, "X-Guest-Id set to a Clerk user id, no token -> 401", str(r.status_code))
        r = c.get("/api/history/chats", headers=hdr(guest="guest_" + "0" * 32))
        check(r.status_code == 401, "unknown X-Guest-Id, no token -> 401", str(r.status_code))
        r = c.post("/api/guest")
        guest = r.json() if r.status_code == 200 else {}
        gid = guest.get("guestId") or guest.get("guest_id") or guest.get("id")
        if gid:
            guest_ids.append(gid)
        check(bool(gid), "POST /api/guest creates a throwaway guest", str(r.status_code))
        r = c.post("/api/guest/migrate", headers=hdr(guest=gid))
        check(r.status_code == 401, "guest id alone cannot call /api/guest/migrate (Clerk-only) -> 401", str(r.status_code))

        # ---- chat as user A
        r = c.post("/api/chat/start", json={"mode": "subject", "subject": "Mathematics"}, headers=hdr(tok_a))
        chat = r.json() if r.status_code == 200 else {}
        check(r.status_code == 200 and chat.get("session_id"), "POST /api/chat/start (Mathematics) as A -> 200", str(r.status_code))
        sess_a, chat_id = chat.get("session_id"), chat.get("chat_id")

        if not args.no_llm and sess_a:
            t0 = time.time()
            with c.stream("POST", "/api/chat/message/stream", headers=hdr(tok_a), json={
                    "session_id": sess_a, "message": "What is 2+2? Reply with just the number."}) as rs:
                status = rs.status_code
                events = []
                for line in rs.iter_lines():
                    if line.startswith("data: "):
                        try:
                            events.append(json.loads(line[6:]))
                        except ValueError:
                            pass
            types = sorted({e.get("type") or e.get("event") or "?" for e in events})
            answer = "".join(str(e.get("text") or e.get("content") or e.get("token") or "") for e in events
                             if (e.get("type") or e.get("event")) in ("token", "answer", "delta", "done"))
            check(status == 200 and len(events) > 0, "POST /api/chat/message/stream as A -> 200 SSE",
                  f"{status} events={len(events)} types={types} {time.time()-t0:.1f}s answer_len={len(answer)}")

        # ---- ownership + isolation
        r = c.get("/api/history/chats", headers=hdr(tok_a))
        items = r.json() if r.status_code == 200 else []
        items = items.get("items", items) if isinstance(items, dict) else items
        ids_a = [i.get("id") for i in items]
        check(r.status_code == 200 and chat_id in ids_a, "A sees own chat in history", f"{r.status_code} n={len(ids_a)}")
        r = c.get("/api/history/chats", headers=hdr(tok_b))
        items_b = r.json() if r.status_code == 200 else []
        items_b = items_b.get("items", items_b) if isinstance(items_b, dict) else items_b
        check(r.status_code == 200 and chat_id not in [i.get("id") for i in items_b], "B does NOT see A's chat in history", f"{r.status_code}")
        from backend import db
        database = db.get_db()
        if database is not None and chat_id:
            doc = database["chats"].find_one({"_id": db.oid(chat_id)}, {"owner": 1})
            check(bool(doc) and doc.get("owner") == ua["id"], "stored chat owner == Clerk user id (not guest/local)")
        r = c.get(f"/api/history/chats/{chat_id}", headers=hdr(tok_b))
        check(r.status_code in (403, 404), "B cannot GET A's chat by id", str(r.status_code))
        r = c.delete(f"/api/history/chats/{chat_id}", headers=hdr(tok_b))
        check(r.status_code in (403, 404) or (r.status_code == 200 and not r.json().get("deleted", True)),
              "B cannot DELETE A's chat", f"{r.status_code} {str(r.json())[:60]}")
        r = c.post("/api/chat/resume", json={"chat_id": chat_id}, headers=hdr(tok_b))
        check(r.status_code in (403, 404), "B cannot resume A's chat", str(r.status_code))
        r = c.post("/api/chat/message/stream", json={"session_id": sess_a, "message": "hi"}, headers=hdr(tok_b))
        check(r.status_code in (403, 404), "B cannot use A's chat session token (stream)", str(r.status_code))
        r = c.post("/api/chat/message", json={"session_id": sess_a, "message": "hi"}, headers=hdr(tok_b))
        check(r.status_code in (403, 404), "B cannot use A's chat session token (message)", str(r.status_code))
        r = c.get(f"/api/chat/overview/{sess_a}", headers=hdr(tok_b))
        check(r.status_code in (403, 404), "B cannot read A's session overview", str(r.status_code))
        # guest (valid guest id) can't see A's data; A's token + guest header stays A
        r = c.get("/api/history/chats", headers=hdr(guest=gid))
        gl = r.json() if r.status_code == 200 else []
        gl = gl.get("items", gl) if isinstance(gl, dict) else gl
        check(r.status_code == 200 and chat_id not in [i.get("id") for i in gl], "a live guest does NOT see A's chat", str(r.status_code))
        r = c.get("/api/history/chats", headers=hdr(tok_a, guest=gid))
        la = r.json() if r.status_code == 200 else []
        la = la.get("items", la) if isinstance(la, dict) else la
        check(r.status_code == 200 and chat_id in [i.get("id") for i in la],
              "valid Clerk JWT + guest header -> still owner A (guest cannot override)", str(r.status_code))
        r = c.get("/api/history/chats", headers=hdr(tok_b, guest=gid))
        lb = r.json() if r.status_code == 200 else []
        lb = lb.get("items", lb) if isinstance(lb, dict) else lb
        check(r.status_code == 200 and chat_id not in [i.get("id") for i in lb], "B's JWT + guest header -> owner B", str(r.status_code))

        # ---- Server-Timing auth: cold (first sight of a token) vs warm (token cache)
        _, tok_c = clerk.session_token(ua["id"])
        r1 = c.get("/api/progress/summary", headers=hdr(tok_c))
        r2 = c.get("/api/progress/summary", headers=hdr(tok_c))
        r3 = c.get("/api/progress/summary", headers=hdr(tok_c))
        print(f"        Server-Timing auth: first-sight={server_timing(r1)}ms  warm={server_timing(r2)}ms/{server_timing(r3)}ms"
              f"  (status {r1.status_code}/{r2.status_code}/{r3.status_code})", flush=True)
        check(r1.status_code == r2.status_code == 200, "same token reused -> 200 each time")

        # ---- JWKS cache / outage (in-process)
        inprocess_jwks_checks(tok_c)

        # ---- server whose JWKS host is unreachable: valid-looking token -> clean 401 fast, no hang
        dead_pk = "pk_test_" + base64.b64encode(b"127.0.0.1:1$").decode().rstrip("=")
        port2 = args.port + 1
        procs.append(start_server(port2, {"CLERK_PUBLISHABLE_KEY": dead_pk}))
        c2 = httpx.Client(base_url=f"http://127.0.0.1:{port2}", timeout=30)
        t = time.time()
        r = c2.get("/api/subjects", headers=hdr(tok_a))
        d = time.time() - t
        check(r.status_code == 503 and d < 6, "live server with unreachable JWKS + bearer -> clean retryable 503, no hang", f"{r.status_code} {d:.2f}s")
        t = time.time()
        r = c2.get("/api/subjects", headers=hdr(tok_a))
        check(r.status_code == 503 and time.time() - t < 1, "second request fails fast (backoff)", f"{r.status_code} {time.time()-t:.3f}s")
        stop_server(procs.pop())

        # ---- expired token (Clerk ignores/honours expires_in; else wait out 60s lifetime + 10s leeway)
        if not args.skip_expiry:
            exp = json.loads(unb64(tok_short.split(".")[1]))["exp"]
            wait = max(0.0, exp + 11 - time.time())
            print(f"        waiting {wait:.0f}s for a real Clerk token to expire (leeway 10s)...", flush=True)
            time.sleep(wait)
            r = c.get("/api/subjects", headers=hdr(tok_short))
            check(r.status_code == 401, "genuinely expired Clerk token -> 401", str(r.status_code))
        # a fresh token still works after all the above (no poisoned state)
        _, tok_fresh = clerk.session_token(ub["id"])
        r = c.get("/api/subjects", headers=hdr(tok_fresh))
        check(r.status_code == 200, "fresh token still 200 after failures", str(r.status_code))
    finally:
        print("--- cleanup", flush=True)
        for pr in procs:
            stop_server(pr)
        try:
            mongo = cleanup_mongo([u["id"] for u in users], guest_ids)
            print("mongo cleanup:", mongo, flush=True)
        except Exception as exc:  # noqa: BLE001
            print("mongo cleanup FAILED:", type(exc).__name__, exc, flush=True)
        for u in users:
            print("delete clerk user", u["id"], "->", clerk.delete_user(u["id"]), flush=True)

    bad = [label for ok, label in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(bad)}/{len(RESULTS)} checks passed")
    for label in bad:
        print("  FAILED:", label)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
