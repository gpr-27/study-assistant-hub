"""Load test for the study-assistant API (no real LLM spend).

Orchestrates everything itself, so a run is reproducible:

  1. starts ``scripts/fake_providers.py`` (fake Anthropic / Groq / Gemini incl. embeddings),
  2. starts ``uvicorn api.index:app`` (N workers) with ``LLM_CHAIN_TEST_BASE_URL`` pointing at the fake
     (the real MongoDB from .env is used; the real LLM hosts are never contacted),
  3. provisions throwaway guests ``guest_loadtest001 ...`` (<= 250) directly in MongoDB,
  4. runs the scenarios, sampling server RSS once per second and probing ``/api/health`` at 10 Hz,
  5. deletes EVERYTHING it created, keyed by the ``guest_loadtest`` owner prefix (idempotent; also
     runs from ``--cleanup-only`` and on Ctrl-C / crash).

Scenarios (``--scenarios a,b,c,d,e,g``, default a..e):
  a  static / health / config / subjects at high concurrency (closed loop, ``--a-conc`` workers)
  b  ``--users`` concurrent users, each with its own guest id: subjects, history, open chat, stream 3
     messages with think-time, quiz, flashcards, history, progress  (1 uvicorn worker)
  c  as b, with fault injection: Anthropic 50% HTTP 529, Groq key(s) 30% HTTP 429 -> fall-through
  d  as b, with ``--workers`` uvicorn workers (mimics parallel serverless instances)
  e  upload path: small PDFs through POST /api/subjects (+ delete), concurrently
  g  chunked upload path (POST /uploads, PUT chunks, POST complete, continue-index loop), ``--g-users``

Usage:
  python scripts/loadtest.py --scenarios a,b --users 200 --json out.json
  python scripts/loadtest.py --cleanup-only

Only needs httpx (+ pymongo via backend.db for the guest provisioning/cleanup). Never prints env values.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import signal
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PREFIX = "guest_loadtest"                  # conforms to guest_[A-Za-z0-9]{1,48}
MAX_GUESTS = 250
FAKE_PORT, API_PORT = int(os.environ.get("LT_FAKE_PORT", "8912")), int(os.environ.get("LT_API_PORT", "8911"))
PDF = ROOT / "documents" / "Mathematics" / "Lec-1.pdf"      # ~1.5 MB, bundled

CHAT_MESSAGES = [
    "What is a matrix and what are its elements?",
    "Explain the order of a matrix with an example.",
    "How do I tell whether two matrices can be multiplied?",
    "What is the difference between a row and a column matrix?",
    "Summarise the main idea of the lecture in two sentences.",
]

procs: list[subprocess.Popen] = []


# ------------------------------------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------------------------------------
class Metrics:
    def __init__(self) -> None:
        self.lat: dict[str, list[float]] = defaultdict(list)
        self.status: dict[str, Counter] = defaultdict(Counter)
        self.ttft: list[float] = []
        self.labels: Counter = Counter()
        self.t0 = time.perf_counter()
        self.t1 = self.t0
        self.stream_total: list[float] = []
        self.stream_bad = 0
        self.srv: dict[str, list[float]] = defaultdict(list)   # server-side ms from Server-Timing "total"

    def add(self, name: str, status, dt: float) -> None:
        self.lat[name].append(dt * 1000.0)
        self.status[name][str(status)] += 1

    def done(self) -> None:
        self.t1 = time.perf_counter()


def _srv_ms(r) -> float | None:
    """Server-side total (ms) from the ``Server-Timing: total;dur=...`` header the app sends."""
    try:
        for part in r.headers.get("server-timing", "").split(","):
            if part.strip().startswith("total;dur="):
                return float(part.strip().split("=", 1)[1])
    except Exception:  # noqa: BLE001
        pass
    return None


def pct(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    return s[min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1))))]


def summarize(m: Metrics) -> dict:
    wall = max(1e-9, m.t1 - m.t0)
    out: dict = {"wall_s": round(wall, 1), "endpoints": {}}
    total = errors5 = errors_other = 0
    for name, vals in sorted(m.lat.items()):
        c = m.status[name]
        n = len(vals)
        total += n
        e5 = sum(v for k, v in c.items() if k.startswith("5"))
        eo = sum(v for k, v in c.items() if not k.startswith(("2", "3", "5")))
        errors5 += e5
        errors_other += eo
        out["endpoints"][name] = {
            "n": n, "rps": round(n / wall, 1), "p50": round(pct(vals, 50)), "p95": round(pct(vals, 95)),
            "p99": round(pct(vals, 99)), "max": round(max(vals)), "status": dict(c),
        }
    out["total_requests"], out["rps"] = total, round(total / wall, 1)
    out["http_5xx"], out["other_errors"] = errors5, errors_other
    if m.ttft:
        out["ttft_ms"] = {"n": len(m.ttft), "p50": round(pct(m.ttft, 50)), "p95": round(pct(m.ttft, 95)),
                          "p99": round(pct(m.ttft, 99))}
        out["stream_total_ms"] = {"p50": round(pct(m.stream_total, 50)), "p95": round(pct(m.stream_total, 95)),
                                  "p99": round(pct(m.stream_total, 99))}
    out["stream_incomplete"] = m.stream_bad
    out["labels"] = dict(m.labels)
    out["server_ms"] = {k: {"p50": round(pct(v, 50)), "p95": round(pct(v, 95)), "p99": round(pct(v, 99)), "n": len(v)}
                        for k, v in sorted(m.srv.items())}
    return out


def print_summary(title: str, s: dict, extra: dict | None = None) -> None:
    print(f"\n=== {title}  (wall {s['wall_s']} s, {s['total_requests']} req, {s['rps']} rps, "
          f"5xx={s['http_5xx']}, other-errors={s['other_errors']})")
    print(f"{'endpoint':28s} {'n':>6s} {'rps':>7s} {'p50':>7s} {'p95':>7s} {'p99':>7s} {'max':>7s}  status")
    for name, e in s["endpoints"].items():
        print(f"{name:28s} {e['n']:6d} {e['rps']:7.1f} {e['p50']:7d} {e['p95']:7d} {e['p99']:7d} {e['max']:7d}  {e['status']}")
    if "ttft_ms" in s:
        t, st = s["ttft_ms"], s["stream_total_ms"]
        print(f"stream TTFT ms  p50={t['p50']} p95={t['p95']} p99={t['p99']} (n={t['n']}); "
              f"total ms p50={st['p50']} p95={st['p95']} p99={st['p99']}; incomplete={s['stream_incomplete']}")
    if s["labels"]:
        print("answer labels:", s["labels"])
    for k, v in s.get("server_ms", {}).items():
        print(f"server-side ms {k:<30} p50={v['p50']} p95={v['p95']} p99={v['p99']} (n={v['n']})")
    for k, v in (extra or {}).items():
        print(f"{k}: {v}")
    if EXC_SAMPLES:
        print("client exceptions (text: count):", dict(EXC_SAMPLES.most_common(5)))
        EXC_SAMPLES.clear()


# ------------------------------------------------------------------------------------------------
# Processes
# ------------------------------------------------------------------------------------------------
def _spawn(cmd: list[str], env: dict, log: Path) -> subprocess.Popen:
    p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=open(log, "w"), stderr=subprocess.STDOUT,
                         start_new_session=True)
    procs.append(p)
    return p


def _kill(p: subprocess.Popen) -> None:
    """Terminate a process WE started (and its children), nothing else."""
    if p.poll() is None:
        try:
            os.killpg(p.pid, signal.SIGTERM)
        except Exception:  # noqa: BLE001
            p.terminate()
        try:
            p.wait(8)
        except Exception:  # noqa: BLE001
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except Exception:  # noqa: BLE001
                p.kill()
    if p in procs:
        procs.remove(p)


def start_fake(logdir: Path, **cfg) -> subprocess.Popen:
    cmd = [sys.executable, "scripts/fake_providers.py", "--port", str(FAKE_PORT)]
    for k, v in cfg.items():
        cmd += ["--" + k.replace("_", "-"), str(v)]
    return _spawn(cmd, dict(os.environ), logdir / "fake.log")


OPCOUNT_FILE = ""


def start_api(logdir: Path, workers: int) -> subprocess.Popen:
    global OPCOUNT_FILE
    env = dict(os.environ)
    env["RATE_LIMIT_DISABLED"] = "1"       # many guests hammer the API on purpose (ignored on Vercel)
    OPCOUNT_FILE = ""
    if workers == 1:                       # count MongoDB operations (command-monitoring listener)
        OPCOUNT_FILE = str(logdir / "opcount.json")
        Path(OPCOUNT_FILE).unlink(missing_ok=True)
        env["LT_OPCOUNT_FILE"] = OPCOUNT_FILE
        env["LT_PORT"] = str(API_PORT)
        if os.environ.get("LT_SAMPLE"):          # profile: in-process thread-leaf sampler (see _lt_server.py)
            env["LT_SAMPLE_FILE"] = str(logdir / "sample.json")
        env["LOG_LEVEL"] = os.environ.get("LT_LOG_LEVEL", "WARNING")
        env.pop("VERCEL", None)
        env["LLM_CHAIN_TEST_BASE_URL"] = f"http://127.0.0.1:{FAKE_PORT}"
        return _spawn([sys.executable, "scripts/_lt_server.py"], env, logdir / "api.log")
    env["LLM_CHAIN_TEST_BASE_URL"] = f"http://127.0.0.1:{FAKE_PORT}"
    env.pop("VERCEL", None)
    env["LOG_LEVEL"] = os.environ.get("LT_LOG_LEVEL", "WARNING")
    cmd = [sys.executable, "-m", "uvicorn", "api.index:app", "--port", str(API_PORT), "--log-level", "warning",
           "--no-access-log"]
    if workers > 1:
        cmd += ["--workers", str(workers)]
    return _spawn(cmd, env, logdir / "api.log")


async def wait_ready(url: str, timeout: float = 120.0) -> None:
    t_end = time.time() + timeout
    async with httpx.AsyncClient() as c:
        while time.time() < t_end:
            try:
                if (await c.get(url, timeout=3)).status_code == 200:
                    return
            except Exception:  # noqa: BLE001
                pass
            await asyncio.sleep(0.3)
    raise RuntimeError(f"{url} did not become ready")


def tree_rss_mb(root_pid: int) -> float:
    try:
        out = subprocess.run(["ps", "-A", "-o", "pid=,ppid=,rss="], capture_output=True, text=True, timeout=5).stdout
    except Exception:  # noqa: BLE001
        return 0.0
    kids: dict[int, list[int]] = defaultdict(list)
    rss: dict[int, int] = {}
    for line in out.splitlines():
        try:
            pid, ppid, r = (int(x) for x in line.split())
        except ValueError:
            continue
        kids[ppid].append(pid)
        rss[pid] = r
    total, stack = 0, [root_pid]
    while stack:
        p = stack.pop()
        total += rss.get(p, 0)
        stack.extend(kids.get(p, []))
    return total / 1024.0


class Sampler:
    """RSS of the API process tree (1 Hz), /api/health probe (10 Hz) and client event-loop lag."""

    def __init__(self, pid: int, base: str) -> None:
        self.pid, self.base = pid, base
        self.rss: list[float] = []
        self.health: list[float] = []
        self.health_err = 0
        self.lag: list[float] = []
        self._tasks: list[asyncio.Task] = []
        self._client = httpx.AsyncClient(limits=httpx.Limits(max_connections=4), timeout=10)

    async def _rss(self) -> None:
        while True:
            self.rss.append(await asyncio.to_thread(tree_rss_mb, self.pid))
            await asyncio.sleep(1.0)

    async def _health(self) -> None:
        while True:
            t = time.perf_counter()
            try:
                r = await self._client.get(self.base + "/api/health")
                if r.status_code == 200:
                    self.health.append((time.perf_counter() - t) * 1000.0)
                else:
                    self.health_err += 1
            except Exception:  # noqa: BLE001
                self.health_err += 1
            await asyncio.sleep(0.1)

    async def _lag(self) -> None:
        while True:
            t = time.perf_counter()
            await asyncio.sleep(0.05)
            self.lag.append(max(0.0, (time.perf_counter() - t - 0.05) * 1000.0))

    def start(self) -> None:
        self._tasks = [asyncio.create_task(f()) for f in (self._rss, self._health, self._lag)]

    async def stop(self) -> dict:
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        await self._client.aclose()
        return {
            "rss_mb_max": round(max(self.rss, default=0)), "rss_mb_first": round(self.rss[0]) if self.rss else 0,
            "rss_mb_last": round(self.rss[-1]) if self.rss else 0,
            "health_ms": {"n": len(self.health), "p50": round(pct(self.health, 50), 1), "p99": round(pct(self.health, 99), 1),
                          "max": round(max(self.health, default=0), 1), "errors": self.health_err},
            "client_loop_lag_ms": {"p99": round(pct(self.lag, 99), 1), "max": round(max(self.lag, default=0), 1)},
        }


# ------------------------------------------------------------------------------------------------
# Mongo: guests and cleanup (ONLY documents whose owner / guestId starts with the loadtest prefix)
# ------------------------------------------------------------------------------------------------
def _mongo():
    from backend import db

    d = db.get_db()
    if d is None:
        raise SystemExit("MongoDB is not configured (MONGODB_URI) - the load test needs it for guest accounts.")
    return db, d


def provision_guests(n: int) -> list[str]:
    if n > MAX_GUESTS:
        raise SystemExit(f"refusing to create more than {MAX_GUESTS} guests")
    db, d = _mongo()
    cleanup()                                       # start from a clean slate (idempotent)
    now = db.now()
    ids = [f"{PREFIX}{i:03d}" for i in range(1, n + 1)]
    d[db.GUESTS].insert_many(
        [{"guestId": g, "isGuest": True, "displayName": "Loadtest", "createdAt": now, "lastActiveAt": now,
          "migratedTo": None, "hasUploads": False} for g in ids], ordered=False)
    return ids


def cleanup() -> dict:
    """Remove every document created by this script, keyed by the guest-id prefix. Idempotent."""
    import shutil

    db, d = _mongo()
    rx = {"$regex": "^" + re.escape(PREFIX)}
    removed: dict[str, int] = {}
    for coll in (*db.KINDS, *db.EPHEMERAL_TTLS):
        n = d[coll].delete_many({"owner": rx}).deleted_count
        if n:
            removed[coll] = n
    removed["guests"] = d[db.GUESTS].delete_many({"guestId": rx}).deleted_count
    # GridFS (uploaded PDFs + retrieval index blobs) owned by loadtest guests
    for bucket in (db.UPLOADS_BUCKET, db.INDEX_BUCKET):
        ids = [f["_id"] for f in d[bucket + ".files"].find({"owner": rx}, {"_id": 1})]
        if ids:
            d[bucket + ".chunks"].delete_many({"files_id": {"$in": ids}})
            d[bucket + ".files"].delete_many({"_id": {"$in": ids}})
            removed[bucket] = len(ids)
    # local on-disk copies of uploaded subjects (documents/user__<owner>__<subject>)
    from backend.config import DOCUMENTS_DIR

    n_dirs = 0
    for p in DOCUMENTS_DIR.glob(f"user__{PREFIX}*"):
        shutil.rmtree(p, ignore_errors=True)
        n_dirs += 1
    if n_dirs:
        removed["local_dirs"] = n_dirs
    # on-disk retrieval-index cache (vector_store._cache_root): a stale copy here is REUSED by the next
    # upload with the same chunk text, so it must go too (otherwise uploads skip embedding entirely)
    from backend import vector_store

    n_idx = 0
    for p in vector_store._cache_root().glob(f"user__{PREFIX}*"):
        shutil.rmtree(p, ignore_errors=True)
        n_idx += 1
    if n_idx:
        removed["index_cache_dirs"] = n_idx
    return {k: v for k, v in removed.items() if v}


# ------------------------------------------------------------------------------------------------
# Request helpers
# ------------------------------------------------------------------------------------------------
EXC_SAMPLES: Counter = Counter()      # client-side exception texts (what the socket layer said)


def _note_exc(exc: BaseException) -> None:
    EXC_SAMPLES[f"{type(exc).__name__}: {str(exc)[:140]}"] += 1


async def timed(c: httpx.AsyncClient, m: Metrics, name: str, method: str, url: str, **kw):
    t = time.perf_counter()
    try:
        r = await c.request(method, url, **kw)
        m.add(name, r.status_code, time.perf_counter() - t)
        sv = _srv_ms(r)
        if sv is not None:
            m.srv[name].append(sv)
        return r
    except Exception as exc:  # noqa: BLE001
        _note_exc(exc)
        m.add(name, "EXC:" + type(exc).__name__, time.perf_counter() - t)
        return None


async def stream_message(c: httpx.AsyncClient, m: Metrics, gid: str, sid: str, text: str) -> None:
    """POST /api/chat/message/stream; record TTFT (first ``token`` event) and total; expect ``done``+llm."""
    t = time.perf_counter()
    first = None
    done = None
    padding = False
    status: object = "EXC"
    try:
        async with c.stream("POST", "/api/chat/message/stream", headers={"X-Guest-Id": gid},
                            json={"session_id": sid, "message": text}) as r:
            status = r.status_code
            sv = _srv_ms(r)                  # server time to the first byte (headers are sent with the first token)
            if sv is not None:
                m.srv["chat stream (server TTFB)"].append(sv)
            if status == 200:
                async for line in r.aiter_lines():
                    if line.startswith(": connected"):
                        padding = True
                    elif line.startswith("data:"):
                        try:
                            ev = json.loads(line[5:].strip())
                        except ValueError:
                            continue
                        if ev.get("type") == "token" and first is None:
                            first = time.perf_counter() - t
                        elif ev.get("type") == "done":
                            done = ev
    except Exception as exc:  # noqa: BLE001
        _note_exc(exc)
        status = "EXC:" + type(exc).__name__
    dt = time.perf_counter() - t
    m.add("chat stream", status, dt)
    if status == 200:
        if first is not None:
            m.ttft.append(first * 1000.0)
        m.stream_total.append(dt * 1000.0)
        label = ((done or {}).get("llm") or {}).get("label")
        if done is None or not label or not padding or first is None:
            m.stream_bad += 1
        else:
            m.labels[label.replace("Model: ", "")[:48]] += 1


async def user_session(c: httpx.AsyncClient, m: Metrics, gid: str, think: float, delay: float, full: bool = True) -> None:
    await asyncio.sleep(delay)
    H = {"X-Guest-Id": gid}
    await timed(c, m, "GET /api/subjects", "GET", "/api/subjects", headers=H)
    await timed(c, m, "GET /api/history/chats", "GET", "/api/history/chats", headers=H)
    r = await timed(c, m, "POST /api/chat/start", "POST", "/api/chat/start", headers=H,
                    json={"mode": "subject", "subject": "Mathematics"})
    if r is None or r.status_code != 200:
        return
    sid = r.json()["session_id"]
    msgs = random.sample(CHAT_MESSAGES, 3)
    for i, text in enumerate(msgs):
        await stream_message(c, m, gid, sid, f"{text} ({gid[-3:]}-{i})")
        await asyncio.sleep(random.uniform(0.5 * think, 1.5 * think))
    if not full:
        return
    r = await timed(c, m, "POST /api/quiz/generate", "POST", "/api/quiz/generate", headers=H,
                    json={"subject": "Mathematics", "pdf": "Lec-1.pdf", "num_questions": 3})
    _check_llm(m, r)
    await asyncio.sleep(random.uniform(0.5 * think, 1.5 * think))
    r = await timed(c, m, "POST /api/flashcards/generate", "POST", "/api/flashcards/generate", headers=H,
                    json={"subject": "Mathematics", "pdf": "Lec-1.pdf", "num_cards": 3})
    _check_llm(m, r)
    await timed(c, m, "GET /api/history/quizzes", "GET", "/api/history/quizzes", headers=H)
    await timed(c, m, "GET /api/progress/summary", "GET", "/api/progress/summary", headers=H)


def _check_llm(m: Metrics, r) -> None:
    if r is not None and r.status_code == 200:
        try:
            lbl = (r.json().get("llm") or {}).get("label")
        except ValueError:
            lbl = None
        if lbl:
            m.labels["(non-stream) " + lbl.replace("Model: ", "")[:40]] += 1
        else:
            m.stream_bad += 1


def new_client(base: str, conns: int = 600) -> httpx.AsyncClient:
    return httpx.AsyncClient(base_url=base, timeout=httpx.Timeout(90.0, connect=30.0),
                             limits=httpx.Limits(max_connections=conns, max_keepalive_connections=conns,
                                                 # uvicorn closes idle keep-alive sockets after 5 s; expire ours
                                                 # first, or a reused socket the server just closed shows as ReadError
                                                 keepalive_expiry=3.0))


# ------------------------------------------------------------------------------------------------
# Scenarios
# ------------------------------------------------------------------------------------------------
async def scenario_a(base: str, guests: list[str], conc: int, dur: float) -> tuple[dict, dict]:
    """Closed-loop hammering of cheap endpoints."""
    results = {}
    async with new_client(base, conc + 50) as c:
        sem = asyncio.Semaphore(20)          # warm each guest's first-touch (Mongo) cost out of the measurement

        async def warm(g):
            async with sem:
                await c.get("/api/subjects", headers={"X-Guest-Id": g})

        await asyncio.gather(*(warm(g) for g in guests[:conc]))
        for name, path, hdr in (("GET /api/health", "/api/health", {}), ("GET /api/config", "/api/config", {}),
                                ("GET /api/subjects", "/api/subjects", None), ("GET / (static shell)", "/", {})):
            m = Metrics()
            stop = time.perf_counter() + dur

            async def worker(i: int):
                h = {"X-Guest-Id": guests[i % len(guests)]} if hdr is None else hdr
                while time.perf_counter() < stop:
                    await timed(c, m, name, "GET", path, headers=h)

            await asyncio.gather(*(worker(i) for i in range(conc)))
            m.done()
            results[name] = summarize(m)
    return results, {}


async def scenario_users(base: str, guests: list[str], users: int, think: float, ramp: float, full: bool = True) -> Metrics:
    m = Metrics()
    async with new_client(base) as c:
        await asyncio.gather(*(user_session(c, m, guests[i], think, ramp * i / max(1, users), full)
                               for i in range(users)))
    m.done()
    return m


async def scenario_e(base: str, guests: list[str], n: int) -> Metrics:
    m = Metrics()
    data = PDF.read_bytes()
    async with new_client(base) as c:
        async def one(i: int):
            gid = guests[i]
            H = {"X-Guest-Id": gid}
            name = f"LoadSubj{i:03d}"
            r = await timed(c, m, "POST /api/subjects (upload)", "POST", "/api/subjects", headers=H, data={"name": name},
                            files=[("files", ("Lec-1.pdf", data, "application/pdf"))])
            if r is not None and r.status_code == 200:
                await timed(c, m, "GET /api/subjects (own)", "GET", "/api/subjects", headers=H)
                await timed(c, m, "GET .../pdfs", "GET", f"/api/subjects/{name}/pdfs", headers=H)
                await timed(c, m, "DELETE /api/subjects/x", "DELETE", f"/api/subjects/{name}", headers=H)

        # health stays responsive while uploads parse/embed in worker threads
        await asyncio.gather(*(one(i) for i in range(n)))
    m.done()
    return m


async def scenario_f(base: str, guests: list[str], users: int, rounds: int, interval: float, ramp: float) -> Metrics:
    """Idle dashboard refresh: each user keeps the dashboard open (progress, history drawer, subjects) and
    it re-fetches every ``interval`` seconds. After the first round almost nothing should reach MongoDB."""
    m = Metrics()
    async with new_client(base) as c:
        async def one(i: int):
            await asyncio.sleep(ramp * i / max(1, users))
            H = {"X-Guest-Id": guests[i]}
            for r in range(rounds):
                await timed(c, m, "GET /api/progress/summary", "GET", "/api/progress/summary", headers=H)
                await timed(c, m, "GET /api/history/chats", "GET", "/api/history/chats", headers=H)
                await timed(c, m, "GET /api/history/quizzes", "GET", "/api/history/quizzes", headers=H)
                await timed(c, m, "GET /api/subjects", "GET", "/api/subjects", headers=H)
                await asyncio.sleep(random.uniform(0.8 * interval, 1.2 * interval))

        await asyncio.gather(*(one(i) for i in range(users)))
    m.done()
    return m


# ------------------------------------------------------------------------------------------------
# Orchestration
# ------------------------------------------------------------------------------------------------
async def with_server(logdir: Path, workers: int, fake_cfg: dict, body):
    fake = start_fake(logdir, **fake_cfg)
    api = start_api(logdir, workers)
    base = f"http://127.0.0.1:{API_PORT}"
    try:
        await wait_ready(f"http://127.0.0.1:{FAKE_PORT}/_stats", 30)
        await wait_ready(base + "/api/health", 120)
        # warm: one Mongo touch per worker so the first-request cost is not measured as load
        async with httpx.AsyncClient(timeout=30) as c:
            for _ in range(workers * 3):
                await c.get(base + "/api/health/deep")
        sampler = Sampler(api.pid, base)
        sampler.start()
        try:
            result = await body(base)
        finally:
            extra = await sampler.stop()
        async with httpx.AsyncClient(timeout=5) as c:
            try:
                extra["fake_stats"] = (await c.get(f"http://127.0.0.1:{FAKE_PORT}/_stats")).json()
            except Exception:  # noqa: BLE001
                pass
        if OPCOUNT_FILE:                     # stop the server so its listener flushes the op counts
            _kill(api)
            await asyncio.sleep(0.5)
            if Path(OPCOUNT_FILE).exists():
                try:
                    extra["mongo_ops"] = json.loads(Path(OPCOUNT_FILE).read_text())
                except Exception:  # noqa: BLE001
                    pass
        return result, extra
    finally:
        _kill(api)
        _kill(fake)
        await asyncio.sleep(1.0)


def _upload_pdf_blobs(n: int, pages: int = 4) -> list[bytes]:
    """n different small PDFs (slices of the bundled lectures), each a few pages."""
    import io
    import uuid

    from pypdf import PdfReader, PdfWriter

    srcs = [PdfReader(str(ROOT / "documents" / s / f)) for s in ("Mathematics", "Operating System")
            for f in ("Lec-1.pdf", "Lec-2.pdf")]
    out = []
    for i in range(n):
        r = srcs[i % len(srcs)]
        start = (i * 3) % max(1, len(r.pages) - pages)
        w = PdfWriter()
        for p in range(start, min(len(r.pages), start + pages)):
            w.add_page(r.pages[p])
        w.add_metadata({"/Title": f"loadtest upload {i} {uuid.uuid4().hex}"})   # unique bytes per run
        b = io.BytesIO()
        w.write(b)
        out.append(b.getvalue())
    return out


async def scenario_g(base: str, guests: list[str], n: int, recover_s: float = 0.0) -> Metrics:
    """Chunked upload path as the frontend drives it: POST /uploads, PUT chunks, POST complete, then
    POST continue-index (sequentially) while the subject is still partial. With ``recover_s`` > 0 the
    fake Gemini starts answering again after that many seconds (a provider outage that ends)."""
    m = Metrics()
    blobs = _upload_pdf_blobs(n)
    e2e: list[float] = []
    cont_calls: list[int] = []
    final: Counter = Counter()
    first_state: list = []
    async with new_client(base) as c:
        async def recover():
            await asyncio.sleep(recover_s)
            async with httpx.AsyncClient(timeout=5) as fc:
                await fc.post(f"http://127.0.0.1:{FAKE_PORT}/_cfg", json={"gemini_fail": 0.0})

        if recover_s > 0:
            asyncio.create_task(recover())
        async def one(i: int):
            gid, blob, name = guests[i], blobs[i], f"LoadUp{i:03d}"
            H = {"X-Guest-Id": gid}
            t0 = time.perf_counter()
            r = await timed(c, m, "POST /api/subjects/uploads", "POST", "/api/subjects/uploads", headers=H,
                            json={"subject": name, "filename": f"lec{i}.pdf", "size": len(blob)})
            if r is None or r.status_code != 200:
                return
            info = r.json()
            uid, cs, total = info["upload_id"], info["chunk_size"], info["total_chunks"]
            for k in range(total):
                await timed(c, m, "PUT .../chunks/{i}", "PUT", f"/api/subjects/uploads/{uid}/chunks/{k}",
                            headers={**H, "Content-Type": "application/octet-stream"},
                            content=blob[k * cs:(k + 1) * cs])
            r = await timed(c, m, "POST .../uploads/{id}/complete", "POST",
                            f"/api/subjects/uploads/{uid}/complete", headers=H)
            if r is None or r.status_code != 200:
                return
            e2e.append((time.perf_counter() - t0) * 1000)
            body = r.json()
            first_state.append((body.get("indexing"), body.get("n_embedded"), body.get("chunk_count")))
            calls = 0
            while body.get("indexing") == "partial" and calls < 40:
                r = await timed(c, m, "POST continue-index", "POST", f"/api/subjects/{name}/continue-index",
                                headers=H)
                calls += 1
                if r is None or r.status_code != 200:
                    break
                cur = r.json()
                if cur.get("busy"):
                    await asyncio.sleep(1.0)
                    continue
                body = {**body, **cur}
                if cur.get("progress", 0) == 0 and cur.get("indexing") == "partial":
                    await asyncio.sleep(2.0)       # quota window full: wait for it to slide
            cont_calls.append(calls)
            final[body.get("indexing", "?")] += 1

        await asyncio.gather(*(one(i) for i in range(n)))
    m.done()
    m.upload_e2e = e2e                               # type: ignore[attr-defined]
    m.continue_calls = cont_calls                    # type: ignore[attr-defined]
    m.final_state = dict(final)                      # type: ignore[attr-defined]
    m.first_state = first_state                      # type: ignore[attr-defined]
    return m


async def main_async(a) -> int:
    import tempfile

    import socket

    for port in (API_PORT, FAKE_PORT):              # never measure someone else's server on our ports
        with socket.socket() as sk:
            sk.settimeout(0.5)
            if sk.connect_ex(("127.0.0.1", port)) == 0:
                raise SystemExit(f"port {port} is already in use; refusing to run against it")
    logdir = Path(a.logdir or tempfile.mkdtemp(prefix="loadtest-"))
    logdir.mkdir(parents=True, exist_ok=True)
    wanted = [s.strip() for s in a.scenarios.split(",") if s.strip()]
    report: dict = {"args": {k: v for k, v in vars(a).items()}, "scenarios": {}}
    need = max(a.users, 20, a.uploads, a.g_users)
    guests = provision_guests(need)
    print(f"provisioned {len(guests)} throwaway guests ({PREFIX}001..)")
    fake_ok = {"latency_ms": a.latency_ms, "jitter_ms": a.latency_ms / 3, "token_ms": 12}
    try:
        if "a" in wanted:
            async def body(base):
                return await scenario_a(base, guests, a.a_conc, a.a_secs)
            (res, _), extra = await with_server(logdir, 1, fake_ok, body)
            for name, s in res.items():
                print_summary(f"A: {name} @ {a.a_conc} concurrent", s)
            print("A server:", {k: v for k, v in extra.items() if k != "fake_stats"})
            report["scenarios"]["a"] = {"results": res, "server": extra}

        for key, title, workers, fcfg, users in (
            ("b", "B: realistic users, 1 worker", 1, fake_ok, a.users),
            ("c", "C: fault injection (Anthropic 50% 529, Groq 30% 429, Gemini 100% 429), 1 worker", 1,
             {**fake_ok, "anthropic_fail": 0.5, "anthropic_status": 529, "groq_fail": 0.3, "groq_status": 429,
              "gemini_fail": 1.0, "gemini_status": 429}, a.users),
            ("d", f"D: realistic users, {a.workers} workers", a.workers, fake_ok, a.users),
        ):
            if key not in wanted:
                continue

            async def body(base, users=users):
                return await scenario_users(base, guests, users, a.think, a.ramp)
            (m), extra = await with_server(logdir, workers, fcfg, body)
            s = summarize(m)
            ops = extra.get("mongo_ops")
            print_summary(f"{title} - {users} users", s, {"mongo ops": (f"total={ops['total']} per session={ops['total'] / users:.1f} {ops['by_command']}" if ops else "n/a (multi-worker)"),
                                                         "server/probe": {k: v for k, v in extra.items() if k not in ("fake_stats", "mongo_ops")},
                                                         "fake calls": extra.get("fake_stats", {}).get("calls"),
                                                         "fake injected": extra.get("fake_stats", {}).get("injected")})
            report["scenarios"][key] = {"summary": s, "server": extra}

        if "f" in wanted:
            async def body(base):
                return await scenario_f(base, guests, a.users, a.f_rounds, a.f_interval, a.ramp)
            (m), extra = await with_server(logdir, 1, fake_ok, body)
            s = summarize(m)
            ops = extra.get("mongo_ops", {})
            print_summary(f"F: idle dashboard refresh - {a.users} users x {a.f_rounds} rounds (every ~{a.f_interval}s)", s,
                          {"mongo ops total": ops.get("total"), "ops per user": round(ops.get("total", 0) / max(1, a.users), 2),
                           "ops by command": ops.get("by_command")})
            report["scenarios"]["f"] = {"summary": s, "server": extra}

        if "g" in wanted:
            async def body(base):
                return await scenario_g(base, guests, a.g_users, a.g_recover_s)
            (m), extra = await with_server(logdir, 1, {**fake_ok, "gemini_fail": a.g_gemini_fail, "gemini_status": 429}, body)
            s = summarize(m)
            e2e = sorted(getattr(m, "upload_e2e", []) or [0])
            cc = getattr(m, "continue_calls", [])
            print_summary(f"G: chunked uploads - {a.g_users} users, distinct small PDFs", s,
                          {"upload end-to-end ms": {"p50": round(pct(e2e, 50)), "p95": round(pct(e2e, 95)),
                                                    "max": round(max(e2e))},
                           "continue-index calls per subject": {"p50": pct(cc, 50), "max": max(cc or [0])},
                           "final indexing state": getattr(m, "final_state", None),
                           "state right after complete (indexing, n_embedded, chunks)": Counter(
                               getattr(m, "first_state", [])).most_common(4)})
            report["scenarios"]["g"] = {"summary": s, "server": extra,
                                        "upload_e2e_ms": e2e, "continue_calls": cc,
                                        "final_state": getattr(m, "final_state", None)}

        if "e" in wanted:
            async def body(base):
                return await scenario_e(base, guests, a.uploads)
            (m), extra = await with_server(logdir, 1, fake_ok, body)
            s = summarize(m)
            print_summary(f"E: upload path - {a.uploads} concurrent uploads of {PDF.name} ({PDF.stat().st_size // 1024} KB)", s,
                          {"server/probe": {k: v for k, v in extra.items() if k != "fake_stats"}})
            report["scenarios"]["e"] = {"summary": s, "server": extra}
    finally:
        print("cleanup:", cleanup())
    if a.json:
        Path(a.json).write_text(json.dumps(report, indent=1, default=str))
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scenarios", default="a,b,c,d,e")
    ap.add_argument("--users", type=int, default=200)
    ap.add_argument("--think", type=float, default=2.0, help="mean think-time between chat messages (s)")
    ap.add_argument("--ramp", type=float, default=10.0, help="spread user arrival over this many seconds")
    ap.add_argument("--workers", type=int, default=4, help="uvicorn workers for scenario d")
    ap.add_argument("--a-conc", type=int, default=200)
    ap.add_argument("--a-secs", type=float, default=10.0)
    ap.add_argument("--uploads", type=int, default=8)
    ap.add_argument("--g-users", type=int, default=20, help="scenario g: concurrent chunked uploads")
    ap.add_argument("--g-recover-s", type=float, default=0.0,
                    help="scenario g: Gemini stops failing after this many seconds (outage ends)")
    ap.add_argument("--g-gemini-fail", type=float, default=0.0,
                    help="scenario g: fraction of Gemini embedding calls answered 429 (forces partial indexing)")
    ap.add_argument("--f-rounds", type=int, default=6)
    ap.add_argument("--f-interval", type=float, default=4.0)
    ap.add_argument("--latency-ms", type=float, default=300.0, help="fake provider time-to-first-byte")
    ap.add_argument("--logdir", default="", help="where server logs go (default: a fresh temp dir)")
    ap.add_argument("--json", default="")
    ap.add_argument("--cleanup-only", action="store_true")
    a = ap.parse_args()
    if a.cleanup_only:
        print("cleanup:", cleanup())
        return

    def _bye(*_):
        for p in list(procs):
            _kill(p)
        try:
            print("cleanup:", cleanup())
        finally:
            os._exit(130)

    signal.signal(signal.SIGINT, _bye)
    signal.signal(signal.SIGTERM, _bye)
    try:
        sys.exit(asyncio.run(main_async(a)))
    finally:
        for p in list(procs):
            _kill(p)


if __name__ == "__main__":
    main()
