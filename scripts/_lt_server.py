"""Load-test helper: run the API under uvicorn with a MongoDB command listener that counts operations.

Used by scripts/loadtest.py (single worker only). Counts every command sent to MongoDB except
connection handshakes/pings and writes ``{"total": N, "by_command": {...}}`` to $LT_OPCOUNT_FILE when the
process exits. TEST ONLY; nothing in the deployed app imports this.
"""
import atexit
import collections
import json
import os
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pymongo import monitoring  # noqa: E402

_IGNORE = {"hello", "isMaster", "ismaster", "ping", "saslStart", "saslContinue", "endSessions", "buildInfo"}
_lock = threading.Lock()
_counts: collections.Counter = collections.Counter()


_durs: list[float] = []          # server round trip per command, seconds (as the driver sees it)


class _Counter(monitoring.CommandListener):
    def started(self, event):
        if event.command_name in _IGNORE:
            return
        target = event.command.get(event.command_name)
        with _lock:
            _counts[f"{event.command_name}:{target if isinstance(target, str) else ''}"] += 1

    def succeeded(self, event):
        if event.command_name not in _IGNORE:
            with _lock:
                _durs.append(event.duration_micros / 1e6)

    def failed(self, event):
        pass


def _pct(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(p / 100 * len(xs)))] * 1000, 1)


def _dump() -> None:
    path = os.environ.get("LT_OPCOUNT_FILE")
    if path:
        with _lock:
            from backend import llm_chain

            Path(path).write_text(json.dumps({
                "total": sum(_counts.values()), "by_command": dict(_counts),
                "breaker": {"opened": llm_chain._state.breaker_opened, "closed": llm_chain._state.breaker_closed},
                "command_ms": {"p50": _pct(_durs, 50), "p95": _pct(_durs, 95), "p99": _pct(_durs, 99),
                               "n": len(_durs)}}))


atexit.register(_dump)


_IDLE_FILES = ("threading.py", "selectors.py", "queue.py", "thread.py")


def _start_sampler(path: str, hz: float = 100.0) -> None:
    """In-process wall-clock sampler (no root needed). Every 1/hz s it looks at each thread's leaf frame and
    counts non-idle leaves ("file:func:line"); a thread parked on a lock, queue or the selector is idle.
    Written to ``path`` (JSON) on shutdown. Test only, enabled by LT_SAMPLE_FILE."""
    import time
    import traceback

    samples: collections.Counter = collections.Counter()
    loop_samples: collections.Counter = collections.Counter()   # main thread = the asyncio event loop
    state = {"n": 0, "busy": 0, "loop_n": 0, "loop_busy": 0}
    stop = threading.Event()
    main_id = threading.main_thread().ident

    def run() -> None:
        me = threading.get_ident()
        while not stop.is_set():
            for tid, frame in sys._current_frames().items():
                if tid == me:
                    continue
                fs = traceback.extract_stack(frame, limit=14 if tid == main_id else 2)
                if not fs:
                    continue
                leaf = fs[-1]
                idle = leaf.filename.endswith(_IDLE_FILES) or leaf.filename.endswith("selectors.py") \
                    or "epoll" in leaf.name
                if tid == main_id:
                    state["loop_n"] += 1
                    if not idle:
                        state["loop_busy"] += 1
                        key = " <- ".join(f"{Path(f.filename).name}:{f.name}:{f.lineno}" for f in reversed(fs[-8:]))
                        loop_samples[key] += 1
                state["n"] += 1
                if idle:
                    continue
                state["busy"] += 1
                samples[f"{Path(leaf.filename).name}:{leaf.name}:{leaf.lineno}"] += 1
            time.sleep(1.0 / hz)

    threading.Thread(target=run, name="lt-sampler", daemon=True).start()

    def dump() -> None:
        stop.set()
        Path(path).write_text(json.dumps({
            "samples_total": state["n"], "busy": state["busy"], "top_leaves": samples.most_common(40),
            "event_loop": {"samples": state["loop_n"], "busy": state["loop_busy"],
                           "busy_top": loop_samples.most_common(12)}}, indent=1))

    atexit.register(dump)
    globals()["_sampler_dump"] = dump

from backend import db  # noqa: E402

if os.environ.get("LT_SAMPLE_FILE"):
    _start_sampler(os.environ["LT_SAMPLE_FILE"])

db._POOL_KWARGS["event_listeners"] = [_Counter()]

if __name__ == "__main__":
    import uvicorn

    from api.index import app

    # uvicorn re-raises SIGTERM after shutting down (atexit never runs), so flush on app shutdown.
    from contextlib import asynccontextmanager

    _orig = app.router.lifespan_context

    @asynccontextmanager
    async def _wrapped(a):
        async with _orig(a):
            yield
        _dump()
        if "_sampler_dump" in globals():
            globals()["_sampler_dump"]()

    app.router.lifespan_context = _wrapped
    uvicorn.run(app, port=int(os.environ.get("LT_PORT", "8891")), log_level="warning", access_log=False)
