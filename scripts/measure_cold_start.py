"""Measure serverless cold-start cost of the API.

Spawns FRESH interpreters (so nothing is cached in-process) and reports:
  * wall time to ``import api.index`` (the function's cold-start import cost),
  * time for the first and second ``GET /api/health`` through FastAPI's TestClient,
  * the top offenders from ``python -X importtime`` (cumulative ms).

Usage:   python scripts/measure_cold_start.py [--runs 3] [--top 15]
Run it with the interpreter whose site-packages mirror the deployed requirements.txt.
Never prints env values; only timings and module names.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

_PROBE = r"""
import json, os, sys, time
sys.path.insert(0, os.getcwd())
t0 = time.perf_counter()
try:
    import api.index as m
    target = "api.index"
except ModuleNotFoundError:
    import backend.main as m
    target = "backend.main"
t1 = time.perf_counter()
from fastapi.testclient import TestClient
client = TestClient(m.app)
t2 = time.perf_counter()
r1 = client.get("/api/health")
t3 = time.perf_counter()
r2 = client.get("/api/health")
t4 = time.perf_counter()
print(json.dumps({"target": target, "import_s": t1 - t0, "testclient_s": t2 - t1,
                  "health1_s": t3 - t2, "health2_s": t4 - t3, "status": r1.status_code,
                  "modules": len(sys.modules)}))
"""


def _env() -> dict:
    env = dict(os.environ)
    env.setdefault("APP_ENV", "production")
    return env


def run_probe() -> dict:
    out = subprocess.run([sys.executable, "-c", _PROBE], cwd=ROOT, env=_env(),
                         capture_output=True, text=True, timeout=180)
    if out.returncode != 0:
        raise SystemExit("probe failed:\n" + out.stderr[-2000:])
    return json.loads(out.stdout.strip().splitlines()[-1])


def import_offenders(top: int) -> list[tuple[float, str]]:
    code = "import sys; sys.path.insert(0, '.')\ntry:\n import api.index\nexcept ModuleNotFoundError:\n import backend.main"
    out = subprocess.run([sys.executable, "-X", "importtime", "-c", code], cwd=ROOT, env=_env(),
                         capture_output=True, text=True, timeout=180)
    rows = []
    for line in out.stderr.splitlines():
        m = re.match(r"import time:\s+(\d+) \|\s+(\d+) \| (\s*)(\S+)", line)
        if m:
            rows.append((int(m.group(2)) / 1000.0, m.group(3).__len__(), m.group(4)))
    # Report top-level-ish offenders: cumulative ms, de-duplicated by root package.
    best: dict[str, float] = {}
    for cum, depth, name in rows:
        root = name.split(".")[0]
        best[root] = max(best.get(root, 0.0), cum)
    # `api`/`backend` roots are the sum of everything below them; list the real third-party/own
    # sub-packages instead (own modules appear as backend.<x> so slow first-party imports show up).
    own = {}
    for cum, _depth, name in rows:
        if name.startswith("backend."):
            own[name] = max(own.get(name, 0.0), cum)
    best.pop("api", None)
    best.pop("backend", None)
    best.update(own)
    ranked = sorted(((v, k) for k, v in best.items()), reverse=True)
    return ranked[:top]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()
    results = [run_probe() for _ in range(args.runs)]
    med = lambda k: statistics.median(r[k] for r in results)  # noqa: E731
    print(f"python            : {sys.version.split()[0]}  ({sys.executable})")
    print(f"target module     : {results[0]['target']}")
    print(f"runs              : {args.runs} (fresh interpreter each)")
    print(f"import (median)   : {med('import_s') * 1000:8.0f} ms   <- cold-start import cost")
    print(f"TestClient create : {med('testclient_s') * 1000:8.0f} ms")
    print(f"first  /api/health: {med('health1_s') * 1000:8.0f} ms   (HTTP {results[0]['status']})")
    print(f"second /api/health: {med('health2_s') * 1000:8.0f} ms")
    print(f"modules loaded    : {results[0]['modules']}")
    print(f"\nTop {args.top} import offenders (cumulative ms, by root package):")
    for ms, name in import_offenders(args.top):
        print(f"  {ms:8.1f}  {name}")


if __name__ == "__main__":
    main()
