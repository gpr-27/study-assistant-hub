"""Size cap + LRU eviction for the per-instance ``/tmp`` caches.

Two directories grow without bound on a long-lived serverless instance:

* ``$TMPDIR/study-documents``  - one directory per hydrated user subject (PDFs from GridFS)
* ``$TMPDIR/study-index``      - one directory per cached user-subject index

Both are pure caches (MongoDB is the source of truth), so the least-recently-used entry can
always be dropped and re-hydrated on demand. :func:`touch` records use; :func:`enforce` evicts
the oldest entries until the total is under ``TMP_CACHE_MAX_MB`` (default 300). An entry is
NEVER evicted while

* it was used in the last ``GRACE_S`` seconds (a request may still be reading it), or
* the caller's ``protect`` predicate says it is in use (e.g. an upload/delete claim, or the
  index is resident in the RAM cache).
"""
from __future__ import annotations

import logging
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Callable

from backend import config

log = logging.getLogger(__name__)

GRACE_S = 180.0                       # never evict anything touched this recently
_MAX_TRACKED = 4096

_lock = threading.Lock()
_last_used: dict[str, float] = {}     # str(path) -> time.monotonic()
_enforce_lock = threading.Lock()


def max_bytes() -> int:
    try:
        return max(1, int(getattr(config, "TMP_CACHE_MAX_MB", 300))) * 1024 * 1024
    except (TypeError, ValueError):
        return 300 * 1024 * 1024


def touch(path: Path | str) -> None:
    """Record that ``path`` (a cache entry directory) was just used."""
    with _lock:
        if len(_last_used) >= _MAX_TRACKED:
            cutoff = time.monotonic() - GRACE_S
            for k in [k for k, v in _last_used.items() if v < cutoff]:
                _last_used.pop(k, None)
        _last_used[str(path)] = time.monotonic()


def forget(path: Path | str) -> None:
    with _lock:
        _last_used.pop(str(path), None)


def last_used(path: Path | str) -> float | None:
    with _lock:
        return _last_used.get(str(path))


def _dir_size(path: Path) -> int:
    total = 0
    try:
        for root, _dirs, files in os.walk(path):
            for name in files:
                try:
                    total += os.lstat(os.path.join(root, name)).st_size
                except OSError:
                    pass
    except OSError:
        pass
    return total


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def enforce(roots: list[Path], *, protect: Callable[[Path], bool] | None = None,
            limit_bytes: int | None = None) -> list[Path]:
    """Evict least-recently-used entries under ``roots`` until their total size fits the cap.

    Entries are the direct child directories of each root (dot-prefixed temp dirs are
    ignored). Returns the evicted paths. Never raises; concurrent callers collapse to one.
    """
    limit = max_bytes() if limit_bytes is None else limit_bytes
    if not _enforce_lock.acquire(blocking=False):
        return []
    evicted: list[Path] = []
    try:
        entries: list[tuple[float, int, Path]] = []
        total = 0
        now = time.monotonic()
        wall = time.time()
        for root in roots:
            try:
                children = [c for c in root.iterdir() if c.is_dir() and not c.name.startswith(".")]
            except OSError:
                continue
            for child in children:
                size = _dir_size(child)
                total += size
                seen = last_used(child)
                # age in seconds since last use (fall back to the directory mtime for entries
                # this process never touched, e.g. left behind by a previous invocation)
                age = (now - seen) if seen is not None else max(0.0, wall - _mtime(child))
                entries.append((age, size, child))
        if total <= limit:
            return []
        entries.sort(key=lambda e: -e[0])               # oldest use first
        for age, size, path in entries:
            if total <= limit:
                break
            if age < GRACE_S:
                continue
            try:
                if protect is not None and protect(path):
                    continue
            except Exception:  # noqa: BLE001 - when unsure, keep it
                continue
            shutil.rmtree(path, ignore_errors=True)
            forget(path)
            total -= size
            evicted.append(path)
            log.info("Evicted cache entry %s (%d KB) to stay under the /tmp cap.", path.name, size // 1024)
        return evicted
    except Exception:  # noqa: BLE001 - housekeeping must never break a request
        log.debug("tmp cache enforcement failed", exc_info=True)
        return evicted
    finally:
        _enforce_lock.release()
