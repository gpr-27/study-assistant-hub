"""In-memory stand-in for the ``chats`` collection (find_one / update_one with the few operators chat uses)."""
from __future__ import annotations

import copy

from bson import ObjectId


class _Res:
    def __init__(self, n):
        self.matched_count = n
        self.modified_count = n


def _elem_match(arr, cond) -> int | None:
    for i, el in enumerate(arr or []):
        if all(el.get(k) == v for k, v in cond.items()):
            return i
    return None


class FakeChats:
    def __init__(self):
        self.docs: list[dict] = []

    def add(self, owner, messages=None, **fields) -> str:
        doc = {"_id": ObjectId(), "owner": owner, "messages": copy.deepcopy(messages or []), **fields}
        self.docs.append(doc)
        return str(doc["_id"])

    def get(self, chat_id):
        return next(d for d in self.docs if str(d["_id"]) == chat_id)

    def _find(self, flt):
        for d in self.docs:
            ok = True
            for k, v in flt.items():
                if isinstance(v, dict) and "$elemMatch" in v:
                    ok = _elem_match(d.get(k), v["$elemMatch"]) is not None
                else:
                    ok = d.get(k) == v
                if not ok:
                    break
            if ok:
                return d
        return None

    def find_one(self, flt, projection=None):
        d = self._find(flt)
        return copy.deepcopy(d) if d else None

    def update_one(self, flt, update, upsert=False):
        d = self._find(flt)
        if d is None:
            return _Res(0)
        for path, value in update.get("$set", {}).items():
            if ".$." in path:
                arr, rest = path.split(".$.")
                i = _elem_match(d[arr], flt[arr]["$elemMatch"])
                d[arr][i][rest] = copy.deepcopy(value)
            elif "." in path:
                parts = path.split(".")
                cur = d
                for p in parts[:-1]:
                    cur = cur[int(p)] if isinstance(cur, list) else cur[p]
                last = parts[-1]
                if isinstance(cur, list):
                    cur[int(last)] = copy.deepcopy(value)
                else:
                    cur[last] = copy.deepcopy(value)
            else:
                d[path] = copy.deepcopy(value)
        for path in update.get("$unset", {}):
            if ".$." in path:
                arr, rest = path.split(".$.")
                i = _elem_match(d[arr], flt[arr]["$elemMatch"])
                d[arr][i].pop(rest, None)
            else:
                parts = path.split(".")
                cur = d
                for p in parts[:-1]:
                    cur = cur[int(p)] if isinstance(cur, list) else cur[p]
                if isinstance(cur, list):
                    cur[int(parts[-1])].pop(parts[-1], None)
                else:
                    cur.pop(parts[-1], None)
        for path, spec in update.get("$push", {}).items():
            arr = d.setdefault(path, [])
            arr.extend(copy.deepcopy(spec["$each"]))
            if spec.get("$slice", 0) < 0:
                del arr[:max(0, len(arr) + spec["$slice"])]
        return _Res(1)
