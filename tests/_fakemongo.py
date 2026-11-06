"""A tiny in-memory stand-in for the slice of pymongo the grading code uses (no network, no mongomock).

Supports: ``db[name]`` collections with ``insert_one`` (+ the partial unique ``(owner, attempt_id)`` index ->
``DuplicateKeyError``), ``find_one`` / ``find().sort().limit()`` with equality, ``$type``, ``$lte`` and ``$in`` filters,
``update_one`` with ``$set`` (dotted keys) and ``create_index``.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone

from bson import ObjectId
from pymongo.errors import DuplicateKeyError


def _get(doc, path):
    cur = doc
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _match(doc, flt) -> bool:
    for k, cond in flt.items():
        v = _get(doc, k) if k != "_id" else doc.get("_id")
        if isinstance(cond, dict) and any(str(c).startswith("$") for c in cond):
            for op, arg in cond.items():
                if op == "$type" and not (arg == "string" and isinstance(v, str)):
                    return False
                if op == "$lte" and not (v is not None and v <= arg):
                    return False
                if op == "$in" and v not in arg:
                    return False
        elif v != cond:
            return False
    return True


def _project(doc, projection):
    inc = [k.split(".")[0] for k, v in (projection or {}).items() if v == 1 and k != "_id"]
    if not inc:
        return doc
    return {k: v for k, v in doc.items() if k in inc or k in ("_id", "owner")}


class _Cursor:
    def __init__(self, docs, projection=None):
        self.docs = docs
        self.projection = projection

    def sort(self, key, direction=1):
        self.docs.sort(key=lambda d: str(_get(d, key) or ""), reverse=direction < 0)
        return self

    def limit(self, n):
        self.docs = self.docs[:n]
        return self

    def __iter__(self):
        return iter(copy.deepcopy([_project(d, self.projection) for d in self.docs]))


class _Res:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class FakeColl:
    def __init__(self):
        self.docs: list[dict] = []
        self.unique_attempt = False

    def create_index(self, keys, **kw):
        if kw.get("unique"):
            self.unique_attempt = True

    def insert_one(self, doc):
        doc = copy.deepcopy(doc)
        if self.unique_attempt and isinstance(doc.get("attempt_id"), str):
            for d in self.docs:
                if d.get("owner") == doc.get("owner") and d.get("attempt_id") == doc["attempt_id"]:
                    raise DuplicateKeyError("E11000 duplicate key")
        doc["_id"] = ObjectId()
        self.docs.append(doc)
        return _Res(inserted_id=doc["_id"])

    def find_one(self, flt, projection=None):
        for d in self.docs:
            if _match(d, flt):
                return copy.deepcopy(_project(d, projection))
        return None

    def find(self, flt, projection=None):
        return _Cursor([d for d in self.docs if _match(d, flt)], projection)

    def count_documents(self, flt):
        return sum(1 for d in self.docs if _match(d, flt))

    def delete_one(self, flt):
        for i, d in enumerate(self.docs):
            if _match(d, flt):
                del self.docs[i]
                return _Res(deleted_count=1)
        return _Res(deleted_count=0)

    def update_one(self, flt, update, upsert=False):
        for d in self.docs:
            if _match(d, flt):
                for path, value in update.get("$set", {}).items():
                    parts = path.split(".")
                    cur = d
                    for p in parts[:-1]:
                        cur = cur.setdefault(p, {})
                    cur[parts[-1]] = copy.deepcopy(value)
                return _Res(matched_count=1, modified_count=1)
        return _Res(matched_count=0, modified_count=0)


class FakeMongo:
    def __init__(self):
        self.colls: dict[str, FakeColl] = {}

    def __getitem__(self, name) -> FakeColl:
        return self.colls.setdefault(name, FakeColl())

    def count(self, name) -> int:
        return len(self[name].docs)


def now():
    return datetime.now(timezone.utc)
