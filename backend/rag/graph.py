"""Concept graph (GraphRAG-style) built from the chunks of ONE subject index, with no LLM by default.

Why: a plain dense/BM25 index finds passages that *look like the question*. Multi-hop questions
("how does caching relate to the storage hierarchy", "compare X and Y"), questions that span
several PDFs and "global" questions ("what are the main themes of this subject") need the
*relations between concepts* too. This module extracts them at index time, for free:

  * **Concepts** - RAKE-style noun-phrase candidates (1-3 words, stop-word delimited) plus
    definition patterns ("X is a ...", "called X", "X (ACRONYM)"), slide/section headings and
    bullet terms. Scored with tf-idf; closed sub-phrases are merged into their longer phrase.
  * **Edges** - concept co-occurrence inside a chunk or two adjacent chunks of the same file,
    weighted by positive PMI (damped by log count); each concept keeps its strongest neighbours.
  * **Membership** - concept -> chunk weights (tf-idf, per chunk max-normalised).
  * **Communities** - weighted label propagation over the concept graph; each community gets an
    *extractive* summary (its most central sentences). ``GRAPH_LLM_SUMMARIES=1`` rewrites them with
    ONE chain call, ``GRAPH_LLM_RELATIONS=1`` labels strong edges (is-a / part-of / requires /
    contrasts) with one call per ~20 chunks. Both are index-time only, JSON mode, ``sensitive`` for
    user content, and best-effort.

Query time (:class:`ConceptGraph`): the query's concepts seed a personalised PageRank (a few
sparse matrix-vector products in numpy) or a bounded BFS; chunk score = sum of concept mass times
membership weight. That list is the third ranker fused with dense + BM25 by ``VectorIndex.rank``.
Cost is well under 5 ms for bundled subjects and needs no network.

Storage: ``graph.json`` next to ``chunks.json`` (also inside the user-index zip stored in Mongo).
It carries a signature of the chunk texts, so a graph that does not belong to the chunks it is
loaded with is ignored and rebuilt in memory - the graph can never outlive the index it describes.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import random
import re
import threading
import time
from collections import Counter, OrderedDict

log = logging.getLogger(__name__)

GRAPH_VERSION = 2
GRAPH_FILE = "graph.json"

MAX_NGRAM = 3
MAX_EDGES_PER_NODE = 10
PPR_ALPHA = 0.30                  # restart probability
PPR_ITERS = 8
SEED_DISCOUNT = 0.0               # chunk-score weight of the query's own concepts vs expanded ones
METHOD = "ppr"                    # "ppr" | "bfs"
HOPS = 2
MAX_CONCEPTS_PER_WINDOW = 25

_EXTRA_STOP = frozenset("""
example examples solution solutions find show then thus hence let following given also may might one two three
four five six first second third next such each every any all some other another more most many much very just
only same different various etc cont case cases note notes called known said value values number numbers question
questions answer answers true false correct option options type types objective choose fill blanks exercise
exercises chapter overview use used using whether without within upon since because both either neither than too
can could would should shall must like get got make makes made take takes taken per via vs ie eg thing things
way ways lot new old good bad big small large little long short high low well way part parts form forms called
refer refers see seen let lets lecture lectures slide slides edition book end introduction
""".split())

_RUN_BREAK = re.compile(r"[^A-Za-z\-'’\s]+")
_WORD = re.compile(r"[A-Za-z][A-Za-z\-'’]*")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\s*[▪•●◦]\s*")

_DEF_PATTERNS = (
    re.compile(r"((?:[A-Za-z][\w\-’']*\s+){0,3}[A-Za-z][\w\-’']*)\s+(?:is|are)\s+"
               r"(?:a|an|the|said to be|called|defined as|known as|termed)\b"),
    re.compile(r"\b(?:called|known as|termed|named|denoted by|referred to as)\s+(?:a |an |the )?"
               r"([A-Za-z][\w\-’']*(?:\s+[A-Za-z][\w\-’']*){0,2})"),
    re.compile(r"((?:[A-Za-z][\w\-]*\s+){1,4}[A-Za-z][\w\-]*)\s*\(([A-Z]{2,6})s?\)"),
    re.compile(r"[▪•●]\s*([A-Z][A-Za-z\-’' ]{2,30}?)\s*[–—:-]\s"),
    re.compile(r"\*\*([^*\n]{3,40})\*\*"),
    re.compile(r"\b\d+(?:\.\d+){1,3}\s+([A-Z][A-Za-z\-’' ]{3,45}?)(?=\s+(?:[A-Z]|\d|▪|If|Let|A |The )|$)"),
)
_EXERCISE = re.compile(r"(?:^|\s)\d{1,3}\.\s+[A-Z]")
_ACRONYM = re.compile(r"\b([A-Z]{2,6})s?\b")


def _stop() -> frozenset:
    from backend import vector_store

    return vector_store._STOP | _EXTRA_STOP


def _stem(w: str) -> str:
    from backend import vector_store

    return vector_store._stem(w)


def chunks_sig(chunks: list[dict]) -> str:
    """Stable signature of the chunk texts (+ order): ties a graph to the chunks it was built from."""
    h = hashlib.blake2b(digest_size=10)
    h.update(f"v{GRAPH_VERSION}:{len(chunks)}:".encode())
    for c in chunks:
        h.update(c["text"].encode("utf-8", "ignore"))
        h.update(b"\x00")
    return h.hexdigest()


# --------------------------------------------------------------------------------------
# Phrase extraction
# --------------------------------------------------------------------------------------
def phrase_runs(text: str, stop: frozenset | None = None) -> list[list[str]]:
    """Lower-cased word runs delimited by stop words / punctuation / digits (RAKE candidates)."""
    stop = stop or _stop()
    out: list[list[str]] = []
    for frag in _RUN_BREAK.split(text or ""):
        run: list[str] = []
        for m in _WORD.finditer(frag):
            w = m.group(0).strip("-'’").lower()
            if len(w) < 3 or w in stop:
                if run:
                    out.append(run)
                    run = []
                continue
            run.append(w)
        if run:
            out.append(run)
    return out


def grams_of(runs: list[list[str]], max_n: int = MAX_NGRAM) -> list[tuple[str, str]]:
    """``(stem key, surface)`` for every contiguous 1..max_n gram inside each run."""
    out: list[tuple[str, str]] = []
    for run in runs:
        stems = [_stem(w) for w in run]
        n = len(run)
        for i in range(n):
            for j in range(i + 1, min(n, i + max_n) + 1):
                out.append((" ".join(stems[i:j]), " ".join(run[i:j])))
    return out


def _pattern_phrases(text: str, stop: frozenset) -> list[tuple[str, str, bool]]:
    """Definition / heading / bullet-term phrases (the ones worth a bonus): ``(key, surface, is_definition)``.
    Only runs of 1-3 words are kept - a longer run is a slide title glued together, not a concept."""
    raws: list[tuple[str, bool]] = []
    acr: list[str] = []
    for pat in _DEF_PATTERNS:
        for m in pat.finditer(text):
            raws.append((m.group(1), True))
            if m.lastindex and m.lastindex >= 2 and m.group(2) and m.group(2).isupper():
                acr.append(m.group(2))
    head = re.split(r"[\u25aa\u2022:]", text[:90], maxsplit=1)[0]
    head = re.sub(r"^\s*\d+(?:\.\d+)*\s+", "", head)
    if 2 < len(head) <= 70:
        raws.append((head, False))
    found: list[tuple[str, str, bool]] = [(a.lower(), a, True) for a in acr]
    for raw, is_def in raws:
        for run in phrase_runs(raw, stop):
            if len(run) <= MAX_NGRAM:
                found.append((" ".join(_stem(w) for w in run), " ".join(run), is_def))
    return found


def _acronyms(text: str) -> list[str]:
    return [m.group(1) for m in _ACRONYM.finditer(text)]


# --------------------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------------------
def build_graph(chunks: list[dict], *, sensitive: bool = False, llm: bool | None = None,
                budget_s: float = 20.0) -> dict | None:
    """Concept graph for ``chunks`` as a JSON-able dict (see module docstring). ``None`` when the
    corpus is too small/empty to carry any structure. Never raises for content reasons."""
    import numpy as np

    t0 = time.monotonic()
    n = len(chunks)
    if n < 2:
        return None
    stop = _stop()
    big = n > 3000
    max_n = 2 if big else MAX_NGRAM

    per_chunk: list[Counter] = []
    def_keys: set[str] = set()
    head_keys: set[str] = set()
    surf: dict[str, str] = {}
    df: Counter = Counter()
    cf: Counter = Counter()
    for c in chunks:
        text = c["text"]
        cnt: Counter = Counter()
        for key, sf in grams_of(phrase_runs(text, stop), max_n):
            cnt[key] += 1
            surf.setdefault(key, sf)
        for key, sf, is_def in _pattern_phrases(text, stop):
            if key:
                (def_keys if is_def else head_keys).add(key)
                surf.setdefault(key, sf)
                cnt.setdefault(key, 1)
        for ac in _acronyms(text):
            k = ac.lower()
            surf[k] = ac                         # keep the upper-case surface
            cnt.setdefault(k, 1)
        per_chunk.append(cnt)
        for key, tf in cnt.items():
            df[key] += 1
            cf[key] += tf
        if time.monotonic() - t0 > budget_s:
            log.info("Concept graph: extraction budget spent; using the first part only.")
            break

    max_df = max(3, int(0.5 * n)) if n >= 20 else n
    cands: dict[str, float] = {}
    for key, d in df.items():
        is_pat = key in def_keys or key in head_keys
        if d < 2 and key not in def_keys:
            continue
        if d > max_df:
            continue
        nw = key.count(" ") + 1
        base = key.replace(" ", "")
        if nw == 1 and len(base) < 3 and not is_pat:
            continue
        idf = math.log(1 + n / d)
        score = idf * (1 + math.log(cf[key] / d)) * (1 + 0.35 * (nw - 1)) * (1.6 if is_pat else 1.0)
        if nw == 1 and len(base) < 4:
            score *= 0.5
        cands[key] = score

    # Closed sub-phrases. "skew symmetric" always followed by "matrix" -> keep the longer phrase when the extra
    # word is specific to it; when the extra word is a common one (df >= 1.5x the phrase's) the shorter phrase
    # is the real concept ("address translation avoids" -> "address translation").
    for key in sorted(cands, key=lambda k: -k.count(" ")):
        if key not in cands:
            continue
        parts = key.split(" ")
        if len(parts) < 2:
            continue
        for sub, extra in ((" ".join(parts[1:]), parts[0]), (" ".join(parts[:-1]), parts[-1])):
            if sub in cands and df[sub] == df[key]:
                if df[extra] >= 1.5 * df[key]:
                    cands.pop(key, None)
                    break
                cands.pop(sub, None)
        else:
            for i in range(len(parts)):
                for j in range(i + 1, len(parts) + 1):
                    sub = " ".join(parts[i:j])
                    if j - i < len(parts) and sub in cands and df[sub] == df[key] and key in cands:
                        cands.pop(sub, None)

    cap = max(40, min(2500, int(1.5 * n) + 40))
    keys = sorted(cands, key=lambda k: (-cands[k], k))[:cap]
    if not keys:
        return None
    kid = {k: i for i, k in enumerate(keys)}
    idf = [round(math.log(1 + n / df[k]), 4) for k in keys]

    # membership + per-window concept lists
    mk: list[int] = []
    mc: list[int] = []
    mw: list[int] = []
    window_sets: list[list[int]] = []
    for ci, cnt in enumerate(per_chunk):
        row = [(kid[k], (1 + math.log(tf)) * idf[kid[k]]) for k, tf in cnt.items() if k in kid and tf > 0]
        row.sort(key=lambda x: (-x[1], x[0]))
        row = row[:40]
        top = row[0][1] if row else 1.0
        for cid, w in row:
            mk.append(ci)
            mc.append(cid)
            mw.append(max(1, int(round(100 * w / top))))
        window_sets.append([cid for cid, _ in row[:MAX_CONCEPTS_PER_WINDOW]])

    # windows: each chunk + each adjacent pair of chunks of the same file
    windows: list[list[int]] = [sorted(set(w)) for w in window_sets]
    for i in range(len(chunks) - 1):
        if chunks[i]["source"] == chunks[i + 1]["source"]:
            windows.append(sorted(set(window_sets[i]) | set(window_sets[i + 1])))
    nk = len(keys)
    na = np.zeros(nk, dtype=np.int64)
    pairs: list = []
    for w in windows:
        if not w:
            continue
        arr = np.asarray(w, dtype=np.int64)
        na[arr] += 1
        if len(arr) > 1:
            a, b = np.triu_indices(len(arr), 1)
            pairs.append(arr[a] * nk + arr[b])
    ea: list[int] = []
    eb: list[int] = []
    ew: list[float] = []
    if pairs:
        codes, counts = np.unique(np.concatenate(pairs), return_counts=True)
        min_co = 2 if n >= 12 else 1
        W = max(1, len(windows))
        keep = counts >= min_co
        codes, counts = codes[keep], counts[keep]
        a_ids, b_ids = codes // nk, codes % nk
        pmi = np.log(counts * W / (na[a_ids] * na[b_ids]).astype(np.float64))
        wgt = np.maximum(pmi, 0.0) * np.log1p(counts)
        order = np.argsort(-wgt, kind="stable")
        deg = np.zeros(nk, dtype=np.int32)
        for idx in order:
            if wgt[idx] <= 0:
                break
            a, b = int(a_ids[idx]), int(b_ids[idx])
            if deg[a] >= MAX_EDGES_PER_NODE and deg[b] >= MAX_EDGES_PER_NODE:
                continue
            ea.append(a)
            eb.append(b)
            ew.append(round(float(wgt[idx]), 3))
            deg[a] += 1
            deg[b] += 1

    quality = []
    for c in chunks:
        t = c["text"]
        non_space = sum(1 for ch in t if not ch.isspace()) or 1
        ratio = sum(1 for ch in t if ch.isalpha()) / non_space
        q = max(0.15, min(1.0, (ratio - 0.5) / 0.3))                                    # formula/table debris -> low
        if len(_EXERCISE.findall(t)) >= 3:
            q *= 0.5                                      # a list of numbered exercises is a poor overview chunk
        quality.append(int(round(100 * q)))
    data: dict = {
        "v": GRAPH_VERSION, "sig": chunks_sig(chunks), "n_chunks": n, "cq": quality,
        "keys": keys, "names": [surf.get(k, k) for k in keys], "idf": idf,
        "mk": mk, "mc": mc, "mw": mw, "ea": ea, "eb": eb, "ew": ew, "rel": [], "comm": [],
    }
    try:
        data["comm"] = _communities(data, chunks, stop)
    except Exception:  # noqa: BLE001 - communities are an extra
        log.info("Community detection failed; graph kept without themes.", exc_info=True)
    use_llm = _env_on("GRAPH_LLM_RELATIONS") if llm is None else llm
    if use_llm:
        try:
            _llm_relations(data, chunks, sensitive=sensitive)
        except Exception:  # noqa: BLE001
            log.info("LLM relation labelling skipped.", exc_info=True)
    if _env_on("GRAPH_LLM_SUMMARIES") if llm is None else llm:
        try:
            _llm_summaries(data, sensitive=sensitive)
        except Exception:  # noqa: BLE001
            log.info("LLM community summaries skipped.", exc_info=True)
    data["built_s"] = round(time.monotonic() - t0, 3)
    return data


def _env_on(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


# --------------------------------------------------------------------------------------
# Communities (weighted label propagation) + extractive summaries
# --------------------------------------------------------------------------------------
def _label_propagation(nk: int, adj: list[dict[int, float]], seed: int = 7, iters: int = 25) -> list[int]:
    labels = list(range(nk))
    order = list(range(nk))
    rng = random.Random(seed)
    for _ in range(iters):
        rng.shuffle(order)
        changed = 0
        for v in order:
            nb = adj[v]
            if not nb:
                continue
            tally: dict[int, float] = {}
            for u, w in nb.items():
                tally[labels[u]] = tally.get(labels[u], 0.0) + w
            best = max(tally.items(), key=lambda kv: (kv[1], -kv[0]))[0]
            if best != labels[v]:
                labels[v] = best
                changed += 1
        if not changed:
            break
    return labels


def _communities(data: dict, chunks: list[dict], stop: frozenset) -> list[dict]:
    keys, names, idf = data["keys"], data["names"], data["idf"]
    nk, n = len(keys), data["n_chunks"]
    adj: list[dict[int, float]] = [dict() for _ in range(nk)]
    for a, b, w in zip(data["ea"], data["eb"], data["ew"]):
        adj[a][b] = adj[a].get(b, 0.0) + w
        adj[b][a] = adj[b].get(a, 0.0) + w
    labels = _label_propagation(nk, adj)
    groups: dict[int, list[int]] = {}
    for v, lab in enumerate(labels):
        if adj[v]:
            groups.setdefault(lab, []).append(v)
    groups = {k: v for k, v in groups.items() if len(v) >= 3}
    if not groups:
        return []
    cmax = max(3, min(24, n // 6 + 2))
    # chunk -> concept weights
    by_chunk: dict[int, list[tuple[int, int]]] = {}
    for ci, cid, w in zip(data["mk"], data["mc"], data["mw"]):
        by_chunk.setdefault(ci, []).append((cid, w))
    comm_of = {}
    for g, members in groups.items():
        for v in members:
            comm_of[v] = g
    mass: dict[int, Counter] = {g: Counter() for g in groups}
    for ci, row in by_chunk.items():
        for cid, w in row:
            g = comm_of.get(cid)
            if g is not None:
                mass[g][ci] += w
    cent = [sum(adj[v].values()) * idf[v] for v in range(nk)]
    ranked_groups = sorted(groups, key=lambda g: (-sum(mass[g].values()), g))[:cmax]
    primary: dict[int, int] = {}              # chunk -> community with the largest mass
    best_mass: dict[int, tuple[float, int]] = {}
    for g in ranked_groups:
        for ci, m in mass[g].items():
            if ci not in best_mass or m > best_mass[ci][0]:
                best_mass[ci] = (m, g)
    for ci, (_m, g) in best_mass.items():
        primary[ci] = g
    out: list[dict] = []
    for g in ranked_groups:
        members = sorted(groups[g], key=lambda v: (-cent[v], v))
        cks = sorted((ci for ci, gg in primary.items() if gg == g), key=lambda ci: (-mass[g][ci], ci))
        if not cks:
            continue
        label = ", ".join(names[v] for v in members[:3])
        mset = {keys[v]: cent[v] for v in members}
        summary = _extractive_summary(chunks, cks[:6], mset, stop)
        out.append({"label": label, "concepts": members[:30], "chunks": cks, "summary": summary,
                    "size": len(members)})
    return out


def _extractive_summary(chunks: list[dict], chunk_ids: list[int], weights: dict[str, float],
                        stop: frozenset, max_sent: int = 2, max_chars: int = 420) -> str:
    scored: list[tuple[float, int, int, str]] = []
    for pos, ci in enumerate(chunk_ids):
        for si, sent in enumerate(_SENT_SPLIT.split(chunks[ci]["text"])):
            s = " ".join(sent.split())
            if not 50 <= len(s) <= 300:
                continue
            letters = sum(ch.isalpha() for ch in s)
            if letters < 0.6 * len(s):
                continue                          # formula / table debris
            keys = {k for k, _ in grams_of(phrase_runs(s, stop))}
            hit = sum(weights.get(k, 0.0) for k in keys)
            if hit <= 0:
                continue
            scored.append((hit / math.sqrt(len(s.split())), pos, si, s))
    scored.sort(key=lambda t: -t[0])
    chosen: list[tuple[int, int, str]] = []
    for _sc, pos, si, s in scored:
        if any(s[:40] == c[2][:40] for c in chosen):
            continue
        chosen.append((pos, si, s))
        if len(chosen) >= max_sent:
            break
    chosen.sort()
    return " ".join(c[2] for c in chosen)[:max_chars]


# --------------------------------------------------------------------------------------
# Optional LLM labelling (index time; flags GRAPH_LLM_RELATIONS / GRAPH_LLM_SUMMARIES)
# --------------------------------------------------------------------------------------
REL_TYPES = ("is-a", "part-of", "requires", "contrasts")
_REL_SYSTEM = ("You label relations between concepts of a student's course notes. Reply with valid JSON only.")
CHUNKS_PER_CALL = 20


def _llm_chat(system: str, user: str, *, sensitive: bool, max_tokens: int = 900):
    from backend import deadline, llm_chain

    t = deadline.clamp(25.0)
    return llm_chain.chat(system, [{"role": "user", "content": user}], max_tokens=max_tokens, temperature=0.0,
                          json_mode=True, deadline_s=t, sensitive=sensitive)


def _llm_relations(data: dict, chunks: list[dict], *, sensitive: bool) -> None:
    from backend.services.llm_json import extract_json_object

    max_calls = int(os.environ.get("GRAPH_LLM_MAX_CALLS", "") or 5)
    names, keys = data["names"], data["keys"]
    members: dict[int, set] = {}
    for ci, cid in zip(data["mk"], data["mc"]):
        members.setdefault(ci, set()).add(cid)
    edges = sorted(range(len(data["ew"])), key=lambda i: -data["ew"][i])
    rel: dict[int, str] = {}
    calls = 0
    for start in range(0, len(chunks), CHUNKS_PER_CALL):
        if calls >= max_calls:
            break
        batch = range(start, min(len(chunks), start + CHUNKS_PER_CALL))
        present = set().union(*(members.get(ci, set()) for ci in batch))
        cand = [i for i in edges if data["ea"][i] in present and data["eb"][i] in present and i not in rel][:12]
        if not cand:
            continue
        excerpts = "\n".join(f"- {' '.join(chunks[ci]['text'].split())[:240]}" for ci in batch)
        pairs = "\n".join(f"{j}: {names[data['ea'][i]]} | {names[data['eb'][i]]}" for j, i in enumerate(cand))
        user = (f"Notes excerpts:\n{excerpts}\n\nConcept pairs:\n{pairs}\n\n"
                f"For each pair choose ONE relation from {list(REL_TYPES)} (a rel b), or \"none\". "
                'Return ONLY JSON: {"relations": [{"i": 0, "type": "is-a"}]}')
        calls += 1
        res = _llm_chat(_REL_SYSTEM, user, sensitive=sensitive)
        arr = extract_json_object(res.text).get("relations")
        for r in arr if isinstance(arr, list) else []:
            try:
                i, t = int(r["i"]), str(r["type"]).lower()
            except (KeyError, TypeError, ValueError):
                continue
            if 0 <= i < len(cand) and t in REL_TYPES:
                rel[cand[i]] = t
    data["rel"] = [[i, t] for i, t in sorted(rel.items())]


def _llm_summaries(data: dict, *, sensitive: bool) -> None:
    from backend.services.llm_json import extract_json_object

    comms = data.get("comm") or []
    if not comms:
        return
    listing = "\n".join(f"{j}: {c['label']} :: {c['summary'][:300]}" for j, c in enumerate(comms[:24]))
    user = (f"Themes of a subject (label :: extracted sentences):\n{listing}\n\n"
            'Write one clear sentence per theme. Return ONLY JSON: {"summaries": ["...", "..."]} in the same order.')
    res = _llm_chat("You summarise themes of a student's course notes. Reply with valid JSON only.", user,
                    sensitive=sensitive, max_tokens=1200)
    arr = extract_json_object(res.text).get("summaries")
    if isinstance(arr, list):
        for c, s in zip(comms, arr):
            if isinstance(s, str) and s.strip():
                c["summary"] = " ".join(s.split())[:420]


# --------------------------------------------------------------------------------------
# Build cache (a /tmp checkpoint loop saves the same chunks many times)
# --------------------------------------------------------------------------------------
_memo: "OrderedDict[str, dict | None]" = OrderedDict()
_memo_lock = threading.Lock()


def build_enabled() -> bool:
    return os.environ.get("GRAPH_RAG_BUILD", "1").strip().lower() not in {"0", "false", "no", "off"}


def get_or_build(chunks: list[dict], *, sensitive: bool = False) -> dict | None:
    """Memoised :func:`build_graph` keyed by the chunk signature. ``None`` on any failure."""
    if not build_enabled() or not chunks:
        return None
    sig = chunks_sig(chunks)
    with _memo_lock:
        if sig in _memo:
            _memo.move_to_end(sig)
            return _memo[sig]
    try:
        data = build_graph(chunks, sensitive=sensitive)
    except Exception:  # noqa: BLE001 - a graph problem must never break indexing
        log.warning("Concept graph build failed; the index is saved without it.", exc_info=True)
        data = None
    with _memo_lock:
        _memo[sig] = data
        while len(_memo) > 4:
            _memo.popitem(last=False)
    return data


def dumps(data: dict) -> bytes:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def stamp(data: dict | None) -> dict | None:
    """The small summary that goes into the index manifest."""
    if not data:
        return None
    return {"v": data["v"], "sig": data["sig"], "n_concepts": len(data["keys"]), "n_edges": len(data["ea"]),
            "n_communities": len(data.get("comm") or []), "relations": len(data.get("rel") or [])}


# --------------------------------------------------------------------------------------
# Query-time graph
# --------------------------------------------------------------------------------------
_GLOBAL_RE = re.compile(
    r"\b(summari[sz]e|summary|overview|main (?:themes?|topics?|ideas?|points?|concepts?)|key (?:themes?|topics?|ideas?|"
    r"concepts?|points?)|big picture|what (?:does|do|is|are) (?:this|the|my|these) (?:[a-z]+ ){0,3}(?:subject|course|notes?|pdfs?|material|"
    r"lectures?|documents?|topics?) (?:cover|about|contain|include|covered)|everything (?:the|my|in)|whole (?:subject|course|"
    r"syllabus)|entire (?:subject|course|syllabus)|across (?:all|the) (?:lectures?|pdfs?|notes?|documents?)|"
    r"(?:all|every) (?:the )?(?:topics?|lectures?|chapters?)|what (?:topics|chapters) )\b", re.I)
_GENERIC = frozenset("""subject subjects course courses note notes pdf pdfs material materials lecture lectures document
documents syllabus topic topics theme themes idea ideas point points concept concepts covered cover covers contain contains
include includes everything whole entire main key big picture overview summary summarise summarize list all each every
across give tell show explain describe outline provide understand study notes slides chapter chapters content contents
lec text understanding overall picture course""".split())


def is_global_question(question: str, graph: "ConceptGraph | None" = None, subject: str = "") -> bool:
    """'Summarise the whole subject' / 'main themes' style questions (answered from the community view).

    A topic word that is not generic ("summarise the main ideas of deadlocks") makes it an ordinary scoped
    question - unless it is part of the subject's own name (``subject``: "Operating System")."""
    q = question or ""
    if not _GLOBAL_RE.search(q):
        return False
    stop = _stop()
    named = {_stem(w) for w in re.findall(r"[a-z]{3,}", subject.lower())} | {w for w in re.findall(r"[a-z]{3,}", subject.lower())}
    rest = [w.lower() for w in re.findall(r"[A-Za-z]{3,}", _GLOBAL_RE.sub(" ", q))]
    for w in rest:
        if w in stop or w in _GENERIC or w in {"the", "and", "for", "from", "with", "that", "this", "these", "those"}:
            continue
        if w in named or _stem(w) in named:
            continue
        return False
    return True


class ConceptGraph:
    """Immutable query-time view of one subject's graph (numpy arrays + a few dicts)."""

    def __init__(self, data: dict) -> None:
        import numpy as np

        self.data = data
        self.keys: list[str] = data["keys"]
        self.names: list[str] = data["names"]
        self.n_chunks: int = int(data["n_chunks"])
        self.nk = len(self.keys)
        self.key2id = {k: i for i, k in enumerate(self.keys)}
        self.idf = np.asarray(data["idf"], dtype=np.float32)
        self.mk = np.asarray(data["mk"], dtype=np.int32)
        self.mc = np.asarray(data["mc"], dtype=np.int32)
        self.mw = np.asarray(data["mw"], dtype=np.float32) / 100.0
        ea = np.asarray(data["ea"], dtype=np.int32)
        eb = np.asarray(data["eb"], dtype=np.int32)
        ew = np.asarray(data["ew"], dtype=np.float32)
        rel = {int(i): t for i, t in (data.get("rel") or [])}
        self.relations = {(int(ea[i]), int(eb[i])): t for i, t in rel.items() if i < len(ea)}
        if rel:
            boost = np.ones(len(ew), dtype=np.float32)
            for i, t in rel.items():
                if i < len(boost):
                    boost[i] = 1.25 if t != "contrasts" else 1.0
            ew = ew * boost
        self.src = np.concatenate([ea, eb])
        self.dst = np.concatenate([eb, ea])
        w = np.concatenate([ew, ew])
        out = np.bincount(self.src, weights=w, minlength=self.nk).astype(np.float32)
        out[out == 0] = 1.0
        self.w_norm = (w / out[self.src]).astype(np.float32)     # column-stochastic transition weights
        self.adj: dict[int, list[tuple[int, float]]] = {}
        for s, d, ww in zip(self.src.tolist(), self.dst.tolist(), w.tolist()):
            self.adj.setdefault(s, []).append((d, ww))
        self.communities: list[dict] = list(data.get("comm") or [])
        self.df = np.bincount(self.mc, minlength=self.nk)
        cq = data.get("cq")
        self.quality = (np.asarray(cq, dtype=np.float32) / 100.0 if cq and len(cq) == self.n_chunks
                        else np.ones(self.n_chunks, dtype=np.float32))

    @property
    def nbytes(self) -> int:
        return int(self.mk.nbytes + self.mc.nbytes + self.mw.nbytes + self.src.nbytes + self.dst.nbytes
                   + self.w_norm.nbytes + 64 * self.nk)

    # -- query -> seed concepts ------------------------------------------------------
    def seeds(self, query: str) -> dict[int, float]:
        stop = _stop()
        found: dict[int, float] = {}
        for key, _sf in grams_of(phrase_runs(query, stop)):
            cid = self.key2id.get(key)
            if cid is not None:
                nw = key.count(" ") + 1
                found[cid] = max(found.get(cid, 0.0), float(self.idf[cid]) * (1 + 0.35 * (nw - 1)))
        for ac in _acronyms(query):
            cid = self.key2id.get(ac.lower())
            if cid is not None:
                found[cid] = max(found.get(cid, 0.0), float(self.idf[cid]) * 1.3)
        return found

    # -- propagation -----------------------------------------------------------------
    def ppr(self, seeds: dict[int, float], alpha: float = PPR_ALPHA, iters: int = PPR_ITERS):
        """Personalised PageRank from ``seeds`` (restart prob ``alpha``, ``iters`` power steps)."""
        import numpy as np

        e = np.zeros(self.nk, dtype=np.float32)
        for cid, w in seeds.items():
            e[cid] = w
        tot = float(e.sum())
        if tot <= 0:
            return e
        e /= tot
        p = e.copy()
        for _ in range(iters):
            p = (1 - alpha) * np.bincount(self.dst, weights=self.w_norm * p[self.src],
                                          minlength=self.nk).astype(np.float32) + alpha * e
        return p

    def bfs(self, seeds: dict[int, float], hops: int = 2, decay: float = 0.5):
        """Bounded breadth-first expansion: concept mass spread ``hops`` steps, ``decay`` per hop."""
        import numpy as np

        mass = np.zeros(self.nk, dtype=np.float32)
        frontier = {c: float(w) for c, w in seeds.items()}
        tot = sum(frontier.values()) or 1.0
        for c, w in frontier.items():
            mass[c] = w / tot
        seen = set(frontier)
        cur = {c: w / tot for c, w in frontier.items()}
        for _h in range(max(0, hops)):
            nxt: dict[int, float] = {}
            for c, w in cur.items():
                nb = self.adj.get(c, [])
                z = sum(x for _, x in nb) or 1.0
                for d, x in nb:
                    nxt[d] = nxt.get(d, 0.0) + decay * w * x / z
            for d, w in nxt.items():
                mass[d] += w
            cur = {d: w for d, w in nxt.items() if d not in seen}
            seen.update(cur)
            if not cur:
                break
        return mass

    def concept_mass(self, query: str, *, method: str = "ppr", hops: int = 2):
        """``(seeds, concept mass)``; mass is ``None`` when the query names no known concept."""
        seeds = self.seeds(query)
        if not seeds:
            return seeds, None
        mass = self.bfs(seeds, hops) if method == "bfs" else self.ppr(seeds, iters=max(2, hops * 3 + 2) if hops else PPR_ITERS)
        return seeds, mass

    def chunk_scores(self, mass, seeds: dict[int, float], seed_discount: float = SEED_DISCOUNT):
        import numpy as np

        m = mass.copy()
        if seeds and seed_discount != 1.0:
            ids = np.fromiter(seeds.keys(), dtype=np.int64, count=len(seeds))
            m[ids] *= seed_discount
        return np.bincount(self.mk, weights=self.mw * m[self.mc], minlength=self.n_chunks)

    def rank_chunks(self, query: str, k: int = 30, *, method: str | None = None, hops: int | None = None,
                    seed_discount: float | None = None) -> list[tuple[int, float]]:
        """``[(chunk, score), …]`` best first (only positive scores). Empty when the query has no known concept."""
        import numpy as np

        seeds, mass = self.concept_mass(query, method=method or METHOD, hops=HOPS if hops is None else hops)
        if mass is None:
            return []
        sc = self.chunk_scores(mass, seeds, SEED_DISCOUNT if seed_discount is None else seed_discount)
        k = min(k, len(sc))
        if k <= 0:
            return []
        top = np.argpartition(-sc, k - 1)[:k]
        top = top[np.argsort(-sc[top], kind="stable")]
        return [(int(i), float(sc[i])) for i in top if sc[i] > 0]

    def related(self, query: str, n: int = 4, *, method: str = "ppr") -> list[str]:
        """Concepts related to the query's concepts (not already in the query), strongest first."""
        seeds, mass = self.concept_mass(query, method=method)
        if mass is None:
            return []
        qstems = set(_stem(w) for w in re.findall(r"[a-z]+", query.lower()))
        out: list[str] = []
        for cid in sorted(range(self.nk), key=lambda c: -mass[c]):
            if mass[cid] <= 0 or len(out) >= n:
                break
            if cid in seeds:
                continue
            if set(self.keys[cid].split(" ")) <= qstems:
                continue
            if any(set(self.keys[cid].split(" ")) <= set(self.keys[s].split(" ")) or
                   set(self.keys[s].split(" ")) <= set(self.keys[cid].split(" ")) for s in seeds):
                continue
            out.append(self.names[cid])
        return out

    # -- global view -----------------------------------------------------------------
    def themes(self) -> list[dict]:
        return [{"label": c["label"], "summary": c.get("summary", ""), "n_chunks": len(c["chunks"])}
                for c in self.communities]

    def global_chunks(self, k: int = 8, query: str = "") -> list[tuple[int, float]]:
        """``k`` chunks that together COVER the subject's central concepts (greedy weighted set cover over
        concept weight = idf x graph degree, discounted for formula-heavy chunks), best first. This is the
        global / 'main themes' retrieval: no query terms are used, so it needs no embedding at all."""
        import numpy as np

        if self.nk == 0 or self.n_chunks == 0:
            return []
        deg = np.bincount(self.src, minlength=self.nk).astype(np.float32)
        weight = self.idf * (1.0 + deg)
        top = np.argsort(-weight)[: max(24, min(160, self.nk))]
        want = np.zeros(self.nk, dtype=np.float32)
        want[top] = weight[top]
        covered = np.zeros(self.nk, dtype=bool)
        out: list[tuple[int, float]] = []
        taken: set[int] = set()
        base = self.mw * want[self.mc]                                  # membership x concept weight
        for _ in range(min(k, self.n_chunks)):
            gain = np.bincount(self.mk, weights=np.where(covered[self.mc], 0.0, base), minlength=self.n_chunks)
            gain = gain * self.quality
            if taken:
                gain[list(taken)] = -1.0
            best = int(np.argmax(gain))
            if gain[best] <= 0:
                break
            taken.add(best)
            out.append((best, float(gain[best])))
            covered[self.mc[self.mk == best]] = True
        top_g = out[0][1] if out else 1.0
        return [(i, g / top_g) for i, g in out]


def load(data: dict | None, chunks: list[dict]) -> ConceptGraph | None:
    """A :class:`ConceptGraph` for ``data`` iff it was built from exactly ``chunks``."""
    if not data or data.get("v") != GRAPH_VERSION or data.get("n_chunks") != len(chunks):
        return None
    if data.get("sig") != chunks_sig(chunks):
        return None
    try:
        return ConceptGraph(data)
    except Exception:  # noqa: BLE001
        log.info("Stored concept graph is unreadable; rebuilding.", exc_info=True)
        return None
