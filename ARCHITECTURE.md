# Architecture

Study Assistant Hub is a single **FastAPI** server that serves a vanilla-JS single-page
app **and** the `/api/*` JSON API on one origin. Text generation runs on **Groq**;
embeddings run on **Google Gemini** (hosted, `EMBEDDING_PROVIDER=gemini`, with a local
fastembed fallback via `EMBEDDING_PROVIDER=local`). Everything is driven by environment
variables — the same image runs identically on a laptop and on Render.

```
                         ┌───────────────────────────────────────────────┐
   Browser  ──────────▶  │  FastAPI (backend/main.py)                     │
   (SPA, frontend/)      │   • GET  /            → SPA (index.html)        │
                         │   • GET  /assets/*    → static JS/CSS           │
                         │   • POST /api/chat/*  → Chat (agentic RAG)      │
                         │   • POST /api/quiz|flashcards|mindmap|planner/* │
                         │   • GET  /api/health, /api/config               │
                         └───────────────┬───────────────────────────────┘
                                         │
              ┌──────────────────────────┼───────────────────────────┐
              ▼                          ▼                           ▼
        Groq (LLM)            Gemini embeddings + FAISS        MongoDB (optional)
     text generation          hosted embeddings / search    history · uploads · indexes
```

The required secrets are `GROQ_API_KEY` and `GEMINI_API_KEY` (embeddings). MongoDB
(persistence) and Clerk (auth) are optional; without them the app runs fully, just open
and stateless.

---

## 1. Configuration — one source of truth

`backend/config.py` is the **only** module that reads environment variables. It loads a
single root `.env`, validates everything at startup (`validate_config()`), and exposes
typed values. Missing/invalid required config makes the server refuse to start with a
clear message — never a hidden default. No model id, host, port, or key is hardcoded;
that is why one image serves any deployment.

See the [Config reference](#config-reference) at the bottom.

---

## 2. Startup sequence

The FastAPI **lifespan** (`backend/main.py`) runs once per worker:

1. Configure logging, **validate config**, print the startup banner (which now includes
   `CHAT RETRIEVAL: agentic | classic`).
2. Spawn a background warmup thread so the first request isn't slow:
   - `restore_uploaded_pdfs()` — re-hydrate user uploads from MongoDB to disk,
   - `warm_embeddings()` — initialize the embeddings client,
   - `warm_all_subjects()` — load (or build) each subject's FAISS index.
3. Start the guest janitor (cleans up expired anonymous accounts; no-op without MongoDB).

`/api/health` answers immediately (it's what Render probes) because warmup is off-thread.

---

## 3. Two content pipelines

There are two distinct ways content reaches the LLM. Don't confuse them.

### 3a. Chat → **RAG pipeline** (`backend/rag/*`, orchestrated by `backend/agentic_rag.py`)

Chat answers a *specific question* from the notes. The default path is retrieval-first: **no LLM
call happens before the answer streams** unless retrieval is weak.

```
plan (no LLM) ─▶ hybrid retrieve (dense+BM25 RRF, pool 24) ─▶ rerank ─▶ MMR + per-PDF balance
   ─▶ neighbour merge ─▶ confidence ─┬─(strong)──────────────────────────────▶ generate (cited)
                                     └─(weak, <0.2)─▶ 1 LLM rephrase + re-retrieve ─▶ generate
generate ─┬─ model asks <search>…</search> (≤ AGENTIC_RAG_HOPS, deadline-bounded) ─▶ fuse ─▶ generate
          └─ verify citations (no LLM) ─▶ done.sources / done.citations
```

- **Query understanding** (`backend/rag/query.py`): follow-ups ("Why is it needed?") are condensed with
  the topic of the previous user question; explicit file names ("in Lec-2") become a reranking hint.
- **Retrieval** (`backend/rag/retrieve.py`): per query, the subject index's hybrid search (embedding RRF
  plus BM25) with a candidate pool larger than the final k; one extra keyword BM25 pass (no embedding
  call); fused by weighted RRF. Rerank = fused rank + keyword coverage + dense cosine + position prior.
  MMR drops near-duplicates, the balance step keeps a strongly-matching second PDF, and the best
  chunks are merged with their adjacent chunks from the same file/page range.
- **Confidence** = best keyword coverage (blended with dense cosine when embeddings are present).
  Below `AGENTIC_RAG_WEAK_CONF` (0.25) the model is told to say "I couldn't find this in your notes"
  and to mark any general-knowledge answer as such.
- **Adaptive pass** (only when confidence < `AGENTIC_RAG_LOW_CONF`, 0.2): one LLM call proposes
  alternative queries and a hypothetical passage (HyDE); results are fused into the first pass.
- **Generation** (`backend/rag/grounding.py`): numbered blocks `[1] Lec-1.pdf p.14-15`; the system
  prompt requires answering only from them and citing inline `[1][2]`.
- **Citation verification** (no LLM): citation numbers that do not exist are stripped. An
  abstention answer has no sources.
- **Failure paths**: embeddings slow or down → BM25-only retrieval (mode `lexical`); every provider
  down → `error` event labelled `Model: none (all providers failed)`; a rephrase step that fails is
  skipped, and generation then reports the dead chain.
- **Cross-subject search** (`subject = "*"`, "All subjects"): one retriever per subject the owner can
  see (bundled + their own), fused like the passages of one subject. Owner isolation is enforced by
  `vector_store` access checks, so a user can never retrieve another user's chunks.

#### Concept graph + agentic planner (`backend/rag/graph.py`, `rag/query.py::decompose`)

Built at INDEX time from the chunks alone (no API call, no LLM): RAKE-style noun phrases + definition /
heading / acronym patterns -> tf-idf-scored **concepts**, PMI-weighted co-occurrence **edges** (same chunk or
adjacent chunks of one file), concept->chunk **membership**, label-propagation **communities** with an
extractive summary each. Stored as `graph.json` beside `chunks.json` (also inside the user-index zip in
Mongo), stamped in the manifest (`graph: {sig, n_concepts, n_edges, n_communities}`) and tied to the exact
chunk texts by a signature: a graph that does not match its chunks (re-index, other blob, old algorithm
version) is ignored and rebuilt in memory, so it can never outlive or leak across the index it describes
(user indexes are only reachable through the owner-checked `vector_store._get_index`). Bundled graphs:
`scripts/build_index.py --graph-only`. Optional LLM labelling, index time only, JSON mode, `sensitive` for
user content: `GRAPH_LLM_RELATIONS=1` (one call per ~20 chunks, max `GRAPH_LLM_MAX_CALLS`=5, edges typed
is-a / part-of / requires / contrasts) and `GRAPH_LLM_SUMMARIES=1` (one call for all community summaries).

Used for: (a) **global questions** ("summarise the whole subject", "main themes"): `global_retrieval` picks
the chunks that jointly cover the central concepts (greedy weighted set cover, formula/exercise chunks
discounted), in reading order, + themes in the prompt; 0 embeddings, 0 LLM, ~2 ms; (b) **related concepts**
(`done.related`, PPR neighbours) for follow-up chips; (c) optionally a **third ranker** (personalised
PageRank / bounded BFS over the concept graph, fused in the same RRF): `GRAPH_RAG_RANK=1`, OFF by default
because it did not help (numbers below). Query-time cost: 0.2 ms bundled, ~0.9 ms at 10k chunks; build 0.3 s
for 135 chunks, ~4 s for 5k, ~10 s for 10k (extraction stops after 20 s).

**Planner** (heuristic, no LLM): compare / contrast / relate ("difference between X and Y", "compare A, B and
C", "X vs Y", "how do X and Y relate") and multi-part questions ("... and how ...", "?" lists) are split into
<= 3 sub-queries. Each part is searched in parallel (`asyncio.gather`, one worker thread per part, inside
the request deadline; speculative by default so latency = max not sum, `AGENTIC_RAG_SPECULATE=0` for lazy);
parts the first pass already covers are dropped (early stop), the rest are fused with the first pass and
`balance_parts` guarantees every part's best passage a place; MMR dedupes across all results. Steps stream as
`rewrite` events. No new LLM call: the existing adaptive LLM expansion (confidence < 0.2) now asks for one
query per part. Hops stay <= `AGENTIC_RAG_HOPS` (2) and only below confidence 0.6. Knobs: `GRAPH_RAG_GLOBAL`,
`GRAPH_RAG_RELATED`, `GRAPH_RAG_RANK`, `GRAPH_RAG_BUILD` (index time), `AGENTIC_RAG_PARTS`,
`AGENTIC_RAG_SUBQUERIES`, `AGENTIC_RAG_SPECULATE`.

**Measured** (`scripts/eval_retrieval.py --graph`, `scripts/eval_rag.py --pipeline`; 32 new questions in
`tests/data/multihop_eval.json`: 6 compare, 18 cross-chunk, 8 global; knobs tuned on the dev half only; the
whole set is small and saturated, differences < ~0.05 are noise):

| retrieval over 135 chunks | hit@1 | MRR | recall@5 | recall@8 |
|---|---|---|---|---|
| dense | 0.875 | 0.938 | 0.820 | 0.863 |
| BM25 | 0.781 | 0.852 | 0.720 | 0.779 |
| hybrid (baseline) | 0.844 | 0.922 | 0.833 | 0.879 |
| hybrid + graph ranker (3rd RRF list) | 0.875 | 0.938 | 0.724 | 0.824 |
| hybrid + global routing (shipped) | 0.938 | 0.969 | 0.909 | 0.948 |

The graph as a RANKER does not help here: hybrid already finds 100 % of the compare / cross-chunk anchors at
k=8 (the bundled subjects are tiny), the graph never won a question at recall@5 and cost ~0.1 recall@5; test
half MRR +0.03 is inside the bootstrap CI (+0.00..+0.05). Single-hop regression: MRR 0.989 -> 0.972 with the
ranker on. What does help is the community view on global questions (8 near-duplicate phrasings of 2
subject-level tasks, so n_eff ~ 2; the global heuristics were adjusted while looking at these two subjects, no
held-out set): topic-group recall@8 0.52 -> 0.79. Whole-pipeline (`retrieve_events`, cached vectors, 300 ms
simulated embedding, no LLM) pre-generation latency: baseline p50 316 / p95 327 ms; parts lazy p95 635 ms
(a second embedding round trip when a part is uncovered); parts speculative p50 341 / p95 401 ms; global
questions 2-3 ms (no embedding). Pipeline recall (final merged blocks): 0.888 -> 0.930 (global 0.65 -> 0.81;
decomposed questions unchanged at 0.975, already saturated). The real gain of the planner is that a comparison
no longer relies on the model emitting `<search>` (one extra LLM round trip, seconds) to find its second half.

**`done` event shape (additive; the frontend keeps reading `sources` as filenames):**

| field | shape |
|---|---|
| `sources` | `["Lec-1.pdf", …]`: distinct files of the cited passages (`[]` when the answer abstains) |
| `citations` | `[{n, source, file, page, text, snippet, score, cited, subject?}]`: one per retrieved passage; `source`/`page`/`text` are the original keys; `page` is 1-based or null |

**Env knobs:** `AGENTIC_RAG` (on), `AGENTIC_RAG_QUERIES` (2), `AGENTIC_RAG_POOL` (24),
`AGENTIC_RAG_FINAL_K` (8), `AGENTIC_RAG_LOW_CONF` (0.2), `AGENTIC_RAG_WEAK_CONF` (0.25),
`AGENTIC_RAG_ADAPTIVE` (on), `AGENTIC_RAG_HOPS` (2), `AGENTIC_RAG_LLM_REWRITE` (off: always run the LLM
query expansion), `AGENTIC_RAG_LLM_RERANK` (off: listwise LLM rerank), `AGENTIC_RAG_MAX_ITERS` (1: the
legacy LLM rewrite/grade loop, kept as an opt-in).

**Evaluation:** `scripts/eval_rag.py` runs `tests/data/rag_eval.json` (37 questions over the bundled
Operating System and Mathematics PDFs, including follow-ups, out-of-scope questions and multi-chunk
questions) through the real chat turn. `--mode fake` checks the plumbing against
`scripts/fake_providers.py`; `--mode live --only-live` makes the real provider calls (22 questions).

### 3b. Quiz / Flashcards / Mind map / Planner → **direct PDF content**

These tools generate over a *whole selected PDF*, not a specific question, so they don't
retrieve. Each request extracts the PDF's text (`backend/documents.py::read_pdf` /
`extract_pdf_text`) and clips it to the per-call budget with
`clip_to_tokens(text, MAX_CONTENT_TOKENS)` before prompting the LLM. The budget is sized
to fit the Groq free-tier TPM limit (input + reserved output tokens count against it).

The Planner additionally offers per-PDF **summaries** ("Generate summary"): each selected
PDF is summarized into a TOPICS/SUMMARY layout (`services/planner.py::summarize_pdfs`)
and the result is saved to the user's history (kind `summaries`, separate from saved
plans) when persistence is enabled.

---

### 3c. Spend guards on the paid provider (`backend/spend.py`)
`llm_chain` bills ACTUAL Anthropic tokens (non-streaming `usage`, streaming `message_start`/`message_delta`; estimated
when absent) to a per-UTC-day ledger: scope `global` plus `owner:<sha256(owner)[:24]>`, kept in memory per instance and
flushed with `$inc` upserts to `usage_daily` (or Redis `INCRBY`) every ~20 s. The owner comes from a request-scoped
ContextVar holder (`spend.begin_request()` in the middleware, filled by `auth.current_owner`) or an explicit `owner=`
kwarg. Before each run `_Run.gate()` checks estimated prompt tokens + usage so far against the global/owner budgets
(shared totals cached ~30 s) and the per-request cap; a tripped budget adds `anthropic budget` / `anthropic too-large`
to `skipped` and the chain continues with Groq/Gemini. Store errors fail open to the per-instance counters.
`LLMRequestTooLarge` (HTTP 413) is raised only when no non-Anthropic model may take an oversize prompt.

## 4. Embeddings & search index (`backend/vector_store.py`)

Each subject's PDFs are split into 1,500-char chunks, embedded via the hosted **Gemini**
embeddings API (`backend/gemini_embeddings.py`; `EMBEDDING_PROVIDER=local` falls back to
the keyless on-device fastembed model), and stored in a per-subject **FAISS** index at
`faiss_index/<subject>/`. A `manifest.json` fingerprints the PDFs + settings, so an index
rebuilds only when its PDFs (or the embedding model) actually change — bundled subjects
ship with pre-baked indexes that are merely *loaded*, never re-embedded (important on
Render's 512MB free tier). This index backs the chat retriever (pipeline 3a). User
subjects' index files are also persisted to MongoDB (GridFS) and restored on startup.

**Storage decision (graph + vectors): keep numpy + BM25 + `graph.json` in-process, one blob per user subject.**
Compared for this deployment (Vercel serverless, Atlas M0 ~100 ops/s, many users, $0 budget):

| | in-process numpy+BM25+graph (chosen) | Atlas Vector Search (`$vectorSearch`, `$graphLookup`) | pgvector (Supabase/Neon) |
|---|---|---|---|
| Mongo ops per chat turn | 0 when the instance is warm (RAM LRU), 1 version check / revalidation window | >= 1 per query (2-4 per turn: main + parts + hops) against the 100 ops/s cap | 0 Mongo, but a second database |
| Cold load of a user subject | 1 GridFS read of the zip: 0.9 MB / 500 chunks, 8.7 MB / 5k (float16 vectors + chunks + graph); unpack + load 5 ms / 38 ms; graph parse 4 / 18 ms | none, but every query pays a 30-80 ms round trip from the function region | none, same round trip |
| Query latency | hybrid rank 0.05 ms @135 chunks, 0.7 ms @10k; graph 0.2-0.9 ms | 30-80 ms network + ANN | 20-60 ms |
| Hybrid + graph | RRF of dense, BM25 and PPR in one numpy pass | BM25 needs a separate Atlas Search index; fusion client side (no `$rankFusion` on M0); PPR is iterative, `$graphLookup` is a parent-pointer walk (one hop per round trip) | would need a tsvector + custom SQL |
| Storage | 8.7 MB zip per 5k-chunk subject | one 768-float document (6 KB) per chunk = ~30 MB per 5k chunks of the 512 MB M0, + index build lag (seconds, eventually consistent) | free tier 500 MB, new infra |
| Isolation | the owner-checked `_get_index` (existing tests) | owner filter must be on every query | row-level security to add |

Atlas Vector Search would ADD Mongo load per request and per-query latency, and it cannot run the graph
step, so it is not implemented behind a flag (it would be dead weight on M0). Revisit only if a single user
subject routinely exceeds ~50k chunks (the 20 ms hybrid rank and ~80 MB RAM stop being cheap) or the
deployment moves to a dedicated cluster (M10+, `$rankFusion`).


---

## 5. Persistence (`backend/db.py`, optional MongoDB)

Active only when `MONGODB_URI` is set; otherwise every function is a graceful no-op.

- **History** — chats, quizzes, flashcards, plans, mindmaps, summaries, scoped per `owner`.
- **Uploaded PDFs** — stored in GridFS and re-hydrated into `documents/` on startup, so
  uploads survive restarts on a diskless host (e.g. Render's free tier). Owner-scoped;
  owner-less legacy records are never restored (they would leak as bundled subjects).
- **FAISS indexes** — user subjects' index files persist in GridFS and restore on startup,
  so a redeploy never re-embeds (no Gemini cost, no rebuild).

`owner` is the Clerk user id (`backend/auth.py`), or an anonymous guest id
(`guest_<uuid>`, auto-expiring via a Mongo TTL index), or `"local"` when auth is off.

---

### 5a. Database load, the shared cache, and the "advanced database" evaluation

**Measured on this app** (200 scripted guest sessions, fake LLMs, real Atlas M0, one local uvicorn worker):

| | before this work | after |
|---|---|---|
| Mongo ops per full session (touch, subjects, 2 history lists, chat start + 3 turns, quiz, flashcards, progress) | 12.0 | see LOADTEST.md section 9 (7-8) |
| ops per idle open dashboard (4 requests / 4 s) | ~0.18 ops/s, 7.9 ops/user per 24 requests | ~2.5 ops/user per 24 requests |
| M0 ceiling (probe, `find_one` loop) | ~100 ops/s | unchanged |

How: one `$unionWith` aggregation per owner (`db.warm_owner`) serves every history list *and* the dashboard; the app's own
writes are patched into it instead of invalidating it; the guest touch is a single `findAndModify` that also returns a
`hasUploads` flag (a guest that never uploaded skips the GridFS aggregation); the chat document is created by the first
turn's upsert (and carries its LLM title), so `chat/start` is free. What remains per session is the floor: 1 guest
validation, 1 snapshot read, 3 chat-turn writes, quiz and flashcards saves (writes are not cacheable and must be durable).

| option | throughput ceiling | monthly cost | migration effort | verdict for this app |
|---|---|---|---|---|
| Atlas M0 + this cache (now) | ~100 ops/s => ~14 full sessions/s per cluster, more with Redis (shared snapshots, touch markers) | $0 | none | enough for hundreds of concurrent users |
| Atlas M10 | no ops cap (provisioned 2 GB RAM, ~1-3K ops/s typical), 3-node replica set | ~$57 (dedicated) | zero code change: change `MONGODB_URI`, `mongodump`/`mongorestore` | **first paid step** when sustained load passes ~8 sessions/s or M0's 512 MB fills |
| Neon/Supabase Postgres + pgvector | thousands of qps | free tiers exist (0.5 GB / 500 MB); Pro ~$19-25 | high: ~14 collections of nested documents (chat messages, quiz questions) -> JSONB tables, GridFS PDFs/indexes -> object storage, rewrite `db.py` and 8 services' queries and the test fake | not justified: the access pattern is document-shaped, owner-keyed and mostly point reads; the vector index is a small per-subject file, not a server-side ANN workload |
| Hybrid (Mongo for documents, Postgres only for vectors) | n/a | n/a | medium | no benefit: retrieval already runs from an in-process index shipped with the deployment |

**Recommendation:** stay on M0 + the shared Redis cache (both free), keep the op budget as a regression test
(`tests/test_cache.py::TestSessionOpBudget`), and when `command` p95 stays above ~150 ms under real traffic move to Atlas
M10 (connection-string change only). Do not migrate to Postgres. Upgrade steps for the rest: (1) enable Upstash (see
DEPLOYMENT.md); (2) M10: create the cluster, restore, set `MONGODB_URI`, raise `MONGO_MAX_POOL`; (3) with M10 the snapshot
TTL and the cache can stay as they are (they only reduce cost).

## 6. Request flow examples

**Chat message** → `POST /api/chat/message` → `chat.ask()` → `chat.stream_turn()` → RAG pipeline (§3a)
→ reply, `sources` and `citations` saved to history (if enabled).

**Generate a quiz** → `POST /api/quiz/generate` → `read_pdf(subject, pdf)` +
`clip_to_tokens()` produce the content → `services/quiz.py::make_quiz()` calls Groq
(JSON mode), validates each question → saved to history → returned.

Errors are normalized by global handlers in `main.py`: rate limit → 429, other Groq errors
→ 502, missing key → 503, unknown subject/PDF → 404, summary/parse failures → 502.

---

## 7. Packaging & deployment

**`Dockerfile`** (multi-stage): a *builder* installs deps into a venv; the *runtime* copies
the venv + `backend/` + `frontend/` + bundled `documents/` + pre-baked `faiss_index/`,
drops to a non-root user, and **pre-bakes the tokenizer** (tiktoken `cl100k_base`) into a
writable cache so the first request needs no download (embeddings are hosted on Gemini —
no local model to bake). Runs under **tini** →
`docker-entrypoint.sh` (seeds bundled PDFs into `documents/` without clobbering) →
**gunicorn** with Uvicorn workers (`gunicorn.conf.py`, binds `0.0.0.0:$PORT`).

**`render.yaml`** (Blueprint): builds the Dockerfile, health-checks `/api/health`, injects
`PORT`, sets all non-secret env, and prompts for secrets (`GROQ_API_KEY` required;
`MONGODB_URI`/Clerk optional). On the free tier, set `MONGODB_URI` (Atlas free tier) to
persist history, uploads, and summaries across restarts. Full guide: `DEPLOYMENT.md`.

---

## 8. Directory map

```
backend/
  main.py            FastAPI app, lifespan, routers, exception handlers
  config.py          all env reading + validation (single source of truth)
  agentic_rag.py     chat RAG orchestration (plan → retrieve → confidence → adaptive → generate → verify)
  rag/               query understanding, hybrid retrieval/rerank/MMR, grounding + citation checks, metrics
  gemini_embeddings.py hosted Gemini embeddings client (LangChain-compatible)
  vector_store.py    per-subject FAISS index (embeddings) for chat retrieval
  documents.py       PDF read/list/save + restore; traversal-safe paths
  db.py              optional MongoDB: history, uploaded PDFs + indexes (GridFS)
  auth.py            optional Clerk JWT verification + owner resolution
  services/          chat, quiz, flashcards, mindmap, planner, subjects (business logic)
  routers/           one router per tool (the /api/* endpoints)
frontend/            vanilla-JS SPA (index.html + assets/)
documents/           bundled subject PDFs (seeded into the container on start)
tests/               unittest suite (stdlib + pytest runner)
Dockerfile, docker-entrypoint.sh, gunicorn.conf.py, render.yaml, DEPLOYMENT.md
```

---

## Config reference

| Variable | Required | Purpose |
|---|---|---|
| `APP_ENV` | yes | `development` or `production` |
| `HOST`, `PORT` | yes | bind address (Render injects `PORT`) |
| `LLM_PROVIDER` | yes | `groq` |
| `GROQ_API_KEY` | yes | Groq secret |
| `AVAILABLE_MODELS`, `DEFAULT_MODEL` | yes | model menu + default |
| `EMBEDDING_PROVIDER` | no (`gemini`) | `gemini` (hosted) or `local` (fastembed fallback) |
| `GEMINI_API_KEY` | with `gemini` | Gemini embeddings secret |
| `EMBEDDING_MODEL` | yes | embedding model id for the active provider |
| `MAX_CONTENT_TOKENS` | no (7000) | per-call content budget for the generation tools |
| `AGENTIC_RAG` | no (true) | enable the agentic chat loop; `false` = classic RAG |
| `AGENTIC_RAG_MAX_ITERS` | no (1) | legacy LLM rewrite/grade rounds (opt-in) |
| `AGENTIC_RAG_QUERIES` | no (2) | search queries per round |
| `AGENTIC_RAG_POOL` / `_FINAL_K` / `_LOW_CONF` / `_WEAK_CONF` / `_HOPS` / `_ADAPTIVE` | no | RAG pipeline knobs, see §3a |
| `CHAT_MODEL` … `MINDMAP_MODEL`, `SUMMARY_MODEL` | no | per-tool model overrides |
| `MONGODB_URI`, `MONGODB_DB_NAME` | no | enable persistence (history, uploads, summaries) |
| `CLERK_PUBLISHABLE_KEY`, `CLERK_SECRET_KEY` | no | enable the sign-in gate / API JWT verification |
| `CORS_ORIGINS`, `FRONTEND_PORT`, `API_BASE_URL` | no | split-origin mode (single-server by default) |
