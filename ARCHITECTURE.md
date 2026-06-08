# Architecture

Study Assistant Hub is a single **FastAPI** server that serves a vanilla-JS single-page
app **and** the `/api/*` JSON API on one origin. Text generation runs on **Groq**;
embeddings run **locally** (keyless) via fastembed. Everything is driven by environment
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
        Groq (LLM)              fastembed + FAISS              MongoDB (optional)
     text generation         local embeddings / search     history · uploads · summaries
```

The only required secret is `GROQ_API_KEY`. MongoDB (persistence) and Clerk (auth) are
optional; without them the app runs fully, just open and stateless.

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
   - `warm_embeddings()` — load the fastembed model,
   - `warm_all_subjects()` — build/load each subject's FAISS index.
3. Start the guest janitor (cleans up expired anonymous accounts; no-op without MongoDB).

`/api/health` answers immediately (it's what Render probes) because warmup is off-thread.

---

## 3. Two content pipelines

There are two distinct ways content reaches the LLM. Don't confuse them.

### 3a. Chat → **Agentic RAG** (`backend/agentic_rag.py`)

Chat answers a *specific question*, so it uses retrieval — but as an **LLM-driven loop**,
not a single fetch:

```
rewrite ─▶ retrieve ─▶ grade ─┬─(sufficient)─▶ generate
   ▲                          │
   └────(insufficient)────────┘   (corrective re-retrieve, bounded)
```

1. **Rewrite** (`_rewrite`) — turn the history-aware question into up to
   `AGENTIC_RAG_QUERIES` focused search queries; resolves follow-ups, decomposes
   multi-part questions ("compare X and Y" → query X, query Y).
2. **Retrieve** (`_retrieve`) — run each query through the session's FAISS retriever;
   merge + de-dupe the chunks.
3. **Grade** (`_grade`) — ask the LLM whether the gathered context can fully answer the
   question, and what's missing. *This is the key agentic step naive RAG lacks.*
4. **Correct** — if insufficient and rounds remain (`AGENTIC_RAG_MAX_ITERS`), rewrite
   using the gap feedback and retrieve again; otherwise stop.
5. **Generate** (`_generate`) — answer grounded only in the gathered context (the persona
   says to admit "I don't know" when unsupported).

Wired into `backend/services/chat.py::ask()` behind `config.AGENTIC_RAG` (default on),
with the classic single-shot path kept as the fallback. Properties:

- **Bounded:** `AGENTIC_RAG_MAX_ITERS` caps re-retrieve rounds; `AGENTIC_RAG_QUERIES`
  caps queries/round → predictable cost (≈ 3–5 Groq calls/message vs 1 for naive).
- **Degrades gracefully:** malformed rewrite/grade output falls back to classic behavior.
- **Errors surface:** Groq rate-limit/API errors propagate to the global 429/502 handlers.
- **Toggle:** `AGENTIC_RAG=false` reverts to classic single-shot RAG.

```
NAIVE:    question ─▶ retrieve top-k once ─▶ answer
AGENTIC:  question ─▶ rewrite(N queries) ─▶ retrieve+merge ─▶ grade ──(loop if weak)──▶ answer
```

### 3b. Quiz / Flashcards / Mind map / Planner → **chunk summaries** (`backend/chunk_summaries.py`)

These tools generate over a *whole selected PDF*, not a specific question, so they don't
retrieve. Previously each request re-read the entire PDF and clipped it to the first
`MAX_CONTENT_TOKENS` — re-reading every time and dropping the document's tail.

Now each PDF is split into chunks **once** and every chunk gets a thorough, faithful,
extractive summary (definitions, formulas, terms, examples preserved). The compact set —
covering the whole document — is what the tools feed to the LLM thereafter.

Built once, cached at three layers, keyed by a **content fingerprint** (hash of the PDF
text + summary model):

- in-process **memory** (per worker),
- **disk** at `faiss_index/chunk_summaries/<fingerprint>.json` (content-addressed),
- **MongoDB** (when configured) — survives restarts/redeploys even with no disk.

Resolution order in `_resolve()`: memory → disk → MongoDB → build (LLM), backfilling
every layer. Because the key is a content hash, the **same bytes from any source** —
a saved subject PDF or a one-off `/upload` — share one cache entry, and a re-upload of an
identical file reuses the saved summaries instead of re-billing the LLM.

- **Saved subject PDFs:** `content_for(subject, pdf)`; built eagerly in the background when
  a subject is added (`services/subjects.py`) and lazily on first use otherwise.
- **One-off uploads:** `content_for_text(text, filename)` — same pipeline; summaries saved.

---

## 4. Embeddings & search index (`backend/vector_store.py`)

Each subject's PDFs are split into 1,500-char chunks, embedded with fastembed
(`BAAI/bge-small-en-v1.5`, local/keyless), and stored in a per-subject **FAISS** index at
`faiss_index/<subject>/`. A `manifest.json` fingerprints the PDFs + settings, so an index
rebuilds only when its PDFs (or the embedding model) actually change. This index backs the
chat retriever (pipeline 3a).

---

## 5. Persistence (`backend/db.py`, optional MongoDB)

Active only when `MONGODB_URI` is set; otherwise every function is a graceful no-op.

- **History** — chats, quizzes, flashcards, plans, mindmaps, scoped per `owner`.
- **Uploaded PDFs** — stored in GridFS and re-hydrated into `documents/` on startup, so
  uploads survive restarts on a diskless host (e.g. Render's free tier).
- **Chunk summaries** — cached by content fingerprint so restarts don't re-bill the LLM.

`owner` is the Clerk user id (`backend/auth.py`), or an anonymous guest id
(`guest_<uuid>`, auto-expiring via a Mongo TTL index), or `"local"` when auth is off.

---

## 6. Request flow examples

**Chat message** → `POST /api/chat/message` → `chat.ask()` → (agentic) `agentic_rag.answer()`
runs rewrite→retrieve→grade→correct→generate → reply saved to history (if enabled).

**Generate a quiz** → `POST /api/quiz/generate` → `content_for(subject, pdf)` returns the
cached chunk summaries → `services/quiz.py::make_quiz()` calls Groq (JSON mode), validates
each question → saved to history → returned.

Errors are normalized by global handlers in `main.py`: rate limit → 429, other Groq errors
→ 502, missing key → 503, unknown subject/PDF → 404, summary/parse failures → 502.

---

## 7. Packaging & deployment

**`Dockerfile`** (multi-stage): a *builder* installs deps into a venv; the *runtime* copies
the venv + `backend/` + `frontend/` + bundled `documents/`, drops to a non-root user, and
**pre-bakes the models** (fastembed embedding model + tiktoken `cl100k_base`) into a
writable cache so the first request needs no download. Runs under **tini** →
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
  agentic_rag.py     chat retrieval loop: rewrite → retrieve → grade → correct → generate
  chunk_summaries.py per-chunk PDF summaries (build-once, cache by content)
  vector_store.py    per-subject FAISS index (embeddings) for chat retrieval
  documents.py       PDF read/list/save + restore; traversal-safe paths
  db.py              optional MongoDB: history, uploaded PDFs (GridFS), summaries
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
| `EMBEDDING_MODEL` | yes | local fastembed model (pre-baked) |
| `MAX_CONTENT_TOKENS` | no (7000) | per-call content budget for the generation tools |
| `AGENTIC_RAG` | no (true) | enable the agentic chat loop; `false` = classic RAG |
| `AGENTIC_RAG_MAX_ITERS` | no (2) | corrective re-retrieve rounds |
| `AGENTIC_RAG_QUERIES` | no (3) | search queries per round |
| `CHAT_MODEL` … `MINDMAP_MODEL`, `SUMMARY_MODEL` | no | per-tool model overrides |
| `MONGODB_URI`, `MONGODB_DB_NAME` | no | enable persistence (history, uploads, summaries) |
| `CLERK_PUBLISHABLE_KEY`, `CLERK_SECRET_KEY` | no | enable the sign-in gate / API JWT verification |
| `CORS_ORIGINS`, `FRONTEND_PORT`, `API_BASE_URL` | no | split-origin mode (single-server by default) |
