# 🪟 Study Assistant Hub — Liquid Glass

An AI study companion with **five tools** — **Chat**, **Quiz**, **Flashcards**, **Study Planner** and **Mind Map** — built on your own lecture PDFs. One fast **FastAPI** server, a hand-built **Apple-style “Liquid Glass”** single-page UI, and **Groq** for all LLM inference. Embeddings run locally, so the only secret you need is a single `GROQ_API_KEY`. No Streamlit, no OpenAI.

> Rebuilt from a four-Streamlit-app prototype into a single SPA + API: one process, instant navigation, a real-time refraction/glass UI, and one key to run everything.

---

## 📑 Table of contents
1. [Features](#-features)
2. [Tech stack](#-tech-stack)
3. [Quick start (run it in 4 steps)](#-quick-start)
4. [Using the app](#-using-the-app)
5. [Add your own study material](#-add-your-own-study-material)
6. [How it works](#-how-it-works)
7. [Configuration (`.env`)](#-configuration-env)
8. [Rate limits & large PDFs](#-rate-limits--large-pdfs)
9. [Project structure](#-project-structure)
10. [API reference](#-api-reference)
11. [Running the tests](#-running-the-tests)
12. [Deploy to production (Docker)](#-deploy-to-production-docker)
13. [Troubleshooting](#-troubleshooting)
14. [Author](#-author)

---

## ✨ Features

Every tool can work from one of **three sources** (just like a segmented toggle at the top of each tool):

| Source | What it means |
|---|---|
| **Subject** | Use the PDFs you’ve filed under a subject (`documents/<Subject>/`). |
| **Topic** | Type a free topic — the model generates from its own knowledge, no document needed. |
| **Upload PDF** | Upload a one-off PDF and work from its contents (not saved). |

The tools:

- **💬 Chat Assistant** — Ask questions and get grounded explanations. In Subject/Upload mode it uses **RAG** (local embeddings + FAISS retrieval) so answers are tied to your material; it also **remembers the last few turns** of the conversation.
- **📝 Quiz Generator** — Multiple-choice quizzes with a one-question-at-a-time runner, instant scoring, explanations, and a review screen.
- **🎴 Flashcards** — Active-recall cards you can flip, rate your confidence on, and review (it resurfaces the ones you didn’t know).
- **📅 Study Planner** — Per-PDF summaries plus a day-by-day markdown study plan paced to your target date (downloadable as `.md`).
- **🧠 Mind Map** — An interactive, zoomable concept map (central idea → branches → points) from a PDF or a topic.
- **➕ Add a Subject** — Upload PDFs to create or extend a subject; its FAISS index is built on the spot so every tool can use it immediately.

Plus: a **light/dark theme** toggle (defaults to light; your choice is remembered), fully **responsive** layout (phones → desktops), and a safe in-house markdown renderer (code blocks with copy, tables, lists).

## 🛠 Tech stack
- **Backend:** FastAPI + Uvicorn — a single server that serves the SPA *and* the `/api/*` routes.
- **Frontend:** vanilla HTML/CSS/JS, **zero build step**. “Liquid Glass” design system (`backdrop-filter` + SVG `feDisplacementMap` refraction), light/dark adaptive, responsive.
- **LLM:** **Groq** via the official `groq` SDK and `langchain-groq`. **No model ids are hardcoded** — the models offered in the UI and the per-tool defaults all come from `.env` (`AVAILABLE_MODELS`, `DEFAULT_MODEL`, optional per-tool overrides). See [Configuration](#-configuration-env).
- **Retrieval (Chat only):** LangChain + **FAISS** with **local, keyless `fastembed`** embeddings (`BAAI/bge-small-en-v1.5`).
- **PDF text:** `pypdf`.

---

## 🚀 Quick start

**Prerequisites:** Python **3.11+** and a free **Groq API key** (get one at <https://console.groq.com/keys>).

1. **Clone & enter**
   ```bash
   git clone https://github.com/gpr-27/study-assistant-hub.git
   cd study-assistant-hub
   ```
2. **Create a virtualenv & install**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate          # Windows: .venv\Scripts\activate
   pip install -r requirements-rag.txt   # full app: base deps + the local retrieval stack
   ```
   > `requirements.txt` is the **slim** base (used by the serverless/Vercel build and CI);
   > `requirements-rag.txt` adds the heavy local-embeddings + FAISS stack and pulls the base
   > in, so the one command above installs everything for the full local app.
3. **Create your `.env`**
   ```bash
   cp .env.example .env               # then edit .env and fill it in
   ```
   `.env.example` ships with sensible values already in place — the one thing you **must** edit is `GROQ_API_KEY=gsk_...`. Every value the app reads is declared in `.env.example`; the server **validates configuration at startup** and refuses to start if anything required is missing or invalid, printing each problem as `❌ Missing/invalid configuration: <VAR>`. See [Configuration](#-configuration-env).
4. **Run**
   ```bash
   python run.py                      # or: uvicorn backend.main:app --reload
   ```
   `python run.py` reads `HOST`/`PORT` from `.env`; with the defaults, open **<http://127.0.0.1:8000>**.

> **First run** downloads a small (~67 MB) local embedding model the first time you use **Chat** in Subject/Upload mode. It’s cached afterwards, so later starts are instant. `run.py` auto-relaunches under the project’s `.venv` if you start it with the wrong Python.

---

## 📖 Using the app

Open the hub and pick a tool. At the top of each tool is a **Subject / Topic / Upload PDF** toggle — choose where the content comes from, then go.

### 💬 Chat Assistant
1. Pick **Subject** (chat across all of a subject’s PDFs), **Topic** (a tutor for any topic), or **Upload PDF**.
2. Click **Start session**, then ask questions in the composer.
3. Answers render as rich markdown (with copy buttons). The assistant keeps the **last few turns** as context, so follow-ups like *“explain that more simply”* work. *(Memory is in-session only — it resets on reload or server restart.)*

### 📝 Quiz Generator
1. Choose a source, set **difficulty** and **number of questions**, click **Generate quiz**.
2. Answer one question at a time — you’ll see if you were right, an explanation, and your running score.
3. Finish to get a percentage, a summary, and a per-question review.

### 🎴 Flashcards
1. Choose a source, set difficulty and card count, click **Generate flashcards**.
2. **Click a card to flip it.** Rate each card *Didn’t know / Somewhat / Knew well*.
3. Switch to **Review** mode to drill only the cards you haven’t mastered. **Reset** clears progress.

### 📅 Study Planner
1. **Subject mode:** pick a subject, tick the PDFs, set a **target date**. Optionally **Preview summaries** first, then **Generate plan**.
2. **Topic mode:** type a topic + target date → a from-scratch learning plan.
3. **Upload mode:** upload one or more PDFs + a target date.
4. The plan renders as markdown and can be **downloaded as `.md`**.

### 🧠 Mind Map
1. Choose a PDF, an uploaded file, or a topic, click generate.
2. Explore the map: **drag** to pan, use the **+/–** controls to zoom, and **expand/collapse** branches.

### ➕ Add a Subject
1. Open **Add subject** (top-right), enter a name, and upload one or more PDFs.
2. The PDFs are saved under `documents/<Subject>/` and embedded immediately — the new subject appears in every tool.

---

## 📚 Add your own study material

Two ways to add content:

- **In the UI:** use **Add a Subject** (above) — best for quick uploads.
- **On disk:** drop files into `documents/<Subject>/your-file.pdf`. The subject appears automatically; the Chat index for that subject builds on first use (or rebuilds when the files change).

```
documents/
  Automata/        Lec-1.pdf  Lec-2.pdf  Lec-3.pdf
  Mathematics/     Lec-1.pdf  Lec-2.pdf
  Economics/       ...
```

> PDFs must contain a **text layer**. Scanned/image-only PDFs extract no text — run them through OCR first.

---

## 🧠 How it works

**Two different content strategies, by design:**

- **Chat uses embeddings (RAG).** Each subject’s PDFs are split into chunks, embedded **once** with the local `fastembed` model, and saved to `faiss_index/<Subject>/`. On later runs the index is **reloaded from disk** (a `manifest.json` fingerprint rebuilds it only when the PDFs or settings change). On each question, only the **most relevant chunks** are retrieved and sent to Groq.

- **Quiz / Flashcards / Planner / Mind Map read the PDF text directly.** These tools generate from the *whole* document, so retrieval doesn’t fit. They extract the text with `pypdf` and send a **single bounded slice** to Groq in **one call** — they do **not** fan out into many calls.

**The single-call budget — `MAX_CONTENT_TOKENS` (default 7,000):**
- If a document **fits** within the budget, it’s sent **in full**.
- If it’s **larger**, only the **first `MAX_CONTENT_TOKENS`** tokens are used.
- Either way it’s **one request**, which keeps it under Groq’s per-minute token limit.

> **Pages ≠ tokens.** A 62-page *slide deck* is often only ~7k tokens (~111 tokens/page) and is sent whole; a dense 60-page *paper/textbook* can be ~40k tokens and gets capped. Text density is what matters, not page count.

---

## ⚙️ Configuration (`.env`)

**All configuration is centralized.** The backend reads environment values in exactly one place — **`backend/config.py`** (the only file that calls `os.getenv`) — and the frontend reads server config in exactly one place — **`frontend/assets/config.js`**, which fetches it once at boot via **`GET /api/config`**. No view, router, or service hardcodes a model id, host, port, or provider; they all read from these two layers.

**Setup:** copy `.env.example` → `.env` and fill it in. `.env.example` declares **every** value the app needs, with comments. The server **validates configuration at startup** and **refuses to start** if any required value is missing or invalid — it prints each problem as a line like:

```
  ❌ Missing/invalid configuration: GROQ_API_KEY (get one at https://console.groq.com/keys)
```

Fix what it lists, then restart. (There are no hidden in-code fallbacks for required values.)

### Environment variables

**Runtime**

| Variable | Example | Purpose |
|---|---|---|
| `APP_ENV` | `development` | `development` or `production`. Controls auto-reload (dev = on, prod = off). **Switching dev ⇄ prod is only this one change — no code edits.** |
| `HOST` | `127.0.0.1` | Address the server binds to. |
| `PORT` | `8000` | TCP port the server listens on. |
| `LOG_LEVEL` | `INFO` | `DEBUG` \| `INFO` \| `WARNING` \| `ERROR` \| `CRITICAL`. |

**Frontend / CORS**

| Variable | Example | Purpose |
|---|---|---|
| `CORS_ORIGINS` | *(empty)* | Comma-separated allowed origins. **Empty = same-origin** (the default — the SPA is served by this same server, so no CORS is needed). Set it to the SPA's origin when you run the frontend on its own port (below). |
| `FRONTEND_PORT` | *(empty)* | **Empty = single-server mode** (one server serves both SPA + API). Set it to also serve the SPA on its own port — see [Split-origin frontend](#split-origin-frontend-optional). |
| `API_BASE_URL` | `http://127.0.0.1:8000/api` | Absolute API URL the browser uses in split-origin mode. **Required when `FRONTEND_PORT` is set**, ignored otherwise. |

**Accounts, history & guests** *(all optional — the app runs fully without them)*

| Variable | Example | Purpose |
|---|---|---|
| `CLERK_PUBLISHABLE_KEY` | `pk_test_…` | **Empty = open app** (no sign-in). Set it to put a Clerk sign-in gate in front of the app. PUBLIC key (safe in the browser). |
| `CLERK_SECRET_KEY` | `sk_…` | *(optional)* Reserved for backend session-token verification. SECRET. |
| `MONGODB_URI` | `mongodb+srv://…` | **Empty = no history** (nothing saved; app identical). Set it to save & browse every chat, quiz, flashcard deck, plan and mind map, scoped per user. SECRET. |
| `MONGODB_DB_NAME` | `study_assistant_hub` | Database name (created on first write). |
| `GUEST_TTL_DAYS` | `30` | Inactive **guest** accounts are auto-deleted this many days after last activity (MongoDB TTL). |
| `GUEST_PURGE_INTERVAL_SECONDS` | `86400` | How often the background janitor reaps artifacts left behind by a TTL-expired guest. |

**LLM provider & key**

| Variable | Example | Purpose |
|---|---|---|
| `LLM_PROVIDER` | `groq` | The LLM provider. `groq` is the only provider with a client integration today. |
| `GROQ_API_KEY` | `gsk_...` | Your Groq key (required when `LLM_PROVIDER=groq`). Get one at <https://console.groq.com/keys>. |

**Models** *(no model ids are hardcoded anywhere — they live here)*

| Variable | Example | Purpose |
|---|---|---|
| `AVAILABLE_MODELS` | `llama-3.3-70b-versatile,llama-3.1-8b-instant,…` | Comma-separated list of model ids offered in the UI's **Model** dropdown. Each entry may optionally carry an explicit label as `id\|Label` (e.g. `openai/gpt-oss-120b\|GPT-OSS 120B`); without one, a readable label is derived from the id. |
| `DEFAULT_MODEL` | `llama-3.3-70b-versatile` | The default model. **Must be one of `AVAILABLE_MODELS`** (validated at startup). |
| `CHAT_MODEL` | *(unset)* | Optional per-tool override. If unset, the tool uses `DEFAULT_MODEL`. Any value set must also appear in `AVAILABLE_MODELS`. |
| `QUIZ_MODEL` | *(unset)* | Optional override for Quiz. |
| `FLASHCARDS_MODEL` | *(unset)* | Optional override for Flashcards. |
| `PLANNER_MODEL` | *(unset)* | Optional override for the study plan. |
| `SUMMARY_MODEL` | *(unset)* | Optional override for per-PDF summaries. |
| `MINDMAP_MODEL` | *(unset)* | Optional override for Mind Map (a lighter model like `llama-3.1-8b-instant` is a sensible choice). |

**Embeddings & generation tuning**

| Variable | Example | Purpose |
|---|---|---|
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Local, keyless `fastembed` model for Chat retrieval (downloaded once). |
| `MAX_CONTENT_TOKENS` | `7000` | Max document text per generation (quiz/flashcards/planner). See [How it works](#-how-it-works). |

### Dynamic model selector

Every tool has a **Model** dropdown that is populated at runtime from `AVAILABLE_MODELS` (delivered to the SPA via `GET /api/config`). **Add or remove a model in `.env` and the UI updates with no code changes** — and there's no way for a request to pick a model that isn't in your list (an out-of-list choice is ignored and the configured default is used). Per-tool overrides above let one tool default to a different model without touching code.

### Accounts & anonymous guests

Auth and history are **layered and optional**, and combine cleanly:

- **No Clerk key** → the app runs open; everything is stored under a single `local` profile (or nothing, if `MONGODB_URI` is also unset).
- **Clerk key set** → a sign-in gate appears. Each signed-in user sees only their own history.
- **Clerk key *and* `MONGODB_URI` set** → the gate also offers **“Continue as guest.”** A guest gets an anonymous, MongoDB-backed identity (`guest_<uuid>`, stored in the browser’s `localStorage`) and can use every tool exactly like a signed-in user — their chats, quizzes, flashcards, plans and mind maps are saved and restored across refreshes.

Everything is scoped by one opaque `owner` string (Clerk user id, `guest_<uuid>`, or `local`), so all three identities flow through the same per-user persistence with no special-casing.

**Upgrading a guest.** If a guest later signs in or signs up with Clerk, all their data is **migrated** to the Clerk account automatically (artifacts are re-keyed `owner: guest → clerk`, then the guest record is retired). The migration is ordered to be crash-safe and resumable — an interrupted upgrade never loses data and a later page load completes it.

**Cleanup.** A MongoDB **TTL index** on `guests.lastActiveAt` deletes guests inactive for `GUEST_TTL_DAYS` (default 30); `lastActiveAt` is refreshed on every guest request, so active guests never expire. A small background janitor sweeps any artifacts left behind by an expired guest.

Guests deliberately do **not** see Clerk account/profile management — they get a “Guest” badge, an **Exit** action, and a persistent CTA: *“Create an account to permanently save and sync your work.”*

### Split-origin frontend (optional)

By default **one server** serves both the SPA and the API on `PORT` (same origin, no CORS). To run the **frontend on its own port**, set `FRONTEND_PORT` (plus `API_BASE_URL` and `CORS_ORIGINS`):

```env
PORT=8000
FRONTEND_PORT=5173
API_BASE_URL=http://127.0.0.1:8000/api
CORS_ORIGINS=http://127.0.0.1:5173
```

`python run.py` then serves the API on `:8000` and the static SPA on `:5173`; the browser calls the API cross-origin at `API_BASE_URL` (the SPA learns it from an injected `window.__API_BASE__`, so no URL is hardcoded). Leave `FRONTEND_PORT` blank for classic single-server mode — fully backward compatible.

### Dev ⇄ prod

The only difference between development and production is `APP_ENV`, which drives auto-reload (`development` → reload on; `production` → reload off). Flip that single value to switch — no code changes.

Changes to `.env` take effect on the **next server start**.

---

## 🚦 Rate limits & large PDFs

Groq’s **free tier** has small per-minute token limits (**TPM**). Pick a model whose TPM comfortably exceeds `MAX_CONTENT_TOKENS` + prompt + output:

| Model | TPM (tokens/min) |
|---|---|
| `llama-3.3-70b-versatile` | **12,000** |
| `openai/gpt-oss-120b` | 8,000 |
| `llama-3.1-8b-instant` | 6,000 |

If you hit a **413 “request too large”** or **429 “rate limit”**:
- Keep `MAX_CONTENT_TOKENS` below your model’s TPM (the default 7,000 fits the 8k/12k models).
- Or pick a higher-TPM model — switch it live in the tool’s **Model** dropdown, set it as `DEFAULT_MODEL`, or set a per-tool override (e.g. `QUIZ_MODEL`) in `.env`. With a 12k-TPM model you can also raise `MAX_CONTENT_TOKENS` (e.g. ~10,000) for fuller coverage.
- The app surfaces these as friendly messages (`429` with guidance, `503` for a missing key) rather than opaque 500s.

---

## 📂 Project structure

```
backend/
  main.py            FastAPI app: serves the SPA + /api/health + /api/config, exception handlers, includes routers
  config.py          THE config layer: the only file that reads os.getenv — env vars, model resolution, MAX_CONTENT_TOKENS, startup validation
  auth.py            owner resolution: Clerk session → guest (X-Guest-Id) → local; require_clerk_user
  db.py              optional MongoDB persistence: per-owner artifacts + guest accounts (TTL, migration, janitor)
  documents.py       subjects / PDFs / text extraction + traversal-safe path guards + token clipping
  vector_store.py    persistent per-subject FAISS index (build-once, reload-from-disk) — Chat only
  schemas.py         Pydantic request/response contract
  routers/           chat.py  quiz.py  flashcards.py  planner.py  mindmap.py  subjects.py  history.py  guest.py
  services/          chat.py  quiz.py  flashcards.py  planner.py  mindmap.py  subjects.py  (Groq logic)
frontend/
  index.html         liquid-glass SPA shell (auth gate: Sign in · Create account · Continue as guest)
  assets/config.js   THE frontend config layer: the only file that reads GET /api/config → window.AppConfig
  assets/auth.js     Clerk + anonymous-guest gate: sign-in, guest create/restore, account-upgrade migration
  assets/styles.css  design system + component kit + responsive layer
  assets/app.js      hash router + api()/ui() helpers (attaches Clerk token + X-Guest-Id) + model selector + markdown + theme
  assets/views/      chat.js  quiz.js  flashcards.js  planner.js  mindmap.js  subjects.js
documents/<Subject>/*.pdf       your study material
faiss_index/<Subject>/          gitignored, rebuildable embedding indexes
tests/                          pytest suite
run.py                          launcher (re-execs into .venv, runs uvicorn; also starts the SPA server in split-origin mode)
frontend_server.py              static SPA server for split-origin mode (serves the SPA on FRONTEND_PORT)
Dockerfile                      production image (multi-stage; gunicorn+uvicorn; pre-baked models; non-root)
docker-compose.yml              production-style run (app + optional MongoDB profile; volumes for indexes/uploads)
docker-entrypoint.sh            seeds the documents dir, then execs gunicorn
gunicorn.conf.py                env-driven gunicorn config (workers, timeouts, logging)
.dockerignore                   keeps the build context/image lean and secret-free
.env.production.example         production .env template (see also .env.example)
requirements-dev.txt            runtime deps + pytest (test/CI only)
.github/workflows/              CI (tests) + Docker (build & publish image to GHCR)
DEPLOYMENT.md                   full production deployment guide
```

---

## 🔌 API reference

All endpoints are under `/api`. The SPA uses them, but they’re also callable directly.

| Method & path | Body | Purpose |
|---|---|---|
| `GET /api/health` | — | `{status, groq_key}` |
| `GET /api/config` | — | Public, non-secret bootstrap config the SPA reads (provider, default model, `availableModels`, app env). Driven entirely by `.env`. |
| `GET /api/subjects` | — | List subjects |
| `GET /api/subjects/{subject}/pdfs` | — | List a subject’s PDFs (with page counts) |
| `POST /api/subjects` | multipart `name`, `files[]` | Add/extend a subject and build its index |
| `POST /api/chat/start` | `{mode, subject?, topic?}` | Start a subject/topic chat session |
| `POST /api/chat/upload` | multipart `file` | Start a chat session from an uploaded PDF |
| `POST /api/chat/message` | `{session_id, message}` | Send a message; returns the answer |
| `POST /api/quiz/generate` | `{subject, pdf, num_questions, difficulty}` | Quiz from a subject PDF |
| `POST /api/quiz/topic` | `{topic, num_questions, difficulty}` | Quiz from a topic |
| `POST /api/quiz/upload` | multipart `file`, `num_questions`, `difficulty` | Quiz from an uploaded PDF |
| `POST /api/flashcards/generate` \| `/topic` \| `/upload` | (same shape as quiz, `num_cards`) | Flashcards from subject / topic / upload |
| `POST /api/planner/summaries` | `{subject, pdfs[]}` | Per-PDF summaries |
| `POST /api/planner/generate` | `{subject, pdfs[], target_date}` | Plan from a subject |
| `POST /api/planner/topic` | `{topic, target_date}` | Plan from a topic |
| `POST /api/planner/upload` | multipart `files[]`, `target_date` | Plan from uploaded PDFs |
| `POST /api/mindmap/generate` | `{mode, subject?, pdf?, topic?}` | Mind map from a subject PDF or topic |
| `POST /api/mindmap/upload` | multipart `file` | Mind map from an uploaded PDF |
| `GET /api/history/{kind}` | `?q=` | List saved items of a kind (`chats`\|`quizzes`\|`flashcards`\|`plans`\|`mindmaps`), scoped to the caller |
| `GET\|PATCH\|DELETE /api/history/{kind}/{id}` | `{title}` (PATCH) | View / rename / delete one saved item |
| `POST /api/guest` | — | Create an anonymous guest (returns `{guestId, isGuest, …}`). Needs `MONGODB_URI`. |
| `GET /api/guest/me` | header `X-Guest-Id` | Restore a guest session (404 if expired/migrated) |
| `POST /api/guest/migrate` | headers `Authorization` + `X-Guest-Id` | Upgrade: move all guest data to the signed-in Clerk user |
| `GET /runtime-config.js` | — | Tiny script seeding `window.__API_BASE__` (same-origin or absolute in split mode) |

> Identity for every scoped route comes from the `Authorization: Bearer <clerk-jwt>` header, or the `X-Guest-Id: guest_<uuid>` header — Clerk wins when both are present.

---

## 🧪 Running the tests

The suite is pure-Python (no network, no Groq key, no real model — the LLM is mocked):

```bash
pip install pytest      # not in requirements.txt; install once in your venv
python -m pytest -q
```

It covers document/path-safety, the vector store, chat session errors, mind-map shaping, and LLM-error propagation.

---

## 🐳 Deploy to production (Docker)

The repo is production-ready and containerized. The image runs the app in
**single-server mode** (one FastAPI process serves the SPA **and** the API) with
**Gunicorn + Uvicorn workers**, a **non-root** user, **pre-baked local models**
(no first-request download), and a `/api/health` healthcheck.

```bash
cp .env.production.example .env     # fill in GROQ_API_KEY (+ optional Mongo/Clerk)
docker compose up --build -d        # build + run
# open http://localhost:8000   ·   health: http://localhost:8000/api/health
```

Your secrets come from `.env` at **run time** (never baked into the image);
production runtime mode and `0.0.0.0` binding are forced by `docker-compose.yml`.
Built FAISS indexes (`/app/faiss_index`) and uploaded subjects (`/app/documents`)
persist in named volumes.

### Two deployment modes

The app ships with **two** runtime modes, chosen automatically by the host:

| | **Full** (container / VPS) | **Lite** (serverless / Vercel) |
|---|---|---|
| Chat grounding | **RAG** — local `fastembed` embeddings + FAISS retrieval | **Direct document text** sent to the LLM (clipped to `MAX_CONTENT_TOKENS`) |
| Heavy deps (`faiss-cpu`, `fastembed`/onnxruntime) | installed (`requirements-rag.txt`) | **omitted** (`requirements.txt` only) |
| Background threads, writable disk | yes | no (state under `/tmp`, no warmup/janitor) |
| Quiz · Flashcards · Planner · Mind Map | ✅ identical | ✅ identical |
| Best for | richest Chat, persistent indexes | one-command deploy, autoscaling, zero servers |

The mode flips on a single flag — `RAG_ENABLED` (defaults to **off** when the platform sets
`VERCEL`, **on** otherwise). No code changes; see `backend/config.py`.

### Deploy to Vercel (lite mode)

The repo is Vercel-ready: `api/index.py` exposes the FastAPI app to the `@vercel/python`
runtime, `vercel.json` rewrites every path to it and bundles `frontend/` + `documents/`,
and the slim `requirements.txt` keeps the function under the size limit.

```bash
vercel            # preview deploy (or: vercel --prod)
```

Set the project's environment variables in the Vercel dashboard (or `vercel env add`):
`GROQ_API_KEY` (+ `LLM_PROVIDER`, `AVAILABLE_MODELS`, `DEFAULT_MODEL`), and optionally
`CLERK_PUBLISHABLE_KEY`/`CLERK_SECRET_KEY` and `MONGODB_URI` for auth + history. `VERCEL=1`
is set by the platform, so the app boots straight into lite mode.

> **When to prefer a container instead.** Full RAG chat, persistent FAISS indexes, and
> durably-saved uploaded subjects want a long-lived process — deploy the Docker image to
> **Render / Railway / Fly.io / Cloud Run / a VPS**. On serverless, uploaded subjects are
> ephemeral (reset on cold start) and Chat grounds on document text rather than retrieval.

Two GitHub Actions are included: **CI** (tests on every push/PR) and **Docker**
(builds & publishes the image to GHCR on `main` / version tags).

**👉 Full guide:** [`DEPLOYMENT.md`](DEPLOYMENT.md) — compose & plain-`docker`
recipes, volumes, scaling/workers, TLS/reverse-proxy, per-platform notes, CI/CD,
and a security checklist.

---

## 🧰 Troubleshooting

| Symptom | Cause & fix |
|---|---|
| **`ERR_CONNECTION_REFUSED` / nothing on :8000** | Started with the wrong Python, so dependencies are missing. Use `python run.py` (it re-execs into `.venv`), or activate the venv first. |
| **`ModuleNotFoundError` after `git clone` on a new machine** | A copied/stale `.venv` points at a Python that doesn’t exist here. Recreate it: `rm -rf .venv && python3 -m venv .venv && pip install -r requirements.txt`. |
| **Server won’t start, prints `❌ Missing/invalid configuration: <VAR>`** | A required value in `.env` is missing or invalid (e.g. `GROQ_API_KEY`, `PORT`, `DEFAULT_MODEL` not in `AVAILABLE_MODELS`). Fix each listed line in `.env` (copy `.env.example` if you don’t have one) and restart. |
| **AI tools return 503 “GROQ_API_KEY is not set”** | The key was cleared after startup. Set `GROQ_API_KEY` in `.env` and restart. Check `GET /api/health` → `"groq_key": true`. |
| **413 “request too large” / 429 “rate limit”** | Your model’s per-minute token limit is too low for the request. See [Rate limits](#-rate-limits--large-pdfs) — lower `MAX_CONTENT_TOKENS`, or pick a higher-TPM model via the **Model** dropdown / `DEFAULT_MODEL` / a per-tool override in `.env`. |
| **No subjects listed** | Add PDFs via **Add a Subject**, or drop them under `documents/<Subject>/`. |
| **A PDF produces nothing** | It’s likely scanned/image-only (no text layer). OCR it first. |
| **First Chat use is slow** | One-time ~67 MB embedding-model download; cached afterwards. |

---

## 👤 Author
**Gandra Praneeth Reddy** — [LinkedIn](https://www.linkedin.com/in/praneeth-gandra-411ab3309/)
Built for the hackathon at **IIITT** by **TEAM404**.
