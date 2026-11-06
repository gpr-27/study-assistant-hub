# Deploying to Vercel

Study Assistant Hub = a static SPA (`frontend/` -> CDN) + **one** Python serverless function
(`api/index.py`, the FastAPI `app`) + MongoDB Atlas + Clerk. The old Render/Docker setup is
retired (kept under `legacy/render/` for reference, unsupported).

```
browser --> Vercel CDN --- /, /assets/* ---> dist/ (built from frontend/)
                      \--- /api/*  --rewrite--> api/index.py (FastAPI, Python 3.12, bom1)
                                                   |-- MongoDB Atlas (AWS ap-south-1, pooled client)
                                                   |-- Clerk JWKS (cached 1 h)
                                                   `-- LLM chain (Anthropic -> Groq x2 -> Gemini x2)
```

## What `vercel.json` does
| Setting | Value / reason |
|---|---|
| `buildCommand` | `python3 scripts/build_frontend.py --out dist` copies `frontend/` to `dist/`, stamps every `/assets/...` URL with a content hash (`?v=`, via `scripts/stamp_assets.py`) (one build step; `frontend/` sources are never modified). |
| `outputDirectory` | `dist` (static files; nothing else is served from the filesystem). |
| `rewrites` | `/api/:path*` -> `/api/index` (the single FastAPI function; original path is preserved). |
| `headers` | `/assets/*?v=...` -> `public, max-age=31536000, immutable`; assets requested without `?v` and `/` -> `max-age=0, must-revalidate`; security headers + a `Content-Security-Policy` on non-API paths (allows jsDelivr, cdnjs, Google Fonts, Clerk `*.clerk.accounts.dev` / `*.clerk.com`, Cloudflare Turnstile). **If you switch Clerk to a production instance with a custom domain (`clerk.<yourdomain>`), add that host to `script-src`, `connect-src` and `frame-src` in `vercel.json`.** All API responses (including `/api/config`) get `Cache-Control: no-store` from the app. |
| `functions.api/index.py` | `maxDuration: 60`; `excludeFiles` keeps tests, `frontend/`, `scripts/`, `legacy/`, per-user data and docs out of the bundle. |
| `regions` | `["bom1"]` (Mumbai) - the Atlas `Cluster0` is AWS `ap-south-1`; the DB is hit several times per request, the LLM once, so co-locate with the DB. Change it if you move the cluster. |
| `crons` | `17 3 * * *` -> `GET /api/cron/janitor` (reaps data of TTL-expired guests, ensures Mongo indexes). Hobby plans allow daily crons only. |
| `.python-version` | `3.12`. |

**Memory/CPU**: with Fluid Compute (default for new projects) memory is set in the dashboard, not
`vercel.json`: *Project -> Settings -> Functions -> Memory* = **Standard (1 GB / 1 vCPU)** is plenty.
Bundle: runtime deps resolve to ~125 MB unzipped (limit 250 MB; Vercel docs quote 500 MB).

## Environment variables
Set for **Production** (and Preview if you want previews to work). Never commit values; the repo's
`.env` is git-ignored and `.env.example` holds blanks.

| Variable | Required | Notes |
|---|---|---|
| `ANTHROPIC_API_KEY` | chain step 1 | `claude-sonnet-5-5` only |
| `GROQ_API_KEY`, `GROQ_API_KEY_2` | chain steps 2, 3 | any empty step is skipped |
| `GEMINI_API_KEY`, `GEMINI_API_KEY_2` | chain steps 4, 5 + embeddings | keys must be restricted "auth keys" |
| `MONGODB_URI` | yes (history, uploads, guests) | Atlas -> Network Access must allow Vercel (`0.0.0.0/0`, or use Vercel's Secure Compute / Atlas Vercel integration) |
| `MONGODB_DB_NAME` | optional | default `study_assistant_hub` |
| `CLERK_PUBLISHABLE_KEY` | yes (auth) | public key; JWKS host is derived from it. **On Vercel the app fails closed without it**: every authenticated route answers `503` ("no sign-in configured") and `/api/health/deep` reports `checks.auth.ok=false`; it never falls back to the shared `local` owner (that fallback exists for local dev only) |
| `CLERK_SECRET_KEY` | recommended | reserved for server-side verification |
| `SESSION_SECRET` | **yes (prod)** | random 32+ bytes (`openssl rand -hex 32`); chat session tokens are HMAC-signed with it. **On Vercel the app fails closed without it** (chat answers 503 + `/api/health/deep` warns); it no longer derives a key from provider keys. Tokens expire after 24 h (re-minted automatically while a chat is active) |
| `RATE_LIMIT_USER_PER_MIN` / `_PER_HOUR`, `RATE_LIMIT_GUEST_PER_MIN` / `_PER_HOUR` | optional | per-owner limits on paid endpoints, defaults 30/300 (signed in) and 10/60 (guest); see "Abuse and spend control" below |
| `ANTHROPIC_DAILY_TOKEN_BUDGET`, `ANTHROPIC_OWNER_DAILY_TOKENS`, `ANTHROPIC_GUEST_DAILY_TOKENS`, `ANTHROPIC_MAX_REQUEST_INPUT_TOKENS` | optional | spend guards on the paid Anthropic step; defaults 2,000,000 / 400,000 / 100,000 tokens per UTC day and 120,000 estimated tokens per request; `0` = unlimited (see "Abuse and spend control") |
| `ALLOW_GEMINI_FOR_USER_CONTENT` | optional | default `0`; see "Privacy decision" below |
| `REQUEST_BUDGET_S`, `MAX_BODY_BYTES`, `TMP_CACHE_MAX_MB` | optional | per-request time budget (50 s), upload body cap (4.4 MB), `/tmp` cache cap (300 MB) |
| `CRON_SECRET` | yes | any long random string; Vercel sends it as `Authorization: Bearer ...` to cron routes. Without it `/api/cron/janitor` always answers 401 |
| `CORS_ORIGINS` | optional | comma-separated; leave empty (same-origin) unless the SPA is hosted elsewhere |
| `GUEST_TTL_DAYS`, `LOG_LEVEL`, `AGENTIC_RAG`, `EMBED_RPM_PER_KEY`, `INDEX_BUDGET_S` | optional | tuning knobs, see `.env.example` |

`VERCEL` / `VERCEL_ENV` are set by the platform (the app uses `VERCEL` to disable local static
serving and the `/docs` schema browser). `HOST`, `PORT`, `APP_ENV` are NOT needed on Vercel.

## First deploy (run these yourself - nothing here deploys automatically)
```bash
npm i -g vercel                 # once
vercel login
vercel link                     # create/link the project (framework preset: Other)

# Env vars: pipe the value on stdin so it never appears in shell history or logs.
printf '%s' "$ANTHROPIC_API_KEY" | vercel env add ANTHROPIC_API_KEY production
#   ...repeat for GROQ_API_KEY GROQ_API_KEY_2 GEMINI_API_KEY GEMINI_API_KEY_2 MONGODB_URI \
#      CLERK_PUBLISHABLE_KEY CLERK_SECRET_KEY, and:
openssl rand -hex 32 | vercel env add CRON_SECRET production
openssl rand -hex 32 | vercel env add SESSION_SECRET production
# Easiest bulk path: read each value from your local .env without echoing it:
#   for k in ANTHROPIC_API_KEY GROQ_API_KEY ...; do grep "^$k=" .env | cut -d= -f2- | tr -d '\n' | vercel env add $k production; done

vercel deploy                   # preview deployment -> URL printed
curl -s https://<preview-url>/api/health          # {"status":"ok"}
curl -s https://<preview-url>/api/health/deep     # Mongo ping + chain configuration (booleans only)
vercel deploy --prod            # or: vercel promote <preview-url>
```
After the first production deploy: Clerk dashboard -> add the Vercel domain to allowed origins/redirect
URLs; Atlas -> allow Vercel egress IPs; verify `vercel crons ls`.

Rollback: `vercel rollback` (or promote a previous deployment in the dashboard).

## Running locally
* **Plain dev server** (single process serves SPA + API, auto-reload):
  ```bash
  python3.12 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
  cp .env.example .env            # fill in keys
  .venv/bin/python run.py         # http://127.0.0.1:8000
  ```
* **Vercel parity** (`vercel dev` builds `dist/`, runs `api/index.py` through the Python runtime):
  ```bash
  vercel link && vercel env pull .env.local     # pulls dev env into a git-ignored file
  python3 scripts/build_frontend.py             # vercel dev serves dist/, so build it first
vercel dev                                     # http://localhost:3000
  ```
* **Cold-start / import cost**: `.venv/bin/python scripts/measure_cold_start.py` (fresh interpreters, prints
  `-X importtime` offenders).
* Tests: `.venv/bin/python -m pytest -q`.

## Abuse and spend control
Every paid endpoint spends real LLM money, so there are three layers:

1. **In-app limiter** (`backend/ratelimit.py`): per-owner token buckets, in memory, per instance. Defaults:
   signed-in **30 req/min and 300/hour**, guests **10 req/min and 60/hour** (`RATE_LIMIT_*` env vars; `0` disables a
   window). Applied to chat, quiz, flashcards, exam, cheatsheet, mindmap, planner, predictor, teach-back,
   scan-notes, insights and chat overview. Over the limit: `429 {"detail": ...}` with `Retry-After` (no `llm`
   block, no model was involved). Serverless instances do not share memory, so the real ceiling is
   "limit x live instances" - that is why layer 2 exists. `RATE_LIMIT_DISABLED=1` switches it off for load
   tests (`scripts/loadtest.py` sets it for its own server) and is **ignored whenever `VERCEL` is set**.
2. **Vercel Firewall rate limits** (edge, shared by all instances). Recommended rules (*Project -> Firewall ->
   Configure -> Custom Rules*, action **Rate Limit**, key **IP**, window = fixed/sliding as offered):

   | Rule | Match | Limit | Then |
   |---|---|---|---|
   | AI endpoints | Request Path matches `^/api/(chat\|quiz\|flashcards\|exam\|cheatsheet\|mindmap\|planner\|predictor\|teachback\|scannotes\|insights\|grade)` and Method is `POST` (or `GET` for insights) | 60 requests / 1 min per IP | 429 for 1 min |
   | Uploads | Request Path is one of `/api/scannotes/extract`, `/api/predictor/analyze`, `/api/chat/upload`, `/api/subjects`, `/api/subjects/uploads` and Method `POST` | 10 requests / 1 min per IP | 429 for 5 min |
   | Guest creation | Request Path is `/api/guest` and Method `POST` | 5 requests / 10 min per IP | 429 for 30 min |
   | Everything else | Request Path starts with `/api/` | 300 requests / 1 min per IP | 429 for 1 min |

   Also turn on **Bot Protection** (Challenge) and, if you see a flood, **Attack Challenge Mode**. Set a
   **spend limit / alert** on every provider key (Anthropic, Groq, Google) - the app cannot cap provider-side spend.
3. **Premium-model spend guards** (`backend/spend.py`, shared across instances, no Redis needed). Anthropic is the
   only paid step, so its tokens (prompt + completion, from the provider's actual `usage`, streaming included) are
   counted per UTC day in the Mongo collection `usage_daily` (TTL 3 days; Redis `INCRBY` instead when
   `CACHE_BACKEND` is shared). Env, `0` = unlimited: `ANTHROPIC_DAILY_TOKEN_BUDGET` (global, default 0 = unlimited),
   `ANTHROPIC_OWNER_DAILY_TOKENS` (signed-in, default 0), `ANTHROPIC_GUEST_DAILY_TOKENS` (guests, default 0) and
   `ANTHROPIC_MAX_REQUEST_INPUT_TOKENS` (per request, estimated chars/3.6, default 0). **All four default to 0 (no budget caps) at the owner's request; set them to enable spend guards**. Exhausted budget or oversize
   prompt: the chain skips Anthropic (`skipped: ["anthropic budget"]` / `["anthropic too-large"]`) and Groq/Gemini
   answer; if no other model may take an oversize prompt (e.g. only Anthropic + Gemini keys and uploaded content)
   the API returns **413** "This request is too large for the available models; use fewer or smaller files".
   Observability: `/api/health/deep` -> `checks.budgets` (`global_used`, `global_limit`, `pct`, `owners_tracked`,
   `anthropic_skipped_today`; no owner ids) and one INFO log line when a budget first trips each day; at 80% a
   warning is logged and the model badge popover shows "Premium model budget nearly used today" (`/api/config`
   `premiumBudget` booleans). Usage is flushed every ~20 s (or at 5% of a budget), so concurrent instances can
   overshoot by roughly one flush window of tokens; a Mongo/Redis outage fails open to per-instance counters.
   Still set a provider-side spend limit on the Anthropic key.
4. **Fail closed**: on Vercel a missing `CLERK_PUBLISHABLE_KEY` or `SESSION_SECRET` is refused with a clear 503
   (see the table above) instead of serving everyone as one shared owner.

## Privacy decision (the lead's call)
Gemini's free tier may train on prompts. **Decision:** content a *user uploaded* is never sent to Gemini unless the
operator opts in. Concretely, with `ALLOW_GEMINI_FOR_USER_CONTENT=0` (the default) these flows call the provider
chain with `sensitive=True`, which skips both Gemini steps:

* scan-notes (every photo);
* chat, quiz, flashcards, exam (generate + grading), cheat sheet, mind map, planner (+ summaries), predictor and
  teach-back over a **user-uploaded subject** or an **uploaded PDF**, and every PDF chat (`chat_pdfs`);
* the chat overview card of a user subject.

Bundled subjects (the public sample PDFs) and generic topic-only requests stay non-sensitive and may use Gemini.
If only Gemini keys are configured, a sensitive request answers `503` with *"This request uses content you uploaded,
which is never sent to Google's free-tier Gemini models ... add an Anthropic or Groq key"* and the model label
`Model: none (private content - Gemini disabled)`. Set `ALLOW_GEMINI_FOR_USER_CONTENT=1` to lift the restriction.
Known gap: retrieval **embeddings** of uploaded PDFs (and of chat queries over them) still go to the Gemini
embeddings API, because there is no other embedding backend; the restriction covers generation only.

## Time and size budgets
* One shared per-request deadline (`REQUEST_BUDGET_S`, default 50 s of Vercel's 60 s; `backend/deadline.py`): a chat
  tool turn (intent detection + generation) and a photo scan (up to two rounds of vision calls) draw from the same
  budget; when it is spent the API answers a clean `503` with the `llm` label (a scan returns the pages it already
  transcribed, failed pages are marked in the Markdown and listed in `failed_pages`). `generate_json` defaults to a
  40 s budget on Vercel.
* Request bodies for scan-notes, predictor, chat-PDF upload and the other single-PDF upload routes are capped at
  **~4.4 MB total** (`MAX_BODY_BYTES`), answered as JSON `413` *before* parsing (Vercel's own 4.5 MB rejection is not
  JSON). The SPA shrinks photos in the browser (long edge 1600 px, JPEG q0.8, <= ~900 KB each), sends at most 3 per
  request and chains requests for more; oversize PDFs are refused client-side with a message.
* `/tmp` caches (`study-documents`, `study-index`) are capped at `TMP_CACHE_MAX_MB` (300) with LRU eviction; an entry
  used in the last 3 minutes or held by an upload / delete is never evicted. A `/tmp` index is revalidated against
  MongoDB (2-minute window) before it is trusted, also after it was evicted from RAM.
* The janitor cron shares one 50 s budget across index ensure, the artifact sweep and the uploads sweep; both sweeps
  are resumable (`partial: true` in the response means the next run continues).

## Operational notes
* **No background work**: there are no threads at startup and no warm-ups. The Mongo client is created
  lazily, once per instance (pool <= 10, 3 s server-selection timeout); indexes are ensured by a one-shot
  background call guarded by a version marker (`_meta` collection) and by the daily cron.
* **Health**: `/api/health` is liveness only (no DB/LLM). `/api/health/deep` returns 503 if Mongo is down.
* **Observability**: every response carries `X-Request-ID` and `Server-Timing` (`auth`, `db`, `total`); one
  access-log line per request (`vercel logs`, dashboard -> Logs).
* **Errors**: LLM chain exhausted -> HTTP 503 `{detail, llm:{label:"Model: none (all providers failed)"}}`;
  MongoDB unreachable -> 503; anything else -> JSON 500 with a `request_id` (no tracebacks to clients).
* **Limits / large PDFs**: function request/response bodies are capped at 4.5 MB on Vercel, so the SPA sends
  selections up to ~4 MB as one multipart `POST /api/subjects` and anything bigger as a **resumable chunked
  upload** assembled server-side: `POST /api/subjects/uploads` (validates name/size/type, caps: 5 open uploads and
  80 MB pending per user) -> `PUT /api/subjects/uploads/{id}/chunks/{i}` (raw <= 3 MB, idempotent) ->
  `POST /api/subjects/uploads/{id}/complete` (same pipeline + response as the multipart add; idempotent, safe
  against concurrent calls) ; `GET` shows stored chunks, `DELETE` aborts. Chunks live in MongoDB
  (`upload_sessions` / `upload_chunks`, TTL 1 h after the last chunk, so abandoned uploads vanish by themselves);
  **MONGODB_URI is required on Vercel for this** (instances do not share `/tmp`; without it the endpoints return
  503). Locally without MongoDB a bounded OS-temp-dir store is used. Limits (all env, defaults shown; the SPA text
  is generated from them via `GET /api/subjects` -> `limits`, so copy never drifts): `UPLOAD_MAX_FILE_MB=200`,
  `UPLOAD_MAX_TOTAL_MB=0` (0 = **no per-upload total cap**), `UPLOAD_MAX_FILES=50`, `UPLOAD_MAX_PDFS_PER_SUBJECT=100`,
  `UPLOAD_MAX_SUBJECTS_PER_OWNER=100`, `UPLOAD_MAX_PAGES_PER_PDF=2000`, `UPLOAD_MAX_PAGES_PER_REQUEST=0` (unlimited),
  `UPLOAD_MAX_OPEN=8` open chunked uploads and `UPLOAD_MAX_PENDING_MB=1024` in flight per owner,
  `UPLOAD_CHUNK_BYTES=3000000` (max 3.5 MB; the SPA keeps 3 pieces in flight). Big PDFs are READ in page ranges
  across calls (`continue-index` drives it, ~35 s each; progress "Reading pages 400/1500", then "Embedding x/y chunks");
  the read text is kept next to the index (`texts.json.gz` inside the stored index zip).
* **Storage on Atlas M0 (512 MB, shared with everything)**: originals up to `RAW_KEEP_MB=25` stay in GridFS; a bigger
  PDF is dropped once fully read and only its compressed text + chunks + vectors remain ("text index only"; local dev
  without MongoDB keeps every file). `STORAGE_CAP_MB=512` / `STORAGE_REFUSE_PCT=90` make uploads answer HTTP 507
  with a clear message when the cluster is nearly full (`dbStats`, cached `STORAGE_CACHE_S=300`; fails open if unreadable).
  Per-owner usage is shown in the subject panel. Truly unlimited raw storage needs Atlas M10+ or object storage
  (Vercel Blob / S3) for the originals; the index stays in MongoDB.
* **Editing subjects**: `PATCH /api/subjects/{name}` renames (files, index zip re-keyed without re-embedding, history
  `subject` labels relabelled with one `update_many` per collection), `DELETE /api/subjects/{name}/pdfs/{file}`,
  `POST /api/subjects/{name}/reindex`, `GET /api/subjects/{name}/files`. Standard subjects can be hidden per account:
  `POST /api/subjects/{name}/hide|restore`, `POST /api/subjects/restore-standard` (collection `owner_prefs`, read
  through a 60 s cache + the shared L2 cache; shared files and history are never touched).
* **Guests/Clerk**: Clerk JWKS is fetched once per instance (3 s timeout, 1 h cache, 30 s back-off on failure).

## Shared cache (optional, free): Upstash Redis

`backend/cache.py` is a small cache/state layer with a pluggable backend. With **no configuration the app behaves
exactly as before** (everything stays in per-process memory). Setting the two variables below turns on a *shared*
Redis (any Redis exposing the Upstash-style HTTPS REST API; plain `httpx`, no client library) so that serverless
instances share state and the database is hit far less often.

| variable | meaning |
|---|---|
| `UPSTASH_REDIS_REST_URL`, `UPSTASH_REDIS_REST_TOKEN` | enable it (the Vercel Marketplace aliases `KV_REST_API_URL` / `KV_REST_API_TOKEN` also work) |
| `CACHE_BACKEND` | `auto` (default: use Redis when both vars are set), `memory` (force off), `upstash` |
| `CACHE_TIMEOUT_MS` | per-call budget, default 120, capped at 150 |

What moves into it (all keys `sah:v1:<namespace>:<sha256(owner)[:24]>:...`, small JSON, always with a TTL, never the raw
owner/guest id):

* the per-owner snapshot behind the history lists and the progress dashboard, the progress summary and the daily insights
  card, each validated by a per-owner **epoch** that every write (on any instance) bumps, so a stale entry is never served
  after a write;
* the per-owner rate-limit buckets (one atomic Lua call; the limit then holds across instances);
* the "guest was just validated/touched" marker (a cold instance skips the Mongo write), the 120 s chat-state cache, the
  user's uploaded-subjects list, the verified-token verdict (only read while Clerk's JWKS is unreachable) and the LLM
  chain's "quota exhausted" / "dead key" memory (read from a background-refreshed snapshot, never on the request path).

Safety: every call has a hard timeout and falls back to the in-process path; 3 consecutive failures open a 20 s circuit
breaker on the cache itself, so a Redis outage costs nothing but the old behaviour. The cache is never a source of truth.
Privacy note: chat context (the last few turns, <= 48 KB) and progress aggregates are stored in the Redis you choose; leave
the variables unset if that is not acceptable.

Enable it (free): Vercel dashboard -> Storage / Marketplace -> **Upstash for Redis** -> plan **Free** (500K commands/month,
no payment method; accept Upstash's terms once), connect to this project. Then `vercel env ls` must show
`KV_REST_API_URL` / `KV_REST_API_TOKEN` (or add `UPSTASH_REDIS_REST_URL` / `_TOKEN`) and redeploy. CLI equivalent once the
terms were accepted: `vercel integration add upstash/upstash-kv --plan free`. Budget: roughly 3-6 commands per session, so
the free 500K/month covers on the order of 100K sessions; if exceeded the breaker simply opens and the app keeps working.

## Privacy decision: embeddings of uploaded content (owner's call, 2026-11-06)

Chat, quizzes, flashcards, exams and every other generation over user-uploaded content skip Gemini
(`backend/privacy.py`, toggle `ALLOW_GEMINI_FOR_USER_CONTENT`, default off). **Embeddings are the one exception, by the owner's
explicit choice:** the text of uploaded PDFs is sent to Google's `gemini-embedding-2` to build the search index, because it gives much
better retrieval than keyword-only search. On the Gemini free tier Google may use prompts to improve its products. To change this,
make `vector_store` skip the embedding step for user subjects (they then fall back to BM25 keyword search automatically).
