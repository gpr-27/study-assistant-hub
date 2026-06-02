# 🚀 Production deployment

This guide covers running **Study Assistant Hub** in production with Docker. The
app is a single FastAPI server that serves both the vanilla‑JS SPA and the
`/api/*` routes on one origin, talks to **Groq** for inference, embeds locally
with **fastembed** (no API key), and optionally persists per‑user history in
**MongoDB**.

> **TL;DR — Render (recommended):** push to GitHub → Render dashboard → **New +
> → Blueprint** → connect the repo → set `GROQ_API_KEY` → done. One Web Service
> serves the SPA **and** the API.
>
> **TL;DR — local Docker:**
> ```bash
> cp .env.production.example .env     # fill in GROQ_API_KEY (+ optional Mongo/Clerk)
> docker compose up --build -d        # build + run
> # open http://localhost:8000  ·  health: http://localhost:8000/api/health
> ```

---

## Table of contents

- [Deploy to Render (one-click Blueprint)](#deploy-to-render-one-click-blueprint)
- [Why a container (and not serverless / Vercel)](#why-a-container-and-not-serverless--vercel)
- [What the image contains](#what-the-image-contains)
- [Configuration & secrets](#configuration--secrets)
- [Run with Docker Compose](#run-with-docker-compose-recommended)
- [Run with plain Docker](#run-with-plain-docker)
- [Data & persistence (volumes)](#data--persistence-volumes)
- [Scaling, workers & resources](#scaling-workers--resources)
- [Health checks](#health-checks)
- [TLS / reverse proxy](#tls--reverse-proxy)
- [Deploying to a managed platform](#deploying-to-a-managed-platform)
- [CI/CD & the published image](#cicd--the-published-image)
- [Security checklist](#security-checklist)
- [Troubleshooting](#troubleshooting)

---

## Deploy to Render (one-click Blueprint)

The repo ships a [`render.yaml`](render.yaml) Blueprint that provisions a single
Render **Web Service** building the production `Dockerfile`. One service serves the
SPA and the `/api/*` routes on one origin, with full RAG (local embeddings + FAISS).

1. Push the repo to GitHub.
2. Open <https://dashboard.render.com> → **New + → Blueprint** and connect the repo.
3. Render reads `render.yaml` and shows the service → **Apply**.
4. Set the env vars it marks as required-at-deploy (`sync: false`): **`GROQ_API_KEY`**
   is the only required one. Add `CLERK_PUBLISHABLE_KEY` (+ optional `CLERK_SECRET_KEY`)
   to enable the sign-in gate, and `MONGODB_URI` to enable saved history.
5. Render builds the image (~5–10 min — it installs FAISS/fastembed and pre-bakes the
   embedding model) and serves at `https://<service-name>.onrender.com`. The health
   check hits `/api/health`.

**Notes & gotchas:**

- **Memory.** The `free` instance is **512 MB**; the embedding model loads per worker
  (`WEB_CONCURRENCY=1` in the blueprint keeps that to one copy). If the first chat OOMs
  or the service restarts, change `plan: free` → `plan: standard` (2 GB) in `render.yaml`
  or the dashboard.
- **Cold starts.** Free services sleep after ~15 min idle and take ~30–60 s to wake.
- **FAISS indexes** rebuild from the bundled PDFs on each start (free tier has no
  persistent disk). Uncomment the `disk:` block in `render.yaml` (paid plans) to cache
  them at `/app/faiss_index` across restarts.
- **History/MongoDB.** Add `0.0.0.0/0` to **Atlas → Network Access** so Render's dynamic
  egress IPs can connect (RAG chat itself works without Mongo).
- **`PORT`.** Render injects `PORT`; the app binds `0.0.0.0:$PORT` automatically — only
  `HOST=0.0.0.0` is set in the blueprint.

The same image runs on **Railway / Fly.io / Cloud Run / a VPS** — see
[Deploying to a managed platform](#deploying-to-a-managed-platform).

## Why a container (and not serverless / Vercel)

This backend is a poor fit for serverless platforms (Vercel Functions, Lambda,
etc.) and a great fit for a long‑lived container, because it:

- **Bundles heavy native dependencies** — `onnxruntime`, `faiss-cpu`, `numpy`,
  the full LangChain stack. The installed footprint (~hundreds of MB) blows past
  typical serverless bundle limits.
- **Loads a local embedding model into memory** and keeps **per‑subject FAISS
  indexes** cached in‑process. Serverless functions are ephemeral and would
  rebuild/reload this on every cold start.
- **Runs background threads** — a startup warmup (embeddings + indexes) and a
  guest‑cleanup janitor — that need a process that stays alive.
- **Holds a persistent MongoDB connection** and writes to disk (built indexes,
  uploaded PDFs), which wants a writable, persistent filesystem.

So: deploy the Docker image to any container host — **Render, Railway, Fly.io,
Google Cloud Run, AWS App Runner/ECS, a plain VPS**, etc.

## What the image contains

- **Multi‑stage build** on `python:3.13-slim`: dependencies are installed into a
  virtualenv in a builder stage and copied into a lean runtime stage.
- **Pre‑baked models**: the fastembed embedding model **and** the tiktoken
  encoding are downloaded at build time into `/opt/cache`, so the **first
  request needs no network** and isn't slow. (If a download is unavailable at
  build time the cache self‑heals on first use at runtime.)
- **Non‑root user** (`app`, uid 10001) and **tini** as PID 1 for clean signals.
- **Gunicorn + Uvicorn workers** (`gunicorn.conf.py`), all tunable via env.
- A **`/api/health`** Docker `HEALTHCHECK`.
- Secrets are **never** baked in — they're injected at run time.

## Configuration & secrets

Every value the app needs is documented in [`.env.example`](.env.example);
[`.env.production.example`](.env.production.example) is the same set with
production defaults. **Required** in production:

| Variable | Notes |
|---|---|
| `GROQ_API_KEY` | **Required.** Your Groq key (`gsk_…`). |
| `LLM_PROVIDER` | `groq`. |
| `AVAILABLE_MODELS`, `DEFAULT_MODEL` | The model list + default (validated at startup). |
| `EMBEDDING_MODEL` | Local fastembed model — **pre‑baked into the image** (see below). |

**Optional** (the app runs fully without them):

| Variable | Notes |
|---|---|
| `MONGODB_URI` | Enables per‑user history (and guests). Use Atlas or the bundled `mongo` service. SECRET. |
| `CLERK_PUBLISHABLE_KEY` | Enables the sign‑in gate. Use a `pk_live_…` key in production. PUBLIC. |
| `CLERK_SECRET_KEY` | **Set this in production if you use Clerk** — it makes the API verify session JWTs server‑side. Without it the gate is client‑side only and the API is open. SECRET. |

`APP_ENV`, `HOST`, `PORT`, and `WEB_CONCURRENCY` are set to production values
automatically by `docker-compose.yml` and by the image defaults, so you mostly
provide the secrets above.

> **Never commit a filled‑in `.env`.** It's gitignored. On a managed platform,
> set these as the platform's environment variables / secret store instead of
> shipping a file.

## Run with Docker Compose (recommended)

```bash
cp .env.production.example .env        # then edit: at minimum set GROQ_API_KEY
docker compose up --build -d           # build image + start in the background
docker compose logs -f app             # follow logs
docker compose ps                      # check health
```

- App: <http://localhost:8000> · Health: <http://localhost:8000/api/health>
- Remap the host port with `APP_PORT=9000 docker compose up` (container stays 8000).
- Stop: `docker compose down` (add `-v` to also delete the data volumes).

**With a local MongoDB** (instead of Atlas):

```bash
# in .env:  MONGODB_URI=mongodb://mongo:27017
docker compose --profile with-mongo up --build -d
```

## Run with plain Docker

```bash
# Build (optionally pin the embedding model to match your .env)
docker build -t study-assistant-hub:latest \
  --build-arg EMBEDDING_MODEL=BAAI/bge-small-en-v1.5 .

# Run — inject secrets via --env-file, force production runtime, persist data
docker run -d --name study-assistant-hub \
  --env-file .env \
  -e APP_ENV=production -e HOST=0.0.0.0 -e PORT=8000 \
  -p 8000:8000 \
  -v sah_faiss:/app/faiss_index \
  -v sah_documents:/app/documents \
  --restart unless-stopped \
  study-assistant-hub:latest
```

## Data & persistence (volumes)

The app writes to two directories; mount volumes so they survive restarts:

| Path | Contents | Notes |
|---|---|---|
| `/app/faiss_index` | Built per‑subject vector indexes | Pure cache — safe to delete; rebuilt automatically from the PDFs on next chat. |
| `/app/documents` | Subject PDFs (bundled **and** uploaded) | The bundled subjects are seeded into this dir on start (without clobbering), so uploads in a mounted volume persist. |

`MongoDB` (history, if enabled) is **separate** from these — it lives in your
Atlas cluster or the `mongo` volume, not in the app's filesystem.

> First chat for a subject builds its index (a few seconds, in a background
> thread) and caches it to `/app/faiss_index`. With the volume mounted this
> happens once, not on every restart.

## Scaling, workers & resources

- Tune workers with **`WEB_CONCURRENCY`** (default `2`). **Each worker loads its
  own copy of the embedding model**, so memory scales with worker count. The app
  is I/O‑bound (waiting on Groq/MongoDB) and each worker already serves sync
  routes from a threadpool, so **1–2 workers** suits most deployments.
- Suggested starting resources: **~1 vCPU, 1–2 GB RAM** for 1–2 workers. Bump
  RAM before adding workers.
- Other gunicorn knobs (timeout, max‑requests recycling, keep‑alive) are env‑
  driven — see [`gunicorn.conf.py`](gunicorn.conf.py).
- Mind **Groq rate limits** (free tier has low tokens‑per‑minute); the app
  surfaces a clear 429 with remediation. Pick a model whose TPM exceeds
  `MAX_CONTENT_TOKENS` + prompt + output.

## Health checks

- Endpoint: **`GET /api/health`** → `{"status":"ok","groq_key":true|false}`.
- The image declares a Docker `HEALTHCHECK` against it (40s start period).
- Point your platform's liveness/readiness probe at `/api/health`.

## TLS / reverse proxy

The container speaks plain HTTP on `:8000`. In production, terminate TLS in
front of it:

- **Managed platforms** (Render/Railway/Fly/Cloud Run) provide HTTPS + a domain
  automatically — just expose port `8000`.
- **Self‑hosted**: put **Nginx / Caddy / Traefik** in front. Forward to the
  container and pass `X-Forwarded-*` headers. Example Caddyfile:
  ```
  app.example.com {
      reverse_proxy 127.0.0.1:8000
  }
  ```

If you ever serve the SPA from a **different origin** than the API, set
`CORS_ORIGINS` to that origin (comma‑separated). In the default single‑server
setup no CORS is needed.

## Deploying to a managed platform

All of these consume the same image / repo. Set the env vars from
[Configuration & secrets](#configuration--secrets), expose port `8000`, point
the health check at `/api/health`, and attach a disk for `/app/faiss_index`
(+`/app/documents` if you allow uploads).

- **Render** — *New → Web Service → Docker*. Add env vars; add a Disk mounted at
  `/app/faiss_index`. Health check path `/api/health`.
- **Railway** — deploy from the repo (Dockerfile detected). Add env vars and a
  volume for `/app/faiss_index`.
- **Fly.io** — `fly launch` (uses the Dockerfile), `fly secrets set GROQ_API_KEY=…`,
  add a volume mounted at `/app/faiss_index`; set the internal port to `8000`.
- **Google Cloud Run** — push the image, deploy with container port `8000`. Note
  Cloud Run has an ephemeral filesystem and scales to zero, so prefer Atlas for
  history and accept that indexes rebuild on cold start (or keep min‑instances ≥ 1).
- **VPS** — install Docker, `git clone`, create `.env`, `docker compose up -d`,
  put Caddy/Nginx in front for TLS.

## CI/CD & the published image

Two GitHub Actions workflows are included:

- **[`ci.yml`](.github/workflows/ci.yml)** — runs the test suite on Python 3.13
  for every push to `main` and every PR. Hermetic (no secrets needed).
- **[`docker.yml`](.github/workflows/docker.yml)** — builds the image and pushes
  it to **`ghcr.io/<owner>/<repo>`** on pushes to `main` and on `v*` tags. Tags:
  branch name, semver (from `v*` tags), and the commit SHA.

Pull and run the published image (after the workflow has run once):

```bash
docker pull ghcr.io/<owner>/study-assistant-hub:main
docker run -d --env-file .env -e APP_ENV=production -e HOST=0.0.0.0 \
  -p 8000:8000 -v sah_faiss:/app/faiss_index \
  ghcr.io/<owner>/study-assistant-hub:main
```

> The package may default to private. Make it public in the repo's *Packages*
> settings, or `docker login ghcr.io` with a token that has `read:packages`.

## Security checklist

- [ ] **Rotate any credentials** that were ever in a local `.env` before going
      live (Groq key, MongoDB user/password). Use the platform's secret store.
- [ ] If using Clerk, set **`CLERK_SECRET_KEY`** so the API verifies JWTs — the
      publishable key alone only gates the UI client‑side.
- [ ] Use a **dedicated MongoDB user** with least privilege and a strong password;
      restrict network access (Atlas IP allow‑list) to your platform.
- [ ] Serve over **HTTPS** (managed platform or reverse proxy).
- [ ] Keep `.env` out of git (it already is) and out of the image (it is — see
      `.dockerignore`).
- [ ] Set `CORS_ORIGINS` only to origins you control (leave empty for single‑origin).

## Troubleshooting

| Symptom | Cause & fix |
|---|---|
| Container exits immediately, logs show `❌ Missing/invalid configuration: …` | A required env var is missing/invalid (e.g. `GROQ_API_KEY`, `DEFAULT_MODEL` not in `AVAILABLE_MODELS`). Fix it in `.env` and restart — the app fails fast by design. |
| `HEALTHCHECK` stuck `starting` then `unhealthy` | App didn't bind. Check logs (`docker compose logs app`); verify `PORT=8000` and that nothing else uses the host port. |
| First chat slow / build pulls the model | The embedding model wasn't pre‑baked (build‑time download failed). It downloads once at runtime into the cache; rebuild with network access to bake it. |
| History not saving | `MONGODB_URI` unset or unreachable. Logs print `MongoDB init failed; persistence disabled for this run`. Check the URI/credentials and network/IP allow‑list. |
| Indexes rebuild on every restart | `/app/faiss_index` isn't on a persistent volume. Mount one (see [Data & persistence](#data--persistence-volumes)). |
| Uploaded subjects disappear on restart | `/app/documents` isn't on a persistent volume. Mount one. |
| 429 from AI tools | Groq rate limit. Use a higher‑TPM model or lower `MAX_CONTENT_TOKENS`. |
