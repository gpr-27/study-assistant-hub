###############################################################################
# Study Assistant Hub — production image
#
# Multi-stage build:
#   • builder  — installs the Python dependencies into an isolated virtualenv.
#   • runtime  — copies that venv, pre-bakes the local models (so the first
#                request never waits on a download), drops to a non-root user,
#                and serves the app with Gunicorn + Uvicorn workers.
#
# The container runs the app in SINGLE-SERVER mode: one FastAPI process serves
# BOTH the vanilla-JS SPA and the /api/* routes on the same origin (no CORS).
# Every value comes from the environment (see .env.example / .env.production.example)
# — nothing deployment-specific is baked into the image. Secrets are injected at
# run time (env_file / -e / platform secret store); they are NEVER built in.
#
# No `# syntax=` directive on purpose: it keeps the build from pulling an external
# BuildKit frontend image (one fewer network dependency at build time). Only stock
# Dockerfile features are used, so the engine's built-in frontend handles it.
###############################################################################

# Python 3.12: mature and has prebuilt wheels for every dependency (faiss-cpu,
# onnxruntime, numpy, …), so the build never falls back to a slow/failing source
# compile. Override with --build-arg PYTHON_VERSION=... if you need a different one.
ARG PYTHON_VERSION=3.12

# -----------------------------------------------------------------------------
# Stage 1 — builder: resolve & install dependencies into /opt/venv
# -----------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

# faiss-cpu / onnxruntime / numpy ship manylinux wheels for CPython, so no
# compiler is needed in practice. build-essential is kept only as a safety net
# for any sdist-only transitive dependency; it stays in this stage and adds
# nothing to the final runtime image.
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install dependencies in their own layer so editing app code never busts the
# (slow) dependency cache.
COPY requirements.txt ./
RUN pip install --upgrade pip \
 && pip install -r requirements.txt

# -----------------------------------------------------------------------------
# Stage 2 — runtime
# -----------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS runtime

# Embeddings are hosted (Google Gemini) — nothing to pre-bake into the image.
# Kept as a build arg only so the FAISS pre-build below can stamp the manifest model.
ARG EMBEDDING_MODEL=gemini-embedding-2

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH="/opt/venv/bin:$PATH" \
    # All model/tokenizer caches live under one baked, writable dir so the first
    # request needs no network — and a cache miss self-heals (re-downloads here).
    HF_HOME=/opt/cache/hf \
    HUGGINGFACE_HUB_CACHE=/opt/cache/hf/hub \
    FASTEMBED_CACHE_PATH=/opt/cache/fastembed \
    TIKTOKEN_CACHE_DIR=/opt/cache/tiktoken \
    # Sensible in-image defaults for the runtime mode; compose / -e still override.
    APP_ENV=production \
    HOST=0.0.0.0 \
    PORT=8000 \
    WEB_CONCURRENCY=2

# tini = a tiny init for correct signal handling / zombie reaping as PID 1.
RUN apt-get update \
 && apt-get install -y --no-install-recommends tini \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --create-home --uid 10001 app

# The installed dependencies (virtualenv) from the builder stage.
COPY --from=builder /opt/venv /opt/venv

WORKDIR /app

# Application code + production server config + entrypoint.
COPY backend ./backend
COPY frontend ./frontend
COPY gunicorn.conf.py ./gunicorn.conf.py
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh

# Seed documents live OUTSIDE the data dir so an empty volume mounted at
# /app/documents doesn't hide them — the entrypoint seeds them in on start.
COPY documents /opt/seed-documents

# Writable data dirs: faiss_index holds pre-built (and any new) indexes,
# documents holds the seed PDFs + any uploads. Created owned by the app user so a
# named volume first-mounted here inherits the correct ownership.
RUN mkdir -p /app/documents /app/faiss_index /opt/cache \
 && chmod +x /usr/local/bin/docker-entrypoint.sh \
 && chown -R app:app /app /opt/cache /opt/seed-documents

USER app

# Pre-bake the tokenizer so the first request needs no network for tokenization.
# (Embeddings are hosted on Gemini now — no local model to bake.) Non-fatal.
RUN python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')" \
      || echo 'WARN: tiktoken encoding not pre-baked; it will download on first use.'

# Seed the bundled-subject documents so their committed FAISS indexes are recognized
# as fresh at runtime (the content-hash manifest matches) and merely LOADED — no
# re-embedding, so no Gemini API call and no key needed at build time. We preserve
# timestamps with -p to mirror what the entrypoint seeds at runtime.
RUN cp -rp /opt/seed-documents/. /app/documents/ || true

EXPOSE 8000

# Liveness probe via the app's own health endpoint (no curl in slim — use Python).
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD python -c "import os,urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','8000')+'/api/health', timeout=4).status==200 else 1)" || exit 1

ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/docker-entrypoint.sh"]
CMD ["gunicorn", "backend.main:app", "-c", "/app/gunicorn.conf.py"]
