"""Centralized configuration — the single source of truth for the whole backend.

Every environment-specific value (host, port, environment name, provider, model
ids, API key, log level, CORS origins) is read **here, exactly once**, validated
at startup, and exposed as typed module attributes / helpers. No other module in
the repository may call ``os.getenv`` or read ``os.environ`` directly — they import
from this module instead.

Design rules enforced here:
  * Required values have **no in-code fallback**. A missing required value makes
    ``validate_config()`` fail loudly at startup (see ``run.py`` / the app lifespan)
    rather than silently defaulting to a hidden hardcoded value.
  * **No model id, host, port, URL, provider, or key literal** appears as a default.
    The only literals are *enumerations of what the code supports* (which environment
    names and which providers have an integration) — never a concrete deployment value.
  * A few non-environment-specific tuning knobs (log level, token budget) carry safe,
    documented defaults because they are identical across dev and prod.

The app runs entirely on Groq for LLM inference; embeddings run locally via fastembed,
so the only secret required is ``GROQ_API_KEY``.
"""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path

from dotenv import load_dotenv

# A SINGLE .env at the repository root is the one source of configuration for BOTH
# the backend and the frontend (the SPA receives its share via GET /api/config — it
# has no .env of its own). Load that root file explicitly so it is found no matter
# the current working directory. This is the ONLY entry point through which
# configuration enters the application.
_REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_REPO_ROOT / ".env")

log = logging.getLogger("backend.config")


# ---------------------------------------------------------------------------
# Paths — derived from the repository layout, NOT environment-specific.
# ---------------------------------------------------------------------------
ROOT_DIR: Path = _REPO_ROOT
DOCUMENTS_DIR: Path = ROOT_DIR / "documents"
FAISS_INDEX_DIR: Path = ROOT_DIR / "faiss_index"
FRONTEND_DIR: Path = ROOT_DIR / "frontend"


# ---------------------------------------------------------------------------
# Supported-capability enumerations.
# These list what the CODE can do (which environments it understands, which
# providers it has a client for). They are NOT environment-specific values —
# the *selected* environment/provider always comes from the .env variables below.
# ---------------------------------------------------------------------------
ENV_DEVELOPMENT: str = "development"
ENV_PRODUCTION: str = "production"
VALID_ENVIRONMENTS: tuple[str, ...] = (ENV_DEVELOPMENT, ENV_PRODUCTION)

# Providers the code has an integration for. Only "groq" is wired today; adding a
# provider here means adding its client, not just an env value.
SUPPORTED_PROVIDERS: tuple[str, ...] = ("groq",)

# Per-tool optional model overrides map to these env vars; each, if set, must be
# one of AVAILABLE_MODELS. Unset tools fall back to DEFAULT_MODEL.
TOOL_MODEL_ENV: dict[str, str] = {
    "chat": "CHAT_MODEL",
    "quiz": "QUIZ_MODEL",
    "flashcards": "FLASHCARDS_MODEL",
    "planner": "PLANNER_MODEL",
    "summary": "SUMMARY_MODEL",
    "mindmap": "MINDMAP_MODEL",
}


class ConfigError(RuntimeError):
    """Raised by ``validate_config()`` when required configuration is missing/invalid."""


class MissingApiKeyError(RuntimeError):
    """Raised when an LLM call is attempted without a Groq API key."""


# ---------------------------------------------------------------------------
# Low-level readers — the ONLY place os.getenv is called.
# ---------------------------------------------------------------------------
def _get(name: str) -> str | None:
    """Return a stripped env value, or None if unset/blank."""
    value = os.getenv(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _get_int(name: str) -> tuple[int | None, bool]:
    """Return (int value, ok). ok is False when set-but-not-an-integer."""
    raw = _get(name)
    if raw is None:
        return None, True
    try:
        return int(raw), True
    except ValueError:
        return None, False


_TRUE_VALUES: frozenset[str] = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES: frozenset[str] = frozenset({"0", "false", "no", "off"})


def _get_bool(name: str, default: bool) -> bool:
    """Read a boolean env value (true/false/yes/no/on/off/1/0); fall back to ``default``."""
    raw = _get(name)
    if raw is None:
        return default
    low = raw.lower()
    if low in _TRUE_VALUES:
        return True
    if low in _FALSE_VALUES:
        return False
    return default


def _get_csv(name: str) -> list[str]:
    """Parse a comma-separated env value into a de-duplicated, order-preserving list."""
    raw = _get(name)
    if not raw:
        return []
    seen: list[str] = []
    for chunk in raw.split(","):
        item = chunk.strip()
        if item and item not in seen:
            seen.append(item)
    return seen


# ---------------------------------------------------------------------------
# Model label derivation.
# ---------------------------------------------------------------------------
_SIZE_RE = re.compile(r"^\d+(?:\.\d+)?[bmkBMK]$")     # 70b, 120b, 9b, 1.5b
_VERSION_RE = re.compile(r"^v?\d+(?:\.\d+)*$")        # 3, 3.3, 3.1, 2.5
_ALWAYS_UPPER = {"gpt", "oss", "ai", "llm", "moe", "vl", "it"}


def humanize_model(model_id: str) -> str:
    """Derive a readable label from a model id, e.g. '<family>-<ver>-<size>-<variant>'
    -> '<Family> <Ver> <SIZE> <Variant>', and '<vendor>/<name>' uses just '<name>'.

    Used only when an explicit label is not provided in AVAILABLE_MODELS. No model
    ids are hardcoded — this transforms whatever ids the environment supplies.
    """
    name = model_id.split("/")[-1]
    out: list[str] = []
    for part in re.split(r"[-_]", name):
        if not part:
            continue
        low = part.lower()
        if _SIZE_RE.match(part):
            out.append(part[:-1] + part[-1].upper())
        elif _VERSION_RE.match(part):
            out.append(part)
        elif low in _ALWAYS_UPPER:
            out.append(part.upper())
        else:
            out.append(part.capitalize())
    return " ".join(out) or model_id


def _parse_models(raw_list: list[str]) -> tuple[list[str], dict[str, str]]:
    """Turn AVAILABLE_MODELS entries into (ids, {id: label}).

    Each entry is either ``id`` (label derived) or ``id|Human Label`` (label given).
    """
    ids: list[str] = []
    labels: dict[str, str] = {}
    for item in raw_list:
        if "|" in item:
            mid, label = item.split("|", 1)
            mid, label = mid.strip(), label.strip()
        else:
            mid, label = item, ""
        if not mid or mid in labels:
            continue
        ids.append(mid)
        labels[mid] = label or humanize_model(mid)
    return ids, labels


# ---------------------------------------------------------------------------
# Resolved configuration (computed once at import from the environment).
# ---------------------------------------------------------------------------
# Runtime
APP_ENV: str | None = _get("APP_ENV")
HOST: str | None = _get("HOST")
PORT, _PORT_OK = _get_int("PORT")
LOG_LEVEL: str = (_get("LOG_LEVEL") or "INFO").upper()

# Derived runtime flags (no extra env needed — driven by APP_ENV).
IS_PRODUCTION: bool = APP_ENV == ENV_PRODUCTION
IS_DEVELOPMENT: bool = APP_ENV == ENV_DEVELOPMENT
RELOAD: bool = IS_DEVELOPMENT          # auto-reload only in development

# Frontend / CORS. Empty by default: the SPA is served by this same server, so no
# cross-origin access is configured (and none is hardcoded).
CORS_ORIGINS: list[str] = _get_csv("CORS_ORIGINS")

# ---- Split-origin frontend (optional) ----
# By default ONE server serves both the SPA and the API on the same origin (no CORS). To run
# the SPA on its OWN port instead, set FRONTEND_PORT: run.py then also serves the static SPA
# there, the browser reaches the API cross-origin at API_BASE_URL, and the SPA's origin must
# be allowed in CORS_ORIGINS. Unset -> classic single-server mode (fully backward compatible).
FRONTEND_PORT, _FRONTEND_PORT_OK = _get_int("FRONTEND_PORT")
SPLIT_ORIGIN: bool = FRONTEND_PORT is not None
# Absolute base URL the browser uses to reach the API in split mode, e.g.
# "http://127.0.0.1:8000/api". Required when FRONTEND_PORT is set; ignored in single-server mode.
API_BASE_URL: str | None = _get("API_BASE_URL")

# Provider / secret
LLM_PROVIDER: str | None = _get("LLM_PROVIDER")
GROQ_API_KEY: str | None = _get("GROQ_API_KEY")
# Embeddings: hosted Google Gemini (default) instead of a local fastembed model.
# Set EMBEDDING_PROVIDER=local to fall back to the keyless on-device model.
GEMINI_API_KEY: str | None = _get("GEMINI_API_KEY")
# Optional second Gemini key (a separate Google project = a separate free-tier
# quota bucket). When the primary key's embedding quota is exhausted, requests
# transparently retry on this key — SAME model, so vectors stay compatible
# with the existing FAISS indexes.
GEMINI_API_KEY_FALLBACK: str | None = _get("GEMINI_API_KEY_FALLBACK")

# Auth (optional, Clerk). CLERK_PUBLISHABLE_KEY is a PUBLIC key (safe to send to the
# browser — that is its purpose); its presence enables the sign-in gate, its absence
# leaves the SPA open (graceful). CLERK_SECRET_KEY is optional and reserved for
# backend session-token verification on the API; without it the gate is client-side.
CLERK_PUBLISHABLE_KEY: str | None = _get("CLERK_PUBLISHABLE_KEY")
CLERK_SECRET_KEY: str | None = _get("CLERK_SECRET_KEY")
AUTH_ENABLED: bool = bool(CLERK_PUBLISHABLE_KEY)

# Persistence (optional, MongoDB). When MONGODB_URI is set, every generated artifact
# (chats/quizzes/flashcards/plans/mindmaps) is saved and browsable, scoped per user.
# Absent -> nothing is persisted (graceful). The database name is not a secret, so it
# carries a safe default; it is created on first write.
MONGODB_URI: str | None = _get("MONGODB_URI")
MONGODB_DB_NAME: str = _get("MONGODB_DB_NAME") or "study_assistant_hub"
PERSISTENCE_ENABLED: bool = bool(MONGODB_URI)

# Guests (optional, anonymous accounts). A guest is just another ``owner`` string
# ("guest_<uuid>") flowing through the SAME per-user scoping as a Clerk user, so guest
# chats/quizzes/flashcards/plans/mindmaps live in the existing collections and become a
# Clerk user's items on upgrade by a single owner re-key. Guests need somewhere to store
# their work, so the feature is offered ONLY when MongoDB persistence is enabled.
GUEST_ENABLED: bool = PERSISTENCE_ENABLED
# Inactivity lifetime for anonymous guests (TTL). Not environment-specific — a tuning
# knob with a documented default (30 days), overridable in .env. ``lastActiveAt`` is
# refreshed on every guest request, so only truly idle guests are reaped.
GUEST_TTL_DAYS: int = (_get_int("GUEST_TTL_DAYS")[0]) or 30
GUEST_TTL_SECONDS: int = max(GUEST_TTL_DAYS, 1) * 86400
# How often the background janitor sweeps artifacts orphaned by a TTL-expired guest
# (Mongo TTL drops the guest record but cannot cascade to its artifacts). Tuning knob
# with a safe default of one day.
GUEST_PURGE_INTERVAL_SECONDS: int = (_get_int("GUEST_PURGE_INTERVAL_SECONDS")[0]) or 86400

# Models
AVAILABLE_MODELS, AVAILABLE_MODEL_LABELS = _parse_models(_get_csv("AVAILABLE_MODELS"))
DEFAULT_MODEL: str | None = _get("DEFAULT_MODEL")
TOOL_MODEL_OVERRIDES: dict[str, str | None] = {
    tool: _get(env_name) for tool, env_name in TOOL_MODEL_ENV.items()
}

# Embedding backend for FAISS retrieval.
#   EMBEDDING_PROVIDER=gemini  → hosted Google Gemini API (default; needs GEMINI_API_KEY).
#                                No model loaded in-process: frees ~150–250 MB of RAM and
#                                avoids the local onnxruntime/faiss libomp clash.
#   EMBEDDING_PROVIDER=local   → keyless on-device fastembed model (legacy fallback).
# EMBEDDING_MODEL is the model id for the active provider (e.g. "gemini-embedding-2"
# for gemini, or "BAAI/bge-small-en-v1.5" for local).
EMBEDDING_PROVIDER: str = (_get("EMBEDDING_PROVIDER") or "gemini").lower()
EMBEDDING_MODEL: str | None = _get("EMBEDDING_MODEL")
# Optional: request a smaller embedding dimension (gemini supports Matryoshka
# truncation; default keeps the model's full, already-normalized output).
EMBEDDING_DIM: int | None = _get_int("EMBEDDING_DIM")[0]

_SUPPORTED_EMBEDDING_PROVIDERS = ("gemini", "local")

# Generation tuning — not environment-specific (same value dev & prod). Documented
# default so the app stays usable if the knob is omitted; override in .env to tune.
# Kept modest (3000) because a provider's per-minute token budget counts the INPUT
# *and* the reserved output of a single request: on Groq's free tier the smallest
# model (llama-3.1-8b-instant) allows only 6000 tokens/minute, so the content slice
# plus the prompt plus the capped completion must all fit under that. A larger value
# is safe only on higher-tier models/plans.
MAX_CONTENT_TOKENS: int = (_get_int("MAX_CONTENT_TOKENS")[0]) or 3000

# Agentic RAG (chat): when enabled, chat retrieval becomes an LLM-driven loop —
# rewrite the question into search queries → retrieve → grade the context → correct
# (re-retrieve) if it's insufficient → answer — instead of a single top-k retrieval.
# Tuning knobs with safe defaults (not deployment-specific); set AGENTIC_RAG=false to
# fall back to classic single-shot RAG. MAX_ITERS bounds the corrective re-retrieve
# rounds so cost/latency stay bounded.
AGENTIC_RAG: bool = _get_bool("AGENTIC_RAG", True)
AGENTIC_RAG_MAX_ITERS: int = max(1, (_get_int("AGENTIC_RAG_MAX_ITERS")[0]) or 1)
AGENTIC_RAG_QUERIES: int = max(1, (_get_int("AGENTIC_RAG_QUERIES")[0]) or 2)


# ---------------------------------------------------------------------------
# Public helpers used by the rest of the app.
# ---------------------------------------------------------------------------
def has_groq_key() -> bool:
    """Return True if a Groq API key is configured."""
    return bool(GROQ_API_KEY)


def require_groq_key() -> None:
    """Raise a clear error if the Groq key is missing (call before LLM use)."""
    if not has_groq_key():
        raise MissingApiKeyError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add your Groq key."
        )


def has_gemini_key() -> bool:
    """Return True if a Gemini API key is configured (for hosted embeddings)."""
    return bool(GEMINI_API_KEY)


def model_for(tool: str, requested: str | None = None) -> str | None:
    """Resolve the model id to use for ``tool``.

    Priority: a valid per-request choice (must be one of AVAILABLE_MODELS) →
    the per-tool env override → DEFAULT_MODEL. An invalid requested model is
    ignored (logged) and falls through to the configured defaults, so an
    untrusted request can never select an unlisted model.
    """
    if requested:
        choice = requested.strip()
        if choice in AVAILABLE_MODEL_LABELS:
            log.debug("Model selection for %s: requested %r (allowed).", tool, choice)
            return choice
        log.warning(
            "Ignoring requested model %r for %s — not in AVAILABLE_MODELS; using default.",
            choice, tool,
        )
    override = TOOL_MODEL_OVERRIDES.get(tool)
    resolved = override or DEFAULT_MODEL
    log.debug("Model selection for %s: %r (override=%r, default=%r).",
              tool, resolved, override, DEFAULT_MODEL)
    return resolved


def available_models_public() -> list[dict[str, str]]:
    """Return [{id, label}, ...] for the frontend model selector (env-driven)."""
    return [{"id": mid, "label": AVAILABLE_MODEL_LABELS[mid]} for mid in AVAILABLE_MODELS]


def runtime_api_base() -> str:
    """The API base the browser should use: the absolute API_BASE_URL in split-origin mode,
    or the same-origin relative "/api" in single-server mode. Both servers inject this into
    the page (window.__API_BASE__) so the SPA bootstraps identically in either mode."""
    return API_BASE_URL or "/api"


def public_config() -> dict:
    """The non-secret configuration the SPA needs to bootstrap itself."""
    return {
        "appEnv": APP_ENV,
        "llmProvider": LLM_PROVIDER,
        "defaultModel": DEFAULT_MODEL,
        "availableModels": available_models_public(),
        "groqKey": has_groq_key(),
        # Auth: the publishable key (or null) and whether the sign-in gate is active.
        # Null/false means the SPA runs open (graceful) — no key configured yet.
        "clerkPublishableKey": CLERK_PUBLISHABLE_KEY,
        "authEnabled": AUTH_ENABLED,
        # Persistence: whether the history/memory features are available.
        "persistenceEnabled": PERSISTENCE_ENABLED,
        # Guests: whether "Continue as guest" is available (needs persistence). This gates the
        # guest button *within* the Clerk sign-in gate — so it appears only when BOTH a Clerk
        # key and persistence are set. With no Clerk key the app runs open (no gate) and never
        # offers a guest session.
        "guestEnabled": GUEST_ENABLED,
        # API base for the SPA: same-origin "/api" by default, or the absolute API_BASE_URL
        # when the frontend runs on its own port (split-origin mode). Driven by .env.
        "apiBase": runtime_api_base(),
    }


# ---------------------------------------------------------------------------
# Startup validation + diagnostics.
# ---------------------------------------------------------------------------
def validate_config() -> None:
    """Validate required configuration; raise ConfigError listing every problem.

    Call once at startup (run.py / app lifespan). Fails fast so the server never
    starts in a half-configured state with hidden defaults.
    """
    errors: list[str] = []

    if not APP_ENV:
        errors.append("APP_ENV (one of: " + ", ".join(VALID_ENVIRONMENTS) + ")")
    elif APP_ENV not in VALID_ENVIRONMENTS:
        errors.append(f"APP_ENV={APP_ENV!r} — must be one of {', '.join(VALID_ENVIRONMENTS)}")

    if not HOST:
        errors.append("HOST (the network address the server binds to)")
    if not _PORT_OK:
        errors.append("PORT (must be an integer)")
    elif PORT is None:
        errors.append("PORT (the TCP port to listen on)")

    if not LLM_PROVIDER:
        errors.append("LLM_PROVIDER (one of: " + ", ".join(SUPPORTED_PROVIDERS) + ")")
    elif LLM_PROVIDER not in SUPPORTED_PROVIDERS:
        errors.append(
            f"LLM_PROVIDER={LLM_PROVIDER!r} — no integration; supported: {', '.join(SUPPORTED_PROVIDERS)}"
        )

    if LLM_PROVIDER == "groq" and not GROQ_API_KEY:
        errors.append("GROQ_API_KEY (get one at https://console.groq.com/keys)")

    if not AVAILABLE_MODELS:
        errors.append("AVAILABLE_MODELS (comma-separated list of model ids)")
    if not DEFAULT_MODEL:
        errors.append("DEFAULT_MODEL (the default model id)")
    elif AVAILABLE_MODELS and DEFAULT_MODEL not in AVAILABLE_MODEL_LABELS:
        errors.append(f"DEFAULT_MODEL={DEFAULT_MODEL!r} is not listed in AVAILABLE_MODELS")

    for tool, mid in TOOL_MODEL_OVERRIDES.items():
        if mid and AVAILABLE_MODELS and mid not in AVAILABLE_MODEL_LABELS:
            errors.append(
                f"{TOOL_MODEL_ENV[tool]}={mid!r} is not listed in AVAILABLE_MODELS"
            )

    if EMBEDDING_PROVIDER not in _SUPPORTED_EMBEDDING_PROVIDERS:
        errors.append(
            f"EMBEDDING_PROVIDER={EMBEDDING_PROVIDER!r} — supported: "
            + ", ".join(_SUPPORTED_EMBEDDING_PROVIDERS)
        )
    if not EMBEDDING_MODEL:
        errors.append("EMBEDDING_MODEL (embedding model id, e.g. gemini-embedding-2)")
    if EMBEDDING_PROVIDER == "gemini" and not GEMINI_API_KEY:
        errors.append("GEMINI_API_KEY (required when EMBEDDING_PROVIDER=gemini; get one at https://aistudio.google.com/apikey)")

    # Auth is optional, but a misformatted publishable key is almost always a typo.
    if CLERK_PUBLISHABLE_KEY and not CLERK_PUBLISHABLE_KEY.startswith("pk_"):
        errors.append("CLERK_PUBLISHABLE_KEY must start with 'pk_' (publishable key from the Clerk dashboard)")

    # Persistence is optional, but a misformatted URI is almost always a typo.
    if MONGODB_URI and not MONGODB_URI.startswith("mongodb"):
        errors.append("MONGODB_URI must start with 'mongodb://' or 'mongodb+srv://'")

    # Split-origin frontend is optional, but if FRONTEND_PORT is set it must be a valid port
    # AND API_BASE_URL must be set (the SPA can't reach the API cross-origin without it).
    if not _FRONTEND_PORT_OK:
        errors.append("FRONTEND_PORT (must be an integer)")
    elif SPLIT_ORIGIN:
        if not API_BASE_URL:
            errors.append(
                "API_BASE_URL (absolute API URL, e.g. http://127.0.0.1:8000/api) is required "
                "when FRONTEND_PORT is set"
            )
        elif not API_BASE_URL.startswith("http"):
            errors.append("API_BASE_URL must be an absolute URL starting with http:// or https://")

    if APP_ENV and APP_ENV not in VALID_ENVIRONMENTS:
        pass  # already reported above

    if errors:
        bullet = "\n".join(f"  ❌ Missing/invalid configuration: {e}" for e in errors)
        raise ConfigError(
            "Configuration is invalid — the server cannot start.\n"
            + bullet
            + "\n\nSet these in your .env file (copy .env.example) and restart."
        )

    log.info("Configuration validated ✓ (%d models, provider=%s, env=%s).",
             len(AVAILABLE_MODELS), LLM_PROVIDER, APP_ENV)


def startup_diagnostics() -> list[str]:
    """Return the startup banner lines (logged by ``log_startup_diagnostics``)."""
    bar = "=" * 49
    models = ", ".join(AVAILABLE_MODELS) if AVAILABLE_MODELS else "(none)"
    return [
        bar,
        " Study Assistant Hub",
        bar,
        f" APP_ENV          {APP_ENV}",
        f" HOST             {HOST}",
        f" PORT             {PORT}",
        f" RELOAD           {RELOAD}",
        f" LOG_LEVEL        {LOG_LEVEL}",
        f" LLM_PROVIDER     {LLM_PROVIDER}",
        f" DEFAULT_MODEL    {DEFAULT_MODEL}",
        f" AVAILABLE_MODELS {len(AVAILABLE_MODELS)} ({models})",
        f" EMBEDDINGS       {EMBEDDING_PROVIDER} · {EMBEDDING_MODEL}"
        + (f" ({EMBEDDING_DIM}d)" if EMBEDDING_DIM else "")
        + (" · key ✓" if (EMBEDDING_PROVIDER == 'gemini' and GEMINI_API_KEY) else ""),
        f" CHAT RETRIEVAL   {('agentic (rewrite→retrieve→grade→correct, ≤' + str(AGENTIC_RAG_MAX_ITERS) + ' rounds)') if AGENTIC_RAG else 'classic single-shot RAG'}",
        f" GROQ_API_KEY     {'✓ loaded' if has_groq_key() else '✗ MISSING'}",
        f" CORS_ORIGINS     {', '.join(CORS_ORIGINS) if CORS_ORIGINS else '(none — same-origin)'}",
        f" FRONTEND         {('split → SPA on :' + str(FRONTEND_PORT) + ', API at ' + (API_BASE_URL or '?')) if SPLIT_ORIGIN else 'single server (SPA + API same origin)'}",
        f" AUTH (Clerk)     {('enabled — ' + ('API verify on' if CLERK_SECRET_KEY else 'client-side gate')) if AUTH_ENABLED else 'disabled (open app)'}",
        f" PERSISTENCE      {('MongoDB → ' + MONGODB_DB_NAME) if PERSISTENCE_ENABLED else 'disabled (no history)'}",
        f" GUESTS           {('enabled — TTL ' + str(GUEST_TTL_DAYS) + 'd') if GUEST_ENABLED else 'disabled (needs MongoDB)'}",
        bar,
    ]


def log_startup_diagnostics() -> None:
    """Log the startup banner + a 'Groq config loaded' marker."""
    for line in startup_diagnostics():
        log.info(line)
    if has_groq_key():
        log.info("Groq config loaded ✓ (provider=%s).", LLM_PROVIDER)
    # In split-origin mode the SPA calls the API cross-origin; without an allowed origin the
    # browser will block every request. Warn loudly (not fatal — a reverse proxy may add it).
    if SPLIT_ORIGIN and not CORS_ORIGINS:
        log.warning(
            "FRONTEND_PORT is set but CORS_ORIGINS is empty — the SPA's cross-origin API calls "
            "will be blocked by the browser. Add the SPA origin (e.g. http://%s:%s) to CORS_ORIGINS.",
            HOST or "127.0.0.1", FRONTEND_PORT,
        )


# ---------------------------------------------------------------------------
# Logging configuration.
# ---------------------------------------------------------------------------
_logging_configured = False


def configure_logging() -> None:
    """Configure the ``backend`` logger from LOG_LEVEL (idempotent).

    Routes backend.* logs to the console with a consistent format and the
    configured level, independent of uvicorn's own handlers.
    """
    global _logging_configured
    if _logging_configured:
        return
    level = getattr(logging, LOG_LEVEL, logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-7s %(name)s | %(message)s", datefmt="%H:%M:%S")
    )
    backend_logger = logging.getLogger("backend")
    backend_logger.setLevel(level)
    backend_logger.handlers = [handler]
    backend_logger.propagate = False
    _logging_configured = True
