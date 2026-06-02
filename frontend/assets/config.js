/* ============================================================
   Study Assistant Hub — centralized FRONTEND configuration.

   This is the ONLY frontend file that knows the bootstrap API path and the
   ONLY one that talks to the server's /config endpoint. Everything else reads
   window.AppConfig.* — no view hardcodes an API URL, a model id, or a provider.

   window.AppConfig shape:
     apiBase          string   — base path for all API calls (server may override)
     appEnv           string   — "development" | "production"
     llmProvider      string   — e.g. "groq"
     defaultModel     string   — id of the default model
     availableModels  [{id,label}] — the configured model list (drives the selector)
     groqKey          boolean  — whether the server has a Groq key
     guestEnabled     boolean  — whether "Continue as guest" is offered (needs persistence)
     loaded           boolean  — true once /config has resolved
     ready            Promise<AppConfig> — resolves after /config loads
   ============================================================ */
(() => {
  "use strict";

  // Bootstrap API base for the very first request (GET /config). It comes from
  // window.__API_BASE__, injected by /runtime-config.js (served by whichever server
  // delivered this page): same-origin "/api" in single-server mode, or the absolute
  // API_BASE_URL when the SPA runs on its own port. Falls back to "/api" if the inject
  // script is missing, so single-server mode keeps working with no extra setup. The
  // server can still confirm/override `apiBase` via the /config response below.
  const BOOTSTRAP_API_BASE =
    (typeof window.__API_BASE__ === "string" && window.__API_BASE__) || "/api";

  const cfg = {
    apiBase: BOOTSTRAP_API_BASE,
    appEnv: null,
    llmProvider: null,
    defaultModel: null,
    availableModels: [],
    groqKey: false,
    clerkPublishableKey: null,
    authEnabled: false,
    persistenceEnabled: false,
    guestEnabled: false,
    loaded: false,
  };

  cfg.ready = fetch(BOOTSTRAP_API_BASE + "/config")
    .then((r) => (r.ok ? r.json() : Promise.reject(new Error("config " + r.status))))
    .then((d) => {
      cfg.apiBase = d.apiBase || BOOTSTRAP_API_BASE;
      cfg.appEnv = d.appEnv || null;
      cfg.llmProvider = d.llmProvider || null;
      cfg.defaultModel = d.defaultModel || null;
      cfg.availableModels = Array.isArray(d.availableModels) ? d.availableModels : [];
      cfg.groqKey = !!d.groqKey;
      cfg.clerkPublishableKey = d.clerkPublishableKey || null;
      cfg.authEnabled = !!d.authEnabled;
      cfg.persistenceEnabled = !!d.persistenceEnabled;
      cfg.guestEnabled = !!d.guestEnabled;
      cfg.loaded = true;
      return cfg;
    })
    .catch((e) => {
      // Keep the bootstrap defaults so the app still renders; surface the failure.
      console.error("Failed to load /api/config:", e);
      cfg.loaded = true;
      return cfg;
    });

  window.AppConfig = cfg;
})();
