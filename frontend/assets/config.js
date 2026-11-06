/* ============================================================
   Study Assistant Hub — centralized FRONTEND configuration.

   This is the ONLY frontend file that knows the bootstrap API path and the
   ONLY one that talks to the server's /config endpoint. Everything else reads
   window.AppConfig.* — no view hardcodes an API URL, a model id, or a provider.

   window.AppConfig shape:
     apiBase          string   — base path for all API calls (default same-origin "/api")
     appEnv           string   — "development" | "production"
     chainOrder       array    — `llmChainOrder` [{step,name,provider,models[]}] (static, no keys)
     chain            object   — `llmChain` booleans {anthropic, groq1, groq2, gemini1, gemini2}; never keys
     modelPolicy      string   — e.g. "auto: best available via chain"
     aiAvailable      boolean  — true unless the server positively reports NO provider key
     llmProvider / defaultModel — legacy; may be null (models are now "Auto")
     availableModels  [{id,label}] — optional override list; usually empty ("Auto")
     groqKey          boolean  — legacy flag (old backend); NOT used to show key errors
     guestEnabled     boolean  — whether "Continue as guest" is offered (needs persistence)
     loaded           boolean  — true once /config has resolved
     ready            Promise<AppConfig> — resolves after /config loads

   Loading strategy (fast path): the <head> inline script starts the /config request
   before any script is parsed (window.__sahCfg). A copy of the last good response is kept in
   localStorage, so a returning visitor's `ready` resolves from cache immediately while the
   network copy revalidates in the background and updates the object in place.
   ============================================================ */
(() => {
  "use strict";

  // Default API base is same-origin "/api" (Vercel rewrites /api/* to the function).
  // window.__API_BASE__ is honoured only if something set it (no /runtime-config.js needed).
  const BOOTSTRAP_API_BASE =
    (typeof window.__API_BASE__ === "string" && window.__API_BASE__) || "/api";

  const CACHE_KEY = "sah-cfg-v1";
  const CACHE_TTL_MS = 10 * 60 * 1000;   // beyond this a cached copy is ignored (not just stale)

  const cfg = {
    apiBase: BOOTSTRAP_API_BASE,
    appEnv: null,
    llmProvider: null,
    defaultModel: null,
    availableModels: [],
    chain: null,
    chainOrder: [],
    aiAvailable: true,
    modelPolicy: null,
    groqKey: false,
    clerkPublishableKey: null,
    authEnabled: false,
    persistenceEnabled: false,
    guestEnabled: false,
    loaded: false,
  };

  function anyTrue(o) {
    if (o === true) return true;
    if (!o || typeof o !== "object") return false;
    return Object.values(o).some((v) => v === true || (v && typeof v === "object" && (v.configured === true || v.available === true || v.enabled === true || anyTrue(v))));
  }

  function apply(d) {
    if (!d || typeof d !== "object") return;
    cfg.apiBase = (typeof d.apiBase === "string" && d.apiBase) || BOOTSTRAP_API_BASE;
    cfg.appEnv = d.appEnv || null;
    cfg.llmProvider = d.llmProvider || null;
    cfg.defaultModel = d.defaultModel || null;
    cfg.availableModels = Array.isArray(d.availableModels) ? d.availableModels : [];
    // New shape: chain status booleans. Accept a few spellings; never expect key material.
    const chain = d.llmChain || d.chain || d.chainStatus || d.llm_chain || d.providers || null;
    cfg.chain = chain && typeof chain === "object" ? chain : null;
    cfg.chainOrder = Array.isArray(d.llmChainOrder) ? d.llmChainOrder : [];
    cfg.premiumBudget = d.premiumBudget && typeof d.premiumBudget === "object" ? d.premiumBudget : null;   // booleans only
    cfg.groqKey = !!d.groqKey;
    // Only report "unavailable" when the server positively says so: a known chain with no
    // true entry. Old servers (groqKey only) are never nagged about a "missing Groq key".
    // Backend contract: `llmAvailable` (bool) is authoritative; fall back to the chain flags.
    cfg.aiAvailable = typeof d.llmAvailable === "boolean" ? d.llmAvailable
      : cfg.chain ? anyTrue(cfg.chain) : (d.aiAvailable !== false);
    cfg.modelPolicy = d.modelPolicy || null;
    cfg.clerkPublishableKey = d.clerkPublishableKey || null;
    cfg.authEnabled = !!d.authEnabled;
    cfg.persistenceEnabled = !!d.persistenceEnabled;
    cfg.guestEnabled = !!d.guestEnabled;
    cfg.loaded = true;
  }

  function readCache() {
    try {
      const raw = localStorage.getItem(CACHE_KEY);
      if (!raw) return null;
      const { at, data } = JSON.parse(raw);
      if (!data || Date.now() - at > CACHE_TTL_MS) return null;
      return data;
    } catch (_) { return null; }
  }
  function writeCache(data) {
    try { localStorage.setItem(CACHE_KEY, JSON.stringify({ at: Date.now(), data })); } catch (_) {}
  }

  // Reuse the request the <head> script already started; otherwise start it now.
  const network = (window.__sahCfg || fetch(BOOTSTRAP_API_BASE + "/config", { credentials: "same-origin" }))
    .then((r) => (r.ok ? r.json() : Promise.reject(new Error("config " + r.status))))
    .then((d) => { writeCache(d); return d; });

  const cached = readCache();
  if (cached) {
    apply(cached);
    cfg.ready = Promise.resolve(cfg);
    // Revalidate in the background; update in place and tell listeners.
    network.then((d) => {
      apply(d);
      try { window.dispatchEvent(new CustomEvent("sah:config", { detail: cfg })); } catch (_) {}
    }).catch((e) => console.warn("Config revalidation failed:", e && e.message));
  } else {
    cfg.ready = network
      .then((d) => { apply(d); return cfg; })
      .catch((e) => {
        // Keep the bootstrap defaults so the app still renders; surface the failure.
        console.error("Failed to load /api/config:", e);
        cfg.loaded = true;
        return cfg;
      });
  }

  window.AppConfig = cfg;
})();
