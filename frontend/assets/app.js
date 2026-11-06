/* ============================================================
   Study Assistant Hub — SPA runtime
   Provides: window.api, window.ui, window.Views, hash router,
   theme toggle, ambient interactions, AI-availability banner.

   ---- Tool-view contract (each views/<name>.js does) ----
   window.Views.quiz = {
     accent: "quiz",                  // chat | quiz | cards | plan
     title: "Quiz Generator",
     async mount(root) { ...build UI into `root`... },
     unmount() {}                     // optional cleanup
   };
   `root` is an empty .view-body element. Build with template strings
   + ui.el(), call window.api for data, use the CSS component classes
   (.panel .btn .btn-accent .field .input .select .seg .opts .flip …).
   ============================================================ */
(() => {
  "use strict";
  const root = document.documentElement;
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* ---------------- api ---------------- */
  // The API base comes from the centralized config layer (window.AppConfig,
  // defined in config.js which loads before this script). No API URL is hardcoded
  // here — config.js seeds apiBase synchronously and the server may override it.
  const apiUrl = (p) => window.AppConfig.apiBase + p;
  // Auth readiness. The Clerk SDK (auth.js — a deferred ES module) only resolves the
  // session AFTER this classic script has already run and mounted the first view. Any
  // authenticated call made during first paint (notably the history panel) must not race
  // ahead of it, or it fires with no token and 401s ("Sign in or continue as a guest…").
  // auth.js calls window.__authReady() once the session is settled (immediately when there
  // is no Clerk gate); the timeout is a backstop so requests can never hang if auth.js
  // never loads. authHeaders() awaits this before reading the token.
  const authReady = new Promise((resolve) => {
    window.__authReady = resolve;
    setTimeout(resolve, 4000);
  });

  // Asset version stamp (written into <meta name="asset-version"> by scripts/stamp_assets.py)
  // so lazily injected view scripts share the same long-cache-safe ?v= as the eager ones.
  const ASSET_V = (() => {
    const m = document.querySelector('meta[name="asset-version"]');
    return (m && m.content) || "";
  })();
  const assetUrl = (p) => p + (ASSET_V ? "?v=" + ASSET_V : "");

  // The last API failure that is worth a "Retry" affordance (503 AI unavailable / network).
  // ui.toast(message, "error") consults it so every existing call site gets Retry for free.
  let lastRetryable = null;
  // The `llm` block ("Model: none (all providers failed)" …) of the last failed API call, so the
  // error toast can say which model state it is reporting (matched by message, valid for 8 s).
  let lastErrLlm = null;
  async function handle(res) {
    if (!res.ok) {
      let detail, body;
      try { body = await res.json(); detail = body && body.detail; } catch (_) {}
      if (detail && typeof detail === "object") detail = detail.message || detail.detail || JSON.stringify(detail);
      let msg = detail;
      if (res.status === 503) msg = detail || "The AI is temporarily unavailable. Please try again in a moment.";
      else if (res.status === 504 || res.status === 502) msg = detail || "The server took too long to respond. Please try again.";
      const err = new Error(msg || `${res.status} ${res.statusText}`);
      err.status = res.status;     // let callers special-case auth (401) vs real failures
      err.detail = detail || null;
      if (body && body.llm) { err.llm = body.llm; lastErrLlm = { message: err.message, llm: body.llm, at: Date.now() }; }
      if (res.status === 503 || res.status === 502 || res.status === 504 || res.status === 429) {
        err.retryable = true;
        lastRetryable = { message: err.message, at: Date.now() };
      }
      throw err;
    }
    const ct = res.headers.get("content-type") || "";
    return ct.includes("application/json") ? res.json() : res.text();
  }
  // localStorage key for the anonymous guest id (written by auth.js). Read directly here
  // so the API layer never depends on auth.js's async load order.
  const GUEST_KEY = "sah-guest-id";
  const GUEST_RE = /^guest_[A-Za-z0-9]{1,48}$/;
  // Attach identity to every API call:
  //  • Clerk session token (Authorization: Bearer …) when signed in — verified server-side.
  //  • Guest id (X-Guest-Id) when a guest session exists — scopes data to that guest.
  // Both are harmless no-ops when absent; the server prefers a valid Clerk session over a
  // guest id, so sending both during the brief upgrade window is safe.
  async function authHeaders(extra) {
    // Wait until the auth layer has settled so first-paint requests carry the Clerk token
    // instead of racing ahead of sign-in and 401ing. Resolves fast when there's no gate.
    try { await authReady; } catch (_) {}
    const headers = Object.assign({}, extra);
    try {
      const token = window.Auth && window.Auth.getToken ? await window.Auth.getToken() : null;
      if (token) headers["Authorization"] = "Bearer " + token;
    } catch (_) { /* unauthenticated or token unavailable — send the request without it */ }
    try {
      const gid = localStorage.getItem(GUEST_KEY);
      // Only well-formed ids go on the wire (backend: guest_[A-Za-z0-9]{1,48}); a stale or
      // corrupted value is dropped here and cleared by auth.js on its next validation.
      if (gid && GUEST_RE.test(gid)) headers["X-Guest-Id"] = gid;
      else if (gid) localStorage.removeItem(GUEST_KEY);
    } catch (_) { /* localStorage unavailable — proceed without the guest header */ }
    return headers;
  }
  // A fetch that dies at the network layer (Safari: "Load failed", Chrome:
  // "Failed to fetch") usually means the connection was cut while the server
  // was still working — e.g. a long LLM/rate-limit wait outliving the phone's
  // ~60s fetch limit. Surface something the user can act on instead.
  function netFail(e) {
    if (e instanceof TypeError) {
      const offline = navigator.onLine === false;
      const err = new Error(offline
        ? "You appear to be offline. Check your connection and try again."
        : "Connection dropped — the server is busy or slow. Wait a moment and try again."
      );
      err.status = 0;
      err.retryable = true;
      lastRetryable = { message: err.message, at: Date.now() };
      throw err;
    }
    throw e;
  }

  // ---- per-view lifetime: GETs issued by a view are dropped when the user navigates away ----
  // A request whose view is gone never settles (its callbacks would only touch a dead DOM
  // and flash a bogus error toast), but the underlying fetch is only aborted when no other
  // caller is sharing it.
  let viewCtl = new AbortController();
  function newViewScope() {
    try { viewCtl.abort(); } catch (_) {}
    viewCtl = new AbortController();
    return viewCtl.signal;
  }
  const NEVER = new Promise(() => {});
  function scoped(promise, signal) {
    if (!signal) return promise;
    if (signal.aborted) return NEVER;
    return new Promise((resolve, reject) => {
      const onAbort = () => { /* stay pending: the owning view is gone */ };
      signal.addEventListener("abort", onAbort, { once: true });
      promise.then(
        (v) => { signal.removeEventListener("abort", onAbort); if (!signal.aborted) resolve(v); },
        (e) => { signal.removeEventListener("abort", onAbort); if (!signal.aborted) reject(e); }
      );
    });
  }

  // ---- GET: in-flight de-duplication + tiny stale-while-revalidate cache ----
  // CACHEABLE paths are read-mostly; entries live in memory (per tab) with a TTL. Fresh hit
  // -> instant; stale hit -> returned instantly and refreshed in the background (the next
  // caller gets the new data). Mutating calls to a related prefix invalidate the entry.
  // /config is additionally mirrored to sessionStorage by config.js (not user-specific);
  // user-scoped data (subjects, history) stays in memory only so it can never leak across
  // accounts in a shared tab.
  const CACHE_TTL = { "/subjects": 60_000, "/config": 300_000, "/progress/summary": 30_000, "/history/chats": 15_000 };
  const cache = new Map();      // path -> { at, data }
  const inflight = new Map();   // identity+path -> Promise
  const clone = (x) => (x && typeof x === "object" && typeof structuredClone === "function") ? structuredClone(x) : x;
  function identityTag() {
    let gid = ""; try { gid = localStorage.getItem(GUEST_KEY) || ""; } catch (_) {}
    return (window.Auth && window.Auth.userId) || gid || "anon";
  }
  function rawGet(p) {
    const key = identityTag() + " " + p;
    if (inflight.has(key)) return inflight.get(key);
    const pr = authHeaders().then((h) => fetch(apiUrl(p), { headers: h }).catch(netFail)).then(handle);
    inflight.set(key, pr);
    const done = () => { if (inflight.get(key) === pr) inflight.delete(key); };
    pr.then(done, done);
    return pr;
  }
  function get(p, opts) {
    const signal = (opts && opts.signal) || viewCtl.signal;
    const ttl = CACHE_TTL[p.split("?")[0]];
    if (ttl && !(opts && opts.fresh)) {
      const key = identityTag() + " " + p;
      const hit = cache.get(key);
      if (hit) {
        const age = Date.now() - hit.at;
        if (age > ttl && !hit.refreshing) {
          hit.refreshing = true;
          rawGet(p).then((d) => cache.set(key, { at: Date.now(), data: d }), () => { hit.refreshing = false; });
        }
        return scoped(Promise.resolve(clone(hit.data)), signal);
      }
      return scoped(rawGet(p).then((d) => { cache.set(key, { at: Date.now(), data: d }); return clone(d); }), signal);
    }
    return scoped(rawGet(p).then(clone), signal);
  }
  function invalidate(prefix) {
    try { sessionStorage.removeItem(BOOT_PREFIX + identityTag()); } catch (_) {}   // the boot mirror is now stale
    for (const k of [...cache.keys()]) if (!prefix || k.split(" ")[1].startsWith(prefix)) cache.delete(k);
  }
  const mutating = (fn) => (p, ...rest) => fn(p, ...rest).then((r) => { invalidate("/subjects"); invalidate("/progress"); invalidate("/history"); return r; });
  // Any other write may change progress / saved chats: drop those SWR entries (cheap; refetched on demand).
  const touches = (fn) => (...a) => fn(...a).then((r) => { invalidate("/progress"); invalidate("/history"); return r; });

  // ---- /api/bootstrap: ONE request for subjects + progress summary + recent chats ----
  // Primes the SWR cache above (so later api.get("/subjects") etc. are instant) and mirrors the payload to
  // sessionStorage under an IDENTITY-SCOPED key: a returning tab paints from it immediately and revalidates.
  // Cleared on sign-out / identity change (clearCache); never shared across accounts; owner data is not
  // cacheable by any CDN (the endpoint is `private, no-store`).
  const BOOT_PREFIX = "sah-boot-v1:";
  const BOOT_MAX_AGE = 10 * 60 * 1000;
  function bootKey() { return BOOT_PREFIX + identityTag(); }
  function primeBootstrap(d, at) {
    const id = identityTag();
    if (d.subjects) cache.set(id + " /subjects", { at, data: d.subjects });
    if (d.progress) cache.set(id + " /progress/summary", { at, data: d.progress });
    if (d.chats) cache.set(id + " /history/chats", { at, data: d.chats });
  }
  function bootFromSession() {
    try {
      const raw = sessionStorage.getItem(bootKey());
      if (!raw) return null;
      const { at, data } = JSON.parse(raw);
      if (!data || Date.now() - at > BOOT_MAX_AGE) return null;
      // `at` is the original fetch time: an entry older than its TTL is served as stale and revalidated by get().
      primeBootstrap(data, at);
      return data;
    } catch (_) { return null; }
  }
  let bootPromise = null;
  function bootstrap() {
    if (bootPromise) return bootPromise;
    bootPromise = rawGet("/bootstrap").then((d) => {
      if (!d || !d.owner) { bootPromise = null; return null; }
      primeBootstrap(d, Date.now());
      try { sessionStorage.setItem(bootKey(), JSON.stringify({ at: Date.now(), data: { subjects: d.subjects, progress: d.progress, chats: d.chats } })); } catch (_) {}
      return d;
    }, () => { bootPromise = null; return null; });
    return bootPromise;
  }
  function clearBootSession() {
    bootPromise = null;
    try { Object.keys(sessionStorage).filter((k) => k.startsWith(BOOT_PREFIX)).forEach((k) => sessionStorage.removeItem(k)); } catch (_) {}
  }

  window.api = {
    get,
    post: touches((p, body, opts) => authHeaders({ "Content-Type": "application/json" }).then((h) =>
      fetch(apiUrl(p), { method: "POST", headers: h, body: JSON.stringify(body || {}), signal: opts && opts.signal }).catch(netFail)
    ).then(handle)),
    postForm: mutating((p, formData) => authHeaders().then((h) =>
      fetch(apiUrl(p), { method: "POST", headers: h, body: formData }).catch(netFail)
    ).then(handle)),
    // Raw binary PUT (used for chunked PDF upload; each body stays well under the 4.5 MB host limit).
    putBlob: (p, blob, opts) => authHeaders({ "Content-Type": "application/octet-stream" }).then((h) =>
      fetch(apiUrl(p), { method: "PUT", headers: h, body: blob, signal: opts && opts.signal }).catch(netFail)
    ).then(handle),
    del: mutating((p) => authHeaders().then((h) => fetch(apiUrl(p), { method: "DELETE", headers: h }).catch(netFail)).then(handle)),
    patch: touches((p, body) => authHeaders({ "Content-Type": "application/json" }).then((h) =>
      fetch(apiUrl(p), { method: "PATCH", headers: h, body: JSON.stringify(body || {}) }).catch(netFail)
    ).then(handle)),
    invalidate,
    bootstrap,
    clearCache: () => { cache.clear(); clearBootSession(); },
    assetUrl,
  };

  /* ---------------- ui helpers ---------------- */
  const esc = (s) => String(s == null ? "" : s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
  function el(html) {
    const t = document.createElement("template");
    t.innerHTML = String(html).trim();
    return t.content.firstElementChild;
  }
  function clear(node) { while (node && node.firstChild) node.removeChild(node.firstChild); }
  function spinner(label = "Working…") {
    return el(`<div class="loading" role="status"><span class="spinner"></span><span>${esc(label)}</span></div>`);
  }
  function empty(message, icon = "✦") {
    return el(`<div class="empty"><div class="big">${icon}</div><div>${esc(message)}</div></div>`);
  }
  // The generate/primary button the user last pressed inside the current view; the toast's
  // Retry re-clicks it so a 503 "AI unavailable" can be retried with the same form state.
  let lastAction = null;
  function toast(message, type = "info", ms = 3200) {
    const wrap = document.getElementById("toasts");
    if (!wrap) return;
    let errLlm = null;
    if (message && typeof message === "object") { errLlm = message.llm || null; message = message.message; }   // accept Error objects
    if (type === "error" && !errLlm && lastErrLlm && lastErrLlm.message === message && Date.now() - lastErrLlm.at < 8000) errLlm = lastErrLlm.llm;
    const t = el(`<div class="toast ${type === "error" ? "err" : type === "ok" ? "ok" : ""}"></div>`);
    const txt = document.createElement("span");
    txt.textContent = String(message == null ? "" : message);
    if (type === "error" && errLlm) { const l = renderModelLabel(errLlm); if (l) txt.append(l); }   // "Model: none (…)" on AI failures
    t.append(txt);
    let life = ms;
    // Retryable API failures (503 AI unavailable, dropped connection, 429) get a Retry action.
    if (type === "error" && lastRetryable && lastRetryable.message === message && Date.now() - lastRetryable.at < 8000
        && lastAction && lastAction.isConnected) {
      const b = el(`<button type="button" class="toast-action">Retry</button>`);
      b.addEventListener("click", () => { t.remove(); const a = lastAction; if (a && a.isConnected && !a.disabled) a.click(); });
      t.append(b);
      life = Math.max(ms, 8000);
    }
    wrap.appendChild(t);
    setTimeout(() => { t.style.opacity = "0"; t.style.transition = "opacity .4s"; setTimeout(() => t.remove(), 400); }, life);
  }

  // ---- mandatory "Model: …" line ------------------------------------------------------
  // Renders the backend's top-level `llm` object ({provider, model, step, step_name, label,
  // skipped}) as a small unobtrusive line. Renders NOTHING when `llm` is absent (older backend)
  // unless the caller passes {stored:true} (an item reopened from history): items saved before
  // model tracking then say so honestly instead of silently showing nothing.
  const LEGACY_LABEL = "Model: not recorded (saved before model labels)";
  const AUTO_MODEL_TEXT = "Auto · best available model (5-key chain)";
  function modelLabelText(llm) {
    if (!llm || typeof llm !== "object") return "";
    if (llm.label) return String(llm.label);
    if (llm.model) return `Model: ${llm.provider || "?"} · ${llm.model}` + (llm.step ? ` · step ${llm.step}${llm.step_name ? ` (${llm.step_name})` : ""}` : "");
    return "";
  }
  function decorateModelLabel(node, llm, text, caption) {
    node.textContent = "";
    if (caption) {
      const c = document.createElement("span");
      c.className = "model-label-cap";
      c.textContent = caption + " · ";
      node.append(c);
    }
    node.append(document.createTextNode(text));     // the backend's authoritative string, verbatim
    node.setAttribute("role", "note");
    node.setAttribute("aria-label", "AI model used" + (caption ? " for the " + caption.toLowerCase() : " for this output") + ". " + text);
    const sk = llm && llm.skipped && llm.skipped.length ? " (fallbacks skipped: " + llm.skipped.join(", ") + ")" : "";
    node.title = text + sk;
  }
  function renderModelLabel(llm, opts) {
    opts = opts || {};
    let text = modelLabelText(llm);
    if (!text && opts.stored) { llm = { label: LEGACY_LABEL }; text = LEGACY_LABEL; }
    if (!text) return null;
    const d = document.createElement("div");
    d.className = "model-label";
    d.setAttribute("data-model-label", "");
    decorateModelLabel(d, llm, text, opts.caption);
    if (!opts.stored && !opts.quiet) noteActiveModel(llm, text);   // history reopens are not "the latest answer"
    return d;
  }
  // Update an existing label node in place (e.g. the live label under a streaming bubble).
  function updateModelLabel(node, llm) {
    const text = modelLabelText(llm);
    if (!node || !text) return node || null;
    decorateModelLabel(node, llm, text, node.getAttribute("data-caption") || "");
    noteActiveModel(llm, text);
    return node;
  }
  // Append (or replace) the label inside `container`. Safe to call with a missing llm.
  function attachModelLabel(container, resp, opts) {
    if (!container) return null;
    container.querySelectorAll(":scope > .model-label").forEach((n) => n.remove());
    const node = renderModelLabel(resp && resp.llm ? resp.llm : resp && resp.label ? resp : null, opts);
    if (node) container.append(node);
    return node;
  }

  // ---- global "AI model" badge (header chip + footer line + chain popover) ------------------
  // Shows the model behind the MOST RECENT AI response of this session; before any answer it says
  // "Auto · best available model (5-key chain)". No network of its own: the chain order and which
  // steps have a key come from /api/config (window.AppConfig.chainOrder / .chain; booleans only).
  const FALLBACK_CHAIN = [   // only used if an older cached /config lacks `llmChainOrder`
    { step: 1, name: "Anthropic", provider: "anthropic", models: ["claude-sonnet-5-5"] },
    { step: 2, name: "Groq key 1", provider: "groq", models: ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"] },
    { step: 3, name: "Groq key 2", provider: "groq", models: ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"] },
    { step: 4, name: "Gemini key 1", provider: "gemini", models: ["gemini-3.8-flash", "gemini-3.6-flash", "gemini-3.1-flash-lite"] },
    { step: 5, name: "Gemini key 2", provider: "gemini", models: ["gemini-3.8-flash", "gemini-3.6-flash", "gemini-3.1-flash-lite"] },
  ];
  const CHAIN_FLAG = { 1: "anthropic", 2: "groq1", 3: "groq2", 4: "gemini1", 5: "gemini2" };
  let activeModel = { text: "", short: "", llm: null };
  function shortModelText(llm, text) {
    if (llm && llm.model) return `${llm.provider || "?"} · ${String(llm.model).split("/").pop()}` + (llm.step ? ` · step ${llm.step}` : "");
    return String(text || "").replace(/^Model:\s*/i, "");
  }
  function paintActiveModel() {
    const full = activeModel.text ? activeModel.text.replace(/^Model:\s*/i, "") : AUTO_MODEL_TEXT;
    const short = activeModel.text ? activeModel.short : AUTO_MODEL_TEXT;
    const btn = document.getElementById("aiModelBtn");
    const t = document.getElementById("aiModelText");
    const f = document.getElementById("aiModelFootText");
    if (t) t.textContent = short;
    if (f) f.textContent = full;
    if (btn) {
      btn.setAttribute("aria-label", "AI model: " + full);
      btn.dataset.state = activeModel.text ? (/^Model:\s*none/i.test(activeModel.text) ? "none" : "live") : "auto";
    }
  }
  function noteActiveModel(llm, text) {
    activeModel = { text, short: shortModelText(llm, text), llm };
    paintActiveModel();
    if (document.getElementById("aiModelPop") && !document.getElementById("aiModelPop").hidden) buildModelPopover();
  }
  function buildModelPopover() {
    const pop = document.getElementById("aiModelPop");
    if (!pop) return;
    const cfg = window.AppConfig || {};
    const order = Array.isArray(cfg.chainOrder) && cfg.chainOrder.length ? cfg.chainOrder : FALLBACK_CHAIN;
    const flags = cfg.chain && typeof cfg.chain === "object" ? cfg.chain : null;
    const curStep = activeModel.llm && activeModel.llm.step;
    clear(pop);
    pop.append(el(`<div class="aim-head"><b>AI model</b><span class="muted">Last answer</span></div>`));
    const last = el(`<div class="aim-last"></div>`);
    last.textContent = activeModel.text || ("Nothing answered yet · " + AUTO_MODEL_TEXT);
    pop.append(last);
    pop.append(el(`<div class="aim-sub">Chain order (first available model answers; the next is used only if one fails)</div>`));
    const ol = el(`<ol class="aim-chain"></ol>`);
    order.forEach((st) => {
      const configured = flags ? !!flags[CHAIN_FLAG[st.step]] : null;
      const li = document.createElement("li");
      if (curStep === st.step) li.className = "is-current";
      const nm = document.createElement("span");
      nm.className = "aim-step";
      nm.textContent = `${st.step}. ${st.name}`;
      const state = document.createElement("span");
      state.className = "aim-state " + (configured === null ? "unknown" : configured ? "on" : "off");
      state.textContent = configured === null ? "" : configured ? "configured" : "no key";
      const models = document.createElement("span");
      models.className = "aim-models";
      models.textContent = (st.models || []).join(" → ");
      li.append(nm, state, models);
      ol.append(li);
    });
    pop.append(ol);
    const pb = cfg.premiumBudget;
    if (pb && (pb.nearlyUsed || pb.exhausted)) {
      const note = el(`<div class="aim-sub aim-budget"></div>`);
      note.textContent = pb.exhausted
        ? "Premium model budget used for today · free models are answering"
        : "Premium model budget nearly used today";
      pop.append(note);
    }
  }
  function initModelBadge() {
    const btn = document.getElementById("aiModelBtn");
    const pop = document.getElementById("aiModelPop");
    if (!btn || !pop || btn.dataset.bound) return;
    btn.dataset.bound = "1";
    paintActiveModel();
    const close = () => { pop.hidden = true; pop.dataset.peek = ""; btn.setAttribute("aria-expanded", "false"); };
    const open = (peek) => { buildModelPopover(); pop.hidden = false; pop.dataset.peek = peek ? "1" : ""; btn.setAttribute("aria-expanded", "true"); };
    // A click on a hover-opened ("peek") popover pins it open; otherwise click toggles.
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      if (pop.hidden) open(false);
      else if (pop.dataset.peek) pop.dataset.peek = "";
      else close();
    });
    document.addEventListener("click", (e) => { if (!pop.hidden && !pop.contains(e.target) && e.target !== btn) close(); });
    document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !pop.hidden) { close(); btn.focus(); } });
    // Hover peek on devices that can hover (click/tap works everywhere); leaving closes a peek only.
    const wrap = document.getElementById("aiModel");
    if (wrap && window.matchMedia && matchMedia("(hover: hover)").matches) {
      wrap.addEventListener("mouseenter", () => { if (pop.hidden) open(true); });
      wrap.addEventListener("mouseleave", () => { if (pop.dataset.peek) close(); });
    }
    addEventListener("sah:config", () => { if (!pop.hidden) buildModelPopover(); });
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", initModelBadge);
  else initModelBadge();

  // Inline error block with an optional Retry button (used for 503 / network failures).
  function errorBlock(err, retry) {
    const msg = (err && err.message) || String(err || "Something went wrong.");
    const box = el(`<div class="panel-soft err-block" role="alert"></div>`);
    const t = document.createElement("span");
    t.textContent = "⚠ " + msg;
    box.append(t);
    const node = renderModelLabel(err && err.llm);
    if (node) box.append(node);
    if (retry) {
      const b = el(`<button type="button" class="btn err-retry">Retry</button>`);
      b.addEventListener("click", () => retry());
      box.append(b);
    }
    return box;
  }

  // ---- rich, SAFE markdown renderer -> enhanced <div class="md"> element ----
  // Math spans are lifted out BEFORE escaping/emphasis (so `_` and `*` inside TeX survive) and typeset lazily.
  const MATH_SPLIT = /(`[^`]+`|\$\$[^$\n]+?\$\$|\$(?!\s)(?:\\.|[^$\\\n])+?(?<!\s)\$(?!\d)|\\\((?:.|\n)+?\\\))/g;
  function mathSpan(raw) {
    let tex, display = false;
    if (raw.startsWith("$$")) { tex = raw.slice(2, -2); display = true; }
    else if (raw.startsWith("\\(")) tex = raw.slice(2, -2);
    else tex = raw.slice(1, -1);
    return `<span class="math-inline" data-tex="${esc(tex.trim())}"${display ? ' data-display="1"' : ""}>${esc(raw)}</span>`;
  }
  function inlineMd(text) {
    return String(text).split(MATH_SPLIT).map((part, i) => {
      if (i % 2 === 1) {
        if (part.startsWith("`")) return `<code class="inline">${esc(part.slice(1, -1))}</code>`;
        return mathSpan(part);
      }
      return esc(part)
        .replace(/!\[([^\]]*)\]\([^)]*\)/g, (m, alt) => `<em class="md-noimg">[image${alt ? ": " + alt : ""} blocked]</em>`)   // images are never loaded
        .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
        .replace(/__([^_]+)__/g, "<strong>$1</strong>")
        .replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<em>$2</em>")
        .replace(/~~([^~]+)~~/g, "<del>$1</del>")
        .replace(/\[([^\]]+)\]\((https:\/\/[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
    }).join("");
  }

  function copyToClipboard(text, btn) {
    const use = btn && btn.querySelector("use");
    const reset = btn ? btn.textContent : "";
    const done = () => {
      if (!btn) return;
      if (use) {   // icon button: swap the glyph to a check (label unchanged), announce via toast-free live title
        const prev = use.getAttribute("href"), prevT = btn.dataset.tip || btn.title || "";
        use.setAttribute("href", "#i-check"); btn.classList.add("copied"); btn.dataset.tip = "Copied"; btn.removeAttribute("title"); try { toast("Copied to clipboard.", "ok", 1800); } catch (_) {}
        setTimeout(() => { use.setAttribute("href", prev); btn.dataset.tip = prevT; btn.classList.remove("copied"); }, 1500);
        return;
      }
      btn.textContent = "Copied!"; btn.classList.add("copied"); setTimeout(() => { btn.textContent = reset; btn.classList.remove("copied"); }, 1500);
    };
    const fallback = () => { const ta = document.createElement("textarea"); ta.value = text; ta.style.position = "fixed"; ta.style.opacity = "0"; document.body.appendChild(ta); ta.select(); try { document.execCommand("copy"); } catch (_) {} ta.remove(); done(); };
    if (navigator.clipboard && navigator.clipboard.writeText) navigator.clipboard.writeText(text).then(done).catch(fallback);
    else fallback();
  }

  // highlight.js (~36 KB + theme) is only fetched the first time a code block is rendered.
  let hljsP = null;
  function loadHljs() {
    if (window.hljs) return Promise.resolve(window.hljs);
    if (hljsP) return hljsP;
    const css = document.createElement("link");
    css.rel = "stylesheet";
    css.href = "https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css";
    document.head.appendChild(css);
    hljsP = new Promise((resolve, reject) => {
      const sc = document.createElement("script");
      sc.src = "https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js";
      sc.async = true;
      sc.onload = () => resolve(window.hljs);
      sc.onerror = () => { hljsP = null; reject(new Error("hljs")); };
      document.head.appendChild(sc);
    });
    return hljsP;
  }
  const hlQueue = new Set();
  let hlRaf = 0;
  function highlightLater(codeEl) {
    hlQueue.add(codeEl);
    if (hlRaf) return;
    hlRaf = requestAnimationFrame(() => {
      hlRaf = 0;
      const batch = [...hlQueue]; hlQueue.clear();
      loadHljs().then((h) => batch.forEach((c) => { if (c.isConnected) { try { h.highlightElement(c); } catch (_) {} } })).catch(() => {});
    });
  }

  // KaTeX (~280 KB + fonts) is only fetched the first time a math span is rendered. Pinned + SRI.
  const KATEX_V = "0.16.9";
  const KATEX_BASE = "https://cdnjs.cloudflare.com/ajax/libs/KaTeX/" + KATEX_V + "/";
  const KATEX_SRI = {
    "katex.min.css": "sha384-n8MVd4RsNIU0tAv4ct0nTaAbDJwPJzDEaqSD1odI+WdtXRGWt2kTvGFasHpSy3SV",
    "katex.min.js": "sha384-XjKyOOlGwcjNTAIQHIpgOno0Hl1YQqzUOEleOLALmuqehneUG+vnGctmUb0ZY0l8",
  };
  let katexP = null;
  function loadKatex() {
    if (window.katex) return Promise.resolve(window.katex);
    if (katexP) return katexP;
    const css = document.createElement("link");
    css.rel = "stylesheet"; css.href = KATEX_BASE + "katex.min.css";
    css.integrity = KATEX_SRI["katex.min.css"]; css.crossOrigin = "anonymous";
    document.head.appendChild(css);
    katexP = new Promise((resolve, reject) => {
      const sc = document.createElement("script");
      sc.src = KATEX_BASE + "katex.min.js"; sc.async = true;
      sc.integrity = KATEX_SRI["katex.min.js"]; sc.crossOrigin = "anonymous";
      sc.onload = () => resolve(window.katex);
      sc.onerror = () => { katexP = null; reject(new Error("katex")); };
      document.head.appendChild(sc);
    });
    return katexP;
  }
  const mathQueue = new Set();
  let mathRaf = 0;
  function mathLater(node) {
    mathQueue.add(node);
    if (mathRaf) return;
    mathRaf = requestAnimationFrame(() => {
      mathRaf = 0;
      const batch = [...mathQueue]; mathQueue.clear();
      loadKatex().then((k) => batch.forEach((n) => {
        if (!n.isConnected || n.dataset.typeset) return;
        try {
          k.render(n.dataset.tex || "", n, { displayMode: n.classList.contains("math-block") || !!n.dataset.display,
            throwOnError: false, trust: false, maxExpand: 1000, output: "htmlAndMathml" });
          n.dataset.typeset = "1";
        } catch (_) { /* leave the raw TeX visible */ }
      })).catch(() => {});
    });
  }
  function queueMath(root) {
    if (LIVE_MD) return;                       // streaming tail: keep the raw TeX until the block is final
    root.querySelectorAll("[data-tex]:not([data-typeset])").forEach(mathLater);
  }
  let LIVE_MD = false;                          // set by renderMarkdown(src, {live:true}): no syntax highlight / typeset

  function buildCodeBlock(lang, code) {
    const wrap = el(`<div class="codeblock"></div>`);
    const head = el(`<div class="cb-head"><span class="cb-lang">${esc(lang || "code")}</span><button class="cb-copy icon-btn is-sm" type="button" aria-label="Copy code" title="Copy code">${iconHtml("copy")}</button></div>`);
    const pre = document.createElement("pre");
    const codeEl = document.createElement("code");
    if (lang) codeEl.className = "language-" + lang;
    codeEl.textContent = code;
    pre.appendChild(codeEl);
    wrap.append(head, pre);
    if (!LIVE_MD) highlightLater(codeEl);
    head.querySelector(".cb-copy").addEventListener("click", () => copyToClipboard(code, head.querySelector(".cb-copy")));
    return wrap;
  }

  function buildList(items) {
    const rootList = document.createElement(items[0].ordered ? "ol" : "ul");
    if (items[0].ordered && items[0].start > 1) rootList.start = items[0].start;
    const stack = [{ indent: items[0].indent, list: rootList }];
    for (const it of items) {
      let top = stack[stack.length - 1];
      if (it.indent > top.indent) {
        const sub = document.createElement(it.ordered ? "ol" : "ul");
        (top.list.lastElementChild || top.list).appendChild(sub);
        stack.push({ indent: it.indent, list: sub });
        top = stack[stack.length - 1];
      } else {
        while (stack.length > 1 && it.indent < top.indent) { stack.pop(); top = stack[stack.length - 1]; }
      }
      const li = document.createElement("li");
      const task = it.text.match(/^\[([ xX])\]\s+(.*)$/);
      if (task) {                                  // GitHub task list: read-only checkbox
        li.className = "md-task";
        const cb = document.createElement("input");
        cb.type = "checkbox"; cb.disabled = true; cb.checked = task[1] !== " ";
        cb.setAttribute("aria-label", task[1] !== " " ? "done" : "not done");
        li.append(cb, " ");
        const span = document.createElement("span"); span.innerHTML = inlineMd(task[2]); li.append(span);
      } else li.innerHTML = inlineMd(it.text);
      top.list.appendChild(li);
    }
    return rootList;
  }

  function splitRow(line) {
    let s = line.trim();
    if (s.startsWith("|")) s = s.slice(1);
    if (s.endsWith("|")) s = s.slice(0, -1);
    return s.split("|").map((c) => c.trim());
  }

  function buildTable(header, aligns, rows) {
    const wrap = el(`<div class="table-wrap"></div>`);
    const table = document.createElement("table");
    table.className = "md-table";
    const thead = document.createElement("thead");
    const htr = document.createElement("tr");
    header.forEach((c, i) => { const th = document.createElement("th"); th.innerHTML = inlineMd(c); if (aligns[i]) th.style.textAlign = aligns[i]; htr.appendChild(th); });
    thead.appendChild(htr);
    const tbody = document.createElement("tbody");
    rows.forEach((r) => { const tr = document.createElement("tr"); header.forEach((_, i) => { const td = document.createElement("td"); td.innerHTML = inlineMd(r[i] || ""); if (aligns[i]) td.style.textAlign = aligns[i]; tr.appendChild(td); }); tbody.appendChild(tr); });
    table.append(thead, tbody);
    wrap.appendChild(table);
    return wrap;
  }

  function mathBlock(tex) {
    const d = document.createElement("div");
    d.className = "math-block"; d.dataset.tex = tex.trim(); d.textContent = "$$" + tex.trim() + "$$";
    return d;
  }

  function renderMarkdown(src, opts) {
    LIVE_MD = !!(opts && opts.live);
    try { return renderMarkdownInner(src); } finally { LIVE_MD = false; }
  }
  function renderMarkdownInner(src) {
    const root = el(`<div class="md"></div>`);
    const lines = String(src == null ? "" : src).replace(/\r\n/g, "\n").split("\n");
    let i = 0;
    while (i < lines.length) {
      const line = lines[i];
      // display math: $$ … $$ (one line or several) and \[ … \]
      const dm = line.match(/^\s*\$\$(.+?)\$\$\s*$/) || line.match(/^\s*\\\[(.+?)\\\]\s*$/);
      if (dm) { root.appendChild(mathBlock(dm[1])); i++; continue; }
      if (/^\s*(\$\$|\\\[)\s*$/.test(line)) {
        const close = /^\s*\$\$/.test(line) ? /^\s*\$\$\s*$/ : /^\s*\\\]\s*$/;
        const buf = []; i++;
        while (i < lines.length && !close.test(lines[i])) { buf.push(lines[i]); i++; }
        i++;
        root.appendChild(mathBlock(buf.join("\n")));
        continue;
      }
      const fence = line.match(/^```(\w+)?\s*$/);
      if (fence) {
        const lang = (fence[1] || "").toLowerCase();
        const buf = []; i++;
        while (i < lines.length && !/^```\s*$/.test(lines[i])) { buf.push(lines[i]); i++; }
        i++;
        root.appendChild(buildCodeBlock(lang, buf.join("\n")));
        continue;
      }
      if (line.includes("|") && i + 1 < lines.length && lines[i + 1].includes("|") && lines[i + 1].includes("-") && /^\s*\|?\s*:?-{2,}/.test(lines[i + 1])) {
        const header = splitRow(line);
        const aligns = splitRow(lines[i + 1]).map((c) => { const l = c.startsWith(":"), r = c.endsWith(":"); return l && r ? "center" : r ? "right" : l ? "left" : ""; });
        i += 2;
        const rows = [];
        while (i < lines.length && lines[i].includes("|") && lines[i].trim() !== "") { rows.push(splitRow(lines[i])); i++; }
        root.appendChild(buildTable(header, aligns, rows));
        continue;
      }
      const h = line.match(/^(#{1,4})\s+(.*)$/);
      if (h) { const e2 = document.createElement("h" + h[1].length); e2.innerHTML = inlineMd(h[2]); root.appendChild(e2); i++; continue; }
      if (/^\s*([-*_])\1\1+\s*$/.test(line)) { root.appendChild(document.createElement("hr")); i++; continue; }
      if (/^\s*>\s?/.test(line)) { const buf = []; while (i < lines.length && /^\s*>\s?/.test(lines[i])) { buf.push(lines[i].replace(/^\s*>\s?/, "")); i++; } const bq = document.createElement("blockquote"); bq.innerHTML = inlineMd(buf.join(" ")); root.appendChild(bq); continue; }
      if (/^\s*([-*+]|\d+\.)\s+/.test(line)) {
        const items = [];
        while (i < lines.length) {
          const m = lines[i].match(/^(\s*)([-*+]|\d+\.)\s+(.*)$/);
          if (m) { items.push({ indent: m[1].length, ordered: /\d+\./.test(m[2]), start: parseInt(m[2], 10) || 1, text: m[3] }); i++; continue; }
          if (items.length && /^\s+\S/.test(lines[i]) && lines[i].trim() !== "") { items[items.length - 1].text += " " + lines[i].trim(); i++; continue; }
          break;
        }
        root.appendChild(buildList(items));
        continue;
      }
      if (line.trim() === "") { i++; continue; }
      const buf = [line]; i++;
      while (i < lines.length && lines[i].trim() !== "" && !/^(```|#{1,4}\s|\s*>\s?|\s*([-*+]|\d+\.)\s)/.test(lines[i])) { buf.push(lines[i]); i++; }
      const p = document.createElement("p"); p.innerHTML = inlineMd(buf.join(" ")); root.appendChild(p);
    }
    queueMath(root);
    return root;
  }

  /* ---------------- model selector ("Auto" via the provider chain) ---------------- */
  // The backend now picks the model itself (strict provider chain), so the selector is
  // "Auto" by default. If a server still advertises `availableModels`, they are offered as
  // an optional override; with an empty list the control renders a single harmless
  // "Auto" option and selectedModel() returns undefined (=> server decides).
  function modelSelect() {
    const field = el(`
      <div class="field" style="margin:0">
        <label class="label">Model</label>
        <select class="select model-select" aria-label="Model"><option value="">Auto (best available via provider chain)</option></select>
        <div class="hint model-hint">Chosen automatically; the model used is shown with each result.</div>
      </div>`);
    const sel = field.querySelector(".model-select");
    const populate = (cfg) => {
      // modelPolicy present => the server's provider chain chooses; never offer a stale override list.
      const models = (cfg && !cfg.modelPolicy && Array.isArray(cfg.availableModels)) ? cfg.availableModels : [];
      models.forEach((m) => {
        if (!m || !m.id) return;
        const o = document.createElement("option");
        o.value = m.id; o.textContent = m.label || m.id;
        sel.append(o);
      });
      if (sel.options.length <= 1) sel.disabled = true;   // nothing to choose: keep it visibly inert
    };
    if (window.AppConfig && window.AppConfig.loaded) populate(window.AppConfig);
    else if (window.AppConfig && window.AppConfig.ready) window.AppConfig.ready.then(populate);
    return field;
  }
  function selectedModel(container) {
    const sel = container && container.querySelector(".model-select");
    const value = sel && sel.value;
    return value || undefined;
  }

  /* ---------------- history panel (per-user saved items: list / search / open / delete) ---------------- */
  // Reusable across every tool. Reads from /api/history/<kind> (scoped to the signed-in
  // user server-side). Returns a .panel element with a .refresh() method so a view can
  // refresh the list after generating a new item.
  //   ui.historyPanel("quizzes", { title: "Saved quizzes", onOpen: (fullDoc) => {...} })
  function historyPanel(kind, opts) {
    opts = opts || {};
    const heading = opts.title || "History";
    const panel = el(`
      <div class="panel stack-sm history-panel" data-kind="${esc(kind)}">
        <div class="between">
          <div class="section-head" style="margin:0"><h3>${esc(heading)}</h3></div>
          <span class="row" style="gap:.3rem;flex:none">
            <button class="icon-btn is-danger hist-clear" type="button" title="Clear all" aria-label="Clear all">${iconHtml("trash")}</button>
            <button class="icon-btn hist-refresh" type="button" title="Refresh" aria-label="Refresh">${iconHtml("refresh")}</button>
          </span>
        </div>
        <input class="input hist-search" type="search" placeholder="Search…" aria-label="Search ${esc(heading)}" />
        <div class="hist-list stack-sm"></div>
      </div>`);
    const search = panel.querySelector(".hist-search");
    const listEl = panel.querySelector(".hist-list");
    const clearBtn = panel.querySelector(".hist-clear");

    const fmtDate = (iso) => { try { return new Date(iso).toLocaleString(); } catch (_) { return ""; } };

    // Inline rename: swap the title for an input; Enter/blur saves (PATCH), Escape cancels.
    function startRename(r, item) {
      const titleEl = r.querySelector(".hist-title");
      if (!titleEl) return;
      const input = el(`<input class="input hist-rename" type="text" aria-label="Rename ${esc(item.title || "item")}" />`);
      input.value = item.title || "";
      titleEl.replaceWith(input);
      input.focus(); input.select();
      let done = false;
      const finish = async (save) => {
        if (done) return; done = true;
        const val = input.value.trim();
        if (save && val && val !== item.title) {
          try { await window.api.patch(`/history/${kind}/${encodeURIComponent(item.id)}`, { title: val }); item.title = val; toast("Renamed.", "ok"); }
          catch (e) { toast(e.message, "error"); }
        }
        const restored = el(`<span class="hist-title"></span>`);
        restored.textContent = item.title || "(untitled)";
        input.replaceWith(restored);
      };
      input.addEventListener("click", (e) => e.stopPropagation());
      input.addEventListener("keydown", (e) => {
        e.stopPropagation();
        if (e.key === "Enter") { e.preventDefault(); finish(true); }
        else if (e.key === "Escape") { e.preventDefault(); finish(false); }
      });
      input.addEventListener("blur", () => finish(true));
    }

    function row(item) {
      const r = el(`
        <div class="hist-item row between" role="button" tabindex="0">
          <span class="hist-meta">
            <span class="hist-title">${esc(item.title || "(untitled)")}</span>
            <span class="muted hist-date">${esc(fmtDate(item.created_at))}</span>
          </span>
          <span class="hist-actions row" style="gap:.15rem;flex:none">
            <button class="icon-btn hist-edit" type="button" title="Rename" aria-label="Rename">${iconHtml("edit")}</button>
            <button class="icon-btn is-danger hist-del" type="button" title="Delete" aria-label="Delete">${iconHtml("trash")}</button>
          </span>
        </div>`);
      const open = async () => {
        if (!opts.onOpen) return;
        try { opts.onOpen(await window.api.get(`/history/${kind}/${encodeURIComponent(item.id)}`)); }
        catch (e) { toast(e.message, "error"); }
      };
      r.addEventListener("click", (ev) => { if (!ev.target.closest(".hist-actions") && !ev.target.closest(".hist-rename")) open(); });
      // role="button" elements must respond to both Enter and Space per the standard
      // button activation contract; Space also needs preventDefault so it doesn't scroll the list.
      r.addEventListener("keydown", (ev) => {
        if (ev.target !== r) return;
        if (ev.key === "Enter") open();
        else if (ev.key === " " || ev.key === "Spacebar") { ev.preventDefault(); open(); }
      });
      r.querySelector(".hist-edit").addEventListener("click", (ev) => { ev.stopPropagation(); startRename(r, item); });
      r.querySelector(".hist-del").addEventListener("click", async (ev) => {
        ev.stopPropagation();
        try { await window.api.del(`/history/${kind}/${encodeURIComponent(item.id)}`); r.remove(); toast("Deleted.", "ok"); }
        catch (e) { toast(e.message, "error"); }
        if (!listEl.querySelector(".hist-item")) load(search.value.trim());
      });
      return r;
    }

    async function load(q) {
      if (!window.AppConfig || !window.AppConfig.persistenceEnabled) {
        search.style.display = "none";
        clearBtn.style.display = "none";
        listEl.replaceChildren(el(`<div class="muted hist-empty">History is off — set <code>MONGODB_URI</code> in .env to save items.</div>`));
        return;
      }
      listEl.replaceChildren(spinner("Loading…"));
      try {
        const data = await window.api.get(`/history/${kind}` + (q ? `?q=${encodeURIComponent(q)}` : ""));
        const items = (data && data.items) || [];
        clear(listEl);
        if (!items.length) {
          listEl.append(el(`<div class="muted hist-empty">${q ? "No matches." : "Nothing saved yet."}</div>`));
          return;
        }
        items.forEach((item) => listEl.append(row(item)));
      } catch (e) {
        // 401 = the session isn't established (signed out, or — pre auth-ready-gating —
        // the token hadn't loaded yet). Show a quiet hint, never an error toast: this is
        // what used to flash "Sign in or continue as a guest…" on every reload.
        if (e.status === 401) {
          listEl.replaceChildren(el(`<div class="muted hist-empty">Sign in to see your saved items.</div>`));
        } else {
          listEl.replaceChildren(el(`<div class="muted hist-empty">Couldn’t load history.</div>`));
          toast(e.message, "error");
        }
      }
    }

    let debounce;
    search.addEventListener("input", () => { clearTimeout(debounce); debounce = setTimeout(() => load(search.value.trim()), 250); });
    panel.querySelector(".hist-refresh").addEventListener("click", () => load(search.value.trim()));
    clearBtn.addEventListener("click", async () => {
      if (!listEl.querySelector(".hist-item")) { toast("Nothing to clear.", "info"); return; }
      if (!confirm(`Clear all ${heading.toLowerCase()}? This can't be undone.`)) return;
      try { await window.api.del(`/history/${kind}`); toast("Cleared.", "ok"); }
      catch (e) { toast(e.message, "error"); }
      load("");
    });
    panel.refresh = () => load(search.value.trim());

    if (window.AppConfig && window.AppConfig.loaded) load("");
    else if (window.AppConfig && window.AppConfig.ready) window.AppConfig.ready.then(() => load(""));
    return panel;
  }

  // Holds the currently-open drawer's close() so other code paths (re-toggling the
  // same button, or navigating away mid-open) can tear it down through the one
  // routine that also removes its Escape/Tab listener and restores focus.
  let _closeHistDrawer = null;

  function openHistoryDrawer(kind, opts) {
    opts = opts || {};
    const anchor = opts.anchor || null;
    // Whoever had focus when the drawer opened (usually the anchor button) — restored on close
    // so keyboard/screen-reader users land back where they were instead of at the top of <body>.
    const opener = document.activeElement;

    if (_closeHistDrawer) { _closeHistDrawer(); return; }

    const backdrop = el(`<div id="histPopoverBg" class="hist-popover-backdrop"></div>`);
    const popover = el(`<div id="histPopover" class="hist-popover stack-sm" role="dialog" aria-modal="true" aria-labelledby="histPopoverTitle"></div>`);
    // On phones the anchored dropdown is cramped and often clips off-screen — render
    // a full-width bottom sheet instead (CSS owns its layout via .hist-popover--sheet).
    const isMobile = window.matchMedia("(max-width: 560px)").matches;

    const close = () => {
      if (_closeHistDrawer !== close) return; // already closed / superseded
      _closeHistDrawer = null;
      backdrop.classList.remove("hist-popover-backdrop--open");
      document.body.classList.remove("hist-scroll-lock");
      document.removeEventListener("keydown", onKeydown, true);
      popover.remove();
      setTimeout(() => backdrop.remove(), 200);
      // Return focus to whatever opened the drawer, so long as it's still on the page.
      const returnTo = (anchor && anchor.isConnected) ? anchor : (opener && opener.isConnected ? opener : null);
      if (returnTo) returnTo.focus();
    };
    _closeHistDrawer = close;
    backdrop.addEventListener("mousedown", close);
    backdrop.addEventListener("touchstart", (e) => { e.preventDefault(); close(); }, { passive: false });

    const head = el(`<div class="hist-popover-head">
      <span class="hist-popover-title" id="histPopoverTitle">${esc(opts.title || "History")}</span>
      <button class="icon-btn" type="button" title="Close" aria-label="Close">${iconHtml("close")}</button>
    </div>`);
    const closeBtn = head.querySelector("button");
    closeBtn.addEventListener("click", close);

    popover.append(head, historyPanel(kind, {
      title: "",
      onOpen: (item) => { close(); opts.onOpen && opts.onOpen(item); },
    }));

    // Escape closes the drawer; Tab is lightly trapped so keyboard focus can't leak
    // back into the (backgrounded) page while the dialog is open.
    function onKeydown(e) {
      if (e.key === "Escape") { e.preventDefault(); close(); return; }
      if (e.key !== "Tab") return;
      const focusable = [...popover.querySelectorAll("button, [href], input, select, textarea, [tabindex]:not([tabindex='-1'])")]
        .filter((n) => !n.disabled && n.offsetParent !== null);
      if (!focusable.length) return;
      const first = focusable[0], last = focusable[focusable.length - 1];
      if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
      else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
    }
    document.addEventListener("keydown", onKeydown, true);

    document.body.append(backdrop, popover);
    requestAnimationFrame(() => backdrop.classList.add("hist-popover-backdrop--open"));
    closeBtn.focus();

    if (isMobile) {
      // Full-width bottom sheet — layout/animation handled by CSS. Lock the page
      // behind it so a scroll gesture moves the list, not the page underneath.
      popover.classList.add("hist-popover--sheet");
      document.body.classList.add("hist-scroll-lock");
    } else if (anchor) {
      // Position below and right-aligned with the anchor button (clamped to viewport)
      const rect = anchor.getBoundingClientRect();
      const right = Math.max(12, window.innerWidth - rect.right);
      const top = rect.bottom + 8;
      // Cap at ~70% of the viewport so the list scrolls instead of running long,
      // but never overflow the space actually available below the button.
      const fit = Math.max(200, window.innerHeight - top - 16);
      const maxH = Math.min(fit, Math.round(window.innerHeight * 0.7));
      popover.style.cssText = `position:fixed;top:${top}px;right:${right}px;width:320px;max-height:${maxH}px;z-index:200`;
    } else {
      popover.style.cssText = `position:fixed;top:80px;right:24px;width:320px;max-height:70vh;z-index:200`;
    }
  }

  /* ---------------- clean print (PDF via the browser's print dialog) ----------------
     Opens a standalone window with a minimal, print-tuned document (white page,
     dark text, document typography) and the rendered markdown HTML inside.
     Far more reliable than @media print visibility hacks over the glassy app
     shell — dark-theme colors, fixed overlays and backdrop filters all used to
     leak into the printed PDF. */
  function printDoc(title, contentHtml, modelLabel) {
    const w = window.open("", "_blank");
    if (!w) { toast("Allow pop-ups to print.", "error"); return; }
    w.document.write(`<!DOCTYPE html><html><head><meta charset="utf-8">
<title>${esc(title || "Document")}</title>
<style>
  @page { margin: 16mm 14mm; }
  * { box-sizing: border-box; }
  html, body { background: #fff; color: #111; margin: 0; }
  body { font: 11pt/1.55 Georgia, "Times New Roman", serif; padding: 24px 28px; }
  .doc-title { font: 700 17pt/1.2 -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
               letter-spacing: -.01em; margin: 0 0 2pt; }
  .doc-sub { color: #666; font: 9pt -apple-system, "Segoe UI", sans-serif; margin: 0 0 14pt;
             padding-bottom: 8pt; border-bottom: 1.5pt solid #111; }
  h1, h2, h3, h4 { font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
                   letter-spacing: -.01em; line-height: 1.25; margin: 14pt 0 5pt;
                   break-after: avoid-page; }
  h1 { font-size: 14.5pt; } h2 { font-size: 12.5pt; }
  h3 { font-size: 11.5pt; } h4 { font-size: 11pt; }
  p { margin: 5pt 0; }
  ul, ol { margin: 5pt 0 5pt 16pt; padding: 0; }
  li { margin: 2.5pt 0; }
  strong { color: #000; }
  code { font: 9.5pt "SF Mono", Menlo, Consolas, monospace; background: #f2f2f2;
         padding: .5pt 3pt; border-radius: 3pt; }
  pre { background: #f6f6f6; border: .75pt solid #ddd; border-radius: 4pt;
        padding: 7pt 9pt; overflow-x: hidden; white-space: pre-wrap;
        word-wrap: break-word; break-inside: avoid-page; }
  pre code { background: none; padding: 0; }
  table { border-collapse: collapse; width: 100%; margin: 7pt 0; font-size: 10pt;
          break-inside: avoid-page; }
  th, td { border: .75pt solid #bbb; padding: 4pt 7pt; text-align: left; vertical-align: top; }
  th { background: #f0f0f0; font-family: -apple-system, "Segoe UI", sans-serif; }
  blockquote { margin: 7pt 0; padding: 2pt 0 2pt 10pt; border-left: 2.5pt solid #888; color: #444; }
  hr { border: 0; border-top: .75pt solid #ccc; margin: 10pt 0; }
  img { max-width: 100%; }
  /* strip app-specific chrome that travels inside rendered markdown */
  .cb-copy, .codeblock-head button { display: none !important; }
</style></head><body>
<div class="doc-title">${esc(title || "Document")}</div>
<div class="doc-sub">Study Assistant Hub${modelLabel ? " · " + esc(modelLabel) : ""}</div>
${contentHtml}
</body></html>`);
    w.document.close();
    // Give the new document a beat to lay out before invoking the dialog.
    setTimeout(() => { try { w.focus(); w.print(); } catch (_) {} }, 350);
  }

  /* ---------------- artifact deep-link handoff ---------------- */
  // A chat artifact card stashes {kind, id} here, navigates to the tool's route,
  // and the destination view claims it on mount to open the real saved item.
  let _pendingArtifact = null;
  function openArtifact(kind, id, route) {
    _pendingArtifact = { kind, id };
    location.hash = route;
  }
  function takeArtifact(kinds) {
    if (_pendingArtifact && kinds.includes(_pendingArtifact.kind)) {
      const art = _pendingArtifact;
      _pendingArtifact = null;
      return art;
    }
    return null;
  }

  // ---- on-demand script loading (views are only fetched on first navigation) ----
  const scriptP = new Map();
  function loadScript(src) {
    if (scriptP.has(src)) return scriptP.get(src);
    const p = new Promise((resolve, reject) => {
      const sc = document.createElement("script");
      sc.src = assetUrl(src);
      sc.async = true;
      sc.onload = () => resolve();
      sc.onerror = () => { scriptP.delete(src); reject(new Error("Couldn’t download this tool (" + src.split("/").pop() + "). Check your connection and retry.")); };
      document.head.appendChild(sc);
    });
    scriptP.set(src, p);
    return p;
  }

  // ---- icon system: <svg class="ic"><use href="#i-name"/></svg> from the sprite in index.html ----
  const iconHtml = (n) => `<svg class="ic" viewBox="0 0 24 24" aria-hidden="true" focusable="false"><use href="#i-${n}"/></svg>`;
  function icon(name, o = {}) {
    const n = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    n.setAttribute("class", "ic" + (o.cls ? " " + o.cls : ""));
    n.setAttribute("viewBox", "0 0 24 24");
    n.setAttribute("aria-hidden", "true");
    n.setAttribute("focusable", "false");
    if (o.size) { n.style.width = n.style.height = typeof o.size === "number" ? o.size + "px" : o.size; }
    const u = document.createElementNS("http://www.w3.org/2000/svg", "use");
    u.setAttribute("href", "#i-" + name);
    n.appendChild(u);
    if (o.label) { n.setAttribute("role", "img"); n.setAttribute("aria-label", o.label); n.removeAttribute("aria-hidden"); }
    return n;
  }
  // Icon-only button: always aria-label + title tooltip; aria-pressed when `pressed` is boolean.
  function iconBtn(o = {}) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "icon-btn" + (o.danger ? " is-danger" : "") + (o.cls ? " " + o.cls : "");
    b.append(icon(o.icon, { size: o.size }));
    if (o.text) { const t = document.createElement("span"); t.className = "icon-btn-text"; t.textContent = o.text; b.append(t); b.classList.add("has-text"); }
    setIconBtn(b, o);
    if (o.onClick) b.addEventListener("click", o.onClick);
    return b;
  }
  function setIconBtn(b, o) {
    if (o.label) { b.setAttribute("aria-label", o.label); b.dataset.tip = o.label; b.removeAttribute("title"); }
    if (o.icon) { const u = b.querySelector("use"); if (u) u.setAttribute("href", "#i-" + o.icon); }
    if (typeof o.pressed === "boolean") b.setAttribute("aria-pressed", String(o.pressed));
    return b;
  }

  /* ---------------- global tooltips: one delegated listener set, no per-button listeners ----------------
     Icon-only controls (svg glyph, no visible text) show their aria-label / data-tip after a short hover
     delay, on keyboard focus-visible and on touch long-press. Native title attributes are moved to data-tip
     on first contact so the browser's own (double) tooltip never appears. */
  (function installTooltips() {
    const SEL = "button, a[href], [role=button], [data-tip]";
    let tip = null, cur = null, showT = 0, pressT = 0, lastShown = 0, swallow = null;
    const reduce = matchMedia("(prefers-reduced-motion: reduce)");
    function hasVisibleText(b) {
      const w = document.createTreeWalker(b, NodeFilter.SHOW_TEXT);
      let n;
      while ((n = w.nextNode())) {
        const t = n.nodeValue.trim();
        if (t.length > 2 && n.parentElement.getBoundingClientRect().width > 2) return true;
      }
      return false;
    }
    function textFor(el) {
      if (el.title) { if (!el.dataset.tip) el.dataset.tip = el.title; el.removeAttribute("title"); }
      const explicit = el.dataset.tip;
      if (explicit) return explicit;
      const label = el.getAttribute("aria-label");
      if (label && el.querySelector("svg") && !hasVisibleText(el)) return label;
      return "";
    }
    function target(node) {
      const el = node && node.closest ? node.closest(SEL) : null;
      return el && !el.closest("[inert]") ? el : null;
    }
    function ensure() {
      if (tip) return tip;
      tip = document.createElement("div");
      tip.className = "sah-tip"; tip.id = "sahTip"; tip.setAttribute("role", "tooltip"); tip.hidden = true;
      document.body.appendChild(tip);
      return tip;
    }
    function hide() {
      clearTimeout(showT); clearTimeout(pressT);
      if (tip && !tip.hidden) { tip.hidden = true; tip.classList.remove("is-on"); lastShown = Date.now(); }
      if (cur && cur.dataset.tipDesc) { cur.removeAttribute("aria-describedby"); delete cur.dataset.tipDesc; }
      cur = null;
    }
    function show(el) {
      const text = textFor(el);
      if (!text || !el.isConnected) return;
      const t = ensure();
      t.replaceChildren(document.createTextNode(text));
      if (el.dataset.shortcut) { const k = document.createElement("kbd"); k.textContent = el.dataset.shortcut; t.append(" ", k); }
      t.hidden = false; t.style.left = "0px"; t.style.top = "0px";
      const r = el.getBoundingClientRect(), tw = t.offsetWidth, th = t.offsetHeight, vw = document.documentElement.clientWidth;
      let top = r.top - th - 8, below = false;
      if (top < 6) { top = r.bottom + 8; below = true; }
      const left = Math.max(8, Math.min(vw - tw - 8, r.left + r.width / 2 - tw / 2));
      t.style.left = Math.round(left) + "px"; t.style.top = Math.round(top) + "px";
      t.classList.toggle("is-below", below);
      t.classList.toggle("no-anim", reduce.matches);
      requestAnimationFrame(() => t.classList.add("is-on"));
      cur = el;
      if (!el.getAttribute("aria-label")) { el.setAttribute("aria-describedby", "sahTip"); el.dataset.tipDesc = "1"; }
      lastShown = Date.now();
    }
    const schedule = (el, delay) => {
      clearTimeout(showT);
      if (!textFor(el)) return;
      const d = Date.now() - lastShown < 350 ? 0 : delay;
      showT = setTimeout(() => show(el), d);
    };
    document.addEventListener("pointerover", (e) => {
      if (e.pointerType === "touch") return;
      const el = target(e.target);
      if (!el || el === cur) return;
      hide(); schedule(el, 120);
    });
    document.addEventListener("pointerout", (e) => {
      if (e.pointerType === "touch") return;
      const el = target(e.target);
      if (el && (!e.relatedTarget || !el.contains(e.relatedTarget))) hide();
    });
    document.addEventListener("focusin", (e) => {
      const el = target(e.target);
      if (!el) return;
      let fv = true; try { fv = el.matches(":focus-visible"); } catch (_) { /* old engines */ }
      if (fv) { hide(); schedule(el, 0); }
    });
    document.addEventListener("focusout", hide);
    document.addEventListener("keydown", (e) => { if (e.key === "Escape") hide(); }, true);
    document.addEventListener("scroll", hide, true);
    window.addEventListener("resize", hide);
    document.addEventListener("click", (e) => {
      if (swallow && swallow.contains(e.target)) { e.preventDefault(); e.stopPropagation(); swallow = null; return; }
      hide();
    }, true);
    // touch long-press (~450 ms) reveals the label; the click that follows the release is swallowed so a
    // destructive control isn't activated by someone only reading its name.
    document.addEventListener("pointerdown", (e) => {
      if (e.pointerType !== "touch") { hide(); return; }
      const el = target(e.target);
      hide();
      if (!el || !textFor(el)) return;
      pressT = setTimeout(() => { show(el); swallow = el; setTimeout(() => { if (swallow === el) swallow = null; hide(); }, 1800); }, 450);
    });
    const cancelPress = () => clearTimeout(pressT);
    document.addEventListener("pointerup", cancelPress);
    document.addEventListener("pointercancel", cancelPress);
    document.addEventListener("contextmenu", (e) => { if (swallow || pressT) { const el = target(e.target); if (el && textFor(el)) e.preventDefault(); } });
  })();

  window.ui = {
    icon, iconBtn, setIconBtn, iconHtml: (n) => `<svg class="ic" viewBox="0 0 24 24" aria-hidden="true" focusable="false"><use href="#i-${n}"/></svg>`,
    el, clear, esc, spinner, empty, toast, md: renderMarkdown, copy: copyToClipboard,
    modelSelect, selectedModel, historyPanel, openHistoryDrawer, printDoc,
    openArtifact, takeArtifact,
    renderModelLabel, attachModelLabel, updateModelLabel, errorBlock, loadScript,
    // AbortSignal tied to the current view (aborted on navigation) for views' own fetches.
    viewSignal: () => viewCtl.signal,
    // run `fn` on the next frame, coalescing repeated calls (streaming / resize work)
    rafBatch: (fn) => { let id = 0, last; return (...a) => { last = a; if (!id) id = requestAnimationFrame(() => { id = 0; fn(...last); }); }; },
    debounce: (fn, ms = 250) => { let t; const d = (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; d.cancel = () => clearTimeout(t); return d; },
  };

  /* ---------------- view registry ---------------- */
  window.Views = window.Views || {};
  const TITLES = {
    chat: "Apex Agent", quiz: "Quiz Generator", flashcards: "Flashcards",
    planner: "Study Planner", mindmap: "Mind Map", subjects: "Add a Subject",
    exam: "Mock Exam", teachback: "Teach-back Tutor", predictor: "Question Predictor",
    progress: "Progress", cheatsheet: "Cheat Sheet", scannotes: "Scan Notes", library: "Library",
  };
  const ROUTES = Object.keys(TITLES);
  const home = document.getElementById("home");
  const view = document.getElementById("view");
  const SITE_TITLE = document.title; // "Study Assistant Hub", captured before any route changes it
  let current = null;

  // Which script(s) each view needs (lazy: fetched on first navigation, then cached by the
  // browser/CDN under the ?v= stamp). The chat view also pulls its optional enhancers.
  const VIEW_DEPS = {
    chat: ["/assets/views/chat.js", "/assets/artifact-inline.js", "/assets/voice.js", "/assets/grade-ui.js"],
    quiz: ["/assets/views/quiz.js", "/assets/grade-ui.js"],
    exam: ["/assets/views/exam.js", "/assets/grade-ui.js"],
    flashcards: ["/assets/views/flashcards.js", "/assets/views/flashcards-decks.js"],
    library: ["/assets/views/library.js"],       // lazy: never part of the bootstrap path
  };
  const viewFiles = (name) => VIEW_DEPS[name] || [`/assets/views/${name}.js`];
  const loadView = (name) => {
    if (window.Views[name]) return Promise.resolve();
    const [main, ...extra] = viewFiles(name);
    // The main view script first (it defines window.Views[name]); enhancers in parallel after.
    return loadScript(main).then(() => { extra.forEach((f) => loadScript(f).catch(() => {})); });
  };
  let navSeq = 0;

  function skeleton() {
    return el(`<div class="skeleton-wrap" role="status" aria-label="Loading"><div class="skeleton sk-line sk-w60"></div><div class="skeleton sk-block"></div><div class="skeleton sk-line sk-w40"></div></div>`);
  }

  function render() {
    const name = (location.hash.replace(/^#\/?/, "") || "home");
    const seq = ++navSeq;
    // Navigating away always closes a stray history sheet (also frees its scroll lock and
    // removes its keydown listener — routed through the drawer's own close() for that).
    if (_closeHistDrawer) _closeHistDrawer();
    if (current && Views[current] && typeof Views[current].unmount === "function") {
      try { Views[current].unmount(); } catch (_) {}
    }
    newViewScope();            // drop the previous view's in-flight GETs
    lastAction = null;
    // The footer belongs to the landing page only — hide it inside tool views.
    const foot = document.querySelector(".foot");
    if (foot) foot.hidden = ROUTES.includes(name);
    if (!ROUTES.includes(name)) {
      home.hidden = false; view.hidden = true; current = null;
      document.title = SITE_TITLE;
      focusRouteHeading(document.getElementById("homeHeading"));
      return;
    }
    home.hidden = true; view.hidden = false; current = name;
    const v0 = Views[name];
    view.dataset.accent = (v0 && v0.accent) || name;
    clear(view);
    document.title = `${(v0 && v0.title) || TITLES[name]} · ${SITE_TITLE}`;
    const head = el(`<div class="view-head"><a class="btn-ghost" href="#home">← Hub</a><h2 tabindex="-1">${esc((v0 && v0.title) || TITLES[name])}</h2><span class="spacer"></span></div>`);
    const body = el(`<div class="view-body"></div>`);
    view.append(head, body);
    window.scrollTo({ top: 0, behavior: "auto" });
    focusRouteHeading(head.querySelector("h2"));
    // Remember the last primary action button so a toast "Retry" can re-fire it.
    body.addEventListener("click", (e) => {
      const b = e.target.closest && e.target.closest("button.btn-accent, button.btn-primary, button.btn-lg, #chatSend");
      if (b) lastAction = b;
    }, true);

    const stale = () => seq !== navSeq;
    const needsLoad = !Views[name];
    if (needsLoad) body.append(skeleton());
    loadView(name).then(() => {
      if (stale()) return;
      const v = Views[name];
      if (!v || typeof v.mount !== "function") {
        clear(body); body.append(empty("This tool isn’t available yet.")); return;
      }
      view.dataset.accent = v.accent || name;
      if (v.title) { document.title = `${v.title} · ${SITE_TITLE}`; const h = head.querySelector("h2"); if (h) h.textContent = v.title; }
      if (needsLoad) clear(body);
      return Promise.resolve(v.mount(body)).then(() => {
        if (!stale()) window.dispatchEvent(new CustomEvent("sah:viewmounted", { detail: { name } }));
      });
    }).catch((e) => {
      if (stale()) return;
      clear(body);
      body.append(errorBlock(e, () => render()));
    });
  }
  // Moves focus to the new route's heading after a hash navigation, the way client-side
  // routers are expected to (WAI-ARIA APG) — screen reader users otherwise stay anchored
  // wherever focus was on the previous page. Skipped on the very first load (no route
  // change has actually happened yet, so stealing focus from the page would be surprising).
  let _hasNavigated = false;
  function focusRouteHeading(heading) {
    if (!heading || !_hasNavigated) { _hasNavigated = true; return; }
    heading.focus({ preventScroll: true });
  }
  addEventListener("hashchange", render);

  /* ---------------- theme (defaults to light; a manual toggle persists) ---------------- */
  let saved = null;
  try { saved = localStorage.getItem("sah-theme"); } catch (_) {}
  // Editorial is a light-only skin: never apply a stored "dark" under it (half-applied dark rules
  // elsewhere, e.g. low-contrast links). The stored choice is kept and returns with Glass.
  if (saved && root.getAttribute("data-skin") !== "editorial") root.setAttribute("data-theme", saved);
  // No saved choice → keep the document default (light, set on <html> in index.html).
  const themeToggle = document.getElementById("themeToggle");
  const syncThemeToggle = () => themeToggle.setAttribute("aria-pressed", String(root.getAttribute("data-theme") === "dark"));
  syncThemeToggle(); // reflect whatever theme was just applied (saved choice or the light default)
  themeToggle.addEventListener("click", () => {
    const next = root.getAttribute("data-theme") === "dark" ? "light" : "dark";
    root.setAttribute("data-theme", next);
    try { localStorage.setItem("sah-theme", next); } catch (_) {}
    syncThemeToggle();
  });

  /* ---------------- skin (glass | editorial; applied early by an inline head script) ---------------- */
  const skinSwitch = document.getElementById("skinSwitch");
  if (skinSwitch) {
    const skinButtons = [...skinSwitch.querySelectorAll("[data-skin-choice]")];
    const syncSkin = (skin) => {
      skinButtons.forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.skinChoice === skin)));
    };
    // Reflect the skin the head script already set (default "glass").
    syncSkin(root.getAttribute("data-skin") || "editorial");
    skinButtons.forEach((btn) => {
      btn.addEventListener("click", () => {
        const skin = btn.dataset.skinChoice;
        syncSkin(skin);
        try { localStorage.setItem("sah-skin-v2", skin); } catch (_) {}
        // The editorial stylesheet + its fonts are fetched only when that skin is first picked
        // (index.html head defines __sahSkinAssets); apply once they're in to avoid a flash.
        const apply = () => {
          root.setAttribute("data-skin", skin);
          let th = null; try { th = localStorage.getItem("sah-theme"); } catch (_) {}
          root.setAttribute("data-theme", skin === "editorial" ? "light" : (th || "light"));
          syncThemeToggle();
        };
        if (typeof window.__sahSkinAssets === "function") window.__sahSkinAssets(skin).then(apply, apply);
        else apply();
      });
    });
  }

  /* ---------------- ambient: cursor glow + orb parallax ---------------- */
  // Idle the rAF loop when parallax deltas settle (cursor still) or the tab is
  // backgrounded — otherwise this runs ~60fps forever for a decorative effect.
  // Restart on the next pointermove. Complements perf-motion.js (CSS pause).
  const glow = document.getElementById("glow");
  const orbs = [...document.querySelectorAll(".orb")];
  if (!reduce) {
    let rx = 0, ry = 0, tx = 0, ty = 0;
    let rafId = 0;
    let settledFrames = 0;
    const EPS = 0.0004;
    const SETTLED_N = 8;

    function tick() {
      rafId = 0;
      if (document.hidden) return;
      rx += (tx - rx) * 0.05; ry += (ty - ry) * 0.05;
      orbs.forEach((o, i) => { o.style.translate = `${rx * (i + 1) * 10}px ${ry * (i + 1) * 10}px`; });
      const dx = Math.abs(tx - rx), dy = Math.abs(ty - ry);
      if (dx < EPS && dy < EPS) {
        settledFrames++;
        if (settledFrames >= SETTLED_N) return; // stop — restart on next pointermove
      } else {
        settledFrames = 0;
      }
      rafId = requestAnimationFrame(tick);
    }
    function kick() {
      if (document.hidden || rafId) return;
      settledFrames = 0;
      rafId = requestAnimationFrame(tick);
    }

    addEventListener("pointermove", (e) => {
      // Editorial hides the glow + orbs (display:none): skip the work and the rAF loop entirely.
      if (root.getAttribute("data-skin") === "editorial") return;
      glow.style.transform = `translate3d(${e.clientX}px, ${e.clientY}px, 0)`;
      tx = e.clientX / innerWidth - 0.5; ty = e.clientY / innerHeight - 0.5;
      kick();
    }, { passive: true });
    document.addEventListener("visibilitychange", () => {
      if (document.hidden) {
        if (rafId) { cancelAnimationFrame(rafId); rafId = 0; }
      } else if (Math.abs(tx - rx) >= EPS || Math.abs(ty - ry) >= EPS) {
        kick();
      }
    }, { passive: true });
  }

  /* ---------------- home card 3D tilt + specular ---------------- */
  // Cache the rect on pointerenter rather than calling getBoundingClientRect() on every
  // pointermove: a forced-layout read on each mouse-move event (dozens per second, per
  // card) risks synchronous reflow and visible jank, especially alongside the orb/glow
  // rAF loop above that's concurrently writing transform styles.
  document.querySelectorAll("#home .card").forEach((card) => {
    let rect = null;
    card.addEventListener("pointerenter", () => { rect = card.getBoundingClientRect(); });
    card.addEventListener("pointermove", (e) => {
      if (!rect) rect = card.getBoundingClientRect();
      const px = (e.clientX - rect.left) / rect.width, py = (e.clientY - rect.top) / rect.height;
      card.style.setProperty("--mx", px * 100 + "%");
      card.style.setProperty("--my", py * 100 + "%");
      if (!reduce && root.getAttribute("data-skin") !== "editorial") card.style.transform = `rotateX(${(0.5 - py) * 12}deg) rotateY(${(px - 0.5) * 12}deg) translateZ(10px) scale(1.012)`;
    });
    card.addEventListener("pointerleave", () => {
      rect = null;
      card.style.transform = "";
      card.style.setProperty("--mx", "30%"); card.style.setProperty("--my", "0%");
    });
  });

  /* ---------------- boot: AI-availability banner + subject count + connectivity ---------------- */
  // App-level requests (not tied to any view) use their own never-aborted signal.
  const appSignal = new AbortController().signal;
  // /config is already in flight (started in <head>); no extra /health round trip is needed.
  // New backend: `chain` = {step: bool}; old backend: `groqKey` boolean. Only warn when the
  // server positively says NO provider key is available.
  function anyTrue(o) {
    if (o === true) return true;
    if (!o || typeof o !== "object") return false;
    return Object.values(o).some((v) => v === true || (v && typeof v === "object" && (v.configured === true || v.available === true || v.enabled === true || anyTrue(v))));
  }
  function showAiBanner(cfg) {
    const host = document.getElementById("keywarn");
    if (!host || host.querySelector(".ai-banner")) return;
    if (cfg.aiAvailable !== false) return;   // `llmAvailable` from /api/config (see config.js)
    host.appendChild(el(
      `<div class="keywarn glass ai-banner">⚠ <span>No AI provider key is configured on the server (set <code>ANTHROPIC_API_KEY</code>, <code>GROQ_API_KEY</code> or <code>GEMINI_API_KEY</code>), so the AI tools are unavailable.</span></div>`
    ));
  }
  if (window.AppConfig && window.AppConfig.ready) window.AppConfig.ready.then(showAiBanner);
  // The home subject count needs an identity: calling /subjects before the visitor is signed in
  // or chose guest only produces a 401. Run it once auth is known (auth.js fires "sah:authed"
  // each time an identity is established), or straight away when the app runs open (no Clerk key).
  let statLoaded = false;
  function paintSubjectCount(d) {
    const n = ((d && d.subjects) || []).length;
    const slot = document.getElementById("stat-subjects");
    if (slot) slot.textContent = n || "·";
  }
  function loadSubjectStat() {
    if (statLoaded) return;
    statLoaded = true;
    const warm = bootFromSession();                    // instant paint from this tab's last visit (same identity only)
    if (warm && warm.subjects) paintSubjectCount(warm.subjects);
    // One round trip for subjects + progress + recent chats; falls back to the plain endpoint.
    bootstrap().then((d) => (d && d.subjects) ? d.subjects : api.get("/subjects", { signal: appSignal }))
      .then(paintSubjectCount)
      .catch(() => { statLoaded = false; });
  }
  addEventListener("sah:authed", loadSubjectStat);
  if (window.AppConfig && window.AppConfig.ready) {
    window.AppConfig.ready.then((cfg) => {
      if (!cfg.clerkPublishableKey) loadSubjectStat();
      else if (window.Auth && (window.Auth.userId || window.Auth.isGuest())) loadSubjectStat();
    });
  }

  // Offline / slow-network handling: a polite, dismiss-on-reconnect banner.
  (function connectivity() {
    const host = document.getElementById("keywarn");
    if (!host) return;
    let bar = null;
    const sync = () => {
      if (navigator.onLine === false) {
        if (bar) return;
        bar = el(`<div class="keywarn glass offline-banner" role="status">📡 <span>You’re offline — changes and AI requests will fail until the connection returns.</span></div>`);
        host.appendChild(bar);
      } else if (bar) { bar.remove(); bar = null; }
    };
    addEventListener("online", sync); addEventListener("offline", sync); sync();
  })();

  /* ---------------- prefetch: warm a tool's script before the click lands ---------------- */
  (function prefetchViews() {
    const warm = (a) => {
      const m = a && a.getAttribute && (a.getAttribute("href") || "").match(/^#\/?([a-z]+)$/);
      if (m && ROUTES.includes(m[1])) loadView(m[1]).catch(() => {});
    };
    document.addEventListener("pointerover", (e) => warm(e.target.closest && e.target.closest("a[href^='#']")), { passive: true });
    document.addEventListener("focusin", (e) => warm(e.target.closest && e.target.closest("a[href^='#']")), { passive: true });
    document.addEventListener("touchstart", (e) => warm(e.target.closest && e.target.closest("a[href^='#']")), { passive: true });
    // After first paint, quietly fetch the most-used tool while the browser is idle.
    const idle = window.requestIdleCallback || ((f) => setTimeout(f, 2500));
    idle(() => { if (!navigator.connection || !navigator.connection.saveData) loadView("chat").catch(() => {}); }, { timeout: 4000 });
  })();

  /* ---------------- skip link ---------------- */
  // Fragment navigation scrolls to #mainContent in every browser, but focus doesn't
  // reliably follow onto a non-interactive tabindex="-1" target everywhere — move it
  // explicitly so keyboard users actually land in the content, not just visually near it.
  const skipLink = document.querySelector(".skip-link");
  if (skipLink) {
    skipLink.addEventListener("click", (e) => {
      const target = document.getElementById("mainContent");
      if (!target) return;
      e.preventDefault();
      target.focus();
      target.scrollIntoView();
    });
  }

  if (document.readyState === "loading") addEventListener("DOMContentLoaded", render);
  else render();
})();
