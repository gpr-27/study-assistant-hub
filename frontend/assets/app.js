/* ============================================================
   Study Assistant Hub — SPA runtime
   Provides: window.api, window.ui, window.Views, hash router,
   theme toggle, ambient interactions, Groq-key banner.

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
  async function handle(res) {
    if (!res.ok) {
      let detail;
      try { detail = (await res.json()).detail; } catch (_) {}
      const err = new Error(detail || `${res.status} ${res.statusText}`);
      err.status = res.status;     // let callers special-case auth (401) vs real failures
      throw err;
    }
    const ct = res.headers.get("content-type") || "";
    return ct.includes("application/json") ? res.json() : res.text();
  }
  // localStorage key for the anonymous guest id (written by auth.js). Read directly here
  // so the API layer never depends on auth.js's async load order.
  const GUEST_KEY = "sah-guest-id";
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
      if (gid) headers["X-Guest-Id"] = gid;
    } catch (_) { /* localStorage unavailable — proceed without the guest header */ }
    return headers;
  }
  window.api = {
    get: (p) => authHeaders().then((h) => fetch(apiUrl(p), { headers: h })).then(handle),
    post: (p, body) => authHeaders({ "Content-Type": "application/json" }).then((h) =>
      fetch(apiUrl(p), { method: "POST", headers: h, body: JSON.stringify(body || {}) })
    ).then(handle),
    postForm: (p, formData) => authHeaders().then((h) =>
      fetch(apiUrl(p), { method: "POST", headers: h, body: formData })
    ).then(handle),
    del: (p) => authHeaders().then((h) => fetch(apiUrl(p), { method: "DELETE", headers: h })).then(handle),
    patch: (p, body) => authHeaders({ "Content-Type": "application/json" }).then((h) =>
      fetch(apiUrl(p), { method: "PATCH", headers: h, body: JSON.stringify(body || {}) })
    ).then(handle),
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
    return el(`<div class="loading"><span class="spinner"></span><span>${esc(label)}</span></div>`);
  }
  function empty(message, icon = "✦") {
    return el(`<div class="empty"><div class="big">${icon}</div><div>${esc(message)}</div></div>`);
  }
  let toastTimer;
  function toast(message, type = "info", ms = 3200) {
    const wrap = document.getElementById("toasts");
    const t = el(`<div class="toast ${type === "error" ? "err" : type === "ok" ? "ok" : ""}">${esc(message)}</div>`);
    wrap.appendChild(t);
    setTimeout(() => { t.style.opacity = "0"; t.style.transition = "opacity .4s"; setTimeout(() => t.remove(), 400); }, ms);
  }
  // ---- rich, SAFE markdown renderer -> enhanced <div class="md"> element ----
  function inlineMd(text) {
    return String(text).split(/(`[^`]+`)/g).map((part) => {
      if (/^`[^`]+`$/.test(part)) return `<code class="inline">${esc(part.slice(1, -1))}</code>`;
      return esc(part)
        .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
        .replace(/__([^_]+)__/g, "<strong>$1</strong>")
        .replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<em>$2</em>")
        .replace(/~~([^~]+)~~/g, "<del>$1</del>")
        .replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
    }).join("");
  }

  function copyToClipboard(text, btn) {
    const reset = btn ? btn.textContent : "";
    const done = () => { if (!btn) return; btn.textContent = "Copied!"; btn.classList.add("copied"); setTimeout(() => { btn.textContent = reset; btn.classList.remove("copied"); }, 1500); };
    const fallback = () => { const ta = document.createElement("textarea"); ta.value = text; ta.style.position = "fixed"; ta.style.opacity = "0"; document.body.appendChild(ta); ta.select(); try { document.execCommand("copy"); } catch (_) {} ta.remove(); done(); };
    if (navigator.clipboard && navigator.clipboard.writeText) navigator.clipboard.writeText(text).then(done).catch(fallback);
    else fallback();
  }

  function buildCodeBlock(lang, code) {
    const wrap = el(`<div class="codeblock"></div>`);
    const head = el(`<div class="cb-head"><span class="cb-lang">${esc(lang || "code")}</span><button class="cb-copy" type="button">Copy</button></div>`);
    const pre = document.createElement("pre");
    const codeEl = document.createElement("code");
    if (lang) codeEl.className = "language-" + lang;
    codeEl.textContent = code;
    pre.appendChild(codeEl);
    wrap.append(head, pre);
    if (window.hljs) { try { window.hljs.highlightElement(codeEl); } catch (_) {} }
    head.querySelector(".cb-copy").addEventListener("click", () => copyToClipboard(code, head.querySelector(".cb-copy")));
    return wrap;
  }

  function buildList(items) {
    const rootList = document.createElement(items[0].ordered ? "ol" : "ul");
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
      li.innerHTML = inlineMd(it.text);
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

  function renderMarkdown(src) {
    const root = el(`<div class="md"></div>`);
    const lines = String(src == null ? "" : src).replace(/\r\n/g, "\n").split("\n");
    let i = 0;
    while (i < lines.length) {
      const line = lines[i];
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
          if (m) { items.push({ indent: m[1].length, ordered: /\d+\./.test(m[2]), text: m[3] }); i++; continue; }
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
    return root;
  }

  /* ---------------- dynamic model selector (config-driven; no hardcoded ids) ---------------- */
  // Returns a `.field` containing a <select.model-select> populated from
  // window.AppConfig.availableModels, with defaultModel preselected. The whole
  // list comes from the server's /config (driven by AVAILABLE_MODELS in .env), so
  // adding/removing/renaming models needs NO code change. Read the chosen id with
  // ui.selectedModel(container); it returns undefined when nothing is selected, so
  // callers can simply omit `model` and let the server use its configured default.
  function modelSelect() {
    const field = el(`
      <div class="field" style="margin:0">
        <label class="label">Model</label>
        <select class="select model-select" aria-label="Model"><option value="">Loading models…</option></select>
        <div class="hint model-hint"></div>
      </div>`);
    const sel = field.querySelector(".model-select");
    const hint = field.querySelector(".model-hint");
    const populate = (cfg) => {
      clear(sel);
      const models = (cfg && cfg.availableModels) || [];
      if (!models.length) {
        sel.append(el(`<option value="">No models configured</option>`));
        hint.textContent = "Set AVAILABLE_MODELS in .env.";
        return;
      }
      models.forEach((m) => sel.append(el(`<option value="${esc(m.id)}">${esc(m.label || m.id)}</option>`)));
      const preferred = cfg.defaultModel && models.some((m) => m.id === cfg.defaultModel)
        ? cfg.defaultModel : models[0].id;
      sel.value = preferred;
      hint.textContent = cfg.llmProvider ? `Provider: ${cfg.llmProvider}` : "";
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
            <button class="btn-ghost hist-clear" type="button" title="Clear all">Clear all</button>
            <button class="btn-ghost hist-refresh" type="button" title="Refresh" aria-label="Refresh">↻</button>
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
      const input = el(`<input class="input hist-rename" type="text" aria-label="Rename" />`);
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
            <button class="btn-ghost hist-edit" type="button" title="Rename" aria-label="Rename">✎</button>
            <button class="btn-ghost hist-del" type="button" title="Delete" aria-label="Delete">✕</button>
          </span>
        </div>`);
      const open = async () => {
        if (!opts.onOpen) return;
        try { opts.onOpen(await window.api.get(`/history/${kind}/${encodeURIComponent(item.id)}`)); }
        catch (e) { toast(e.message, "error"); }
      };
      r.addEventListener("click", (ev) => { if (!ev.target.closest(".hist-actions") && !ev.target.closest(".hist-rename")) open(); });
      r.addEventListener("keydown", (ev) => { if (ev.key === "Enter" && ev.target === r) open(); });
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

  function openHistoryDrawer(kind, opts) {
    opts = opts || {};
    const anchor = opts.anchor || null;

    const existing = document.getElementById("histPopover");
    if (existing) { existing.remove(); document.getElementById("histPopoverBg")?.remove(); return; }

    const backdrop = el(`<div id="histPopoverBg" class="hist-popover-backdrop"></div>`);
    const popover = el(`<div id="histPopover" class="hist-popover stack-sm"></div>`);
    // On phones the anchored dropdown is cramped and often clips off-screen — render
    // a full-width bottom sheet instead (CSS owns its layout via .hist-popover--sheet).
    const isMobile = window.matchMedia("(max-width: 560px)").matches;

    const close = () => {
      backdrop.classList.remove("hist-popover-backdrop--open");
      popover.classList.remove("hist-popover--in");
      document.body.classList.remove("hist-scroll-lock");
      popover.remove();
      setTimeout(() => backdrop.remove(), 200);
    };
    backdrop.addEventListener("mousedown", close);
    backdrop.addEventListener("touchstart", (e) => { e.preventDefault(); close(); }, { passive: false });

    const head = el(`<div class="hist-popover-head">
      <span class="hist-popover-title">${esc(opts.title || "History")}</span>
      <button class="btn-ghost" type="button" title="Close" style="padding:.25rem .5rem;font-size:1rem">✕</button>
    </div>`);
    head.querySelector("button").addEventListener("click", close);

    popover.append(head, historyPanel(kind, {
      title: "",
      onOpen: (item) => { close(); opts.onOpen && opts.onOpen(item); },
    }));

    document.body.append(backdrop, popover);
    requestAnimationFrame(() => backdrop.classList.add("hist-popover-backdrop--open"));

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

  window.ui = {
    el, clear, esc, spinner, empty, toast, md: renderMarkdown, copy: copyToClipboard,
    modelSelect, selectedModel, historyPanel, openHistoryDrawer,
  };

  /* ---------------- view registry ---------------- */
  window.Views = window.Views || {};
  const TITLES = { chat: "Chat Assistant", quiz: "Quiz Generator", flashcards: "Flashcards", planner: "Study Planner", mindmap: "Mind Map", subjects: "Add a Subject" };
  const ROUTES = Object.keys(TITLES);
  const home = document.getElementById("home");
  const view = document.getElementById("view");
  let current = null;

  function render() {
    const name = (location.hash.replace(/^#\/?/, "") || "home");
    // Navigating away always closes a stray history sheet and frees the scroll lock.
    document.getElementById("histPopover")?.remove();
    document.getElementById("histPopoverBg")?.remove();
    document.body.classList.remove("hist-scroll-lock");
    if (current && Views[current] && typeof Views[current].unmount === "function") {
      try { Views[current].unmount(); } catch (_) {}
    }
    if (!ROUTES.includes(name)) { home.hidden = false; view.hidden = true; current = null; return; }
    const v = Views[name];
    home.hidden = true; view.hidden = false; current = name;
    view.dataset.accent = (v && v.accent) || name;
    clear(view);
    const head = el(`<div class="view-head"><a class="btn-ghost" href="#home">← Hub</a><h2>${esc((v && v.title) || TITLES[name])}</h2><span class="spacer"></span></div>`);
    const body = el(`<div class="view-body"></div>`);
    view.append(head, body);
    window.scrollTo({ top: 0, behavior: reduce ? "auto" : "smooth" });
    if (v && typeof v.mount === "function") {
      Promise.resolve()
        .then(() => v.mount(body))
        .catch((e) => { clear(body); body.append(empty("Couldn’t load this tool: " + e.message, "⚠")); });
    } else {
      body.append(empty("This tool isn’t available yet."));
    }
  }
  addEventListener("hashchange", render);

  /* ---------------- theme (defaults to light; a manual toggle persists) ---------------- */
  const saved = localStorage.getItem("sah-theme");
  if (saved) root.setAttribute("data-theme", saved);
  // No saved choice → keep the document default (light, set on <html> in index.html).
  document.getElementById("themeToggle").addEventListener("click", () => {
    const next = root.getAttribute("data-theme") === "dark" ? "light" : "dark";
    root.setAttribute("data-theme", next);
    localStorage.setItem("sah-theme", next);
  });

  /* ---------------- ambient: cursor glow + orb parallax ---------------- */
  const glow = document.getElementById("glow");
  const orbs = [...document.querySelectorAll(".orb")];
  if (!reduce) {
    let rx = 0, ry = 0, tx = 0, ty = 0;
    addEventListener("pointermove", (e) => {
      glow.style.transform = `translate3d(${e.clientX}px, ${e.clientY}px, 0)`;
      tx = e.clientX / innerWidth - 0.5; ty = e.clientY / innerHeight - 0.5;
    }, { passive: true });
    (function loop() {
      rx += (tx - rx) * 0.05; ry += (ty - ry) * 0.05;
      orbs.forEach((o, i) => { o.style.translate = `${rx * (i + 1) * 10}px ${ry * (i + 1) * 10}px`; });
      requestAnimationFrame(loop);
    })();
  }

  /* ---------------- home card 3D tilt + specular ---------------- */
  document.querySelectorAll("#home .card").forEach((card) => {
    card.addEventListener("pointermove", (e) => {
      const r = card.getBoundingClientRect();
      const px = (e.clientX - r.left) / r.width, py = (e.clientY - r.top) / r.height;
      card.style.setProperty("--mx", px * 100 + "%");
      card.style.setProperty("--my", py * 100 + "%");
      if (!reduce) card.style.transform = `rotateX(${(0.5 - py) * 12}deg) rotateY(${(px - 0.5) * 12}deg) translateZ(10px) scale(1.012)`;
    });
    card.addEventListener("pointerleave", () => {
      card.style.transform = "";
      card.style.setProperty("--mx", "30%"); card.style.setProperty("--my", "0%");
    });
  });

  /* ---------------- boot: health + subject count + key banner ---------------- */
  api.get("/health").then((h) => {
    if (!h.groq_key) {
      document.getElementById("keywarn").appendChild(el(
        `<div class="keywarn glass">⚠ <span>No Groq key detected. Add <code>GROQ_API_KEY</code> to a <code>.env</code> file, then restart the server to enable the AI tools.</span></div>`
      ));
    }
  }).catch(() => {});
  api.get("/subjects").then((d) => {
    const n = (d.subjects || []).length;
    const slot = document.getElementById("stat-subjects");
    if (slot) slot.textContent = n || "·";
  }).catch(() => {});

  if (document.readyState === "loading") addEventListener("DOMContentLoaded", render);
  else render();
})();
