/* ============================================================
   Chat Assistant (Apex Agent) — ChatGPT-style experience.

   • Generation control: Stop (aborts the SSE fetch; the server cancels the provider stream and keeps the partial
     answer), Regenerate (older answers stay as versions), Edit & resend, Continue generating.
   • Message UX: copy, thumbs up/down, timestamps, model label + token-usage tooltip, jump-to-latest, auto-growing
     composer, keyboard shortcuts, retry-on-error, offline state.
   • Conversations: grouped/searchable drawer with pin, inline rename, in-page delete confirm, export (MD / PDF).
   • Smart help: starter chips, follow-up chips (no extra LLM call: the backend parses them from the answer),
     citation chips that expand to the retrieved passage.
   Rendering is XSS-safe (ui.md escapes first). While streaming, finished blocks are rendered once and only the
   unfinished tail block is re-rendered per animation frame.
   ============================================================ */
(() => {
  "use strict";
  const { el, esc, clear, toast, md, spinner, copy } = window.ui;

  // ── State ──────────────────────────────────────────────────
  let session = null;       // { id, label, chatId, mode } — id is a long URL-safe token: never truncate/parse it
  let msgs = [];           // message records (see newRec); each keeps its DOM node in rec.node
  let lastStart = null;     // {path, body} of the last subject/topic start, to auto-recreate an expired session
  let source = "subject";   // "subject" | "topic" | "upload"
  let pdfFile = null;
  let generating = false;   // a response is currently streaming
  let currentAbort = null;  // AbortController for the in-flight stream (the Stop button)
  let teardownLayout = null, teardownThreadScroll = null, toolsMenuCleanup = null;
  let scrollRaf = 0;
  let drawerSeq = 0;        // discards stale drawer responses
  let hiddenBefore = 0;     // records [0, hiddenBefore) are not rendered yet ("Load earlier")

  const RENDER_LAST = 40;   // long chats render only the last N messages at first
  const HISTORY_SEND = 40;  // prior turns sent with each message when persistence is off
  const reduceMotion = () => !!(window.matchMedia && matchMedia("(prefers-reduced-motion: reduce)").matches);
  const $ = (id) => document.getElementById(id);
  const thread = () => $("chatThread");
  const uid = () => {
    try { return crypto.randomUUID().replace(/-/g, "").slice(0, 12); }
    catch (_) { return "m" + Math.random().toString(36).slice(2, 10) + Date.now().toString(36).slice(-4); }
  };
  const nowIso = () => new Date().toISOString();

  function fmtTime(iso) {
    if (!iso) return "";
    const d = new Date(iso);
    if (isNaN(d)) return "";
    const t = d.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
    const now = new Date();
    if (d.toDateString() === now.toDateString()) return t;
    if (d.toDateString() === new Date(now - 864e5).toDateString()) return "Yesterday " + t;
    return d.toLocaleDateString([], { month: "short", day: "numeric" }) + " " + t;
  }

  // Polite status for screen readers ("Searching your notes…", "Answer ready."). Never the token stream itself.
  function announce(text) {
    const s = $("chatStatus");
    if (s && s.textContent !== text) s.textContent = text || "";
  }

  // ── Sticky-bottom autoscroll + "Jump to latest" ───────────
  //    Layout reads are kept off the streaming path: the sticky flag is maintained by the scroll listener
  //    (and set directly on programmatic scrolls), so a frame costs one write and no forced reflow.
  const NEAR_BOTTOM_PX = 100;
  let unseen = false;       // new content arrived while the user was scrolled up
  function isNearBottom(t) { return !t || (t.scrollHeight - t.scrollTop - t.clientHeight) <= NEAR_BOTTOM_PX; }
  function updateJump() {
    const t = thread(), j = $("chatJump");
    if (!t || !j) return;
    const away = t.dataset.stickBottom === "0";
    if (j.hidden === !away && j.classList.contains("has-new") === (away && unseen)) return;   // no DOM churn
    j.hidden = !away;
    j.classList.toggle("has-new", away && unseen);
    if (!away) unseen = false;
  }
  function scrollThreadToBottom(t, { force = false } = {}) {
    if (!t) return;
    if (!force && t.dataset.stickBottom === "0") { if (generating) unseen = true; updateJump(); return; }
    t.scrollTop = t.scrollHeight;
    t.dataset.stickBottom = "1";
    updateJump();
  }
  function scheduleScrollToBottom(t, opts) {
    if (!t) return;
    if (!(opts && opts.force) && t.dataset.stickBottom === "0") { if (generating) { unseen = true; updateJump(); } return; }
    if (scrollRaf) return;
    scrollRaf = requestAnimationFrame(() => { scrollRaf = 0; scrollThreadToBottom(t, opts); });
  }
  function installThreadScroll(t) {
    if (teardownThreadScroll) teardownThreadScroll();
    if (!t) return;
    t.dataset.stickBottom = "1";
    const onScroll = () => {
      // Reads here happen inside a scroll event (layout is already clean) — no forced reflow.
      t.dataset.stickBottom = isNearBottom(t) ? "1" : "0";
      updateJump();
    };
    t.addEventListener("scroll", onScroll, { passive: true });
    teardownThreadScroll = () => { t.removeEventListener("scroll", onScroll); teardownThreadScroll = null; };
  }

  // ── Tools (slash menu) ─────────────────────────────────────
  const TOOLS = [
    { key: "quiz", label: "Quiz", icon: "help" }, { key: "flashcards", label: "Flashcards", icon: "layers" },
    { key: "summary", label: "Summarize", icon: "sparkle" }, { key: "plan", label: "Study Plan", icon: "calendar" },
    { key: "mindmap", label: "Mind Map", icon: "network" }, { key: "cheatsheet", label: "Cheat Sheet", icon: "list" },
    { key: "predictor", label: "Predict Questions", icon: "target" }, { key: "exam", label: "Mock Exam", icon: "graduation-cap" },
    { key: "teachback", label: "Teach-back", icon: "mic" }, { key: "progress", label: "My Progress", icon: "chart" },
  ];
  const TOOL_KEYS = new Set(TOOLS.map((t) => t.key));
  function parseForcedTool(text) {
    const m = (text || "").match(/^\/([a-z0-9]+)(?=\s|$)/i);
    if (m && TOOL_KEYS.has(m[1].toLowerCase())) return { tool: m[1].toLowerCase(), rest: text.slice(m[0].length).trim() };
    return { tool: null, rest: text };
  }

  // ── Composer ───────────────────────────────────────────────
  const COMPOSER_READY_PH = "Message Apex Agent…  (Enter to send, Shift+Enter for a new line)";
  const COMPOSER_IDLE_PH = "Start a session to chat…";
  const COMPOSER_IDLE_TITLE =
    "Start a session first (Options → Start session, or open Chats), then you can type or pick a tool.";
  const inputEl = () => document.querySelector("#chatComposer .input");

  // Auto-grow: one forced layout per animation frame at most. Keystrokes only mark the box dirty; the
  // read (scrollHeight) + write (height) pair runs once in rAF, so a burst of input events (paste,
  // held key, IME) costs a single reflow. Pass `now` when the caller needs the height immediately.
  const sizing = new WeakSet();
  function autosizeNow(input) {
    const max = Math.round(Math.min(window.innerHeight * 0.4, 200));
    input.style.height = "auto";
    const sh = input.scrollHeight + 2;
    input.style.height = Math.min(sh, max) + "px";
    input.style.overflowY = sh > max ? "auto" : "hidden";
  }
  function autosize(input, now) {
    if (!input) return;
    if (now) { autosizeNow(input); return; }
    if (sizing.has(input)) return;
    sizing.add(input);
    requestAnimationFrame(() => { sizing.delete(input); if (input.isConnected) autosizeNow(input); });
  }
  function setStopMode(on) {
    const btn = $("chatSend");
    if (!btn) return;
    btn.replaceChildren(window.ui.icon(on ? "stop" : "send"));
    const lab = document.createElement("span"); lab.className = "icon-btn-text"; lab.textContent = on ? "Stop" : "Send"; btn.append(lab);
    btn.dataset.tip = on ? "Stop generating" : "Send message"; btn.removeAttribute("title");
    if (on) btn.removeAttribute("data-shortcut"); else btn.dataset.shortcut = "Enter";
    btn.classList.toggle("is-stop", on);
    btn.setAttribute("aria-label", on ? "Stop generating" : "Send message");
    btn.disabled = false;
  }
  function stopGenerating() {
    if (currentAbort) { try { currentAbort.abort(); } catch { /* already aborted */ } }
  }
  function setComposerEnabled(on) {
    const input = inputEl(), sendBtn = $("chatSend");
    if (!input || !sendBtn) return;
    input.disabled = !on;
    sendBtn.disabled = !on;
    if (on) { input.placeholder = COMPOSER_READY_PH; input.removeAttribute("title"); }
    else { input.placeholder = COMPOSER_IDLE_PH; input.title = COMPOSER_IDLE_TITLE; }
    syncOnline();
  }
  // The tool rows are built on first open (not at mount): keeps the view's first paint lean.
  function buildToolsMenu(menu) {
    if (menu.childElementCount) return;
    const frag = document.createDocumentFragment();
    TOOLS.forEach((tl) => {
      const item = el(`<button class="chat-tool-item" type="button" role="menuitem" data-tool="${tl.key}"></button>`);
      item.innerHTML = `<span class="chat-tool-ic" aria-hidden="true">${window.ui.iconHtml(tl.icon)}</span><span class="chat-tool-label">${esc(tl.label)}</span><span class="chat-tool-slash">/${tl.key}</span>`;
      item.addEventListener("click", () => pickTool(tl.key));
      frag.append(item);
    });
    menu.append(frag);
  }
  function openToolsMenu() {
    if (!session) {
      toast("Start a session first, then pick a tool.", "error");
      setComposerEnabled(false);
      return;
    }
    const menu = $("chatToolsMenu");
    if (menu) { buildToolsMenu(menu); menu.hidden = false; }
    $("chatToolsBtn")?.setAttribute("aria-expanded", "true");
  }
  function closeToolsMenu() {
    const menu = $("chatToolsMenu");
    if (menu) menu.hidden = true;
    $("chatToolsBtn")?.setAttribute("aria-expanded", "false");
  }
  function pickTool(key) {
    closeToolsMenu();
    const input = inputEl();
    if (!input) return;
    input.value = `/${key} ` + input.value.replace(/^\/[a-z0-9]+\s*/i, "");
    input.disabled = false;
    input.removeAttribute("title");
    input.placeholder = COMPOSER_READY_PH;
    autosize(input);
    input.focus();
  }

  // Offline: sending is paused and explained (the chat stays readable).
  function syncOnline() {
    const off = navigator.onLine === false;
    const net = $("chatNet"), send = $("chatSend");
    if (net) net.hidden = !off;
    if (send && session && !generating) send.disabled = off;
  }

  // ── Responsive height ──────────────────────────────────────
  //    Measures the shell's DOCUMENT offset (scroll-invariant) and only reacts to resize, never to scroll.
  function installShellSizing(shell) {
    if (teardownLayout) teardownLayout();
    const vp = window.visualViewport;
    const fit = () => {
      if (!shell.isConnected) return;
      const rect = shell.getBoundingClientRect();
      const offsetTop = rect.top + (window.scrollY || window.pageYOffset || 0);
      const viewportH = vp ? vp.height : window.innerHeight;
      const host = shell.closest(".shell");
      const padB = host ? (parseFloat(getComputedStyle(host).paddingBottom) || 0) : 0;
      shell.style.height = Math.max(340, viewportH - offsetTop - Math.max(12, padB)) + "px";
    };
    requestAnimationFrame(fit);
    window.addEventListener("resize", fit);
    window.addEventListener("orientationchange", fit);
    if (vp) vp.addEventListener("resize", fit);
    let ro = null;
    if (typeof ResizeObserver === "function") {
      let raf = 0;
      ro = new ResizeObserver(() => { cancelAnimationFrame(raf); raf = requestAnimationFrame(fit); });
      ro.observe(document.body);
    }
    teardownLayout = () => {
      if (ro) ro.disconnect();
      window.removeEventListener("resize", fit);
      window.removeEventListener("orientationchange", fit);
      if (vp) vp.removeEventListener("resize", fit);
      teardownLayout = null;
    };
  }

  // ── Source control (setup card) ────────────────────────────
  function buildSourceRow(host) {
    clear(host);
    host.dataset.src = source;
    if (source === "subject") {
      host.append(el(`<select id="chatSubject" class="select" aria-label="Subject" style="max-width:220px"><option value="">Loading…</option></select>`));
      const sel = host.querySelector("#chatSubject");
      window.api.get("/subjects").then((d) => {
        const subjects = d.subjects || [];
        clear(sel);
        if (!subjects.length) { sel.append(el(`<option value="">No subjects</option>`)); return; }
        subjects.forEach((s) => sel.append(el(`<option value="${esc(s)}">${esc(s)}</option>`)));
        sel.value = subjects[0];
      }).catch(() => { clear(sel); sel.append(el(`<option value="">Error loading</option>`)); });
    } else if (source === "topic") {
      host.append(el(`<input id="chatTopic" class="input" type="text" placeholder="e.g. Binary search trees" aria-label="Topic" autocomplete="off" style="max-width:260px" />`));
    } else {
      const wrap = el(`<div class="row"></div>`);
      const lbl = el(`<label class="btn" for="chatPdfUp" style="cursor:pointer;padding:.55rem .9rem">Choose PDF…</label>`);
      const name = el(`<span id="chatPdfName" class="muted" style="font-size:var(--fs-sm)">No file chosen</span>`);
      const inp = el(`<input id="chatPdfUp" type="file" accept="application/pdf" hidden />`);
      inp.addEventListener("change", () => {
        pdfFile = inp.files[0] || null;
        if (pdfFile && pdfFile.size > 4 * 1000 * 1000) {
          toast(`${pdfFile.name} is ${(pdfFile.size / 1e6).toFixed(1)} MB — PDFs for chat can be up to 4 MB. Compress it or add it as a subject instead.`, "error");
          pdfFile = null; inp.value = "";
        }
        name.textContent = pdfFile ? pdfFile.name : "No file chosen";
      });
      wrap.append(lbl, name, inp);
      host.append(wrap);
    }
  }

  // ── Attachment chips (above the composer) ──────────────────
  function addChip({ icon, text, busy = false, title = "", onRemove = null, id = "" }) {
    const bar = $("chatAttach");
    if (!bar) return null;
    if (id) bar.querySelector(`[data-chip="${id}"]`)?.remove();
    const chip = el(`<span class="chat-chip"></span>`);
    if (id) chip.dataset.chip = id;
    if (title) chip.title = title;
    const ic = document.createElement("span");
    ic.setAttribute("aria-hidden", "true");
    if (busy) ic.className = "spinner chat-chip-spin"; else ic.append(window.ui.icon(icon === "📷" ? "camera" : "book"));
    chip.append(ic);
    const t = document.createElement("span");
    t.className = "chat-chip-text"; t.textContent = text;
    chip.append(t);
    if (onRemove) {
      const x = el(`<button type="button" class="chat-chip-x icon-btn is-sm" aria-label="Remove attachment" title="Remove attachment">${window.ui.iconHtml("close")}</button>`);
      x.addEventListener("click", () => { onRemove(); chip.remove(); });
      chip.append(x);
    }
    bar.append(chip);
    return { el: chip, remove: () => chip.remove() };
  }
  function sessionChip() {
    $("chatAttach")?.querySelector('[data-chip="session"]')?.remove();
    if (session && session.mode === "pdf") {
      addChip({ icon: "📄", text: (session.label || "").replace(/^PDF · /, "") || "PDF", id: "session", title: "This chat answers from this PDF" });
    }
  }

  // ── Message records ────────────────────────────────────────
  function newRec(role, content, extra) {
    return Object.assign({ id: uid(), role, content: content || "", ts: nowIso(), llm: null, stopped: false, finish: null,
      versions: [], vIdx: null, feedback: null, citations: [], steps: [], sources: [], usage: null, followups: [],
      node: null, extras: [], fromHistory: false }, extra || {});
  }
  const lastRec = () => msgs[msgs.length - 1] || null;
  const lastAssistant = () => { const r = lastRec(); return r && r.role === "assistant" ? r : null; };
  // all versions of an assistant message, oldest first; the live one is last
  const allVersions = (r) => [...(r.versions || []), { content: r.content, llm: r.llm, ts: r.ts }];
  const shownIdx = (r) => (r.vIdx == null ? allVersions(r).length - 1 : r.vIdx);

  // Without server-side persistence the backend cannot rebuild the conversation: send the prior turns.
  function historyFor(before) {
    if (session && session.chatId) return undefined;
    return before.filter((m) => (m.role === "user" || m.role === "assistant") && m.content)
      .map((m) => ({ role: m.role, content: m.content })).slice(-HISTORY_SEND);
  }

  // Remove records [idx, end) from state + DOM; returns them so a failed turn can put them back.
  function detachFrom(idx) {
    const removed = msgs.splice(idx);
    removed.forEach((r) => { r.node?.remove(); r.extras.forEach((n) => n.remove()); });
    thread()?.querySelectorAll(".chat-followups, .chat-related").forEach((n) => n.remove());
    return removed;
  }
  function restore(removed) {
    const t = thread();
    if (!t || !removed) return;
    t.querySelectorAll(".chat-live-tail").forEach((n) => n.remove());
    removed.forEach((r) => { msgs.push(r); if (r.node) { t.append(r.node); r.extras.forEach((n) => t.append(n)); } });
  }

  // ── Rendering: user + assistant messages ───────────────────
  // Icon-only message/drawer action. `glyph` is a sprite icon name (see ui.icon in app.js).
  function actionBtn(label, glyph, onClick, extra = "", text = "") {
    const b = el(`<button type="button" class="msg-act icon-btn ${extra}"></button>`);
    b.append(window.ui.icon(glyph));
    if (text) { const t = document.createElement("span"); t.className = "icon-btn-text"; t.textContent = text; b.append(t); b.classList.add("has-text"); }
    b.setAttribute("aria-label", label);
    b.dataset.tip = label;
    b.addEventListener("click", onClick);
    return b;
  }

  function renderUser(rec) {
    const bubble = el(`<div class="msg me"><div class="msg-text"></div></div>`);
    bubble.querySelector(".msg-text").textContent = rec.content;
    const node = el(`<div class="msg-row me" data-id="${esc(rec.id)}"></div>`);
    const bar = el(`<div class="msg-actions" role="group" aria-label="Message actions"></div>`);
    bar.append(timeEl(rec.ts));
    bar.append(actionBtn("Copy message", "copy", (e) => copy(rec.content, e.currentTarget)));
    bar.append(actionBtn("Edit and resend this message", "edit", () => startEdit(rec)));
    if (lastRec() === rec) node.classList.add("is-last");
    node.append(bubble, bar);
    rec.node = node;
    return node;
  }

  function timeEl(ts) {
    const time = el(`<time class="msg-time"></time>`);
    time.textContent = fmtTime(ts);
    if (ts) {
      const full = new Date(ts).toLocaleString();
      time.dateTime = ts; time.dataset.full = full; time.tabIndex = 0; time.setAttribute("aria-label", full);   // CSS tooltip (above the row) instead of a native title that covers the icons
    }
    return time;
  }

  function usageText(u) {
    if (!u || !u.total_tokens) return "";
    return `${u.total_tokens.toLocaleString()} tokens (prompt ${(u.prompt_tokens || 0).toLocaleString()} · answer ${(u.completion_tokens || 0).toLocaleString()})`;
  }

  function citationChips(rec) {
    const cites = rec.citations || [];
    if (!cites.length && !(rec.sources || []).length) return null;
    const wrap = el(`<div class="msg-sources"></div>`);
    if (!cites.length) {
      rec.sources.forEach((s) => { const c = el(`<span class="msg-source-chip"></span>`); c.textContent = s; wrap.append(c); });
      return wrap;
    }
    const detail = el(`<div class="cite-detail" hidden></div>`);
    let openChip = null;
    cites.forEach((c, i) => {
      const chip = el(`<button type="button" class="msg-source-chip cite-chip" aria-expanded="false"></button>`);
      chip.textContent = c.source + (c.page ? ` · p.${c.page}` : "");
      chip.title = "Show the passage this answer used";
      chip.addEventListener("click", () => {
        if (openChip === chip) { detail.hidden = true; chip.setAttribute("aria-expanded", "false"); openChip = null; return; }
        if (openChip) openChip.setAttribute("aria-expanded", "false");
        openChip = chip; chip.setAttribute("aria-expanded", "true");
        detail.hidden = false;
        detail.textContent = "";
        const h = el(`<div class="cite-head"></div>`);
        h.textContent = `[${i + 1}] ${c.source}` + (c.page ? ` — page ${c.page}` : "");
        const q = el(`<div class="cite-text"></div>`);
        q.textContent = c.text || "";
        detail.append(h, q);
      });
      wrap.append(chip);
    });
    const box = el(`<div class="msg-cites"></div>`);
    box.append(wrap, detail);
    return box;
  }

  function renderAssistant(rec) {
    const vers = allVersions(rec), idx = shownIdx(rec), v = vers[idx];
    const isLatestV = idx === vers.length - 1;
    const bubble = el(`<div class="msg ai"></div>`);
    const node = el(`<div class="msg-row ai" data-id="${esc(rec.id)}"></div>`);
    bubble.append(md(v.content || ""));
    if (isLatestV && rec.stopped) bubble.append(el(`<div class="msg-stopped-tag">⏹ Stopped</div>`));
    else if (isLatestV && rec.finish === "length") bubble.append(el(`<div class="msg-stopped-tag">Cut off — hit the length limit</div>`));
    else if (isLatestV && rec.finish === "interrupted") bubble.append(el(`<div class="msg-stopped-tag">Connection interrupted mid-answer</div>`));

    const footer = el(`<div class="msg-footer"></div>`);
    const cites = isLatestV ? citationChips(rec) : null;
    if (cites) footer.append(cites);

    const bar = el(`<div class="msg-actions" role="group" aria-label="Answer actions"></div>`);
    bar.append(timeEl(v.ts));
    if (vers.length > 1) {
      const nav = el(`<span class="msg-versions" role="group" aria-label="Answer versions"></span>`);
      const prev = actionBtn("Previous version", "chevron-left", () => { rec.vIdx = Math.max(0, idx - 1); rerender(rec); });
      const next = actionBtn("Next version", "chevron-right", () => { rec.vIdx = Math.min(vers.length - 1, idx + 1); rerender(rec); });
      prev.disabled = idx === 0; next.disabled = isLatestV;
      const lbl = el(`<span class="msg-vlabel" aria-live="polite"></span>`);
      lbl.textContent = `${idx + 1}/${vers.length}`;
      nav.append(prev, lbl, next);
      bar.append(nav);
    }
    bar.append(actionBtn("Copy answer", "copy", (e) => copy(v.content || "", e.currentTarget)));
    const up = actionBtn("Rate: good answer", "thumb-up", () => setFeedback(rec, "up"), "msg-fb");
    const down = actionBtn("Rate: bad answer", "thumb-down", () => setFeedback(rec, "down"), "msg-fb");
    up.setAttribute("aria-pressed", String(rec.feedback === "up"));
    down.setAttribute("aria-pressed", String(rec.feedback === "down"));
    bar.append(up, down);
    if (speechSupported()) bar.append(speakBtn(rec, v));
    if (lastRec() === rec && isLatestV && !generating) {
      if (rec.stopped || rec.finish === "length" || rec.finish === "interrupted") {
        bar.append(actionBtn("Continue generating", "play", () => continueLast(), "msg-act-strong"));
      }
      bar.append(actionBtn("Regenerate this answer", "refresh", () => regenerateLast()));
      node.classList.add("is-last");
    }

    if (isLatestV && rec.steps && rec.steps.length) footer.append(buildTrace(rec.steps));
    const label = window.ui.renderModelLabel(v.llm, { stored: !!rec.fromHistory });
    if (label) {
      const ut = isLatestV ? usageText(rec.usage) : "";
      if (ut) label.title = (label.title ? label.title + "\n" : "") + ut;
      footer.append(label);
    }
    bubble.append(footer);
    node.append(bubble, bar);
    if (speaking && speaking.id === rec.id) node.classList.add("is-speaking");
    rec.node = node;
    return node;
  }

  // ── Read aloud (speech code lives in voice.js: window.Voice) ──
  let speaking = null;          // { id } of the message currently being read
  const speechSupported = () => "speechSynthesis" in window;
  function stopSpeak() {
    speaking = null;
    try { window.Voice && window.Voice.stop(); } catch { /* ignore */ }
    paintSpeakBtns();
  }
  function paintSpeakBtns() {
    document.querySelectorAll("#chatThread .msg-row").forEach((row) => {
      const on = !!speaking && row.dataset.id === speaking.id;
      row.classList.toggle("is-speaking", on);
      const b = row.querySelector(".msg-speak");
      if (!b) return;
      b.setAttribute("aria-pressed", String(on));
      window.ui.setIconBtn(b, { icon: on ? "stop" : "speaker", label: on ? "Stop reading aloud" : "Read aloud" });
    });
  }
  function speakBtn(rec, v) {
    const b = actionBtn("Read aloud", "speaker", () => {
      const V = window.Voice;
      if (!V) { toast("Read aloud isn't ready yet — try again in a moment.", "info"); return; }
      if (speaking && speaking.id === rec.id) { stopSpeak(); return; }
      V.stop();
      const mine = { id: rec.id };
      speaking = mine;
      V.speak(v.content || "", () => { if (speaking === mine) { speaking = null; paintSpeakBtns(); } });
      paintSpeakBtns();
    }, "msg-speak");
    b.setAttribute("aria-pressed", String(!!speaking && speaking.id === rec.id));
    if (speaking && speaking.id === rec.id) window.ui.setIconBtn(b, { icon: "stop", label: "Stop reading aloud" });
    return b;
  }

  function rerender(rec) {
    const old = rec.node;
    const fresh = rec.role === "user" ? renderUser(rec) : renderAssistant(rec);
    if (old && old.isConnected) old.replaceWith(fresh);
  }
  // The action bar of the last answer changes (Regenerate / Continue appear once the turn is over).
  function refreshLastBar() {
    thread()?.querySelectorAll(".msg-row.is-last").forEach((n) => n.classList.remove("is-last"));
    const r = lastAssistant();
    if (r && r.node && r.node.isConnected) rerender(r);
  }

  function appendRec(rec) {
    const node = rec.role === "user" ? renderUser(rec) : renderAssistant(rec);
    thread().append(node);
    return node;
  }

  async function setFeedback(rec, value) {
    const prev = rec.feedback;
    rec.feedback = prev === value ? null : value;
    rerender(rec);
    if (!(session && session.chatId)) return;             // persistence off: the rating lives in this tab only
    try {
      await window.api.post(`/chat/${encodeURIComponent(session.chatId)}/feedback`, { message_id: rec.id, value: rec.feedback });
    } catch (e) {
      rec.feedback = prev; rerender(rec);
      toast("Couldn't save your feedback." + (e && e.status === 404 ? " (That message isn't saved yet.)" : ""), "error");
    }
  }

  // ── Artifact card (the chat created something in another tool) ──
  function appendArtifactCard(host, art, llm) {
    if (!art || typeof art !== "object") return;
    if (art.data && typeof window.renderArtifactInline === "function") {
      const block = el(`<div class="chat-live"></div>`);
      let ok = false;
      try { ok = window.renderArtifactInline(block, art) === true; } catch { ok = false; }
      if (ok) {
        const foot = el(`<div class="cl-foot"></div>`);
        const label = el(`<span class="cl-foot-label"></span>`);
        label.textContent = "Saved in " + (art.tool || "the tool");
        foot.append(label);
        const open = el(`<button class="cl-foot-open" type="button">Open →</button>`);
        open.addEventListener("click", () => window.ui.openArtifact(art.kind, art.id, art.route || "#home"));
        foot.append(open);
        block.append(foot);
        { const l = window.ui.renderModelLabel(art.llm || llm); if (l) block.append(l); }
        host.append(block);
        return;
      }
    }
    const card = el(`<a class="chat-artifact" href="${esc(art.route || "#home")}"></a>`);
    card.append(el(`<span class="chat-artifact-icon" aria-hidden="true">✦</span>`));
    const meta = el(`<span class="chat-artifact-meta"></span>`);
    const t = el(`<span class="chat-artifact-title"></span>`);
    t.textContent = art.title || "New artifact";
    const s = el(`<span class="chat-artifact-sub"></span>`);
    s.textContent = "Open in " + (art.tool || "the tool") + " →";
    meta.append(t, s);
    { const l = window.ui.renderModelLabel(art.llm || llm); if (l) meta.append(l); }
    card.append(meta);
    if (art.id) {
      card.addEventListener("click", (e) => { e.preventDefault(); window.ui.openArtifact(art.kind, art.id, art.route || "#home"); });
    }
    host.append(card);
  }

  // ── Agentic trace renderer ────────────────────────────────
  function stepIcon(step) {
    if (step.kind === "rewrite") return "✎";
    if (step.kind === "retrieve") return "⌕";
    if (step.kind === "grade") {
      if (step.verdict === "sufficient" || step.verdict === "final") return "✓";
      if (step.verdict === "limit") return "◐";
      return "⚠";
    }
    return "✦";
  }
  function traceNote(text) { const n = el(`<div class="trace-note"></div>`); n.textContent = text; return n; }
  function passageCard(p) {
    const card = el(`<div class="trace-pass"></div>`);
    const head = el(`<div class="trace-pass-head"></div>`);
    const src = el(`<span class="trace-pass-src"></span>`);
    src.textContent = p.source || "—";
    head.append(src);
    if (p.page) { const pg = el(`<span class="trace-pass-pg"></span>`); pg.textContent = "p." + p.page; head.append(pg); }
    const body = el(`<div class="trace-pass-text"></div>`);
    body.textContent = p.text || "";
    card.append(head, body);
    return card;
  }
  function stepDetail(step) {
    const d = el(`<div class="trace-detail-inner"></div>`);
    if (step.kind === "rewrite") {
      if (step.missing) d.append(traceNote("Addressing gap → " + step.missing));
      if (step.queries && step.queries.length) {
        const ql = el(`<div class="trace-queries"></div>`);
        step.queries.forEach((q) => { const chip = el(`<span class="trace-q"></span>`); chip.textContent = q; ql.append(chip); });
        d.append(ql);
      }
    } else if (step.kind === "retrieve") {
      if (step.passages && step.passages.length) step.passages.forEach((p) => d.append(passageCard(p)));
      else d.append(traceNote("No new passages this round."));
    } else if (step.kind === "grade") {
      const label = step.verdict === "sufficient" ? "✓ Sufficient" : step.verdict === "final" ? "→ Answering"
        : step.verdict === "limit" ? "◐ Limit reached" : "⚠ Insufficient";
      const pill = el(`<span class="trace-verdict trace-verdict--${esc(step.verdict || "")}"></span>`);
      pill.textContent = label;
      d.append(pill);
      if (step.missing) d.append(traceNote("Missing → " + step.missing));
    } else if (step.kind === "generate" && step.passages_used != null) {
      const n = step.passages_used;
      const chars = step.context_chars ? " · " + step.context_chars.toLocaleString() + " chars of context" : "";
      d.append(traceNote(n + " passage" + (n === 1 ? "" : "s") + " grounding the answer" + chars));
    }
    return d;
  }
  function buildTrace(steps) {
    const wrap = el(`<div class="msg-trace-wrap"></div>`);
    const btn = el(`<button class="msg-trace-btn" type="button" aria-expanded="false"></button>`);
    const timeline = el(`<ol class="trace-timeline" hidden></ol>`);
    const n = steps.length;
    const setLabel = (open) => {
      btn.setAttribute("aria-expanded", String(open));
      btn.innerHTML = `<span class="trace-chev">${open ? "▾" : "▸"}</span> ${open ? "Hide" : "Show"} reasoning`
        + ` <span class="trace-count">${n} step${n === 1 ? "" : "s"}</span>`;
    };
    steps.forEach((step) => {
      const kind = step.kind || "generate";
      const node = el(`<li class="trace-node trace-node--${esc(kind)}${step.verdict ? " trace-v--" + esc(step.verdict) : ""}"></li>`);
      const dot = el(`<span class="trace-dot"></span>`);
      dot.textContent = stepIcon(step);
      const body = el(`<div class="trace-node-body"></div>`);
      const head = el(`<button class="trace-node-head" type="button"></button>`);
      const title = el(`<span class="trace-node-title"></span>`);
      title.textContent = step.title || kind;
      head.append(title, el(`<span class="trace-node-chev">›</span>`));
      const detail = el(`<div class="trace-node-detail" hidden></div>`);
      detail.append(stepDetail(step));
      head.addEventListener("click", () => { detail.hidden = !detail.hidden; node.classList.toggle("is-open", !detail.hidden); });
      body.append(head, detail);
      node.append(dot, body);
      timeline.append(node);
    });
    setLabel(false);
    btn.addEventListener("click", () => { timeline.hidden = !timeline.hidden; setLabel(!timeline.hidden); });
    wrap.append(btn, timeline);
    return wrap;
  }

  // ── Empty state: welcome card with starter chips ───────────
  const BUILD_PROMPTS = ["Make a quiz", "Create a cheat sheet", "Generate flashcards", "Build a mind map", "Make a 2-week study plan"];
  function genericStarters() {
    const topic = session && session.mode === "topic" ? (session.label || "").replace(/^Topic · /, "") : "";
    return topic
      ? [`Give me an overview of ${topic}`, `What are the key ideas in ${topic}?`, `Quiz me on ${topic}`]
      : ["Summarize the key ideas", "What are the most important concepts here?", "Explain the hardest idea simply"];
  }
  function chipRow(items, cls) {
    const row = el(`<div class="${cls}"></div>`);
    items.forEach((q) => {
      const b = el(`<button class="chat-overview-q" type="button"></button>`);
      b.textContent = q;
      b.addEventListener("click", () => askQuestion(q));
      row.append(b);
    });
    return row;
  }
  // "Related concepts" (concept-graph suggestions from the `done` event): compact chips, max 6, each sends
  // "Explain <concept> in the context of <subject>" through the normal composer path. Live only: the
  // backend does not persist `related`, so a reopened chat simply has none. Returns null when empty.
  function relatedRow(list) {
    const items = (Array.isArray(list) ? list : []).map((x) => String((x && (x.name || x.title || x.label)) || x || "").trim())
      .filter(Boolean).filter((v, i, a) => a.indexOf(v) === i).slice(0, 6);
    if (!items.length) return null;
    const subject = ((session && session.label) || "").replace(/^(PDF|Topic|Subject) · /, "").trim();
    const row = el(`<div class="chat-related" role="group" aria-label="Related concepts"><span class="chat-related-label">Related concepts</span></div>`);
    items.forEach((c) => {
      const b = el(`<button class="chat-related-chip" type="button"></button>`);
      b.innerHTML = esc(c);
      b.addEventListener("click", () => askQuestion(`Explain ${c} in the context of ${subject || "this material"}`));
      row.append(b);
    });
    return row;
  }
  function showWelcome(t, label, hint) {
    clear(t);
    const card = el(`<div class="chat-overview chat-welcome"></div>`);
    const title = el(`<div class="chat-overview-title"></div>`);
    title.textContent = label || "Ask me anything";
    const slot = el(`<div class="chat-overview-slot"></div>`);
    if (hint) { const h = el(`<div class="chat-overview-body muted"></div>`); h.textContent = hint; slot.append(h); }
    const qs = chipRow(genericStarters(), "chat-overview-qs");
    const build = el(`<div class="chat-overview-build"></div>`);
    build.append(el(`<div class="chat-overview-build-label muted">Or ask me to build something:</div>`));
    const chips = el(`<div class="agent-prompt-chips"></div>`);
    BUILD_PROMPTS.forEach((p) => {
      const chip = el(`<button class="agent-prompt-chip" type="button"></button>`);
      chip.textContent = p;
      chip.addEventListener("click", () => askQuestion(p));
      chips.append(chip);
    });
    build.append(chips);
    card.append(title, slot, qs, build);
    t.append(card);
    return card;
  }
  // Best-effort overview (pre-baked overviews are instant): upgrades the welcome card with a summary and
  // subject-aware starter questions. Never blocks the chat.
  async function renderOverview(card, sessionId) {
    let data;
    try { data = await window.api.get(`/chat/overview/${encodeURIComponent(sessionId)}`); } catch { return; }
    if (!session || session.id !== sessionId || !card.isConnected || msgs.length) return;
    const overview = (data && typeof data.overview === "string") ? data.overview : "";
    const questions = (data && Array.isArray(data.questions)) ? data.questions.filter(Boolean) : [];
    if (!overview && !questions.length) return;
    const slot = card.querySelector(".chat-overview-slot");
    if (data.label) card.querySelector(".chat-overview-title").textContent = data.label;
    const sources = (data && Number(data.sources)) || 0;
    clear(slot);
    if (sources > 0) { const s = el(`<div class="chat-overview-sources"></div>`); s.textContent = sources + " source" + (sources === 1 ? "" : "s"); slot.append(s); }
    if (overview) { const b = el(`<div class="chat-overview-body"></div>`); b.append(md(overview)); slot.append(b); }
    if (questions.length) card.querySelector(".chat-overview-qs").replaceWith(chipRow(questions.slice(0, 3), "chat-overview-qs"));
    { const l = window.ui.renderModelLabel(data && data.llm); if (l) slot.append(l); }
  }

  function askQuestion(text) {
    const input = inputEl();
    if (!input || input.disabled || generating) return;
    input.value = text;
    autosize(input);
    send();
  }

  // ── Live reasoning rails ──────────────────────────────────
  function liveStepCopy(step) {
    const kind = step.kind || "generate";
    if (kind === "tool") return { rail: "tool", ico: "🛠", label: step.title || ("Building " + (step.tool || "tool")) };
    if (kind === "rewrite") {
      const n = (step.queries && step.queries.length) || 0;
      return { rail: "ground", ico: "⌕", label: n ? `Searching your notes · ${n} quer${n === 1 ? "y" : "ies"}` : "Searching your notes…" };
    }
    if (kind === "retrieve") {
      const n = Array.isArray(step.passages) ? step.passages.length : (step.count || 0);
      return { rail: "ground", ico: "⌕", label: n ? `Retrieved ${n} passage${n === 1 ? "" : "s"}` : "Retrieving passages…" };
    }
    if (kind === "grade") {
      const v = step.verdict;
      const label = v === "sufficient" ? "Verified — context is sufficient" : v === "final" ? "→ Answering"
        : v === "limit" ? "Verifying — reached search limit" : v === "insufficient" ? "Verifying — looking for more" : "Verifying context";
      return { rail: "ground", ico: "✓", label };
    }
    return { rail: "ground", ico: "✦", label: step.title || "Answering from your notes…" };
  }
  function appendLiveStep(panel, step) {
    panel.querySelectorAll(".reason-step.is-active").forEach((r) => r.classList.remove("is-active"));
    const c = liveStepCopy(step);
    const row = el(`<div class="reason-step reason-step--${c.rail === "tool" ? "tool" : "ground"} is-active"></div>`);
    const ico = el(`<span class="reason-ico" aria-hidden="true"></span>`);
    ico.textContent = c.ico;
    const label = el(`<span class="reason-label"></span>`);
    label.textContent = c.label;
    row.append(ico, label);
    panel.append(row);
    announce(c.label);
    return row;
  }
  function appendThinking(t) {
    const wrap = el(`<div class="msg ai msg-thinking chat-live-tail"></div>`);
    const panel = el(`<div class="reason-live"></div>`);
    const dots = el(`<div class="thinking-dots"><span class="typing"><i></i><i></i><i></i></span><span class="thinking-label">Thinking…</span></div>`);
    panel.append(dots);
    wrap.append(panel);
    t.append(wrap);
    let cleared = false;
    return {
      el: wrap, panel,
      step(s) { if (!cleared) { dots.remove(); cleared = true; } appendLiveStep(panel, s); scheduleScrollToBottom(t); },
      stop() { dots.remove(); panel.querySelectorAll(".reason-step.is-active").forEach((r) => r.classList.remove("is-active")); },
    };
  }

  // Streaming markdown: the boundary (last blank line outside code fences / math) splits the answer into a
  // stable prefix (rendered once, never touched again) and a tail (re-rendered per frame). Math/highlight are
  // deferred for the tail (LIVE) so nothing flickers while the block is still open.
  function stableBoundary(text) {
    let fence = false, math = false, pos = 0, boundary = 0;
    const lines = text.split("\n");
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      const end = pos + line.length + 1;
      if (/^\s*```/.test(line)) fence = !fence;
      else if (!fence && /^\s*\$\$\s*$/.test(line)) math = !math;
      else if (!fence && !math && line.trim() === "" && end <= text.length) boundary = end;
      pos = end;
    }
    return boundary;
  }
  function liveRenderer(host) {
    const stable = el(`<div class="md md-stream-stable"></div>`);
    const tail = el(`<div class="md md-stream-tail"></div>`);
    host.append(stable, tail);
    let stableLen = 0;
    return {
      update(text) {
        const b = stableBoundary(text.slice(stableLen)) + stableLen;
        if (b > stableLen) {
          const frag = md(text.slice(stableLen, b));
          while (frag.firstChild) stable.append(frag.firstChild);
          stableLen = b;
        }
        const t = md(text.slice(stableLen), { live: true });
        tail.replaceChildren(...t.childNodes);
      },
    };
  }

  // ── Streaming headers (mirror app.js authHeaders) ──────────
  async function streamHeaders() {
    const headers = { "Content-Type": "application/json", "Accept": "text/event-stream" };
    try {
      const token = window.Auth && window.Auth.getToken ? await window.Auth.getToken() : null;
      if (token) headers["Authorization"] = "Bearer " + token;
    } catch (_) {}
    try {
      const gid = localStorage.getItem("sah-guest-id");
      if (gid) headers["X-Guest-Id"] = gid;
    } catch (_) {}
    return headers;
  }

  // ── Resume a saved chat ────────────────────────────────────
  function recFromStored(m) {
    return newRec(m.role, m.content || "", {
      id: m.id || uid(), ts: m.ts || null, llm: m.llm || null, stopped: !!m.stopped, finish: m.finish || null,
      versions: Array.isArray(m.versions) ? m.versions : [], feedback: m.feedback || null,
      citations: Array.isArray(m.citations) ? m.citations : [], usage: m.usage || null, fromHistory: true,
    });
  }
  function showEarlierButton() {
    const t = thread();
    t.querySelector(".chat-earlier")?.remove();
    if (hiddenBefore <= 0) return;
    const b = el(`<button type="button" class="chat-earlier btn"></button>`);
    b.textContent = `Load earlier messages (${hiddenBefore} hidden)`;
    b.addEventListener("click", () => {
      const prevH = t.scrollHeight, prevTop = t.scrollTop;
      const from = Math.max(0, hiddenBefore - RENDER_LAST);
      const frag = document.createDocumentFragment();
      for (let i = from; i < hiddenBefore; i++) { const r = msgs[i]; frag.append(r.role === "user" ? renderUser(r) : renderAssistant(r)); }
      b.after(frag);
      hiddenBefore = from;
      showEarlierButton();
      t.scrollTop = prevTop + (t.scrollHeight - prevH);      // keep the reading position
    });
    t.prepend(b);
  }

  async function resumeChat(saved) {
    const t = thread();
    if (!t) return;
    if (generating) { stopGenerating(); generating = false; currentAbort = null; }
    clear(t);
    t.append(spinner("Re-opening chat…"));
    try {
      const resp = await window.api.post("/chat/resume", { chat_id: saved.id });
      session = { id: resp.session_id, label: resp.label, chatId: resp.chat_id || saved.id || null, mode: resp.mode };
      lastStart = null;
      clear(t);
      msgs = (resp.messages || []).map(recFromStored);
      hiddenBefore = Math.max(0, msgs.length - RENDER_LAST);
      if (!msgs.length) showWelcome(t, resp.label);
      else {
        const frag = document.createDocumentFragment();
        for (let i = hiddenBefore; i < msgs.length; i++) frag.append(msgs[i].role === "user" ? renderUser(msgs[i]) : renderAssistant(msgs[i]));
        t.append(frag);
        showEarlierButton();
      }
      setComposerEnabled(true);
      updateSessionBadge(resp.label);
      setChatSetup(false);
      sessionChip();
      refreshExport();
      scrollThreadToBottom(t, { force: true });
      inputEl()?.focus();
      markDrawerActive();
    } catch (e) {
      clear(t);
      t.append(el(`<div class="empty"><div class="big">✦</div><div>Couldn't re-open that chat.</div></div>`));
      toast(e.message, "error");
    }
  }

  function updateSessionBadge(label) {
    const slot = $("chatSessionBadge");
    if (!slot) return;
    clear(slot);
    if (label) slot.append(el(`<span class="badge badge-accent" title="${esc(label)}">${esc(label)}</span>`));
  }

  // ── Start a session ────────────────────────────────────────
  async function startSession() {
    const t = thread(), startBtn = $("chatStart"), srcRow = $("chatSrcRow");
    if (!t || !startBtn) return;
    let payload, upload = false;
    if (source === "subject") {
      const subject = srcRow.querySelector("#chatSubject")?.value;
      if (!subject) { toast("Pick a subject first.", "error"); return; }
      payload = { mode: "subject", subject, model: window.ui.selectedModel(srcRow) };
    } else if (source === "topic") {
      const topic = srcRow.querySelector("#chatTopic")?.value.trim();
      if (!topic) { toast("Enter a topic first.", "error"); return; }
      payload = { mode: "topic", topic, model: window.ui.selectedModel(srcRow) };
    } else {
      if (!pdfFile) { toast("Choose a PDF first.", "error"); return; }
      upload = true;
    }
    startBtn.disabled = true;
    startBtn.textContent = "Starting…";
    clear(t);
    t.append(spinner(source === "topic" ? "Setting up your tutor…"
      : "Preparing knowledge base… a new or updated subject is indexed once — this can take a moment."));
    try {
      let resp;
      if (upload) {
        const fd = new FormData();
        fd.append("file", pdfFile);
        const m = window.ui.selectedModel(srcRow); if (m) fd.append("model", m);
        resp = await window.api.postForm("/chat/upload", fd);
      } else {
        resp = await window.api.post("/chat/start", payload);
      }
      session = { id: resp.session_id, label: resp.label, chatId: resp.chat_id || null, mode: resp.mode || (upload ? "pdf" : source) };
      msgs = []; hiddenBefore = 0;
      lastStart = upload ? null : { path: "/chat/start", body: payload };
      const card = showWelcome(t, resp.label, upload ? "Ask anything about this PDF." : "");
      setComposerEnabled(true);
      updateSessionBadge(resp.label);
      setChatSetup(false);
      sessionChip();
      refreshExport();
      inputEl()?.focus();
      toast("Session started.", "ok");
      if (!upload) renderOverview(card, session.id);
      refreshDrawer();
    } catch (e) {
      clear(t);
      const msg = e && e.message ? e.message : "Couldn't start the session.";
      t.append(el(`<div class="empty"><div class="big">✦</div><div>${esc(msg)}</div></div>`));
      toast(msg, "error");
    } finally {
      startBtn.disabled = false;
      startBtn.innerHTML = `${window.ui.iconHtml("play")}<span>Start session</span>`;
    }
  }

  // ── Expired session recovery ───────────────────────────────
  async function recoverSession(errMsg, op) {
    const note = el(`<span class="muted" style="display:block;margin-top:.4rem">Starting a new session…</span>`);
    errMsg.append(note);
    if (!lastStart) {
      note.textContent = "Your session expired. Start a new chat from Options to continue.";
      session = null; setComposerEnabled(false);
      return;
    }
    try {
      const resp = await window.api.post(lastStart.path, lastStart.body);
      session = { id: resp.session_id, label: resp.label, chatId: resp.chat_id || null, mode: resp.mode };
      setComposerEnabled(true);
      updateSessionBadge(resp.label);
      note.textContent = "Your session expired, so I started a new one. ";
      const rb = el(`<button type="button" class="btn err-retry">Resend my question</button>`);
      rb.addEventListener("click", () => { if (generating) return; errMsg.remove(); retryOp(op); });
      note.append(rb);
    } catch (err) {
      note.textContent = "Your session expired and a new one couldn’t be started: " + err.message;
      session = null; setComposerEnabled(false);
    }
  }

  // ── Turn runner (send / edit / regenerate / continue) ──────
  //    op = { action, text, tool, targetId, before:[records], userRec, oldAsst, removed, display }
  async function runTurn(op) {
    stopSpeak();
    const t = thread();
    generating = true;
    const controller = new AbortController();
    currentAbort = controller;
    setStopMode(true);
    const isCurrent = () => currentAbort === controller;
    let finished = false;
    let collectedSteps = [];
    let gotDone = false, started = false;
    t.querySelectorAll(".chat-followups, .chat-related").forEach((n) => n.remove());
    t.querySelectorAll(".chat-error").forEach((n) => n.remove());
    t.dataset.stickBottom = "1";

    const asstRec = newRec("assistant", "", { id: uid() });
    const prefix = op.action === "continue" && op.oldAsst ? op.oldAsst.content : "";
    if (op.action === "regenerate" && op.oldAsst) {
      asstRec.versions = [...(op.oldAsst.versions || []), { content: op.oldAsst.content, llm: op.oldAsst.llm, ts: op.oldAsst.ts }].slice(-5);
    } else if (op.action === "continue" && op.oldAsst) {
      asstRec.versions = op.oldAsst.versions || [];
    }

    const thinking = appendThinking(t);
    scrollThreadToBottom(t, { force: true });
    announce("Thinking…");

    // Live bubble: deltas are buffered and flushed once per animation frame.
    const live = (() => {
      let bubble = null, renderer = null, text = prefix, raf = 0, liveLabel = null, started = false;
      const flush = () => { raf = 0; if (!bubble) return; renderer.update(text); scheduleScrollToBottom(t); };
      return {
        push(txt) {
          if (!txt || !isCurrent()) return;
          if (!bubble) {
            thinking.stop();
            bubble = el(`<div class="msg ai msg-streaming chat-live-tail"></div>`);
            renderer = liveRenderer(bubble);
            thinking.el.after(bubble);
            if (liveLabel) bubble.after(liveLabel);
            announce("Answering…");
          }
          started = true;
          text += txt;
          if (!raf) raf = requestAnimationFrame(flush);
        },
        model(llm) {
          if (!llm || !isCurrent()) return;
          asstRec.llm = llm;
          if (liveLabel) { window.ui.updateModelLabel(liveLabel, llm); return; }
          liveLabel = window.ui.renderModelLabel(llm);
          if (!liveLabel) return;
          liveLabel.classList.add("is-live");
          (bubble || thinking.el).after(liveLabel);
          scheduleScrollToBottom(t);
        },
        get text() { return text; },
        get started() { return started; },
        discard() {
          if (raf) cancelAnimationFrame(raf); raf = 0;
          bubble?.remove(); bubble = null; liveLabel?.remove(); liveLabel = null;
        },
      };
    })();

    const clearLive = () => {
      thinking.stop(); thinking.el.remove(); live.discard();
      t.querySelectorAll(".chat-live-tail").forEach((n) => n.remove());
    };
    // A failed edit / regenerate / continue puts the previous messages back (the server kept them too).
    const rollback = () => { if (op.removed && op.removed.length) { restore(op.removed); } op.removed = null; };

    const addFinal = (data) => {
      clearLive();
      Object.assign(asstRec, {
        content: data.answer || "", ts: nowIso(), stopped: !!data.stopped,
        finish: data.finish === "length" || data.finish === "interrupted" ? data.finish : null,
        citations: data.citations || [], steps: data.steps || [], sources: data.sources || [], usage: data.usage || null,
        llm: data.llm || asstRec.llm, followups: data.followups || [], fromHistory: false,
      });
      msgs.push(asstRec);
      appendRec(asstRec);
      const list = Array.isArray(data.artifacts) ? data.artifacts : (data.artifacts ? [data.artifacts] : []);
      if (list.length) {
        const host = el(`<div class="msg-extras"></div>`);
        list.forEach((art) => appendArtifactCard(host, art, asstRec.llm));
        t.append(host);
        asstRec.extras.push(host);
      }
      if (asstRec.followups.length && !asstRec.stopped && !list.length) {
        const row = chipRow(asstRec.followups.slice(0, 3), "chat-followups");
        row.setAttribute("aria-label", "Suggested follow-up questions");
        t.append(row);
        asstRec.extras.push(row);
      }
      if (!asstRec.stopped) {
        const rel = relatedRow(data.related);
        if (rel) { t.append(rel); asstRec.extras.push(rel); }
      }
      scheduleScrollToBottom(t);
      announce("Answer ready.");
    };

    const finish = () => {
      if (finished || !isCurrent()) return;
      finished = true;
      generating = false; currentAbort = null;
      setStopMode(false);
      syncOnline();
      refreshLastBar();
      refreshExport();
      scheduleScrollToBottom(t);
      if (session) {
        const input = inputEl();
        if (input) { input.disabled = false; input.placeholder = COMPOSER_READY_PH; input.removeAttribute("title"); }
      } else setComposerEnabled(false);
      refreshDrawer(msgs.length <= 2 ? 3500 : 0);
    };

    // Stop pressed: keep what streamed (the server saves the same text as a "stopped" answer).
    const onStopped = () => {
      const partial = live.text.replace(/\s+$/, "");
      const had = live.started && partial.trim() && partial.length > prefix.length;
      clearLive();
      if (had) {
        addFinal({ answer: partial, stopped: true, llm: asstRec.llm, steps: collectedSteps });
        op.removed = null;
      } else {
        rollback();
        if (op.action === "send" && op.userRec) {          // nothing answered: hand the message back to the composer
          const idx = msgs.indexOf(op.userRec);
          if (idx >= 0) detachFrom(idx);
          const input = inputEl();
          if (input && !input.value) { input.value = op.display || op.text; autosize(input); }
        }
        thread().append(el(`<div class="msg ai msg-stopped chat-error">⏹ Stopped</div>`));
      }
      announce("Stopped.");
    };

    const onError = (e) => {
      clearLive();
      rollback();
      const errMsg = el(`<div class="msg ai chat-error msg-error" role="alert"></div>`);
      const errTxt = document.createElement("span");
      errTxt.textContent = "⚠ " + (e.message || "Something went wrong.");
      errMsg.append(errTxt);
      { const l = window.ui.renderModelLabel(e.llm); if (l) errMsg.append(l); }
      if (e.status === 404 && /session|expired/i.test(e.message || "")) {
        recoverSession(errMsg, op);
      } else if (e.status === 409) {
        const rb = el(`<button type="button" class="btn err-retry">Reload this chat</button>`);
        rb.addEventListener("click", () => { if (session && session.chatId) resumeChat({ id: session.chatId }); });
        errMsg.append(rb);
      } else if (e.retryable !== false && session) {
        const rb = el(`<button type="button" class="btn err-retry">Retry</button>`);
        rb.addEventListener("click", () => { if (generating) return; errMsg.remove(); retryOp(op); });
        errMsg.append(rb);
      }
      t.append(errMsg);
      toast(e.message, "error");
      if (/expired/i.test(e.message || "") && e.status !== 404) session = null;
      announce("Error: " + (e.message || ""));
      scheduleScrollToBottom(t);
    };

    const reqBody = () => ({
      session_id: session.id, message: op.text, tool: op.tool || null, history: historyFor(op.before || []),
      action: op.action, target_id: op.targetId || null, prefix: prefix || null,
      user_id: op.userRec ? op.userRec.id : null, assistant_id: asstRec.id,
    });

    // Non-stream fallback (no ReadableStream / stream never started). The server's answer already includes the
    // continued prefix, so it is used as-is.
    const sendNonStream = async () => {
      try {
        const resp = await window.api.post("/chat/message", Object.assign(reqBody(), { tool: op.tool || undefined }), { signal: controller.signal });
        if (isCurrent()) addFinal(Object.assign({}, resp, { artifacts: resp.artifacts || resp.artifact, steps: resp.steps || [] }));
      } catch (e) {
        if (controller.signal.aborted || e.name === "AbortError") { if (isCurrent()) onStopped(); }
        else if (isCurrent()) onError(e);
      } finally { finish(); }
    };

    if (!(window.ReadableStream && window.TextDecoder && window.fetch)) { await sendNonStream(); return; }

    try {
      const headers = await streamHeaders();
      const res = await fetch(window.AppConfig.apiBase + "/chat/message/stream", {
        method: "POST", headers, body: JSON.stringify(reqBody()), signal: controller.signal,
      });
      if (!res.ok && [401, 403, 404, 409, 429, 503].includes(res.status)) {
        let body = null; try { body = await res.json(); } catch (_) {}
        const d = body && body.detail;
        const er = new Error((d && typeof d === "object" ? (d.message || JSON.stringify(d)) : d) ||
          (res.status === 503 ? "The AI is temporarily unavailable. Please try again in a moment." : "Request failed (" + res.status + ")"));
        er.status = res.status; er.retryable = res.status === 503 || res.status === 429;
        if (body && body.llm) er.llm = body.llm;
        er.final = true;
        throw er;
      }
      if (!res.ok || !res.body) throw new Error("stream_unavailable_" + (res.status || 0));

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";
      while (true) {
        if (!isCurrent()) { try { await reader.cancel(); } catch { /* ignore */ } return; }
        const { value, done } = await reader.read();
        if (done) break;
        started = true;
        buf += decoder.decode(value, { stream: true });
        let sep;
        while ((sep = buf.indexOf("\n\n")) !== -1) {
          const frame = buf.slice(0, sep);
          buf = buf.slice(sep + 2);
          const line = frame.split("\n").find((l) => l.startsWith("data:"));
          if (!line) continue;
          let ev;
          try { ev = JSON.parse(line.slice(5).trim()); } catch { continue; }
          if (ev.type === "token" && ev.text) {
            live.push(ev.text);
          } else if (ev.type === "model" && ev.llm) {
            live.model(ev.llm);
          } else if (ev.type === "step" && ev.step) {
            collectedSteps.push(ev.step);
            if (isCurrent()) thinking.step(ev.step);
          } else if (ev.type === "done") {
            gotDone = true;
            if (ev.session_id && session) session.id = ev.session_id;
            if (isCurrent()) addFinal(Object.assign({}, ev, { steps: collectedSteps, artifacts: ev.artifacts || ev.artifact }));
            finish();
            try { await reader.cancel(); } catch { /* ignore */ }
            return;
          } else if (ev.type === "error") {
            const er = new Error(ev.detail || ev.message || "The assistant hit an error.");
            if (ev.llm) er.llm = ev.llm;
            if (ev.status) er.status = ev.status;
            er.retryable = !(ev.status >= 400 && ev.status < 500 && ev.status !== 429);
            throw er;
          }
        }
      }
      if (controller.signal.aborted) { if (isCurrent() && !gotDone) onStopped(); finish(); return; }
      if (!started) { await sendNonStream(); return; }
      // Stream ended without a final event: keep what arrived as an interrupted answer the user can continue.
      if (live.started && live.text.trim()) {
        addFinal({ answer: live.text, finish: "interrupted", llm: asstRec.llm, steps: collectedSteps });
        finish();
        toast("The connection dropped mid-answer. You can continue it.", "error");
        return;
      }
      throw new Error("Connection dropped mid-answer. Please try again.");
    } catch (e) {
      if (controller.signal.aborted || e.name === "AbortError") {
        if (isCurrent() && !gotDone) onStopped();
        finish();
        return;
      }
      if (!started && !gotDone && !e.final && session && !/expired/i.test(e.message || "")) { await sendNonStream(); return; }
      if (!gotDone && isCurrent()) onError(e);
      finish();
    }
  }

  function retryOp(op) {
    if (!session || generating) return;
    runTurn(Object.assign({}, op, { removed: null }));
  }

  // ── Actions: send / edit / regenerate / continue ───────────
  function send() {
    if (!session || generating) return;
    if (navigator.onLine === false) { toast("You're offline — check your connection.", "error"); return; }
    const t = thread(), input = inputEl();
    const text = input.value.trim();
    if (!text) return;
    const { tool: forcedTool, rest } = parseForcedTool(text);
    const outgoing = forcedTool ? (rest || `Make a ${forcedTool}.`) : text;
    t.querySelector(".empty")?.remove();
    t.querySelector(".chat-overview")?.remove();
    const before = msgs.slice();
    const userRec = newRec("user", text);
    msgs.push(userRec);
    appendRec(userRec);
    input.value = "";
    autosize(input);
    runTurn({ action: "send", text: outgoing, display: text, tool: forcedTool, before, userRec });
  }

  // Regenerate / Continue act on the latest answer: its bubble is detached (kept for rollback), the user's
  // message stays where it is.
  function latestTurn() {
    const asst = lastAssistant();
    if (!asst) return null;
    const idx = msgs.indexOf(asst);
    const userRec = msgs[idx - 1];
    if (!userRec || userRec.role !== "user") return null;
    return { asst, idx, userRec, before: msgs.slice(0, idx - 1) };
  }
  function regenerateLast() {
    if (generating || !session) return;
    const lt = latestTurn();
    if (!lt) return;
    const removed = detachFrom(lt.idx);
    runTurn({ action: "regenerate", text: lt.userRec.content, tool: null, targetId: lt.asst.id, before: lt.before,
      userRec: lt.userRec, oldAsst: lt.asst, removed });
  }
  function continueLast() {
    if (generating || !session) return;
    const lt = latestTurn();
    if (!lt) return;
    const removed = detachFrom(lt.idx);
    runTurn({ action: "continue", text: lt.userRec.content, tool: null, targetId: lt.asst.id, before: lt.before,
      userRec: lt.userRec, oldAsst: lt.asst, removed });
  }

  function startEdit(rec) {
    if (generating || !session || !rec.node || rec.node.classList.contains("is-editing")) return;
    const node = rec.node;
    node.classList.add("is-editing");
    const box = el(`<div class="msg-edit"><textarea class="input msg-edit-input" rows="2" aria-label="Edit your message"></textarea>
      <div class="msg-edit-bar"><button type="button" class="btn-ghost msg-edit-cancel">Cancel</button><button type="button" class="btn-accent msg-edit-save">Save &amp; send</button></div></div>`);
    const ta = box.querySelector("textarea");
    ta.value = rec.content;
    const bubbleEl = node.querySelector(".msg.me") || node;
    const textEl = node.querySelector(".msg-text"), bar = node.querySelector(".msg-actions");
    textEl.hidden = true; bar.hidden = true;
    bubbleEl.append(box);
    autosize(ta); ta.focus(); ta.setSelectionRange(ta.value.length, ta.value.length);
    ta.addEventListener("input", () => autosize(ta));
    const close = () => { box.remove(); textEl.hidden = false; bar.hidden = false; node.classList.remove("is-editing"); };
    const save = () => {
      const val = ta.value.trim();
      if (!val) { toast("The message can't be empty.", "error"); return; }
      if (val === rec.content) { close(); return; }
      close();
      const idx = msgs.indexOf(rec);
      if (idx < 0) return;
      const before = msgs.slice(0, idx);
      const removed = detachFrom(idx);                       // this message + everything after it (kept for rollback)
      const { tool, rest } = parseForcedTool(val);
      const userRec = newRec("user", val);
      msgs.push(userRec);
      appendRec(userRec);
      runTurn({ action: "edit", text: tool ? (rest || `Make a ${tool}.`) : val, display: val, tool, targetId: rec.id,
        before, userRec, removed });
    };
    box.querySelector(".msg-edit-cancel").addEventListener("click", close);
    box.querySelector(".msg-edit-save").addEventListener("click", save);
    ta.addEventListener("keydown", (e) => {
      if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); close(); }
      else if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); save(); }
    });
  }
  function editLastUser() {
    const lastUser = [...msgs].reverse().find((m) => m.role === "user" && m.node && m.node.isConnected);
    if (lastUser) { lastUser.node.scrollIntoView({ block: "nearest" }); startEdit(lastUser); }
  }

  // ── Scan notes via photo upload (composer 📷) ──────────────
  async function scanPhoto(file) {
    if (!file) return;
    const t = thread();
    if (!t) return;
    t.querySelector(".empty")?.remove();
    t.querySelector(".chat-overview")?.remove();
    const pending = el(`<div class="msg me chat-scan-pending"></div>`);
    let objectUrl = null;
    try {
      objectUrl = URL.createObjectURL(file);
      const thumb = el(`<img class="chat-scan-thumb" alt="Photo of your notes" />`);
      thumb.src = objectUrl;
      pending.append(thumb);
    } catch (_) {}
    pending.append(el(`<span class="chat-scan-status">Scanning your notes…</span>`));
    t.append(pending);
    const chip = addChip({ icon: "📷", text: file.name || "photo", busy: true, id: "scan", title: "Scanning your handwritten notes" });
    announce("Scanning your notes…");
    scrollThreadToBottom(t, { force: true });
    try {
      let toSend = file;
      try {
        if (!(window.Views.scannotes && window.Views.scannotes.shrinkPhoto)) await window.ui.loadScript("/assets/views/scannotes.js");
        if (window.Views.scannotes && window.Views.scannotes.shrinkPhoto) toSend = await window.Views.scannotes.shrinkPhoto(file);
      } catch (_) { /* send the original; the server answers with a clear message if it is too big */ }
      const fd = new FormData();
      fd.append("images", toSend, toSend.name || "note.jpg");
      const resp = await window.api.postForm("/scannotes/extract", fd);
      if (objectUrl) { try { URL.revokeObjectURL(objectUrl); } catch (_) {} }
      pending.remove(); chip?.remove();
      const host = el(`<div class="msg-extras"></div>`);
      appendArtifactCard(host, {
        kind: "scans", id: (resp && resp.id) || null, route: "#scannotes", tool: "Scan Notes",
        title: (resp && resp.title) || "Scanned notes", data: { markdown: (resp && resp.markdown) || "" },
      }, resp && resp.llm);
      t.append(host);
      announce("Notes scanned.");
      scheduleScrollToBottom(t);
    } catch (e) {
      if (objectUrl) { try { URL.revokeObjectURL(objectUrl); } catch (_) {} }
      pending.remove(); chip?.remove();
      const errMsg = el(`<div class="msg ai chat-error msg-error" role="alert"></div>`);
      errMsg.textContent = "⚠ " + (e && e.message ? e.message : "Couldn't scan that photo.");
      { const l = window.ui.renderModelLabel(e && e.llm); if (l) errMsg.append(l); }
      const rb = el(`<button type="button" class="btn err-retry">Retry</button>`);
      rb.addEventListener("click", () => { errMsg.remove(); scanPhoto(file); });
      errMsg.append(rb);
      t.append(errMsg);
      toast(e && e.message ? e.message : "Couldn't scan that photo.", "error");
      scheduleScrollToBottom(t);
    }
  }

  // ── Setup card ─────────────────────────────────────────────
  function setChatSetup(open) {
    const setup = $("chatSetup"), btn = $("chatOptionsBtn");
    if (setup) setup.hidden = !open;
    if (btn) { btn.setAttribute("aria-expanded", open ? "true" : "false"); btn.classList.toggle("is-open", !!open); }
  }

  // ── New chat ───────────────────────────────────────────────
  function resetChat() {
    stopSpeak();
    stopGenerating();
    generating = false; currentAbort = null;
    session = null; msgs = []; hiddenBefore = 0; lastStart = null; source = "subject"; pdfFile = null;
    updateSessionBadge("");
    setChatSetup(true);
    const t = thread();
    if (t) {
      clear(t);
      t.append(el(`<div class="empty"><div class="big">✦</div><div>Start a new session or open a saved chat.</div></div>`));
    }
    const att = $("chatAttach");
    if (att) clear(att);
    setStopMode(false);
    setComposerEnabled(false);
    const input = inputEl();
    if (input) { input.value = ""; autosize(input); }
    closeDrawer();
    refreshExport();
    updateJump();
    const srcRow = $("chatSrcRow");
    if (srcRow) buildSourceRow(srcRow);
    $("chatSrcSeg")?.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", b.dataset.src === "subject" ? "true" : "false"));
    markDrawerActive();
  }

  // ── Conversations drawer: grouped, searchable, pin / rename / delete ──
  const drawer = { items: [], q: "", open: false, loading: false, err: "" };
  let drawerOpener = null, refreshTimer = null;

  function groupOf(item) {
    const d = new Date(item.updated_at || item.created_at || 0);
    const start = new Date(); start.setHours(0, 0, 0, 0);
    const day = 864e5;
    if (d >= start) return "Today";
    if (d >= new Date(start - day)) return "Yesterday";
    if (d >= new Date(start - 7 * day)) return "Previous 7 days";
    return "Older";
  }
  function paintDrawer() {
    const list = $("chatDrawerList");
    if (!list) return;
    list.replaceChildren();
    if (!window.AppConfig || !window.AppConfig.persistenceEnabled) {
      list.append(el(`<div class="muted hist-empty">History is off — set <code>MONGODB_URI</code> to save chats.</div>`));
      return;
    }
    if (drawer.loading && !drawer.items.length) { list.append(spinner("Loading…")); return; }
    if (drawer.err) {
      const e = el(`<div class="muted hist-empty"></div>`);
      e.textContent = drawer.err;
      list.append(e);
      return;
    }
    if (!drawer.items.length) {
      list.append(el(`<div class="muted hist-empty">${drawer.q ? "No matching chats." : "No saved chats yet."}</div>`));
      return;
    }
    const ts = (i) => new Date(i.updated_at || i.created_at || 0).getTime();
    const sorted = [...drawer.items].sort((a, b) => ts(b) - ts(a));
    const sections = [];
    const pinned = sorted.filter((i) => i.pinned);
    if (pinned.length) sections.push(["Pinned", pinned]);
    ["Today", "Yesterday", "Previous 7 days", "Older"].forEach((g) => {
      const rows = sorted.filter((i) => !i.pinned && groupOf(i) === g);
      if (rows.length) sections.push([g, rows]);
    });
    sections.forEach(([name, rows]) => {
      const sec = el(`<section class="cd-group"><h4 class="cd-group-h"></h4></section>`);
      sec.querySelector("h4").textContent = name;
      rows.forEach((item) => sec.append(drawerRow(item)));
      list.append(sec);
    });
    markDrawerActive();
  }
  function markDrawerActive() {
    document.querySelectorAll("#chatDrawerList .cd-row").forEach((r) =>
      r.classList.toggle("is-active", !!(session && session.chatId && r.dataset.id === session.chatId)));
  }
  function drawerRow(item) {
    const row = el(`<div class="cd-row" role="listitem"></div>`);
    row.dataset.id = item.id;
    const open = el(`<button type="button" class="cd-open"></button>`);
    const title = el(`<span class="cd-title"></span>`);
    title.textContent = item.title || item.label || "Untitled chat";
    if (item.pinned) { title.prepend(window.ui.icon("pin", { cls: "cd-pin-ic" })); title.classList.add("is-pinned"); }
    const sub = el(`<span class="cd-sub"></span>`);
    sub.textContent = item.label || "";
    open.append(title, sub);
    open.addEventListener("click", () => { closeDrawer(); resumeChat(item); });
    const acts = el(`<span class="cd-acts"></span>`);
    const pin = actionBtn(item.pinned ? "Unpin chat" : "Pin chat", item.pinned ? "unpin" : "pin", async () => {
      try { await window.api.patch(`/chat/${encodeURIComponent(item.id)}`, { pinned: !item.pinned }); item.pinned = !item.pinned; paintDrawer(); }
      catch (e) { toast(e.message, "error"); }
    });
    const ren = actionBtn("Rename chat", "edit", () => renameRow(row, item));
    const del = actionBtn("Delete chat", "trash", () => confirmDelete(row, item), "cd-del is-danger");
    pin.setAttribute("aria-pressed", String(!!item.pinned));
    acts.append(pin, ren, del);
    row.append(open, acts);
    return row;
  }
  function renameRow(row, item) {
    const open = row.querySelector(".cd-open"), acts = row.querySelector(".cd-acts");
    const input = el(`<input class="input cd-rename" type="text" maxlength="120" aria-label="Chat name" />`);
    input.value = item.title || "";
    open.hidden = true; acts.hidden = true;
    row.prepend(input);
    input.focus(); input.select();
    let done = false;
    const end = async (save) => {
      if (done) return; done = true;
      const val = input.value.trim();
      if (save && val && val !== item.title) {
        try { await window.api.patch(`/chat/${encodeURIComponent(item.id)}`, { title: val }); item.title = val; }
        catch (e) { toast(e.message, "error"); }
      }
      paintDrawer();
    };
    input.addEventListener("keydown", (e) => {
      e.stopPropagation();
      if (e.key === "Enter" && !e.isComposing) { e.preventDefault(); end(true); }
      else if (e.key === "Escape") { e.preventDefault(); end(false); }
    });
    input.addEventListener("blur", () => end(true));
  }
  function confirmDelete(row, item) {
    row.replaceChildren();
    row.classList.add("is-confirm");
    const msg = el(`<span class="cd-confirm-msg"></span>`);
    msg.textContent = `Delete “${item.title || "this chat"}”?`;
    const yes = el(`<button type="button" class="btn cd-yes">Delete</button>`);
    const no = el(`<button type="button" class="btn-ghost cd-no">Cancel</button>`);
    row.append(msg, yes, no);
    no.focus();
    no.addEventListener("click", paintDrawer);
    row.addEventListener("keydown", (e) => { if (e.key === "Escape") { e.stopPropagation(); paintDrawer(); } });
    yes.addEventListener("click", async () => {
      yes.disabled = true;
      try {
        await window.api.del(`/history/chats/${encodeURIComponent(item.id)}`);
        drawer.items = drawer.items.filter((i) => i.id !== item.id);
        if (session && session.chatId === item.id) resetChat(); else paintDrawer();
        toast("Chat deleted.", "ok");
      } catch (e) { toast(e.message, "error"); paintDrawer(); }
    });
  }
  async function loadDrawer() {
    if (!window.AppConfig || !window.AppConfig.persistenceEnabled) { paintDrawer(); return; }
    const seq = ++drawerSeq;
    drawer.loading = true; drawer.err = "";
    paintDrawer();
    try {
      const data = await window.api.get("/history/chats" + (drawer.q ? `?q=${encodeURIComponent(drawer.q)}` : ""));
      if (seq !== drawerSeq) return;
      drawer.items = (data && data.items) || [];
    } catch (e) {
      if (seq !== drawerSeq) return;
      drawer.err = e.status === 401 ? "Sign in to see your saved chats." : "Couldn’t load your chats.";
      drawer.items = [];
    }
    drawer.loading = false;
    paintDrawer();
  }
  function refreshDrawer(delay = 0) {
    if (refreshTimer) { clearTimeout(refreshTimer); refreshTimer = null; }
    if (!drawer.open) return;
    if (delay) refreshTimer = setTimeout(() => { refreshTimer = null; if (drawer.open) loadDrawer(); }, delay);
    else loadDrawer();
  }
  function openDrawer(opener) {
    const d = $("chatDrawer");
    if (!d) return;
    drawerOpener = opener || document.activeElement;
    drawer.open = true;
    d.hidden = false;
    $("chatDrawerBg").hidden = false;
    $("chatHistBtn")?.setAttribute("aria-expanded", "true");
    requestAnimationFrame(() => d.classList.add("is-open"));
    $("chatDrawerSearch").value = drawer.q;
    $("chatDrawerSearch").focus();
    loadDrawer();
  }
  function closeDrawer() {
    const d = $("chatDrawer");
    if (!d || !drawer.open) return;
    drawer.open = false;
    d.classList.remove("is-open");
    d.hidden = true;
    $("chatDrawerBg").hidden = true;
    $("chatHistBtn")?.setAttribute("aria-expanded", "false");
    if (drawerOpener && drawerOpener.isConnected) drawerOpener.focus();
  }

  // ── Export (Markdown / PDF) ────────────────────────────────
  function chatTitle() {
    const hit = drawer.items.find((i) => session && i.id === session.chatId);
    return (hit && hit.title) || (session && session.label) || "Chat";
  }
  function refreshExport() {
    const b = $("chatExportBtn");
    if (b) b.disabled = !msgs.some((m) => m.role === "assistant" && m.content);
  }
  function exportMarkdown() {
    const lines = [`# ${chatTitle()}`, "", `_Exported from Apex Agent · ${new Date().toLocaleString()}_`, ""];
    msgs.forEach((m) => {
      const who = m.role === "user" ? "You" : "Apex Agent";
      const v = m.role === "assistant" ? allVersions(m)[shownIdx(m)] : null;
      const content = v ? v.content : m.content;
      lines.push(`### ${who}${m.ts ? " · " + new Date(m.ts).toLocaleString() : ""}`, "", content, "");
      if (v && v.llm && v.llm.label) lines.push(`> ${v.llm.label}`, "");
    });
    const blob = new Blob([lines.join("\n")], { type: "text/markdown;charset=utf-8" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = (chatTitle().replace(/[^\w\- ]+/g, "").trim().replace(/\s+/g, "-").slice(0, 60) || "chat") + ".md";
    document.body.append(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 2000);
  }
  function exportPdf() {
    const wrap = document.createElement("div");
    msgs.forEach((m) => {
      const h = document.createElement("h3");
      h.textContent = (m.role === "user" ? "You" : "Apex Agent") + (m.ts ? " · " + new Date(m.ts).toLocaleString() : "");
      wrap.append(h);
      if (m.role === "user") { const p = document.createElement("p"); p.textContent = m.content; wrap.append(p); }
      else wrap.append(md(allVersions(m)[shownIdx(m)].content || ""));
    });
    window.ui.printDoc(chatTitle(), wrap.innerHTML);
  }
  function toggleExportMenu(force) {
    const m = $("chatExportMenu");
    if (!m) return;
    const show = force == null ? m.hidden : force;
    m.hidden = !show;
    $("chatExportBtn")?.setAttribute("aria-expanded", String(show));
  }

  // ── Mount ──────────────────────────────────────────────────
  window.Views = window.Views || {};
  window.Views.chat = {
    accent: "chat",
    title: "Apex Agent",

    async mount(root) {
      root.append(el(`
        <div class="chat-shell stack">
          <div class="chat-minibar panel">
            <button id="chatHistBtn" class="btn tb-btn" type="button" aria-haspopup="dialog" aria-expanded="false" aria-controls="chatDrawer" title="Your chats" aria-label="Your chats">${window.ui.iconHtml("history")}<span class="tb-text">Chats</span></button>
            <button id="chatOptionsBtn" class="btn tb-btn" type="button" aria-expanded="true" aria-controls="chatSetup" title="Show or hide chat setup" aria-label="Chat options">${window.ui.iconHtml("options")}<span class="tb-text">Options</span></button>
            <span id="chatSessionBadge"></span>
            <span style="flex:1"></span>
            <span class="chat-export">
              <button id="chatExportBtn" class="btn tb-btn" type="button" aria-haspopup="menu" aria-expanded="false" title="Export chat" aria-label="Export chat" disabled>${window.ui.iconHtml("download")}<span class="tb-text">Export</span></button>
              <div id="chatExportMenu" class="chat-tools-menu chat-export-menu" role="menu" hidden>
                <button class="chat-tool-item" type="button" role="menuitem" data-x="md">Markdown (.md)</button>
                <button class="chat-tool-item" type="button" role="menuitem" data-x="pdf">PDF (print)</button>
              </div>
            </span>
            <button id="chatNewBtn" class="btn-accent tb-btn" type="button" title="New chat (Ctrl/⌘+K)" aria-label="New chat">${window.ui.iconHtml("plus")}<span class="tb-text">New</span></button>
          </div>

          <div id="chatSetup" class="chat-src-bar panel">
            <div class="row chat-setup-row">
              <div class="seg" role="group" aria-label="Source" id="chatSrcSeg">
                <button type="button" data-src="subject" aria-pressed="true">Subject</button>
                <button type="button" data-src="topic"   aria-pressed="false">Topic</button>
                <button type="button" data-src="upload"  aria-pressed="false">Upload PDF</button>
              </div>
              <div id="chatSrcRow" class="row chat-src-row" style="gap:.6rem"></div>
              <div id="chatModelCtl"></div>
              <label class="chat-auto-read" id="chatAutoReadLbl" hidden><input type="checkbox" id="chatAutoRead" /> <span>Read replies aloud automatically</span></label>
              <button id="chatStart" class="btn-accent" type="button">${window.ui.iconHtml("play")}<span>Start session</span></button>
            </div>
          </div>

          <div class="panel chat-panel">
            <div class="thread" id="chatThread" tabindex="-1">
              <div class="empty"><div class="big">✦</div><div>Start a new session or open a saved chat.</div></div>
            </div>
            <button id="chatJump" class="chat-jump" type="button" hidden aria-label="Jump to latest message" title="Jump to latest message">${window.ui.iconHtml("chevron-down")}<span>Jump to latest</span></button>
            <div id="chatNet" class="chat-net" hidden>You're offline — you can read this chat, but sending is paused until you're back online.</div>
            <div id="chatAttach" class="chat-attach" aria-label="Attachments"></div>
            <div class="composer" id="chatComposer">
              <button id="chatScanBtn" class="chat-scan-btn" type="button" title="Scan handwritten notes from a photo" aria-label="Scan notes from a photo">${window.ui.iconHtml("camera")}</button>
              <input id="chatScanInput" type="file" accept="image/*" hidden />
              <button id="chatToolsBtn" class="chat-tools-btn" type="button" title="Tools — run a tool on demand" aria-label="Choose a tool" aria-haspopup="true" aria-expanded="false">${window.ui.iconHtml("menu")}</button>
              <div id="chatToolsMenu" class="chat-tools-menu" role="menu" hidden></div>
              <textarea class="input" rows="1" placeholder="Start a session to chat…" aria-label="Message" title="${esc(COMPOSER_IDLE_TITLE)}" disabled></textarea>
              <button id="chatSend" class="btn-accent icon-btn has-text send-btn" type="button" aria-label="Send message" data-tip="Send message" data-shortcut="Enter" disabled>${window.ui.iconHtml("send")}<span class="icon-btn-text">Send</span></button>
            </div>
            <div id="chatStatus" class="sr-only" role="status" aria-live="polite"></div>
          </div>

          <div id="chatDrawerBg" class="chat-drawer-bg" hidden></div>
          <aside id="chatDrawer" class="chat-drawer" role="dialog" aria-modal="false" aria-label="Your chats" hidden>
            <div class="cd-head">
              <input id="chatDrawerSearch" class="input" type="search" placeholder="Search chats…" aria-label="Search chats" autocomplete="off" />
              <button id="chatDrawerClose" class="icon-btn" type="button" aria-label="Close chats" title="Close chats">${window.ui.iconHtml("close")}</button>
            </div>
            <button id="chatDrawerNew" class="btn-accent cd-new" type="button">${window.ui.iconHtml("plus")}<span>New chat</span></button>
            <div id="chatDrawerList" class="cd-list" role="list"></div>
          </aside>
        </div>`));

      const shell = root.querySelector(".chat-shell");
      installShellSizing(shell);
      installThreadScroll($("chatThread"));

      $("chatModelCtl").append(window.ui.modelSelect());
      if (speechSupported()) {
        $("chatAutoReadLbl").hidden = false;
        $("chatAutoRead").addEventListener("change", (e) => { try { window.Voice && window.Voice.setAuto(e.target.checked); } catch { /* voice.js not ready */ } });
      }

      const seg = $("chatSrcSeg"), srcRow = $("chatSrcRow");
      buildSourceRow(srcRow);
      seg.addEventListener("click", (e) => {
        const btn = e.target.closest("button[data-src]");
        if (!btn) return;
        source = btn.dataset.src; pdfFile = null;
        seg.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", b === btn ? "true" : "false"));
        buildSourceRow(srcRow);
      });

      $("chatStart").addEventListener("click", startSession);
      $("chatNewBtn").addEventListener("click", resetChat);
      $("chatHistBtn").addEventListener("click", (e) => (drawer.open ? closeDrawer() : openDrawer(e.currentTarget)));
      $("chatOptionsBtn").addEventListener("click", () => setChatSetup($("chatSetup")?.hidden));
      $("chatDrawerClose").addEventListener("click", closeDrawer);
      $("chatDrawerBg").addEventListener("click", closeDrawer);
      $("chatDrawerNew").addEventListener("click", resetChat);
      const dSearch = $("chatDrawerSearch");
      const dDebounced = window.ui.debounce(() => { drawer.q = dSearch.value.trim(); loadDrawer(); }, 250);
      dSearch.addEventListener("input", dDebounced);
      $("chatDrawer").addEventListener("keydown", (e) => { if (e.key === "Escape" && e.target.tagName !== "INPUT") closeDrawer(); });
      dSearch.addEventListener("keydown", (e) => {
        if (e.key === "Escape") { if (dSearch.value) { dSearch.value = ""; drawer.q = ""; loadDrawer(); } else closeDrawer(); }
      });

      $("chatExportBtn").addEventListener("click", (e) => { e.stopPropagation(); toggleExportMenu(); });
      $("chatExportMenu").addEventListener("click", (e) => {
        const b = e.target.closest("[data-x]");
        if (!b) return;
        toggleExportMenu(false);
        if (b.dataset.x === "md") exportMarkdown(); else exportPdf();
      });

      $("chatJump").addEventListener("click", () => {
        const t = $("chatThread");
        // Instant (not smooth): a smooth scroll emits scroll events mid-flight that would re-mark the thread as
        // "not at bottom" and keep the button up.
        t.scrollTop = t.scrollHeight;
        t.dataset.stickBottom = "1"; unseen = false;
        updateJump();
      });

      // Composer
      const input = inputEl();
      $("chatSend").addEventListener("click", () => { if (generating) stopGenerating(); else send(); });
      input.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.isComposing && !e.altKey) {
          if (e.shiftKey) return;                              // Shift+Enter: newline
          e.preventDefault(); if (!generating) send();
        }
        else if (e.key === "ArrowUp" && !input.value && !e.shiftKey) { e.preventDefault(); editLastUser(); }
        else if (e.key === "Escape" && generating) { e.preventDefault(); stopGenerating(); }
      });
      input.addEventListener("input", () => { autosize(input); if (input.value === "/") openToolsMenu(); });

      const toolsBtn = $("chatToolsBtn"), toolsMenu = $("chatToolsMenu");
      const toggleToolsMenu = () => { if (toolsMenu.hidden) openToolsMenu(); else closeToolsMenu(); };
      toolsBtn.addEventListener("click", (e) => { e.stopPropagation(); toggleToolsMenu(); });

      const scanInput = $("chatScanInput");
      $("chatScanBtn").addEventListener("click", () => scanInput.click());
      scanInput.addEventListener("change", () => {
        const file = scanInput.files && scanInput.files[0];
        scanInput.value = "";
        if (file) scanPhoto(file);
      });

      // Document-level shortcuts + outside clicks (removed on unmount)
      const onDocClick = (e) => {
        if (!toolsMenu.hidden && !toolsMenu.contains(e.target) && e.target !== toolsBtn) closeToolsMenu();
        const em = $("chatExportMenu");
        if (em && !em.hidden && !em.contains(e.target) && e.target !== $("chatExportBtn")) toggleExportMenu(false);
      };
      const onDocKey = (e) => {
        const typing = /^(INPUT|TEXTAREA|SELECT)$/.test(e.target.tagName) || e.target.isContentEditable;
        if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") { e.preventDefault(); resetChat(); return; }
        if (e.key === "Escape") {
          if (!toolsMenu.hidden) { closeToolsMenu(); return; }
          if (!$("chatExportMenu").hidden) { toggleExportMenu(false); return; }
          if (generating) { stopGenerating(); return; }
          return;
        }
        if (e.key === "/" && !typing && !e.ctrlKey && !e.metaKey && !e.altKey) {
          const i = inputEl();
          if (i && !i.disabled) { e.preventDefault(); i.focus(); }
        }
      };
      const onNet = () => syncOnline();
      document.addEventListener("click", onDocClick);
      document.addEventListener("keydown", onDocKey);
      window.addEventListener("online", onNet);
      window.addEventListener("offline", onNet);
      if (toolsMenuCleanup) toolsMenuCleanup();
      toolsMenuCleanup = () => {
        document.removeEventListener("click", onDocClick);
        document.removeEventListener("keydown", onDocKey);
        window.removeEventListener("online", onNet);
        window.removeEventListener("offline", onNet);
      };

      // Hide the "Auto" model row when there is nothing to choose (keeps the setup card compact).
      const ms = root.querySelector("#chatModelCtl .model-select");
      const hideIfInert = () => { if (ms && ms.disabled) $("chatModelCtl").hidden = true; };
      if (window.AppConfig && window.AppConfig.loaded) setTimeout(hideIfInert, 0);
      else if (window.AppConfig && window.AppConfig.ready) window.AppConfig.ready.then(() => setTimeout(hideIfInert, 0));
      syncOnline();
      const _libArt = window.ui.takeArtifact(["chats"]);      // Library deep-link: re-open that chat
      if (_libArt) resumeChat({ id: _libArt.id });
    },

    unmount() {
      stopSpeak();
      stopGenerating();
      generating = false; currentAbort = null;
      session = null; msgs = []; hiddenBefore = 0; lastStart = null; source = "subject"; pdfFile = null;
      drawer.open = false; drawer.items = []; drawer.q = "";
      if (refreshTimer) { clearTimeout(refreshTimer); refreshTimer = null; }
      if (scrollRaf) { cancelAnimationFrame(scrollRaf); scrollRaf = 0; }
      if (teardownLayout) teardownLayout();
      if (teardownThreadScroll) teardownThreadScroll();
      if (toolsMenuCleanup) { toolsMenuCleanup(); toolsMenuCleanup = null; }
      document.getElementById("histPopover")?.remove();
      document.getElementById("histPopoverBg")?.remove();
      document.body.classList.remove("hist-scroll-lock");
    },
  };
})();
