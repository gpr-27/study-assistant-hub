/* ============================================================
   Chat Assistant — redesigned: single-panel layout, floating
   history drawer, source attribution, agentic trace display.
   ============================================================ */
(() => {
  "use strict";
  const { el, esc, clear, toast, md, spinner, copy } = window.ui;

  let session = null;       // { id, label }
  let source  = "subject";  // "subject" | "topic" | "upload"
  let pdfFile = null;
  let teardownLayout = null; // removes the responsive shell-sizing listeners on unmount
  let generating = false;    // a response is currently streaming
  let currentAbort = null;   // AbortController for the in-flight stream (the Stop button)
  let toolsMenuCleanup = null; // removes the tools-menu document listeners on unmount

  // Tools the Apex Agent can run on demand. Picking one prefixes "/key " into the
  // composer so the NEXT message forces that tool (ChatGPT-plugin style). key = backend id.
  const TOOLS = [
    { key: "quiz",       label: "Quiz",              icon: "❓" },
    { key: "flashcards", label: "Flashcards",        icon: "🃏" },
    { key: "summary",    label: "Summarize",         icon: "📝" },
    { key: "plan",       label: "Study Plan",        icon: "🗓" },
    { key: "mindmap",    label: "Mind Map",          icon: "🧠" },
    { key: "cheatsheet", label: "Cheat Sheet",       icon: "📋" },
    { key: "predictor",  label: "Predict Questions", icon: "🔮" },
    { key: "exam",       label: "Mock Exam",         icon: "📄" },
    { key: "teachback",  label: "Teach-back",        icon: "🎙" },
    { key: "progress",   label: "My Progress",       icon: "📊" },
  ];
  const TOOL_KEYS = new Set(TOOLS.map((t) => t.key));

  // Parse a leading "/tool" slash command → { tool, rest } (rest = message minus the token).
  function parseForcedTool(text) {
    // whole token up to whitespace/end → tested against TOOL_KEYS, so "/plan2week" is NOT "plan".
    const m = (text || "").match(/^\/([a-z0-9]+)(?=\s|$)/i);
    if (m && TOOL_KEYS.has(m[1].toLowerCase())) {
      return { tool: m[1].toLowerCase(), rest: text.slice(m[0].length).trim() };
    }
    return { tool: null, rest: text };
  }

  // Send button doubles as a Stop control while a response streams.
  function setStopMode(on) {
    const btn = document.getElementById("chatSend");
    if (!btn) return;
    btn.textContent = on ? "Stop" : "Send";
    btn.classList.toggle("is-stop", on);
    btn.disabled = false;
  }
  function stopGenerating() {
    if (currentAbort) { try { currentAbort.abort(); } catch { /* already aborted */ } }
  }

  // ── On-demand tool picker (slash menu) ─────────────────────
  function openToolsMenu() {
    if (!session) { toast("Start a session first, then pick a tool.", "error"); return; }
    const menu = document.getElementById("chatToolsMenu");
    if (menu) menu.hidden = false;
    document.getElementById("chatToolsBtn")?.setAttribute("aria-expanded", "true");
  }
  function closeToolsMenu() {
    const menu = document.getElementById("chatToolsMenu");
    if (menu) menu.hidden = true;
    document.getElementById("chatToolsBtn")?.setAttribute("aria-expanded", "false");
  }
  function toggleToolsMenu() {
    const menu = document.getElementById("chatToolsMenu");
    if (menu && menu.hidden) openToolsMenu(); else closeToolsMenu();
  }
  function pickTool(key) {
    closeToolsMenu();
    const input = document.querySelector("#chatComposer .input");
    if (!input) return;
    const rest = input.value.replace(/^\/[a-z0-9]+\s*/i, "");   // replace any existing /tool prefix
    input.value = `/${key} ` + rest;
    input.disabled = false;
    input.focus();
  }

  // ── Responsive height: size the chat shell to the space actually left below the
  //    nav + view header so the composer always stays pinned on-screen — on any
  //    phone or tablet, and above the on-screen keyboard. CSS offsets can only
  //    guess the chrome height; measuring it is exact.
  //
  //    CRITICAL: measure the shell's offset from the top of the DOCUMENT
  //    (rect.top + scrollY), NOT its viewport-relative top. Viewport-relative top
  //    shrinks as the page scrolls, so deriving the height from it — and listening
  //    to scroll — formed a feedback loop that grew the shell without bound
  //    ("chat goes infinite down"). Document-relative offset is scroll-invariant,
  //    so the height is stable and we only need to react to resize (URL bar /
  //    keyboard / orientation), never to scroll.
  function installShellSizing(shell) {
    if (teardownLayout) teardownLayout();   // self-heal: drop any prior listeners on re-mount
    const vp = window.visualViewport;
    const fit = () => {
      if (!shell.isConnected) return;
      const rect = shell.getBoundingClientRect();
      const offsetTop = rect.top + (window.scrollY || window.pageYOffset || 0);
      const viewportH = vp ? vp.height : window.innerHeight;
      shell.style.height = Math.max(320, viewportH - offsetTop - 12) + "px";
    };
    requestAnimationFrame(fit);
    window.addEventListener("resize", fit);
    window.addEventListener("orientationchange", fit);
    if (vp) vp.addEventListener("resize", fit);
    teardownLayout = () => {
      window.removeEventListener("resize", fit);
      window.removeEventListener("orientationchange", fit);
      if (vp) vp.removeEventListener("resize", fit);
      teardownLayout = null;
    };
  }

  // ── History popover (shared solid drawer, same as the other tools) ────────
  function openHistoryDrawer(anchor) {
    window.ui.openHistoryDrawer("chats", {
      title: "Recent chats",
      anchor,
      onOpen: (saved) => resumeChat(saved),
    });
  }

  // ── Source control (inline compact) ───────────────────────
  function buildSourceRow(host) {
    clear(host);
    host.dataset.src = source;

    if (source === "subject") {
      host.append(el(`
        <select id="chatSubject" class="select" style="max-width:220px">
          <option value="">Loading…</option>
        </select>`));
      const sel = host.querySelector("#chatSubject");
      window.api.get("/subjects").then((d) => {
        const subjects = d.subjects || [];
        clear(sel);
        if (!subjects.length) { sel.append(el(`<option value="">No subjects</option>`)); return; }
        subjects.forEach((s) => sel.append(el(`<option value="${esc(s)}">${esc(s)}</option>`)));
        sel.value = subjects[0];
      }).catch(() => {
        clear(sel); sel.append(el(`<option value="">Error loading</option>`));
      });
    } else if (source === "topic") {
      host.append(el(`
        <input id="chatTopic" class="input" type="text" placeholder="e.g. Binary search trees"
          autocomplete="off" style="max-width:260px" />`));
    } else {
      const wrap = el(`<div class="row"></div>`);
      const lbl = el(`<label class="btn" for="chatPdfUp" style="cursor:pointer;padding:.55rem .9rem">Choose PDF…</label>`);
      const name = el(`<span id="chatPdfName" class="muted" style="font-size:.85rem">No file chosen</span>`);
      const inp = el(`<input id="chatPdfUp" type="file" accept="application/pdf" hidden />`);
      inp.addEventListener("change", () => {
        pdfFile = inp.files[0] || null;
        name.textContent = pdfFile ? pdfFile.name : "No file chosen";
      });
      wrap.append(lbl, name, inp);
      host.append(wrap);
    }
  }

  // ── Render one message (user or AI) ───────────────────────
  function appendMessage(thread, role, content, { steps = [], sources = [] } = {}) {
    if (role === "user") {
      const me = el(`<div class="msg me"></div>`);
      me.textContent = content;
      thread.append(me);
      return;
    }

    const ai = el(`<div class="msg ai"></div>`);
    ai.append(md(content));

    const footer = el(`<div class="msg-footer"></div>`);

    // Copy button
    const cp = el(`<button class="msg-copy" type="button">Copy</button>`);
    cp.addEventListener("click", () => copy(content, cp));
    footer.append(cp);

    // Source chips
    if (sources.length) {
      const srcs = el(`<div class="msg-sources"></div>`);
      sources.forEach((s) => srcs.append(el(`<span class="msg-source-chip">${esc(s)}</span>`)));
      footer.append(srcs);
    }

    // Agentic trace — structured, expandable timeline of what actually happened
    if (steps.length) footer.append(buildTrace(steps));

    ai.append(footer);
    thread.append(ai);
  }

  // ── Artifact card: the chat created something in another tool ──
  //    When the backend ships a renderable `data` payload AND the inline renderer
  //    succeeds, show the artifact as a LIVE interactive widget in the thread plus
  //    a muted "Saved in <tool>" footer with an Open button. Otherwise fall back to
  //    the plain link card below.
  function appendArtifactCard(thread, art) {
    if (!art || typeof art !== "object") return;   // tolerate a missing/empty artifact
    if (art.data && typeof window.renderArtifactInline === "function") {
      const block = el(`<div class="chat-live"></div>`);
      let ok = false;
      try {
        ok = window.renderArtifactInline(block, art) === true;
      } catch {
        ok = false;
      }
      if (ok) {
        const foot = el(`<div class="cl-foot"></div>`);
        const label = el(`<span class="cl-foot-label"></span>`);
        label.textContent = "Saved in " + (art.tool || "the tool");
        foot.append(label);
        const open = el(`<button class="cl-foot-open" type="button">Open →</button>`);
        open.addEventListener("click", () => {
          window.ui.openArtifact(art.kind, art.id, art.route || "#home");
        });
        foot.append(open);
        block.append(foot);
        thread.append(block);
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
    card.append(meta);
    // Deep-link: when the artifact was persisted (has an id), hand it off so the
    // target tool opens the ACTUAL saved item instead of its blank Setup page.
    // Without an id (persistence off) keep the plain href navigation to the tool.
    if (art.id) {
      card.addEventListener("click", (e) => {
        e.preventDefault();
        window.ui.openArtifact(art.kind, art.id, art.route || "#home");
      });
    }
    thread.append(card);
  }

  // ── Agentic trace renderer ────────────────────────────────
  function stepIcon(step) {
    if (step.kind === "rewrite")  return "✎";
    if (step.kind === "retrieve") return "⌕";
    if (step.kind === "grade")    return step.verdict === "sufficient" ? "✓" : step.verdict === "limit" ? "◐" : "⚠";
    return "✦"; // generate
  }

  function traceNote(text) {
    const n = el(`<div class="trace-note"></div>`);
    n.textContent = text;
    return n;
  }

  function passageCard(p) {
    const card = el(`<div class="trace-pass"></div>`);
    const head = el(`<div class="trace-pass-head"></div>`);
    const src = el(`<span class="trace-pass-src"></span>`);
    src.textContent = p.source || "—";
    head.append(src);
    if (p.page) {
      const pg = el(`<span class="trace-pass-pg"></span>`);
      pg.textContent = "p." + p.page;
      head.append(pg);
    }
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
        step.queries.forEach((q) => {
          const chip = el(`<span class="trace-q"></span>`);
          chip.textContent = q;
          ql.append(chip);
        });
        d.append(ql);
      }
    } else if (step.kind === "retrieve") {
      if (step.passages && step.passages.length) {
        step.passages.forEach((p) => d.append(passageCard(p)));
      } else {
        d.append(traceNote("No new passages this round."));
      }
    } else if (step.kind === "grade") {
      const label = step.verdict === "sufficient" ? "✓ Sufficient"
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
    const btn = el(`<button class="msg-trace-btn" type="button"></button>`);
    const timeline = el(`<ol class="trace-timeline" hidden></ol>`);
    const n = steps.length;
    const setLabel = (open) =>
      btn.innerHTML = `<span class="trace-chev">${open ? "▾" : "▸"}</span> ${open ? "Hide" : "Show"} reasoning`
        + ` <span class="trace-count">${n} step${n === 1 ? "" : "s"}</span>`;

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
      head.addEventListener("click", () => {
        detail.hidden = !detail.hidden;
        node.classList.toggle("is-open", !detail.hidden);
      });
      body.append(head, detail);
      node.append(dot, body);
      timeline.append(node);
    });

    setLabel(false);
    btn.addEventListener("click", () => {
      timeline.hidden = !timeline.hidden;
      setLabel(!timeline.hidden);
    });
    wrap.append(btn, timeline);
    return wrap;
  }

  // ── Session-start overview card (NotebookLM-style empty state) ──
  //    Best-effort: ONE cheap backend call. Renders a short overview + up to 3
  //    tappable starter-question chips into the thread's empty state. On failure
  //    (or empty result) we silently keep the plain empty state — never block the
  //    chat, never toast an error.
  async function renderOverview(thread, sessionId) {
    // Only meaningful before any message exists.
    if (thread.querySelector(".msg")) return;
    const placeholder = thread.querySelector(".empty");

    // Light loading hint appended under the placeholder while we fetch.
    const loading = el(`<div class="chat-overview-loading muted" style="text-align:center;font-size:.85rem;margin-top:.6rem">Preparing an overview…</div>`);
    if (placeholder) placeholder.append(loading); else thread.append(loading);

    let data;
    try {
      data = await window.api.get(`/chat/overview/${encodeURIComponent(sessionId)}`);
    } catch {
      loading.remove();
      return; // graceful fallback: keep the plain empty state
    }
    loading.remove();

    // Session may have changed/ended while we waited, or a message already sent.
    if (!session || session.id !== sessionId) return;
    if (thread.querySelector(".msg")) return;

    const overview = (data && typeof data.overview === "string") ? data.overview : "";
    const questions = (data && Array.isArray(data.questions)) ? data.questions.filter(Boolean) : [];
    if (!overview && !questions.length) return; // fall back to plain empty state

    const card = el(`<div class="chat-overview"></div>`);

    const title = el(`<div class="chat-overview-title"></div>`);
    title.textContent = (data && data.label) || session.label || "Overview";
    card.append(title);

    const sources = (data && Number(data.sources)) || 0;
    if (sources > 0) {
      const src = el(`<div class="chat-overview-sources"></div>`);
      src.textContent = sources + " source" + (sources === 1 ? "" : "s");
      card.append(src);
    }

    if (overview) {
      const body = el(`<div class="chat-overview-body"></div>`);
      body.append(md(overview));
      card.append(body);
    }

    if (questions.length) {
      const qs = el(`<div class="chat-overview-qs"></div>`);
      questions.slice(0, 3).forEach((q) => {
        const btn = el(`<button class="chat-overview-q" type="button"></button>`);
        btn.textContent = q;
        btn.addEventListener("click", () => askQuestion(q));
        qs.append(btn);
      });
      card.append(qs);
    }

    // The merged chat both answers AND builds — surface that here with a small
    // labelled row of tappable "build" prompts, each of which sends through the
    // same path a typed message uses.
    card.append(buildPromptChips());

    // Replace the plain "Session ready" placeholder with the overview card so the
    // empty state IS the NotebookLM-style card (not the card plus a leftover line).
    if (placeholder) placeholder.remove();
    thread.append(card);
    thread.scrollTop = thread.scrollHeight;
  }

  // ── "Build" example prompts (discoverability) ─────────────
  //    Reuses the shared .agent-prompt-chips / .agent-prompt-chip styling.
  const BUILD_PROMPTS = [
    "Make a quiz",
    "Create a cheat sheet",
    "Generate flashcards",
    "Build a mind map",
    "Make a 2-week study plan",
  ];

  function buildPromptChips() {
    const wrap = el(`<div class="chat-overview-build"></div>`);
    wrap.append(el(`<div class="chat-overview-build-label muted">Or ask me to build something:</div>`));
    const chips = el(`<div class="agent-prompt-chips"></div>`);
    BUILD_PROMPTS.forEach((p) => {
      const chip = el(`<button class="agent-prompt-chip" type="button"></button>`);
      chip.textContent = p;
      chip.addEventListener("click", () => askQuestion(p));
      chips.append(chip);
    });
    wrap.append(chips);
    return wrap;
  }

  // Send a starter question through the EXISTING send path (same code a typed
  // message uses): drop it into the composer input and trigger send().
  function askQuestion(text) {
    const composer = document.querySelector("#chatComposer");
    if (!composer) return;
    const input = composer.querySelector(".input");
    if (!input || input.disabled) return;
    input.value = text;
    send();
  }

  // ── Live reasoning rails ──────────────────────────────────
  //    Replaces the old hardcoded THINKING_STEPS cycle. As the backend streams
  //    real steps over SSE we append a row into a "reasoning" panel inside the
  //    pending AI bubble. Two visually distinct rails by step.kind:
  //      • kind === "tool"  → TOOL rail (building an artifact)
  //      • rewrite/retrieve/grade/generate → GROUNDING rail (RAG over notes)
  //    On done the bubble is removed and the final answer + collapsed trace render.

  // Friendly, rail-specific copy for one live step.
  function liveStepCopy(step) {
    const kind = step.kind || "generate";
    if (kind === "tool") {
      const tool = step.tool || step.title || "tool";
      return { rail: "tool", ico: "🛠", label: step.title || ("Building " + tool) };
    }
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
      const label = v === "sufficient" ? "Verified — context is sufficient"
                  : v === "limit" ? "Verifying — reached search limit"
                  : v === "insufficient" ? "Verifying — looking for more"
                  : "Verifying context";
      return { rail: "ground", ico: "✓", label };
    }
    // generate
    return { rail: "ground", ico: "✦", label: step.title || "Answering from your notes…" };
  }

  // Render one live row and append it; previous rows lose their "active" pulse.
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
    return row;
  }

  // The pending AI bubble that hosts the live reasoning panel.
  function appendThinking(thread) {
    const wrap = el(`<div class="msg ai msg-thinking"></div>`);
    const panel = el(`<div class="reason-live"></div>`);
    wrap.append(panel);
    thread.append(wrap);
    return {
      el: wrap,
      panel,
      step(s) { appendLiveStep(panel, s); thread.scrollTop = thread.scrollHeight; },
      stop() { panel.querySelectorAll(".reason-step.is-active").forEach((r) => r.classList.remove("is-active")); },
    };
  }

  // ── Build the same identity headers the API layer attaches, for raw fetch ──
  //    Mirrors app.js authHeaders(): Clerk bearer token + guest id. Kept local so
  //    the streaming fetch carries auth without exposing app.js internals.
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

  // ── Resume saved chat ──────────────────────────────────────
  async function resumeChat(saved) {
    const thread = document.querySelector("#chatThread");
    const composer = document.querySelector("#chatComposer");
    if (!thread || !composer) return;
    if (generating) { stopGenerating(); generating = false; currentAbort = null; }  // stop the old stream first

    clear(thread);
    thread.append(spinner("Re-opening chat…"));
    try {
      const resp = await window.api.post("/chat/resume", { chat_id: saved.id });
      session = { id: resp.session_id, label: resp.label };
      clear(thread);
      const messages = resp.messages || [];
      if (!messages.length) {
        thread.append(el(`<div class="empty"><div class="big">✦</div><div>Session ready — ask your first question.</div></div>`));
      } else {
        messages.forEach((m) => appendMessage(thread, m.role, m.content));
      }
      composer.querySelector(".input").disabled = false;
      composer.querySelector("#chatSend").disabled = false;
      composer.querySelector(".input").focus();
      updateSessionBadge(resp.label);
      setChatSetup(false);   // collapse setup → full-page chat
      thread.scrollTop = thread.scrollHeight;
    } catch (e) {
      clear(thread);
      thread.append(el(`<div class="empty"><div class="big">✦</div><div>Couldn't re-open that chat.</div></div>`));
      toast(e.message, "error");
    }
  }

  // ── Update the session label badge ────────────────────────
  function updateSessionBadge(label) {
    const slot = document.getElementById("chatSessionBadge");
    if (!slot) return;
    clear(slot);
    if (label) slot.append(el(`<span class="badge badge-accent">${esc(label)}</span>`));
  }

  // ── Start session ──────────────────────────────────────────
  async function startSession() {
    const thread   = document.querySelector("#chatThread");
    const composer = document.querySelector("#chatComposer");
    const startBtn = document.getElementById("chatStart");
    const srcRow   = document.getElementById("chatSrcRow");
    if (!thread || !composer || !startBtn) return;

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
    clear(thread);
    thread.append(spinner(
      source === "topic"
        ? "Setting up your tutor…"
        : "Preparing knowledge base… a new or updated subject is indexed once — this can take a moment."
    ));

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
      session = { id: resp.session_id, label: resp.label };
      clear(thread);
      thread.append(el(`<div class="empty"><div class="big">✦</div><div>Session ready — ask your first question.</div></div>`));
      composer.querySelector(".input").disabled = false;
      composer.querySelector("#chatSend").disabled = false;
      composer.querySelector(".input").focus();
      updateSessionBadge(resp.label);
      setChatSetup(false);   // collapse setup → full-page chat
      toast("Session started.", "ok");
      // NotebookLM-style empty-state overview for subject/topic sessions only.
      // Best-effort and non-blocking — never gate the chat on it.
      if (!upload) renderOverview(thread, session.id);
      // refresh history list if popover is open
      const hp = document.querySelector('.history-panel[data-kind="chats"]');
      if (hp?.refresh) hp.refresh();
    } catch (e) {
      clear(thread);
      const msg = e && e.message ? e.message : "Couldn't start the session.";
      thread.append(el(`<div class="empty"><div class="big">✦</div><div>${esc(msg)}</div></div>`));
      toast(msg, "error");
    } finally {
      startBtn.disabled = false;
      startBtn.textContent = "Start session";
    }
  }

  // ── Send a message ─────────────────────────────────────────
  async function send() {
    if (!session || generating) return;   // one stream at a time; Stop handles the in-flight one
    const thread   = document.querySelector("#chatThread");
    const composer = document.querySelector("#chatComposer");
    const input    = composer.querySelector(".input");
    const sendBtn  = composer.querySelector("#chatSend");
    const text     = input.value.trim();
    if (!text) return;

    // A leading "/tool" (from the picker or typed) forces that tool; strip it from the
    // message we send (but keep what the user typed in their bubble).
    const { tool: forcedTool, rest } = parseForcedTool(text);
    const outgoing = forcedTool ? (rest || `Make a ${forcedTool}.`) : text;

    const placeholder = thread.querySelector(".empty");
    if (placeholder) placeholder.remove();
    // Remove the session-start overview card (+ any in-flight loading hint) too.
    thread.querySelector(".chat-overview")?.remove();
    thread.querySelector(".chat-overview-loading")?.remove();
    const firstTurn = !thread.querySelector(".msg");

    appendMessage(thread, "user", text);
    input.value = "";
    // NOTE: the input stays ENABLED so the user can type their next question while this
    // one streams. Only the Send button is repurposed (→ Stop) during generation.
    generating = true;
    const controller = new AbortController();
    currentAbort = controller;
    setStopMode(true);
    // A later "+ New"/resume/unmount supersedes this stream by reassigning currentAbort;
    // isCurrent() lets late callbacks no-op instead of writing into another session's thread.
    const isCurrent = () => currentAbort === controller;
    let finished = false;

    const thinking = appendThinking(thread);
    thread.scrollTop = thread.scrollHeight;

    const onFirstTurn = () => {
      if (!firstTurn) return;
      const hp = document.querySelector('.history-panel[data-kind="chats"]');
      if (hp?.refresh) hp.refresh();
      setTimeout(() => { if (hp?.refresh) hp.refresh(); }, 3500);
    };

    const renderDone = (answer, sources, artifacts, steps) => {
      if (!isCurrent()) return;            // superseded by a new session — don't write stale output
      thinking.stop();
      thinking.el.remove();
      appendMessage(thread, "assistant", answer, {
        steps: steps || [],
        sources: sources || [],
      });
      // A single turn can now create several artifacts (e.g. "a quiz AND flashcards").
      // `artifacts` may be a list (new payload) or a single object (legacy) — render each.
      const list = Array.isArray(artifacts) ? artifacts : (artifacts ? [artifacts] : []);
      list.forEach((art) => appendArtifactCard(thread, art));
      onFirstTurn();
    };

    const onError = (e) => {
      thinking.stop();
      thinking.el.remove();
      const errMsg = el(`<div class="msg ai"></div>`);
      errMsg.textContent = "⚠ " + e.message;
      thread.append(errMsg);
      toast(e.message, "error");
      if (/expired/i.test(e.message)) session = null;
    };

    const finish = () => {
      if (finished || !isCurrent()) return;   // run exactly once, and only for the current stream
      finished = true;
      generating = false;
      currentAbort = null;
      setStopMode(false);                 // Stop → Send
      thread.scrollTop = thread.scrollHeight;
      if (session) {
        input.disabled = false;
        sendBtn.disabled = false;
      } else {
        sendBtn.disabled = true;
      }
      // Don't steal focus — the user may already be typing their next question.
    };

    // ── Non-stream fallback (the original path) ──────────────
    const sendNonStream = async () => {
      try {
        const resp = await window.api.post("/chat/message",
          { session_id: session.id, message: outgoing, tool: forcedTool || undefined },
          { signal: controller.signal });
        renderDone(resp.answer, resp.sources, resp.artifacts || resp.artifact, resp.steps);
      } catch (e) {
        if (controller.signal.aborted || e.name === "AbortError") {   // Stop pressed during fallback
          thinking.stop();
          if (isCurrent()) { thinking.el.remove(); thread.append(el(`<div class="msg ai msg-stopped">⏹ Stopped</div>`)); }
        } else {
          onError(e);
        }
      } finally {
        finish();
      }
    };

    // ── Live SSE stream (primary path) ───────────────────────
    //    Streams real reasoning steps into the live panel, then renders the final
    //    answer + artifact on the "done" event. Any failure (no body-stream
    //    support, HTTP error, malformed frame, mid-stream drop) falls back to the
    //    non-stream window.api.post("/chat/message", …) path above.
    if (!(window.ReadableStream && window.TextDecoder && window.fetch)) {
      await sendNonStream();
      return;
    }

    const sid = session.id;
    let collected = [];          // step dicts seen, for the final collapsed trace
    let gotDone = false;
    let started = false;         // any byte processed → past the point of clean fallback
    try {
      const headers = await streamHeaders();
      const res = await fetch(window.AppConfig.apiBase + "/chat/message/stream", {
        method: "POST",
        headers,
        body: JSON.stringify({ session_id: sid, message: outgoing, tool: forcedTool || null }),
        signal: controller.signal,
      });
      if (!res.ok || !res.body) throw new Error("stream_unavailable_" + (res.status || 0));

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";
      // Parse SSE frames separated by a blank line; each frame is "data: <json>".
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
          if (ev.type === "step" && ev.step) {
            collected.push(ev.step);
            if (isCurrent()) thinking.step(ev.step);
          } else if (ev.type === "done") {
            gotDone = true;
            renderDone(ev.answer, ev.sources, ev.artifacts || ev.artifact, collected);
            finish();                                 // back to "Send" the instant the answer is complete
            try { await reader.cancel(); } catch { /* ignore */ }
            return;
          } else if (ev.type === "error") {
            throw new Error(ev.detail || ev.message || "The assistant hit an error.");
          }
        }
      }
      // Reader closed without a 'done' frame.
      if (controller.signal.aborted) {                // Stop fired before any 'done' arrived
        thinking.stop();
        if (isCurrent() && !gotDone) thread.append(el(`<div class="msg ai msg-stopped">⏹ Stopped</div>`));
        finish();
        return;
      }
      if (!started) { await sendNonStream(); return; }
      throw new Error("Connection dropped mid-answer. Please try again.");
    } catch (e) {
      // User pressed Stop → graceful end (abort), not an error.
      if (controller.signal.aborted || e.name === "AbortError") {
        thinking.stop();
        if (isCurrent()) {
          thinking.el.remove();
          if (!gotDone) thread.append(el(`<div class="msg ai msg-stopped">⏹ Stopped</div>`));
        }
        finish();
        return;
      }
      // Clean fallback only if nothing was streamed yet and the session looks alive
      // — otherwise we'd duplicate a partially-rendered answer.
      if (!started && !gotDone && session && !/expired/i.test(e.message || "")) {
        await sendNonStream();
        return;
      }
      if (!gotDone) onError(e);
      finish();
    }
  }

  // ── Scan notes via photo upload (composer 📷 button) ──────────
  //    Independent of the chat session/streaming path. Picks an image, shows a
  //    pending "Scanning…" user bubble, POSTs it to the scan-notes transcribe
  //    endpoint (multipart field `images`, which transcribes AND saves to the
  //    "scans" history), then renders the returned markdown inline as an artifact
  //    card. On error the pending bubble is swapped for a friendly notice.
  async function scanPhoto(file) {
    if (!file) return;
    const thread = document.querySelector("#chatThread");
    if (!thread) return;

    // Clear any empty/overview placeholders so the scan bubble isn't crowded.
    thread.querySelector(".empty")?.remove();
    thread.querySelector(".chat-overview")?.remove();
    thread.querySelector(".chat-overview-loading")?.remove();

    // Pending user bubble: a thumbnail of the chosen photo + a "Scanning…" hint.
    const pending = el(`<div class="msg me chat-scan-pending"></div>`);
    let objectUrl = null;
    try {
      objectUrl = URL.createObjectURL(file);
      const thumb = el(`<img class="chat-scan-thumb" alt="Scanned note" />`);
      thumb.src = objectUrl;
      pending.append(thumb);
    } catch (_) {}
    pending.append(el(`<span class="chat-scan-status">Scanning your notes…</span>`));
    thread.append(pending);
    thread.scrollTop = thread.scrollHeight;

    try {
      const fd = new FormData();
      fd.append("images", file, file.name || "note.jpg");
      const resp = await window.api.postForm("/scannotes/extract", fd);
      if (objectUrl) { try { URL.revokeObjectURL(objectUrl); } catch (_) {} }
      pending.remove();
      const markdown = (resp && resp.markdown) || "";
      appendArtifactCard(thread, {
        kind: "scans",
        id: (resp && resp.id) || null,
        route: "#scannotes",
        tool: "Scan Notes",
        title: (resp && resp.title) || "Scanned notes",
        data: { markdown },
      });
      thread.scrollTop = thread.scrollHeight;
    } catch (e) {
      if (objectUrl) { try { URL.revokeObjectURL(objectUrl); } catch (_) {} }
      pending.remove();
      const errMsg = el(`<div class="msg ai"></div>`);
      errMsg.textContent = "⚠ " + (e && e.message ? e.message : "Couldn't scan that photo.");
      thread.append(errMsg);
      toast(e && e.message ? e.message : "Couldn't scan that photo.", "error");
      thread.scrollTop = thread.scrollHeight;
    }
  }

  // ── Show / hide the collapsible setup panel (Options toggle) ──
  function setChatSetup(open) {
    const setup = document.getElementById("chatSetup");
    const btn = document.getElementById("chatOptionsBtn");
    if (setup) setup.hidden = !open;
    if (btn) {
      btn.setAttribute("aria-expanded", open ? "true" : "false");
      btn.classList.toggle("is-open", !!open);
    }
  }

  // ── Reset to new chat state ────────────────────────────────
  function resetChat() {
    stopGenerating();                          // abort any in-flight stream before tearing down
    generating = false; currentAbort = null;   // (the superseded stream's callbacks will no-op)
    session = null; source = "subject"; pdfFile = null;
    updateSessionBadge("");

    setChatSetup(true);   // reopen setup so a new session can be started

    const thread = document.querySelector("#chatThread");
    if (thread) {
      clear(thread);
      thread.append(el(`<div class="empty"><div class="big">✦</div><div>Start a new session or open a saved chat.</div></div>`));
    }
    const composer = document.querySelector("#chatComposer");
    if (composer) {
      composer.querySelector(".input").disabled = true;
      composer.querySelector("#chatSend").disabled = true;
    }
    // rebuild source row for subject mode
    const srcRow = document.getElementById("chatSrcRow");
    if (srcRow) buildSourceRow(srcRow);
  }

  // ── Mount ──────────────────────────────────────────────────
  window.Views = window.Views || {};
  window.Views.chat = {
    accent: "chat",
    title: "Apex Agent",

    async mount(root) {
      root.append(el(`
        <div class="chat-shell stack">

          <!-- Slim always-on bar: the Options toggle reveals/hides the setup below,
               so the chat thread can take the full page (ChatGPT-style). -->
          <div class="chat-minibar panel">
            <button id="chatOptionsBtn" class="btn" type="button" aria-expanded="true" aria-controls="chatSetup" title="Show or hide chat setup" style="padding:.42rem .8rem;font-size:.85rem">⚙ Options</button>
            <span id="chatSessionBadge"></span>
            <span style="flex:1"></span>
            <button id="chatHistBtn" class="btn" type="button" title="Chat history" style="padding:.42rem .8rem;font-size:.85rem">≡ History</button>
            <button id="chatNewBtn" class="btn-accent" type="button" style="padding:.42rem .85rem;font-size:.85rem">+ New</button>
          </div>

          <!-- Collapsible setup: source selection + model + start. -->
          <div id="chatSetup" class="chat-src-bar panel">
            <div class="row" style="gap:.75rem;flex-wrap:wrap;align-items:center">
              <div class="seg" role="tablist" id="chatSrcSeg">
                <button type="button" data-src="subject" aria-pressed="true">Subject</button>
                <button type="button" data-src="topic"   aria-pressed="false">Topic</button>
                <button type="button" data-src="upload"  aria-pressed="false">Upload PDF</button>
              </div>
              <div id="chatSrcRow" class="row" style="flex:1;gap:.6rem"></div>
              <div id="chatModelCtl"></div>
              <button id="chatStart" class="btn-accent" type="button">Start session</button>
            </div>
          </div>

          <!-- Chat area -->
          <div class="panel" style="flex:1;min-height:0;display:flex;flex-direction:column;padding:0;overflow:hidden">
            <div class="thread" id="chatThread">
              <div class="empty"><div class="big">✦</div><div>Start a new session or open a saved chat.</div></div>
            </div>
            <div class="composer" id="chatComposer">
              <button id="chatScanBtn" class="chat-scan-btn" type="button" title="Scan handwritten notes from a photo" aria-label="Scan notes from a photo">📷</button>
              <input id="chatScanInput" type="file" accept="image/*" hidden />
              <button id="chatToolsBtn" class="chat-tools-btn" type="button" title="Tools — run a tool on demand" aria-label="Choose a tool" aria-haspopup="true" aria-expanded="false">☰</button>
              <div id="chatToolsMenu" class="chat-tools-menu" role="menu" hidden></div>
              <input class="input" type="text" placeholder="Ask a question…" aria-label="Message" disabled />
              <button id="chatSend" class="btn-accent" type="button" disabled>Send</button>
            </div>
          </div>

        </div>`));

      // Keep the composer pinned on-screen across devices and keyboard states.
      installShellSizing(root.querySelector(".chat-shell"));

      // Wire model selector
      document.getElementById("chatModelCtl").append(window.ui.modelSelect());

      // Wire source segmented control
      const seg = document.getElementById("chatSrcSeg");
      const srcRow = document.getElementById("chatSrcRow");
      buildSourceRow(srcRow);
      seg.addEventListener("click", (e) => {
        const btn = e.target.closest("button[data-src]");
        if (!btn) return;
        source = btn.dataset.src;
        pdfFile = null;
        seg.querySelectorAll("button").forEach((b) =>
          b.setAttribute("aria-pressed", b === btn ? "true" : "false"));
        buildSourceRow(srcRow);
      });

      // Wire buttons
      document.getElementById("chatStart").addEventListener("click", startSession);
      document.getElementById("chatNewBtn").addEventListener("click", resetChat);
      document.getElementById("chatHistBtn").addEventListener("click", (e) =>
        openHistoryDrawer(e.currentTarget));
      document.getElementById("chatOptionsBtn").addEventListener("click", () =>
        setChatSetup(document.getElementById("chatSetup")?.hidden));

      // Wire composer. The Send button doubles as Stop while a response streams.
      const composer = document.getElementById("chatComposer");
      composer.querySelector("#chatSend").addEventListener("click", () => {
        if (generating) stopGenerating(); else send();
      });
      const inputEl = composer.querySelector(".input");
      inputEl.addEventListener("keydown", (e) => {
        if (e.key === "Enter") { e.preventDefault(); if (!generating) send(); }
      });
      // Typing "/" on an empty composer opens the tool menu (slash-command affordance).
      inputEl.addEventListener("input", () => { if (inputEl.value === "/") openToolsMenu(); });

      // Manual tool picker: list every tool; picking one prefixes "/key " so the next
      // message forces that tool (ChatGPT-plugin style).
      const toolsBtn = composer.querySelector("#chatToolsBtn");
      const toolsMenu = composer.querySelector("#chatToolsMenu");
      TOOLS.forEach((t) => {
        const item = el(`<button class="chat-tool-item" type="button" role="menuitem" data-tool="${t.key}"></button>`);
        item.innerHTML = `<span class="chat-tool-ic" aria-hidden="true">${t.icon}</span><span class="chat-tool-label">${esc(t.label)}</span><span class="chat-tool-slash">/${t.key}</span>`;
        item.addEventListener("click", () => pickTool(t.key));
        toolsMenu.append(item);
      });
      toolsBtn.addEventListener("click", (e) => { e.stopPropagation(); toggleToolsMenu(); });
      const onDocClick = (e) => {
        if (!toolsMenu.hidden && !toolsMenu.contains(e.target) && e.target !== toolsBtn) closeToolsMenu();
      };
      const onDocKey = (e) => { if (e.key === "Escape") closeToolsMenu(); };
      if (toolsMenuCleanup) toolsMenuCleanup();   // self-heal: drop any prior listeners on re-mount
      document.addEventListener("click", onDocClick);
      document.addEventListener("keydown", onDocKey);
      toolsMenuCleanup = () => {
        document.removeEventListener("click", onDocClick);
        document.removeEventListener("keydown", onDocKey);
      };

      // Wire 📷 scan-notes upload (independent of the chat session/streaming path).
      const scanInput = composer.querySelector("#chatScanInput");
      composer.querySelector("#chatScanBtn").addEventListener("click", () => scanInput.click());
      scanInput.addEventListener("change", () => {
        const file = scanInput.files && scanInput.files[0];
        scanInput.value = "";  // allow re-picking the same file
        if (file) scanPhoto(file);
      });
    },

    unmount() {
      stopGenerating();                 // abort any in-flight stream
      generating = false; currentAbort = null;
      session = null; source = "subject"; pdfFile = null;
      if (teardownLayout) teardownLayout();
      if (toolsMenuCleanup) { toolsMenuCleanup(); toolsMenuCleanup = null; }
      document.getElementById("histPopover")?.remove();
      document.getElementById("histPopoverBg")?.remove();
      document.body.classList.remove("hist-scroll-lock");
    },
  };
})();
