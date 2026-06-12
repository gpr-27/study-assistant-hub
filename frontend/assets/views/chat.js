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

  function promptDisclosure(text) {
    if (!text) return null;
    const wrap = el(`<div class="trace-prompt-wrap"></div>`);
    const btn = el(`<button class="trace-prompt-btn" type="button">⌗ View prompt</button>`);
    const pre = el(`<pre class="trace-prompt" hidden></pre>`);
    pre.textContent = text;
    btn.addEventListener("click", () => {
      pre.hidden = !pre.hidden;
      btn.textContent = (pre.hidden ? "⌗ View prompt" : "⌗ Hide prompt");
      btn.classList.toggle("is-open", !pre.hidden);
    });
    wrap.append(btn, pre);
    return wrap;
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
    const pd = promptDisclosure(step.prompt);
    if (pd) d.append(pd);
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

  // ── Thinking animation while waiting ──────────────────────
  const THINKING_STEPS = [
    "Rewriting question into search queries…",
    "Retrieving relevant passages…",
    "Grading context quality…",
    "Generating answer…",
  ];

  function appendThinking(thread) {
    const wrap = el(`<div class="msg ai msg-thinking"></div>`);
    const stepEl = el(`<span class="thinking-label"></span>`);
    const dots = el(`<span class="typing"><i></i><i></i><i></i></span>`);
    wrap.append(stepEl, dots);
    thread.append(wrap);

    let idx = 0;
    stepEl.textContent = THINKING_STEPS[0];
    const timer = setInterval(() => {
      idx = (idx + 1) % THINKING_STEPS.length;
      stepEl.textContent = THINKING_STEPS[idx];
    }, 1800);

    return { el: wrap, stop: () => clearInterval(timer) };
  }

  // ── Resume saved chat ──────────────────────────────────────
  async function resumeChat(saved) {
    const thread = document.querySelector("#chatThread");
    const composer = document.querySelector("#chatComposer");
    if (!thread || !composer) return;

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
    if (!session) return;
    const thread   = document.querySelector("#chatThread");
    const composer = document.querySelector("#chatComposer");
    const input    = composer.querySelector(".input");
    const sendBtn  = composer.querySelector("#chatSend");
    const text     = input.value.trim();
    if (!text) return;

    const placeholder = thread.querySelector(".empty");
    if (placeholder) placeholder.remove();
    const firstTurn = !thread.querySelector(".msg");

    appendMessage(thread, "user", text);
    input.value = "";
    input.disabled = true;
    sendBtn.disabled = true;

    const thinking = appendThinking(thread);
    thread.scrollTop = thread.scrollHeight;

    try {
      const resp = await window.api.post("/chat/message", { session_id: session.id, message: text });
      thinking.stop();
      thinking.el.remove();
      appendMessage(thread, "assistant", resp.answer, {
        steps: resp.steps || [],
        sources: resp.sources || [],
      });
      if (firstTurn) {
        const hp = document.querySelector('.history-panel[data-kind="chats"]');
        if (hp?.refresh) hp.refresh();
        setTimeout(() => { if (hp?.refresh) hp.refresh(); }, 3500);
      }
    } catch (e) {
      thinking.stop();
      thinking.el.remove();
      const errMsg = el(`<div class="msg ai"></div>`);
      errMsg.textContent = "⚠ " + e.message;
      thread.append(errMsg);
      toast(e.message, "error");
      if (/expired/i.test(e.message)) session = null;
    } finally {
      thread.scrollTop = thread.scrollHeight;
      if (session) {
        input.disabled = false;
        sendBtn.disabled = false;
        input.focus();
      }
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
    title: "Chat Assistant",

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
          <div class="panel" style="flex:1;display:flex;flex-direction:column;padding:0;overflow:hidden">
            <div class="thread" id="chatThread">
              <div class="empty"><div class="big">✦</div><div>Start a new session or open a saved chat.</div></div>
            </div>
            <div class="composer" id="chatComposer">
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

      // Wire composer
      const composer = document.getElementById("chatComposer");
      composer.querySelector("#chatSend").addEventListener("click", send);
      composer.querySelector(".input").addEventListener("keydown", (e) => {
        if (e.key === "Enter") { e.preventDefault(); send(); }
      });
    },

    unmount() {
      session = null; source = "subject"; pdfFile = null;
      if (teardownLayout) teardownLayout();
      document.getElementById("histPopover")?.remove();
      document.getElementById("histPopoverBg")?.remove();
      document.body.classList.remove("hist-scroll-lock");
    },
  };
})();
