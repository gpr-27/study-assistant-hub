/* ============================================================
   Autonomous Agent — clone of the Chat Assistant shell, but it
   orchestrates the app's tools. Ask in plain English and it
   builds quizzes, cheat sheets, flashcards, plans, mind maps,
   summaries and exam-question predictions, files each artifact
   in that tool's own history, and links back to it via an
   artifact card. Sessions start with autonomous:true.
   ============================================================ */
(() => {
  "use strict";
  const { el, esc, clear, toast, md, spinner, copy } = window.ui;

  let session = null;       // { id, label }
  let source  = "subject";  // "subject" | "topic"
  let teardownLayout = null; // removes the responsive shell-sizing listeners on unmount

  // Example prompts shown in the empty state — clicking one prefills the
  // composer and sends, so a fresh user sees what the agent can do.
  const EXAMPLE_PROMPTS = [
    "Create a cheat sheet for this subject",
    "Make a 10-question quiz",
    "Generate 15 flashcards",
    "Build a mind map of this subject",
    "Make a 2-week study plan",
    "Predict likely exam questions",
  ];

  // ── Responsive height: size the agent shell to the space actually left below
  //    the nav + view header so the composer always stays pinned on-screen — on
  //    any phone or tablet, and above the on-screen keyboard. CSS offsets can
  //    only guess the chrome height; measuring it is exact.
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
  //    Autonomous sessions are still chats, persisted under the same "chats" kind.
  function openHistoryDrawer(anchor) {
    window.ui.openHistoryDrawer("chats", {
      title: "Recent sessions",
      anchor,
      onOpen: (saved) => resumeSession(saved),
    });
  }

  // ── Source control (inline compact) ───────────────────────
  function buildSourceRow(host) {
    clear(host);
    host.dataset.src = source;

    if (source === "subject") {
      host.append(el(`
        <select id="agentSubject" class="select" style="max-width:220px">
          <option value="">Loading…</option>
        </select>`));
      const sel = host.querySelector("#agentSubject");
      window.api.get("/subjects").then((d) => {
        const subjects = d.subjects || [];
        clear(sel);
        if (!subjects.length) { sel.append(el(`<option value="">No subjects</option>`)); return; }
        subjects.forEach((s) => sel.append(el(`<option value="${esc(s)}">${esc(s)}</option>`)));
        sel.value = subjects[0];
      }).catch(() => {
        clear(sel); sel.append(el(`<option value="">Error loading</option>`));
      });
    } else {
      host.append(el(`
        <input id="agentTopic" class="input" type="text" placeholder="e.g. Binary search trees"
          autocomplete="off" style="max-width:260px" />`));
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

  // ── Artifact card: the agent created something in another tool ──
  function appendArtifactCard(thread, art) {
    const card = el(`<a class="chat-artifact" href="${esc(art.route || "#home")}"></a>`);
    card.append(el(`<span class="chat-artifact-icon" aria-hidden="true">✦</span>`));
    const meta = el(`<span class="chat-artifact-meta"></span>`);
    const t = el(`<span class="chat-artifact-title"></span>`);
    t.textContent = art.title || "New artifact";
    const s = el(`<span class="chat-artifact-sub"></span>`);
    s.textContent = "Open in " + (art.tool || "the tool") + " →";
    meta.append(t, s);
    card.append(meta);
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
    "Reading your request…",
    "Picking the right tool…",
    "Gathering subject content…",
    "Building your artifact…",
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

  // ── Empty state: short intro + clickable example-prompt chips ──
  function buildEmptyState() {
    const empty = el(`<div class="empty"></div>`);
    empty.append(el(`<div class="big">✦</div>`));
    empty.append(el(`<div>Ask in plain English — the agent builds the artifact and files it in the right tool for you.</div>`));
    const chips = el(`<div class="agent-prompt-chips"></div>`);
    EXAMPLE_PROMPTS.forEach((p) => {
      const chip = el(`<button class="agent-prompt-chip" type="button"></button>`);
      chip.textContent = p;
      chip.addEventListener("click", () => prefillAndSend(p));
      chips.append(chip);
    });
    empty.append(chips);
    return empty;
  }

  // Drop an example prompt into the composer and fire it.
  function prefillAndSend(text) {
    if (!session) { toast("Start a session first.", "error"); return; }
    const composer = document.querySelector("#agentComposer");
    const input = composer?.querySelector(".input");
    if (!input || input.disabled) return;
    input.value = text;
    send();
  }

  // ── Resume saved session ───────────────────────────────────
  async function resumeSession(saved) {
    const thread = document.querySelector("#agentThread");
    const composer = document.querySelector("#agentComposer");
    if (!thread || !composer) return;

    clear(thread);
    thread.append(spinner("Re-opening session…"));
    try {
      const resp = await window.api.post("/chat/resume", { chat_id: saved.id });
      session = { id: resp.session_id, label: resp.label };
      clear(thread);
      const messages = resp.messages || [];
      if (!messages.length) {
        thread.append(buildEmptyState());
      } else {
        messages.forEach((m) => appendMessage(thread, m.role, m.content));
      }
      composer.querySelector(".input").disabled = false;
      composer.querySelector("#agentSend").disabled = false;
      composer.querySelector(".input").focus();
      updateSessionBadge(resp.label);
      setAgentSetup(false);   // collapse setup → full-page agent
      thread.scrollTop = thread.scrollHeight;
    } catch (e) {
      clear(thread);
      thread.append(el(`<div class="empty"><div class="big">✦</div><div>Couldn't re-open that session.</div></div>`));
      toast(e.message, "error");
    }
  }

  // ── Update the session label badge ────────────────────────
  function updateSessionBadge(label) {
    const slot = document.getElementById("agentSessionBadge");
    if (!slot) return;
    clear(slot);
    if (label) slot.append(el(`<span class="badge badge-accent">${esc(label)}</span>`));
  }

  // ── Start session ──────────────────────────────────────────
  async function startSession() {
    const thread   = document.querySelector("#agentThread");
    const composer = document.querySelector("#agentComposer");
    const startBtn = document.getElementById("agentStart");
    const srcRow   = document.getElementById("agentSrcRow");
    if (!thread || !composer || !startBtn) return;

    let payload;
    if (source === "subject") {
      const subject = srcRow.querySelector("#agentSubject")?.value;
      if (!subject) { toast("Pick a subject first.", "error"); return; }
      payload = { mode: "subject", subject, model: window.ui.selectedModel(srcRow), autonomous: true };
    } else {
      const topic = srcRow.querySelector("#agentTopic")?.value.trim();
      if (!topic) { toast("Enter a topic first.", "error"); return; }
      payload = { mode: "topic", topic, model: window.ui.selectedModel(srcRow), autonomous: true };
    }

    startBtn.disabled = true;
    startBtn.textContent = "Starting…";
    clear(thread);
    thread.append(spinner(
      source === "topic"
        ? "Setting up your agent…"
        : "Preparing knowledge base… a new or updated subject is indexed once — this can take a moment."
    ));

    try {
      const resp = await window.api.post("/chat/start", payload);
      session = { id: resp.session_id, label: resp.label };
      clear(thread);
      thread.append(buildEmptyState());
      composer.querySelector(".input").disabled = false;
      composer.querySelector("#agentSend").disabled = false;
      composer.querySelector(".input").focus();
      updateSessionBadge(resp.label);
      setAgentSetup(false);   // collapse setup → full-page agent
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
    const thread   = document.querySelector("#agentThread");
    const composer = document.querySelector("#agentComposer");
    const input    = composer.querySelector(".input");
    const sendBtn  = composer.querySelector("#agentSend");
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
      if (resp.artifact) appendArtifactCard(thread, resp.artifact);
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
  function setAgentSetup(open) {
    const setup = document.getElementById("agentSetup");
    const btn = document.getElementById("agentOptionsBtn");
    if (setup) setup.hidden = !open;
    if (btn) {
      btn.setAttribute("aria-expanded", open ? "true" : "false");
      btn.classList.toggle("is-open", !!open);
    }
  }

  // ── Reset to new session state ─────────────────────────────
  function resetSession() {
    session = null; source = "subject";
    updateSessionBadge("");

    setAgentSetup(true);   // reopen setup so a new session can be started

    const thread = document.querySelector("#agentThread");
    if (thread) {
      clear(thread);
      thread.append(el(`<div class="empty"><div class="big">✦</div><div>Start a new session or open a saved one.</div></div>`));
    }
    const composer = document.querySelector("#agentComposer");
    if (composer) {
      composer.querySelector(".input").disabled = true;
      composer.querySelector("#agentSend").disabled = true;
    }
    // rebuild source row for subject mode
    const srcRow = document.getElementById("agentSrcRow");
    if (srcRow) buildSourceRow(srcRow);
  }

  // ── Mount ──────────────────────────────────────────────────
  window.Views = window.Views || {};
  window.Views.agent = {
    accent: "agent",
    title: "Autonomous Agent",

    async mount(root) {
      root.append(el(`
        <div class="chat-shell stack">

          <!-- Slim always-on bar: the Options toggle reveals/hides the setup below,
               so the agent thread can take the full page (ChatGPT-style). -->
          <div class="chat-minibar panel">
            <button id="agentOptionsBtn" class="btn" type="button" aria-expanded="true" aria-controls="agentSetup" title="Show or hide session setup" style="padding:.42rem .8rem;font-size:.85rem">⚙ Options</button>
            <span id="agentSessionBadge"></span>
            <span style="flex:1"></span>
            <button id="agentHistBtn" class="btn" type="button" title="Session history" style="padding:.42rem .8rem;font-size:.85rem">≡ History</button>
            <button id="agentNewBtn" class="btn-accent" type="button" style="padding:.42rem .85rem;font-size:.85rem">+ New</button>
          </div>

          <!-- Collapsible setup: source selection + model + start. -->
          <div id="agentSetup" class="chat-src-bar panel">
            <div class="row" style="gap:.75rem;flex-wrap:wrap;align-items:center">
              <div class="seg" role="tablist" id="agentSrcSeg">
                <button type="button" data-src="subject" aria-pressed="true">Subject</button>
                <button type="button" data-src="topic"   aria-pressed="false">Topic</button>
              </div>
              <div id="agentSrcRow" class="row" style="flex:1;gap:.6rem"></div>
              <div id="agentModelCtl"></div>
              <button id="agentStart" class="btn-accent" type="button">Start session</button>
            </div>
          </div>

          <!-- Agent area -->
          <div class="panel" style="flex:1;display:flex;flex-direction:column;padding:0;overflow:hidden">
            <div class="thread" id="agentThread">
              <div class="empty"><div class="big">✦</div><div>Start a new session or open a saved one.</div></div>
            </div>
            <div class="composer" id="agentComposer">
              <input class="input" type="text" placeholder="Ask the agent to build something…" aria-label="Message" disabled />
              <button id="agentSend" class="btn-accent" type="button" disabled>Send</button>
            </div>
          </div>

        </div>`));

      // Keep the composer pinned on-screen across devices and keyboard states.
      installShellSizing(root.querySelector(".chat-shell"));

      // Wire model selector
      document.getElementById("agentModelCtl").append(window.ui.modelSelect());

      // Wire source segmented control
      const seg = document.getElementById("agentSrcSeg");
      const srcRow = document.getElementById("agentSrcRow");
      buildSourceRow(srcRow);
      seg.addEventListener("click", (e) => {
        const btn = e.target.closest("button[data-src]");
        if (!btn) return;
        source = btn.dataset.src;
        seg.querySelectorAll("button").forEach((b) =>
          b.setAttribute("aria-pressed", b === btn ? "true" : "false"));
        buildSourceRow(srcRow);
      });

      // Wire buttons
      document.getElementById("agentStart").addEventListener("click", startSession);
      document.getElementById("agentNewBtn").addEventListener("click", resetSession);
      document.getElementById("agentHistBtn").addEventListener("click", (e) =>
        openHistoryDrawer(e.currentTarget));
      document.getElementById("agentOptionsBtn").addEventListener("click", () =>
        setAgentSetup(document.getElementById("agentSetup")?.hidden));

      // Wire composer
      const composer = document.getElementById("agentComposer");
      composer.querySelector("#agentSend").addEventListener("click", send);
      composer.querySelector(".input").addEventListener("keydown", (e) => {
        if (e.key === "Enter") { e.preventDefault(); send(); }
      });
    },

    unmount() {
      session = null; source = "subject";
      if (teardownLayout) teardownLayout();
      document.getElementById("histPopover")?.remove();
      document.getElementById("histPopoverBg")?.remove();
      document.body.classList.remove("hist-scroll-lock");
    },
  };
})();
