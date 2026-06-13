/* ============================================================
   Socratic Teach-back Tutor — window.Views.teachback
   You explain a concept in your own words; a Feynman-technique
   tutor probes with one question at a time, then scores you.
   Chat-like UI copying views/chat.js (thread + composer + minibar)
   but simpler. Uses window.api, window.ui and the existing CSS kit.
   ============================================================ */
(() => {
  "use strict";
  const { el, esc, clear, toast, md, spinner } = window.ui;

  let session = null;        // { id, concept }
  let source = "topic";      // "topic" | "subject" | "upload"
  let pdfFile = null;        // upload mode: the chosen PDF File
  let teardownLayout = null; // removes resize listeners on unmount

  // ── Responsive height (same approach as chat.js: document-relative offset,
  //    react to resize only, never to scroll) ─────────────────
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

  // ── Source row: Topic input OR Subject + PDF selects ──────
  function buildSourceRow(host) {
    clear(host);
    pdfFile = null;
    if (source === "topic") {
      host.append(el(`
        <input id="tbTopic" class="input" type="text"
          placeholder="e.g. TCP congestion control" autocomplete="off" style="max-width:280px" />`));
      return;
    }
    if (source === "upload") {
      const wrap = el(`
        <div class="row" style="gap:.5rem;align-items:center;flex-wrap:wrap">
          <label class="btn" for="tbFile" style="cursor:pointer">Choose PDF…</label>
          <span class="muted" id="tbFileName">No file selected</span>
          <input id="tbFile" type="file" accept="application/pdf" hidden />
        </div>`);
      host.append(wrap);
      const input = wrap.querySelector("#tbFile");
      const nameSlot = wrap.querySelector("#tbFileName");
      input.addEventListener("change", () => {
        pdfFile = input.files && input.files[0] ? input.files[0] : null;
        nameSlot.textContent = pdfFile ? pdfFile.name : "No file selected";
      });
      return;
    }
    host.append(el(`<select id="tbSubject" class="select" style="max-width:200px"><option value="">Loading…</option></select>`));
    host.append(el(`<select id="tbPdf" class="select" style="max-width:220px" disabled><option value="">Pick a subject first</option></select>`));
    const subjectSel = host.querySelector("#tbSubject");
    const pdfSel = host.querySelector("#tbPdf");

    async function loadPdfs(subject) {
      pdfSel.disabled = true;
      clear(pdfSel);
      pdfSel.append(el(`<option value="">Loading documents…</option>`));
      try {
        const data = await window.api.get(`/subjects/${encodeURIComponent(subject)}/pdfs`);
        const pdfs = data.pdfs || [];
        clear(pdfSel);
        if (!pdfs.length) { pdfSel.append(el(`<option value="">No PDFs in this subject</option>`)); return; }
        pdfs.forEach((p) => pdfSel.append(el(`<option value="${esc(p.name)}">${esc(p.name)}</option>`)));
        pdfSel.disabled = false;
      } catch (e) {
        clear(pdfSel);
        pdfSel.append(el(`<option value="">Couldn’t load documents</option>`));
        toast(e.message, "error");
      }
    }

    window.api.get("/subjects").then(async (d) => {
      const subjects = d.subjects || [];
      clear(subjectSel);
      if (!subjects.length) { subjectSel.append(el(`<option value="">No subjects</option>`)); return; }
      subjects.forEach((s) => subjectSel.append(el(`<option value="${esc(s)}">${esc(s)}</option>`)));
      subjectSel.value = subjects[0];
      subjectSel.addEventListener("change", () => loadPdfs(subjectSel.value));
      await loadPdfs(subjectSel.value);
    }).catch(() => {
      clear(subjectSel);
      subjectSel.append(el(`<option value="">Error loading</option>`));
    });
  }

  // ── Messages ───────────────────────────────────────────────
  function appendMessage(thread, role, content) {
    if (role === "user") {
      const me = el(`<div class="msg me"></div>`);
      me.textContent = content;
      thread.append(me);
    } else {
      const ai = el(`<div class="msg ai"></div>`);
      ai.append(md(content));
      thread.append(ai);
    }
    thread.scrollTop = thread.scrollHeight;
  }

  function appendThinking(thread) {
    const wrap = el(`<div class="msg ai msg-thinking"><span>Thinking it over</span><span class="typing"><i></i><i></i><i></i></span></div>`);
    thread.append(wrap);
    thread.scrollTop = thread.scrollHeight;
    return wrap;
  }

  // ── Evaluation card ────────────────────────────────────────
  function listBlock(title, items) {
    if (!items || !items.length) return null;
    const block = el(`<div class="stack-sm"><span class="label">${esc(title)}</span></div>`);
    const ul = el(`<ul style="margin:0;padding-left:1.2rem"></ul>`);
    items.forEach((s) => { const li = document.createElement("li"); li.textContent = s; ul.append(li); });
    block.append(ul);
    return block;
  }

  function evaluationCard(ev, opts) {
    opts = opts || {};
    const score = Math.max(0, Math.min(10, Number(ev.understanding_score_0_to_10) || 0));
    const card = el(`
      <div class="panel-soft stack" style="align-self:stretch;max-width:none">
        <div class="between">
          <strong style="font-family:var(--font-display);letter-spacing:-.01em">Your evaluation</strong>
          <span class="badge badge-accent">${score} / 10</span>
        </div>
      </div>`);
    const strengths = listBlock("Strengths", ev.strengths);
    if (strengths) card.append(strengths);
    const gaps = listBlock("Gaps", ev.gaps);
    if (gaps) card.append(gaps);
    if (ev.suggested_review && ev.suggested_review.length) {
      const block = el(`<div class="stack-sm"><span class="label">Suggested review</span></div>`);
      const chips = el(`<div class="row row-wrap" style="gap:.35rem"></div>`);
      ev.suggested_review.forEach((s) => {
        const chip = el(`<span class="badge"></span>`);
        chip.textContent = s;
        chips.append(chip);
      });
      block.append(chips);
      card.append(block);
    }
    if (!opts.readonly) {
      const again = el(`<button class="btn-accent btn-block" type="button">Start another</button>`);
      again.addEventListener("click", resetView);
      card.append(again);
    }
    return card;
  }

  // ── UI state helpers ───────────────────────────────────────
  function setSetup(open) {
    const setup = document.getElementById("tbSetup");
    const btn = document.getElementById("tbOptionsBtn");
    if (setup) setup.hidden = !open;
    if (btn) btn.setAttribute("aria-expanded", open ? "true" : "false");
  }

  function setComposer(enabled) {
    const composer = document.getElementById("tbComposer");
    if (!composer) return;
    composer.querySelector(".input").disabled = !enabled;
    composer.querySelector("#tbSend").disabled = !enabled;
    composer.querySelector("#tbEval").disabled = !enabled;
  }

  function setBadge(label) {
    const slot = document.getElementById("tbBadge");
    if (!slot) return;
    clear(slot);
    if (label) slot.append(el(`<span class="badge badge-accent">${esc(label)}</span>`));
  }

  function resetView() {
    session = null;
    setBadge("");
    setSetup(true);
    setComposer(false);
    const thread = document.getElementById("tbThread");
    if (thread) {
      clear(thread);
      thread.append(el(`<div class="empty"><div class="big">🗣</div><div>Pick a topic, a subject PDF, or upload a PDF, then teach the concept back in your own words.</div></div>`));
    }
  }

  // ── Start a session ────────────────────────────────────────
  async function startSession() {
    const thread = document.getElementById("tbThread");
    const startBtn = document.getElementById("tbStart");
    const srcRow = document.getElementById("tbSrcRow");
    const setup = document.getElementById("tbSetup");
    if (!thread || !startBtn || !srcRow) return;

    const model = window.ui.selectedModel(setup);
    let request;
    if (source === "topic") {
      const topic = srcRow.querySelector("#tbTopic")?.value.trim();
      if (!topic) { toast("Enter a topic first.", "error"); return; }
      request = () => window.api.post("/teachback/start", { mode: "topic", topic, model });
    } else if (source === "subject") {
      const subject = srcRow.querySelector("#tbSubject")?.value;
      const pdf = srcRow.querySelector("#tbPdf")?.value;
      if (!subject || !pdf) { toast("Pick a subject and a document first.", "error"); return; }
      request = () => window.api.post("/teachback/start", { mode: "subject", subject, pdf, model });
    } else {
      if (!pdfFile) { toast("Choose a PDF first.", "error"); return; }
      const fd = new FormData();
      fd.append("file", pdfFile);
      if (model) fd.append("model", model);
      request = () => window.api.postForm("/teachback/start-upload", fd);
    }

    startBtn.disabled = true;
    startBtn.textContent = "Starting…";
    clear(thread);
    thread.append(spinner("Choosing a concept for you to teach back…"));
    try {
      const resp = await request();
      session = { id: resp.session_id, concept: resp.concept };
      clear(thread);
      appendMessage(thread, "assistant", resp.opening);
      setBadge("Teach-back · " + resp.concept);
      setComposer(true);
      setSetup(false);
      document.querySelector("#tbComposer .input")?.focus();
    } catch (e) {
      clear(thread);
      thread.append(el(`<div class="empty"><div class="big">⚠</div><div>${esc(e.message || "Couldn’t start the session.")}</div></div>`));
      toast(e.message, "error");
    } finally {
      startBtn.disabled = false;
      startBtn.textContent = "Start";
    }
  }

  // ── Send a student turn ────────────────────────────────────
  async function send(textOverride) {
    if (!session) return;
    const thread = document.getElementById("tbThread");
    const composer = document.getElementById("tbComposer");
    const input = composer.querySelector(".input");
    const text = (textOverride != null ? textOverride : input.value).trim();
    if (!text) return;

    appendMessage(thread, "user", text);
    input.value = "";
    setComposer(false);
    const thinking = appendThinking(thread);

    try {
      const resp = await window.api.post("/teachback/message", { session_id: session.id, message: text });
      thinking.remove();
      appendMessage(thread, "assistant", resp.reply);
      if (resp.done && resp.evaluation) {
        thread.append(evaluationCard(resp.evaluation));
        thread.scrollTop = thread.scrollHeight;
        session = null;
        const hp = document.querySelector('.history-panel[data-kind="teachbacks"]');
        if (hp?.refresh) hp.refresh();
      } else {
        setComposer(true);
        input.focus();
      }
    } catch (e) {
      thinking.remove();
      const errMsg = el(`<div class="msg ai"></div>`);
      errMsg.textContent = "⚠ " + e.message;
      thread.append(errMsg);
      toast(e.message, "error");
      if (e.status === 403 || e.status === 404) { session = null; setSetup(true); }
      else setComposer(true);
    }
  }

  // ── Re-open a saved session (read-only) ────────────────────
  function openSaved(saved) {
    const thread = document.getElementById("tbThread");
    if (!thread) return;
    session = null;
    setComposer(false);
    setSetup(false);
    setBadge(saved.title || "Teach-back");
    clear(thread);
    (saved.messages || []).forEach((m) => appendMessage(thread, m.role, m.content || ""));
    if (saved.evaluation) thread.append(evaluationCard(saved.evaluation, { readonly: true }));
    const again = el(`<button class="btn btn-block" type="button" style="align-self:stretch">Start another</button>`);
    again.addEventListener("click", resetView);
    thread.append(again);
    thread.scrollTop = 0;
  }

  // ── Mount ──────────────────────────────────────────────────
  window.Views = window.Views || {};
  window.Views.teachback = {
    accent: "chat",
    title: "Teach-back Tutor",

    async mount(root) {
      root.append(el(`
        <div class="chat-shell stack">

          <div class="chat-minibar panel">
            <button id="tbOptionsBtn" class="btn" type="button" aria-expanded="true" aria-controls="tbSetup" title="Show or hide setup" style="padding:.42rem .8rem;font-size:.85rem">⚙ Options</button>
            <span id="tbBadge"></span>
            <span style="flex:1"></span>
            <button id="tbHistBtn" class="btn" type="button" title="Past teach-backs" style="padding:.42rem .8rem;font-size:.85rem">≡ History</button>
            <button id="tbNewBtn" class="btn-accent" type="button" style="padding:.42rem .85rem;font-size:.85rem">+ New</button>
          </div>

          <div id="tbSetup" class="chat-src-bar panel">
            <div class="row" style="gap:.75rem;flex-wrap:wrap;align-items:center">
              <div class="seg" role="tablist" id="tbSrcSeg" aria-label="Teach-back source">
                <button type="button" data-src="topic" aria-pressed="true">Topic</button>
                <button type="button" data-src="subject" aria-pressed="false">Subject</button>
                <button type="button" data-src="upload" aria-pressed="false">Upload PDF</button>
              </div>
              <div id="tbSrcRow" class="row" style="flex:1;gap:.6rem;flex-wrap:wrap"></div>
              <div id="tbModelCtl"></div>
              <button id="tbStart" class="btn-accent" type="button">Start</button>
            </div>
            <p class="hint" style="margin:.5rem 0 0">The tutor picks a concept — you explain it in your own words. It only asks questions; say “done” (or hit Evaluate me now) for your score.</p>
          </div>

          <div class="panel" style="flex:1;display:flex;flex-direction:column;padding:0;overflow:hidden">
            <div class="thread" id="tbThread">
              <div class="empty"><div class="big">🗣</div><div>Pick a topic, a subject PDF, or upload a PDF, then teach the concept back in your own words.</div></div>
            </div>
            <div class="composer" id="tbComposer">
              <input class="input" type="text" placeholder="Explain it in your own words…" aria-label="Your explanation" disabled />
              <button id="tbEval" class="btn" type="button" disabled title="Finish and get your evaluation">Evaluate me now</button>
              <button id="tbSend" class="btn-accent" type="button" disabled>Send</button>
            </div>
          </div>

        </div>`));

      installShellSizing(root.querySelector(".chat-shell"));
      document.getElementById("tbModelCtl").append(window.ui.modelSelect());

      const seg = document.getElementById("tbSrcSeg");
      const srcRow = document.getElementById("tbSrcRow");
      buildSourceRow(srcRow);
      seg.addEventListener("click", (e) => {
        const btn = e.target.closest("button[data-src]");
        if (!btn) return;
        source = btn.dataset.src;
        seg.querySelectorAll("button").forEach((b) =>
          b.setAttribute("aria-pressed", b === btn ? "true" : "false"));
        buildSourceRow(srcRow);
      });

      document.getElementById("tbStart").addEventListener("click", startSession);
      document.getElementById("tbNewBtn").addEventListener("click", resetView);
      document.getElementById("tbHistBtn").addEventListener("click", (e) =>
        window.ui.openHistoryDrawer("teachbacks", {
          title: "Past teach-backs",
          anchor: e.currentTarget,
          onOpen: openSaved,
        }));
      document.getElementById("tbOptionsBtn").addEventListener("click", () =>
        setSetup(document.getElementById("tbSetup")?.hidden));

      const composer = document.getElementById("tbComposer");
      composer.querySelector("#tbSend").addEventListener("click", () => send());
      composer.querySelector("#tbEval").addEventListener("click", () => send("evaluate"));
      composer.querySelector(".input").addEventListener("keydown", (e) => {
        if (e.key === "Enter") { e.preventDefault(); send(); }
      });
    },

    unmount() {
      session = null; source = "topic"; pdfFile = null;
      if (teardownLayout) teardownLayout();
      document.getElementById("histPopover")?.remove();
      document.getElementById("histPopoverBg")?.remove();
      document.body.classList.remove("hist-scroll-lock");
    },
  };
})();
