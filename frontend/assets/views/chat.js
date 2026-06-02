/* ============================================================
   Chat Assistant (RAG) view.
   Pick a source — Subject / Topic / Upload PDF — start a session,
   then chat. A sidebar lists your saved chats (search + delete +
   resume) and a "New chat" button. Answers render as rich, safe
   markdown. Composer is locked until a session exists.
   ============================================================ */
(() => {
  "use strict";
  const { el, esc, clear, toast, md, spinner, copy } = window.ui;

  let session = null;       // { id, label }
  let source = "subject";   // "subject" | "topic" | "upload"
  let pdfFile = null;       // File chosen for upload

  function buildSourceControl(host) {
    clear(host);
    if (source === "subject") {
      const field = el(`
        <div class="field">
          <label class="label" for="chatSubject">Subject</label>
          <select id="chatSubject" class="select"><option value="">Loading subjects…</option></select>
          <div class="hint">Chat across every PDF filed under this subject.</div>
        </div>`);
      host.append(field);
      const sel = field.querySelector("#chatSubject");
      window.api.get("/subjects").then((d) => {
        const subjects = d.subjects || [];
        clear(sel);
        if (!subjects.length) {
          sel.append(el(`<option value="">No subjects found</option>`));
          return;
        }
        subjects.forEach((s) => sel.append(el(`<option value="${esc(s)}">${esc(s)}</option>`)));
        sel.value = subjects[0];
      }).catch((e) => {
        clear(sel);
        sel.append(el(`<option value="">Couldn’t load subjects</option>`));
        toast(e.message, "error");
      });
    } else if (source === "topic") {
      host.append(el(`
        <div class="field">
          <label class="label" for="chatTopic">Topic</label>
          <input id="chatTopic" class="input" type="text" placeholder="e.g. Binary search trees" autocomplete="off" />
          <div class="hint">Learn a topic with a specialised tutor — no documents needed.</div>
        </div>`));
    } else {
      const field = el(`
        <div class="field">
          <label class="label">PDF</label>
          <div class="row row-wrap">
            <label class="btn" for="chatPdf" style="cursor:pointer;">Choose PDF…</label>
            <span class="muted" id="chatPdfName">No file selected</span>
          </div>
          <input id="chatPdf" type="file" accept="application/pdf" hidden />
          <div class="hint">Upload a PDF and chat about its contents.</div>
        </div>`);
      host.append(field);
      const input = field.querySelector("#chatPdf");
      const nameSlot = field.querySelector("#chatPdfName");
      input.addEventListener("change", () => {
        pdfFile = input.files && input.files[0] ? input.files[0] : null;
        nameSlot.textContent = pdfFile ? pdfFile.name : "No file selected";
      });
    }
  }

  function setReady(thread, text) {
    clear(thread);
    thread.append(el(`<div class="empty"><div class="big">✦</div><div>${esc(text)}</div></div>`));
  }

  // Render one stored/streamed message into the thread (user = plain, assistant = markdown + copy).
  function appendMessage(thread, role, content) {
    if (role === "user") {
      const me = el(`<div class="msg me"></div>`);
      me.textContent = content;
      thread.append(me);
    } else {
      const ai = el(`<div class="msg ai"></div>`);
      ai.append(md(content));
      const cp = el(`<button class="msg-copy" type="button">Copy</button>`);
      cp.addEventListener("click", () => copy(content, cp));
      ai.append(cp);
      thread.append(ai);
    }
  }

  function refreshChats() {
    const hp = document.querySelector('.history-panel[data-kind="chats"]');
    if (hp && hp.refresh) hp.refresh();
  }

  async function startSession(controls, statusSlot, chatHost, composer) {
    const startBtn = controls.querySelector("#chatStart");
    const thread = chatHost.querySelector(".thread");

    let payload;
    let upload = false;
    if (source === "subject") {
      const subject = controls.querySelector("#chatSubject").value;
      if (!subject) { toast("Pick a subject first.", "error"); return; }
      payload = { mode: "subject", subject, model: window.ui.selectedModel(controls) };
    } else if (source === "topic") {
      const topic = controls.querySelector("#chatTopic").value.trim();
      if (!topic) { toast("Enter a topic first.", "error"); return; }
      payload = { mode: "topic", topic, model: window.ui.selectedModel(controls) };
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
        : "Building a knowledge base from your PDFs… the first run also downloads a small embedding model, so give it a few seconds."
    ));

    try {
      let resp;
      if (upload) {
        const fd = new FormData();
        fd.append("file", pdfFile);
        const __m = window.ui.selectedModel(controls); if (__m) fd.append("model", __m);
        resp = await window.api.postForm("/chat/upload", fd);
      } else {
        resp = await window.api.post("/chat/start", payload);
      }
      session = { id: resp.session_id, label: resp.label };
      clear(statusSlot);
      statusSlot.append(el(`<span class="badge badge-accent">${esc(resp.label)}</span>`));
      setReady(thread, "Session ready — ask your first question.");
      composer.querySelector(".input").disabled = false;
      composer.querySelector("#chatSend").disabled = false;
      composer.querySelector(".input").focus();
      toast("Session started.", "ok");
      refreshChats();   // the new chat now appears in the sidebar
    } catch (e) {
      setReady(thread, "Couldn’t start the session. Check your Groq key and try again.");
      toast(e.message, "error");
    } finally {
      startBtn.disabled = false;
      startBtn.textContent = "Start session";
    }
  }

  // Re-open a saved chat from the sidebar and replay its messages.
  async function resumeChat(saved) {
    const main = document.getElementById("chatMain");
    if (!main) return;
    const statusSlot = main.querySelector("#chatStatus");
    const thread = main.querySelector(".thread");
    const composer = main.querySelector(".composer");
    clear(thread);
    thread.append(spinner("Re-opening your chat…"));
    try {
      const resp = await window.api.post("/chat/resume", { chat_id: saved.id });
      session = { id: resp.session_id, label: resp.label };
      clear(statusSlot);
      statusSlot.append(el(`<span class="badge badge-accent">${esc(resp.label)}</span>`));
      clear(thread);
      const messages = resp.messages || [];
      if (!messages.length) setReady(thread, "Session ready — ask your first question.");
      else messages.forEach((m) => appendMessage(thread, m.role, m.content));
      composer.querySelector(".input").disabled = false;
      composer.querySelector("#chatSend").disabled = false;
      composer.querySelector(".input").focus();
      thread.scrollTop = thread.scrollHeight;
    } catch (e) {
      setReady(thread, "Couldn’t re-open that chat.");
      toast(e.message, "error");
    }
  }

  async function send(composer, thread) {
    if (!session) return;
    const input = composer.querySelector(".input");
    const sendBtn = composer.querySelector("#chatSend");
    const text = input.value.trim();
    if (!text) return;

    const placeholder = thread.querySelector(".empty");
    if (placeholder) placeholder.remove();
    const firstTurn = !thread.querySelector(".msg");

    appendMessage(thread, "user", text);

    input.value = "";
    input.disabled = true;
    sendBtn.disabled = true;

    const aiMsg = el(`<div class="msg ai"><span class="typing"><i></i><i></i><i></i></span></div>`);
    thread.append(aiMsg);
    thread.scrollTop = thread.scrollHeight;

    try {
      const resp = await window.api.post("/chat/message", { session_id: session.id, message: text });
      clear(aiMsg);
      aiMsg.append(md(resp.answer));
      const cp = el(`<button class="msg-copy" type="button">Copy</button>`);
      cp.addEventListener("click", () => copy(resp.answer, cp));
      aiMsg.append(cp);
      if (firstTurn) { refreshChats(); setTimeout(refreshChats, 3500); }   // refresh again once the LLM auto-title lands
    } catch (e) {
      aiMsg.textContent = "⚠ " + e.message;
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

  // Build (or rebuild, for "New chat") the main column: source controls + chat panel.
  function buildMain(main) {
    clear(main);
    session = null;
    source = "subject";
    pdfFile = null;
    // NOTE: wrapped in a single .stack so el() (which returns one root element)
    // keeps BOTH panels; #chatMain is itself a stack, so the nesting is harmless.
    main.append(el(`
      <div class="stack">
        <div class="panel stack">
          <div class="seg" role="tablist" aria-label="Chat source">
            <button type="button" data-src="subject" aria-pressed="true">Subject</button>
            <button type="button" data-src="topic" aria-pressed="false">Topic</button>
            <button type="button" data-src="upload" aria-pressed="false">Upload PDF</button>
          </div>
          <div id="chatSourceCtl"></div>
          <div id="chatModelCtl"></div>
          <div class="row row-wrap between">
            <button id="chatStart" class="btn-accent" type="button">Start session</button>
            <span id="chatStatus"></span>
          </div>
        </div>
        <div class="panel">
          <div class="chat">
            <div class="thread">
              <div class="empty"><div class="big">✦</div><div>Start a new session or open a saved chat.</div></div>
            </div>
            <div class="composer">
              <input class="input" type="text" placeholder="Ask a question…" aria-label="Message" disabled />
              <button id="chatSend" class="btn-accent" type="button" disabled>Send</button>
            </div>
          </div>
        </div>
      </div>`));

    const seg = main.querySelector(".seg");
    const sourceCtl = main.querySelector("#chatSourceCtl");
    const statusSlot = main.querySelector("#chatStatus");
    const chatHost = main.querySelector(".chat");
    const thread = main.querySelector(".thread");
    const composer = main.querySelector(".composer");

    buildSourceControl(sourceCtl);
    main.querySelector("#chatModelCtl").append(window.ui.modelSelect());

    seg.addEventListener("click", (e) => {
      const btn = e.target.closest("button[data-src]");
      if (!btn) return;
      source = btn.dataset.src;
      pdfFile = null;
      seg.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", b === btn ? "true" : "false"));
      buildSourceControl(sourceCtl);
    });

    main.querySelector("#chatStart").addEventListener("click", () => {
      startSession(main, statusSlot, chatHost, composer);
    });
    composer.querySelector("#chatSend").addEventListener("click", () => send(composer, thread));
    composer.querySelector(".input").addEventListener("keydown", (e) => {
      if (e.key === "Enter") { e.preventDefault(); send(composer, thread); }
    });
  }

  window.Views = window.Views || {};
  window.Views.chat = {
    accent: "chat",
    title: "Chat Assistant",
    async mount(root) {
      const layout = el(`
        <div class="layout-aside">
          <aside class="panel stack" id="chatAside">
            <button id="chatNew" class="btn-accent btn-block" type="button">+ New chat</button>
          </aside>
          <section class="stack" id="chatMain"></section>
        </div>`);
      root.append(layout);

      const aside = layout.querySelector("#chatAside");
      const main = layout.querySelector("#chatMain");

      buildMain(main);
      aside.append(window.ui.historyPanel("chats", { title: "Your chats", onOpen: resumeChat }));

      layout.querySelector("#chatNew").addEventListener("click", () => buildMain(main));
    },
    unmount() {
      session = null;
      source = "subject";
      pdfFile = null;
    },
  };
})();
