/* ============================================================
   Flashcards view — generate active-recall cards from a subject
   PDF, a free topic, or an uploaded PDF (mirrors the Chat tool's
   three sources), then flip / rate confidence / review misses.
   Registers window.Views.flashcards (accent "cards").
   ============================================================ */
(() => {
  "use strict";
  const { el, clear, esc, spinner, empty, toast } = window.ui;

  const DIFFICULTIES = ["easy", "medium", "hard"];
  const RATINGS = {
    didnt_know: { label: "Didn’t know", short: "didn’t" },
    somewhat_knew: { label: "Somewhat", short: "somewhat" },
    knew_well: { label: "Knew well", short: "knew" },
  };

  // Mutable per-mount state (reset in mount, cleared in unmount).
  let state;

  function freshState() {
    return {
      cards: [],            // [{ id, front, back, topic }]
      knowledge: {},        // id -> "didnt_know" | "somewhat_knew" | "knew_well"
      mode: "learn",        // "learn" | "review"
      index: 0,             // explicit index into the active list (never a mod counter)
      flipped: false,
      difficulty: "medium",
      numCards: 10,
      source: "subject",    // "subject" | "topic" | "upload"
      pdfFile: null,        // File chosen in upload mode
      busy: false,
    };
  }

  /* ---------- active list (the fix: review iterates an explicit, clamped index) ---------- */
  function activeList() {
    if (state.mode === "review") {
      return state.cards.filter((c) => state.knowledge[c.id] !== "knew_well");
    }
    return state.cards;
  }

  function clampIndex(list) {
    if (list.length === 0) { state.index = 0; return; }
    if (state.index < 0) state.index = 0;
    if (state.index > list.length - 1) state.index = list.length - 1;
  }

  /* ---------- mount ---------- */
  async function mount(root) {
    state = freshState();
    clear(root);

    const layout = el(`<div class="layout-aside"></div>`);
    const aside = el(`<aside class="panel stack"></aside>`);
    const main = el(`<div id="fc-main" class="stack"></div>`);
    layout.append(aside, main);
    root.append(layout);

    // --- source segmented control ---
    const srcSeg = el(`<div class="seg" role="tablist" aria-label="Card source">
      <button type="button" data-src="subject" aria-pressed="true">Subject</button>
      <button type="button" data-src="topic" aria-pressed="false">Topic</button>
      <button type="button" data-src="upload" aria-pressed="false">Upload PDF</button>
    </div>`);
    const srcCtl = el(`<div class="stack-sm"></div>`);

    // --- difficulty segmented control ---
    const diffSeg = el(`<div class="seg" role="group" aria-label="Difficulty"></div>`);
    DIFFICULTIES.forEach((d) => {
      const b = el(`<button type="button" aria-pressed="${d === state.difficulty}">${esc(d)}</button>`);
      b.addEventListener("click", () => {
        state.difficulty = d;
        diffSeg.querySelectorAll("button").forEach((btn) => btn.setAttribute("aria-pressed", String(btn === b)));
      });
      diffSeg.append(b);
    });

    // --- number range + count badge ---
    const countBadge = el(`<span class="badge badge-accent count">${state.numCards}</span>`);
    const range = el(
      `<input type="range" id="fc-num" min="5" max="20" step="5" value="${state.numCards}" aria-label="Number of cards" style="width:100%" />`
    );
    range.addEventListener("input", () => {
      state.numCards = Number(range.value);
      countBadge.textContent = state.numCards;
    });

    const genBtn = el(`<button class="btn-accent btn-lg btn-block">Generate flashcards</button>`);

    aside.append(
      el(`<div class="between" style="align-items:center;margin-bottom:.1rem">
        <div class="section-head" style="margin:0"><h3>Setup</h3><span>build a deck</span></div>
        <button id="fc-hist" class="btn" type="button" style="padding:.4rem .8rem;font-size:.85rem">≡ History</button>
      </div>`),
      buildField("Source", srcSeg),
      srcCtl,
      buildField("Difficulty", diffSeg),
      buildField("Number of cards", wrapRange(range, countBadge)),
      window.ui.modelSelect(),
      genBtn
    );

    function openSavedDeck(saved) {
      state.cards = ((saved && saved.cards) || []).map((c, i) => ({ id: "c" + i, front: c.front, back: c.back, topic: c.topic }));
      state.knowledge = {}; state.mode = "learn"; state.index = 0; state.flipped = false;
      renderDeck(main);
    }
    aside.querySelector("#fc-hist").addEventListener("click", (e) =>
      window.ui.openHistoryDrawer("flashcards", { title: "Saved decks", anchor: e.currentTarget, onOpen: openSavedDeck }));

    // --- source builders (closures: need srcCtl / genBtn / main) ---
    async function buildSubjectMode() {
      const subjectSel = el(`<select class="select" id="fc-subject" aria-label="Subject"></select>`);
      const pdfSel = el(`<select class="select" id="fc-pdf" aria-label="PDF"></select>`);
      srcCtl.append(buildField("Subject", subjectSel), buildField("PDF", pdfSel));
      try {
        const { subjects } = await window.api.get("/subjects");
        if (!subjects || subjects.length === 0) {
          subjectSel.append(el(`<option value="">No subjects</option>`));
          clear(main);
          main.append(empty("No subjects yet — add one, or use Topic / Upload PDF.", "📁"));
          return;
        }
        subjects.forEach((s) => subjectSel.append(el(`<option value="${esc(s)}">${esc(s)}</option>`)));
        subjectSel.addEventListener("change", () => loadPdfs(subjectSel.value, pdfSel, genBtn));
        await loadPdfs(subjectSel.value, pdfSel, genBtn);
      } catch (e) {
        clear(main);
        main.append(empty("Couldn’t load subjects: " + e.message, "⚠"));
      }
    }

    function buildTopicMode() {
      srcCtl.append(buildField(
        "Topic",
        el(`<input class="input" id="fc-topic" type="text" placeholder="e.g. Photosynthesis" autocomplete="off" />`)
      ));
      srcCtl.append(el(`<p class="hint">Cards are generated from the topic — no document needed.</p>`));
    }

    function buildUploadMode() {
      const wrap = el(`
        <div class="field" style="margin:0">
          <label class="label">PDF</label>
          <div class="row row-wrap">
            <label class="btn" for="fc-file" style="cursor:pointer">Choose PDF…</label>
            <span class="muted" id="fc-file-name">No file selected</span>
          </div>
          <input id="fc-file" type="file" accept="application/pdf" hidden />
          <div class="hint">Upload a PDF and build cards from its contents.</div>
        </div>`);
      srcCtl.append(wrap);
      const input = wrap.querySelector("#fc-file");
      const nameSlot = wrap.querySelector("#fc-file-name");
      input.addEventListener("change", () => {
        state.pdfFile = input.files && input.files[0] ? input.files[0] : null;
        nameSlot.textContent = state.pdfFile ? state.pdfFile.name : "No file selected";
      });
    }

    function buildSource() {
      clear(srcCtl);
      state.pdfFile = null;
      genBtn.disabled = false;
      if (state.source === "subject") buildSubjectMode();
      else if (state.source === "topic") buildTopicMode();
      else buildUploadMode();
    }

    srcSeg.addEventListener("click", (e) => {
      const btn = e.target.closest("button[data-src]");
      if (!btn) return;
      state.source = btn.dataset.src;
      srcSeg.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", String(b === btn)));
      buildSource();
    });

    genBtn.addEventListener("click", () => generate(genBtn, main));

    clear(main);
    main.append(empty("Choose a source, then generate a deck.", "🎴"));
    buildSource();
  }

  function buildField(labelText, control) {
    const f = el(`<div class="field" style="margin:0"></div>`);
    f.append(el(`<label class="label">${esc(labelText)}</label>`));
    f.append(control);
    return f;
  }

  function wrapRange(range, badge) {
    const wrap = el(`<div class="stack-sm"></div>`);
    const top = el(`<div class="between"><span class="hint">5 – 20 cards</span></div>`);
    top.append(badge);
    wrap.append(top, range);
    return wrap;
  }

  async function loadPdfs(subject, pdfSel, genBtn) {
    clear(pdfSel);
    pdfSel.append(el(`<option>Loading…</option>`));
    pdfSel.disabled = true;
    genBtn.disabled = true;
    try {
      const { pdfs } = await window.api.get(`/subjects/${encodeURIComponent(subject)}/pdfs`);
      clear(pdfSel);
      if (!pdfs || pdfs.length === 0) {
        pdfSel.append(el(`<option value="">No PDFs</option>`));
        return;
      }
      pdfs.forEach((p) =>
        pdfSel.append(el(`<option value="${esc(p.name)}">${esc(p.name)} · ${p.pages} pg</option>`))
      );
      pdfSel.disabled = false;
      genBtn.disabled = false;
    } catch (e) {
      clear(pdfSel);
      pdfSel.append(el(`<option value="">Error</option>`));
      toast("Couldn’t load PDFs: " + e.message, "error");
    }
  }

  /* ---------- generate (branches on the chosen source) ---------- */
  async function generate(genBtn, main) {
    if (state.busy) return;

    const model = window.ui.selectedModel(genBtn.closest("aside"));

    let request;
    if (state.source === "subject") {
      const subjEl = document.querySelector("#fc-subject");
      const pdfEl = document.querySelector("#fc-pdf");
      const subject = subjEl && subjEl.value;
      const pdf = pdfEl && pdfEl.value;
      if (!subject || !pdf) { toast("Pick a subject and PDF first.", "error"); return; }
      request = () => window.api.post("/flashcards/generate", {
        subject, pdf, num_cards: state.numCards, difficulty: state.difficulty, model,
      });
    } else if (state.source === "topic") {
      const topicEl = document.querySelector("#fc-topic");
      const topic = topicEl ? topicEl.value.trim() : "";
      if (!topic) { toast("Enter a topic first.", "error"); return; }
      request = () => window.api.post("/flashcards/topic", {
        topic, num_cards: state.numCards, difficulty: state.difficulty, model,
      });
    } else {
      if (!state.pdfFile) { toast("Choose a PDF first.", "error"); return; }
      const fd = new FormData();
      fd.append("file", state.pdfFile);
      fd.append("num_cards", String(state.numCards));
      fd.append("difficulty", state.difficulty);
      if (model) fd.append("model", model);
      request = () => window.api.postForm("/flashcards/upload", fd);
    }

    state.busy = true;
    genBtn.disabled = true;
    clear(main);
    main.append(spinner("Creating your flashcards…"));
    try {
      const data = await request();
      const cards = (data.cards || []).map((c, i) => ({
        id: `c${i}`,
        front: c.front,
        back: c.back,
        topic: c.topic,
      }));
      if (cards.length === 0) {
        clear(main);
        main.append(empty("No flashcards came back. Try again or change the source.", "🤔"));
        return;
      }
      state.cards = cards;
      state.knowledge = {};
      state.mode = "learn";
      state.index = 0;
      state.flipped = false;
      toast(`Generated ${cards.length} flashcards.`, "ok");
      const __hp = document.querySelector('.history-panel[data-kind="flashcards"]');
      if (__hp && __hp.refresh) __hp.refresh();
      renderDeck(main);
    } catch (e) {
      clear(main);
      main.append(empty("Couldn’t generate flashcards: " + e.message, "⚠"));
    } finally {
      state.busy = false;
      genBtn.disabled = false;
    }
  }

  /* ---------- deck rendering ---------- */
  function renderDeck(main) {
    clear(main);

    const list = activeList();
    clampIndex(list);

    // Mode toggle + reset
    const head = el(`<div class="between"></div>`);
    const modeSeg = el(`<div class="seg" role="group" aria-label="Study mode"></div>`);
    const learnBtn = el(`<button type="button" aria-pressed="${state.mode === "learn"}">Learn</button>`);
    const reviewBtn = el(`<button type="button" aria-pressed="${state.mode === "review"}">Review</button>`);
    learnBtn.addEventListener("click", () => switchMode("learn", main));
    reviewBtn.addEventListener("click", () => switchMode("review", main));
    modeSeg.append(learnBtn, reviewBtn);

    const resetBtn = el(`<button class="btn-ghost">Reset</button>`);
    resetBtn.addEventListener("click", () => {
      state.knowledge = {};
      state.mode = "learn";
      state.index = 0;
      state.flipped = false;
      toast("Progress reset.", "info");
      renderDeck(main);
    });
    head.append(modeSeg, resetBtn);
    main.append(head);

    // Review with nothing to review -> celebrate, offer back to learn.
    if (state.mode === "review" && list.length === 0) {
      main.append(statsRow());
      const done = empty("Nothing to review — you knew every card well!", "🌟");
      const back = el(`<button class="btn-accent btn-block">Back to all cards</button>`);
      back.addEventListener("click", () => switchMode("learn", main));
      const wrap = el(`<div class="panel stack center"></div>`);
      wrap.append(done, back);
      main.append(wrap);
      return;
    }

    const card = list[state.index];

    // Progress
    const progressWrap = el(`<div class="stack-sm"></div>`);
    const cap = el(
      `<div class="between"><span class="muted">Card ${state.index + 1} of ${list.length}</span>` +
      `<span class="badge">${esc(state.mode === "review" ? "review" : "learn")}</span></div>`
    );
    const bar = el(`<div class="progress"><i style="width:${((state.index + 1) / list.length) * 100}%"></i></div>`);
    progressWrap.append(cap, bar);
    main.append(progressWrap);

    // The card itself
    const deck = el(`<div class="deck"></div>`);
    const flip = el(`<div class="flip${state.flipped ? " is-back" : ""}" tabindex="0" role="button" ` +
      `aria-label="Flashcard, press Enter or Space to flip"></div>`);
    const front = el(`<div class="flip-face flip-front"></div>`);
    front.append(
      el(`<span class="topic badge badge-accent">${esc(card.topic)}</span>`),
      el(`<div class="q">${esc(card.front)}</div>`),
      el(`<span class="fliphint">click to flip</span>`)
    );
    const back = el(`<div class="flip-face flip-back"></div>`);
    back.append(
      el(`<span class="topic badge">${esc(card.topic)}</span>`),
      el(`<div class="a">${esc(card.back)}</div>`),
      el(`<span class="fliphint">click to flip back</span>`)
    );
    flip.append(front, back);
    const doFlip = () => { state.flipped = !state.flipped; flip.classList.toggle("is-back", state.flipped); };
    flip.addEventListener("click", doFlip);
    flip.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); doFlip(); }
    });
    deck.append(flip);
    main.append(deck);

    // Nav row (prev / flip / next)
    const nav = el(`<div class="row between"></div>`);
    const prev = el(`<button class="btn">← Prev</button>`);
    const flipBtn = el(`<button class="btn-ghost">Flip</button>`);
    const next = el(`<button class="btn">Next →</button>`);
    prev.disabled = state.index === 0;
    next.disabled = state.index >= list.length - 1;
    prev.addEventListener("click", () => move(-1, main));
    next.addEventListener("click", () => move(1, main));
    flipBtn.addEventListener("click", doFlip);
    nav.append(prev, flipBtn, next);
    main.append(nav);

    // Rating row
    const rate = el(`<div class="rate"></div>`);
    Object.entries(RATINGS).forEach(([key, meta]) => {
      const active = state.knowledge[card.id] === key;
      const b = el(`<button class="btn${active ? "-accent" : ""}">${esc(meta.label)}</button>`);
      b.addEventListener("click", () => {
        state.knowledge[card.id] = key;
        // Advance within the CURRENT active list using an explicit index (no mod on a shrinking list).
        const after = activeList();
        if (after.length === 0) { renderDeck(main); return; }
        if (state.index > after.length - 1) state.index = after.length - 1;
        else if (state.index < after.length - 1) state.index += 1;
        state.flipped = false;
        renderDeck(main);
      });
      rate.append(b);
    });
    main.append(buildField("How well did you know it?", rate));

    main.append(statsRow());
  }

  function move(delta, main) {
    const list = activeList();
    state.index = Math.min(Math.max(state.index + delta, 0), Math.max(list.length - 1, 0));
    state.flipped = false;
    renderDeck(main);
  }

  function switchMode(mode, main) {
    state.mode = mode;
    state.index = 0;
    state.flipped = false;
    renderDeck(main);
  }

  function statsRow() {
    const counts = { didnt_know: 0, somewhat_knew: 0, knew_well: 0 };
    Object.values(state.knowledge).forEach((v) => { if (v in counts) counts[v] += 1; });
    const rated = counts.didnt_know + counts.somewhat_knew + counts.knew_well;
    const row = el(`<div class="panel-soft row-wrap between"></div>`);
    const left = el(`<div class="row-wrap"></div>`);
    left.append(
      el(`<span class="badge badge-accent">Knew ${counts.knew_well}</span>`),
      el(`<span class="badge">Somewhat ${counts.somewhat_knew}</span>`),
      el(`<span class="badge">Didn’t ${counts.didnt_know}</span>`)
    );
    row.append(left, el(`<span class="muted">${rated} of ${state.cards.length} rated</span>`));
    return row;
  }

  function unmount() {
    state = null;
  }

  window.Views.flashcards = { accent: "cards", title: "Flashcards", mount, unmount };
})();
