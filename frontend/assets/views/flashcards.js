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
  // SM-2 ratings (POST /api/flashcards/review schedules the next review): keys 1-4.
  const RATINGS = {
    again: { label: "Again", short: "again" },
    hard: { label: "Hard", short: "hard" },
    good: { label: "Good", short: "good" },
    easy: { label: "Easy", short: "easy" },
  };
  const PASSED = (v) => v === "good" || v === "easy";

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
      meta: null,           // {subject, topic} of the current deck — for progress recording
      llm: null,            // top-level `llm` of the generating response (Model: … line)
      recorded: false,      // legacy flag (unused: reviews are recorded server-side by /flashcards/review)
      deckId: null,         // saved deck id => spaced repetition + analytics; null when history is off
      pending: [],          // ratings not yet sent
      batchId: null,        // idempotency key of the batch in flight
      due: null,            // {new, due, learning, later, next_due} from /flashcards/due
    };
  }

  /* ---------- active list (the fix: review iterates an explicit, clamped index) ---------- */
  function activeList() {
    if (state.mode === "review") {
      return state.cards.filter((c) => !PASSED(state.knowledge[c.id]));
    }
    return state.cards;
  }

  function clampIndex(list) {
    if (list.length === 0) { state.index = 0; return; }
    if (state.index < 0) state.index = 0;
    if (state.index > list.length - 1) state.index = list.length - 1;
  }

  function mdDiv(cls, text) {
    const d = el(`<div class="${cls}" style="overflow-wrap:anywhere"></div>`);
    d.append(window.ui.md(String(text || "")));
    const n = String(text || "").length;   // length step: <=80 large, <=200 medium, longer smaller (never < 18px)
    if (n > 80) d.dataset.len = n > 200 ? "l" : "m";
    return d;
  }

  // Open a managed deck for study: server-built queue (SM-2 due / cram / starred, optional tag); cloze & reversed arrive as units.
  async function studyDeck(deckId, mode, tag, main) {
    clear(main);
    main.append(spinner("Building your queue…"));
    try {
      const qs = new URLSearchParams({ deck_id: deckId, mode, new_limit: "20", limit: "200" });
      if (tag) qs.set("tag", tag);
      const q = await window.api.get("/flashcards/due?" + qs);
      clear(main);
      if (!q.cards.length) {
        main.append(empty(mode === "due" ? "Nothing due today. Use “Cram all” to practise anyway." : "No cards match.", "🌟"));
        const back = ui.iconBtn({ icon: "chevron-left", label: "Back to my decks", cls: "ib-outline" });
        back.addEventListener("click", () => showDecks(main));
        main.append(back);
        return;
      }
      let deckMeta = { tags: [], cards: [] };
      try { deckMeta = await window.api.get("/flashcards/decks/" + encodeURIComponent(deckId)); } catch (_) {}
      state.studyMeta = { deckId, mode, tag: tag || "", tags: deckMeta.tags || [], raw: deckMeta.cards || [], main };
      state.cards = q.cards.map((c) => ({ id: c.cid, cid: c.cid, starred: c.starred, front: c.front, back: c.back, topic: c.topic || c.type, type: c.type, options: c.options, answer_index: c.answer_index, hint: c.hint, explanation: c.explanation }));
      state.deckId = deckId; state.due = q.counts; state.pending = []; state.knowledge = {}; state.mode = "learn"; state.index = 0;
      state.flipped = false; state.llm = null; state.stored = true; state.session = { start: Date.now(), mode };
      renderDeck(main);
    } catch (e) {
      clear(main);
      main.append(window.ui.errorBlock(e, () => studyDeck(deckId, mode, tag, main)));
    }
  }

  async function showDecks(main) {
    clear(main);
    try { await window.ui.loadScript("/assets/views/flashcards-decks.js"); } catch (e) { main.append(window.ui.errorBlock(e, () => showDecks(main))); return; }
    window.FlashDecks.render(main, { onStudy: (id, mode, tag) => studyDeck(id, mode, tag, main) });
  }

  // `e`: edit the card being studied (front/back/hint/tags) without leaving the session.
  function editCurrent() {
    const main = document.getElementById("fc-main");
    const card = activeList()[state.index];
    if (!main || !card || !state.studyMeta || main.querySelector("#fc-edit")) return;
    const parent = card.cid.split(":")[0];
    const raw = (state.studyMeta.raw || []).find((c) => c.id === parent) || { front: card.front, back: card.back, tags: [], hint: card.hint || "" };
    const box = el(`<div class="panel-soft stack-sm" id="fc-edit" role="dialog" aria-label="Edit card"></div>`);
    const f = el(`<textarea class="textarea" rows="2" aria-label="Front">${esc(raw.front)}</textarea>`);
    const b = el(`<textarea class="textarea" rows="2" aria-label="Back">${esc(raw.back || "")}</textarea>`);
    const t = el(`<input class="input" aria-label="Tags" value="${esc((raw.tags || []).join(" "))}" />`);
    const h = el(`<input class="input" aria-label="Hint" value="${esc(raw.hint || "")}" placeholder="Hint" />`);
    const row = el(`<div class="row" style="gap:.4rem"></div>`);
    const save = el(`<button class="btn-accent" type="button">Save</button>`);
    const cancel = el(`<button class="btn" type="button">Cancel</button>`);
    cancel.addEventListener("click", () => box.remove());
    save.addEventListener("click", async () => {
      save.disabled = true;
      try {
        const r = await window.api.patch(`/flashcards/decks/${encodeURIComponent(state.studyMeta.deckId)}/cards/${parent}`,
          { front: f.value, back: b.value, tags: t.value.split(/[\s,]+/).filter(Boolean), hint: h.value });
        Object.assign(raw, r.card);
        if ((raw.type || "basic") === "basic" || raw.type === "typed") { card.front = r.card.front; card.back = r.card.back; }
        card.hint = r.card.hint;
        box.remove(); toast("Card updated.", "ok"); renderDeck(main);
      } catch (x) { toast(x.message, "error"); save.disabled = false; }
    });
    f.addEventListener("keydown", (ev) => ev.stopPropagation()); b.addEventListener("keydown", (ev) => ev.stopPropagation());
    row.append(save, cancel); box.append(f, b, t, h, row);
    main.prepend(box); f.focus();
  }

  function onKey(e) {
    if (!state || e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.target && /^(INPUT|TEXTAREA|SELECT)$/.test(e.target.tagName)) return;
    const idx = ["1", "2", "3", "4"].indexOf(e.key);
    if (idx >= 0) {
      const b = document.querySelectorAll('#fc-main .rate button')[idx];
      if (b) { e.preventDefault(); b.click(); }
    } else if (e.key === "e") {
      if (state.studyMeta && document.getElementById("fc-main")) { e.preventDefault(); editCurrent(); }
    } else if (e.key === "z" || e.key === "s") {
      const main = document.getElementById("fc-main");
      if (e.key === "s") {
        const c = activeList()[state.index];
        if (c && state.deckId && c.cid) {
          const parent = c.cid.split(":")[0];
          window.api.patch(`/flashcards/decks/${encodeURIComponent(state.deckId)}/cards/${parent}`, { starred: !c.starred })
            .then(() => { c.starred = !c.starred; const raw = (state.studyMeta.raw || []).find((x) => x.id === parent); if (raw) raw.starred = c.starred; toast(c.starred ? "Starred." : "Unstarred.", "ok"); }).catch((x) => toast(x.message, "error"));
        }
      } else if (state.undo && state.undo.length && main) {
        const last = state.undo[state.undo.length - 1];
        const at = state.pending.map((r) => r.card_id).lastIndexOf(last.id);
        if (at < 0) { toast("That rating was already saved.", "info"); return; }
        state.undo.pop(); state.pending.splice(at, 1); delete state.knowledge[last.id];
        state.index = last.idx; state.flipped = false; renderDeck(main);
      }
    } else if (e.key === " " && !(e.target && e.target.tagName === "BUTTON")) {
      const f = document.querySelector("#fc-main .flip");
      if (f) { e.preventDefault(); f.click(); }
    }
  }

  /* ---------- mount ---------- */
  async function mount(root) {
    state = freshState();
    document.addEventListener("keydown", onKey);
    clear(root);

    const layout = el(`<div class="layout-aside"></div>`);
    const aside = el(`<aside class="panel stack"></aside>`);
    const main = el(`<div id="fc-main" class="stack"></div>`);
    layout.append(aside, main);
    root.append(layout);

    // --- source segmented control ---
    const srcSeg = el(`<div class="seg" role="tablist" aria-label="Card source" style="flex-wrap:wrap">
      <button type="button" data-src="subject" aria-pressed="true">Subject</button>
      <button type="button" data-src="topic" aria-pressed="false">Topic</button>
      <button type="button" data-src="upload" aria-pressed="false">Upload PDF</button>
    </div>`);
    const srcCtl = el(`<div class="stack-sm"></div>`);

    // --- difficulty segmented control ---
    const diffSeg = el(`<div class="seg" role="group" aria-label="Difficulty" style="flex-wrap:wrap"></div>`);
    DIFFICULTIES.forEach((d) => {
      const label = d[0].toUpperCase() + d.slice(1);
      const b = el(`<button type="button" aria-pressed="${d === state.difficulty}">${esc(label)}</button>`);
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

    const genBtn = el(`<button type="button" class="btn-accent btn-lg btn-block">Generate flashcards</button>`);

    aside.append(
      el(`<div class="between" style="align-items:center;margin-bottom:.1rem;gap:.6rem">
        <div class="section-head" style="margin:0;min-width:0"><h3>Setup</h3><span>build a deck</span></div>
        <button id="fc-hist" class="icon-btn ib-outline" type="button" aria-label="Open flashcard history" title="Open flashcard history">${ui.iconHtml("history")}</button>
      </div>`),
      buildField("Source", srcSeg),
      srcCtl,
      buildField("Difficulty", diffSeg),
      buildField("Number of cards", wrapRange(range, countBadge)),
      window.ui.modelSelect(),
      genBtn
    );

    const decksBtn = el(`<button type="button" class="btn btn-block">${ui.iconHtml("layers")}<span>My decks</span></button>`);
    decksBtn.addEventListener("click", () => showDecks(main));
    aside.insertBefore(decksBtn, aside.children[1]);

    function openSavedDeck(saved) {
      state.cards = ((saved && saved.cards) || []).map((c, i) => ({ id: "c" + i, front: c.front, back: c.back, topic: c.topic }));
      state.knowledge = {}; state.mode = "learn"; state.index = 0; state.flipped = false;
      state.meta = { topic: (saved && saved.title) || null }; state.recorded = false; state.llm = (saved && saved.llm) || null; state.stored = true;
      state.deckId = (saved && saved.id) || null; state.pending = [];
      loadDue();
      renderDeck(main);
    }
    aside.querySelector("#fc-hist").addEventListener("click", (e) =>
      window.ui.openHistoryDrawer("flashcards", { title: "Saved decks", anchor: e.currentTarget, onOpen: openSavedDeck }));

    // Deep-link handoff: if a chat artifact card sent us here, open that saved deck (best-effort).
    const _art = window.ui.takeArtifact(["flashcards"]);
    if (_art) window.api.get(`/history/${_art.kind}/${encodeURIComponent(_art.id)}`).then(openSavedDeck).catch(() => {});

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
          <label class="label" for="fc-file">PDF</label>
          <div class="row row-wrap" style="align-items:center">
            <label class="btn" for="fc-file" style="cursor:pointer;flex:none">Choose PDF…</label>
            <span class="muted" id="fc-file-name" style="min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">No file selected</span>
          </div>
          <input id="fc-file" type="file" accept="application/pdf" hidden aria-label="Upload PDF for flashcards" />
          <div class="hint">Upload a PDF and build cards from its contents.</div>
        </div>`);
      srcCtl.append(wrap);
      const input = wrap.querySelector("#fc-file");
      const nameSlot = wrap.querySelector("#fc-file-name");
      input.addEventListener("change", () => {
        state.pdfFile = input.files && input.files[0] ? input.files[0] : null;
        nameSlot.textContent = state.pdfFile ? state.pdfFile.name : "No file selected";
        nameSlot.title = state.pdfFile ? state.pdfFile.name : "";
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
    const id = control.id || control.getAttribute?.("id");
    if (id) {
      f.append(el(`<label class="label" for="${esc(id)}">${esc(labelText)}</label>`));
    } else {
      const lid = "fc-lbl-" + labelText.toLowerCase().replace(/\s+/g, "-");
      f.append(el(`<span class="label" id="${lid}">${esc(labelText)}</span>`));
      if (control.setAttribute) {
        if (!control.getAttribute("aria-label") && !control.getAttribute("aria-labelledby")) {
          control.setAttribute("aria-labelledby", lid);
        }
      }
    }
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
    const aside = genBtn.closest("aside") || document;
    if (state.source === "subject") {
      const subjEl = aside.querySelector("#fc-subject");
      const pdfEl = aside.querySelector("#fc-pdf");
      const subject = subjEl && subjEl.value;
      const pdf = pdfEl && pdfEl.value;
      if (!subject || !pdf) { toast("Pick a subject and PDF first.", "error"); return; }
      state.meta = { subject, topic: pdf };
      request = () => window.api.post("/flashcards/generate", {
        subject, pdf, num_cards: state.numCards, difficulty: state.difficulty, model,
      });
    } else if (state.source === "topic") {
      const topicEl = aside.querySelector("#fc-topic");
      const topic = topicEl ? topicEl.value.trim() : "";
      if (!topic) { toast("Enter a topic first.", "error"); return; }
      state.meta = { topic };
      request = () => window.api.post("/flashcards/topic", {
        topic, num_cards: state.numCards, difficulty: state.difficulty, model,
      });
    } else {
      if (!state.pdfFile) { toast("Choose a PDF first.", "error"); return; }
      state.meta = { topic: state.pdfFile.name };
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
      state.llm = data.llm || null;
      state.stored = false;
      state.deckId = data.id || null; state.pending = [];
      loadDue();
      state.knowledge = {};
      state.recorded = false;
      state.mode = "learn";
      state.index = 0;
      state.flipped = false;
      toast(`Generated ${cards.length} flashcards.`, "ok");
      const __hp = document.querySelector('.history-panel[data-kind="flashcards"]');
      if (__hp && __hp.refresh) __hp.refresh();
      renderDeck(main);
    } catch (e) {
      clear(main);
      main.append(window.ui.errorBlock(e, e.retryable ? () => generate(genBtn, main) : null));
    } finally {
      state.busy = false;
      genBtn.disabled = false;
    }
  }

  /* ---------- spaced repetition (server-side SM-2) ---------- */
  async function loadDue() {
    const st = state;
    if (!st || !st.deckId) return;
    try {
      const q = await window.api.get(`/flashcards/due?deck_id=${encodeURIComponent(st.deckId)}&new_limit=50&limit=200`);
      if (state !== st) return;
      const by = {}; (q.cards || []).forEach((x) => { by[x.front + "\u241f" + x.back] = x.cid; });
      st.cards.forEach((c) => { c.cid = by[c.front + "\u241f" + c.back]; });
      st.due = q.counts || null;
      const badge = document.getElementById("fc-due");
      if (badge) badge.textContent = dueText();
    } catch (_) { /* spaced repetition is best-effort; the deck still works */ }
  }
  function dueText() {
    const d = state && state.due;
    if (!d) return "";
    const n = (d.due || 0) + (d.learning || 0);
    return `${n} due today \u00b7 ${d.new || 0} new` + (d.next_due ? ` \u00b7 next ${d.next_due}` : "");
  }
  async function flushReviews() {
    const st = state;
    if (!st || !st.deckId || !st.pending.length) return;
    if (!st.batchId) st.batchId = "fc" + Date.now().toString(36) + Math.random().toString(36).slice(2, 10);
    const reviews = st.pending.slice();
    try {
      const res = await window.api.post("/flashcards/review", { deck_id: st.deckId, batch_id: st.batchId, reviews });
      st.pending.splice(0, reviews.length); st.batchId = null;
      if (res && res.srs) st.due = res.srs;
      window.api.invalidate && window.api.invalidate("/progress");
    } catch (_) { /* keep the batch + its id: the next flush retries it (idempotent server-side) */ }
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

    const resetBtn = ui.iconBtn({ icon: "undo", label: "Reset progress", cls: "ib-outline" });
    resetBtn.addEventListener("click", () => {
      state.knowledge = {};
      state.recorded = false;
      state.mode = "learn";
      state.index = 0;
      state.flipped = false;
      toast("Progress reset.", "info");
      renderDeck(main);
    });
    head.append(modeSeg, resetBtn);
    main.append(head);
    if (state.studyMeta) {      // managed deck: what to study (due / cram all / starred) and which tag
      const m = state.studyMeta;
      const bar = el(`<div class="row row-wrap" style="gap:.5rem" role="group" aria-label="What to study"></div>`);
      const seg = el(`<div class="seg"></div>`);
      [["due", "Due"], ["cram", "Cram all"], ["starred", "Starred"]].forEach(([k, label]) => {
        const b = el(`<button type="button" aria-pressed="${m.mode === k}">${label}</button>`);
        b.addEventListener("click", () => { flushReviews().then(() => studyDeck(m.deckId, k, m.tag, m.main)); });
        seg.append(b);
      });
      bar.append(seg);
      if (m.tags.length) {
        const sel = el(`<select class="select" aria-label="Study only cards with tag" style="max-width:12rem"><option value="">All tags</option>${m.tags.map((t) => `<option ${t === m.tag ? "selected" : ""}>${esc(t)}</option>`).join("")}</select>`);
        sel.addEventListener("change", () => { flushReviews().then(() => studyDeck(m.deckId, m.mode, sel.value, m.main)); });
        bar.append(sel);
      }
      const back = ui.iconBtn({ icon: "chevron-left", label: "Back to my decks", cls: "ib-outline" });
      back.addEventListener("click", () => { flushReviews().then(() => { state.studyMeta = null; state.deckId = null; showDecks(m.main); }); });
      bar.append(back);
      main.append(bar);
    }

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
      el(`<span class="topic badge badge-accent" style="max-width:calc(100% - 2.2rem);overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${esc(card.topic || "")}">${esc(card.topic || "")}</span>`),
      mdDiv("q", card.front),
      el(`<span class="fliphint">click to flip</span>`)
    );
    const back = el(`<div class="flip-face flip-back"></div>`);
    back.append(
      el(`<span class="topic badge" style="max-width:calc(100% - 2.2rem);overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${esc(card.topic || "")}">${esc(card.topic || "")}</span>`),
      mdDiv("a", card.back),
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
    const prev = ui.iconBtn({ icon: "chevron-left", label: "Previous card", cls: "ib-outline" });
    const flipBtn = ui.iconBtn({ icon: "flip", label: "Flip card", cls: "ib-outline" });
    const next = ui.iconBtn({ icon: "chevron-right", label: "Next card", cls: "ib-outline" });
    prev.disabled = state.index === 0;
    next.disabled = state.index >= list.length - 1;
    prev.addEventListener("click", () => move(-1, main));
    next.addEventListener("click", () => move(1, main));
    flipBtn.addEventListener("click", doFlip);
    nav.append(prev, flipBtn, next);
    main.append(nav);
    if (card.hint) main.append(el(`<details class="hint"><summary>Hint</summary><div>${esc(card.hint)}</div></details>`));
    if (card.type === "mcq" && Array.isArray(card.options)) {
      const g = el(`<div class="opts" role="group" aria-label="Choose an answer"></div>`);
      card.options.forEach((o, i) => {
        const b = el(`<button class="opt" type="button"><span class="key">${String.fromCharCode(65 + i)}</span><span style="min-width:0;overflow-wrap:anywhere">${esc(o)}</span></button>`);
        b.addEventListener("click", () => {
          g.querySelectorAll("button").forEach((x, j) => { x.disabled = true; if (j === card.answer_index) x.dataset.state = "correct"; else if (x === b) x.dataset.state = "wrong"; });
          if (!state.flipped) doFlip();
          g.after(el(`<div class="explain" role="status">${i === card.answer_index ? "Correct." : "Not quite."} ${esc(card.explanation || "")}</div>`));
        });
        g.append(b);
      });
      main.append(g);
    } else if (card.type === "typed" && state.deckId) {
      const wrap = el(`<div class="stack-sm fc-typed"></div>`);
      const inp = el(`<input class="input" aria-label="Type your answer" placeholder="Type your answer…" autocomplete="off" />`);
      const chk = el(`<button class="btn" type="button">Check</button>`);
      const out = el(`<div role="status" aria-live="polite"></div>`);
      chk.addEventListener("click", async () => {
        chk.disabled = true; chk.textContent = "Checking…";
        try {
          const r = await window.api.post(`/flashcards/decks/${encodeURIComponent(state.deckId)}/check`, { card_id: card.cid, answer: inp.value });
          out.textContent = (r.correct ? "✓ " : "✗ ") + r.feedback + (r.provisional ? " (provisional)" : "") + " ";
          const l = window.ui.renderModelLabel(r.llm, { caption: "Checked by" }); if (l) out.append(l);
          if (!state.flipped) doFlip();
        } catch (e) { out.textContent = e.message; }
        chk.disabled = false; chk.textContent = "Check";
      });
      inp.addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); chk.click(); } e.stopPropagation(); });
      wrap.append(inp, chk, out);
      main.append(wrap);
    }

    // Rating row
    const rate = el(`<div class="rate" role="group" aria-label="How well did you know this card"></div>`);
    Object.entries(RATINGS).forEach(([key, meta]) => {
      const active = state.knowledge[card.id] === key;
      const b = el(`<button type="button" class="btn${active ? "-accent" : ""}" aria-pressed="${active}">${esc(meta.label)}</button>`);
      b.title = `Key ${Object.keys(RATINGS).indexOf(key) + 1}`;
      b.addEventListener("click", () => {
        state.knowledge[card.id] = key;
        (state.undo = state.undo || []).push({ id: card.id, idx: state.index });
        if (state.deckId && card.cid) {
          state.pending.push({ card_id: card.cid, rating: key });
          const allRated = Object.keys(state.knowledge).length >= state.cards.length;
          if (state.pending.length >= 10 || allRated) flushReviews();
        }
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
    if (state.session && Object.keys(state.knowledge).length >= state.cards.length && state.cards.length) {
      const k = Object.values(state.knowledge); const kept = k.filter((v) => v === "good" || v === "easy").length;
      const mins = Math.max(1, Math.round((Date.now() - state.session.start) / 60000));
      main.append(el(`<div class="panel-soft" role="status"><strong>Session complete</strong> · ${k.length} cards · retention ${Math.round(100 * kept / k.length)}% · ${mins} min</div>`));
    }
    if (state.deckId) main.append(el(`<div class="hint" id="fc-due" aria-live="polite">${esc(dueText())}</div>`));
    main.append(el(`<div class="hint">Shortcuts: Space flips \u00b7 E edit \u00b7 S star \u00b7 Z undo rating \u00b7 1 Again \u00b7 2 Hard \u00b7 3 Good \u00b7 4 Easy</div>`));
    const llmLine = window.ui.renderModelLabel(state.llm, { stored: !!state.stored });
    if (llmLine) main.append(llmLine);
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
    const counts = { again: 0, hard: 0, good: 0, easy: 0 };
    Object.values(state.knowledge).forEach((v) => { if (v in counts) counts[v] += 1; });
    const rated = counts.again + counts.hard + counts.good + counts.easy;
    const row = el(`<div class="panel-soft row-wrap between"></div>`);
    const left = el(`<div class="row-wrap"></div>`);
    left.append(
      el(`<span class="badge badge-accent">Good/Easy ${counts.good + counts.easy}</span>`),
      el(`<span class="badge">Hard ${counts.hard}</span>`),
      el(`<span class="badge">Again ${counts.again}</span>`)
    );
    row.append(left, el(`<span class="muted">${rated} of ${state.cards.length} rated</span>`));
    return row;
  }

  function unmount() {
    document.removeEventListener("keydown", onKey);
    flushReviews();
    state = null;
  }

  window.Views.flashcards = { accent: "cards", title: "Flashcards", mount, unmount };
})();
