/* ============================================================
   Flashcard deck manager — window.FlashDecks
   "My decks": list / create / rename / duplicate / delete decks, then edit one: search + tag filter, add / edit /
   delete cards (basic, reversed, cloze, multiple choice, type-the-answer) with undo, bulk paste, CSV import,
   CSV / Anki-TSV export, AI cards from pasted text. All state lives server-side (/api/flashcards/decks).
     FlashDecks.render(container, { onStudy(deckId, mode, tag) })
   ============================================================ */
(() => {
  "use strict";
  const { el, clear, esc, spinner, toast } = window.ui;
  const api = () => window.api;
  const TYPES = { basic: "Basic", reversed: "Reversed (two-way)", cloze: "Cloze {{c1::…}}", mcq: "Multiple choice", typed: "Type the answer" };
  const base = "/flashcards/decks";
  const when = (iso) => (iso ? new Date(iso).toLocaleDateString() : "never");

  function render(container, hooks) {
    clear(container);
    const root = el(`<div class="stack"></div>`);
    container.append(root);
    list();

    async function list() {
      clear(root);
      root.append(spinner("Loading your decks…"));
      let data;
      try { data = await api().get(base); } catch (e) { clear(root); root.append(window.ui.errorBlock(e, list)); return; }
      clear(root);
      const head = el(`<div class="between"><div class="section-head" style="margin:0"><h3>My decks</h3><span>${(data.decks || []).length} of ${data.limit || 200}</span></div></div>`);
      const add = el(`<button class="btn-accent" type="button">+ New deck</button>`);
      add.addEventListener("click", async () => {
        const title = (window.prompt ? "" : "") || "New deck";
        try { const r = await api().post(base, { title }); toast("Deck created.", "ok"); edit(r.id); } catch (e) { toast(e.message, "error"); }
      });
      head.append(add);
      root.append(head);
      if (data.persistence === false) { root.append(el(`<div class="panel-soft">Decks need history to be enabled on this server.</div>`)); return; }
      if (!data.decks.length) { root.append(el(`<div class="empty"><div class="big">🗂</div><div>No decks yet. Create one and type your own cards, paste notes for AI cards, or generate from a subject in Setup.</div></div>`)); }
      data.decks.forEach((d) => root.append(deckRow(d)));
    }

    function deckRow(d) {
      const row = el(`<div class="panel-soft stack-sm"></div>`);
      row.append(el(`<div class="between" style="gap:.5rem"><strong style="overflow-wrap:anywhere;min-width:0">${esc(d.title || "Untitled")}</strong><span class="badge ${d.due ? "badge-accent" : ""}">${d.due} due</span></div>`));
      row.append(el(`<div class="muted" style="font-size:var(--fs-sm)">${d.cards} cards · ${d.new} new · last studied ${esc(when(d.last_studied))}</div>`));
      const bar = el(`<div class="row row-wrap" style="gap:.4rem"></div>`);
      const btn = (label, fn, cls) => { const b = el(`<button type="button" class="${cls || "btn"}" style="padding:.35rem .75rem;font-size:var(--fs-sm)">${window.ui.iconHtml("play")}<span>${label}</span></button>`); b.addEventListener("click", fn); bar.append(b); return b; };
      const ib = (icon, label, fn, danger) => { const b = window.ui.iconBtn({ icon, label, danger, cls: "ib-outline", onClick: fn }); bar.append(b); return b; };
      btn("Study", () => hooks.onStudy(d.id, "due"), "btn-accent");
      ib("shuffle", "Cram all cards", () => hooks.onStudy(d.id, "cram"));
      ib("star", "Study starred cards", () => hooks.onStudy(d.id, "starred"));
      ib("edit", "Edit cards", () => edit(d.id));
      ib("copy", "Duplicate deck", async () => { try { await api().post(`${base}/${d.id}/duplicate`, {}); toast("Duplicated.", "ok"); list(); } catch (e) { toast(e.message, "error"); } });
      const disarm = () => { del.dataset.armed = ""; del.classList.remove("has-text"); del.querySelector(".icon-btn-text")?.remove(); window.ui.setIconBtn(del, { label: "Delete deck" }); };
      const del = ib("trash", "Delete deck", () => {
        if (del.dataset.armed !== "1") {
          del.dataset.armed = "1"; del.classList.add("has-text");
          const t = document.createElement("span"); t.className = "icon-btn-text"; t.textContent = "Confirm delete"; del.append(t);
          window.ui.setIconBtn(del, { label: "Confirm delete deck" }); setTimeout(disarm, 4000); return;
        }
        api().del(`${base}/${d.id}`).then(() => { toast("Deck deleted.", "ok"); list(); }).catch((e) => toast(e.message, "error"));
      }, true);
      row.append(bar);
      return row;
    }

    /* ---------- one deck ---------- */
    async function edit(id, q, tag) {
      clear(root);
      root.append(spinner("Opening deck…"));
      let deck;
      try { deck = await api().get(`${base}/${id}` + (q || tag ? `?${new URLSearchParams(Object.assign({}, q ? { q } : {}, tag ? { tag } : {}))}` : "")); }
      catch (e) { clear(root); root.append(window.ui.errorBlock(e, () => edit(id, q, tag))); return; }
      clear(root);
      const back = el(`<button class="btn" type="button" style="align-self:flex-start">← All decks</button>`);
      back.addEventListener("click", list);
      const title = el(`<input class="input" aria-label="Deck name" value="${esc(deck.title || "")}" maxlength="120" />`);
      title.addEventListener("change", async () => { try { await api().patch(`${base}/${id}`, { title: title.value.trim() || "Untitled" }); toast("Renamed.", "ok"); } catch (e) { toast(e.message, "error"); } });
      root.append(back, el(`<label class="label">Deck name</label>`), title);

      // search + tags
      const bar = el(`<div class="row row-wrap" style="gap:.5rem"></div>`);
      const search = el(`<input class="input" type="search" placeholder="Search cards…" aria-label="Search cards" value="${esc(q || "")}" style="flex:1;min-width:10rem" />`);
      const tagSel = el(`<select class="select" aria-label="Filter by tag"><option value="">All tags</option>${(deck.tags || []).map((t) => `<option ${t === tag ? "selected" : ""}>${esc(t)}</option>`).join("")}</select>`);
      let timer; search.addEventListener("input", () => { clearTimeout(timer); timer = setTimeout(() => edit(id, search.value.trim(), tagSel.value), 350); });
      tagSel.addEventListener("change", () => edit(id, search.value.trim(), tagSel.value));
      bar.append(search, tagSel);
      const study = el(`<button class="btn-accent" type="button">Study this deck</button>`);
      study.addEventListener("click", () => hooks.onStudy(id, "due", tagSel.value || undefined));
      bar.append(study);
      root.append(bar);

      const more = el(`<div class="row row-wrap" style="gap:.5rem;align-items:center"></div>`);
      const moreBtn = el(`<button class="btn" type="button">✨ Generate 5 more like these</button>`);
      const moreNote = el(`<span class="hint" role="status" aria-live="polite"></span>`);
      moreBtn.addEventListener("click", async () => {
        moreBtn.disabled = true; moreBtn.textContent = "Writing cards…";
        try {
          const r = await api().post(`${base}/${id}/similar`, { num_cards: 5 });
          moreNote.textContent = `Added ${r.added.length} new cards (${r.duplicates_skipped} duplicates skipped). `;
          const l = window.ui.renderModelLabel(r.llm, { caption: "Cards" }); if (l) moreNote.append(l);
          setTimeout(() => edit(id), 1800);
        } catch (e) { moreNote.textContent = e.message; moreBtn.disabled = false; moreBtn.textContent = "✨ Generate 5 more like these"; }
      });
      more.append(moreBtn, moreNote);
      root.append(more);
      root.append(adder(id, deck));
      root.append(importer(id, deck));

      const cards = el(`<div class="stack-sm" aria-live="polite"></div>`);
      cards.append(el(`<div class="muted">${deck.cards.length} shown of ${deck.total_cards}</div>`));
      deck.cards.forEach((c) => cards.append(cardRow(id, c, () => edit(id, search.value.trim(), tagSel.value))));
      if (!deck.cards.length) cards.append(el(`<div class="empty"><div>No cards ${q || tag ? "match" : "yet"}.</div></div>`));
      root.append(cards);
    }

    function cardRow(id, c, refresh) {
      const row = el(`<div class="panel-soft stack-sm"></div>`);
      row.append(el(`<div class="between" style="gap:.5rem"><span class="badge">${esc(TYPES[c.type || "basic"] || c.type)}</span><span class="muted" style="font-size:var(--fs-sm)">${esc((c.tags || []).join(" "))}</span></div>`));
      row.append(el(`<div style="overflow-wrap:anywhere"><strong>${esc(c.front)}</strong></div>`));
      if (c.type === "mcq") row.append(el(`<div class="muted" style="overflow-wrap:anywhere">${(c.options || []).map((o, i) => `${i === c.answer_index ? "✓ " : ""}${esc(o)}`).join(" · ")}</div>`));
      else if (c.back) row.append(el(`<div class="muted" style="overflow-wrap:anywhere">${esc(c.back)}</div>`));
      const tools = el(`<div class="row row-wrap" style="gap:.4rem"></div>`);
      const b = (icon, fn, aria, o = {}) => { const x = window.ui.iconBtn(Object.assign({ icon, label: aria, cls: "ib-outline", onClick: fn }, o)); tools.append(x); return x; };
      b("star", async () => { try { await api().patch(`${base}/${id}/cards/${c.id}`, { starred: !c.starred }); refresh(); } catch (e) { toast(e.message, "error"); } }, c.starred ? "Unstar card" : "Star card", { pressed: !!c.starred });
      b("edit", () => {
        row.replaceWith(editor(id, c, refresh));
      }, "Edit card");
      b("trash", async () => {
        try {
          const r = await api().del(`${base}/${id}/cards/${c.id}`);
          toast("Card deleted.", "ok", 8000, { label: "Undo", fn: async () => { try { await api().post(`${base}/${id}/cards`, { cards: [r.removed] }); refresh(); } catch (e) { toast(e.message, "error"); } } });
          undoBar(r.removed, id, refresh);
          refresh();
        } catch (e) { toast(e.message, "error"); }
      }, "Delete card", { danger: true });
      row.append(tools);
      return row;
    }

    // Undo also as an inline button (toast actions are not supported everywhere)
    function undoBar(card, id, refresh) {
      const bar = el(`<div class="panel-soft row between" role="status"><span>Card deleted.</span></div>`);
      const u = window.ui.iconBtn({ icon: "undo", label: "Undo delete", cls: "ib-outline" });
      u.addEventListener("click", async () => { u.disabled = true; try { await api().post(`${base}/${id}/cards`, { cards: [card] }); bar.remove(); refresh(); } catch (e) { toast(e.message, "error"); u.disabled = false; } });
      bar.append(u);
      root.prepend(bar);
      setTimeout(() => bar.remove(), 12000);
    }

    function cardForm(c) {
      c = c || {};
      const f = el(`<div class="stack-sm"></div>`);
      const type = el(`<select class="select" aria-label="Card type">${Object.entries(TYPES).map(([k, v]) => `<option value="${k}" ${k === (c.type || "basic") ? "selected" : ""}>${v}</option>`).join("")}</select>`);
      const front = el(`<textarea class="textarea" rows="2" aria-label="Front" placeholder="Front / question (cloze: use {{c1::hidden}})">${esc(c.front || "")}</textarea>`);
      const back = el(`<textarea class="textarea" rows="2" aria-label="Back" placeholder="Back / answer (optional for cloze)">${esc(c.back || "")}</textarea>`);
      const opts = el(`<textarea class="textarea" rows="3" aria-label="Options" placeholder="Options, one per line (first line = correct answer)">${esc((c.options || []).length ? [c.options[c.answer_index || 0], ...c.options.filter((_, i) => i !== (c.answer_index || 0))].join("\n") : "")}</textarea>`);
      const tags = el(`<input class="input" aria-label="Tags" placeholder="Tags (space separated)" value="${esc((c.tags || []).join(" "))}" />`);
      const hint = el(`<input class="input" aria-label="Hint" placeholder="Hint (optional)" value="${esc(c.hint || "")}" />`);
      const expl = el(`<input class="input" aria-label="Explanation" placeholder="Explanation (optional)" value="${esc(c.explanation || "")}" />`);
      const sync = () => { opts.hidden = type.value !== "mcq"; back.hidden = type.value === "mcq"; };
      type.addEventListener("change", sync); sync();
      f.append(type, front, back, opts, tags, hint, expl);
      f.value = () => {
        const o = { type: type.value, front: front.value.trim(), back: back.value.trim(), tags: tags.value.split(/[\s,]+/).filter(Boolean), hint: hint.value.trim(), explanation: expl.value.trim() };
        if (type.value === "mcq") { o.options = opts.value.split("\n").map((x) => x.trim()).filter(Boolean); o.answer_index = 0; delete o.back; }
        return o;
      };
      f.reset = () => { front.value = ""; back.value = ""; opts.value = ""; front.focus(); };
      return f;
    }

    function adder(id, deck) {
      const d = el(`<details class="panel-soft"><summary style="cursor:pointer;font-weight:600">Add a card</summary></details>`);
      const form = cardForm(); const save = el(`<button class="btn-accent" type="button">Add card</button>`);
      const note = el(`<div class="hint" role="status" aria-live="polite"></div>`);
      save.addEventListener("click", async () => {
        save.disabled = true;
        try {
          const r = await api().post(`${base}/${id}/cards`, { cards: [form.value()] });
          note.textContent = r.duplicates.length ? `Added - note: looks like a duplicate of “${r.duplicates[0].match_front}” (${r.duplicates[0].how}).` : "Added.";
          form.reset();
          const list2 = root.querySelector("[aria-live=polite] .muted"); if (list2) list2.textContent = `${r.total} cards`;
        } catch (e) { note.textContent = e.message; }
        save.disabled = false;
      });
      d.append(form, save, note);
      return d;
    }

    function editor(id, c, refresh) {
      const box = el(`<div class="panel-soft stack-sm"></div>`);
      const form = cardForm(c);
      const row = el(`<div class="row" style="gap:.4rem"></div>`);
      const ok = el(`<button class="btn-accent" type="button">Save</button>`); const cancel = el(`<button class="btn" type="button">Cancel</button>`);
      ok.addEventListener("click", async () => { try { await api().patch(`${base}/${id}/cards/${c.id}`, form.value()); refresh(); } catch (e) { toast(e.message, "error"); } });
      cancel.addEventListener("click", refresh);
      row.append(ok, cancel); box.append(form, row);
      return box;
    }

    function importer(id, deck) {
      const d = el(`<details class="panel-soft"><summary style="cursor:pointer;font-weight:600">Import · export · AI from notes</summary></details>`);
      const body = el(`<div class="stack-sm"></div>`);
      const ta = el(`<textarea class="textarea" rows="5" aria-label="Paste cards" placeholder="One card per line:  front :: back&#10;or  Q: …  then  A: …&#10;or lines containing {{c1::cloze}}"></textarea>`);
      const file = el(`<input type="file" accept=".csv,.tsv,.txt,text/csv" aria-label="Import CSV file" />`);
      const go = el(`<button class="btn-accent" type="button">Import</button>`);
      const note = el(`<div class="hint" role="status" aria-live="polite"></div>`);
      let fmt = "lines";
      file.addEventListener("change", async () => { const f = file.files[0]; if (f) { ta.value = await f.text(); fmt = "csv"; } });
      go.addEventListener("click", async () => {
        go.disabled = true;
        try { const r = await api().post(`${base}/${id}/import`, { text: ta.value, format: fmt, skip_duplicates: true }); note.textContent = `Imported ${r.added.length} of ${r.parsed} (${r.duplicates.length} duplicates skipped, ${r.skipped_invalid} invalid).`; edit(id); }
        catch (e) { note.textContent = e.message; }
        go.disabled = false;
      });
      const ex = el(`<div class="row row-wrap" style="gap:.4rem"></div>`);
      [["csv", "Export CSV"], ["tsv", "Export for Anki (TSV)"], ["json", "Export JSON"]].forEach(([f, label]) => {
        const b = el(`<button class="btn" type="button">${label}</button>`);
        b.addEventListener("click", async () => {
          try {
            const text = await api().get(`${base}/${id}/export?format=${f}`);
            const url = URL.createObjectURL(new Blob([typeof text === "string" ? text : JSON.stringify(text, null, 1)], { type: "text/plain" }));
            const a = document.createElement("a"); a.href = url; a.download = `${(deck.title || "deck").replace(/[^\w-]+/g, "-")}.${f === "tsv" ? "txt" : f}`; document.body.appendChild(a); a.click(); a.remove();
            setTimeout(() => URL.revokeObjectURL(url), 2000);
          } catch (e) { toast(e.message, "error"); }
        });
        ex.append(b);
      });
      // AI cards from pasted notes
      const notes = el(`<textarea class="textarea" rows="4" aria-label="Notes for AI cards" placeholder="Paste notes here - the AI turns them into cards (the full text is used)."></textarea>`);
      const kind = el(`<select class="select" aria-label="AI card type">${Object.entries(TYPES).map(([k, v]) => `<option value="${k}">${v}</option>`).join("")}</select>`);
      const gen = el(`<button class="btn" type="button">Generate cards from notes</button>`);
      const gnote = el(`<div class="hint" role="status" aria-live="polite"></div>`);
      gen.addEventListener("click", async () => {
        if (!notes.value.trim()) { gnote.textContent = "Paste some notes first."; return; }
        gen.disabled = true; gen.textContent = "Writing cards…";
        try {
          const r = await api().post("/flashcards/text", { text: notes.value, card_type: kind.value, num_cards: 10, title: (deck.title || "Notes") + " (AI)" });
          gnote.textContent = `Created deck “${r.title}” with ${r.cards.length} cards. `;
          const l = window.ui.renderModelLabel(r.llm, { caption: "Cards" }); if (l) gnote.append(l);
        } catch (e) { gnote.textContent = e.message; }
        gen.disabled = false; gen.textContent = "Generate cards from notes";
      });
      body.append(ta, file, go, note, ex, el(`<hr>`), notes, kind, gen, gnote);
      d.append(body);
      return d;
    }
  }

  window.FlashDecks = { render };
})();
