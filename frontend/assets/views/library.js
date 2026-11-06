/* ============================================================
   Library view — window.Views.library
   A subject-by-subject overview: files + index health, everything the user did per subject, a grouped activity
   timeline and the AI model chain that produced the answers. Read-only; data from GET /api/library (one cached
   aggregate) and GET /api/library/{subject}/timeline (paginated). No model is called here.
   ============================================================ */
(() => {
  "use strict";
  const { el, clear, esc, empty, toast } = window.ui;

  const KIND_ICON = {
    chats: "chat", quizzes: "help", attempts: "target", flashcards: "layers", exams: "graduation-cap",
    cheatsheets: "list", mindmaps: "network", plans: "calendar", summaries: "book", predictions: "sparkle",
    teachbacks: "speaker", scans: "camera", uploads: "upload",
  };
  const KIND_NAME = {
    chats: "Chats", quizzes: "Quizzes", attempts: "Attempts", flashcards: "Flashcards", exams: "Exams",
    cheatsheets: "Cheat sheets", mindmaps: "Mind maps", plans: "Study plans", summaries: "Summaries",
    predictions: "Predictions", teachbacks: "Teach-backs", scans: "Scan notes", uploads: "Uploads",
  };
  // count key (card.counts) -> [label, icon, kind]
  const COUNT_CHIPS = [
    ["chats", "Chats", "chat"], ["quizzes", "Quizzes", "help"], ["flashcard_decks", "Decks", "layers"],
    ["exams", "Exams", "graduation-cap"], ["cheatsheets", "Cheat sheets", "list"], ["mindmaps", "Mind maps", "network"],
    ["plans", "Plans", "calendar"], ["teachbacks", "Teach-backs", "speaker"], ["scans", "Scans", "camera"],
    ["predictions", "Predictions", "sparkle"], ["attempts", "Attempts", "target"],
  ];
  const ic = (n) => window.ui.iconHtml(n);
  const OTHER = "__other__";

  let root = null;
  let data = null;
  let detail = null;              // { name, key, card, kind, items, cursor, hasMore, loading, meta }
  let seq = 0;                    // drops stale responses after navigation / retry

  /* ---------- formatting ---------- */
  const fmtBytes = (n) => {
    n = Number(n) || 0;
    if (n >= 1048576) return (n / 1048576).toFixed(n >= 10485760 ? 0 : 1) + " MB";
    if (n >= 1024) return Math.round(n / 1024) + " KB";
    return n ? n + " B" : "—";
  };
  const fmtDate = (iso) => { try { return iso ? new Date(iso).toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric" }) : ""; } catch (_) { return ""; } };
  const fmtTime = (iso) => { try { return new Date(iso).toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" }); } catch (_) { return ""; } };
  const ago = (iso) => {
    if (!iso) return "no activity yet";
    const s = Math.max(0, (Date.now() - new Date(iso).getTime()) / 1000);
    if (s < 90) return "just now";
    if (s < 3600) return Math.round(s / 60) + " min ago";
    if (s < 86400) return Math.round(s / 3600) + " h ago";
    if (s < 86400 * 14) return Math.round(s / 86400) + " d ago";
    return fmtDate(iso);
  };
  const plural = (n, w) => `${n} ${w}${n === 1 ? "" : "s"}`;
  const modelName = (m) => `${m.provider} · ${m.model}${m.step ? ` · step ${m.step}` : ""}`;
  function groupOf(iso) {
    const d = new Date(iso), n = new Date();
    const day = (x) => new Date(x.getFullYear(), x.getMonth(), x.getDate()).getTime();
    const diff = Math.round((day(n) - day(d)) / 86400000);
    return diff <= 0 ? "Today" : diff === 1 ? "Yesterday" : diff < 7 ? "Earlier this week" : "Older";
  }

  /* ---------- shared pieces ---------- */
  function modelBars(models, opts = {}) {
    const wrap = el(`<ul class="library-bars" aria-label="${esc(opts.label || "AI answers by model")}"></ul>`);
    const max = Math.max(1, ...models.map((m) => m.count));
    const total = models.reduce((a, m) => a + m.count, 0) || 1;
    models.slice(0, opts.limit || 8).forEach((m) => {
      wrap.append(el(`<li class="library-bar-row">
        <span class="library-bar-name" title="${esc(modelName(m))}">${esc(modelName(m))}</span>
        <span class="library-bar" role="img" aria-label="${m.count} answers, ${Math.round(100 * m.count / total)} percent"><i style="width:${Math.max(4, Math.round(100 * m.count / max))}%"></i></span>
        <span class="library-bar-n">${m.count}</span></li>`));
    });
    return wrap;
  }
  function modelLabel(llm) {
    if (!llm) return null;
    return window.ui.renderModelLabel(llm, { stored: true, quiet: true });
  }
  function skeleton() {
    return el(`<div class="library-skel" role="status" aria-label="Loading your library">
      <div class="library-skel-row"><div class="skeleton"></div><div class="skeleton"></div><div class="skeleton"></div><div class="skeleton"></div></div>
      <div class="library-skel-grid"><div class="skeleton"></div><div class="skeleton"></div><div class="skeleton"></div></div></div>`);
  }
  function fail(e, retry) {
    const box = window.ui.errorBlock ? window.ui.errorBlock(e, retry) : el(`<div class="panel stack-sm" role="alert"><p>${esc((e && e.message) || "Something went wrong.")}</p><button class="btn" type="button">Retry</button></div>`);
    return box;
  }

  /* ---------- overview ---------- */
  function summarySection(s) {
    const sec = el(`<section class="library-summary panel stack" aria-labelledby="lib-sum-h"></section>`);
    sec.append(el(`<div class="between"><div class="section-head" style="margin:0"><h3 id="lib-sum-h">All your subjects</h3><span>${plural(s.subjects, "subject")}</span></div>
      <button class="icon-btn" type="button" id="libRefresh" aria-label="Refresh library" title="Refresh">${ic("refresh")}</button></div>`));
    const tiles = el(`<div class="library-tiles"></div>`);
    [[s.files, "Files"], [s.pages || "—", "Pages"], [s.chunks || "—", "Index chunks"], [s.activity_total, "Things you did"],
      [s.storage && s.storage.owner_mb != null ? (s.storage.owner_mb < 0.1 ? "<0.1" : s.storage.owner_mb) + " MB" : "—", "Stored"]].forEach(([v, l]) =>
      tiles.append(el(`<div class="panel-soft library-tile"><b>${esc(v)}</b><span class="muted">${esc(l)}</span></div>`)));
    sec.append(tiles);
    const kinds = Object.entries(s.activity_by_kind || {});
    if (kinds.length) {
      const row = el(`<ul class="library-kinds" aria-label="Activity by kind"></ul>`);
      kinds.forEach(([k, n]) => row.append(el(`<li class="library-kind">${ic(KIND_ICON[k] || "info")}<span>${esc(KIND_NAME[k] || k)}</span><b>${n}</b></li>`)));
      sec.append(row);
    }
    if ((s.models || []).length) {
      const box = el(`<div class="library-models stack-sm"><h4>Which AI answered</h4></div>`);
      const prov = el(`<ul class="library-prov"></ul>`);
      (s.providers || []).forEach((p) => prov.append(el(`<li><span class="library-prov-dot library-prov-${esc(p.provider.replace(/[^a-z0-9]/gi, ""))}"></span>${esc(p.provider)} <b>${p.pct}%</b> <span class="muted">(${p.count})</span></li>`)));
      box.append(prov, modelBars(s.models, { limit: 6 }));
      const ch = s.chain || {};
      if (ch.answers) box.append(el(`<p class="muted library-note">${plural(ch.answers, "AI answer")}: ${ch.step1} from the first model in the chain, ${ch.later_steps} fell through to a later step${ch.reasons && ch.reasons.length ? " (" + ch.reasons.map((r) => `${esc(r.reason)} ×${r.count}`).join(", ") + ")" : ""}.</p>`));
      sec.append(box);
    }
    const foot = [];
    if (s.hidden && s.hidden.count) foot.push(`Standard subjects hidden: <b>${s.hidden.count}</b> (${esc(s.hidden.names.join(", "))}) · <a href="#subjects" data-restore-std>Restore</a>`);
    if (s.storage && s.storage.near_cap) foot.push(`<span class="library-warn">Storage is nearly full. Remove files you no longer need.</span>`);
    if (s.truncated) foot.push(`Counts and the timeline cover your newest 100 items of each kind.`);
    if (foot.length) sec.append(el(`<p class="muted library-note">${foot.join("<br>")}</p>`));
    const rs = sec.querySelector("[data-restore-std]");
    if (rs) rs.addEventListener("click", async (e) => {      // the Restore link calls the API directly
      e.preventDefault();
      try {
        await window.api.post("/subjects/restore-standard", {});
        refreshAfterEdit();
        toast("Standard subjects restored.", "ok");
        load(true);
      } catch (err) { toast((err && err.message) || "Could not restore.", "error"); }
    });
    return sec;
  }

  function fileRow(f) {
    const idx = f.text_only ? `<span class="library-flag" title="The original PDF was too large to keep; its text is still searchable">text index only</span>` : "";
    const meta = [f.pages ? plural(f.pages, "page") : null, f.bytes ? fmtBytes(f.bytes) : null, f.added ? "added " + fmtDate(f.added) : "included with the app"].filter(Boolean).join(" · ");
    return el(`<li class="library-file">${ic("book")}<div><span class="library-file-name" title="${esc(f.name)}">${esc(f.name)}</span><span class="muted library-file-meta">${esc(meta)}</span></div>${idx}</li>`);
  }
  function indexLine(ix) {
    const pct = ix.n_chunks ? Math.round(100 * Math.min(ix.n_embedded, ix.n_chunks) / ix.n_chunks) : 0;
    const txt = ix.state === "complete" ? `Fully indexed · ${plural(ix.n_chunks, "chunk")}`
      : ix.state === "partial" ? `Partly indexed · ${ix.n_embedded}/${ix.n_chunks} chunks`
      : ix.state === "none" ? "Not indexed yet" : "Index status not available";
    return el(`<div class="library-index library-index-${esc(ix.state)}"><span>${esc(txt)}</span>${ix.state === "partial" || ix.state === "complete" ? `<span class="library-meter" role="img" aria-label="${pct} percent indexed"><i style="width:${pct}%"></i></span>` : ""}</div>`);
  }

  function subjectCard(c) {
    const card = el(`<article class="panel library-card" aria-labelledby="lc-${seq}-${esc(c.key.replace(/\W/g, "_"))}"></article>`);
    card.append(el(`<header class="library-card-head"><h3 id="lc-${seq}-${esc(c.key.replace(/\W/g, "_"))}">${esc(c.name)}</h3><span class="badge ${c.own ? "badge-accent" : ""}">${c.own ? "Yours" : "Standard"}</span></header>`));
    const t = c.totals || {};
    card.append(el(`<p class="library-line">${plural(t.files || 0, "file")}${t.pages ? " · " + plural(t.pages, "page") : ""}${t.chunks ? " · " + plural(t.chunks, "chunk") : ""}${t.bytes && c.own ? " · " + fmtBytes(t.bytes) : ""}</p>`));
    card.append(indexLine(c.index));
    const files = el(`<ul class="library-files" aria-label="Files in ${esc(c.name)}"></ul>`);
    (c.files || []).slice(0, 3).forEach((f) => files.append(fileRow(f)));
    if (!(c.files || []).length) files.append(el(`<li class="muted library-file">No file list available${c.own ? "" : " for this subject"}.</li>`));
    if ((c.files || []).length > 3) files.append(el(`<li class="muted library-more">+ ${(c.files || []).length - 3} more in details</li>`));
    card.append(files);
    const m = c.mastery || {};
    if (m.pct != null) {
      card.append(el(`<div class="library-mastery"><span>Mastery <b>${m.pct}%</b></span><span class="library-meter" role="img" aria-label="Mastery ${m.pct} percent"><i style="width:${m.pct}%"></i></span></div>`));
      if ((m.weak || []).length) card.append(el(`<p class="library-weak"><span class="muted">Weakest:</span> ${m.weak.map((w) => `<span class="badge">${esc(w.topic)} ${w.mastery}%</span>`).join(" ")}</p>`));
    }
    const counts = c.counts || {};
    const chips = el(`<ul class="library-counts" aria-label="What you did in ${esc(c.name)}"></ul>`);
    COUNT_CHIPS.forEach(([k, label, icon]) => {
      if (!counts[k]) return;
      const extra = k === "chats" && counts.chat_messages ? ` <small>${plural(counts.chat_messages, "msg")}</small>` : k === "flashcard_decks" && counts.flashcard_cards ? ` <small>${plural(counts.flashcard_cards, "card")}</small>` : "";
      chips.append(el(`<li class="library-count" title="${esc(label)}">${ic(icon)}<b>${counts[k]}</b> ${esc(label)}${extra}</li>`));
    });
    if (!chips.children.length) chips.append(el(`<li class="muted library-count">No activity yet</li>`));
    card.append(chips);
    const foot = el(`<footer class="library-card-foot"><span class="muted">Last activity: ${esc(ago(c.last_activity))}</span></footer>`);
    const btn = el(`<button class="btn" type="button" aria-label="Open details for ${esc(c.name)}">Details ${ic("chevron-right")}</button>`);
    btn.dataset.key = c.key;
    btn.addEventListener("click", () => openDetail(c));
    foot.append(btn);
    card.append(foot);
    return card;
  }

  function renderOverview() {
    clear(root);
    detail = null;
    const s = data.summary;
    if (!data.subjects.length && !data.other) {
      root.append(el(`<div class="panel stack library-empty"><h3>Your library is empty</h3><p class="muted">Upload a PDF as a subject and your files, chats, quizzes and more will be organised here, subject by subject.</p><div><a class="btn-accent btn-lg" href="#subjects">Add a subject</a></div></div>`));
      return;
    }
    root.append(summarySection(s));
    const grid = el(`<div class="library-grid"></div>`);
    data.subjects.forEach((c) => grid.append(subjectCard(c)));
    if (data.other) {
      const o = data.other;
      const card = el(`<article class="panel library-card"><header class="library-card-head"><h3>${esc(o.name)}</h3><span class="badge">Other</span></header>
        <p class="muted library-line">Topic chats, scans and quizzes that are not tied to one of your subjects.</p>
        <p class="muted">${Object.entries(o.counts || {}).map(([k, n]) => `${n} ${esc((KIND_NAME[k] || k).toLowerCase())}`).join(" · ")}</p>
        <footer class="library-card-foot"><span class="muted">Last activity: ${esc(ago(o.last_activity))}</span></footer></article>`);
      const btn = el(`<button class="btn" type="button" aria-label="Open details for activity not tied to a subject">Details ${ic("chevron-right")}</button>`);
      btn.dataset.key = OTHER;
      btn.addEventListener("click", () => openDetail({ name: o.name, key: OTHER, own: false, files: [], totals: {}, index: { state: "unknown", n_chunks: 0, n_embedded: 0 }, counts: o.counts || {}, mastery: {} }));
      card.querySelector("footer").append(btn);
      grid.append(card);
    }
    root.append(grid);
    const rf = root.querySelector("#libRefresh");
    if (rf) rf.addEventListener("click", () => load(true));
  }

  /* ---------- detail ---------- */
  function openItem(ev) {
    if (ev.item_id) window.ui.openArtifact(ev.kind, ev.item_id, ev.route);
    else location.hash = ev.route;
  }
  function eventRow(ev) {
    const li = el(`<li class="library-event"></li>`);
    const body = el(`<div class="library-event-body"><span class="library-event-ic">${ic(KIND_ICON[ev.kind] || "info")}</span>
      <div class="library-event-text"><span class="library-event-title">${esc(ev.title)}</span><span class="library-event-sum">${esc(ev.summary)}</span></div>
      <time class="muted library-event-time" datetime="${esc(ev.ts)}">${esc(fmtTime(ev.ts))}</time></div>`);
    const lab = modelLabel(ev.llm);
    if (lab) body.querySelector(".library-event-text").append(lab);
    const open = el(`<button type="button" class="library-event-open" aria-label="${esc(ev.item_id ? "Open " : "Go to ")}${esc(ev.title)}">${ic("chevron-right")}</button>`);
    open.addEventListener("click", () => openItem(ev));
    li.append(body, open);
    li.addEventListener("click", (e) => { if (!e.target.closest("button")) openItem(ev); });
    return li;
  }
  function paintTimeline(box) {
    clear(box);
    if (!detail.items.length) {
      box.append(el(`<p class="muted library-empty-line">${detail.loading ? "Loading…" : detail.kind ? "Nothing of this kind yet." : "No activity recorded for this subject yet."}</p>`));
      return;
    }
    let group = null, ul = null;
    detail.items.forEach((ev) => {
      const g = groupOf(ev.ts);
      if (g !== group) {
        group = g;
        box.append(el(`<h4 class="library-group">${esc(g)}</h4>`));
        ul = el(`<ul class="library-events"></ul>`);
        box.append(ul);
      }
      ul.append(eventRow(ev));
    });
    if (detail.hasMore) {
      const more = el(`<button class="btn library-loadmore" type="button">${detail.loading ? "Loading…" : "Load more"}</button>`);
      more.disabled = detail.loading;
      more.addEventListener("click", () => loadTimeline(false));
      box.append(more);
    }
  }
  function weekStrip(w) {
    const d = (a, b) => (a === b ? "same as last week" : a > b ? `+${a - b} vs last week` : `${a - b} vs last week`);
    return el(`<div class="library-week" aria-label="This week compared with last week">
      <div class="panel-soft library-tile"><b>${w.this_week}</b><span class="muted">things done this week · ${esc(d(w.this_week, w.last_week))}</span></div>
      <div class="panel-soft library-tile"><b>${w.this_week_ai}</b><span class="muted">AI answers this week · ${esc(d(w.this_week_ai, w.last_week_ai))}</span></div>
      <div class="panel-soft library-tile"><b>${w.last_week}</b><span class="muted">things done last week</span></div></div>`);
  }
  function chainPanel(meta) {
    const p = el(`<section class="panel stack-sm" aria-labelledby="lib-chain-h"><div class="section-head" style="margin:0"><h3 id="lib-chain-h">Model chain</h3><span>which AI answered</span></div></section>`);
    const ch = meta.chain || { answers: 0 };
    if (!ch.answers) { p.append(el(`<p class="muted">No AI answers recorded for this subject yet.</p>`)); return p; }
    p.append(modelBars(meta.models, { limit: 10 }));
    p.append(el(`<p class="library-note">${plural(ch.answers, "AI answer")} in total: <b>${ch.step1}</b> came from the first model in the chain, <b>${ch.later_steps}</b> fell through to step 2–5${ch.later_steps && ch.reasons.length ? ". Why the earlier models were skipped: " + ch.reasons.map((r) => `${esc(r.reason)} ×${r.count}`).join(", ") : ""}.</p>`));
    return p;
  }
  /* ---------- editing (own subjects only; standard subjects stay read-only, they can only be hidden) ---------- */
  function refreshAfterEdit() {
    if (!window.api.invalidate) return;
    window.api.invalidate("/library");
    window.api.invalidate("/subjects");
  }
  // In-page confirm: swaps `btn` for "<question> Yes / No" inside `host`; runs `action` on Yes.
  function confirmInline(host, btn, question, yesLabel, action) {
    btn.hidden = true;
    const box = el(`<span class="row library-confirm" role="group" style="gap:.35rem;align-items:center"><span class="muted">${esc(question)}</span></span>`);
    const yes = el(`<button type="button" class="btn-accent">${esc(yesLabel)}</button>`);
    const no = el(`<button type="button" class="btn-ghost">Keep</button>`);
    box.append(yes, no);
    host.append(box);
    no.focus();
    const close = () => { box.remove(); btn.hidden = false; btn.focus(); };
    no.addEventListener("click", close);
    yes.addEventListener("click", async () => {
      yes.disabled = no.disabled = true;
      try { await action(); } catch (e) { toast((e && e.message) || "That didn't work.", "error"); close(); }
    });
  }
  function editControls(c, p, acts) {
    const enc = encodeURIComponent(c.name);
    const reopen = (key) => load(true, key);
    // per-file remove
    p.querySelectorAll(".library-file").forEach((li, i) => {
      const f = (c.files || [])[i];
      if (!f) return;
      const rm = window.ui.iconBtn({ icon: "close", label: "Remove " + f.name, danger: true });
      li.append(rm);
      rm.addEventListener("click", () => confirmInline(li, rm, "Remove this file?", "Remove", async () => {
        await window.api.del(`/subjects/${enc}/pdfs/${encodeURIComponent(f.name)}`);
        refreshAfterEdit();
        toast(`Removed “${f.name}”.`, "ok");
        reopen(c.key);
      }));
    });
    // re-index
    const re = window.ui.iconBtn({ icon: "refresh", label: "Re-index subject" });
    re.addEventListener("click", async () => {
      re.disabled = true;
      try {
        const r = await window.api.post(`/subjects/${enc}/reindex`, {});
        refreshAfterEdit();
        toast(r.indexing === "complete" ? "Re-indexed." : `Re-indexed; ${r.n_embedded}/${r.chunk_count} chunks embedded so far.`, "ok");
        reopen(c.key);
      } catch (e) { toast((e && e.message) || "Could not re-index.", "error"); re.disabled = false; }
    });
    // rename (inline form)
    const rn = window.ui.iconBtn({ icon: "edit", label: "Rename subject" });
    const form = el(`<form class="row library-rename" hidden style="gap:.4rem"><input class="input" type="text" maxlength="80" aria-label="New subject name" value="${esc(c.name)}"><button type="submit" class="btn-accent">Rename</button><button type="button" class="btn-ghost" data-cancel>Cancel</button></form>`);
    rn.addEventListener("click", () => { form.hidden = false; const i = form.querySelector("input"); i.focus(); i.select(); });
    form.querySelector("[data-cancel]").addEventListener("click", () => { form.hidden = true; rn.focus(); });
    form.addEventListener("submit", async (ev) => {
      ev.preventDefault();
      const nn = form.querySelector("input").value.trim();
      if (!nn || nn === c.name) { form.hidden = true; return; }
      try {
        const r = await window.api.patch(`/subjects/${enc}`, { name: nn });
        refreshAfterEdit();
        toast(`Renamed to “${r.subject}”.`, "ok");
        reopen(r.subject);
      } catch (e) { toast((e && e.message) || "Could not rename.", "error"); }
    });
    acts.prepend(re);
    acts.prepend(rn);
    p.append(form);
  }

  function filesPanel(c) {
    const p = el(`<section class="panel stack-sm" aria-labelledby="lib-files-h"><div class="between"><div class="section-head" style="margin:0"><h3 id="lib-files-h">Files</h3><span>${plural((c.files || []).length, "file")}</span></div></div></section>`);
    p.append(indexLine(c.index));
    const ul = el(`<ul class="library-files"></ul>`);
    (c.files || []).forEach((f) => ul.append(fileRow(f)));
    if (!(c.files || []).length) ul.append(el(`<li class="muted">No file list available.</li>`));
    p.append(ul);
    if (c.own) {
      const acts = el(`<div class="row library-actions"></div>`);
      acts.append(el(`<a class="btn" href="#subjects">${ic("plus")} Add or manage files</a>`));
      editControls(c, p, acts);
      if (c.index.state === "partial") {
        const cont = el(`<button class="btn" type="button">${ic("refresh")} Continue indexing</button>`);
        cont.addEventListener("click", async () => {
          cont.disabled = true;
          try {
            const r = await window.api.post(`/subjects/${encodeURIComponent(c.name)}/continue-index`, {});
            toast(r.complete ? "Indexing finished." : `Indexed ${r.n_embedded}/${r.n_chunks} chunks so far.`, "ok");
            window.api.invalidate && window.api.invalidate("/library");
            load(true, c.key);
          } catch (e) { toast(e.message || "Could not continue indexing.", "error"); cont.disabled = false; }
        });
        acts.append(cont);
      }
      p.append(acts);
    } else if (c.key !== OTHER) {
      // Standard (shared sample) subject: files are read-only; it can only be hidden for this account.
      const acts = el(`<div class="row library-actions"><span class="muted">Shared sample subject; read-only.</span></div>`);
      const hide = window.ui.iconBtn({ icon: "eye", label: "Remove " + c.name + " from my account" });
      acts.append(hide);
      hide.addEventListener("click", () => confirmInline(acts, hide, `Remove ${c.name} from your account? You can restore it any time.`, "Remove", async () => {
        await window.api.post(`/subjects/${encodeURIComponent(c.name)}/hide`, {});
        refreshAfterEdit();
        toast(`Removed “${c.name}” from your account.`, "ok");
        load(true);
      }));
      p.append(acts);
    }
    return p;
  }

  async function openDetail(c) {
    detail = { name: c.name, key: c.key, card: c, kind: "", items: [], cursor: null, hasMore: false, loading: true, meta: null, mine: ++seq };
    clear(root);
    const back = el(`<button class="btn library-back" type="button">${ic("chevron-left")} All subjects</button>`);
    back.addEventListener("click", () => {
      const k = detail && detail.key;
      seq++;                                           // drop any in-flight timeline response
      renderOverview();
      const b = Array.from(root.querySelectorAll("[data-key]")).find((x) => x.dataset.key === k);
      if (b) b.focus();
    });
    const head = el(`<div class="library-detail-head"><h3 tabindex="-1" id="lib-detail-h">${esc(c.name)}</h3><span class="badge ${c.own ? "badge-accent" : ""}">${c.key === OTHER ? "Other" : c.own ? "Yours" : "Standard"}</span></div>`);
    root.append(el(`<div class="library-detail-top"></div>`));
    root.firstChild.append(back, head);
    const slot = el(`<div class="library-detail-body stack"><div class="library-skel" role="status" aria-label="Loading"><div class="skeleton" style="height:5rem"></div><div class="skeleton" style="height:12rem"></div></div></div>`);
    root.append(slot);
    head.querySelector("h3").focus({ preventScroll: true });
    window.scrollTo({ top: 0, behavior: "auto" });
    await loadTimeline(true, slot);
  }

  async function loadTimeline(reset, slotArg) {
    const d = detail;
    if (!d) return;
    const mine = d.mine;
    const slot = slotArg || root.querySelector(".library-detail-body");
    if (reset) { d.items = []; d.cursor = null; d.hasMore = false; }
    d.loading = true;
    const q = new URLSearchParams({ limit: "20" });
    if (d.kind) q.set("kind", d.kind);
    if (d.cursor) q.set("before", d.cursor);
    let res;
    try {
      res = await window.api.get(`/library/${encodeURIComponent(d.key)}/timeline?${q}`, { fresh: true });
    } catch (e) {
      if (!detail || detail.mine !== mine) return;
      d.loading = false;
      if (reset && !slot.querySelector("#lib-timeline")) { clear(slot); slot.append(fail(e, () => openDetail(d.card))); }
      else toast((e && e.message) || "Could not load more.", "error");
      return;
    }
    if (!detail || detail.mine !== mine) return;
    d.loading = false;
    d.items = d.items.concat(res.items);
    d.cursor = res.next_cursor;
    d.hasMore = res.has_more;
    d.meta = res;
    if (!slot.querySelector("#lib-timeline")) renderDetailBody(slot, res);
    else paintTimeline(slot.querySelector("#lib-timeline"));
    const chipRow = slot.querySelector(".library-chips");
    if (chipRow) chipRow.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", String((b.dataset.kind || "") === (d.kind || ""))));
  }

  function renderDetailBody(slot, res) {
    const d = detail, c = d.card;
    clear(slot);
    slot.append(weekStrip(res.week));
    const cols = el(`<div class="library-cols"></div>`);
    if (c.key !== OTHER) cols.append(filesPanel(c));
    cols.append(chainPanel(res));
    slot.append(cols);
    const tl = el(`<section class="panel stack" aria-labelledby="lib-tl-h"><div class="section-head" style="margin:0"><h3 id="lib-tl-h">Timeline</h3><span>newest first</span></div></section>`);
    const chips = el(`<div class="library-chips" role="group" aria-label="Filter by kind"></div>`);
    const total = Object.values(res.counts).reduce((a, b) => a + b, 0);
    const mk = (kind, label, n) => {
      const b = el(`<button type="button" class="library-chip" data-kind="${esc(kind)}" aria-pressed="${kind === (d.kind || "")}">${esc(label)} <b>${n}</b></button>`);
      b.addEventListener("click", () => { if ((d.kind || "") === kind) return; d.kind = kind; d.loading = true; paintTimeline(slot.querySelector("#lib-timeline")); loadTimeline(true, slot); });
      return b;
    };
    chips.append(mk("", "All", total));
    Object.entries(res.counts).sort((a, b) => b[1] - a[1]).forEach(([k, n]) => chips.append(mk(k, KIND_NAME[k] || k, n)));
    tl.append(chips);
    const box = el(`<div id="lib-timeline" aria-live="polite"></div>`);
    tl.append(box);
    if (res.truncated) tl.append(el(`<p class="muted library-note">Showing your newest 100 items of each kind.</p>`));
    slot.append(tl);
    paintTimeline(box);
  }

  /* ---------- load ---------- */
  async function load(fresh, reopenKey) {
    const mine = ++seq;
    clear(root);
    root.append(skeleton());
    try {
      const res = await window.api.get("/library" + (fresh ? "?refresh=1" : ""), { fresh: true });
      if (mine !== seq || !root) return;
      data = res;
      renderOverview();
      if (reopenKey) {
        const c = reopenKey === OTHER ? null : data.subjects.find((x) => x.key === reopenKey);
        if (c) openDetail(c);
      }
    } catch (e) {
      if (mine !== seq || !root) return;
      clear(root);
      root.append(fail(e, () => load(false)));
    }
  }

  window.Views.library = {
    accent: "progress",
    title: "Library",
    async mount(node) {
      root = node;
      root.classList.add("library-root");
      await load(false);
    },
    unmount() {
      seq++;
      if (root) { clear(root); root.classList.remove("library-root"); }
      root = null; data = null; detail = null;
    },
  };
})();
