/* ============================================================
   Add a Subject view — window.Views.subjects
   Type a subject name, attach one or more PDFs, and create it.
   The PDFs are saved into documents/<subject>/ and embedded once
   (persisted to disk), so every tool can use the new subject and
   later runs load the index instead of rebuilding it.
   Adding with an existing subject name appends PDFs to it.
   ============================================================ */
(() => {
  "use strict";
  const { el, clear, esc, spinner, toast } = window.ui;

  let chosen = [];          // File[] queued for upload
  // Vercel rejects request bodies > 4.5 MB, so selections up to ~4 MB use the single multipart POST;
  // anything bigger is sent as small raw chunks (see uploadChunked) and assembled on the server.
  const SINGLE_POST_MAX = 4 * 1024 * 1024;
  // The limits are owned by the server (GET /api/subjects -> limits; env UPLOAD_MAX_*). These values are only
  // the fallback used until the first answer arrives, so the copy below can never drift from the backend.
  let limits = { max_file_mb: 200, max_total_mb: 0, max_files: 50, max_pdfs_per_subject: 100, max_subjects: 100,
                 max_pages_per_pdf: 2000, parallel_chunks: 3, raw_keep_mb: 25 };
  const MAX_FILE_BYTES_ = () => limits.max_file_mb * 1048576;
  const MAX_TOTAL_BYTES_ = () => (limits.max_total_mb || 0) * 1048576;   // 0 = no per-upload total cap
  const MAX_FILES_ = () => limits.max_files;
  function introText() {
    const L = limits;
    const total = L.max_total_mb > 0 ? `${L.max_total_mb} MB per upload` : "no total size limit per upload";
    return `Name it, attach lecture PDFs (up to ${L.max_file_mb} MB each, ${total}, up to ${L.max_files} files at a time; ` +
      `a subject holds up to ${L.max_pdfs_per_subject} PDFs and you can keep ${L.max_subjects} subjects). Big files are sent in pieces. ` +
      `Large files (over ${L.raw_keep_mb} MB) keep only their text index. Every tool picks the subject up as soon as it’s indexed.`;
  }
  function paintIntro(layout) {
    const p = layout.querySelector("#subj-intro");
    if (p) p.textContent = introText();
  }
  const CHUNK_TRIES = 3;                         // attempts per chunk (backoff 0.6 s, 1.8 s)
  const COMPLETE_TRIES = 6;                      // complete is idempotent server-side, so retrying is safe
  let activeUpload = null;                       // { cancel() } while a chunked upload is running
  const MAX_ROUNDS = 40;                         // continue-index rounds (~35 s each; stops early when nothing progresses)
  const indexing = new Set();                    // subjects currently being finished in the background
  const undeletable = new Set();                 // names the server said are built-in (403)
  const fmtMB = (b) => (b / 1048576).toFixed(1) + " MB";
  const totalBytes = () => chosen.reduce((n, f) => n + f.size, 0);

  function refreshHomeSubjectStat() {
    window.api.get("/subjects").then((d) => {
      const slot = document.getElementById("stat-subjects");
      if (slot) slot.textContent = (d.subjects || []).length || "·";
    }).catch(() => {});
  }

  function renderFileList(host) {
    clear(host);
    if (!chosen.length) {
      host.append(el(`<span class="muted">No PDFs chosen yet.</span>`));
      return;
    }
    chosen.forEach((file, i) => {
      const kb = Math.max(1, Math.round(file.size / 1024));
      const row = el(`
        <div class="row between panel-soft" style="padding:.55rem .8rem;gap:.5rem;min-width:0">
          <span class="row" style="gap:.5rem;min-width:0;flex:1">
            <span aria-hidden="true">📄</span>
            <span style="min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${esc(file.name)}">${esc(file.name)}</span>
            <span class="muted" style="flex:none">${kb} KB</span>
          </span>
          <button type="button" class="icon-btn is-danger" data-rm="${i}" aria-label="Remove ${esc(file.name)}" title="Remove ${esc(file.name)}">${window.ui.iconHtml("close")}</button>
        </div>`);
      row.querySelector("[data-rm]").addEventListener("click", () => {
        chosen.splice(i, 1);
        renderFileList(host);
      });
      host.append(row);
    });
    // Total size vs. the upload limits; large selections are sent in pieces automatically.
    const total = totalBytes();
    const tooBigFile = chosen.find((f) => f.size > MAX_FILE_BYTES_());
    const problem = tooBigFile
      ? `“${tooBigFile.name}” is ${fmtMB(tooBigFile.size)} — the limit is ${fmtMB(MAX_FILE_BYTES_())} per PDF. Remove it or split the PDF.`
      : MAX_TOTAL_BYTES_() && total > MAX_TOTAL_BYTES_()
        ? `Selected PDFs total ${fmtMB(total)} — the limit is ${fmtMB(MAX_TOTAL_BYTES_())} per upload. Remove a file, or add the PDFs in two batches (the same subject name appends).`
        : chosen.length > MAX_FILES_()
          ? `You selected ${chosen.length} PDFs — the limit is ${MAX_FILES_()} per upload. Remove some, or add them in batches.`
          : "";
    if (problem) {
      host.append(el(`<div class="panel-soft" role="alert" style="border-color:hsl(0 80% 60% / .5);overflow-wrap:anywhere">⛔ ${esc(problem)}</div>`));
    } else if (total > SINGLE_POST_MAX) {
      host.append(el(`<div class="panel-soft" role="status" style="overflow-wrap:anywhere">ℹ Selected PDFs total <b>${fmtMB(total)}</b>. They’ll be uploaded in small pieces with a progress bar — you can cancel any time.</div>`));
    }
  }

  function addFiles(fileList, host) {
    for (const file of fileList) {
      if (!/\.pdf$/i.test(file.name) && file.type !== "application/pdf") {
        toast(`Skipped “${file.name}” — not a PDF.`, "error");
        continue;
      }
      // De-dupe by name + size so re-picking the same file doesn't double it.
      if (!chosen.some((f) => f.name === file.name && f.size === file.size)) {
        chosen.push(file);
      }
    }
    renderFileList(host);
  }

  function showResult(slot, result) {
    clear(slot);
    const list = (result.saved_pdfs || []).map((n) => `<li>${esc(n)}</li>`).join("");
    const dups = (result.skipped_duplicates || []);
    const partial = result.indexing === "partial";
    const box = el(`
      <div class="panel-soft stack-sm" style="border-color:hsl(var(--accent) / .45)">
        <div class="row" style="gap:.6rem">
          <span class="badge badge-accent">Subject ready</span>
          <strong style="overflow-wrap:anywhere">${esc(result.subject)}</strong>
        </div>
        <p class="muted" style="margin:0">Added ${(result.saved_pdfs || []).length} PDF(s) — ${result.chunk_count} text chunk(s) are searchable now${partial ? ", and smarter semantic search is still being built in the background" : ""}. Ready to use in Chat, Quiz, Flashcards, Planner and Mind Map.</p>
        <ul class="md" style="margin:.2rem 0 .2rem 1.1rem;overflow-wrap:anywhere">${list}</ul>
        ${dups.length ? `<p class="hint" style="margin:0">Skipped ${dups.length} duplicate file(s): ${esc(dups.join(", "))}</p>` : ""}
        <div class="subj-progress" role="status" aria-live="polite" ${partial ? "" : "hidden"}></div>
        <div class="row row-wrap" style="gap:.6rem">
          <a class="btn-accent" href="#chat">Open in Chat →</a>
          <a class="btn" href="#quiz">Make a quiz</a>
        </div>
      </div>`);
    slot.append(box);
    if (partial) continueIndexing(result.subject, box.querySelector(".subj-progress"), result);
  }

  // Finish embedding a partially indexed subject: sequential calls (each bounded to ~35 s
  // server-side) until complete, no progress (quota/keys exhausted) or the server is busy.
  // Non-blocking: the subject already works through keyword search while this runs.
  async function continueIndexing(subject, note, first) {
    if (indexing.has(subject)) return;
    indexing.add(subject);
    const paint = (msg) => { if (note) { note.hidden = false; note.className = "subj-progress muted"; note.textContent = msg; } };
    let nEmb = first && first.n_embedded, nTotal = first && first.chunk_count;
    let reading = (first && first.reading) || {};
    const label = () => {
      const rd = Object.values(reading);
      if (rd.length) {
        const done = rd.reduce((n, v) => n + (v.done || 0), 0), tot = rd.reduce((n, v) => n + (v.total || 0), 0);
        return `Reading pages ${done}/${tot} — search already works on what’s been read.`;
      }
      return `Embedding ${nEmb != null && nTotal ? nEmb + "/" + nTotal + " chunks" : "in progress"} — search already works.`;
    };
    try {
      paint(label());
      for (let round = 0; round < MAX_ROUNDS; round++) {
        let r;
        try {
          r = await window.api.post(`/subjects/${encodeURIComponent(subject)}/continue-index`, {});
        } catch (e) {
          paint(`Semantic indexing paused (${e.message}). Search still works; it will resume the next time you add to this subject.`);
          return;
        }
        reading = r.reading || {};
        if (r.n_embedded != null) nEmb = r.n_embedded;
        if (r.n_chunks != null) nTotal = r.n_chunks;
        if (r.complete || r.indexing === "complete") { paint(`Indexing complete — ${nEmb != null ? nEmb + " chunks " : ""}embedded.`); return; }
        if (r.busy) { paint("Indexing is already running for this subject — search works meanwhile."); return; }
        if (!r.progress) { paint(`Semantic indexing is paused (${nEmb != null && nTotal ? nEmb + "/" + nTotal + " chunks" : "provider limit reached"}). Search still works; try again later.`); return; }
        paint(label());
      }
      paint(`${label().replace("Indexing…", "Indexing continues…")} Add more later to finish.`);
    } finally {
      indexing.delete(subject);
      window.api.invalidate("/subjects");
    }
  }

  async function submit(layout) {
    const nameInput = layout.querySelector("#subj-name");
    const goBtn = layout.querySelector("#subj-go");
    const resultSlot = layout.querySelector("#subj-result");
    const fileHost = layout.querySelector("#subj-files");

    const name = nameInput.value.trim();
    if (!name) { toast("Name the subject first.", "error"); nameInput.focus(); return; }
    if (!chosen.length) { toast("Choose at least one PDF.", "error"); return; }
    const limitError = uploadLimitError();
    if (limitError) { toast(limitError, "error"); return; }

    goBtn.disabled = true;
    goBtn.textContent = "Building…";
    clear(resultSlot);
    const chunked = totalBytes() > SINGLE_POST_MAX;
    const ui = chunked ? makeProgressUi(resultSlot) : null;
    if (!chunked) resultSlot.append(spinner("Saving and indexing your PDFs — this runs once, then it’s cached for instant answers…"));

    try {
      let result;
      if (chunked) {
        const out = await uploadChunked(name, chosen.slice(), ui);
        result = out.result;
      } else {
        const fd = new FormData();
        fd.append("name", name);
        chosen.forEach((file) => fd.append("files", file));
        result = await window.api.postForm("/subjects", fd);
      }
      showResult(resultSlot, result);
      toast(`“${result.subject}” is ready.`, "ok");
      // Reset the form for the next subject; keep the result visible.
      chosen = [];
      nameInput.value = "";
      renderFileList(fileHost);
      refreshHomeSubjectStat();
      loadExisting(layout);
    } catch (e) {
      clear(resultSlot);
      const done0 = (e && e.doneFiles) || [];
      if (done0.length) {          // earlier files in this batch were added: don't re-send them on retry
        chosen = chosen.filter((f) => !done0.includes(f));
        renderFileList(fileHost);
        refreshHomeSubjectStat();
        loadExisting(layout);
      }
      if (e && e.cancelled) {
        resultSlot.append(el(`<div class="panel-soft" role="status">Upload cancelled.${done0.length ? ` ${done0.length} PDF(s) finished before you cancelled and were added.` : " Nothing was added."}</div>`));
      } else {
        const extra = done0.length ? ` ${done0.length} earlier PDF(s) in this batch were added; only the remaining files are still selected.` : "";
        resultSlot.append(el(`<div class="panel-soft" role="alert" style="border-color:hsl(0 80% 60% / .5);overflow-wrap:anywhere">⚠ ${esc(e.message)}${esc(extra)}</div>`));
        toast(e.message, "error");
      }
    } finally {
      activeUpload = null;
      goBtn.disabled = false;
      goBtn.textContent = "Create subject & index PDFs";
    }
  }

  function uploadLimitError() {
    return filesLimitError(chosen);
  }

  function filesLimitError(files) {
    if (files.length > MAX_FILES_()) return `Too many PDFs (${files.length}); the limit is ${MAX_FILES_()} per upload.`;
    const big = files.find((f) => f.size > MAX_FILE_BYTES_());
    if (big) return `“${big.name}” is ${fmtMB(big.size)} — the limit is ${fmtMB(MAX_FILE_BYTES_())} per PDF.`;
    const total = files.reduce((n, f) => n + f.size, 0);
    if (MAX_TOTAL_BYTES_() && total > MAX_TOTAL_BYTES_()) return `Selected PDFs total ${fmtMB(total)} — the limit is ${fmtMB(MAX_TOTAL_BYTES_())} per upload. Add them in smaller batches.`;
    return "";
  }

  // ---- chunked upload -----------------------------------------------------------------------
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const retryable = (e) => e && (e.status === 0 || e.status === 408 || e.status === 429 || e.status >= 500);
  const cancelledError = () => Object.assign(new Error("Upload cancelled."), { cancelled: true });

  function makeProgressUi(slot) {
    const box = el(`
      <div class="panel-soft stack-sm" role="group" aria-label="Upload progress">
        <div class="row between" style="gap:.6rem">
          <span class="subj-up-label" role="status" aria-live="polite">Preparing upload…</span>
          <button type="button" class="btn-ghost subj-up-cancel" style="flex:none;padding:.3rem .8rem">Cancel</button>
        </div>
        <div class="progress" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="0"><i style="width:0%"></i></div>
      </div>`);
    slot.append(box);
    const label = box.querySelector(".subj-up-label");
    const bar = box.querySelector(".progress");
    const fill = bar.querySelector("i");
    const cancel = box.querySelector(".subj-up-cancel");
    return {
      cancel,
      text(t) { label.textContent = t; },
      pct(p) { const v = Math.max(0, Math.min(100, Math.round(p))); fill.style.width = v + "%"; bar.setAttribute("aria-valuenow", String(v)); },
      lockCancel(why) { cancel.disabled = true; cancel.title = why || ""; },
    };
  }

  async function sha256Hex(file) {
    try {
      if (!(window.crypto && window.crypto.subtle)) return null;
      const digest = await window.crypto.subtle.digest("SHA-256", await file.arrayBuffer());
      return Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, "0")).join("");
    } catch (_) { return null; }
  }

  // Upload files one after another: create -> chunks (3 tries each) -> complete (idempotent, retried).
  // Resolves { result, doneFiles }; rejects with an Error carrying `doneFiles` (+ `cancelled`).
  async function uploadChunked(subject, files, ui) {
    const ctl = new AbortController();
    let cancelled = false, uploadId = null;
    const state = { cancel() {
      cancelled = true; ctl.abort();
      if (uploadId) window.api.del(`/subjects/uploads/${encodeURIComponent(uploadId)}`).catch(() => {});
    } };
    activeUpload = state;
    ui.cancel.addEventListener("click", () => { ui.text("Cancelling…"); state.cancel(); });

    const doneFiles = [];
    const results = [];
    const check = () => { if (cancelled) throw cancelledError(); };
    try {
      for (let fi = 0; fi < files.length; fi++) {
        const file = files[fi];
        const head = files.length > 1 ? `Uploading file ${fi + 1}/${files.length}` : "Uploading";
        uploadId = null;
        ui.pct(0);
        ui.text(`${head} — preparing…`);
        const sha = await sha256Hex(file);
        check();
        const info = await window.api.post("/subjects/uploads", { subject, filename: file.name, size: file.size, sha256: sha || undefined }, { signal: ctl.signal });
        uploadId = info.upload_id;
        const base = `/subjects/uploads/${encodeURIComponent(uploadId)}`;
        // Up to `parallel_chunks` (3) pieces in flight; each piece retries on its own, so a hiccup never restarts the file.
        let next = 0, failed = false, sentCount = 0;
        const putOne = async (i) => {
          const blob = file.slice(i * info.chunk_size, Math.min(file.size, (i + 1) * info.chunk_size));
          for (let attempt = 1; ; attempt++) {
            try {
              await window.api.putBlob(`${base}/chunks/${i}`, blob, { signal: ctl.signal });
              break;
            } catch (e) {
              if (cancelled || (e && e.name === "AbortError")) throw cancelledError();
              if (attempt >= CHUNK_TRIES || !retryable(e)) throw e;
              ui.text(`${head} — connection hiccup, retrying…`);
              await sleep(600 * 3 ** (attempt - 1));
              check();
            }
          }
          sentCount++;
          const pct = (sentCount / info.total_chunks) * 100;
          ui.pct(pct);
          ui.text(`${head} — ${Math.round(pct)}% (piece ${sentCount}/${info.total_chunks})`);
        };
        const lanes = Math.max(1, Math.min(limits.parallel_chunks || 3, info.total_chunks));
        await Promise.all(Array.from({ length: lanes }, async () => {
          while (!failed) {
            check();
            const i = next++;
            if (i >= info.total_chunks) return;
            try { await putOne(i); } catch (e) { failed = true; throw e; }
          }
        }));
        check();
        ui.lockCancel("Already being processed");
        ui.text(`${files.length > 1 ? `File ${fi + 1}/${files.length}: ` : ""}reading and indexing the PDF…`);
        results.push(await completeWithRetry(base, ui));
        doneFiles.push(file);
        uploadId = null;                               // finished: nothing to abort any more
        ui.cancel.disabled = false;
      }
    } catch (e) {
      if (e && e.name === "AbortError") e = cancelledError();
      if (uploadId && !(e && e.cancelled)) window.api.del(`/subjects/uploads/${encodeURIComponent(uploadId)}`).catch(() => {});
      e.doneFiles = doneFiles;
      throw e;
    } finally {
      if (activeUpload === state) activeUpload = null;
    }
    const last = results[results.length - 1];
    const result = {
      subject: last.subject,
      saved_pdfs: results.flatMap((r) => r.saved_pdfs || []),
      skipped_duplicates: results.flatMap((r) => r.skipped_duplicates || []),
      chunk_count: last.chunk_count,
      n_embedded: last.n_embedded,
      reading: Object.assign({}, ...results.map((r) => r.reading || {})),
      indexing: results.some((r) => r.indexing === "partial") ? "partial" : "complete",
    };
    return { result, doneFiles };
  }

  // `complete` is idempotent server-side (a finished upload returns its stored result; a concurrent
  // call gets 409 "already being completed"), so network errors / 5xx / 409-busy are safe to retry.
  async function completeWithRetry(base, ui) {
    for (let attempt = 1; ; attempt++) {
      try {
        return await window.api.post(`${base}/complete`, {});
      } catch (e) {
        const busy = e && e.status === 409 && /already being completed/i.test(e.message || "");
        if (attempt >= COMPLETE_TRIES || !(busy || retryable(e))) throw e;
        ui.text("Still processing the PDF — checking again…");
        await sleep(busy ? 4000 : 2500);
      }
    }
  }

  function loadExisting(layout) {
    const host = layout.querySelector("#subj-existing");
    clear(host);
    host.append(el(`<span class="muted">Loading…</span>`));
    window.api.get("/subjects", { fresh: true }).then((d) => {
      const subjects = d.subjects || [];
      if (d.limits) { limits = Object.assign({}, limits, d.limits); paintIntro(layout); renderFileList(layout.querySelector("#subj-files")); }
      // The server lists which subjects are the caller's own (deletable) vs built-in (standard, hide-able).
      const own = new Set(Array.isArray(d.own) ? d.own : []);
      const bundled = new Set(Array.isArray(d.bundled) ? d.bundled : []);
      const hidden = Array.isArray(d.hidden) ? d.hidden : [];
      clear(host);
      if (!subjects.length) {
        host.append(el(`<span class="muted">No subjects yet — add your first above.</span>`));
      }
      subjects.forEach((name) => {
        if (bundled.has(name) && !own.has(name)) { host.append(standardChip(layout, name)); return; }
        const canDelete = own.has(name) && !undeletable.has(name);
        host.append(subjectChip(layout, name, canDelete));
      });
      paintStandardNote(layout, hidden, bundled.size > 0);
    }).catch(() => {
      clear(host);
      host.append(el(`<span class="muted">Couldn’t load subjects.</span>`));
    });
  }

  // One subject badge; when deletable, a ✕ that swaps in an in-page "Delete? Yes / No" confirm.
  function subjectChip(layout, name, canDelete) {
    const chip = el(`<span class="badge row" style="gap:.35rem;align-items:center"></span>`);
    const label = document.createElement(canDelete ? "button" : "span");
    label.textContent = name;
    if (canDelete) {
      label.type = "button";
      label.className = "subj-chip-open";
      label.style.cssText = "background:none;border:0;padding:0;color:inherit;font:inherit;cursor:pointer;text-decoration:underline dotted";
      label.setAttribute("aria-label", "Manage subject " + name);
      label.title = "Manage files, rename, re-index";
      label.addEventListener("click", () => openDetail(layout, name));
    }
    chip.append(label);
    if (!canDelete) return chip;
    const del = el(`<button type="button" class="icon-btn is-danger is-sm" title="Delete this subject"></button>`);
    del.append(window.ui.icon("trash"));
    del.setAttribute("aria-label", "Delete subject " + name);
    chip.append(del);
    del.addEventListener("click", () => {
      del.hidden = true;
      const confirmBox = el(`<span class="row" role="group" style="gap:.3rem;align-items:center"><span class="muted" style="font-size:var(--fs-sm)">Delete with its PDFs?</span></span>`);
      const yes = el(`<button type="button" class="btn-accent" style="padding:.1rem .6rem;font-size:var(--fs-sm)">Delete</button>`);
      const no = el(`<button type="button" class="btn-ghost" style="padding:.1rem .6rem;font-size:var(--fs-sm)">Keep</button>`);
      confirmBox.append(yes, no);
      chip.append(confirmBox);
      no.focus();
      no.addEventListener("click", () => { confirmBox.remove(); del.hidden = false; del.focus(); });
      yes.addEventListener("click", async () => {
        yes.disabled = no.disabled = true;
        yes.textContent = "Deleting…";
        try {
          await window.api.del(`/subjects/${encodeURIComponent(name)}`);
          toast(`Deleted “${name}”.`, "ok");
          chip.remove();
          refreshHomeSubjectStat();
        } catch (e) {
          if (e.status === 403) { undeletable.add(name); toast("Built-in subjects can’t be deleted.", "error"); chip.replaceWith(subjectChip(layout, name, false)); }
          else { toast(e.message, "error"); confirmBox.remove(); del.hidden = false; }
        }
      });
    });
    return chip;
  }


  // ---- standard (shared sample) subjects: hide for this account only -----------------------------
  function standardChip(layout, name) {
    const chip = el(`<span class="badge row" style="gap:.35rem;align-items:center"></span>`);
    const label = document.createElement("span");
    label.textContent = name;
    chip.append(label);
    const rm = el(`<button type="button" class="icon-btn is-sm" title="Remove from my account"></button>`);
    rm.append(window.ui.icon("close"));
    rm.setAttribute("aria-label", "Remove " + name + " from my account");
    chip.append(rm);
    rm.addEventListener("click", () => {
      rm.hidden = true;
      const box = el(`<span class="row" role="group" style="gap:.3rem;align-items:center"><span class="muted" style="font-size:var(--fs-sm)">Remove ${esc(name)} from your account? You can restore it any time.</span></span>`);
      const yes = el(`<button type="button" class="btn-accent" style="padding:.1rem .6rem;font-size:var(--fs-sm)">Remove</button>`);
      const no = el(`<button type="button" class="btn-ghost" style="padding:.1rem .6rem;font-size:var(--fs-sm)">Keep</button>`);
      box.append(yes, no);
      chip.append(box);
      no.focus();
      no.addEventListener("click", () => { box.remove(); rm.hidden = false; rm.focus(); });
      yes.addEventListener("click", async () => {
        yes.disabled = no.disabled = true;
        yes.textContent = "Removing…";
        try {
          await window.api.post(`/subjects/${encodeURIComponent(name)}/hide`, {});
          window.api.invalidate("/subjects");
          toast(`Removed “${name}” from your account. You can restore it below.`, "ok");
          loadExisting(layout);
          refreshHomeSubjectStat();
        } catch (e) { toast(e.message, "error"); box.remove(); rm.hidden = false; }
      });
    });
    return chip;
  }

  function paintStandardNote(layout, hidden, anyStandard) {
    const host = layout.querySelector("#subj-std");
    if (!host) return;
    clear(host);
    if (!anyStandard && !hidden.length) return;
    host.append(el(`<p class="hint" style="margin:0">Standard subjects are shared samples — removing one only hides it for you; it is never deleted for anyone else.</p>`));
    if (hidden.length) {
      const row = el(`<div class="row row-wrap" style="gap:.5rem;align-items:center"><span class="muted">Hidden: ${esc(hidden.join(", "))}</span></div>`);
      const btn = el(`<button type="button" class="btn">Restore standard subjects</button>`);
      btn.addEventListener("click", async () => {
        btn.disabled = true;
        try {
          await window.api.post("/subjects/restore-standard", {});
          window.api.invalidate("/subjects");
          toast("Standard subjects restored.", "ok");
          loadExisting(layout);
          refreshHomeSubjectStat();
        } catch (e) { toast(e.message, "error"); btn.disabled = false; }
      });
      row.append(btn);
      host.append(row);
    }
  }

  // ---- subject detail panel: files, rename, re-index, add files, delete ---------------------------
  const fmtSize = (b) => (b >= 1048576 ? (b / 1048576).toFixed(1) + " MB" : Math.max(1, Math.round(b / 1024)) + " KB");

  async function openDetail(layout, name) {
    const host = layout.querySelector("#subj-detail");
    if (!host) return;
    clear(host);
    host.append(spinner("Loading files…"));
    try {
      const d = await window.api.get(`/subjects/${encodeURIComponent(name)}/files`, { fresh: true });
      paintDetail(layout, host, d);
    } catch (e) {
      clear(host);
      host.append(el(`<div class="panel-soft" role="alert" style="overflow-wrap:anywhere">⚠ ${esc(e.message)}</div>`));
    }
  }

  function paintDetail(layout, host, d) {
    clear(host);
    const name = d.subject;
    const enc = encodeURIComponent(name);
    const L = limits;
    const reading = Object.values(d.reading || {});
    const box = el(`
      <div class="panel-soft stack-sm" role="region" aria-label="Subject ${esc(name)}">
        <div class="row between" style="gap:.5rem;flex-wrap:wrap">
          <strong style="overflow-wrap:anywhere;min-width:0">${esc(name)}</strong>
          <span class="row subj-d-tools" style="gap:.15rem"></span>
        </div>
        <form class="row row-wrap subj-d-rename" hidden style="gap:.4rem">
          <input class="input" type="text" maxlength="80" aria-label="New subject name" value="${esc(name)}" style="flex:1;min-width:10rem" />
          <button type="submit" class="btn-accent">Rename</button>
          <button type="button" class="btn-ghost subj-d-rename-cancel">Cancel</button>
        </form>
        <div class="row row-wrap subj-d-confirm" role="group" hidden style="gap:.4rem;align-items:center">
          <span class="muted">Delete “${esc(name)}” with all its files?</span>
          <button type="button" class="btn-accent subj-d-del-yes">Delete</button>
          <button type="button" class="btn-ghost subj-d-del-no">Keep</button>
        </div>
        <p class="hint" style="margin:0">${d.files.length} file(s) · ${d.n_chunks} chunks (${d.n_embedded} embedded) · you use ${d.usage ? d.usage.owner_mb : 0} MB of library space. Files over ${L.raw_keep_mb} MB keep only their text index.${d.usage && d.usage.near_cap ? " The shared library is nearly full." : ""}</p>
        <div class="stack-sm subj-d-files" role="list"></div>
        <div class="subj-progress muted" role="status" aria-live="polite" ${(d.indexing === "partial" || reading.length) ? "" : "hidden"}></div>
        <div class="subj-d-up"></div>
      </div>`);
    const tools = box.querySelector(".subj-d-tools");
    const progress = box.querySelector(".subj-progress");
    const upSlot = box.querySelector(".subj-d-up");
    const mk = (icon, label, danger) => { const b = el(`<button type="button" class="icon-btn${danger ? " is-danger" : ""}"></button>`); b.append(window.ui.icon(icon)); b.setAttribute("aria-label", label); b.title = label; return b; };
    const bRename = mk("edit", "Rename subject"), bRe = mk("refresh", "Re-index subject"), bAdd = mk("plus", "Add PDFs"),
      bDel = mk("trash", "Delete subject", true), bClose = mk("close", "Close");
    tools.append(bRename, bRe, bAdd, bDel, bClose);
    const picker = el(`<input type="file" accept="application/pdf" multiple hidden aria-label="Add PDFs to ${esc(name)}" />`);
    box.append(picker);

    d.files.forEach((f) => {
      const status = f.status === "reading" ? `reading ${f.read_pages || 0}/${f.pages}` : "ready";
      const row = el(`
        <div class="row between panel-soft" role="listitem" style="padding:.45rem .7rem;gap:.5rem;min-width:0">
          <span style="min-width:0;flex:1">
            <span style="display:block;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${esc(f.name)}">${esc(f.name)}</span>
            <span class="muted" style="font-size:var(--fs-sm)">${fmtSize(f.size)} · ${f.pages} pages · ${status}${f.text_only ? " · text index only" : ""}</span>
          </span>
          <span class="subj-d-row-act row" style="gap:.3rem;flex:none"></span>
        </div>`);
      const act = row.querySelector(".subj-d-row-act");
      const rm = mk("close", "Remove " + f.name, true);
      act.append(rm);
      rm.addEventListener("click", () => {
        rm.hidden = true;
        const yes = el(`<button type="button" class="btn-accent" style="padding:.1rem .6rem;font-size:var(--fs-sm)">Remove</button>`);
        const no = el(`<button type="button" class="btn-ghost" style="padding:.1rem .6rem;font-size:var(--fs-sm)">Keep</button>`);
        act.append(yes, no);
        no.focus();
        no.addEventListener("click", () => { yes.remove(); no.remove(); rm.hidden = false; rm.focus(); });
        yes.addEventListener("click", async () => {
          yes.disabled = no.disabled = true;
          yes.textContent = "Removing…";
          try {
            await window.api.del(`/subjects/${enc}/pdfs/${encodeURIComponent(f.name)}`);
            toast(`Removed “${f.name}”.`, "ok");
            openDetail(layout, name);
            refreshHomeSubjectStat();
          } catch (e) { toast(e.message, "error"); yes.remove(); no.remove(); rm.hidden = false; }
        });
      });
      box.querySelector(".subj-d-files").append(row);
    });

    bClose.addEventListener("click", () => clear(host));
    const form = box.querySelector(".subj-d-rename");
    bRename.addEventListener("click", () => { form.hidden = false; form.querySelector("input").focus(); form.querySelector("input").select(); });
    box.querySelector(".subj-d-rename-cancel").addEventListener("click", () => { form.hidden = true; bRename.focus(); });
    form.addEventListener("submit", async (ev) => {
      ev.preventDefault();
      const nn = form.querySelector("input").value.trim();
      if (!nn || nn === name) { form.hidden = true; return; }
      form.querySelector("button[type=submit]").disabled = true;
      try {
        const r = await window.api.patch(`/subjects/${enc}`, { name: nn });
        window.api.invalidate("/subjects");
        toast(`Renamed to “${r.subject}”.`, "ok");
        loadExisting(layout);
        openDetail(layout, r.subject);
      } catch (e) { toast(e.message, "error"); form.querySelector("button[type=submit]").disabled = false; }
    });
    bRe.addEventListener("click", async () => {
      bRe.disabled = true;
      progress.hidden = false;
      progress.textContent = "Re-indexing…";
      try {
        const r = await window.api.post(`/subjects/${enc}/reindex`, {});
        if (r.indexing === "partial") await continueIndexing(name, progress, r);
      } catch (e) { toast(e.message, "error"); }
      openDetail(layout, name);
    });
    const confirmBox = box.querySelector(".subj-d-confirm");
    bDel.addEventListener("click", () => { confirmBox.hidden = false; confirmBox.querySelector(".subj-d-del-no").focus(); });
    box.querySelector(".subj-d-del-no").addEventListener("click", () => { confirmBox.hidden = true; bDel.focus(); });
    box.querySelector(".subj-d-del-yes").addEventListener("click", async () => {
      try {
        await window.api.del(`/subjects/${enc}`);
        toast(`Deleted “${name}”.`, "ok");
        clear(host);
        loadExisting(layout);
        refreshHomeSubjectStat();
      } catch (e) { toast(e.message, "error"); confirmBox.hidden = true; }
    });
    bAdd.addEventListener("click", () => picker.click());
    picker.addEventListener("change", async () => {
      const files = Array.from(picker.files || []).filter((f) => /\.pdf$/i.test(f.name) || f.type === "application/pdf");
      picker.value = "";
      if (!files.length) { toast("Choose PDF files.", "error"); return; }
      const problem = filesLimitError(files);
      if (problem) { toast(problem, "error"); return; }
      await uploadInto(name, files, upSlot);
      openDetail(layout, name);
    });
    host.append(box);
    if (reading.length || d.indexing === "partial") continueIndexing(name, progress, { n_embedded: d.n_embedded, chunk_count: d.n_chunks, reading: d.reading });
  }

  // Upload files into an existing subject (same single-POST / chunked paths as the main form).
  async function uploadInto(name, files, slot) {
    clear(slot);
    const total = files.reduce((n, f) => n + f.size, 0);
    const chunked = total > SINGLE_POST_MAX;
    const ui = chunked ? makeProgressUi(slot) : null;
    if (!chunked) slot.append(spinner("Saving and indexing…"));
    try {
      let result;
      if (chunked) result = (await uploadChunked(name, files, ui)).result;
      else {
        const fd = new FormData();
        fd.append("name", name);
        files.forEach((f) => fd.append("files", f));
        result = await window.api.postForm("/subjects", fd);
      }
      clear(slot);
      toast(`Added ${(result.saved_pdfs || []).length} PDF(s) to “${name}”.`, "ok");
      if (result.indexing === "partial") await continueIndexing(name, slot.closest(".panel-soft").querySelector(".subj-progress"), result);
    } catch (e) {
      clear(slot);
      if (e && e.cancelled) slot.append(el(`<div class="panel-soft" role="status">Upload cancelled.</div>`));
      else { slot.append(el(`<div class="panel-soft" role="alert" style="overflow-wrap:anywhere">⚠ ${esc(e.message)}</div>`)); toast(e.message, "error"); }
    } finally { activeUpload = null; }
  }

  window.Views = window.Views || {};
  window.Views.subjects = {
    accent: "subjects",
    title: "Add a Subject",
    async mount(root) {
      const layout = el(`
        <div class="stack">
          <div class="panel stack">
            <div class="stack-sm">
              <h3 style="font-family:var(--font-display);font-weight:700;letter-spacing:-.02em">Create a new subject</h3>
              <p class="muted" id="subj-intro"></p>
            </div>

            <div class="field">
              <label class="label" for="subj-name">Subject name</label>
              <input id="subj-name" class="input" type="text" placeholder="e.g. Machine Learning" autocomplete="off" />
              <div class="hint">Using an existing subject’s name adds these PDFs to it.</div>
            </div>

            <div class="field">
              <label class="label" for="subj-pdf">PDFs</label>
              <div class="row row-wrap">
                <label class="btn" for="subj-pdf" style="cursor:pointer">Choose PDFs…</label>
              </div>
              <input id="subj-pdf" type="file" accept="application/pdf" multiple hidden aria-label="Choose PDF files for this subject" />
              <div id="subj-files" class="stack-sm" style="margin-top:.7rem" role="list" aria-label="Selected PDFs"></div>
            </div>

            <button id="subj-go" class="btn-accent btn-lg btn-block" type="button">Create subject &amp; index PDFs</button>
            <div id="subj-result"></div>
          </div>

          <div class="panel stack-sm">
            <h3 style="font-family:var(--font-display);font-weight:700;letter-spacing:-.02em">Your subjects</h3>
            <div id="subj-existing" class="row row-wrap"></div>
            <div id="subj-std" class="stack-sm"></div>
            <div id="subj-detail"></div>
          </div>
        </div>`);
      root.append(layout);

      const fileInput = layout.querySelector("#subj-pdf");
      const fileHost = layout.querySelector("#subj-files");
      renderFileList(fileHost);
      paintIntro(layout);

      fileInput.addEventListener("change", () => {
        if (fileInput.files && fileInput.files.length) addFiles(fileInput.files, fileHost);
        fileInput.value = "";   // allow re-picking the same file
      });
      layout.querySelector("#subj-go").addEventListener("click", () => submit(layout));
      layout.querySelector("#subj-name").addEventListener("keydown", (e) => {
        if (e.key === "Enter") { e.preventDefault(); submit(layout); }
      });

      loadExisting(layout);
    },
    unmount() {
      if (activeUpload) { try { activeUpload.cancel(); } catch (_) {} activeUpload = null; }
      chosen = [];
    },
  };
})();
