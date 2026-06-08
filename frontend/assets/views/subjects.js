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
        <div class="row between panel-soft" style="padding:.55rem .8rem">
          <span class="row" style="gap:.5rem;min-width:0">
            <span aria-hidden="true">📄</span>
            <span style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${esc(file.name)}</span>
            <span class="muted" style="flex:none">${kb} KB</span>
          </span>
          <button type="button" class="btn-ghost" data-rm="${i}" aria-label="Remove ${esc(file.name)}" style="padding:.3rem .7rem">✕</button>
        </div>`);
      row.querySelector("[data-rm]").addEventListener("click", () => {
        chosen.splice(i, 1);
        renderFileList(host);
      });
      host.append(row);
    });
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
    slot.append(el(`
      <div class="panel-soft stack-sm" style="border-color:hsl(var(--accent) / .45)">
        <div class="row" style="gap:.6rem">
          <span class="badge badge-accent">Subject ready</span>
          <strong>${esc(result.subject)}</strong>
        </div>
        <p class="muted" style="margin:0">Embedded ${result.chunk_count} chunk(s) from ${(result.saved_pdfs || []).length} PDF(s) — ready to use in Chat, Quiz, Flashcards, Planner and Mind Map.</p>
        <ul class="md" style="margin:.2rem 0 .2rem 1.1rem">${list}</ul>
        <div class="row row-wrap" style="gap:.6rem">
          <a class="btn-accent" href="#chat">Open in Chat →</a>
          <a class="btn" href="#quiz">Make a quiz</a>
        </div>
      </div>`));
  }

  async function submit(layout) {
    const nameInput = layout.querySelector("#subj-name");
    const goBtn = layout.querySelector("#subj-go");
    const resultSlot = layout.querySelector("#subj-result");
    const fileHost = layout.querySelector("#subj-files");

    const name = nameInput.value.trim();
    if (!name) { toast("Name the subject first.", "error"); nameInput.focus(); return; }
    if (!chosen.length) { toast("Choose at least one PDF.", "error"); return; }

    goBtn.disabled = true;
    goBtn.textContent = "Building…";
    clear(resultSlot);
    resultSlot.append(spinner("Saving PDFs and building embeddings — this runs once, then it's cached for instant answers…"));

    try {
      const fd = new FormData();
      fd.append("name", name);
      chosen.forEach((file) => fd.append("files", file));
      const result = await window.api.postForm("/subjects", fd);
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
      resultSlot.append(el(`<div class="panel-soft" style="border-color:hsl(0 80% 60% / .5)">⚠ ${esc(e.message)}</div>`));
      toast(e.message, "error");
    } finally {
      goBtn.disabled = false;
      goBtn.textContent = "Create subject & build embeddings";
    }
  }

  function loadExisting(layout) {
    const host = layout.querySelector("#subj-existing");
    clear(host);
    host.append(el(`<span class="muted">Loading…</span>`));
    window.api.get("/subjects").then((d) => {
      const subjects = d.subjects || [];
      clear(host);
      if (!subjects.length) {
        host.append(el(`<span class="muted">No subjects yet — add your first above.</span>`));
        return;
      }
      subjects.forEach((s) => host.append(el(`<span class="badge">${esc(s)}</span>`)));
    }).catch(() => {
      clear(host);
      host.append(el(`<span class="muted">Couldn’t load subjects.</span>`));
    });
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
              <p class="muted">Name it, attach lecture PDFs, and we’ll save them under <code>documents/</code> and embed them once. Every tool — chat, quiz, flashcards, planner — picks it up immediately.</p>
            </div>

            <div class="field">
              <label class="label" for="subj-name">Subject name</label>
              <input id="subj-name" class="input" type="text" placeholder="e.g. Machine Learning" autocomplete="off" />
              <div class="hint">Using an existing subject’s name adds these PDFs to it.</div>
            </div>

            <div class="field">
              <span class="label">PDFs</span>
              <div class="row row-wrap">
                <label class="btn" for="subj-pdf" style="cursor:pointer">Choose PDFs…</label>
              </div>
              <input id="subj-pdf" type="file" accept="application/pdf" multiple hidden />
              <div id="subj-files" class="stack-sm" style="margin-top:.7rem"></div>
            </div>

            <button id="subj-go" class="btn-accent btn-lg btn-block" type="button">Create subject &amp; build embeddings</button>
            <div id="subj-result"></div>
          </div>

          <div class="panel stack-sm">
            <h3 style="font-family:var(--font-display);font-weight:700;letter-spacing:-.02em">Your subjects</h3>
            <div id="subj-existing" class="row row-wrap"></div>
          </div>
        </div>`);
      root.append(layout);

      const fileInput = layout.querySelector("#subj-pdf");
      const fileHost = layout.querySelector("#subj-files");
      renderFileList(fileHost);

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
      chosen = [];
    },
  };
})();
