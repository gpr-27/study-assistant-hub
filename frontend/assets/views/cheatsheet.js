/* ============================================================
   Cheat-sheet Generator view — window.Views.cheatsheet
   Condense one PDF (from a subject, or uploaded) into an
   ultra-dense one-page exam crib sheet. Renders the markdown
   in a printable panel with "Download .md" and "Print" actions.
   Talks to /api/cheatsheet/{generate,upload}.
   ============================================================ */
(() => {
  "use strict";
  const { el, clear, esc, spinner, toast, md } = window.ui;

  let source = "subject";   // "subject" | "upload"
  let pdfFile = null;       // File for upload mode

  function buildSourceControl(host) {
    clear(host);
    if (source === "subject") {
      const field = el(`
        <div class="stack-sm">
          <div class="field" style="margin:0">
            <label class="label" for="csSubject">Subject</label>
            <select id="csSubject" class="select"><option value="">Loading subjects…</option></select>
          </div>
          <div class="field" style="margin:0">
            <label class="label" for="csPdf">Document (pick one)</label>
            <select id="csPdf" class="select" disabled><option value="">Select a subject first</option></select>
          </div>
        </div>`);
      host.append(field);
      const subjectSel = field.querySelector("#csSubject");
      const pdfSel = field.querySelector("#csPdf");

      function loadPdfs(subject) {
        pdfSel.disabled = true;
        clear(pdfSel);
        pdfSel.append(el(`<option>Loading documents…</option>`));
        window.api.get(`/subjects/${encodeURIComponent(subject)}/pdfs`).then((data) => {
          const pdfs = data.pdfs || [];
          clear(pdfSel);
          if (!pdfs.length) {
            pdfSel.append(el(`<option value="">No PDFs in this subject</option>`));
            return;
          }
          pdfs.forEach((p) => pdfSel.append(el(`<option value="${esc(p.name)}">${esc(p.name)}</option>`)));
          pdfSel.value = pdfs[0].name;
          pdfSel.disabled = false;
        }).catch((e) => { clear(pdfSel); pdfSel.append(el(`<option value="">Couldn’t load documents</option>`)); toast(e.message, "error"); });
      }

      window.api.get("/subjects").then((d) => {
        const subjects = d.subjects || [];
        clear(subjectSel);
        if (!subjects.length) { subjectSel.append(el(`<option value="">No subjects found</option>`)); return; }
        subjects.forEach((s) => subjectSel.append(el(`<option value="${esc(s)}">${esc(s)}</option>`)));
        subjectSel.value = subjects[0];
        loadPdfs(subjects[0]);
      }).catch((e) => { clear(subjectSel); subjectSel.append(el(`<option value="">Couldn’t load subjects</option>`)); toast(e.message, "error"); });

      subjectSel.addEventListener("change", () => { if (subjectSel.value) loadPdfs(subjectSel.value); });
    } else {
      const field = el(`
        <div class="field" style="margin:0">
          <label class="label">PDF</label>
          <div class="row row-wrap">
            <label class="btn" for="csPdfFile" style="cursor:pointer">Choose PDF…</label>
            <span class="muted" id="csPdfName">No file selected</span>
          </div>
          <input id="csPdfFile" type="file" accept="application/pdf" hidden />
          <div class="hint">Upload one PDF and squeeze it onto one page.</div>
        </div>`);
      host.append(field);
      const input = field.querySelector("#csPdfFile");
      const nameSlot = field.querySelector("#csPdfName");
      input.addEventListener("change", () => {
        pdfFile = input.files && input.files[0] ? input.files[0] : null;
        nameSlot.textContent = pdfFile ? pdfFile.name : "No file selected";
      });
    }
  }

  // Render a generated/saved sheet into the result panel with its actions.
  function showSheet(result, markdown, title) {
    clear(result);
    const head = el(`
      <div class="row row-wrap between" style="margin-bottom:1rem">
        <h3 style="font-family:var(--font-display);font-weight:700;letter-spacing:-.02em;margin:0">${esc(title || "Cheat sheet")}</h3>
        <span class="row row-wrap" style="gap:.4rem;flex:none"></span>
      </div>`);
    const actions = head.querySelector("span");

    const dl = el(`<button class="btn-ghost" type="button">Download .md</button>`);
    dl.addEventListener("click", () => {
      const blob = new Blob([markdown], { type: "text/markdown" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${(title || "cheat_sheet").replace(/[^\w.-]+/g, "_")}.md`;
      a.click();
      URL.revokeObjectURL(url);
      toast("Downloaded cheat sheet.", "ok");
    });
    const sheet = md(markdown);
    sheet.classList.add("panel", "day");

    const printBtn = el(`<button class="btn" type="button">🖨 Print / PDF</button>`);
    printBtn.addEventListener("click", () =>
      window.ui.printDoc(title || "Cheat sheet", sheet.innerHTML));
    actions.append(dl, printBtn);

    result.append(head, sheet);
  }

  async function generate(layout) {
    const goBtn = layout.querySelector("#csGo");
    const result = layout.querySelector("#csResult");
    const model = window.ui.selectedModel(layout);

    let call;
    if (source === "subject") {
      const subject = layout.querySelector("#csSubject")?.value;
      const pdf = layout.querySelector("#csPdf")?.value;
      if (!subject) { toast("Pick a subject.", "error"); return; }
      if (!pdf) { toast("Pick a document.", "error"); return; }
      call = () => window.api.post("/cheatsheet/generate", { mode: "subject", subject, pdf, model });
    } else {
      if (!pdfFile) { toast("Choose a PDF.", "error"); return; }
      call = () => { const fd = new FormData(); fd.append("file", pdfFile); if (model) fd.append("model", model); return window.api.postForm("/cheatsheet/upload", fd); };
    }

    goBtn.disabled = true;
    goBtn.textContent = "Condensing…";
    clear(result);
    result.append(spinner("Squeezing the essentials onto one page…"));

    try {
      const resp = await call();
      showSheet(result, resp.markdown || "", resp.title);
      toast("Cheat sheet ready. Download it or print it.", "ok");
      const __hp = document.querySelector('.history-panel[data-kind="cheatsheets"]');
      if (__hp && __hp.refresh) __hp.refresh();
    } catch (e) {
      clear(result);
      result.append(el(`<div class="empty"><div class="big">⚠</div><div>${esc(e.message)}</div></div>`));
      toast(e.message, "error");
    } finally {
      goBtn.disabled = false;
      goBtn.textContent = "Generate cheat sheet";
    }
  }

  window.Views = window.Views || {};
  window.Views.cheatsheet = {
    accent: "cheatsheet",
    title: "Cheat Sheet",
    async mount(root) {
      const layout = el(`
        <div class="stack">
          <div class="panel stack">
            <div class="between" style="align-items:flex-start">
              <div class="stack-sm" style="flex:1">
                <h3 style="font-family:var(--font-display);font-weight:700;letter-spacing:-.02em">Generate a cheat sheet</h3>
                <p class="muted">One PDF in, one ultra-dense exam crib sheet out — definitions, formulas, contrasts, pitfalls.</p>
              </div>
              <button id="cs-hist" class="btn" type="button" style="flex:none;padding:.4rem .8rem;font-size:.85rem">≡ History</button>
            </div>
            <div class="seg" role="tablist" aria-label="Cheat sheet source">
              <button type="button" data-src="subject" aria-pressed="true">Subject PDF</button>
              <button type="button" data-src="upload" aria-pressed="false">Upload PDF</button>
            </div>
            <div id="csSourceCtl"></div>
            <div id="csModelCtl"></div>
            <button id="csGo" class="btn-accent btn-lg btn-block" type="button">Generate cheat sheet</button>
          </div>
          <div class="panel" id="csResult">
            <div class="empty"><div class="big">✦</div><div>Pick a PDF and generate your one-page crib sheet.</div></div>
          </div>
        </div>`);
      root.append(layout);

      function openSavedSheet(saved) {
        if (!saved || !saved.markdown) return;
        showSheet(layout.querySelector("#csResult"), saved.markdown, saved.title);
      }
      layout.querySelector("#cs-hist").addEventListener("click", (e) =>
        window.ui.openHistoryDrawer("cheatsheets", { title: "Saved cheat sheets", anchor: e.currentTarget, onOpen: openSavedSheet }));

      const seg = layout.querySelector(".seg");
      const sourceCtl = layout.querySelector("#csSourceCtl");
      buildSourceControl(sourceCtl);
      layout.querySelector("#csModelCtl").append(window.ui.modelSelect());

      seg.addEventListener("click", (e) => {
        const btn = e.target.closest("button[data-src]");
        if (!btn) return;
        source = btn.dataset.src;
        pdfFile = null;
        seg.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", b === btn ? "true" : "false"));
        buildSourceControl(sourceCtl);
      });

      layout.querySelector("#csGo").addEventListener("click", () => generate(layout));
    },
    unmount() {
      source = "subject";
      pdfFile = null;
    },
  };
})();
