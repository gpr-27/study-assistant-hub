/* ============================================================
   Study Planner view — window.Views.planner
   Source (Subject / Topic / Upload PDF) + a target date -> a
   day-by-day markdown study plan. Subject mode also previews
   per-PDF summaries. Mirrors the Chat tool's three sources.
   Talks to /api/planner/{summaries,generate,topic,upload};
   renders the plan via ui.md() with a "Download .md" action.
   ============================================================ */
(() => {
  "use strict";

  const { el, clear, esc, spinner, empty, toast, md } = window.ui;

  function todayPlus(days) {
    const d = new Date();
    d.setDate(d.getDate() + days);
    return d.toISOString().slice(0, 10);
  }

  function splitSummary(text) {
    // Returns { topics, summary } when the TOPICS:/SUMMARY: layout is present,
    // otherwise { body } with the raw text.
    const src = String(text || "");
    if (src.includes("TOPICS:") && src.includes("SUMMARY:")) {
      const afterTopics = src.split("TOPICS:")[1];
      const topics = afterTopics.split("SUMMARY:")[0].trim();
      const summary = src.split("SUMMARY:")[1].trim();
      return { topics, summary };
    }
    return { body: src.trim() };
  }

  window.Views.planner = {
    accent: "plan",
    title: "Study Planner",

    async mount(root) {
      this._busy = false;
      let source = "subject";   // "subject" | "topic" | "upload"
      let pdfFiles = [];        // File[] chosen in upload mode

      root.append(el(`
        <div class="layout-aside">
          <aside class="panel stack" id="pl-controls">
            <div class="between" style="align-items:center">
              <span style="font-family:var(--font-display);font-weight:700;font-size:1.05rem;letter-spacing:-.01em;color:hsl(var(--ink))">Planner</span>
              <button id="pl-hist" class="btn" type="button" style="padding:.4rem .8rem;font-size:.85rem">≡ History</button>
            </div>
            <div class="field" style="margin:0">
              <span class="label">Source</span>
              <div class="seg" id="pl-src" role="tablist" aria-label="Plan source">
                <button type="button" data-src="subject" aria-pressed="true">Subject</button>
                <button type="button" data-src="topic" aria-pressed="false">Topic</button>
                <button type="button" data-src="upload" aria-pressed="false">Upload PDF</button>
              </div>
            </div>

            <div id="pl-srcctl" class="stack"></div>

            <div class="field" style="margin:0">
              <label class="label" for="pl-date">Target completion date</label>
              <input class="input" type="date" id="pl-date" />
              <p class="hint">Defaults to one week from today.</p>
            </div>

            <div id="pl-modelctl"></div>

            <div class="stack-sm">
              <button class="btn btn-block" id="pl-preview" type="button">Preview summaries</button>
              <button class="btn-accent btn-block" id="pl-generate" type="button">Generate plan</button>
            </div>
          </aside>

          <section class="stack" id="pl-results"></section>
        </div>
      `));

      const srcSeg = root.querySelector("#pl-src");
      const srcCtl = root.querySelector("#pl-srcctl");
      const dateInput = root.querySelector("#pl-date");
      const previewBtn = root.querySelector("#pl-preview");
      const generateBtn = root.querySelector("#pl-generate");
      const results = root.querySelector("#pl-results");

      function openSavedPlan(saved) {
        const wrap = el('<div class="stack"></div>');
        wrap.append(el(`<div class="section-head" style="margin:0"><h3>${esc((saved && saved.title) || "Study plan")}</h3></div>`));
        const planBox = md((saved && saved.plan) || "");
        planBox.classList.add("panel", "day");
        wrap.append(planBox);
        showResults(wrap);
      }
      root.querySelector("#pl-hist").addEventListener("click", (e) =>
        window.ui.openHistoryDrawer("plans", { title: "Saved plans", anchor: e.currentTarget, onOpen: openSavedPlan }));

      root.querySelector("#pl-modelctl").append(window.ui.modelSelect());

      dateInput.value = todayPlus(7);
      dateInput.min = todayPlus(0);

      const selectedPdfs = () =>
        [...srcCtl.querySelectorAll("input[type=checkbox]:checked")].map((c) => c.value);

      const setBusy = (on) => {
        this._busy = on;
        previewBtn.disabled = on;
        generateBtn.disabled = on;
        srcSeg.querySelectorAll("button").forEach((b) => (b.disabled = on));
      };

      const showResults = (node) => {
        clear(results);
        results.append(node);
      };

      // ---- subject mode: subject select + PDF checklist ----
      async function buildSubjectMode() {
        const wrap = el(`
          <div class="stack">
            <div class="field" style="margin:0">
              <label class="label" for="pl-subject">Subject</label>
              <select class="select" id="pl-subject" aria-label="Select subject">
                <option value="">Loading subjects…</option>
              </select>
            </div>
            <div class="field" style="margin:0">
              <div class="between">
                <label class="label" for="pl-pdflist">Lecture PDFs</label>
                <span class="badge badge-accent" id="pl-count">0 selected</span>
              </div>
              <div class="panel-soft stack-sm" id="pl-pdflist" role="group" aria-label="Choose PDFs">
                <div class="muted">Pick a subject to list its PDFs.</div>
              </div>
              <p class="hint">Tick the documents to include in your plan.</p>
            </div>
          </div>`);
        srcCtl.append(wrap);

        const subjectSel = wrap.querySelector("#pl-subject");
        const pdfList = wrap.querySelector("#pl-pdflist");
        const countBadge = wrap.querySelector("#pl-count");

        const refreshCount = () => { countBadge.textContent = `${selectedPdfs().length} selected`; };

        const loadPdfs = async (subject) => {
          pdfList.replaceChildren(spinner("Loading PDFs…"));
          refreshCount();
          try {
            const data = await window.api.get(`/subjects/${encodeURIComponent(subject)}/pdfs`);
            const pdfs = data.pdfs || [];
            if (!pdfs.length) {
              pdfList.replaceChildren(el(`<div class="muted">No PDFs found for “${esc(subject)}”.</div>`));
              return;
            }
            clear(pdfList);
            for (const p of pdfs) {
              const row = el(`
                <label class="row between" style="cursor:pointer;gap:.6rem">
                  <span class="row" style="gap:.55rem">
                    <input type="checkbox" value="${esc(p.name)}" />
                    <span class="soft">${esc(p.name)}</span>
                  </span>
                  <span class="badge">${esc(p.pages)} pg</span>
                </label>
              `);
              row.querySelector("input").addEventListener("change", refreshCount);
              pdfList.append(row);
            }
          } catch (e) {
            pdfList.replaceChildren(el(`<div class="muted">Couldn’t load PDFs.</div>`));
            toast("Couldn’t load PDFs: " + e.message, "error");
          }
        };

        try {
          const data = await window.api.get("/subjects");
          const subjects = data.subjects || [];
          clear(subjectSel);
          if (!subjects.length) {
            subjectSel.append(el(`<option value="">No subjects found</option>`));
            pdfList.replaceChildren(el(`<div class="muted">Add a subject, or use Topic / Upload PDF.</div>`));
            return;
          }
          subjectSel.append(el(`<option value="" disabled selected>Select a subject…</option>`));
          for (const s of subjects) subjectSel.append(el(`<option value="${esc(s)}">${esc(s)}</option>`));
          subjectSel.addEventListener("change", () => loadPdfs(subjectSel.value));
        } catch (e) {
          clear(subjectSel);
          subjectSel.append(el(`<option value="">Couldn’t load subjects</option>`));
          toast("Couldn’t load subjects: " + e.message, "error");
        }
      }

      function buildTopicMode() {
        srcCtl.append(el(`
          <div class="field" style="margin:0">
            <label class="label" for="pl-topic">Topic</label>
            <input class="input" id="pl-topic" type="text" placeholder="e.g. Linear algebra" autocomplete="off" />
            <p class="hint">A plan to learn this topic by your target date — no documents needed.</p>
          </div>`));
      }

      function buildUploadMode() {
        const wrap = el(`
          <div class="field" style="margin:0">
            <label class="label">PDFs</label>
            <div class="row row-wrap">
              <label class="btn" for="pl-files" style="cursor:pointer">Choose PDF(s)…</label>
              <span class="muted" id="pl-files-name">No files selected</span>
            </div>
            <input id="pl-files" type="file" accept="application/pdf" multiple hidden />
            <p class="hint">Upload one or more PDFs to plan around.</p>
          </div>`);
        srcCtl.append(wrap);
        const input = wrap.querySelector("#pl-files");
        const nameSlot = wrap.querySelector("#pl-files-name");
        input.addEventListener("change", () => {
          pdfFiles = input.files ? [...input.files] : [];
          nameSlot.textContent = pdfFiles.length
            ? (pdfFiles.length === 1 ? pdfFiles[0].name : `${pdfFiles.length} files`)
            : "No files selected";
        });
      }

      function buildSource() {
        clear(srcCtl);
        pdfFiles = [];
        previewBtn.style.display = source === "subject" ? "" : "none";   // preview is subject-only
        if (source === "subject") buildSubjectMode();
        else if (source === "topic") buildTopicMode();
        else buildUploadMode();
      }

      srcSeg.addEventListener("click", (e) => {
        const btn = e.target.closest("button[data-src]");
        if (!btn) return;
        source = btn.dataset.src;
        srcSeg.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", String(b === btn)));
        buildSource();
      });

      // ---- preview summaries (subject mode only) ----
      previewBtn.addEventListener("click", async () => {
        if (this._busy) return;
        const subjectSel = srcCtl.querySelector("#pl-subject");
        const subject = subjectSel && subjectSel.value;
        const pdfs = selectedPdfs();
        if (!subject) { toast("Choose a subject first.", "error"); return; }
        if (!pdfs.length) { toast("Select at least one PDF.", "error"); return; }

        showResults(spinner("Summarizing PDFs…"));
        setBusy(true);
        try {
          const data = await window.api.post("/planner/summaries", { subject, pdfs, model: window.ui.selectedModel(root) });
          const summaries = data.summaries || {};
          const names = Object.keys(summaries);
          if (!names.length) { showResults(empty("No summaries were produced.")); return; }

          const wrap = el(`<div class="stack"></div>`);
          wrap.append(el(`<div class="section-head"><h3>PDF summaries</h3><span>${names.length} document(s)</span></div>`));
          for (const name of names) {
            const parts = splitSummary(summaries[name]);
            const card = el(`<article class="panel-soft stack-sm"></article>`);
            card.append(el(`<h3 style="font-family:var(--font-display);font-weight:700">${esc(name)}</h3>`));
            if (parts.topics !== undefined) {
              card.append(el(`<div class="label">Topics &amp; concepts</div>`));
              card.append(md(parts.topics));
              card.append(el(`<div class="label">Overall summary</div>`));
              card.append(md(parts.summary));
            } else {
              card.append(md(parts.body));
            }
            wrap.append(card);
          }
          showResults(wrap);
        } catch (e) {
          showResults(empty("Couldn’t generate summaries.", "⚠"));
          toast("Summaries failed: " + e.message, "error");
        } finally {
          setBusy(false);
        }
      });

      // ---- generate plan (branches on the chosen source) ----
      generateBtn.addEventListener("click", async () => {
        if (this._busy) return;
        const target_date = dateInput.value;
        if (!target_date) { toast("Pick a target date.", "error"); return; }

        let request;
        let dlName;
        if (source === "subject") {
          const subjectSel = srcCtl.querySelector("#pl-subject");
          const subject = subjectSel && subjectSel.value;
          const pdfs = selectedPdfs();
          if (!subject) { toast("Choose a subject first.", "error"); return; }
          if (!pdfs.length) { toast("Select at least one PDF.", "error"); return; }
          request = () => window.api.post("/planner/generate", { subject, pdfs, target_date, model: window.ui.selectedModel(root) });
          dlName = `study_plan_${subject}_${target_date}.md`;
        } else if (source === "topic") {
          const topicEl = srcCtl.querySelector("#pl-topic");
          const topic = topicEl ? topicEl.value.trim() : "";
          if (!topic) { toast("Enter a topic first.", "error"); return; }
          request = () => window.api.post("/planner/topic", { topic, target_date, model: window.ui.selectedModel(root) });
          dlName = `study_plan_${topic}_${target_date}.md`;
        } else {
          if (!pdfFiles.length) { toast("Choose at least one PDF.", "error"); return; }
          const fd = new FormData();
          pdfFiles.forEach((f) => fd.append("files", f));
          fd.append("target_date", target_date);
          const __m = window.ui.selectedModel(root); if (__m) fd.append("model", __m);
          request = () => window.api.postForm("/planner/upload", fd);
          dlName = `study_plan_${target_date}.md`;
        }

        showResults(spinner("Building your study plan…"));
        setBusy(true);
        try {
          const data = await request();
          const plan = data.plan || "";
          if (!plan.trim()) { showResults(empty("The model returned an empty plan.")); return; }

          const wrap = el(`<div class="stack"></div>`);
          const head = el(`<div class="between"><div class="section-head" style="margin:0"><h3>Your study plan</h3></div></div>`);
          const dl = el(`<button class="btn-ghost" type="button">Download .md</button>`);
          dl.addEventListener("click", () => {
            const blob = new Blob([plan], { type: "text/markdown" });
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = dlName;
            a.click();
            URL.revokeObjectURL(url);
            toast("Downloaded study plan.", "ok");
          });
          head.append(dl);
          wrap.append(head);
          const planBox = md(plan);
          planBox.classList.add("panel", "day");
          wrap.append(planBox);
          showResults(wrap);
          const __hp = document.querySelector('.history-panel[data-kind="plans"]');
          if (__hp && __hp.refresh) __hp.refresh();
        } catch (e) {
          showResults(empty("Couldn’t generate the plan.", "⚠"));
          toast("Plan failed: " + e.message, "error");
        } finally {
          setBusy(false);
        }
      });

      showResults(empty("Choose a source, set a date, then preview summaries or generate a plan.", "🗓"));
      buildSource();
    },

    unmount() {
      this._busy = false;
    },
  };
})();
