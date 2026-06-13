/* ============================================================
   Past-paper Question Predictor view — window.Views.predictor
   Upload 1–5 past-paper PDFs (+ optional subject hint) -> Groq
   finds recurring topics/patterns and predicts likely next-exam
   questions, grouped by likelihood with expandable reasoning.
   Talks to /api/predictor/analyze; history drawer uses the
   "predictions" kind.
   ============================================================ */
(() => {
  "use strict";

  const { el, clear, esc, spinner, empty, toast } = window.ui;

  const MAX_FILES = 5;
  const LIKELIHOOD_ORDER = ["high", "medium", "low"];
  const LIKELIHOOD_LABEL = { high: "High likelihood", medium: "Medium likelihood", low: "Low likelihood" };

  function likelihoodBadge(level) {
    const cls = level === "high" ? "badge badge-accent" : "badge";
    const style = level === "low" ? ' style="opacity:.65"' : "";
    return `<span class="${cls}"${style}>${esc(level)}</span>`;
  }

  function renderPatterns(patterns) {
    const wrap = el(`<div class="panel stack-sm"></div>`);
    wrap.append(el(`<div class="section-head" style="margin:0"><h3>Recurring patterns</h3><span>${patterns.length} topic(s)</span></div>`));
    if (!patterns.length) {
      wrap.append(el(`<div class="muted">No clear recurring patterns were identified.</div>`));
      return wrap;
    }
    for (const p of patterns) {
      const row = el(`
        <div class="panel-soft stack-sm">
          <div class="row row-wrap between" style="gap:.5rem">
            <strong style="font-family:var(--font-display)">${esc(p.topic)}</strong>
            ${p.trend ? `<span class="badge badge-accent">${esc(p.trend)}</span>` : ""}
          </div>
          ${p.frequency_note ? `<p class="muted" style="font-size:.88rem">${esc(p.frequency_note)}</p>` : ""}
        </div>`);
      wrap.append(row);
    }
    return wrap;
  }

  function renderPredictions(predictions) {
    const wrap = el(`<div class="panel stack"></div>`);
    wrap.append(el(`<div class="section-head" style="margin:0"><h3>Predicted questions</h3><span>${predictions.length} question(s)</span></div>`));
    if (!predictions.length) {
      wrap.append(el(`<div class="muted">No predictions were produced.</div>`));
      return wrap;
    }
    for (const level of LIKELIHOOD_ORDER) {
      const group = predictions.filter((q) => (q.likelihood || "low") === level);
      if (!group.length) continue;
      const section = el(`<div class="stack-sm"></div>`);
      section.append(el(`<div class="label">${esc(LIKELIHOOD_LABEL[level])}</div>`));
      for (const q of group) {
        const card = el(`
          <div class="panel-soft stack-sm">
            <div class="row row-wrap between" style="gap:.5rem;align-items:flex-start">
              <p style="flex:1;min-width:200px;font-weight:600">${esc(q.question)}</p>
              ${likelihoodBadge(level)}
            </div>
            ${q.topic ? `<div class="row row-wrap" style="gap:.35rem"><span class="badge">${esc(q.topic)}</span></div>` : ""}
          </div>`);
        if (q.reasoning) {
          const toggle = el(`<button class="btn-ghost" type="button" aria-expanded="false" style="align-self:flex-start;padding:.3rem .6rem;font-size:.82rem">Why? ▸</button>`);
          const reasonBox = el(`<div class="hint" hidden style="margin:0">${esc(q.reasoning)}</div>`);
          toggle.addEventListener("click", () => {
            const open = reasonBox.hidden;
            reasonBox.hidden = !open;
            toggle.setAttribute("aria-expanded", String(open));
            toggle.textContent = open ? "Why? ▾" : "Why? ▸";
          });
          card.append(toggle, reasonBox);
        }
        section.append(card);
      }
      wrap.append(section);
    }
    return wrap;
  }

  function renderResults(data, heading) {
    const wrap = el(`<div class="stack"></div>`);
    if (heading) wrap.append(el(`<div class="section-head" style="margin:0"><h3>${esc(heading)}</h3></div>`));
    wrap.append(renderPatterns(data.patterns || []));
    wrap.append(renderPredictions(data.predicted_questions || []));
    wrap.append(el(`<p class="hint">Predictions are pattern-based guidance from past papers — not a guarantee of what will appear. Study the whole syllabus.</p>`));
    return wrap;
  }

  window.Views.predictor = {
    accent: "predictor",
    title: "Question Predictor",

    async mount(root) {
      this._busy = false;
      let pdfFiles = [];

      root.append(el(`
        <div class="layout-aside">
          <aside class="panel stack" id="pr-controls">
            <div class="between" style="align-items:center;gap:.4rem">
              <span style="font-family:var(--font-display);font-weight:700;font-size:1.05rem;letter-spacing:-.01em;color:hsl(var(--ink))">Predictor</span>
              <button id="pr-hist" class="btn" type="button" style="padding:.4rem .7rem;font-size:.82rem">≡ History</button>
            </div>

            <div class="field" style="margin:0">
              <span class="label">Past papers</span>
              <div class="row row-wrap">
                <label class="btn" for="pr-files" style="cursor:pointer">Choose PDF(s)…</label>
                <span class="muted" id="pr-files-name">No files selected</span>
              </div>
              <input id="pr-files" type="file" accept="application/pdf" multiple hidden />
              <p class="hint">Upload 1–${MAX_FILES} past exam papers (PDF). More papers give better pattern detection.</p>
            </div>

            <div class="field" style="margin:0">
              <label class="label" for="pr-hint">Subject (optional)</label>
              <input class="input" id="pr-hint" type="text" placeholder="e.g. Operating Systems" autocomplete="off" />
            </div>

            <div id="pr-modelctl"></div>

            <button class="btn-accent btn-lg btn-block" id="pr-analyze" type="button">Analyze papers</button>
            <p class="hint">This tool spots patterns in past papers and suggests likely questions — guidance, not a guarantee.</p>
          </aside>

          <section class="stack" id="pr-results"></section>
        </div>
      `));

      const filesInput = root.querySelector("#pr-files");
      const filesName = root.querySelector("#pr-files-name");
      const hintInput = root.querySelector("#pr-hint");
      const analyzeBtn = root.querySelector("#pr-analyze");
      const results = root.querySelector("#pr-results");

      root.querySelector("#pr-modelctl").append(window.ui.modelSelect());

      const showResults = (node) => { clear(results); results.append(node); };

      const openSavedPrediction = (saved) =>
        showResults(renderResults(saved || {}, (saved && saved.title) || "Saved predictions"));

      root.querySelector("#pr-hist").addEventListener("click", (e) =>
        window.ui.openHistoryDrawer("predictions", {
          title: "Saved predictions",
          anchor: e.currentTarget,
          onOpen: openSavedPrediction,
        }));

      // Deep-link handoff: a chat artifact card may have asked us to open a
      // specific saved prediction — load it via the same saved-open path.
      const _art = window.ui.takeArtifact(["predictions"]);
      if (_art) {
        window.api.get(`/history/${_art.kind}/${encodeURIComponent(_art.id)}`)
          .then(openSavedPrediction)
          .catch(() => {});
      }

      filesInput.addEventListener("change", () => {
        pdfFiles = filesInput.files ? [...filesInput.files] : [];
        if (pdfFiles.length > MAX_FILES) {
          toast(`Using only the first ${MAX_FILES} PDFs.`, "info");
          pdfFiles = pdfFiles.slice(0, MAX_FILES);
        }
        filesName.textContent = pdfFiles.length
          ? (pdfFiles.length === 1 ? pdfFiles[0].name : `${pdfFiles.length} files`)
          : "No files selected";
      });

      const setBusy = (on) => {
        this._busy = on;
        analyzeBtn.disabled = on;
        analyzeBtn.textContent = on ? "Analyzing…" : "Analyze papers";
      };

      analyzeBtn.addEventListener("click", async () => {
        if (this._busy) return;
        if (!pdfFiles.length) { toast("Choose at least one past-paper PDF.", "error"); return; }

        const fd = new FormData();
        pdfFiles.forEach((f) => fd.append("files", f));
        const hint = hintInput.value.trim();
        if (hint) fd.append("subject_hint", hint);
        const model = window.ui.selectedModel(root);
        if (model) fd.append("model", model);

        showResults(spinner("Reading papers and spotting patterns…"));
        setBusy(true);
        try {
          const data = await window.api.postForm("/predictor/analyze", fd);
          showResults(renderResults(data));
          if (data.id) toast("Predictions saved to history.", "ok");
          const hp = document.querySelector('.history-panel[data-kind="predictions"]');
          if (hp && hp.refresh) hp.refresh();
        } catch (e) {
          showResults(empty("Couldn’t analyze the papers.", "⚠"));
          toast("Analysis failed: " + e.message, "error");
        } finally {
          setBusy(false);
        }
      });

      showResults(empty("Upload past papers, then analyze to see recurring patterns and likely questions.", "🔮"));
    },

    unmount() {
      this._busy = false;
    },
  };
})();
