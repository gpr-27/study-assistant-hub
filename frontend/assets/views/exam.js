/* ============================================================
   Mock Exam Simulator view — window.Views.exam
   Setup (subject + PDF checklist + counts + optional timer +
   model) -> one-page exam (MCQ option buttons + short-answer
   textareas, countdown auto-submit) -> POST /api/exam/grade ->
   report (big score, per-question feedback, topic bars).
   History drawer (kind "exams") reopens past reports.
   Uses window.api, window.ui and the existing CSS kit only.
   ============================================================ */
(() => {
  "use strict";
  const { el, clear, esc, spinner, empty, toast, md } = window.ui;

  const MCQ_COUNTS = [2, 4, 6, 8, 10];
  const SHORT_COUNTS = [0, 1, 2, 3, 4];
  const TIMERS = [0, 10, 20, 30]; // minutes; 0 = no timer

  let root = null;
  let timerHandle = null;
  const GU = () => (window.GradeUI ? Promise.resolve(window.GradeUI) : window.ui.loadScript("/assets/grade-ui.js").then(() => window.GradeUI));

  function stopTimer() {
    if (timerHandle) { clearInterval(timerHandle); timerHandle = null; }
  }

  function histButton(anchorPanel) {
    const btn = el(`<button class="icon-btn ib-outline" type="button" aria-label="Open past exams" title="Open past exams">${window.ui.iconHtml("history")}</button>`);
    btn.addEventListener("click", (e) =>
      window.ui.openHistoryDrawer("exams", {
        title: "Past exams",
        anchor: e.currentTarget,
        onOpen: (saved) => renderReport(saved, true),
      }));
    return btn;
  }

  /* ---------------- setup screen ---------------- */
  async function renderSetup() {
    stopTimer();
    clear(root);

    let source = "subject";   // "subject" | "topic" | "upload"
    let pdfFile = null;

    const panel = el(`
      <div class="panel stack" style="max-width:640px;margin:0 auto">
        <div class="between" style="align-items:flex-start;gap:.75rem">
          <div class="stack-sm" style="flex:1;min-width:0">
            <h3 style="font-family:var(--font-display);font-weight:700;letter-spacing:-.02em;overflow-wrap:anywhere">Mock exam</h3>
            <p class="muted">Pick a source — the AI sets a timed paper of MCQs and short answers, then grades it.</p>
          </div>
          <span id="ex-hist-slot" style="flex:none"></span>
        </div>

        <div class="field">
          <span class="label" id="ex-src-label">Source</span>
          <div class="seg" id="ex-src" role="tablist" aria-labelledby="ex-src-label" style="flex-wrap:wrap">
            <button type="button" data-src="subject" aria-pressed="true">Subject</button>
            <button type="button" data-src="topic" aria-pressed="false">Topic</button>
            <button type="button" data-src="upload" aria-pressed="false">Upload PDF</button>
          </div>
        </div>

        <div id="ex-srcctl"></div>

        <div class="grid-2">
          <div class="field" style="margin:0">
            <label class="label" for="ex-mcq">Multiple choice</label>
            <select class="select" id="ex-mcq" aria-label="Number of MCQs"></select>
          </div>
          <div class="field" style="margin:0">
            <label class="label" for="ex-short">Short answers</label>
            <select class="select" id="ex-short" aria-label="Number of short-answer questions"></select>
          </div>
        </div>

        <div class="field" style="margin:0">
          <label class="label" for="ex-timer">Timer</label>
          <select class="select" id="ex-timer" aria-label="Exam timer"></select>
          <p class="hint">The exam auto-submits when the timer hits zero.</p>
        </div>

        <div id="ex-modelctl"></div>

        <button id="ex-go" class="btn-accent btn-lg btn-block" type="button">Start exam</button>
      </div>
    `);
    root.append(panel);
    panel.querySelector("#ex-hist-slot").append(histButton(panel));

    const srcSeg = panel.querySelector("#ex-src");
    const srcCtl = panel.querySelector("#ex-srcctl");
    const mcqSel = panel.querySelector("#ex-mcq");
    const shortSel = panel.querySelector("#ex-short");
    const timerSel = panel.querySelector("#ex-timer");
    const goBtn = panel.querySelector("#ex-go");

    MCQ_COUNTS.forEach((n) => mcqSel.append(el(`<option value="${n}">${n} questions</option>`)));
    mcqSel.value = "6";
    SHORT_COUNTS.forEach((n) => shortSel.append(el(`<option value="${n}">${n === 0 ? "None" : n + " question" + (n === 1 ? "" : "s")}</option>`)));
    shortSel.value = "2";
    TIMERS.forEach((m) => timerSel.append(el(`<option value="${m}">${m === 0 ? "No timer" : m + " minutes"}</option>`)));
    timerSel.value = "0";

    panel.querySelector("#ex-modelctl").append(window.ui.modelSelect());

    // ---- subject mode: load a subject's PDFs into a tickable checklist ----
    const loadPdfs = async (subject, pdfList, countBadge) => {
      const selected = () =>
        [...pdfList.querySelectorAll("input[type=checkbox]:checked")].map((c) => c.value);
      const refreshCount = () => { countBadge.textContent = `${selected().length} selected`; };
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
            <label class="row between" style="cursor:pointer;gap:.6rem;min-width:0">
              <span class="row" style="gap:.55rem;min-width:0;flex:1">
                <input type="checkbox" value="${esc(p.name)}" aria-label="${esc(p.name)}" />
                <span class="soft" style="min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${esc(p.name)}">${esc(p.name)}</span>
              </span>
              <span class="badge" style="flex:none">${esc(p.pages)} pg</span>
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

    async function buildSubjectMode() {
      const wrap = el(`
        <div class="stack-sm">
          <div class="field" style="margin:0">
            <label class="label" for="ex-subject">Subject</label>
            <select class="select" id="ex-subject" aria-label="Select subject"><option value="">Loading subjects…</option></select>
          </div>
          <div class="field" style="margin:0">
            <div class="between">
              <label class="label" for="ex-pdflist">Lecture PDFs</label>
              <span class="badge badge-accent" id="ex-count">0 selected</span>
            </div>
            <div class="panel-soft stack-sm" id="ex-pdflist" role="group" aria-label="Choose PDFs">
              <div class="muted">Pick a subject to list its PDFs.</div>
            </div>
            <p class="hint">Tick the documents the exam should cover.</p>
          </div>
        </div>`);
      srcCtl.append(wrap);
      const subjectSel = wrap.querySelector("#ex-subject");
      const pdfList = wrap.querySelector("#ex-pdflist");
      const countBadge = wrap.querySelector("#ex-count");
      try {
        const data = await window.api.get("/subjects");
        const subjects = data.subjects || [];
        clear(subjectSel);
        if (!subjects.length) {
          subjectSel.append(el(`<option value="">No subjects available</option>`));
          pdfList.replaceChildren(el(`<div class="muted">Add a subject, or try Topic / Upload PDF.</div>`));
        } else {
          subjectSel.append(el(`<option value="" disabled selected>Select a subject…</option>`));
          subjects.forEach((s) => subjectSel.append(el(`<option value="${esc(s)}">${esc(s)}</option>`)));
          subjectSel.addEventListener("change", () => loadPdfs(subjectSel.value, pdfList, countBadge));
        }
      } catch (e) {
        clear(subjectSel);
        subjectSel.append(el(`<option value="">Couldn’t load subjects</option>`));
        toast("Couldn’t load subjects: " + e.message, "error");
      }
    }

    function buildTopicMode() {
      srcCtl.append(el(`
        <div class="field" style="margin:0">
          <label class="label" for="ex-topic">Topic</label>
          <input id="ex-topic" class="input" type="text" placeholder="e.g. Binary search trees" autocomplete="off" />
          <div class="hint">The paper is written from the topic — no document needed.</div>
        </div>`));
    }

    function buildUploadMode() {
      const wrap = el(`
        <div class="field" style="margin:0">
          <label class="label" for="ex-file">PDF</label>
          <div class="row row-wrap" style="align-items:center">
            <label class="btn" for="ex-file" style="cursor:pointer;flex:none">Choose PDF…</label>
            <span class="muted" id="ex-file-name" style="min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">No file selected</span>
          </div>
          <input id="ex-file" type="file" accept="application/pdf" hidden aria-label="Upload PDF for mock exam" />
          <div class="hint">Upload a PDF and sit an exam on its contents.</div>
        </div>`);
      srcCtl.append(wrap);
      const input = wrap.querySelector("#ex-file");
      const nameSlot = wrap.querySelector("#ex-file-name");
      input.addEventListener("change", () => {
        pdfFile = input.files && input.files[0] ? input.files[0] : null;
        nameSlot.textContent = pdfFile ? pdfFile.name : "No file selected";
        nameSlot.title = pdfFile ? pdfFile.name : "";
      });
    }

    function buildSource() {
      clear(srcCtl);
      pdfFile = null;
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

    buildSource();

    goBtn.addEventListener("click", async () => {
      const numMcq = Number(mcqSel.value);
      const numShort = Number(shortSel.value);
      if (numMcq + numShort === 0) { toast("Pick at least one question.", "error"); return; }
      const model = window.ui.selectedModel(panel);
      const minutes = Number(timerSel.value);

      let label, request;
      if (source === "subject") {
        const subjectSel = srcCtl.querySelector("#ex-subject");
        const subject = subjectSel ? subjectSel.value : "";
        const pdfs = [...srcCtl.querySelectorAll("#ex-pdflist input[type=checkbox]:checked")].map((c) => c.value);
        if (!subject) { toast("Choose a subject first.", "error"); return; }
        if (!pdfs.length) { toast("Select at least one PDF.", "error"); return; }
        label = subject;
        request = () => window.api.post("/exam/generate", {
          subject, pdfs, num_mcq: numMcq, num_short: numShort, model,
        });
      } else if (source === "topic") {
        const topicEl = srcCtl.querySelector("#ex-topic");
        const topic = topicEl ? topicEl.value.trim() : "";
        if (!topic) { toast("Enter a topic first.", "error"); return; }
        label = topic;
        request = () => window.api.post("/exam/topic", {
          topic, num_mcq: numMcq, num_short: numShort, model,
        });
      } else {
        if (!pdfFile) { toast("Choose a PDF first.", "error"); return; }
        label = pdfFile.name;
        const fd = new FormData();
        fd.append("file", pdfFile);
        fd.append("num_mcq", String(numMcq));
        fd.append("num_short", String(numShort));
        if (model) fd.append("model", model);
        request = () => window.api.postForm("/exam/upload", fd);
      }

      goBtn.disabled = true;
      const original = goBtn.textContent;
      goBtn.textContent = "Setting the paper…";
      const load = spinner(source === "topic" ? "Writing the exam…" : "Reading and writing the exam…");
      panel.append(load);
      try {
        const res = await request();
        const questions = (res && res.questions) || [];
        if (!questions.length) throw new Error("No questions came back; try again.");
        renderExam(label, questions, minutes, model, res && res.llm);
      } catch (e) {
        toast(e.message, "error");
        goBtn.disabled = false;
        goBtn.textContent = original;
        load.replaceWith(window.ui.errorBlock(e, e.retryable ? () => goBtn.click() : null));
      }
    });
  }

  /* ---------------- exam runner (one-page paper) ---------------- */
  function renderExam(subject, questions, minutes, model, llm) {
    stopTimer();
    clear(root);
    const answers = {};   // id -> selected option index (mcq) | text (short)
    const startedAt = Date.now();
    const attemptId = "ex" + Date.now().toString(36) + Math.random().toString(36).slice(2, 10);   // idempotency key: retry / double-click safe
    let submitted = false;
    GU().catch(() => {});

    const panel = el(`
      <div class="panel stack" style="max-width:680px;margin:0 auto">
        <div class="between">
          <span class="badge" id="ex-meta"></span>
          <span class="badge badge-accent" id="ex-clock" hidden>--:--</span>
        </div>
        <div id="ex-paper" class="stack"></div>
        <button id="ex-submit" class="btn-accent btn-lg btn-block" type="button">Submit exam</button>
      </div>
    `);
    root.append(panel);
    { const l = window.ui.renderModelLabel(llm); if (l) panel.insertBefore(l, panel.querySelector("#ex-paper")); }   // "Model: …" (exam generator)

    const metaEl = panel.querySelector("#ex-meta");
    metaEl.textContent = `${subject} · ${questions.length} question${questions.length === 1 ? "" : "s"}`;
    metaEl.title = metaEl.textContent;
    metaEl.style.cssText = "min-width:0;max-width:100%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap";
    const paper = panel.querySelector("#ex-paper");
    const clock = panel.querySelector("#ex-clock");
    clock.setAttribute("aria-live", "polite");
    const submitBtn = panel.querySelector("#ex-submit");

    questions.forEach((q, i) => {
      const card = el(`<article class="panel-soft stack-sm"></article>`);
      card.append(el(`
        <div class="between" style="gap:.5rem;align-items:flex-start">
          <span class="badge" style="flex:none">Q${i + 1}</span>
          <span class="badge" style="min-width:0;max-width:100%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${esc((q.type === "mcq" ? "1 mark" : "5 marks") + " · " + (q.topic || "General"))}">${q.type === "mcq" ? "1 mark" : "5 marks"} · ${esc(q.topic || "General")}</span>
        </div>`));
      card.append(el(`<p style="font-weight:600;line-height:1.45;overflow-wrap:anywhere">${esc(q.question)}</p>`));

      if (q.type === "mcq") {
        const opts = el(`<div class="opts" role="group" aria-label="Answer options for question ${i + 1}" style="margin-top:.2rem"></div>`);
        const buttons = [];
        (q.options || []).forEach((text, idx) => {
          const key = String.fromCharCode(65 + idx);
          const b = el(`<button class="opt" type="button" aria-pressed="false" aria-label="Option ${key}: ${esc(text)}"><span class="key">${key}</span><span style="min-width:0;overflow-wrap:anywhere">${esc(text)}</span></button>`);
          b.addEventListener("click", () => {
            answers[q.id] = idx;
            buttons.forEach((other, j) => other.setAttribute("aria-pressed", String(j === idx)));
          });
          buttons.push(b);
          opts.append(b);
        });
        card.append(opts);
      } else {
        const ta = el(`<textarea class="textarea" rows="4" placeholder="Type your answer…" aria-label="Your answer for question ${i + 1}" style="width:100%;box-sizing:border-box;resize:vertical"></textarea>`);
        ta.addEventListener("input", () => { answers[q.id] = ta.value; });
        card.append(ta);
      }
      paper.append(card);
    });

    async function submit(auto) {
      if (submitted) return;
      submitted = true;
      stopTimer();
      submitBtn.disabled = true;
      submitBtn.textContent = "Grading…";
      const load = spinner(auto ? "Time’s up — grading your answers…" : "Grading your answers…");
      panel.append(load);
      try {
        const payload = {
          kind: "exam", subject, title: "Exam \u00b7 " + subject, items: questions, answers,
          attempt_id: attemptId, duration_seconds: Math.round((Date.now() - startedAt) / 1000),
          generated_llm: llm || undefined,       // lets the saved report name who wrote the paper
        };
        const GradeUI = await GU();
        const result = await window.api.post("/grade", payload);
        if (result && result.saved) toast("Exam report saved to history.", "ok");
        window.api.invalidate && window.api.invalidate("/progress");
        renderGraded(result, { payload, items: questions });
      } catch (e) {
        toast("Grading failed: " + e.message, "error");
        submitted = false;
        submitBtn.disabled = false;
        submitBtn.textContent = "Submit exam";
        load.replaceWith(window.ui.errorBlock(e, e.retryable ? () => submit(false) : null));
      }
    }

    submitBtn.addEventListener("click", () => submit(false));

    if (minutes > 0) {
      clock.hidden = false;
      const deadline = Date.now() + minutes * 60000;
      const tick = () => {
        const left = Math.max(0, Math.round((deadline - Date.now()) / 1000));
        const mm = String(Math.floor(left / 60)).padStart(2, "0");
        const ss = String(left % 60).padStart(2, "0");
        clock.textContent = `⏱ ${mm}:${ss}`;
        if (left <= 60) clock.style.color = "hsl(0 80% 55%)";
        if (left <= 0) { toast("Time’s up — submitting your exam.", "info"); submit(true); }
      };
      tick();
      timerHandle = setInterval(tick, 1000);
    }
  }

  /* ---------------- report (fresh grade or reopened from history) ---------------- */
  async function renderGraded(result, extra) {
    stopTimer();
    clear(root);
    const panel = el(`
      <div class="panel stack" style="max-width:680px;margin:0 auto">
        <div class="between" style="align-items:flex-start"><span></span><span id="ex-rep-hist"></span></div>
        <div id="ex-result"></div>
      </div>`);
    root.append(panel);
    panel.querySelector("#ex-rep-hist").append(histButton(panel));
    const box = panel.querySelector("#ex-result");
    try {
      const GradeUI = await GU();
      GradeUI.renderResult(box, result, Object.assign({ newLabel: "New exam", onNew: () => renderSetup() }, extra || {}));
    } catch (e) { box.append(window.ui.errorBlock(e, () => renderGraded(result, extra))); }
  }

  function renderReport(report, fromHistory) {
    if (report && report.result && Array.isArray(report.result.items)) { renderGraded(report.result, {}); return; }   // unified (new) report
    stopTimer();
    clear(root);
    const per = (report && report.per_question) || [];
    const topicRows = (report && report.topic_breakdown) || [];
    const total = (report && report.total_score) || 0;
    const max = (report && report.max_score) || 0;
    const pct = max ? Math.round((total / max) * 100) : 0;

    let line;
    if (pct >= 90) line = "Outstanding — exam-ready.";
    else if (pct >= 75) line = "Strong performance — nearly there.";
    else if (pct >= 50) line = "A fair attempt — review the weak topics below.";
    else line = "Keep studying — the breakdown shows where to focus.";

    const dur = report && report.duration_seconds;
    const durText = dur ? ` · ${Math.floor(dur / 60)}m ${dur % 60}s` : "";

    const panel = el(`
      <div class="panel stack" style="max-width:680px;margin:0 auto">
        <div class="between" style="align-items:flex-start">
          <div class="section-head" style="margin:0"><h3>${esc((report && report.title) || "Exam report")}</h3></div>
          <span id="ex-rep-hist"></span>
        </div>
        <div class="center stack-sm" style="text-align:center">
          <div class="score-ring">${pct}%</div>
          <p style="font-family:var(--font-display);font-weight:600;font-size:1.1rem">${esc(line)}</p>
          <p class="muted">${total} of ${max} marks${esc(durText)}</p>
        </div>
        <div class="stack-sm" id="ex-topics"></div>
        <div class="stack-sm" id="ex-review"></div>
        <button id="ex-new" class="btn-accent btn-lg btn-block" type="button">New exam</button>
      </div>
    `);
    root.append(panel);
    {   // "Model: …" for the questions (generator) AND for the marking/feedback (grader), when each ran
      const anchor = panel.querySelector("#ex-new");
      const gen = window.ui.renderModelLabel(report && report.generated_llm, { caption: "Questions", stored: !!fromHistory });
      const gr = window.ui.renderModelLabel(report && report.llm, { caption: "Marking", quiet: !!fromHistory });
      if (gen) panel.insertBefore(gen, anchor);
      if (gr) panel.insertBefore(gr, anchor);
    }
    panel.querySelector("#ex-rep-hist").append(histButton(panel));

    const topicsBox = panel.querySelector("#ex-topics");
    if (topicRows.length) {
      topicsBox.append(el(`<div class="section-head" style="margin:0"><h3>Topic breakdown</h3></div>`));
      topicRows.forEach((t) => {
        const width = t.possible ? Math.round((t.earned / t.possible) * 100) : 0;
        const row = el(`
          <div class="stack-sm" style="gap:.3rem">
            <div class="between" style="gap:.5rem">
              <span class="soft" style="min-width:0;overflow-wrap:anywhere">${esc(t.topic)}</span>
              <span class="badge" style="flex:none">${esc(t.earned)} / ${esc(t.possible)}</span>
            </div>
            <div class="progress"><i style="width:${width}%"></i></div>
          </div>`);
        topicsBox.append(row);
      });
    }

    const review = panel.querySelector("#ex-review");
    if (per.length) {
      review.append(el(`<div class="section-head" style="margin:0"><h3>Per-question feedback</h3></div>`));
      per.forEach((q, i) => {
        const full = q.score === q.max_score;
        const item = el(`
          <details class="panel-soft">
            <summary style="cursor:pointer;font-weight:600">
              <span class="badge ${full ? "badge-accent" : ""}">${esc(q.score)}/${esc(q.max_score)}</span>
              Question ${i + 1} <span class="muted">· ${esc(q.topic || "General")}</span>
            </summary>
            <div class="stack-sm" style="margin-top:.8rem">
              ${q.question ? `<p><strong>${esc(q.question)}</strong></p>` : ""}
              <div class="explain" data-md></div>
            </div>
          </details>`);
        item.querySelector("[data-md]").append(md(q.feedback || ""));
        review.append(item);
      });
    } else {
      review.append(empty("No per-question details in this report."));
    }

    panel.querySelector("#ex-new").addEventListener("click", () => renderSetup());
  }

  /* ---------------- view registration ---------------- */
  window.Views.exam = {
    accent: "quiz",
    title: "Mock Exam",
    async mount(node) {
      root = node;
      await renderSetup();
      // Library deep-link: open the saved exam report instead of leaving Setup showing.
      const _art = window.ui.takeArtifact(["exams"]);
      if (_art) window.api.get(`/history/${_art.kind}/${encodeURIComponent(_art.id)}`).then((saved) => renderReport(saved, true)).catch(() => {});
    },
    unmount() {
      stopTimer();
      if (root) clear(root);
      root = null;
    },
  };
})();
