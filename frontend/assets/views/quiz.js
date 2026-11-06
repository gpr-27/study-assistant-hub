/* ============================================================
   Quiz Generator view — window.Views.quiz
   Source (Subject / Topic / Upload PDF) -> Groq quiz ->
   one-question-at-a-time runner -> score. Mirrors the Chat tool's
   three sources. Uses window.api, window.ui and the CSS kit.
   ============================================================ */
(() => {
  "use strict";
  const { el, clear, esc, spinner, empty, toast } = window.ui;
  const KEYS = ["A", "B", "C", "D"];
  const DIFFICULTIES = ["easy", "medium", "hard"];

  let root = null;
  let llmLine = null;
  let lastMeta = null; // {subject, topic} of the most recent quiz — for progress recording
  const GU = () => (window.GradeUI ? Promise.resolve(window.GradeUI) : window.ui.loadScript("/assets/grade-ui.js").then(() => window.GradeUI));

  // Open a saved quiz doc (history drawer item or chat-artifact deep-link).
  function openSavedQuiz(saved) {
    lastMeta = { topic: (saved && saved.title) || null };
    renderRunner((saved && saved.questions) || [], saved && saved.llm, true, saved && saved.id);
  }

  /* ---------------- setup screen ---------------- */
  async function renderSetup() {
    clear(root);
    let source = "subject";   // "subject" | "topic" | "upload"
    let difficulty = "medium";
    let pdfFile = null;

    const panel = el(`
      <div class="panel stack" style="max-width:640px;margin:0 auto">
        <div class="between" style="align-items:flex-start;gap:.75rem">
          <div class="stack-sm" style="flex:1;min-width:0">
            <h3 style="font-family:var(--font-display);font-weight:700;letter-spacing:-.02em;overflow-wrap:anywhere">Build a quiz</h3>
            <p class="muted">Pick a source, choose how hard and how many — the AI writes the questions.</p>
          </div>
          <button id="q-hist" class="icon-btn ib-outline" type="button" aria-label="Open quiz history" title="Open quiz history">${window.ui.iconHtml("history")}</button>
        </div>

        <div class="field">
          <span class="label" id="q-src-label">Source</span>
          <div class="seg" id="q-src" role="tablist" aria-labelledby="q-src-label" style="flex-wrap:wrap">
            <button type="button" data-src="subject" aria-pressed="true">Subject</button>
            <button type="button" data-src="topic" aria-pressed="false">Topic</button>
            <button type="button" data-src="upload" aria-pressed="false">Upload PDF</button>
          </div>
        </div>

        <div id="q-srcctl"></div>

        <div class="field">
          <span class="label" id="q-diff-label">Difficulty</span>
          <div class="seg" id="q-diff" role="group" aria-labelledby="q-diff-label" style="flex-wrap:wrap"></div>
        </div>

        <div class="field">
          <div class="between">
            <label class="label" for="q-num" style="margin:0">Number of questions</label>
            <span class="badge badge-accent count" id="q-num-out" aria-live="polite">5</span>
          </div>
          <input id="q-num" type="range" min="5" max="15" step="5" value="5" aria-valuemin="5" aria-valuemax="15" aria-valuenow="5" aria-label="Number of questions">
        </div>

        <button id="q-go" class="btn-accent btn-lg btn-block" type="button">Generate quiz</button>
      </div>
    `);
    root.append(panel);

    panel.querySelector("#q-hist").addEventListener("click", (e) =>
      window.ui.openHistoryDrawer("quizzes", {
        title: "Saved quizzes",
        anchor: e.currentTarget,
        onOpen: openSavedQuiz,
      }));

    const srcSeg = panel.querySelector("#q-src");
    const srcCtl = panel.querySelector("#q-srcctl");
    const diffSeg = panel.querySelector("#q-diff");
    const numInput = panel.querySelector("#q-num");
    const numOut = panel.querySelector("#q-num-out");
    const goBtn = panel.querySelector("#q-go");

    // difficulty segmented control
    DIFFICULTIES.forEach((level) => {
      const b = el(`<button type="button" aria-pressed="${level === difficulty}">${esc(level[0].toUpperCase() + level.slice(1))}</button>`);
      b.addEventListener("click", () => {
        difficulty = level;
        diffSeg.querySelectorAll("button").forEach((other) => other.setAttribute("aria-pressed", String(other === b)));
      });
      diffSeg.append(b);
    });

    numInput.addEventListener("input", () => {
      numOut.textContent = numInput.value;
      numInput.setAttribute("aria-valuenow", numInput.value);
    });

    // ---- subject mode: load a subject's PDFs into the document select ----
    async function loadPdfs(subject, pdfSel, pdfHint) {
      pdfSel.disabled = true;
      pdfHint.textContent = "";
      clear(pdfSel);
      pdfSel.append(el(`<option>Loading documents…</option>`));
      try {
        const data = await api.get(`/subjects/${encodeURIComponent(subject)}/pdfs`);
        const pdfs = data.pdfs || [];
        clear(pdfSel);
        if (!pdfs.length) {
          pdfSel.append(el(`<option value="">No PDFs in this subject</option>`));
          pdfHint.textContent = "No documents found here.";
          return;
        }
        pdfs.forEach((p) => pdfSel.append(el(`<option value="${esc(p.name)}">${esc(p.name)}</option>`)));
        pdfSel.disabled = false;
        const showPages = () => {
          const sel = pdfs.find((p) => p.name === pdfSel.value);
          pdfHint.textContent = sel ? `${sel.pages} page${sel.pages === 1 ? "" : "s"}` : "";
        };
        pdfSel.addEventListener("change", showPages);
        showPages();
      } catch (e) {
        clear(pdfSel);
        pdfSel.append(el(`<option value="">Couldn’t load documents</option>`));
        toast(e.message, "error");
      }
    }

    async function buildSubjectMode() {
      const wrap = el(`
        <div class="stack-sm">
          <div class="field" style="margin:0">
            <label class="label" for="q-subject">Subject</label>
            <select id="q-subject" class="select"><option>Loading…</option></select>
          </div>
          <div class="field" style="margin:0">
            <label class="label" for="q-pdf">Document</label>
            <select id="q-pdf" class="select" disabled><option>Select a subject first</option></select>
            <div class="hint" id="q-pdf-hint"></div>
          </div>
        </div>`);
      srcCtl.append(wrap);
      const subjectSel = wrap.querySelector("#q-subject");
      const pdfSel = wrap.querySelector("#q-pdf");
      const pdfHint = wrap.querySelector("#q-pdf-hint");
      try {
        const data = await api.get("/subjects");
        const subjects = data.subjects || [];
        clear(subjectSel);
        if (!subjects.length) {
          subjectSel.append(el(`<option value="">No subjects available</option>`));
          pdfHint.textContent = "Add a subject, or try Topic / Upload PDF.";
          return;
        }
        subjects.forEach((s) => subjectSel.append(el(`<option value="${esc(s)}">${esc(s)}</option>`)));
        subjectSel.addEventListener("change", () => loadPdfs(subjectSel.value, pdfSel, pdfHint));
        await loadPdfs(subjectSel.value, pdfSel, pdfHint);
      } catch (e) {
        clear(subjectSel);
        subjectSel.append(el(`<option value="">Couldn’t load subjects</option>`));
        toast(e.message, "error");
      }
    }

    function buildTopicMode() {
      srcCtl.append(el(`
        <div class="field" style="margin:0">
          <label class="label" for="q-topic">Topic</label>
          <input id="q-topic" class="input" type="text" placeholder="e.g. Binary search trees" autocomplete="off" />
          <div class="hint">Questions are written from the topic — no document needed.</div>
        </div>`));
    }

    function buildUploadMode() {
      const wrap = el(`
        <div class="field" style="margin:0">
          <label class="label" for="q-file">PDF</label>
          <div class="row row-wrap" style="align-items:center">
            <label class="btn" for="q-file" style="cursor:pointer;flex:none">Choose PDF…</label>
            <span class="muted" id="q-file-name" style="min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">No file selected</span>
          </div>
          <input id="q-file" type="file" accept="application/pdf" hidden aria-label="Upload PDF for quiz" />
          <div class="hint">Upload a PDF and quiz yourself on its contents.</div>
        </div>`);
      srcCtl.append(wrap);
      const input = wrap.querySelector("#q-file");
      const nameSlot = wrap.querySelector("#q-file-name");
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
    goBtn.before(window.ui.modelSelect());

    // generate
    goBtn.addEventListener("click", async () => {
      const num = Number(numInput.value);
      const model = window.ui.selectedModel(panel);
      let request;
      if (source === "subject") {
        const subject = srcCtl.querySelector("#q-subject") && srcCtl.querySelector("#q-subject").value;
        const pdf = srcCtl.querySelector("#q-pdf") && srcCtl.querySelector("#q-pdf").value;
        if (!subject || !pdf) { toast("Pick a subject and a document first.", "error"); return; }
        lastMeta = { subject, topic: pdf };
        request = () => api.post("/quiz/generate", { subject, pdf, num_questions: num, difficulty, model });
      } else if (source === "topic") {
        const topicEl = srcCtl.querySelector("#q-topic");
        const topic = topicEl ? topicEl.value.trim() : "";
        if (!topic) { toast("Enter a topic first.", "error"); return; }
        lastMeta = { topic };
        request = () => api.post("/quiz/topic", { topic, num_questions: num, difficulty, model });
      } else {
        if (!pdfFile) { toast("Choose a PDF first.", "error"); return; }
        lastMeta = { topic: pdfFile.name };
        const fd = new FormData();
        fd.append("file", pdfFile);
        fd.append("num_questions", String(num));
        fd.append("difficulty", difficulty);
        if (model) fd.append("model", model);
        request = () => api.postForm("/quiz/upload", fd);
      }

      goBtn.disabled = true;
      const original = goBtn.textContent;
      goBtn.textContent = "Generating…";
      const load = spinner(source === "topic" ? "Writing your quiz…" : "Reading and writing your quiz…");
      panel.append(load);
      try {
        const res = await request();
        const questions = (res && res.questions) || [];
        if (!questions.length) throw new Error("No questions came back; try again.");
        renderRunner(questions, res && res.llm, false, res && res.id);
      } catch (e) {
        toast(e.message, "error");
        goBtn.disabled = false;
        goBtn.textContent = original;
        load.replaceWith(window.ui.errorBlock(e, e.retryable ? () => goBtn.click() : null));
      }
    });
  }

  /* ---------------- quiz runner ---------------- */
  function renderRunner(questions, llm, stored, quizId) {
    clear(root);
    GU().catch(() => {});                              // warm the shared results component while the student answers
    const attemptId = "qz" + Date.now().toString(36) + Math.random().toString(36).slice(2, 10);   // idempotency key for the submit
    const startedAt = Date.now();
    const total = questions.length;
    let index = 0;
    const answers = new Array(total).fill(null); // selected key per question

    const panel = el(`
      <div class="panel stack" style="max-width:680px;margin:0 auto">
        <div class="stack-sm">
          <div class="between">
            <span class="badge" id="qz-count"></span>
            <span class="badge badge-accent" id="qz-score">0 correct</span>
          </div>
          <div class="progress"><i id="qz-bar" style="width:0%"></i></div>
        </div>
        <div id="qz-body" class="stack"></div>
      </div>
    `);
    root.append(panel);
    llmLine = window.ui.renderModelLabel(llm, { stored: !!stored });   // mandatory "Model: …" line (stored = reopened from history)
    if (llmLine) panel.append(llmLine);

    const countEl = panel.querySelector("#qz-count");
    const scoreEl = panel.querySelector("#qz-score");
    const bar = panel.querySelector("#qz-bar");
    const body = panel.querySelector("#qz-body");

    const correctCount = () => answers.reduce((n, sel, i) => n + (sel === questions[i].correct ? 1 : 0), 0);

    function paintQuestion() {
      const q = questions[index];
      countEl.textContent = `Question ${index + 1} of ${total}`;
      scoreEl.textContent = `${correctCount()} correct`;
      bar.style.width = `${((index) / total) * 100}%`;

      clear(body);
      body.append(el(`<h3 style="font-family:var(--font-display);font-weight:700;line-height:1.3;letter-spacing:-.01em;overflow-wrap:anywhere">${esc(q.question)}</h3>`));

      const opts = el(`<div class="opts" role="group" aria-label="Answer options"></div>`);
      const buttons = {};
      KEYS.forEach((key) => {
        const text = (q.options && q.options[key]) || "";
        const b = el(`<button class="opt" type="button" data-key="${key}" aria-label="Option ${key}: ${esc(text)}"><span class="key">${key}</span><span style="min-width:0;overflow-wrap:anywhere">${esc(text)}</span></button>`);
        b.addEventListener("click", () => lockIn(key, buttons, q));
        buttons[key] = b;
        opts.append(b);
      });
      body.append(opts);
    }

    function lockIn(chosen, buttons, q) {
      if (answers[index] !== null) return; // already answered
      answers[index] = chosen;

      KEYS.forEach((key) => {
        const b = buttons[key];
        b.disabled = true;
        if (key === q.correct) b.dataset.state = "correct";
        else if (key === chosen) b.dataset.state = "wrong";
      });

      scoreEl.textContent = `${correctCount()} correct`;
      bar.style.width = `${((index + 1) / total) * 100}%`;

      const verdict = chosen === q.correct ? "Correct" : `Not quite — the answer is ${q.correct}`;
      body.append(el(
        `<div class="explain" role="status" style="overflow-wrap:anywhere"><strong>${esc(verdict)}.</strong> ${esc(q.explanation || "")}</div>`
      ));

      const nextLabel = index < total - 1 ? "Next question →" : "See results →";
      const next = el(`<button class="btn-accent btn-block" type="button">${nextLabel}</button>`);
      next.addEventListener("click", () => {
        if (index < total - 1) { index += 1; paintQuestion(); }
        else renderResults(questions, answers, { llm, quizId, attemptId, startedAt });
      });
      body.append(next);
      next.focus();
    }

    paintQuestion();
  }

  /* ---------------- results (marked + recorded by POST /api/grade; same component as the exam and chat widgets) ---------------- */
  async function renderResults(questions, answers, run) {
    clear(root);
    const total = questions.length;
    const localCorrect = answers.reduce((n, sel, i) => n + (sel === questions[i].correct ? 1 : 0), 0);
    const panel = el(`
      <div class="panel stack" style="max-width:680px;margin:0 auto">
        <div id="rv-local" class="muted" role="status" aria-live="polite" hidden></div>
        <div id="rv-result"></div>
      </div>
    `);
    root.append(panel);
    const box = panel.querySelector("#rv-result");
    const local = panel.querySelector("#rv-local");
    const meta = lastMeta || {};
    const ans = {};
    answers.forEach((sel, i) => { if (sel) ans["q" + (i + 1)] = sel; });
    let GradeUI;
    try { GradeUI = await GU(); } catch (e) { box.append(window.ui.errorBlock(e, () => renderResults(questions, answers, run))); return; }
    GradeUI.gradeInto(box, {
      kind: "quiz", subject: meta.subject || undefined, topic: meta.topic || undefined,
      title: "Quiz \u00b7 " + (meta.topic || meta.subject || "practice"),
      items: questions, answers: ans, attempt_id: run.attemptId, quiz_id: run.quizId || undefined,
      duration_seconds: Math.round((Date.now() - run.startedAt) / 1000), generated_llm: run.llm || undefined,
    }, {
      newLabel: "New quiz", onNew: () => renderSetup(), loading: "Marking and saving your quiz\u2026",
      onError: () => { local.hidden = false; local.textContent = `You answered ${localCorrect} of ${total} correctly - this result is not saved yet. Use Retry to save it.`; },
      onResult: () => { local.hidden = true; window.api.invalidate && window.api.invalidate("/progress"); },
    });
  }

  /* ---------------- view registration ---------------- */
  window.Views.quiz = {
    accent: "quiz",
    title: "Quiz Generator",
    async mount(node) {
      root = node;
      await renderSetup();
      // Chat-artifact deep-link: open the saved quiz instead of leaving Setup showing.
      const _art = window.ui.takeArtifact(["quizzes"]);
      if (_art) window.api.get(`/history/${_art.kind}/${encodeURIComponent(_art.id)}`).then(openSavedQuiz).catch(() => {});
    },
    unmount() {
      if (root) clear(root);
      root = null;
    },
  };
})();
