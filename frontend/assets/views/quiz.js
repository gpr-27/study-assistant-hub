/* ============================================================
   Quiz Generator view — window.Views.quiz
   Source (Subject / Topic / Upload PDF) -> Groq quiz ->
   one-question-at-a-time runner -> score. Mirrors the Chat tool's
   three sources. Uses window.api, window.ui and the CSS kit.
   ============================================================ */
(() => {
  "use strict";
  const { el, clear, esc, spinner, toast } = window.ui;
  const KEYS = ["A", "B", "C", "D"];
  const DIFFICULTIES = ["easy", "medium", "hard"];

  let root = null;

  /* ---------------- setup screen ---------------- */
  async function renderSetup() {
    clear(root);
    let source = "subject";   // "subject" | "topic" | "upload"
    let difficulty = "medium";
    let pdfFile = null;

    const panel = el(`
      <div class="panel stack" style="max-width:640px;margin:0 auto">
        <div class="between" style="align-items:flex-start">
          <div class="stack-sm" style="flex:1">
            <h3 style="font-family:var(--font-display);font-weight:700;letter-spacing:-.02em">Build a quiz</h3>
            <p class="muted">Pick a source, choose how hard and how many — Groq writes the questions.</p>
          </div>
          <button id="q-hist" class="btn" type="button" style="flex:none;padding:.4rem .8rem;font-size:.85rem;gap:.35rem">≡ History</button>
        </div>

        <div class="field">
          <span class="label">Source</span>
          <div class="seg" id="q-src" role="tablist" aria-label="Quiz source">
            <button type="button" data-src="subject" aria-pressed="true">Subject</button>
            <button type="button" data-src="topic" aria-pressed="false">Topic</button>
            <button type="button" data-src="upload" aria-pressed="false">Upload PDF</button>
          </div>
        </div>

        <div id="q-srcctl"></div>

        <div class="field">
          <span class="label">Difficulty</span>
          <div class="seg" id="q-diff" role="group" aria-label="Difficulty"></div>
        </div>

        <div class="field">
          <div class="between">
            <label class="label" for="q-num" style="margin:0">Number of questions</label>
            <span class="badge badge-accent count" id="q-num-out">5</span>
          </div>
          <input id="q-num" type="range" min="5" max="15" step="5" value="5" aria-label="Number of questions">
        </div>

        <button id="q-go" class="btn-accent btn-lg btn-block">Generate quiz</button>
      </div>
    `);
    root.append(panel);

    panel.querySelector("#q-hist").addEventListener("click", (e) =>
      window.ui.openHistoryDrawer("quizzes", {
        title: "Saved quizzes",
        anchor: e.currentTarget,
        onOpen: (saved) => renderRunner((saved && saved.questions) || []),
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

    numInput.addEventListener("input", () => { numOut.textContent = numInput.value; });

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
          <label class="label">PDF</label>
          <div class="row row-wrap">
            <label class="btn" for="q-file" style="cursor:pointer">Choose PDF…</label>
            <span class="muted" id="q-file-name">No file selected</span>
          </div>
          <input id="q-file" type="file" accept="application/pdf" hidden />
          <div class="hint">Upload a PDF and quiz yourself on its contents.</div>
        </div>`);
      srcCtl.append(wrap);
      const input = wrap.querySelector("#q-file");
      const nameSlot = wrap.querySelector("#q-file-name");
      input.addEventListener("change", () => {
        pdfFile = input.files && input.files[0] ? input.files[0] : null;
        nameSlot.textContent = pdfFile ? pdfFile.name : "No file selected";
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
        request = () => api.post("/quiz/generate", { subject, pdf, num_questions: num, difficulty, model });
      } else if (source === "topic") {
        const topicEl = srcCtl.querySelector("#q-topic");
        const topic = topicEl ? topicEl.value.trim() : "";
        if (!topic) { toast("Enter a topic first.", "error"); return; }
        request = () => api.post("/quiz/topic", { topic, num_questions: num, difficulty, model });
      } else {
        if (!pdfFile) { toast("Choose a PDF first.", "error"); return; }
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
        renderRunner(questions);
      } catch (e) {
        toast(e.message, "error");
        goBtn.disabled = false;
        goBtn.textContent = original;
        load.remove();
      }
    });
  }

  /* ---------------- quiz runner ---------------- */
  function renderRunner(questions) {
    clear(root);
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
      body.append(el(`<h3 style="font-family:var(--font-display);font-weight:700;line-height:1.3;letter-spacing:-.01em">${esc(q.question)}</h3>`));

      const opts = el(`<div class="opts" role="group" aria-label="Answer options"></div>`);
      const buttons = {};
      KEYS.forEach((key) => {
        const text = (q.options && q.options[key]) || "";
        const b = el(`<button class="opt" type="button" data-key="${key}"><span class="key">${key}</span><span>${esc(text)}</span></button>`);
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
        `<div class="explain"><strong>${esc(verdict)}.</strong> ${esc(q.explanation)}</div>`
      ));

      const nextLabel = index < total - 1 ? "Next question →" : "See results →";
      const next = el(`<button class="btn-accent btn-block" type="button">${nextLabel}</button>`);
      next.addEventListener("click", () => {
        if (index < total - 1) { index += 1; paintQuestion(); }
        else renderResults(questions, answers);
      });
      body.append(next);
      next.focus();
    }

    paintQuestion();
  }

  /* ---------------- results ---------------- */
  function renderResults(questions, answers) {
    clear(root);
    const total = questions.length;
    const correct = answers.reduce((n, sel, i) => n + (sel === questions[i].correct ? 1 : 0), 0);
    const pct = total ? Math.round((correct / total) * 100) : 0;

    let line;
    if (pct === 100) line = "Perfect score — outstanding work.";
    else if (pct >= 80) line = "Great job! A solid understanding.";
    else if (pct >= 60) line = "Good effort — keep practising.";
    else line = "Keep studying — you’ll get there.";

    const panel = el(`
      <div class="panel stack" style="max-width:680px;margin:0 auto">
        <div class="center stack-sm">
          <div class="score-ring">${pct}%</div>
          <p style="font-family:var(--font-display);font-weight:600;font-size:1.1rem">${esc(line)}</p>
          <p class="muted">You answered ${correct} of ${total} correctly.</p>
        </div>
        <div class="stack-sm" id="rv-list"></div>
        <button id="rv-new" class="btn-accent btn-lg btn-block" type="button">New quiz</button>
      </div>
    `);
    root.append(panel);

    const list = panel.querySelector("#rv-list");
    questions.forEach((q, i) => {
      const sel = answers[i];
      const ok = sel === q.correct;
      const review = el(`
        <details class="panel-soft">
          <summary style="cursor:pointer;font-weight:600">
            <span class="badge ${ok ? "badge-accent" : ""}">${ok ? "✓" : "✗"}</span>
            Question ${i + 1}
          </summary>
          <div class="stack-sm" style="margin-top:.8rem">
            <p><strong>${esc(q.question)}</strong></p>
            <div class="opts" style="margin-top:.4rem"></div>
            <div class="explain">${esc(q.explanation)}</div>
          </div>
        </details>
      `);
      const opts = review.querySelector(".opts");
      KEYS.forEach((key) => {
        const text = (q.options && q.options[key]) || "";
        const o = el(`<div class="opt" aria-disabled="true"><span class="key">${key}</span><span>${esc(text)}</span></div>`);
        if (key === q.correct) o.dataset.state = "correct";
        else if (key === sel) o.dataset.state = "wrong";
        opts.append(o);
      });
      list.append(review);
    });

    panel.querySelector("#rv-new").addEventListener("click", () => renderSetup());
  }

  /* ---------------- view registration ---------------- */
  window.Views.quiz = {
    accent: "quiz",
    title: "Quiz Generator",
    async mount(node) {
      root = node;
      await renderSetup();
    },
    unmount() {
      if (root) clear(root);
      root = null;
    },
  };
})();
