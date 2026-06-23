/* ============================================================
   Inline artifact renderer — window.renderArtifactInline
   The Apex Agent chat creates artifacts (quiz, flashcards, cheat
   sheet, summary, plan, mind map, prediction) from the CURRENT
   session subject and returns them with a `data` payload. This
   module turns that payload into a LIVE, interactive widget right
   inside the chat thread — playable quiz, flippable deck, readable
   doc, collapsible map, prediction list — while the artifact stays
   saved and openable full-screen.

   Subject-locked by construction: it renders only artifact.data,
   which the backend already produced from the session subject. It
   never fetches, never crosses subjects, never mutates the artifact.

   Exposes exactly one global:
     window.renderArtifactInline(container, artifact) -> boolean
   Returns true when it built a widget, false when the data is
   missing/mismatched so the caller can fall back to a link card.
   Uses only window.ui (el, esc, md) and plain DOM. CSS owns styling
   via the cl-* class names below.
   ============================================================ */
(() => {
  "use strict";

  const ui = window.ui || {};
  const el = ui.el;
  const esc = ui.esc;
  const md = ui.md;

  const KEYS = ["A", "B", "C", "D"];

  // ---- small helpers (no external deps) ----
  function isObj(v) { return v && typeof v === "object"; }
  function nonEmptyArr(v) { return Array.isArray(v) && v.length > 0; }
  function str(v) { return typeof v === "string" ? v : ""; }

  /* ---------------- quizzes: playable MCQ runner ---------------- */
  function renderQuiz(container, data) {
    const questions = (data && data.questions) || [];
    if (!nonEmptyArr(questions)) return false;

    const total = questions.length;
    let index = 0;
    let score = 0;
    const answered = new Array(total).fill(false);

    const wrap = el(`<div class="cl-quiz"></div>`);
    const head = el(`<div class="cl-qhead"></div>`);
    const counter = el(`<span></span>`);
    const score_b = el(`<span class="cl-score"></span>`);
    head.append(counter, score_b);
    const body = el(`<div></div>`);
    wrap.append(head, body);
    container.append(wrap);

    function updateHead() {
      counter.textContent = `Question ${index + 1} of ${total}`;
      score_b.textContent = `${score} / ${total}`;
    }

    function paint() {
      while (body.firstChild) body.removeChild(body.firstChild);
      const q = questions[index] || {};
      updateHead();

      body.append(el(`<div class="cl-prompt">${esc(str(q.question))}</div>`));

      const opts = el(`<div class="cl-opts"></div>`);
      const options = isObj(q.options) ? q.options : {};
      const correct = str(q.correct);
      const buttons = {};
      KEYS.forEach((key) => {
        if (!(key in options)) return;
        const b = el(`<button class="cl-opt" type="button"><b>${esc(key)}</b> <span>${esc(str(options[key]))}</span></button>`);
        b.addEventListener("click", () => choose(key, q, correct, buttons, opts));
        buttons[key] = b;
        opts.append(b);
      });
      body.append(opts);
    }

    function choose(chosen, q, correct, buttons, opts) {
      if (answered[index]) return;
      answered[index] = true;
      if (chosen === correct) score += 1;

      Object.keys(buttons).forEach((key) => {
        const b = buttons[key];
        b.disabled = true;
        if (key === correct) b.classList.add("is-correct");
        else if (key === chosen) b.classList.add("is-wrong");
      });
      updateHead();

      const expl = str(q.explanation);
      if (expl) {
        const verdict = chosen === correct ? "Correct. " : `Answer: ${esc(correct)}. `;
        opts.after(el(`<div class="cl-expl"><b>${verdict}</b>${esc(expl)}</div>`));
      }

      const last = index >= total - 1;
      const next = el(`<button class="cl-next" type="button">${last ? "See score" : "Next →"}</button>`);
      next.addEventListener("click", () => {
        if (last) finish();
        else { index += 1; paint(); }
      });
      body.append(next);
    }

    function finish() {
      while (body.firstChild) body.removeChild(body.firstChild);
      const pct = total ? Math.round((score / total) * 100) : 0;
      counter.textContent = "Finished";
      score_b.textContent = `${score} / ${total}`;
      body.append(el(`<div class="cl-score">You scored ${score} of ${total} (${pct}%).</div>`));
      const again = el(`<button class="cl-next" type="button">Try again</button>`);
      again.addEventListener("click", () => { index = 0; score = 0; answered.fill(false); paint(); });
      body.append(again);
    }

    paint();
    return true;
  }

  /* ---------------- mock exams: take → submit → graded report ---------------- */
  function renderExam(container, data) {
    const raw = (data && data.questions) || [];
    if (!nonEmptyArr(raw)) return false;

    // Assign stable ids (q1, q2, …) in order — matches the grader's fallback so
    // answers and questions line up. We send these ids in the grade request.
    const questions = raw.map((q, i) => {
      const item = isObj(q) ? q : {};
      const id = (typeof item.id === "string" && item.id.trim()) ? item.id.trim() : `q${i + 1}`;
      return Object.assign({}, item, { id: id });
    });
    const subject = str(data.subject);

    const answers = {};          // id -> chosen index (mcq) | text (short)
    const wrap = el(`<div class="cl-exam"></div>`);

    // ── custom countdown timer (optional) — set minutes, Start, auto-submits at 0 ──
    let timerId = null, startSecs = 0, remainSecs = 0;
    const fmtClock = (s) => Math.floor(s / 60) + ":" + String(s % 60).padStart(2, "0");
    const stopTimer = () => { if (timerId) { clearInterval(timerId); timerId = null; } };
    const defMin = Math.max(5, Math.min(120, questions.length || 10));
    const timerBar = el(`<div class="cl-exam-timer"></div>`);
    const tMin = el(`<input class="cl-exam-timer-min" type="number" min="1" max="180" value="${defMin}" aria-label="Timer minutes" />`);
    const tBtn = el(`<button class="cl-exam-timer-btn" type="button">Start timer</button>`);
    const tClock = el(`<span class="cl-exam-timer-clock"></span>`);
    timerBar.append(
      el(`<span class="cl-exam-timer-label">⏱ Timer</span>`), tMin,
      el(`<span class="cl-exam-timer-unit">min</span>`), tBtn, tClock,
    );
    function tickTimer() {
      if (!tClock.isConnected) { stopTimer(); return; }     // widget gone — stop ticking
      remainSecs -= 1;
      tClock.textContent = fmtClock(Math.max(0, remainSecs));
      if (remainSecs <= 0) {
        stopTimer();
        tClock.classList.add("is-up"); tClock.textContent = "Time's up";
        tMin.disabled = true; tBtn.disabled = true;
        if (!wrap.classList.contains("is-graded")) submitExam();   // auto-submit when time runs out
      }
    }
    tBtn.addEventListener("click", () => {
      if (wrap.classList.contains("is-graded")) return;
      if (timerId) {                                          // running → cancel/reset
        stopTimer(); tBtn.textContent = "Start timer"; tMin.disabled = false;
        tClock.textContent = ""; tClock.classList.remove("is-up"); startSecs = 0;
        return;
      }
      const mins = Math.max(1, Math.min(180, parseInt(tMin.value, 10) || 10));
      startSecs = remainSecs = mins * 60;
      tClock.textContent = fmtClock(remainSecs);
      tMin.disabled = true; tBtn.textContent = "Stop";
      timerId = setInterval(tickTimer, 1000);
    });
    wrap.append(timerBar);

    const body = el(`<div class="cl-exam-body"></div>`);
    wrap.append(body);
    container.append(wrap);

    questions.forEach((q, i) => {
      const type = str(q.type) || (nonEmptyArr(q.options) ? "mcq" : "short");
      const qEl = el(`<div class="cl-exam-q" data-qid="${esc(q.id)}"></div>`);
      const topic = str(q.topic);
      const head = el(`<div class="cl-exam-qhead"></div>`);
      head.append(el(`<span class="cl-exam-num">Q${i + 1}</span>`));
      if (topic) head.append(el(`<span class="cl-exam-topic">${esc(topic)}</span>`));
      qEl.append(head);
      qEl.append(el(`<div class="cl-exam-prompt">${esc(str(q.question))}</div>`));

      if (type === "mcq") {
        const options = Array.isArray(q.options) ? q.options : [];
        const opts = el(`<div class="cl-exam-opts"></div>`);
        options.forEach((opt, oi) => {
          const b = el(`<button class="cl-exam-opt" type="button"><b>${esc(KEYS[oi] || String(oi + 1))}</b> <span>${esc(str(opt))}</span></button>`);
          b.addEventListener("click", () => {
            if (wrap.classList.contains("is-graded")) return;
            answers[q.id] = oi;
            opts.querySelectorAll(".cl-exam-opt").forEach((o) => o.classList.remove("is-chosen"));
            b.classList.add("is-chosen");
          });
          opts.append(b);
        });
        qEl.append(opts);
      } else {
        const ta = el(`<textarea class="cl-exam-short" rows="3" placeholder="Your answer…"></textarea>`);
        ta.addEventListener("input", () => { answers[q.id] = ta.value; });
        qEl.append(ta);
      }
      body.append(qEl);
    });

    const submit = el(`<button class="cl-exam-submit" type="button">Submit exam</button>`);
    submit.addEventListener("click", submitExam);
    wrap.append(submit);

    async function submitExam() {
      if (wrap.classList.contains("is-graded")) return;
      stopTimer();
      const elapsed = startSecs ? (startSecs - Math.max(0, remainSecs)) : null;   // seconds taken, if timer used
      submit.disabled = true;
      submit.textContent = "Grading…";
      try {
        const report = await window.api.post("/exam/grade", {
          subject: subject,
          questions: questions,
          answers: answers,
          duration_seconds: elapsed,
          model: null,
        });
        renderReport(report);
      } catch (err) {
        submit.disabled = false;
        submit.textContent = "Submit exam";
        body.after(el(`<div class="cl-exam-err">Could not grade the exam — ${esc((err && err.message) || "please try again")}.</div>`));
      }
    }

    function renderReport(report) {
      if (!isObj(report)) return;
      wrap.classList.add("is-graded");
      submit.remove();

      // Mark each question's options correct/wrong; disable inputs.
      const byId = {};
      (Array.isArray(report.per_question) ? report.per_question : []).forEach((r) => {
        if (isObj(r) && r.id != null) byId[String(r.id)] = r;
      });
      questions.forEach((q) => {
        const qEl = body.querySelector(`.cl-exam-q[data-qid="${cssId(q.id)}"]`);
        if (!qEl) return;
        const r = byId[String(q.id)] || {};
        const optBtns = qEl.querySelectorAll(".cl-exam-opt");
        optBtns.forEach((b, oi) => {
          b.disabled = true;
          if (typeof r.answer_index === "number" && oi === r.answer_index) b.classList.add("is-correct");
          else if (typeof r.user_answer === "number" && oi === r.user_answer) b.classList.add("is-wrong");
        });
        const ta = qEl.querySelector(".cl-exam-short");
        if (ta) ta.disabled = true;
        const fb = str(r.feedback);
        if (fb) {
          const d = el(`<div class="cl-exam-fb"></div>`);
          d.append(md(fb));
          qEl.append(d);
        }
      });

      const total = Number(report.total_score) || 0;
      const max = Number(report.max_score) || 0;
      const pct = max ? Math.round((total / max) * 100) : 0;
      const rep = el(`<div class="cl-exam-report"></div>`);
      rep.append(el(`<div class="cl-exam-score">You scored ${total} of ${max} (${pct}%).</div>`));

      const breakdown = Array.isArray(report.topic_breakdown) ? report.topic_breakdown : [];
      if (breakdown.length) {
        const tb = el(`<div class="cl-exam-topics"></div>`);
        tb.append(el(`<strong>By topic</strong>`));
        breakdown.forEach((t) => {
          if (!isObj(t)) return;
          const tn = str(t.topic) || "General";
          const e = Number(t.earned) || 0;
          const p = Number(t.possible) || 0;
          tb.append(el(`<div class="cl-exam-topic-row"><span>${esc(tn)}</span><span>${e} / ${p}</span></div>`));
        });
        rep.append(tb);
      }
      wrap.append(rep);
    }

    return true;
  }

  // Escape an id for use inside a CSS attribute selector value.
  function cssId(id) { return String(id).replace(/["\\]/g, "\\$&"); }

  /* ---------------- flashcards: flip deck ---------------- */
  function renderFlashcards(container, data) {
    const cards = (data && data.cards) || [];
    if (!nonEmptyArr(cards)) return false;

    const total = cards.length;
    let index = 0;
    let flipped = false;

    const wrap = el(`<div class="cl-deck"></div>`);
    const card = el(`<button class="cl-card" type="button" aria-label="Flashcard, click to flip"></button>`);
    const nav = el(`<div class="cl-nav"></div>`);
    const prev = el(`<button class="cl-navbtn" type="button">← Prev</button>`);
    const pos = el(`<span></span>`);
    const next = el(`<button class="cl-navbtn" type="button">Next →</button>`);
    nav.append(prev, pos, next);
    wrap.append(card, nav);
    container.append(wrap);

    function paint() {
      while (card.firstChild) card.removeChild(card.firstChild);
      const c = cards[index] || {};
      const topic = str(c.topic);
      const text = flipped ? str(c.back) : str(c.front);
      const face = el(`<div class="cl-face"></div>`);
      if (topic) face.append(el(`<small>${esc(topic)}</small>`));
      face.append(el(`<div>${esc(text)}</div>`));
      face.append(el(`<span class="cl-flip">${flipped ? "back · click to flip" : "front · click to flip"}</span>`));
      card.append(face);
      card.classList.toggle("cl-flip", flipped);
      pos.textContent = `${index + 1} / ${total}`;
      prev.disabled = index === 0;
      next.disabled = index >= total - 1;
    }

    card.addEventListener("click", () => { flipped = !flipped; paint(); });
    prev.addEventListener("click", () => { if (index > 0) { index -= 1; flipped = false; paint(); } });
    next.addEventListener("click", () => { if (index < total - 1) { index += 1; flipped = false; paint(); } });

    paint();
    return true;
  }

  /* ---------------- cheat sheets + plans: markdown doc ---------------- */
  function renderDoc(container, markdown) {
    const text = str(markdown);
    if (!text.trim()) return false;
    const doc = el(`<div class="cl-doc"></div>`);
    doc.append(md(text));
    container.append(doc);
    return true;
  }

  /* ---------------- summaries: per-pdf sections ---------------- */
  function renderSummaries(container, data) {
    const summaries = data && data.summaries;
    if (!isObj(summaries)) return false;
    const names = Object.keys(summaries);
    if (!names.length) return false;

    const doc = el(`<div class="cl-doc"></div>`);
    let rendered = 0;
    names.forEach((name) => {
      const text = str(summaries[name]);
      if (!text.trim()) return;
      doc.append(el(`<div class="cl-doc-h">${esc(name)}</div>`));
      doc.append(md(text));
      rendered += 1;
    });
    if (!rendered) return false;
    container.append(doc);
    return true;
  }

  /* ---------------- mindmaps: collapsible tree ---------------- */
  function renderMindmap(container, data) {
    const root = data && data.root;
    if (!isObj(root) || typeof root.label !== "string") return false;

    const tree = el(`<div class="cl-mm"></div>`);
    tree.append(buildMmNode(root, 0));
    container.append(tree);
    return true;
  }

  function buildMmNode(node, depth) {
    const children = Array.isArray(node.children) ? node.children : [];
    const hasKids = children.length > 0;
    const row = el(`<div class="cl-mm-row" style="--mm-depth:${depth}"></div>`);

    const line = el(`<div class="cl-mm-label-line"></div>`);
    let kidsWrap = null;
    if (hasKids) {
      const toggle = el(`<button class="cl-mm-toggle" type="button" aria-expanded="${depth === 0}">${depth === 0 ? "−" : "+"}</button>`);
      const label = el(`<span class="cl-mm-label">${esc(str(node.label))}</span>`);
      line.append(toggle, label);
      kidsWrap = el(`<div class="cl-mm-kids"></div>`);
      if (depth !== 0) kidsWrap.hidden = true;
      children.forEach((child) => { if (isObj(child)) kidsWrap.append(buildMmNode(child, depth + 1)); });
      toggle.addEventListener("click", () => {
        const open = kidsWrap.hidden;
        kidsWrap.hidden = !open;
        toggle.textContent = open ? "−" : "+";
        toggle.setAttribute("aria-expanded", String(open));
      });
    } else {
      line.append(el(`<span class="cl-mm-toggle is-leaf" aria-hidden="true">·</span>`));
      line.append(el(`<span class="cl-mm-label">${esc(str(node.label))}</span>`));
    }
    row.append(line);
    if (kidsWrap) row.append(kidsWrap);
    return row;
  }

  /* ---------------- predictions: question list + patterns ---------------- */
  function renderPredictions(container, data) {
    const predicted = (data && data.predicted_questions) || [];
    const patterns = (data && data.patterns) || [];
    if (!nonEmptyArr(predicted) && !nonEmptyArr(patterns)) return false;

    const wrap = el(`<div class="cl-pred"></div>`);

    if (nonEmptyArr(predicted)) {
      predicted.forEach((item) => {
        const itemEl = el(`<div class="cl-pred-item"></div>`);
        // Items may be plain strings or {question/text, reason/why}.
        let qText = "";
        let why = "";
        if (typeof item === "string") {
          qText = item;
        } else if (isObj(item)) {
          qText = str(item.question || item.text || item.q || "");
          why = str(item.reason || item.why || item.rationale || "");
        }
        if (!qText) return;
        itemEl.append(el(`<div class="cl-pred-q">${esc(qText)}</div>`));
        if (why) itemEl.append(el(`<div class="cl-pred-why">${esc(why)}</div>`));
        wrap.append(itemEl);
      });
    }

    if (nonEmptyArr(patterns)) {
      const pat = el(`<div class="cl-pred-patterns"></div>`);
      pat.append(el(`<strong>Patterns</strong>`));
      patterns.forEach((p) => {
        const text = typeof p === "string" ? p : (isObj(p) ? str(p.pattern || p.text || p.label || "") : "");
        if (text) pat.append(el(`<div>${esc(text)}</div>`));
      });
      if (pat.childElementCount > 1) wrap.append(pat);
    }

    if (!wrap.childElementCount) return false;
    container.append(wrap);
    return true;
  }

  /* ---------------- teach-back: live Socratic mini-chat ---------------- */
  function renderTeachback(container, data) {
    const sessionId = str(data && data.session_id);
    const opening = str(data && data.opening);
    if (!sessionId || !opening) return false;

    const concept = str(data.concept);
    let done = false;
    let busy = false;

    const wrap = el(`<div class="cl-tb"></div>`);
    if (concept) wrap.append(el(`<div class="cl-tb-concept">Explain: <strong>${esc(concept)}</strong></div>`));
    const thread = el(`<div class="cl-tb-thread"></div>`);
    wrap.append(thread);
    container.append(wrap);

    function bubble(role, text) {
      const b = el(`<div class="cl-tb-msg cl-tb-${role}"></div>`);
      b.append(el(`<div>${esc(text)}</div>`));
      thread.append(b);
      b.scrollIntoView({ block: "nearest" });
      return b;
    }

    bubble("tutor", opening);

    const form = el(`<div class="cl-tb-form"></div>`);
    const input = el(`<textarea class="cl-tb-input" rows="2" placeholder="Explain it in your own words…"></textarea>`);
    const send = el(`<button class="cl-tb-send" type="button">Send</button>`);
    form.append(input, send);
    wrap.append(form);

    async function submit() {
      if (done || busy) return;
      const message = input.value.trim();
      if (!message) return;
      busy = true;
      send.disabled = true;
      input.disabled = true;
      bubble("student", message);
      input.value = "";
      const pending = bubble("tutor", "…");
      pending.classList.add("is-pending");
      try {
        const res = await window.api.post("/teachback/message", {
          session_id: sessionId,
          message: message,
        });
        pending.remove();
        bubble("tutor", str(res && res.reply) || "…");
        if (res && res.done) {
          done = true;
          form.remove();
          renderEval(res.evaluation);
        } else {
          busy = false;
          send.disabled = false;
          input.disabled = false;
          input.focus();
        }
      } catch (err) {
        pending.remove();
        busy = false;
        send.disabled = false;
        input.disabled = false;
        thread.append(el(`<div class="cl-tb-err">Couldn't reach the tutor — ${esc((err && err.message) || "try again")}.</div>`));
      }
    }

    send.addEventListener("click", submit);
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) { e.preventDefault(); submit(); }
    });

    function renderEval(evaluation) {
      if (!isObj(evaluation)) return;
      const score = Number(evaluation.understanding_score_0_to_10);
      const ev = el(`<div class="cl-tb-eval"></div>`);
      if (Number.isFinite(score)) {
        ev.append(el(`<div class="cl-tb-score">Understanding: ${score} / 10</div>`));
      }
      const block = (title, items, cls) => {
        const arr = Array.isArray(items) ? items.filter((s) => str(s).trim()) : [];
        if (!arr.length) return;
        const sec = el(`<div class="cl-tb-eval-sec ${cls}"></div>`);
        sec.append(el(`<strong>${esc(title)}</strong>`));
        const list = el(`<ul></ul>`);
        arr.forEach((s) => list.append(el(`<li>${esc(str(s))}</li>`)));
        sec.append(list);
        ev.append(sec);
      };
      block("Strengths", evaluation.strengths, "is-strength");
      block("Gaps", evaluation.gaps, "is-gap");
      block("Review next", evaluation.suggested_review, "is-review");
      wrap.append(ev);
    }

    return true;
  }

  /* ---------------- progress: read-only insight tiles ---------------- */
  function renderProgress(container, data) {
    if (!isObj(data)) return false;

    const overall = isObj(data.overall) ? data.overall : null;
    const subjects = Array.isArray(data.subjects) ? data.subjects : [];
    const weakest = Array.isArray(data.weakest) ? data.weakest : [];
    const activity = Array.isArray(data.activity) ? data.activity : [];
    // Insights-style fields (when the agent returns the richer insights payload).
    const headline = str(data.headline);
    const observations = Array.isArray(data.observations) ? data.observations : [];
    const recommendations = Array.isArray(data.recommendations) ? data.recommendations : [];
    const focusTopics = Array.isArray(data.focus_topics) ? data.focus_topics : [];

    const totalAttempts = overall ? Number(overall.total_attempts) || 0 : 0;
    const hasAnything = totalAttempts > 0 || subjects.length || weakest.length ||
      activity.length || headline || observations.length || recommendations.length;
    if (!hasAnything) {
      container.append(el(`<div class="cl-prog cl-prog-empty">No activity yet — take a quiz, run a teach-back, or sit a mock exam and your progress will show up here.</div>`));
      return true;
    }

    const wrap = el(`<div class="cl-prog"></div>`);

    if (headline) wrap.append(el(`<div class="cl-prog-headline">${esc(headline)}</div>`));

    if (overall) {
      const tiles = el(`<div class="cl-prog-tiles"></div>`);
      const tile = (label, value) => {
        const t = el(`<div class="cl-prog-tile"></div>`);
        t.append(el(`<div class="cl-prog-tile-v">${esc(String(value))}</div>`));
        t.append(el(`<div class="cl-prog-tile-l">${esc(label)}</div>`));
        tiles.append(t);
      };
      tile("Attempts", totalAttempts);
      tile("Accuracy", `${Math.round((Number(overall.overall_accuracy) || 0) * 100)}%`);
      tile("Day streak", Number(overall.streak_days) || 0);
      wrap.append(tiles);
    }

    if (observations.length) {
      const obs = el(`<div class="cl-prog-obs"></div>`);
      observations.forEach((o) => { if (str(o).trim()) obs.append(el(`<div>${esc(str(o))}</div>`)); });
      if (obs.childElementCount) wrap.append(obs);
    }

    if (activity.length) {
      const act = el(`<div class="cl-prog-activity"></div>`);
      act.append(el(`<strong>Recent activity</strong>`));
      activity.forEach((a) => {
        if (!isObj(a)) return;
        const label = str(a.label) || str(a.tool);
        if (!label) return;
        const count = Number(a.count) || 0;
        const metric = str(a.metric);
        const row = el(`<div class="cl-prog-act-row"></div>`);
        row.append(el(`<span>${esc(label)}</span>`));
        row.append(el(`<span>${esc(String(count))}${metric ? " · " + esc(metric) : ""}</span>`));
        act.append(row);
      });
      if (act.childElementCount > 1) wrap.append(act);
    }

    if (weakest.length) {
      const weak = el(`<div class="cl-prog-weak"></div>`);
      weak.append(el(`<strong>Weak spots</strong>`));
      weakest.forEach((w) => {
        if (!isObj(w)) return;
        const topic = str(w.topic);
        if (!topic) return;
        const subj = str(w.subject);
        const acc = Math.round((Number(w.accuracy) || 0) * 100);
        const row = el(`<div class="cl-prog-weak-row"></div>`);
        row.append(el(`<span>${esc(topic)}${subj ? " · " + esc(subj) : ""}</span>`));
        row.append(el(`<span>${acc}%</span>`));
        weak.append(row);
      });
      if (weak.childElementCount > 1) wrap.append(weak);
    } else if (focusTopics.length) {
      const focus = el(`<div class="cl-prog-weak"></div>`);
      focus.append(el(`<strong>Focus topics</strong>`));
      focusTopics.forEach((t) => { if (str(t).trim()) focus.append(el(`<div class="cl-prog-weak-row"><span>${esc(str(t))}</span></div>`)); });
      if (focus.childElementCount > 1) wrap.append(focus);
    }

    if (recommendations.length) {
      const recs = el(`<div class="cl-prog-recs"></div>`);
      recs.append(el(`<strong>Recommendations</strong>`));
      recommendations.forEach((r) => {
        let action = "";
        let reason = "";
        if (typeof r === "string") action = r;
        else if (isObj(r)) { action = str(r.action || r.text || ""); reason = str(r.reason || r.why || ""); }
        if (!action) return;
        const rec = el(`<div class="cl-prog-rec"></div>`);
        rec.append(el(`<div class="cl-prog-rec-a">${esc(action)}</div>`));
        if (reason) rec.append(el(`<div class="cl-prog-rec-r">${esc(reason)}</div>`));
        recs.append(rec);
      });
      if (recs.childElementCount > 1) wrap.append(recs);
    }

    if (!wrap.childElementCount) {
      container.append(el(`<div class="cl-prog cl-prog-empty">No activity yet — take a quiz, run a teach-back, or sit a mock exam and your progress will show up here.</div>`));
      return true;
    }
    container.append(wrap);
    return true;
  }

  /* ---------------- dispatch ---------------- */
  function renderArtifactInline(container, artifact) {
    if (!container || !isObj(artifact)) return false;
    if (typeof el !== "function" || typeof esc !== "function" || typeof md !== "function") return false;
    const data = artifact.data;
    if (!isObj(data)) return false;

    try {
      switch (artifact.kind) {
        case "quizzes":     return renderQuiz(container, data);
        case "exams":       return renderExam(container, data);
        case "flashcards":  return renderFlashcards(container, data);
        case "cheatsheets": return renderDoc(container, data.markdown);
        case "plans":       return renderDoc(container, data.plan);
        case "summaries":   return renderSummaries(container, data);
        case "mindmaps":    return renderMindmap(container, data);
        case "predictions": return renderPredictions(container, data);
        case "teachback":   return renderTeachback(container, data);
        case "teachbacks":  return renderTeachback(container, data);
        case "progress":    return renderProgress(container, data);
        case "scans":       return renderDoc(container, data.markdown);
        default:            return false;
      }
    } catch (_) {
      return false;
    }
  }

  window.renderArtifactInline = renderArtifactInline;
})();
