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

  /* ---------------- dispatch ---------------- */
  function renderArtifactInline(container, artifact) {
    if (!container || !isObj(artifact)) return false;
    if (typeof el !== "function" || typeof esc !== "function" || typeof md !== "function") return false;
    const data = artifact.data;
    if (!isObj(data)) return false;

    try {
      switch (artifact.kind) {
        case "quizzes":     return renderQuiz(container, data);
        case "flashcards":  return renderFlashcards(container, data);
        case "cheatsheets": return renderDoc(container, data.markdown);
        case "plans":       return renderDoc(container, data.plan);
        case "summaries":   return renderSummaries(container, data);
        case "mindmaps":    return renderMindmap(container, data);
        case "predictions": return renderPredictions(container, data);
        default:            return false;
      }
    } catch (_) {
      return false;
    }
  }

  window.renderArtifactInline = renderArtifactInline;
})();
