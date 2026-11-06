/* ============================================================
   Shared grading UI — window.GradeUI
   ONE results component + ONE practice runner used by the Quiz and Mock Exam views, the Apex Agent's
   inline widgets, teach-back and flashcard reviews, so every surface marks (POST /api/grade) and
   shows results the same way.

     GradeUI.newId()                                   -> client attempt id (idempotency key)
     GradeUI.gradeInto(container, payload, opts)       -> POST /grade, then render the result (loading / error+Retry)
     GradeUI.renderResult(container, result, opts)     -> the result component
     GradeUI.runPaper(container, rawItems, meta, opts) -> a one-page practice paper that grades itself
     GradeUI.toMarkdown(result)                        -> string (export)

   opts: { payload, onNew, newLabel, onStudyWeak(topics), items (raw signed items for "retry wrong"), compact }
   Scoped classes: gr-*. Design tokens only; no globals restyled. Results are announced via aria-live.
   ============================================================ */
(() => {
  "use strict";
  const ui = window.ui || {};
  const { el, clear, esc, md, spinner } = ui;
  const api = () => window.api;

  /* ---------------- styles (scoped, token based) ---------------- */
  if (!document.getElementById("gr-styles")) {
    const st = document.createElement("style");
    st.id = "gr-styles";
    st.textContent = `
.gr-result{display:grid;gap:1rem;min-width:0}
.gr-head{display:flex;align-items:center;gap:1rem;flex-wrap:wrap}
.gr-head .score-ring{font-size:3rem;line-height:1}
.gr-band{font-family:var(--font-display);font-weight:700;font-size:1.15rem}
.gr-sub{color:hsl(var(--ink-soft));font-size:.9rem}
.gr-notice{border-left:3px solid hsl(38 92% 55%);padding:.7rem .9rem;border-radius:var(--r-sm);background:hsl(38 92% 55% / .12);font-size:.9rem;overflow-wrap:anywhere}
.gr-actions{display:flex;gap:.5rem;flex-wrap:wrap}
.gr-actions button{padding:.45rem .85rem;font-size:var(--fs-sm)}
.gr-sec{display:grid;gap:.5rem}
.gr-sec h4{font-family:var(--font-display);font-size:1rem;margin:0}
.gr-row{display:grid;gap:.25rem}
.gr-row-top{display:flex;justify-content:space-between;gap:.5rem;font-size:.9rem}
.gr-row-top span:first-child{overflow-wrap:anywhere;min-width:0}
.gr-chips{display:flex;gap:.35rem;flex-wrap:wrap}
.gr-item{padding:.2rem 0}
.gr-item>summary{cursor:pointer;display:flex;gap:.6rem;align-items:center;font-weight:600;min-width:0}
.gr-item>summary .gr-q{min-width:0;overflow-wrap:anywhere;flex:1}
.gr-body{display:grid;gap:.6rem;margin-top:.7rem;min-width:0;overflow-wrap:anywhere}
.gr-v{font-size:var(--fs-xs);font-weight:700;padding:.25rem .6rem;border-radius:var(--r-pill);border:1px solid;flex:none;white-space:nowrap}
.gr-v-correct{background:hsl(150 60% 45% / .18);border-color:hsl(150 60% 45% / .5)}
.gr-v-partial{background:hsl(38 92% 55% / .18);border-color:hsl(38 92% 55% / .5)}
.gr-v-incorrect{background:hsl(0 78% 60% / .16);border-color:hsl(0 78% 60% / .5)}
.gr-prov{font-size:var(--fs-xs);opacity:.85}
.gr-kv{display:grid;gap:.15rem;font-size:.9rem}
.gr-kv b{font-size:var(--fs-xs);text-transform:uppercase;letter-spacing:.06em;color:hsl(var(--ink-faint))}
.gr-opts{display:grid;gap:.3rem}
.gr-opt{display:flex;gap:.5rem;padding:.4rem .6rem;border-radius:var(--r-sm);border:1px solid var(--glass-border);font-size:.9rem}
.gr-opt[data-state=correct]{background:hsl(150 60% 45% / .16);border-color:hsl(150 60% 45% / .5)}
.gr-opt[data-state=wrong]{background:hsl(0 78% 60% / .14);border-color:hsl(0 78% 60% / .5)}
.gr-tools{display:flex;gap:.4rem;flex-wrap:wrap}
.gr-tools button{padding:.35rem .7rem;font-size:var(--fs-sm)}
.gr-slot{display:grid;gap:.5rem}
.gr-paper{display:grid;gap:1rem;min-width:0}
.gr-q{display:grid;gap:.5rem}
.gr-ans{width:100%;box-sizing:border-box;resize:vertical}
.gr-choice{display:flex;gap:.5rem;text-align:left;width:100%}
.gr-choice[aria-pressed=true]{border-color:hsl(var(--accent));background:hsl(var(--accent) / .16)}
@media (max-width:520px){.gr-head .score-ring{font-size:2.4rem}.gr-actions button{flex:1 1 auto}}
@media print{.gr-actions,.gr-tools{display:none!important}}
`;
    document.head.appendChild(st);
  }

  /* ---------------- helpers ---------------- */
  function newId() {
    try { if (window.crypto && crypto.randomUUID) return crypto.randomUUID().replace(/-/g, ""); } catch (_) {}
    return "a" + Date.now().toString(36) + Math.random().toString(36).slice(2, 12);
  }
  const isObj = (v) => v && typeof v === "object";
  const fmtNum = (n) => (Number.isFinite(+n) ? String(Math.round(+n * 100) / 100) : "0");
  const fmtTime = (s) => (s ? `${Math.floor(s / 60)}m ${s % 60}s` : "");
  const VERDICT = { correct: "Correct", partial: "Partial", incorrect: "Incorrect" };

  function toMarkdown(r) {
    const L = [`# ${r.title || "Results"}`, "", `**Score:** ${r.total} / ${r.max} (${r.percent}%) - ${r.grade_band}`];
    if (r.duration_seconds) L.push(`**Time:** ${fmtTime(r.duration_seconds)}`);
    if (r.llm && r.llm.label) L.push(`**Marking:** ${r.llm.label}`);
    L.push("");
    if ((r.topic_breakdown || []).length) {
      L.push("## By topic", "");
      r.topic_breakdown.forEach((b) => L.push(`- ${b.topic}: ${b.earned} / ${b.possible} (${b.percent}%)`));
      L.push("");
    }
    if ((r.weak_topics || []).length) { L.push("## Weak topics", ""); r.weak_topics.forEach((t) => L.push(`- ${t}`)); L.push(""); }
    L.push("## Questions", "");
    (r.items || []).forEach((it, i) => {
      L.push(`${i + 1}. **${it.question}** (${it.score}/${it.max}, ${it.verdict}${it.provisional ? ", provisional" : ""})`);
      if (it.user_answer != null && it.user_answer !== "") L.push(`   - Your answer: ${it.user_answer}`);
      if (it.correct_answer) L.push(`   - Correct answer: ${it.correct_answer}`);
      if (it.feedback) L.push(`   - Feedback: ${it.feedback}`);
    });
    return L.join("\n") + "\n";
  }

  function download(name, text) {
    const url = URL.createObjectURL(new Blob([text], { type: "text/markdown;charset=utf-8" }));
    const a = document.createElement("a");
    a.href = url; a.download = name; document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 2000);
  }

  function busy(btn, label, fn) {
    if (btn.querySelector(".ic")) {   // icon-only button: keep the glyph, show a spinning state + tooltip
      const t = btn.dataset.tip, a = btn.getAttribute("aria-label");
      btn.disabled = true; btn.classList.add("is-busy"); btn.dataset.tip = label; btn.setAttribute("aria-busy", "true");
      return Promise.resolve().then(fn).finally(() => { btn.disabled = false; btn.classList.remove("is-busy"); btn.dataset.tip = t; btn.removeAttribute("aria-busy"); if (a) btn.setAttribute("aria-label", a); });
    }
    const orig = btn.textContent;
    btn.disabled = true; btn.textContent = label;
    return Promise.resolve().then(fn).finally(() => { btn.disabled = false; btn.textContent = orig; });
  }

  /* raw item that produced a result row (by id, else by q<n> position) */
  function rawFor(rawItems, id) {
    if (!Array.isArray(rawItems)) return null;
    const byId = rawItems.find((x) => isObj(x) && String(x.id) === String(id));
    if (byId) return byId;
    const m = /^q(\d+)$/.exec(String(id));
    return m ? rawItems[+m[1] - 1] || null : null;
  }

  /* ---------------- grading call ---------------- */
  async function gradeInto(container, payload, opts) {
    opts = opts || {};
    clear(container);
    container.append(spinner(opts.loading || "Marking your answers…"));
    try {
      const res = await api().post("/grade", payload);
      clear(container);
      renderResult(container, res, Object.assign({}, opts, { payload }));
      if (opts.onResult) opts.onResult(res);
      return res;
    } catch (err) {
      clear(container);
      container.append(ui.errorBlock(err, () => gradeInto(container, payload, opts)));   // same attempt_id: safe to retry
      if (opts.onError) opts.onError(err);
      return null;
    }
  }

  /* ---------------- result component ---------------- */
  function renderResult(container, r, opts) {
    opts = opts || {};
    const raw = opts.items || (opts.payload && opts.payload.items) || [];
    const root = el(`<section class="gr-result" aria-label="Results"></section>`);
    const pct = Math.round(Number(r.percent) || 0);

    const head = el(`<div class="gr-head" role="status" aria-live="polite" tabindex="-1"></div>`);
    head.append(el(`<div class="score-ring">${pct}%</div>`));
    const ht = el(`<div></div>`);
    ht.append(el(`<div class="gr-band">${esc(r.grade_band || "")}</div>`));
    ht.append(el(`<div class="gr-sub">${esc(fmtNum(r.total))} of ${esc(fmtNum(r.max))} marks${r.duration_seconds ? " · " + esc(fmtTime(r.duration_seconds)) : ""}${r.title ? " · " + esc(r.title) : ""}</div>`));
    head.append(ht);
    root.append(head);

    if (r.notice) root.append(el(`<div class="gr-notice" role="note">${esc(r.notice)}</div>`));
    if (r.replayed) root.append(el(`<div class="gr-sub">This submission was already marked - showing the saved result (not counted twice).</div>`));
    if (r.saved === false && !r.provisional && !r.local) root.append(el(`<div class="gr-sub">History is off, so this result was not saved.</div>`));

    // ---- actions ----
    const actions = el(`<div class="gr-actions"></div>`);
    const wrong = (r.items || []).filter((it) => it.verdict !== "correct");
    const wrongRaw = wrong.map((it) => rawFor(raw, it.id)).filter(Boolean);
    if (wrongRaw.length) {
      const b = ui.iconBtn({ icon: "retry", label: `Retry the ${wrongRaw.length} I got wrong`, cls: "ib-accent" });
      b.addEventListener("click", () => {
        clear(container);
        runPaper(container, wrongRaw, { kind: r.kind === "exam" ? "practice" : (r.kind || "practice"), subject: r.subject, topic: r.topic,
          title: "Retry · " + (r.title || "wrong answers") }, opts);
      });
      actions.append(b);
    }
    if ((r.weak_topics || []).length && r.kind !== "flashcards") {
      const b = ui.iconBtn({ icon: "target", label: "Study weak topics", cls: "ib-outline" });
      b.addEventListener("click", () => {
        if (opts.onStudyWeak) return opts.onStudyWeak(r.weak_topics, r);
        studyWeak(container, r, opts, b);
      });
      actions.append(b);
    }
    if (r.provisional && opts.payload) {
      const b = ui.iconBtn({ icon: "refresh", label: "Re-grade now", cls: "ib-outline" });
      b.addEventListener("click", () => gradeInto(container, Object.assign({}, opts.payload, { regrade: true }), opts));
      actions.append(b);
    }
    const exp = ui.iconBtn({ icon: "download", label: "Export as Markdown (.md)", cls: "ib-outline" });
    exp.addEventListener("click", () => download(((r.title || "results").replace(/[^\w\-]+/g, "-").slice(0, 60) || "results") + ".md", toMarkdown(r)));
    const pr = ui.iconBtn({ icon: "print", label: "Print or save as PDF", cls: "ib-outline" });
    pr.addEventListener("click", () => {
      const label = r.llm && r.llm.label ? r.llm.label : "";
      const body = md(toMarkdown(r));
      ui.printDoc(r.title || "Results", body ? body.outerHTML : "", label);
    });
    actions.append(exp, pr);
    if (opts.onNew) {
      const b = ui.iconBtn({ icon: /close/i.test(opts.newLabel || "") ? "close" : "plus", label: opts.newLabel || "New", cls: "ib-outline" });
      b.addEventListener("click", opts.onNew);
      actions.append(b);
    }
    root.append(actions);

    // ---- topic breakdown ----
    const tb = r.topic_breakdown || [];
    if (tb.length) {
      const sec = el(`<div class="gr-sec"><h4>By topic</h4></div>`);
      tb.forEach((t) => {
        const row = el(`<div class="gr-row"><div class="gr-row-top"><span>${esc(t.topic)}</span><span>${esc(fmtNum(t.earned))} / ${esc(fmtNum(t.possible))}</span></div><div class="progress" role="img" aria-label="${esc(t.topic)}: ${Math.round(t.percent)} percent"><i style="width:${Math.max(0, Math.min(100, t.percent))}%"></i></div></div>`);
        sec.append(row);
      });
      if ((r.weak_topics || []).length) {
        const chips = el(`<div class="gr-chips" aria-label="Weak topics"></div>`);
        r.weak_topics.forEach((t) => chips.append(el(`<span class="badge">Needs work: ${esc(t)}</span>`)));
        sec.append(chips);
      }
      root.append(sec);
    }

    // ---- per item review ----
    const items = r.items || [];
    if (items.length && r.kind !== "teachback") {
      const sec = el(`<div class="gr-sec"><h4>Question by question</h4></div>`);
      items.forEach((it, i) => sec.append(itemCard(it, i, r, rawFor(raw, it.id))));
      root.append(sec);
    }

    // ---- model labels ----
    if (r.generated_llm) { const l = ui.renderModelLabel(r.generated_llm, { caption: "Questions", quiet: true }); if (l) root.append(l); }
    const ml = ui.renderModelLabel(r.llm, { caption: "Marking" });
    if (ml) root.append(ml);
    else if (!r.llm) root.append(el(`<div class="model-label gr-sub">Marked automatically (no AI model was needed).</div>`));

    container.append(root);
    try { head.focus({ preventScroll: false }); } catch (_) {}
    return root;
  }

  function itemCard(it, i, r, rawItem) {
    const d = el(`<details class="panel-soft gr-item"></details>`);
    if (it.verdict !== "correct") d.open = r.items.length <= 3;
    d.append(el(`<summary><span class="gr-v gr-v-${esc(it.verdict)}">${esc(fmtNum(it.score))}/${esc(fmtNum(it.max))} ${esc(VERDICT[it.verdict] || it.verdict)}</span><span class="gr-q">${i + 1}. ${esc(it.question)}</span>${it.provisional ? '<span class="gr-v gr-v-partial gr-prov">provisional</span>' : ""}</summary>`));
    const body = el(`<div class="gr-body"></div>`);
    if (Array.isArray(it.options) && it.options.length) {
      const opts = el(`<div class="gr-opts" role="list"></div>`);
      it.options.forEach((o, idx) => {
        const state = idx === it.answer_index ? "correct" : idx === it.user_index ? "wrong" : "";
        opts.append(el(`<div class="gr-opt" role="listitem" ${state ? `data-state="${state}"` : ""}><b>${String.fromCharCode(65 + idx)}</b><span>${esc(o)}${state === "correct" ? " ✓" : state === "wrong" ? " ✗ (your answer)" : ""}</span></div>`));
      });
      body.append(opts);
    } else {
      if (it.user_answer != null && it.user_answer !== "") body.append(el(`<div class="gr-kv"><b>Your answer</b><span>${esc(it.user_answer)}</span></div>`));
      else body.append(el(`<div class="gr-kv"><b>Your answer</b><span>(no answer)</span></div>`));
      if (it.correct_answer) body.append(el(`<div class="gr-kv"><b>${it.kind === "short" || it.kind === "long" ? "Key points" : "Correct answer"}</b><span>${esc(it.correct_answer)}</span></div>`));
    }
    if (it.feedback) { const fb = el(`<div class="gr-kv"><b>Feedback</b></div>`); fb.append(md(it.feedback)); body.append(fb); }
    if ((it.missing_points || []).length) {
      const m = el(`<div class="gr-kv"><b>Missing</b></div>`);
      const ul = el(`<ul style="margin:0;padding-left:1.1rem"></ul>`);
      it.missing_points.forEach((p) => { const li = document.createElement("li"); li.textContent = p; ul.append(li); });
      m.append(ul); body.append(m);
    }
    if (it.model_answer_hint) body.append(el(`<div class="gr-kv"><b>Hint</b><span>${esc(it.model_answer_hint)}</span></div>`));
    if (it.explanation) body.append(el(`<div class="gr-kv"><b>Explanation</b><span>${esc(it.explanation)}</span></div>`));

    const slot = el(`<div class="gr-slot" aria-live="polite"></div>`);
    const tools = el(`<div class="gr-tools"></div>`);
    const ex = ui.iconBtn({ icon: "info", label: "Explain this answer", cls: "ib-outline" });
    ex.addEventListener("click", () => busy(ex, "Explaining…", async () => {
      clear(slot);
      try {
        const res = await api().post("/grade/explain", {
          question: it.question, topic: it.topic, options: Array.isArray(it.options) ? it.options : [],
          correct_answer: String(it.correct_answer || ""), user_answer: String(it.user_answer == null ? "" : it.user_answer),
          explanation: it.explanation || "",
        });
        slot.append(md(res.explanation || ""));
        const l = ui.renderModelLabel(res.llm, { caption: "Explanation" }); if (l) slot.append(l);
      } catch (e) { slot.append(ui.errorBlock(e, () => ex.click())); }
    }));
    tools.append(ex);
    if (rawItem) {
      const sim = ui.iconBtn({ icon: "shuffle", label: "Similar question", cls: "ib-outline" });
      sim.addEventListener("click", () => busy(sim, "Writing…", async () => {
        clear(slot);
        try {
          const res = await api().post("/grade/similar", { item: rawItem });
          const box = el(`<div class="panel-soft"></div>`);
          slot.append(box);
          const l = ui.renderModelLabel(res.llm, { caption: "Question" });
          runPaper(box, [res.item], { kind: "practice", subject: r.subject, topic: it.topic, title: "Similar · " + it.topic }, { compact: true });
          if (l) slot.append(l);
        } catch (e) { slot.append(ui.errorBlock(e, () => sim.click())); }
      }));
      tools.append(sim);
    }
    body.append(tools, slot);
    d.append(body);
    return d;
  }

  /* "Study weak topics" default: a fresh 5-question quiz on exactly those topics, right here */
  async function studyWeak(container, r, opts, btn) {
    const topics = r.weak_topics.slice(0, 4);
    await busy(btn, "Writing a quiz…", async () => {
      try {
        const res = await api().post("/quiz/topic", { topic: topics.join("; "), num_questions: 5, difficulty: "medium" });
        clear(container);
        runPaper(container, res.questions || [], { kind: "quiz", topic: topics.join(", "), title: "Weak topics · " + topics.join(", "),
          quiz_id: res.id || undefined, generated_llm: res.llm }, opts);
      } catch (e) {
        container.prepend(ui.errorBlock(e, () => studyWeak(container, r, opts, btn)));
      }
    });
  }

  /* ---------------- one-page practice paper (retry wrong / similar / weak-topic quiz) ---------------- */
  function runPaper(container, rawItems, meta, opts) {
    opts = opts || {};
    meta = meta || {};
    const attemptId = newId();
    const answers = {};
    const started = Date.now();
    const paper = el(`<div class="gr-paper"></div>`);
    if (!opts.compact) paper.append(el(`<div class="gr-band">${esc(meta.title || "Practice")}</div>`));
    // ids: keep the item's own id, else position, so results map back to the raw list
    const items = rawItems.map((x, i) => Object.assign({}, x, { id: (x && x.id) ? String(x.id) : "q" + (i + 1) }));
    items.forEach((it, i) => {
      const block = el(`<fieldset class="gr-q" style="border:0;padding:0;margin:0;min-width:0"><legend style="font-weight:600;overflow-wrap:anywhere">${i + 1}. ${esc(it.question)}</legend></fieldset>`);
      const t = String(it.type || "").toLowerCase();
      const opts2 = Array.isArray(it.options) ? it.options : (isObj(it.options) ? Object.keys(it.options).sort().map((k) => it.options[k]) : null);
      if (opts2 && opts2.length) {
        const btns = [];
        opts2.forEach((text, idx) => {
          const b = el(`<button class="btn gr-choice" type="button" aria-pressed="false"><b>${String.fromCharCode(65 + idx)}</b><span>${esc(text)}</span></button>`);
          b.addEventListener("click", () => { answers[it.id] = idx; btns.forEach((o, j) => o.setAttribute("aria-pressed", String(j === idx))); });
          btns.push(b); block.append(b);
        });
      } else if (t === "tf") {
        const btns = [];
        [["True", true], ["False", false]].forEach(([label, val]) => {
          const b = el(`<button class="btn gr-choice" type="button" aria-pressed="false">${label}</button>`);
          b.addEventListener("click", () => { answers[it.id] = val; btns.forEach((o) => o.setAttribute("aria-pressed", String(o === b))); });
          btns.push(b); block.append(b);
        });
      } else if (t === "numeric") {
        const inp = el(`<input class="input gr-ans" type="text" inputmode="decimal" aria-label="Your answer to question ${i + 1}" placeholder="Your answer…" />`);
        inp.addEventListener("input", () => { answers[it.id] = inp.value; });
        block.append(inp);
      } else {
        const ta = el(`<textarea class="textarea gr-ans" rows="3" aria-label="Your answer to question ${i + 1}" placeholder="Your answer…"></textarea>`);
        ta.addEventListener("input", () => { answers[it.id] = ta.value; });
        block.append(ta);
      }
      paper.append(block);
    });
    const submit = el(`<button class="btn-accent" type="button">${items.length === 1 ? "Check answer" : "Submit"}</button>`);
    const out = el(`<div></div>`);
    submit.addEventListener("click", () => {
      if (submit.disabled) return;
      submit.disabled = true;
      const payload = {
        kind: meta.kind || "practice", subject: meta.subject || undefined, topic: meta.topic || undefined, title: meta.title || undefined,
        items, answers, attempt_id: attemptId, quiz_id: meta.quiz_id || undefined,
        duration_seconds: Math.round((Date.now() - started) / 1000), generated_llm: meta.generated_llm || undefined,
      };
      clear(paper); paper.append(out);
      gradeInto(out, payload, Object.assign({}, opts, { items, onNew: opts.onNew }));
    });
    paper.append(submit);
    container.append(paper);
    return paper;
  }

  window.GradeUI = { newId, gradeInto, renderResult, runPaper, toMarkdown };
})();
