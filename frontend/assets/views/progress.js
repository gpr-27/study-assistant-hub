/* ============================================================
   Progress / Weakness Heatmap view — window.Views.progress
   Reads /api/progress/summary (events recorded automatically by
   the Quiz and Flashcards tools) and renders: stat tiles, a
   "weakest topics" list with accuracy bars, and a per-subject
   heatmap grid of topic cells colored red→green by accuracy.
   No model selector — this view never calls an LLM.
   ============================================================ */
(() => {
  "use strict";
  const { el, clear, esc, spinner, empty, toast } = window.ui;

  let root = null;

  const pct = (acc) => Math.round((Number(acc) || 0) * 100);
  // red (0) → green (120) hue ramp; saturation/lightness tuned to read on glass.
  const accColor = (acc, alpha) =>
    `hsl(${Math.round(Math.max(0, Math.min(1, Number(acc) || 0)) * 120)} 72% 46%${alpha ? " / " + alpha : ""})`;
  const fmtDate = (iso) => {
    if (!iso) return "";
    try { return new Date(iso).toLocaleDateString(); } catch (_) { return ""; }
  };

  function statTiles(overall) {
    const wrap = el(`<div class="prog-tiles"></div>`);
    const tiles = [
      { value: String(overall.total_attempts || 0), label: "Sessions logged" },
      { value: pct(overall.overall_accuracy) + "%", label: "Overall accuracy" },
      { value: String(overall.streak_days || 0) + (overall.streak_days === 1 ? " day" : " days"), label: "Study streak" },
    ];
    tiles.forEach((t) => {
      wrap.append(el(`
        <div class="panel-soft prog-tile">
          <div class="prog-tile-value">${esc(t.value)}</div>
          <div class="muted">${esc(t.label)}</div>
        </div>`));
    });
    return wrap;
  }

  function weakestList(weakest) {
    const panel = el(`<div class="panel stack-sm"></div>`);
    panel.append(el(`<div class="section-head" style="margin:0"><h3>Weakest topics</h3><span>lowest accuracy first</span></div>`));
    if (!weakest.length) {
      panel.append(el(`<p class="muted">No weak spots yet — topics show up here once they have 2+ attempts.</p>`));
      return panel;
    }
    weakest.forEach((w) => {
      const p = pct(w.accuracy);
      const row = el(`
        <div class="stack-sm" style="gap:.3rem">
          <div class="between" style="gap:.5rem;align-items:flex-start">
            <span class="prog-weak-name" style="min-width:0;overflow-wrap:anywhere">${esc(w.topic)}${w.subject && w.subject !== w.topic ? ` <span class="muted">· ${esc(w.subject)}</span>` : ""}</span>
            <span class="badge" style="flex:none">${p}% · ${w.attempts}×</span>
          </div>
          <div class="prog-bar"><i style="width:${p}%;background:${accColor(w.accuracy)}"></i></div>
        </div>`);
      panel.append(row);
    });
    return panel;
  }

  function heatmap(subjects) {
    const panel = el(`<div class="panel stack"></div>`);
    panel.append(el(`<div class="section-head" style="margin:0"><h3>Subject heatmap</h3><span>red = needs work, green = solid</span></div>`));
    subjects.forEach((s) => {
      const group = el(`<div class="stack-sm" style="gap:.45rem"></div>`);
      group.append(el(`
        <div class="between" style="gap:.5rem;align-items:flex-start">
          <span class="prog-subj" style="min-width:0;overflow-wrap:anywhere">${esc(s.subject)}</span>
          <span class="badge badge-accent" style="flex:none">${pct(s.accuracy)}% · ${s.attempts} session${s.attempts === 1 ? "" : "s"}</span>
        </div>`));
      const grid = el(`<div class="prog-heat"></div>`);
      s.topics.forEach((t) => {
        const cell = el(`
          <div class="prog-cell" style="background:${accColor(t.accuracy, ".82")}"
               title="${esc(t.topic)} — ${pct(t.accuracy)}% over ${t.attempts} attempt${t.attempts === 1 ? "" : "s"}${t.last_seen ? " · last " + esc(fmtDate(t.last_seen)) : ""}">
            <span class="prog-cell-topic">${esc(t.topic)}</span>
            <span class="prog-cell-acc">${pct(t.accuracy)}%</span>
          </div>`);
        grid.append(cell);
      });
      group.append(grid);
      panel.append(group);
    });
    return panel;
  }

  // ── AI Insights: one cross-tool analysis (chats, quizzes, exams, teach-backs…)
  const TOOL_ROUTES = {
    quiz: "#quiz", flashcards: "#flashcards", exam: "#exam", teachback: "#teachback",
    chat: "#chat", planner: "#planner", cheatsheet: "#cheatsheet", mindmap: "#mindmap",
  };

  function renderInsights(panel, payload) {
    clear(panel);
    const head = el(`<div class="between" style="align-items:center;gap:.5rem">
      <div class="section-head" style="margin:0;min-width:0"><h3>✦ AI insights</h3></div>
      <button class="icon-btn ib-outline" type="button" title="Re-analyze now" aria-label="Refresh AI insights">${window.ui.iconHtml("refresh")}</button>
    </div>`);
    head.querySelector("button").addEventListener("click", () => loadInsights(panel, true));
    panel.append(head);

    if (payload.empty || !payload.insights) {
      panel.append(el(`<p class="muted">Use the tools for a bit — insights appear once there's activity to analyze.</p>`));
      return;
    }
    const ins = payload.insights;
    const hl = el(`<p class="prog-insight-headline"></p>`);
    hl.textContent = ins.headline || "";
    panel.append(hl);
    { const l = window.ui.renderModelLabel(payload.llm); if (l) panel.append(l); }   // "Model: …" for the AI insights, under the headline

    (ins.observations || []).forEach((o) => {
      const li = el(`<p class="muted" style="margin:0;font-size:.92rem">• <span></span></p>`);
      li.querySelector("span").textContent = o;
      panel.append(li);
    });

    if ((ins.recommendations || []).length) {
      const recs = el(`<div class="stack-sm" style="gap:.5rem;margin-top:.3rem"></div>`);
      ins.recommendations.forEach((r) => {
        const route = r.tool && TOOL_ROUTES[r.tool];
        const card = el(`<${route ? `a href="${esc(route)}"` : "div"} class="panel-soft prog-rec"></${route ? "a" : "div"}>`);
        const action = el(`<div class="prog-rec-action"></div>`);
        action.textContent = r.action;
        const reason = el(`<div class="muted" style="font-size:var(--fs-sm)"></div>`);
        reason.textContent = r.reason || "";
        card.append(action, reason);
        if (route) card.append(el(`<span class="prog-rec-go">Open ${esc(r.tool)} →</span>`));
        recs.append(card);
      });
      panel.append(recs);
    }

    if ((ins.focus_topics || []).length) {
      const chips = el(`<div class="row row-wrap" style="gap:.35rem;margin-top:.2rem"></div>`);
      ins.focus_topics.forEach((t) => {
        const chip = el(`<span class="badge badge-accent"></span>`);
        chip.textContent = t;
        chips.append(chip);
      });
      panel.append(el(`<div class="label" style="margin-top:.3rem">Focus next</div>`), chips);
    }
  }

  async function loadInsights(panel, refresh) {
    clear(panel);
    panel.append(spinner(refresh ? "Re-analyzing your activity…" : "Analyzing your activity across all tools…"));
    try {
      renderInsights(panel, await window.api.get("/insights" + (refresh ? "?refresh=1" : "")));
    } catch (e) {
      clear(panel);
      const err = new Error("Couldn't generate insights right now" + (e && e.message ? " — " + e.message : "") + ".");
      if (e && e.llm) err.llm = e.llm;       // keep "Model: none (all providers failed)" on the failure card
      panel.append(window.ui.errorBlock(err, () => loadInsights(panel, true)));
    }
  }

  // ── All-app activity: one tile per tool (chats, quizzes, exams, scans…) ──
  function activityGrid(activity) {
    const panel = el(`<div class="panel stack-sm"></div>`);
    panel.append(el(`<div class="section-head" style="margin:0"><h3>Your activity across the app</h3><span>tap a tile to open the tool</span></div>`));
    const grid = el(`<div class="prog-acts"></div>`);
    activity.forEach((a) => {
      const tile = el(`
        <a class="panel-soft prog-act${a.count ? "" : " prog-act--zero"}" href="${esc(a.route)}">
          <span class="prog-act-count">${a.count}</span>
          <span class="prog-act-label">${esc(a.label)}</span>
          ${a.metric ? `<span class="badge badge-accent">${esc(a.metric)}</span>` : ""}
        </a>`);
      grid.append(tile);
    });
    panel.append(grid);
    return panel;
  }

  // Recent graded attempts (GET /api/attempts): click one to reopen its full unified result.
  async function recentAttempts(panel) {
    panel.append(el(`<div class="section-head" style="margin:0"><h3>Recent attempts</h3><span>marked quizzes, exams, reviews and teach-backs</span></div>`));
    const body = el(`<div class="stack-sm" aria-live="polite"></div>`);
    panel.append(body);
    body.append(spinner("Loading attempts…"));
    try {
      window.api.invalidate && window.api.invalidate("/attempts");
      const data = await window.api.get("/attempts?limit=8");
      clear(body);
      if (!data.items.length) { body.append(el(`<p class="hint">No graded attempts yet.</p>`)); return; }
      data.items.forEach((a) => {
        const row = el(`<button type="button" class="panel-soft" style="text-align:left;width:100%;display:grid;gap:.2rem;cursor:pointer"></button>`);
        row.append(el(`<div class="between" style="gap:.5rem"><strong style="overflow-wrap:anywhere;min-width:0">${esc(a.title || a.kind)}</strong><span class="badge ${a.percent >= 75 ? "badge-accent" : ""}">${Math.round(a.percent || 0)}% · ${esc(a.grade_band || "")}</span></div>`));
        row.append(el(`<div class="muted" style="font-size:var(--fs-sm)">${esc(a.kind)} · ${esc(fmtDate(a.created_at))}${a.verified === false ? " · not counted" : ""}${a.provisional ? " · provisional" : ""}</div>`));
        const l = window.ui.renderModelLabel(a.llm, { stored: false, quiet: true, caption: "Marking" }); if (l) row.append(l);
        row.addEventListener("click", async () => {
          row.disabled = true;
          try {
            await window.ui.loadScript("/assets/grade-ui.js");
            const full = await window.api.get("/attempts/" + encodeURIComponent(a.id));
            const host = el(`<div class="panel" id="prog-attempt"></div>`);
            const old = panel.querySelector("#prog-attempt"); if (old) old.remove();
            panel.append(host);
            window.GradeUI.renderResult(host, full, { newLabel: "Close", onNew: () => host.remove() });
          } catch (e) { toast(e.message, "error"); }
          row.disabled = false;
        });
        body.append(row);
      });
    } catch (e) { clear(body); body.append(window.ui.errorBlock(e, () => { clear(panel); recentAttempts(panel); })); }
  }

  function renderSummary(data) {
    clear(root);
    const overall = (data && data.overall) || { total_attempts: 0, overall_accuracy: 0, streak_days: 0 };
    const subjects = (data && data.subjects) || [];
    const weakest = (data && data.weakest) || [];
    const activity = (data && data.activity) || [];
    const anyActivity = activity.some((a) => a.count > 0);

    const wrap = el(`<div class="stack" style="max-width:840px;margin:0 auto"></div>`);

    if (!anyActivity && !overall.total_attempts) {
      const intro = el(`<div class="panel stack"></div>`);
      intro.append(empty("Nothing to analyze yet. Use any tool — chats, quizzes, exams, scans — and your activity, performance and AI insights all land here.", "📊"));
      intro.append(el(`
        <div class="row-wrap" style="justify-content:center">
          <a class="btn" href="#chat">Start a chat</a>
          <a class="btn" href="#quiz">Take a quiz</a>
          <a class="btn" href="#exam">Sit a mock exam</a>
        </div>`));
      wrap.append(intro);
      root.append(wrap);
      return;
    }

    // 1) Whole-app activity (every tool), 2) AI insights, 3) quiz/flashcard performance.
    if (activity.length) wrap.append(activityGrid(activity));
    const insPanel = el(`<div class="panel stack-sm"></div>`);
    wrap.append(insPanel);
    loadInsights(insPanel, false);

    const attemptsPanel = el(`<div class="panel stack-sm"></div>`);
    wrap.append(attemptsPanel);
    recentAttempts(attemptsPanel);

    if (overall.total_attempts) {
      wrap.append(statTiles(overall));
      wrap.append(weakestList(weakest));
      if (subjects.length) wrap.append(heatmap(subjects));
      wrap.append(el(`<p class="hint center">Accuracy stats are based on your last 100 quiz and flashcard sessions.</p>`));
    } else {
      wrap.append(el(`<p class="hint center">Finish a quiz or rate a flashcard deck to unlock accuracy stats and the weakness heatmap.</p>`));
    }
    root.append(wrap);
  }

  async function load() {
    clear(root);
    root.append(spinner("Crunching your study data…"));
    try {
      renderSummary(await window.api.get("/progress/summary"));
    } catch (e) {
      clear(root);
      if (e.status === 401) {
        root.append(empty("Sign in (or continue as a guest) to track your progress.", "🔒"));
      } else {
        root.append(window.ui.errorBlock(new Error("Couldn’t load your progress: " + e.message), () => load()));
      }
    }
  }

  window.Views.progress = {
    accent: "plan",
    title: "Progress",
    async mount(node) {
      root = node;
      await load();
    },
    unmount() {
      if (root) clear(root);
      root = null;
    },
  };
})();
