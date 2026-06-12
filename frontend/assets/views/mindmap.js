/* ============================================================
   Mind Map view — window.Views.mindmap
   Build a hierarchical concept map from one of three sources:
   a single PDF in a subject, a typed topic, or an uploaded PDF.
   The backend returns a bounded {label, children} tree; we render
   it as a connector-railed mind map and offer a copy-as-outline.
   ============================================================ */
(() => {
  "use strict";
  const { el, clear, esc, spinner, toast, copy } = window.ui;

  let source = "subject";   // "subject" | "topic" | "upload"
  let pdfFile = null;       // File for upload mode

  function buildSourceControl(host) {
    clear(host);
    if (source === "subject") {
      const field = el(`
        <div class="stack-sm">
          <div class="field" style="margin:0">
            <label class="label" for="mmSubject">Subject</label>
            <select id="mmSubject" class="select"><option value="">Loading subjects…</option></select>
          </div>
          <div class="field" style="margin:0">
            <label class="label" for="mmPdf">Document (pick one)</label>
            <select id="mmPdf" class="select" disabled><option value="">Select a subject first</option></select>
          </div>
        </div>`);
      host.append(field);
      const subjectSel = field.querySelector("#mmSubject");
      const pdfSel = field.querySelector("#mmPdf");

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
    } else if (source === "topic") {
      host.append(el(`
        <div class="field" style="margin:0">
          <label class="label" for="mmTopic">Topic</label>
          <input id="mmTopic" class="input" type="text" placeholder="e.g. Photosynthesis" autocomplete="off" />
          <div class="hint">A concept map is built from the topic — no document needed.</div>
        </div>`));
    } else {
      const field = el(`
        <div class="field" style="margin:0">
          <label class="label">PDF</label>
          <div class="row row-wrap">
            <label class="btn" for="mmPdfFile" style="cursor:pointer">Choose PDF…</label>
            <span class="muted" id="mmPdfName">No file selected</span>
          </div>
          <input id="mmPdfFile" type="file" accept="application/pdf" hidden />
          <div class="hint">Upload one PDF and map its key concepts.</div>
        </div>`);
      host.append(field);
      const input = field.querySelector("#mmPdfFile");
      const nameSlot = field.querySelector("#mmPdfName");
      input.addEventListener("change", () => {
        pdfFile = input.files && input.files[0] ? input.files[0] : null;
        nameSlot.textContent = pdfFile ? pdfFile.name : "No file selected";
      });
    }
  }

  const SVG_NS = "http://www.w3.org/2000/svg";

  // Default fold state: root open, every branch collapsed (drill in on click).
  function prepTree(node, depth) {
    node.expanded = depth === 0;
    for (const child of node.children || []) prepTree(child, depth + 1);
  }

  function hasChildren(node) {
    return Array.isArray(node.children) && node.children.length > 0;
  }

  // Mount an interactive left-to-right concept graph into `host`. Nodes are
  // absolutely positioned on a stage; curved SVG rails connect each parent to
  // its visible children. The stage auto-centres in the scroll layer (margin
  // auto) and a CSS transform handles zoom. Toggling re-runs the layout.
  function mountMindmap(root, host) {
    const PAD = 32;     // outer margin around the graph
    const H_GAP = 96;   // horizontal space between depth columns
    const V_GAP = 16;   // vertical space between stacked leaves
    const START_X = 28; // where a parent's rail leaves its card (clears the toggle)
    const ZOOM_MIN = 0.4;
    const ZOOM_MAX = 2.2;

    const canvas = el(`
      <div class="mm-canvas">
        <div class="mm-scroll"><div class="mm-viewport"><div class="mm-stage"></div></div></div>
        <div class="mm-zoom">
          <button type="button" data-z="out" aria-label="Zoom out">−</button>
          <button type="button" data-z="reset" aria-label="Reset zoom">100%</button>
          <button type="button" data-z="in" aria-label="Zoom in">+</button>
        </div>
      </div>`);
    const scroll = canvas.querySelector(".mm-scroll");
    const viewport = canvas.querySelector(".mm-viewport");
    const stage = canvas.querySelector(".mm-stage");
    const zoomLabel = canvas.querySelector('[data-z="reset"]');
    host.append(canvas);

    let zoom = 1;
    let contentW = 0;
    let contentH = 0;
    let centeredOnce = false;

    function applyZoom() {
      viewport.style.width = `${contentW * zoom}px`;
      viewport.style.height = `${contentH * zoom}px`;
      stage.style.transform = `scale(${zoom})`;
      zoomLabel.textContent = `${Math.round(zoom * 100)}%`;
    }

    // Re-zoom while keeping the content point under (clientX, clientY) fixed.
    function setZoom(next, clientX, clientY) {
      const z = Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, next));
      if (z === zoom) return;
      const rect = scroll.getBoundingClientRect();
      const ax = clientX == null ? rect.width / 2 : clientX - rect.left;
      const ay = clientY == null ? rect.height / 2 : clientY - rect.top;
      const cx = (scroll.scrollLeft + ax) / zoom;
      const cy = (scroll.scrollTop + ay) / zoom;
      zoom = z;
      applyZoom();
      scroll.scrollLeft = cx * zoom - ax;
      scroll.scrollTop = cy * zoom - ay;
    }

    // `animateFrom` (optional) is the node just expanded — its freshly-revealed
    // descendants ease in, so opening a branch feels smooth instead of popping.
    function render(animateFrom) {
      const keepLeft = scroll.scrollLeft;
      const keepTop = scroll.scrollTop;
      clear(stage);

      const svg = document.createElementNS(SVG_NS, "svg");
      svg.setAttribute("class", "mm-links");
      stage.append(svg);

      const visible = [];
      (function walk(node, depth, inNew) {
        node._depth = depth;
        node._enter = inNew && node !== animateFrom;   // a newly-revealed child
        visible.push(node);
        if (node.expanded) for (const child of node.children || []) walk(child, depth + 1, inNew || node === animateFrom);
      })(root, 0, false);

      // Create + measure every visible card (offsetWidth needs a live DOM).
      for (const node of visible) {
        const wrap = el(`<div class="mm-node-wrap${node._enter ? " mm-enter" : ""}"></div>`);
        const card = el(`<div class="mm-card lvl${Math.min(node._depth, 2)}">${esc(node.label)}</div>`);
        wrap.append(card);
        if (hasChildren(node)) {
          const toggle = el(`<button class="mm-toggle" type="button" aria-label="Expand or collapse" aria-expanded="${node.expanded}">${node.expanded ? "‹" : "›"}</button>`);
          toggle.addEventListener("click", () => { node.expanded = !node.expanded; render(node.expanded ? node : null); });
          wrap.append(toggle);
        }
        stage.append(wrap);
        node._wrap = wrap;
        node._w = card.offsetWidth;
        node._h = card.offsetHeight;
      }

      // X by depth: each column starts past the widest card of the column before it.
      const colWidth = {};
      let maxDepth = 0;
      for (const node of visible) {
        colWidth[node._depth] = Math.max(colWidth[node._depth] || 0, node._w);
        maxDepth = Math.max(maxDepth, node._depth);
      }
      const colX = { 0: PAD };
      for (let d = 1; d <= maxDepth; d += 1) colX[d] = colX[d - 1] + colWidth[d - 1] + H_GAP;

      // Y via post-order: leaves stack, parents centre on their visible children.
      let cursor = PAD;
      (function place(node) {
        const kids = node.expanded ? node.children || [] : [];
        if (!kids.length) {
          node._cy = cursor + node._h / 2;
          cursor += node._h + V_GAP;
          return;
        }
        kids.forEach(place);
        node._cy = (kids[0]._cy + kids[kids.length - 1]._cy) / 2;
      })(root);

      let maxX = 0;
      let maxY = 0;
      for (const node of visible) {
        node._x = colX[node._depth];
        node._wrap.style.left = `${node._x}px`;
        node._wrap.style.top = `${node._cy - node._h / 2}px`;
        maxX = Math.max(maxX, node._x + node._w);
        maxY = Math.max(maxY, node._cy + node._h / 2);
      }
      contentW = maxX + PAD + 32;
      contentH = maxY + PAD;
      stage.style.width = `${contentW}px`;
      stage.style.height = `${contentH}px`;
      svg.setAttribute("viewBox", `0 0 ${contentW} ${contentH}`);

      for (const node of visible) {
        if (!node.expanded) continue;
        const sx = node._x + node._w + START_X;
        const sy = node._cy;
        for (const child of node.children || []) {
          const ex = child._x;
          const ey = child._cy;
          const dx = Math.max(30, (ex - sx) / 2);
          const path = document.createElementNS(SVG_NS, "path");
          path.setAttribute("class", "mm-link" + (child._enter ? " mm-link-enter" : ""));
          path.setAttribute("d", `M ${sx} ${sy} C ${sx + dx} ${sy} ${ex - dx} ${ey} ${ex} ${ey}`);
          svg.append(path);
        }
      }

      applyZoom();
      if (centeredOnce) {
        scroll.scrollLeft = keepLeft;
        scroll.scrollTop = keepTop;
      } else {
        // First paint: keep the root in view, vertically centred on its row.
        scroll.scrollLeft = 0;
        scroll.scrollTop = Math.max(0, root._cy * zoom - scroll.clientHeight / 2);
        centeredOnce = true;
      }
    }

    // Pan + pinch-zoom via Pointer Events — one code path for mouse, touch and pen.
    // One pointer drags to pan; two pointers pinch to zoom toward their midpoint.
    // On touch, CSS sets `touch-action: none` on .mm-scroll so the browser hands us
    // the gesture instead of native-scrolling or page-zooming. (Toggles and the zoom
    // buttons keep their own clicks.)
    const pointers = new Map();   // pointerId -> {x, y}
    let pinchPrev = 0;            // last finger-to-finger distance during a pinch

    scroll.addEventListener("pointerdown", (e) => {
      if (e.target.closest(".mm-toggle") || e.target.closest(".mm-zoom")) return;
      pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
      if (pointers.size === 1) canvas.classList.add("is-panning");
      if (pointers.size === 2) {
        const [a, b] = [...pointers.values()];
        pinchPrev = Math.hypot(a.x - b.x, a.y - b.y);
      }
      try { scroll.setPointerCapture(e.pointerId); } catch (_) {}
    });

    scroll.addEventListener("pointermove", (e) => {
      const p = pointers.get(e.pointerId);
      if (!p) return;
      const dx = e.clientX - p.x, dy = e.clientY - p.y;
      p.x = e.clientX; p.y = e.clientY;

      if (pointers.size >= 2) {
        // Pinch: scale by the change in finger distance, anchored at the midpoint.
        const [a, b] = [...pointers.values()];
        const dist = Math.hypot(a.x - b.x, a.y - b.y);
        if (pinchPrev > 0 && dist > 0) setZoom(zoom * (dist / pinchPrev), (a.x + b.x) / 2, (a.y + b.y) / 2);
        pinchPrev = dist;
        return;
      }
      // Single pointer: drag to pan.
      scroll.scrollLeft -= dx;
      scroll.scrollTop -= dy;
    });

    const endPointer = (e) => {
      if (!pointers.delete(e.pointerId)) return;
      if (pointers.size < 2) pinchPrev = 0;
      if (pointers.size === 0) canvas.classList.remove("is-panning");
    };
    scroll.addEventListener("pointerup", endPointer);
    scroll.addEventListener("pointercancel", endPointer);

    // Wheel (or trackpad pinch) zooms toward the pointer — desktop fast path.
    scroll.addEventListener("wheel", (e) => {
      e.preventDefault();
      const factor = e.deltaY < 0 ? 1.12 : 1 / 1.12;
      setZoom(zoom * factor, e.clientX, e.clientY);
    }, { passive: false });

    canvas.querySelector(".mm-zoom").addEventListener("click", (e) => {
      const btn = e.target.closest("button[data-z]");
      if (!btn) return;
      if (btn.dataset.z === "in") setZoom(zoom * 1.2);
      else if (btn.dataset.z === "out") setZoom(zoom / 1.2);
      else setZoom(1);
    });

    render();
  }

  function toOutline(node, depth) {
    const prefix = depth === 0 ? "# " : "  ".repeat(depth - 1) + "- ";
    let out = prefix + node.label + "\n";
    for (const child of node.children || []) out += toOutline(child, depth + 1);
    return out;
  }

  async function generate(layout) {
    const goBtn = layout.querySelector("#mmGo");
    const result = layout.querySelector("#mmResult");
    const model = window.ui.selectedModel(layout);

    let call;
    if (source === "subject") {
      const subject = layout.querySelector("#mmSubject")?.value;
      const pdf = layout.querySelector("#mmPdf")?.value;
      if (!subject) { toast("Pick a subject.", "error"); return; }
      if (!pdf) { toast("Pick a document.", "error"); return; }
      call = () => window.api.post("/mindmap/generate", { mode: "subject", subject, pdf, model });
    } else if (source === "topic") {
      const topic = layout.querySelector("#mmTopic")?.value.trim();
      if (!topic) { toast("Enter a topic.", "error"); return; }
      call = () => window.api.post("/mindmap/generate", { mode: "topic", topic, model });
    } else {
      if (!pdfFile) { toast("Choose a PDF.", "error"); return; }
      call = () => { const fd = new FormData(); fd.append("file", pdfFile); if (model) fd.append("model", model); return window.api.postForm("/mindmap/upload", fd); };
    }

    goBtn.disabled = true;
    goBtn.textContent = "Mapping…";
    clear(result);
    result.append(spinner("Distilling the key concepts into a mind map…"));

    try {
      const resp = await call();
      const root = resp.root;
      prepTree(root, 0);
      clear(result);
      const head = el(`<div class="row row-wrap between" style="margin-bottom:1rem"><h3 style="font-family:var(--font-display);font-weight:700;letter-spacing:-.02em;margin:0">Mind map</h3></div>`);
      const copyBtn = el(`<button class="btn" type="button">Copy outline</button>`);
      copyBtn.addEventListener("click", () => copy(toOutline(root, 0), copyBtn));
      head.append(copyBtn);
      result.append(head);
      mountMindmap(root, result);
      toast("Mind map ready. Drag to pan, click ‹ › to fold.", "ok");
      const __hp = document.querySelector('.history-panel[data-kind="mindmaps"]');
      if (__hp && __hp.refresh) __hp.refresh();
    } catch (e) {
      clear(result);
      result.append(el(`<div class="empty"><div class="big">⚠</div><div>${esc(e.message)}</div></div>`));
      toast(e.message, "error");
    } finally {
      goBtn.disabled = false;
      goBtn.textContent = "Generate mind map";
    }
  }

  window.Views = window.Views || {};
  window.Views.mindmap = {
    accent: "mindmap",
    title: "Mind Map",
    async mount(root) {
      const layout = el(`
        <div class="stack">
          <div class="panel stack">
            <div class="between" style="align-items:flex-start">
              <div class="stack-sm" style="flex:1">
                <h3 style="font-family:var(--font-display);font-weight:700;letter-spacing:-.02em">Create a mind map</h3>
                <p class="muted">From one PDF in a subject, a topic you type, or a PDF you upload.</p>
              </div>
              <button id="mm-hist" class="btn" type="button" style="flex:none;padding:.4rem .8rem;font-size:.85rem">≡ History</button>
            </div>
            <div class="seg" role="tablist" aria-label="Mind map source">
              <button type="button" data-src="subject" aria-pressed="true">Subject PDF</button>
              <button type="button" data-src="topic" aria-pressed="false">Topic</button>
              <button type="button" data-src="upload" aria-pressed="false">Upload PDF</button>
            </div>
            <div id="mmSourceCtl"></div>
            <div id="mmModelCtl"></div>
            <button id="mmGo" class="btn-accent btn-lg btn-block" type="button">Generate mind map</button>
          </div>
          <div class="panel" id="mmResult">
            <div class="empty"><div class="big">✦</div><div>Pick a source and generate your mind map.</div></div>
          </div>
        </div>`);
      root.append(layout);

      function openSavedMindmap(saved) {
        const result = layout.querySelector("#mmResult");
        const tree = saved && saved.root;
        if (!tree) return;
        prepTree(tree, 0);
        clear(result);
        const head = el('<div class="row row-wrap between" style="margin-bottom:1rem"><h3 style="font-family:var(--font-display);font-weight:700;letter-spacing:-.02em;margin:0">Mind map</h3></div>');
        const copyBtn = el('<button class="btn" type="button">Copy outline</button>');
        copyBtn.addEventListener("click", () => copy(toOutline(tree, 0), copyBtn));
        head.append(copyBtn);
        result.append(head);
        mountMindmap(tree, result);
      }
      layout.querySelector("#mm-hist").addEventListener("click", (e) =>
        window.ui.openHistoryDrawer("mindmaps", { title: "Saved mind maps", anchor: e.currentTarget, onOpen: openSavedMindmap }));

      const seg = layout.querySelector(".seg");
      const sourceCtl = layout.querySelector("#mmSourceCtl");
      buildSourceControl(sourceCtl);
      layout.querySelector("#mmModelCtl").append(window.ui.modelSelect());

      seg.addEventListener("click", (e) => {
        const btn = e.target.closest("button[data-src]");
        if (!btn) return;
        source = btn.dataset.src;
        pdfFile = null;
        seg.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", b === btn ? "true" : "false"));
        buildSourceControl(sourceCtl);
      });

      layout.querySelector("#mmGo").addEventListener("click", () => generate(layout));
    },
    unmount() {
      source = "subject";
      pdfFile = null;
    },
  };
})();
