/* ============================================================
   Scan Handwritten Notes view — window.Views.scannotes
   Camera-friendly photo picker (accept="image/*" capture) ->
   thumbnail preview -> POST /api/scannotes/extract (Gemini OCR)
   -> markdown via ui.md with Copy / Download .md actions.
   History kind: "scans". Uses window.api, window.ui, CSS kit.
   ============================================================ */
(() => {
  "use strict";
  const { el, clear, esc, spinner, toast, md } = window.ui;

  const MAX_IMAGES = 6;
  const MAX_BYTES = 8 * 1024 * 1024;

  let root = null;
  let files = [];        // File[] queued for the next extract
  let objectUrls = [];   // createObjectURL handles — revoked on repaint/unmount

  function revokeThumbs() {
    objectUrls.forEach((u) => { try { URL.revokeObjectURL(u); } catch (_) {} });
    objectUrls = [];
  }

  function isHeic(f) {
    const name = (f.name || "").toLowerCase();
    const type = (f.type || "").toLowerCase();
    return type === "image/heic" || type === "image/heif" ||
      name.endsWith(".heic") || name.endsWith(".heif");
  }

  /* ---------------- picker screen ---------------- */
  function renderPicker() {
    clear(root);
    revokeThumbs();
    files = [];

    const panel = el(`
      <div class="panel stack" style="max-width:640px;margin:0 auto">
        <div class="between" style="align-items:flex-start">
          <div class="stack-sm" style="flex:1">
            <h3 style="font-family:var(--font-display);font-weight:700;letter-spacing:-.02em">Scan handwritten notes</h3>
            <p class="muted">Photograph your notes — Gemini transcribes them into clean markdown.</p>
          </div>
          <button id="sn-hist" class="btn" type="button" style="flex:none;padding:.4rem .8rem;font-size:.85rem;gap:.35rem">≡ History</button>
        </div>

        <div class="field">
          <span class="label">Photos</span>
          <div class="row row-wrap">
            <label class="btn" for="sn-file" style="cursor:pointer">📷 Take / choose photos…</label>
            <span class="muted" id="sn-count">No photos selected</span>
          </div>
          <input id="sn-file" type="file" accept="image/*" multiple hidden />
          <div class="hint">Works best with clear photos and good lighting. Up to ${MAX_IMAGES} images, 8 MB each (JPEG/PNG/WebP — HEIC isn’t supported).</div>
        </div>

        <div id="sn-thumbs" class="row row-wrap" style="gap:.5rem"></div>

        <button id="sn-go" class="btn-accent btn-lg btn-block" type="button" disabled>Extract notes</button>
      </div>
    `);
    root.append(panel);

    const input = panel.querySelector("#sn-file");
    const thumbs = panel.querySelector("#sn-thumbs");
    const countEl = panel.querySelector("#sn-count");
    const goBtn = panel.querySelector("#sn-go");

    panel.querySelector("#sn-hist").addEventListener("click", (e) =>
      window.ui.openHistoryDrawer("scans", {
        title: "Saved scans",
        anchor: e.currentTarget,
        onOpen: (saved) => renderResult(saved),
      }));

    function paintThumbs() {
      revokeThumbs();
      clear(thumbs);
      files.forEach((f, i) => {
        const url = URL.createObjectURL(f);
        objectUrls.push(url);
        const t = el(`
          <button type="button" title="Remove ${esc(f.name || "photo")}"
            style="padding:0;border:1px solid rgba(127,127,127,.35);border-radius:10px;overflow:hidden;background:none;cursor:pointer;line-height:0">
            <img src="${url}" alt="${esc(f.name || "photo")}" style="width:84px;height:84px;object-fit:cover;display:block">
          </button>`);
        t.addEventListener("click", () => { files.splice(i, 1); paintThumbs(); });
        thumbs.append(t);
      });
      countEl.textContent = files.length
        ? `${files.length} photo${files.length === 1 ? "" : "s"} ready — tap a thumbnail to remove it`
        : "No photos selected";
      goBtn.disabled = !files.length;
    }

    input.addEventListener("change", () => {
      const chosen = Array.from(input.files || []);
      input.value = ""; // allow re-picking the same file
      if (!chosen.length) return;
      for (const f of chosen) {
        if (files.length >= MAX_IMAGES) { toast(`Up to ${MAX_IMAGES} photos per scan.`, "error"); break; }
        if (isHeic(f)) { toast(`${f.name || "Photo"}: HEIC isn’t supported — use JPEG or PNG.`, "error"); continue; }
        if (f.size > MAX_BYTES) { toast(`${f.name || "Photo"} is over 8 MB — resize it first.`, "error"); continue; }
        files.push(f);
      }
      paintThumbs();
    });

    goBtn.addEventListener("click", async () => {
      if (!files.length) { toast("Choose at least one photo first.", "error"); return; }
      const fd = new FormData();
      files.forEach((f) => fd.append("images", f, f.name || "photo.jpg"));
      goBtn.disabled = true;
      const original = goBtn.textContent;
      goBtn.textContent = "Extracting…";
      const load = spinner(files.length > 1 ? "Reading your pages…" : "Reading your handwriting…");
      panel.append(load);
      try {
        const res = await api.postForm("/scannotes/extract", fd);
        renderResult(res);
      } catch (e) {
        toast(e.message, "error");
        goBtn.disabled = false;
        goBtn.textContent = original;
        load.remove();
      }
    });
  }

  /* ---------------- result screen (fresh scan or saved item) ---------------- */
  function renderResult(res) {
    clear(root);
    revokeThumbs();
    files = [];
    const markdown = (res && res.markdown) || "";
    const pages = (res && res.pages) || 0;
    const title = (res && res.title) || "Scanned notes";

    const panel = el(`
      <div class="panel stack" style="max-width:760px;margin:0 auto">
        <div class="stack-sm">
          <h3 style="font-family:var(--font-display);font-weight:700;letter-spacing:-.02em;overflow-wrap:anywhere">${esc(title)}</h3>
          <span class="badge badge-accent" style="align-self:flex-start">${pages} page${pages === 1 ? "" : "s"}</span>
        </div>
        <div class="row row-wrap">
          <button id="sn-copy" class="btn" type="button">Copy</button>
          <button id="sn-dl" class="btn" type="button">Download .md</button>
        </div>
        <div class="panel-soft" id="sn-md" style="overflow-x:auto"></div>
        <button id="sn-again" class="btn-accent btn-block" type="button">Scan more notes</button>
      </div>
    `);
    root.append(panel);

    panel.querySelector("#sn-md").append(md(markdown));
    panel.querySelector("#sn-copy").addEventListener("click", (e) =>
      window.ui.copy(markdown, e.currentTarget));
    panel.querySelector("#sn-dl").addEventListener("click", () => downloadMd(title, markdown));
    panel.querySelector("#sn-again").addEventListener("click", renderPicker);
  }

  function downloadMd(title, markdown) {
    const blob = new Blob([markdown], { type: "text/markdown" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = (String(title).replace(/[^\w\s.()-]/g, "").trim() || "scanned-notes") + ".md";
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => { try { URL.revokeObjectURL(url); } catch (_) {} }, 1000);
  }

  /* ---------------- view registration ---------------- */
  window.Views.scannotes = {
    accent: "chat",
    title: "Scan Notes",
    async mount(node) {
      root = node;
      renderPicker();
    },
    unmount() {
      revokeThumbs();
      files = [];
      if (root) clear(root);
      root = null;
    },
  };
})();
