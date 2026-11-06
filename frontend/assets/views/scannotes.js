/* ============================================================
   Scan Handwritten Notes view — window.Views.scannotes
   Camera-friendly photo picker (accept="image/*" capture) ->
   thumbnail preview -> POST /api/scannotes/extract (Gemini OCR)
   -> markdown via ui.md with Copy / Download .md actions.
   History kind: "scans". Uses window.api, window.ui, CSS kit.
   ============================================================ */
(() => {
  "use strict";
  const { el, clear, esc, spinner, empty, toast, md } = window.ui;

  const MAX_IMAGES = 6;
  const MAX_BYTES = 25 * 1024 * 1024;       // raw phone photo cap (it is shrunk before upload)
  // Vercel rejects request bodies above ~4.5 MB, so photos are shrunk in the browser and sent in
  // small sequential requests (<= PER_REQUEST photos and <= REQUEST_BYTES in total each).
  const PER_REQUEST = 3;
  const LONG_EDGE = 1600;
  const JPEG_QUALITY = 0.8;
  const TARGET_BYTES = 900 * 1024;
  const REQUEST_BYTES = 3.8 * 1000 * 1000;

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

  /* ---------------- client-side photo shrinking ---------------- */
  function canvasToBlob(canvas, quality) {
    return new Promise((resolve) => canvas.toBlob((b) => resolve(b), "image/jpeg", quality));
  }

  async function decode(file) {
    if (window.createImageBitmap) {
      try { return await createImageBitmap(file, { imageOrientation: "from-image" }); } catch (_) { /* fall through */ }
    }
    const url = URL.createObjectURL(file);
    try {
      return await new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => resolve(img);
        img.onerror = () => reject(new Error("decode"));
        img.src = url;
      });
    } finally { setTimeout(() => URL.revokeObjectURL(url), 1000); }
  }

  /* Downscale (long edge ~1600 px) and re-encode as JPEG (q 0.8, stepping down until <= ~900 KB).
     Returns a File. A photo that cannot be decoded is sent as-is when it is already small. */
  async function shrinkPhoto(file) {
    const small = file.size <= TARGET_BYTES && /^image\/jpe?g$/i.test(file.type || "");
    let src;
    try { src = await decode(file); } catch (_) {
      if (file.size <= TARGET_BYTES) return file;
      throw new Error(`${file.name || "Photo"} couldn’t be read in the browser — try a JPEG or PNG.`);
    }
    const w0 = src.width || src.naturalWidth, h0 = src.height || src.naturalHeight;
    if (small && Math.max(w0, h0) <= LONG_EDGE) { if (src.close) src.close(); return file; }
    let edge = Math.min(LONG_EDGE, Math.max(w0, h0));
    let quality = JPEG_QUALITY;
    let blob = null;
    for (let attempt = 0; attempt < 6; attempt++) {
      const k = edge / Math.max(w0, h0);
      const canvas = document.createElement("canvas");
      canvas.width = Math.max(1, Math.round(w0 * k));
      canvas.height = Math.max(1, Math.round(h0 * k));
      const ctx = canvas.getContext("2d");
      ctx.fillStyle = "#fff";                       // PNG transparency -> white paper
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(src, 0, 0, canvas.width, canvas.height);
      blob = await canvasToBlob(canvas, quality);
      if (!blob) break;
      if (blob.size <= TARGET_BYTES) break;
      if (quality > 0.55) quality -= 0.1; else edge = Math.round(edge * 0.8);
    }
    if (src.close) src.close();
    if (!blob) {
      if (file.size <= TARGET_BYTES) return file;
      throw new Error(`${file.name || "Photo"} couldn’t be shrunk in the browser — try a smaller photo.`);
    }
    const name = (file.name || "photo").replace(/\.[^.]+$/, "") + ".jpg";
    return new File([blob], name, { type: "image/jpeg" });
  }

  /* Group photos into requests of <= PER_REQUEST files and <= REQUEST_BYTES total. */
  function batchPhotos(list) {
    const batches = [];
    let cur = [], bytes = 0;
    for (const f of list) {
      if (cur.length && (cur.length >= PER_REQUEST || bytes + f.size > REQUEST_BYTES)) {
        batches.push(cur); cur = []; bytes = 0;
      }
      cur.push(f); bytes += f.size;
    }
    if (cur.length) batches.push(cur);
    return batches;
  }

  /* Each request numbers its pages from 1: shift them so the merged document reads Page 1..N. */
  function offsetPages(markdown, offset) {
    if (!offset) return markdown;
    return String(markdown || "").replace(/^## Page (\d+)/gm, (_m, n) => `## Page ${Number(n) + offset}`);
  }

  /* ---------------- picker screen ---------------- */
  function renderPicker() {
    clear(root);
    revokeThumbs();
    files = [];

    const panel = el(`
      <div class="panel stack" style="max-width:640px;margin:0 auto">
        <div class="between" style="align-items:flex-start;gap:.75rem">
          <div class="stack-sm" style="flex:1;min-width:0">
            <h3 style="font-family:var(--font-display);font-weight:700;letter-spacing:-.02em;overflow-wrap:anywhere">Scan handwritten notes</h3>
            <p class="muted">Photograph your notes — vision AI transcribes them into clean markdown.</p>
          </div>
          <button id="sn-hist" class="icon-btn ib-outline" type="button" aria-label="Open scan history" title="Open scan history">${window.ui.iconHtml("history")}</button>
        </div>

        <div class="field">
          <label class="label" for="sn-file">Photos</label>
          <div class="row row-wrap" style="align-items:center">
            <label class="btn" for="sn-file" style="cursor:pointer;flex:none">📷 Take / choose photos…</label>
            <span class="muted" id="sn-count" style="min-width:0">No photos selected</span>
          </div>
          <input id="sn-file" type="file" accept="image/*" multiple capture="environment" hidden aria-label="Take or choose photos of handwritten notes" />
          <div class="hint">Works best with clear photos and good lighting. Up to ${MAX_IMAGES} images (JPEG/PNG/WebP — HEIC isn’t supported). Photos are shrunk automatically before upload and sent ${PER_REQUEST} at a time.</div>
        </div>

        <div id="sn-thumbs" class="row row-wrap" style="gap:.5rem" role="list" aria-label="Selected photos"></div>

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
        onOpen: (saved) => renderResult(saved, true),
      }));

    function paintThumbs() {
      revokeThumbs();
      clear(thumbs);
      files.forEach((f, i) => {
        const url = URL.createObjectURL(f);
        objectUrls.push(url);
        const t = el(`
          <button type="button" role="listitem" title="Remove ${esc(f.name || "photo")}"
            aria-label="Remove ${esc(f.name || "photo")}"
            style="padding:0;border:1px solid rgba(127,127,127,.35);border-radius:10px;overflow:hidden;background:none;cursor:pointer;line-height:0;flex:none">
            <img src="${url}" alt="" style="width:84px;height:84px;object-fit:cover;display:block">
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
        if (f.size > MAX_BYTES) { toast(`${f.name || "Photo"} is over 25 MB — choose a smaller photo.`, "error"); continue; }
        files.push(f);
      }
      paintThumbs();
    });

    goBtn.addEventListener("click", async () => {
      if (!files.length) { toast("Choose at least one photo first.", "error"); return; }
      goBtn.disabled = true;
      const original = goBtn.textContent;
      goBtn.textContent = "Preparing photos…";
      const load = spinner(files.length > 1 ? "Reading your pages…" : "Reading your handwriting…");
      panel.append(load);
      const merged = { markdown: "", pages: 0, title: "", llm: null, failed_pages: [], partial: false };
      try {
        const prepared = [];
        for (const f of files) prepared.push(await shrinkPhoto(f));
        const batches = batchPhotos(prepared);
        for (let i = 0; i < batches.length; i++) {
          goBtn.textContent = batches.length > 1 ? `Extracting ${i + 1}/${batches.length}…` : "Extracting…";
          const fd = new FormData();
          batches[i].forEach((f) => fd.append("images", f, f.name || "photo.jpg"));
          let res;
          try {
            res = await api.postForm("/scannotes/extract", fd);
          } catch (e) {
            if (!merged.pages) throw e;                         // nothing paid for yet: normal error UI
            toast(`${e.message} — showing the pages scanned so far.`, "error");
            break;
          }
          const offset = merged.pages;
          merged.markdown += (merged.markdown ? "\n\n" : "") + offsetPages(res.markdown, offset);
          merged.pages += res.pages || batches[i].length;
          merged.title = merged.title || res.title || "";
          merged.llm = merged.llm || res.llm || null;
          (res.failed_pages || []).forEach((n) => merged.failed_pages.push(n + offset));
        }
        merged.partial = merged.failed_pages.length > 0;
        renderResult(merged);
        if (merged.partial) toast(`Page${merged.failed_pages.length === 1 ? "" : "s"} ${merged.failed_pages.join(", ")} couldn’t be read — scan ${merged.failed_pages.length === 1 ? "it" : "them"} again.`, "error");
      } catch (e) {
        toast(e.message, "error");
        goBtn.disabled = false;
        goBtn.textContent = original;
        load.replaceWith(window.ui.errorBlock(e, e.retryable ? () => goBtn.click() : null));
      }
    });
  }

  /* ---------------- result screen (fresh scan or saved item) ---------------- */
  function renderResult(res, fromHistory) {
    clear(root);
    revokeThumbs();
    files = [];
    const markdown = (res && res.markdown) || "";
    const pages = (res && res.pages) || 0;
    const title = (res && res.title) || "Scanned notes";

    const llmLine = window.ui.renderModelLabel(res && res.llm, { stored: !!fromHistory });   // "Model: …" (OCR) under the title
    const modelText = llmLine ? llmLine.textContent : "";
    const panel = el(`
      <div class="panel stack" style="max-width:760px;margin:0 auto">
        <div class="stack-sm">
          <h3 style="font-family:var(--font-display);font-weight:700;letter-spacing:-.02em;overflow-wrap:anywhere">${esc(title)}</h3>
          <span class="badge badge-accent" style="align-self:flex-start">${pages} page${pages === 1 ? "" : "s"}</span>
        </div>
        <div class="row row-wrap">
          <button id="sn-print" class="icon-btn ib-outline" type="button" aria-label="Print or save as PDF" title="Print or save as PDF">${window.ui.iconHtml("print")}</button>
          <button id="sn-dl" class="icon-btn ib-outline" type="button" aria-label="Download as Markdown (.md)" title="Download as Markdown (.md)">${window.ui.iconHtml("download")}</button>
          <button id="sn-copy" class="icon-btn ib-outline" type="button" aria-label="Copy notes" title="Copy notes">${window.ui.iconHtml("copy")}</button>
        </div>
        <div class="panel-soft" id="sn-md" style="overflow-x:auto;min-width:0"></div>
        <button id="sn-again" class="btn-accent btn-block" type="button">Scan more notes</button>
      </div>
    `);
    root.append(panel);
    if (llmLine) panel.querySelector(".stack-sm").append(llmLine);

    const mdHost = panel.querySelector("#sn-md");
    if (!markdown.trim()) {
      mdHost.append(empty("No text was extracted from these photos. Try clearer images.", "∅"));
      panel.querySelector("#sn-print").disabled = true;
      panel.querySelector("#sn-dl").disabled = true;
      panel.querySelector("#sn-copy").disabled = true;
    } else {
      const rendered = md(markdown);
      mdHost.append(rendered);
      panel.querySelector("#sn-print").addEventListener("click", () =>
        window.ui.printDoc(title, rendered.innerHTML, modelText));
      panel.querySelector("#sn-dl").addEventListener("click", () => downloadMd(title, markdown + (modelText ? "\n\n---\n*" + modelText + "*\n" : "")));
      panel.querySelector("#sn-copy").addEventListener("click", (e) =>
        window.ui.copy(markdown, e.currentTarget));
    }
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
    shrinkPhoto,                       // reused by the chat composer's 📷 button
    async mount(node) {
      root = node;
      renderPicker();
      const _art = window.ui.takeArtifact(["scans"]);       // Library deep-link
      if (_art) window.api.get(`/history/${_art.kind}/${encodeURIComponent(_art.id)}`).then((saved) => renderResult(saved, true)).catch(() => {});
    },
    unmount() {
      revokeThumbs();
      files = [];
      if (root) clear(root);
      root = null;
    },
  };
})();
