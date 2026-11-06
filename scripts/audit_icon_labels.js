/* Dev-time audit: lists icon-only controls (an <svg> glyph or a 1-2 char symbol, no visible words) that lack an
   accessible name (aria-label / aria-labelledby / title / text). Paste into DevTools, or run through Playwright:
     await page.evaluate(fs.readFileSync("scripts/audit_icon_labels.js", "utf8"))
   Returns an array of {tag, id, class, html}; an empty array means every icon-only control is labelled.
   Also flags buttons that carry BOTH a native title and data-tip (double tooltip). */
(() => {
  const visibleText = (b) => {
    const w = document.createTreeWalker(b, NodeFilter.SHOW_TEXT); let n;
    while ((n = w.nextNode())) { if (n.nodeValue.trim().length > 2 && n.parentElement.getBoundingClientRect().width > 2) return true; }
    return false;
  };
  const bad = [];
  document.querySelectorAll("button, a[href], [role=button], input[type=button]").forEach((b) => {
    if (b.closest("[hidden]") && !b.matches("[hidden]")) { /* still audit hidden menus */ }
    const iconOnly = (b.querySelector("svg") && !visibleText(b)) || /^[\s -⯿\u{1F000}-\u{1FFFF}+×✕✎↻≡⚙☰−›‹]{1,3}$/u.test(b.textContent.trim());
    if (!iconOnly) return;
    const named = b.getAttribute("aria-label") || b.getAttribute("aria-labelledby") || b.getAttribute("title") || (b.textContent || "").replace(/[^\p{L}\p{N}]/gu, "").length > 0;
    const dup = b.hasAttribute("title") && b.dataset.tip;
    if (!named || dup) bad.push({ tag: b.tagName, id: b.id, class: b.className && b.className.baseVal === undefined ? b.className : "", issue: !named ? "no accessible name" : "title + data-tip", html: b.outerHTML.slice(0, 140) });
  });
  return bad;
})();
