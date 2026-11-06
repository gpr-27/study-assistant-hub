/* ============================================================
   Study Assistant Hub — auth gate (Clerk + anonymous guests, framework-agnostic).

   Uses Clerk's vanilla SDK (@clerk/clerk-js), NOT @clerk/react — this app is a
   plain-JS SPA. The gate offers THREE paths:
     • No CLERK_PUBLISHABLE_KEY configured  -> hide the gate, app is open (graceful, unchanged).
     • Key configured, signed OUT, no guest -> show the landing/sign-in gate
                                               (Sign in · Create account · Continue as guest).
     • Signed IN (Clerk) OR a live guest    -> hide the gate, reveal the full app.

   Identities, in priority order (must mirror backend/auth.py):
     1. Clerk user  — session token sent as Authorization: Bearer <jwt> (by app.js).
     2. Guest       — "guest_<uuid>" kept in localStorage, sent as X-Guest-Id (by app.js).

   Guest lifecycle:
     • Continue as guest -> POST /api/guest, store guestId, reveal the app.
     • Refresh/startup   -> GET /api/guest/me validates the stored guestId (restore or clear).
     • Upgrade           -> if a guest later signs in/up with Clerk, POST /api/guest/migrate
                            moves all their data to the Clerk account and clears the guestId.

   The publishable key and the guestEnabled flag come from the centralized config layer
   (window.AppConfig, from GET /api/config) — never hardcoded.
   ============================================================ */
const gate = document.getElementById("authGate");
const actions = document.getElementById("authActions");
const foot = document.getElementById("authFoot");
const userSlot = document.getElementById("userButton");
const keywarn = document.getElementById("keywarn");
const appShell = document.getElementById("appShell");

// Clerk's vanilla SDK, pinned to v5, loaded from a CDN (no bundler in this project).
const CLERK_CDN = "https://cdn.jsdelivr.net/npm/@clerk/clerk-js@5/+esm";
// Clerk themes CDN, used to load the dark theme.
const THEMES_CDN = "https://cdn.jsdelivr.net/npm/@clerk/themes@2/+esm";
// Must match GUEST_KEY in app.js (the API layer reads the same key).
const GUEST_KEY = "sah-guest-id";

function el(html) {
  const t = document.createElement("template");
  t.innerHTML = String(html).trim();
  return t.content.firstElementChild;
}
function revealApp() {
  gate.classList.add("is-hidden");
  gate.setAttribute("aria-hidden", "true");
  // The gate is done blocking the app — let keyboard/screen-reader users back into the
  // shell (it ships `inert` by default in index.html so it's never briefly reachable
  // behind the gate before this script decides which branch applies).
  appShell?.removeAttribute("inert");
}
function showGate() {
  gate.classList.remove("is-hidden");
  gate.setAttribute("aria-hidden", "false");
  appShell?.setAttribute("inert", "");
}
function gateMessage(text) {
  actions.replaceChildren(el(`<div class="auth-loading" role="status">${text}</div>`));
}
// Backend accepts guest_[A-Za-z0-9]{1,48}; anything else is stale/corrupt -> drop it before use.
const GUEST_RE = /^guest_[A-Za-z0-9]{1,48}$/;
function getGuestId() {
  try {
    const id = localStorage.getItem(GUEST_KEY);
    if (!id) return null;
    if (!GUEST_RE.test(id)) { localStorage.removeItem(GUEST_KEY); return null; }
    return id;
  } catch (_) { return null; }
}
function setGuestId(id) {
  try { id ? localStorage.setItem(GUEST_KEY, id) : localStorage.removeItem(GUEST_KEY); } catch (_) {}
}
function apiBase() {
  return (window.AppConfig && window.AppConfig.apiBase) || "/api";
}
function announceAuthed() {
  try { window.dispatchEvent(new CustomEvent("sah:authed")); } catch (_) {}
}
function notify(message, type) {
  if (window.ui && window.ui.toast) window.ui.toast(message, type);
}

// ---- Guest CTA banner (lives in #keywarn, reusing the existing .keywarn .glass style) ----
function showGuestCta(onUpgrade) {
  if (!keywarn || keywarn.querySelector(".guest-cta")) return;
  const bar = el(
    `<div class="keywarn glass guest-cta">👤 <span>You’re browsing as a <b>guest</b>. ` +
    `<a href="#" class="cta-upgrade" role="button">Create an account</a> to permanently save and sync your work.</span></div>`
  );
  const upgrade = bar.querySelector(".cta-upgrade");
  const goUpgrade = (e) => { e.preventDefault(); onUpgrade && onUpgrade(); };
  upgrade.addEventListener("click", goUpgrade);
  upgrade.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") goUpgrade(e);
  });
  keywarn.appendChild(bar);
}
function hideGuestCta() {
  if (!keywarn) return;
  const bar = keywarn.querySelector(".guest-cta");
  if (bar) bar.remove();
}

async function boot() {
  const cfg = await window.AppConfig.ready;

  // Tell the API layer (app.js) the session is settled, so authenticated first-paint
  // requests stop waiting and fire WITH the token. Must run on every exit path below —
  // once Clerk has loaded (or right away when there's no gate / Clerk can't load), the
  // token situation is final and further blocking would only stall the UI.
  const signalAuthReady = () => { try { window.__authReady && window.__authReady(); } catch (_) {} };

  // Graceful path: no Clerk key -> the app runs open, exactly as before. (Guests are only
  // meaningful as an alternative to a Clerk gate; with no gate everyone is already "local".)
  if (!cfg.clerkPublishableKey) {
    signalAuthReady();
    revealApp();
    return;
  }

  // Optimistic reveal: if a previous visit left evidence of a session (Clerk's own
  // `__client_uat` cookie is non-zero, or a stored guest id), show the app shell NOW instead of
  // waiting ~1 s for the Clerk SDK. The authoritative check below still runs and re-shows the
  // gate if the session turns out to be invalid; API calls stay token-gated server-side.
  // `clerkHint` (Clerk cookie non-zero) is the ONLY case that loads the SDK at boot: a likely-signed-in
  // user needs getToken() before the first authed request. Anonymous visitors and stored guests run
  // "lean" (no 800 KB SDK); Clerk loads on the first Sign in / Create account click (spinner), after an
  // idle modulepreload so that click feels instant.
  let clerkHint = false;
  (function optimisticReveal() {
    let hint = false;
    try {
      const m = document.cookie.match(/(?:^|;\s*)__client_uat(?:_[A-Za-z0-9]+)?=(\d+)/);
      clerkHint = !!(m && m[1] !== "0");
      hint = clerkHint || (!!getGuestId() && !!cfg.guestEnabled);
    } catch (_) {}
    if (hint) revealApp();
  })();

  let clerk = null, clerkPromise = null, darkTheme = null;
  // @clerk/themes (dark) is only needed in dark mode; fetch it lazily, not on every load.
  async function ensureDarkTheme() {
    if (darkTheme) return darkTheme;
    try {
      const themesModule = await import(THEMES_CDN);
      darkTheme = themesModule.default.dark;
    } catch (e) {
      console.warn("Couldn’t load the Clerk dark theme:", e);
    }
    return darkTheme;
  }
  // Match Clerk modals to the app's data-theme (light default / darkTheme). Hardcoding
  // darkTheme made Sign-in look wrong on light + Editorial. Re-applied on theme toggle.
  const isAppDark = () => document.documentElement.getAttribute("data-theme") === "dark";
  function clerkAppearance() {
    const dark = isAppDark();
    const sharedElements = {
      card: {
        borderRadius: "24px !important",
        padding: "2.5rem !important",
        overflow: "visible !important",
      },
      headerTitle: {
        fontFamily: '"Bricolage Grotesque", "Segoe UI", sans-serif !important',
        fontWeight: "800 !important",
        fontSize: "1.8rem !important",
        letterSpacing: "-0.02em !important",
        marginTop: "0.5rem !important",
      },
      headerSubtitle: {
        fontFamily: '"Figtree", sans-serif !important',
        fontSize: "0.95rem !important",
      },
      socialButtonsBlockButton: {
        borderRadius: "12px !important",
        transition: "transform 0.2s, background 0.2s !important",
        height: "44px !important",
      },
      socialButtonsBlockButtonText: {
        fontFamily: '"Figtree", sans-serif !important',
        fontWeight: "600 !important",
      },
      dividerText: {
        fontFamily: '"Figtree", sans-serif !important',
      },
      formFieldLabel: {
        fontFamily: '"Figtree", sans-serif !important',
        fontWeight: "600 !important",
        fontSize: "0.88rem !important",
        marginBottom: "0.4rem !important",
      },
      formFieldInput: {
        borderRadius: "12px !important",
        fontFamily: '"Figtree", sans-serif !important',
        padding: "0.75rem 1rem !important",
        height: "44px !important",
        transition: "border-color 0.2s, box-shadow 0.2s !important",
      },
      formFieldInput__focus: {
        borderColor: "#7c3aed !important",
        boxShadow: "0 0 0 3px rgba(124, 58, 237, 0.25) !important",
      },
      formButtonPrimary: {
        background: "linear-gradient(150deg, #7c3aed, #6366f1) !important",
        borderRadius: "999px !important",
        fontFamily: '"Bricolage Grotesque", sans-serif !important',
        fontWeight: "600 !important",
        fontSize: "0.95rem !important",
        boxShadow: "0 8px 22px rgba(124, 58, 237, 0.45) !important",
        border: "0 !important",
        height: "44px !important",
        transition: "transform 0.2s, box-shadow 0.2s !important",
        cursor: "pointer !important",
      },
      formButtonPrimary__hover: {
        transform: "translateY(-1px) !important",
        boxShadow: "0 12px 28px rgba(124, 58, 237, 0.55) !important",
      },
      footerActionLink: {
        color: dark ? "#a78bfa !important" : "#6d28d9 !important",
        fontWeight: "600 !important",
        fontFamily: '"Figtree", sans-serif !important',
      },
      footerActionText: {
        fontFamily: '"Figtree", sans-serif !important',
      },
    };
    if (dark) {
      return {
        baseTheme: darkTheme || undefined,
        variables: {
          colorPrimary: "#7c3aed",
          colorBackground: "#131520",
          colorText: "#ffffff",
          colorTextSecondary: "#a7a7d0",
          colorInputBackground: "#1e1e2e",
          colorInputText: "#ffffff",
          colorBorder: "rgba(255, 255, 255, 0.12)",
          colorTextOnPrimaryBackground: "#ffffff",
        },
        elements: {
          ...sharedElements,
          card: {
            ...sharedElements.card,
            background: "#131520 !important",
            border: "1px solid rgba(255, 255, 255, 0.12) !important",
            boxShadow: "0 1px 1px rgba(255, 255, 255, 0.05), 0 16px 40px rgba(0, 0, 0, 0.6) !important",
          },
          headerTitle: { ...sharedElements.headerTitle, color: "#ffffff !important" },
          headerSubtitle: { ...sharedElements.headerSubtitle, color: "#a7a7d0 !important" },
          socialButtonsBlockButton: {
            ...sharedElements.socialButtonsBlockButton,
            background: "rgba(255, 255, 255, 0.04) !important",
            border: "1px solid rgba(255, 255, 255, 0.1) !important",
          },
          socialButtonsBlockButton__hover: {
            background: "rgba(255, 255, 255, 0.08) !important",
            transform: "translateY(-1px) !important",
          },
          socialButtonsBlockButtonText: { ...sharedElements.socialButtonsBlockButtonText, color: "#ffffff !important" },
          dividerLine: { background: "rgba(255, 255, 255, 0.1) !important" },
          dividerText: { ...sharedElements.dividerText, color: "#a7a7d0 !important" },
          formFieldLabel: { ...sharedElements.formFieldLabel, color: "#a7a7d0 !important" },
          formFieldInput: {
            ...sharedElements.formFieldInput,
            background: "rgba(255, 255, 255, 0.05) !important",
            border: "1px solid rgba(255, 255, 255, 0.1) !important",
            color: "#ffffff !important",
          },
          footerActionText: { ...sharedElements.footerActionText, color: "#a7a7d0 !important" },
        },
      };
    }
    // Light / Editorial: Clerk's default (light) base — no darkTheme.
    return {
      baseTheme: undefined,
      variables: {
        colorPrimary: "#7c3aed",
        colorBackground: "#ffffff",
        colorText: "#1a1a2e",
        colorTextSecondary: "#5b5b7a",
        colorInputBackground: "#f6f6fb",
        colorInputText: "#1a1a2e",
        colorBorder: "rgba(26, 26, 46, 0.12)",
        colorTextOnPrimaryBackground: "#ffffff",
      },
      elements: {
        ...sharedElements,
        card: {
          ...sharedElements.card,
          background: "#ffffff !important",
          border: "1px solid rgba(26, 26, 46, 0.1) !important",
          boxShadow: "0 1px 2px rgba(26, 26, 46, 0.04), 0 16px 40px rgba(26, 26, 46, 0.12) !important",
        },
        headerTitle: { ...sharedElements.headerTitle, color: "#1a1a2e !important" },
        headerSubtitle: { ...sharedElements.headerSubtitle, color: "#5b5b7a !important" },
        socialButtonsBlockButton: {
          ...sharedElements.socialButtonsBlockButton,
          background: "rgba(26, 26, 46, 0.03) !important",
          border: "1px solid rgba(26, 26, 46, 0.12) !important",
        },
        socialButtonsBlockButton__hover: {
          background: "rgba(26, 26, 46, 0.06) !important",
          transform: "translateY(-1px) !important",
        },
        socialButtonsBlockButtonText: { ...sharedElements.socialButtonsBlockButtonText, color: "#1a1a2e !important" },
        dividerLine: { background: "rgba(26, 26, 46, 0.12) !important" },
        dividerText: { ...sharedElements.dividerText, color: "#5b5b7a !important" },
        formFieldLabel: { ...sharedElements.formFieldLabel, color: "#5b5b7a !important" },
        formFieldInput: {
          ...sharedElements.formFieldInput,
          background: "#f6f6fb !important",
          border: "1px solid rgba(26, 26, 46, 0.12) !important",
          color: "#1a1a2e !important",
        },
        footerActionText: { ...sharedElements.footerActionText, color: "#5b5b7a !important" },
      },
    };
  }

  // Auth surface for the API layer (app.js): token + guest id. Works with and without the Clerk SDK:
  // before it loads (lean mode) there is no Clerk session, so getToken() resolves null.
  window.Auth = {
    get clerk() { return clerk; },
    getToken: () => (clerk && clerk.session ? clerk.session.getToken() : Promise.resolve(null)),
    getGuestId,
    get userId() { return clerk && clerk.user ? clerk.user.id : null; },
    isGuest: () => !(clerk && clerk.user) && !!getGuestId(),
  };

  // Keep Clerk in sync when the user toggles light/dark (app.js sets data-theme).
  let lastClerkTheme = isAppDark() ? "dark" : "light";
  async function syncClerkTheme() {
    const next = isAppDark() ? "dark" : "light";
    if (next === lastClerkTheme) return;
    lastClerkTheme = next;
    if (next === "dark") await ensureDarkTheme();
    try {
      if (clerk && typeof clerk.__unstable__updateProps === "function") {
        clerk.__unstable__updateProps({ appearance: clerkAppearance() });
      }
    } catch (e) {
      console.warn("Couldn’t sync Clerk theme:", e);
    }
  }

  // Load + initialise the Clerk SDK exactly once (memoised; a failure may be retried by the next click).
  function initClerk() {
    if (clerkPromise) return clerkPromise;
    clerkPromise = (async () => {
      let mod;
      try {
        const wantDark = isAppDark();
        [mod] = await Promise.all([import(CLERK_CDN), wantDark ? ensureDarkTheme() : null]);
      } catch (e) {
        console.error("Failed to load the Clerk SDK:", e);
        throw Object.assign(new Error("sdk"), { code: "sdk" });
      }
      const c = new mod.Clerk(cfg.clerkPublishableKey);
      try {
        await c.load({ afterSignOutUrl: window.location.origin, appearance: clerkAppearance() });
      } catch (e) {
        console.error("Clerk failed to initialize:", e);
        throw Object.assign(new Error("init"), { code: "init" });
      }
      clerk = c;
      lastClerkTheme = isAppDark() ? "dark" : "light";
      new MutationObserver(syncClerkTheme).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
      c.addListener(render);      // fires render() for sign-in/out; migration + UserButton live there
      return c;
    })();
    clerkPromise.catch(() => { clerkPromise = null; });
    return clerkPromise;
  }

  // Open Clerk's sign-in / sign-up UI. First use loads the SDK: the clicked button shows a small
  // spinner (and is disabled) until the modal is ready.
  async function openAuthUI(kind, btn) {
    let restore = null;
    if (!clerk && btn) {
      const html = btn.innerHTML;
      btn.disabled = true;
      btn.setAttribute("aria-busy", "true");
      btn.replaceChildren(el(`<span class="spinner" aria-hidden="true"></span>`), document.createTextNode(" Loading…"));
      restore = () => { btn.disabled = false; btn.removeAttribute("aria-busy"); btn.innerHTML = html; };
    }
    try {
      await initClerk();
    } catch (e) {
      if (restore && btn.isConnected) restore();
      gateMessage(e && e.code === "init" ? "⚠ Sign-in is misconfigured — verify CLERK_PUBLISHABLE_KEY in your .env."
        : "⚠ Couldn’t load the sign-in module. Check your connection and try again.");
      setTimeout(() => { try { render(); } catch (_) {} }, 2500);
      return;
    }
    if (restore && btn.isConnected) restore();
    const href = window.location.href;
    if (kind === "signIn") clerk.openSignIn({ afterSignInUrl: href });
    else clerk.openSignUp({ afterSignUpUrl: href });
  }

  // After first paint + load, when the browser is idle: fetch (not execute) the SDK so the first click is instant.
  function prefetchClerk() {
    const c = navigator.connection;
    if (c && (c.saveData || /(^|-)2g$/.test(c.effectiveType || ""))) return;
    const go = () => {
      const idle = window.requestIdleCallback || ((f) => setTimeout(f, 1500));
      idle(() => {
        const urls = [CLERK_CDN];
        if (isAppDark()) urls.push(THEMES_CDN);
        urls.forEach((u) => {
          if (document.querySelector(`link[rel="modulepreload"][href="${u}"]`)) return;
          const l = document.createElement("link");
          l.rel = "modulepreload"; l.href = u; l.crossOrigin = "anonymous";
          document.head.appendChild(l);
        });
      }, { timeout: 6000 });
    };
    if (document.readyState === "complete") go(); else addEventListener("load", go, { once: true });
  }

  // Clerk has loaded and window.Auth is in place: clerk.session (if any) is restored, so
  // getToken() returns the real token. The API layer's first-paint requests are released after
  // the FIRST render() below, i.e. once a stored guest id has been validated (or cleared) —
  // so no request is ever sent with a stale guest id.

  let migrating = false;
  let mountedUserId = null;     // Clerk user id whose <UserButton> is currently mounted
  let guestValidated = null;    // memoized GET /guest/me result for this page load

  // Account upgrade: a guest who just signed in/up with Clerk -> move their data across.
  // CRITICAL: only clear the stored guest id when migration is actually SETTLED (HTTP 200 —
  // which covers both "moved" and "already migrated/nothing to move"). On a transient failure
  // (token not ready, 401/5xx, network error) we KEEP the guest id so the next page load
  // retries; clearing it there would orphan the user's data (the migration never ran).
  async function maybeMigrate() {
    const gid = getGuestId();
    if (!(clerk && clerk.user) || !gid || migrating) return;
    migrating = true;
    try {
      const token = await window.Auth.getToken();
      if (!token) return; // session token not ready yet — retry on the next auth event
      const res = await fetch(apiBase() + "/guest/migrate", {
        method: "POST",
        headers: { "Authorization": "Bearer " + token, "X-Guest-Id": gid },
      });
      if (res.ok) {
        const data = await res.json().catch(() => ({ total: 0 }));
        setGuestId(null); // settled — stop sending the guest id
        if (data && data.total) notify(`Welcome! Moved ${data.total} item(s) to your account.`, "ok");
      } else {
        // Transient failure — keep the guest id and let the next boot retry.
        console.error("Guest migration failed (will retry on next load):", res.status);
      }
    } catch (e) {
      console.error("Guest migration error (will retry on next load):", e);
    } finally {
      migrating = false;
    }
  }

  // "Continue as guest" -> create a guest, store it, reveal the app.
  async function startGuest() {
    gateMessage("Setting up your guest session…");
    try {
      const res = await fetch(apiBase() + "/guest", { method: "POST" });
      if (!res.ok) throw new Error("guest " + res.status);
      const guest = await res.json();
      setGuestId(guest.guestId);
      guestValidated = true; // just created — no need to re-validate via /guest/me
      await render(); // re-render: the guest branch reveals the app
    } catch (e) {
      console.error("Couldn’t start a guest session:", e);
      gateMessage("⚠ Couldn’t start a guest session. Try again, or sign in.");
    }
  }

  // Validate a stored guestId against the server (restoration on startup/refresh). The result
  // is memoized for the page load so the (multiple) Clerk listener firings don't each re-hit
  // GET /guest/me; startGuest()/Exit reset it.
  async function guestSessionValid() {
    const gid = getGuestId();
    if (!gid) { guestValidated = false; return false; }
    if (guestValidated === true) return true;
    try {
      const res = await fetch(apiBase() + "/guest/me", { headers: { "X-Guest-Id": gid } });
      if (res.ok) { guestValidated = true; return true; }
      if (res.status === 404 || res.status === 400) { setGuestId(null); guestValidated = false; } // expired/migrated/invalid
      return false;
    } catch (e) {
      console.error("Couldn’t restore guest session:", e);
      return false; // transient network error — keep the id, just can't restore right now
    }
  }

  // Nav menu for an active guest: a "Guest" tag + an End-session button (reusing existing
  // classes; the upgrade CTA lives in the banner). Deliberately does NOT mount Clerk's
  // <UserButton> — guests have no Clerk account/profile/billing to manage.
  function mountGuestNav() {
    userSlot.classList.add("is-on");
    const badge = el(
      `<span class="row guest-nav" style="gap:.4rem;align-items:center;flex:none">` +
      `<span class="tag">Guest</span>` +
      `<button class="btn-ghost guest-exit" type="button" title="End guest session">Exit</button>` +
      `</span>`
    );
    badge.querySelector(".guest-exit").addEventListener("click", () => {
      setGuestId(null);
      window.location.reload();
    });
    userSlot.replaceChildren(badge);
  }

  let lastIdentity = null;
  async function render() {
    // Drop any in-memory API cache when the identity changes (sign in/out, guest -> account).
    const user = clerk && clerk.user;
    const ident = user ? user.id : (getGuestId() || "anon");
    if (lastIdentity !== null && lastIdentity !== ident && window.api && window.api.clearCache) window.api.clearCache();
    // Signed out with no guest either: also drop the tab's identity-scoped bootstrap mirror (sessionStorage).
    if (ident === "anon" && window.api && window.api.clearCache) window.api.clearCache();
    lastIdentity = ident;
    if (user) {
      // Signed in with Clerk: run any pending guest->account upgrade, then reveal.
      await maybeMigrate();
      // Mount the Clerk <UserButton> ONCE per user. Clerk fires its listener several times
      // per load / on token refresh; re-mounting onto the same node without unmounting leaks
      // the prior component, so guard on the mounted user id.
      if (mountedUserId !== user.id) {
        userSlot.classList.add("is-on");
        userSlot.replaceChildren();
        clerk.mountUserButton(userSlot, { afterSignOutUrl: window.location.origin });
        mountedUserId = user.id;
      }
      hideGuestCta();
      revealApp();
      announceAuthed();
      return;
    }

    // Signed out: forget any previously-mounted Clerk button so a later sign-in re-mounts.
    mountedUserId = null;

    // Signed out: is there a valid guest session to restore?
    if (await guestSessionValid()) {
      mountGuestNav();
      showGuestCta(() => openAuthUI("signUp"));
      revealApp();
      announceAuthed();
      return;
    }

    // No identity: show the gate (Sign in · Create account · Continue as guest).
    // clerk.addListener(render) can fire this branch repeatedly before the user acts
    // (token refreshes, etc.) — only steal focus on the actual hidden->visible transition,
    // not on every redundant re-render while the gate is already up.
    const gateWasHidden = gate.classList.contains("is-hidden");
    userSlot.classList.remove("is-on");
    userSlot.replaceChildren();
    hideGuestCta();
    showGate();
    const signIn = el(`<button class="btn-accent btn-lg" type="button">Sign in</button>`);
    const signUp = el(`<button class="btn btn-lg" type="button">Create account</button>`);
    signIn.addEventListener("click", () => openAuthUI("signIn", signIn));
    signUp.addEventListener("click", () => openAuthUI("signUp", signUp));
    const children = [signIn, signUp];
    if (cfg.guestEnabled) {
      const guestBtn = el(`<button class="btn-ghost btn-lg" type="button">Continue as guest</button>`);
      guestBtn.addEventListener("click", startGuest);
      children.push(guestBtn);
    }
    actions.replaceChildren(...children);
    foot.textContent = cfg.appEnv ? `Secured by Clerk · ${cfg.appEnv}` : "Secured by Clerk";
    if (gateWasHidden) signIn.focus();
  }

  if (clerkHint) {            // likely signed in: SDK now (the head script already modulepreloaded it)
    try { await initClerk(); } catch (e) {
      gateMessage(e && e.code === "init" ? "⚠ Sign-in is misconfigured — verify CLERK_PUBLISHABLE_KEY in your .env."
        : "⚠ Couldn’t load the sign-in module. Check your connection and reload.");
      signalAuthReady();
      return;
    }
  }
  try { await render(); } finally { signalAuthReady(); }
  if (!clerk) prefetchClerk();
}

boot().catch((e) => {
  // Never trap the user on the loader: if anything unexpected fails, reveal the app and
  // release the API layer (it also has its own timeout backstop).
  console.error("Auth gate error:", e);
  try { window.__authReady && window.__authReady(); } catch (_) {}
  revealApp();
});
