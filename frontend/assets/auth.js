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
}
function showGate() {
  gate.classList.remove("is-hidden");
  gate.setAttribute("aria-hidden", "false");
}
function gateMessage(text) {
  actions.replaceChildren(el(`<div class="auth-loading">${text}</div>`));
}
function getGuestId() {
  try { return localStorage.getItem(GUEST_KEY) || null; } catch (_) { return null; }
}
function setGuestId(id) {
  try { id ? localStorage.setItem(GUEST_KEY, id) : localStorage.removeItem(GUEST_KEY); } catch (_) {}
}
function apiBase() {
  return (window.AppConfig && window.AppConfig.apiBase) || "/api";
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
  bar.querySelector(".cta-upgrade").addEventListener("click", (e) => { e.preventDefault(); onUpgrade && onUpgrade(); });
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

  let Clerk, darkTheme;
  try {
    const [clerkModule, themesModule] = await Promise.all([
      import(CLERK_CDN),
      import(THEMES_CDN)
    ]);
    Clerk = clerkModule.Clerk;
    darkTheme = themesModule.default.dark;
  } catch (e) {
    console.error("Failed to load the Clerk SDK or themes:", e);
    gateMessage("⚠ Couldn’t load the sign-in module. Check your connection and reload.");
    signalAuthReady();
    return;
  }

  const clerk = new Clerk(cfg.clerkPublishableKey);
  try {
    await clerk.load({
      afterSignOutUrl: window.location.origin,
      appearance: {
        baseTheme: darkTheme,
        variables: {
          colorPrimary: "#7c3aed", // Vibrant purple accent
          colorBackground: "#131520", // Dark card background
          colorText: "#ffffff", // White text
          colorTextSecondary: "#a7a7d0", // Soft secondary text
          colorInputBackground: "#1e1e2e", // Dark input background
          colorInputText: "#ffffff", // White input text
          colorBorder: "rgba(255, 255, 255, 0.12)", // Subtle border
          colorTextOnPrimaryBackground: "#ffffff",
        },
        elements: {
          card: {
            background: "#131520 !important",
            border: "1px solid rgba(255, 255, 255, 0.12) !important",
            borderRadius: "24px !important",
            boxShadow: "0 1px 1px rgba(255, 255, 255, 0.05), 0 16px 40px rgba(0, 0, 0, 0.6) !important",
            padding: "2.5rem !important",
            overflow: "visible !important", // Prevent clipping of headers
          },
          headerTitle: {
            fontFamily: '"Bricolage Grotesque", "Segoe UI", sans-serif !important',
            fontWeight: "800 !important",
            fontSize: "1.8rem !important",
            letterSpacing: "-0.02em !important",
            color: "#ffffff !important",
            marginTop: "0.5rem !important",
          },
          headerSubtitle: {
            fontFamily: '"Figtree", sans-serif !important',
            color: "#a7a7d0 !important",
            fontSize: "0.95rem !important",
          },
          socialButtonsBlockButton: {
            background: "rgba(255, 255, 255, 0.04) !important",
            border: "1px solid rgba(255, 255, 255, 0.1) !important",
            borderRadius: "12px !important",
            transition: "transform 0.2s, background 0.2s !important",
            height: "44px !important",
          },
          socialButtonsBlockButton__hover: {
            background: "rgba(255, 255, 255, 0.08) !important",
            transform: "translateY(-1px) !important",
          },
          socialButtonsBlockButtonText: {
            color: "#ffffff !important",
            fontFamily: '"Figtree", sans-serif !important',
            fontWeight: "600 !important",
          },
          dividerLine: {
            background: "rgba(255, 255, 255, 0.1) !important",
          },
          dividerText: {
            color: "#a7a7d0 !important",
            fontFamily: '"Figtree", sans-serif !important',
          },
          formFieldLabel: {
            fontFamily: '"Figtree", sans-serif !important',
            color: "#a7a7d0 !important",
            fontWeight: "600 !important",
            fontSize: "0.88rem !important",
            marginBottom: "0.4rem !important",
          },
          formFieldInput: {
            background: "rgba(255, 255, 255, 0.05) !important",
            border: "1px solid rgba(255, 255, 255, 0.1) !important",
            borderRadius: "12px !important",
            color: "#ffffff !important",
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
            color: "#a78bfa !important",
            fontWeight: "600 !important",
            fontFamily: '"Figtree", sans-serif !important',
          },
          footerActionText: {
            fontFamily: '"Figtree", sans-serif !important',
            color: "#a7a7d0 !important",
          }
        }
      }
    });
  } catch (e) {
    console.error("Clerk failed to initialize:", e);
    gateMessage("⚠ Sign-in is misconfigured — verify CLERK_PUBLISHABLE_KEY in your .env.");
    signalAuthReady();
    return;
  }

  // Expose a tiny auth surface so the API layer can attach the session token + guest id,
  // and so other scripts can ask "is this a guest?".
  window.Auth = {
    clerk,
    getToken: () => (clerk.session ? clerk.session.getToken() : Promise.resolve(null)),
    getGuestId,
    isGuest: () => !clerk.user && !!getGuestId(),
  };

  // Clerk has loaded and window.Auth is in place: clerk.session (if any) is restored, so
  // getToken() returns the real token. Release the API layer's first-paint requests now —
  // after this, authHeaders() will attach the token instead of racing ahead and 401ing.
  signalAuthReady();

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
    if (!clerk.user || !gid || migrating) return;
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

  async function render() {
    if (clerk.user) {
      // Signed in with Clerk: run any pending guest->account upgrade, then reveal.
      await maybeMigrate();
      // Mount the Clerk <UserButton> ONCE per user. Clerk fires its listener several times
      // per load / on token refresh; re-mounting onto the same node without unmounting leaks
      // the prior component, so guard on the mounted user id.
      if (mountedUserId !== clerk.user.id) {
        userSlot.classList.add("is-on");
        userSlot.replaceChildren();
        clerk.mountUserButton(userSlot, { afterSignOutUrl: window.location.origin });
        mountedUserId = clerk.user.id;
      }
      hideGuestCta();
      revealApp();
      return;
    }

    // Signed out: forget any previously-mounted Clerk button so a later sign-in re-mounts.
    mountedUserId = null;

    // Signed out: is there a valid guest session to restore?
    if (await guestSessionValid()) {
      mountGuestNav();
      showGuestCta(() => clerk.openSignUp({ afterSignUpUrl: window.location.href }));
      revealApp();
      return;
    }

    // No identity: show the gate (Sign in · Create account · Continue as guest).
    userSlot.classList.remove("is-on");
    userSlot.replaceChildren();
    hideGuestCta();
    showGate();
    const signIn = el(`<button class="btn-accent btn-lg" type="button">Sign in</button>`);
    const signUp = el(`<button class="btn btn-lg" type="button">Create account</button>`);
    signIn.addEventListener("click", () => clerk.openSignIn({ afterSignInUrl: window.location.href }));
    signUp.addEventListener("click", () => clerk.openSignUp({ afterSignUpUrl: window.location.href }));
    const children = [signIn, signUp];
    if (cfg.guestEnabled) {
      const guestBtn = el(`<button class="btn-ghost btn-lg" type="button">Continue as guest</button>`);
      guestBtn.addEventListener("click", startGuest);
      children.push(guestBtn);
    }
    actions.replaceChildren(...children);
    foot.textContent = cfg.appEnv ? `Secured by Clerk · ${cfg.appEnv}` : "Secured by Clerk";
  }

  clerk.addListener(render);
  render();
}

boot().catch((e) => {
  // Never trap the user on the loader: if anything unexpected fails, reveal the app and
  // release the API layer (it also has its own timeout backstop).
  console.error("Auth gate error:", e);
  try { window.__authReady && window.__authReady(); } catch (_) {}
  revealApp();
});
