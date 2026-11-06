/* ============================================================
   Tab-visibility motion guard — perf-only, no visual/UX change.
   Pauses decorative CSS keyframe animations (ambient orbs, status
   LEDs, spinners, skeleton shimmer, reasoning pulses, voice-rec
   pulse, scan spinners, …) while the tab is hidden/backgrounded,
   then resumes them the instant it's visible again.

   Independent of and complementary to the `prefers-reduced-motion`
   handling already in styles.css: that one is a permanent user
   preference; this one is purely "nobody can see this right now,
   so don't spend GPU/CPU (and battery) animating it".
   Sets [data-tab-hidden] on <html>; styles.css does the pausing.
   ============================================================ */
(() => {
  "use strict";
  const root = document.documentElement;
  function sync() {
    root.toggleAttribute("data-tab-hidden", document.hidden);
  }
  sync();
  document.addEventListener("visibilitychange", sync, { passive: true });
})();
