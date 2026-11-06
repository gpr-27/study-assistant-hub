/* ============================================================
   Voice tutor — progressive enhancement for the Chat Assistant.
   No backend, no view of its own. When the #chat route is active
   it injects into #chatComposer:
     • #voiceMicBtn — dictate a question via SpeechRecognition
       (interim results stream into the composer input; the final
       text is LEFT in the input, never auto-sent).
     • window.Voice — speak()/stop()/setAuto(): per-message "Read aloud"
       buttons (chat.js) and the "Read replies aloud automatically" option
       use it (speechSynthesis). There is no speaker button in the composer.
   Fully feature-checked: the mic button is never created when
   SpeechRecognition is unsupported (e.g. Firefox), and the speaker
   button is never created without speechSynthesis. Navigating away
   cancels speech, stops recording and disconnects all observers.
   ============================================================ */
(() => {
  "use strict";

  const SR = window.SpeechRecognition || window.webkitSpeechRecognition || null;
  const TTS = "speechSynthesis" in window ? window.speechSynthesis : null;
  if (!SR && !TTS) return; // nothing this enhancement can offer

  const LANG = navigator.language || "en-US";

  // Ephemeral feedback (e.g. "mic blocked") via the app's toast, if present.
  function notify(message, kind) {
    try { window.ui && window.ui.toast && window.ui.toast(message, kind || "info"); } catch (_) {}
  }

  let recognition = null;  // active SpeechRecognition instance (null = idle)
  let observer = null;     // MutationObserver watching #chatThread
  let ttsOn = false;       // speaker toggle state
  let pollTimer = null;    // short poll waiting for #chatComposer to mount

  /* ---------------- text extraction ---------------- */
  // Crude markdown strip — good enough for speech: drop code blocks,
  // links keep their label, headings/emphasis markers removed.
  function stripMarkdown(text) {
    return String(text == null ? "" : text)
      .replace(/```[\s\S]*?(```|$)/g, " code block omitted. ")
      .replace(/`([^`]*)`/g, "$1")
      .replace(/!\[([^\]]*)\]\([^)]*\)/g, "$1")
      .replace(/\[([^\]]+)\]\([^)]*\)/g, "$1")
      .replace(/https?:\/\/\S+/g, " link ")
      .replace(/^#{1,6}\s*/gm, "")
      .replace(/^\s*[-*+]\s+/gm, "")
      .replace(/[*_~#>|]/g, "")
      .replace(/\s+/g, " ")
      .trim();
  }

  // Speakable text for one .msg.ai node: clone it, drop the footer
  // (Copy button, source chips, trace) and rendered code blocks,
  // then strip any residual markdown from the remaining text.
  function messageText(msg) {
    let clone;
    try { clone = msg.cloneNode(true); } catch (_) { return ""; }
    clone.querySelectorAll(".msg-footer, .msg-actions, .codeblock, pre").forEach((n) => n.remove());
    return stripMarkdown(clone.textContent);
  }

  /* ---------------- speech synthesis ---------------- */
  function speak(text, onEnd) {
    if (!TTS || !text) { if (onEnd) onEnd(); return; }
    try {
      const u = new SpeechSynthesisUtterance(text);
      u.lang = LANG;
      if (onEnd) { u.onend = onEnd; u.onerror = onEnd; }
      TTS.speak(u);
    } catch (_) { /* synthesis unavailable mid-session — stay silent */ }
  }

  function stopSpeaking() {
    if (TTS) { try { TTS.cancel(); } catch (_) {} }
  }

  function startObserver() {
    const thread = document.getElementById("chatThread");
    if (!thread || observer) return;
    observer = new MutationObserver((mutations) => {
      if (!ttsOn) return;
      // Collect AI messages added in this batch. A restored/resumed chat
      // appends its whole transcript in one synchronous batch — reading
      // an entire saved conversation aloud helps nobody, so only voice
      // small additions (live answers arrive one at a time).
      const added = [];
      for (const m of mutations) {
        m.addedNodes.forEach((node) => {
          if (node.nodeType !== 1) return;
          if (!node.classList.contains("msg-row") || !node.classList.contains("ai")) return;   // finished answers only (not the live bubble)
          added.push(node);
        });
      }
      if (!added.length || added.length > 2) return;
      added.forEach((node) => speak(messageText(node)));
    });
    observer.observe(thread, { childList: true });
  }

  function stopObserver() {
    if (observer) { try { observer.disconnect(); } catch (_) {} observer = null; }
  }

  /* ---------------- speech recognition (mic) ---------------- */
  function stopRecognition() {
    const rec = recognition;
    recognition = null;
    if (rec) { try { rec.stop(); } catch (_) {} }
    const btn = document.getElementById("voiceMicBtn");
    if (btn) { btn.classList.remove("is-rec"); btn.setAttribute("aria-pressed", "false"); }
  }

  function buildMicBtn(composer) {
    const btn = document.createElement("button");
    btn.id = "voiceMicBtn";
    btn.type = "button";
    btn.className = "icon-btn ib-outline";
    btn.title = "Speak your question (tap again to stop)";
    btn.setAttribute("aria-label", "Speak your question (tap again to stop)");
    btn.setAttribute("aria-pressed", "false");
    btn.append(window.ui.icon("mic"));

    btn.addEventListener("click", () => {
      if (recognition) { stopRecognition(); return; } // tap again to stop
      const input = composer.querySelector(".input");
      if (!input) return;

      let rec;
      try { rec = new SR(); } catch (_) { return; }
      rec.lang = LANG;
      rec.interimResults = true;
      // continuous=true keeps the session open across natural pauses. With
      // false, desktop Chrome ends recognition the instant it senses an
      // initial silence — the mic appears to "turn on then immediately off".
      // Phones capture the phrase fast enough to mask it; laptops don't. The
      // button is a toggle, so the user taps again to stop.
      rec.continuous = true;
      rec.maxAlternatives = 1;

      // Stream into the input on top of whatever was already typed.
      const base = input.value ? input.value.replace(/\s+$/, "") + " " : "";
      rec.onresult = (ev) => {
        let finals = "", interim = "";
        for (let i = 0; i < ev.results.length; i++) {
          const r = ev.results[i];
          if (r[0]) { if (r.isFinal) finals += r[0].transcript; else interim += r[0].transcript; }
        }
        // Final text is LEFT in the input — the user reviews and hits Send.
        input.value = (base + finals + interim).replace(/^\s+/, "");
      };
      // Surface ONLY errors the user can act on. "not-allowed"/"audio-capture"
      // are actionable (fix a permission, plug in a mic). "network" is NOT a
      // user-connection problem — the Web Speech API routes audio to Google's
      // speech backend, and "network" fires on transient/region/browser issues
      // even when the user's internet is fine; blaming their connection is wrong
      // and just noise. "aborted" is normal (tapping the mic to stop). So those
      // fail quietly and the user can simply type instead.
      rec.onerror = (ev) => {
        const err = (ev && ev.error) || "";
        if (err === "not-allowed" || err === "service-not-allowed") {
          notify("Microphone is blocked — allow mic access in your browser/system settings.", "error");
        } else if (err === "audio-capture") {
          notify("No microphone found.", "error");
        } else if (err === "no-speech") {
          notify("Didn't catch anything — tap the mic and speak again.");
        }
        // "network" / "aborted" / anything else: stop silently, no toast.
        stopRecognition();
      };
      rec.onend = () => {
        recognition = null;
        btn.classList.remove("is-rec");
        btn.setAttribute("aria-pressed", "false");
      };

      recognition = rec;
      btn.classList.add("is-rec");
      btn.setAttribute("aria-pressed", "true");
      try { rec.start(); } catch (_) { stopRecognition(); }
    });

    return btn;
  }

  /* ---------------- public API (used by the chat view) ---------------- */
  // Per-message "Read aloud" buttons live in the message footer (chat.js); the composer has no speaker button.
  // "Read replies aloud automatically" is a checkbox in the chat Options panel (default off).
  window.Voice = {
    supported: !!TTS,
    stop: stopSpeaking,
    speak: (text, onEnd) => { stopSpeaking(); speak(stripMarkdown(text), onEnd); },
    setAuto: (on) => { ttsOn = !!on; if (ttsOn) startObserver(); else { stopObserver(); stopSpeaking(); } },
  };

  /* ---------------- injection ---------------- */
  // Idempotent — guarded by element ids so repeated calls (poll ticks,
  // hashchange replays) never duplicate buttons.
  function inject() {
    const composer = document.getElementById("chatComposer");
    if (!composer) return false;
    const sendBtn = composer.querySelector("#chatSend");

    if (SR && !document.getElementById("voiceMicBtn")) {
      // Mic sits first: mic, speaker, then Send. Hidden entirely
      // (never created) when SpeechRecognition is unsupported.
      const mic = buildMicBtn(composer);
      const anchor = sendBtn;
      if (anchor) composer.insertBefore(mic, anchor);
      else composer.appendChild(mic);
    }
    return true;
  }

  /* ---------------- routing ---------------- */
  const isChat = () => /^#\/?chat$/.test(location.hash);

  function clearPoll() {
    if (pollTimer) { removeEventListener("sah:viewmounted", pollTimer); pollTimer = null; }
  }

  // The chat view mounts asynchronously after the router fires. Instead of polling, wait for
  // the router's "sah:viewmounted" event (dispatched after Views.chat.mount resolves).
  function schedulePoll() {
    clearPoll();
    if (inject()) return;
    pollTimer = () => { if (inject() || !isChat()) clearPoll(); };
    addEventListener("sah:viewmounted", pollTimer);
  }

  function teardown() {
    clearPoll();
    stopRecognition();
    stopObserver();
    stopSpeaking();
    ttsOn = false; // buttons are rebuilt on return — state must match
  }

  function onRoute() {
    if (isChat()) schedulePoll();
    else teardown();
  }

  addEventListener("hashchange", onRoute);
  if (document.readyState === "loading") addEventListener("DOMContentLoaded", onRoute);
  else onRoute();
})();
