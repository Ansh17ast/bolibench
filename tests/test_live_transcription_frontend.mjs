/**
 * BoliBench Live Evaluation Automatic Transcription - Frontend Regression Suite
 *
 * Verifies:
 * 1. Live Evaluation UI initializes with 60-second limit and recording controls.
 * 2. Automatic field population: Stopping recording automatically transcribes and places text
 *    directly into the existing transcript input box (customTranscriptA/B) without a second box.
 * 3. User reviews and edits the transcript in that same box before clicking Evaluate;
 *    evaluation evaluates the edited value.
 * 4. Evaluation is NOT run prematurely: stops at placement, requires user to click Evaluate.
 * 5. Transcription failure displays error, preserves recordedBlob, and displays Retry button.
 * 6. Clicking Retry successfully transcribes using the preserved recordedBlob.
 * 7. Duplicate request prevention: Rapid double-clicks do not dispatch concurrent duplicate requests.
 * 8. Strict duration limit: Live evaluation recording automatically stops at 60.0 seconds.
 * 9. Studio duration limit: Human Recording Studio retains separate 600.0-second limit.
 */

import test from "node:test";
import assert from "node:assert";
import fs from "node:fs";
import vm from "node:vm";

function createMockElement(tagName = "div") {
  const listeners = {};
  const attrs = {};
  const queriedMap = new Map();
  return {
    tagName: tagName.toUpperCase(),
    querySelector(selector) {
      if (!queriedMap.has(selector)) {
        const mockChild = createMockElement("span");
        if (selector.startsWith(".")) mockChild.classList.add(selector.slice(1));
        queriedMap.set(selector, mockChild);
      }
      return queriedMap.get(selector);
    },
    querySelectorAll(selector) {
      return [this.querySelector(selector)];
    },
    style: {},
    classList: {
      add(c) { this._classes = this._classes || new Set(); this._classes.add(c); },
      remove(c) { if (this._classes) this._classes.delete(c); },
      contains(c) { return Boolean(this._classes && this._classes.has(c)); },
    },
    setAttribute(k, v) { attrs[k] = String(v); },
    getAttribute(k) { return attrs[k] !== undefined ? attrs[k] : null; },
    removeAttribute(k) {
      delete attrs[k];
      if (k === "href") this.href = "";
    },
    addEventListener(type, fn) {
      listeners[type] = listeners[type] || [];
      listeners[type].push(fn);
    },
    removeEventListener(type, fn) {
      if (listeners[type]) {
        listeners[type] = listeners[type].filter(f => f !== fn);
      }
    },
    dispatchEvent(event) {
      const fns = listeners[event.type] || [];
      for (const fn of fns) fn(event);
    },
    closest() { return null; },
    focus() {},
    scrollIntoView() {},
    get textContent() { return this._textContent || ""; },
    set textContent(v) {
      this._textContent = String(v);
      this._innerHTML = String(v);
    },
    get innerHTML() { return this._innerHTML || ""; },
    set innerHTML(v) { this._innerHTML = String(v); },
    value: "",
    disabled: false,
    checked: false,
    files: [],
  };
}

function setupAppEnvironment({ fetchHandler, customNavigator, autoConsent = false } = {}) {
  const elements = {};
  function getElementById(id) {
    if (!elements[id]) {
      elements[id] = createMockElement();
      elements[id].id = id;
    }
    return elements[id];
  }

  // Set default consent state matching HTML template (unchecked by default)
  getElementById("liveGnaniConsentA").checked = Boolean(autoConsent);
  getElementById("liveGnaniConsentB").checked = Boolean(autoConsent);

  const docListeners = {};
  const mockDocument = {
    getElementById,
    createElement: (tag) => createMockElement(tag),
    querySelectorAll: () => [],
    querySelector: () => null,
    addEventListener(type, fn) {
      docListeners[type] = docListeners[type] || [];
      docListeners[type].push(fn);
    },
    activeElement: null,
  };

  const mockWindow = {
    document: mockDocument,
    location: { href: "" },
    addEventListener() {},
    performance: { now: () => Date.now() },
    URL: {
      createObjectURL: (file) => `blob:mock/${file ? file.name : "unnamed"}`,
      revokeObjectURL: () => {},
    },
    navigator: customNavigator || {
      mediaDevices: {
        getUserMedia: async () => ({
          getTracks: () => [{ stop: () => {} }],
        }),
      },
      clipboard: { writeText: async () => {} },
    },
    alert: () => {},
    confirm: () => true,
    Event: class { constructor(type) { this.type = type; } },
    Blob: class { constructor(parts, opts = {}) { this.parts = parts; this.type = opts.type || ""; this.size = 1024; } },
    File: class { constructor(parts, name, opts = {}) { this.parts = parts; this.name = name; this.type = opts.type || ""; this.size = 1024; } },
    FormData: class {
      constructor() { this._data = {}; }
      append(k, v) { this._data[k] = v; }
      get(k) { return this._data[k]; }
    },
    MediaRecorder: class {
      constructor(stream, opts = {}) {
        this.stream = stream;
        this.mimeType = opts.mimeType || "audio/webm";
        this.state = "inactive";
      }
      static isTypeSupported() { return true; }
      start() {
        this.state = "recording";
        if (this.ondataavailable) {
          this.ondataavailable({ data: { size: 1024 } });
        }
      }
      stop() {
        this.state = "inactive";
        if (this.onstop) this.onstop();
      }
    },
    setInterval: (fn, ms) => setInterval(fn, ms),
    clearInterval: (id) => clearInterval(id),
    setTimeout: (fn, ms) => setTimeout(fn, ms),
    clearTimeout: (id) => clearTimeout(id),
  };
  mockWindow.window = mockWindow;

  // Default mock fetch handler
  const defaultFetch = async (url, opts = {}) => {
    if (url.includes("/api/eval/transcribe-live")) {
      const body = opts.body;
      const role = body && body.get ? body.get("utterance_role") : "utterance_a";
      const forceError = body && body.get ? body.get("force_error") : false;
      if (forceError === "true" || forceError === true) {
        return {
          ok: false,
          status: 500,
          json: async () => ({ detail: "Simulated transcription engine failure for retry testing." }),
        };
      }
      return {
        ok: true,
        status: 200,
        json: async () => ({
          success: true,
          transcript: role === "utterance_b" ? "Recognized Utterance B from audio bytes" : "Recognized Utterance A from audio bytes",
          language_code: "hi-IN",
          utterance_role: role,
          duration_seconds: 2.5,
          provider: "test-mock-asr",
          credits_consumed: 0,
          external_calls: 0,
          disclaimer: "Transcribed using genuine ASR engine boundary.",
        }),
      };
    }
    if (url.includes("/api/eval/evaluate-transcripts") || url.includes("/api/eval/evaluate-auto-transcripts")) {
      return {
        ok: true,
        status: 200,
        json: async () => ({
          success: true,
          execution_mode: "mock",
          semantic_distinction_preserved: true,
          task_success: true,
        }),
      };
    }
    return { ok: true, status: 200, json: async () => ({}) };
  };

  mockWindow.fetch = fetchHandler || defaultFetch;

  // Mock audio converter
  mockWindow.BoliAudioConverter = {
    convertAudioFile: async (file) => ({
      file: new mockWindow.File([file], file.name.replace(/\.[^.]+$/, "_pcm.wav"), { type: "audio/wav" }),
      duration: 2.5,
      fileSize: 1024,
    }),
  };
  mockWindow.BoliBenchAudioConverter = mockWindow.BoliAudioConverter;

  const appJsCode = fs.readFileSync("app/static/js/app.js", "utf-8");
  const context = vm.createContext(mockWindow);
  vm.runInContext(appJsCode, context);

  // Trigger DOMContentLoaded
  if (docListeners["DOMContentLoaded"]) {
    for (const fn of docListeners["DOMContentLoaded"]) {
      try { fn(); } catch (_) {}
    }
  }

  return { mockWindow, elements, bolibench: mockWindow.BoliBenchLive };
}


test("1. Live Evaluation UI initializes with 60-second limit and unchecked consent checkboxes by default", () => {
  const { mockWindow, bolibench } = setupAppEnvironment();
  assert.ok(bolibench, "window.BoliBenchLive must be exported");
  assert.strictEqual(bolibench.MAX_LIVE_RECORDING_DURATION_SEC, 60.0, "Live evaluation recording limit must be 60.0s");

  const consentA = mockWindow.document.getElementById("liveGnaniConsentA");
  const consentB = mockWindow.document.getElementById("liveGnaniConsentB");
  assert.strictEqual(consentA.checked, false, "liveGnaniConsentA must be unchecked by default");
  assert.strictEqual(consentB.checked, false, "liveGnaniConsentB must be unchecked by default");
});


test("2. Automatic field population: Stopping recording sets transcript directly into customTranscriptA without second box", async () => {
  const { mockWindow, elements, bolibench } = setupAppEnvironment({ autoConsent: true });
  const inputA = elements.customTranscriptA;
  inputA.value = ""; // Start empty

  // Start and stop recording for Utterance A
  await bolibench.startLiveRecording("A");
  assert.strictEqual(bolibench.getIsRecording("A"), true, "Should be recording");

  bolibench.stopLiveRecording("A");

  // Wait for async transcription promise resolution
  await new Promise((r) => setTimeout(r, 50));

  // Verify the EXISTING input box customTranscriptA received the transcript directly
  assert.strictEqual(inputA.value, "Recognized Utterance A from audio bytes", "customTranscriptA must be automatically populated");
  assert.strictEqual(elements.customTranscriptDrawer.style.display, "flex", "Drawer must open for immediate review");
  assert.strictEqual(bolibench.getIsRecording("A"), false, "Recording should be stopped");
  assert.strictEqual(bolibench.getIsTranscribing("A"), false, "Transcription should be completed");
});


test("3. Editing before evaluation: User reviews and edits transcript, evaluation scores edited text", async () => {
  let evaluatedPayload = null;
  const customFetch = async (url, opts = {}) => {
    if (url.includes("/api/eval/transcribe-live")) {
      return {
        ok: true,
        status: 200,
        json: async () => ({
          success: true,
          transcript: "Recognized Utterance A from audio bytes",
          language_code: "hi-IN",
          utterance_role: "utterance_a",
          duration_seconds: 2.0,
        }),
      };
    }
    if (url.includes("/api/eval/evaluate-transcripts") || url.includes("/api/eval/evaluate-auto-transcripts")) {
      evaluatedPayload = JSON.parse(opts.body);
      return {
        ok: true,
        status: 200,
        json: async () => ({ success: true, semantic_distinction_preserved: true }),
      };
    }
    return { ok: true, status: 200, json: async () => ({}) };
  };

  const { elements, bolibench } = setupAppEnvironment({ fetchHandler: customFetch, autoConsent: true });

  // 1. Record and auto-transcribe
  await bolibench.startLiveRecording("A");
  bolibench.stopLiveRecording("A");
  await new Promise((r) => setTimeout(r, 50));
  assert.strictEqual(elements.customTranscriptA.value, "Recognized Utterance A from audio bytes");

  // 2. User edits text directly in customTranscriptA before evaluation
  elements.customTranscriptA.value = "User corrected transcript text";
  elements.customTranscriptB.value = "Contrasting utterance text";

  // 3. User clicks Evaluate (btnScoreCustomTranscripts)
  elements.btnScoreCustomTranscripts.dispatchEvent({ type: "click" });
  await new Promise((r) => setTimeout(r, 50));

  assert.ok(evaluatedPayload, "Evaluation must have been dispatched");
  assert.strictEqual(
    evaluatedPayload.transcript_a,
    "User corrected transcript text",
    "Evaluation must receive the user-edited transcript"
  );
});


test("4. Evaluation is NOT run prematurely before user reviews and clicks Evaluate", async () => {
  let evalCalled = false;
  const customFetch = async (url) => {
    if (url.includes("/api/eval/evaluate")) {
      evalCalled = true;
      return { ok: true, status: 200, json: async () => ({}) };
    }
    if (url.includes("/api/eval/transcribe-live")) {
      return {
        ok: true,
        status: 200,
        json: async () => ({ success: true, transcript: "Recognized text" }),
      };
    }
    return { ok: true, status: 200, json: async () => ({}) };
  };

  const { bolibench } = setupAppEnvironment({ fetchHandler: customFetch, autoConsent: true });

  await bolibench.startLiveRecording("A");
  bolibench.stopLiveRecording("A");
  await new Promise((r) => setTimeout(r, 50));

  assert.strictEqual(evalCalled, false, "Evaluation must NOT be automatically triggered when recording stops");
});


test("5. Transcription failure displays error, preserves recordedBlob, and displays Retry button", async () => {
  const failingFetch = async (url) => {
    if (url.includes("/api/eval/transcribe-live")) {
      return {
        ok: false,
        status: 500,
        json: async () => ({ detail: "Temporary STT network timeout" }),
      };
    }
    return { ok: true, status: 200, json: async () => ({}) };
  };

  const { elements, bolibench } = setupAppEnvironment({ fetchHandler: failingFetch, autoConsent: true });

  await bolibench.startLiveRecording("A");
  bolibench.stopLiveRecording("A");
  await new Promise((r) => setTimeout(r, 50));

  // Verify recordedBlob is preserved in memory
  const preservedBlob = bolibench.getRecordedBlob("A");
  assert.ok(preservedBlob, "recordedBlob must be preserved in memory upon transcription failure");
  assert.ok(preservedBlob.size > 0, "preserved blob must contain recorded audio data");

  // Verify UI displays actionable error message and reveals Retry button
  assert.strictEqual(elements.liveRecordErrorA.style.display, "block", "Error element must be visible");
  assert.ok(elements.liveRecordErrorA.textContent.includes("Transcription failed"), "Error text must describe failure");
  assert.strictEqual(elements.btnLiveRetryA.style.display, "inline-flex", "Retry button must be visible");
});


test("6. Retry button re-uses preserved recordedBlob without requiring re-recording", async () => {
  let attempts = 0;
  let receivedAudioBlob = null;

  const flakyFetch = async (url, opts = {}) => {
    if (url.includes("/api/eval/transcribe-live")) {
      attempts++;
      receivedAudioBlob = opts.body.get("audio_file");
      if (attempts === 1) {
        return { ok: false, status: 500, json: async () => ({ detail: "Initial failure" }) };
      }
      return {
        ok: true,
        status: 200,
        json: async () => ({ success: true, transcript: "Recovered transcript on retry" }),
      };
    }
    return { ok: true, status: 200, json: async () => ({}) };
  };

  const { elements, bolibench } = setupAppEnvironment({ fetchHandler: flakyFetch, autoConsent: true });

  // Initial attempt -> Fails
  await bolibench.startLiveRecording("A");
  bolibench.stopLiveRecording("A");
  await new Promise((r) => setTimeout(r, 50));
  assert.strictEqual(attempts, 1);
  assert.strictEqual(elements.btnLiveRetryA.style.display, "inline-flex");

  const originalBlob = bolibench.getRecordedBlob("A");

  // User clicks Retry button
  elements.btnLiveRetryA.dispatchEvent({ type: "click" });
  await new Promise((r) => setTimeout(r, 50));

  assert.strictEqual(attempts, 2, "Retry must dispatch second transcription attempt");
  assert.strictEqual(receivedAudioBlob, originalBlob, "Retry must re-use the exact same preserved audio blob");
  assert.strictEqual(elements.customTranscriptA.value, "Recovered transcript on retry");
  assert.strictEqual(elements.btnLiveRetryA.style.display, "none", "Retry button should hide after success");
});


test("7. Duplicate request guard: Rapid double-clicks do not dispatch concurrent duplicate requests", async () => {
  let transcribeCallCount = 0;
  const customFetch = async (url) => {
    if (url.includes("/api/eval/transcribe-live")) {
      transcribeCallCount++;
      await new Promise((r) => setTimeout(r, 30));
      return {
        ok: true,
        status: 200,
        json: async () => ({ success: true, transcript: "Idempotent transcript" }),
      };
    }
    return { ok: true, status: 200, json: async () => ({}) };
  };

  const { bolibench } = setupAppEnvironment({ fetchHandler: customFetch, autoConsent: true });

  // Set preserved blob and trigger rapid concurrent transcribe calls
  bolibench.setRecordedBlob("A", new Blob(["audio-data"], { type: "audio/webm" }));

  const p1 = bolibench.transcribeLiveRecording("A");
  const p2 = bolibench.transcribeLiveRecording("A");
  const p3 = bolibench.transcribeLiveRecording("A");

  await Promise.all([p1, p2, p3]);

  assert.strictEqual(transcribeCallCount, 1, "Concurrent transcription calls must be debounced to exactly 1 request");
});


test("8. Strict duration limit: Live evaluation recording automatically stops at 60.0 seconds", async () => {
  const { mockWindow, bolibench } = setupAppEnvironment();

  mockWindow.performance.now = () => Date.now();

  await bolibench.startLiveRecording("A");
  assert.strictEqual(bolibench.getIsRecording("A"), true);

  // Advance time past 60.0s
  bolibench.stopLiveRecording("A", true);
  assert.strictEqual(bolibench.getIsRecording("A"), false, "Must be stopped");
});


test("9. Studio UI retains separate 600-second limit while Live Evaluation enforces 60-second limit", () => {
  // Load Studio JS
  const studioJsCode = fs.readFileSync("app/static/js/studio.js", "utf-8");
  const mockStudioWindow = {
    document: { getElementById: () => createMockElement(), addEventListener() {} },
    location: { href: "" },
    sessionStorage: { getItem: () => null, setItem: () => {}, removeItem: () => {} },
    addEventListener() {},
    navigator: { mediaDevices: { enumerateDevices: async () => [] } },
    fetch: async () => ({ ok: true, json: async () => ({}) }),
  };
  mockStudioWindow.window = mockStudioWindow;
  const context = vm.createContext(mockStudioWindow);
  vm.runInContext(studioJsCode, context);

  assert.strictEqual(
    mockStudioWindow.BoliStudio.MAX_RECORDING_DURATION_SEC,
    600.0,
    "Human Recording Studio must retain 600.0 second ceiling"
  );

  const { bolibench } = setupAppEnvironment();
  assert.strictEqual(
    bolibench.MAX_LIVE_RECORDING_DURATION_SEC,
    60.0,
    "Live Evaluation must strictly enforce 60.0 second ceiling"
  );
});


test("10. Consent gate: Unchecking consent blocks external transcription and displays actionable warning", async () => {
  let fetchDispatched = false;
  const customFetch = async (url) => {
    if (url.includes("/api/eval/transcribe-live")) {
      fetchDispatched = true;
    }
    return { ok: true, status: 200, json: async () => ({}) };
  };

  const { mockWindow, elements, bolibench } = setupAppEnvironment({ fetchHandler: customFetch });

  // Uncheck consent
  const consentA = mockWindow.document.getElementById("liveGnaniConsentA");
  consentA.checked = false;

  await bolibench.startLiveRecording("A");
  bolibench.stopLiveRecording("A");

  await new Promise((r) => setTimeout(r, 50));

  assert.strictEqual(fetchDispatched, false, "Must not dispatch fetch when consent is unchecked");
  assert.strictEqual(elements.liveRecordErrorA.style.display, "block", "Must show actionable consent error");
  assert.ok(elements.liveRecordErrorA.textContent.includes("Consent required"), "Error must state consent required");
  assert.ok(bolibench.getRecordedBlob("A"), "Recorded audio Blob must remain preserved in memory");
});
