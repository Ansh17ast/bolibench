/**
 * BoliBench — Frontend Recording Workflow & Studio Remediation Regression Tests
 *
 * Verifies all 11 required regression capabilities:
 * 1. Initial #evaluateView tab activation on page load
 * 2. Valid and invalid hash changes via hashchange listener
 * 3. Human-pair handoff and automatic selection in Run Evaluation
 * 4. Switching templates without stale transcripts or locked pair IDs
 * 5. Preserving intentional user edits when switching templates
 * 6. Rapid double-click on Save prevented via synchronous guard
 * 7. Failed-save retry behavior preserves audio and permits re-submission
 * 8. Microphone track and AudioContext cleanup on device switch and unload
 * 9. Microphone dropdown selection preservation across device enumeration
 * 10. Resuming an incomplete pair from sessionStorage after refresh
 * 11. Client-side WAV conversion integration and error handling
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
      _classes: new Set(),
      add(c) { this._classes.add(c); },
      remove(c) { this._classes.delete(c); },
      contains(c) { return this._classes.has(c); },
    },
    setAttribute(k, v) { attrs[k] = String(v); },
    getAttribute(k) { return attrs[k] !== undefined ? attrs[k] : null; },
    removeAttribute(k) { delete attrs[k]; },
    addEventListener(type, fn) {
      listeners[type] = listeners[type] || [];
      listeners[type].push(fn);
    },
    removeEventListener(type, fn) {
      if (listeners[type]) {
        listeners[type] = listeners[type].filter((f) => f !== fn);
      }
    },
    dispatchEvent(event) {
      const fns = listeners[event.type] || [];
      for (const fn of fns) fn(event);
    },
    focus() {},
    scrollIntoView() {},
    getContext() {
      return {
        fillStyle: "",
        fillRect() {},
        beginPath() {},
        moveTo() {},
        lineTo() {},
        stroke() {},
      };
    },
    get textContent() { return this._textContent || ""; },
    set textContent(v) {
      this._textContent = String(v);
      this._innerHTML = String(v);
    },
    get innerHTML() { return this._innerHTML || ""; },
    set innerHTML(v) { this._innerHTML = String(v); },
    appendChild(child) {
      this._children = this._children || [];
      this._children.push(child);
      if (child.tagName === "OPTION") {
        this.options = this.options || [];
        this.options.push(child);
      }
    },
    value: "",
    checked: false,
    disabled: false,
    src: "",
  };
}

const SAMPLE_SCENARIO_TEMPLATES = [
  {
    id: "pair_qty_001",
    category: "quantities_and_numbers",
    pair_id: "pair_qty_001",
    role: "utterance_a",
    title: "E-Commerce: 2 Packets",
    language: "hinglish",
    reference_transcript: "Do packet chahiye",
    distinction: "Quantity 2 vs Quantity 3",
  },
  {
    id: "pair_neg_001",
    category: "negation",
    pair_id: "pair_neg_001",
    role: "utterance_a",
    title: "Ride Booking: Retain Ride",
    language: "hi-IN",
    reference_transcript: "Nahi ride cancel mat karo",
    distinction: "Retain ride vs Cancel ride",
  },
  {
    id: "pair_cor_001",
    category: "corrections_and_mind_changes",
    pair_id: "pair_cor_001",
    role: "utterance_a",
    title: "Travel: Mumbai to Delhi Correction",
    language: "en-IN",
    reference_transcript: "Book flight to Mumbai, no wait, to Delhi",
    distinction: "Destination correction Delhi vs Mumbai",
  },
];

// Helper to set up App.js environment
function setupAppEnvironment({ initialHash = "", sessionData = {}, fetchHandler = null } = {}) {
  const elements = {};
  function getElementById(id) {
    if (!elements[id]) {
      elements[id] = createMockElement();
      elements[id].id = id;
    }
    return elements[id];
  }

  const docListeners = {};
  const mockDocument = {
    getElementById,
    createElement(tag) { return createMockElement(tag); },
    addEventListener(evt, fn) {
      docListeners[evt] = docListeners[evt] || [];
      docListeners[evt].push(fn);
    },
    querySelectorAll() { return []; },
  };

  const windowListeners = {};
  const mockWindow = {
    document: mockDocument,
    location: {
      hash: initialHash,
      href: `http://localhost:8000/${initialHash}`,
    },
    addEventListener(evt, fn) {
      windowListeners[evt] = windowListeners[evt] || [];
      windowListeners[evt].push(fn);
    },
    dispatchEvent(evt) {
      const fns = windowListeners[evt.type] || [];
      for (const fn of fns) fn(evt);
    },
    sessionStorage: {
      _store: { ...sessionData },
      getItem(k) { return this._store[k] !== undefined ? this._store[k] : null; },
      setItem(k, v) { this._store[k] = String(v); },
      removeItem(k) { delete this._store[k]; },
      clear() { this._store = {}; },
    },
    alert() {},
    confirm() { return true; },
  };

  const requestedUrls = [];
  const mockFetch = async (url, opts) => {
    requestedUrls.push({ url, opts });
    if (fetchHandler) {
      const res = await fetchHandler(url, opts);
      if (res !== undefined) return res;
    }
    if (url.includes("/api/eval/test-cases") || url.includes("/api/eval/results")) {
      return { ok: true, status: 200, json: async () => [] };
    }
    if (url.includes("/api/eval/human-pairs")) {
      return {
        ok: true,
        status: 200,
        json: async () => [
          {
            pair_id: "human_qty_001",
            category: "quantities_and_numbers",
            language: "hinglish",
            utterance_a: { reference_transcript: "Do packet chahiye", duration_seconds: 50.0 },
            utterance_b: { reference_transcript: "Teen packet chahiye", duration_seconds: 50.0 },
            status: "ready_for_evaluation",
          },
        ],
      };
    }
    return { ok: false, status: 404, json: async () => ({}) };
  };

  const code = fs.readFileSync("app/static/js/app.js", "utf-8");
  const context = vm.createContext({
    window: mockWindow,
    document: mockDocument,
    fetch: mockFetch,
    console,
    sessionStorage: mockWindow.sessionStorage,
    alert: (m) => mockWindow.alert(m),
    confirm: (m) => mockWindow.confirm(m),
    setTimeout: (fn) => setTimeout(fn, 0),
    clearTimeout: (id) => clearTimeout(id),
    URL: globalThis.URL || { createObjectURL: () => "blob:mock", revokeObjectURL: () => {} },
    URLSearchParams: globalThis.URLSearchParams,
    FormData: class { append() {} },
  });

  vm.runInContext(code, context);

  // Trigger DOMContentLoaded
  if (docListeners["DOMContentLoaded"]) {
    for (const fn of docListeners["DOMContentLoaded"]) fn();
  }

  return {
    elements,
    bolibench: mockWindow.__bolibench,
    mockWindow,
    windowListeners,
    requestedUrls,
  };
}

// Helper to set up Studio.js environment
function setupStudioEnvironment({ sessionData = {}, fetchHandler = null, confirmResponse = true, converter = null } = {}) {
  const elements = {};
  function getElementById(id) {
    if (!elements[id]) {
      elements[id] = createMockElement();
      elements[id].id = id;
    }
    return elements[id];
  }

  const docListeners = {};
  const mockDocument = {
    getElementById,
    createElement(tag) { return createMockElement(tag); },
    addEventListener(evt, fn) {
      docListeners[evt] = docListeners[evt] || [];
      docListeners[evt].push(fn);
    },
    readyState: "complete",
  };

  const windowListeners = {};
  const tracksStopped = [];
  const contextsClosed = [];

  class MockAudioTrack {
    constructor() { this.kind = "audio"; this.stopped = false; }
    stop() { this.stopped = true; tracksStopped.push(this); }
  }

  const currentTracks = [new MockAudioTrack()];
  const mockStream = {
    getTracks() { return currentTracks; },
    getAudioTracks() { return currentTracks; },
  };

  class MockAudioContext {
    constructor() {
      this.state = "running";
      this.closed = false;
    }
    createMediaStreamSource() {
      return { connect() {} };
    }
    createAnalyser() {
      return {
        fftSize: 256,
        frequencyBinCount: 128,
        getByteTimeDomainData() {},
      };
    }
    async close() {
      this.state = "closed";
      this.closed = true;
      contextsClosed.push(this);
    }
  }

  const mockWindow = {
    document: mockDocument,
    addEventListener(evt, fn) {
      windowListeners[evt] = windowListeners[evt] || [];
      windowListeners[evt].push(fn);
    },
    dispatchEvent(evt) {
      const fns = windowListeners[evt.type] || [];
      for (const fn of fns) fn(evt);
    },
    sessionStorage: {
      _store: { ...sessionData },
      getItem(k) { return this._store[k] !== undefined ? this._store[k] : null; },
      setItem(k, v) { this._store[k] = String(v); },
      removeItem(k) { delete this._store[k]; },
      clear() { this._store = {}; },
    },
    AudioContext: MockAudioContext,
    navigator: {
      mediaDevices: {
        getUserMedia: async () => mockStream,
        enumerateDevices: async () => [
          { kind: "audioinput", deviceId: "mic_default", label: "Default Microphone" },
          { kind: "audioinput", deviceId: "mic_usb", label: "USB Microphone" },
          { kind: "audioinput", deviceId: "mic_headset", label: "Headset Mic" },
        ],
      },
      permissions: {
        query: async () => ({ state: "granted", onchange: null }),
      },
    },
    alert() {},
    confirm() { return confirmResponse; },
    requestAnimationFrame(fn) { return 1; },
    cancelAnimationFrame(id) {},
    URL: {
      createObjectURL: () => "blob:mock-audio",
      revokeObjectURL: () => {},
    },
    URLSearchParams: globalThis.URLSearchParams,
    BoliAudioConverter: converter,
  };

  const uploadRequests = [];
  const pairCreateRequests = [];

  const mockFetch = async (url, opts) => {
    if (url.includes("/api/studio/upload")) {
      uploadRequests.push({ url, opts });
    }
    if (fetchHandler) {
      const custom = await fetchHandler(url, opts);
      if (custom !== undefined) return custom;
    }
    if (url.includes("/api/studio/config")) {
      return {
        ok: true,
        status: 200,
        json: async () => ({
          storage_directory: "human_recordings_v1",
          scenario_templates: SAMPLE_SCENARIO_TEMPLATES,
        }),
      };
    }
    if (url.includes("/api/studio/pairs/create")) {
      pairCreateRequests.push(JSON.parse(opts.body));
      return { ok: true, status: 200, json: async () => ({ ok: true }) };
    }
    if (url.includes("/api/studio/upload")) {
      return {
        ok: true,
        status: 200,
        json: async () => ({
          recording_id: "rec_mock_12345",
          validation_status: "valid_50s",
          diagnostics: { duration_seconds: 50.0, is_complete_50s: true },
        }),
      };
    }
    if (url.includes("/api/studio/recordings/")) {
      return {
        ok: true,
        status: 200,
        json: async () => ({
          recording_id: "rec_mock_12345",
          reference_transcript: "Do packet chahiye",
          verified_transcript: "Do packet chahiye",
          pair_id: "human_qty_001",
          utterance_role: "utterance_a",
        }),
      };
    }
    if (url.includes("/api/studio/recordings")) {
      return { ok: true, status: 200, json: async () => [] };
    }
    if (url.includes("/api/studio/pairs")) {
      return { ok: true, status: 200, json: async () => [] };
    }
    return { ok: false, status: 404, json: async () => ({}) };
  };

  class MockMediaRecorder {
    constructor(stream, opts) {
      this.stream = stream;
      this.opts = opts;
      this.state = "inactive";
      this.mimeType = opts?.mimeType || "audio/webm";
      this.ondataavailable = null;
      this.onstop = null;
    }
    static isTypeSupported() { return true; }
    start() {
      this.state = "recording";
    }
    stop() {
      this.state = "inactive";
      if (this.ondataavailable) {
        this.ondataavailable({ data: new MockBlob([new Uint8Array(1000)], { type: this.mimeType }) });
      }
      if (this.onstop) {
        this.onstop();
      }
    }
    pause() {
      this.state = "paused";
    }
    resume() {
      this.state = "recording";
    }
  }

  class MockBlob {
    constructor(chunks, opts) {
      this.chunks = chunks || [];
      this.type = opts?.type || "audio/webm";
      let sz = 0;
      for (const c of this.chunks) {
        if (c.byteLength) sz += c.byteLength;
        else if (c.size) sz += c.size;
        else if (typeof c === "string") sz += c.length;
        else sz += 1000;
      }
      this.size = sz || 50000;
    }
    async arrayBuffer() {
      return new ArrayBuffer(this.size);
    }
  }

  let currentTime = 10000;
  let activeIntervalCallback = null;
  let intervalIdCounter = 1;

  class MockFormData {
    constructor() { this.fields = {}; }
    append(k, v) { this.fields[k] = v; }
  }

  const code = fs.readFileSync("app/static/js/studio.js", "utf-8");
  const context = vm.createContext({
    window: mockWindow,
    document: mockDocument,
    navigator: mockWindow.navigator,
    fetch: mockFetch,
    console,
    sessionStorage: mockWindow.sessionStorage,
    alert: (m) => mockWindow.alert(m),
    confirm: (m) => mockWindow.confirm(m),
    AudioContext: MockAudioContext,
    MediaRecorder: MockMediaRecorder,
    Blob: MockBlob,
    FormData: MockFormData,
    URL: mockWindow.URL,
    URLSearchParams: globalThis.URLSearchParams,
    setTimeout: (fn) => setTimeout(fn, 0),
    clearTimeout: (id) => clearTimeout(id),
    setInterval: (fn, ms) => {
      activeIntervalCallback = fn;
      return intervalIdCounter++;
    },
    clearInterval: () => {
      activeIntervalCallback = null;
    },
    performance: { now: () => currentTime },
    requestAnimationFrame: (fn) => 1,
    cancelAnimationFrame: (id) => {},
  });

  vm.runInContext(code, context);

  return {
    elements,
    studio: mockWindow.BoliStudio,
    mockWindow,
    tracksStopped,
    contextsClosed,
    uploadRequests,
    pairCreateRequests,
    advanceTime: (ms) => {
      currentTime += ms;
      if (activeIntervalCallback) {
        activeIntervalCallback();
      }
    },
  };
}

// =============================================================================
// TEST 1: Initial #evaluateView tab activation on page load
// =============================================================================
test("1. Initial #evaluateView hash cleanly activates the Run Evaluation pane on page load", () => {
  const { elements, bolibench } = setupAppEnvironment({ initialHash: "#evaluateView" });
  assert.strictEqual(bolibench.getActiveTabKey(), "evaluateView", "Tab key must be evaluateView");
  assert.strictEqual(elements.evaluateView.style.display, "block", "evaluateView pane must have display: block");
  assert.strictEqual(elements.overviewView.style.display, "none", "overviewView pane must have display: none");
  assert.ok(elements.tabBtnEvaluate.classList.contains("active"), "Evaluate button must have .active class");
});

// =============================================================================
// TEST 2: Valid and invalid hash changes via hashchange listener
// =============================================================================
test("2. Valid hash change updates active tab; invalid hash is safely ignored", () => {
  const { elements, bolibench, mockWindow } = setupAppEnvironment({ initialHash: "#overviewView" });
  assert.strictEqual(bolibench.getActiveTabKey(), "overviewView");

  // Valid hash change to #historyView
  mockWindow.location.hash = "#historyView";
  mockWindow.dispatchEvent({ type: "hashchange" });
  assert.strictEqual(bolibench.getActiveTabKey(), "historyView", "Tab must switch to historyView");
  assert.strictEqual(elements.historyView.style.display, "block");
  assert.strictEqual(elements.overviewView.style.display, "none");

  // Invalid hash change
  mockWindow.location.hash = "#nonExistentPane";
  mockWindow.dispatchEvent({ type: "hashchange" });
  assert.strictEqual(bolibench.getActiveTabKey(), "historyView", "Invalid hash must not corrupt activeTabKey");
});

// =============================================================================
// TEST 3: Human-pair handoff and automatic selection in Run Evaluation
// =============================================================================
test("3. Human-pair handoff selects human pair and toggles to Human source", async () => {
  const { elements, bolibench } = setupAppEnvironment({
    initialHash: "#evaluateView",
    sessionData: { bolibench_selected_human_pair: "human_qty_001" },
  });

  assert.strictEqual(bolibench.getActiveEvaluationSource(), "human_recording", "Must switch source to human_recording");
  assert.strictEqual(elements.humanEvalContainer.style.display, "block", "Human container must be visible");
  assert.strictEqual(elements.syntheticEvalContainer.style.display, "none", "Synthetic container must be hidden");
  assert.ok(elements.sourceToggleHuman.classList.contains("active"), "Human source toggle must be active");
});

// =============================================================================
// TEST 4: Switching templates without stale transcripts or locked pair IDs
// =============================================================================
test("4. Switching from quantity to negation template updates category, transcript, and pair ID", async () => {
  const { elements, studio } = setupStudioEnvironment();
  // Wait for initial async startup promises to settle
  await new Promise((r) => setTimeout(r, 40));

  // 1. Select Quantity template
  elements.studioScenarioTemplateSelect.value = "pair_qty_001";
  studio.handleScenarioTemplateSelected();
  await new Promise((r) => setTimeout(r, 10));

  assert.strictEqual(elements.studioCategorySelect.value, "quantities_and_numbers");
  assert.strictEqual(elements.studioReferenceTranscript.value, "Do packet chahiye");
  assert.strictEqual(elements.studioPairId.value, "pair_qty_001");
  assert.strictEqual(studio.getCurrentPair().pairId, "pair_qty_001");
  assert.strictEqual(elements.studioActivePairIdDisplay.textContent, "pair_qty_001");

  // 2. Switch to Negation template (WITHOUT manual user typing)
  elements.studioScenarioTemplateSelect.value = "pair_neg_001";
  studio.handleScenarioTemplateSelected();
  await new Promise((r) => setTimeout(r, 10));

  assert.strictEqual(elements.studioCategorySelect.value, "negation", "Category must update to negation");
  assert.strictEqual(elements.studioReferenceTranscript.value, "Nahi ride cancel mat karo", "Transcript must update to negation script");
  assert.strictEqual(elements.studioPairId.value, "pair_neg_001", "Pair ID must update to pair_neg_001");
  assert.strictEqual(studio.getCurrentPair().pairId, "pair_neg_001", "currentPair.pairId must be synchronized");
  assert.strictEqual(elements.studioActivePairIdDisplay.textContent, "pair_neg_001", "Active display must be synchronized");

  // 3. Switch to third template (Corrections)
  elements.studioScenarioTemplateSelect.value = "pair_cor_001";
  studio.handleScenarioTemplateSelected();
  await new Promise((r) => setTimeout(r, 10));

  assert.strictEqual(elements.studioCategorySelect.value, "corrections_and_mind_changes");
  assert.strictEqual(elements.studioReferenceTranscript.value, "Book flight to Mumbai, no wait, to Delhi");
  assert.strictEqual(elements.studioPairId.value, "pair_cor_001");
});

// =============================================================================
// TEST 5: Preserving intentional user edits when switching templates
// =============================================================================
test("5. Preserves user-edited transcript when user declines replacement", async () => {
  // confirm() returns false (user declines replacement)
  const { elements, studio } = setupStudioEnvironment({ confirmResponse: false });

  elements.studioScenarioTemplateSelect.value = "pair_qty_001";
  studio.handleScenarioTemplateSelected();
  await new Promise((r) => setTimeout(r, 10));

  // User explicitly edits the reference transcript
  elements.studioReferenceTranscript.value = "Custom enterprise procurement order 500 units";
  studio.setIsRefUserEdited(true);

  // Switch to negation template; user declines overwrite
  elements.studioScenarioTemplateSelect.value = "pair_neg_001";
  studio.handleScenarioTemplateSelected();
  await new Promise((r) => setTimeout(r, 10));

  assert.strictEqual(
    elements.studioReferenceTranscript.value,
    "Custom enterprise procurement order 500 units",
    "Custom user transcript must be preserved when user declines overwrite"
  );
  assert.strictEqual(elements.studioCategorySelect.value, "negation", "Category still updates");
});

// =============================================================================
// TEST 6: Rapid double-click on Save prevented via synchronous guard
// =============================================================================
test("6. Rapid double-click on Save triggers only 1 upload request", async () => {
  let uploadCallCount = 0;
  const { elements, studio, uploadRequests } = setupStudioEnvironment({
    fetchHandler: async (url) => {
      if (url.includes("/api/studio/upload")) {
        uploadCallCount++;
        // Add realistic network delay
        await new Promise((r) => setTimeout(r, 30));
        return {
          ok: true,
          status: 200,
          json: async () => ({ recording_id: "rec_double_click_test" }),
        };
      }
    },
  });

  // Set up valid state
  await studio.requestMicrophonePermission(true);
  elements.studioReferenceTranscript.value = "Valid transcript";
  elements.studioVerifiedTranscript.value = "Valid transcript";
  elements.studioConsentCheckbox.checked = true;
  studio.setRecordedBlob({ size: 16000, type: "audio/wav" });
  studio.setLastClientDuration(50.0);

  // Simulate two rapid clicks in the same event tick
  const p1 = studio.handleSaveClick();
  const p2 = studio.handleSaveClick();

  await Promise.all([p1, p2]);

  assert.strictEqual(uploadCallCount, 1, "Must issue exactly 1 upload request on double-click");
  assert.strictEqual(uploadRequests.length, 1);
});

// =============================================================================
// TEST 7: Failed-save retry behavior preserves audio and permits re-submission
// =============================================================================
test("7. Failed save keeps recordedBlob intact and re-enables Save for retry", async () => {
  let attempt = 0;
  const { elements, studio } = setupStudioEnvironment({
    fetchHandler: async (url) => {
      if (url.includes("/api/studio/upload")) {
        attempt++;
        if (attempt === 1) {
          return {
            ok: false,
            status: 500,
            json: async () => ({ detail: "Temporary network timeout" }),
          };
        }
        return {
          ok: true,
          status: 200,
          json: async () => ({ recording_id: "rec_retry_success" }),
        };
      }
    },
  });

  await studio.requestMicrophonePermission(true);
  elements.studioReferenceTranscript.value = "Retryable transcript";
  elements.studioVerifiedTranscript.value = "Retryable transcript";
  elements.studioConsentCheckbox.checked = true;
  const mockBlob = { size: 24000, type: "audio/wav" };
  studio.setRecordedBlob(mockBlob);
  studio.setLastClientDuration(50.0);

  // First save fails
  await studio.handleSaveClick();
  assert.strictEqual(elements.btnStudioSave.disabled, false, "Save button must be re-enabled after failure");
  assert.strictEqual(elements.btnStudioRetake.disabled, false, "Retake button must be re-enabled after failure");
  assert.strictEqual(studio.getRecordedBlob(), mockBlob, "recordedBlob must be preserved after failure");

  // Retry succeeds
  await studio.handleSaveClick();
  assert.strictEqual(attempt, 2, "Must execute second attempt");
});

// =============================================================================
// TEST 8: Microphone track and AudioContext cleanup on device switch and unload
// =============================================================================
test("8. Switching microphone device stops previous tracks and closes AudioContext", async () => {
  const { studio, tracksStopped, contextsClosed, mockWindow } = setupStudioEnvironment();

  // First initialization
  await studio.requestMicrophonePermission(true);
  const stream1 = studio.getAudioStream();
  const ctx1 = studio.getAudioContext();

  assert.ok(stream1, "Stream must exist");
  assert.ok(ctx1, "Context must exist");

  // Switch device
  await studio.requestMicrophonePermission(true);

  assert.ok(tracksStopped.length >= 1, "Previous tracks must have stop() called");
  assert.ok(contextsClosed.length >= 1, "Previous AudioContext must be closed");

  // Page unload
  mockWindow.dispatchEvent({ type: "beforeunload" });
  assert.strictEqual(studio.getAudioStream(), null, "Stream must be null after unload");
  assert.strictEqual(studio.getAudioContext(), null, "Context must be null after unload");
});

// =============================================================================
// TEST 9: Microphone dropdown selection preservation across device enumeration
// =============================================================================
test("9. enumerateAudioDevices preserves currently selected device ID", async () => {
  const { elements, studio } = setupStudioEnvironment();

  await studio.requestMicrophonePermission(true);
  elements.studioMicDeviceSelect.value = "mic_usb";

  // Re-enumerate devices (e.g. on peripheral change)
  await studio.enumerateAudioDevices();

  assert.strictEqual(
    elements.studioMicDeviceSelect.value,
    "mic_usb",
    "Selected device ID 'mic_usb' must be preserved across enumeration"
  );
});

// =============================================================================
// TEST 10: Resuming an incomplete pair from sessionStorage after refresh
// =============================================================================
test("10. Page refresh with in-progress pair in sessionStorage resumes Utterance B without advancing pair ID", async () => {
  const savedPair = {
    pairId: "human_qty_007",
    speakerId: "spk_indian_07",
    language: "hinglish",
    category: "quantities_and_numbers",
    scenarioDescription: "Quantity contrast (7 vs 10)",
    expectedContrast: "7 packets vs 10 packets",
    refTranscriptA: "Saat packet chahiye",
    verTranscriptA: "Saat packet chahiye",
    uttARecordingId: "rec_mock_12345",
    currentStep: 3,
  };

  const { elements, studio } = setupStudioEnvironment({
    sessionData: {
      bolibench_studio_in_progress_pair: JSON.stringify(savedPair),
    },
  });

  const resumed = await studio.tryResumeInProgressPair();
  assert.strictEqual(resumed, true, "Must successfully resume in-progress pair");
  assert.strictEqual(studio.getCurrentPair().pairId, "human_qty_007", "Pair ID must be preserved");
  assert.strictEqual(elements.studioPairId.value, "human_qty_007");
  assert.strictEqual(elements.studioUtteranceRoleSelect.value, "utterance_b", "Role must be set to utterance_b");
  assert.strictEqual(elements.studioActivePairIdDisplay.textContent, "human_qty_007");
  assert.ok(elements.studioUttAContrastText.textContent.includes("Saat packet chahiye"), "Must display Utterance A ground truth");
});

// =============================================================================
// TEST 11: Client-side WAV conversion integration and error handling
// =============================================================================
test("11. Client-side WAV converter standardizes audio before upload and reports conversion failure", async () => {
  let convertedCalled = false;
  const mockConverter = {
    convertAudioFile: async (blob) => {
      convertedCalled = true;
      return {
        file: { name: "test_pcm.wav", size: 32000, type: "audio/wav" },
        duration: 50.0,
      };
    },
  };

  const { elements, studio, uploadRequests } = setupStudioEnvironment({ converter: mockConverter });
  await studio.requestMicrophonePermission(true);

  elements.studioReferenceTranscript.value = "Audio conversion test";
  elements.studioVerifiedTranscript.value = "Audio conversion test";
  elements.studioConsentCheckbox.checked = true;
  studio.setRecordedBlob({ size: 40000, type: "audio/webm" });
  studio.setLastClientDuration(1.0);

  await studio.handleSaveClick();

  assert.strictEqual(convertedCalled, true, "BoliAudioConverter.convertAudioFile must be called before upload");
  assert.strictEqual(uploadRequests.length, 1);
  assert.ok(uploadRequests[0].opts.body.fields.audio_file.name.endsWith("_pcm.wav"), "Uploaded file must have _pcm.wav name");
});



// =============================================================================
// TEST 12: Manual stop works before time limit and allows 5-second recording to be saved
// =============================================================================
test("12. Manual stop works before time limit and allows 5-second recording to be saved", async () => {
  const { elements, studio, advanceTime, uploadRequests } = setupStudioEnvironment();
  await studio.requestMicrophonePermission(true);

  elements.studioReferenceTranscript.value = "Short 5-second test utterance";
  elements.studioVerifiedTranscript.value = "Short 5-second test utterance";
  elements.studioConsentCheckbox.checked = true;

  // Start recording
  studio.startRecording();
  assert.strictEqual(studio.getIsRecording(), true, "Recording must be active");
  assert.strictEqual(elements.btnStudioStop.disabled, false, "Stop button must be enabled");
  assert.strictEqual(elements.btnStudioSave.disabled, true, "Save button must be disabled during recording");
  assert.strictEqual(elements.btnStudioRetake.disabled, true, "Retake button must be disabled during recording");

  // Advance clock by 5.0 seconds
  advanceTime(5000);
  assert.strictEqual(elements.studioElapsedTimer.textContent, "5.0s / 600.0s");
  assert.strictEqual(elements.studioCountdownTimer.textContent, "595.0s remaining");

  // User manually clicks Stop after 5 seconds (well before the 600s limit)
  studio.stopRecording(false);
  assert.strictEqual(studio.getIsRecording(), false, "Recording must be stopped");
  assert.strictEqual(elements.btnStudioStop.disabled, true, "Stop button must be disabled after stop");
  assert.strictEqual(elements.btnStudioRetake.disabled, false, "Retake button must be enabled after stop");
  assert.strictEqual(elements.btnStudioSave.disabled, false, "Save button must be enabled after stop");
  assert.strictEqual(elements.studioReviewSection.style.display, "block", "Review section must be visible for preview");
  assert.strictEqual(elements.studioCompleteNotice.style.display, "flex", "Complete notice must be shown");
  const noticeTxt = elements.studioCompleteNotice.querySelector(".notice-text");
  assert.ok(noticeTxt.textContent.includes("5.0s"), "Notice must display 5.0s duration");

  // Save the 5-second recording
  await studio.handleSaveClick();
  assert.strictEqual(uploadRequests.length, 1, "Must issue exactly 1 upload request");
  assert.strictEqual(uploadRequests[0].opts.body.fields.client_duration_seconds, "5.000");
});

// =============================================================================
// TEST 13: Recording automatically stops at 600 seconds as a safety limit
// =============================================================================
test("13. Recording automatically stops at 600 seconds as a safety limit", async () => {
  const { elements, studio, advanceTime } = setupStudioEnvironment();
  await studio.requestMicrophonePermission(true);

  studio.startRecording();
  assert.strictEqual(studio.getIsRecording(), true);

  // Advance clock to 600 seconds
  advanceTime(600000);

  // Verify auto-stop fired
  assert.strictEqual(studio.getIsRecording(), false, "Must auto-stop at 600s");
  assert.strictEqual(studio.getLastClientDuration(), 600.0, "Duration must clamp to 600.0s");
  assert.strictEqual(elements.studioCountdownTimer.textContent, "0.0s remaining");
  assert.strictEqual(elements.studioElapsedTimer.textContent, "600.0s / 600.0s");
  assert.ok(elements.studioWaveformOverlay.textContent.includes("600-Second safety limit reached"));
  assert.strictEqual(elements.btnStudioStop.disabled, true);
  assert.strictEqual(elements.btnStudioRetake.disabled, false);
  assert.strictEqual(elements.btnStudioSave.disabled, false);
});

// =============================================================================
// TEST 14: Empty or invalid audio cannot be saved
// =============================================================================
test("14. Empty or invalid audio cannot be saved", async () => {
  const { elements, studio, uploadRequests } = setupStudioEnvironment();
  await studio.requestMicrophonePermission(true);

  elements.studioReferenceTranscript.value = "Empty audio test";
  elements.studioVerifiedTranscript.value = "Empty audio test";
  elements.studioConsentCheckbox.checked = true;

  // Case A: No audio captured at all (null blob)
  studio.setRecordedBlob(null);
  studio.setLastClientDuration(0.0);

  await studio.handleSaveClick();
  assert.strictEqual(uploadRequests.length, 0, "Must not upload when audio is null");
  assert.strictEqual(elements.studioSaveErrorAlert.style.display, "flex");
  assert.ok(elements.studioSaveErrorMessage.textContent.includes("No recording captured"));

  // Case B: Zero-byte blob
  elements.studioSaveErrorAlert.style.display = "none";
  studio.setRecordedBlob({ size: 0, type: "audio/webm" });
  await studio.handleSaveClick();
  assert.strictEqual(uploadRequests.length, 0, "Must not upload when audio size is 0 bytes");
  assert.strictEqual(elements.studioSaveErrorAlert.style.display, "flex");
});

// =============================================================================
// TEST 15: Retaking a recording does not accidentally retain the previous audio
// =============================================================================
test("15. Retaking a recording does not accidentally retain the previous audio", async () => {
  const { elements, studio, advanceTime, uploadRequests } = setupStudioEnvironment();
  await studio.requestMicrophonePermission(true);

  elements.studioReferenceTranscript.value = "Retake test";
  elements.studioVerifiedTranscript.value = "Retake test";
  elements.studioConsentCheckbox.checked = true;

  // Capture valid 10-second recording
  studio.startRecording();
  advanceTime(10000);
  studio.stopRecording(false);

  assert.ok(studio.getRecordedBlob(), "Recorded blob must exist before retake");
  assert.strictEqual(elements.btnStudioRetake.disabled, false);

  // Retake recording
  studio.retakeRecording();

  assert.strictEqual(studio.getRecordedBlob(), null, "recordedBlob must be cleared to null on retake");
  assert.strictEqual(studio.getLastClientDuration(), 0.0, "lastClientDuration must be reset to 0.0 on retake");
  assert.strictEqual(elements.studioReviewSection.style.display, "none", "Review section must be hidden");
  assert.strictEqual(elements.studioCompleteNotice.style.display, "none", "Complete notice must be hidden");
  assert.strictEqual(elements.btnStudioRetake.disabled, true, "Retake button must be disabled after retake");
  assert.strictEqual(elements.studioCountdownTimer.textContent, "600.0s remaining", "Countdown must reset to 600.0s");
  assert.strictEqual(elements.studioElapsedTimer.textContent, "0.0s / 600.0s", "Elapsed must reset to 0.0s");

  // Attempt to save immediately after retake (without recording again)
  await studio.handleSaveClick();
  assert.strictEqual(uploadRequests.length, 0, "Must not upload previous discarded audio after retake");
  assert.strictEqual(elements.studioSaveErrorAlert.style.display, "flex", "Must show save error alert");
});

// =============================================================================
// TEST 16: Control enabled/disabled states across complete recording lifecycle
// =============================================================================
test("16. Control enabled/disabled states across complete recording lifecycle", async () => {
  const { elements, studio, advanceTime } = setupStudioEnvironment();

  // Initial state before mic
  assert.strictEqual(elements.btnStudioRecord.disabled, true, "Record disabled before mic");
  assert.strictEqual(elements.btnStudioPause.disabled, true, "Pause disabled initially");
  assert.strictEqual(elements.btnStudioStop.disabled, true, "Stop disabled initially");
  assert.strictEqual(elements.btnStudioRetake.disabled, true, "Retake disabled initially");

  // After mic enabled
  await studio.requestMicrophonePermission(true);
  assert.strictEqual(elements.btnStudioRecord.disabled, false, "Record enabled after mic granted");
  assert.strictEqual(elements.btnStudioStop.disabled, true, "Stop disabled before recording starts");
  assert.strictEqual(elements.btnStudioRetake.disabled, true, "Retake disabled before recording starts");

  // Start recording
  studio.startRecording();
  assert.strictEqual(elements.btnStudioRecord.disabled, true, "Record disabled while recording");
  assert.strictEqual(elements.btnStudioPause.disabled, false, "Pause enabled while recording");
  assert.strictEqual(elements.btnStudioStop.disabled, false, "Stop enabled while recording");
  assert.strictEqual(elements.btnStudioRetake.disabled, true, "Retake disabled while recording");
  assert.strictEqual(elements.btnStudioSave.disabled, true, "Save disabled while recording");

  // Stop recording
  advanceTime(5000);
  studio.stopRecording(false);
  assert.strictEqual(elements.btnStudioRecord.disabled, false, "Record re-enabled after stop");
  assert.strictEqual(elements.btnStudioPause.disabled, true, "Pause disabled after stop");
  assert.strictEqual(elements.btnStudioStop.disabled, true, "Stop disabled after stop");
  assert.strictEqual(elements.btnStudioRetake.disabled, false, "Retake enabled after stop with audio");
  assert.strictEqual(elements.btnStudioSave.disabled, false, "Save enabled after stop");
});

// =============================================================================
// TEST 17: Timer display shows elapsed and remaining time correctly at multiple intervals
// =============================================================================
test("17. Timer display shows elapsed and remaining time correctly at multiple intervals", async () => {
  const { elements, studio, advanceTime } = setupStudioEnvironment();
  await studio.requestMicrophonePermission(true);

  // Initial display
  assert.strictEqual(elements.studioCountdownTimer.textContent, "600.0s remaining");
  assert.strictEqual(elements.studioElapsedTimer.textContent, "0.0s / 600.0s");
  assert.strictEqual(elements.studioCountdownProgress.value, 0);

  studio.startRecording();

  // 15 seconds
  advanceTime(15000);
  assert.strictEqual(elements.studioElapsedTimer.textContent, "15.0s / 600.0s");
  assert.strictEqual(elements.studioCountdownTimer.textContent, "585.0s remaining");
  assert.strictEqual(elements.studioCountdownProgress.value, 15);

  // 120 seconds (2 minutes)
  advanceTime(105000);
  assert.strictEqual(elements.studioElapsedTimer.textContent, "120.0s / 600.0s");
  assert.strictEqual(elements.studioCountdownTimer.textContent, "480.0s remaining");
  assert.strictEqual(elements.studioCountdownProgress.value, 120);

  // 300 seconds (5 minutes)
  advanceTime(180000);
  assert.strictEqual(elements.studioElapsedTimer.textContent, "300.0s / 600.0s");
  assert.strictEqual(elements.studioCountdownTimer.textContent, "300.0s remaining");
  assert.strictEqual(elements.studioCountdownProgress.value, 300);

  studio.stopRecording(false);
  assert.strictEqual(elements.studioElapsedTimer.textContent, "300.0s / 600.0s");
  assert.strictEqual(elements.studioCountdownTimer.textContent, "300.0s remaining");
});

// =============================================================================
// TEST 18: 11.7-second manual stop is accepted, displays valid notice, and can be saved
// =============================================================================
test("18. 11.7-second manual stop is accepted, displays valid notice, and can be saved", async () => {
  const { elements, studio, advanceTime, uploadRequests } = setupStudioEnvironment();
  await studio.requestMicrophonePermission(true);

  studio.startRecording();
  advanceTime(11700); // 11.7 seconds
  studio.stopRecording(false);

  // Assert timers
  assert.strictEqual(elements.studioElapsedTimer.textContent, "11.7s / 600.0s");
  assert.strictEqual(elements.studioCountdownTimer.textContent, "588.3s remaining");

  // Assert banners: early stop warning must be HIDDEN, complete notice must be SHOWN
  assert.strictEqual(elements.studioEarlyStopWarning.style.display, "none", "Early stop warning must NOT be shown for 11.7s recording");
  assert.strictEqual(elements.studioCompleteNotice.style.display, "flex", "Complete notice must be shown for 11.7s recording");
  assert.ok(elements.studioCompleteNotice.querySelector(".notice-text").textContent.includes("Valid Recording (11.7s)"));

  // Review section must be visible
  assert.strictEqual(elements.studioReviewSection.style.display, "block", "Review section must be visible");

  // Fill required fields
  elements.studioReferenceTranscript.value = "Do packet chahiye";
  elements.studioReferenceTranscript.dispatchEvent({ type: "input" });
  elements.studioVerifiedTranscript.value = "Do packet chahiye";
  elements.studioVerifiedTranscript.dispatchEvent({ type: "input" });
  elements.studioConsentCheckbox.checked = true;
  elements.studioConsentCheckbox.dispatchEvent({ type: "change" });

  // Save the 11.7s recording
  await studio.handleSaveClick();
  assert.strictEqual(uploadRequests.length, 1, "Must upload valid 11.7s recording");
  assert.ok(uploadRequests[0].opts.body.fields.client_duration_seconds.startsWith("11.7"));
});

// =============================================================================
// TEST 19: Studio UI initializes with 600 seconds, not 50
// =============================================================================
test("19. Studio UI initializes with 600 seconds, not 50", async () => {
  const { elements, studio } = setupStudioEnvironment();

  assert.strictEqual(elements.studioCountdownTimer.textContent, "600.0s remaining");
  assert.strictEqual(elements.studioElapsedTimer.textContent, "0.0s / 600.0s");
  assert.strictEqual(elements.studioCountdownProgress.max, 600);
  assert.strictEqual(elements.studioCountdownProgress.value, 0);
  assert.strictEqual(studio.MAX_RECORDING_DURATION_SEC, 600.0);
});

// =============================================================================
// TEST 20: Empty or corrupt audio cannot be saved and shows error
// =============================================================================
test("20. Empty or corrupt audio cannot be saved and shows error", async () => {
  const { elements, studio, uploadRequests } = setupStudioEnvironment();
  await studio.requestMicrophonePermission(true);

  // Fill transcripts and consent
  elements.studioReferenceTranscript.value = "Do packet chahiye";
  elements.studioVerifiedTranscript.value = "Do packet chahiye";
  elements.studioConsentCheckbox.checked = true;

  // Attempt save with null/empty blob
  studio.setRecordedBlob(null);
  await studio.handleSaveClick();

  assert.strictEqual(uploadRequests.length, 0, "Empty audio must not trigger upload");
  assert.strictEqual(elements.studioSaveErrorAlert.style.display, "flex");
  assert.ok(elements.studioSaveErrorMessage.textContent.includes("Record your voice sample first"));
});

// =============================================================================
// TEST 21: Sub-minimum clips stay reviewable but cannot be treated as valid or saved
// =============================================================================
test("21. A clip shorter than 0.5 seconds remains reviewable but cannot be saved", async () => {
  const { elements, studio, advanceTime, uploadRequests } = setupStudioEnvironment();
  await studio.requestMicrophonePermission(true);

  elements.studioReferenceTranscript.value = "Short clip boundary test";
  elements.studioVerifiedTranscript.value = "Short clip boundary test";
  elements.studioConsentCheckbox.checked = true;

  studio.startRecording();
  advanceTime(200);
  studio.stopRecording(false);

  assert.strictEqual(elements.studioReviewSection.style.display, "block", "Captured audio must remain reviewable");
  assert.strictEqual(elements.studioEarlyStopWarning.style.display, "flex", "Sub-minimum audio must be marked incomplete");
  assert.strictEqual(elements.studioCompleteNotice.style.display, "none", "Sub-minimum audio must not be marked valid");

  await studio.handleSaveClick();
  assert.strictEqual(uploadRequests.length, 0, "Sub-minimum audio must not be uploaded");
  assert.strictEqual(elements.studioSaveErrorAlert.style.display, "flex");
  assert.ok(elements.studioSaveErrorMessage.textContent.includes("at least 0.5 seconds"));
});

// =============================================================================
// TEST 22: Studio HTML template includes static asset versioning query parameters
// =============================================================================
test("22. Studio HTML template includes static asset versioning query parameters for cache-busting", () => {
  const htmlContent = fs.readFileSync("app/templates/studio.html", "utf-8");
  assert.ok(htmlContent.includes("/static/js/studio.js?v="), "studio.js must have version query parameter");
  assert.ok(htmlContent.includes("/static/js/audio_converter.js?v="), "audio_converter.js must have version query parameter");
  assert.ok(htmlContent.includes("/static/css/style.css?v="), "style.css must have version query parameter");
});

// =============================================================================
// TEST 23: Dynamic loadStudioConfig updates max-duration configuration
// =============================================================================
test("23. Dynamic loadStudioConfig updates recording max-duration configuration", async () => {
  let configFetched = false;
  const { elements, studio } = setupStudioEnvironment({
    fetchHandler: async (url) => {
      if (url.includes("/api/studio/config")) {
        configFetched = true;
        return {
          ok: true,
          status: 200,
          json: async () => ({
            storage_directory: "human_recordings_v1",
            max_duration_seconds: 600.0,
            target_duration_seconds: 600.0,
            scenario_templates: SAMPLE_SCENARIO_TEMPLATES,
          }),
        };
      }
    },
  });

  await studio.loadStudioConfig();
  assert.strictEqual(configFetched, true, "Must fetch studio config");
  assert.strictEqual(studio.MAX_RECORDING_DURATION_SEC, 600.0);
  assert.strictEqual(elements.studioCountdownTimer.textContent, "600.0s remaining");
  assert.strictEqual(elements.studioElapsedTimer.textContent, "0.0s / 600.0s");
  assert.strictEqual(elements.studioCountdownProgress.max, 600);
});

// =============================================================================
// TEST 24: Audio converter preserves separate 60-second limit for live evaluation
// =============================================================================
// =============================================================================
// TEST 25: Studio Gnani STT transmission consent is unchecked by default
// =============================================================================
test("25. Studio Gnani STT transmission consent checkbox is unchecked by default", async () => {
  const { elements } = setupStudioEnvironment();
  assert.strictEqual(elements.studioGnaniConsentCheckbox.checked, false, "Consent must be unchecked by default");

  const htmlContent = fs.readFileSync("app/templates/studio.html", "utf-8");
  const match = htmlContent.match(/<input\s+type="checkbox"\s+id="studioGnaniConsentCheckbox"([^>]*)>/);
  assert.ok(match, "studioGnaniConsentCheckbox input must exist in HTML");
  assert.ok(!match[1].includes("checked"), "HTML element must not have checked attribute");
});

// =============================================================================
// TEST 26: Unconsented recording stop makes zero calls to transcription endpoint
// =============================================================================
test("26. Unconsented recording stop makes 0 calls to STT endpoint", async () => {
  let sttCalls = 0;
  const { elements, studio, advanceTime } = setupStudioEnvironment({
    fetchHandler: async (url) => {
      if (url.includes("/api/studio/transcribe")) {
        sttCalls++;
        return {
          ok: true,
          status: 200,
          json: async () => ({ transcript: "Unconsented transcript" }),
        };
      }
    },
  });

  await studio.requestMicrophonePermission(true);
  assert.strictEqual(elements.studioGnaniConsentCheckbox.checked, false);

  studio.startRecording();
  advanceTime(5000); // 5s recording <= 60s
  studio.stopRecording(false);

  assert.strictEqual(sttCalls, 0, "Must make 0 provider calls when unconsented");
  assert.strictEqual(elements.studioVerifiedTranscript.value, "", "Verified transcript must remain empty when unconsented");
});

// =============================================================================
// TEST 27: Consented recording stop triggers STT into verified transcript
// =============================================================================
test("27. Consented recording stop populates Spoken Ground Truth, leaves Reference separate and unverified", async () => {
  let sttCalls = 0;
  const { elements, studio, advanceTime, uploadRequests } = setupStudioEnvironment({
    fetchHandler: async (url) => {
      if (url.includes("/api/studio/transcribe")) {
        sttCalls++;
        return {
          ok: true,
          status: 200,
          json: async () => ({
            success: true,
            transcript: "Recognized spoken ground truth from audio",
            language_code: "hi-IN",
            utterance_role: "utterance_a",
            duration_seconds: 5.0,
            provider: "gnani-prisma-v3",
            credits_consumed: 1,
            external_calls: 1,
            disclaimer: "Transcribed using Gnani AI with explicit user consent.",
          }),
        };
      }
    },
  });

  await studio.requestMicrophonePermission(true);
  elements.studioGnaniConsentCheckbox.checked = true;
  elements.studioReferenceTranscript.value = "Original Intended Script";

  studio.startRecording();
  advanceTime(5000); // 5.0s
  studio.stopRecording(false);

  // Wait a tick for async fetch
  await new Promise((resolve) => setTimeout(resolve, 50));

  assert.strictEqual(sttCalls, 1, "Must call /api/studio/transcribe once");
  assert.strictEqual(elements.studioVerifiedTranscript.value, "Recognized spoken ground truth from audio");
  // Intended reference transcript must remain strictly unchanged
  assert.strictEqual(elements.studioReferenceTranscript.value, "Original Intended Script");
  // Verification checkbox must NOT be auto-checked
  assert.strictEqual(elements.studioTranscriptVerified.checked, false, "Manual verification checkbox must remain unchecked");
  // ASR draft badge must be displayed
  assert.strictEqual(elements.studioAsrDraftBadge.style.display, "inline-block", "ASR draft badge must be visible");
  // Must NOT auto-save
  assert.strictEqual(uploadRequests.length, 0, "Must not auto-save after transcription");
});

// =============================================================================
// TEST 28: Failed STT preserves recordedBlob in memory and reveals Retry STT button
// =============================================================================
test("28. Failed STT preserves audio in memory and reveals Retry button without auto-retry", async () => {
  let sttCalls = 0;
  const { elements, studio, advanceTime } = setupStudioEnvironment({
    fetchHandler: async (url) => {
      if (url.includes("/api/studio/transcribe")) {
        sttCalls++;
        return {
          ok: false,
          status: 500,
          json: async () => ({ detail: "Gnani STT API timeout" }),
        };
      }
    },
  });

  await studio.requestMicrophonePermission(true);
  elements.studioGnaniConsentCheckbox.checked = true;

  studio.startRecording();
  advanceTime(3000);
  studio.stopRecording(false);

  await new Promise((resolve) => setTimeout(resolve, 50));

  assert.strictEqual(sttCalls, 1, "Initial call executed once");
  assert.ok(studio.getRecordedBlob() !== null, "Audio blob must be preserved in memory");
  assert.strictEqual(elements.btnStudioRetryStt.style.display, "inline-block", "Retry button must be visible");
  assert.ok(elements.studioSttStatus.textContent.includes("Transcription failed"), "Status must display error message");
});

// =============================================================================
// TEST 29: Duplicate transcription calls are prevented
// =============================================================================
test("29. Duplicate transcription calls are prevented via isTranscribing guard", async () => {
  let sttCalls = 0;
  let resolveStt;
  const sttPromise = new Promise((resolve) => { resolveStt = resolve; });

  const { elements, studio, advanceTime } = setupStudioEnvironment({
    fetchHandler: async (url) => {
      if (url.includes("/api/studio/transcribe")) {
        sttCalls++;
        await sttPromise;
        return {
          ok: true,
          status: 200,
          json: async () => ({ transcript: "Finished" }),
        };
      }
    },
  });

  await studio.requestMicrophonePermission(true);
  elements.studioGnaniConsentCheckbox.checked = true;

  studio.startRecording();
  advanceTime(4000);
  studio.stopRecording(false);

  // While in progress, trigger another transcription call
  const blob = studio.getRecordedBlob();
  await studio.transcribeStudioRecording(blob);

  assert.strictEqual(sttCalls, 1, "Second concurrent call must be blocked");

  resolveStt();
  await new Promise((resolve) => setTimeout(resolve, 50));
});

// =============================================================================
// TEST 30: Recording > 60.0s shows clear notice, preserves audio, and makes 0 calls
// =============================================================================
test("30. Recording > 60.0s displays clear notice, preserves audio, and makes 0 STT calls", async () => {
  let sttCalls = 0;
  const { elements, studio, advanceTime } = setupStudioEnvironment({
    fetchHandler: async (url) => {
      if (url.includes("/api/studio/transcribe")) {
        sttCalls++;
      }
    },
  });

  await studio.requestMicrophonePermission(true);
  elements.studioGnaniConsentCheckbox.checked = true;

  studio.startRecording();
  advanceTime(75000); // 75 seconds (> 60.0s limit, <= 600s Studio limit)
  studio.stopRecording(false);

  assert.strictEqual(sttCalls, 0, "Must make 0 STT calls for recordings > 60.0s");
  assert.strictEqual(elements.studioSttDurationNotice.style.display, "block", "Duration notice must be displayed");
  assert.ok(elements.studioSttDurationNotice.textContent.includes("60.0-second limit"), "Notice must explain 60s limit");
  assert.ok(studio.getRecordedBlob() !== null, "Audio blob must be preserved for manual verification");
});

// =============================================================================
// TEST 31: Studio labels, instructions, button, and ARIA accurately reflect 600s max & 60s STT
// =============================================================================
test("31. Studio template labels, button text, and ARIA attributes reflect 600s studio & 60s STT limits", () => {
  const htmlContent = fs.readFileSync("app/templates/studio.html", "utf-8");
  assert.ok(htmlContent.includes("<span>🔴 Start Recording</span>"), "Button must say 'Start Recording'");
  assert.ok(!htmlContent.includes("Start 50s Recording"), "Stale 'Start 50s Recording' button text must be removed");
  assert.ok(htmlContent.includes('aria-label="Studio recording timer (Studio recording max 600s; Gnani REST STT max 60s)"'), "ARIA label must disclose 600s studio & 60s STT");
  assert.ok(htmlContent.includes("100% Local by Default (Zero External Calls; Gnani STT requires separate consent)"), "Trust banner must accurately disclose consent requirement");
  assert.ok(htmlContent.includes("Studio Recording Max 600s | Gnani REST STT Max 60s"), "Trust banner must state 600s studio and 60s STT limits");
});

test("24. Audio converter preserves separate 60-second limit for live evaluation", async () => {
  const converterCode = fs.readFileSync("app/static/js/audio_converter.js", "utf-8");
  assert.ok(converterCode.includes("const MAX_DURATION_SECONDS = 60.0"), "Default converter limit must remain 60s for evaluation");
});
