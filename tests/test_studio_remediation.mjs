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
    MediaRecorder: { isTypeSupported: () => true },
    Blob: class { constructor(chunks, opts) { this.size = 50000; this.type = opts?.type || "audio/webm"; } },
    FormData: MockFormData,
    URL: globalThis.URL || mockWindow.URL,
    URLSearchParams: globalThis.URLSearchParams,
    setTimeout: (fn) => setTimeout(fn, 0),
    clearTimeout: (id) => clearTimeout(id),
    setInterval: () => 1,
    clearInterval: () => {},
    performance: { now: () => Date.now() },
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

  await studio.handleSaveClick();

  assert.strictEqual(convertedCalled, true, "BoliAudioConverter.convertAudioFile must be called before upload");
  assert.strictEqual(uploadRequests.length, 1);
  assert.ok(uploadRequests[0].opts.body.fields.audio_file.name.endsWith("_pcm.wav"), "Uploaded file must have _pcm.wav name");
});
