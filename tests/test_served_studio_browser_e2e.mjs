/**
 * BoliBench — Served Studio Browser E2E Lifecycle Verification
 *
 * Fetches the actual served HTML, CSS, and JS assets directly from http://127.0.0.1:8000/studio
 * and executes a completely isolated browser DOM simulation with deterministic mocked STT.
 *
 * Guarantees:
 * - Does NOT reload, focus, touch, or interact with any existing user browser window or tab.
 * - Uses only disposable synthetic audio and deterministic mock responses (0 Gnani API calls, 0 credits).
 * - Verifies the full Stop -> Automatic ASR Draft lifecycle, field separation, consent gating, and error/retry paths.
 */

import test from "node:test";
import assert from "node:assert";
import http from "node:http";
import vm from "node:vm";

// Helper to fetch content over HTTP from running 127.0.0.1:8000 server
function fetchServedUrl(path) {
  return new Promise((resolve, reject) => {
    http.get(`http://127.0.0.1:8000${path}`, (res) => {
      let data = "";
      res.on("data", (chunk) => { data += chunk; });
      res.on("end", () => {
        resolve({
          statusCode: res.statusCode,
          headers: res.headers,
          body: data,
        });
      });
    }).on("error", reject);
  });
}

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
    getContext() {
      return {
        clearRect() {},
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

class MockBlob {
  constructor(parts = [], options = {}) {
    this.parts = parts;
    this.type = options.type || "audio/webm";
    let sz = 0;
    for (const p of parts) sz += p.length || p.byteLength || 0;
    this.size = sz || 32000;
  }
  async arrayBuffer() {
    return new ArrayBuffer(this.size);
  }
}

// Sets up a completely fresh isolated DOM context with the served JS code
function setupServedStudioContext(servedJsCode, { fetchHandler = null } = {}) {
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
    createMediaStreamSource() { return { connect() {} }; }
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
      _store: {},
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
        ],
      },
      permissions: {
        query: async () => ({ state: "granted", onchange: null }),
      },
    },
    alert() {},
    confirm() { return true; },
    requestAnimationFrame(fn) { return 1; },
    cancelAnimationFrame(id) {},
    URL: {
      createObjectURL: () => "blob:mock-audio-url",
      revokeObjectURL: () => {},
    },
    URLSearchParams: globalThis.URLSearchParams,
  };

  const uploadRequests = [];
  const sttRequests = [];

  const mockFetch = async (url, opts) => {
    if (url.includes("/api/studio/upload")) {
      uploadRequests.push({ url, opts });
      return {
        ok: true,
        status: 200,
        json: async () => ({
          recording_id: "rec_e2e_mock",
          validation_status: "valid_50s",
        }),
      };
    }
    if (url.includes("/api/studio/transcribe")) {
      sttRequests.push({ url, opts });
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
          max_duration_seconds: 600.0,
          max_stt_duration_seconds: 60.0,
          live_transcription_enabled: true,
          scenario_templates: [
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
          ],
        }),
      };
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
    start() { this.state = "recording"; }
    stop() {
      this.state = "inactive";
      if (this.ondataavailable) {
        this.ondataavailable({ data: new MockBlob([new Uint8Array(16000)], { type: this.mimeType }) });
      }
      if (this.onstop) {
        this.onstop();
      }
    }
  }

  let currentTime = 10000;
  let activeIntervalCallback = null;
  let intervalIdCounter = 1;

  const context = vm.createContext({
    window: mockWindow,
    document: mockDocument,
    navigator: mockWindow.navigator,
    AudioContext: MockAudioContext,
    MediaRecorder: MockMediaRecorder,
    Blob: MockBlob,
    File: class MockFile extends MockBlob {
      constructor(parts, name, opts) {
        super(parts, opts);
        this.name = name;
      }
    },
    FormData: class MockFormData {
      constructor() { this.entries = {}; }
      append(k, v, filename) { this.entries[k] = { value: v, filename }; }
    },
    fetch: mockFetch,
    console,
    sessionStorage: mockWindow.sessionStorage,
    alert: (m) => mockWindow.alert(m),
    confirm: (m) => mockWindow.confirm(m),
    setTimeout: (fn, delay) => setTimeout(fn, Math.min(delay, 10)),
    clearTimeout: (id) => clearTimeout(id),
    setInterval: (fn) => {
      activeIntervalCallback = fn;
      return intervalIdCounter++;
    },
    clearInterval: () => { activeIntervalCallback = null; },
    requestAnimationFrame: (fn) => 1,
    cancelAnimationFrame: () => {},
    performance: { now: () => currentTime },
    Date: globalThis.Date,
    URL: mockWindow.URL,
    URLSearchParams: globalThis.URLSearchParams,
  });

  vm.runInContext(servedJsCode, context);

  // Fire DOMContentLoaded
  if (docListeners["DOMContentLoaded"]) {
    for (const fn of docListeners["DOMContentLoaded"]) fn();
  }

  return {
    elements,
    studio: mockWindow.BoliStudio || mockWindow.__bolibenchStudio,
    advanceTime: (ms) => {
      currentTime += ms;
      if (activeIntervalCallback) activeIntervalCallback();
    },
    sttRequests,
    uploadRequests,
  };
}

// =============================================================================
// TEST 1: Live Served /studio Page Assets Verification
// =============================================================================
test("E2E-1. Served HTML includes cache-busting v=2.2.1 for all scripts and styles", async () => {
  const page = await fetchServedUrl("/studio");
  assert.strictEqual(page.statusCode, 200, "GET /studio must return 200 OK");
  assert.ok(page.body.includes('href="/static/css/style.css?v=2.2.1"'), "style.css must have v=2.2.1");
  assert.ok(page.body.includes('src="/static/js/studio.js?v=2.2.1"'), "studio.js must have v=2.2.1");
  assert.ok(page.body.includes('src="/static/js/audio_converter.js?v=2.2.1"'), "audio_converter.js must have v=2.2.1");
  assert.ok(page.body.includes('id="studioGnaniConsentCheckbox"'), "Gnani consent checkbox must be present in served HTML");
  assert.ok(page.body.includes('id="studioAsrDraftBadge"'), "ASR draft badge must be present in served HTML");
  assert.ok(page.body.includes('id="btnStudioRetryStt"'), "Retry STT button must be present in served HTML");
});

// =============================================================================
// TEST 2: Stop Recording with Consent -> Automatic ASR Draft Population
// =============================================================================
test("E2E-2. Stop Recording triggers exactly one STT call, sets Spoken Ground Truth as unverified draft, leaves Reference separate, and makes 0 saves", async () => {
  const servedJs = await fetchServedUrl("/static/js/studio.js?v=2.2.1");
  assert.strictEqual(servedJs.statusCode, 200);

  const { elements, studio, advanceTime, sttRequests, uploadRequests } = setupServedStudioContext(servedJs.body, {
    fetchHandler: async (url) => {
      if (url.includes("/api/studio/transcribe")) {
        return {
          ok: true,
          status: 200,
          json: async () => ({
            success: true,
            transcript: "दो पैकेट चाहिए",
            language_code: "hi-IN",
            utterance_role: "utterance_a",
            duration_seconds: 12.1,
            provider: "gnani-prisma-v3",
            credits_consumed: 1,
            external_calls: 1,
            disclaimer: "Transcribed using Gnani AI with explicit user consent. 1 Gnani credit consumed.",
          }),
        };
      }
    },
  });

  await studio.requestMicrophonePermission(true);

  // Entrant checks the Gnani consent box
  elements.studioGnaniConsentCheckbox.checked = true;

  // Intended reference script is populated from scenario
  elements.studioReferenceTranscript.value = "Do packet chahiye";

  // Spoken ground truth starts blank
  elements.studioVerifiedTranscript.value = "";
  elements.studioTranscriptVerified.checked = false;

  // 1. Start recording
  studio.startRecording();

  // 2. Simulate 12.1-second recording (<= 60.0s REST limit)
  advanceTime(12100);

  // 3. Click Stop Recording
  studio.stopRecording(false);

  // Wait a tick for async dispatch and DOM updates
  await new Promise((resolve) => setTimeout(resolve, 60));

  // Assertions:
  // (a) Exactly 1 STT request triggered
  assert.strictEqual(sttRequests.length, 1, "Stop must trigger exactly one STT request");

  // (b) Spoken ground truth field receives the ASR draft transcript
  assert.strictEqual(elements.studioVerifiedTranscript.value, "दो पैकेट चाहिए", "Spoken ground truth field must receive ASR draft");

  // (c) ASR Draft Badge is visible
  assert.strictEqual(elements.studioAsrDraftBadge.style.display, "inline-block", "ASR Draft badge must be displayed");

  // (d) Intended reference transcript remains completely unchanged
  assert.strictEqual(elements.studioReferenceTranscript.value, "Do packet chahiye", "Intended reference transcript must NOT be modified");

  // (e) Manual verification checkbox remains UNCHECKED
  assert.strictEqual(elements.studioTranscriptVerified.checked, false, "Manual verification checkbox must remain unchecked");

  // (f) Zero auto-save upload calls
  assert.strictEqual(uploadRequests.length, 0, "No auto-save upload must be dispatched");
});

// =============================================================================
// TEST 3: Consent-Off Makes Zero STT Calls
// =============================================================================
test("E2E-3. Consent-off recording stop makes zero STT requests and leaves ground truth blank", async () => {
  const servedJs = await fetchServedUrl("/static/js/studio.js?v=2.2.1");

  const { elements, studio, advanceTime, sttRequests } = setupServedStudioContext(servedJs.body, {
    fetchHandler: async (url) => {
      if (url.includes("/api/studio/transcribe")) {
        return {
          ok: true,
          status: 200,
          json: async () => ({ transcript: "Unconsented" }),
        };
      }
    },
  });

  await studio.requestMicrophonePermission(true);

  // Consent checkbox is UNCHECKED (default)
  elements.studioGnaniConsentCheckbox.checked = false;
  elements.studioReferenceTranscript.value = "Do packet chahiye";

  studio.startRecording();
  advanceTime(8000); // 8.0s recording
  studio.stopRecording(false);

  await new Promise((resolve) => setTimeout(resolve, 50));

  // Must make 0 STT calls
  assert.strictEqual(sttRequests.length, 0, "Zero provider calls must be made when consent is unchecked");
  assert.strictEqual(elements.studioVerifiedTranscript.value, "", "Spoken ground truth must remain empty");
  assert.strictEqual(elements.studioReferenceTranscript.value, "Do packet chahiye");
});

// =============================================================================
// TEST 4: Mocked HTTP 504 Gateway Timeout Preserves Audio and Displays Retry STT
// =============================================================================
test("E2E-4. Mocked HTTP 504 Gateway Timeout preserves audio in memory, shows sanitized error, and exposes manual retry", async () => {
  const servedJs = await fetchServedUrl("/static/js/studio.js?v=2.2.1");

  let sttCalls = 0;
  const { elements, studio, advanceTime } = setupServedStudioContext(servedJs.body, {
    fetchHandler: async (url) => {
      if (url.includes("/api/studio/transcribe")) {
        sttCalls++;
        return {
          ok: false,
          status: 504,
          json: async () => ({ detail: "Gnani transcription service timed out (504 Gateway Timeout); audio is preserved; retry later." }),
        };
      }
    },
  });

  await studio.requestMicrophonePermission(true);
  elements.studioGnaniConsentCheckbox.checked = true;

  studio.startRecording();
  advanceTime(10000); // 10.0s
  studio.stopRecording(false);

  await new Promise((resolve) => setTimeout(resolve, 60));

  assert.strictEqual(sttCalls, 1, "Initial call executed once");
  assert.ok(studio.getRecordedBlob() !== null, "Audio blob must be preserved in memory");
  assert.strictEqual(elements.btnStudioRetryStt.style.display, "inline-block", "Retry button must be visible");
  assert.ok(elements.studioSttStatus.textContent.includes("Gnani transcription service timed out"), "Status must report timeout message");
  assert.ok(elements.studioSttStatus.textContent.includes("audio is preserved; retry later"), "Status must assure audio is preserved");

  // Click Retry STT button
  elements.btnStudioRetryStt.dispatchEvent({ type: "click" });
  await new Promise((resolve) => setTimeout(resolve, 60));

  assert.strictEqual(sttCalls, 2, "Manual retry must execute exactly once more");
  assert.ok(studio.getRecordedBlob() !== null, "Audio blob must remain preserved");
});

// =============================================================================
// TEST 5: Duplicate Transcription Prevention Guard
// =============================================================================
test("E2E-5. Duplicate transcription calls are prevented via isTranscribingStudio guard", async () => {
  const servedJs = await fetchServedUrl("/static/js/studio.js?v=2.2.1");

  let sttCalls = 0;
  let finishStt;
  const inFlightPromise = new Promise((resolve) => { finishStt = resolve; });

  const { elements, studio, advanceTime } = setupServedStudioContext(servedJs.body, {
    fetchHandler: async (url) => {
      if (url.includes("/api/studio/transcribe")) {
        sttCalls++;
        await inFlightPromise;
        return {
          ok: true,
          status: 200,
          json: async () => ({ transcript: "Done" }),
        };
      }
    },
  });

  await studio.requestMicrophonePermission(true);
  elements.studioGnaniConsentCheckbox.checked = true;

  studio.startRecording();
  advanceTime(5000);
  studio.stopRecording(false);

  // While first request is in-flight, simulate duplicate trigger
  const blob = studio.getRecordedBlob();
  await studio.transcribeStudioRecording(blob);

  assert.strictEqual(sttCalls, 1, "Duplicate concurrent call must be blocked");

  // Complete in-flight call
  finishStt();
  await new Promise((resolve) => setTimeout(resolve, 50));
});
