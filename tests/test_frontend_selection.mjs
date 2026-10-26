/**
 * BoliBench Phase 6.16.1 — Frontend Dynamic Selection & Modal State Isolation Regression Tests
 *
 * Verifies:
 * 1. Dynamically rendered history rows carry correct data-run-id, data-mode, and data-status.
 * 2. Inspecting a selected eligible run uses that row's actual run ID (not hardcoded eval_live_758277ca81d8).
 * 3. Inspecting a mock, failed, or incomplete run never requests a derived re-score.
 * 4. Modal state isolation prevents stale values from surviving between selections or on error.
 * 5. Clean empty and ineligible states are displayed when no run is available.
 */

import test from "node:test";
import assert from "node:assert";
import fs from "node:fs";
import vm from "node:vm";

function createMockElement(tagName = "div") {
  const listeners = {};
  const attrs = {};
  return {
    tagName: tagName.toUpperCase(),
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
    closest(selector) { return null; },
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
    files: [],
  };
}

function setupEnvironment(fetchHandler, customConverter) {
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
    createElement: (tag) => createMockElement(tag),
    querySelectorAll: () => [],
    querySelector: () => null,
    addEventListener(type, fn) {
      docListeners[type] = docListeners[type] || [];
      docListeners[type].push(fn);
    },
    activeElement: null,
  };

  const revokedUrls = [];
  const mockWindow = {
    document: mockDocument,
    addEventListener() {},
    performance: { now: () => Date.now() },
    URL: {
      createObjectURL: (file) => `blob:mock/${file ? file.name : "unnamed"}`,
      revokeObjectURL: (url) => { revokedUrls.push(url); },
    },
    navigator: { clipboard: { writeText: async () => {} } },
    alert: () => {},
    confirm: () => true,
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
    return { ok: false, status: 404, json: async () => ({}) };
  };

  const defaultConverter = customConverter || {
    convertAudioFile: async (file) => {
      return {
        file: {
          name: file.name.replace(/\.[^/.]+$/, "") + "_pcm.wav",
          size: 32000,
          type: "audio/wav",
          _isConvertedWav: true,
        },
        duration: 1.0,
        sampleRate: 16000,
        channels: 1,
        bitDepth: 16,
        fileSize: 32044,
      };
    },
  };

  class MockFormData {
    constructor() {
      this._entries = [];
    }
    append(key, val, filename) {
      this._entries.push({ key, val, filename });
    }
    get(key) {
      const entry = this._entries.find((e) => e.key === key);
      return entry ? entry.val : null;
    }
    has(key) {
      return this._entries.some((e) => e.key === key);
    }
  }

  const code = fs.readFileSync("app/static/js/app.js", "utf-8");
  const context = vm.createContext({
    window: mockWindow,
    document: mockDocument,
    fetch: mockFetch,
    console,
    alert: (msg) => { mockWindow.alert(msg); },
    confirm: (msg) => mockWindow.confirm(msg),
    setTimeout: (fn) => setTimeout(fn, 0),
    clearTimeout: (id) => clearTimeout(id),
    performance: mockWindow.performance,
    URL: mockWindow.URL,
    FormData: MockFormData,
    BoliAudioConverter: defaultConverter,
  });

  vm.runInContext(code, context);

  // Trigger DOMContentLoaded
  if (docListeners["DOMContentLoaded"]) {
    for (const fn of docListeners["DOMContentLoaded"]) {
      fn();
    }
  }

  return {
    elements,
    bolibench: mockWindow.__bolibench,
    requestedUrls,
    revokedUrls,
    mockWindow,
  };
}

// =============================================================================
// Test 1: Dynamically rendered history row carries correct attributes
// =============================================================================
test("1. Dynamically rendered history rows carry correct run ID, execution mode, and status", () => {
  const { elements, bolibench } = setupEnvironment();
  assert.ok(bolibench, "window.__bolibench must be exposed for testing");

  const tbody = createMockElement("tbody");
  const sampleRuns = [
    {
      run_id: "eval_live_test_001",
      test_case_id: "pair_qty_001",
      execution_mode: "live",
      overall_status: "completed",
      semantic_distinction_preserved: true,
      task_success: true,
      average_wer: 0.0,
      timestamp: "2026-10-21T05:00:00Z",
    },
    {
      run_id: "eval_mock_test_002",
      test_case_id: "pair_neg_001",
      execution_mode: "mock",
      overall_status: "completed",
      semantic_distinction_preserved: true,
      task_success: true,
      average_wer: 0.0,
      timestamp: "2026-10-21T05:01:00Z",
    },
    {
      run_id: "eval_live_failed_003",
      test_case_id: "pair_item_001",
      execution_mode: "live",
      overall_status: "failed",
      primary_failure_category: "upstream_504_gateway_timeout",
      timestamp: "2026-10-21T05:02:00Z",
    },
    {
      run_id: "eval_live_inc_004",
      test_case_id: "pair_addr_001",
      execution_mode: "live",
      overall_status: "incomplete",
      primary_failure_category: "partial_provider_timeout",
      timestamp: "2026-10-21T05:03:00Z",
    },
  ];

  bolibench.renderHistoryRows(tbody, sampleRuns, true);
  const html = tbody.innerHTML;

  // Verify run 1 (live completed)
  assert.ok(html.includes('data-run-id="eval_live_test_001"'), "Must carry actual live run ID");
  assert.ok(html.includes('data-mode="live"'), "Must carry live mode attribute");
  assert.ok(html.includes('data-status="completed"'), "Must carry completed status attribute");

  // Verify run 2 (mock completed)
  assert.ok(html.includes('data-run-id="eval_mock_test_002"'), "Must carry mock run ID");
  assert.ok(html.includes('data-mode="mock"'), "Must carry mock mode attribute");
  assert.ok(html.includes('data-status="completed"'), "Must carry completed status attribute");

  // Verify run 3 (live failed)
  assert.ok(html.includes('data-run-id="eval_live_failed_003"'), "Must carry failed run ID");
  assert.ok(html.includes('data-mode="live"'), "Must carry live mode attribute");
  assert.ok(html.includes('data-status="failed"'), "Must carry failed status attribute");

  // Verify run 4 (live incomplete)
  assert.ok(html.includes('data-run-id="eval_live_inc_004"'), "Must carry incomplete run ID");
  assert.ok(html.includes('data-mode="live"'), "Must carry live mode attribute");
  assert.ok(html.includes('data-status="incomplete"'), "Must carry incomplete status attribute");

  // Verify inspect button class
  assert.ok(html.includes('class="btn-row-inspect"'), "Inspect button must carry class btn-row-inspect");
});

// =============================================================================
// Test 2: Inspecting an eligible live run uses actual run ID and does not silently mutate
// =============================================================================
test("2. Inspecting a selected eligible run uses that row's actual run ID (not eval_live_758277ca81d8) and guards re-score generation", async () => {
  const selectedRunId = "eval_live_custom_dyn_999";
  let derivedRescoreCalled = false;

  const fetchHandler = async (url) => {
    if (url === `/api/eval/results/${encodeURIComponent(selectedRunId)}`) {
      return {
        ok: true,
        status: 200,
        json: async () => ({
          run_id: selectedRunId,
          test_case_id: "pair_qty_001",
          audio_condition: "clean_recorded",
          language_code: "hi-IN",
          execution_mode: "live",
          overall_status: "completed",
          task_success: false,
          primary_failure_category: "unrecognized_intent",
          failure_explanation: "Devanagari verb phrasing",
          client_request_elapsed_ms: 2100,
          utterance_a: {
            request_id: "req_alpha_999",
            actual_transcript: "2 पैकेट चाहिए",
            reference_transcript: "Do packet chahiye",
            wer: 1.0,
            latency_ms: 1050,
          },
          utterance_b: {
            request_id: "req_beta_999",
            actual_transcript: "3 पैकेट चाहिए",
            reference_transcript: "Teen packet chahiye",
            wer: 1.0,
            latency_ms: 1050,
          },
          average_wer: 1.0,
        }),
      };
    }
    if (url.startsWith("/api/eval/derived-assessments")) {
      // No derived assessment exists yet for this run
      return { ok: true, status: 200, json: async () => [] };
    }
    if (url === `/api/eval/runs/${encodeURIComponent(selectedRunId)}/derived-rescore`) {
      derivedRescoreCalled = true;
      return {
        ok: true,
        status: 200,
        json: async () => ({
          derived_id: `rescore_${selectedRunId}`,
          source_run_id: selectedRunId,
          test_case_id: "pair_qty_001",
          created_at: "2026-10-21T05:10:00Z",
          normalized_wer_a: 0.0,
          normalized_wer_b: 0.0,
          average_normalized_wer: 0.0,
          rescore_task_success: true,
          rescore_detected_intent_a: "ORDER_PRODUCT",
          rescore_primary_failure_category: "none",
          rescore_failure_explanation: "Downstream order intent recognized from Devanagari",
          actual_transcript_a: "2 पैकेट चाहिए",
          actual_transcript_b: "3 पैकेट चाहिए",
          reference_a: "Do packet chahiye",
          reference_b: "Teen packet chahiye",
        }),
      };
    }
    return { ok: false, status: 404, json: async () => ({}) };
  };

  const { elements, bolibench, requestedUrls } = setupEnvironment(fetchHandler);

  // Inspect the dynamically selected run
  await bolibench.openRescoreCompareModal(selectedRunId, "live", "completed");

  // Invariant 1: Source run fetched using actual run ID, NOT eval_live_758277ca81d8
  assert.ok(
    requestedUrls.some((r) => r.url === `/api/eval/results/${encodeURIComponent(selectedRunId)}`),
    `Must request source result with actual run ID ${selectedRunId}`
  );
  assert.strictEqual(
    requestedUrls.some((r) => r.url.includes("eval_live_758277ca81d8")),
    false,
    "Must not fall back to eval_live_758277ca81d8 when an explicit run ID is passed"
  );

  // Invariant 2: Derived re-score route must NOT be called merely on modal open!
  assert.strictEqual(
    derivedRescoreCalled,
    false,
    "Must NOT silently create/call derived-rescore on modal open"
  );

  // Invariant 3: Source column populated with actual run's data
  assert.strictEqual(elements.modalLiveRunId.textContent, selectedRunId);
  assert.strictEqual(elements.modalLiveReqA.textContent, "req_alpha_999");
  assert.strictEqual(elements.modalLiveReqB.textContent, "req_beta_999");
  assert.strictEqual(elements.modalLiveActualA.textContent, "2 पैकेट चाहिए");

  // Invariant 4: Explanatory notice and explicit button displayed
  assert.strictEqual(elements.modalComputeRescoreArea.style.display, "block");
  assert.ok(elements.btnComputeDerivedRescore.onclick != null, "Action button must have click handler wired");

  // Now simulate explicit user action to compute derived re-score
  await elements.btnComputeDerivedRescore.onclick();

  assert.strictEqual(derivedRescoreCalled, true, "Derived re-score must be requested after explicit user action");
  assert.strictEqual(elements.modalDerivedId.textContent, `rescore_${selectedRunId}`);
  assert.strictEqual(elements.modalDerivedSrcId.textContent, selectedRunId);
  assert.strictEqual(elements.modalDerivedContent.style.display, "flex");
  assert.strictEqual(elements.modalDerivedUnavailable.style.display, "none");
  assert.ok(elements.modalBtnExportJson.href.includes(selectedRunId), "Export link must reference actual run ID");
});

// =============================================================================
// Test 3: Inspecting mock, failed, or incomplete runs never requests derived re-score
// =============================================================================
test("3. Inspecting a mock, failed, or incomplete run does not request a derived re-score", async () => {
  let derivedEndpointHit = false;

  const fetchHandler = async (url) => {
    if (url.includes("/derived-rescore") || url.includes("/derived-assessments")) {
      derivedEndpointHit = true;
      return { ok: true, json: async () => ({}) };
    }
    if (url === "/api/eval/results/eval_mock_123") {
      return {
        ok: true,
        status: 200,
        json: async () => ({
          run_id: "eval_mock_123",
          execution_mode: "mock",
          overall_status: "completed",
          test_case_id: "pair_qty_001",
          task_success: true,
        }),
      };
    }
    if (url === "/api/eval/results/eval_live_fail_456") {
      return {
        ok: true,
        status: 200,
        json: async () => ({
          run_id: "eval_live_fail_456",
          execution_mode: "live",
          overall_status: "failed",
          test_case_id: "pair_qty_001",
          task_success: false,
          primary_failure_category: "upstream_504_gateway_timeout",
        }),
      };
    }
    if (url === "/api/eval/results/eval_live_inc_789") {
      return {
        ok: true,
        status: 200,
        json: async () => ({
          run_id: "eval_live_inc_789",
          execution_mode: "live",
          overall_status: "incomplete",
          test_case_id: "pair_qty_001",
          task_success: false,
        }),
      };
    }
    return { ok: false, status: 404, json: async () => ({}) };
  };

  const { elements, bolibench } = setupEnvironment(fetchHandler);

  // 3a. Inspect Mock Run
  derivedEndpointHit = false;
  await bolibench.openRescoreCompareModal("eval_mock_123", "mock", "completed");
  assert.strictEqual(derivedEndpointHit, false, "Mock run must not call derived re-score or listing endpoints");
  assert.strictEqual(elements.modalDerivedContent.style.display, "none");
  assert.strictEqual(elements.modalDerivedUnavailable.style.display, "block");
  assert.ok(elements.modalDerivedUnavailableMsg.textContent.includes("Deterministic Mock Simulation"));

  // 3b. Inspect Failed Run
  derivedEndpointHit = false;
  await bolibench.openRescoreCompareModal("eval_live_fail_456", "live", "failed");
  assert.strictEqual(derivedEndpointHit, false, "Failed run must not call derived endpoints");
  assert.strictEqual(elements.modalDerivedContent.style.display, "none");
  assert.strictEqual(elements.modalDerivedUnavailable.style.display, "block");
  assert.ok(elements.modalDerivedUnavailableMsg.textContent.includes("Operational Provider Error / Incomplete Run"));

  // 3c. Inspect Incomplete Run
  derivedEndpointHit = false;
  await bolibench.openRescoreCompareModal("eval_live_inc_789", "live", "incomplete");
  assert.strictEqual(derivedEndpointHit, false, "Incomplete run must not call derived endpoints");
  assert.strictEqual(elements.modalDerivedContent.style.display, "none");
  assert.strictEqual(elements.modalDerivedUnavailable.style.display, "block");
  assert.ok(elements.modalDerivedUnavailableMsg.textContent.includes("Operational Provider Error / Incomplete Run"));
});

// =============================================================================
// Test 4: Modal state isolation prevents stale values between selections
// =============================================================================
test("4. A missing or invalid API response cannot leave stale values from a previously opened modal", async () => {
  const fetchHandler = async (url) => {
    if (url === "/api/eval/results/eval_live_run_A") {
      return {
        ok: true,
        status: 200,
        json: async () => ({
          run_id: "eval_live_run_A",
          execution_mode: "live",
          overall_status: "completed",
          test_case_id: "pair_qty_001",
          audio_condition: "clean_recorded",
          language_code: "hi-IN",
          client_request_elapsed_ms: 1800,
          utterance_a: { request_id: "req_A_1", actual_transcript: "Transcript A from Run 1", wer: 0.1 },
          utterance_b: { request_id: "req_A_2", actual_transcript: "Transcript B from Run 1", wer: 0.2 },
          average_wer: 0.15,
          task_success: true,
        }),
      };
    }
    if (url.startsWith("/api/eval/derived-assessments")) {
      return {
        ok: true,
        status: 200,
        json: async () => [
          {
            derived_id: "rescore_eval_live_run_A",
            source_run_id: "eval_live_run_A",
            actual_transcript_a: "Transcript A from Run 1",
            actual_transcript_b: "Transcript B from Run 1",
            normalized_wer_a: 0.0,
            normalized_wer_b: 0.0,
            average_normalized_wer: 0.0,
            rescore_task_success: true,
          },
        ],
      };
    }
    if (url === "/api/eval/results/eval_nonexistent_run_B") {
      // Run B returns 404 (failed or invalid)
      return { ok: false, status: 404, json: async () => ({ detail: "Run not found" }) };
    }
    return { ok: false, status: 404, json: async () => ({}) };
  };

  const { elements, bolibench } = setupEnvironment(fetchHandler);

  // Step 1: Open Run A (successful run with existing derived assessment)
  await bolibench.openRescoreCompareModal("eval_live_run_A", "live", "completed");
  assert.strictEqual(elements.modalLiveRunId.textContent, "eval_live_run_A");
  assert.strictEqual(elements.modalDerivedId.textContent, "rescore_eval_live_run_A");
  assert.strictEqual(elements.modalDerivedActualA.textContent, "Transcript A from Run 1");
  assert.strictEqual(elements.modalDerivedContent.style.display, "flex");
  assert.ok(elements.modalBtnExportJson.href.includes("eval_live_run_A"));

  // Step 2: Now open Run B (which returns 404)
  await bolibench.openRescoreCompareModal("eval_nonexistent_run_B", "live", "completed");

  // Invariant: Zero stale data from Run A must survive!
  assert.strictEqual(elements.modalDerivedContent.style.display, "none", "Derived content must be hidden");
  assert.strictEqual(elements.modalDerivedUnavailable.style.display, "block", "Unavailable banner must be shown");
  assert.ok(elements.modalDerivedUnavailableMsg.textContent.includes("not found in active database"));

  // Crucial: Fields must NOT retain values from Run A
  assert.strictEqual(elements.modalDerivedId.textContent, "—", "Derived ID must be reset");
  assert.strictEqual(elements.modalDerivedActualA.textContent, "—", "Derived transcript must be reset");
  assert.strictEqual(elements.modalDerivedNormWerA.textContent, "Normalized WER: —", "WER must be reset");
  assert.strictEqual(elements.modalLiveActualA.textContent, "—", "Live transcript must be reset");
  assert.strictEqual(elements.modalLiveReqA.textContent, "—", "Live request ID must be reset");
  assert.strictEqual(elements.modalLiveTaskSuccess.textContent, "—", "Task success badge must be reset");
  assert.ok(!elements.modalBtnExportJson.href, "Export link must be empty/cleared");
});

// =============================================================================
// Test 5: Clean empty and ineligible states
// =============================================================================
test("5. Correct empty or ineligible state is shown when no run ID is selected", async () => {
  const { elements, bolibench, requestedUrls } = setupEnvironment();

  // Clear boot-time fetch requests
  requestedUrls.length = 0;

  // Call openRescoreCompareModal with null and no latestLiveRunId
  await bolibench.openRescoreCompareModal(null, null, null);

  assert.strictEqual(requestedUrls.length, 0, "No network calls should be made if no target run is available");
  assert.strictEqual(elements.modalDerivedContent.style.display, "none");
  assert.strictEqual(elements.modalDerivedUnavailable.style.display, "block");
  assert.ok(
    elements.modalDerivedUnavailableMsg.textContent.includes("No completed live evaluation run found or selected"),
    "Must display clean empty state message"
  );
});

// =============================================================================
// Test 6: Selecting a supported file starts conversion and updates UI states
// =============================================================================
test("6. Selecting a supported audio file automatically converts to 16 kHz WAV and updates preview", async () => {
  const { elements, bolibench, requestedUrls } = setupEnvironment();
  requestedUrls.length = 0; // reset boot requests

  const mockInputFile = { name: "interview_recording.m4a", size: 65536, type: "audio/mp4" };
  elements.liveAudioInputA.files = [mockInputFile];

  // Await async conversion handler
  await bolibench.handleAudioSelection("A", mockInputFile);

  const convertedFiles = bolibench.getCurrentConvertedFiles();
  assert.ok(convertedFiles.a, "Utterance A converted file should be populated");
  assert.strictEqual(convertedFiles.a.name, "interview_recording_pcm.wav");
  assert.strictEqual(convertedFiles.a.type, "audio/wav");

  // Verify UI feedback
  assert.strictEqual(elements.origFileNameA.textContent, "Source: interview_recording.m4a");
  assert.strictEqual(elements.fileNameA.textContent, "interview_recording_pcm.wav");
  assert.strictEqual(elements.conversionFeedbackA.classList.contains("status-success"), true);
  assert.ok(elements.conversionFeedbackA.textContent.includes("Converted successfully"));

  // Verify preview player src
  assert.ok(elements.liveAudioPreviewA.src.includes("blob:mock/interview_recording_pcm.wav"));
  assert.strictEqual(elements.previewContainerA.style.display, "flex");

  // Verify zero network/provider requests were made during conversion
  assert.strictEqual(requestedUrls.length, 0, "Audio conversion must not invoke any backend or provider endpoints");
});

// =============================================================================
// Test 7: Converted WAV files (not original recordings) are passed to FormData
// =============================================================================
test("7. Live evaluation submission passes converted WAV files instead of original recordings", async () => {
  let submittedFormData = null;
  const fetchHandler = async (url, opts) => {
    if (url === "/api/eval/authorize-live") {
      return {
        ok: true,
        status: 200,
        json: async () => ({
          status: "authorized",
          confirmation_token: "test_token_xyz_123",
          required_header: "X-BoliBench-Confirm-Live",
          required_header_value: "authorize-confirmed",
        }),
      };
    }
    if (url.startsWith("/api/eval/run-live/")) {
      submittedFormData = opts.body;
      return {
        ok: true,
        status: 200,
        json: async () => ({
          run_id: "eval_live_test_conversion_999",
          test_case_id: "pair_qty_001",
          execution_mode: "live",
          overall_status: "completed",
          semantic_distinction_preserved: true,
          task_success: true,
          average_wer: 0.05,
          timestamp: new Date().toISOString(),
          utterance_a: {
            actual_transcript: "Do packet chahiye",
            stt_latency_ms: 320,
            fields_match: true,
            action_match: true,
          },
          utterance_b: {
            actual_transcript: "Teen packet chahiye",
            stt_latency_ms: 340,
            fields_match: true,
            action_match: true,
          },
        }),
      };
    }
    return { ok: false, status: 404, json: async () => ({}) };
  };

  const { elements, bolibench } = setupEnvironment(fetchHandler);

  // Set test case ID
  elements.evalTestCaseSelect.value = "pair_qty_001";

  // Simulate converting file A (M4A) and file B (MP3)
  const origFileA = { name: "raw_phone_speech.m4a", size: 120000, type: "audio/mp4" };
  const origFileB = { name: "raw_phone_contrast.mp3", size: 140000, type: "audio/mpeg" };

  elements.liveAudioInputA.files = [origFileA];
  elements.liveAudioInputB.files = [origFileB];

  await bolibench.handleAudioSelection("A", origFileA);
  await bolibench.handleAudioSelection("B", origFileB);

  // Check confirmation box
  elements.confirmLiveCreditUsage.checked = true;
  bolibench.checkLiveEvalFormValidity();

  assert.strictEqual(elements.btnExecuteLiveEval.disabled, false, "Submit button should be enabled after conversion & confirmation");

  // Trigger live evaluation execution
  elements.btnExecuteLiveEval.dispatchEvent({ type: "click" });
  await new Promise((r) => setTimeout(r, 20));

  assert.ok(submittedFormData, "FormData must have been sent to run-live endpoint");
  const fileA = submittedFormData.get("audio_file_a");
  const fileB = submittedFormData.get("audio_file_b");

  assert.ok(fileA, "audio_file_a must be in FormData");
  assert.ok(fileB, "audio_file_b must be in FormData");
  assert.strictEqual(fileA.name, "raw_phone_speech_pcm.wav", "Must submit converted WAV, not original M4A");
  assert.strictEqual(fileB.name, "raw_phone_contrast_pcm.wav", "Must submit converted WAV, not original MP3");
  assert.strictEqual(fileA.type, "audio/wav");
  assert.strictEqual(fileB.type, "audio/wav");
});

// =============================================================================
// Test 8: Submission remains disabled while either file is converting
// =============================================================================
test("8. Submission remains disabled while either audio file is in converting state", async () => {
  let resolveConversionA;
  const slowConverter = {
    convertAudioFile: async (file) => {
      if (file.name.includes("slow")) {
        return new Promise((resolve) => {
          resolveConversionA = () => resolve({
            file: { name: file.name + "_pcm.wav", size: 32000, type: "audio/wav" },
            duration: 1.0,
            sampleRate: 16000,
            channels: 1,
            bitDepth: 16,
            fileSize: 32044,
          });
        });
      }
      return {
        file: { name: file.name + "_pcm.wav", size: 32000, type: "audio/wav" },
        duration: 1.0,
        sampleRate: 16000,
        channels: 1,
        bitDepth: 16,
        fileSize: 32044,
      };
    },
  };

  const { elements, bolibench } = setupEnvironment(null, slowConverter);
  elements.confirmLiveCreditUsage.checked = true;

  // File B converts immediately
  await bolibench.handleAudioSelection("B", { name: "clip_b.wav", size: 32000 });

  // File A starts slow conversion
  const pendingConversionA = bolibench.handleAudioSelection("A", { name: "clip_slow.m4a", size: 40000 });

  // While converting, verify disabled state and status
  const states = bolibench.getConversionStates();
  assert.strictEqual(states.isConvertingA, true, "Utterance A must be in converting state");
  assert.strictEqual(elements.btnExecuteLiveEval.disabled, true, "Submit button must be disabled while A is converting");
  assert.strictEqual(elements.btnLiveEvalText.textContent, "Utterance A is Converting...");

  // Finish conversion A
  resolveConversionA();
  await pendingConversionA;

  // Now both are converted and confirmation checked
  assert.strictEqual(bolibench.getConversionStates().isConvertingA, false);
  assert.strictEqual(elements.btnExecuteLiveEval.disabled, false, "Submit button should enable when both conversions complete");
});

// =============================================================================
// Test 9: Conversion failure prevents live submission and displays actionable error
// =============================================================================
test("9. Conversion failure on either recording prevents live submission and displays actionable error", async () => {
  const failingConverter = {
    convertAudioFile: async (file) => {
      if (file.name.includes("overlength")) {
        throw new Error("Audio duration 75.2s exceeds the 60.0-second limit.");
      }
      return {
        file: { name: file.name + "_pcm.wav", size: 32000, type: "audio/wav" },
        duration: 1.0,
        sampleRate: 16000,
        channels: 1,
        bitDepth: 16,
        fileSize: 32044,
      };
    },
  };

  const { elements, bolibench } = setupEnvironment(null, failingConverter);
  elements.confirmLiveCreditUsage.checked = true;

  // Utterance A succeeds
  await bolibench.handleAudioSelection("A", { name: "valid_clip_a.m4a", size: 50000 });
  assert.ok(bolibench.getCurrentConvertedFiles().a, "Utterance A should be converted");

  // Utterance B fails due to duration > 60s
  await bolibench.handleAudioSelection("B", { name: "overlength_clip_b.m4a", size: 500000 });

  // Utterance B must be null, error message displayed
  assert.strictEqual(bolibench.getCurrentConvertedFiles().b, null, "Utterance B must not have converted file");
  assert.strictEqual(elements.conversionFeedbackB.classList.contains("status-error"), true);
  assert.ok(elements.conversionFeedbackB.textContent.includes("Audio exceeds the permitted duration"));

  // Live submission must strictly remain disabled
  assert.strictEqual(elements.btnExecuteLiveEval.disabled, true, "Submit button must remain disabled if B failed");
});

// =============================================================================
// Test 10: Corrupt or unsupported audio format provides clear decode error
// =============================================================================
test("10. Corrupt or unsupported audio format provides actionable message and rejects submission", async () => {
  const unsupportedConverter = {
    convertAudioFile: async () => {
      throw new Error("Unable to decode audio data — codec or container not supported by browser.");
    },
  };

  const { elements, bolibench } = setupEnvironment(null, unsupportedConverter);

  await bolibench.handleAudioSelection("A", { name: "corrupted.amr", size: 2048 });

  assert.strictEqual(bolibench.getCurrentConvertedFiles().a, null);
  assert.strictEqual(elements.conversionFeedbackA.classList.contains("status-error"), true);
  assert.ok(elements.conversionFeedbackA.textContent.includes("this browser could not decode the selected audio format"));
  assert.strictEqual(elements.btnExecuteLiveEval.disabled, true);
});

// =============================================================================
// Test 11: Rapid file replacement prevents stale async conversions from overwriting
// =============================================================================
test("11. Rapid file replacement discards stale async conversion results", async () => {
  let resolveFirst;
  const delayedConverter = {
    convertAudioFile: async (file) => {
      if (file.name === "file_first.m4a") {
        return new Promise((resolve) => {
          resolveFirst = () => resolve({
            file: { name: "stale_first_pcm.wav", size: 10000, type: "audio/wav" },
            duration: 0.5,
            sampleRate: 16000,
            channels: 1,
            bitDepth: 16,
            fileSize: 10044,
          });
        });
      }
      return {
        file: { name: "newer_second_pcm.wav", size: 20000, type: "audio/wav" },
        duration: 1.0,
        sampleRate: 16000,
        channels: 1,
        bitDepth: 16,
        fileSize: 20044,
      };
    },
  };

  const { elements, bolibench } = setupEnvironment(null, delayedConverter);

  // User starts selecting first file
  const firstPromise = bolibench.handleAudioSelection("A", { name: "file_first.m4a", size: 10000 });

  // User rapidly replaces with second file
  const secondPromise = bolibench.handleAudioSelection("A", { name: "file_second.m4a", size: 20000 });
  await secondPromise;

  assert.strictEqual(bolibench.getCurrentConvertedFiles().a.name, "newer_second_pcm.wav");

  // Now the first slow conversion finishes late
  resolveFirst();
  await firstPromise;

  // Crucial: The late result from file_first must NOT overwrite the newer file_second!
  assert.strictEqual(
    bolibench.getCurrentConvertedFiles().a.name,
    "newer_second_pcm.wav",
    "Stale async conversion must not overwrite newer selection"
  );
  assert.strictEqual(elements.fileNameA.textContent, "newer_second_pcm.wav");
});

// =============================================================================
// Test 12: Clearing an audio input resets state and revokes object URLs
// =============================================================================
test("12. Clearing audio input releases object URLs and resets conversion state", async () => {
  const { elements, bolibench, revokedUrls } = setupEnvironment();

  // Select and convert file
  await bolibench.handleAudioSelection("A", { name: "test_speech.ogg", size: 30000 });
  assert.ok(bolibench.getCurrentConvertedFiles().a);
  assert.strictEqual(elements.previewContainerA.style.display, "flex");

  // User clears input (null file)
  await bolibench.handleAudioSelection("A", null);

  assert.strictEqual(bolibench.getCurrentConvertedFiles().a, null);
  assert.strictEqual(elements.previewContainerA.style.display, "none");
  assert.strictEqual(elements.origFileNameA.style.display, "none");
  assert.ok(revokedUrls.length > 0, "Obsolete object URL must be revoked when cleared");
  assert.strictEqual(elements.btnExecuteLiveEval.disabled, true);
});

// =============================================================================
// Test 13: Zero Gnani credits or provider endpoints called during conversion
// =============================================================================
test("13. Conversion never calls live evaluation or transcription endpoints", async () => {
  const forbiddenCalled = [];
  const monitoringFetch = async (url) => {
    if (url.includes("authorize-live") || url.includes("run-live") || url.includes("transcribe")) {
      forbiddenCalled.push(url);
    }
    return { ok: true, status: 200, json: async () => ({}) };
  };

  const { bolibench } = setupEnvironment(monitoringFetch);

  // Perform multiple conversions
  await bolibench.handleAudioSelection("A", { name: "audio1.m4a", size: 20000 });
  await bolibench.handleAudioSelection("B", { name: "audio2.m4a", size: 25000 });

  assert.strictEqual(forbiddenCalled.length, 0, "Zero credit-consuming endpoints may be called during conversion");
});

// =============================================================================
// Test 14: Script loading chain: audio_converter.js executes before app.js and exposes window.BoliAudioConverter
// =============================================================================
test("14. Script loading chain: audio_converter.js exposes window.BoliAudioConverter to app.js", async () => {
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
    createElement: (tag) => createMockElement(tag),
    querySelectorAll: () => [],
    querySelector: () => null,
    addEventListener(type, fn) {
      docListeners[type] = docListeners[type] || [];
      docListeners[type].push(fn);
    },
    activeElement: null,
  };

  class MockAudioContext {
    constructor() {}
    async decodeAudioData(buffer, success) {
      const audioBuf = {
        sampleRate: 44100,
        numberOfChannels: 1,
        duration: 1.2,
        length: Math.floor(44100 * 1.2),
        getChannelData: () => new Float32Array(Math.floor(44100 * 1.2)),
      };
      if (typeof success === "function") success(audioBuf);
      return audioBuf;
    }
    async close() {}
  }

  const mockWindow = {
    document: mockDocument,
    addEventListener() {},
    performance: { now: () => Date.now() },
    AudioContext: MockAudioContext,
    URL: {
      createObjectURL: (file) => `blob:mock/${file ? file.name : "unnamed"}`,
      revokeObjectURL: () => {},
    },
    File: class File {
      constructor(chunks, name, opts) {
        this.name = name;
        this.size = chunks.reduce((acc, c) => acc + (c.byteLength || c.size || 0), 0);
        this.type = (opts && opts.type) || "audio/wav";
      }
    },
    navigator: { clipboard: { writeText: async () => {} } },
    alert: () => {},
    confirm: () => true,
  };

  const mockFetch = async () => ({ ok: true, status: 200, json: async () => [] });

  const context = vm.createContext({
    window: mockWindow,
    document: mockDocument,
    globalThis: mockWindow,
    fetch: mockFetch,
    console,
    alert: (msg) => { mockWindow.alert(msg); },
    confirm: (msg) => mockWindow.confirm(msg),
    setTimeout: (fn) => setTimeout(fn, 0),
    clearTimeout: (id) => clearTimeout(id),
    performance: mockWindow.performance,
    URL: mockWindow.URL,
    File: mockWindow.File,
    FormData: class { append() {} get() {} has() {} },
  });

  // Step 1: Execute audio_converter.js in context (simulating first <script>)
  const convCode = fs.readFileSync("app/static/js/audio_converter.js", "utf-8");
  vm.runInContext(convCode, context);

  // Assert window.BoliAudioConverter is populated immediately
  assert.ok(mockWindow.BoliAudioConverter, "window.BoliAudioConverter must exist after loading audio_converter.js");
  assert.strictEqual(typeof mockWindow.BoliAudioConverter.convertAudioFile, "function");

  // Step 2: Execute app.js in context (simulating second <script>)
  const appCode = fs.readFileSync("app/static/js/app.js", "utf-8");
  vm.runInContext(appCode, context);

  for (const fn of (docListeners["DOMContentLoaded"] || [])) fn();

  // Step 3: Test that app.js successfully uses window.BoliAudioConverter for audio conversion
  const testInput = {
    name: "phone_sample.m4a",
    size: 20000,
    arrayBuffer: async () => new ArrayBuffer(20000),
  };

  await mockWindow.__bolibench.handleAudioSelection("A", testInput);

  const converted = mockWindow.__bolibench.getCurrentConvertedFiles();
  assert.ok(converted.a, "Utterance A must be converted");
  assert.strictEqual(converted.a.name, "phone_sample_pcm.wav");
  assert.strictEqual(elements.conversionFeedbackA.classList.contains("status-success"), true);
  assert.ok(elements.conversionFeedbackA.textContent.includes("Converted successfully"));
});

// =============================================================================
// Test 15: Changing test case scenario resets audio selection to prevent mismatch
// =============================================================================
test("15. Changing test case scenario resets audio conversion and selection state", async () => {
  const { elements, bolibench } = setupEnvironment();

  // Populate converted file A and B
  const mockFileA = { name: "clip_a.wav", size: 4000, type: "audio/wav" };
  const mockFileB = { name: "clip_b.wav", size: 4000, type: "audio/wav" };

  await bolibench.handleAudioSelection("A", mockFileA);
  await bolibench.handleAudioSelection("B", mockFileB);

  elements.confirmLiveCreditUsage.checked = true;
  bolibench.checkLiveEvalFormValidity();

  let converted = bolibench.getCurrentConvertedFiles();
  assert.ok(converted.a, "Utterance A should be converted");
  assert.ok(converted.b, "Utterance B should be converted");
  assert.strictEqual(elements.btnExecuteLiveEval.disabled, false);

  // User changes the test case dropdown to another scenario
  elements.evalTestCaseSelect.value = "pair_neg_order_001";
  elements.evalTestCaseSelect.dispatchEvent({ type: "change" });

  // Verification: Audio selection must be completely reset!
  converted = bolibench.getCurrentConvertedFiles();
  assert.strictEqual(converted.a, null, "Utterance A converted file must be cleared on scenario switch");
  assert.strictEqual(converted.b, null, "Utterance B converted file must be cleared on scenario switch");
  assert.strictEqual(elements.btnExecuteLiveEval.disabled, true, "Submit button must be disabled until new audio is provided");
});

// =============================================================================
// Test 16: Live evaluation passes expected_category to authorize-live and run-live
// =============================================================================
test("16. Live evaluation passes expected_category matching selected test case to guard credit usage", async () => {
  let authPayload = null;
  let runLiveFormData = null;

  const fetchHandler = async (url, opts) => {
    if (url === "/api/eval/authorize-live") {
      authPayload = JSON.parse(opts.body);
      return {
        ok: true,
        status: 200,
        json: async () => ({
          status: "authorized",
          confirmation_token: "token_guard_check_456",
          required_header: "X-BoliBench-Confirm-Live",
          required_header_value: "authorize-confirmed",
        }),
      };
    }
    if (url.startsWith("/api/eval/run-live/")) {
      runLiveFormData = opts.body;
      return {
        ok: true,
        status: 200,
        json: async () => ({
          run_id: "eval_live_category_guard_test",
          test_case_id: "pair_time_deliv_001",
          execution_mode: "live",
          overall_status: "completed",
          semantic_distinction_preserved: true,
          task_success: true,
          average_wer: 0.0,
          timestamp: new Date().toISOString(),
          utterance_a: { actual_transcript: "डिलीवरी सुबह नौ बजे चाहिए", fields_match: true },
          utterance_b: { actual_transcript: "डिलीवरी सुबह दस बजे चाहिए", fields_match: true },
        }),
      };
    }
    return { ok: false, status: 404, json: async () => ({}) };
  };

  const { elements, bolibench } = setupEnvironment(fetchHandler);

  // Set test case and matching category
  elements.evalTestCaseSelect.value = "pair_time_deliv_001";
  const option = createMockElement("option");
  option.setAttribute("data-category", "time_extraction");
  elements.evalTestCaseSelect.options = [option];
  elements.evalTestCaseSelect.selectedIndex = 0;

  const fileA = { name: "time_9am.wav", size: 8000, type: "audio/wav" };
  const fileB = { name: "time_10am.wav", size: 8000, type: "audio/wav" };

  await bolibench.handleAudioSelection("A", fileA);
  await bolibench.handleAudioSelection("B", fileB);

  elements.confirmLiveCreditUsage.checked = true;
  bolibench.checkLiveEvalFormValidity();

  elements.btnExecuteLiveEval.dispatchEvent({ type: "click" });
  await new Promise((r) => setTimeout(r, 20));

  assert.ok(authPayload, "authorize-live must be called");
  assert.strictEqual(authPayload.test_id, "pair_time_deliv_001");
  assert.strictEqual(authPayload.expected_category, "time_extraction");

  assert.ok(runLiveFormData, "run-live FormData must be sent");
  assert.strictEqual(runLiveFormData.get("expected_category"), "time_extraction");
});

// =============================================================================
// Test 17: Credit Authorization Modal Interaction & Promise Resolution
// =============================================================================
test("17. Credit authorization modal correctly displays clip metadata and resolves user confirmation/cancellation", async () => {
  const { elements, bolibench } = setupEnvironment();

  // Attach dialog methods to liveCreditAuthModal
  elements.liveCreditAuthModal.showModal = function() {
    this.open = true;
  };
  elements.liveCreditAuthModal.close = function() {
    this.open = false;
    this.dispatchEvent({ type: "close" });
  };

  // Test 17a: User authorizes by clicking confirm button
  const authPromiseA = bolibench.requestCreditAuthorization("Curated Scenario", "recording_a.wav", "recording_b.wav");
  assert.strictEqual(elements.liveCreditAuthModal.open, true, "Dialog must be opened");
  assert.strictEqual(elements.modalAuthFileA.textContent, "recording_a.wav");
  assert.strictEqual(elements.modalAuthFileB.textContent, "recording_b.wav");
  assert.strictEqual(elements.modalAuthEvalMode.textContent, "Curated Scenario");

  // User clicks confirm button
  elements.btnConfirmLiveModal.dispatchEvent({ type: "click" });
  const resultA = await authPromiseA;
  assert.strictEqual(resultA, true, "Clicking confirm must resolve authorization to true");
  assert.strictEqual(elements.liveCreditAuthModal.open, false, "Dialog must be closed after confirmation");

  // Test 17b: User cancels by clicking cancel button
  const authPromiseB = bolibench.requestCreditAuthorization("Curated Scenario", "recording_a.wav", "recording_b.wav");
  assert.strictEqual(elements.liveCreditAuthModal.open, true, "Dialog must be opened");

  elements.btnCancelLiveModal.dispatchEvent({ type: "click" });
  const resultB = await authPromiseB;
  assert.strictEqual(resultB, false, "Clicking cancel must resolve authorization to false");
  assert.strictEqual(elements.liveCreditAuthModal.open, false, "Dialog must be closed after cancellation");

  // Test 17c: User closes via header close button
  const authPromiseC = bolibench.requestCreditAuthorization("Curated Scenario", "recording_a.wav", "recording_b.wav");
  assert.strictEqual(elements.liveCreditAuthModal.open, true, "Dialog must be opened");

  elements.modalCloseBtn.dispatchEvent({ type: "click" });
  const resultC = await authPromiseC;
  assert.strictEqual(resultC, false, "Clicking header close button must resolve authorization to false");
  assert.strictEqual(elements.liveCreditAuthModal.open, false, "Dialog must be closed");
});
