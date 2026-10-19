/**
 * BoliBench — Frontend Interaction, Telemetry & Paired-Contrast Evaluation Handler
 * Manages audio upload, live playback preview, API interaction, latency telemetry,
 * error handling, and the paired-contrast evaluation engine.
 */

document.addEventListener("DOMContentLoaded", () => {
  // ===========================================================================
  // 1. Tab Navigation
  // ===========================================================================
  const tabBtnTranscribe = document.getElementById("tabBtnTranscribe");
  const tabBtnEvaluate = document.getElementById("tabBtnEvaluate");
  const transcribeView = document.getElementById("transcribeView");
  const evaluateView = document.getElementById("evaluateView");

  function switchTab(targetTab) {
    if (targetTab === "evaluateView") {
      tabBtnEvaluate.classList.add("active");
      tabBtnEvaluate.setAttribute("aria-selected", "true");
      tabBtnTranscribe.classList.remove("active");
      tabBtnTranscribe.setAttribute("aria-selected", "false");

      transcribeView.style.display = "none";
      evaluateView.style.display = "block";

      // Refresh benchmark metrics and current test-case specs
      loadBenchmarkMetrics();
      loadEvaluationHistory();
      updateSelectedTestCaseSpecs();
    } else {
      tabBtnTranscribe.classList.add("active");
      tabBtnTranscribe.setAttribute("aria-selected", "true");
      tabBtnEvaluate.classList.remove("active");
      tabBtnEvaluate.setAttribute("aria-selected", "false");

      evaluateView.style.display = "none";
      transcribeView.style.display = "block";
    }
  }

  tabBtnTranscribe.addEventListener("click", () => switchTab("transcribeView"));
  tabBtnEvaluate.addEventListener("click", () => switchTab("evaluateView"));

  // ===========================================================================
  // 2. Single Clip Audio Transcription Lab (Existing Working Functionality)
  // ===========================================================================
  const dropzone = document.getElementById("dropzone");
  const audioFileInput = document.getElementById("audioFileInput");
  const dropzonePrompt = document.getElementById("dropzonePrompt");
  const selectedFilePreview = document.getElementById("selectedFilePreview");
  const fileNameDisplay = document.getElementById("fileNameDisplay");
  const fileSizeDisplay = document.getElementById("fileSizeDisplay");
  const audioPreview = document.getElementById("audioPreview");
  const btnClearFile = document.getElementById("btnClearFile");

  const transcribeForm = document.getElementById("transcribeForm");
  const languageCodeSelect = document.getElementById("languageCode");
  const formatModeSelect = document.getElementById("formatMode");
  const itnNativeNumerals = document.getElementById("itnNativeNumerals");
  const mockModeCheckbox = document.getElementById("mockMode");

  const btnSubmit = document.getElementById("btnSubmit");
  const btnText = document.getElementById("btnText");
  const btnSpinner = document.getElementById("btnSpinner");

  const resultStatusBadge = document.getElementById("resultStatusBadge");
  const clientLatencyDisplay = document.getElementById("clientLatencyDisplay");
  const backendLatencyDisplay = document.getElementById("backendLatencyDisplay");
  const modelDisplay = document.getElementById("modelDisplay");
  const requestIdDisplay = document.getElementById("requestIdDisplay");

  const transcriptDisplay = document.getElementById("transcriptDisplay");
  const btnCopyTranscript = document.getElementById("btnCopyTranscript");

  const errorAlert = document.getElementById("errorAlert");
  const errorTypeDisplay = document.getElementById("errorTypeDisplay");
  const errorMessageDisplay = document.getElementById("errorMessageDisplay");
  const errorRemediationDisplay = document.getElementById("errorRemediationDisplay");

  let currentAudioUrl = null;

  function handleFileSelected(file) {
    if (!file) return;

    if (currentAudioUrl) {
      URL.revokeObjectURL(currentAudioUrl);
      currentAudioUrl = null;
    }

    const sizeKB = (file.size / 1024).toFixed(1);
    const sizeMB = (file.size / (1024 * 1024)).toFixed(2);
    const sizeFormatted = file.size > 1024 * 1024 ? `${sizeMB} MB` : `${sizeKB} KB`;

    fileNameDisplay.textContent = file.name;
    fileSizeDisplay.textContent = sizeFormatted;

    currentAudioUrl = URL.createObjectURL(file);
    audioPreview.src = currentAudioUrl;

    dropzonePrompt.style.display = "none";
    selectedFilePreview.style.display = "flex";
    hideError();
  }

  function clearSelectedFile() {
    audioFileInput.value = "";
    if (currentAudioUrl) {
      URL.revokeObjectURL(currentAudioUrl);
      currentAudioUrl = null;
    }
    audioPreview.removeAttribute("src");
    audioPreview.load();

    selectedFilePreview.style.display = "none";
    dropzonePrompt.style.display = "block";
  }

  audioFileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFileSelected(e.target.files[0]);
    }
  });

  btnClearFile.addEventListener("click", (e) => {
    e.stopPropagation();
    clearSelectedFile();
  });

  ["dragenter", "dragover"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.add("drag-over");
    });
  });

  ["dragleave", "drop"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.remove("drag-over");
    });
  });

  dropzone.addEventListener("drop", (e) => {
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      audioFileInput.files = e.dataTransfer.files;
      handleFileSelected(file);
    }
  });

  formatModeSelect.addEventListener("change", () => {
    if (formatModeSelect.value === "verbatim") {
      itnNativeNumerals.disabled = true;
      itnNativeNumerals.checked = false;
    } else {
      itnNativeNumerals.disabled = false;
    }
  });

  if (formatModeSelect.value === "verbatim") {
    itnNativeNumerals.disabled = true;
  }

  transcribeForm.addEventListener("submit", async (e) => {
    e.preventDefault();

    const file = audioFileInput.files[0];
    if (!file) {
      showError("MISSING_FILE", "Please select an audio file to transcribe.");
      return;
    }

    setLoadingState(true);
    hideError();

    const formData = new FormData();
    formData.append("audio_file", file);
    formData.append("language_code", languageCodeSelect.value);
    formData.append("format_mode", formatModeSelect.value);
    formData.append("itn_native_numerals", itnNativeNumerals.checked);
    formData.append("mock_mode", mockModeCheckbox.checked);

    const clientStartTime = performance.now();

    try {
      const response = await fetch("/api/transcribe", {
        method: "POST",
        body: formData,
      });

      const clientLatencyMs = Math.round(performance.now() - clientStartTime);
      const data = await response.json();

      if (!response.ok) {
        throw {
          status: response.status,
          error_type: data.error_type || `HTTP_${response.status}`,
          message: data.message || data.detail || "Transcription request failed",
          remediation: data.remediation || null,
        };
      }

      displaySuccess(data, clientLatencyMs);
    } catch (err) {
      const clientLatencyMs = Math.round(performance.now() - clientStartTime);
      displayError(err, clientLatencyMs);
    } finally {
      setLoadingState(false);
    }
  });

  function setLoadingState(isLoading) {
    btnSubmit.disabled = isLoading;
    if (isLoading) {
      btnText.textContent = "Transcribing Speech...";
      btnSpinner.style.display = "inline-block";
      resultStatusBadge.className = "status-badge status-running";
      resultStatusBadge.textContent = "Processing...";
      transcriptDisplay.innerHTML = `<p class="placeholder-text" style="color: #93c5fd;">Streaming to Gnani Prisma v2.5 STT engine...</p>`;
      btnCopyTranscript.style.display = "none";
    } else {
      btnText.textContent = "Transcribe with Gnani Prisma";
      btnSpinner.style.display = "none";
    }
  }

  function displaySuccess(data, clientLatencyMs) {
    clientLatencyDisplay.textContent = `${clientLatencyMs} ms`;
    backendLatencyDisplay.textContent = `${data.backend_latency_ms} ms`;
    modelDisplay.textContent = data.model || "gnani-prisma-v2.5";
    requestIdDisplay.textContent = data.request_id || "req_local";

    if (data.is_mocked) {
      resultStatusBadge.className = "status-badge status-mock";
      resultStatusBadge.textContent = "Mocked Response (200 OK)";
    } else {
      resultStatusBadge.className = "status-badge status-success";
      resultStatusBadge.textContent = "Success (200 OK)";
    }

    transcriptDisplay.innerHTML = `<p style="color: #f3f4f6; font-size: 1.05rem;">${escapeHtml(data.transcript)}</p>`;
    btnCopyTranscript.style.display = "block";
    hideError();
  }

  function displayError(err, clientLatencyMs) {
    clientLatencyDisplay.textContent = `${clientLatencyMs} ms`;
    resultStatusBadge.className = "status-badge status-error";
    resultStatusBadge.textContent = `Failed (${err.status || "ERR"})`;

    transcriptDisplay.innerHTML = `<p class="placeholder-text" style="color: #fca5a5;">Transcription failed. Review diagnostic details below.</p>`;
    btnCopyTranscript.style.display = "none";

    showError(err.error_type || "API_ERROR", err.message, err.remediation);
  }

  function showError(type, message, remediation = null) {
    errorTypeDisplay.textContent = type;
    errorMessageDisplay.textContent = message;

    if (remediation) {
      errorRemediationDisplay.textContent = remediation;
      errorRemediationDisplay.style.display = "block";
    } else {
      errorRemediationDisplay.style.display = "none";
    }

    errorAlert.style.display = "flex";
  }

  function hideError() {
    errorAlert.style.display = "none";
  }

  btnCopyTranscript.addEventListener("click", async () => {
    const text = transcriptDisplay.innerText;
    if (!text) return;

    try {
      await navigator.clipboard.writeText(text);
      btnCopyTranscript.textContent = "Copied!";
      setTimeout(() => {
        btnCopyTranscript.textContent = "Copy";
      }, 2000);
    } catch {
      btnCopyTranscript.textContent = "Failed";
    }
  });

  // ===========================================================================
  // 3. Paired-Contrast Evaluation Engine Handler
  // ===========================================================================
  const evalTestCaseSelect = document.getElementById("evalTestCaseSelect");
  const testCategoryBadge = document.getElementById("testCategoryBadge");
  const testDescriptionText = document.getElementById("testDescriptionText");

  const specRefA = document.getElementById("specRefA");
  const specIntentA = document.getElementById("specIntentA");
  const specFieldsA = document.getElementById("specFieldsA");
  const specActionA = document.getElementById("specActionA");
  const specLangA = document.getElementById("specLangA");

  const specRefB = document.getElementById("specRefB");
  const specIntentB = document.getElementById("specIntentB");
  const specFieldsB = document.getElementById("specFieldsB");
  const specActionB = document.getElementById("specActionB");
  const specLangB = document.getElementById("specLangB");

  const specDistinctionField = document.getElementById("specDistinctionField");
  const specExpectedDistinctionText = document.getElementById("specExpectedDistinctionText");

  const btnRunMockIdeal = document.getElementById("btnRunMockIdeal");
  const btnRunMockCrashed = document.getElementById("btnRunMockCrashed");
  const btnToggleCustomInputs = document.getElementById("btnToggleCustomInputs");
  const customTranscriptDrawer = document.getElementById("customTranscriptDrawer");
  const customTranscriptA = document.getElementById("customTranscriptA");
  const customTranscriptB = document.getElementById("customTranscriptB");
  const btnScoreCustomTranscripts = document.getElementById("btnScoreCustomTranscripts");

  const btnToggleLiveEval = document.getElementById("btnToggleLiveEval");
  const liveEvalDrawer = document.getElementById("liveEvalDrawer");
  const liveRefPromptA = document.getElementById("liveRefPromptA");
  const liveRefPromptB = document.getElementById("liveRefPromptB");
  const liveAudioInputA = document.getElementById("liveAudioInputA");
  const liveAudioInputB = document.getElementById("liveAudioInputB");
  const confirmLiveCreditUsage = document.getElementById("confirmLiveCreditUsage");
  const btnExecuteLiveEval = document.getElementById("btnExecuteLiveEval");
  const btnLiveEvalText = document.getElementById("btnLiveEvalText");
  const btnLiveSpinner = document.getElementById("btnLiveSpinner");

  const evalStatusBadge = document.getElementById("evalStatusBadge");
  const evalResultsPlaceholder = document.getElementById("evalResultsPlaceholder");
  const evalResultsContainer = document.getElementById("evalResultsContainer");

  const evalOutcomeBanner = document.getElementById("evalOutcomeBanner");
  const evalOutcomeIcon = document.getElementById("evalOutcomeIcon");
  const evalOutcomeTitle = document.getElementById("evalOutcomeTitle");
  const evalOutcomeExplanation = document.getElementById("evalOutcomeExplanation");
  const evalExecModePill = document.getElementById("evalExecModePill");

  const metricDistinctionDisplay = document.getElementById("metricDistinctionDisplay");
  const metricTaskSuccessDisplay = document.getElementById("metricTaskSuccessDisplay");
  const metricMeanWerDisplay = document.getElementById("metricMeanWerDisplay");
  const metricFieldAccDisplay = document.getElementById("metricFieldAccDisplay");

  const evalComparisonTableBody = document.getElementById("evalComparisonTableBody");
  const crashImpactDescription = document.getElementById("crashImpactDescription");
  const crashFieldTags = document.getElementById("crashFieldTags");

  const runIdDisplay = document.getElementById("runIdDisplay");
  const runLatencyDisplay = document.getElementById("runLatencyDisplay");
  const runTimestampDisplay = document.getElementById("runTimestampDisplay");

  const statTotalRuns = document.getElementById("statTotalRuns");
  const statDistinctionRate = document.getElementById("statDistinctionRate");
  const statTaskSuccessRate = document.getElementById("statTaskSuccessRate");
  const statAvgWer = document.getElementById("statAvgWer");
  const historyTableBody = document.getElementById("historyTableBody");

  let currentTestCaseData = null;

  async function updateSelectedTestCaseSpecs() {
    const testId = evalTestCaseSelect.value;
    if (!testId) return;

    try {
      const response = await fetch(`/api/eval/test-cases/${testId}`);
      if (!response.ok) return;
      const tc = await response.json();
      currentTestCaseData = tc;

      testCategoryBadge.textContent = tc.category;
      testDescriptionText.textContent = tc.description;

      specRefA.textContent = `"${tc.utterance_a.reference_transcript}"`;
      specIntentA.textContent = tc.utterance_a.expected_intent;
      specFieldsA.textContent = JSON.stringify(tc.utterance_a.expected_critical_fields);
      specActionA.textContent = tc.utterance_a.expected_action;
      specLangA.textContent = tc.language_code;

      specRefB.textContent = `"${tc.utterance_b.reference_transcript}"`;
      specIntentB.textContent = tc.utterance_b.expected_intent;
      specFieldsB.textContent = JSON.stringify(tc.utterance_b.expected_critical_fields);
      specActionB.textContent = tc.utterance_b.expected_action;
      specLangB.textContent = tc.language_code;

      specDistinctionField.textContent = `Critical Distinction: ${tc.distinction_field}`;
      specExpectedDistinctionText.textContent = tc.expected_distinction;

      if (liveRefPromptA) liveRefPromptA.textContent = tc.utterance_a.reference_transcript;
      if (liveRefPromptB) liveRefPromptB.textContent = tc.utterance_b.reference_transcript;

      // Populate custom inputs as starting template
      customTranscriptA.value = tc.utterance_a.reference_transcript;
      customTranscriptB.value = tc.utterance_b.reference_transcript;
    } catch (err) {
      console.error("Failed to load test case details:", err);
    }
  }

  evalTestCaseSelect.addEventListener("change", updateSelectedTestCaseSpecs);

  btnToggleCustomInputs.addEventListener("click", () => {
    const isHidden = customTranscriptDrawer.style.display === "none";
    customTranscriptDrawer.style.display = isHidden ? "flex" : "none";
    btnToggleCustomInputs.textContent = isHidden ? "Hide Custom Inputs" : "Custom Transcript Test";
    if (isHidden) {
      liveEvalDrawer.style.display = "none";
      btnToggleLiveEval.textContent = "🎙️ Live Audio Pair (Consumes Credits)";
    }
  });

  btnToggleLiveEval.addEventListener("click", () => {
    const isHidden = liveEvalDrawer.style.display === "none";
    liveEvalDrawer.style.display = isHidden ? "flex" : "none";
    btnToggleLiveEval.textContent = isHidden ? "Hide Live Audio Pair" : "🎙️ Live Audio Pair (Consumes Credits)";
    if (isHidden) {
      customTranscriptDrawer.style.display = "none";
      btnToggleCustomInputs.textContent = "Custom Transcript Test";
    }
  });

  function checkLiveEvalFormValidity() {
    const hasA = liveAudioInputA.files && liveAudioInputA.files.length > 0;
    const hasB = liveAudioInputB.files && liveAudioInputB.files.length > 0;
    const isConfirmed = confirmLiveCreditUsage.checked;
    btnExecuteLiveEval.disabled = !(hasA && hasB && isConfirmed);
  }

  liveAudioInputA.addEventListener("change", checkLiveEvalFormValidity);
  liveAudioInputB.addEventListener("change", checkLiveEvalFormValidity);
  confirmLiveCreditUsage.addEventListener("change", checkLiveEvalFormValidity);

  btnExecuteLiveEval.addEventListener("click", async () => {
    const testId = evalTestCaseSelect.value;
    if (!testId) return;

    if (!confirmLiveCreditUsage.checked) {
      alert("Please confirm the credit usage authorization before executing live evaluation.");
      return;
    }

    if (!liveAudioInputA.files || !liveAudioInputA.files[0] || !liveAudioInputB.files || !liveAudioInputB.files[0]) {
      alert("Please select audio recordings for both Utterance A and Utterance B.");
      return;
    }

    // Explicit confirmation action protecting user credits
    const userConfirmed = window.confirm(
      "Live Evaluation Credit Authorization:\n\n" +
      "Evaluating this pair makes 2 real requests to Gnani Prisma STT v3 and may consume programme credits.\n\n" +
      "Do you authorize 2 live requests for Utterance A and Utterance B?"
    );
    if (!userConfirmed) return;

    btnExecuteLiveEval.disabled = true;
    btnLiveSpinner.style.display = "inline-block";
    btnLiveEvalText.textContent = "Authorizing Live Evaluation...";
    evalStatusBadge.className = "status-badge status-running";
    evalStatusBadge.textContent = "Authorizing Credit Guard...";

    try {
      // Step 1: Request single-use authorization token from server
      const authResp = await fetch("/api/eval/authorize-live", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-Requested-With": "XMLHttpRequest",
        },
        body: JSON.stringify({ test_id: testId }),
      });

      if (!authResp.ok) {
        const authErr = await authResp.json().catch(() => ({}));
        throw new Error(authErr.detail || `Live authorization rejected (HTTP ${authResp.status})`);
      }

      const authData = await authResp.json();
      const authToken = authData.confirmation_token;

      // Step 2: Build multipart payload with confirmation token
      const formData = new FormData();
      formData.append("audio_file_a", liveAudioInputA.files[0]);
      formData.append("audio_file_b", liveAudioInputB.files[0]);
      formData.append("audio_condition", "clean_recorded");
      formData.append("confirmation_token", authToken);

      btnLiveEvalText.textContent = "Transcribing Live via Gnani...";
      evalStatusBadge.textContent = "Evaluating Live Audio Pair...";

      const response = await fetch(`/api/eval/run-live/${testId}`, {
        method: "POST",
        headers: {
          "X-BoliBench-Confirm-Live": "CONFIRM_LIVE_EVALUATION",
        },
        body: formData,
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || `Live evaluation failed with HTTP ${response.status}`);
      }

      const result = await response.json();
      displayEvaluationScorecard(result);
      loadBenchmarkMetrics();
      loadEvaluationHistory();
    } catch (err) {
      console.error("Live paired evaluation error:", err);
      evalStatusBadge.className = "status-badge status-error";
      evalStatusBadge.textContent = "Live Evaluation Failed";
      alert(`Live Evaluation Error: ${err.message}`);
    } finally {
      btnLiveSpinner.style.display = "none";
      btnLiveEvalText.textContent = "Execute Paired Live Evaluation";
      checkLiveEvalFormValidity();
    }
  });

  async function executeMockRun(scenario) {
    const testId = evalTestCaseSelect.value;
    if (!testId) return;

    evalStatusBadge.className = "status-badge status-running";
    evalStatusBadge.textContent = "Scoring Pair...";

    try {
      const response = await fetch(`/api/eval/run-mock/${testId}?scenario=${scenario}`, {
        method: "POST",
      });

      if (!response.ok) {
        throw new Error(`Mock evaluation failed with HTTP ${response.status}`);
      }

      const result = await response.json();
      displayEvaluationScorecard(result);
      loadBenchmarkMetrics();
      loadEvaluationHistory();
    } catch (err) {
      console.error("Error executing mock evaluation:", err);
      evalStatusBadge.className = "status-badge status-error";
      evalStatusBadge.textContent = "Evaluation Failed";
    }
  }

  btnRunMockIdeal.addEventListener("click", () => executeMockRun("ideal_preserved"));
  btnRunMockCrashed.addEventListener("click", () => executeMockRun("crashed_distinction"));

  btnScoreCustomTranscripts.addEventListener("click", async () => {
    const testId = evalTestCaseSelect.value;
    const txtA = customTranscriptA.value.trim();
    const txtB = customTranscriptB.value.trim();

    if (!txtA || !txtB) {
      alert("Please provide transcripts for both Utterance A and Utterance B.");
      return;
    }

    evalStatusBadge.className = "status-badge status-running";
    evalStatusBadge.textContent = "Scoring Transcripts...";

    try {
      const response = await fetch("/api/eval/evaluate-transcripts", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          test_case_id: testId,
          transcript_a: txtA,
          transcript_b: txtB,
          execution_mode: "mock",
          audio_condition: "text_only_simulation",
        }),
      });

      if (!response.ok) {
        throw new Error("Evaluation request failed");
      }

      const result = await response.json();
      displayEvaluationScorecard(result);
      loadBenchmarkMetrics();
      loadEvaluationHistory();
    } catch (err) {
      console.error("Custom transcript evaluation error:", err);
      evalStatusBadge.className = "status-badge status-error";
      evalStatusBadge.textContent = "Scoring Error";
    }
  });

  function displayEvaluationScorecard(res) {
    evalResultsPlaceholder.style.display = "none";
    evalResultsContainer.style.display = "block";

    // Execution Mode
    evalExecModePill.textContent = res.execution_mode.toUpperCase() + " EVALUATION";

    // Distinction Preserved Banner
    if (res.semantic_distinction_preserved && res.task_success) {
      evalOutcomeBanner.className = "outcome-banner outcome-success";
      evalOutcomeIcon.textContent = "✅";
      evalOutcomeTitle.textContent = "CRITICAL DISTINCTION PRESERVED";
      evalStatusBadge.className = "status-badge status-success";
      evalStatusBadge.textContent = "Passed (Preserved)";
    } else {
      evalOutcomeBanner.className = "outcome-banner outcome-crashed";
      evalOutcomeIcon.textContent = "🚨";
      evalOutcomeTitle.textContent = `DISTINCTION COLLAPSED (${res.primary_failure_category.toUpperCase()})`;
      evalStatusBadge.className = "status-badge status-error";
      evalStatusBadge.textContent = "Action Crashed";
    }

    evalOutcomeExplanation.textContent = res.failure_explanation;

    // Metrics Row
    metricDistinctionDisplay.innerHTML = res.semantic_distinction_preserved
      ? `<span style="color: #34d399;">PRESERVED</span>`
      : `<span style="color: #f87171;">COLLAPSED</span>`;

    metricTaskSuccessDisplay.innerHTML = res.task_success
      ? `<span style="color: #34d399;">SUCCESS</span>`
      : `<span style="color: #f87171;">FAILED</span>`;

    metricMeanWerDisplay.textContent = `${(res.average_wer * 100).toFixed(1)}%`;
    metricFieldAccDisplay.textContent = `${(res.critical_field_accuracy * 100).toFixed(0)}%`;

    // Utterance Comparison Table
    evalComparisonTableBody.innerHTML = `
      <tr>
        <td><span class="spec-badge badge-a">Utterance A</span></td>
        <td><code>${escapeHtml(res.utterance_a.reference_transcript)}</code></td>
        <td><strong>${escapeHtml(res.utterance_a.actual_transcript)}</strong></td>
        <td>${(res.utterance_a.wer * 100).toFixed(1)}% (S:${res.utterance_a.substitutions}/D:${res.utterance_a.deletions}/I:${res.utterance_a.insertions})</td>
        <td><code>${escapeHtml(JSON.stringify(res.utterance_a.extracted_fields))}</code></td>
        <td>${res.utterance_a.task_success ? '🟢 Correct' : '🔴 Wrong Action'}</td>
      </tr>
      <tr>
        <td><span class="spec-badge badge-b">Utterance B</span></td>
        <td><code>${escapeHtml(res.utterance_b.reference_transcript)}</code></td>
        <td><strong>${escapeHtml(res.utterance_b.actual_transcript)}</strong></td>
        <td>${(res.utterance_b.wer * 100).toFixed(1)}% (S:${res.utterance_b.substitutions}/D:${res.utterance_b.deletions}/I:${res.utterance_b.insertions})</td>
        <td><code>${escapeHtml(JSON.stringify(res.utterance_b.extracted_fields))}</code></td>
        <td>${res.utterance_b.task_success ? '🟢 Correct' : '🔴 Wrong Action'}</td>
      </tr>
    `;

    // Crash Analysis
    crashImpactDescription.textContent = res.failure_explanation;
    crashFieldTags.innerHTML = `
      <span class="code-pill">Category: ${res.category}</span>
      <span class="code-pill">Audio Condition: ${res.audio_condition}</span>
      <span class="code-pill">Mean WER: ${(res.average_wer * 100).toFixed(1)}%</span>
    `;

    // Telemetry footer with strict distinction between mock and live
    runIdDisplay.textContent = `Run ID: ${res.run_id}`;
    if (res.execution_mode === "mock") {
      if (res.simulated_latency_ms != null) {
        runLatencyDisplay.textContent = `Simulated Latency: ${res.simulated_latency_ms} ms (Text Simulation — 0 Network Calls)`;
      } else {
        runLatencyDisplay.textContent = "Measured Latency: null (Mock Mode — 0 Network Calls)";
      }
    } else {
      const clientLat = (res.client_request_elapsed_ms != null)
        ? `${res.client_request_elapsed_ms} ms`
        : (res.client_measured_latency_ms != null ? `${res.client_measured_latency_ms} ms` : "null");
      const provLat = res.provider_processing_time_ms != null
        ? `${res.provider_processing_time_ms} ms`
        : "null (not reported by provider)";
      runLatencyDisplay.textContent = `Client Request Elapsed: ${clientLat} | Provider Processing: ${provLat}`;
    }
    runTimestampDisplay.textContent = `Time: ${res.timestamp}`;
  }

  async function loadBenchmarkMetrics() {
    try {
      const response = await fetch("/api/eval/metrics");
      if (!response.ok) return;
      const m = await response.json();

      statTotalRuns.textContent = m.total_runs;
      statDistinctionRate.textContent = `${m.distinction_preservation_rate.toFixed(1)}%`;
      statTaskSuccessRate.textContent = `${m.task_success_rate.toFixed(1)}%`;
      statAvgWer.textContent = `${(m.average_wer * 100).toFixed(1)}%`;
    } catch (err) {
      console.error("Failed to load metrics:", err);
    }
  }

  async function loadEvaluationHistory() {
    try {
      const response = await fetch("/api/eval/results?limit=15");
      if (!response.ok) return;
      const runs = await response.json();

      if (runs.length === 0) {
        historyTableBody.innerHTML = `<tr><td colspan="10" style="text-align: center; color: var(--text-subtle); padding: 1.5rem;">No evaluation runs recorded in SQLite yet.</td></tr>`;
        return;
      }

      historyTableBody.innerHTML = runs.map((r) => `
        <tr>
          <td><code style="font-size: 0.75rem;">${r.run_id}</code></td>
          <td><strong>${r.test_case_id}</strong></td>
          <td>${r.category}</td>
          <td>${r.language_code}</td>
          <td>${r.semantic_distinction_preserved ? '<span class="badge-pill-success">PRESERVED</span>' : '<span class="badge-pill-danger">COLLAPSED</span>'}</td>
          <td>${r.task_success ? '<span class="badge-pill-success">SUCCESS</span>' : '<span class="badge-pill-danger">FAILED</span>'}</td>
          <td>${(r.average_wer * 100).toFixed(1)}%</td>
          <td><code style="font-size: 0.72rem;">${r.primary_failure_category}</code></td>
          <td><span class="lang-pill">${r.execution_mode}</span></td>
          <td style="font-size: 0.72rem; color: var(--text-muted);">${r.timestamp.split('T')[0]} ${r.timestamp.split('T')[1]?.substring(0, 8) || ''}</td>
        </tr>
      `).join("");
    } catch (err) {
      console.error("Failed to load history runs:", err);
    }
  }

  function escapeHtml(text) {
    if (!text) return "";
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
  }

  // Initial load
  updateSelectedTestCaseSpecs();
  loadBenchmarkMetrics();
});
