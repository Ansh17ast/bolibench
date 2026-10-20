/**
 * BoliBench — Frontend Interaction, Telemetry & Paired-Contrast Evaluation Handler
 * Branded: BoliBench — Indian Voice AI Crash-Test Lab
 * Multi-view Developer Tool for the Great Indian AI Internship Challenge.
 *
 * Implements:
 * 1. 6-Tab Navigation (Overview Dashboard, Paired Crash Lab, Catalogue, History, Methodology, Single-Clip Lab)
 * 2. Strict metric separation between mock demonstrations and live Gnani evaluations
 * 3. Audio upload validation, in-browser playback previews for Utterance A & B
 * 4. Transparent latency telemetry separating client round-trip from provider-reported processing
 * 5. Full audit history with status filtering (all, completed, failed, incomplete)
 * 6. Benchmark catalogue browsing with text-only simulation labeling
 */

document.addEventListener("DOMContentLoaded", () => {
  // ===========================================================================
  // 1. Tab Navigation
  // ===========================================================================
  const tabs = {
    overviewView: document.getElementById("tabBtnOverview"),
    evaluateView: document.getElementById("tabBtnEvaluate"),
    catalogueView: document.getElementById("tabBtnCatalogue"),
    historyView: document.getElementById("tabBtnHistory"),
    methodologyView: document.getElementById("tabBtnMethodology"),
    transcribeView: document.getElementById("tabBtnTranscribe"),
  };

  const panes = {
    overviewView: document.getElementById("overviewView"),
    evaluateView: document.getElementById("evaluateView"),
    catalogueView: document.getElementById("catalogueView"),
    historyView: document.getElementById("historyView"),
    methodologyView: document.getElementById("methodologyView"),
    transcribeView: document.getElementById("transcribeView"),
  };

  let activeTabKey = "overviewView";

  function switchTab(targetKey) {
    if (!panes[targetKey]) return;
    activeTabKey = targetKey;

    Object.keys(tabs).forEach((key) => {
      const btn = tabs[key];
      const pane = panes[key];
      if (!btn || !pane) return;

      const isTarget = key === targetKey;
      if (isTarget) {
        btn.classList.add("active");
        btn.setAttribute("aria-selected", "true");
        pane.style.display = "block";
      } else {
        btn.classList.remove("active");
        btn.setAttribute("aria-selected", "false");
        pane.style.display = "none";
      }
    });

    // Tab-specific data loading
    if (targetKey === "overviewView") {
      loadBenchmarkMetrics();
    } else if (targetKey === "evaluateView") {
      loadBenchmarkMetrics();
      loadEvaluationHistory();
      updateSelectedTestCaseSpecs();
    } else if (targetKey === "catalogueView") {
      loadBenchmarkCatalogue();
    } else if (targetKey === "historyView") {
      loadFullHistory(currentStatusFilter);
    }
  }

  Object.keys(tabs).forEach((key) => {
    const btn = tabs[key];
    if (btn) {
      btn.addEventListener("click", () => switchTab(key));
    }
  });

  // ===========================================================================
  // 2. Overview Dashboard & Metric Separation
  // ===========================================================================
  const btnRefreshMetrics = document.getElementById("btnRefreshMetrics");
  if (btnRefreshMetrics) {
    btnRefreshMetrics.addEventListener("click", () => {
      loadBenchmarkMetrics();
    });
  }

  const healthTotalRuns = document.getElementById("healthTotalRuns");
  const healthCompletedRuns = document.getElementById("healthCompletedRuns");
  const healthIncompleteRuns = document.getElementById("healthIncompleteRuns");
  const healthFailedRuns = document.getElementById("healthFailedRuns");
  const healthLiveCalls = document.getElementById("healthLiveCalls");
  const healthMockRuns = document.getElementById("healthMockRuns");

  const liveAttemptsPill = document.getElementById("liveAttemptsPill");
  const liveEmptyStateBox = document.getElementById("liveEmptyStateBox");
  const liveStatsGrid = document.getElementById("liveStatsGrid");
  const liveStatDistinctionRate = document.getElementById("liveStatDistinctionRate");
  const liveStatTaskSuccessRate = document.getElementById("liveStatTaskSuccessRate");
  const liveStatAvgWer = document.getElementById("liveStatAvgWer");
  const liveStatFieldAcc = document.getElementById("liveStatFieldAcc");
  const liveStatCompleted = document.getElementById("liveStatCompleted");
  const liveStatIncomplete = document.getElementById("liveStatIncomplete");
  const liveStatFailed = document.getElementById("liveStatFailed");

  const mockAttemptsPill = document.getElementById("mockAttemptsPill");
  const mockEmptyStateBox = document.getElementById("mockEmptyStateBox");
  const mockStatsGrid = document.getElementById("mockStatsGrid");
  const mockStatDistinctionRate = document.getElementById("mockStatDistinctionRate");
  const mockStatTaskSuccessRate = document.getElementById("mockStatTaskSuccessRate");
  const mockStatAvgWer = document.getElementById("mockStatAvgWer");
  const mockStatFieldAcc = document.getElementById("mockStatFieldAcc");
  const mockStatCompleted = document.getElementById("mockStatCompleted");
  const mockStatFailed = document.getElementById("mockStatFailed");

  const categoryBreakdownBody = document.getElementById("categoryBreakdownBody");

  // Paired Lab Summary Cards (Backward-compatible)
  const statTotalRuns = document.getElementById("statTotalRuns");
  const statDistinctionRate = document.getElementById("statDistinctionRate");
  const statTaskSuccessRate = document.getElementById("statTaskSuccessRate");
  const statAvgWer = document.getElementById("statAvgWer");

  async function loadBenchmarkMetrics() {
    try {
      const response = await fetch("/api/eval/metrics");
      if (!response.ok) return;
      const m = await response.json();

      // Overall Health Ribbon
      if (healthTotalRuns) healthTotalRuns.textContent = m.total_runs;
      if (healthCompletedRuns) healthCompletedRuns.textContent = m.completed_runs;
      if (healthIncompleteRuns) healthIncompleteRuns.textContent = m.incomplete_runs;
      if (healthFailedRuns) healthFailedRuns.textContent = m.failed_runs;
      if (healthLiveCalls) healthLiveCalls.textContent = m.live_runs;
      if (healthMockRuns) healthMockRuns.textContent = m.mock_runs;

      // Backward-compatible Lab stats
      if (statTotalRuns) statTotalRuns.textContent = m.total_runs;
      if (statDistinctionRate) {
        statDistinctionRate.textContent = m.distinction_preservation_rate != null ? `${m.distinction_preservation_rate.toFixed(1)}%` : "—%";
      }
      if (statTaskSuccessRate) {
        statTaskSuccessRate.textContent = m.task_success_rate != null ? `${m.task_success_rate.toFixed(1)}%` : "—%";
      }
      if (statAvgWer) {
        statAvgWer.textContent = m.average_wer != null ? `${(m.average_wer * 100).toFixed(1)}%` : "—";
      }

      // 1. LIVE GNANI EVALUATION METRICS
      if (m.live) {
        if (liveAttemptsPill) liveAttemptsPill.textContent = `${m.live.total_attempts} Attempts`;
        if (liveStatCompleted) liveStatCompleted.textContent = m.live.completed_runs;
        if (liveStatIncomplete) liveStatIncomplete.textContent = m.live.incomplete_runs;
        if (liveStatFailed) liveStatFailed.textContent = m.live.failed_runs;

        if (m.live.completed_runs > 0) {
          if (liveEmptyStateBox) liveEmptyStateBox.style.display = "none";
          if (liveStatsGrid) liveStatsGrid.style.display = "grid";

          if (liveStatDistinctionRate) {
            liveStatDistinctionRate.textContent = m.live.distinction_preservation_rate != null
              ? `${m.live.distinction_preservation_rate.toFixed(1)}%`
              : "—%";
          }
          if (liveStatTaskSuccessRate) {
            liveStatTaskSuccessRate.textContent = m.live.task_success_rate != null
              ? `${m.live.task_success_rate.toFixed(1)}%`
              : "—%";
          }
          if (liveStatAvgWer) {
            liveStatAvgWer.textContent = m.live.average_wer != null
              ? `${(m.live.average_wer * 100).toFixed(1)}%`
              : "—";
          }
          if (liveStatFieldAcc) {
            liveStatFieldAcc.textContent = m.live.average_critical_field_accuracy != null
              ? `${m.live.average_critical_field_accuracy.toFixed(1)}%`
              : "—%";
          }
        } else {
          if (liveEmptyStateBox) liveEmptyStateBox.style.display = "block";
          if (liveStatsGrid) liveStatsGrid.style.display = "none";
        }
      }

      // 2. MOCK DEMONSTRATION METRICS
      if (m.mock) {
        if (mockAttemptsPill) mockAttemptsPill.textContent = `${m.mock.total_attempts} Attempts`;
        if (mockStatCompleted) mockStatCompleted.textContent = m.mock.completed_runs;
        if (mockStatFailed) mockStatFailed.textContent = m.mock.failed_runs;

        if (m.mock.completed_runs > 0) {
          if (mockEmptyStateBox) mockEmptyStateBox.style.display = "none";
          if (mockStatsGrid) mockStatsGrid.style.display = "grid";

          if (mockStatDistinctionRate) {
            mockStatDistinctionRate.textContent = m.mock.distinction_preservation_rate != null
              ? `${m.mock.distinction_preservation_rate.toFixed(1)}%`
              : "—%";
          }
          if (mockStatTaskSuccessRate) {
            mockStatTaskSuccessRate.textContent = m.mock.task_success_rate != null
              ? `${m.mock.task_success_rate.toFixed(1)}%`
              : "—%";
          }
          if (mockStatAvgWer) {
            mockStatAvgWer.textContent = m.mock.average_wer != null
              ? `${(m.mock.average_wer * 100).toFixed(1)}%`
              : "—";
          }
          if (mockStatFieldAcc) {
            mockStatFieldAcc.textContent = m.mock.average_critical_field_accuracy != null
              ? `${m.mock.average_critical_field_accuracy.toFixed(1)}%`
              : "—%";
          }
        } else {
          if (mockEmptyStateBox) mockEmptyStateBox.style.display = "block";
          if (mockStatsGrid) mockStatsGrid.style.display = "none";
        }
      }

      // 3. CATEGORY BREAKDOWN TABLE
      renderCategoryBreakdown(m);
    } catch (err) {
      console.error("Failed to load metrics:", err);
    }
  }

  function renderCategoryBreakdown(metrics) {
    if (!categoryBreakdownBody) return;
    const catData = metrics.category_breakdown || {};
    const categories = Object.keys(catData);

    if (categories.length === 0) {
      categoryBreakdownBody.innerHTML = `
        <tr>
          <td colspan="8" class="table-empty-cell">No evaluation runs recorded in SQLite yet. Execute a test case in the Crash Lab to view linguistic breakdown.</td>
        </tr>
      `;
      return;
    }

    const rowsHtml = categories.map((cat) => {
      const row = catData[cat];
      const distStr = row.distinction_rate != null ? `${row.distinction_rate.toFixed(1)}%` : "—%";
      const taskStr = row.task_success_rate != null ? `${row.task_success_rate.toFixed(1)}%` : "—%";
      const werStr = row.average_wer != null ? `${(row.average_wer * 100).toFixed(1)}%` : "—";
      const failedCount = (row.failed || 0) + (row.incomplete || 0);

      // Check mode breakdowns if available
      const mockCat = metrics.mock?.category_breakdown?.[cat];
      const liveCat = metrics.live?.category_breakdown?.[cat];
      let modeBadges = [];
      if (liveCat && liveCat.total > 0) modeBadges.push('<span class="badge-live">LIVE</span>');
      if (mockCat && mockCat.total > 0) modeBadges.push('<span class="badge-mock">MOCK</span>');
      const modeHtml = modeBadges.length > 0 ? modeBadges.join(" ") : '<span class="count-tag">NONE</span>';

      return `
        <tr>
          <td><strong>${escapeHtml(cat)}</strong></td>
          <td>${modeHtml}</td>
          <td>${row.total}</td>
          <td><span style="color: #34d399; font-weight: 700;">${row.completed}</span></td>
          <td><span style="color: ${failedCount > 0 ? '#f87171' : 'var(--text-subtle)'}; font-weight: 700;">${failedCount}</span></td>
          <td>${distStr}</td>
          <td>${taskStr}</td>
          <td>${werStr}</td>
        </tr>
      `;
    }).join("");

    categoryBreakdownBody.innerHTML = rowsHtml;
  }

  // ===========================================================================
  // 3. Paired-Contrast Crash Lab Execution & Audio Previews
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
  const btnToggleLiveEval = document.getElementById("btnToggleLiveEval");

  const liveEvalDrawer = document.getElementById("liveEvalDrawer");
  const liveRefPromptA = document.getElementById("liveRefPromptA");
  const liveRefPromptB = document.getElementById("liveRefPromptB");
  const liveAudioInputA = document.getElementById("liveAudioInputA");
  const liveAudioInputB = document.getElementById("liveAudioInputB");
  const previewContainerA = document.getElementById("previewContainerA");
  const previewContainerB = document.getElementById("previewContainerB");
  const liveAudioPreviewA = document.getElementById("liveAudioPreviewA");
  const liveAudioPreviewB = document.getElementById("liveAudioPreviewB");
  const fileNameA = document.getElementById("fileNameA");
  const fileNameB = document.getElementById("fileNameB");
  const fileSizeA = document.getElementById("fileSizeA");
  const fileSizeB = document.getElementById("fileSizeB");

  const confirmLiveCreditUsage = document.getElementById("confirmLiveCreditUsage");
  const btnExecuteLiveEval = document.getElementById("btnExecuteLiveEval");
  const btnLiveEvalText = document.getElementById("btnLiveEvalText");
  const btnLiveSpinner = document.getElementById("btnLiveSpinner");

  const customTranscriptDrawer = document.getElementById("customTranscriptDrawer");
  const customTranscriptA = document.getElementById("customTranscriptA");
  const customTranscriptB = document.getElementById("customTranscriptB");
  const btnScoreCustomTranscripts = document.getElementById("btnScoreCustomTranscripts");

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
  const transliterationNote = document.getElementById("transliterationNote");
  const crashImpactDescription = document.getElementById("crashImpactDescription");
  const crashFieldTags = document.getElementById("crashFieldTags");

  const runIdDisplay = document.getElementById("runIdDisplay");
  const runLatencyDisplay = document.getElementById("runLatencyDisplay");
  const runTimestampDisplay = document.getElementById("runTimestampDisplay");

  let currentLiveAudioUrlA = null;
  let currentLiveAudioUrlB = null;

  function formatSize(bytes) {
    if (!bytes || bytes <= 0) return "0 KB";
    const kb = (bytes / 1024).toFixed(1);
    if (bytes >= 1024 * 1024) {
      return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
    }
    return `${kb} KB`;
  }

  // Audio Playback Previews for Utterance A & B
  if (liveAudioInputA) {
    liveAudioInputA.addEventListener("change", (e) => {
      const file = e.target.files && e.target.files[0];
      if (currentLiveAudioUrlA) {
        URL.revokeObjectURL(currentLiveAudioUrlA);
        currentLiveAudioUrlA = null;
      }

      if (file) {
        fileNameA.textContent = file.name;
        fileSizeA.textContent = formatSize(file.size);
        currentLiveAudioUrlA = URL.createObjectURL(file);
        liveAudioPreviewA.src = currentLiveAudioUrlA;
        previewContainerA.style.display = "flex";
      } else {
        previewContainerA.style.display = "none";
        liveAudioPreviewA.removeAttribute("src");
      }
      checkLiveEvalFormValidity();
    });
  }

  if (liveAudioInputB) {
    liveAudioInputB.addEventListener("change", (e) => {
      const file = e.target.files && e.target.files[0];
      if (currentLiveAudioUrlB) {
        URL.revokeObjectURL(currentLiveAudioUrlB);
        currentLiveAudioUrlB = null;
      }

      if (file) {
        fileNameB.textContent = file.name;
        fileSizeB.textContent = formatSize(file.size);
        currentLiveAudioUrlB = URL.createObjectURL(file);
        liveAudioPreviewB.src = currentLiveAudioUrlB;
        previewContainerB.style.display = "flex";
      } else {
        previewContainerB.style.display = "none";
        liveAudioPreviewB.removeAttribute("src");
      }
      checkLiveEvalFormValidity();
    });
  }

  function checkLiveEvalFormValidity() {
    if (!btnExecuteLiveEval) return;
    const hasA = liveAudioInputA && liveAudioInputA.files && liveAudioInputA.files.length > 0;
    const hasB = liveAudioInputB && liveAudioInputB.files && liveAudioInputB.files.length > 0;
    const isConfirmed = confirmLiveCreditUsage && confirmLiveCreditUsage.checked;
    btnExecuteLiveEval.disabled = !(hasA && hasB && isConfirmed);
  }

  if (confirmLiveCreditUsage) {
    confirmLiveCreditUsage.addEventListener("change", checkLiveEvalFormValidity);
  }

  async function updateSelectedTestCaseSpecs() {
    if (!evalTestCaseSelect) return;
    const testId = evalTestCaseSelect.value;
    if (!testId) return;

    try {
      const response = await fetch(`/api/eval/test-cases/${testId}`);
      if (!response.ok) return;
      const tc = await response.json();

      if (testCategoryBadge) testCategoryBadge.textContent = tc.category;
      if (testDescriptionText) testDescriptionText.textContent = tc.description;

      if (specRefA) specRefA.textContent = `"${tc.utterance_a.reference_transcript}"`;
      if (specIntentA) specIntentA.textContent = tc.utterance_a.expected_intent;
      if (specFieldsA) specFieldsA.textContent = JSON.stringify(tc.utterance_a.expected_critical_fields);
      if (specActionA) specActionA.textContent = tc.utterance_a.expected_action;
      if (specLangA) specLangA.textContent = tc.language_code;

      if (specRefB) specRefB.textContent = `"${tc.utterance_b.reference_transcript}"`;
      if (specIntentB) specIntentB.textContent = tc.utterance_b.expected_intent;
      if (specFieldsB) specFieldsB.textContent = JSON.stringify(tc.utterance_b.expected_critical_fields);
      if (specActionB) specActionB.textContent = tc.utterance_b.expected_action;
      if (specLangB) specLangB.textContent = tc.language_code;

      if (specDistinctionField) specDistinctionField.textContent = `Critical Distinction: ${tc.distinction_field}`;
      if (specExpectedDistinctionText) specExpectedDistinctionText.textContent = tc.expected_distinction;

      if (liveRefPromptA) liveRefPromptA.textContent = tc.utterance_a.reference_transcript;
      if (liveRefPromptB) liveRefPromptB.textContent = tc.utterance_b.reference_transcript;

      // Populate custom inputs as starting template
      if (customTranscriptA) customTranscriptA.value = tc.utterance_a.reference_transcript;
      if (customTranscriptB) customTranscriptB.value = tc.utterance_b.reference_transcript;
    } catch (err) {
      console.error("Failed to load test case details:", err);
    }
  }

  if (evalTestCaseSelect) {
    evalTestCaseSelect.addEventListener("change", updateSelectedTestCaseSpecs);
  }

  if (btnToggleCustomInputs) {
    btnToggleCustomInputs.addEventListener("click", () => {
      const isHidden = customTranscriptDrawer.style.display === "none";
      customTranscriptDrawer.style.display = isHidden ? "flex" : "none";
      btnToggleCustomInputs.textContent = isHidden ? "Hide Custom Inputs" : "Custom Transcript Test";
      if (isHidden && liveEvalDrawer) {
        liveEvalDrawer.style.display = "none";
        btnToggleLiveEval.textContent = "🎙️ Live Audio Pair (WAV • Consumes Credits)";
      }
    });
  }

  if (btnToggleLiveEval) {
    btnToggleLiveEval.addEventListener("click", () => {
      const isHidden = liveEvalDrawer.style.display === "none";
      liveEvalDrawer.style.display = isHidden ? "flex" : "none";
      btnToggleLiveEval.textContent = isHidden ? "Hide Live Audio Pair" : "🎙️ Live Audio Pair (WAV • Consumes Credits)";
      if (isHidden && customTranscriptDrawer) {
        customTranscriptDrawer.style.display = "none";
        btnToggleCustomInputs.textContent = "Custom Transcript Test";
      }
    });
  }

  // Deterministic Mock Execution
  async function executeMockScenario(scenario) {
    const testId = evalTestCaseSelect.value;
    if (!testId) return;

    evalStatusBadge.className = "status-badge status-running";
    evalStatusBadge.textContent = "Executing Mock Evaluation...";

    try {
      const response = await fetch(`/api/eval/run-mock/${testId}?scenario=${encodeURIComponent(scenario)}`, {
        method: "POST",
      });

      if (!response.ok) {
        throw new Error(`Mock evaluation failed (HTTP ${response.status})`);
      }

      const result = await response.json();
      displayEvaluationScorecard(result);
      loadBenchmarkMetrics();
      loadEvaluationHistory();
    } catch (err) {
      console.error("Mock evaluation error:", err);
      evalStatusBadge.className = "status-badge status-error";
      evalStatusBadge.textContent = "Execution Error";
      alert(`Mock evaluation error: ${err.message}`);
    }
  }

  if (btnRunMockIdeal) {
    btnRunMockIdeal.addEventListener("click", () => executeMockScenario("ideal_preserved"));
  }

  if (btnRunMockCrashed) {
    btnRunMockCrashed.addEventListener("click", () => executeMockScenario("crashed_distinction"));
  }

  // Custom Spoken Transcript Scoring
  if (btnScoreCustomTranscripts) {
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
  }

  // Live Paired Evaluation Execution with Strict Safeguards
  if (btnExecuteLiveEval) {
    btnExecuteLiveEval.addEventListener("click", async () => {
      const testId = evalTestCaseSelect.value;
      if (!testId) return;

      if (!confirmLiveCreditUsage.checked) {
        alert("Please confirm the credit usage authorization before executing live evaluation.");
        return;
      }

      if (!liveAudioInputA.files || !liveAudioInputA.files[0] || !liveAudioInputB.files || !liveAudioInputB.files[0]) {
        alert("Please select WAV audio recordings for both Utterance A and Utterance B.");
        return;
      }

      // Explicit confirmation protecting programme credits
      const userConfirmed = window.confirm(
        "Live Evaluation Credit Authorization:\n\n" +
        "Evaluating this pair makes 2 real requests to Gnani Prisma STT v3 and consumes account credits.\n\n" +
        "Duration policy: Each clip must be a valid WAV file strictly <= 60 seconds.\n\n" +
        "Do you authorize 2 live requests for Utterance A and Utterance B?"
      );
      if (!userConfirmed) return;

      btnExecuteLiveEval.disabled = true;
      btnLiveSpinner.style.display = "inline-block";
      btnLiveEvalText.textContent = "Authorizing Live Evaluation...";
      evalStatusBadge.className = "status-badge status-running";
      evalStatusBadge.textContent = "Authorizing Credit Guard...";

      try {
        // Step 1: Request single-use authorization token
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

        // Step 2: Build multipart payload
        const liveAudioConditionSelect = document.getElementById("liveAudioCondition");
        const selectedCondition = liveAudioConditionSelect ? liveAudioConditionSelect.value : "clean_recorded";

        const formData = new FormData();
        formData.append("audio_file_a", liveAudioInputA.files[0]);
        formData.append("audio_file_b", liveAudioInputB.files[0]);
        formData.append("audio_condition", selectedCondition);
        formData.append("confirmation_token", authToken);

        btnLiveEvalText.textContent = "Sending Audio A to Gnani Prisma...";
        evalStatusBadge.textContent = "Running Live STT Evaluation...";

        const evalResp = await fetch(`/api/eval/run-live/${testId}`, {
          method: "POST",
          headers: {
            "X-Requested-With": "XMLHttpRequest",
            "X-BoliBench-Confirm-Live": "authorized-by-local-operator",
          },
          body: formData,
        });

        if (!evalResp.ok) {
          const evalErr = await evalResp.json().catch(() => ({}));
          throw new Error(evalErr.detail || `Live evaluation failed (HTTP ${evalResp.status})`);
        }

        const evalResult = await evalResp.json();
        displayEvaluationScorecard(evalResult);
        loadBenchmarkMetrics();
        loadEvaluationHistory();
      } catch (err) {
        console.error("Live paired evaluation failed:", err);
        evalStatusBadge.className = "status-badge status-error";
        evalStatusBadge.textContent = "Live Evaluation Failed";
        alert(`Live Evaluation Error:\n\n${err.message}`);
        loadBenchmarkMetrics();
        loadEvaluationHistory();
      } finally {
        btnExecuteLiveEval.disabled = false;
        btnLiveSpinner.style.display = "none";
        btnLiveEvalText.textContent = "Execute Paired Live Evaluation";
        checkLiveEvalFormValidity();
      }
    });
  }

  // ===========================================================================
  // 4. Evaluation Scorecard & Results Analysis
  // ===========================================================================
  function displayEvaluationScorecard(res) {
    if (!evalResultsPlaceholder || !evalResultsContainer) return;

    evalResultsPlaceholder.style.display = "none";
    evalResultsContainer.style.display = "block";

    // Mode Pill
    const isLive = res.execution_mode === "live";
    evalExecModePill.textContent = isLive ? "LIVE GNANI EVALUATION" : "MOCK EVALUATION";
    evalExecModePill.className = isLive ? "exec-mode-pill badge-live" : "exec-mode-pill badge-mock";

    // Status Banner
    if (res.overall_status === "failed") {
      evalOutcomeBanner.className = "outcome-banner outcome-crashed";
      evalOutcomeIcon.textContent = "⚠️";
      evalOutcomeTitle.textContent = "OPERATIONAL PROVIDER FAILURE (ABORTED)";
      evalStatusBadge.className = "status-badge status-error";
      evalStatusBadge.textContent = "Operational Failure";
    } else if (res.overall_status === "incomplete") {
      evalOutcomeBanner.className = "outcome-banner outcome-crashed";
      evalOutcomeIcon.textContent = "⚠️";
      evalOutcomeTitle.textContent = "PARTIAL LIVE FAILURE (INCOMPLETE RUN)";
      evalStatusBadge.className = "status-badge status-warning";
      evalStatusBadge.textContent = "Incomplete Run";
    } else if (res.semantic_distinction_preserved && res.task_success) {
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
    if (res.overall_status !== "completed") {
      metricDistinctionDisplay.innerHTML = `<span class="val-unavailable">Unavailable</span>`;
      metricTaskSuccessDisplay.innerHTML = `<span class="val-unavailable">Unavailable</span>`;
      metricMeanWerDisplay.innerHTML = `<span class="val-unavailable">Unavailable</span>`;
      metricFieldAccDisplay.innerHTML = `<span class="val-unavailable">Unavailable</span>`;
    } else {
      metricDistinctionDisplay.innerHTML = res.semantic_distinction_preserved
        ? `<span style="color: #34d399;">PRESERVED</span>`
        : `<span style="color: #f87171;">COLLAPSED</span>`;

      metricTaskSuccessDisplay.innerHTML = res.task_success
        ? `<span style="color: #34d399;">SUCCESS</span>`
        : `<span style="color: #f87171;">FAILED</span>`;

      metricMeanWerDisplay.textContent = `${(res.average_wer * 100).toFixed(1)}%`;
      metricFieldAccDisplay.textContent = `${(res.critical_field_accuracy * 100).toFixed(0)}%`;
    }

    // Comparison Table
    evalComparisonTableBody.innerHTML = `
      <tr>
        <td><span class="spec-badge badge-a">Utterance A</span></td>
        <td><code>${escapeHtml(res.utterance_a.reference_transcript)}</code></td>
        <td><strong>${escapeHtml(res.utterance_a.actual_transcript || '—')}</strong></td>
        <td>${res.utterance_a.status === 'completed' ? `${(res.utterance_a.wer * 100).toFixed(1)}% (S:${res.utterance_a.substitutions}/D:${res.utterance_a.deletions}/I:${res.utterance_a.insertions})` : `<span class="val-unavailable">${res.utterance_a.status}</span>`}</td>
        <td><code>${escapeHtml(JSON.stringify(res.utterance_a.extracted_fields))}</code></td>
        <td>${res.utterance_a.status === 'completed' ? (res.utterance_a.task_success ? '🟢 Correct' : '🔴 Wrong Action') : `<span class="val-unavailable">${res.utterance_a.status}</span>`}</td>
      </tr>
      <tr>
        <td><span class="spec-badge badge-b">Utterance B</span></td>
        <td><code>${escapeHtml(res.utterance_b.reference_transcript)}</code></td>
        <td><strong>${escapeHtml(res.utterance_b.actual_transcript || '—')}</strong></td>
        <td>${res.utterance_b.status === 'completed' ? `${(res.utterance_b.wer * 100).toFixed(1)}% (S:${res.utterance_b.substitutions}/D:${res.utterance_b.deletions}/I:${res.utterance_b.insertions})` : `<span class="val-unavailable">${res.utterance_b.status}</span>`}</td>
        <td><code>${escapeHtml(JSON.stringify(res.utterance_b.extracted_fields))}</code></td>
        <td>${res.utterance_b.status === 'completed' ? (res.utterance_b.task_success ? '🟢 Correct' : '🔴 Wrong Action') : `<span class="val-unavailable">${res.utterance_b.status}</span>`}</td>
      </tr>
    `;

    // Script & Transliteration check
    if (transliterationNote) {
      const hasDevanagariHyp = /[\u0900-\u097F]/.test(res.utterance_a.actual_transcript || "") || /[\u0900-\u097F]/.test(res.utterance_b.actual_transcript || "");
      const hasLatinRef = /^[A-Za-z0-9\s.,?!'"-]+$/.test(res.utterance_a.reference_transcript);
      if (hasDevanagariHyp && hasLatinRef) {
        transliterationNote.style.display = "flex";
      } else {
        transliterationNote.style.display = "none";
      }
    }

    // Crash Impact Description
    crashImpactDescription.textContent = res.failure_explanation;
    crashFieldTags.innerHTML = `
      <span class="code-pill">Category: ${res.category}</span>
      <span class="code-pill">Audio Condition: ${res.audio_condition}</span>
      <span class="code-pill">Status: ${res.overall_status.toUpperCase()}</span>
      ${res.overall_status === 'completed' ? `<span class="code-pill">Mean WER: ${(res.average_wer * 100).toFixed(1)}%</span>` : ''}
    `;

    // Latency & Telemetry Separation
    runIdDisplay.textContent = `Run ID: ${res.run_id}`;
    if (res.execution_mode === "mock") {
      if (res.simulated_latency_ms != null) {
        runLatencyDisplay.textContent = `Simulated Latency: ${res.simulated_latency_ms} ms (Text Simulation — 0 Network Calls)`;
      } else {
        runLatencyDisplay.textContent = "Measured Latency: null (Mock Mode — 0 Network Calls)";
      }
    } else {
      const clientLat = res.client_request_elapsed_ms != null
        ? `${res.client_request_elapsed_ms} ms`
        : (res.client_measured_latency_ms != null ? `${res.client_measured_latency_ms} ms` : "null");
      const provLat = res.provider_processing_time_ms != null
        ? `${res.provider_processing_time_ms} ms`
        : "Not reported";
      runLatencyDisplay.textContent = `Client Elapsed: ${clientLat} | Provider Processing: ${provLat}`;
    }
    runTimestampDisplay.textContent = `Time: ${res.timestamp}`;
  }

  // ===========================================================================
  // 5. Benchmark Catalogue View
  // ===========================================================================
  const catalogueSearchInput = document.getElementById("catalogueSearchInput");
  const catalogueCategoryFilter = document.getElementById("catalogueCategoryFilter");
  const catalogueGrid = document.getElementById("catalogueGrid");
  const catalogueCountBadge = document.getElementById("catalogueCountBadge");

  let allTestCases = [];

  async function loadBenchmarkCatalogue() {
    if (!catalogueGrid) return;
    try {
      const response = await fetch("/api/eval/test-cases");
      if (!response.ok) return;
      allTestCases = await response.json();
      renderCatalogue();
    } catch (err) {
      console.error("Failed to load benchmark catalogue:", err);
    }
  }

  function renderCatalogue() {
    if (!catalogueGrid) return;
    const search = catalogueSearchInput ? catalogueSearchInput.value.toLowerCase().trim() : "";
    const cat = catalogueCategoryFilter ? catalogueCategoryFilter.value : "all";

    const filtered = allTestCases.filter((tc) => {
      const matchesCat = cat === "all" || tc.category === cat;
      const matchesSearch = !search ||
        tc.id.toLowerCase().includes(search) ||
        tc.description.toLowerCase().includes(search) ||
        tc.utterance_a.reference_transcript.toLowerCase().includes(search) ||
        tc.utterance_b.reference_transcript.toLowerCase().includes(search);
      return matchesCat && matchesSearch;
    });

    if (catalogueCountBadge) {
      catalogueCountBadge.textContent = `${filtered.length} of ${allTestCases.length} Test Cases`;
    }

    if (filtered.length === 0) {
      catalogueGrid.innerHTML = `
        <div style="grid-column: 1 / -1; text-align: center; padding: 3rem 1.5rem; color: var(--text-muted);">
          <p>No test cases match your search filter.</p>
        </div>
      `;
      return;
    }

    catalogueGrid.innerHTML = filtered.map((tc) => `
      <div class="catalogue-card-item">
        <div class="cat-header">
          <span class="cat-id">${tc.id}</span>
          <div class="cat-tag-row">
            <span class="lang-tag">${tc.language_code}</span>
            <span class="sim-badge" title="Hand-authored minimal contrast pair for semantic crash testing; not an observed speech error">TEXT-ONLY SIMULATION</span>
          </div>
        </div>
        <p class="cat-desc">${escapeHtml(tc.description)}</p>
        <div class="cat-contrast-box">
          <div class="cat-utt">
            <span class="cat-utt-label">Utterance A</span>
            <span class="cat-utt-ref">"${escapeHtml(tc.utterance_a.reference_transcript)}"</span>
            <code style="font-size: 0.72rem; color: #93c5fd;">Action: ${tc.utterance_a.expected_action}</code>
          </div>
          <div class="cat-utt">
            <span class="cat-utt-label">Utterance B (Contrast)</span>
            <span class="cat-utt-ref">"${escapeHtml(tc.utterance_b.reference_transcript)}"</span>
            <code style="font-size: 0.72rem; color: #93c5fd;">Action: ${tc.utterance_b.expected_action}</code>
          </div>
        </div>
        <div class="cat-distinction">
          <strong>Distinction (${tc.distinction_field}):</strong> ${escapeHtml(tc.expected_distinction)}
        </div>
        <div class="cat-footer">
          <span class="provenance-tag">${tc.provenance || 'hand_authored_text_simulation'}</span>
          <button type="button" class="btn-cat-test" data-test-id="${tc.id}">Test in Crash Lab ⚡</button>
        </div>
      </div>
    `).join("");

    // Wire "Test in Crash Lab" buttons
    catalogueGrid.querySelectorAll(".btn-cat-test").forEach((btn) => {
      btn.addEventListener("click", () => {
        const targetId = btn.getAttribute("data-test-id");
        if (evalTestCaseSelect) {
          evalTestCaseSelect.value = targetId;
        }
        switchTab("evaluateView");
      });
    });
  }

  if (catalogueSearchInput) {
    catalogueSearchInput.addEventListener("input", renderCatalogue);
  }
  if (catalogueCategoryFilter) {
    catalogueCategoryFilter.addEventListener("change", renderCatalogue);
  }

  // ===========================================================================
  // 6. Full History & Operational Status Audit Trail
  // ===========================================================================
  const historyFullTableBody = document.getElementById("historyFullTableBody");
  const historyTableBody = document.getElementById("historyTableBody");
  let currentStatusFilter = "all";

  const countFilterAll = document.getElementById("countFilterAll");
  const countFilterCompleted = document.getElementById("countFilterCompleted");
  const countFilterFailed = document.getElementById("countFilterFailed");
  const countFilterIncomplete = document.getElementById("countFilterIncomplete");

  const statusFilterBtns = document.querySelectorAll(".status-filter-btn");
  statusFilterBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      statusFilterBtns.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      currentStatusFilter = btn.getAttribute("data-status");
      loadFullHistory(currentStatusFilter);
    });
  });

  async function loadFullHistory(statusFilter = "all") {
    if (!historyFullTableBody && !historyTableBody) return;
    try {
      const url = statusFilter && statusFilter !== "all"
        ? `/api/eval/results?limit=100&status=${encodeURIComponent(statusFilter)}`
        : `/api/eval/results?limit=100`;

      const response = await fetch(url);
      if (!response.ok) return;
      const runs = await response.json();

      // Fetch overall counts for pills if viewing all
      if (statusFilter === "all") {
        updateHistoryFilterCounts(runs);
      }

      renderHistoryRows(historyFullTableBody, runs, true);
      renderHistoryRows(historyTableBody, runs.slice(0, 15), false);
    } catch (err) {
      console.error("Failed to load full history:", err);
    }
  }

  function updateHistoryFilterCounts(allRuns) {
    let completed = 0;
    let failed = 0;
    let incomplete = 0;

    allRuns.forEach((r) => {
      const st = r.overall_status || "completed";
      if (st === "completed") completed++;
      else if (st === "failed") failed++;
      else if (st === "incomplete") incomplete++;
    });

    if (countFilterAll) countFilterAll.textContent = allRuns.length;
    if (countFilterCompleted) countFilterCompleted.textContent = completed;
    if (countFilterFailed) countFilterFailed.textContent = failed;
    if (countFilterIncomplete) countFilterIncomplete.textContent = incomplete;
  }

  function renderHistoryRows(tbody, runs, isFullView) {
    if (!tbody) return;

    if (runs.length === 0) {
      tbody.innerHTML = `
        <tr>
          <td colspan="${isFullView ? 10 : 11}" class="table-empty-cell">
            No evaluation runs found matching status filter '${currentStatusFilter}'.
          </td>
        </tr>
      `;
      return;
    }

    tbody.innerHTML = runs.map((r) => {
      const status = r.overall_status || "completed";
      let statusBadge = `<span class="status-pill-completed">COMPLETED</span>`;
      if (status === "failed") {
        statusBadge = `<span class="status-pill-failed">FAILED</span>`;
      } else if (status === "incomplete") {
        statusBadge = `<span class="status-pill-incomplete">INCOMPLETE</span>`;
      }

      // Neutral display for unavailable metrics on failed/incomplete runs
      let distinctionHtml = `<span class="val-unavailable">Unavailable</span>`;
      let taskHtml = `<span class="val-unavailable">Unavailable</span>`;
      let werHtml = `<span class="val-unavailable">Unavailable</span>`;

      if (status === "completed") {
        distinctionHtml = r.semantic_distinction_preserved
          ? `<span class="badge-pill-success">PRESERVED</span>`
          : `<span class="badge-pill-danger">COLLAPSED</span>`;
        taskHtml = r.task_success
          ? `<span class="badge-pill-success">SUCCESS</span>`
          : `<span class="badge-pill-danger">FAILED</span>`;
        werHtml = `${(r.average_wer * 100).toFixed(1)}%`;
      }

      const modeBadge = r.execution_mode === "live"
        ? `<span class="badge-live">LIVE</span>`
        : `<span class="badge-mock">MOCK</span>`;

      // Operational details
      let opDetails = `<span style="color: var(--text-subtle);">All completed</span>`;
      if (status === "failed") {
        const errMsg = r.utterance_a?.error_message || r.failure_explanation || "Operation failed";
        opDetails = `<div class="operational-detail-box"><strong>Error:</strong> ${escapeHtml(errMsg)}</div>`;
      } else if (status === "incomplete") {
        opDetails = `<div class="operational-detail-box" style="color: #fcd34d;"><strong>Partial:</strong> A completed, B failed (${escapeHtml(r.utterance_b?.error_message || 'provider error')})</div>`;
      }

      const dateStr = r.timestamp ? (r.timestamp.split("T")[0] + " " + (r.timestamp.split("T")[1]?.substring(0, 8) || "")) : "—";

      if (isFullView) {
        return `
          <tr>
            <td><code style="font-size: 0.75rem;">${r.run_id}</code></td>
            <td><strong>${escapeHtml(r.test_case_id)}</strong></td>
            <td>${statusBadge}</td>
            <td>${modeBadge}</td>
            <td>${distinctionHtml}</td>
            <td>${taskHtml}</td>
            <td>${werHtml}</td>
            <td><code style="font-size: 0.72rem;">${r.primary_failure_category}</code></td>
            <td>${opDetails}</td>
            <td style="font-size: 0.72rem; color: var(--text-muted);">${dateStr}</td>
          </tr>
        `;
      } else {
        return `
          <tr>
            <td><code style="font-size: 0.75rem;">${r.run_id}</code></td>
            <td><strong>${escapeHtml(r.test_case_id)}</strong></td>
            <td>${statusBadge}</td>
            <td>${r.category}</td>
            <td>${r.language_code}</td>
            <td>${distinctionHtml}</td>
            <td>${taskHtml}</td>
            <td>${werHtml}</td>
            <td><code style="font-size: 0.72rem;">${r.primary_failure_category}</code></td>
            <td>${modeBadge}</td>
            <td style="font-size: 0.72rem; color: var(--text-muted);">${dateStr}</td>
          </tr>
        `;
      }
    }).join("");
  }

  async function loadEvaluationHistory() {
    loadFullHistory(currentStatusFilter);
  }

  // ===========================================================================
  // 7. Single Clip Audio Transcription Lab (Existing Working Functionality)
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
  const providerProcessingDisplay = document.getElementById("providerProcessingDisplay");
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

    fileNameDisplay.textContent = file.name;
    fileSizeDisplay.textContent = formatSize(file.size);

    currentAudioUrl = URL.createObjectURL(file);
    audioPreview.src = currentAudioUrl;

    dropzonePrompt.style.display = "none";
    selectedFilePreview.style.display = "flex";
    hideError();
  }

  function clearSelectedFile() {
    if (!audioFileInput) return;
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

  if (audioFileInput) {
    audioFileInput.addEventListener("change", (e) => {
      if (e.target.files && e.target.files[0]) {
        handleFileSelected(e.target.files[0]);
      }
    });
  }

  if (btnClearFile) {
    btnClearFile.addEventListener("click", (e) => {
      e.stopPropagation();
      clearSelectedFile();
    });
  }

  if (dropzone) {
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
  }

  function showError(type, message, remediation = null) {
    if (!errorAlert) return;
    errorTypeDisplay.textContent = type;
    errorMessageDisplay.textContent = message;

    if (remediation) {
      errorRemediationDisplay.innerHTML = remediation;
      errorRemediationDisplay.style.display = "block";
    } else {
      errorRemediationDisplay.style.display = "none";
    }

    errorAlert.style.display = "block";
    resultStatusBadge.className = "status-badge status-error";
    resultStatusBadge.textContent = "Error";
  }

  function hideError() {
    if (!errorAlert) return;
    errorAlert.style.display = "none";
  }

  if (btnCopyTranscript) {
    btnCopyTranscript.addEventListener("click", () => {
      const text = transcriptDisplay.innerText;
      navigator.clipboard.writeText(text).then(() => {
        btnCopyTranscript.textContent = "Copied!";
        setTimeout(() => {
          btnCopyTranscript.textContent = "Copy";
        }, 2000);
      });
    });
  }

  if (transcribeForm) {
    transcribeForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      hideError();

      if (!audioFileInput.files || !audioFileInput.files[0]) {
        showError("VALIDATION_ERROR", "Please select an audio file to transcribe.");
        return;
      }

      btnSubmit.disabled = true;
      btnSpinner.style.display = "inline-block";
      btnText.textContent = "Transcribing...";
      resultStatusBadge.className = "status-badge status-running";
      resultStatusBadge.textContent = "Processing...";

      const formData = new FormData();
      formData.append("audio_file", audioFileInput.files[0]);
      formData.append("language_code", languageCodeSelect.value);
      formData.append("format_mode", formatModeSelect.value);
      formData.append("itn_native_numerals", itnNativeNumerals.checked ? "true" : "false");
      formData.append("mock_mode", mockModeCheckbox.checked ? "true" : "false");

      const startTime = performance.now();

      try {
        const response = await fetch("/api/transcribe", {
          method: "POST",
          body: formData,
        });

        const elapsedMs = Math.round(performance.now() - startTime);
        clientLatencyDisplay.textContent = `${elapsedMs} ms`;

        if (!response.ok) {
          const errData = await response.json().catch(() => ({}));
          const errType = errData.error_type || `HTTP_${response.status}`;
          const errMsg = errData.message || errData.detail || "Transcription request failed.";
          let remediation = null;

          if (response.status === 401 || errType === "AUTHENTICATION_ERROR") {
            remediation = "<strong>Remediation:</strong> Check that <code>GNANI_API_KEY</code> is correctly set in your <code>.env</code> file. Alternatively, enable <strong>Mock Mode</strong> to test locally without an API key.";
          } else if (response.status === 429) {
            remediation = "<strong>Remediation:</strong> Upstream rate limit reached. Please wait a few seconds before submitting another audio clip.";
          }

          showError(errType, errMsg, remediation);
          return;
        }

        const data = await response.json();

        // Populate Telemetry
        const backendLatency = data.client_request_elapsed_ms != null
          ? data.client_request_elapsed_ms
          : data.backend_latency_ms;
        backendLatencyDisplay.textContent = backendLatency != null ? `${backendLatency} ms` : "—";
        providerProcessingDisplay.textContent = data.provider_processing_time_ms != null
          ? `${data.provider_processing_time_ms} ms`
          : "Not reported";
        modelDisplay.textContent = data.model || "gnani-prisma-v2.5";
        requestIdDisplay.textContent = data.request_id || "mock-request-id";

        // Populate Transcript
        transcriptDisplay.innerHTML = `<p class="transcript-text">${escapeHtml(data.transcript)}</p>`;
        if (btnCopyTranscript) btnCopyTranscript.style.display = "inline-block";

        resultStatusBadge.className = "status-badge status-success";
        resultStatusBadge.textContent = data.is_mocked ? "Completed (Mock)" : "Completed (Live)";
      } catch (err) {
        console.error("Transcription request failed:", err);
        showError("NETWORK_ERROR", "Could not connect to the BoliBench server. Check your connection.", "<strong>Remediation:</strong> Ensure the FastAPI server is running on localhost:8000.");
      } finally {
        btnSubmit.disabled = false;
        btnSpinner.style.display = "none";
        btnText.textContent = "Transcribe with Gnani Prisma";
      }
    });
  }

  function escapeHtml(text) {
    if (!text) return "";
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
  }

  // ===========================================================================
  // Initial Boot
  // ===========================================================================
  updateSelectedTestCaseSpecs();
  loadBenchmarkMetrics();
  loadBenchmarkCatalogue();
  loadFullHistory("all");
});
