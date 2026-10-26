/**
 * BoliBench — Frontend Interaction, Telemetry & Paired-Contrast Evaluation Handler
 * Branded: BoliBench — Indian Voice AI Crash-Test Lab
 * Multi-view Developer Tool for the Great Indian AI Internship Challenge.
 *
 * Implements:
 * 1. 3-Primary Navigation + Secondary Reference Tools
 * 2. Product-First Hero with Paired-Contrast Concept Example & CTAs
 * 3. 4-Step Paired-Contrast Crash Lab Workflow with In-Browser WAV Previews
 * 4. Transparent Latency Telemetry & Strict Mode Metric Separation
 * 5. Dynamic Live-vs-Derived Audit Modal per selected history row
 * 6. Accessible Modal Focus Management, Keyboard Navigation, and Reduced-Motion Support
 */

document.addEventListener("DOMContentLoaded", () => {
  // ===========================================================================
  // 1. Navigation & View Routing
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

    // Tab-specific asynchronous refresh
    if (targetKey === "overviewView") {
      loadBenchmarkMetrics();
    } else if (targetKey === "evaluateView") {
      loadBenchmarkMetrics();
      loadEvaluationHistory();
      updateSelectedTestCaseSpecs();
      checkHumanPairHandoff();
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

  // Hero CTAs
  const btnHeroRunDemo = document.getElementById("btnHeroRunDemo");
  if (btnHeroRunDemo) {
    btnHeroRunDemo.addEventListener("click", () => {
      if (evalTestCaseSelect) {
        evalTestCaseSelect.value = "pair_qty_001";
        updateSelectedTestCaseSpecs();
      }
      switchTab("evaluateView");
      const evalSection = document.getElementById("evaluateView");
      if (evalSection) {
        evalSection.scrollIntoView({ behavior: "smooth" });
      }
    });
  }

  const btnHeroExploreResults = document.getElementById("btnHeroExploreResults");
  if (btnHeroExploreResults) {
    btnHeroExploreResults.addEventListener("click", () => {
      switchTab("historyView");
      const histSection = document.getElementById("historyView");
      if (histSection) {
        histSection.scrollIntoView({ behavior: "smooth" });
      }
    });
  }

  // ===========================================================================
  // 2. Overview Dashboard & Metric Separation
  // ===========================================================================
  // Live Evaluation Authorization Contract (Must exactly match live_guard.py on the backend)
  const REQUIRED_CONFIRM_HEADER = "X-BoliBench-Confirm-Live";
  const EXPECTED_CONFIRM_HEADER_VALUE = "CONFIRM_LIVE_EVALUATION";

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
  const healthLiveAttempts = document.getElementById("healthLiveAttempts") || document.getElementById("healthLiveCalls");
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

  // Mode-Separated Category Breakdown Elements
  const liveCategoryBreakdownBody = document.getElementById("liveCategoryBreakdownBody");
  const liveCatEmptyBox = document.getElementById("liveCatEmptyBox");
  const liveCatTableWrapper = document.getElementById("liveCatTableWrapper");

  const mockCategoryBreakdownBody = document.getElementById("mockCategoryBreakdownBody");
  const mockCatEmptyBox = document.getElementById("mockCatEmptyBox");
  const mockCatTableWrapper = document.getElementById("mockCatTableWrapper");

  // Crash Lab Mode-Separated Summaries (Strict separation: zero unlabelled mixed-mode presentation)
  const labMockCompleted = document.getElementById("labMockCompleted");
  const labMockDistinction = document.getElementById("labMockDistinction");
  const labMockTaskSuccess = document.getElementById("labMockTaskSuccess");
  const labMockWer = document.getElementById("labMockWer");

  const labLiveCompleted = document.getElementById("labLiveCompleted");
  const labLiveDistinction = document.getElementById("labLiveDistinction");
  const labLiveTaskSuccess = document.getElementById("labLiveTaskSuccess");
  const labLiveWer = document.getElementById("labLiveWer");
  const labLiveEmptyState = document.getElementById("labLiveEmptyState");
  const labLiveStatsGrid = document.getElementById("labLiveStatsGrid");

  let latestLiveRunId = null;

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
      if (healthLiveAttempts) healthLiveAttempts.textContent = m.live_runs;
      if (healthMockRuns) healthMockRuns.textContent = m.mock_runs;

      // Crash Lab: Mode-Separated Performance Summaries (No mixed-mode presentation)
      if (m.mock) {
        if (labMockCompleted) labMockCompleted.textContent = m.mock.completed_runs;
        if (labMockDistinction) {
          labMockDistinction.textContent = m.mock.distinction_preservation_rate != null
            ? `${m.mock.distinction_preservation_rate.toFixed(1)}%`
            : "—%";
        }
        if (labMockTaskSuccess) {
          labMockTaskSuccess.textContent = m.mock.task_success_rate != null
            ? `${m.mock.task_success_rate.toFixed(1)}%`
            : "—%";
        }
        if (labMockWer) {
          labMockWer.textContent = m.mock.average_wer != null
            ? `${(m.mock.average_wer * 100).toFixed(1)}%`
            : "—";
        }
      }

      if (m.live) {
        if (m.live.completed_runs > 0) {
          if (labLiveEmptyState) labLiveEmptyState.style.display = "none";
          if (labLiveStatsGrid) labLiveStatsGrid.style.display = "grid";
          if (labLiveCompleted) labLiveCompleted.textContent = m.live.completed_runs;
          if (labLiveDistinction) {
            labLiveDistinction.textContent = m.live.distinction_preservation_rate != null
              ? `${m.live.distinction_preservation_rate.toFixed(1)}%`
              : "—%";
          }
          if (labLiveTaskSuccess) {
            labLiveTaskSuccess.textContent = m.live.task_success_rate != null
              ? `${m.live.task_success_rate.toFixed(1)}%`
              : "—%";
          }
          if (labLiveWer) {
            labLiveWer.textContent = m.live.average_wer != null
              ? `${(m.live.average_wer * 100).toFixed(1)}%`
              : "—";
          }
        } else {
          if (labLiveEmptyState) labLiveEmptyState.style.display = "block";
          if (labLiveStatsGrid) labLiveStatsGrid.style.display = "none";
        }
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

      // 3. MODE-SEPARATED CATEGORY BREAKDOWN TABLES
      renderCategoryBreakdown(m);
    } catch (err) {
      console.error("Failed to load metrics:", err);
    }
  }

  function renderCategoryBreakdown(metrics) {
    if (!metrics) return;
    renderModeCategoryTable(
      liveCategoryBreakdownBody,
      liveCatEmptyBox,
      liveCatTableWrapper,
      metrics.live?.category_breakdown,
      true
    );
    renderModeCategoryTable(
      mockCategoryBreakdownBody,
      mockCatEmptyBox,
      mockCatTableWrapper,
      metrics.mock?.category_breakdown,
      false
    );
  }

  function renderModeCategoryTable(tbody, emptyBox, tableWrapper, catData, isLive) {
    if (!tbody) return;
    const categories = catData ? Object.keys(catData) : [];
    const hasAnyAttempts = categories.length > 0 && categories.some((cat) => (catData[cat].total || 0) > 0);

    if (!hasAnyAttempts) {
      if (emptyBox) emptyBox.style.display = "block";
      if (tableWrapper) tableWrapper.style.display = "none";
      tbody.innerHTML = `
        <tr>
          <td colspan="7" class="table-empty-cell">
            ${isLive ? "No completed live evaluations recorded yet." : "No mock demonstrations recorded yet."}
          </td>
        </tr>
      `;
      return;
    }

    if (emptyBox) emptyBox.style.display = "none";
    if (tableWrapper) tableWrapper.style.display = "block";

    const rowsHtml = categories.map((cat) => {
      const row = catData[cat];
      const distStr = row.distinction_rate != null
        ? `${row.distinction_rate.toFixed(1)}%`
        : '<span class="val-unavailable">Unavailable</span>';
      const taskStr = row.task_success_rate != null
        ? `${row.task_success_rate.toFixed(1)}%`
        : '<span class="val-unavailable">Unavailable</span>';
      const werStr = row.average_wer != null
        ? `${(row.average_wer * 100).toFixed(1)}%`
        : '<span class="val-unavailable">Unavailable</span>';
      const failedCount = (row.failed || 0) + (row.incomplete || 0);

      return `
        <tr>
          <td><strong>${escapeHtml(cat)}</strong></td>
          <td>${row.total}</td>
          <td><span style="color: #34d399; font-weight: 700;">${row.completed}</span></td>
          <td><span style="color: ${failedCount > 0 ? '#f87171' : 'var(--text-subtle)'}; font-weight: 700;">${failedCount}</span></td>
          <td>${distStr}</td>
          <td>${taskStr}</td>
          <td>${werStr}</td>
        </tr>
      `;
    }).join("");

    tbody.innerHTML = rowsHtml;
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
  const conversionFeedbackA = document.getElementById("conversionFeedbackA");
  const conversionFeedbackB = document.getElementById("conversionFeedbackB");
  const origFileNameA = document.getElementById("origFileNameA");
  const origFileNameB = document.getElementById("origFileNameB");
  const audioSpecsA = document.getElementById("audioSpecsA");
  const audioSpecsB = document.getElementById("audioSpecsB");

  const confirmLiveCreditUsage = document.getElementById("confirmLiveCreditUsage");
  const btnExecuteLiveEval = document.getElementById("btnExecuteLiveEval");
  const btnLiveEvalText = document.getElementById("btnLiveEvalText");
  const btnLiveSpinner = document.getElementById("btnLiveSpinner");
  const liveEvalBtnHint = document.getElementById("liveEvalBtnHint");
  const liveEvalInlineError = document.getElementById("liveEvalInlineError");
  const curatedBenchmarkDrawer = document.getElementById("curatedBenchmarkDrawer");

  const liveCreditAuthModal = document.getElementById("liveCreditAuthModal");
  const modalCloseBtn = document.getElementById("modalCloseBtn") || document.getElementById("btnCloseLiveModal");
  const modalCancelBtn = document.getElementById("btnCancelLiveModal") || document.getElementById("modalCancelBtn");
  const modalConfirmBtn = document.getElementById("btnConfirmLiveModal") || document.getElementById("modalConfirmBtn");
  const modalClipNameA = document.getElementById("modalAuthFileA") || document.getElementById("modalClipNameA");
  const modalClipNameB = document.getElementById("modalAuthFileB") || document.getElementById("modalClipNameB");
  const modalTargetScenario = document.getElementById("modalAuthEvalMode") || document.getElementById("modalTargetScenario");

  const customTranscriptDrawer = document.getElementById("customTranscriptDrawer");
  const customTranscriptA = document.getElementById("customTranscriptA");
  const customTranscriptB = document.getElementById("customTranscriptB");
  const btnScoreCustomTranscripts = document.getElementById("btnScoreCustomTranscripts");
  const btnScoreAutoTranscripts = document.getElementById("btnScoreAutoTranscripts");

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
  let currentConvertedFileA = null;
  let currentConvertedFileB = null;
  let isConvertingA = false;
  let isConvertingB = false;
  let conversionSeqA = 0;
  let conversionSeqB = 0;

  function formatSize(bytes) {
    if (!bytes || bytes <= 0) return "0 KB";
    const kb = (bytes / 1024).toFixed(1);
    if (bytes >= 1024 * 1024) {
      return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
    }
    return `${kb} KB`;
  }

  function showConversionStatus(which, status, message) {
    const feedbackEl = which === "A" ? conversionFeedbackA : conversionFeedbackB;
    if (!feedbackEl) return;

    feedbackEl.className = "conversion-feedback";
    if (status === "clear") {
      feedbackEl.style.display = "none";
      feedbackEl.textContent = "";
      return;
    }

    if (status === "converting") {
      feedbackEl.classList.add("status-converting");
      feedbackEl.innerHTML = `<span class="conv-spinner-mini"></span> Converting audio locally to 16 kHz WAV...`;
      feedbackEl.style.display = "flex";
    } else if (status === "success") {
      feedbackEl.classList.add("status-success");
      feedbackEl.textContent = message || "Converted successfully — WAV, 16 kHz, mono, PCM 16-bit.";
      feedbackEl.style.display = "flex";
    } else if (status === "error") {
      feedbackEl.classList.add("status-error");
      feedbackEl.textContent = message || "Audio conversion failed.";
      feedbackEl.style.display = "flex";
    }
  }

  async function handleAudioSelection(which, file) {
    const isA = which === "A";
    const previewContainer = isA ? previewContainerA : previewContainerB;
    const previewAudio = isA ? liveAudioPreviewA : liveAudioPreviewB;
    const origFileNameEl = isA ? origFileNameA : origFileNameB;
    const fileNameEl = isA ? fileNameA : fileNameB;
    const fileSizeEl = isA ? fileSizeA : fileSizeB;
    const audioSpecsEl = isA ? audioSpecsA : audioSpecsB;

    // Increment sequence number to prevent stale async conversions from overwriting newer selections
    const seq = isA ? ++conversionSeqA : ++conversionSeqB;

    // Revoke previous object URL
    if (isA) {
      if (currentLiveAudioUrlA) {
        URL.revokeObjectURL(currentLiveAudioUrlA);
        currentLiveAudioUrlA = null;
      }
      currentConvertedFileA = null;
      isConvertingA = false;
    } else {
      if (currentLiveAudioUrlB) {
        URL.revokeObjectURL(currentLiveAudioUrlB);
        currentLiveAudioUrlB = null;
      }
      currentConvertedFileB = null;
      isConvertingB = false;
    }

    // If no file was selected (user cleared selection)
    if (!file) {
      showConversionStatus(which, "clear");
      if (origFileNameEl) origFileNameEl.style.display = "none";
      if (previewContainer) previewContainer.style.display = "none";
      if (previewAudio) previewAudio.removeAttribute("src");
      checkLiveEvalFormValidity();
      return;
    }

    // Set converting state
    if (isA) {
      isConvertingA = true;
    } else {
      isConvertingB = true;
    }
    checkLiveEvalFormValidity();

    // Show initial UI
    if (origFileNameEl) {
      origFileNameEl.textContent = `Source: ${file.name}`;
      origFileNameEl.style.display = "inline";
    }
    showConversionStatus(which, "converting");
    if (previewContainer) previewContainer.style.display = "none";
    if (previewAudio) previewAudio.removeAttribute("src");

    try {
      const converter =
        (typeof BoliAudioConverter !== "undefined" && BoliAudioConverter) ||
        (typeof BoliBenchAudioConverter !== "undefined" && BoliBenchAudioConverter) ||
        (typeof window !== "undefined" && (window.BoliAudioConverter || window.BoliBenchAudioConverter));

      if (!converter || typeof converter.convertAudioFile !== "function") {
        throw new Error("Audio conversion engine is not available.");
      }

      const result = await converter.convertAudioFile(file);

      // Check if sequence is still current (prevent stale async resolution)
      const currentSeq = isA ? conversionSeqA : conversionSeqB;
      if (seq !== currentSeq) {
        return;
      }

      const convertedFile = result.convertedFile || result.file;
      if (!convertedFile) {
        throw new Error("Audio conversion engine did not produce a converted WAV file.");
      }

      const duration =
        typeof result.duration === "number"
          ? result.duration
          : (result.metadata && typeof result.metadata.duration === "number" ? result.metadata.duration : 0);
      const fileSize =
        typeof result.fileSize === "number"
          ? result.fileSize
          : (result.metadata && typeof result.metadata.fileSize === "number" ? result.metadata.fileSize : convertedFile.size);

      if (isA) {
        currentConvertedFileA = convertedFile;
        isConvertingA = false;
      } else {
        currentConvertedFileB = convertedFile;
        isConvertingB = false;
      }

      // Update UI with converted specs
      if (fileNameEl) fileNameEl.textContent = convertedFile.name;
      if (fileSizeEl) fileSizeEl.textContent = formatSize(fileSize);
      if (audioSpecsEl) {
        audioSpecsEl.textContent = `WAV • 16 kHz • Mono • 16-bit • ${duration.toFixed(2)}s`;
      }

      // Set playback preview with converted WAV
      const newUrl = URL.createObjectURL(convertedFile);
      if (isA) {
        currentLiveAudioUrlA = newUrl;
      } else {
        currentLiveAudioUrlB = newUrl;
      }
      if (previewAudio) {
        previewAudio.src = newUrl;
      }
      if (previewContainer) {
        previewContainer.style.display = "flex";
      }

      showConversionStatus(which, "success", `Converted successfully — WAV, 16 kHz, mono, PCM 16-bit.`);
      checkLiveEvalFormValidity();
    } catch (err) {
      const currentSeq = isA ? conversionSeqA : conversionSeqB;
      if (seq !== currentSeq) return;

      if (isA) {
        currentConvertedFileA = null;
        isConvertingA = false;
      } else {
        currentConvertedFileB = null;
        isConvertingB = false;
      }

      let errorMsg = err.message || "Conversion failed.";
      if (errorMsg === "Audio conversion engine is not available.") {
        errorMsg = "Audio conversion engine is not available. Please ensure audio_converter.js is loaded.";
      } else if (errorMsg.includes("duration") || errorMsg.includes("60 seconds")) {
        errorMsg = "Audio exceeds the permitted duration (maximum 60 seconds).";
      } else if (errorMsg.includes("decode") || errorMsg.includes("codec")) {
        errorMsg = "Conversion failed — this browser could not decode the selected audio format. Try exporting the recording to a supported format.";
      } else if (errorMsg.includes("size limit")) {
        errorMsg = "Selected file exceeds the maximum input size limit.";
      }

      showConversionStatus(which, "error", errorMsg);
      if (previewContainer) previewContainer.style.display = "none";
      if (previewAudio) previewAudio.removeAttribute("src");
      checkLiveEvalFormValidity();
    }
  }

  // Audio Playback Previews and Automatic Conversion for Utterance A & B
  if (liveAudioInputA) {
    liveAudioInputA.addEventListener("change", (e) => {
      const file = e.target.files && e.target.files[0];
      handleAudioSelection("A", file);
    });
  }

  if (liveAudioInputB) {
    liveAudioInputB.addEventListener("change", (e) => {
      const file = e.target.files && e.target.files[0];
      handleAudioSelection("B", file);
    });
  }

  function showLiveInlineError(msg) {
    if (!liveEvalInlineError) return;
    if (!msg) {
      liveEvalInlineError.style.display = "none";
      liveEvalInlineError.textContent = "";
      return;
    }
    liveEvalInlineError.style.display = "flex";
    liveEvalInlineError.textContent = msg;
  }

  function requestCreditAuthorization(scenarioName, clipNameA, clipNameB) {
    return new Promise((resolve) => {
      if (!liveCreditAuthModal || typeof liveCreditAuthModal.showModal !== "function") {
        const userConfirmed = window.confirm(
          "LIVE EVALUATION CREDIT AUTHORIZATION\n\n" +
          `Target Scenario: ${scenarioName}\n` +
          `Recording A: ${clipNameA}\n` +
          `Recording B: ${clipNameB}\n\n` +
          "Evaluating this pair makes 2 real requests to Gnani Prisma STT v3 and consumes account credits.\n\n" +
          "Duration policy: Each clip must be a valid WAV file strictly <= 60 seconds.\n\n" +
          "Do you authorize this paired live execution?"
        );
        resolve(userConfirmed);
        return;
      }

      const clipAEl = document.getElementById("modalAuthFileA") || document.getElementById("modalClipNameA");
      const clipBEl = document.getElementById("modalAuthFileB") || document.getElementById("modalClipNameB");
      const scenarioEl = document.getElementById("modalAuthEvalMode") || document.getElementById("modalTargetScenario");

      if (clipAEl) clipAEl.textContent = clipNameA || "Recording A";
      if (clipBEl) clipBEl.textContent = clipNameB || "Recording B";
      if (scenarioEl) scenarioEl.textContent = scenarioName || "Automatic Semantic Contrast (Uncurated Pair)";

      let settled = false;
      const finish = (result) => {
        if (settled) return;
        settled = true;
        cleanup();
        if (liveCreditAuthModal.open) {
          liveCreditAuthModal.close();
        }
        resolve(result);
      };

      const handleConfirm = (e) => {
        if (e && typeof e.preventDefault === "function") e.preventDefault();
        finish(true);
      };

      const handleCancel = (e) => {
        if (e && typeof e.preventDefault === "function") e.preventDefault();
        finish(false);
      };

      const handleDialogClose = () => {
        const isConfirmed = liveCreditAuthModal.returnValue === "confirm";
        finish(isConfirmed);
      };

      const confirmBtn = document.getElementById("btnConfirmLiveModal") || document.getElementById("modalConfirmBtn");
      const cancelBtn = document.getElementById("btnCancelLiveModal") || document.getElementById("modalCancelBtn");
      const closeBtn = document.getElementById("modalCloseBtn") || document.getElementById("btnCloseLiveModal");

      const cleanup = () => {
        if (confirmBtn) confirmBtn.removeEventListener("click", handleConfirm);
        if (cancelBtn) cancelBtn.removeEventListener("click", handleCancel);
        if (closeBtn) closeBtn.removeEventListener("click", handleCancel);
        liveCreditAuthModal.removeEventListener("close", handleDialogClose);
        liveCreditAuthModal.removeEventListener("cancel", handleCancel);
      };

      if (confirmBtn) confirmBtn.addEventListener("click", handleConfirm, { once: true });
      if (cancelBtn) cancelBtn.addEventListener("click", handleCancel, { once: true });
      if (closeBtn) closeBtn.addEventListener("click", handleCancel, { once: true });
      liveCreditAuthModal.addEventListener("close", handleDialogClose, { once: true });
      liveCreditAuthModal.addEventListener("cancel", handleCancel, { once: true });

      liveCreditAuthModal.returnValue = "";
      liveCreditAuthModal.showModal();
    });
  }

  function checkLiveEvalFormValidity() {
    if (!btnExecuteLiveEval) return;
    const hasA = Boolean(currentConvertedFileA);
    const hasB = Boolean(currentConvertedFileB);
    const converting = isConvertingA || isConvertingB;
    const isConfirmed = Boolean(confirmLiveCreditUsage && confirmLiveCreditUsage.checked);

    // Dynamic hint text explaining the missing requirements right beside the control
    if (liveEvalBtnHint) {
      if (converting) {
        liveEvalBtnHint.textContent = "Converting audio locally to 16 kHz WAV...";
        liveEvalBtnHint.style.color = "#f59e0b";
      } else if (!hasA && !hasB) {
        liveEvalBtnHint.textContent = "Upload both Recording A & B above to enable analysis.";
        liveEvalBtnHint.style.color = "#94a3b8";
      } else if (!hasA) {
        liveEvalBtnHint.textContent = "Upload Recording A above to enable analysis.";
        liveEvalBtnHint.style.color = "#94a3b8";
      } else if (!hasB) {
        liveEvalBtnHint.textContent = "Upload Recording B above to enable analysis.";
        liveEvalBtnHint.style.color = "#94a3b8";
      } else if (!isConfirmed) {
        liveEvalBtnHint.textContent = "Check credit authorization box to enable analysis.";
        liveEvalBtnHint.style.color = "#f59e0b";
      } else {
        liveEvalBtnHint.textContent = "Ready to analyze paired recordings.";
        liveEvalBtnHint.style.color = "#34d399";
      }
    }

    btnExecuteLiveEval.disabled = !hasA || !hasB || converting || !isConfirmed;

    const isSpinning = Boolean(btnLiveSpinner && btnLiveSpinner.style && btnLiveSpinner.style.display && btnLiveSpinner.style.display.includes("inline"));
    if (btnLiveEvalText && !isSpinning) {
      if (isConvertingA && isConvertingB) {
        btnLiveEvalText.textContent = "Converting Both Recordings...";
      } else if (isConvertingA) {
        btnLiveEvalText.textContent = "Utterance A is Converting...";
      } else if (isConvertingB) {
        btnLiveEvalText.textContent = "Utterance B is Converting...";
      } else {
        btnLiveEvalText.textContent = "Analyze Paired Recordings";
      }
    }
  }

  function resetLiveAudioSelection() {
    if (liveAudioInputA) liveAudioInputA.value = "";
    if (liveAudioInputB) liveAudioInputB.value = "";
    handleAudioSelection("A", null);
    handleAudioSelection("B", null);
    showLiveInlineError(null);
  }

  if (confirmLiveCreditUsage) {
    confirmLiveCreditUsage.addEventListener("change", () => {
      showLiveInlineError(null);
      checkLiveEvalFormValidity();
    });
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

      // Update Active Scenario indicator in Live Audio Drawer
      const liveDrawerCategoryBadge = document.getElementById("liveDrawerCategoryBadge");
      const liveDrawerTestTitle = document.getElementById("liveDrawerTestTitle");
      const liveDrawerDescription = document.getElementById("liveDrawerDescription");
      const liveDrawerPromptA = document.getElementById("liveDrawerPromptA");
      const liveDrawerPromptB = document.getElementById("liveDrawerPromptB");

      if (liveDrawerCategoryBadge) liveDrawerCategoryBadge.textContent = tc.category;
      if (liveDrawerTestTitle) liveDrawerTestTitle.textContent = tc.id;
      if (liveDrawerDescription) liveDrawerDescription.textContent = tc.description;
      if (liveDrawerPromptA) liveDrawerPromptA.textContent = `"${tc.utterance_a.reference_transcript}"`;
      if (liveDrawerPromptB) liveDrawerPromptB.textContent = `"${tc.utterance_b.reference_transcript}"`;

      // Populate custom inputs as starting template
      if (customTranscriptA) customTranscriptA.value = tc.utterance_a.reference_transcript;
      if (customTranscriptB) customTranscriptB.value = tc.utterance_b.reference_transcript;
    } catch (err) {
      console.error("Failed to load test case details:", err);
    }
  }

  if (evalTestCaseSelect) {
    evalTestCaseSelect.addEventListener("change", () => {
      resetLiveAudioSelection();
      updateSelectedTestCaseSpecs();
    });
  }

  if (btnToggleCustomInputs) {
    btnToggleCustomInputs.addEventListener("click", () => {
      const isHidden = customTranscriptDrawer.style.display === "none";
      customTranscriptDrawer.style.display = isHidden ? "flex" : "none";
      btnToggleCustomInputs.textContent = isHidden ? "Hide Custom Inputs" : "Custom Transcript Test";
      if (isHidden && liveEvalDrawer) {
        liveEvalDrawer.style.display = "none";
        btnToggleLiveEval.textContent = "🎙️ Configure Live Audio Pair (WAV)";
      }
    });
  }

  if (btnToggleLiveEval) {
    btnToggleLiveEval.addEventListener("click", () => {
      const isHidden = liveEvalDrawer.style.display === "none";
      liveEvalDrawer.style.display = isHidden ? "flex" : "none";
      btnToggleLiveEval.textContent = isHidden ? "Hide Live Audio Pair" : "🎙️ Configure Live Audio Pair (WAV)";
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

    if (btnRunMockIdeal) btnRunMockIdeal.disabled = true;
    if (btnRunMockCrashed) btnRunMockCrashed.disabled = true;

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

      const scorecard = document.getElementById("evalScorecardSection");
      if (scorecard) {
        scorecard.scrollIntoView({ behavior: "smooth" });
      }
    } catch (err) {
      console.error("Mock evaluation error:", err);
      evalStatusBadge.className = "status-badge status-error";
      evalStatusBadge.textContent = "Execution Error";
    } finally {
      if (btnRunMockIdeal) btnRunMockIdeal.disabled = false;
      if (btnRunMockCrashed) btnRunMockCrashed.disabled = false;
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

      btnScoreCustomTranscripts.disabled = true;
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

        const scorecard = document.getElementById("evalScorecardSection");
        if (scorecard) {
          scorecard.scrollIntoView({ behavior: "smooth" });
        }
      } catch (err) {
        console.error("Custom transcript evaluation error:", err);
        evalStatusBadge.className = "status-badge status-error";
        evalStatusBadge.textContent = "Scoring Error";
      } finally {
        btnScoreCustomTranscripts.disabled = false;
      }
    });
  }

  // Automatic Transcript Evaluation Handler
  if (btnScoreAutoTranscripts) {
    btnScoreAutoTranscripts.addEventListener("click", async () => {
      const txtA = customTranscriptA.value.trim();
      const txtB = customTranscriptB.value.trim();

      if (!txtA || !txtB) {
        alert("Please provide transcripts for both Utterance A and Utterance B.");
        return;
      }

      btnScoreAutoTranscripts.disabled = true;
      evalStatusBadge.className = "status-badge status-running";
      evalStatusBadge.textContent = "Auto-Detecting Contrast...";

      try {
        const response = await fetch("/api/eval/evaluate-auto-transcripts", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            transcript_a: txtA,
            transcript_b: txtB,
            audio_condition: "text_only_simulation",
          }),
        });

        if (!response.ok) {
          throw new Error("Automatic evaluation request failed");
        }

        const result = await response.json();
        displayEvaluationScorecard(result);
        loadBenchmarkMetrics();
        loadEvaluationHistory();

        const scorecard = document.getElementById("evalScorecardSection");
        if (scorecard) {
          scorecard.scrollIntoView({ behavior: "smooth" });
        }
      } catch (err) {
        console.error("Automatic transcript evaluation error:", err);
        evalStatusBadge.className = "status-badge status-error";
        evalStatusBadge.textContent = "Scoring Error";
      } finally {
        btnScoreAutoTranscripts.disabled = false;
      }
    });
  }

  // Live Paired Evaluation Execution with Strict Safeguards & Automatic Scenario Detection
  if (btnExecuteLiveEval) {
    btnExecuteLiveEval.addEventListener("click", async () => {
      showLiveInlineError(null);

      if (!currentConvertedFileA || !currentConvertedFileB) {
        showLiveInlineError("Missing audio recordings: Please select and convert both Recording A and Recording B before analyzing.");
        return;
      }

      if (!confirmLiveCreditUsage || !confirmLiveCreditUsage.checked) {
        showLiveInlineError("Credit Authorization Required: Please check the credit authorization box above before executing.");
        return;
      }

      // Check whether a curated test is selected or if we use automatic scenario detection
      const isCurated = Boolean(evalTestCaseSelect && evalTestCaseSelect.value && evalTestCaseSelect.value !== "auto");
      const testId = isCurated ? evalTestCaseSelect.value : "auto";

      const selectedOption = isCurated && evalTestCaseSelect.options ? evalTestCaseSelect.options[evalTestCaseSelect.selectedIndex] : null;
      const selectedCategory = selectedOption ? selectedOption.getAttribute("data-category") : (isCurated && testCategoryBadge ? testCategoryBadge.textContent.trim() : null);

      const scenarioLabel = isCurated
        ? `Curated Pair Benchmark: [${selectedCategory || 'curated'}] ${testId}`
        : "Automatic Scenario Contrast Detection (Uncurated Pair)";

      const confirmed = await requestCreditAuthorization(scenarioLabel, currentConvertedFileA.name, currentConvertedFileB.name);
      if (!confirmed) return;

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
          body: JSON.stringify({
            test_id: testId,
            expected_category: selectedCategory,
          }),
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
        formData.append("audio_file_a", currentConvertedFileA, currentConvertedFileA.name);
        formData.append("audio_file_b", currentConvertedFileB, currentConvertedFileB.name);
        formData.append("audio_condition", selectedCondition);
        formData.append("confirmation_token", authToken);
        if (selectedCategory) {
          formData.append("expected_category", selectedCategory);
        }

        btnLiveEvalText.textContent = "Transcribing Recording A & B with Provider...";
        evalStatusBadge.textContent = "Running Live STT Evaluation...";

        const confirmHeader = authData.required_header || REQUIRED_CONFIRM_HEADER;
        const confirmValue = authData.required_header_value || EXPECTED_CONFIRM_HEADER_VALUE;

        const evalResp = await fetch(`/api/eval/run-live/${testId}`, {
          method: "POST",
          headers: {
            "X-Requested-With": "XMLHttpRequest",
            [confirmHeader]: confirmValue,
          },
          body: formData,
        });

        if (!evalResp.ok) {
          const evalErr = await evalResp.json().catch(() => ({}));
          throw new Error(evalErr.detail || `Live evaluation failed (HTTP ${evalResp.status})`);
        }

        const evalResult = await evalResp.json();
        latestLiveRunId = evalResult.run_id;
        displayEvaluationScorecard(evalResult);
        loadBenchmarkMetrics();
        loadEvaluationHistory();

        const scorecard = document.getElementById("evalScorecardSection");
        if (scorecard) {
          scorecard.scrollIntoView({ behavior: "smooth" });
        }
      } catch (err) {
        console.error("Live paired evaluation failed:", err);
        evalStatusBadge.className = "status-badge status-error";
        evalStatusBadge.textContent = "Live Evaluation Failed";
        showLiveInlineError(`Live Evaluation Error: ${err.message}`);
        loadBenchmarkMetrics();
        loadEvaluationHistory();
      } finally {
        btnExecuteLiveEval.disabled = false;
        btnLiveSpinner.style.display = "none";
        btnLiveEvalText.textContent = "Analyze Paired Recordings";
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

    const isLive = res.execution_mode === "live";
    const isAuto = res.evaluation_mode === "automatic_inference" || res.is_curated_reference === false;

    // Mode Pill
    if (isAuto) {
      evalExecModePill.textContent = isLive ? "LIVE GNANI — AUTO INFERRED" : "MOCK — AUTO INFERRED";
      evalExecModePill.className = isLive ? "exec-mode-pill badge-live" : "exec-mode-pill badge-mock";
    } else {
      evalExecModePill.textContent = isLive ? "LIVE GNANI EVALUATION" : "MOCK EVALUATION";
      evalExecModePill.className = isLive ? "exec-mode-pill badge-live" : "exec-mode-pill badge-mock";
    }

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
    } else if (res.category === "unknown" || res.category === "ambiguous") {
      evalOutcomeBanner.className = "outcome-banner outcome-crashed";
      evalOutcomeIcon.textContent = "❓";
      evalOutcomeTitle.textContent = "AMBIGUOUS / UNKNOWN SCENARIO CONTRAST";
      evalStatusBadge.className = "status-badge status-warning";
      evalStatusBadge.textContent = "Ambiguous Result";
    } else if (res.semantic_distinction_preserved && res.task_success) {
      evalOutcomeBanner.className = "outcome-banner outcome-success";
      evalOutcomeIcon.textContent = "✅";
      evalOutcomeTitle.textContent = isAuto
        ? `AUTOMATIC SCENARIO DETECTED: [${(res.category || '').toUpperCase()}] CRITICAL DISTINCTION PRESERVED`
        : "CRITICAL DISTINCTION PRESERVED";
      evalStatusBadge.className = "status-badge status-success";
      evalStatusBadge.textContent = "Passed (Preserved)";
    } else {
      evalOutcomeBanner.className = "outcome-banner outcome-crashed";
      evalOutcomeIcon.textContent = "🚨";
      evalOutcomeTitle.textContent = `DISTINCTION COLLAPSED (${res.primary_failure_category ? res.primary_failure_category.toUpperCase() : 'CRASH'})`;
      evalStatusBadge.className = "status-badge status-error";
      evalStatusBadge.textContent = "Action Crashed";
    }

    evalOutcomeExplanation.textContent = res.failure_explanation || "";

    // Metrics Row
    if (res.overall_status !== "completed") {
      metricDistinctionDisplay.innerHTML = `<span class="val-unavailable">Unavailable</span>`;
      metricTaskSuccessDisplay.innerHTML = `<span class="val-unavailable">Unavailable</span>`;
      metricMeanWerDisplay.innerHTML = `<span class="val-unavailable">Unavailable</span>`;
      metricFieldAccDisplay.innerHTML = `<span class="val-unavailable">Unavailable</span>`;
    } else if (res.category === "unknown" || res.category === "ambiguous") {
      metricDistinctionDisplay.innerHTML = `<span style="color: #f59e0b;">AMBIGUOUS</span>`;
      metricTaskSuccessDisplay.innerHTML = `<span class="val-unavailable">Indeterminate</span>`;
      metricMeanWerDisplay.innerHTML = `<span class="val-unavailable" title="Reference-free upload">N/A (Reference-Free)</span>`;
      metricFieldAccDisplay.innerHTML = `<span class="val-unavailable">N/A</span>`;
    } else {
      metricDistinctionDisplay.innerHTML = res.semantic_distinction_preserved
        ? `<span style="color: #34d399;">PRESERVED</span>`
        : `<span style="color: #f87171;">COLLAPSED</span>`;

      metricTaskSuccessDisplay.innerHTML = res.task_success
        ? `<span style="color: #34d399;">SUCCESS</span>`
        : `<span style="color: #f87171;">FAILED</span>`;

      if (isAuto) {
        metricMeanWerDisplay.innerHTML = `<span class="val-unavailable" title="Reference-free upload: Ground-truth reference unavailable">N/A (Reference-Free)</span>`;
        metricFieldAccDisplay.textContent = res.critical_field_accuracy != null ? `${(res.critical_field_accuracy * 100).toFixed(0)}%` : "—";
      } else {
        metricMeanWerDisplay.textContent = res.average_wer != null ? `${(res.average_wer * 100).toFixed(1)}%` : "—";
        metricFieldAccDisplay.textContent = res.critical_field_accuracy != null ? `${(res.critical_field_accuracy * 100).toFixed(0)}%` : "—";
      }
    }

    // Comparison Table
    const refA = isAuto ? '<em style="color: var(--text-muted);">N/A (Reference-Free Upload)</em>' : `<code>${escapeHtml(res.utterance_a?.reference_transcript || '')}</code>`;
    const refB = isAuto ? '<em style="color: var(--text-muted);">N/A (Reference-Free Upload)</em>' : `<code>${escapeHtml(res.utterance_b?.reference_transcript || '')}</code>`;
    const werA = isAuto
      ? `<span class="val-unavailable">N/A (No Ground-Truth)</span>`
      : (res.utterance_a?.status === 'completed' ? `${(res.utterance_a.wer * 100).toFixed(1)}% (Raw: S:${res.utterance_a.substitutions}/D:${res.utterance_a.deletions}/I:${res.utterance_a.insertions})${res.utterance_a.normalized_wer != null ? `<br><small style="color: #60a5fa; font-weight: 500;">Normalized WER: ${(res.utterance_a.normalized_wer * 100).toFixed(1)}%</small>` : ''}` : `<span class="val-unavailable">${res.utterance_a?.status || '—'}</span>`);
    const werB = isAuto
      ? `<span class="val-unavailable">N/A (No Ground-Truth)</span>`
      : (res.utterance_b?.status === 'completed' ? `${(res.utterance_b.wer * 100).toFixed(1)}% (Raw: S:${res.utterance_b.substitutions}/D:${res.utterance_b.deletions}/I:${res.utterance_b.insertions})${res.utterance_b.normalized_wer != null ? `<br><small style="color: #60a5fa; font-weight: 500;">Normalized WER: ${(res.utterance_b.normalized_wer * 100).toFixed(1)}%</small>` : ''}` : `<span class="val-unavailable">${res.utterance_b?.status || '—'}</span>`);

    evalComparisonTableBody.innerHTML = `
      <tr>
        <td><span class="spec-badge badge-a">Recording A</span></td>
        <td>${refA}</td>
        <td><strong>${escapeHtml(res.utterance_a?.actual_transcript || '—')}</strong></td>
        <td>${werA}</td>
        <td><code>${escapeHtml(JSON.stringify(res.utterance_a?.extracted_fields || {}))}</code></td>
        <td>${res.utterance_a?.status === 'completed' ? (res.utterance_a.task_success ? '🟢 Correct' : '🔴 Wrong Action') : `<span class="val-unavailable">${res.utterance_a?.status || '—'}</span>`}</td>
      </tr>
      <tr>
        <td><span class="spec-badge badge-b">Recording B</span></td>
        <td>${refB}</td>
        <td><strong>${escapeHtml(res.utterance_b?.actual_transcript || '—')}</strong></td>
        <td>${werB}</td>
        <td><code>${escapeHtml(JSON.stringify(res.utterance_b?.extracted_fields || {}))}</code></td>
        <td>${res.utterance_b?.status === 'completed' ? (res.utterance_b.task_success ? '🟢 Correct' : '🔴 Wrong Action') : `<span class="val-unavailable">${res.utterance_b?.status || '—'}</span>`}</td>
      </tr>
    `;

    // Script & Transliteration check
    if (transliterationNote) {
      const hasDevanagariHyp = /[\u0900-\u097F]/.test(res.utterance_a?.actual_transcript || "") || /[\u0900-\u097F]/.test(res.utterance_b?.actual_transcript || "");
      const hasLatinRef = !isAuto && /^[A-Za-z0-9\s.,?!'"-]+$/.test(res.utterance_a?.reference_transcript || "");
      if (hasDevanagariHyp && hasLatinRef) {
        transliterationNote.style.display = "flex";
      } else {
        transliterationNote.style.display = "none";
      }
    }

    // Crash Impact Description & Metadata Tags
    crashImpactDescription.textContent = res.failure_explanation || "Semantic comparison evaluated across paired utterances.";
    crashFieldTags.innerHTML = `
      <span class="code-pill">Category: ${escapeHtml(res.category || 'unknown')}</span>
      <span class="code-pill" style="color: ${isAuto ? '#a78bfa' : '#60a5fa'};">Mode: ${isAuto ? 'Automatic Semantic Inference' : 'Curated Reference Benchmark'}</span>
      <span class="code-pill">Audio Condition: ${escapeHtml(res.audio_condition || '')}</span>
      <span class="code-pill">Status: ${escapeHtml((res.overall_status || '').toUpperCase())}</span>
      ${res.distinction_field ? `<span class="code-pill">Distinction Field: ${escapeHtml(res.distinction_field)}</span>` : ''}
      ${res.detection_confidence != null ? `<span class="code-pill" style="color: #34d399;">Detection Confidence: ${typeof res.detection_confidence === 'number' ? (res.detection_confidence * 100).toFixed(0) + '%' : escapeHtml(String(res.detection_confidence).toUpperCase())}</span>` : ''}
      ${!isAuto && res.overall_status === 'completed' && res.average_wer != null ? `<span class="code-pill">Mean Raw WER: ${(res.average_wer * 100).toFixed(1)}%</span>` : ''}
      ${!isAuto && res.average_normalized_wer != null ? `<span class="code-pill" style="color: #60a5fa;">Cross-Script WER: ${(res.average_normalized_wer * 100).toFixed(1)}%</span>` : ''}
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
          <span class="cat-id">${escapeHtml(tc.id)}</span>
          <div class="cat-tag-row">
            <span class="lang-tag">${escapeHtml(tc.language_code)}</span>
            <span class="sim-badge" title="Hand-authored minimal contrast pair for semantic crash testing; not an observed speech error">TEXT-ONLY SIMULATION</span>
          </div>
        </div>
        <p class="cat-desc">${escapeHtml(tc.description)}</p>
        <div class="cat-contrast-box">
          <div class="cat-utt">
            <span class="cat-utt-label">Utterance A</span>
            <span class="cat-utt-ref">"${escapeHtml(tc.utterance_a.reference_transcript)}"</span>
            <code style="font-size: 0.72rem; color: #93c5fd;">Action: ${escapeHtml(tc.utterance_a.expected_action)}</code>
          </div>
          <div class="cat-utt">
            <span class="cat-utt-label">Utterance B (Contrast)</span>
            <span class="cat-utt-ref">"${escapeHtml(tc.utterance_b.reference_transcript)}"</span>
            <code style="font-size: 0.72rem; color: #93c5fd;">Action: ${escapeHtml(tc.utterance_b.expected_action)}</code>
          </div>
        </div>
        <div class="cat-distinction">
          <strong>Distinction (${escapeHtml(tc.distinction_field)}):</strong> ${escapeHtml(tc.expected_distinction)}
        </div>
        <div class="cat-footer">
          <span class="provenance-tag">${escapeHtml(tc.provenance || 'hand_authored_text_simulation')}</span>
          <button type="button" class="btn-cat-test" data-test-id="${escapeHtml(tc.id)}">Test in Crash Lab ⚡</button>
        </div>
      </div>
    `).join("");

    // Wire "Test in Crash Lab" buttons
    catalogueGrid.querySelectorAll(".btn-cat-test").forEach((btn) => {
      btn.addEventListener("click", () => {
        const targetId = btn.getAttribute("data-test-id");
        if (evalTestCaseSelect) {
          evalTestCaseSelect.value = targetId;
          resetLiveAudioSelection();
          updateSelectedTestCaseSpecs();
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

  let currentSourceFilter = "all";

  const statusFilterBtns = document.querySelectorAll(".status-filter-btn");
  statusFilterBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      statusFilterBtns.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      currentStatusFilter = btn.getAttribute("data-status");
      loadFullHistory(currentStatusFilter, currentSourceFilter);
    });
  });

  const datasetFilterBtns = document.querySelectorAll(".dataset-filter-btn");
  datasetFilterBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      datasetFilterBtns.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      currentSourceFilter = btn.getAttribute("data-source") || "all";
      loadFullHistory(currentStatusFilter, currentSourceFilter);
    });
  });

  async function loadFullHistory(statusFilter = "all", sourceFilter = "all") {
    if (!historyFullTableBody && !historyTableBody) return;
    try {
      const params = new URLSearchParams({ limit: "100" });
      if (statusFilter && statusFilter !== "all") {
        params.append("status", statusFilter);
      }
      if (sourceFilter && sourceFilter !== "all") {
        params.append("dataset_source", sourceFilter);
      }
      const url = `/api/eval/results?${params.toString()}`;

      const response = await fetch(url);
      if (!response.ok) return;
      const runs = await response.json();

      // Track most recent live run for the header shortcut
      const liveRuns = runs.filter((r) => r.execution_mode === "live" && r.overall_status === "completed");
      if (liveRuns.length > 0) {
        latestLiveRunId = liveRuns[0].run_id;
      }

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
            No evaluation runs found matching status filter '${escapeHtml(currentStatusFilter)}'.
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
        werHtml = r.average_wer != null ? `${(r.average_wer * 100).toFixed(1)}%` : "—";
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

      // Dataset source badge
      const sourceBadge = r.dataset_source === "human_recording"
        ? `<span class="badge-pill-source-human">HUMAN</span>`
        : `<span class="badge-pill-source-synth">SYNTHETIC</span>`;

      // Dynamic inspect button for every row
      const inspectBtn = ` <button type="button" class="btn-row-inspect" data-run-id="${escapeHtml(r.run_id)}" data-mode="${escapeHtml(r.execution_mode || 'mock')}" data-status="${escapeHtml(status)}" title="Inspect side-by-side contrast & offline derived re-score">🔍 Inspect</button>`;

      if (isFullView) {
        return `
          <tr>
            <td><code style="font-size: 0.75rem;">${escapeHtml(r.run_id)}</code>${inspectBtn}</td>
            <td><strong>${escapeHtml(r.test_case_id)}</strong> ${sourceBadge}</td>
            <td>${statusBadge}</td>
            <td>${modeBadge}</td>
            <td>${distinctionHtml}</td>
            <td>${taskHtml}</td>
            <td>${werHtml}</td>
            <td><code style="font-size: 0.72rem;">${escapeHtml(r.primary_failure_category || '')}</code></td>
            <td>${opDetails}</td>
            <td style="font-size: 0.72rem; color: var(--text-muted);">${escapeHtml(dateStr)}</td>
          </tr>
        `;
      } else {
        return `
          <tr>
            <td><code style="font-size: 0.75rem;">${escapeHtml(r.run_id)}</code>${inspectBtn}</td>
            <td><strong>${escapeHtml(r.test_case_id)}</strong> ${sourceBadge}</td>
            <td>${statusBadge}</td>
            <td>${escapeHtml(r.category || '')}</td>
            <td>${escapeHtml(r.language_code || '')}</td>
            <td>${distinctionHtml}</td>
            <td>${taskHtml}</td>
            <td>${werHtml}</td>
            <td><code style="font-size: 0.72rem;">${escapeHtml(r.primary_failure_category || '')}</code></td>
            <td>${modeBadge}</td>
            <td style="font-size: 0.72rem; color: var(--text-muted);">${escapeHtml(dateStr)}</td>
          </tr>
        `;
      }
    }).join("");
  }

  async function loadEvaluationHistory() {
    loadFullHistory(currentStatusFilter);
  }

  // ===========================================================================
  // 7. Single Clip Audio Transcription Lab
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
          } else if (response.status === 403) {
            remediation = "<strong>Remediation:</strong> Live single-audio transcription is disabled by default to protect challenge credits. Check <strong>Mock Mode</strong> for offline testing, or set <code>LIVE_TRANSCRIPTION_ENABLED=true</code> in your <code>.env</code> file.";
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
    if (text === null || text === undefined) return "";
    const div = document.createElement("div");
    div.textContent = String(text);
    return div.innerHTML;
  }

  // ===========================================================================
  // 8. Dynamic Live vs. Derived Re-Score Audit Modal
  // ===========================================================================
  const rescoreCompareModal = document.getElementById("rescoreCompareModal");
  const btnModalClose = document.getElementById("btnModalClose");
  const modalBtnCloseFooter = document.getElementById("modalBtnCloseFooter");
  const btnOverviewOpenCompare = document.getElementById("btnOverviewOpenCompare");
  const btnHistoryOpenCompare = document.getElementById("btnHistoryOpenCompare");

  // Modal Context Elements
  const modalTestCaseId = document.getElementById("modalTestCaseId");
  const modalAudioCondition = document.getElementById("modalAudioCondition");
  const modalLangCode = document.getElementById("modalLangCode");

  // Source Run Elements (Left Column)
  const modalLiveRunId = document.getElementById("modalLiveRunId");
  const modalLiveExecMode = document.getElementById("modalLiveExecMode");
  const modalLiveReqA = document.getElementById("modalLiveReqA");
  const modalLiveReqB = document.getElementById("modalLiveReqB");
  const modalLiveClientLatency = document.getElementById("modalLiveClientLatency");
  const modalLiveUttLatency = document.getElementById("modalLiveUttLatency");
  const modalLiveProvProcLatency = document.getElementById("modalLiveProvProcLatency");
  const modalLiveActualA = document.getElementById("modalLiveActualA");
  const modalLiveActualB = document.getElementById("modalLiveActualB");
  const modalLiveRefA = document.getElementById("modalLiveRefA");
  const modalLiveRefB = document.getElementById("modalLiveRefB");
  const modalLiveWerA = document.getElementById("modalLiveWerA");
  const modalLiveWerB = document.getElementById("modalLiveWerB");
  const modalLiveAvgWer = document.getElementById("modalLiveAvgWer");
  const modalLiveFields = document.getElementById("modalLiveFields");
  const modalLiveIntent = document.getElementById("modalLiveIntent");
  const modalLiveTaskSuccess = document.getElementById("modalLiveTaskSuccess");
  const modalLiveFailCat = document.getElementById("modalLiveFailCat");
  const modalLiveFailExp = document.getElementById("modalLiveFailExp");

  // Derived Assessment Elements (Right Column)
  const modalDerivedId = document.getElementById("modalDerivedId");
  const modalDerivedSrcId = document.getElementById("modalDerivedSrcId");
  const modalDerivedTime = document.getElementById("modalDerivedTime");
  const modalDerivedActualA = document.getElementById("modalDerivedActualA");
  const modalDerivedActualB = document.getElementById("modalDerivedActualB");
  const modalDerivedRefA = document.getElementById("modalDerivedRefA");
  const modalDerivedRefB = document.getElementById("modalDerivedRefB");
  const modalDerivedNormWerA = document.getElementById("modalDerivedNormWerA");
  const modalDerivedNormWerB = document.getElementById("modalDerivedNormWerB");
  const modalDerivedAvgWer = document.getElementById("modalDerivedAvgWer");
  const modalDerivedFields = document.getElementById("modalDerivedFields");
  const modalDerivedIntent = document.getElementById("modalDerivedIntent");
  const modalDerivedTaskSuccess = document.getElementById("modalDerivedTaskSuccess");
  const modalDerivedFailCat = document.getElementById("modalDerivedFailCat");
  const modalDerivedFailExp = document.getElementById("modalDerivedFailExp");
  const modalBtnExportJson = document.getElementById("modalBtnExportJson");

  // Unavailable & Compute Containers
  const modalDerivedUnavailable = document.getElementById("modalDerivedUnavailable");
  const modalDerivedUnavailableMsg = document.getElementById("modalDerivedUnavailableMsg");
  const modalComputeRescoreArea = document.getElementById("modalComputeRescoreArea");
  const modalDerivedContent = document.getElementById("modalDerivedContent");

  let lastFocusedElement = null;

  function closeCompareModal() {
    if (rescoreCompareModal) {
      rescoreCompareModal.style.display = "none";
      if (lastFocusedElement && typeof lastFocusedElement.focus === "function") {
        lastFocusedElement.focus();
      }
    }
  }

  if (btnModalClose) btnModalClose.addEventListener("click", closeCompareModal);
  if (modalBtnCloseFooter) modalBtnCloseFooter.addEventListener("click", closeCompareModal);
  if (rescoreCompareModal) {
    rescoreCompareModal.addEventListener("click", (e) => {
      if (e.target === rescoreCompareModal) closeCompareModal();
    });
  }

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && rescoreCompareModal && rescoreCompareModal.style.display !== "none") {
      closeCompareModal();
    }
  });

  function resetModalFields() {
    // Reset Context Pills
    if (modalTestCaseId) modalTestCaseId.textContent = "Loading...";
    if (modalAudioCondition) modalAudioCondition.textContent = "—";
    if (modalLangCode) modalLangCode.textContent = "—";

    // Reset Source Column Fields
    if (modalLiveRunId) modalLiveRunId.textContent = "—";
    if (modalLiveExecMode) modalLiveExecMode.textContent = "—";
    if (modalLiveReqA) modalLiveReqA.textContent = "—";
    if (modalLiveReqB) modalLiveReqB.textContent = "—";
    if (modalLiveClientLatency) modalLiveClientLatency.textContent = "—";
    if (modalLiveUttLatency) modalLiveUttLatency.textContent = "—";
    if (modalLiveProvProcLatency) modalLiveProvProcLatency.textContent = "Not reported";
    if (modalLiveRefA) modalLiveRefA.textContent = "—";
    if (modalLiveRefB) modalLiveRefB.textContent = "—";
    if (modalLiveActualA) modalLiveActualA.textContent = "—";
    if (modalLiveActualB) modalLiveActualB.textContent = "—";
    if (modalLiveWerA) modalLiveWerA.textContent = "Raw WER: —";
    if (modalLiveWerB) modalLiveWerB.textContent = "Raw WER: —";
    if (modalLiveAvgWer) modalLiveAvgWer.textContent = "—";
    if (modalLiveFields) modalLiveFields.textContent = "—";
    if (modalLiveIntent) modalLiveIntent.textContent = "—";
    if (modalLiveTaskSuccess) {
      modalLiveTaskSuccess.textContent = "—";
      modalLiveTaskSuccess.className = "badge-pill-neutral";
    }
    if (modalLiveFailCat) modalLiveFailCat.textContent = "—";
    if (modalLiveFailExp) modalLiveFailExp.textContent = "—";

    // Reset Derived Column Fields
    if (modalDerivedId) modalDerivedId.textContent = "—";
    if (modalDerivedSrcId) modalDerivedSrcId.textContent = "—";
    if (modalDerivedTime) modalDerivedTime.textContent = "—";
    if (modalDerivedRefA) modalDerivedRefA.textContent = "—";
    if (modalDerivedRefB) modalDerivedRefB.textContent = "—";
    if (modalDerivedActualA) modalDerivedActualA.textContent = "—";
    if (modalDerivedActualB) modalDerivedActualB.textContent = "—";
    if (modalDerivedNormWerA) modalDerivedNormWerA.textContent = "Normalized WER: —";
    if (modalDerivedNormWerB) modalDerivedNormWerB.textContent = "Normalized WER: —";
    if (modalDerivedAvgWer) modalDerivedAvgWer.textContent = "—";
    if (modalDerivedFields) modalDerivedFields.textContent = "—";
    if (modalDerivedIntent) modalDerivedIntent.textContent = "—";
    if (modalDerivedTaskSuccess) {
      modalDerivedTaskSuccess.textContent = "—";
      modalDerivedTaskSuccess.className = "badge-pill-neutral";
    }
    if (modalDerivedFailCat) modalDerivedFailCat.textContent = "—";
    if (modalDerivedFailExp) modalDerivedFailExp.textContent = "—";
    if (modalBtnExportJson) {
      modalBtnExportJson.removeAttribute("href");
      modalBtnExportJson.href = "";
      modalBtnExportJson.style.display = "none";
    }

    // Hide derived content and compute action until verified
    if (modalDerivedContent) modalDerivedContent.style.display = "none";
    if (modalComputeRescoreArea) modalComputeRescoreArea.style.display = "none";
    if (modalDerivedUnavailable) {
      modalDerivedUnavailable.style.display = "block";
      if (modalDerivedUnavailableMsg) modalDerivedUnavailableMsg.textContent = "Loading evaluation data...";
    }
  }

  function renderDerivedAssessment(d) {
    if (modalDerivedUnavailable) modalDerivedUnavailable.style.display = "none";
    if (modalComputeRescoreArea) modalComputeRescoreArea.style.display = "none";
    if (modalDerivedContent) modalDerivedContent.style.display = "flex";

    if (modalDerivedId) modalDerivedId.textContent = d.derived_id;
    if (modalDerivedSrcId) modalDerivedSrcId.textContent = d.source_run_id;
    if (modalDerivedTime) modalDerivedTime.textContent = d.created_at || "—";
    if (modalDerivedRefA) modalDerivedRefA.textContent = d.reference_a || "—";
    if (modalDerivedRefB) modalDerivedRefB.textContent = d.reference_b || "—";
    if (modalDerivedActualA) modalDerivedActualA.textContent = d.actual_transcript_a || "—";
    if (modalDerivedActualB) modalDerivedActualB.textContent = d.actual_transcript_b || "—";
    if (modalDerivedNormWerA) modalDerivedNormWerA.textContent = `Normalized WER: ${(d.normalized_wer_a * 100).toFixed(1)}%`;
    if (modalDerivedNormWerB) modalDerivedNormWerB.textContent = `Normalized WER: ${(d.normalized_wer_b * 100).toFixed(1)}%`;
    if (modalDerivedAvgWer) modalDerivedAvgWer.textContent = `${(d.average_normalized_wer * 100).toFixed(1)}%`;
    if (modalDerivedFields) modalDerivedFields.textContent = "Preserved";
    if (modalDerivedIntent) modalDerivedIntent.textContent = d.rescore_detected_intent_a || "ORDER_PRODUCT";
    if (modalDerivedTaskSuccess) {
      modalDerivedTaskSuccess.textContent = d.rescore_task_success ? "SUCCESS" : "FAILED";
      modalDerivedTaskSuccess.className = d.rescore_task_success ? "badge-pill-success" : "badge-pill-danger";
    }
    if (modalDerivedFailCat) modalDerivedFailCat.textContent = d.rescore_primary_failure_category || "none";
    if (modalDerivedFailExp) modalDerivedFailExp.textContent = d.rescore_failure_explanation || "Downstream order intent recognized; actions valid.";
    if (modalBtnExportJson) {
      modalBtnExportJson.href = `/api/eval/runs/${encodeURIComponent(d.source_run_id)}/derived-rescore`;
      modalBtnExportJson.style.display = "inline-flex";
    }
  }

  async function openRescoreCompareModal(runId = null, mode = null, status = null) {
    if (!rescoreCompareModal) return;
    lastFocusedElement = document.activeElement;

    // Reset previous record's state completely
    resetModalFields();

    const targetRunId = runId || latestLiveRunId;
    if (modalLiveRunId && targetRunId) {
      modalLiveRunId.textContent = targetRunId;
    }
    if (modalLiveExecMode && mode) {
      modalLiveExecMode.textContent = mode.toUpperCase();
    }

    rescoreCompareModal.style.display = "flex";
    if (btnModalClose) btnModalClose.focus();

    if (!targetRunId) {
      if (modalTestCaseId) modalTestCaseId.textContent = "—";
      if (modalDerivedUnavailableMsg) {
        modalDerivedUnavailableMsg.textContent = "No completed live evaluation run found or selected.";
      }
      return;
    }

    try {
      // Step 1: Always fetch the selected source record using its actual run ID
      const runResp = await fetch(`/api/eval/results/${encodeURIComponent(targetRunId)}`);
      if (!runResp.ok) {
        if (modalTestCaseId) modalTestCaseId.textContent = "—";
        if (modalDerivedUnavailableMsg) {
          modalDerivedUnavailableMsg.textContent = `Evaluation run '${targetRunId}' not found in active database.`;
        }
        return;
      }

      const run = await runResp.json();

      // Step 2: Populate source run column strictly from API response
      if (modalTestCaseId) modalTestCaseId.textContent = run.test_case_id || "—";
      if (modalAudioCondition) modalAudioCondition.textContent = run.audio_condition || "—";
      if (modalLangCode) modalLangCode.textContent = run.language_code || "—";

      if (modalLiveRunId) modalLiveRunId.textContent = run.run_id;
      if (modalLiveExecMode) modalLiveExecMode.textContent = (run.execution_mode || "mock").toUpperCase();
      if (modalLiveReqA) modalLiveReqA.textContent = run.utterance_a?.request_id || "Not reported";
      if (modalLiveReqB) modalLiveReqB.textContent = run.utterance_b?.request_id || "Not reported";
      if (modalLiveClientLatency) modalLiveClientLatency.textContent = run.client_request_elapsed_ms != null ? `${run.client_request_elapsed_ms} ms` : "Not reported";
      if (modalLiveUttLatency) {
        const latA = run.utterance_a?.latency_ms != null ? `${run.utterance_a.latency_ms} ms` : "—";
        const latB = run.utterance_b?.latency_ms != null ? `${run.utterance_b.latency_ms} ms` : "—";
        modalLiveUttLatency.textContent = `A: ${latA} | B: ${latB}`;
      }
      if (modalLiveProvProcLatency) {
        modalLiveProvProcLatency.textContent = run.provider_processing_time_ms != null ? `${run.provider_processing_time_ms} ms` : "Not reported";
      }
      if (modalLiveRefA) modalLiveRefA.textContent = run.utterance_a?.reference_transcript || "—";
      if (modalLiveRefB) modalLiveRefB.textContent = run.utterance_b?.reference_transcript || "—";
      if (modalLiveActualA) modalLiveActualA.textContent = run.utterance_a?.actual_transcript || "—";
      if (modalLiveActualB) modalLiveActualB.textContent = run.utterance_b?.actual_transcript || "—";
      if (modalLiveWerA) {
        modalLiveWerA.textContent = run.utterance_a?.wer != null
          ? `Raw WER: ${(run.utterance_a.wer * 100).toFixed(1)}% (S:${run.utterance_a.substitutions ?? 0}/D:${run.utterance_a.deletions ?? 0}/I:${run.utterance_a.insertions ?? 0})`
          : "Unavailable";
      }
      if (modalLiveWerB) {
        modalLiveWerB.textContent = run.utterance_b?.wer != null
          ? `Raw WER: ${(run.utterance_b.wer * 100).toFixed(1)}% (S:${run.utterance_b.substitutions ?? 0}/D:${run.utterance_b.deletions ?? 0}/I:${run.utterance_b.insertions ?? 0})`
          : "Unavailable";
      }
      if (modalLiveAvgWer) {
        modalLiveAvgWer.textContent = run.average_wer != null ? `${(run.average_wer * 100).toFixed(1)}%` : "Unavailable";
      }
      if (modalLiveFields) {
        modalLiveFields.textContent = JSON.stringify(run.utterance_a?.extracted_fields || {});
      }
      if (modalLiveIntent) modalLiveIntent.textContent = run.utterance_a?.detected_intent || "UNKNOWN";
      if (modalLiveTaskSuccess) {
        modalLiveTaskSuccess.textContent = run.task_success ? "SUCCESS" : "FAILED";
        modalLiveTaskSuccess.className = run.task_success ? "badge-pill-success" : "badge-pill-danger";
      }
      if (modalLiveFailCat) modalLiveFailCat.textContent = run.primary_failure_category || "none";
      if (modalLiveFailExp) modalLiveFailExp.textContent = run.failure_explanation || "No failure explanation.";

      // Step 3: Check eligibility for offline derived re-score
      const execMode = run.execution_mode || "mock";
      const overallStatus = run.overall_status || "completed";
      const hasTranscripts = Boolean(run.utterance_a?.actual_transcript && run.utterance_b?.actual_transcript);

      if (execMode !== "live") {
        // Ineligible: Deterministic mock simulation (never call /derived-rescore)
        if (modalDerivedContent) modalDerivedContent.style.display = "none";
        if (modalDerivedUnavailable) {
          modalDerivedUnavailable.style.display = "block";
          if (modalDerivedUnavailableMsg) {
            modalDerivedUnavailableMsg.textContent = "Deterministic Mock Simulation (0 credits consumed, 0 network calls). Offline cross-script re-scoring is designed for upstream provider evaluations.";
          }
        }
        if (modalComputeRescoreArea) modalComputeRescoreArea.style.display = "none";
        return;
      }

      if (overallStatus !== "completed" || !hasTranscripts) {
        // Ineligible: Operational error or incomplete run (never call /derived-rescore)
        if (modalDerivedContent) modalDerivedContent.style.display = "none";
        if (modalDerivedUnavailable) {
          modalDerivedUnavailable.style.display = "block";
          if (modalDerivedUnavailableMsg) {
            modalDerivedUnavailableMsg.textContent = "Operational Provider Error / Incomplete Run. Offline re-scoring is unavailable because complete paired transcripts were not returned by the upstream provider.";
          }
        }
        if (modalComputeRescoreArea) modalComputeRescoreArea.style.display = "none";
        return;
      }

      // Step 4: Eligible completed live run! Check if a derived assessment ALREADY exists (read-only query)
      let existingDerived = null;
      try {
        const derivedListResp = await fetch("/api/eval/derived-assessments?limit=500");
        if (derivedListResp.ok) {
          const derivedList = await derivedListResp.json();
          existingDerived = derivedList.find((d) => d.source_run_id === targetRunId);
        }
      } catch (err) {
        console.warn("Could not list derived assessments:", err);
      }

      if (existingDerived) {
        // Existing derived assessment found: render it directly without side effects
        renderDerivedAssessment(existingDerived);
      } else {
        // Eligible live run without a saved derived assessment:
        // Do NOT silently invoke /derived-rescore. Display explanatory notice and provide an explicit button.
        if (modalDerivedContent) modalDerivedContent.style.display = "none";
        if (modalDerivedUnavailable) {
          modalDerivedUnavailable.style.display = "block";
          if (modalDerivedUnavailableMsg) {
            modalDerivedUnavailableMsg.textContent = "No saved offline derived re-score recorded yet for this completed live run.";
          }
        }
        if (modalComputeRescoreArea) {
          modalComputeRescoreArea.style.display = "block";
          const btnCompute = document.getElementById("btnComputeDerivedRescore");
          if (btnCompute) {
            btnCompute.disabled = false;
            btnCompute.innerHTML = "<span>⚡ Compute &amp; Save Offline Derived Re-Score</span>";
            btnCompute.onclick = async () => {
              btnCompute.disabled = true;
              btnCompute.innerHTML = "<span>⏳ Computing &amp; saving derived re-score...</span>";
              try {
                const rescoreResp = await fetch(`/api/eval/runs/${encodeURIComponent(targetRunId)}/derived-rescore`);
                if (rescoreResp.ok) {
                  const created = await rescoreResp.json();
                  renderDerivedAssessment(created);
                } else {
                  const errData = await rescoreResp.json().catch(() => ({}));
                  if (modalDerivedUnavailableMsg) {
                    modalDerivedUnavailableMsg.textContent = `Error generating derived re-score: ${errData.detail || 'Request failed'}`;
                  }
                  btnCompute.disabled = false;
                  btnCompute.innerHTML = "<span>⚡ Compute &amp; Save Offline Derived Re-Score</span>";
                }
              } catch (err) {
                if (modalDerivedUnavailableMsg) {
                  modalDerivedUnavailableMsg.textContent = `Network error generating derived re-score: ${err.message}`;
                }
                btnCompute.disabled = false;
                btnCompute.innerHTML = "<span>⚡ Compute &amp; Save Offline Derived Re-Score</span>";
              }
            };
          }
        }
      }
    } catch (err) {
      console.error("Failed to load audit modal:", err);
      if (modalDerivedContent) modalDerivedContent.style.display = "none";
      if (modalDerivedUnavailable) {
        modalDerivedUnavailable.style.display = "block";
        if (modalDerivedUnavailableMsg) {
          modalDerivedUnavailableMsg.textContent = "Network error loading evaluation record.";
        }
      }
    }
  }

  if (btnOverviewOpenCompare) {
    btnOverviewOpenCompare.addEventListener("click", () => openRescoreCompareModal());
  }
  if (btnHistoryOpenCompare) {
    btnHistoryOpenCompare.addEventListener("click", () => openRescoreCompareModal());
  }

  // Delegated handler for inspect buttons on history table rows
  document.addEventListener("click", (e) => {
    const btn = e.target.closest(".btn-row-inspect");
    if (btn) {
      const runId = btn.getAttribute("data-run-id");
      const mode = btn.getAttribute("data-mode");
      const status = btn.getAttribute("data-status");
      openRescoreCompareModal(runId, mode, status);
    }
  });

  // Expose methods for automated headless testing
  if (typeof window !== "undefined") {
    window.__bolibench = {
      renderHistoryRows,
      openRescoreCompareModal,
      resetModalFields,
      renderDerivedAssessment,
      handleAudioSelection,
      resetLiveAudioSelection,
      checkLiveEvalFormValidity,
      showConversionStatus,
      requestCreditAuthorization,
      getCurrentConvertedFiles: () => ({ a: currentConvertedFileA, b: currentConvertedFileB }),
      getConversionStates: () => ({ isConvertingA, isConvertingB }),
    };
  }


  // ===========================================================================
  // EVALUATION DATASET SOURCE SELECTOR & HUMAN PAIR RUNNER
  // ===========================================================================

  let activeEvaluationSource = "synthetic_benchmark";
  let currentSelectedHumanPair = null;

  const sourceToggleSynthetic = document.getElementById("sourceToggleSynthetic");
  const sourceToggleHuman = document.getElementById("sourceToggleHuman");
  const syntheticEvalContainer = document.getElementById("syntheticEvalContainer");
  const humanEvalContainer = document.getElementById("humanEvalContainer");

  const humanEvalCategoryFilter = document.getElementById("humanEvalCategoryFilter");
  const humanEvalLanguageFilter = document.getElementById("humanEvalLanguageFilter");
  const humanEvalStatusFilter = document.getElementById("humanEvalStatusFilter");
  const btnRefreshHumanPairs = document.getElementById("btnRefreshHumanPairs");
  const humanPairsTableBody = document.getElementById("humanPairsTableBody");

  // Inspector Elements
  const inspectHumanPairTitle = document.getElementById("inspectHumanPairTitle");
  const inspectHumanPairBadge = document.getElementById("inspectHumanPairBadge");
  const inspectHumanRefA = document.getElementById("inspectHumanRefA");
  const inspectHumanRefB = document.getElementById("inspectHumanRefB");
  const inspectHumanAudioA = document.getElementById("inspectHumanAudioA");
  const inspectHumanAudioB = document.getElementById("inspectHumanAudioB");
  const inspectHumanDistinctionText = document.getElementById("inspectHumanDistinctionText");
  const confirmLiveHumanCreditUsage = document.getElementById("confirmLiveHumanCreditUsage");
  const btnExecuteHumanLiveEval = document.getElementById("btnExecuteHumanLiveEval");
  const btnExecuteHumanMockEval = document.getElementById("btnExecuteHumanMockEval");
  const humanEvalInlineNotice = document.getElementById("humanEvalInlineNotice");
  const humanEvalResultCard = document.getElementById("humanEvalResultCard");
  const humanResultTitle = document.getElementById("humanResultTitle");
  const humanResultDistinctionBadge = document.getElementById("humanResultDistinctionBadge");
  const humanResultBody = document.getElementById("humanResultBody");

  function setEvaluationSource(source) {
    activeEvaluationSource = source;
    if (source === "human_recording") {
      if (sourceToggleHuman) {
        sourceToggleHuman.classList.add("active");
        sourceToggleHuman.setAttribute("aria-checked", "true");
      }
      if (sourceToggleSynthetic) {
        sourceToggleSynthetic.classList.remove("active");
        sourceToggleSynthetic.setAttribute("aria-checked", "false");
      }
      if (syntheticEvalContainer) syntheticEvalContainer.style.display = "none";
      if (humanEvalContainer) humanEvalContainer.style.display = "block";
      loadHumanPairsForEvaluation();
    } else {
      if (sourceToggleSynthetic) {
        sourceToggleSynthetic.classList.add("active");
        sourceToggleSynthetic.setAttribute("aria-checked", "true");
      }
      if (sourceToggleHuman) {
        sourceToggleHuman.classList.remove("active");
        sourceToggleHuman.setAttribute("aria-checked", "false");
      }
      if (syntheticEvalContainer) syntheticEvalContainer.style.display = "block";
      if (humanEvalContainer) humanEvalContainer.style.display = "none";
    }
  }

  if (sourceToggleSynthetic) {
    sourceToggleSynthetic.addEventListener("click", () => setEvaluationSource("synthetic_benchmark"));
  }
  if (sourceToggleHuman) {
    sourceToggleHuman.addEventListener("click", () => setEvaluationSource("human_recording"));
  }

  async function loadHumanPairsForEvaluation() {
    if (!humanPairsTableBody) return;
    try {
      const category = humanEvalCategoryFilter ? humanEvalCategoryFilter.value : "all";
      const language = humanEvalLanguageFilter ? humanEvalLanguageFilter.value : "all";
      const status = humanEvalStatusFilter ? humanEvalStatusFilter.value : "all";

      // Check if sessionStorage specified a pair
      const handoffPairId = (typeof sessionStorage !== "undefined" && sessionStorage && sessionStorage.getItem)
        ? sessionStorage.getItem("bolibench_selected_human_pair")
        : null;

      const params = new URLSearchParams();
      if (!handoffPairId) {
        if (category && category !== "all") params.append("category", category);
        if (language && language !== "all") params.append("language", language);
        if (status && status !== "all") params.append("status", status);
      } else {
        if (humanEvalStatusFilter) humanEvalStatusFilter.value = "all";
        if (humanEvalCategoryFilter) humanEvalCategoryFilter.value = "all";
        if (humanEvalLanguageFilter) humanEvalLanguageFilter.value = "all";
      }

      const resp = await fetch(`/api/eval/human-pairs?${params.toString()}`);
      if (!resp.ok) return;
      const pairs = await resp.json();

      if (pairs.length === 0) {
        humanPairsTableBody.innerHTML = `
          <tr>
            <td colspan="6" class="text-center" style="padding: 2rem; color: var(--text-muted);">
              No saved human contrast pairs matching criteria. Record pairs in the <a href="/studio" style="color: var(--primary);">Human Recording Studio</a>.
            </td>
          </tr>
        `;
        return;
      }

      humanPairsTableBody.innerHTML = pairs.map((p, idx) => {
        const isEligible = Boolean(p.utterance_a && p.utterance_b);
        const isSelected = handoffPairId ? (p.pair_id === handoffPairId) : (idx === 0);

        let statusBadge = '<span class="status-pill status-guard">Incomplete</span>';
        if (p.status === "completed") {
          statusBadge = '<span class="status-pill status-active">✓ Completed</span>';
        } else if (p.status === "ready_for_evaluation") {
          statusBadge = '<span class="status-pill status-active" style="background: rgba(245, 158, 11, 0.2); color: #fbbf24;">⚡ Ready</span>';
        } else if (p.status === "awaiting_verification") {
          statusBadge = '<span class="status-pill status-warning">Awaiting Verification</span>';
        } else if (p.status === "provider_failure") {
          statusBadge = '<span class="status-pill status-error">Provider Failure</span>';
        }

        const transcriptA = p.utterance_a ? (p.utterance_a.verified_transcript || p.utterance_a.reference_transcript || '') : '';
        const transcriptB = p.utterance_b ? (p.utterance_b.verified_transcript || p.utterance_b.reference_transcript || '') : '';

        const uttAText = p.utterance_a
          ? `<strong>"${escapeHtml(transcriptA)}"</strong> <span style="font-size: 0.75rem; color: var(--text-muted);">(${(p.utterance_a.duration_seconds || 0).toFixed(1)}s)</span>`
          : '<span class="text-muted">—</span>';

        const uttBText = p.utterance_b
          ? `<strong>"${escapeHtml(transcriptB)}"</strong> <span style="font-size: 0.75rem; color: var(--text-muted);">(${(p.utterance_b.duration_seconds || 0).toFixed(1)}s)</span>`
          : '<span class="text-muted">—</span>';

        return `
          <tr class="human-pair-row ${isSelected ? 'row-selected' : ''}" data-pair-id="${escapeHtml(p.pair_id)}" style="cursor: pointer;">
            <td style="text-align: center;">
              <input type="radio" name="selectedHumanPairRadio" value="${escapeHtml(p.pair_id)}" ${isSelected ? 'checked' : ''} ${!isEligible ? 'disabled' : ''}>
            </td>
            <td><code>${escapeHtml(p.pair_id)}</code></td>
            <td><span class="category-pill">${escapeHtml(p.category || '')}</span> <span class="lang-pill">${escapeHtml(p.language || '')}</span></td>
            <td>${uttAText}</td>
            <td>${uttBText}</td>
            <td>${statusBadge}</td>
          </tr>
        `;
      }).join("");

      // Bind row clicks
      const rows = humanPairsTableBody.querySelectorAll(".human-pair-row");
      rows.forEach((row) => {
        row.addEventListener("click", () => {
          const pId = row.getAttribute("data-pair-id");
          const pairObj = pairs.find((item) => item.pair_id === pId);
          if (pairObj) {
            rows.forEach((r) => r.classList.remove("row-selected"));
            row.classList.add("row-selected");
            const radio = row.querySelector("input[type=radio]");
            if (radio) radio.checked = true;
            selectHumanPair(pairObj);
          }
        });
      });

      // Initial selection
      const activePair = handoffPairId
        ? pairs.find((item) => item.pair_id === handoffPairId) || pairs[0]
        : pairs[0];

      if (activePair) {
        selectHumanPair(activePair);
      }

      if (handoffPairId) {
        sessionStorage.removeItem("bolibench_selected_human_pair");
      }
    } catch (err) {
      console.error("Failed to load human pairs for evaluation:", err);
    }
  }

  function selectHumanPair(pair) {
    currentSelectedHumanPair = pair;
    if (!pair) return;

    if (inspectHumanPairTitle) {
      inspectHumanPairTitle.textContent = `${pair.pair_id} (${pair.category || 'Contrast Pair'})`;
    }

    const hasBoth = Boolean(pair.utterance_a && pair.utterance_b);
    if (inspectHumanPairBadge) {
      inspectHumanPairBadge.textContent = hasBoth ? "1 Pair Selected (2 Planned STT Requests)" : "Incomplete Pair";
    }

    const transcriptA = pair.utterance_a ? (pair.utterance_a.verified_transcript || pair.utterance_a.reference_transcript || '') : '';
    const transcriptB = pair.utterance_b ? (pair.utterance_b.verified_transcript || pair.utterance_b.reference_transcript || '') : '';

    if (inspectHumanRefA && pair.utterance_a) {
      inspectHumanRefA.textContent = `"${transcriptA}" (${(pair.utterance_a.duration_seconds || 0).toFixed(1)}s)`;
    }
    if (inspectHumanAudioA && pair.utterance_a) {
      inspectHumanAudioA.src = pair.utterance_a.audio_url || `/api/studio/recordings/${encodeURIComponent(pair.utterance_a.recording_id)}/audio`;
    }

    if (inspectHumanRefB && pair.utterance_b) {
      inspectHumanRefB.textContent = `"${transcriptB}" (${(pair.utterance_b.duration_seconds || 0).toFixed(1)}s)`;
    }
    if (inspectHumanAudioB && pair.utterance_b) {
      inspectHumanAudioB.src = pair.utterance_b.audio_url || `/api/studio/recordings/${encodeURIComponent(pair.utterance_b.recording_id)}/audio`;
    }

    if (inspectHumanDistinctionText) {
      inspectHumanDistinctionText.textContent = pair.expected_distinction_desc || pair.expected_contrast || pair.scenario_description || `Tests semantic distinction in ${pair.category}.`;
    }

    updateHumanEvaluationButtonStates();
  }

  function updateHumanEvaluationButtonStates() {
    const hasPair = Boolean(currentSelectedHumanPair && currentSelectedHumanPair.utterance_a && currentSelectedHumanPair.utterance_b);
    const hasConsent = confirmLiveHumanCreditUsage ? confirmLiveHumanCreditUsage.checked : false;

    if (btnExecuteHumanLiveEval) {
      btnExecuteHumanLiveEval.disabled = !(hasPair && hasConsent);
    }
    if (btnExecuteHumanMockEval) {
      btnExecuteHumanMockEval.disabled = !hasPair;
    }

    const hint = document.getElementById("humanEvalBtnHint");
    if (hint) {
      if (!hasPair) {
        hint.textContent = "Please select an eligible pair with both Utterance A & B.";
        hint.style.color = "var(--text-muted)";
      } else if (!hasConsent) {
        hint.textContent = "Please authorize live credit deduction checkbox above to enable live evaluation.";
        hint.style.color = "#fcd34d";
      } else {
        hint.textContent = "✓ 2 STT requests authorized. Ready for live evaluation.";
        hint.style.color = "#6ee7b7";
      }
    }
  }

  if (confirmLiveHumanCreditUsage) {
    confirmLiveHumanCreditUsage.addEventListener("change", updateHumanEvaluationButtonStates);
  }

  if (humanEvalCategoryFilter) humanEvalCategoryFilter.addEventListener("change", loadHumanPairsForEvaluation);
  if (humanEvalLanguageFilter) humanEvalLanguageFilter.addEventListener("change", loadHumanPairsForEvaluation);
  if (humanEvalStatusFilter) humanEvalStatusFilter.addEventListener("change", loadHumanPairsForEvaluation);
  if (btnRefreshHumanPairs) btnRefreshHumanPairs.addEventListener("click", loadHumanPairsForEvaluation);

  // Live Human Evaluation Trigger
  if (btnExecuteHumanLiveEval) {
    btnExecuteHumanLiveEval.addEventListener("click", async () => {
      if (!currentSelectedHumanPair) return;

      if (!confirmLiveHumanCreditUsage || !confirmLiveHumanCreditUsage.checked) {
        alert("Please authorize the live credit usage checkbox before executing live STT requests.");
        return;
      }

      btnExecuteHumanLiveEval.disabled = true;
      if (btnExecuteHumanMockEval) btnExecuteHumanMockEval.disabled = true;
      const textSpan = document.getElementById("btnHumanLiveText");
      if (textSpan) textSpan.textContent = "⏳ Running Live Gnani STT (2 requests)...";

      try {
        // Step 1: Request live authorization token from server-side credit guard
        const authResp = await fetch("/api/eval/authorize-live", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "X-Requested-With": "XMLHttpRequest",
          },
          body: JSON.stringify({
            test_id: currentSelectedHumanPair.pair_id,
          }),
        });

        if (!authResp.ok) {
          const authErr = await authResp.json().catch(() => ({}));
          throw new Error(authErr.detail || `Live evaluation authorization rejected (HTTP ${authResp.status})`);
        }

        const authData = await authResp.json();
        const confirmHeader = authData.required_header || REQUIRED_CONFIRM_HEADER;
        const confirmValue = authData.required_header_value || EXPECTED_CONFIRM_HEADER_VALUE;

        // Step 2: Execute Live Evaluation with issued token and CSRF header
        const resp = await fetch(`/api/eval/run-human-pair-live/${encodeURIComponent(currentSelectedHumanPair.pair_id)}`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "X-Requested-With": "XMLHttpRequest",
            [confirmHeader]: confirmValue,
          },
          body: JSON.stringify({
            confirmation_token: authData.confirmation_token,
            audio_condition: "clean_recorded",
          }),
        });

        if (!resp.ok) {
          const err = await resp.json().catch(() => ({}));
          throw new Error(err.detail || `Server error (${resp.status})`);
        }

        const result = await resp.json();
        renderHumanEvaluationScorecard(result);
        await loadHumanPairsForEvaluation();
        loadBenchmarkMetrics();
        loadFullHistory(currentStatusFilter, currentSourceFilter);
      } catch (err) {
        console.error("Live human evaluation failed:", err);
        if (humanEvalInlineNotice) {
          humanEvalInlineNotice.style.display = "block";
          humanEvalInlineNotice.textContent = `Evaluation execution failed: ${err.message}`;
        }
      } finally {
        if (textSpan) textSpan.textContent = "⚡ Run Live Evaluation (Gnani Prisma)";
        updateHumanEvaluationButtonStates();
      }
    });
  }

  // Mock Human Evaluation Trigger
  if (btnExecuteHumanMockEval) {
    btnExecuteHumanMockEval.addEventListener("click", async () => {
      if (!currentSelectedHumanPair) return;

      btnExecuteHumanMockEval.disabled = true;
      try {
        const resp = await fetch(`/api/eval/run-human-pair-mock/${encodeURIComponent(currentSelectedHumanPair.pair_id)}`, {
          method: "POST",
        });

        if (!resp.ok) {
          const err = await resp.json().catch(() => ({}));
          throw new Error(err.detail || `Server error (${resp.status})`);
        }

        const result = await resp.json();
        renderHumanEvaluationScorecard(result);
        await loadHumanPairsForEvaluation();
        loadBenchmarkMetrics();
        loadFullHistory(currentStatusFilter, currentSourceFilter);
      } catch (err) {
        console.error("Mock human evaluation failed:", err);
        if (humanEvalInlineNotice) {
          humanEvalInlineNotice.style.display = "block";
          humanEvalInlineNotice.textContent = `Mock evaluation failed: ${err.message}`;
        }
      } finally {
        updateHumanEvaluationButtonStates();
      }
    });
  }

  function renderHumanEvaluationScorecard(result) {
    if (!humanEvalResultCard || !humanResultBody) return;
    humanEvalResultCard.style.display = "block";

    const isPreserved = result.semantic_distinction_preserved;
    if (humanResultTitle) {
      humanResultTitle.textContent = `Evaluation Result: ${result.test_case_id} (${result.execution_mode.toUpperCase()})`;
    }

    if (humanResultDistinctionBadge) {
      if (isPreserved) {
        humanResultDistinctionBadge.textContent = "✓ DISTINCTION PRESERVED";
        humanResultDistinctionBadge.className = "status-pill status-active";
      } else {
        humanResultDistinctionBadge.textContent = "❌ DISTINCTION COLLAPSED";
        humanResultDistinctionBadge.className = "status-pill status-error";
      }
    }

    const meanWer = result.average_wer != null ? `${(result.average_wer * 100).toFixed(1)}%` : "N/A";
    const meanNormWer = result.average_normalized_wer != null ? `${(result.average_normalized_wer * 100).toFixed(1)}%` : "N/A";

    const uttA = result.utterance_a || {};
    const uttB = result.utterance_b || {};

    humanResultBody.innerHTML = `
      <div class="stat-box">
        <span class="stat-number">${meanWer}</span>
        <span class="stat-title">Raw Mean WER</span>
      </div>
      <div class="stat-box highlight-stat">
        <span class="stat-number" style="color: #60a5fa;">${meanNormWer}</span>
        <span class="stat-title">Normalized WER</span>
      </div>
      <div class="stat-box">
        <span class="stat-number" style="color: ${isPreserved ? '#34d399' : '#f87171'};">${isPreserved ? 'PRESERVED' : 'COLLAPSED'}</span>
        <span class="stat-title">Semantic Preservation</span>
      </div>
      <div class="stat-box">
        <span class="stat-number">${(result.total_latency_seconds || 0).toFixed(2)}s</span>
        <span class="stat-title">Pair Latency</span>
      </div>
      <div class="stat-box" style="grid-column: 1 / -1; text-align: left; background: rgba(255,255,255,0.03); padding: 0.85rem;">
        <div style="margin-bottom: 0.35rem;">
          <span class="spec-badge badge-a">Utterance A</span>
          Ref: <code>${escapeHtml(uttA.reference_transcript || '')}</code> &bull;
          STT: <strong>${escapeHtml(uttA.actual_transcript || 'Error')}</strong>
          ${uttA.error_message ? `<span style="color: #f87171;">(Error: ${escapeHtml(uttA.error_message)})</span>` : ''}
        </div>
        <div>
          <span class="spec-badge badge-b">Utterance B</span>
          Ref: <code>${escapeHtml(uttB.reference_transcript || '')}</code> &bull;
          STT: <strong>${escapeHtml(uttB.actual_transcript || 'Error')}</strong>
          ${uttB.error_message ? `<span style="color: #f87171;">(Error: ${escapeHtml(uttB.error_message)})</span>` : ''}
        </div>
      </div>
    `;

    humanEvalResultCard.scrollIntoView({ behavior: "smooth", block: "center" });
  }

  function checkHumanPairHandoff() {
    try {
      if (typeof window !== "undefined" && typeof sessionStorage !== "undefined" && sessionStorage && sessionStorage.getItem) {
        const pairId = sessionStorage.getItem("bolibench_selected_human_pair");
        if (pairId) {
          setEvaluationSource("human_recording");
        }
      }
    } catch (e) {
      // Ignore in non-browser or mock environments
    }
  }

  // Handle URL hash routing on initial boot and window hashchange
  function handleUrlHashRouting() {
    try {
      if (typeof window !== "undefined" && window && window.location && window.location.hash) {
        const targetHash = window.location.hash.replace(/^#/, "");
        if (targetHash && panes[targetHash]) {
          switchTab(targetHash);
          return true;
        }
      }
    } catch (e) {
      // Ignore in non-browser or mock environments
    }
    return false;
  }

  if (typeof window !== "undefined" && window && typeof window.addEventListener === "function") {
    window.addEventListener("hashchange", () => {
      handleUrlHashRouting();
    });
  }

  // Also check on initial page load if hash points to a tab pane or handoff exists
  let initialTabRouted = false;
  try {
    initialTabRouted = handleUrlHashRouting();
    if (!initialTabRouted && typeof sessionStorage !== "undefined" && sessionStorage && sessionStorage.getItem && sessionStorage.getItem("bolibench_selected_human_pair")) {
      switchTab("evaluateView");
      initialTabRouted = true;
    }
  } catch (e) {
    // Ignore in non-browser or mock environments
  }


  // ===========================================================================
  // 9. Initial Boot
  // ===========================================================================
  updateSelectedTestCaseSpecs();
  loadBenchmarkMetrics();
  loadBenchmarkCatalogue();
  loadFullHistory("all");

  // Augment window.__bolibench for headless regression testing
  if (typeof window !== "undefined" && window.__bolibench) {
    Object.assign(window.__bolibench, {
      switchTab,
      handleUrlHashRouting,
      checkHumanPairHandoff,
      setEvaluationSource,
      getActiveTabKey: () => activeTabKey,
      getActiveEvaluationSource: () => activeEvaluationSource,
      getCurrentSelectedHumanPair: () => currentSelectedHumanPair,
      panes,
      tabs,
    });
  }
});
