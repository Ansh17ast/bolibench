/**
 * BoliBench — In-Website Human Audio Recording Studio Controller
 *
 * Capabilities:
 * - End-to-end paired contrast workflow: Create Pair -> Record A -> Save A -> Record B -> Save B -> Verify Pair -> Live Evaluate.
 * - Explicit browser microphone permission workflow.
 * - Record, Pause, Resume, Stop, Playback, Save, and Retake controls.
 * - Actionable validation checklist replacing silent button disabling.
 * - Visible 50-second countdown with native accessible <progress> element.
 * - Automatic 50-second cutoff; early-stop detection and incomplete marking.
 * - Real-time Web Audio API waveform visualizer.
 * - Upload to /api/studio/upload preserving unmodified raw audio and receiving standardized 16kHz WAV diagnostics.
 * - Automatic reuse of Pair ID, Speaker, Category, and Language from Utterance A to Utterance B.
 * - Genuine semantic contrast verification (flags identical transcripts and audio).
 * - Direct transition from studio completion to Run Evaluation.
 */

(function () {
  "use strict";

  const TARGET_DURATION_SEC = 50.0;
  const COMPLETION_THRESHOLD_SEC = 49.5;

  // Studio State
  let audioStream = null;
  let audioContext = null;
  let analyserNode = null;
  let mediaRecorder = null;
  let audioChunks = [];
  let recordedBlob = null;
  let recordedAudioUrl = null;
  let animFrameId = null;

  let recordingStartTime = 0;
  let pausedElapsedTime = 0;
  let pauseStartTime = 0;
  let countdownTimerId = null;
  let isRecording = false;
  let isPaused = false;
  let lastClientDuration = 0.0;
  let isRefUserEdited = false;
  let isVerUserEdited = false;
  let isSaveInProgress = false;
  let isMicRequestInProgress = false;
  const SESSION_PAIR_KEY = "bolibench_studio_in_progress_pair";

  // Active Contrast Pair Context
  let currentPair = {
    pairId: "human_qty_001",
    speakerId: "spk_indian_01",
    language: "hinglish",
    category: "quantities_and_numbers",
    scenarioDescription: "",
    expectedContrast: "",
    refTranscriptA: "",
    verTranscriptA: "",
    uttARecordingId: null,
    uttBRecordingId: null,
    currentStep: 1, // 1: Define Pair, 2: Record A, 3: Record B, 4: Complete
  };

  // DOM Elements Cache
  const el = {};

  function initElements() {
    // Stepper
    el.pairsProgressBadge = document.getElementById("studioPairsProgressBadge");
    el.stepNav1 = document.getElementById("stepNav1");
    el.stepNav2 = document.getElementById("stepNav2");
    el.stepNav3 = document.getElementById("stepNav3");
    el.stepNav4 = document.getElementById("stepNav4");

    // Microphone & Status
    el.micPermissionBtn = document.getElementById("btnStudioRequestMic");
    el.micStatusBadge = document.getElementById("studioMicStatusBadge");
    el.micStatusText = document.getElementById("studioMicStatusText");
    el.micDeviceSelect = document.getElementById("studioMicDeviceSelect");

    // Controls
    el.btnRecord = document.getElementById("btnStudioRecord");
    el.btnPause = document.getElementById("btnStudioPause");
    el.btnStop = document.getElementById("btnStudioStop");
    el.btnRetake = document.getElementById("btnStudioRetake");
    el.btnSave = document.getElementById("btnStudioSave");
    el.btnSaveText = document.getElementById("btnStudioSaveText");

    // Displays
    el.statePill = document.getElementById("studioStatePill");
    el.countdownTimer = document.getElementById("studioCountdownTimer");
    el.elapsedTimer = document.getElementById("studioElapsedTimer");
    el.countdownProgress = document.getElementById("studioCountdownProgress");
    el.waveformCanvas = document.getElementById("studioWaveformCanvas");
    el.waveformStatusOverlay = document.getElementById("studioWaveformOverlay");
    el.currentRolePill = document.getElementById("studioCurrentRolePill");
    el.activePairIdDisplay = document.getElementById("studioActivePairIdDisplay");
    el.formRoleTag = document.getElementById("studioFormRoleTag");

    // Contrast Comparison Box
    el.uttAContrastRefBox = document.getElementById("studioUttAContrastRefBox");
    el.uttAContrastText = document.getElementById("studioUttAContrastText");

    // Actionable Validation Checklist & Alert
    el.validationChecklist = document.getElementById("studioValidationChecklist");
    el.readinessBadge = document.getElementById("studioReadinessBadge");
    el.checkItemMic = document.getElementById("checkItemMic");
    el.checkItemAudio = document.getElementById("checkItemAudio");
    el.checkItemRef = document.getElementById("checkItemRef");
    el.checkItemVer = document.getElementById("checkItemVer");
    el.checkItemConsent = document.getElementById("checkItemConsent");
    el.saveErrorAlert = document.getElementById("studioSaveErrorAlert");
    el.saveErrorMessage = document.getElementById("studioSaveErrorMessage");

    // Transitions
    el.transitionToBBanner = document.getElementById("studioTransitionToBBanner");
    el.btnProceedToB = document.getElementById("btnStudioProceedToB");
    el.pairCompleteBanner = document.getElementById("studioPairCompleteBanner");
    el.btnLaunchEval = document.getElementById("btnStudioLaunchEval");
    el.btnCreateNextPair = document.getElementById("btnStudioCreateNextPair");
    el.btnNewPairReset = document.getElementById("btnStudioNewPairReset");

    // Review & Warning Elements
    el.reviewSection = document.getElementById("studioReviewSection");
    el.reviewPlayer = document.getElementById("studioReviewPlayer");
    el.earlyStopWarning = document.getElementById("studioEarlyStopWarning");
    el.completeNotice = document.getElementById("studioCompleteNotice");
    el.uploadProgressCard = document.getElementById("studioUploadProgressCard");
    el.uploadProgress = document.getElementById("studioUploadProgress");
    el.uploadStatusText = document.getElementById("studioUploadStatusText");
    el.diagnosticsCard = document.getElementById("studioDiagnosticsCard");
    el.diagnosticsBody = document.getElementById("studioDiagnosticsBody");

    // Step 1: Pair Definition Inputs
    el.pairIdInput = document.getElementById("studioPairId");
    el.btnGeneratePairId = document.getElementById("btnStudioGeneratePairId");
    el.categorySelect = document.getElementById("studioCategorySelect");
    el.speakerIdInput = document.getElementById("studioSpeakerId");
    el.btnRandomSpeaker = document.getElementById("btnStudioRandomSpeaker");
    el.languageSelect = document.getElementById("studioLanguageSelect");
    el.scenarioDescription = document.getElementById("studioScenarioDescription");
    el.expectedContrast = document.getElementById("studioExpectedContrast");
    el.scenarioTemplateSelect = document.getElementById("studioScenarioTemplateSelect");
    el.btnSavePairSpec = document.getElementById("btnStudioSavePairSpec");

    // Step 2 & 3: Utterance Form Inputs
    el.recordingIdInput = document.getElementById("studioRecordingId");
    el.btnRefreshId = document.getElementById("btnStudioRefreshId");
    el.utteranceRoleSelect = document.getElementById("studioUtteranceRoleSelect");
    el.referenceTranscript = document.getElementById("studioReferenceTranscript");
    el.verifiedTranscript = document.getElementById("studioVerifiedTranscript");
    el.transcriptVerifiedCheckbox = document.getElementById("studioTranscriptVerified");
    el.btnCopyRefToVer = document.getElementById("btnStudioCopyRefToVer");
    el.consentCheckbox = document.getElementById("studioConsentCheckbox");
    el.consentText = document.getElementById("studioConsentText");

    // Library & Pair Elements
    el.libraryTableBody = document.getElementById("studioLibraryTableBody");
    el.pairsTableBody = document.getElementById("studioPairsTableBody");
    el.btnRefreshLibrary = document.getElementById("btnStudioRefreshLibrary");

    // Pair Verification Modal
    el.verifyModal = document.getElementById("studioVerifyPairModal");
    el.verifyModalPairId = document.getElementById("studioVerifyModalPairId");
    el.verifyModalUttAText = document.getElementById("studioVerifyModalUttAText");
    el.verifyModalUttBText = document.getElementById("studioVerifyModalUttBText");
    el.verifyModalDistinction = document.getElementById("studioVerifyModalDistinction");
    el.verifyModalCheckbox = document.getElementById("studioVerifyModalCheckbox");
    el.verifyModalNotes = document.getElementById("studioVerifyModalNotes");
    el.btnModalConfirmVerify = document.getElementById("btnStudioModalConfirmVerify");
    el.btnModalCancelVerify = document.getElementById("btnStudioModalCancelVerify");
  }

  // ---------------------------------------------------------------------------
  // Workflow Stepper Management
  // ---------------------------------------------------------------------------

  function setWorkflowStep(stepNum) {
    currentPair.currentStep = stepNum;

    const steps = [el.stepNav1, el.stepNav2, el.stepNav3, el.stepNav4];
    steps.forEach((stepEl, idx) => {
      if (!stepEl) return;
      const stepIdx = idx + 1;
      stepEl.classList.remove("active", "completed");
      if (stepIdx < stepNum) {
        stepEl.classList.add("completed");
      } else if (stepIdx === stepNum) {
        stepEl.classList.add("active");
      }
    });

    if (stepNum === 2) {
      if (el.currentRolePill) {
        el.currentRolePill.className = "spec-badge badge-a";
        el.currentRolePill.textContent = "Utterance A (Baseline)";
      }
      if (el.formRoleTag) el.formRoleTag.textContent = "Role: Utterance A";
      if (el.btnSaveText) el.btnSaveText.textContent = "💾 Save Utterance A";
      if (el.utteranceRoleSelect) el.utteranceRoleSelect.value = "utterance_a";
      if (el.uttAContrastRefBox) el.uttAContrastRefBox.style.display = "none";
    } else if (stepNum === 3) {
      if (el.currentRolePill) {
        el.currentRolePill.className = "spec-badge badge-b";
        el.currentRolePill.textContent = "Utterance B (Contrast)";
      }
      if (el.formRoleTag) el.formRoleTag.textContent = "Role: Utterance B";
      if (el.btnSaveText) el.btnSaveText.textContent = "💾 Save Utterance B";
      if (el.utteranceRoleSelect) el.utteranceRoleSelect.value = "utterance_b";
      if (el.uttAContrastRefBox) {
        el.uttAContrastRefBox.style.display = "flex";
        if (el.uttAContrastText) {
          el.uttAContrastText.textContent = `"${currentPair.refTranscriptA || ''}"`;
        }
      }
    }
  }

  // ---------------------------------------------------------------------------
  // Resource Cleanup & Session Persistence Helpers
  // ---------------------------------------------------------------------------

  function stopActiveAudioStream() {
    if (audioStream) {
      try {
        audioStream.getTracks().forEach((track) => {
          try { track.stop(); } catch (_) {}
        });
      } catch (e) {
        console.debug("Error stopping audio tracks:", e);
      }
      audioStream = null;
    }
  }

  function closeActiveAudioContext() {
    if (animFrameId) {
      try { cancelAnimationFrame(animFrameId); } catch (_) {}
      animFrameId = null;
    }
    if (audioContext) {
      try {
        if (audioContext.state !== "closed" && typeof audioContext.close === "function") {
          audioContext.close().catch((err) => console.debug("AudioContext close error:", err));
        }
      } catch (e) {
        console.debug("AudioContext close error:", e);
      }
      audioContext = null;
      analyserNode = null;
    }
  }

  function persistPairSessionState() {
    try {
      if (typeof sessionStorage !== "undefined" && sessionStorage && sessionStorage.setItem) {
        const stateToSave = {
          pairId: currentPair.pairId,
          speakerId: currentPair.speakerId,
          language: currentPair.language,
          category: currentPair.category,
          scenarioDescription: currentPair.scenarioDescription,
          expectedContrast: currentPair.expectedContrast,
          refTranscriptA: currentPair.refTranscriptA,
          verTranscriptA: currentPair.verTranscriptA,
          uttARecordingId: currentPair.uttARecordingId,
          currentStep: currentPair.currentStep,
          timestamp: Date.now(),
        };
        sessionStorage.setItem(SESSION_PAIR_KEY, JSON.stringify(stateToSave));
      }
    } catch (e) {
      console.warn("Could not persist pair session state:", e);
    }
  }

  function clearPairSessionState() {
    try {
      if (typeof sessionStorage !== "undefined" && sessionStorage && sessionStorage.removeItem) {
        sessionStorage.removeItem(SESSION_PAIR_KEY);
      }
    } catch (e) {
      console.warn("Could not clear pair session state:", e);
    }
  }

  async function tryResumeInProgressPair() {
    try {
      if (typeof sessionStorage === "undefined" || !sessionStorage || !sessionStorage.getItem) {
        return false;
      }
      const raw = sessionStorage.getItem(SESSION_PAIR_KEY);
      if (!raw) return false;

      let saved;
      try {
        saved = JSON.parse(raw);
      } catch (e) {
        clearPairSessionState();
        return false;
      }

      if (!saved || !saved.pairId || !saved.uttARecordingId) {
        clearPairSessionState();
        return false;
      }

      // Verify with backend that the recording still exists
      try {
        const checkResp = await fetch(`/api/studio/recordings/${encodeURIComponent(saved.uttARecordingId)}`);
        if (!checkResp.ok) {
          console.info("In-progress pair recording not found on server; resetting session.");
          clearPairSessionState();
          return false;
        }
      } catch (err) {
        console.warn("Could not verify in-progress pair with server:", err);
        return false;
      }

      // Restore pair context
      currentPair.pairId = saved.pairId;
      currentPair.speakerId = saved.speakerId || "spk_indian_01";
      currentPair.language = saved.language || "hinglish";
      currentPair.category = saved.category || "quantities_and_numbers";
      currentPair.scenarioDescription = saved.scenarioDescription || "";
      currentPair.expectedContrast = saved.expectedContrast || "";
      currentPair.refTranscriptA = saved.refTranscriptA || "";
      currentPair.verTranscriptA = saved.verTranscriptA || "";
      currentPair.uttARecordingId = saved.uttARecordingId;
      currentPair.uttBRecordingId = null;

      // Update UI elements
      if (el.pairIdInput) el.pairIdInput.value = currentPair.pairId;
      if (el.activePairIdDisplay) el.activePairIdDisplay.textContent = currentPair.pairId;
      if (el.speakerIdInput) el.speakerIdInput.value = currentPair.speakerId;
      if (el.languageSelect) el.languageSelect.value = currentPair.language;
      if (el.categorySelect) el.categorySelect.value = currentPair.category;
      if (el.scenarioDescription) el.scenarioDescription.value = currentPair.scenarioDescription;
      if (el.expectedContrast) el.expectedContrast.value = currentPair.expectedContrast;

      // Set workflow to step 3 (Utterance B)
      setWorkflowStep(3);
      if (el.utteranceRoleSelect) el.utteranceRoleSelect.value = "utterance_b";
      if (el.formRoleTag) el.formRoleTag.textContent = "Role: Utterance B";
      if (el.btnSaveText) el.btnSaveText.textContent = "💾 Save Utterance B";

      // Show Utterance A reference box
      if (el.uttAContrastRefBox) {
        el.uttAContrastRefBox.style.display = "flex";
        if (el.uttAContrastText) {
          el.uttAContrastText.textContent = `"${currentPair.verTranscriptA || currentPair.refTranscriptA || ''}"`;
        }
      }

      // Show transition banner offering to record B or start fresh
      if (el.transitionToBBanner) {
        el.transitionToBBanner.style.display = "flex";
        const label = el.transitionToBBanner.querySelector(".active-pair-label");
        if (label) label.textContent = currentPair.pairId;
      }

      generateNewRecordingId();
      updateValidationChecklist();
      return true;
    } catch (e) {
      console.warn("Error in tryResumeInProgressPair:", e);
      clearPairSessionState();
      return false;
    }
  }

  // ---------------------------------------------------------------------------
  // Microphone & Permissions Workflow
  // ---------------------------------------------------------------------------

  async function checkInitialPermission() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setMicStatus("unsupported", "Microphone API not supported in this browser.");
      if (el.btnRecord) el.btnRecord.disabled = true;
      updateValidationChecklist();
      return;
    }

    if (navigator.permissions && navigator.permissions.query) {
      try {
        const permissionStatus = await navigator.permissions.query({ name: "microphone" });
        if (permissionStatus.state === "granted") {
          await requestMicrophonePermission(false);
        } else if (permissionStatus.state === "denied") {
          setMicStatus("denied", "Microphone access blocked. Enable permissions in browser address bar.");
        } else {
          setMicStatus("prompt", "Microphone ready to initialize. Click 'Enable Microphone'.");
        }

        permissionStatus.onchange = () => {
          if (permissionStatus.state === "granted") {
            requestMicrophonePermission(false);
          } else if (permissionStatus.state === "denied") {
            setMicStatus("denied", "Microphone access was denied.");
          }
          updateValidationChecklist();
        };
      } catch (err) {
        setMicStatus("prompt", "Click 'Enable Microphone' to grant audio permission.");
      }
    } else {
      setMicStatus("prompt", "Click 'Enable Microphone' to grant audio permission.");
    }
    updateValidationChecklist();
  }

  async function requestMicrophonePermission(userInitiated = true) {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      if (typeof alert === "function") {
        alert("Microphone recording is not supported in this browser environment.");
      }
      return;
    }

    if (isMicRequestInProgress) return;
    isMicRequestInProgress = true;

    setMicStatus("requesting", "Requesting microphone permission from browser...");

    try {
      const constraints = {
        audio: {
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
        },
      };

      if (el.micDeviceSelect && el.micDeviceSelect.value) {
        constraints.audio.deviceId = { exact: el.micDeviceSelect.value };
      }

      // Stop previous tracks and close previous context BEFORE requesting new one
      stopActiveAudioStream();
      closeActiveAudioContext();

      audioStream = await navigator.mediaDevices.getUserMedia(constraints);

      // Populate input devices list preserving selection
      await enumerateAudioDevices();

      // Initialize Web Audio API Analyser for live visualizer
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      if (AudioCtx) {
        audioContext = new AudioCtx();
        const source = audioContext.createMediaStreamSource(audioStream);
        analyserNode = audioContext.createAnalyser();
        analyserNode.fftSize = 256;
        source.connect(analyserNode);
        startVisualizer();
      }

      setMicStatus("granted", "Microphone active & ready.");
      if (el.btnRecord) el.btnRecord.disabled = false;
      if (el.micPermissionBtn) {
        el.micPermissionBtn.textContent = "✓ Microphone Active";
        el.micPermissionBtn.classList.remove("btn-primary");
        el.micPermissionBtn.classList.add("btn-secondary");
      }
      setStatePill("MIC READY", "status-active");
    } catch (err) {
      console.warn("Microphone access error:", err);
      setMicStatus("denied", `Permission denied or mic error: ${err.name || err.message}`);
      if (!audioStream && el.btnRecord) el.btnRecord.disabled = true;
      setStatePill("MIC BLOCKED", "status-error");
    } finally {
      isMicRequestInProgress = false;
    }
    updateValidationChecklist();
  }

  async function enumerateAudioDevices() {
    if (!navigator.mediaDevices.enumerateDevices || !el.micDeviceSelect) return;
    try {
      const currentSelectedId = el.micDeviceSelect.value;
      const devices = await navigator.mediaDevices.enumerateDevices();
      const audioInputs = devices.filter((d) => d.kind === "audioinput");

      el.micDeviceSelect.innerHTML = "";
      if (audioInputs.length === 0) {
        const opt = document.createElement("option");
        opt.textContent = "Default Microphone";
        el.micDeviceSelect.appendChild(opt);
        return;
      }

      let hasMatch = false;
      audioInputs.forEach((device, idx) => {
        const opt = document.createElement("option");
        opt.value = device.deviceId;
        opt.textContent = device.label || `Microphone ${idx + 1}`;
        if (currentSelectedId && device.deviceId === currentSelectedId) {
          opt.selected = true;
          hasMatch = true;
        }
        el.micDeviceSelect.appendChild(opt);
      });

      if (hasMatch) {
        el.micDeviceSelect.value = currentSelectedId;
      } else if (audioInputs.length > 0 && !currentSelectedId) {
        el.micDeviceSelect.value = audioInputs[0].deviceId;
      }
    } catch (e) {
      console.debug("Could not enumerate audio devices:", e);
    }
  }

  function setMicStatus(state, message) {
    if (!el.micStatusBadge || !el.micStatusText) return;
    el.micStatusText.textContent = message;

    el.micStatusBadge.classList.remove("status-active", "status-warning", "status-error");
    if (state === "granted") {
      el.micStatusBadge.classList.add("status-active");
    } else if (state === "denied" || state === "unsupported") {
      el.micStatusBadge.classList.add("status-error");
    } else {
      el.micStatusBadge.classList.add("status-warning");
    }
  }

  function setStatePill(label, className) {
    if (!el.statePill) return;
    el.statePill.textContent = label;
    el.statePill.className = `status-pill ${className || ""}`;
  }

  // ---------------------------------------------------------------------------
  // Waveform Visualizer
  // ---------------------------------------------------------------------------

  function startVisualizer() {
    if (!el.waveformCanvas || !analyserNode) return;
    const canvas = el.waveformCanvas;
    const ctx = canvas.getContext("2d");
    const bufferLength = analyserNode.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    function draw() {
      animFrameId = requestAnimationFrame(draw);
      analyserNode.getByteTimeDomainData(dataArray);

      ctx.fillStyle = "#0b0f19";
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      if (!isRecording && !isPaused) {
        ctx.lineWidth = 1.5;
        ctx.strokeStyle = "rgba(59, 130, 246, 0.25)";
        ctx.beginPath();
        ctx.moveTo(0, canvas.height / 2);
        ctx.lineTo(canvas.width, canvas.height / 2);
        ctx.stroke();
        return;
      }

      ctx.lineWidth = 2.5;
      ctx.strokeStyle = isPaused ? "#f59e0b" : "#3b82f6";

      ctx.beginPath();
      const sliceWidth = (canvas.width * 1.0) / bufferLength;
      let x = 0;

      for (let i = 0; i < bufferLength; i++) {
        const v = dataArray[i] / 128.0;
        const y = (v * canvas.height) / 2;

        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
        x += sliceWidth;
      }

      ctx.lineTo(canvas.width, canvas.height / 2);
      ctx.stroke();
    }

    if (animFrameId) cancelAnimationFrame(animFrameId);
    draw();
  }

  // ---------------------------------------------------------------------------
  // Recording & 50-Second Countdown Control
  // ---------------------------------------------------------------------------

  function startRecording() {
    if (!audioStream) {
      showSaveError("Microphone is not active. Please click 'Enable Microphone' first.");
      if (el.micPermissionBtn) el.micPermissionBtn.focus();
      return;
    }

    hideSaveError();

    // Reset recording buffers and review state
    audioChunks = [];
    recordedBlob = null;
    if (recordedAudioUrl) {
      URL.revokeObjectURL(recordedAudioUrl);
      recordedAudioUrl = null;
    }

    if (el.reviewSection) el.reviewSection.style.display = "none";
    if (el.earlyStopWarning) el.earlyStopWarning.style.display = "none";
    if (el.completeNotice) el.completeNotice.style.display = "none";
    if (el.diagnosticsCard) el.diagnosticsCard.style.display = "none";
    if (el.uploadProgressCard) el.uploadProgressCard.style.display = "none";
    if (el.transitionToBBanner) el.transitionToBBanner.style.display = "none";
    if (el.pairCompleteBanner) el.pairCompleteBanner.style.display = "none";

    let options = {};
    if (MediaRecorder.isTypeSupported("audio/webm;codecs=opus")) {
      options.mimeType = "audio/webm;codecs=opus";
    } else if (MediaRecorder.isTypeSupported("audio/ogg;codecs=opus")) {
      options.mimeType = "audio/ogg;codecs=opus";
    } else if (MediaRecorder.isTypeSupported("audio/wav")) {
      options.mimeType = "audio/wav";
    }

    try {
      mediaRecorder = new MediaRecorder(audioStream, options);
    } catch (e) {
      mediaRecorder = new MediaRecorder(audioStream);
    }

    mediaRecorder.ondataavailable = (event) => {
      if (event.data && event.data.size > 0) {
        audioChunks.push(event.data);
      }
    };

    mediaRecorder.onstop = handleRecordingStopped;
    mediaRecorder.start(100);

    isRecording = true;
    isPaused = false;
    pausedElapsedTime = 0;
    recordingStartTime = performance.now();

    el.btnRecord.disabled = true;
    el.btnPause.disabled = false;
    el.btnPause.textContent = "⏸ Pause";
    el.btnStop.disabled = false;
    el.btnRetake.disabled = true;

    setStatePill("RECORDING (LIVE)", "status-active status-recording-pulse");
    if (el.waveformStatusOverlay) {
      el.waveformStatusOverlay.textContent = "🔴 Recording in progress — 50-second benchmark standard";
    }

    runCountdownLoop();
    updateValidationChecklist();
  }

  function runCountdownLoop() {
    clearInterval(countdownTimerId);

    countdownTimerId = setInterval(() => {
      if (!isRecording) {
        clearInterval(countdownTimerId);
        return;
      }

      if (isPaused) return;

      const now = performance.now();
      const elapsedMs = now - recordingStartTime - pausedElapsedTime;
      const elapsedSec = Math.min(TARGET_DURATION_SEC, Math.max(0.0, elapsedMs / 1000.0));
      const remainingSec = Math.max(0.0, TARGET_DURATION_SEC - elapsedSec);

      lastClientDuration = elapsedSec;

      if (el.countdownTimer) {
        el.countdownTimer.textContent = `${remainingSec.toFixed(1)}s remaining`;
      }
      if (el.elapsedTimer) {
        el.elapsedTimer.textContent = `${elapsedSec.toFixed(1)}s / ${TARGET_DURATION_SEC.toFixed(1)}s`;
      }
      if (el.countdownProgress) {
        el.countdownProgress.value = elapsedSec;
      }

      if (elapsedSec >= TARGET_DURATION_SEC) {
        clearInterval(countdownTimerId);
        stopRecording(true);
      }
    }, 50);
  }

  function pauseRecording() {
    if (!mediaRecorder || !isRecording) return;

    if (!isPaused) {
      if (typeof mediaRecorder.pause === "function") {
        mediaRecorder.pause();
      }
      isPaused = true;
      pauseStartTime = performance.now();
      el.btnPause.textContent = "▶ Resume";
      setStatePill("PAUSED", "status-warning");
      if (el.waveformStatusOverlay) {
        el.waveformStatusOverlay.textContent = "⏸ Recording paused";
      }
    } else {
      if (typeof mediaRecorder.resume === "function") {
        mediaRecorder.resume();
      }
      pausedElapsedTime += performance.now() - pauseStartTime;
      isPaused = false;
      el.btnPause.textContent = "⏸ Pause";
      setStatePill("RECORDING (LIVE)", "status-active status-recording-pulse");
      if (el.waveformStatusOverlay) {
        el.waveformStatusOverlay.textContent = "🔴 Recording in progress";
      }
    }
  }

  function stopRecording(autoStop = false) {
    if (!isRecording || !mediaRecorder) return;

    clearInterval(countdownTimerId);
    isRecording = false;
    isPaused = false;

    if (mediaRecorder.state !== "inactive") {
      mediaRecorder.stop();
    }

    if (el.waveformStatusOverlay) {
      el.waveformStatusOverlay.textContent = autoStop
        ? "✓ 50-Second automatic cutoff reached."
        : "⏹ Recording stopped by user.";
    }
  }

  function handleRecordingStopped() {
    const mime = (mediaRecorder && mediaRecorder.mimeType) || "audio/webm";
    recordedBlob = new Blob(audioChunks, { type: mime });
    recordedAudioUrl = URL.createObjectURL(recordedBlob);

    if (el.reviewPlayer) {
      el.reviewPlayer.src = recordedAudioUrl;
      el.reviewPlayer.load();

      // Listen for decoded audio metadata to verify actual playable duration
      el.reviewPlayer.onloadedmetadata = () => {
        if (el.reviewPlayer.duration && isFinite(el.reviewPlayer.duration) && el.reviewPlayer.duration > 0) {
          const actualSec = el.reviewPlayer.duration;
          if (Math.abs(actualSec - lastClientDuration) > 1.5) {
            console.warn(`Duration discrepancy: timer=${lastClientDuration.toFixed(2)}s, actual=${actualSec.toFixed(2)}s`);
            lastClientDuration = actualSec;
            updateRecordingCompletionNotice();
            updateValidationChecklist();
          }
        }
      };
    }

    // Decode actual audio frames using Web Audio API to calculate true decoded duration
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    if (AudioCtx && recordedBlob) {
      recordedBlob.arrayBuffer().then((buffer) => {
        const decodeCtx = new AudioCtx();
        decodeCtx.decodeAudioData(buffer.slice(0))
          .then((decodedAudio) => {
            const actualSec = decodedAudio.duration;
            console.info(`Decoded audio frames: duration = ${actualSec.toFixed(2)}s, sampleRate = ${decodedAudio.sampleRate}, length = ${decodedAudio.length}`);
            lastClientDuration = actualSec;
            updateRecordingCompletionNotice();
            updateValidationChecklist();
          })
          .catch((err) => {
            console.warn("Could not decode audio frames via Web Audio API:", err);
          })
          .finally(() => {
            decodeCtx.close().catch(() => {});
          });
      }).catch((e) => {
        console.warn("ArrayBuffer conversion error:", e);
      });
    }

    if (el.reviewSection) {
      el.reviewSection.style.display = "block";
    }

    updateRecordingCompletionNotice();

    // If verified transcript is empty and user hasn't typed in it yet,
    // pre-fill with reference transcript as a convenient starting point for manual review
    // (preserving whatever reference transcript the user actually entered)
    if (el.verifiedTranscript && !isVerUserEdited && !el.verifiedTranscript.value.trim()) {
      const refVal = (el.referenceTranscript && el.referenceTranscript.value.trim()) || "";
      if (refVal) {
        el.verifiedTranscript.value = refVal;
      }
    }

    el.btnRecord.disabled = false;
    el.btnPause.disabled = true;
    el.btnPause.textContent = "⏸ Pause";
    el.btnStop.disabled = true;
    el.btnRetake.disabled = false;

    updateValidationChecklist();
  }

  function updateRecordingCompletionNotice() {
    const isComplete = lastClientDuration >= COMPLETION_THRESHOLD_SEC;

    if (isComplete) {
      if (el.completeNotice) {
        el.completeNotice.style.display = "flex";
        const txt = el.completeNotice.querySelector(".notice-text");
        if (txt) {
          txt.textContent = `Valid 50-Second Recording (${lastClientDuration.toFixed(1)}s). Ready for review and storage.`;
        }
      }
      if (el.earlyStopWarning) el.earlyStopWarning.style.display = "none";
      setStatePill("50S COMPLETE", "status-active");
    } else {
      if (el.earlyStopWarning) {
        el.earlyStopWarning.style.display = "flex";
        const txt = el.earlyStopWarning.querySelector(".warning-text");
        if (txt) {
          txt.textContent = `Clip stopped early at ${lastClientDuration.toFixed(1)} seconds. Marked as INCOMPLETE. Per BoliBench benchmark rules, this clip will be saved as 'incomplete_early_stop' and does not count as a valid 50-second recording.`;
        }
      }
      if (el.completeNotice) el.completeNotice.style.display = "none";
      setStatePill("INCOMPLETE (EARLY STOP)", "status-warning");
    }
  }

  function retakeRecording() {
    if (isRecording) {
      stopRecording(false);
    }

    if (recordedAudioUrl) {
      URL.revokeObjectURL(recordedAudioUrl);
      recordedAudioUrl = null;
    }
    recordedBlob = null;
    audioChunks = [];
    lastClientDuration = 0.0;

    if (el.reviewSection) el.reviewSection.style.display = "none";
    if (el.earlyStopWarning) el.earlyStopWarning.style.display = "none";
    if (el.completeNotice) el.completeNotice.style.display = "none";
    if (el.diagnosticsCard) el.diagnosticsCard.style.display = "none";
    if (el.reviewPlayer) el.reviewPlayer.src = "";

    if (el.countdownTimer) el.countdownTimer.textContent = `${TARGET_DURATION_SEC.toFixed(1)}s remaining`;
    if (el.elapsedTimer) el.elapsedTimer.textContent = `0.0s / ${TARGET_DURATION_SEC.toFixed(1)}s`;
    if (el.countdownProgress) el.countdownProgress.value = 0;

    if (el.waveformStatusOverlay) {
      el.waveformStatusOverlay.textContent = "Microphone ready. Click 'Start 50s Recording'.";
    }

    el.btnRecord.disabled = !audioStream;
    el.btnPause.disabled = true;
    el.btnStop.disabled = true;
    el.btnRetake.disabled = true;

    setStatePill("MIC READY", "status-active");
    updateValidationChecklist();
  }

  // ---------------------------------------------------------------------------
  // Actionable Save Validation Checklist & Feedback
  // ---------------------------------------------------------------------------

  function getValidationState() {
    const micActive = Boolean(audioStream);
    const hasAudio = Boolean(recordedBlob && recordedBlob.size > 0 && !isRecording);
    const hasRef = Boolean(el.referenceTranscript && el.referenceTranscript.value.trim().length > 0);
    const hasVer = Boolean(el.verifiedTranscript && el.verifiedTranscript.value.trim().length > 0);
    const hasConsent = Boolean(el.consentCheckbox && el.consentCheckbox.checked);

    const isAllValid = micActive && hasAudio && hasRef && hasVer && hasConsent;

    return {
      micActive,
      hasAudio,
      hasRef,
      hasVer,
      hasConsent,
      isAllValid,
    };
  }

  function updateValidationChecklist() {
    const state = getValidationState();

    updateChecklistItem(el.checkItemMic, state.micActive, "Microphone active");
    updateChecklistItem(el.checkItemAudio, state.hasAudio, state.hasAudio ? `Audio captured (${lastClientDuration.toFixed(1)}s)` : "Audio captured (> 0s)");
    updateChecklistItem(el.checkItemRef, state.hasRef, "Intended transcript");
    updateChecklistItem(el.checkItemVer, state.hasVer, "Verified transcript");
    updateChecklistItem(el.checkItemConsent, state.hasConsent, "Explicit consent checkbox");

    if (el.readinessBadge) {
      if (state.isAllValid) {
        el.readinessBadge.textContent = "✓ Ready to Save";
        el.readinessBadge.className = "spec-badge badge-active";
      } else if (!state.hasAudio) {
        el.readinessBadge.textContent = "Needs Audio Recording";
        el.readinessBadge.className = "badge-pill-source-synth";
      } else if (!state.hasConsent) {
        el.readinessBadge.textContent = "Explicit Consent Required";
        el.readinessBadge.className = "spec-badge badge-warning";
      } else {
        el.readinessBadge.textContent = "Incomplete Fields";
        el.readinessBadge.className = "badge-pill-source-synth";
      }
    }
  }

  function updateChecklistItem(itemEl, isValid, labelText) {
    if (!itemEl) return;
    itemEl.classList.remove("valid", "invalid");
    itemEl.classList.add(isValid ? "valid" : "invalid");
    const icon = itemEl.querySelector(".check-icon");
    if (icon) icon.textContent = isValid ? "✓" : "✕";
  }

  function showSaveError(message) {
    if (!el.saveErrorAlert || !el.saveErrorMessage) return;
    el.saveErrorMessage.textContent = message;
    el.saveErrorAlert.style.display = "flex";
  }

  function hideSaveError() {
    if (el.saveErrorAlert) el.saveErrorAlert.style.display = "none";
  }

  // ---------------------------------------------------------------------------
  // Save & Upload to Backend
  // ---------------------------------------------------------------------------

  async function handleSaveClick() {
    if (isSaveInProgress) return;
    const state = getValidationState();

    if (isRecording) {
      showSaveError("Recording is currently active. Please click '⏹ Stop' before saving.");
      if (el.btnStop) el.btnStop.focus();
      return;
    }

    if (!state.micActive) {
      showSaveError("Microphone permission required. Click '🎙️ Enable Microphone' to initialize your audio device.");
      if (el.micPermissionBtn) el.micPermissionBtn.focus();
      return;
    }

    if (!state.hasAudio) {
      showSaveError("No recording captured. Click '🔴 Start 50s Recording' and record your voice sample first.");
      if (el.btnRecord) el.btnRecord.focus();
      return;
    }

    if (!state.hasRef) {
      showSaveError("Intended reference transcript required. Please specify what the speaker intended to say.");
      if (el.referenceTranscript) el.referenceTranscript.focus();
      return;
    }

    if (!state.hasConsent) {
      showSaveError("Explicit consent confirmation required! You must review the consent terms and check the consent box before saving.");
      if (el.consentCheckbox) el.consentCheckbox.focus();
      const consentBox = document.getElementById("studioConsentBox");
      if (consentBox) {
        consentBox.scrollIntoView({ behavior: "smooth", block: "center" });
        consentBox.style.boxShadow = "0 0 15px rgba(239, 68, 68, 0.4)";
        setTimeout(() => { consentBox.style.boxShadow = ""; }, 2500);
      }
      return;
    }

    // Semantic contrast check for Utterance B
    const utteranceRole = el.utteranceRoleSelect ? el.utteranceRoleSelect.value : "utterance_a";
    const refTranscript = (el.referenceTranscript && el.referenceTranscript.value.trim()) || "";

    if (utteranceRole === "utterance_b" && currentPair.refTranscriptA) {
      if (refTranscript.toLowerCase() === currentPair.refTranscriptA.toLowerCase()) {
        showSaveError(`Contrast check failed: Utterance B reference transcript cannot be identical to Utterance A ("${currentPair.refTranscriptA}"). BoliBench paired benchmark requires a genuine semantic distinction!`);
        if (el.referenceTranscript) el.referenceTranscript.focus();
        return;
      }
    }

    hideSaveError();
    await saveRecordingToBackend();
  }

  async function saveRecordingToBackend() {
    if (isSaveInProgress) return;
    isSaveInProgress = true;

    // Immediately disable Save and Retake synchronously before ANY await
    if (el.btnSave) el.btnSave.disabled = true;
    if (el.btnRetake) el.btnRetake.disabled = true;

    const speakerId = (el.speakerIdInput && el.speakerIdInput.value.trim()) || "spk_indian_01";
    const language = el.languageSelect ? el.languageSelect.value : "hinglish";
    const category = el.categorySelect ? el.categorySelect.value : "quantities_and_numbers";
    const refTranscript = (el.referenceTranscript && el.referenceTranscript.value.trim()) || "";
    const verTranscript = (el.verifiedTranscript && el.verifiedTranscript.value.trim()) || refTranscript;
    const isTranscriptVerified = el.transcriptVerifiedCheckbox ? el.transcriptVerifiedCheckbox.checked : false;
    const recId = (el.recordingIdInput && el.recordingIdInput.value.trim()) || "";
    const pairId = (el.pairIdInput && el.pairIdInput.value.trim()) || "human_qty_001";
    const utteranceRole = el.utteranceRoleSelect ? el.utteranceRoleSelect.value : "utterance_a";
    const consentStatement = el.consentText ? el.consentText.textContent.trim() : "I consent to record and use audio for BoliBench.";

    if (el.uploadProgressCard) el.uploadProgressCard.style.display = "block";
    if (el.uploadProgress) el.uploadProgress.value = 15;
    if (el.uploadStatusText) el.uploadStatusText.textContent = "Preparing recording upload...";

    try {
      // Ensure contrast pair definition exists with accurate scenario and expected distinction
      try {
        await fetch("/api/studio/pairs/create", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            pair_id: pairId,
            category: category,
            language: language,
            speaker_id: speakerId,
            scenario_description: (el.scenarioDescription && el.scenarioDescription.value.trim()) || null,
            expected_distinction: (el.expectedContrast && el.expectedContrast.value.trim()) || null,
          }),
        });
      } catch (e) {
        console.warn("Pair definition creation warning:", e);
      }

      // Client-side WAV conversion integration via BoliAudioConverter if available
      let uploadFile = recordedBlob;
      let uploadFilename = `recording${recordedBlob && recordedBlob.type && recordedBlob.type.includes("wav") ? ".wav" : recordedBlob && recordedBlob.type && recordedBlob.type.includes("ogg") ? ".ogg" : ".webm"}`;

      const converter = (typeof window !== "undefined" && (window.BoliAudioConverter || window.BoliBenchAudioConverter)) || null;
      if (converter && typeof converter.convertAudioFile === "function" && recordedBlob) {
        try {
          if (el.uploadProgress) el.uploadProgress.value = 35;
          if (el.uploadStatusText) el.uploadStatusText.textContent = "Converting audio to standard 16kHz mono PCM WAV in-browser...";

          const convResult = await converter.convertAudioFile(recordedBlob);
          if (convResult && (convResult.file || convResult.convertedFile)) {
            uploadFile = convResult.file || convResult.convertedFile;
            uploadFilename = `${recId || "recording"}_pcm.wav`;
            if (convResult.duration && isFinite(convResult.duration) && convResult.duration > 0) {
              lastClientDuration = convResult.duration;
            }
            console.info(`Client WAV conversion successful: ${lastClientDuration.toFixed(2)}s, ${uploadFile.size} bytes`);
          }
        } catch (convErr) {
          console.warn("Client-side WAV conversion failed:", convErr);
          throw new Error(`In-browser WAV audio standardization failed: ${convErr.message}`);
        }
      }

      if (el.uploadProgress) el.uploadProgress.value = 60;
      if (el.uploadStatusText) el.uploadStatusText.textContent = "Uploading audio to BoliBench server...";

      const formData = new FormData();
      formData.append("audio_file", uploadFile, uploadFilename);
      formData.append("speaker_id", speakerId);
      formData.append("language", language);
      formData.append("category", category);
      formData.append("reference_transcript", refTranscript);
      formData.append("consent_given", "true");
      formData.append("consent_statement", consentStatement);
      if (recId) formData.append("recording_id", recId);
      if (pairId) formData.append("pair_id", pairId);
      formData.append("utterance_role", utteranceRole);
      if (verTranscript) formData.append("verified_transcript", verTranscript);
      formData.append("is_transcript_verified", isTranscriptVerified ? "true" : "false");
      formData.append("client_duration_seconds", lastClientDuration.toFixed(3));

      const response = await fetch("/api/studio/upload", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Upload failed with HTTP status ${response.status}`);
      }

      const result = await response.json();

      if (el.uploadProgress) el.uploadProgress.value = 100;
      if (el.uploadStatusText) el.uploadStatusText.textContent = `✓ Utterance ${utteranceRole === "utterance_a" ? "A" : "B"} stored successfully! (ID: ${result.recording_id})`;

      // Render Diagnostics
      renderDiagnostics(result);

      // Refresh Library Table and Candidate Pairs
      await loadStudioRecordings();
      await loadStudioPairs();

      setStatePill("SAVED TO STORAGE", "status-active");

      // Handle Workflow Transitions
      if (utteranceRole === "utterance_a") {
        currentPair.pairId = pairId;
        currentPair.speakerId = speakerId;
        currentPair.language = language;
        currentPair.category = category;
        currentPair.refTranscriptA = refTranscript;
        currentPair.verTranscriptA = verTranscript;
        currentPair.uttARecordingId = result.recording_id;

        // Persist in-progress pair to session storage so refresh resumes Utterance B
        persistPairSessionState();

        // Advance to step 3 in Stepper
        setWorkflowStep(3);

        if (el.transitionToBBanner) {
          el.transitionToBBanner.style.display = "flex";
          const label = el.transitionToBBanner.querySelector(".active-pair-label");
          if (label) label.textContent = pairId;
          el.transitionToBBanner.scrollIntoView({ behavior: "smooth", block: "center" });
        }
      } else if (utteranceRole === "utterance_b") {
        currentPair.uttBRecordingId = result.recording_id;

        // Pair is complete: clear in-progress pair session storage
        clearPairSessionState();

        // Advance to step 4 in Stepper
        setWorkflowStep(4);

        if (el.pairCompleteBanner) {
          el.pairCompleteBanner.style.display = "flex";
          el.pairCompleteBanner.scrollIntoView({ behavior: "smooth", block: "center" });
        }
      }

      generateNewRecordingId();
    } catch (err) {
      console.error("Save error:", err);
      showSaveError(`Upload failed: ${err.message}. Please retry.`);
      if (el.uploadStatusText) el.uploadStatusText.textContent = `❌ Error: ${err.message}`;
      if (err.message && err.message.includes("already exists")) {
        generateNewRecordingId();
      }
    } finally {
      isSaveInProgress = false;
      if (el.btnSave) el.btnSave.disabled = false;
      if (el.btnRetake) el.btnRetake.disabled = !recordedBlob;
      updateValidationChecklist();
    }
  }


  // ---------------------------------------------------------------------------
  // Transition Actions: Advance to B, Next Pair, Evaluate
  // ---------------------------------------------------------------------------

  function proceedToUtteranceB() {
    if (el.transitionToBBanner) el.transitionToBBanner.style.display = "none";
    if (el.pairCompleteBanner) el.pairCompleteBanner.style.display = "none";

    // Set role to Utterance B
    if (el.utteranceRoleSelect) el.utteranceRoleSelect.value = "utterance_b";
    if (el.formRoleTag) el.formRoleTag.textContent = "Role: Utterance B";

    // Reuse Pair ID, Speaker ID, Language, Category from A
    if (el.pairIdInput) el.pairIdInput.value = currentPair.pairId;
    if (el.speakerIdInput) el.speakerIdInput.value = currentPair.speakerId;
    if (el.languageSelect) el.languageSelect.value = currentPair.language;
    if (el.categorySelect) el.categorySelect.value = currentPair.category;

    // Display Utterance A's ACTUAL reference transcript in the contrast comparison box
    if (el.uttAContrastRefBox) el.uttAContrastRefBox.style.display = "block";
    if (el.uttAContrastText) el.uttAContrastText.textContent = `"${currentPair.refTranscriptA || ''}"`;

    // Clear transcripts for Utterance B - NEVER overwrite with hardcoded suggestions
    if (el.referenceTranscript) el.referenceTranscript.value = "";
    if (el.verifiedTranscript) el.verifiedTranscript.value = "";
    if (el.transcriptVerifiedCheckbox) el.transcriptVerifiedCheckbox.checked = false;
    isRefUserEdited = false;
    isVerUserEdited = false;

    // Clear recorded audio buffer for new recording
    retakeRecording();
    generateNewRecordingId();

    setWorkflowStep(3);

    // Focus reference transcript input or record button
    if (el.referenceTranscript) {
      el.referenceTranscript.focus();
      el.referenceTranscript.scrollIntoView({ behavior: "smooth", block: "center" });
    } else if (el.btnRecord) {
      el.btnRecord.focus();
    }
  }

  function createNextPair() {
    clearPairSessionState();
    if (el.pairCompleteBanner) el.pairCompleteBanner.style.display = "none";
    if (el.transitionToBBanner) el.transitionToBBanner.style.display = "none";
    if (el.uttAContrastRefBox) el.uttAContrastRefBox.style.display = "none";

    // Auto-generate next pair ID
    generateNextPairId();

    // Reset role to utterance_a
    if (el.utteranceRoleSelect) el.utteranceRoleSelect.value = "utterance_a";
    if (el.formRoleTag) el.formRoleTag.textContent = "Role: Utterance A";

    // Clear transcripts cleanly - do not set hardcoded defaults
    if (el.referenceTranscript) el.referenceTranscript.value = "";
    if (el.verifiedTranscript) el.verifiedTranscript.value = "";
    if (el.transcriptVerifiedCheckbox) el.transcriptVerifiedCheckbox.checked = false;
    isRefUserEdited = false;
    isVerUserEdited = false;

    currentPair.refTranscriptA = "";
    currentPair.verTranscriptA = "";
    currentPair.uttARecordingId = null;
    currentPair.uttBRecordingId = null;

    retakeRecording();
    generateNewRecordingId();

    setWorkflowStep(1);
    if (el.scenarioDescription) el.scenarioDescription.focus();
  }

  async function savePairSpecification() {
    const pairId = (el.pairIdInput && el.pairIdInput.value.trim()) || currentPair.pairId;
    const category = el.categorySelect ? el.categorySelect.value : "quantities_and_numbers";
    const language = el.languageSelect ? el.languageSelect.value : "hinglish";
    const speakerId = (el.speakerIdInput && el.speakerIdInput.value.trim()) || "spk_indian_01";
    const scenarioDesc = (el.scenarioDescription && el.scenarioDescription.value.trim()) || "";
    const expectedDist = (el.expectedContrast && el.expectedContrast.value.trim()) || "";

    if (!pairId) return;

    try {
      const resp = await fetch("/api/studio/pairs/create", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          pair_id: pairId,
          category: category,
          language: language,
          speaker_id: speakerId,
          scenario_description: scenarioDesc,
          expected_distinction: expectedDist,
        }),
      });
      if (resp.ok) {
        currentPair.pairId = pairId;
        currentPair.speakerId = speakerId;
        currentPair.language = language;
        currentPair.category = category;
        currentPair.scenarioDescription = scenarioDesc;
        currentPair.expectedContrast = expectedDist;
        if (el.btnSavePairSpec) {
          const oldText = el.btnSavePairSpec.innerHTML;
          el.btnSavePairSpec.innerHTML = "<span>✓ Pair Saved</span>";
          setTimeout(() => { if (el.btnSavePairSpec) el.btnSavePairSpec.innerHTML = oldText; }, 2000);
        }
        await loadStudioPairs();
      }
    } catch (e) {
      console.warn("Failed to create pair definition:", e);
    }
  }

  function launchEvaluationForPair() {
    const pairId = (el.pairIdInput && el.pairIdInput.value.trim()) || currentPair.pairId;
    if (!pairId) return;

    // Save selected pair to sessionStorage for Run Evaluation page to pick up automatically
    sessionStorage.setItem("bolibench_selected_human_pair", pairId);
    window.location.href = "/#evaluateView";
  }

  function generateNextPairId() {
    if (!el.pairIdInput) return;
    const cat = el.categorySelect ? el.categorySelect.value : "quantities_and_numbers";
    const prefixMap = {
      quantities_and_numbers: "human_qty",
      negation: "human_neg",
      corrections_and_mind_changes: "human_cor",
      named_entities: "human_ent",
      command_intent: "human_cmd",
      time_extraction: "human_tim",
    };
    const pfx = prefixMap[cat] || "human_pair";

    fetch("/api/studio/pairs")
      .then((r) => r.json())
      .then((pairs) => {
        let maxNum = 0;
        pairs.forEach((p) => {
          if (p.pair_id && p.pair_id.startsWith(pfx)) {
            const parts = p.pair_id.split("_");
            const num = parseInt(parts[parts.length - 1], 10);
            if (!isNaN(num) && num > maxNum) maxNum = num;
          }
        });
        const nextNum = String(maxNum + 1).padStart(3, "0");
        const nextId = `${pfx}_${nextNum}`;
        el.pairIdInput.value = nextId;
        if (el.activePairIdDisplay) el.activePairIdDisplay.textContent = nextId;
        currentPair.pairId = nextId;
      })
      .catch(() => {
        const fallbackId = `${pfx}_001`;
        el.pairIdInput.value = fallbackId;
        if (el.activePairIdDisplay) el.activePairIdDisplay.textContent = fallbackId;
      });
  }

  function renderDiagnostics(meta) {
    if (!el.diagnosticsCard || !el.diagnosticsBody) return;
    el.diagnosticsCard.style.display = "block";

    const diag = meta.diagnostics || {};
    const statusVal = meta.validation_status || "valid_50s";

    let statusBadge = '<span class="spec-badge badge-active">✓ Valid 50s</span>';
    if (statusVal === "incomplete_early_stop") {
      statusBadge = '<span class="spec-badge badge-warning">⚠️ Incomplete Early Stop</span>';
    } else if (statusVal === "silent_warning") {
      statusBadge = '<span class="spec-badge badge-warning">⚠️ Silent Audio Detected</span>';
    } else if (statusVal === "clipped_warning") {
      statusBadge = '<span class="spec-badge badge-error">⚠️ Acoustic Clipping</span>';
    }

    el.diagnosticsBody.innerHTML = `
      <div class="diagnostic-grid">
        <div class="diag-item">
          <span class="diag-label">Recording ID</span>
          <span class="diag-value"><code>${meta.recording_id}</code></span>
        </div>
        <div class="diag-item">
          <span class="diag-label">Validation Status</span>
          <span class="diag-value">${statusBadge}</span>
        </div>
        <div class="diag-item">
          <span class="diag-label">Standard Format</span>
          <span class="diag-value">${diag.sample_rate_hz || 16000} Hz, ${diag.channels === 1 ? "Mono" : "Stereo"}, 16-bit PCM WAV</span>
        </div>
        <div class="diag-item">
          <span class="diag-label">Validated Duration</span>
          <span class="diag-value"><strong>${(diag.duration_seconds || 0).toFixed(2)}s</strong> (${diag.is_complete_50s ? "50s standard met" : "Early stop"})</span>
        </div>
        <div class="diag-item">
          <span class="diag-label">RMS Amplitude (Silence Check)</span>
          <span class="diag-value">${(diag.rms_amplitude || 0).toFixed(4)} (${diag.is_silent ? "⚠️ Below speech threshold" : "✓ Audible speech"})</span>
        </div>
        <div class="diag-item">
          <span class="diag-label">Clipping / Distortion</span>
          <span class="diag-value">${diag.clipped_samples_count || 0} clipped frames (${diag.is_clipped ? "⚠️ Saturation warning" : "✓ Clean headroom"})</span>
        </div>
        <div class="diag-item">
          <span class="diag-label">Storage Locations</span>
          <span class="diag-value">
            Raw: <code>${meta.raw_audio_path}</code><br>
            Standard WAV: <code>${meta.standardized_audio_path}</code>
          </span>
        </div>
        <div class="diag-item">
          <span class="diag-label">Transcript Verification</span>
          <span class="diag-value">${meta.is_transcript_verified ? "✓ Manually Verified" : "⏳ Pending Manual Verification"}</span>
        </div>
      </div>
    `;
  }

  // ---------------------------------------------------------------------------
  // Library & Matched Pairs
  // ---------------------------------------------------------------------------

  async function loadStudioRecordings() {
    if (!el.libraryTableBody) return;
    try {
      const resp = await fetch("/api/studio/recordings");
      if (!resp.ok) return;
      const records = await resp.json();

      if (records.length === 0) {
        el.libraryTableBody.innerHTML = `
          <tr>
            <td colspan="8" class="text-center" style="padding: 2rem; color: var(--text-muted);">
              No human audio recordings saved yet. Record a 50-second clip above to build your dataset.
            </td>
          </tr>
        `;
        return;
      }

      el.libraryTableBody.innerHTML = records
        .map((r) => {
          const statusBadge = r.is_complete_50s
            ? '<span class="status-pill status-active">✓ 50s Valid</span>'
            : '<span class="status-pill status-warning">⚠️ Incomplete Early Stop</span>';

          const verBadge = r.is_transcript_verified
            ? '<span class="spec-badge badge-active">✓ Verified</span>'
            : '<span class="spec-badge badge-muted">Unverified</span>';

          const roleBadge = r.utterance_role === "utterance_a"
            ? '<span class="spec-badge badge-a">Part A</span>'
            : r.utterance_role === "utterance_b"
            ? '<span class="spec-badge badge-b">Part B</span>'
            : '<span class="spec-badge">Single</span>';

          const transcriptText = r.verified_transcript || r.reference_transcript || "—";

          return `
            <tr>
              <td><code>${escapeHtml(r.recording_id)}</code></td>
              <td><code>${escapeHtml(r.speaker_id)}</code></td>
              <td><span class="category-pill">${escapeHtml(r.category)}</span></td>
              <td>${roleBadge} ${r.pair_id ? `<code>${escapeHtml(r.pair_id)}</code>` : ""}</td>
              <td><strong>${(r.duration_seconds || 0).toFixed(1)}s</strong></td>
              <td>${statusBadge}</td>
              <td>
                <div style="max-width: 260px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;" title="${escapeHtml(transcriptText)}">
                  ${escapeHtml(transcriptText)}
                </div>
                <div style="margin-top: 0.25rem;">${verBadge}</div>
              </td>
              <td>
                <audio controls preload="none" src="/api/studio/recordings/${encodeURIComponent(r.recording_id)}/audio" style="height: 32px; max-width: 180px;"></audio>
              </td>
            </tr>
          `;
        })
        .join("");
    } catch (e) {
      console.error("Failed to load studio recordings:", e);
    }
  }

  async function loadStudioPairs() {
    if (!el.pairsTableBody) return;
    try {
      const resp = await fetch("/api/studio/pairs");
      if (!resp.ok) return;
      const pairs = await resp.json();

      // Update progress counter
      const completedCount = pairs.filter((p) => p.utterance_a && p.utterance_b).length;
      if (el.pairsProgressBadge) {
        el.pairsProgressBadge.textContent = `${completedCount} / 50 pairs recorded`;
      }

      if (pairs.length === 0) {
        el.pairsTableBody.innerHTML = `
          <tr>
            <td colspan="6" class="text-center" style="padding: 2rem; color: var(--text-muted);">
              No candidate contrastive pairs found. Record Utterance A and Utterance B to test contrastive distinctions.
            </td>
          </tr>
        `;
        return;
      }

      el.pairsTableBody.innerHTML = pairs
        .map((p) => {
          const statusVal = p.status || (p.verification?.is_verified_pair ? "ready_for_evaluation" : "incomplete");
          let statusBadge = "";

          if (statusVal === "completed") {
            statusBadge = '<span class="status-pill status-active">✓ Completed</span>';
          } else if (statusVal === "ready_for_evaluation") {
            statusBadge = '<span class="status-pill status-active" style="background: rgba(245, 158, 11, 0.2); color: #fbbf24;">⚡ Ready for Evaluation</span>';
          } else if (statusVal === "awaiting_verification") {
            statusBadge = '<span class="status-pill status-warning">Awaiting Verification</span>';
          } else if (statusVal === "provider_failure") {
            statusBadge = '<span class="status-pill status-error">Provider Failure</span>';
          } else {
            statusBadge = '<span class="status-pill status-guard">Incomplete (Clips Pending)</span>';
          }

          const uttAText = p.utterance_a
            ? `<div style="display: flex; flex-direction: column; gap: 0.2rem;">
                 <code>${p.utterance_a.recording_id}</code>
                 <span style="font-size: 0.8rem; color: var(--text-muted);">${(p.utterance_a.duration_seconds || 0).toFixed(1)}s &bull; ${p.utterance_a.is_transcript_verified ? "✓ Verified" : "⏳ Pending"}</span>
                 <audio controls preload="none" src="${p.utterance_a.audio_url || `/api/studio/recordings/${encodeURIComponent(p.utterance_a.recording_id)}/audio`}" style="height: 28px; width: 140px;"></audio>
               </div>`
            : '<span class="text-muted">Missing Utterance A</span>';

          const uttBText = p.utterance_b
            ? `<div style="display: flex; flex-direction: column; gap: 0.2rem;">
                 <code>${p.utterance_b.recording_id}</code>
                 <span style="font-size: 0.8rem; color: var(--text-muted);">${(p.utterance_b.duration_seconds || 0).toFixed(1)}s &bull; ${p.utterance_b.is_transcript_verified ? "✓ Verified" : "⏳ Pending"}</span>
                 <audio controls preload="none" src="${p.utterance_b.audio_url || `/api/studio/recordings/${encodeURIComponent(p.utterance_b.recording_id)}/audio`}" style="height: 28px; width: 140px;"></audio>
               </div>`
            : '<span class="text-muted">Missing Utterance B</span>';

          let actionButtons = "";
          if (p.utterance_a && !p.utterance_b) {
            actionButtons = `<button type="button" class="btn-primary btn-sm" onclick="window.BoliStudio.prepareRecordB('${escapeHtml(p.pair_id)}')">Record B &rarr;</button>`;
          } else if (statusVal === "awaiting_verification" || (!p.verification?.is_verified_pair && p.is_ready_for_verification)) {
            actionButtons = `
              <div style="display: flex; gap: 0.35rem; flex-wrap: wrap;">
                <button type="button" class="btn-secondary btn-sm" onclick="window.BoliStudio.openVerifyModal('${escapeHtml(p.pair_id)}')">Verify</button>
                <button type="button" class="btn-primary btn-sm" onclick="window.BoliStudio.goToEvaluation('${escapeHtml(p.pair_id)}')">⚡ Evaluate</button>
              </div>`;
          } else if (statusVal === "ready_for_evaluation" || statusVal === "completed") {
            actionButtons = `<button type="button" class="btn-primary btn-sm" style="background: linear-gradient(135deg, #d97706, #b45309);" onclick="window.BoliStudio.goToEvaluation('${escapeHtml(p.pair_id)}')">⚡ Evaluate</button>`;
          } else {
            actionButtons = `<span class="text-muted" style="font-size: 0.8rem;">Clips Required</span>`;
          }

          return `
            <tr>
              <td><code>${escapeHtml(p.pair_id)}</code></td>
              <td><span class="category-pill">${escapeHtml(p.category || "")}</span></td>
              <td>${uttAText}</td>
              <td>${uttBText}</td>
              <td>${statusBadge}</td>
              <td>${actionButtons}</td>
            </tr>
          `;
        })
        .join("");
    } catch (e) {
      console.error("Failed to load studio pairs:", e);
    }
  }

  // ---------------------------------------------------------------------------
  // Pair Verification Modal Workflow
  // ---------------------------------------------------------------------------

  let currentVerifyingPairId = null;

  async function openVerifyModal(pairId) {
    currentVerifyingPairId = pairId;
    if (!el.verifyModal) return;

    try {
      const resp = await fetch("/api/studio/pairs");
      const pairs = await resp.json();
      const pair = pairs.find((p) => p.pair_id === pairId);
      if (!pair) return;

      const metaA = await (await fetch(`/api/studio/recordings/${encodeURIComponent(pair.utterance_a.recording_id)}`)).json();
      const metaB = await (await fetch(`/api/studio/recordings/${encodeURIComponent(pair.utterance_b.recording_id)}`)).json();

      if (el.verifyModalPairId) el.verifyModalPairId.textContent = pairId;
      if (el.verifyModalUttAText) el.verifyModalUttAText.textContent = metaA.verified_transcript || metaA.reference_transcript;
      if (el.verifyModalUttBText) el.verifyModalUttBText.textContent = metaB.verified_transcript || metaB.reference_transcript;
      if (el.verifyModalDistinction) {
        el.verifyModalDistinction.textContent = `A (${(metaA.verified_transcript || metaA.reference_transcript)}) vs B (${(metaB.verified_transcript || metaB.reference_transcript)})`;
      }
      if (el.verifyModalCheckbox) el.verifyModalCheckbox.checked = false;
      if (el.verifyModalNotes) el.verifyModalNotes.value = "";

      el.verifyModal.style.display = "flex";
    } catch (e) {
      alert(`Could not load pair details: ${e.message}`);
    }
  }

  async function submitPairVerification() {
    if (!currentVerifyingPairId) return;
    if (!el.verifyModalCheckbox || !el.verifyModalCheckbox.checked) {
      alert("You must explicitly confirm that this pair tests a genuine semantic distinction.");
      return;
    }

    const notes = el.verifyModalNotes ? el.verifyModalNotes.value.trim() : "";

    try {
      const resp = await fetch(`/api/studio/pairs/${encodeURIComponent(currentVerifyingPairId)}/verify`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          confirmed_semantic_distinction: true,
          operator_notes: notes,
          verified_by: "human_studio_operator",
        }),
      });

      if (!resp.ok) {
        const err = await resp.json();
        throw new Error(err.detail || "Pair verification rejected.");
      }

      alert(`Pair '${currentVerifyingPairId}' verified and sealed successfully! Ready for live evaluation.`);
      closeVerifyModal();
      await loadStudioPairs();
    } catch (e) {
      alert(`Verification error: ${e.message}`);
    }
  }

  function closeVerifyModal() {
    if (el.verifyModal) el.verifyModal.style.display = "none";
    currentVerifyingPairId = null;
  }

  // ---------------------------------------------------------------------------
  // External Shortcuts & Helpers
  // ---------------------------------------------------------------------------

  function prepareRecordB(pairId) {
    fetch("/api/studio/pairs")
      .then((r) => r.json())
      .then((pairs) => {
        const p = pairs.find((item) => item.pair_id === pairId);
        if (p && p.utterance_a) {
          currentPair.pairId = p.pair_id;
          currentPair.speakerId = p.speaker_id || (el.speakerIdInput && el.speakerIdInput.value) || "spk_indian_01";
          currentPair.language = p.language || (el.languageSelect && el.languageSelect.value) || "hinglish";
          currentPair.category = p.category || (el.categorySelect && el.categorySelect.value) || "quantities_and_numbers";
          currentPair.refTranscriptA = p.utterance_a.reference_transcript || "";
          if (p.scenario_description && el.scenarioDescription) {
            el.scenarioDescription.value = p.scenario_description;
          }
          if ((p.expected_distinction_desc || p.expected_distinction) && el.expectedContrast) {
            el.expectedContrast.value = p.expected_distinction_desc || p.expected_distinction;
          }
          proceedToUtteranceB();
        }
      });
  }

  function goToEvaluation(pairId) {
    sessionStorage.setItem("bolibench_selected_human_pair", pairId);
    window.location.href = "/#evaluateView";
  }

  function generateNewRecordingId() {
    if (!el.recordingIdInput) return;
    const now = new Date();
    const dateStr = now.toISOString().slice(0, 10).replace(/-/g, "");
    const rand = Math.random().toString(36).substring(2, 8);
    const spk = (el.speakerIdInput && el.speakerIdInput.value.replace(/[^a-zA-Z0-9]/g, "").slice(0, 8)) || "spk01";
    el.recordingIdInput.value = `rec_${dateStr}_${spk}_${rand}`;
  }

  function generateRandomSpeakerId() {
    if (!el.speakerIdInput) return;
    const rand = Math.random().toString(36).substring(2, 6);
    el.speakerIdInput.value = `spk_indian_${rand}`;
    generateNewRecordingId();
  }

  function handleScenarioTemplateSelected() {
    if (!el.scenarioTemplateSelect) return;
    const selectedId = el.scenarioTemplateSelect.value;
    if (!selectedId) return;

    // Guard against active recording
    if (isRecording) {
      if (typeof alert === "function") {
        alert("Please stop or cancel the current recording before changing scenario templates.");
      }
      return;
    }

    // Guard against overwriting an in-progress pair without explicit user confirmation
    if (currentPair.uttARecordingId && currentPair.currentStep >= 3) {
      const confirmChange = (typeof confirm === "function")
        ? confirm(
            `Pair "${currentPair.pairId}" is currently in progress (Utterance A has been saved).\nLoading a new scenario template will abandon this in-progress pair.\n\nDo you want to proceed and start a new pair?`
          )
        : true;
      if (!confirmChange) {
        if (el.scenarioTemplateSelect) el.scenarioTemplateSelect.value = "";
        return;
      }
      createNextPair();
    }

    fetch("/api/studio/config")
      .then((r) => r.json())
      .then((cfg) => {
        const tmpl = (cfg.scenario_templates || []).find((t) => t.id === selectedId);
        if (tmpl) {
          if (el.categorySelect) el.categorySelect.value = tmpl.category;
          if (el.languageSelect) el.languageSelect.value = tmpl.language;
          currentPair.category = tmpl.category;
          currentPair.language = tmpl.language;

          // Keep Pair ID, currentPair.pairId, and active pair display strictly synchronized
          const targetPairId = tmpl.pair_id || selectedId.replace(/_b$/, "");
          if (el.pairIdInput) {
            el.pairIdInput.value = targetPairId;
          }
          currentPair.pairId = targetPairId;
          if (el.activePairIdDisplay) {
            el.activePairIdDisplay.textContent = targetPairId;
          }

          // Update scenario description and expected distinction from the template
          const desc = tmpl.title || tmpl.distinction || "";
          if (el.scenarioDescription) {
            el.scenarioDescription.value = desc;
          }
          currentPair.scenarioDescription = desc;

          const distinction = tmpl.distinction || "";
          if (el.expectedContrast) {
            el.expectedContrast.value = distinction;
          }
          currentPair.expectedContrast = distinction;

          // Distinguish intentionally customized transcript from stale template transcript
          if (isRefUserEdited) {
            const confirmOverwrite = (typeof confirm === "function")
              ? confirm("You have entered a custom reference transcript. Do you want to replace it with the template script?")
              : true;
            if (confirmOverwrite) {
              if (el.referenceTranscript) {
                el.referenceTranscript.value = tmpl.reference_transcript || "";
                isRefUserEdited = false;
              }
            } else {
              console.info("Preserving custom reference transcript; scenario template loaded without overwriting your script.");
            }
          } else {
            if (el.referenceTranscript) {
              el.referenceTranscript.value = tmpl.reference_transcript || "";
              isRefUserEdited = false;
            }
          }

          // If verified transcript has not been user-edited, update it to match new reference transcript
          if (el.verifiedTranscript && !isVerUserEdited) {
            el.verifiedTranscript.value = tmpl.reference_transcript || "";
          }

          if (tmpl.role === "utterance_b") {
            setWorkflowStep(3);
            if (el.utteranceRoleSelect) el.utteranceRoleSelect.value = "utterance_b";
            if (el.formRoleTag) el.formRoleTag.textContent = "Role: Utterance B";
          } else {
            setWorkflowStep(2);
            if (el.utteranceRoleSelect) el.utteranceRoleSelect.value = "utterance_a";
            if (el.formRoleTag) el.formRoleTag.textContent = "Role: Utterance A";
          }

          generateNewRecordingId();
          updateValidationChecklist();
        }
      });
  }

  function escapeHtml(str) {
    if (!str) return "";
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  // ---------------------------------------------------------------------------
  // Initialization & Event Binding
  // ---------------------------------------------------------------------------

  function initStudio() {
    initElements();

    if (el.micPermissionBtn) {
      el.micPermissionBtn.addEventListener("click", () => requestMicrophonePermission(true));
    }
    if (el.micDeviceSelect) {
      el.micDeviceSelect.addEventListener("change", () => requestMicrophonePermission(true));
    }

    if (el.btnRecord) el.btnRecord.addEventListener("click", startRecording);
    if (el.btnPause) el.btnPause.addEventListener("click", pauseRecording);
    if (el.btnStop) el.btnStop.addEventListener("click", () => stopRecording(false));
    if (el.btnRetake) el.btnRetake.addEventListener("click", retakeRecording);
    if (el.btnSave) el.btnSave.addEventListener("click", handleSaveClick);

    if (el.btnProceedToB) el.btnProceedToB.addEventListener("click", proceedToUtteranceB);
    if (el.btnCreateNextPair) el.btnCreateNextPair.addEventListener("click", createNextPair);
    if (el.btnLaunchEval) el.btnLaunchEval.addEventListener("click", launchEvaluationForPair);
    if (el.btnNewPairReset) el.btnNewPairReset.addEventListener("click", createNextPair);
    if (el.btnGeneratePairId) el.btnGeneratePairId.addEventListener("click", generateNextPairId);
    if (el.btnSavePairSpec) el.btnSavePairSpec.addEventListener("click", savePairSpecification);

    if (el.btnRefreshId) el.btnRefreshId.addEventListener("click", generateNewRecordingId);
    if (el.btnRandomSpeaker) el.btnRandomSpeaker.addEventListener("click", generateRandomSpeakerId);
    if (el.scenarioTemplateSelect) el.scenarioTemplateSelect.addEventListener("change", handleScenarioTemplateSelected);

    // Live Validation Checklist Listeners
    if (el.consentCheckbox) {
      el.consentCheckbox.addEventListener("change", () => {
        hideSaveError();
        updateValidationChecklist();
      });
    }
    if (el.referenceTranscript) {
      ["input", "change", "paste"].forEach((evt) => {
        el.referenceTranscript.addEventListener(evt, () => {
          isRefUserEdited = true;
          updateValidationChecklist();
        });
      });
    }
    if (el.verifiedTranscript) {
      ["input", "change", "paste"].forEach((evt) => {
        el.verifiedTranscript.addEventListener(evt, () => {
          isVerUserEdited = true;
          updateValidationChecklist();
        });
      });
    }
    if (el.transcriptVerifiedCheckbox) {
      el.transcriptVerifiedCheckbox.addEventListener("change", () => {
        updateValidationChecklist();
      });
    }
    if (el.btnCopyRefToVer) {
      el.btnCopyRefToVer.addEventListener("click", () => {
        const refVal = (el.referenceTranscript && el.referenceTranscript.value.trim()) || "";
        if (el.verifiedTranscript) {
          el.verifiedTranscript.value = refVal;
          isVerUserEdited = true;
          updateValidationChecklist();
        }
      });
    }
    if (el.utteranceRoleSelect) {
      el.utteranceRoleSelect.addEventListener("change", () => {
        const role = el.utteranceRoleSelect.value;
        if (role === "utterance_b") {
          setWorkflowStep(3);
        } else {
          setWorkflowStep(2);
        }
        generateNewRecordingId();
        updateValidationChecklist();
      });
    }
    if (el.categorySelect) {
      el.categorySelect.addEventListener("change", () => {
        generateNextPairId();
      });
    }

    if (el.btnRefreshLibrary) {
      el.btnRefreshLibrary.addEventListener("click", () => {
        loadStudioRecordings();
        loadStudioPairs();
      });
    }

    if (el.btnModalConfirmVerify) el.btnModalConfirmVerify.addEventListener("click", submitPairVerification);
    if (el.btnModalCancelVerify) el.btnModalCancelVerify.addEventListener("click", closeVerifyModal);

    // Page lifecycle resource disposal
    if (typeof window !== "undefined") {
      const cleanupResources = () => {
        stopActiveAudioStream();
        closeActiveAudioContext();
      };
      window.addEventListener("beforeunload", cleanupResources);
      window.addEventListener("pagehide", cleanupResources);
    }

    // Initial state
    generateNewRecordingId();
    checkInitialPermission();
    loadStudioRecordings();
    loadStudioPairs();

    // Check if an in-progress pair should be resumed from sessionStorage
    tryResumeInProgressPair().then((resumed) => {
      if (!resumed) {
        generateNextPairId();
        setWorkflowStep(2); // Ready for Utterance A
      }
    });
  }

  // Expose global methods for inline HTML callbacks and automated regression tests
  window.BoliStudio = {
    init: initStudio,
    openVerifyModal: openVerifyModal,
    closeVerifyModal: closeVerifyModal,
    prepareRecordB: prepareRecordB,
    goToEvaluation: goToEvaluation,
    // Test inspection hooks
    handleScenarioTemplateSelected: handleScenarioTemplateSelected,
    saveRecordingToBackend: saveRecordingToBackend,
    handleSaveClick: handleSaveClick,
    requestMicrophonePermission: requestMicrophonePermission,
    enumerateAudioDevices: enumerateAudioDevices,
    stopActiveAudioStream: stopActiveAudioStream,
    closeActiveAudioContext: closeActiveAudioContext,
    tryResumeInProgressPair: tryResumeInProgressPair,
    persistPairSessionState: persistPairSessionState,
    clearPairSessionState: clearPairSessionState,
    createNextPair: createNextPair,
    proceedToUtteranceB: proceedToUtteranceB,
    getCurrentPair: () => currentPair,
    getIsSaveInProgress: () => isSaveInProgress,
    getIsRefUserEdited: () => isRefUserEdited,
    setIsRefUserEdited: (v) => { isRefUserEdited = Boolean(v); },
    getIsVerUserEdited: () => isVerUserEdited,
    setIsVerUserEdited: (v) => { isVerUserEdited = Boolean(v); },
    getRecordedBlob: () => recordedBlob,
    setRecordedBlob: (b) => { recordedBlob = b; },
    getAudioStream: () => audioStream,
    getAudioContext: () => audioContext,
    getValidationState: getValidationState,
    el: el,
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initStudio);
  } else {
    initStudio();
  }
})();
