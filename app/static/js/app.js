/**
 * BoliBench — Frontend Interaction & Telemetry Handler
 * Manages audio upload, live playback preview, API interaction, latency telemetry, and error handling.
 */

document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements
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

  // ---------------------------------------------------------------------------
  // Audio File Selection & Drag-and-Drop
  // ---------------------------------------------------------------------------

  function handleFileSelected(file) {
    if (!file) return;

    // Release old preview URL
    if (currentAudioUrl) {
      URL.revokeObjectURL(currentAudioUrl);
      currentAudioUrl = null;
    }

    // Format file size
    const sizeKB = (file.size / 1024).toFixed(1);
    const sizeMB = (file.size / (1024 * 1024)).toFixed(2);
    const sizeFormatted = file.size > 1024 * 1024 ? `${sizeMB} MB` : `${sizeKB} KB`;

    fileNameDisplay.textContent = file.name;
    fileSizeDisplay.textContent = sizeFormatted;

    // Attach to audio player
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

  // Drag & drop visual feedback
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

  // ---------------------------------------------------------------------------
  // Format / ITN Options Interaction
  // ---------------------------------------------------------------------------

  formatModeSelect.addEventListener("change", () => {
    if (formatModeSelect.value === "verbatim") {
      itnNativeNumerals.disabled = true;
      itnNativeNumerals.checked = false;
    } else {
      itnNativeNumerals.disabled = false;
    }
  });

  // Initialize state
  if (formatModeSelect.value === "verbatim") {
    itnNativeNumerals.disabled = true;
  }

  // ---------------------------------------------------------------------------
  // Form Submission & Latency Telemetry
  // ---------------------------------------------------------------------------

  transcribeForm.addEventListener("submit", async (e) => {
    e.preventDefault();

    const file = audioFileInput.files[0];
    if (!file) {
      showError("MISSING_FILE", "Please select an audio file to transcribe.");
      return;
    }

    // Set UI to loading state
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

      // Success Display
      displaySuccess(data, clientLatencyMs);
    } catch (err) {
      const clientLatencyMs = Math.round(performance.now() - clientStartTime);
      displayError(err, clientLatencyMs);
    } finally {
      setLoadingState(false);
    }
  });

  // ---------------------------------------------------------------------------
  // UI Display Helpers
  // ---------------------------------------------------------------------------

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

  function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
  }

  // ---------------------------------------------------------------------------
  // Copy to Clipboard
  // ---------------------------------------------------------------------------

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
});
