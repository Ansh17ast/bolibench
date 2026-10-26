/**
 * BoliBench — Client-Side Audio Conversion & WAV Encoding Service
 * Converts user-uploaded audio (M4A/AAC, MP3, OGG, WAV, FLAC, WebM) locally in-browser
 * to standard 16 kHz mono 16-bit PCM WAV before live evaluation upload.
 *
 * Guarantees:
 * - 0 external network requests or Gnani credits consumed during conversion.
 * - Standards-compliant RIFF/WAVE container generation (linear PCM format 1).
 * - Enforces client-side 50MB pre-decode limit, 25MB post-encode limit, and 60s duration limit.
 * - Channel downmixing and resampling to 16,000 Hz.
 */

(function (root, factory) {
  if (typeof module === "object" && module.exports) {
    module.exports = factory();
  } else {
    var api = factory();
    root.BoliAudioConverter = api;
    root.BoliBenchAudioConverter = api;
  }
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  const TARGET_SAMPLE_RATE = 16000;
  const TARGET_CHANNELS = 1;
  const BITS_PER_SAMPLE = 16;
  const MAX_INPUT_SIZE_BYTES = 52428800; // 50 MB safety guard before decode
  const MAX_UPLOAD_SIZE_BYTES = 26214400; // 25 MB max WAV size
  const MAX_DURATION_SECONDS = 60.0;

  /**
   * Encodes an array of floating-point audio samples [-1.0, 1.0] into a standard
   * 16-bit signed PCM, little-endian mono WAV ArrayBuffer.
   *
   * @param {Float32Array|Array<number>} samples Audio samples
   * @param {number} sampleRate Audio sample rate in Hz (default 16000)
   * @returns {ArrayBuffer} Formatted WAV byte buffer
   */
  function encodeWav(samples, sampleRate = TARGET_SAMPLE_RATE) {
    if (!samples || samples.length === 0) {
      throw new Error("Cannot encode WAV from empty or null sample data.");
    }
    if (typeof sampleRate !== "number" || sampleRate <= 0) {
      throw new Error(`Invalid sample rate: ${sampleRate}`);
    }

    const numChannels = TARGET_CHANNELS;
    const bitsPerSample = BITS_PER_SAMPLE;
    const bytesPerSample = bitsPerSample / 8;
    const blockAlign = numChannels * bytesPerSample;
    const byteRate = sampleRate * blockAlign;
    const dataSize = samples.length * bytesPerSample;
    const bufferSize = 44 + dataSize;

    const buffer = new ArrayBuffer(bufferSize);
    const view = new DataView(buffer);

    function writeAscii(offset, str) {
      for (let i = 0; i < str.length; i++) {
        view.setUint8(offset + i, str.charCodeAt(i));
      }
    }

    // 1. RIFF Header
    writeAscii(0, "RIFF");
    view.setUint32(4, 36 + dataSize, true); // ChunkSize = 36 + Subchunk2Size
    writeAscii(8, "WAVE");

    // 2. "fmt " Sub-chunk
    writeAscii(12, "fmt ");
    view.setUint32(16, 16, true);           // Subchunk1Size (16 for PCM)
    view.setUint16(20, 1, true);            // AudioFormat (1 = Linear PCM)
    view.setUint16(22, numChannels, true);  // NumChannels (1 = Mono)
    view.setUint32(24, sampleRate, true);   // SampleRate (16000)
    view.setUint32(28, byteRate, true);     // ByteRate (32000)
    view.setUint16(32, blockAlign, true);   // BlockAlign (2)
    view.setUint16(34, bitsPerSample, true);// BitsPerSample (16)

    // 3. "data" Sub-chunk
    writeAscii(36, "data");
    view.setUint32(40, dataSize, true);     // Subchunk2Size

    // 4. PCM Samples (16-bit signed, little-endian)
    let offset = 44;
    for (let i = 0; i < samples.length; i++) {
      let s = samples[i];
      // Clamping to [-1.0, 1.0]
      if (s > 1.0) s = 1.0;
      else if (s < -1.0) s = -1.0;
      // Quantize to 16-bit signed integer
      const val = s < 0 ? Math.round(s * 32768) : Math.round(s * 32767);
      view.setInt16(offset, val, true);
      offset += 2;
    }

    return buffer;
  }

  /**
   * Parses and validates a RIFF/WAVE ArrayBuffer, verifying header integrity,
   * linear PCM format, sample rate, duration, and file size limits.
   *
   * @param {ArrayBuffer} arrayBuffer WAV byte buffer
   * @param {number} maxDurationSec Maximum allowed duration in seconds
   * @param {number} maxSizeBytes Maximum allowed file size in bytes
   * @returns {Object} Validated audio metadata
   */
  function parseAndValidateWav(
    arrayBuffer,
    maxDurationSec = MAX_DURATION_SECONDS,
    maxSizeBytes = MAX_UPLOAD_SIZE_BYTES
  ) {
    if (!arrayBuffer || arrayBuffer.byteLength < 44) {
      throw new Error("Invalid WAV file: buffer is smaller than minimum 44-byte header.");
    }
    if (arrayBuffer.byteLength > maxSizeBytes) {
      const actualMb = (arrayBuffer.byteLength / (1024 * 1024)).toFixed(2);
      const maxMb = (maxSizeBytes / (1024 * 1024)).toFixed(2);
      throw new Error(`Converted file size (${actualMb} MB) exceeds maximum upload limit of ${maxMb} MB.`);
    }

    const view = new DataView(arrayBuffer);

    // Verify RIFF identifier
    const riff = String.fromCharCode(view.getUint8(0), view.getUint8(1), view.getUint8(2), view.getUint8(3));
    if (riff !== "RIFF") {
      throw new Error("Invalid WAV file header: missing 'RIFF' container signature.");
    }

    // Verify WAVE format
    const wave = String.fromCharCode(view.getUint8(8), view.getUint8(9), view.getUint8(10), view.getUint8(11));
    if (wave !== "WAVE") {
      throw new Error("Invalid WAV file header: missing 'WAVE' format identifier.");
    }

    let offset = 12;
    let fmtFound = false;
    let audioFormat = 0;
    let numChannels = 0;
    let sampleRate = 0;
    let byteRate = 0;
    let blockAlign = 0;
    let bitsPerSample = 0;
    let dataSize = 0;
    let dataFound = false;

    while (offset + 8 <= arrayBuffer.byteLength) {
      const chunkId = String.fromCharCode(
        view.getUint8(offset),
        view.getUint8(offset + 1),
        view.getUint8(offset + 2),
        view.getUint8(offset + 3)
      );
      const chunkSize = view.getUint32(offset + 4, true);

      if (chunkId === "fmt ") {
        fmtFound = true;
        audioFormat = view.getUint16(offset + 8, true);
        numChannels = view.getUint16(offset + 10, true);
        sampleRate = view.getUint32(offset + 12, true);
        byteRate = view.getUint32(offset + 16, true);
        blockAlign = view.getUint16(offset + 20, true);
        bitsPerSample = view.getUint16(offset + 22, true);
      } else if (chunkId === "data") {
        dataFound = true;
        dataSize = chunkSize;
      }

      offset += 8 + chunkSize;
      if (chunkSize % 2 !== 0) offset += 1; // RIFF chunk 2-byte word padding
    }

    if (!fmtFound) throw new Error("Invalid WAV file: missing 'fmt ' header subchunk.");
    if (!dataFound) throw new Error("Invalid WAV file: missing 'data' audio subchunk.");
    if (audioFormat !== 1) {
      throw new Error(`Unsupported WAV compression format (${audioFormat}). Audio must be linear PCM (1).`);
    }
    if (byteRate <= 0) {
      throw new Error("Invalid WAV header: byteRate must be greater than 0.");
    }

    const duration = dataSize / byteRate;
    if (duration > maxDurationSec) {
      throw new Error(
        `Audio duration (${duration.toFixed(1)}s) exceeds the maximum allowed limit of ${maxDurationSec.toFixed(0)}s.`
      );
    }

    return {
      isValid: true,
      audioFormat,
      numChannels,
      sampleRate,
      byteRate,
      blockAlign,
      bitsPerSample,
      dataSize,
      duration,
      sizeBytes: arrayBuffer.byteLength,
      fileSize: arrayBuffer.byteLength,
      formatDescription: `WAV, ${(sampleRate / 1000).toFixed(0)} kHz, ${numChannels === 1 ? "mono" : "stereo"}, PCM ${bitsPerSample}-bit`,
    };
  }

  /**
   * Pure algorithmic channel downmixing from multi-channel AudioBuffer or arrays to mono Float32Array.
   */
  function downmixToMono(audioBuffer) {
    const numChannels = audioBuffer.numberOfChannels;
    const length = audioBuffer.length;
    if (numChannels === 1) {
      return audioBuffer.getChannelData(0);
    }

    const mono = new Float32Array(length);
    const channels = [];
    for (let c = 0; c < numChannels; c++) {
      channels.push(audioBuffer.getChannelData(c));
    }

    for (let i = 0; i < length; i++) {
      let sum = 0;
      for (let c = 0; c < numChannels; c++) {
        sum += channels[c][i];
      }
      mono[i] = sum / numChannels;
    }
    return mono;
  }

  /**
   * Pure linear interpolation resampling algorithm from any source sample rate to a target rate.
   */
  function resampleLinear(samples, fromRate, toRate) {
    if (fromRate === toRate || samples.length === 0) {
      return samples;
    }
    const ratio = fromRate / toRate;
    const newLength = Math.max(1, Math.round(samples.length / ratio));
    const result = new Float32Array(newLength);

    for (let i = 0; i < newLength; i++) {
      const srcIdx = i * ratio;
      const index = Math.floor(srcIdx);
      const frac = srcIdx - index;

      if (index + 1 < samples.length) {
        result[i] = samples[index] * (1 - frac) + samples[index + 1] * frac;
      } else if (index < samples.length) {
        result[i] = samples[index];
      } else {
        result[i] = 0;
      }
    }
    return result;
  }

  /**
   * Resamples and downmixes an AudioBuffer to target sample rate mono Float32Array.
   * Uses OfflineAudioContext in supported browsers for hardware-accelerated polyphase sinc resampling,
   * falling back to pure algorithmic downmixing and linear resampling.
   */
  async function resampleAndDownmix(audioBuffer, targetSampleRate = TARGET_SAMPLE_RATE) {
    const hasOfflineContext =
      typeof window !== "undefined" &&
      (typeof window.OfflineAudioContext === "function" ||
        typeof window.webkitOfflineAudioContext === "function");

    if (hasOfflineContext) {
      try {
        const OfflineContextClass = window.OfflineAudioContext || window.webkitOfflineAudioContext;
        const targetLength = Math.max(1, Math.ceil(audioBuffer.duration * targetSampleRate));
        const offlineCtx = new OfflineContextClass(TARGET_CHANNELS, targetLength, targetSampleRate);

        const source = offlineCtx.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(offlineCtx.destination);
        source.start(0);

        const rendered = await offlineCtx.startRendering();
        return rendered.getChannelData(0);
      } catch (err) {
        console.warn("OfflineAudioContext resampling failed, using algorithmic fallback:", err);
      }
    }

    // Pure Algorithmic Fallback
    const mono = downmixToMono(audioBuffer);
    return resampleLinear(mono, audioBuffer.sampleRate, targetSampleRate);
  }

  /**
   * Converts an AudioBuffer to 16 kHz mono 16-bit PCM WAV ArrayBuffer.
   */
  async function convertAudioBufferToWav(audioBuffer, targetSampleRate = TARGET_SAMPLE_RATE) {
    if (!audioBuffer) {
      throw new Error("Cannot convert null or undefined AudioBuffer.");
    }
    const samples = await resampleAndDownmix(audioBuffer, targetSampleRate);
    const wavBuffer = encodeWav(samples, targetSampleRate);
    const metadata = parseAndValidateWav(wavBuffer, MAX_DURATION_SECONDS, MAX_UPLOAD_SIZE_BYTES);
    return { wavBuffer, metadata };
  }

  /**
   * Converts a user-selected audio File/Blob to a verified 16 kHz mono PCM WAV File.
   *
   * @param {File|Blob} file Selected input audio file
   * @param {Object} options Configuration overrides
   * @returns {Promise<Object>} Converted File, object preview URL, and metadata
   */
  async function convertAudioFile(file, options = {}) {
    if (!file) {
      throw new Error("No audio file provided for conversion.");
    }

    const maxInputSize = options.maxInputSizeBytes || MAX_INPUT_SIZE_BYTES;
    if (file.size === 0) {
      throw new Error("Selected audio file is empty (0 bytes).");
    }
    if (file.size > maxInputSize) {
      const mb = (file.size / (1024 * 1024)).toFixed(1);
      const limitMb = (maxInputSize / (1024 * 1024)).toFixed(0);
      throw new Error(
        `Selected file size (${mb} MB) exceeds maximum pre-conversion limit of ${limitMb} MB to prevent browser memory exhaustion.`
      );
    }

    const rawArrayBuffer = await file.arrayBuffer();

    // Check if the file is ALREADY a valid 16 kHz mono 16-bit PCM WAV
    try {
      const existingMeta = parseAndValidateWav(rawArrayBuffer, MAX_DURATION_SECONDS, MAX_UPLOAD_SIZE_BYTES);
      if (
        existingMeta.isValid &&
        existingMeta.audioFormat === 1 &&
        existingMeta.numChannels === 1 &&
        existingMeta.sampleRate === TARGET_SAMPLE_RATE &&
        existingMeta.bitsPerSample === BITS_PER_SAMPLE
      ) {
        // Pristine 16kHz mono PCM WAV already: return with zero re-encoding loss
        const convertedName = file.name.toLowerCase().endsWith(".wav")
          ? file.name
          : `${file.name.replace(/\.[^/.]+$/, "")}_pcm.wav`;
        const readyFile = new File([rawArrayBuffer], convertedName, { type: "audio/wav" });
        const previewUrl = typeof URL !== "undefined" && typeof URL.createObjectURL === "function"
          ? URL.createObjectURL(readyFile)
          : null;

        return {
          convertedFile: readyFile,
          file: readyFile,
          previewUrl,
          metadata: existingMeta,
          duration: existingMeta.duration,
          sampleRate: existingMeta.sampleRate,
          channels: existingMeta.numChannels,
          bitDepth: existingMeta.bitsPerSample,
          fileSize: existingMeta.sizeBytes || existingMeta.fileSize || rawArrayBuffer.byteLength,
          sizeBytes: existingMeta.sizeBytes || existingMeta.fileSize || rawArrayBuffer.byteLength,
          originalName: file.name,
          originalSize: file.size,
          wasPassthrough: true,
        };
      }
    } catch (_) {
      // Not a compliant 16kHz mono PCM WAV: proceed to full Web Audio decode and resample
    }

    // Decode actual audio content via Web Audio API
    const AudioContextClass =
      options.AudioContext ||
      (typeof window !== "undefined" && (window.AudioContext || window.webkitAudioContext));

    if (!AudioContextClass) {
      throw new Error(
        "Web Audio API (AudioContext) is not available in this browser environment to decode audio."
      );
    }

    const audioCtx = new AudioContextClass();
    let audioBuffer = null;

    try {
      // decodeAudioData detaches the input ArrayBuffer in some browsers; use slice to be safe
      const bufferToDecode = rawArrayBuffer.slice(0);
      audioBuffer = await new Promise((resolve, reject) => {
        const promise = audioCtx.decodeAudioData(bufferToDecode, resolve, reject);
        if (promise && typeof promise.then === "function") {
          promise.then(resolve).catch(reject);
        }
      });
    } catch (decodeErr) {
      const fname = file.name || "selected file";
      throw new Error(
        `Conversion failed \u2014 this browser could not decode the selected audio format ('${fname}'). Try exporting the recording to a supported WAV, MP3, or OGG format.`
      );
    } finally {
      if (typeof audioCtx.close === "function") {
        audioCtx.close().catch(() => {});
      }
    }

    if (!audioBuffer) {
      throw new Error("Audio decoding returned an empty buffer.");
    }

    // Pre-check duration before spending time on resampling and encoding
    if (audioBuffer.duration > MAX_DURATION_SECONDS) {
      throw new Error(
        `Audio duration (${audioBuffer.duration.toFixed(1)}s) exceeds the maximum allowed limit of ${MAX_DURATION_SECONDS.toFixed(0)}s.`
      );
    }

    // Resample to 16 kHz, downmix to mono, and encode to 16-bit PCM WAV
    const { wavBuffer, metadata } = await convertAudioBufferToWav(audioBuffer, TARGET_SAMPLE_RATE);

    // Compute converted filename
    const baseName = (file.name || "utterance").replace(/\.[^/.]+$/, "");
    const convertedFileName = `${baseName}_pcm.wav`;

    const convertedFile = new File([wavBuffer], convertedFileName, { type: "audio/wav" });
    const previewUrl =
      typeof URL !== "undefined" && typeof URL.createObjectURL === "function"
        ? URL.createObjectURL(convertedFile)
        : null;

    return {
      convertedFile,
      file: convertedFile,
      previewUrl,
      metadata,
      duration: metadata.duration,
      sampleRate: metadata.sampleRate,
      channels: metadata.numChannels,
      bitDepth: metadata.bitsPerSample,
      fileSize: metadata.sizeBytes || metadata.fileSize || wavBuffer.byteLength,
      sizeBytes: metadata.sizeBytes || metadata.fileSize || wavBuffer.byteLength,
      originalName: file.name || "unnamed_audio",
      originalSize: file.size,
      wasPassthrough: false,
    };
  }

  return {
    encodeWav,
    parseAndValidateWav,
    downmixToMono,
    resampleLinear,
    convertAudioBufferToWav,
    convertAudioFile,
    TARGET_SAMPLE_RATE,
    TARGET_CHANNELS,
    BITS_PER_SAMPLE,
    MAX_DURATION_SECONDS,
    MAX_UPLOAD_SIZE_BYTES,
    MAX_INPUT_SIZE_BYTES,
  };
});
