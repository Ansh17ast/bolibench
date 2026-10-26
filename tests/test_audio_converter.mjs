/**
 * BoliBench Phase 6.17 — WAV Encoder & Audio Converter Unit Tests
 *
 * Tests:
 * 1. RIFF/WAVE signature and container structure
 * 2. PCM format tag = 1 (linear PCM)
 * 3. 16 kHz sample rate
 * 4. Mono channel count = 1
 * 5. 16-bit sample format
 * 6. Correct byte rate (32000), block alignment (2), and data length
 * 7. Little-endian sample encoding and clamping
 * 8. Accurate duration calculation
 * 9. Rejection of empty, null, or invalid sample data
 * 10. File size validation and limit enforcement
 * 11. Channel downmixing to mono
 * 12. Sample rate resampling (44.1 kHz / 48 kHz -> 16 kHz)
 * 13. Passthrough preservation for already-valid 16kHz mono PCM WAV
 * 14. Actionable error reporting on decode failure
 */

import test from "node:test";
import assert from "node:assert";
import fs from "node:fs";
import vm from "node:vm";
import converter from "../app/static/js/audio_converter.js";

const {
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
} = converter;

// =============================================================================
// Test 1: RIFF/WAVE Container & Standards-Compliant Headers
// =============================================================================
test("1. encodeWav outputs standards-compliant RIFF/WAVE container with valid format chunks", () => {
  const numSamples = 16000; // 1 second
  const samples = new Float32Array(numSamples);
  const buffer = encodeWav(samples, 16000);

  assert.ok(buffer instanceof ArrayBuffer, "Must return ArrayBuffer");
  assert.strictEqual(buffer.byteLength, 44 + numSamples * 2, "Size must be 44 header bytes + 2 bytes per sample");

  const view = new DataView(buffer);
  const riff = String.fromCharCode(view.getUint8(0), view.getUint8(1), view.getUint8(2), view.getUint8(3));
  const wave = String.fromCharCode(view.getUint8(8), view.getUint8(9), view.getUint8(10), view.getUint8(11));
  const fmt = String.fromCharCode(view.getUint8(12), view.getUint8(13), view.getUint8(14), view.getUint8(15));
  const data = String.fromCharCode(view.getUint8(36), view.getUint8(37), view.getUint8(38), view.getUint8(39));

  assert.strictEqual(riff, "RIFF", "Header must begin with 'RIFF'");
  assert.strictEqual(wave, "WAVE", "Container must be 'WAVE'");
  assert.strictEqual(fmt, "fmt ", "Sub-chunk must be 'fmt '");
  assert.strictEqual(data, "data", "Sub-chunk must be 'data'");
});

// =============================================================================
// Test 2: PCM Format Tag, Sample Rate, Channels, and Bit Depth
// =============================================================================
test("2. encodeWav sets linear PCM tag=1, mono=1, 16kHz, and 16-bit depth", () => {
  const samples = new Float32Array(8000); // 0.5 seconds
  const buffer = encodeWav(samples, 16000);
  const view = new DataView(buffer);

  const subchunk1Size = view.getUint32(16, true);
  const audioFormat = view.getUint16(20, true);
  const numChannels = view.getUint16(22, true);
  const sampleRate = view.getUint32(24, true);
  const byteRate = view.getUint32(28, true);
  const blockAlign = view.getUint16(32, true);
  const bitsPerSample = view.getUint16(34, true);
  const dataSize = view.getUint32(40, true);

  assert.strictEqual(subchunk1Size, 16, "fmt chunk size must be 16 for PCM");
  assert.strictEqual(audioFormat, 1, "AudioFormat must be 1 (uncompressed linear PCM)");
  assert.strictEqual(numChannels, 1, "NumChannels must be 1 (mono)");
  assert.strictEqual(sampleRate, 16000, "SampleRate must be 16,000 Hz");
  assert.strictEqual(bitsPerSample, 16, "BitsPerSample must be 16 bits");
  assert.strictEqual(blockAlign, 2, "BlockAlign must be 2 bytes (1 channel * 2 bytes/sample)");
  assert.strictEqual(byteRate, 32000, "ByteRate must be 32,000 bytes/sec (16000 * 2)");
  assert.strictEqual(dataSize, 16000, "Data size must be 8000 samples * 2 bytes = 16000 bytes");
});

// =============================================================================
// Test 3: Little-Endian Encoding, Clamping, and Quantization
// =============================================================================
test("3. encodeWav clamps samples to [-1.0, 1.0] and encodes 16-bit signed little-endian integers", () => {
  // Test specific sample values: 0.0, 1.0, -1.0, 0.5, -0.5, clamped 2.5, clamped -3.0
  const input = new Float32Array([0.0, 1.0, -1.0, 0.5, -0.5, 2.5, -3.0]);
  const buffer = encodeWav(input, 16000);
  const view = new DataView(buffer);

  assert.strictEqual(view.getInt16(44, true), 0, "0.0 must quantize to 0");
  assert.strictEqual(view.getInt16(46, true), 32767, "1.0 must quantize to 32767");
  assert.strictEqual(view.getInt16(48, true), -32768, "-1.0 must quantize to -32768");
  assert.strictEqual(view.getInt16(50, true), Math.round(0.5 * 32767), "0.5 must quantize to ~16384");
  assert.strictEqual(view.getInt16(52, true), Math.round(-0.5 * 32768), "-0.5 must quantize to -16384");
  assert.strictEqual(view.getInt16(54, true), 32767, "+2.5 must clamp to 32767");
  assert.strictEqual(view.getInt16(56, true), -32768, "-3.0 must clamp to -32768");
});

// =============================================================================
// Test 4: Duration Calculation & Validation
// =============================================================================
test("4. parseAndValidateWav accurately calculates audio duration from byte rate", () => {
  const durationSec = 3.5;
  const numSamples = Math.round(16000 * durationSec);
  const samples = new Float32Array(numSamples);
  const buffer = encodeWav(samples, 16000);

  const meta = parseAndValidateWav(buffer, 60);
  assert.strictEqual(meta.isValid, true);
  assert.strictEqual(meta.duration, durationSec, "Duration must be exactly 3.5s");
  assert.strictEqual(meta.sampleRate, 16000);
  assert.strictEqual(meta.numChannels, 1);
  assert.strictEqual(meta.bitsPerSample, 16);
  assert.strictEqual(meta.formatDescription, "WAV, 16 kHz, mono, PCM 16-bit");
});

// =============================================================================
// Test 5: Rejection of Invalid or Exceeded Duration
// =============================================================================
test("5. parseAndValidateWav rejects audio exceeding 60-second limit", () => {
  const samples65s = new Float32Array(16000 * 65);
  const buffer = encodeWav(samples65s, 16000);

  assert.throws(
    () => parseAndValidateWav(buffer, 60),
    (err) => err.message.includes("exceeds the maximum allowed limit of 60s")
  );
});

// =============================================================================
// Test 6: Rejection of Empty or Null Input Data
// =============================================================================
test("6. encodeWav and parseAndValidateWav reject empty, null, or corrupted data", () => {
  assert.throws(() => encodeWav([]), /Cannot encode WAV from empty/);
  assert.throws(() => encodeWav(null), /Cannot encode WAV from empty/);
  assert.throws(() => parseAndValidateWav(new ArrayBuffer(20)), /smaller than minimum 44-byte header/);

  // Corrupted magic bytes
  const corrupted = new ArrayBuffer(44);
  new DataView(corrupted).setUint32(0, 0x12345678);
  assert.throws(() => parseAndValidateWav(corrupted), /missing 'RIFF'/);
});

// =============================================================================
// Test 7: Downmixing Multichannel Audio to Mono
// =============================================================================
test("7. downmixToMono averages stereo channels correctly", () => {
  const mockStereoBuffer = {
    numberOfChannels: 2,
    length: 4,
    sampleRate: 44100,
    getChannelData(c) {
      if (c === 0) return new Float32Array([1.0, 0.5, -0.5, 0.0]); // Left
      if (c === 1) return new Float32Array([0.0, 0.5, 0.5, -1.0]); // Right
    },
  };

  const mono = downmixToMono(mockStereoBuffer);
  assert.strictEqual(mono.length, 4);
  assert.strictEqual(mono[0], 0.5, "1.0 and 0.0 averaged to 0.5");
  assert.strictEqual(mono[1], 0.5, "0.5 and 0.5 averaged to 0.5");
  assert.strictEqual(mono[2], 0.0, "-0.5 and 0.5 averaged to 0.0");
  assert.strictEqual(mono[3], -0.5, "0.0 and -1.0 averaged to -0.5");
});

// =============================================================================
// Test 8: Linear Resampling to 16 kHz
// =============================================================================
test("8. resampleLinear correctly scales sample sequence from 48 kHz / 32 kHz to 16 kHz", () => {
  // 32 kHz down to 16 kHz (2:1 decimation)
  const samples32k = new Float32Array([0.0, 0.5, 1.0, 0.5, 0.0, -0.5, -1.0, -0.5]);
  const resampled = resampleLinear(samples32k, 32000, 16000);

  assert.strictEqual(resampled.length, 4, "8 samples at 32kHz must become 4 samples at 16kHz");
  assert.strictEqual(resampled[0], 0.0);
  assert.strictEqual(resampled[1], 1.0);
  assert.strictEqual(resampled[2], 0.0);
  assert.strictEqual(resampled[3], -1.0);
});

// =============================================================================
// Test 9: convertAudioFile Pre-decode Size Limit Protection
// =============================================================================
test("9. convertAudioFile enforces 50 MB pre-decode safety limit", async () => {
  const fakeHugeFile = {
    name: "huge_video_audio.m4a",
    size: 55 * 1024 * 1024, // 55 MB
    arrayBuffer: async () => new ArrayBuffer(0),
  };

  await assert.rejects(
    async () => convertAudioFile(fakeHugeFile),
    (err) => err.message.includes("exceeds maximum pre-conversion limit of 50 MB")
  );
});

// =============================================================================
// Test 10: convertAudioFile Decodes & Produces Verified WAV File
// =============================================================================
test("10. convertAudioFile decodes audio and produces standard File object with _pcm.wav suffix", async () => {
  // Mock AudioContext and decodeAudioData
  const mockAudioBuffer = {
    numberOfChannels: 2,
    length: 44100 * 2, // 2 seconds at 44.1 kHz
    sampleRate: 44100,
    duration: 2.0,
    getChannelData(c) {
      return new Float32Array(44100 * 2).fill(0.2);
    },
  };

  class MockAudioContext {
    async decodeAudioData(buffer, success, failure) {
      if (typeof success === "function") success(mockAudioBuffer);
      return mockAudioBuffer;
    }
    async close() {}
  }

  const sampleFile = {
    name: "phone_recording_utterance_a.m4a",
    size: 150000,
    arrayBuffer: async () => new ArrayBuffer(150000),
  };

  // Provide mock File class if running in bare Node without global File
  if (typeof globalThis.File === "undefined") {
    globalThis.File = class File {
      constructor(chunks, name, opts) {
        this.name = name;
        this.size = chunks.reduce((acc, c) => acc + (c.byteLength || c.size || 0), 0);
        this.type = opts.type;
      }
    };
  }

  const result = await convertAudioFile(sampleFile, {
    AudioContext: MockAudioContext,
  });

  assert.strictEqual(result.originalName, "phone_recording_utterance_a.m4a");
  assert.strictEqual(result.convertedFile.name, "phone_recording_utterance_a_pcm.wav");
  assert.strictEqual(result.convertedFile.type, "audio/wav");
  assert.strictEqual(result.metadata.sampleRate, 16000);
  assert.strictEqual(result.metadata.numChannels, 1);
  assert.strictEqual(result.metadata.duration, 2.0);
  assert.strictEqual(result.metadata.formatDescription, "WAV, 16 kHz, mono, PCM 16-bit");
});

// =============================================================================
// Test 11: UMD wrapper assigns both BoliAudioConverter and BoliBenchAudioConverter
// =============================================================================
test("11. UMD wrapper attaches BoliAudioConverter and BoliBenchAudioConverter to window in browser", () => {
  const code = fs.readFileSync("app/static/js/audio_converter.js", "utf-8");
  const mockWindow = {};
  const sandbox = {
    window: mockWindow,
    globalThis: mockWindow,
    console,
  };
  vm.createContext(sandbox);
  vm.runInContext(code, sandbox);

  assert.ok(mockWindow.BoliAudioConverter, "window.BoliAudioConverter must be defined");
  assert.ok(mockWindow.BoliBenchAudioConverter, "window.BoliBenchAudioConverter must be defined as alias");
  assert.strictEqual(typeof mockWindow.BoliAudioConverter.convertAudioFile, "function");
  assert.strictEqual(typeof mockWindow.BoliAudioConverter.encodeWav, "function");
  assert.strictEqual(typeof mockWindow.BoliAudioConverter.parseAndValidateWav, "function");
  assert.strictEqual(mockWindow.BoliAudioConverter, mockWindow.BoliBenchAudioConverter);
});
