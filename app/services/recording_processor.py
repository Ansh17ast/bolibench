"""Server-side audio conversion and acoustic validation processor for BoliBench.

Guarantees:
- Zero external provider calls.
- Preserves raw uploaded audio without modification.
- Converts uploads to standardized 16 kHz, mono, 16-bit linear PCM WAV.
- Rigorously validates duration, decoding, sample rate, channels, silence, and clipping.
- Enforces duration validation (up to 600s maximum; short recordings >= 0.5s accepted).
"""

import io
import logging
from pathlib import Path
import shutil
import subprocess
from typing import Optional, Tuple
import wave

import numpy as np

from app.recording_schemas import AudioDiagnostics, ValidationStatus

logger = logging.getLogger("bolibench.recording_processor")


class AudioProcessingError(Exception):
    """Exception raised when audio conversion or decoding fails."""
    pass


class HumanAudioProcessor:
    """Handles audio format normalization to 16kHz mono PCM WAV and acoustic validation."""

    TARGET_SAMPLE_RATE = 16000
    TARGET_CHANNELS = 1
    TARGET_BITS_PER_SAMPLE = 16
    MAX_DURATION_SECONDS = 600.0        # Maximum 10 minutes (600s)
    MIN_DURATION_SECONDS = 0.5          # Minimum audible duration (0.5s)
    TARGET_DURATION_SECONDS = 600.0     # Backward-compatible alias
    COMPLETION_TOLERANCE_SECONDS = 0.5

    # Silence and clipping thresholds
    SILENCE_RMS_THRESHOLD = 0.002       # ~ -54 dBFS
    SILENCE_PEAK_THRESHOLD = 0.005      # ~ -46 dBFS
    CLIPPING_THRESHOLD_RATIO = 0.001    # > 0.1% of samples at digital limit
    CLIPPING_SAMPLE_COUNT_MIN = 10

    def __init__(self, ffmpeg_path: Optional[str] = None):
        # Locate ffmpeg binary (check PATH or explicit override)
        self.ffmpeg_path = ffmpeg_path or shutil.which("ffmpeg")
        if not self.ffmpeg_path:
            # Common Windows winget install fallback
            winget_path = Path(
                r"C:\Users\astru\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.2-full_build\bin\ffmpeg.EXE"
            )
            if winget_path.exists():
                self.ffmpeg_path = str(winget_path)

    def convert_to_standard_wav(self, input_path: Path, output_wav_path: Path) -> str:
        """Converts any audio file to standard 16 kHz, mono, 16-bit PCM WAV.

        Returns:
            The name of the conversion engine used ('ffmpeg' or 'wave_fallback').
        """
        output_wav_path.parent.mkdir(parents=True, exist_ok=True)

        if self.ffmpeg_path:
            cmd = [
                self.ffmpeg_path,
                "-y",  # overwrite output if exists
                "-i", str(input_path),
                "-vn",  # no video streams
                "-ar", str(self.TARGET_SAMPLE_RATE),
                "-ac", str(self.TARGET_CHANNELS),
                "-c:a", "pcm_s16le",
                "-f", "wav",
                str(output_wav_path),
            ]
            try:
                proc = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=30,
                    check=False,
                )
                if proc.returncode != 0:
                    err_msg = proc.stderr.decode("utf-8", errors="replace")
                    logger.warning("FFmpeg conversion failed: %s", err_msg)
                    # Attempt fallback if input happens to be WAV
                    if self._is_wav_file(input_path):
                        return self._convert_wav_fallback(input_path, output_wav_path)
                    raise AudioProcessingError(f"FFmpeg conversion failed with code {proc.returncode}: {err_msg[:300]}")

                if not output_wav_path.exists() or output_wav_path.stat().st_size < 44:
                    raise AudioProcessingError("FFmpeg produced empty or incomplete WAV file.")

                return "ffmpeg"
            except subprocess.TimeoutExpired:
                raise AudioProcessingError("Audio conversion timed out after 30 seconds.")
            except Exception as e:
                if isinstance(e, AudioProcessingError):
                    raise
                logger.warning("FFmpeg invocation exception: %s", str(e))
                if self._is_wav_file(input_path):
                    return self._convert_wav_fallback(input_path, output_wav_path)
                raise AudioProcessingError(f"Audio conversion failed: {str(e)}")

        # Fallback if ffmpeg is missing and input is WAV
        if self._is_wav_file(input_path):
            return self._convert_wav_fallback(input_path, output_wav_path)

        raise AudioProcessingError(
            "Server conversion requires FFmpeg for non-WAV containers (e.g. WebM/OGG). FFmpeg was not found in PATH."
        )

    def analyze_standard_wav(self, wav_path: Path, client_duration: Optional[float] = None) -> Tuple[AudioDiagnostics, ValidationStatus]:
        """Performs acoustic validation on the standardized 16kHz mono PCM WAV copy.

        Validates:
        - Decoding integrity
        - Sample rate == 16000 Hz
        - Channels == 1 (mono)
        - Bit depth == 16-bit
        - Duration check (up to 600.0s maximum; short recordings >= 0.5s accepted)
        - Silence level (RMS and peak)
        - Clipping / saturation detection
        """
        if not wav_path.exists():
            raise AudioProcessingError(f"Standardized WAV file not found at {wav_path}")

        try:
            with wave.open(str(wav_path), "rb") as wf:
                sample_rate = wf.getframerate()
                channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                nframes = wf.getnframes()
                raw_frames = wf.readframes(nframes)
        except Exception as exc:
            raise AudioProcessingError(f"Failed to decode standardized WAV: {str(exc)}")

        if sample_rate != self.TARGET_SAMPLE_RATE:
            raise AudioProcessingError(f"Invalid sample rate {sample_rate} Hz. Expected {self.TARGET_SAMPLE_RATE} Hz.")
        if channels != self.TARGET_CHANNELS:
            raise AudioProcessingError(f"Invalid channel count {channels}. Expected {self.TARGET_CHANNELS} (mono).")
        if sampwidth != (self.TARGET_BITS_PER_SAMPLE // 8):
            raise AudioProcessingError(f"Invalid sample width {sampwidth} bytes. Expected 2 bytes (16-bit).")

        duration_seconds = round(nframes / float(sample_rate), 3) if sample_rate > 0 else 0.0

        # Duration completeness check (0.5s to 600.0s is valid)
        is_complete_50s = (self.MIN_DURATION_SECONDS <= duration_seconds <= self.MAX_DURATION_SECONDS)

        # Integrity Check: Compare client-reported duration against actual decoded playable audio duration
        if client_duration is not None and client_duration >= self.MIN_DURATION_SECONDS:
            if duration_seconds < self.MIN_DURATION_SECONDS:
                is_complete_50s = False
                logger.warning(
                    "Duration discrepancy detected: client reported %.2fs, but decoded WAV is only %.2fs",
                    client_duration,
                    duration_seconds,
                )

        # Acoustic Signal Analysis using NumPy
        samples = np.frombuffer(raw_frames, dtype=np.int16)
        total_samples = len(samples)

        if total_samples > 0:
            samples_float = samples.astype(np.float32) / 32768.0
            rms_amplitude = float(np.sqrt(np.mean(samples_float ** 2)))
            peak_amplitude = float(np.max(np.abs(samples_float)))
            clipped_samples = int(np.sum(np.abs(samples) >= 32767))
            clipping_ratio = float(clipped_samples / total_samples)
        else:
            rms_amplitude = 0.0
            peak_amplitude = 0.0
            clipped_samples = 0
            clipping_ratio = 0.0

        is_silent = bool(rms_amplitude < self.SILENCE_RMS_THRESHOLD or peak_amplitude < self.SILENCE_PEAK_THRESHOLD)
        is_clipped = bool(
            clipped_samples >= 100
            or (clipped_samples >= self.CLIPPING_SAMPLE_COUNT_MIN and clipping_ratio >= self.CLIPPING_THRESHOLD_RATIO)
        )


        # Determine overall validation classification
        if duration_seconds < self.MIN_DURATION_SECONDS:
            status = ValidationStatus.INCOMPLETE_EARLY_STOP
        elif duration_seconds > self.MAX_DURATION_SECONDS:
            status = ValidationStatus.DURATION_EXCEEDED
        elif is_silent:
            status = ValidationStatus.SILENT_WARNING
        elif is_clipped:
            status = ValidationStatus.CLIPPED_WARNING
        else:
            status = ValidationStatus.VALID_50S

        diagnostics = AudioDiagnostics(
            sample_rate_hz=sample_rate,
            channels=channels,
            duration_seconds=duration_seconds,
            is_complete_50s=is_complete_50s,
            rms_amplitude=round(rms_amplitude, 5),
            peak_amplitude=round(peak_amplitude, 5),
            is_silent=is_silent,
            clipped_samples_count=clipped_samples,
            clipping_ratio=round(clipping_ratio, 6),
            is_clipped=is_clipped,
            decoding_successful=True,
            converter_used="ffmpeg" if self.ffmpeg_path else "wave_fallback",
        )

        return diagnostics, status

    def _is_wav_file(self, path: Path) -> bool:
        """Returns True if the file header starts with RIFF/WAVE."""
        try:
            with open(path, "rb") as f:
                header = f.read(12)
                return header.startswith(b"RIFF") and header[8:12] == b"WAVE"
        except Exception:
            return False

    def _convert_wav_fallback(self, input_path: Path, output_wav_path: Path) -> str:
        """Fallback converter for WAV inputs using pure Python wave and numpy."""
        with wave.open(str(input_path), "rb") as src:
            src_rate = src.getframerate()
            src_channels = src.getnchannels()
            src_width = src.getsampwidth()
            nframes = src.getnframes()
            raw_data = src.readframes(nframes)

        # Parse source samples
        if src_width == 2:
            data = np.frombuffer(raw_data, dtype=np.int16)
        elif src_width == 1:
            # 8-bit unsigned -> 16-bit signed
            u8 = np.frombuffer(raw_data, dtype=np.uint8)
            data = ((u8.astype(np.int32) - 128) * 256).astype(np.int16)
        else:
            raise AudioProcessingError(f"Unsupported sample width for fallback conversion: {src_width} bytes")

        # Downmix to mono if stereo
        if src_channels > 1:
            data = data.reshape(-1, src_channels).mean(axis=1).astype(np.int16)

        # Resample to 16,000 Hz if necessary
        if src_rate != self.TARGET_SAMPLE_RATE:
            num_output_samples = int(len(data) * (self.TARGET_SAMPLE_RATE / float(src_rate)))
            indices = np.linspace(0, len(data) - 1, num_output_samples)
            data = np.interp(indices, np.arange(len(data)), data).astype(np.int16)

        # Write standardized WAV
        with wave.open(str(output_wav_path), "wb") as dst:
            dst.setnchannels(self.TARGET_CHANNELS)
            dst.setsampwidth(self.TARGET_BITS_PER_SAMPLE // 8)
            dst.setframerate(self.TARGET_SAMPLE_RATE)
            dst.writeframes(data.tobytes())

        return "wave_fallback"
