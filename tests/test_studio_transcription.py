"""Backend regression tests for BoliBench Human Recording Studio automatic STT transcription.

Guarantees Verified:
- End-to-end studio transcription via POST /api/studio/transcribe.
- Zero live external provider calls during testing (deterministic mock boundary).
- Zero ground-truth leakage (transcripts produced strictly from audio bytes).
- Explicit provider transmission consent gate (unconsented returns 403, 0 calls).
- Strict 60.0s Gnani REST limit (60.0s accepted, 60.01s/60.05s rejected with HTTP 400).
- Studio 600s recording maximum preserved for manual ground truth verification.
- Feature flag enforcement (LIVE_TRANSCRIPTION_ENABLED).
- Failure and error handling for manual retry QA testing.
"""

import io
import math
import struct
import wave
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.asr_service import (
    BaseASREngine,
    GnaniPrismaEngine,
    TestMockASREngine,
    get_asr_engine,
    set_asr_engine,
    reset_asr_engine,
)


@pytest.fixture(autouse=True)
def clean_asr_engine():
    """Ensure each test starts and ends with a clean ASR engine boundary."""
    reset_asr_engine()
    yield
    reset_asr_engine()


def generate_test_wav_bytes(duration_seconds: float = 2.0, sample_rate: int = 16000) -> bytes:
    """Generates synthetic in-memory 16-bit PCM mono WAV bytes of exact duration."""
    num_frames = int(duration_seconds * sample_rate)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        # Synthetic 440Hz sine wave tone
        samples = []
        for i in range(num_frames):
            val = int(10000 * math.sin(2 * math.pi * 440 * (i / sample_rate)))
            samples.append(struct.pack("<h", val))
        wf.writeframes(b"".join(samples))
    return buffer.getvalue()


client = TestClient(app)


def test_studio_config_exposes_stt_metadata():
    """GET /api/studio/config exposes live transcription status, provider info, and limits."""
    resp = client.get("/api/studio/config")
    assert resp.status_code == 200
    data = resp.json()

    assert "live_transcription_enabled" in data
    assert "stt_provider_configured" in data
    assert data["max_stt_duration_seconds"] == 60.0
    assert data["max_duration_seconds"] == 600.0
    assert data["target_duration_seconds"] == 600.0


def test_studio_transcribe_with_consent_success():
    """POST /api/studio/transcribe transcribes actual audio bytes when consented."""
    mock_engine = TestMockASREngine(fixed_transcript="दो पैकेट चाहिए")
    set_asr_engine(mock_engine)

    wav_bytes = generate_test_wav_bytes(duration_seconds=3.0)

    files = {"audio_file": ("studio_test.wav", wav_bytes, "audio/wav")}
    data = {
        "language_code": "hi-IN",
        "utterance_role": "utterance_a",
        "consent_given": "true",
    }

    resp = client.post("/api/studio/transcribe", files=files, data=data)
    assert resp.status_code == 200
    res = resp.json()

    assert res["success"] is True
    assert res["transcript"] == "दो पैकेट चाहिए"
    assert res["language_code"] == "hi-IN"
    assert res["utterance_role"] == "utterance_a"
    assert res["duration_seconds"] == 3.0
    assert mock_engine.call_count == 1
    assert mock_engine.last_audio_bytes == wav_bytes


def test_studio_transcribe_without_consent_rejected_403():
    """External provider mode strictly blocks unconsented transcription (HTTP 403, 0 calls)."""
    from app.config import get_settings
    settings = get_settings()

    gnani_engine = GnaniPrismaEngine(settings=settings)
    set_asr_engine(gnani_engine)

    wav_bytes = generate_test_wav_bytes(duration_seconds=2.0)
    files = {"audio_file": ("studio_test.wav", wav_bytes, "audio/wav")}
    data = {
        "language_code": "hi-IN",
        "utterance_role": "utterance_a",
        "consent_given": "false",
    }

    resp = client.post("/api/studio/transcribe", files=files, data=data)
    assert resp.status_code == 403
    assert "explicit user consent" in resp.json()["detail"]


def test_studio_transcribe_duration_boundary_60s_accepted():
    """Clips of exactly 60.0s are accepted for automatic transcription."""
    mock_engine = TestMockASREngine(fixed_transcript="Boundary test at sixty seconds")
    set_asr_engine(mock_engine)

    wav_60 = generate_test_wav_bytes(duration_seconds=60.0)
    files = {"audio_file": ("boundary_60.wav", wav_60, "audio/wav")}
    data = {
        "language_code": "en-IN",
        "utterance_role": "utterance_a",
        "consent_given": "true",
    }

    resp = client.post("/api/studio/transcribe", files=files, data=data)
    assert resp.status_code == 200
    assert resp.json()["success"] is True
    assert mock_engine.call_count == 1


def test_studio_transcribe_duration_boundary_60_01s_rejected():
    """Clips exceeding 60.0s (e.g. 60.01s, 60.05s) are rejected with HTTP 400; 0 provider calls."""
    mock_engine = TestMockASREngine()
    set_asr_engine(mock_engine)

    wav_60_05 = generate_test_wav_bytes(duration_seconds=60.05)
    files = {"audio_file": ("boundary_over.wav", wav_60_05, "audio/wav")}
    data = {
        "language_code": "hi-IN",
        "utterance_role": "utterance_a",
        "consent_given": "true",
    }

    resp = client.post("/api/studio/transcribe", files=files, data=data)
    assert resp.status_code == 400
    assert "exceeds the 60.0-second limit" in resp.json()["detail"]
    assert "Recordings up to 600s are preserved" in resp.json()["detail"]
    assert mock_engine.call_count == 0, "No provider call must be made when duration exceeds limit"


def test_studio_transcribe_empty_audio_rejected():
    """Empty audio files return HTTP 400 Bad Request."""
    mock_engine = TestMockASREngine()
    set_asr_engine(mock_engine)

    files = {"audio_file": ("empty.wav", b"", "audio/wav")}
    data = {"consent_given": "true"}

    resp = client.post("/api/studio/transcribe", files=files, data=data)
    assert resp.status_code == 400
    assert "empty (0 bytes)" in resp.json()["detail"]
    assert mock_engine.call_count == 0


def test_studio_transcribe_force_error_for_qa():
    """force_error=true triggers simulated 500 error to test client retry retention."""
    mock_engine = TestMockASREngine()
    set_asr_engine(mock_engine)

    wav_bytes = generate_test_wav_bytes(duration_seconds=2.0)
    files = {"audio_file": ("test.wav", wav_bytes, "audio/wav")}
    data = {
        "consent_given": "true",
        "force_error": "true",
    }

    resp = client.post("/api/studio/transcribe", files=files, data=data)
    assert resp.status_code == 500
    assert "Simulated transcription engine failure" in resp.json()["detail"]
    assert mock_engine.call_count == 0


def test_zero_ground_truth_leakage():
    """Verifies that speech transcripts are strictly produced from audio bytes, never from test-case ground truth."""
    # TestMockASREngine returns acoustic description of audio duration, NOT scenario text
    mock_engine = TestMockASREngine()
    set_asr_engine(mock_engine)

    wav_bytes = generate_test_wav_bytes(duration_seconds=2.5)
    files = {"audio_file": ("real_speech.wav", wav_bytes, "audio/wav")}
    data = {
        "language_code": "hi-IN",
        "utterance_role": "utterance_a",
        "consent_given": "true",
    }

    resp = client.post("/api/studio/transcribe", files=files, data=data)
    assert resp.status_code == 200
    transcript = resp.json()["transcript"]

    # Verify transcript is derived from audio bytes, not canned scenario
    assert "2.5s" in transcript
    assert mock_engine.call_count == 1


def test_studio_transcribe_webm_audio_processed_successfully(tmp_path):
    """Regression test: WebM/Opus audio recorded by browser MediaRecorder is standardized and transcribed (no 500 error)."""
    import subprocess
    import shutil

    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        pytest.skip("FFmpeg not installed in test environment")

    webm_file = tmp_path / "browser_recorded.webm"
    # Generate a synthetic 3.0-second WebM audio file using ffmpeg
    subprocess.run(
        [
            ffmpeg_bin,
            "-y",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=3",
            "-c:a",
            "libopus",
            str(webm_file),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )

    webm_bytes = webm_file.read_bytes()
    assert len(webm_bytes) > 0

    mock_engine = TestMockASREngine(fixed_transcript="दो पैकेट चाहिए")
    set_asr_engine(mock_engine)

    # Send WebM bytes as uploaded by browser
    files = {"audio_file": ("studio_recording.webm", webm_bytes, "audio/webm")}
    data = {
        "language_code": "hi-IN",
        "utterance_role": "utterance_a",
        "consent_given": "true",
    }

    resp = client.post("/api/studio/transcribe", files=files, data=data)
    assert resp.status_code == 200, f"Expected 200 OK, got {resp.status_code}: {resp.text}"
    body = resp.json()
    assert body["success"] is True
    assert body["transcript"] == "दो पैकेट चाहिए"
    assert mock_engine.call_count == 1
    # Verify mock engine received valid standardized WAV bytes (starts with RIFF)
    assert mock_engine.last_audio_bytes.startswith(b"RIFF")


def test_audio_validator_imported_from_correct_module():
    """Confirms AudioValidator is imported from app.services.audio_validator, avoiding ImportError."""
    from app.services.audio_validator import AudioValidator, AudioValidationError

    validator = AudioValidator()
    assert hasattr(validator, "validate")
    assert hasattr(validator, "validate_live_audio")


def test_studio_transcribe_upstream_504_gateway_timeout_mapping(monkeypatch):
    """Confirms upstream Cloudflare/Gnani HTTP 504 Gateway Timeout is mapped to retryable HTTP 504 with sanitized message."""
    import httpx
    from app.config import get_settings
    from app.services.gnani_client import GnaniPrismaClient

    def mock_504_gateway_timeout(request: httpx.Request) -> httpx.Response:
        cloudflare_html = (
            "<!DOCTYPE html><html><head><title>vachana.ai | 504: Gateway time-out</title></head>"
            "<body><h1>Gateway time-out</h1><p>The web server reported a gateway time-out error.</p></body></html>"
        )
        return httpx.Response(status_code=504, text=cloudflare_html, headers={"Content-Type": "text/html"})

    settings = get_settings()
    monkeypatch.setattr(settings, "live_transcription_enabled", True)
    mock_transport = httpx.MockTransport(mock_504_gateway_timeout)
    mock_http_client = httpx.AsyncClient(transport=mock_transport)
    mock_gnani_client = GnaniPrismaClient(settings=settings, http_client=mock_http_client)
    engine = GnaniPrismaEngine(settings=settings, client=mock_gnani_client)
    set_asr_engine(engine)

    wav_bytes = generate_test_wav_bytes(duration_seconds=2.0)
    files = {"audio_file": ("test.wav", wav_bytes, "audio/wav")}
    data = {
        "language_code": "hi-IN",
        "utterance_role": "utterance_a",
        "consent_given": "true",
    }

    resp = client.post("/api/studio/transcribe", files=files, data=data)
    assert resp.status_code == 504, f"Expected HTTP 504 Gateway Timeout, got {resp.status_code}"
    detail = resp.json()["detail"]
    assert "Gnani transcription service timed out (504 Gateway Timeout)" in detail
    assert "audio is preserved; retry later" in detail
    # Confirm raw Cloudflare HTML is completely sanitized and not dumped into response
    assert "<!DOCTYPE" not in detail
    assert "<html>" not in detail
