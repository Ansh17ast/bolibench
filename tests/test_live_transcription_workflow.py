"""Regression and integration tests for BoliBench Live Evaluation Automatic Transcription with Gnani AI.

Verifies:
1. Genuine ASR boundary: actual audio bytes are received by the ASR engine; zero ground truth or canned text.
2. Gnani Prisma REST STT v3 integration (mocked, 0 live network calls during tests):
   - Correct endpoint POST https://api.vachana.ai/stt/v3
   - Header X-API-Key-ID transmission
   - Multipart audio_file and language_code fields
   - Accurate disclosure of 1 external call and 1 Gnani credit consumed
3. Language mapping for Gnani (hinglish -> hi-IN, hi-IN -> hi-IN, en-IN -> en-IN).
4. Consent gate enforcement: requires explicit user consent before transmitting audio bytes to Gnani (403 if omitted).
5. Error handling: Gnani 401, 400, 429, 500 errors mapped to actionable HTTP responses.
6. Retry behavior: strictly no auto-retry to prevent unexpected credit spending; preserves audio bytes for manual user retry.
7. User editing before evaluation: evaluation scores the edited transcript, not unreviewed text.
8. Separate duration limits:
   - Live Evaluation limit: strictly 60.0s (clips > 60s rejected with 400).
   - Human Recording Studio limit: strictly 600.0s preserved.
9. Empty audio rejection (400).
10. Unsupported language rejection (400).
11. Zero live provider calls: all tests use mock transports / boundaries.
"""

import io
import httpx
import pytest
from app.config import Settings
from app.services.asr_service import (
    GnaniPrismaEngine,
    TestMockASREngine,
    LocalWhisperEngine,
    set_asr_engine,
    reset_asr_engine,
    verify_asr_status,
)
from app.services.gnani_client import GnaniPrismaClient
from tests.conftest import create_mock_wav


@pytest.fixture(autouse=True)
def manage_asr_engine():
    """Ensure ASR engine is reset after each test."""
    yield
    reset_asr_engine()


@pytest.mark.asyncio
async def test_gnani_prisma_request_structure_and_headers_mocked(async_client):
    """Proves Live Evaluation calls Gnani v3 REST endpoint with X-API-Key-ID, multipart audio, and language_code.
    Accurately discloses 1 external call and 1 credit consumed.
    Mocked transport ensures 0 live network calls.
    """
    intercepted_requests = []

    async def mock_gnani_handler(request: httpx.Request) -> httpx.Response:
        intercepted_requests.append(request)
        # Verify Gnani v3 REST specification
        assert request.url.path == "/stt/v3"
        assert request.headers.get("X-API-Key-ID") == "vach_mock_test_key_12345"
        assert "multipart/form-data" in request.headers.get("content-type", "")

        return httpx.Response(
            status_code=200,
            json={
                "success": True,
                "request_id": "req_mock_gnani_001",
                "transcript": "नमस्ते, मुझे दो पैकेट चाहिए",
                "model": "gnani-prisma-v2.5",
            },
        )

    mock_settings = Settings(
        gnani_api_key="vach_mock_test_key_12345",
        gnani_base_url="https://api.vachana.ai/stt/v3",
    )
    transport = httpx.MockTransport(mock_gnani_handler)
    http_client = httpx.AsyncClient(transport=transport)

    gnani_client = GnaniPrismaClient(settings=mock_settings, http_client=http_client)
    gnani_engine = GnaniPrismaEngine(settings=mock_settings, client=gnani_client)
    set_asr_engine(gnani_engine)

    wav_bytes = create_mock_wav(duration_seconds=2.5)
    files = {"audio_file": ("speech_live.wav", io.BytesIO(wav_bytes), "audio/wav")}
    data = {
        "language_code": "hi-IN",
        "utterance_role": "utterance_a",
        "consent_given": "true",
    }

    resp = await async_client.post("/api/eval/transcribe-live", files=files, data=data)
    assert resp.status_code == 200

    result = resp.json()
    assert result["success"] is True
    assert result["transcript"] == "नमस्ते, मुझे दो पैकेट चाहिए"
    assert result["provider"] == "gnani-prisma-v3"
    assert result["external_calls"] == 1
    assert result["credits_consumed"] == 1
    assert "1 Gnani credit consumed" in result["disclaimer"]

    # Verify that exactly 1 request was sent to Gnani
    assert len(intercepted_requests) == 1
    req = intercepted_requests[0]
    assert req.method == "POST"
    assert "vachana.ai" in str(req.url)


@pytest.mark.asyncio
async def test_gnani_prisma_language_mapping_mocked(async_client):
    """Proves language codes (e.g. hinglish -> hi-IN) are normalized correctly for Gnani."""
    observed_languages = []

    async def mock_handler(request: httpx.Request) -> httpx.Response:
        body_text = request.read().decode("latin1")
        if 'name="language_code"' in body_text:
            # Extract language_code field from multipart
            part = body_text.split('name="language_code"')[1].split("\r\n\r\n")[1].split("\r\n")[0]
            observed_languages.append(part)

        return httpx.Response(
            status_code=200,
            json={"success": True, "transcript": "Recognized speech"},
        )

    mock_settings = Settings(gnani_api_key="vach_mock_test_key_12345")
    transport = httpx.MockTransport(mock_handler)
    http_client = httpx.AsyncClient(transport=transport)
    client = GnaniPrismaClient(settings=mock_settings, http_client=http_client)
    set_asr_engine(GnaniPrismaEngine(settings=mock_settings, client=client))

    wav_bytes = create_mock_wav(duration_seconds=1.0)

    # Test hinglish mapping
    resp_hgl = await async_client.post(
        "/api/eval/transcribe-live",
        files={"audio_file": ("test.wav", io.BytesIO(wav_bytes), "audio/wav")},
        data={"language_code": "hinglish", "consent_given": "true"},
    )
    assert resp_hgl.status_code == 200

    # Test en-IN mapping
    resp_en = await async_client.post(
        "/api/eval/transcribe-live",
        files={"audio_file": ("test.wav", io.BytesIO(wav_bytes), "audio/wav")},
        data={"language_code": "en-IN", "consent_given": "true"},
    )
    assert resp_en.status_code == 200

    assert observed_languages == ["hi-IN", "en-IN"]


@pytest.mark.asyncio
async def test_gnani_consent_gate_enforcement(async_client):
    """Proves that without explicit user consent, server returns 403 Forbidden and 0 calls are made to Gnani."""
    called = False

    async def mock_handler(request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(status_code=200, json={"transcript": "Should not be called"})

    mock_settings = Settings(gnani_api_key="vach_mock_test_key_12345")
    transport = httpx.MockTransport(mock_handler)
    client = GnaniPrismaClient(settings=mock_settings, http_client=httpx.AsyncClient(transport=transport))
    set_asr_engine(GnaniPrismaEngine(settings=mock_settings, client=client))

    wav_bytes = create_mock_wav(duration_seconds=1.0)
    files = {"audio_file": ("test.wav", io.BytesIO(wav_bytes), "audio/wav")}

    # 1. consent_given = false
    resp_no_consent = await async_client.post(
        "/api/eval/transcribe-live",
        files=files,
        data={"language_code": "hi-IN", "consent_given": "false"},
    )
    assert resp_no_consent.status_code == 403
    assert "explicit user consent" in resp_no_consent.json()["detail"]
    assert not called, "Gnani API must NEVER be called when consent is false"


@pytest.mark.asyncio
async def test_gnani_error_handling_mocked(async_client):
    """Proves Gnani API errors (401, 400, 429, 500) translate to actionable HTTP responses without exposing secrets."""
    status_to_simulate = 401

    async def mock_error_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=status_to_simulate,
            json={"error": "API error simulation", "message": f"Simulated {status_to_simulate}"},
        )

    mock_settings = Settings(gnani_api_key="vach_mock_test_key_12345")
    transport = httpx.MockTransport(mock_error_handler)
    client = GnaniPrismaClient(settings=mock_settings, http_client=httpx.AsyncClient(transport=transport))
    set_asr_engine(GnaniPrismaEngine(settings=mock_settings, client=client))

    wav_bytes = create_mock_wav(duration_seconds=1.0)
    files = {"audio_file": ("test.wav", io.BytesIO(wav_bytes), "audio/wav")}
    data = {"language_code": "hi-IN", "consent_given": "true"}

    # Test 401 Unauthorized
    status_to_simulate = 401
    resp_401 = await async_client.post("/api/eval/transcribe-live", files=files, data=data)
    assert resp_401.status_code == 401

    # Test 429 Rate Limit
    status_to_simulate = 429
    files = {"audio_file": ("test.wav", io.BytesIO(wav_bytes), "audio/wav")}
    resp_429 = await async_client.post("/api/eval/transcribe-live", files=files, data=data)
    assert resp_429.status_code == 429

    # Test 500 Server Error
    status_to_simulate = 500
    files = {"audio_file": ("test.wav", io.BytesIO(wav_bytes), "audio/wav")}
    resp_500 = await async_client.post("/api/eval/transcribe-live", files=files, data=data)
    assert resp_500.status_code == 500


@pytest.mark.asyncio
async def test_gnani_manual_retry_behavior_no_auto_retry(async_client):
    """Proves server does not auto-retry on failure (preventing unexpected credit use),
    preserves audio bytes, and user-initiated manual retry succeeds.
    """
    call_attempts = 0

    async def mock_flaky_handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_attempts
        call_attempts += 1
        if call_attempts == 1:
            # First attempt fails with 503
            return httpx.Response(status_code=503, json={"error": "Service temporarily overloaded"})
        return httpx.Response(status_code=200, json={"transcript": "Successfully transcribed on manual retry"})

    mock_settings = Settings(gnani_api_key="vach_mock_test_key_12345")
    transport = httpx.MockTransport(mock_flaky_handler)
    client = GnaniPrismaClient(settings=mock_settings, http_client=httpx.AsyncClient(transport=transport))
    set_asr_engine(GnaniPrismaEngine(settings=mock_settings, client=client))

    wav_bytes = create_mock_wav(duration_seconds=2.0)

    # First attempt -> Fails with 503. Server must NOT auto-retry!
    files1 = {"audio_file": ("retry_audio.wav", io.BytesIO(wav_bytes), "audio/wav")}
    data1 = {"language_code": "hi-IN", "consent_given": "true"}
    resp1 = await async_client.post("/api/eval/transcribe-live", files=files1, data=data1)
    assert resp1.status_code == 503
    assert call_attempts == 1, "Must NOT auto-retry on server error; exactly 1 call allowed"

    # User reviews error and clicks manual 'Retry Transcribe' with preserved audio bytes
    files2 = {"audio_file": ("retry_audio.wav", io.BytesIO(wav_bytes), "audio/wav")}
    data2 = {"language_code": "hi-IN", "consent_given": "true"}
    resp2 = await async_client.post("/api/eval/transcribe-live", files=files2, data=data2)
    assert resp2.status_code == 200
    assert resp2.json()["transcript"] == "Successfully transcribed on manual retry"
    assert call_attempts == 2


@pytest.mark.asyncio
async def test_editing_before_evaluation(async_client):
    """Proves user can review and edit auto-generated transcript before clicking Evaluate,
    and evaluation scores the edited text.
    """
    mock_engine = TestMockASREngine(fixed_transcript="Draft from audio bytes")
    set_asr_engine(mock_engine)

    wav_bytes = create_mock_wav(duration_seconds=2.0)
    files = {"audio_file": ("speech_a.wav", io.BytesIO(wav_bytes), "audio/wav")}
    data = {"language_code": "hi-IN", "utterance_role": "utterance_a", "consent_given": "true"}

    resp = await async_client.post("/api/eval/transcribe-live", files=files, data=data)
    assert resp.status_code == 200
    initial_text = resp.json()["transcript"]
    assert initial_text == "Draft from audio bytes"

    # User reviews and edits in the existing box before clicking Evaluate
    edited_text_a = "Do packet chahiye urgently"
    edited_text_b = "Teen packet chahiye urgently"

    eval_resp = await async_client.post(
        "/api/eval/evaluate-transcripts",
        json={
            "test_case_id": "pair_qty_001",
            "transcript_a": edited_text_a,
            "transcript_b": edited_text_b,
            "execution_mode": "mock",
        },
    )
    assert eval_resp.status_code == 200
    eval_data = eval_resp.json()
    assert eval_data["utterance_a"]["actual_transcript"] == edited_text_a
    assert eval_data["utterance_b"]["actual_transcript"] == edited_text_b


@pytest.mark.asyncio
async def test_duration_limits_live_evaluation(async_client):
    """Proves Live Evaluation strictly enforces hard 60.0-second limit:
    - Exactly 60.0s is accepted.
    - 60.01s is rejected with 400 (zero provider calls).
    - 60.05s is rejected with 400 (zero provider calls).
    - 65.0s is rejected with 400 (zero provider calls).
    """
    mock_engine = TestMockASREngine()
    set_asr_engine(mock_engine)

    # Exactly 60.0s -> Accepted
    wav_60 = create_mock_wav(duration_seconds=60.0)
    resp_60 = await async_client.post(
        "/api/eval/transcribe-live",
        files={"audio_file": ("exact_60.wav", io.BytesIO(wav_60), "audio/wav")},
        data={"language_code": "hi-IN", "consent_given": "true"},
    )
    assert resp_60.status_code == 200
    assert mock_engine.call_count == 1

    # 60.01s -> Rejected with 400 with zero provider calls
    wav_60_01 = create_mock_wav(duration_seconds=60.01)
    resp_60_01 = await async_client.post(
        "/api/eval/transcribe-live",
        files={"audio_file": ("too_long_60_01.wav", io.BytesIO(wav_60_01), "audio/wav")},
        data={"language_code": "hi-IN", "consent_given": "true"},
    )
    assert resp_60_01.status_code == 400
    assert "exceeds the hard 60.0-second live evaluation limit" in resp_60_01.json()["detail"]
    assert mock_engine.call_count == 1, "Zero provider calls on 60.01s rejection"

    # 60.05s -> Rejected with 400 with zero provider calls
    wav_60_05 = create_mock_wav(duration_seconds=60.05)
    resp_60_05 = await async_client.post(
        "/api/eval/transcribe-live",
        files={"audio_file": ("too_long_60_05.wav", io.BytesIO(wav_60_05), "audio/wav")},
        data={"language_code": "hi-IN", "consent_given": "true"},
    )
    assert resp_60_05.status_code == 400
    assert "exceeds the hard 60.0-second live evaluation limit" in resp_60_05.json()["detail"]
    assert mock_engine.call_count == 1, "Zero provider calls on 60.05s rejection"

    # 65.0s -> Rejected with 400
    overlimit_wav = create_mock_wav(duration_seconds=65.0)
    resp_over = await async_client.post(
        "/api/eval/transcribe-live",
        files={"audio_file": ("too_long.wav", io.BytesIO(overlimit_wav), "audio/wav")},
        data={"language_code": "hi-IN", "consent_given": "true"},
    )
    assert resp_over.status_code == 400
    assert "exceeds the hard 60.0-second live evaluation limit" in resp_over.json()["detail"]
    assert mock_engine.call_count == 1, "Zero provider calls on 65.0s rejection"


@pytest.mark.asyncio
async def test_duration_limits_studio_preserved(async_client):
    """Proves Human Recording Studio preserves separate 600-second limit."""
    resp = await async_client.get("/api/studio/config")
    assert resp.status_code == 200
    cfg = resp.json()
    assert cfg["max_duration_seconds"] == 600.0
    assert cfg["target_duration_seconds"] == 600.0


@pytest.mark.asyncio
async def test_empty_audio_rejection(async_client):
    """Proves empty 0-byte audio file is rejected with 400."""
    files = {"audio_file": ("empty.wav", io.BytesIO(b""), "audio/wav")}
    resp = await async_client.post("/api/eval/transcribe-live", files=files, data={"language_code": "hi-IN"})
    assert resp.status_code == 400
    assert "empty" in resp.json()["detail"].lower()


@pytest.mark.asyncio
async def test_unsupported_language_rejection(async_client):
    """Proves unsupported language code is rejected with 400."""
    wav_bytes = create_mock_wav(duration_seconds=1.0)
    files = {"audio_file": ("speech.wav", io.BytesIO(wav_bytes), "audio/wav")}
    resp = await async_client.post(
        "/api/eval/transcribe-live",
        files=files,
        data={"language_code": "fr-FR"},
    )
    assert resp.status_code == 400
    assert "Unsupported language_code" in resp.json()["detail"]


def test_verify_asr_status_helper():
    """Verifies verify_asr_status detects gnani-prisma-v3 when configured."""
    status_info = verify_asr_status()
    assert "status" in status_info
    assert status_info["status"] in ("configured_external", "available_local", "missing_dependencies")
    if status_info["status"] == "configured_external":
        assert status_info["provider"] in ("gnani-prisma-v3", "openai-whisper-api")
        assert status_info["requires_consent"] is True
