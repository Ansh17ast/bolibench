"""Route tests for FastAPI application endpoints."""

import io
from unittest.mock import patch
import pytest


@pytest.mark.asyncio
async def test_health_check(async_client):
    """Test /health endpoint returns valid system status."""
    response = await async_client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "ok"
    assert "gnani_api_configured" in data
    assert "app_environment" in data
    assert len(data["supported_languages"]) == 10
    assert "hi-IN" in data["supported_languages"]
    assert "en-IN" in data["supported_languages"]
    assert "verbatim" in data["supported_formats"]
    assert "transcribe" in data["supported_formats"]
    assert data["live_transcription_enabled"] is False
    assert data["live_evaluation_enabled"] is False


@pytest.mark.asyncio
async def test_index_page(async_client):
    """Test root page serves interactive HTML."""
    response = await async_client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "BoliBench" in response.text
    assert "Indian Voice AI Crash-Test Lab" in response.text


@pytest.mark.asyncio
async def test_transcribe_mock_mode(async_client, valid_wav_bytes):
    """Test transcription in mock mode returns cleanly formatted mock response."""
    files = {
        "audio_file": ("test_speech.wav", io.BytesIO(valid_wav_bytes), "audio/wav"),
    }
    data = {
        "language_code": "hi-IN",
        "format_mode": "verbatim",
        "mock_mode": "true",
    }

    response = await async_client.post("/api/transcribe", files=files, data=data)
    assert response.status_code == 200

    result = response.json()
    assert result["success"] is True
    assert result["is_mocked"] is True
    assert result["language_code"] == "hi-IN"
    assert "MOCK TRANSCRIPTION" in result["transcript"]
    assert result["backend_latency_ms"] > 0


@pytest.mark.asyncio
async def test_transcribe_invalid_language(async_client, valid_wav_bytes):
    """Test validation fails when language code is not in supported list."""
    files = {
        "audio_file": ("sample.wav", io.BytesIO(valid_wav_bytes), "audio/wav"),
    }
    data = {
        "language_code": "fr-FR",  # Not a supported Indian language
        "mock_mode": "true",
    }

    response = await async_client.post("/api/transcribe", files=files, data=data)
    assert response.status_code == 400
    assert "Unsupported language_code" in response.json()["detail"]


@pytest.mark.asyncio
async def test_transcribe_unsupported_file_extension(async_client):
    """Test validation fails when file has an invalid extension."""
    files = {
        "audio_file": ("script.py", io.BytesIO(b"print('hello')"), "text/x-python"),
    }
    data = {
        "language_code": "hi-IN",
        "mock_mode": "true",
    }

    response = await async_client.post("/api/transcribe", files=files, data=data)
    assert response.status_code == 400
    result = response.json()
    assert result["error_type"] == "UNSUPPORTED_EXTENSION"


@pytest.mark.asyncio
async def test_transcribe_empty_file(async_client):
    """Test validation fails when file content is 0 bytes."""
    files = {
        "audio_file": ("empty.wav", io.BytesIO(b""), "audio/wav"),
    }
    data = {
        "language_code": "hi-IN",
        "mock_mode": "true",
    }

    response = await async_client.post("/api/transcribe", files=files, data=data)
    assert response.status_code == 400
    result = response.json()
    assert result["error_type"] == "EMPTY_FILE"


@pytest.mark.asyncio
async def test_transcribe_missing_credentials_unmocked(async_client, valid_wav_bytes, monkeypatch):
    """Test unmocked request without GNANI_API_KEY returns 401 with remediation instructions when live transcription is enabled."""
    monkeypatch.setenv("GNANI_API_KEY", "")
    monkeypatch.setenv("LIVE_TRANSCRIPTION_ENABLED", "true")
    from app.config import get_settings
    get_settings.cache_clear()

    files = {
        "audio_file": ("sample.wav", io.BytesIO(valid_wav_bytes), "audio/wav"),
    }
    data = {
        "language_code": "hi-IN",
        "format_mode": "verbatim",
        "mock_mode": "false",  # Request real API without credentials
    }

    with patch("httpx.AsyncHTTPTransport.handle_async_request") as mock_transport:
        response = await async_client.post("/api/transcribe", files=files, data=data)
        assert response.status_code == 401
        result = response.json()
        assert result["error_type"] == "MISSING_CREDENTIALS"
        assert "remediation" in result
        assert "GNANI_API_KEY" in result["remediation"]
        mock_transport.assert_not_called()


@pytest.mark.asyncio
async def test_transcribe_live_disabled_by_default_rejects_with_403(async_client, valid_wav_bytes, monkeypatch):
    """Proves unmocked requests to /api/transcribe are rejected with 403 Forbidden when LIVE_TRANSCRIPTION_ENABLED=false."""
    monkeypatch.setenv("LIVE_TRANSCRIPTION_ENABLED", "false")
    monkeypatch.setenv("GNANI_API_KEY", "test_mock_secret_key_12345")
    from app.config import get_settings
    get_settings.cache_clear()

    files = {
        "audio_file": ("sample.wav", io.BytesIO(valid_wav_bytes), "audio/wav"),
    }
    data = {
        "language_code": "hi-IN",
        "format_mode": "verbatim",
        "mock_mode": "false",
    }

    with patch("app.services.gnani_client.GnaniPrismaClient.transcribe") as mock_transcribe, \
         patch("httpx.AsyncHTTPTransport.handle_async_request") as mock_transport:
        response = await async_client.post("/api/transcribe", files=files, data=data)
        assert response.status_code == 403
        assert "disabled by default to protect programme credits" in response.json()["detail"].lower()
        mock_transcribe.assert_not_called()
        mock_transport.assert_not_called()


@pytest.mark.asyncio
async def test_transcribe_live_disabled_with_mock_credentials_makes_zero_calls(async_client, valid_wav_bytes, monkeypatch):
    """Proves live transcription safety gate blocks requests even if GNANI_API_KEY is configured in test settings."""
    monkeypatch.setenv("GNANI_API_KEY", "test_mock_secret_key_12345")
    monkeypatch.setenv("LIVE_TRANSCRIPTION_ENABLED", "false")
    from app.config import get_settings
    get_settings.cache_clear()

    files = {
        "audio_file": ("sample.wav", io.BytesIO(valid_wav_bytes), "audio/wav"),
    }
    data = {
        "language_code": "hi-IN",
        "format_mode": "verbatim",
        "mock_mode": "false",
    }

    with patch("app.services.gnani_client.GnaniPrismaClient.transcribe") as mock_transcribe:
        response = await async_client.post("/api/transcribe", files=files, data=data)
        assert response.status_code == 403
        assert "disabled by default to protect programme credits" in response.json()["detail"].lower()
        mock_transcribe.assert_not_called()


@pytest.mark.asyncio
async def test_single_audio_latency_labels_and_semantics(async_client, valid_wav_bytes):
    """Proves single-audio transcription UI labels and response field semantics.

    - Replaces ambiguous 'Backend Latency' with 'Backend-measured Gnani API request time'.
    - Confirms 'Browser End-to-End Latency' is kept separate.
    - Confirms 'Provider-Internal Processing Time' shows 'Not reported' when not provided by Gnani.
    - Confirms backend_latency_ms is preserved as an alias for client_request_elapsed_ms.
    """
    # 1. UI presentation labels
    resp_index = await async_client.get("/")
    assert resp_index.status_code == 200
    html = resp_index.text
    assert "Browser End-to-End Latency" in html
    assert "Backend-measured Gnani API request time" in html
    assert "Provider-Internal Processing Time" in html
    assert "Not reported" in html

    # 2. API response data semantics
    files = {
        "audio_file": ("sample.wav", io.BytesIO(valid_wav_bytes), "audio/wav"),
    }
    data = {
        "language_code": "hi-IN",
        "format_mode": "verbatim",
        "mock_mode": "true",
    }
    resp = await async_client.post("/api/transcribe", files=files, data=data)
    assert resp.status_code == 200
    result = resp.json()
    assert result["backend_latency_ms"] == 15.5
    assert result["client_request_elapsed_ms"] == 15.5
    assert result["provider_processing_time_ms"] is None


@pytest.mark.asyncio
async def test_health_check_live_evaluation_disabled_by_default(async_client, monkeypatch):
    """Proves /health endpoint reflects that live evaluation is disabled by default."""
    monkeypatch.setenv("LIVE_EVALUATION_ENABLED", "false")
    monkeypatch.setenv("LIVE_TRANSCRIPTION_ENABLED", "false")
    from app.config import get_settings
    get_settings.cache_clear()

    resp = await async_client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["live_evaluation_enabled"] is False
    assert data["live_transcription_enabled"] is False


@pytest.mark.asyncio
async def test_health_check_live_transcription_disabled_by_default(async_client, monkeypatch):
    """Proves /health endpoint reflects that live transcription is disabled by default."""
    monkeypatch.setenv("LIVE_TRANSCRIPTION_ENABLED", "false")
    from app.config import get_settings
    get_settings.cache_clear()

    resp = await async_client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["live_transcription_enabled"] is False


@pytest.mark.asyncio
async def test_transcribe_missing_credentials_safe_when_env_file_supplies_key(async_client, valid_wav_bytes, monkeypatch, tmp_path):
    """Regression test: Proves missing-credentials request remains safe with 401 response and ZERO
    outbound provider requests, even when an underlying .env file supplies a real non-empty API key.
    """
    # 1. Create an isolated simulated .env file containing a non-empty credentials definition
    simulated_env = tmp_path / "simulated_user.env"
    simulated_env.write_text(
        "GNANI_API_KEY=simulated_real_customer_key_abcdef123456\n"
        "LIVE_TRANSCRIPTION_ENABLED=true\n",
        encoding="utf-8",
    )

    from app.config import get_settings
    # Patch Settings model_config to load this simulated file
    with patch("app.config.Settings.model_config", {"env_file": str(simulated_env), "env_file_encoding": "utf-8", "extra": "ignore"}):
        # Explicit empty override prevents leaking the key from the .env file
        monkeypatch.setenv("GNANI_API_KEY", "")
        monkeypatch.setenv("LIVE_TRANSCRIPTION_ENABLED", "true")
        get_settings.cache_clear()

        safe_settings = get_settings()
        assert safe_settings.is_api_configured is False
        assert safe_settings.gnani_api_key == ""

        files = {
            "audio_file": ("sample.wav", io.BytesIO(valid_wav_bytes), "audio/wav"),
        }
        data = {
            "language_code": "hi-IN",
            "format_mode": "verbatim",
            "mock_mode": "false",
        }

        # Verify the unmocked transcribe request returns 401 without any network request
        with patch("httpx.AsyncHTTPTransport.handle_async_request") as mock_transport:
            response = await async_client.post("/api/transcribe", files=files, data=data)
            assert response.status_code == 401
            result = response.json()
            assert result["error_type"] == "MISSING_CREDENTIALS"
            assert "GNANI_API_KEY" in result["remediation"]
            mock_transport.assert_not_called()
