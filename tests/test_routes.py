"""Route tests for FastAPI application endpoints."""

import io
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
    """Test unmocked request without GNANI_API_KEY returns 401 with remediation instructions."""
    monkeypatch.delenv("GNANI_API_KEY", raising=False)
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

    response = await async_client.post("/api/transcribe", files=files, data=data)
    assert response.status_code == 401
    result = response.json()
    assert result["error_type"] == "MISSING_CREDENTIALS"
    assert "remediation" in result
    assert "GNANI_API_KEY" in result["remediation"]
