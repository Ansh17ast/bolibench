"""Pytest test configuration and fixtures for BoliBench."""

import io
import struct
import wave
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.config import Settings, get_settings
from app.main import app


@pytest.fixture(autouse=True)
def reset_settings_cache():
    """Ensures get_settings lru_cache is cleared before and after every test."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def create_mock_wav(duration_seconds: float = 1.0, sample_rate: int = 16000) -> bytes:
    """Generates valid in-memory PCM mono WAV audio bytes for testing."""
    buffer = io.BytesIO()
    num_frames = int(duration_seconds * sample_rate)

    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)

        # Generate simple silence / tone frames
        frames = struct.pack(f"<{num_frames}h", *(0 for _ in range(num_frames)))
        wav_file.writeframes(frames)

    return buffer.getvalue()


@pytest.fixture
def valid_wav_bytes() -> bytes:
    """Fixture providing 1-second valid WAV audio."""
    return create_mock_wav(duration_seconds=1.0)


@pytest.fixture
def long_wav_bytes() -> bytes:
    """Fixture providing 65-second WAV audio (exceeds 60s limit)."""
    return create_mock_wav(duration_seconds=65.0)


@pytest.fixture
def mock_settings() -> Settings:
    """Provides test settings with configured mock credentials."""
    return Settings(
        gnani_api_key="test_mock_api_key_12345",
        gnani_base_url="https://api.vachana.ai/stt/v3",
        gnani_request_timeout_seconds=5.0,
        max_audio_size_bytes=10 * 1024 * 1024,  # 10 MB for testing
        app_env="testing",
    )


@pytest.fixture
def unconfigured_settings() -> Settings:
    """Provides test settings with no API key."""
    return Settings(
        gnani_api_key="",
        gnani_base_url="https://api.vachana.ai/stt/v3",
        gnani_request_timeout_seconds=5.0,
        max_audio_size_bytes=10 * 1024 * 1024,
        app_env="testing",
    )


@pytest_asyncio.fixture
async def async_client():
    """Async client for testing FastAPI routes."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
