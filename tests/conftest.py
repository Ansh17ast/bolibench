"""Pytest test configuration and fixtures for BoliBench."""

import io
import os
from pathlib import Path
import struct
import tempfile
import wave
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from unittest.mock import patch

# Pre-import safety isolation: Ensure that any module-level singletons (such as
# engine = EvaluationEngine() in app.routers.evaluation) created during module imports
# resolve to a temporary session database instead of the real bolibench_evaluation.db.
_SESSION_TEST_DIR = Path(tempfile.gettempdir()) / "bolibench_pytest_isolated"
_SESSION_TEST_DIR.mkdir(parents=True, exist_ok=True)
_SESSION_TEST_DB = _SESSION_TEST_DIR / f"bolibench_session_test_{os.getpid()}.db"
if "BOLIBENCH_DB_PATH" not in os.environ:
    os.environ["BOLIBENCH_DB_PATH"] = str(_SESSION_TEST_DB)

from app.config import Settings, get_settings
from app.main import app


@pytest.fixture(autouse=True)
def isolate_test_environment(monkeypatch):
    """Guarantees tests run isolated from developer's local .env live flags.

    Defaults live flags to false and ensures Settings cache is cleared before and after each test.
    """
    monkeypatch.setenv("LIVE_EVALUATION_ENABLED", "false")
    monkeypatch.setenv("LIVE_TRANSCRIPTION_ENABLED", "false")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def isolate_evaluation_storage(tmp_path, monkeypatch):
    """Guarantees each test runs with a unique, isolated SQLite database in a temp directory.

    Redirects all StorageManager instances created during tests (both direct construction
    and module-level singletons) to an isolated temporary database.
    """
    import app.evaluation.storage
    from app.evaluation.storage import StorageManager
    import app.routers.evaluation

    test_db_path = tmp_path / "test_isolated_evaluation.db"

    # 1. Set environment variable so any process/thread/import resolves to this test DB
    monkeypatch.setenv("BOLIBENCH_DB_PATH", str(test_db_path))

    # 2. Monkeypatch module-level DEFAULT_DB_PATH in storage module
    monkeypatch.setattr(app.evaluation.storage, "DEFAULT_DB_PATH", test_db_path)

    # 3. Create isolated StorageManager instance for this test
    test_storage = StorageManager(db_path=test_db_path)

    # 4. Redirect module-level singleton engine in app.routers.evaluation
    orig_engine_storage = getattr(app.routers.evaluation.engine, "storage", None)
    monkeypatch.setattr(app.routers.evaluation.engine, "storage", test_storage)

    try:
        yield test_storage
    finally:
        # Keep engine.storage pointed to an isolated DB; never restore to the production database
        real_db_path = Path(__file__).resolve().parent.parent / "bolibench_evaluation.db"
        if orig_engine_storage and Path(orig_engine_storage.db_path).resolve() != real_db_path.resolve():
            app.routers.evaluation.engine.storage = orig_engine_storage
        else:
            app.routers.evaluation.engine.storage = StorageManager(db_path=_SESSION_TEST_DB)


@pytest.fixture(autouse=True)
def isolate_recording_storage(tmp_path, monkeypatch):
    """Guarantees each test runs with an isolated human_recordings directory in a temp directory.

    Prevents automated tests from polluting the application's real human_recordings_v1/ directory,
    raw audio, standardized audio, metadata, pairs, or manifest.json with test mock recordings.
    """
    import app.services.recording_storage
    from app.services.recording_storage import RecordingStorage
    import app.routers.recording

    test_recordings_dir = tmp_path / "test_isolated_human_recordings"

    # 1. Monkeypatch the default recordings directory so all new RecordingStorage() calls use tmp_path
    monkeypatch.setattr(app.services.recording_storage, "DEFAULT_RECORDINGS_DIR", test_recordings_dir)

    # 2. Swap router's module-level storage instance
    orig_router_storage = app.routers.recording.storage
    test_storage = RecordingStorage(base_dir=test_recordings_dir)
    monkeypatch.setattr(app.routers.recording, "storage", test_storage)

    try:
        yield test_storage
    finally:
        app.routers.recording.storage = orig_router_storage



@pytest.fixture(autouse=True)
def block_outbound_network_calls():
    """Safety guard: prevents any unmocked outbound HTTP transport calls in tests.

    If any test accidentally attempts an unmocked outbound HTTP request to Gnani or any external API,
    this fixture intercepts it at the transport layer, immediately failing the test and preventing
    credit consumption.
    """
    with patch(
        "httpx.AsyncHTTPTransport.handle_async_request",
        side_effect=RuntimeError(
            "Outbound network request blocked in test suite to protect programme credits! "
            "All provider interactions must be mocked in tests."
        ),
    ):
        yield


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
