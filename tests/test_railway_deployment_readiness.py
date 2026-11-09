"""Tests for Railway deployment readiness and persistent storage configuration.

Validates:
1. Centralized persistent data directory resolution (BOLIBENCH_DATA_DIR).
2. SQLite database path resolution under BOLIBENCH_DATA_DIR.
3. Database safeguard: BOLIBENCH_REQUIRE_EXISTING_DB fails closed when DB is absent.
4. RecordingStorage path resolution under BOLIBENCH_DATA_DIR.
5. Human recording ingestion and manifest persistence under BOLIBENCH_DATA_DIR.
6. Benchmark results and checkpoints directory resolution under BOLIBENCH_DATA_DIR.
7. /health endpoint behavior, status, and zero-leakage contract.
8. CORS allowed origins configuration via ALLOWED_ORIGINS.
9. Single worker / process safety contract.
"""

import io
import os
from pathlib import Path
import sqlite3
import tempfile
import pytest
from httpx import ASGITransport, AsyncClient

from app.config import (
    Settings,
    get_benchmark_checkpoints_dir,
    get_benchmark_results_dir,
    get_data_dir,
    get_recordings_dir,
    get_settings,
)
from app.evaluation.storage import StorageManager, get_default_db_path
from app.main import app
from app.services.recording_storage import RecordingStorage


def test_data_dir_resolution(tmp_path, monkeypatch):
    """Proves BOLIBENCH_DATA_DIR routes all persistent storage directories to the data root."""
    data_dir = tmp_path / "railway_data"
    data_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("BOLIBENCH_DATA_DIR", str(data_dir))
    get_settings.cache_clear()

    settings = get_settings()
    assert settings.data_dir_path == data_dir
    assert get_data_dir() == data_dir
    assert get_recordings_dir() == data_dir / "human_recordings_v1"
    assert get_benchmark_results_dir() == data_dir / "benchmark_results"
    assert get_benchmark_checkpoints_dir() == data_dir / "benchmark_checkpoints"

    # Clear cache on teardown
    get_settings.cache_clear()


def test_sqlite_db_resolution_under_data_dir(tmp_path, monkeypatch):
    """Proves SQLite database path resolves to data_dir/bolibench_evaluation.db when BOLIBENCH_DB_PATH is unset."""
    data_dir = tmp_path / "railway_data"
    data_dir.mkdir(parents=True, exist_ok=True)

    import app.evaluation.storage
    # Reset DEFAULT_DB_PATH from conftest fixture to standard PROD_DB_PATH
    monkeypatch.setattr(app.evaluation.storage, "DEFAULT_DB_PATH", app.evaluation.storage.PROD_DB_PATH)
    monkeypatch.delenv("BOLIBENCH_DB_PATH", raising=False)
    monkeypatch.setenv("BOLIBENCH_DATA_DIR", str(data_dir))
    monkeypatch.setenv("BOLIBENCH_ALLOW_REAL_DB", "1")  # Bypass pytest safety redirect
    get_settings.cache_clear()

    db_path = get_default_db_path()
    assert db_path == data_dir / "bolibench_evaluation.db"

    # Initializing StorageManager creates tables in data_dir
    sm = StorageManager(db_path=db_path)
    assert (data_dir / "bolibench_evaluation.db").exists()

    with sm._get_connection() as conn:
        c = conn.cursor()
        c.execute("PRAGMA journal_mode;")
        mode = c.fetchone()[0]
        assert mode.lower() == "wal"

    get_settings.cache_clear()


def test_database_safeguard_fails_closed_when_expected_db_missing(tmp_path, monkeypatch):
    """Proves BOLIBENCH_REQUIRE_EXISTING_DB=true raises RuntimeError if database does not exist."""
    data_dir = tmp_path / "empty_railway_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    target_db = data_dir / "bolibench_evaluation.db"

    monkeypatch.setenv("BOLIBENCH_DATA_DIR", str(data_dir))
    monkeypatch.setenv("BOLIBENCH_REQUIRE_EXISTING_DB", "true")
    get_settings.cache_clear()

    assert not target_db.exists()

    with pytest.raises(RuntimeError) as exc_info:
        StorageManager(db_path=target_db)

    assert "Production database safeguard" in str(exc_info.value)
    assert "Expected existing database" in str(exc_info.value)
    assert not target_db.exists()  # Did not silently create empty database

    get_settings.cache_clear()


def test_recording_storage_under_data_dir(tmp_path, monkeypatch):
    """Proves RecordingStorage initializes and stores files within BOLIBENCH_DATA_DIR."""
    data_dir = tmp_path / "railway_data"
    data_dir.mkdir(parents=True, exist_ok=True)

    import app.services.recording_storage
    proj_default = Path(__file__).resolve().parent.parent / "human_recordings_v1"
    monkeypatch.setattr(app.services.recording_storage, "DEFAULT_RECORDINGS_DIR", proj_default)
    monkeypatch.setenv("BOLIBENCH_DATA_DIR", str(data_dir))
    get_settings.cache_clear()

    storage = RecordingStorage()
    assert storage.base_dir == data_dir / "human_recordings_v1"
    assert (data_dir / "human_recordings_v1" / "raw").exists()
    assert (data_dir / "human_recordings_v1" / "standardized_16k").exists()
    assert (data_dir / "human_recordings_v1" / "manifest.json").exists()

    # Ingest a valid short test recording
    import wave
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        # 1.0 second of audio
        wf.writeframes(b"\x00\x00" * 16000)
    wav_bytes = buf.getvalue()

    storage.register_trusted_speaker("spk_railway_test")
    meta = storage.save_recording(
        raw_bytes=wav_bytes,
        original_filename="test_railway.wav",
        speaker_id="spk_railway_test",
        language="hinglish",
        category="quantities_and_numbers",
        reference_transcript="Testing railway storage",
        verified_transcript="Testing railway storage",
        is_transcript_verified=True,
        consent_given=True,
        consent_statement="Consent confirmed.",
    )

    assert meta.recording_id.startswith("rec_")
    assert (data_dir / "human_recordings_v1" / "raw" / meta.raw_audio_filename).exists()
    assert (data_dir / "human_recordings_v1" / "standardized_16k" / meta.standardized_audio_filename).exists()

    # Re-instantiate storage to verify persistence from manifest
    storage2 = RecordingStorage()
    recs = storage2.list_recordings()
    assert any(r.get('recording_id') == meta.recording_id for r in recs)

    get_settings.cache_clear()


@pytest.mark.asyncio
async def test_health_check_endpoint():
    """Proves /health returns 200 OK without disclosing credentials or filesystem paths."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()

        assert data["status"] == "ok"
        assert "gnani_api_configured" in data
        assert "live_evaluation_enabled" in data
        assert "live_transcription_enabled" in data
        assert "app_environment" in data
        assert "timestamp" in data

        # Security check: Zero credential disclosure
        resp_text = resp.text.lower()
        assert "api_key" not in data
        assert "admin_api_key" not in data
        assert "secret" not in resp_text
        assert "password" not in resp_text
        assert "c:\\" not in resp_text
        assert "/data" not in resp_text


@pytest.mark.asyncio
async def test_cors_allowed_origins_configuration(monkeypatch):
    """Proves ALLOWED_ORIGINS settings cleanly extends allowed CORS origins for Railway deployment."""
    monkeypatch.setenv("ALLOWED_ORIGINS", "https://bolibench.up.railway.app,https://eval.example.com")
    get_settings.cache_clear()

    from app.main import _resolve_allowed_origins
    origins = _resolve_allowed_origins()

    assert "https://bolibench.up.railway.app" in origins
    assert "https://eval.example.com" in origins
    assert "http://localhost:8000" in origins  # Preserves default local origins

    get_settings.cache_clear()


@pytest.mark.asyncio
async def test_studio_page_and_config_served():
    """Proves studio page and studio config are served cleanly."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        studio_resp = await client.get("/studio")
        assert studio_resp.status_code == 200
        assert "In-Website Human Audio Recording Studio" in studio_resp.text
        assert "600.0s remaining" in studio_resp.text

        cfg_resp = await client.get("/api/studio/config")
        assert cfg_resp.status_code == 200
        cfg = cfg_resp.json()
        assert cfg["target_duration_seconds"] == 600.0
        assert cfg["max_duration_seconds"] == 600.0


def test_dockerfile_and_entrypoint_integrity():
    """Proves Dockerfile and entrypoint.sh implement safe non-root startup and volume permission fixing."""
    repo_root = Path(__file__).resolve().parent.parent
    dockerfile_path = repo_root / "Dockerfile"
    entrypoint_path = repo_root / "entrypoint.sh"

    assert dockerfile_path.exists(), "Dockerfile must exist"
    assert entrypoint_path.exists(), "entrypoint.sh must exist"

    df_content = dockerfile_path.read_text(encoding="utf-8")
    assert "gosu" in df_content, "Dockerfile must install gosu for dropping privileges"
    assert 'ENTRYPOINT ["/app/entrypoint.sh"]' in df_content, "Dockerfile must set entrypoint.sh"
    assert "--workers 1" in df_content, "Dockerfile must run single Uvicorn worker for SQLite safety"
    assert "useradd -m -u 10001" in df_content, "Dockerfile must configure UID 10001 appuser"

    # Check entrypoint.sh line endings and content
    ep_bytes = entrypoint_path.read_bytes()
    assert b"\r" not in ep_bytes, "entrypoint.sh must use pure LF line endings (no CRLF)"

    ep_content = ep_bytes.decode("utf-8")
    assert ep_content.startswith("#!/bin/sh"), "entrypoint.sh must start with #!/bin/sh"
    assert "10001" in ep_content, "entrypoint.sh must reference UID 10001"
    assert "gosu" in ep_content, "entrypoint.sh must use gosu to drop privileges"
    assert "chown" in ep_content, "entrypoint.sh must fix volume ownership when running as root"
    assert "perm_check" in ep_content, "entrypoint.sh must verify write permission before startup"


def test_storage_unwritable_directory_fails_closed(tmp_path):
    """Proves StorageManager fails closed when the target persistent directory cannot be written."""
    data_dir = tmp_path / "test_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    bad_db_path = data_dir / "uncreated_nonexistent_parent" / "sub" / "bolibench_evaluation.db"

    blocking_file = data_dir / "uncreated_nonexistent_parent"
    blocking_file.write_text("blocker")

    try:
        with pytest.raises((NotADirectoryError, OSError)):
            StorageManager(db_path=bad_db_path)
    finally:
        if blocking_file.exists():
            blocking_file.unlink()
