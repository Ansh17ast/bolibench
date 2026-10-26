"""Regression tests verifying human recording storage isolation in the test suite.

Proves that recording-related endpoint and service tests NEVER mutate or leak mock records
into the real production `human_recordings_v1/` directory, manifest.json, or audio folders.
"""

import json
from pathlib import Path
import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.services.recording_storage import RecordingStorage
from tests.test_human_workflow_acceptance import generate_pcm_wav


def snapshot_real_storage():
    """Captures a snapshot of the real human_recordings_v1 directory state."""
    real_dir = Path(__file__).resolve().parent.parent / "human_recordings_v1"
    manifest_path = real_dir / "manifest.json"

    manifest_mtime = manifest_path.stat().st_mtime if manifest_path.exists() else None
    manifest_size = manifest_path.stat().st_size if manifest_path.exists() else None

    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)
        recordings_count = len(manifest_data.get("recordings", []))
        pairs_count = len(manifest_data.get("pairs", []))
        total_recs = manifest_data.get("total_recordings", 0)
    else:
        recordings_count = 0
        pairs_count = 0
        total_recs = 0

    raw_files = set(f.name for f in (real_dir / "raw").glob("*")) if (real_dir / "raw").exists() else set()
    std_files = set(f.name for f in (real_dir / "standardized_16k").glob("*")) if (real_dir / "standardized_16k").exists() else set()
    meta_files = set(f.name for f in (real_dir / "metadata").glob("*")) if (real_dir / "metadata").exists() else set()
    pair_files = set(f.name for f in (real_dir / "pairs").glob("*")) if (real_dir / "pairs").exists() else set()

    return {
        "mtime": manifest_mtime,
        "size": manifest_size,
        "recordings_count": recordings_count,
        "pairs_count": pairs_count,
        "total_recordings": total_recs,
        "raw_files": raw_files,
        "std_files": std_files,
        "meta_files": meta_files,
        "pair_files": pair_files,
    }


@pytest.mark.asyncio
async def test_recording_endpoint_does_not_mutate_real_storage():
    """Proves that performing studio uploads and pair creation via FastAPI endpoints
    leaves the real human_recordings_v1 storage completely untouched."""
    before = snapshot_real_storage()

    # Perform studio actions via client
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Create a test pair
        pair_resp = await client.post("/api/studio/pairs/create", json={
            "pair_id": "pair_isolation_check_001",
            "speaker_id": "spk_iso_check",
            "language": "hi-IN",
            "category": "quantities_and_numbers",
            "scenario_description": "Isolation regression verification",
            "expected_distinction": "Isolation test",
        })
        assert pair_resp.status_code == 200

        # Upload a recording
        wav_bytes = generate_pcm_wav(duration_sec=50.0)
        upload_resp = await client.post("/api/studio/upload", data={
            "recording_id": "rec_iso_check_001",
            "pair_id": "pair_isolation_check_001",
            "utterance_role": "utterance_a",
            "speaker_id": "spk_iso_check",
            "language": "hi-IN",
            "category": "quantities_and_numbers",
            "reference_transcript": "Do packet chahiye",
            "verified_transcript": "Do packet chahiye",
            "is_transcript_verified": "true",
            "consent_given": "true",
            "consent_statement": "Consent for isolation test",
            "client_duration_seconds": "50.0",
        }, files={"audio_file": ("test.wav", wav_bytes, "audio/wav")})
        assert upload_resp.status_code == 200

    after = snapshot_real_storage()

    # Assert real storage remained strictly unchanged
    assert after["mtime"] == before["mtime"], "Real manifest.json mtime was modified during test!"
    assert after["size"] == before["size"], "Real manifest.json size changed during test!"
    assert after["total_recordings"] == before["total_recordings"], "Real total_recordings count changed!"
    assert after["raw_files"] == before["raw_files"], "New file created in real raw/ directory!"
    assert after["std_files"] == before["std_files"], "New file created in real standardized_16k/ directory!"
    assert after["meta_files"] == before["meta_files"], "New file created in real metadata/ directory!"
    assert after["pair_files"] == before["pair_files"], "New file created in real pairs/ directory!"


def test_direct_recording_storage_instantiation_uses_isolated_dir():
    """Proves that direct RecordingStorage() instantiations inside tests use the isolated temp directory."""
    before = snapshot_real_storage()

    storage = RecordingStorage()
    assert "test_isolated_human_recordings" in str(storage.base_dir)

    wav_bytes = generate_pcm_wav(duration_sec=50.0)
    meta = storage.save_recording(
        raw_bytes=wav_bytes,
        original_filename="direct_test.wav",
        speaker_id="spk_direct_iso",
        language="en-IN",
        category="named_entities",
        reference_transcript="Send invoice to Rahul Sharma",
        consent_given=True,
        consent_statement="Direct test consent",
    )
    assert meta is not None

    after = snapshot_real_storage()
    assert after["raw_files"] == before["raw_files"]
    assert after["std_files"] == before["std_files"]
    assert after["meta_files"] == before["meta_files"]
    assert after["pair_files"] == before["pair_files"]
    assert after["total_recordings"] == before["total_recordings"]
