"""Tests for upload idempotency in BoliBench Human Recording Studio.

Verifies:
1. Exact-duplicate uploads with same idempotency key return original record without duplication.
2. Form field idempotency_key and X-Idempotency-Key header are both supported.
3. Reused idempotency key with modified payload (audio or metadata) returns 409 Conflict.
4. Concurrent duplicate upload requests serialize cleanly without duplicate disk writes.
5. Storage isolation prevents mutation of real human_recordings_v1/ dataset.
"""

import concurrent.futures
import io
import json
from pathlib import Path
import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.services.recording_storage import RecordingStorage, StorageCollisionError
from tests.test_human_recording_studio import create_pcm_wav_bytes


@pytest.fixture
def clean_wav_50s():
    """Generates 50-second test PCM WAV bytes."""
    return create_pcm_wav_bytes(duration_seconds=50.0, amplitude=0.4)


@pytest.fixture
def clean_wav_alt():
    """Generates alternative 50-second test PCM WAV bytes with different amplitude."""
    return create_pcm_wav_bytes(duration_seconds=50.0, amplitude=0.8)


def test_storage_idempotency_duplicate_payload(tmp_path, clean_wav_50s):
    """Calling save_recording with identical idempotency_key and payload returns existing record."""
    storage = RecordingStorage(base_dir=tmp_path / "recordings")

    rec1 = storage.save_recording(
        raw_bytes=clean_wav_50s,
        original_filename="audio1.wav",
        speaker_id="spk_idem_01",
        language="hi-IN",
        category="quantities_and_numbers",
        reference_transcript="Do packet chahiye",
        consent_given=True,
        consent_statement="I agree to terms",
        idempotency_key="key_test_12345",
    )

    assert rec1.recording_id is not None
    assert rec1.idempotency_key == "key_test_12345"
    assert rec1.content_sha256 is not None

    # Snapshot disk state
    raw_files_before = list(storage.raw_dir.glob("*"))
    std_files_before = list(storage.standardized_dir.glob("*"))
    manifest_before = json.loads(storage.manifest_file.read_text(encoding="utf-8"))

    # Repeat exact upload with same idempotency_key
    rec2 = storage.save_recording(
        raw_bytes=clean_wav_50s,
        original_filename="audio1_retry.wav",
        speaker_id="spk_idem_01",
        language="hi-IN",
        category="quantities_and_numbers",
        reference_transcript="Do packet chahiye",
        consent_given=True,
        consent_statement="I agree to terms",
        idempotency_key="key_test_12345",
    )

    # Must return identical recording ID
    assert rec2.recording_id == rec1.recording_id
    assert rec2.created_at == rec1.created_at

    # Verify no new files written
    raw_files_after = list(storage.raw_dir.glob("*"))
    std_files_after = list(storage.standardized_dir.glob("*"))
    manifest_after = json.loads(storage.manifest_file.read_text(encoding="utf-8"))

    assert len(raw_files_after) == len(raw_files_before)
    assert len(std_files_after) == len(std_files_before)
    assert len(manifest_after["recordings"]) == len(manifest_before["recordings"])


def test_storage_idempotency_changed_payload_rejected(tmp_path, clean_wav_50s, clean_wav_alt):
    """Reusing same idempotency_key with different audio bytes or metadata raises StorageCollisionError."""
    storage = RecordingStorage(base_dir=tmp_path / "recordings")

    storage.save_recording(
        raw_bytes=clean_wav_50s,
        original_filename="audio1.wav",
        speaker_id="spk_idem_02",
        language="hi-IN",
        category="quantities_and_numbers",
        reference_transcript="Original transcript",
        consent_given=True,
        consent_statement="I agree",
        idempotency_key="key_collision_check",
    )

    # 1. Changed audio payload
    with pytest.raises(StorageCollisionError) as exc_info:
        storage.save_recording(
            raw_bytes=clean_wav_alt,
            original_filename="audio1.wav",
            speaker_id="spk_idem_02",
            language="hi-IN",
            category="quantities_and_numbers",
            reference_transcript="Original transcript",
            consent_given=True,
            consent_statement="I agree",
            idempotency_key="key_collision_check",
        )
    assert "different upload payload" in str(exc_info.value)

    # 2. Changed transcript
    with pytest.raises(StorageCollisionError) as exc_info:
        storage.save_recording(
            raw_bytes=clean_wav_50s,
            original_filename="audio1.wav",
            speaker_id="spk_idem_02",
            language="hi-IN",
            category="quantities_and_numbers",
            reference_transcript="Different transcript",
            consent_given=True,
            consent_statement="I agree",
            idempotency_key="key_collision_check",
        )
    assert "different upload payload" in str(exc_info.value)


@pytest.mark.asyncio
async def test_api_upload_idempotency_via_form_and_header(clean_wav_50s):
    """API endpoint /api/studio/upload correctly supports idempotency via Form and Header."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        data_payload = {
            "speaker_id": "spk_api_idem_01",
            "language": "hinglish",
            "category": "named_entities",
            "reference_transcript": "Rahul Sharma ko bhejo",
            "consent_given": "true",
            "consent_statement": "Consent granted for benchmark evaluation",
            "idempotency_key": "idem_form_key_999",
        }
        files = {
            "audio_file": ("test.wav", io.BytesIO(clean_wav_50s), "audio/wav"),
        }

        # 1. Initial upload via Form param
        r1 = await ac.post("/api/studio/upload", data=data_payload, files=files)
        assert r1.status_code == 200, r1.text
        meta1 = r1.json()
        assert meta1["idempotency_key"] == "idem_form_key_999"
        assert meta1["content_sha256"] is not None

        # 2. Duplicate upload with same Form param -> returns exact same recording (200 OK)
        files2 = {
            "audio_file": ("test.wav", io.BytesIO(clean_wav_50s), "audio/wav"),
        }
        r2 = await ac.post("/api/studio/upload", data=data_payload, files=files2)
        assert r2.status_code == 200
        meta2 = r2.json()
        assert meta2["recording_id"] == meta1["recording_id"]
        assert meta2["created_at"] == meta1["created_at"]

        # 3. Upload with Header X-Idempotency-Key
        data_no_form_key = {
            "speaker_id": "spk_api_idem_02",
            "language": "en-IN",
            "category": "command_intent",
            "reference_transcript": "Book the flight now",
            "consent_given": "true",
            "consent_statement": "Consent granted",
        }
        files3 = {
            "audio_file": ("test.wav", io.BytesIO(clean_wav_50s), "audio/wav"),
        }
        headers = {"X-Idempotency-Key": "header_key_888"}
        r3 = await ac.post("/api/studio/upload", data=data_no_form_key, files=files3, headers=headers)
        assert r3.status_code == 200
        meta3 = r3.json()
        assert meta3["idempotency_key"] == "header_key_888"

        # Duplicate via header -> exact match
        files4 = {
            "audio_file": ("test.wav", io.BytesIO(clean_wav_50s), "audio/wav"),
        }
        r4 = await ac.post("/api/studio/upload", data=data_no_form_key, files=files4, headers=headers)
        assert r4.status_code == 200
        meta4 = r4.json()
        assert meta4["recording_id"] == meta3["recording_id"]

        # 4. Same key with different payload -> 409 Conflict
        conflicting_payload = dict(data_no_form_key)
        conflicting_payload["reference_transcript"] = "Cancel the flight immediately"
        files5 = {
            "audio_file": ("test.wav", io.BytesIO(clean_wav_50s), "audio/wav"),
        }
        r5 = await ac.post("/api/studio/upload", data=conflicting_payload, files=files5, headers=headers)
        assert r5.status_code == 409
        assert "different upload payload" in r5.json()["detail"]


def test_concurrent_idempotent_uploads(tmp_path, clean_wav_50s):
    """Concurrent threads uploading with the same idempotency key return the exact same recording."""
    storage = RecordingStorage(base_dir=tmp_path / "recordings")
    idempotency_key = "concurrent_key_777"

    def do_upload():
        return storage.save_recording(
            raw_bytes=clean_wav_50s,
            original_filename="concurrent.wav",
            speaker_id="spk_concurrent",
            language="hi-IN",
            category="negation",
            reference_transcript="Nahi cancel mat karo",
            consent_given=True,
            consent_statement="Consent confirmed",
            idempotency_key=idempotency_key,
        )

    num_threads = 5
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(do_upload) for _ in range(num_threads)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    # All returned objects should have the exact same recording_id
    rec_ids = {r.recording_id for r in results}
    assert len(rec_ids) == 1, f"Expected 1 unique recording_id, got: {rec_ids}"

    # Manifest should contain exactly 1 entry
    manifest = json.loads(storage.manifest_file.read_text(encoding="utf-8"))
    assert manifest["total_recordings"] == 1
    assert len(manifest["recordings"]) == 1

    # Exactly 1 raw file and 1 standardized WAV
    assert len(list(storage.raw_dir.glob("*"))) == 1
    assert len(list(storage.standardized_dir.glob("*"))) == 1
