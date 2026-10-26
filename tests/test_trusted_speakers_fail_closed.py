"""Tests for Trusted Speaker Registry Fail-Closed Security & Integrity Gates.

Verifies:
1. Missing trusted_speakers.json registry defaults to empty set (fails closed; never trusts arbitrary speakers).
2. Corrupted or unreadable registry file safely defaults to empty set.
3. Empty registry file fails closed for all speakers.
4. Unknown speakers remain untrusted even when other speakers are registered.
5. Test/mock speaker prefixes (spk_test, spk_mock, spk_fail, etc.) are strictly rejected during registration and is_speaker_trusted checks.
6. Pair verification fails closed if either or both speakers are not registered trusted speakers.
7. Explicitly registered trusted speakers succeed in pair verification and yield human_verified classifications.
8. API endpoints (/api/recording/trusted-speakers) enforce validation against test/mock prefixes.
"""

import io
import json
import wave
import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.recording_schemas import (
    AudioDiagnostics,
    DatasetClassification,
    RecordingMetadata,
    UtteranceRole,
    ValidationStatus,
)
from app.services.recording_storage import (
    RecordingStorage,
    StorageValidationError,
    TEST_SPEAKER_PREFIXES,
)


def create_pcm_wav_bytes(duration_seconds: float = 1.0, sample_rate: int = 16000) -> bytes:
    """Generates valid 16kHz 16-bit mono PCM WAV bytes."""
    num_samples = int(sample_rate * duration_seconds)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        samples = (0.2 * 32767.0 * np.sin(2 * np.pi * 440.0 * np.linspace(0, duration_seconds, num_samples, endpoint=False))).astype(np.int16)
        wf.writeframes(samples.tobytes())
    return buf.getvalue()


def test_missing_registry_file_fails_closed(tmp_path):
    """When trusted_speakers.json does not exist, all speakers must be untrusted."""
    storage_dir = tmp_path / "recordings_missing"
    storage = RecordingStorage(base_dir=storage_dir)

    assert not storage.trusted_speakers_file.exists()
    assert storage.list_trusted_speakers() == []
    assert storage.is_speaker_trusted("any_speaker") is False
    assert storage.is_speaker_trusted("spk_native_speaker") is False
    assert storage.is_speaker_trusted("") is False
    assert storage.is_speaker_trusted(None) is False


def test_corrupted_registry_file_fails_closed(tmp_path):
    """When trusted_speakers.json is corrupted or invalid JSON, registry must fail closed."""
    storage_dir = tmp_path / "recordings_corrupted"
    storage_dir.mkdir(parents=True, exist_ok=True)
    bad_file = storage_dir / "trusted_speakers.json"
    bad_file.write_text("{malformed_json: true,", encoding="utf-8")

    storage = RecordingStorage(base_dir=storage_dir)
    assert storage.list_trusted_speakers() == []
    assert storage.is_speaker_trusted("speaker_01") is False


def test_empty_registry_file_fails_closed(tmp_path):
    """When trusted_speakers.json exists but list is empty, all speakers must be untrusted."""
    storage_dir = tmp_path / "recordings_empty"
    storage_dir.mkdir(parents=True, exist_ok=True)
    empty_file = storage_dir / "trusted_speakers.json"
    empty_file.write_text(json.dumps({"speakers": []}), encoding="utf-8")

    storage = RecordingStorage(base_dir=storage_dir)
    assert storage.list_trusted_speakers() == []
    assert storage.is_speaker_trusted("speaker_01") is False


def test_unknown_speaker_when_registry_has_other_speakers(tmp_path):
    """Unknown speakers must not be trusted even when the registry contains other speakers."""
    storage_dir = tmp_path / "recordings_registry"
    storage = RecordingStorage(base_dir=storage_dir)

    storage.register_trusted_speaker("spk_trusted_alice")
    assert storage.is_speaker_trusted("spk_trusted_alice") is True
    assert storage.is_speaker_trusted("spk_untrusted_bob") is False
    assert storage.is_speaker_trusted("spk_random_user") is False
    assert "spk_trusted_alice" in storage.list_trusted_speakers()


def test_test_and_mock_speaker_prefixes_strictly_rejected(tmp_path):
    """Test and mock speaker prefixes must never qualify as trusted human speakers."""
    storage_dir = tmp_path / "recordings_prefixes"
    storage = RecordingStorage(base_dir=storage_dir)

    test_speakers = [
        "spk_mock_01",
        "spk_test_alpha",
        "spk_fail_negative",
        "spk_duration_test_10",
        "spk_acc_verify",
        "spk_isolated_env",
        "spk_enterprise_01",
        "spk_trans_sep_qa",
    ]

    for prefix in test_speakers:
        with pytest.raises(StorageValidationError, match="Cannot register test/mock speaker"):
            storage.register_trusted_speaker(prefix)
        assert storage.is_speaker_trusted(prefix) is False


def test_pair_verification_fails_closed_without_trusted_speakers(tmp_path):
    """Pair verification must fail closed if either or both speakers are untrusted."""
    storage_dir = tmp_path / "recordings_pair_gate"
    storage = RecordingStorage(base_dir=storage_dir)
    wav_bytes = create_pcm_wav_bytes(duration_seconds=1.0)
    pair_id = "human_gate_pair_01"

    # Save Utterance A (speaker_a) & Utterance B (speaker_b)
    storage.save_recording(
        raw_bytes=wav_bytes,
        original_filename="a.wav",
        speaker_id="spk_human_alice",
        language="hi-IN",
        category="quantities_and_numbers",
        reference_transcript="Do plate momos",
        verified_transcript="Do plate momos",
        is_transcript_verified=True,
        consent_given=True,
        consent_statement="Consent given.",
        pair_id=pair_id,
        utterance_role="utterance_a",
    )
    storage.save_recording(
        raw_bytes=wav_bytes,
        original_filename="b.wav",
        speaker_id="spk_human_bob",
        language="hi-IN",
        category="quantities_and_numbers",
        reference_transcript="Teen plate momos",
        verified_transcript="Teen plate momos",
        is_transcript_verified=True,
        consent_given=True,
        consent_statement="Consent given.",
        pair_id=pair_id,
        utterance_role="utterance_b",
    )

    # Neither speaker registered -> fails closed
    with pytest.raises(StorageValidationError, match="Both speakers must be registered in the trusted speaker registry"):
        storage.verify_contrast_pair(
            pair_id=pair_id,
            confirmed_semantic_distinction=True,
            distinction_field="quantity",
            expected_distinction_desc="2 vs 3 plates",
        )

    # Only Utterance A speaker registered -> still fails closed
    storage.register_trusted_speaker("spk_human_alice")
    with pytest.raises(StorageValidationError, match="Both speakers must be registered in the trusted speaker registry"):
        storage.verify_contrast_pair(
            pair_id=pair_id,
            confirmed_semantic_distinction=True,
            distinction_field="quantity",
            expected_distinction_desc="2 vs 3 plates",
        )

    # Both speakers registered -> verification succeeds and seals pair as human_verified
    storage.register_trusted_speaker("spk_human_bob")
    pair_rec = storage.verify_contrast_pair(
        pair_id=pair_id,
        confirmed_semantic_distinction=True,
        distinction_field="quantity",
        expected_distinction_desc="2 vs 3 plates",
    )
    assert pair_rec.is_verified_pair is True
    assert pair_rec.classification == DatasetClassification.HUMAN_VERIFIED.value


@pytest.mark.asyncio
async def test_trusted_speakers_api_validation(isolate_recording_storage, monkeypatch):
    """Verifies GET and POST /api/studio/trusted-speakers behavior and validation with administrative authorization."""
    from app.config import get_settings
    test_key = "test_adm_secret_key_8849"
    monkeypatch.setenv("ADMIN_API_KEY", test_key)
    get_settings.cache_clear()

    auth_headers = {"Authorization": f"Bearer {test_key}"}

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Initial empty state (public read)
        get_res = await client.get("/api/studio/trusted-speakers")
        assert get_res.status_code == 200
        assert get_res.json() == {"trusted_speakers": []}

        # Attempt to register without credentials -> 401 Unauthorized
        unauth_res = await client.post(
            "/api/studio/trusted-speakers",
            json={"speaker_id": "spk_authorized_human_10"},
        )
        assert unauth_res.status_code == 401

        # Attempt to register test speaker with valid admin credentials -> 400 Bad Request
        bad_res = await client.post(
            "/api/studio/trusted-speakers",
            json={"speaker_id": "spk_test_intruder"},
            headers=auth_headers,
        )
        assert bad_res.status_code == 400
        assert "Cannot register test/mock speaker" in bad_res.json()["detail"]

        # Register valid speaker with valid admin credentials -> 200 OK
        ok_res = await client.post(
            "/api/studio/trusted-speakers",
            json={"speaker_id": "spk_authorized_human_10"},
            headers=auth_headers,
        )
        assert ok_res.status_code == 200
        assert ok_res.json() == {"status": "registered", "speaker_id": "spk_authorized_human_10"}

        # Verify listed
        listed_res = await client.get("/api/studio/trusted-speakers")
        assert listed_res.status_code == 200
        assert "spk_authorized_human_10" in listed_res.json()["trusted_speakers"]
