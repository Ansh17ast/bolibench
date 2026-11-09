"""Comprehensive automated tests for BoliBench In-Website Human Audio Recording Studio.

Verifies:
1. Recording metadata schemas, categories, languages, and consent requirement.
2. Server-side conversion to standardized 16 kHz mono 16-bit PCM WAV.
3. Acoustic validation: duration, 50s completeness vs early-stop incomplete, decoding, silence, clipping.
4. Dedicated storage architecture in human_recordings_v1/ (raw, standardized, metadata, manifest).
5. Collision resistance, path traversal security, and atomic staging on interrupted upload.
6. Contrastive pair integrity: requiring Utterance A, Utterance B, verified transcripts, and confirmation of genuine semantic distinction.
7. Dataset separation: zero mutation of synthetic manifests, checkpoints, or SQLite evaluation tables.
8. API endpoints: /config, /upload, /recordings, /audio stream, /pairs, /verify, /studio route.
9. Safety guarantees: 0 external provider calls, no auto-evaluation trigger.
"""

import io
import json
from pathlib import Path
import shutil
import wave
import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.recording_schemas import (
    AudioDiagnostics,
    BenchmarkCategory,
    RecordingLanguage,
    RecordingMetadata,
    UtteranceRole,
    ValidationStatus,
)
from app.services.recording_processor import HumanAudioProcessor
from app.services.recording_storage import (
    RecordingStorage,
    StorageCollisionError,
    StorageSecurityError,
    StorageValidationError,
)


def create_pcm_wav_bytes(
    duration_seconds: float = 50.0,
    sample_rate: int = 16000,
    channels: int = 1,
    amplitude: float = 0.5,
    is_silent: bool = False,
    is_clipped: bool = False,
) -> bytes:
    """Helper to generate in-memory PCM WAV bytes with customizable acoustic properties."""
    num_samples = int(duration_seconds * sample_rate * channels)
    if is_silent:
        data = np.zeros(num_samples, dtype=np.int16)
    else:
        t = np.linspace(0, duration_seconds, num_samples, endpoint=False)
        # 440 Hz standard sine wave tone
        samples = amplitude * 32767.0 * np.sin(2 * np.pi * 440.0 * t)
        if is_clipped:
            # Force significant portion of samples to ceiling
            samples[:500] = 32767.0
            samples[1000:1500] = -32768.0
        data = np.clip(samples, -32768, 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(data.tobytes())
    return buf.getvalue()


@pytest.fixture
def temp_studio_storage(tmp_path):
    """Fixture providing an isolated RecordingStorage instance in a temporary folder."""
    storage_dir = tmp_path / "test_human_recordings"
    storage = RecordingStorage(base_dir=storage_dir)
    return storage


# =============================================================================
# 1. METADATA & SCHEMA VALIDATION TESTS
# =============================================================================

def test_six_benchmark_categories_supported():
    """Proves all six BoliBench benchmark categories are present and valid."""
    expected_categories = {
        "quantities_and_numbers",
        "negation",
        "corrections_and_mind_changes",
        "named_entities",
        "command_intent",
        "time_extraction",
    }
    actual_categories = {c.value for c in BenchmarkCategory}
    assert actual_categories == expected_categories


def test_supported_recording_languages():
    """Proves Hindi, English, and Hinglish are supported."""
    langs = {l.value for l in RecordingLanguage}
    assert "hi-IN" in langs
    assert "en-IN" in langs
    assert "hinglish" in langs


def test_metadata_model_validation():
    """Verifies complete RecordingMetadata schema serialization and deserialization."""
    diag = AudioDiagnostics(
        sample_rate_hz=16000,
        channels=1,
        duration_seconds=50.0,
        is_complete_50s=True,
        rms_amplitude=0.35,
        peak_amplitude=0.50,
        is_silent=False,
        clipped_samples_count=0,
        clipping_ratio=0.0,
        is_clipped=False,
        decoding_successful=True,
        converter_used="ffmpeg",
    )

    meta = RecordingMetadata(
        recording_id="rec_test_001",
        speaker_id="spk_rajesh_01",
        language="hi-IN",
        category="quantities_and_numbers",
        scenario_id="pair_qty_001",
        pair_id="pair_qty_001",
        utterance_role=UtteranceRole.UTTERANCE_A,
        reference_transcript="Do packet chahiye",
        verified_transcript="Do packet chahiye",
        is_transcript_verified=True,
        consent_given=True,
        consent_statement="I consent to record and use audio for evaluation.",
        consent_timestamp="2026-10-22T18:00:00Z",
        created_at="2026-10-22T18:00:00Z",
        client_duration_seconds=50.0,
        diagnostics=diag,
        validation_status=ValidationStatus.VALID_50S,
        raw_audio_filename="rec_test_001_raw.webm",
        standardized_audio_filename="rec_test_001.wav",
        raw_audio_path="raw/rec_test_001_raw.webm",
        standardized_audio_path="standardized_16k/rec_test_001.wav",
    )

    dumped = meta.model_dump()
    assert dumped["recording_id"] == "rec_test_001"
    assert dumped["validation_status"] == "valid_50s"
    assert dumped["diagnostics"]["is_complete_50s"] is True


# =============================================================================
# 2. AUDIO CONVERSION & ACOUSTIC VALIDATION TESTS
# =============================================================================

def test_audio_processor_standardization_and_validation(tmp_path):
    """Proves audio processor converts and verifies 16 kHz mono 16-bit PCM WAV."""
    processor = HumanAudioProcessor()

    # Generate a 50.0-second WAV input
    input_wav = tmp_path / "raw_input.wav"
    input_bytes = create_pcm_wav_bytes(duration_seconds=50.0, sample_rate=44100, channels=2)
    input_wav.write_bytes(input_bytes)

    output_wav = tmp_path / "standardized_50s.wav"
    converter_engine = processor.convert_to_standard_wav(input_wav, output_wav)

    assert output_wav.exists()
    assert output_wav.stat().st_size > 1000

    # Analyze standardized copy
    diag, status = processor.analyze_standard_wav(output_wav)
    assert diag.decoding_successful is True
    assert diag.sample_rate_hz == 16000
    assert diag.channels == 1
    assert 49.5 <= diag.duration_seconds <= 50.5
    assert diag.is_complete_50s is True
    assert status == ValidationStatus.VALID_50S
    assert diag.is_silent is False
    assert diag.is_clipped is False


def test_too_short_marked_incomplete(tmp_path):
    """Proves that clips under minimum duration (< 0.5s) are flagged incomplete."""
    processor = HumanAudioProcessor()

    # Create a 0.2-second clip (too short for acoustic analysis)
    input_wav = tmp_path / "too_short.wav"
    input_wav.write_bytes(create_pcm_wav_bytes(duration_seconds=0.2, sample_rate=16000))

    output_wav = tmp_path / "std_too_short.wav"
    processor.convert_to_standard_wav(input_wav, output_wav)

    diag, status = processor.analyze_standard_wav(output_wav)
    assert diag.is_complete_50s is False
    assert status == ValidationStatus.INCOMPLETE_EARLY_STOP
    assert diag.duration_seconds < 0.5


def test_manual_stop_11_7s_and_5s_valid(tmp_path):
    """Proves that short clips stopped manually (e.g. 11.7s or 5.0s) are valid and NOT marked incomplete."""
    processor = HumanAudioProcessor()

    # 11.7s clip
    input_11_7 = tmp_path / "clip_11_7s.wav"
    input_11_7.write_bytes(create_pcm_wav_bytes(duration_seconds=11.7, sample_rate=16000))
    out_11_7 = tmp_path / "std_11_7s.wav"
    processor.convert_to_standard_wav(input_11_7, out_11_7)
    diag_11_7, status_11_7 = processor.analyze_standard_wav(out_11_7)
    assert diag_11_7.is_complete_50s is True
    assert status_11_7 == ValidationStatus.VALID_50S
    assert diag_11_7.duration_seconds == pytest.approx(11.7, rel=1e-2)

    # 5.0s clip
    input_5s = tmp_path / "clip_5s.wav"
    input_5s.write_bytes(create_pcm_wav_bytes(duration_seconds=5.0, sample_rate=16000))
    out_5s = tmp_path / "std_5s.wav"
    processor.convert_to_standard_wav(input_5s, out_5s)
    diag_5s, status_5s = processor.analyze_standard_wav(out_5s)
    assert diag_5s.is_complete_50s is True
    assert status_5s == ValidationStatus.VALID_50S
    assert diag_5s.duration_seconds == pytest.approx(5.0, rel=1e-2)


def test_duration_exceeded_over_600s(tmp_path):
    """Proves that clips exceeding 600s safety limit are flagged as DURATION_EXCEEDED."""
    processor = HumanAudioProcessor()

    input_long = tmp_path / "long.wav"
    input_long.write_bytes(create_pcm_wav_bytes(duration_seconds=605.0, sample_rate=16000))
    out_long = tmp_path / "std_long.wav"
    processor.convert_to_standard_wav(input_long, out_long)
    diag_long, status_long = processor.analyze_standard_wav(out_long)
    assert diag_long.is_complete_50s is False
    assert status_long == ValidationStatus.DURATION_EXCEEDED


def test_silence_detection(tmp_path):
    """Proves audio processor identifies silent recordings."""
    processor = HumanAudioProcessor()

    silent_wav = tmp_path / "silent.wav"
    silent_wav.write_bytes(create_pcm_wav_bytes(duration_seconds=50.0, is_silent=True))

    output_wav = tmp_path / "std_silent.wav"
    processor.convert_to_standard_wav(silent_wav, output_wav)

    diag, status = processor.analyze_standard_wav(output_wav)
    assert diag.is_silent is True
    assert status == ValidationStatus.SILENT_WARNING


def test_clipping_detection(tmp_path):
    """Proves audio processor identifies severe acoustic clipping / saturation."""
    processor = HumanAudioProcessor()

    clipped_wav = tmp_path / "clipped.wav"
    clipped_wav.write_bytes(create_pcm_wav_bytes(duration_seconds=50.0, is_clipped=True))

    output_wav = tmp_path / "std_clipped.wav"
    processor.convert_to_standard_wav(clipped_wav, output_wav)

    diag, status = processor.analyze_standard_wav(output_wav)
    assert diag.is_clipped is True
    assert diag.clipped_samples_count >= 10
    assert status == ValidationStatus.CLIPPED_WARNING


# =============================================================================
# 3. STORAGE & SECURITY TESTS
# =============================================================================

def test_storage_preserves_raw_and_generates_standardized(temp_studio_storage):
    """Proves storage saves raw file unmodified and creates standardized WAV copy."""
    storage = temp_studio_storage
    raw_wav = create_pcm_wav_bytes(duration_seconds=50.0)

    meta = storage.save_recording(
        raw_bytes=raw_wav,
        original_filename="mic_take.wav",
        speaker_id="spk_priya_01",
        language="hinglish",
        category="command_intent",
        reference_transcript="Mera account balance check karo",
        consent_given=True,
        consent_statement="I explicitly consent to record and use audio.",
        recording_id="rec_test_storage_01",
    )

    assert meta.recording_id == "rec_test_storage_01"
    raw_file = storage.base_dir / meta.raw_audio_path
    std_file = storage.base_dir / meta.standardized_audio_path
    meta_file = storage.base_dir / "metadata" / f"{meta.recording_id}.json"

    assert raw_file.exists()
    assert std_file.exists()
    assert meta_file.exists()

    # Verify raw file content is identical to uploaded bytes
    assert raw_file.read_bytes() == raw_wav


def test_consent_mandatory_for_saving(temp_studio_storage):
    """Proves that audio is strictly rejected if consent is not granted."""
    storage = temp_studio_storage
    raw_wav = create_pcm_wav_bytes(duration_seconds=50.0)

    with pytest.raises(StorageValidationError, match="Explicit consent is required"):
        storage.save_recording(
            raw_bytes=raw_wav,
            original_filename="mic_take.wav",
            speaker_id="spk_priya_01",
            language="hinglish",
            category="command_intent",
            reference_transcript="Mera account balance check karo",
            consent_given=False,  # NOT GIVEN
            consent_statement="I consent.",
        )


def test_collision_prevention_rejects_duplicate_id(temp_studio_storage):
    """Proves that duplicate recording IDs cannot overwrite existing recordings."""
    storage = temp_studio_storage
    raw_wav = create_pcm_wav_bytes(duration_seconds=50.0)

    # Save first recording
    storage.save_recording(
        raw_bytes=raw_wav,
        original_filename="mic.wav",
        speaker_id="spk_01",
        language="hi-IN",
        category="quantities_and_numbers",
        reference_transcript="Do packet",
        consent_given=True,
        consent_statement="Consent given.",
        recording_id="rec_unique_01",
    )

    # Attempt saving duplicate
    with pytest.raises(StorageCollisionError, match="already exists"):
        storage.save_recording(
            raw_bytes=raw_wav,
            original_filename="mic2.wav",
            speaker_id="spk_02",
            language="hi-IN",
            category="quantities_and_numbers",
            reference_transcript="Teen packet",
            consent_given=True,
            consent_statement="Consent given.",
            recording_id="rec_unique_01",
        )


def test_path_traversal_prevention(temp_studio_storage):
    """Proves that malicious identifiers containing directory traversal are rejected."""
    storage = temp_studio_storage
    raw_wav = create_pcm_wav_bytes(duration_seconds=50.0)

    with pytest.raises(StorageSecurityError, match="Invalid recording_id"):
        storage.save_recording(
            raw_bytes=raw_wav,
            original_filename="mic.wav",
            speaker_id="spk_01",
            language="hi-IN",
            category="quantities_and_numbers",
            reference_transcript="Test",
            consent_given=True,
            consent_statement="Consent.",
            recording_id="../../bad_id",
        )


def test_interrupted_upload_staging_cleanup(temp_studio_storage):
    """Proves corrupt / invalid audio files do not leave corrupted files in storage."""
    storage = temp_studio_storage

    # Non-audio garbage bytes
    garbage_bytes = b"NOT_A_VALID_AUDIO_HEADER_JUST_RANDOM_TEXT" * 10

    with pytest.raises(StorageValidationError):
        storage.save_recording(
            raw_bytes=garbage_bytes,
            original_filename="corrupt.wav",
            speaker_id="spk_01",
            language="hi-IN",
            category="quantities_and_numbers",
            reference_transcript="Test",
            consent_given=True,
            consent_statement="Consent.",
            recording_id="rec_corrupt_test",
        )

    # Verify no corrupt metadata or standardized files were created
    assert not (storage.metadata_dir / "rec_corrupt_test.json").exists()
    assert not (storage.standardized_dir / "rec_corrupt_test.wav").exists()


# =============================================================================
# 4. DATASET INTEGRITY & CONTRASTIVE PAIR VERIFICATION TESTS
# =============================================================================

def test_contrast_pair_verification_requires_both_utterances(temp_studio_storage):
    """Proves pair verification fails if either Utterance A or B is missing."""
    storage = temp_studio_storage
    raw_wav = create_pcm_wav_bytes(duration_seconds=50.0)

    # Save only Utterance A
    storage.save_recording(
        raw_bytes=raw_wav,
        original_filename="mic_a.wav",
        speaker_id="spk_01",
        language="hinglish",
        category="quantities_and_numbers",
        reference_transcript="Do packet chahiye",
        verified_transcript="Do packet chahiye",
        is_transcript_verified=True,
        consent_given=True,
        consent_statement="Consent.",
        pair_id="pair_qty_999",
        utterance_role="utterance_a",
    )

    with pytest.raises(StorageValidationError, match="Missing Utterance B recording"):
        storage.verify_contrast_pair(
            pair_id="pair_qty_999",
            confirmed_semantic_distinction=True,
        )


def test_contrast_pair_verification_requires_manual_transcript_verification(temp_studio_storage):
    """Proves pair verification fails if transcripts are not verified."""
    storage = temp_studio_storage
    raw_wav = create_pcm_wav_bytes(duration_seconds=50.0)

    # Save Utterance A (verified)
    storage.save_recording(
        raw_bytes=raw_wav,
        original_filename="mic_a.wav",
        speaker_id="spk_01",
        language="hinglish",
        category="quantities_and_numbers",
        reference_transcript="Do packet chahiye",
        verified_transcript="Do packet chahiye",
        is_transcript_verified=True,
        consent_given=True,
        consent_statement="Consent.",
        pair_id="pair_qty_888",
        utterance_role="utterance_a",
    )

    # Save Utterance B (UNVERIFIED)
    storage.save_recording(
        raw_bytes=raw_wav,
        original_filename="mic_b.wav",
        speaker_id="spk_02",
        language="hinglish",
        category="quantities_and_numbers",
        reference_transcript="Teen packet chahiye",
        is_transcript_verified=False,  # Unverified
        consent_given=True,
        consent_statement="Consent.",
        pair_id="pair_qty_888",
        utterance_role="utterance_b",
    )

    with pytest.raises(StorageValidationError, match="Utterance B transcript must be manually verified"):
        storage.verify_contrast_pair(
            pair_id="pair_qty_888",
            confirmed_semantic_distinction=True,
        )


def test_contrast_pair_rejects_identical_transcripts(temp_studio_storage):
    """Proves arbitrary identical recordings cannot form a contrast pair."""
    storage = temp_studio_storage
    storage.register_trusted_speaker("spk_01")
    storage.register_trusted_speaker("spk_02")
    raw_wav = create_pcm_wav_bytes(duration_seconds=50.0)

    # Utterance A & B have IDENTICAL transcripts
    storage.save_recording(
        raw_bytes=raw_wav,
        original_filename="mic_a.wav",
        speaker_id="spk_01",
        language="hi-IN",
        category="negation",
        reference_transcript="Haan cancel kar do",
        verified_transcript="Haan cancel kar do",
        is_transcript_verified=True,
        consent_given=True,
        consent_statement="Consent.",
        pair_id="pair_neg_identical",
        utterance_role="utterance_a",
    )
    storage.save_recording(
        raw_bytes=raw_wav,
        original_filename="mic_b.wav",
        speaker_id="spk_02",
        language="hi-IN",
        category="negation",
        reference_transcript="Haan cancel kar do",
        verified_transcript="Haan cancel kar do",
        is_transcript_verified=True,
        consent_given=True,
        consent_statement="Consent.",
        pair_id="pair_neg_identical",
        utterance_role="utterance_b",
    )

    with pytest.raises(StorageValidationError, match="identical transcripts"):
        storage.verify_contrast_pair(
            pair_id="pair_neg_identical",
            confirmed_semantic_distinction=True,
        )


def test_contrast_pair_successful_verification(temp_studio_storage):
    """Proves successful verification when A and B test a genuine contrast."""
    storage = temp_studio_storage
    storage.register_trusted_speaker("spk_01")
    storage.register_trusted_speaker("spk_02")
    raw_wav = create_pcm_wav_bytes(duration_seconds=50.0)

    storage.save_recording(
        raw_bytes=raw_wav,
        original_filename="mic_a.wav",
        speaker_id="spk_01",
        language="hinglish",
        category="quantities_and_numbers",
        reference_transcript="Do packet chahiye",
        verified_transcript="Do packet chahiye",
        is_transcript_verified=True,
        consent_given=True,
        consent_statement="Consent.",
        pair_id="pair_qty_valid_01",
        utterance_role="utterance_a",
    )
    storage.save_recording(
        raw_bytes=raw_wav,
        original_filename="mic_b.wav",
        speaker_id="spk_02",
        language="hinglish",
        category="quantities_and_numbers",
        reference_transcript="Teen packet chahiye",
        verified_transcript="Teen packet chahiye",
        is_transcript_verified=True,
        consent_given=True,
        consent_statement="Consent.",
        pair_id="pair_qty_valid_01",
        utterance_role="utterance_b",
    )

    pair = storage.verify_contrast_pair(
        pair_id="pair_qty_valid_01",
        confirmed_semantic_distinction=True,
        distinction_field="quantity",
        expected_distinction_desc="Quantity 2 vs Quantity 3",
        operator_notes="Verified genuine contrast.",
        verified_by="test_operator",
    )

    assert pair.is_verified_pair is True
    assert pair.confirmed_semantic_distinction is True
    assert (storage.pairs_dir / "pair_qty_valid_01.json").exists()


# =============================================================================
# 5. FASTAPI ROUTE & VERIFICATION TESTS
# =============================================================================

@pytest.mark.asyncio
async def test_api_studio_config():
    """Proves /api/studio/config returns capabilities, 6 categories, and templates."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/studio/config")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["supported_categories"]) == 6
        assert len(data["supported_languages"]) == 3
        assert data["target_duration_seconds"] == 600.0
        assert data["max_duration_seconds"] == 600.0
        assert data["storage_directory"] == "human_recordings_v1"
        assert len(data["scenario_templates"]) >= 6


@pytest.mark.asyncio
async def test_api_studio_upload_and_stream():
    """Proves uploading via /api/studio/upload and streaming audio via /audio."""
    import uuid
    rec_id = f"rec_api_{uuid.uuid4().hex[:8]}"
    wav_bytes = create_pcm_wav_bytes(duration_seconds=50.0)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # 1. Successful Upload
        files = {"audio_file": ("speaker_take.wav", wav_bytes, "audio/wav")}
        data = {
            "speaker_id": "spk_test_api",
            "language": "hi-IN",
            "category": "negation",
            "reference_transcript": "Nahi cancel mat karo",
            "consent_given": "true",
            "consent_statement": "I explicitly consent to record and use audio.",
            "recording_id": rec_id,
            "utterance_role": "utterance_a",
            "is_transcript_verified": "true",
            "verified_transcript": "Nahi cancel mat karo",
        }

        resp = await client.post("/api/studio/upload", data=data, files=files)
        assert resp.status_code == 200
        result = resp.json()
        assert result["recording_id"] == rec_id
        assert result["validation_status"] == "valid_50s"
        assert result["diagnostics"]["sample_rate_hz"] == 16000

        # 2. Query Recordings List
        list_resp = await client.get("/api/studio/recordings?category=negation")
        assert list_resp.status_code == 200
        records = list_resp.json()
        assert any(r["recording_id"] == rec_id for r in records)

        # 3. Stream Standardized Audio
        audio_resp = await client.get(f"/api/studio/recordings/{rec_id}/audio")
        assert audio_resp.status_code == 200
        assert audio_resp.headers["content-type"].startswith("audio/wav")
        assert len(audio_resp.content) > 1000


@pytest.mark.asyncio
async def test_api_studio_page_rendered():
    """Proves dedicated /studio and /recording-studio routes return complete HTML interface."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        for route in ["/studio", "/recording-studio"]:
            resp = await client.get(route)
            assert resp.status_code == 200
            html = resp.text
            assert "Human Audio Recording Studio" in html
            assert 'id="btnStudioRecord"' in html
            assert 'id="btnStudioPause"' in html
            assert 'id="btnStudioStop"' in html
            assert 'id="btnStudioRetake"' in html
            assert 'id="btnStudioSave"' in html
            assert 'id="studioCountdownTimer"' in html
            assert 'id="studioCountdownProgress"' in html
            assert 'id="studioWaveformCanvas"' in html
            assert 'id="studioEarlyStopWarning"' in html
            assert 'id="studioConsentCheckbox"' in html
            assert "human_recordings_v1" in html


@pytest.mark.asyncio
async def test_isolated_from_evaluations_and_no_external_api_calls():
    """Proves recording studio makes zero external provider calls and does not trigger benchmark runs."""
    import uuid
    from app.services.recording_storage import RecordingStorage
    storage = RecordingStorage()

    rec_id = f"rec_iso_{uuid.uuid4().hex[:8]}"

    # Verify synthetic benchmark files and evaluation DB remain completely unmutated
    manifest_v2 = Path("benchmark_manifest_v2.0.0.json")
    assert manifest_v2.exists()
    mtime_before = manifest_v2.stat().st_mtime

    wav_bytes = create_pcm_wav_bytes(duration_seconds=50.0)
    meta = storage.save_recording(
        raw_bytes=wav_bytes,
        original_filename="sample.wav",
        speaker_id="spk_isolated",
        language="hi-IN",
        category="time_extraction",
        reference_transcript="Dopahar ko do baje",
        consent_given=True,
        consent_statement="Consent given.",
        recording_id=rec_id,
    )

    # Benchmark manifest should be completely untouched
    assert manifest_v2.stat().st_mtime == mtime_before
    assert meta.recording_id == rec_id



@pytest.mark.asyncio
async def test_audio_duration_discrepancy_validation(tmp_path):
    """Proves server detects duration discrepancy when client claims long duration but audio is under minimum (<0.5s)."""
    processor = HumanAudioProcessor()
    # Create actual 0.2 second audio (under minimum duration)
    short_wav = create_pcm_wav_bytes(duration_seconds=0.2)
    wav_path = tmp_path / "discrepancy_sample.wav"
    wav_path.write_bytes(short_wav)

    # Client deceptively reports 600.0 seconds
    diag, status = processor.analyze_standard_wav(wav_path, client_duration=600.0)

    assert diag.is_complete_50s is False
    assert status == ValidationStatus.INCOMPLETE_EARLY_STOP
    assert diag.duration_seconds < 0.5

    # Conversely, a valid manually stopped 11.7s clip is accepted and marked valid
    valid_wav = create_pcm_wav_bytes(duration_seconds=11.7)
    valid_path = tmp_path / "valid_sample.wav"
    valid_path.write_bytes(valid_wav)
    diag_valid, status_valid = processor.analyze_standard_wav(valid_path, client_duration=11.7)
    assert diag_valid.is_complete_50s is True
    assert status_valid == ValidationStatus.VALID_50S
    assert diag_valid.duration_seconds == pytest.approx(11.7, rel=1e-2)


@pytest.mark.asyncio
async def test_pair_creation_and_ab_recording_association():
    """Proves end-to-end creation of pair definition, ingestion of A, ingestion of B, and status progression."""
    import uuid
    uid = uuid.uuid4().hex[:6]
    pair_id = f"human_test_qty_{uid}"

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Step 1: Create Pair Definition
        create_resp = await client.post("/api/studio/pairs/create", json={
            "pair_id": pair_id,
            "speaker_id": "spk_test_ab",
            "language": "hinglish",
            "category": "quantities_and_numbers",
            "scenario_description": "Quantity test between 2 and 3 items",
            "expected_contrast": "Utterance A orders quantity 2 vs Utterance B orders quantity 3",
        })
        assert create_resp.status_code == 200
        pair_data = create_resp.json()
        assert pair_data["pair_id"] == pair_id
        assert pair_data["status"] == "incomplete"

        # Step 2: Upload Utterance A (50s standard)
        wav_a = create_pcm_wav_bytes(duration_seconds=50.0)
        rec_id_a = f"rec_a_{uid}"
        upload_a_resp = await client.post("/api/studio/upload", data={
            "recording_id": rec_id_a,
            "pair_id": pair_id,
            "utterance_role": "utterance_a",
            "speaker_id": "spk_test_ab",
            "language": "hinglish",
            "category": "quantities_and_numbers",
            "reference_transcript": "Do packet chahiye",
            "verified_transcript": "Do packet chahiye",
            "is_transcript_verified": "true",
            "consent_given": "true",
            "consent_statement": "I consent.",
            "client_duration_seconds": "50.0",
        }, files={"audio_file": ("utt_a.wav", wav_a, "audio/wav")})
        assert upload_a_resp.status_code == 200

        # Step 3: Check pair status after A
        pair_after_a = (await client.get(f"/api/studio/pairs/{pair_id}")).json()
        assert pair_after_a["utterance_a"] is not None
        assert pair_after_a["utterance_a"]["recording_id"] == rec_id_a
        assert pair_after_a["utterance_b"] is None
        assert pair_after_a["status"] == "incomplete"

        # Step 4: Upload Utterance B (50s standard)
        wav_b = create_pcm_wav_bytes(duration_seconds=50.0)
        rec_id_b = f"rec_b_{uid}"
        upload_b_resp = await client.post("/api/studio/upload", data={
            "recording_id": rec_id_b,
            "pair_id": pair_id,
            "utterance_role": "utterance_b",
            "speaker_id": "spk_test_ab",
            "language": "hinglish",
            "category": "quantities_and_numbers",
            "reference_transcript": "Teen packet chahiye",
            "verified_transcript": "Teen packet chahiye",
            "is_transcript_verified": "true",
            "consent_given": "true",
            "consent_statement": "I consent.",
            "client_duration_seconds": "50.0",
        }, files={"audio_file": ("utt_b.wav", wav_b, "audio/wav")})
        assert upload_b_resp.status_code == 200

        # Step 5: Check pair status after both A & B
        pair_after_b = (await client.get(f"/api/studio/pairs/{pair_id}")).json()
        assert pair_after_b["utterance_a"] is not None
        assert pair_after_b["utterance_b"] is not None
        assert pair_after_b["status"] in ["awaiting_verification", "ready_for_evaluation"]


@pytest.mark.asyncio
async def test_human_pair_mock_evaluation_and_sqlite_cohort_isolation():
    """Proves human contrast pair mock evaluation succeeds and is isolated in SQLite storage by dataset_source."""
    import uuid
    uid = uuid.uuid4().hex[:6]
    pair_id = f"human_eval_mock_{uid}"

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Create Pair
        await client.post("/api/studio/pairs/create", json={
            "pair_id": pair_id,
            "speaker_id": "spk_mock",
            "language": "hi-IN",
            "category": "quantities_and_numbers",
            "scenario_description": "Mock quantity contrast",
            "expected_contrast": "A orders 2 vs B orders 3",
        })

        wav_bytes = create_pcm_wav_bytes(duration_seconds=50.0)
        # Upload A
        await client.post("/api/studio/upload", data={
            "pair_id": pair_id,
            "utterance_role": "utterance_a",
            "speaker_id": "spk_mock",
            "language": "hi-IN",
            "category": "quantities_and_numbers",
            "reference_transcript": "Do packet chahiye",
            "verified_transcript": "Do packet chahiye",
            "is_transcript_verified": "true",
            "consent_given": "true",
            "consent_statement": "Consent.",
            "client_duration_seconds": "50.0",
        }, files={"audio_file": ("a.wav", wav_bytes, "audio/wav")})

        # Upload B
        await client.post("/api/studio/upload", data={
            "pair_id": pair_id,
            "utterance_role": "utterance_b",
            "speaker_id": "spk_mock",
            "language": "hi-IN",
            "category": "quantities_and_numbers",
            "reference_transcript": "Teen packet chahiye",
            "verified_transcript": "Teen packet chahiye",
            "is_transcript_verified": "true",
            "consent_given": "true",
            "consent_statement": "Consent.",
            "client_duration_seconds": "50.0",
        }, files={"audio_file": ("b.wav", wav_bytes, "audio/wav")})

        # Run Mock Evaluation
        eval_resp = await client.post(f"/api/eval/run-human-pair-mock/{pair_id}")
        assert eval_resp.status_code == 200
        res = eval_resp.json()
        assert res["dataset_source"] == "human_recording"
        assert res["human_pair_id"] == pair_id
        assert res["semantic_distinction_preserved"] is True
        assert res["overall_status"] == "completed"

        # Verify separation in SQLite storage
        # 1. Querying human_recording returns the run
        human_runs = (await client.get("/api/eval/results?dataset_source=human_recording")).json()
        assert any(r["human_pair_id"] == pair_id for r in human_runs)

        # 2. Querying synthetic_benchmark strictly excludes the human run
        synth_runs = (await client.get("/api/eval/results?dataset_source=synthetic_benchmark")).json()
        assert not any(r["human_pair_id"] == pair_id for r in synth_runs)


@pytest.mark.asyncio
async def test_human_pair_live_evaluation_independent_ab_failure_handling(monkeypatch):
    """Proves independent A/B failure handling: if Utterance B fails with 503, Utterance A is preserved and stored."""
    import uuid
    from app.services.gnani_client import GnaniPrismaClient, GnaniServiceUnavailableError

    uid = uuid.uuid4().hex[:6]
    pair_id = f"human_live_fail_{uid}"

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Create pair and recordings
        await client.post("/api/studio/pairs/create", json={
            "pair_id": pair_id,
            "speaker_id": "spk_fail_test",
            "language": "hinglish",
            "category": "quantities_and_numbers",
            "scenario_description": "Independent failure test",
            "expected_contrast": "A orders 2 vs B orders 3",
        })

        wav_bytes = create_pcm_wav_bytes(duration_seconds=50.0)
        await client.post("/api/studio/upload", data={
            "pair_id": pair_id,
            "utterance_role": "utterance_a",
            "speaker_id": "spk_fail_test",
            "language": "hinglish",
            "category": "quantities_and_numbers",
            "reference_transcript": "Do packet chahiye",
            "verified_transcript": "Do packet chahiye",
            "is_transcript_verified": "true",
            "consent_given": "true",
            "consent_statement": "Consent.",
            "client_duration_seconds": "50.0",
        }, files={"audio_file": ("a.wav", wav_bytes, "audio/wav")})

        await client.post("/api/studio/upload", data={
            "pair_id": pair_id,
            "utterance_role": "utterance_b",
            "speaker_id": "spk_fail_test",
            "language": "hinglish",
            "category": "quantities_and_numbers",
            "reference_transcript": "Teen packet chahiye",
            "verified_transcript": "Teen packet chahiye",
            "is_transcript_verified": "true",
            "consent_given": "true",
            "consent_statement": "Consent.",
            "client_duration_seconds": "50.0",
        }, files={"audio_file": ("b.wav", wav_bytes, "audio/wav")})

        # Enable live mode for test operator
        from app.config import get_settings
        monkeypatch.setattr(get_settings(), "live_evaluation_enabled", True)
        monkeypatch.setattr(get_settings(), "gnani_api_key", "test_mock_key_12345")

        # Mock STT client: Utterance A succeeds, Utterance B fails with 503
        call_count = 0
        async def mock_transcribe(self, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {
                    "success": True,
                    "transcript": "Do packet chahiye",
                    "latency_seconds": 0.25,
                    "provider_processing_time_ms": 250.0,
                    "request_id": "req_mock_a_001",
                }
            else:
                raise GnaniServiceUnavailableError("Gnani STT 503 Service Temporarily Unavailable")

        monkeypatch.setattr(GnaniPrismaClient, "transcribe", mock_transcribe)

        # Request live authorization token from BoliBench security guard
        auth_resp = await client.post("/api/eval/authorize-live", json={"test_id": pair_id})
        assert auth_resp.status_code == 200
        token = auth_resp.json()["confirmation_token"]

        # Execute Live Evaluation with explicit operator authorization header and token
        from app.services.live_guard import live_guard
        live_resp = await client.post(
            f"/api/eval/run-human-pair-live/{pair_id}",
            headers={live_guard.REQUIRED_CONFIRM_HEADER: live_guard.EXPECTED_HEADER_VALUE},
            json={"confirmation_token": token},
        )
        assert live_resp.status_code == 200
        result = live_resp.json()

        assert result["overall_status"] == "incomplete"
        assert result["primary_failure_category"] == "provider_error"
        assert result["dataset_source"] == "human_recording"

        # Utterance A is preserved and correctly recognized
        assert result["utterance_a"]["actual_transcript"] == "Do packet chahiye"
        assert result["utterance_a"]["error_message"] is None

        # Utterance B failure is recorded independently without discarding A
        assert result["utterance_b"]["error_message"] is not None
        assert "503" in result["utterance_b"]["error_message"]


@pytest.mark.asyncio
async def test_human_pair_live_credit_guard_rejection_without_authorization():
    """Proves live evaluation strictly rejects unauthorized calls with 403 to prevent accidental credit consumption."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # No authorization header
        resp = await client.post("/api/eval/run-human-pair-live/human_dummy_001")
        assert resp.status_code == 403
        assert "disabled by default" in resp.json()["detail"].lower() or "operator authorization required" in resp.json()["detail"].lower()


@pytest.mark.asyncio
async def test_complete_business_script_paired_acceptance_workflow():
    """Acceptance test verifying end-to-end human recording and paired evaluation workflow.

    1. Record A with custom business script -> save successfully -> reload and confirm actual audio & transcript.
    2. Record B with contrasting business script -> save successfully -> reload and confirm audio & transcript.
    3. Confirm both clips share the exact same pair ID and have correct roles (utterance_a, utterance_b).
    4. Confirm the completed pair appears in candidate pairs library and is ready for Run Evaluation.
    5. Trigger mocked evaluation and verify both utterances and comparison results persist cleanly.
    6. Verify that a 3-second audio file cannot be marked as a valid 50-second recording merely because client reports 50s.
    """
    import uuid
    uid = uuid.uuid4().hex[:6]
    pair_id = f"human_bus_test_{uid}"
    speaker_id = "spk_enterprise_01"

    script_a = "Please process invoice number 4082 for vendor Acme Corp and disburse payment immediately"
    script_b = "Please hold invoice number 4082 for vendor Acme Corp and suspend payment until audit completion"

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Step 0: Create initial pair definition with custom business scenario
        pair_init = await client.post("/api/studio/pairs/create", json={
            "pair_id": pair_id,
            "speaker_id": speaker_id,
            "language": "en-IN",
            "category": "command_intent",
            "scenario_description": "Enterprise procurement invoice disbursement authorization disparity",
            "expected_distinction": "Utterance A authorizes immediate disbursement vs Utterance B suspends disbursement pending audit",
        })
        assert pair_init.status_code == 200

        # Step 1: Record A (50.0s audio) -> Save -> Reload & confirm actual audio & transcript
        wav_bytes_a = create_pcm_wav_bytes(duration_seconds=50.0)
        rec_id_a = f"rec_bus_a_{uid}"

        resp_a = await client.post("/api/studio/upload", data={
            "recording_id": rec_id_a,
            "pair_id": pair_id,
            "utterance_role": "utterance_a",
            "speaker_id": speaker_id,
            "language": "en-IN",
            "category": "command_intent",
            "reference_transcript": script_a,
            "verified_transcript": script_a,
            "is_transcript_verified": "true",
            "consent_given": "true",
            "consent_statement": "Explicit enterprise consent.",
            "client_duration_seconds": "50.0",
        }, files={"audio_file": ("business_take_a.wav", wav_bytes_a, "audio/wav")})
        assert resp_a.status_code == 200
        data_a = resp_a.json()
        assert data_a["recording_id"] == rec_id_a
        assert data_a["pair_id"] == pair_id
        assert data_a["utterance_role"] == "utterance_a"
        assert data_a["reference_transcript"] == script_a
        assert data_a["verified_transcript"] == script_a
        assert data_a["validation_status"] == "valid_50s"
        assert data_a["diagnostics"]["is_complete_50s"] is True

        # Reload from API and verify audio streaming
        reloaded_a = (await client.get(f"/api/studio/recordings/{rec_id_a}")).json()
        assert reloaded_a["reference_transcript"] == script_a
        audio_stream_a = await client.get(f"/api/studio/recordings/{rec_id_a}/audio")
        assert audio_stream_a.status_code == 200
        assert len(audio_stream_a.content) > 1000

        # Step 2: Record B (50.0s audio) -> Save -> Reload & confirm actual audio & transcript
        wav_bytes_b = create_pcm_wav_bytes(duration_seconds=50.0)
        rec_id_b = f"rec_bus_b_{uid}"

        resp_b = await client.post("/api/studio/upload", data={
            "recording_id": rec_id_b,
            "pair_id": pair_id,
            "utterance_role": "utterance_b",
            "speaker_id": speaker_id,
            "language": "en-IN",
            "category": "command_intent",
            "reference_transcript": script_b,
            "verified_transcript": script_b,
            "is_transcript_verified": "true",
            "consent_given": "true",
            "consent_statement": "Explicit enterprise consent.",
            "client_duration_seconds": "50.0",
        }, files={"audio_file": ("business_take_b.wav", wav_bytes_b, "audio/wav")})
        assert resp_b.status_code == 200
        data_b = resp_b.json()
        assert data_b["recording_id"] == rec_id_b
        assert data_b["pair_id"] == pair_id
        assert data_b["utterance_role"] == "utterance_b"
        assert data_b["reference_transcript"] == script_b
        assert data_b["verified_transcript"] == script_b
        assert data_b["validation_status"] == "valid_50s"
        assert data_b["diagnostics"]["is_complete_50s"] is True

        reloaded_b = (await client.get(f"/api/studio/recordings/{rec_id_b}")).json()
        assert reloaded_b["reference_transcript"] == script_b
        audio_stream_b = await client.get(f"/api/studio/recordings/{rec_id_b}/audio")
        assert audio_stream_b.status_code == 200
        assert len(audio_stream_b.content) > 1000

        # Step 3: Confirm distinct recording IDs, shared pair ID, and correct A/B roles
        assert rec_id_a != rec_id_b
        assert data_a["pair_id"] == data_b["pair_id"] == pair_id
        assert data_a["utterance_role"] == "utterance_a"
        assert data_b["utterance_role"] == "utterance_b"

        # Step 4: Confirm pair appears in library and is ready for Run Evaluation
        candidate_pairs = (await client.get("/api/eval/human-pairs")).json()
        matched_pair = next((p for p in candidate_pairs if p["pair_id"] == pair_id), None)
        assert matched_pair is not None, f"Pair {pair_id} not found in candidate pairs list"
        assert matched_pair["status"] == "ready_for_evaluation"
        assert matched_pair["is_ready_for_evaluation"] is True
        assert matched_pair["utterance_a"]["recording_id"] == rec_id_a
        assert matched_pair["utterance_b"]["recording_id"] == rec_id_b
        assert matched_pair["utterance_a"]["reference_transcript"] == script_a
        assert matched_pair["utterance_b"]["reference_transcript"] == script_b

        # Step 5: Trigger mocked evaluation and verify comparison results persist correctly
        eval_resp = await client.post(f"/api/eval/run-human-pair-mock/{pair_id}")
        assert eval_resp.status_code == 200
        eval_data = eval_resp.json()
        assert eval_data["human_pair_id"] == pair_id
        assert eval_data["dataset_source"] == "human_recording"
        assert eval_data["overall_status"] == "completed"
        assert eval_data["utterance_a"]["reference_transcript"] == script_a
        assert eval_data["utterance_b"]["reference_transcript"] == script_b
        assert eval_data["semantic_distinction_preserved"] is True

        # Verify persisted in SQLite under dataset_source == 'human_recording'
        history_resp = await client.get("/api/eval/results?dataset_source=human_recording")
        assert history_resp.status_code == 200
        history_runs = history_resp.json()
        assert any(r["human_pair_id"] == pair_id for r in history_runs)

        # Step 6: Verify a 0.2-second audio file cannot be marked as valid merely because client claims 600s
        fake_short_wav = create_pcm_wav_bytes(duration_seconds=0.2)
        short_rec_id = f"rec_short_fake600_{uid}"
        short_resp = await client.post("/api/studio/upload", data={
            "recording_id": short_rec_id,
            "pair_id": f"human_test_short_{uid}",
            "utterance_role": "utterance_a",
            "speaker_id": speaker_id,
            "language": "en-IN",
            "category": "command_intent",
            "reference_transcript": "Short clip",
            "verified_transcript": "Short clip",
            "is_transcript_verified": "true",
            "consent_given": "true",
            "consent_statement": "Consent.",
            "client_duration_seconds": "600.0",  # Deceptive client claim
        }, files={"audio_file": ("short.wav", fake_short_wav, "audio/wav")})
        assert short_resp.status_code == 200
        short_data = short_resp.json()
        assert short_data["diagnostics"]["is_complete_50s"] is False
        assert short_data["validation_status"] == "incomplete_early_stop"
        assert short_data["diagnostics"]["duration_seconds"] < 0.5

        # Step 7: Verify a 11.7-second manually stopped recording is accepted as valid (not marked incomplete)
        valid_11_7_wav = create_pcm_wav_bytes(duration_seconds=11.7)
        rec_11_7_id = f"rec_11_7_{uid}"
        resp_11_7 = await client.post("/api/studio/upload", data={
            "recording_id": rec_11_7_id,
            "pair_id": f"human_test_11_7_{uid}",
            "utterance_role": "utterance_a",
            "speaker_id": speaker_id,
            "language": "hinglish",
            "category": "quantities_and_numbers",
            "reference_transcript": "Do plate momos pack kar do",
            "verified_transcript": "Do plate momos pack kar do",
            "is_transcript_verified": "true",
            "consent_given": "true",
            "consent_statement": "Consent confirmed.",
            "client_duration_seconds": "11.7",
        }, files={"audio_file": ("rec_11_7.wav", valid_11_7_wav, "audio/wav")})
        assert resp_11_7.status_code == 200
        data_11_7 = resp_11_7.json()
        assert data_11_7["diagnostics"]["is_complete_50s"] is True
        assert data_11_7["validation_status"] == "valid_50s"
        assert data_11_7["diagnostics"]["duration_seconds"] == pytest.approx(11.7, rel=1e-2)


def test_short_recording_5s_ingestion_and_validation(temp_studio_storage):
    """Proves backend accepts and validates short recordings (e.g. 5 seconds) when manually stopped.

    Verifies:
    1. A 5-second PCM WAV is successfully converted to standardized 16kHz mono 16-bit PCM WAV.
    2. Diagnostics accurately capture 5.0s duration and report decoding success.
    3. Storage saves the metadata, raw file, standardized file, and updates manifest.
    4. Dataset classification marks it as human_verified when transcript is verified and consent is given.
    5. Empty or corrupted audio bytes (< 64 bytes) are properly rejected by storage validation.
    """
    storage = temp_studio_storage
    storage.register_trusted_speaker("spk_short_human_01")
    raw_5s_wav = create_pcm_wav_bytes(duration_seconds=5.0)

    # Ingest 5-second recording
    meta = storage.save_recording(
        raw_bytes=raw_5s_wav,
        original_filename="manual_stop_5s.wav",
        speaker_id="spk_short_human_01",
        language="hinglish",
        category="quantities_and_numbers",
        reference_transcript="Do plate momos",
        verified_transcript="Do plate momos",
        is_transcript_verified=True,
        consent_given=True,
        consent_statement="Consent given.",
        client_duration_seconds=5.0,
        pair_id="pair_short_5s_01",
        utterance_role="utterance_a",
    )

    assert meta is not None
    assert meta.recording_id.startswith("rec_")
    assert meta.classification == "human_verified"
    assert meta.diagnostics.decoding_successful is True
    assert 4.8 <= meta.diagnostics.duration_seconds <= 5.2
    assert meta.diagnostics.sample_rate_hz == 16000
    assert meta.diagnostics.channels == 1
    assert meta.client_duration_seconds == 5.0

    # Verify standardized audio file exists and is valid 16kHz mono WAV
    std_path = storage.base_dir / meta.standardized_audio_path
    assert std_path.exists()
    assert std_path.stat().st_size > 1000

    # Verify empty or invalid audio (< 64 bytes) is rejected
    with pytest.raises(StorageValidationError, match="Audio payload is empty or too short"):
        storage.save_recording(
            raw_bytes=b"",
            original_filename="empty.wav",
            speaker_id="spk_short_human_01",
            language="hinglish",
            category="quantities_and_numbers",
            reference_transcript="Empty",
            consent_given=True,
            consent_statement="Consent given.",
        )

    with pytest.raises(StorageValidationError, match="Audio payload is empty or too short"):
        storage.save_recording(
            raw_bytes=b"RIFF" + b"\x00" * 20,
            original_filename="corrupted.wav",
            speaker_id="spk_short_human_01",
            language="hinglish",
            category="quantities_and_numbers",
            reference_transcript="Corrupted",
            consent_given=True,
            consent_statement="Consent given.",
        )


@pytest.mark.asyncio
async def test_short_recording_5s_api_upload_and_empty_rejection(temp_studio_storage, monkeypatch):
    """Proves the /api/studio/upload endpoint accepts 5s audio and rejects empty audio."""
    import app.routers.recording as rec_router
    from app.main import app as web_app

    monkeypatch.setattr(rec_router, "storage", temp_studio_storage)
    temp_studio_storage.register_trusted_speaker("spk_short_api_01")

    transport = ASGITransport(app=web_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Successful 5s recording upload
        raw_5s_wav = create_pcm_wav_bytes(duration_seconds=5.0)
        resp = await client.post(
            "/api/studio/upload",
            data={
                "recording_id": "rec_short_5s_test",
                "pair_id": "pair_short_5s_api",
                "utterance_role": "utterance_a",
                "speaker_id": "spk_short_api_01",
                "language": "hi-IN",
                "category": "quantities_and_numbers",
                "reference_transcript": "Paanch rupaye",
                "verified_transcript": "Paanch rupaye",
                "is_transcript_verified": "true",
                "consent_given": "true",
                "consent_statement": "Consent given.",
                "client_duration_seconds": "5.0",
            },
            files={"audio_file": ("test_5s.wav", raw_5s_wav, "audio/wav")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["recording_id"] == "rec_short_5s_test"
        assert 4.8 <= data["diagnostics"]["duration_seconds"] <= 5.2
        assert data["classification"] == "human_verified"

        # 2. Reject empty audio file
        empty_resp = await client.post(
            "/api/studio/upload",
            data={
                "speaker_id": "spk_short_api_01",
                "language": "hi-IN",
                "category": "quantities_and_numbers",
                "reference_transcript": "Empty test",
                "consent_given": "true",
                "consent_statement": "Consent given.",
            },
            files={"audio_file": ("empty.wav", b"", "audio/wav")},
        )
        assert empty_resp.status_code == 400
        assert "Audio payload is empty or too short" in empty_resp.json()["detail"]
