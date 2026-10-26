"""Tests for Phase 3: Hinglish Dataset Integrity and Logical Segregation.

Verifies:
1. Dynamic and persisted classification of recordings and pairs (human_verified, synthetic, test_artifact, unverified).
2. Pair filtering endpoints (curated_only, exclude_test_artifacts, classification).
3. Segregated metrics computation across dataset cohorts (human vs synthetic vs test_artifacts).
4. Backward compatibility with existing human recording studio and evaluation endpoints.
5. Invariance of production database bolibench_evaluation.db and recording files.
"""

import io
import json
import uuid
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
    determine_pair_classification,
    determine_recording_classification,
)
from app.evaluation.schemas import (
    AudioCondition,
    ExecutionMode,
    FailureCategory,
    PairEvaluationResult,
    TestCategory,
    UtteranceEvaluationResult,
)
from app.evaluation.storage import StorageManager


def generate_test_wav_bytes(duration_sec: float = 1.0, sample_rate: int = 16000) -> bytes:
    """Generate in-memory 16kHz mono PCM WAV bytes."""
    num_samples = int(duration_sec * sample_rate)
    samples = (0.2 * 32767.0 * np.sin(2 * np.pi * 440.0 * np.linspace(0, duration_sec, num_samples, endpoint=False))).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())
    return buf.getvalue()


def make_dummy_metadata(
    recording_id: str,
    speaker_id: str,
    reference_transcript: str = "Do packet chahiye",
    verified_transcript: str = "Do packet chahiye",
    is_transcript_verified: bool = True,
    utterance_role: UtteranceRole = UtteranceRole.UTTERANCE_A,
    pair_id: str = "pair_test_01",
    scenario_id: str = "pair_test_01",
    classification: str = None,
) -> RecordingMetadata:
    diag = AudioDiagnostics(
        duration_seconds=50.0,
        sample_rate_hz=16000,
        channels=1,
        is_complete_50s=True,
        rms_amplitude=0.1,
        peak_amplitude=0.5,
        is_silent=False,
        clipped_samples_count=0,
        clipping_ratio=0.0,
        is_clipped=False,
        decoding_successful=True,
        converter_used="ffmpeg",
    )
    kwargs = dict(
        recording_id=recording_id,
        speaker_id=speaker_id,
        language="hi-IN",
        category="quantities_and_numbers",
        scenario_id=scenario_id,
        pair_id=pair_id,
        utterance_role=utterance_role,
        reference_transcript=reference_transcript,
        verified_transcript=verified_transcript,
        is_transcript_verified=is_transcript_verified,
        consent_given=True,
        consent_statement="I consent to record and use audio for evaluation.",
        consent_timestamp="2026-10-23T12:00:00Z",
        created_at="2026-10-23T12:00:00Z",
        client_duration_seconds=50.0,
        diagnostics=diag,
        validation_status=ValidationStatus.VALID_50S,
        raw_audio_filename=f"{recording_id}_raw.webm",
        standardized_audio_filename=f"{recording_id}.wav",
        raw_audio_path=f"raw/{recording_id}_raw.webm",
        standardized_audio_path=f"standardized_16k/{recording_id}.wav",
    )
    if classification is not None:
        kwargs["classification"] = classification
    return RecordingMetadata(**kwargs)


def make_dummy_pair_result(
    run_id: str,
    classification: str,
    wer_avg: float,
    pass_status: bool,
) -> PairEvaluationResult:
    u_a = UtteranceEvaluationResult(
        utterance_id=f"{run_id}_a",
        reference_transcript="Ref A",
        actual_transcript="Actual A",
        wer=wer_avg,
        substitutions=0,
        deletions=0,
        insertions=0,
        reference_length=2,
        critical_fields_match=pass_status,
        detected_intent="ORDER",
        intent_match=pass_status,
        task_success=pass_status,
        status="completed",
    )
    u_b = UtteranceEvaluationResult(
        utterance_id=f"{run_id}_b",
        reference_transcript="Ref B",
        actual_transcript="Actual B",
        wer=wer_avg,
        substitutions=0,
        deletions=0,
        insertions=0,
        reference_length=2,
        critical_fields_match=pass_status,
        detected_intent="ORDER",
        intent_match=pass_status,
        task_success=pass_status,
        status="completed",
    )
    return PairEvaluationResult(
        run_id=run_id,
        test_case_id=f"tc_{run_id}",
        category="quantities_and_numbers",
        language_code="hi-IN",
        execution_mode=ExecutionMode.MOCK,
        audio_condition="clean_recorded",
        timestamp="2026-10-23T12:00:00Z",
        overall_status="completed",
        distinction_preserved=pass_status,
        semantic_distinction_preserved=pass_status,
        task_success=pass_status,
        wer_a=wer_avg,
        wer_b=wer_avg,
        average_wer=wer_avg,
        average_normalized_wer=wer_avg,
        critical_field_accuracy=1.0 if pass_status else 0.0,
        primary_failure_category=FailureCategory.NONE if pass_status else FailureCategory.ENTITY_SUBSTITUTION,
        failure_explanation="Preserved" if pass_status else "Failed",
        utterance_a=u_a,
        utterance_b=u_b,
        dataset_source="human_recording",
        dataset_classification=classification,
    )


# ---------------------------------------------------------------------------
# 1. Classification Logic Tests
# ---------------------------------------------------------------------------

def test_determine_recording_classification_logic():
    """Verifies heuristic and metadata-driven classification of recordings."""
    # Test artifact by speaker prefix
    rec_test = make_dummy_metadata("rec_001", "spk_test_123")
    assert determine_recording_classification(rec_test) == DatasetClassification.TEST_ARTIFACT.value

    # Test artifact by mock speaker prefix
    rec_mock = make_dummy_metadata("rec_002", "spk_mock_01")
    assert determine_recording_classification(rec_mock) == DatasetClassification.TEST_ARTIFACT.value

    # Human verified: real speaker, verified transcript present
    rec_verified = make_dummy_metadata("rec_003", "spk_curated_native_01")
    assert determine_recording_classification(rec_verified) == DatasetClassification.HUMAN_VERIFIED.value

    # Unverified: real speaker, but no verified transcript
    rec_unverified = make_dummy_metadata(
        "rec_004",
        "spk_curated_native_01",
        verified_transcript=None,
        is_transcript_verified=False,
    )
    assert determine_recording_classification(rec_unverified) == DatasetClassification.UNVERIFIED.value


def test_determine_pair_classification_logic():
    """Verifies heuristic and metadata-driven classification of contrast pairs."""
    # Test artifact pair ID
    pair_test = {
        "pair_id": "human_acc_pair_xyz123",
        "verified": True,
        "speaker_id": "spk_acc_xyz",
    }
    assert determine_pair_classification(pair_test) == DatasetClassification.TEST_ARTIFACT.value

    # Verified human pair: real ID, marked verified
    pair_curated = {
        "pair_id": "human_curated_qty_001",
        "verified": True,
        "speaker_id": "spk_curated_native_01",
    }
    assert determine_pair_classification(pair_curated) == DatasetClassification.HUMAN_VERIFIED.value

    # Unverified human pair: real ID, but verified=False
    pair_incomplete = {
        "pair_id": "human_qty_001",
        "verified": False,
        "speaker_id": "spk_curated_native_01",
    }
    assert determine_pair_classification(pair_incomplete) == DatasetClassification.UNVERIFIED.value


def save_meta_recording(storage: RecordingStorage, wav_bytes: bytes, meta: RecordingMetadata) -> RecordingMetadata:
    return storage.save_recording(
        raw_bytes=wav_bytes,
        original_filename=meta.raw_audio_filename,
        speaker_id=meta.speaker_id,
        language=meta.language,
        category=meta.category,
        reference_transcript=meta.reference_transcript,
        consent_given=meta.consent_given,
        consent_statement=meta.consent_statement,
        recording_id=meta.recording_id,
        scenario_id=meta.scenario_id,
        pair_id=meta.pair_id,
        utterance_role=meta.utterance_role.value if hasattr(meta.utterance_role, "value") else str(meta.utterance_role),
        verified_transcript=meta.verified_transcript,
        is_transcript_verified=meta.is_transcript_verified,
        client_duration_seconds=meta.client_duration_seconds,
        classification=meta.classification if meta.classification != DatasetClassification.UNVERIFIED.value else None,
    )


# ---------------------------------------------------------------------------
# 2. Recording Storage Dynamic Classification & Filtering
# ---------------------------------------------------------------------------

def test_recording_storage_classification_and_filtering(tmp_path):
    """Verifies saving, dynamic backfilling, and filtering in RecordingStorage."""
    storage = RecordingStorage(base_dir=tmp_path / "recordings")
    wav_bytes = generate_test_wav_bytes(duration_sec=1.0)

    # 1. Save test artifact recording
    meta_test = make_dummy_metadata("rec_test_01", "spk_test_user")
    rec_test = save_meta_recording(storage, wav_bytes, meta_test)
    assert rec_test.classification == DatasetClassification.TEST_ARTIFACT.value

    # 2. Save human verified recording (register trusted speaker first)
    storage.register_trusted_speaker("spk_real_native")
    meta_verified = make_dummy_metadata("rec_human_01", "spk_real_native")
    rec_v = save_meta_recording(storage, wav_bytes, meta_verified)
    assert rec_v.classification == DatasetClassification.HUMAN_VERIFIED.value

    # 3. Save unverified recording
    meta_unv = make_dummy_metadata(
        "rec_human_unv",
        "spk_real_native_2",
        verified_transcript=None,
        is_transcript_verified=False,
    )
    rec_u = save_meta_recording(storage, wav_bytes, meta_unv)
    assert rec_u.classification == DatasetClassification.UNVERIFIED.value

    # Test filtering in list_recordings
    all_recs = storage.list_recordings()
    assert len(all_recs) == 3

    non_test_recs = storage.list_recordings(exclude_test_artifacts=True)
    assert len(non_test_recs) == 2
    assert all(r["classification"] != DatasetClassification.TEST_ARTIFACT.value for r in non_test_recs)

    curated_recs = storage.list_recordings(classification=DatasetClassification.HUMAN_VERIFIED.value)
    assert len(curated_recs) == 1
    assert curated_recs[0]["recording_id"] == "rec_human_01"


# ---------------------------------------------------------------------------
# 3. API Filtering for Human Pairs and Curated Selection
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_api_human_pairs_filtering(isolate_recording_storage):
    """Verifies /api/eval/human-pairs and /api/studio/pairs query filters."""
    storage = isolate_recording_storage
    wav_bytes = generate_test_wav_bytes(duration_sec=1.0)

    # Create a test pair (untrusted speaker prefix cannot be verified via verify_contrast_pair)
    from app.services.recording_storage import StorageValidationError
    pair_test_id = "test_pair_alpha"
    m_ta = make_dummy_metadata("rec_ta", "spk_test_qa", reference_transcript="Do packet chahiye", verified_transcript="Do packet chahiye", pair_id=pair_test_id, scenario_id=pair_test_id, utterance_role=UtteranceRole.UTTERANCE_A)
    m_tb = make_dummy_metadata("rec_tb", "spk_test_qa", reference_transcript="Teen packet chahiye", verified_transcript="Teen packet chahiye", pair_id=pair_test_id, scenario_id=pair_test_id, utterance_role=UtteranceRole.UTTERANCE_B)
    save_meta_recording(storage, wav_bytes, m_ta)
    save_meta_recording(storage, wav_bytes, m_tb)
    # Fail-closed: pair verification must reject untrusted test speakers
    with pytest.raises(StorageValidationError, match="trusted speaker registry"):
        storage.verify_contrast_pair(pair_test_id, confirmed_semantic_distinction=True, distinction_field="quantity", expected_distinction_desc="test delta")

    # Create a curated human pair with registered trusted speaker
    pair_curated_id = "human_curated_001"
    storage.register_trusted_speaker("spk_native_speaker_01")
    m_ca = make_dummy_metadata("rec_ca", "spk_native_speaker_01", reference_transcript="Do plate biryani pack kar do", verified_transcript="Do plate biryani pack kar do", pair_id=pair_curated_id, scenario_id=pair_curated_id, utterance_role=UtteranceRole.UTTERANCE_A)
    m_cb = make_dummy_metadata("rec_cb", "spk_native_speaker_01", reference_transcript="Teen plate biryani pack kar do", verified_transcript="Teen plate biryani pack kar do", pair_id=pair_curated_id, scenario_id=pair_curated_id, utterance_role=UtteranceRole.UTTERANCE_B)
    save_meta_recording(storage, wav_bytes, m_ca)
    save_meta_recording(storage, wav_bytes, m_cb)
    storage.verify_contrast_pair(pair_curated_id, confirmed_semantic_distinction=True, distinction_field="quantity", expected_distinction_desc="2 vs 3 plates")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Default behavior: backward compatible, returns both
        res_default = await client.get("/api/eval/human-pairs")
        assert res_default.status_code == 200
        pids_default = [p["pair_id"] for p in res_default.json()]
        assert pair_test_id in pids_default
        assert pair_curated_id in pids_default

        # Filter: exclude_test_artifacts=true
        res_clean = await client.get("/api/eval/human-pairs?exclude_test_artifacts=true")
        assert res_clean.status_code == 200
        pids_clean = [p["pair_id"] for p in res_clean.json()]
        assert pair_test_id not in pids_clean
        assert pair_curated_id in pids_clean

        # Filter: curated_only=true
        res_curated = await client.get("/api/eval/human-pairs?curated_only=true")
        assert res_curated.status_code == 200
        pids_curated = [p["pair_id"] for p in res_curated.json()]
        assert pids_curated == [pair_curated_id]


# ---------------------------------------------------------------------------
# 4. Segregated Metrics Computation in Evaluation Storage
# ---------------------------------------------------------------------------

def test_evaluation_storage_segregated_metrics(tmp_path):
    """Verifies that SQLite storage persists dataset_classification and computes segregated cohort metrics."""
    db_file = tmp_path / "test_segregation.db"
    storage = StorageManager(db_path=db_file)

    # 1. Save 2 human_verified runs (both pass, WER 0.05)
    r1 = make_dummy_pair_result("run_hv_1", DatasetClassification.HUMAN_VERIFIED.value, 0.05, True)
    r2 = make_dummy_pair_result("run_hv_2", DatasetClassification.HUMAN_VERIFIED.value, 0.05, True)
    storage.save_run(r1)
    storage.save_run(r2)

    # 2. Save 2 synthetic runs (both pass, WER 0.02)
    r3 = make_dummy_pair_result("run_synth_1", DatasetClassification.SYNTHETIC.value, 0.02, True)
    r4 = make_dummy_pair_result("run_synth_2", DatasetClassification.SYNTHETIC.value, 0.02, True)
    storage.save_run(r3)
    storage.save_run(r4)

    # 3. Save 1 test_artifact run (fails, WER 0.80)
    r5 = make_dummy_pair_result("run_art_1", DatasetClassification.TEST_ARTIFACT.value, 0.80, False)
    storage.save_run(r5)

    # Total runs in DB
    all_runs = storage.list_runs(limit=10)
    assert len(all_runs) == 5

    # Filtered list_runs
    hv_runs = storage.list_runs(dataset_classification=DatasetClassification.HUMAN_VERIFIED.value)
    assert len(hv_runs) == 2
    assert all(r.dataset_classification == DatasetClassification.HUMAN_VERIFIED.value for r in hv_runs)

    # Segregated Aggregate Metrics
    # Aggregate over ALL runs (contaminated by test_artifact failure)
    all_metrics = storage.get_aggregate_metrics()
    assert all_metrics.total_runs == 5
    assert all_metrics.distinction_preservation_rate == 80.0  # 4 out of 5 passed

    # Curated human cohort metrics ONLY
    hv_metrics = storage.get_aggregate_metrics(dataset_classification=DatasetClassification.HUMAN_VERIFIED.value)
    assert hv_metrics.total_runs == 2
    assert hv_metrics.distinction_preservation_rate == 100.0
    assert hv_metrics.average_wer == 0.05
    assert hv_metrics.failure_category_breakdown == {"none": 2}

    # Synthetic cohort metrics ONLY
    synth_metrics = storage.get_aggregate_metrics(dataset_classification=DatasetClassification.SYNTHETIC.value)
    assert synth_metrics.total_runs == 2
    assert synth_metrics.distinction_preservation_rate == 100.0
    assert synth_metrics.average_wer == 0.02


@pytest.mark.asyncio
async def test_api_metrics_and_results_segregation(isolate_evaluation_storage):
    """Verifies /api/eval/metrics and /api/eval/results with dataset_classification filter."""
    storage = isolate_evaluation_storage

    res_hv = make_dummy_pair_result("run_api_hv", "human_verified", 0.05, True)
    res_art = make_dummy_pair_result("run_api_art", "test_artifact", 0.90, False)
    storage.save_run(res_hv)
    storage.save_run(res_art)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Check filtered results
        res_all = await client.get("/api/eval/results")
        assert res_all.status_code == 200
        assert len(res_all.json()) == 2

        res_filtered = await client.get("/api/eval/results?dataset_classification=human_verified")
        assert res_filtered.status_code == 200
        results_json = res_filtered.json()
        assert len(results_json) == 1
        assert results_json[0]["run_id"] == "run_api_hv"

        # Check segregated metrics
        metrics_all = await client.get("/api/eval/metrics")
        assert metrics_all.status_code == 200
        assert metrics_all.json()["total_runs"] == 2
        assert metrics_all.json()["distinction_preservation_rate"] == 50.0

        metrics_hv = await client.get("/api/eval/metrics?dataset_classification=human_verified")
        assert metrics_hv.status_code == 200
        assert metrics_hv.json()["total_runs"] == 1
        assert metrics_hv.json()["distinction_preservation_rate"] == 100.0
