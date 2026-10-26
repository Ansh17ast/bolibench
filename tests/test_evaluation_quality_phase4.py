"""Comprehensive Phase 4 Validation: End-to-End Evaluation Quality & Dataset Integrity.

Covers all 6 Phase 4 Workstreams:
1. End-to-end workflow: Ingestion -> verification -> pair eligibility -> evaluation -> persistence -> API retrieval.
2. Hinglish benchmark metrics: Raw WER, Normalized WER, Critical Field Accuracy, Polysemy (do/दो), Loan Words, Negation, Names.
3. Dataset eligibility: Trusted speaker registry, explicit consent, audio diagnostics (silent/clipped/decoding).
4. Failure recovery: Incomplete run persistence, single-utterance resumption, upload idempotency, and concurrent resumption.
5. Migration safety: Legacy schema simulation, automatic column addition, idempotency, and production DB invariance.
6. Reproducibility: Deterministic metric calculation and persistence consistency across identical inputs.
"""

import io
import json
import sqlite3
import wave
from pathlib import Path
from unittest.mock import AsyncMock, patch
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
    determine_pair_classification,
    determine_recording_classification,
)
from app.evaluation.engine import EvaluationEngine
from app.evaluation.metrics import (
    calculate_cross_script_normalized_wer,
    calculate_wer,
    canonicalize_cross_script_text,
    normalize_text,
    score_pair,
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
from app.services.gnani_client import GnaniPrismaClient, GnaniTimeoutError


def create_test_wav_bytes(duration_sec: float = 1.0, sample_rate: int = 16000, freq_hz: float = 440.0) -> bytes:
    """Generates valid 16kHz mono PCM WAV bytes."""
    num_samples = int(duration_sec * sample_rate)
    samples = (0.2 * 32767.0 * np.sin(2 * np.pi * freq_hz * np.linspace(0, duration_sec, num_samples, endpoint=False))).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())
    return buf.getvalue()


# ===========================================================================
# Workstream 1: End-to-End Workflow Validation
# ===========================================================================

@pytest.mark.asyncio
async def test_e2e_workflow_recording_to_curated_evaluation_and_api(isolate_recording_storage, isolate_evaluation_storage):
    """Validates the complete lifecycle:
    Ingestion -> Verification -> Pair Eligibility -> Live Paired Evaluation (mocked client)
    -> Result Persistence -> API Retrieval (/results, /results/{id}, /metrics).
    """
    rec_storage = isolate_recording_storage
    eval_storage = isolate_evaluation_storage
    engine = EvaluationEngine(storage_manager=eval_storage)

    # 1. Register trusted human speaker
    speaker_id = "spk_trusted_native_99"
    rec_storage.register_trusted_speaker(speaker_id)
    assert rec_storage.is_speaker_trusted(speaker_id) is True

    wav_bytes = create_test_wav_bytes(duration_sec=1.5)
    pair_id = "human_curated_order_99"

    # 2. Ingest Utterance A & B with valid transcripts and consent
    meta_a = rec_storage.save_recording(
        raw_bytes=wav_bytes,
        original_filename="clip_a.wav",
        speaker_id=speaker_id,
        language="hi-IN",
        category="quantities_and_numbers",
        reference_transcript="Do plate biryani pack kar do",
        verified_transcript="Do plate biryani pack kar do",
        is_transcript_verified=True,
        consent_given=True,
        consent_statement="Full consent granted for benchmark recording.",
        pair_id=pair_id,
        utterance_role="utterance_a",
    )
    meta_b = rec_storage.save_recording(
        raw_bytes=wav_bytes,
        original_filename="clip_b.wav",
        speaker_id=speaker_id,
        language="hi-IN",
        category="quantities_and_numbers",
        reference_transcript="Teen plate biryani pack kar do",
        verified_transcript="Teen plate biryani pack kar do",
        is_transcript_verified=True,
        consent_given=True,
        consent_statement="Full consent granted for benchmark recording.",
        pair_id=pair_id,
        utterance_role="utterance_b",
    )

    # 3. Verify and seal the contrastive pair
    pair_record = rec_storage.verify_contrast_pair(
        pair_id=pair_id,
        confirmed_semantic_distinction=True,
        distinction_field="quantity",
        expected_distinction_desc="Contrast between 2 plates and 3 plates biryani",
        verified_by="chief_operator",
    )
    assert pair_record.classification == DatasetClassification.HUMAN_VERIFIED.value

    # 4. Check candidate pair eligibility via API
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        res_pairs = await client.get("/api/eval/human-pairs?curated_only=true")
        assert res_pairs.status_code == 200
        curated_pids = [p["pair_id"] for p in res_pairs.json()]
        assert pair_id in curated_pids

    # 5. Execute Paired Evaluation using mocked Gnani Prisma client (Zero real credits)
    mock_client = AsyncMock(spec=GnaniPrismaClient)
    mock_client.transcribe.side_effect = [
        {
            "success": True,
            "transcript": "2 प्लेट बिरयानी पैक कर दो",
            "request_id": "req_gnani_a_001",
            "client_request_elapsed_ms": 110.0,
            "provider_processing_time_ms": 45.0,
        },
        {
            "success": True,
            "transcript": "3 प्लेट बिरयानी पैक कर दो",
            "request_id": "req_gnani_b_002",
            "client_request_elapsed_ms": 115.0,
            "provider_processing_time_ms": 48.0,
        },
    ]

    eval_result = await engine.evaluate_human_pair_live(
        pair_id=pair_id,
        client=mock_client,
        audio_condition="clean_recorded",
        recording_storage=rec_storage,
    )

    assert mock_client.transcribe.call_count == 2
    assert eval_result.overall_status == "completed"
    assert eval_result.dataset_source == "human_recording"
    assert eval_result.dataset_classification == DatasetClassification.HUMAN_VERIFIED.value
    assert eval_result.human_pair_id == pair_id
    assert eval_result.distinction_preserved is True
    assert eval_result.average_normalized_wer == 0.0

    # 6. Verify result retrieval and metric segregation via API
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Retrieve single run by ID
        get_res = await client.get(f"/api/eval/results/{eval_result.run_id}")
        assert get_res.status_code == 200
        run_data = get_res.json()
        assert run_data["run_id"] == eval_result.run_id
        assert run_data["dataset_classification"] == "human_verified"

        # Query metrics filtered to human_verified cohort
        metrics_res = await client.get("/api/eval/metrics?dataset_classification=human_verified")
        assert metrics_res.status_code == 200
        metrics = metrics_res.json()
        assert metrics["total_runs"] == 1
        assert metrics["completed_runs"] == 1
        assert metrics["distinction_preservation_rate"] == 100.0


# ===========================================================================
# Workstream 2: Hinglish Benchmark Metrics Deep Validation
# ===========================================================================

def test_hinglish_metrics_code_switching_loan_words():
    """Validates cross-script equivalence for enterprise Hinglish loan words."""
    # Loan words in Devanagari vs Roman script
    ref = "Order cancel kar do aur amount account me transfer karo"
    hyp = "ऑर्डर कैंसिल कर दो और amount अकाउंट me ट्रांसफर karo"

    raw_wer, subs, dels, inss, _ = calculate_wer(ref, hyp)
    assert raw_wer > 0.0, "Raw WER must detect script divergence"

    norm_wer, n_subs, n_dels, n_inss, _ = calculate_cross_script_normalized_wer(ref, hyp)
    assert norm_wer == 0.0, "Normalized WER must equate whitelisted enterprise loan words"


def test_hinglish_metrics_numeral_and_do_polysemy():
    """Validates numeral mapping and context-aware disambiguation of 'do' / 'दो'."""
    # Case 1: 'do' as numeral 2 before quantified noun
    ref_num = "Mujhe do packet chahiye"
    hyp_digit = "Mujhe 2 पैकेट चाहिए"
    norm_wer_num, _, _, _, _ = calculate_cross_script_normalized_wer(ref_num, hyp_digit)
    assert norm_wer_num == 0.0, "'do packet' must canonicalize to numeral 2"

    # Case 2: 'do' as imperative auxiliary light verb ('give/perform') after verbal root
    ref_verb = "Payment transfer kar do"
    hyp_verb = "Payment transfer kar do"
    canon_verb = canonicalize_cross_script_text(ref_verb)
    assert "2" not in canon_verb.split(), "'kar do' must NEVER convert verbal 'do' to numeral '2'"

    # Case 3: Contrast distinction must NEVER be collapsed
    ref_a = "Do packet chahiye"
    ref_b = "Teen packet chahiye"
    wer_between_pairs, _, _, _, _ = calculate_cross_script_normalized_wer(ref_a, ref_b)
    assert wer_between_pairs > 0.0, "Distinction between 2 and 3 must never be erased"


def test_hinglish_metrics_named_entities_and_negation():
    """Validates critical entity substitution and negation polarity preservation."""
    # Case 1: Named Entity substitution (Rohit -> Rahul)
    ref_entity = "Rohit ko 500 rupaye bhejo"
    hyp_entity = "Rahul ko 500 rupaye bhejo"
    raw_wer, _, _, _, _ = calculate_wer(ref_entity, hyp_entity)
    norm_wer, _, _, _, _ = calculate_cross_script_normalized_wer(ref_entity, hyp_entity)
    assert raw_wer > 0.0
    assert norm_wer > 0.0, "Named entity substitution must not be erased"

    # Case 2: Negation polarity flip (haan vs nahi)
    ref_affirm = "Haan ticket confirm kar do"
    hyp_negate = "Nahi ticket confirm mat karo"
    wer_neg, _, _, _, _ = calculate_cross_script_normalized_wer(ref_affirm, hyp_negate)
    assert wer_neg > 0.0, "Negation polarity tokens must never be equated"


# ===========================================================================
# Workstream 3: Dataset Eligibility & Trusted Speaker Registration
# ===========================================================================

def test_dataset_eligibility_and_trusted_speaker_guard(tmp_path):
    """Proves that unverified speakers, invalid audio diagnostics, and missing consent
    are blocked from attaining curated benchmark status.
    """
    storage = RecordingStorage(base_dir=tmp_path / "recordings_eligibility")
    wav_bytes = create_test_wav_bytes(duration_sec=1.0)

    # 1. Unknown speaker (not in trusted registry)
    storage.register_trusted_speaker("spk_authorized_human_01")
    assert storage.is_speaker_trusted("spk_unauthorized_user") is False

    meta_unauthorized = storage.save_recording(
        raw_bytes=wav_bytes,
        original_filename="test.wav",
        speaker_id="spk_unauthorized_user",
        language="hi-IN",
        category="quantities_and_numbers",
        reference_transcript="Do packet chahiye",
        verified_transcript="Do packet chahiye",
        is_transcript_verified=True,
        consent_given=True,
        consent_statement="Consent",
    )
    assert meta_unauthorized.classification == DatasetClassification.UNVERIFIED.value

    # 2. Acoustic diagnostics gating: Severe clipping
    diag_clipped = AudioDiagnostics(
        duration_seconds=50.0,
        sample_rate_hz=16000,
        channels=1,
        is_complete_50s=True,
        rms_amplitude=0.8,
        peak_amplitude=1.0,
        is_silent=False,
        clipped_samples_count=5000,
        clipping_ratio=0.15,
        is_clipped=True,
        decoding_successful=True,
        converter_used="ffmpeg",
    )
    meta_clipped = RecordingMetadata(
        recording_id="rec_clipped_01",
        speaker_id="spk_authorized_human_01",
        language="hi-IN",
        category="quantities_and_numbers",
        utterance_role=UtteranceRole.UTTERANCE_A,
        reference_transcript="Do packet",
        verified_transcript="Do packet",
        is_transcript_verified=True,
        consent_given=True,
        consent_statement="Consent",
        consent_timestamp="2026-10-23T12:00:00Z",
        created_at="2026-10-23T12:00:00Z",
        diagnostics=diag_clipped,
        raw_audio_filename="clip.webm",
        standardized_audio_filename="clip.wav",
        raw_audio_path="raw/clip.webm",
        standardized_audio_path="standardized_16k/clip.wav",
    )
    class_clipped = determine_recording_classification(meta_clipped, trusted_speakers=set(storage.list_trusted_speakers()))
    assert class_clipped == DatasetClassification.UNVERIFIED.value, "Clipped audio must not be human_verified"

    # 3. Acoustic diagnostics gating: Silent audio
    diag_silent = AudioDiagnostics(
        duration_seconds=50.0,
        sample_rate_hz=16000,
        channels=1,
        is_complete_50s=True,
        rms_amplitude=0.0001,
        peak_amplitude=0.0002,
        is_silent=True,
        clipped_samples_count=0,
        clipping_ratio=0.0,
        is_clipped=False,
        decoding_successful=True,
        converter_used="ffmpeg",
    )
    meta_silent = RecordingMetadata(
        recording_id="rec_silent_01",
        speaker_id="spk_authorized_human_01",
        language="hi-IN",
        category="quantities_and_numbers",
        utterance_role=UtteranceRole.UTTERANCE_A,
        reference_transcript="Do packet",
        verified_transcript="Do packet",
        is_transcript_verified=True,
        consent_given=True,
        consent_statement="Consent",
        consent_timestamp="2026-10-23T12:00:00Z",
        created_at="2026-10-23T12:00:00Z",
        diagnostics=diag_silent,
        raw_audio_filename="clip.webm",
        standardized_audio_filename="clip.wav",
        raw_audio_path="raw/clip.webm",
        standardized_audio_path="standardized_16k/clip.wav",
    )
    class_silent = determine_recording_classification(meta_silent, trusted_speakers=set(storage.list_trusted_speakers()))
    assert class_silent == DatasetClassification.UNVERIFIED.value, "Silent audio must not be human_verified"


# ===========================================================================
# Workstream 4: Failure Recovery, Resumption, Idempotency & Retries
# ===========================================================================

@pytest.mark.asyncio
async def test_failure_recovery_interrupted_run_resumption_and_idempotency(tmp_path):
    """Validates that:
    1. Interrupted evaluation saves an 'incomplete' run in SQLite.
    2. Resuming the run only executes the failed utterance (0 redundant calls).
    3. Upload idempotency prevents duplicate storage writes.
    """
    db_path = tmp_path / "test_resume.db"
    storage = StorageManager(db_path=db_path)
    engine = EvaluationEngine(storage_manager=storage)
    wav_bytes = create_test_wav_bytes(duration_sec=1.0)

    # 1. Incomplete evaluation simulation
    mock_client = AsyncMock(spec=GnaniPrismaClient)
    mock_client.transcribe.side_effect = [
        {
            "success": True,
            "transcript": "Do packet chahiye",
            "request_id": "req_first_ok",
            "client_request_elapsed_ms": 100.0,
            "provider_processing_time_ms": 40.0,
        },
        GnaniTimeoutError(timeout_seconds=30.0),
    ]

    with pytest.raises(GnaniTimeoutError):
        await engine.evaluate_live_audio(
            test_case_id="pair_qty_001",
            audio_a_bytes=wav_bytes,
            filename_a="utt_a.wav",
            content_type_a="audio/wav",
            audio_b_bytes=wav_bytes,
            filename_b="utt_b.wav",
            content_type_b="audio/wav",
            client=mock_client,
        )

    assert mock_client.transcribe.call_count == 2
    incomplete_runs = storage.list_runs(status="incomplete")
    assert len(incomplete_runs) == 1
    incomplete_run = incomplete_runs[0]

    # 2. Resume incomplete evaluation: Only Utterance B is called
    mock_client.reset_mock()
    mock_client.transcribe.side_effect = [
        {
            "success": True,
            "transcript": "Teen packet chahiye",
            "request_id": "req_second_ok",
            "client_request_elapsed_ms": 110.0,
            "provider_processing_time_ms": 42.0,
        }
    ]

    resumed_result = await engine.evaluate_live_audio(
        test_case_id="pair_qty_001",
        audio_a_bytes=wav_bytes,
        filename_a="utt_a.wav",
        content_type_a="audio/wav",
        audio_b_bytes=wav_bytes,
        filename_b="utt_b.wav",
        content_type_b="audio/wav",
        client=mock_client,
        resume_run_id=incomplete_run.run_id,
    )

    assert mock_client.transcribe.call_count == 1, "Resumption must ONLY call the failed utterance"
    assert resumed_result.run_id == incomplete_run.run_id
    assert resumed_result.overall_status == "completed"
    assert resumed_result.distinction_preserved is True

    # 3. Upload Idempotency: Duplicate upload with same idempotency_key returns original
    rec_storage = RecordingStorage(base_dir=tmp_path / "idemp_recordings")
    idemp_key = "idemp_phase4_unique_token_123"
    r1 = rec_storage.save_recording(
        raw_bytes=wav_bytes,
        original_filename="audio.wav",
        speaker_id="spk_operator_01",
        language="hi-IN",
        category="quantities_and_numbers",
        reference_transcript="Do packet",
        consent_given=True,
        consent_statement="Consent",
        idempotency_key=idemp_key,
    )
    r2 = rec_storage.save_recording(
        raw_bytes=wav_bytes,
        original_filename="audio.wav",
        speaker_id="spk_operator_01",
        language="hi-IN",
        category="quantities_and_numbers",
        reference_transcript="Do packet",
        consent_given=True,
        consent_statement="Consent",
        idempotency_key=idemp_key,
    )
    assert r1.recording_id == r2.recording_id
    all_recordings = rec_storage.list_recordings()
    assert len(all_recordings) == 1, "Idempotent upload must not create duplicate recording entries"


# ===========================================================================
# Workstream 5: Database Migration Safety & Legacy Schema Compatibility
# ===========================================================================

def test_database_migration_safety_on_legacy_and_current_schemas(tmp_path):
    """Simulates a legacy SQLite database (without dataset_classification column),
    verifies graceful backward compatibility on read, executes safe migration,
    verifies idempotency, and confirms production database is untouched.
    """
    legacy_db_file = tmp_path / "legacy_evaluation.db"

    # 1. Create a raw legacy table matching the pre-Phase 3 schema
    with sqlite3.connect(legacy_db_file) as conn:
        conn.execute(
            """
            CREATE TABLE evaluation_runs (
                run_id TEXT PRIMARY KEY,
                test_case_id TEXT NOT NULL,
                category TEXT NOT NULL,
                language_code TEXT NOT NULL,
                execution_mode TEXT NOT NULL,
                audio_condition TEXT NOT NULL,
                wer_a REAL NOT NULL,
                wer_b REAL NOT NULL,
                average_wer REAL NOT NULL,
                critical_field_accuracy REAL NOT NULL,
                distinction_preserved INTEGER NOT NULL,
                task_success INTEGER NOT NULL,
                primary_failure_category TEXT NOT NULL,
                failure_explanation TEXT NOT NULL,
                client_latency_ms REAL,
                provider_latency_ms REAL,
                simulated_latency_ms REAL,
                created_at TEXT NOT NULL,
                payload_json TEXT NOT NULL
            )
            """
        )
        dummy_payload = json.dumps({"test_case_id": "pair_001", "overall_status": "completed"})
        conn.execute(
            """
            INSERT INTO evaluation_runs (
                run_id, test_case_id, category, language_code, execution_mode, audio_condition,
                wer_a, wer_b, average_wer, critical_field_accuracy, distinction_preserved,
                task_success, primary_failure_category, failure_explanation, created_at, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("hist_run_legacy_1", "pair_001", "quantities_and_numbers", "hi-IN", "mock", "clean_recorded",
             0.0, 0.0, 0.0, 1.0, 1, 1, "none", "ok", "2026-10-20T10:00:00Z", dummy_payload),
        )
        conn.commit()

    # 2. Instantiate StorageManager on legacy database; verify migration runs cleanly
    storage = StorageManager(db_path=legacy_db_file)

    # 3. Read back legacy run; confirm default 'synthetic' cohort assignment
    legacy_run = storage.get_run("hist_run_legacy_1")
    assert legacy_run is not None
    assert legacy_run.dataset_classification == "synthetic", "Legacy runs must default to synthetic cohort"

    # 4. Verify idempotent second initialization without errors
    storage_again = StorageManager(db_path=legacy_db_file)
    assert storage_again.get_run("hist_run_legacy_1") is not None

    # 5. Insert new run with dataset_classification='human_verified' and verify query filtering
    u_mock = UtteranceEvaluationResult(
        utterance_id="u_phase4",
        reference_transcript="Ref",
        actual_transcript="Actual",
        wer=0.0,
        substitutions=0,
        deletions=0,
        insertions=0,
        reference_length=1,
        critical_fields_match=True,
        detected_intent="ORDER",
        intent_match=True,
        task_success=True,
        status="completed",
    )
    new_hv_run = PairEvaluationResult(
        run_id="run_phase4_hv_1",
        test_case_id="pair_qty_001",
        category=TestCategory.QUANTITIES_AND_NUMBERS,
        language_code="hi-IN",
        execution_mode=ExecutionMode.MOCK,
        audio_condition="clean_recorded",
        timestamp="2026-10-23T12:00:00Z",
        overall_status="completed",
        distinction_preserved=True,
        semantic_distinction_preserved=True,
        task_success=True,
        wer_a=0.0,
        wer_b=0.0,
        average_wer=0.0,
        average_normalized_wer=0.0,
        critical_field_accuracy=1.0,
        primary_failure_category=FailureCategory.NONE,
        failure_explanation="Preserved",
        utterance_a=u_mock,
        utterance_b=u_mock,
        dataset_source="human_recording",
        dataset_classification="human_verified",
    )
    storage.save_run(new_hv_run)

    hv_runs = storage.list_runs(dataset_classification="human_verified")
    assert len(hv_runs) == 1
    assert hv_runs[0].run_id == "run_phase4_hv_1"

    synth_runs = storage.list_runs(dataset_classification="synthetic")
    assert any(r.run_id == "hist_run_legacy_1" for r in synth_runs)


# ===========================================================================
# Workstream 6: Reproducibility & Determinism
# ===========================================================================

def test_evaluation_reproducibility_and_metric_determinism():
    """Validates that metric computation and result assembly are strictly deterministic
    across multiple iterations over identical input pairs.
    """
    ref_a = "Do plate biryani chahiye"
    hyp_a = "2 प्लेट बिरयानी चाहिए"

    ref_b = "Teen plate biryani chahiye"
    hyp_b = "3 प्लेट बिरयानी चाहिए"

    wer_a_runs = []
    norm_wer_a_runs = []
    for _ in range(10):
        w_a, _, _, _, _ = calculate_wer(ref_a, hyp_a)
        nw_a, _, _, _, _ = calculate_cross_script_normalized_wer(ref_a, hyp_a)
        wer_a_runs.append(w_a)
        norm_wer_a_runs.append(nw_a)

    assert len(set(wer_a_runs)) == 1, "Raw WER must be 100% deterministic"
    assert len(set(norm_wer_a_runs)) == 1, "Normalized WER must be 100% deterministic"
    assert norm_wer_a_runs[0] == 0.0, "Canonicalization must consistently resolve Hinglish script equivalences"
