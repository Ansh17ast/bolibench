"""Unit and integration tests for BoliBench Batch Benchmark Runner.

STRICT VERIFICATION CONSTRAINTS:
- ZERO real provider network calls. All STT provider calls are mocked.
- Validates offline manifest validation, dry-run, live safeguards, request accounting,
  failure isolation, resume mechanics, and historical DB record preservation.
"""

import asyncio
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import struct
import wave
import pytest

from app.config import Settings
from app.evaluation.batch_manifest import (
    AudioSourceType,
    BatchManifest,
    BatchManifestItem,
    create_sample_manifest,
    load_manifest,
    save_manifest,
    validate_manifest,
)
from app.evaluation.batch_runner import (
    BatchBenchmarkRunner,
    BatchDryRunReport,
    BatchExecutionConfig,
    LiveExecutionBlockedError,
    MAX_PILOT_PAIRS,
)
from app.evaluation.schemas import FailureCategory, TestCategory
from app.evaluation.storage import StorageManager
from app.services.audio_validator import AudioValidationError, AudioValidator
from app.services.gnani_client import GnaniClientError, GnaniPrismaClient


def _create_synthetic_pcm_wav(duration_seconds: float = 1.0, sample_rate: int = 16000, freq_mod: int = 16) -> bytes:
    """Generates standard linear PCM WAV bytes for testing."""
    num_samples = int(sample_rate * duration_seconds)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)  # Mono
        w.setsampwidth(2)  # 16-bit
        w.setframerate(sample_rate)
        # Write dummy sine with freq_mod to allow generating distinct non-identical recordings
        samples = [int(1000 * (i % freq_mod)) for i in range(num_samples)]
        raw_data = struct.pack(f"<{len(samples)}h", *samples)
        w.writeframes(raw_data)
    return buf.getvalue()


class MockGnaniClient:
    """Mock client that simulates STT responses without making external calls."""

    def __init__(self, response_map=None, should_fail_on=None):
        self.response_map = response_map or {}
        self.should_fail_on = should_fail_on or set()
        self.calls = []

    async def transcribe(
        self,
        audio_bytes: bytes,
        filename: str,
        content_type: str = "audio/wav",
        language_code: str = "hi-IN",
        preferred_language: str = "hi-IN",
        format_mode: str = "transcribe",
    ):
        self.calls.append({
            "filename": filename,
            "language_code": language_code,
            "size": len(audio_bytes),
        })

        if filename in self.should_fail_on:
            raise GnaniClientError(f"Simulated network timeout for {filename}", status_code=504)

        if filename in self.response_map:
            return self.response_map[filename]

        # Default fallback transcript
        return {
            "transcript": "परीक्षण प्रतिलेख",
            "confidence": 0.95,
            "request_id": f"mock_req_{len(self.calls)}",
            "audio_duration": 2.5,
        }


# ==============================================================================
# 1. Manifest Validation Tests
# ==============================================================================

def test_manifest_valid_pcm_audio(tmp_path: Path):
    """Verifies that standard linear PCM WAV files pass manifest validation."""
    wav_a = tmp_path / "test_a.wav"
    wav_b = tmp_path / "test_b.wav"
    wav_a.write_bytes(_create_synthetic_pcm_wav(2.0))
    wav_b.write_bytes(_create_synthetic_pcm_wav(3.0))

    item = BatchManifestItem(
        pair_id="pair_valid_001",
        category="negation",
        language_code="hi-IN",
        audio_path_a=str(wav_a),
        audio_path_b=str(wav_b),
        reference_transcript_a="ऑर्डर कैंसल करो",
        reference_transcript_b="ऑर्डर कैंसल मत करो",
        source_type=AudioSourceType.HUMAN_RECORDED,
    )
    manifest = BatchManifest(created_at="2026-10-21T00:00:00Z", items=[item])

    report = validate_manifest(manifest, base_dir=tmp_path)
    assert report.total_items == 1
    assert report.ready_count == 1
    assert len(report.invalid_items) == 0
    assert report.planned_requests_count == 2
    assert report.counts_by_source[AudioSourceType.HUMAN_RECORDED.value] == 1


def test_manifest_missing_audio_files(tmp_path: Path):
    """Verifies that missing audio files are rejected before live calls."""
    item = BatchManifestItem(
        pair_id="pair_missing_001",
        category="negation",
        language_code="hi-IN",
        audio_path_a="non_existent_a.wav",
        audio_path_b="non_existent_b.wav",
        reference_transcript_a="A",
        reference_transcript_b="B",
    )
    manifest = BatchManifest(created_at="2026-10-21T00:00:00Z", items=[item])

    report = validate_manifest(manifest, base_dir=tmp_path)
    assert report.ready_count == 0
    assert len(report.invalid_items) == 1
    assert report.invalid_items[0]["error_type"] == "AUDIO_FILE_NOT_FOUND"


def test_manifest_malformed_wav_rejection(tmp_path: Path):
    """Verifies that non-RIFF corrupted audio files are rejected."""
    corrupt_a = tmp_path / "corrupt_a.wav"
    valid_b = tmp_path / "valid_b.wav"
    corrupt_a.write_bytes(b"NOT_A_WAV_HEADER_DATA_123456789")
    valid_b.write_bytes(_create_synthetic_pcm_wav(1.0))

    item = BatchManifestItem(
        pair_id="pair_corrupt_001",
        category="command_intent",
        audio_path_a=str(corrupt_a),
        audio_path_b=str(valid_b),
        reference_transcript_a="A",
        reference_transcript_b="B",
    )
    manifest = BatchManifest(created_at="2026-10-21T00:00:00Z", items=[item])

    report = validate_manifest(manifest, base_dir=tmp_path)
    assert report.ready_count == 0
    assert len(report.invalid_items) == 1
    assert "INVALID_AUDIO_A" in report.invalid_items[0]["error_type"]


def test_manifest_duration_exceeding_limit(tmp_path: Path):
    """Verifies that audio longer than 60.0s is rejected to protect credits."""
    long_a = tmp_path / "long_a.wav"
    valid_b = tmp_path / "valid_b.wav"
    # Create WAV with duration 65 seconds
    long_a.write_bytes(_create_synthetic_pcm_wav(duration_seconds=65.0, sample_rate=8000))
    valid_b.write_bytes(_create_synthetic_pcm_wav(2.0))

    item = BatchManifestItem(
        pair_id="pair_long_001",
        category="time_extraction",
        audio_path_a=str(long_a),
        audio_path_b=str(valid_b),
        reference_transcript_a="A",
        reference_transcript_b="B",
    )
    manifest = BatchManifest(created_at="2026-10-21T00:00:00Z", items=[item])

    report = validate_manifest(manifest, base_dir=tmp_path)
    assert report.ready_count == 0
    assert len(report.invalid_items) == 1
    assert "DURATION_EXCEEDED" in report.invalid_items[0]["error_type"]


def test_manifest_duplicate_pair_id_rejection(tmp_path: Path):
    """Verifies that duplicate pair IDs are identified and rejected."""
    wav_a = tmp_path / "test_a.wav"
    wav_b = tmp_path / "test_b.wav"
    wav_a.write_bytes(_create_synthetic_pcm_wav(1.0, freq_mod=16))
    wav_b.write_bytes(_create_synthetic_pcm_wav(1.0, freq_mod=17))

    item1 = BatchManifestItem(
        pair_id="pair_dup_001",
        category="negation",
        audio_path_a=str(wav_a),
        audio_path_b=str(wav_b),
        reference_transcript_a="A1",
        reference_transcript_b="B1",
    )
    item2 = BatchManifestItem(
        pair_id="pair_dup_001",  # Same ID
        category="negation",
        audio_path_a=str(wav_a),
        audio_path_b=str(wav_b),
        reference_transcript_a="A2",
        reference_transcript_b="B2",
    )
    manifest = BatchManifest(created_at="2026-10-21T00:00:00Z", items=[item1, item2])

    report = validate_manifest(manifest, base_dir=tmp_path)
    assert report.total_items == 2
    assert report.ready_count == 1
    assert "pair_dup_001" in report.duplicated_items
    assert len(report.invalid_items) == 1


def test_manifest_missing_ground_truth(tmp_path: Path):
    """Verifies that pairs with missing ground-truth references are excluded."""
    wav_a = tmp_path / "test_a.wav"
    wav_b = tmp_path / "test_b.wav"
    wav_a.write_bytes(_create_synthetic_pcm_wav(1.0))
    wav_b.write_bytes(_create_synthetic_pcm_wav(1.0))

    item = BatchManifestItem(
        pair_id="pair_no_gt_001",
        category="named_entities",
        audio_path_a=str(wav_a),
        audio_path_b=str(wav_b),
        reference_transcript_a="",  # Empty
        reference_transcript_b="Valid reference",
    )
    manifest = BatchManifest(created_at="2026-10-21T00:00:00Z", items=[item])

    report = validate_manifest(manifest, base_dir=tmp_path)
    assert report.ready_count == 0
    assert "pair_no_gt_001" in report.missing_ground_truth_items


# ==============================================================================
# 2. Dry-Run Mode Tests
# ==============================================================================

def test_dry_run_offline_inspection(tmp_path: Path):
    """Verifies dry-run returns planned requests and consumes zero credits."""
    wav_a = tmp_path / "a.wav"
    wav_b = tmp_path / "b.wav"
    wav_a.write_bytes(_create_synthetic_pcm_wav(1.5, freq_mod=16))
    wav_b.write_bytes(_create_synthetic_pcm_wav(1.5, freq_mod=17))

    item = BatchManifestItem(
        pair_id="pair_dry_001",
        category="quantities_and_numbers",
        audio_path_a=str(wav_a),
        audio_path_b=str(wav_b),
        reference_transcript_a="Do packet",
        reference_transcript_b="Teen packet",
        source_type=AudioSourceType.HUMAN_RECORDED,
    )
    manifest = BatchManifest(created_at="2026-10-21T00:00:00Z", items=[item])

    config = BatchExecutionConfig(live=False, output_dir=tmp_path / "out")
    runner = BatchBenchmarkRunner(config=config)

    report = runner.run_dry(manifest=manifest, base_dir=tmp_path)
    assert report.total_pairs == 1
    assert report.ready_pairs == 1
    assert report.planned_requests == 2
    assert report.validation_passed is True
    assert "DRY RUN ONLY" in report.disclaimer


# ==============================================================================
# 3. Live Execution Guard Tests
# ==============================================================================

def test_live_guard_rejects_by_default():
    """Live execution must be disabled by default."""
    config = BatchExecutionConfig(live=False)
    runner = BatchBenchmarkRunner(config=config)
    with pytest.raises(LiveExecutionBlockedError, match="disabled by default"):
        runner.check_live_preflight_guards(pair_count=1)


def test_live_guard_requires_max_pairs():
    """Live execution must reject if max_pairs is not provided."""
    config = BatchExecutionConfig(live=True, max_pairs=None)
    runner = BatchBenchmarkRunner(config=config)
    with pytest.raises(LiveExecutionBlockedError, match="--max-pairs limit is strictly required"):
        runner.check_live_preflight_guards(pair_count=1)


def test_live_guard_pilot_limit_exceeded():
    """Pilot mode must strictly enforce maximum 5 pairs."""
    config = BatchExecutionConfig(live=True, max_pairs=6, pilot_mode=True)
    runner = BatchBenchmarkRunner(config=config)
    with pytest.raises(LiveExecutionBlockedError, match="Pilot execution limit exceeded"):
        runner.check_live_preflight_guards(pair_count=6)


def test_live_guard_requires_explicit_confirm():
    """Live execution must reject if --confirm is False."""
    settings = Settings(
        live_evaluation_enabled=True,
        gnani_api_key="test_dummy_key",
        gnani_token="test_dummy_token",
    )
    config = BatchExecutionConfig(live=True, max_pairs=5, confirm=False)
    runner = BatchBenchmarkRunner(config=config, settings=settings)
    with pytest.raises(LiveExecutionBlockedError, match="Explicit operator authorization confirmation required"):
        runner.check_live_preflight_guards(pair_count=5)


def test_live_guard_rejects_when_feature_flag_disabled():
    """Live execution must reject if LIVE_EVALUATION_ENABLED is False."""
    settings = Settings(
        live_evaluation_enabled=False,
        gnani_api_key="test_dummy_key",
        gnani_token="test_dummy_token",
    )
    config = BatchExecutionConfig(live=True, max_pairs=3, confirm=True)
    runner = BatchBenchmarkRunner(config=config, settings=settings)
    with pytest.raises(LiveExecutionBlockedError, match="LIVE_EVALUATION_ENABLED is false"):
        runner.check_live_preflight_guards(pair_count=3)


# ==============================================================================
# 4. Request Accounting & Sequential Failure Isolation
# ==============================================================================

@pytest.mark.asyncio
async def test_failure_isolation_skips_utterance_b(tmp_path: Path):
    """If Utterance A fails, Utterance B must be skipped to protect provider credits."""
    wav_a = tmp_path / "fail_a.wav"
    wav_b = tmp_path / "skip_b.wav"
    wav_a.write_bytes(_create_synthetic_pcm_wav(1.0, freq_mod=16))
    wav_b.write_bytes(_create_synthetic_pcm_wav(1.0, freq_mod=17))

    item = BatchManifestItem(
        pair_id="pair_fail_iso_001",
        category="negation",
        audio_path_a=str(wav_a),
        audio_path_b=str(wav_b),
        reference_transcript_a="Cancel",
        reference_transcript_b="Do not cancel",
    )
    manifest = BatchManifest(created_at="2026-10-21T00:00:00Z", items=[item])

    settings = Settings(
        live_evaluation_enabled=True,
        gnani_api_key="test_dummy_key",
        gnani_token="test_dummy_token",
    )
    config = BatchExecutionConfig(
        live=True,
        max_pairs=5,
        confirm=True,
        output_dir=tmp_path / "results",
        pacing_delay_seconds=0.0,
    )

    mock_client = MockGnaniClient(should_fail_on={"fail_a.wav"})
    runner = BatchBenchmarkRunner(config=config, settings=settings, client=mock_client)

    summary = await runner.run(manifest=manifest, base_dir=tmp_path)

    # Utterance A attempted and failed; Utterance B was never dispatched!
    assert summary["attempted_requests"] == 1
    assert summary["successful_requests"] == 0
    assert summary["failed_requests"] == 1
    assert summary["retried_requests"] == 0
    assert summary["failed_pairs"] == 1

    # Only 1 call was recorded on the client
    assert len(mock_client.calls) == 1
    assert mock_client.calls[0]["filename"] == "fail_a.wav"


@pytest.mark.asyncio
async def test_successful_evaluation_accounting(tmp_path: Path):
    """Verifies separate request counts (attempted, successful, retries=0) for successful run."""
    wav_a = tmp_path / "success_a.wav"
    wav_b = tmp_path / "success_b.wav"
    wav_a.write_bytes(_create_synthetic_pcm_wav(1.0, freq_mod=16))
    wav_b.write_bytes(_create_synthetic_pcm_wav(1.0, freq_mod=17))

    item = BatchManifestItem(
        pair_id="pair_succ_001",
        category="negation",
        audio_path_a=str(wav_a),
        audio_path_b=str(wav_b),
        reference_transcript_a="मेरा ऑर्डर कैंसल कर दो।",
        reference_transcript_b="मेरा ऑर्डर कैंसल मत करना।",
        expected_critical_fields_a={"is_negated": False},
        expected_critical_fields_b={"is_negated": True},
        distinction_field="is_negated",
    )
    manifest = BatchManifest(created_at="2026-10-21T00:00:00Z", items=[item])

    mock_responses = {
        "success_a.wav": {"transcript": "मेरा ऑर्डर कैंसल कर दो।", "request_id": "req_succ_a"},
        "success_b.wav": {"transcript": "मेरा ऑर्डर कैंसल मत करना।", "request_id": "req_succ_b"},
    }
    mock_client = MockGnaniClient(response_map=mock_responses)

    settings = Settings(
        live_evaluation_enabled=True,
        gnani_api_key="test_dummy_key",
        gnani_token="test_dummy_token",
    )
    config = BatchExecutionConfig(
        live=True,
        max_pairs=5,
        confirm=True,
        output_dir=tmp_path / "results",
        pacing_delay_seconds=0.0,
    )
    runner = BatchBenchmarkRunner(config=config, settings=settings, client=mock_client)

    summary = await runner.run(manifest=manifest, base_dir=tmp_path)

    assert summary["attempted_requests"] == 2
    assert summary["successful_requests"] == 2
    assert summary["failed_requests"] == 0
    assert summary["retried_requests"] == 0
    assert summary["completed_pairs"] == 1
    assert summary["mean_raw_wer"] == 0.0
    assert summary["distinction_preservation_rate"] == 1.0


# ==============================================================================
# 5. Persistent Resume Behavior
# ==============================================================================

@pytest.mark.asyncio
async def test_resume_skips_already_completed_pairs(tmp_path: Path):
    """Verifies resume uses state.json to avoid re-transcribing completed items."""
    wav_a1 = tmp_path / "a1.wav"
    wav_b1 = tmp_path / "b1.wav"
    wav_a2 = tmp_path / "a2.wav"
    wav_b2 = tmp_path / "b2.wav"
    wav_a1.write_bytes(_create_synthetic_pcm_wav(1.0, freq_mod=10))
    wav_b1.write_bytes(_create_synthetic_pcm_wav(1.0, freq_mod=11))
    wav_a2.write_bytes(_create_synthetic_pcm_wav(1.0, freq_mod=12))
    wav_b2.write_bytes(_create_synthetic_pcm_wav(1.0, freq_mod=13))

    item1 = BatchManifestItem(
        pair_id="pair_resume_001",
        category="negation",
        audio_path_a=str(wav_a1),
        audio_path_b=str(wav_b1),
        reference_transcript_a="A1",
        reference_transcript_b="B1",
    )
    item2 = BatchManifestItem(
        pair_id="pair_resume_002",
        category="command_intent",
        audio_path_a=str(wav_a2),
        audio_path_b=str(wav_b2),
        reference_transcript_a="A2",
        reference_transcript_b="B2",
    )
    manifest = BatchManifest(created_at="2026-10-21T00:00:00Z", items=[item1, item2])

    settings = Settings(
        live_evaluation_enabled=True,
        gnani_api_key="test_dummy_key",
        gnani_token="test_dummy_token",
    )
    out_dir = tmp_path / "resume_test_out"
    batch_id = "test_resume_batch_123"

    # Pre-populate state file marking item1 as completed
    out_dir.mkdir(parents=True, exist_ok=True)
    state_file = out_dir / f"{batch_id}_state.json"
    with open(state_file, "w", encoding="utf-8") as f:
        json.dump({
            "batch_id": batch_id,
            "items": {
                "pair_resume_001": {
                    "item_id": f"{batch_id}_pair_resume_001",
                    "pair_id": "pair_resume_001",
                    "status": "completed",
                    "attempted_calls": 2,
                    "successful_calls": 2,
                    "failed_calls": 0,
                    "average_raw_wer": 0.0,
                    "average_normalized_wer": 0.0,
                    "critical_field_accuracy": 1.0,
                    "distinction_preserved": True,
                    "task_success": True,
                }
            }
        }, f)

    mock_client = MockGnaniClient()
    config = BatchExecutionConfig(
        live=True,
        max_pairs=5,
        confirm=True,
        resume=True,
        output_dir=out_dir,
        batch_id=batch_id,
        pacing_delay_seconds=0.0,
    )
    runner = BatchBenchmarkRunner(config=config, settings=settings, client=mock_client)

    summary = await runner.run(manifest=manifest, base_dir=tmp_path)

    # Item1 was skipped; only Item2 (2 calls) was dispatched to provider!
    assert len(mock_client.calls) == 2
    assert mock_client.calls[0]["filename"] == "a2.wav"
    assert mock_client.calls[1]["filename"] == "b2.wav"
    assert summary["total_pairs"] == 2
    assert summary["completed_pairs"] == 2


# ==============================================================================
# 6. Historical Record Preservation
# ==============================================================================

def test_historical_database_preservation(tmp_path: Path):
    """Verifies batch storage operations do not mutate or delete existing evaluation records."""
    db_path = tmp_path / "test_bolibench.db"
    storage = StorageManager(db_path=db_path)

    # 1. Insert an existing historical run in evaluation_runs
    from app.evaluation.schemas import ExecutionMode, PairEvaluationResult, UtteranceEvaluationResult
    historical_run = PairEvaluationResult(
        run_id="eval_historical_001",
        test_case_id="pair_qty_001",
        category=TestCategory.QUANTITIES_AND_NUMBERS.value,
        language_code="hi-IN",
        execution_mode=ExecutionMode.LIVE,
        audio_condition="clean_recorded",
        timestamp="2026-10-20T12:00:00Z",
        wer_a=0.05,
        wer_b=0.05,
        average_wer=0.05,
        critical_field_accuracy=1.0,
        semantic_distinction_preserved=True,
        task_success=True,
        primary_failure_category=FailureCategory.NONE,
        failure_explanation="No failure",
        utterance_a=UtteranceEvaluationResult(
            utterance_id="pair_qty_001_a",
            reference_transcript="Do packet",
            actual_transcript="Do packet",
            wer=0.0,
            substitutions=0,
            deletions=0,
            insertions=0,
            reference_length=2,
            critical_fields_match=True,
            detected_intent="ORDER",
            intent_match=True,
            task_success=True,
        ),
        utterance_b=UtteranceEvaluationResult(
            utterance_id="pair_qty_001_b",
            reference_transcript="Teen packet",
            actual_transcript="Teen packet",
            wer=0.0,
            substitutions=0,
            deletions=0,
            insertions=0,
            reference_length=2,
            critical_fields_match=True,
            detected_intent="ORDER",
            intent_match=True,
            task_success=True,
        ),
    )
    storage.save_run(historical_run)

    # 2. Perform batch benchmark operations
    storage.save_batch_run({
        "batch_id": "batch_test_storage_999",
        "name": "Storage Preservation Test",
        "execution_mode": "dry_run",
        "created_at": "2026-10-21T00:00:00Z",
        "finished_at": "2026-10-21T00:01:00Z",
        "total_pairs": 5,
        "completed_pairs": 5,
        "failed_pairs": 0,
        "incomplete_pairs": 0,
        "skipped_pairs": 0,
        "attempted_requests": 0,
        "successful_requests": 0,
        "failed_requests": 0,
        "retried_requests": 0,
    })

    # 3. Verify historical record is 100% intact and unchanged
    loaded_run = storage.get_run("eval_historical_001")
    assert loaded_run is not None
    assert loaded_run.run_id == "eval_historical_001"
    assert loaded_run.average_wer == 0.05
    assert loaded_run.semantic_distinction_preserved is True

    # 4. Verify batch table holds new record separately
    batch_run = storage.get_batch_run("batch_test_storage_999")
    assert batch_run is not None
    assert batch_run["batch_id"] == "batch_test_storage_999"
    assert batch_run["total_pairs"] == 5


# ==============================================================================
# 7. Partial Failure, Metric Integrity, and Modality Separation Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_partial_failure_utterance_b_fails(tmp_path: Path):
    """When Utterance A succeeds but Utterance B fails, the run is incomplete and Utterance A is preserved."""
    wav_a = tmp_path / "part_a.wav"
    wav_b = tmp_path / "part_b.wav"
    wav_a.write_bytes(_create_synthetic_pcm_wav(1.0, freq_mod=16))
    wav_b.write_bytes(_create_synthetic_pcm_wav(1.0, freq_mod=17))

    item = BatchManifestItem(
        pair_id="pair_part_001",
        category="named_entities",
        audio_path_a=str(wav_a),
        audio_path_b=str(wav_b),
        reference_transcript_a="Rahul",
        reference_transcript_b="Rohit",
    )
    manifest = BatchManifest(created_at="2026-10-21T00:00:00Z", items=[item])

    settings = Settings(
        live_evaluation_enabled=True,
        gnani_api_key="test_dummy_key",
        gnani_token="test_dummy_token",
    )
    config = BatchExecutionConfig(
        live=True,
        max_pairs=5,
        confirm=True,
        output_dir=tmp_path / "results_part",
        pacing_delay_seconds=0.0,
    )

    mock_client = MockGnaniClient(
        response_map={"part_a.wav": {"transcript": "Rahul", "request_id": "req_a"}},
        should_fail_on={"part_b.wav"},
    )
    runner = BatchBenchmarkRunner(config=config, settings=settings, client=mock_client)

    summary = await runner.run(manifest=manifest, base_dir=tmp_path)

    assert summary["attempted_requests"] == 2
    assert summary["successful_requests"] == 1
    assert summary["failed_requests"] == 1
    assert summary["retried_requests"] == 0
    assert summary["incomplete_pairs"] == 1
    assert summary["completed_pairs"] == 0


@pytest.mark.asyncio
async def test_modality_separation_and_secret_safety(tmp_path: Path):
    """Verifies human, synthetic, and transformed variants are separated, and no secrets leak into artifacts."""
    wav_h_a = tmp_path / "h_a.wav"
    wav_h_b = tmp_path / "h_b.wav"
    wav_s_a = tmp_path / "s_a.wav"
    wav_s_b = tmp_path / "s_b.wav"
    wav_t_a = tmp_path / "t_a.wav"
    wav_t_b = tmp_path / "t_b.wav"
    wav_h_a.write_bytes(_create_synthetic_pcm_wav(1.0, freq_mod=10))
    wav_h_b.write_bytes(_create_synthetic_pcm_wav(1.0, freq_mod=11))
    wav_s_a.write_bytes(_create_synthetic_pcm_wav(1.0, freq_mod=12))
    wav_s_b.write_bytes(_create_synthetic_pcm_wav(1.0, freq_mod=13))
    wav_t_a.write_bytes(_create_synthetic_pcm_wav(1.0, freq_mod=14))
    wav_t_b.write_bytes(_create_synthetic_pcm_wav(1.0, freq_mod=15))

    items = [
        BatchManifestItem(
            pair_id="pair_human",
            category="quantities_and_numbers",
            audio_path_a=str(wav_h_a),
            audio_path_b=str(wav_h_b),
            reference_transcript_a="Do packet",
            reference_transcript_b="Teen packet",
            source_type=AudioSourceType.HUMAN_RECORDED,
            expected_critical_fields_a={"qty": 2},
            expected_critical_fields_b={"qty": 3},
            distinction_field="qty",
        ),
        BatchManifestItem(
            pair_id="pair_tts",
            category="quantities_and_numbers",
            audio_path_a=str(wav_s_a),
            audio_path_b=str(wav_s_b),
            reference_transcript_a="Char packet",
            reference_transcript_b="Paanch packet",
            source_type=AudioSourceType.TTS_SYNTHETIC,
            expected_critical_fields_a={"qty": 4},
            expected_critical_fields_b={"qty": 5},
            distinction_field="qty",
        ),
        BatchManifestItem(
            pair_id="pair_trans",
            category="quantities_and_numbers",
            audio_path_a=str(wav_t_a),
            audio_path_b=str(wav_t_b),
            reference_transcript_a="Chhah packet",
            reference_transcript_b="Saat packet",
            source_type=AudioSourceType.ACOUSTIC_TRANSFORMED,
            expected_critical_fields_a={"qty": 6},
            expected_critical_fields_b={"qty": 7},
            distinction_field="qty",
        ),
    ]
    manifest = BatchManifest(created_at="2026-10-21T00:00:00Z", items=items)

    secret_key = "SECRET_SUPER_CONFIDENTIAL_KEY_9999"
    secret_token = "SECRET_SUPER_CONFIDENTIAL_TOKEN_8888"
    settings = Settings(
        live_evaluation_enabled=True,
        gnani_api_key=secret_key,
        gnani_token=secret_token,
    )
    out_dir = tmp_path / "artifacts_test"
    batch_id = "batch_secrets_audit"
    config = BatchExecutionConfig(
        live=True,
        max_pairs=5,
        confirm=True,
        output_dir=out_dir,
        batch_id=batch_id,
        pacing_delay_seconds=0.0,
    )

    mock_client = MockGnaniClient()
    runner = BatchBenchmarkRunner(config=config, settings=settings, client=mock_client)

    summary = await runner.run(manifest=manifest, base_dir=tmp_path)

    # 1. Modality separation verified
    pbreak = summary["provenance_breakdown"]
    assert AudioSourceType.HUMAN_RECORDED.value in pbreak
    assert AudioSourceType.TTS_SYNTHETIC.value in pbreak
    assert AudioSourceType.ACOUSTIC_TRANSFORMED.value in pbreak
    assert pbreak[AudioSourceType.HUMAN_RECORDED.value]["count"] == 1
    assert pbreak[AudioSourceType.TTS_SYNTHETIC.value]["count"] == 1
    assert pbreak[AudioSourceType.ACOUSTIC_TRANSFORMED.value]["count"] == 1

    # 2. Disclaimer present
    assert "TTS-generated and transformed audio variants do not establish performance" in summary["disclaimer"]

    # 3. Secret leak audit in files
    csv_content = (out_dir / f"{batch_id}_results.csv").read_text(encoding="utf-8")
    json_content = (out_dir / f"{batch_id}_results.json").read_text(encoding="utf-8")
    md_content = (out_dir / f"{batch_id}_summary.md").read_text(encoding="utf-8")
    state_content = (out_dir / f"{batch_id}_state.json").read_text(encoding="utf-8")

    for f_content in [csv_content, json_content, md_content, state_content]:
        assert secret_key not in f_content
        assert secret_token not in f_content


# ==============================================================================
# 8. Enhanced Offline Validation Tests (Identical Hashes, PCM, Refs, Fields)
# ==============================================================================

def test_manifest_identical_file_hashes_rejected(tmp_path: Path):
    """Verifies that pairs with identical file content are rejected."""
    wav_a = tmp_path / "identical_a.wav"
    wav_b = tmp_path / "identical_b.wav"
    audio_data = _create_synthetic_pcm_wav(1.0)
    wav_a.write_bytes(audio_data)
    wav_b.write_bytes(audio_data)

    item = BatchManifestItem(
        pair_id="pair_ident_files_001",
        category="negation",
        audio_path_a=str(wav_a),
        audio_path_b=str(wav_b),
        reference_transcript_a="Cancel order",
        reference_transcript_b="Do not cancel order",
        distinction_field="is_negated",
        expected_critical_fields_a={"is_negated": False},
        expected_critical_fields_b={"is_negated": True},
    )
    manifest = BatchManifest(created_at="2026-10-21T00:00:00Z", items=[item])

    report = validate_manifest(manifest, base_dir=tmp_path)
    assert report.ready_count == 0
    assert len(report.invalid_items) == 1
    assert report.invalid_items[0]["error_type"] == "IDENTICAL_AUDIO_FILES"
    assert "Recordings must be distinct" in report.invalid_items[0]["message"]


def test_manifest_identical_pcm_samples_differing_wav_metadata(tmp_path: Path):
    """Verifies that pairs with identical decoded PCM frames but differing RIFF metadata are rejected."""
    wav_a = tmp_path / "meta_a.wav"
    wav_b = tmp_path / "meta_b.wav"
    data_a = _create_synthetic_pcm_wav(1.0)
    # Append a JUNK chunk to data_b so file hash is different but decoded PCM samples are identical
    junk_chunk = b"JUNK\x04\x00\x00\x00TEST"
    riff_size = len(data_a) - 8 + len(junk_chunk)
    data_b = data_a[:4] + struct.pack("<I", riff_size) + data_a[8:] + junk_chunk

    wav_a.write_bytes(data_a)
    wav_b.write_bytes(data_b)

    import hashlib
    assert hashlib.sha256(data_a).hexdigest() != hashlib.sha256(data_b).hexdigest()

    item = BatchManifestItem(
        pair_id="pair_ident_pcm_001",
        category="quantities_and_numbers",
        audio_path_a=str(wav_a),
        audio_path_b=str(wav_b),
        reference_transcript_a="Do packet",
        reference_transcript_b="Teen packet",
        distinction_field="quantity",
        expected_critical_fields_a={"quantity": 2},
        expected_critical_fields_b={"quantity": 3},
    )
    manifest = BatchManifest(created_at="2026-10-21T00:00:00Z", items=[item])

    report = validate_manifest(manifest, base_dir=tmp_path)
    assert report.ready_count == 0
    assert len(report.invalid_items) == 1
    assert report.invalid_items[0]["error_type"] == "IDENTICAL_PCM_AUDIO"
    assert "identical decoded PCM audio samples" in report.invalid_items[0]["message"]


def test_manifest_identical_reference_transcripts_rejected(tmp_path: Path):
    """Verifies that pairs with identical normalized reference transcripts are rejected."""
    wav_a = tmp_path / "a.wav"
    wav_b = tmp_path / "b.wav"
    wav_a.write_bytes(_create_synthetic_pcm_wav(1.0, freq_mod=16))
    wav_b.write_bytes(_create_synthetic_pcm_wav(1.0, freq_mod=17))

    item = BatchManifestItem(
        pair_id="pair_ident_ref_001",
        category="negation",
        audio_path_a=str(wav_a),
        audio_path_b=str(wav_b),
        reference_transcript_a="Cancel my order",
        reference_transcript_b="  cancel   MY  ORDER  ",  # normalized identical
        distinction_field="is_negated",
    )
    manifest = BatchManifest(created_at="2026-10-21T00:00:00Z", items=[item])

    report = validate_manifest(manifest, base_dir=tmp_path)
    assert report.ready_count == 0
    assert len(report.invalid_items) == 1
    assert report.invalid_items[0]["error_type"] == "IDENTICAL_REFERENCE_TRANSCRIPTS"


def test_manifest_missing_or_contradictory_critical_fields(tmp_path: Path):
    """Verifies detection of missing distinction_field and contradictory business fields."""
    wav_a = tmp_path / "a.wav"
    wav_b = tmp_path / "b.wav"
    wav_a.write_bytes(_create_synthetic_pcm_wav(1.0, freq_mod=16))
    wav_b.write_bytes(_create_synthetic_pcm_wav(1.0, freq_mod=17))

    # Case 1: missing distinction_field when critical fields are specified
    item1 = BatchManifestItem(
        pair_id="pair_missing_df_001",
        category="negation",
        audio_path_a=str(wav_a),
        audio_path_b=str(wav_b),
        reference_transcript_a="Cancel order",
        reference_transcript_b="Do not cancel",
        expected_critical_fields_a={"is_negated": False},
        expected_critical_fields_b={"is_negated": True},
        distinction_field="",
    )
    report1 = validate_manifest(BatchManifest(created_at="2026-10-21T00:00:00Z", items=[item1]), base_dir=tmp_path)
    assert len(report1.invalid_items) == 1
    assert report1.invalid_items[0]["error_type"] == "MISSING_DISTINCTION_FIELD"

    # Case 2: contradictory quantity
    item2 = BatchManifestItem(
        pair_id="pair_contra_qty_001",
        category="quantities_and_numbers",
        audio_path_a=str(wav_a),
        audio_path_b=str(wav_b),
        reference_transcript_a="Do packet",
        reference_transcript_b="Teen packet",
        distinction_field="quantity",
        expected_critical_fields_a={"quantity": 2},
        expected_critical_fields_b={"quantity": 2},  # Contradiction: same quantity
    )
    report2 = validate_manifest(BatchManifest(created_at="2026-10-21T00:00:00Z", items=[item2]), base_dir=tmp_path)
    assert len(report2.invalid_items) == 1
    assert report2.invalid_items[0]["error_type"] == "CONTRADICTORY_CRITICAL_FIELDS"

    # Case 3: contradictory named entity
    item3 = BatchManifestItem(
        pair_id="pair_contra_ent_001",
        category="named_entities",
        audio_path_a=str(wav_a),
        audio_path_b=str(wav_b),
        reference_transcript_a="Pay Aarav",
        reference_transcript_b="Pay Vikram",
        distinction_field="name",
        expected_critical_fields_a={"name": "Aarav"},
        expected_critical_fields_b={"name": "Aarav"},  # Contradiction: same name
    )
    report3 = validate_manifest(BatchManifest(created_at="2026-10-21T00:00:00Z", items=[item3]), base_dir=tmp_path)
    assert len(report3.invalid_items) == 1
    assert report3.invalid_items[0]["error_type"] == "CONTRADICTORY_CRITICAL_FIELDS"


def test_manifest_high_lexical_overlap_manual_review_not_rejected(tmp_path: Path):
    """Verifies that high lexical overlap is flagged for manual audit but NOT rejected."""
    wav_a = tmp_path / "a.wav"
    wav_b = tmp_path / "b.wav"
    wav_a.write_bytes(_create_synthetic_pcm_wav(1.0, freq_mod=16))
    wav_b.write_bytes(_create_synthetic_pcm_wav(1.5, freq_mod=17))

    item = BatchManifestItem(
        pair_id="pair_overlap_001",
        category="negation",
        audio_path_a=str(wav_a),
        audio_path_b=str(wav_b),
        reference_transcript_a="mera order cancel kar do",
        reference_transcript_b="mera order cancel mat kar do",
        distinction_field="is_negated",
        expected_critical_fields_a={"is_negated": False},
        expected_critical_fields_b={"is_negated": True},
    )
    manifest = BatchManifest(created_at="2026-10-21T00:00:00Z", items=[item])

    report = validate_manifest(manifest, base_dir=tmp_path)
    assert report.is_valid is True
    assert report.ready_count == 1
    assert len(report.invalid_items) == 0
    # Flagged in manual_review_items due to lexical overlap >= 75%
    assert len(report.manual_review_items) >= 1
    assert report.manual_review_items[0]["review_type"] == "HIGH_LEXICAL_OVERLAP"


# ==============================================================================
# 9. BenchmarkRunner Checkpoint Resume & Safeguards (Offline Mocked)
# ==============================================================================

@pytest.mark.asyncio
async def test_benchmark_runner_checkpoint_resume_and_safeguards(tmp_path: Path, monkeypatch):
    """Verifies scripts/benchmark_runner.py resume skipping, request accounting, and ID reuse."""
    from scripts.benchmark_runner import BenchmarkRunner

    # 1. Setup mock workspace directory with audio
    audio_dir = tmp_path / "benchmark_audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = tmp_path / "benchmark_checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # Create 4 pilot pairs + 1 repaired pair + 1 additional pair
    pairs_spec = [
        ("bench_qty_001", "quantities_and_numbers", "Do packet", "Teen packet", {"quantity": 2}, {"quantity": 3}, "quantity"),
        ("bench_neg_001", "negation", "Cancel order", "Do not cancel", {"is_negated": False}, {"is_negated": True}, "is_negated"),
        ("bench_cor_001", "corrections", "Two items", "Not two, three items", {"quantity": 2}, {"quantity": 3}, "quantity"),
        ("bench_cmd_001", "command_intent", "Return item", "Modify item", {"intent": "return"}, {"intent": "modify"}, "intent"),
        ("bench_ent_001_v2", "named_entities", "Pay Aarav", "Pay Vikram", {"name": "Aarav"}, {"name": "Vikram"}, "name"),
        ("bench_tim_001", "time_extraction", "Deliver at 2pm", "Deliver at 5pm", {"time": "2pm"}, {"time": "5pm"}, "time"),
    ]

    items = []
    for idx, (pid, cat, ref_a, ref_b, fa, fb, df) in enumerate(pairs_spec):
        wav_a = audio_dir / f"{pid}_a.wav"
        wav_b = audio_dir / f"{pid}_b.wav"
        wav_a.write_bytes(_create_synthetic_pcm_wav(1.0, freq_mod=10 + idx * 2))
        wav_b.write_bytes(_create_synthetic_pcm_wav(1.0, freq_mod=11 + idx * 2))
        items.append(
            BatchManifestItem(
                pair_id=pid,
                category=cat,
                audio_path_a=f"benchmark_audio/{pid}_a.wav",
                audio_path_b=f"benchmark_audio/{pid}_b.wav",
                reference_transcript_a=ref_a,
                reference_transcript_b=ref_b,
                expected_critical_fields_a=fa,
                expected_critical_fields_b=fb,
                distinction_field=df,
            )
        )

    manifest = BatchManifest(
        manifest_version="1.1.0",
        name="Test 6-Pair Manifest",
        created_at="2026-10-21T00:00:00Z",
        items=items,
    )
    manifest_file = tmp_path / "benchmark_manifest.json"
    save_manifest(manifest, manifest_file)

    # 2. Seed a historical pilot checkpoint with the 4 pilot pairs + old bench_ent_001
    historical_cp_file = checkpoints_dir / "batch_pilot_historical_checkpoint.json"
    pilot_checkpoint_data = {}
    for pid in ["bench_qty_001", "bench_neg_001", "bench_cor_001", "bench_cmd_001", "bench_ent_001"]:
        pilot_checkpoint_data[f"{pid}_a"] = {
            "utterance_id": f"{pid}_a",
            "pair_id": pid,
            "side": "a",
            "status": "completed",
            "request_id": f"pilot_req_{pid}_a",
            "actual_transcript": f"mock transcript for {pid} a",
            "client_elapsed_ms": 100.0,
            "provider_processing_ms": None,
            "wer_raw": 0.0,
            "wer_normalized": 0.0,
            "resumed": False,
        }
        pilot_checkpoint_data[f"{pid}_b"] = {
            "utterance_id": f"{pid}_b",
            "pair_id": pid,
            "side": "b",
            "status": "completed",
            "request_id": f"pilot_req_{pid}_b",
            "actual_transcript": f"mock transcript for {pid} b",
            "client_elapsed_ms": 105.0,
            "provider_processing_ms": None,
            "wer_raw": 0.0,
            "wer_normalized": 0.0,
            "resumed": False,
        }
    with open(historical_cp_file, "w", encoding="utf-8") as f:
        json.dump(pilot_checkpoint_data, f, indent=2)

    # 3. Instantiate BenchmarkRunner pointing to tmp_path
    monkeypatch.setattr("scripts.benchmark_runner.WORKSPACE_DIR", tmp_path)
    monkeypatch.setattr("scripts.benchmark_runner.DRY_RUN_REPORT_PATH", tmp_path / "benchmark_dry_run_report.json")

    runner = BenchmarkRunner(
        manifest_path=manifest_file,
        checkpoint_dir=checkpoints_dir,
        pacing_seconds=0.0,
    )
    runner.settings = Settings(
        live_evaluation_enabled=True,
        gnani_api_key="mock_test_key",
        gnani_token="mock_test_token",
    )

    # 4. Test dry-run accounting
    dry_data = runner.execute_dry_run()
    assert dry_data["total_pairs"] == 6
    assert dry_data["ready_pairs"] == 6
    assert dry_data["completed_pilot_pairs_count"] == 4
    assert set(dry_data["completed_pilot_pair_ids"]) == {
        "bench_qty_001", "bench_neg_001", "bench_cor_001", "bench_cmd_001"
    }
    assert dry_data["untested_pairs_remaining"] == 2  # bench_ent_001_v2 and bench_tim_001
    assert dry_data["planned_new_provider_requests"] == 4  # 2 pairs * 2 requests

    # 5. Test mocked live batch execution
    mock_client = MockGnaniClient()
    monkeypatch.setattr("scripts.benchmark_runner.GnaniPrismaClient", lambda settings: mock_client)

    summary = await runner.execute_batch(mode="live", max_pairs=50, authorize_live=True)

    # Verifications:
    # A. 4 completed pilot pairs were resumed with 0 requests made (8 resumed calls)
    assert summary["requests_resumed"] == 8
    # B. Exactly 4 new provider requests were dispatched (for bench_ent_001_v2 and bench_tim_001)
    assert summary["requests_dispatched"] == 4
    assert len(mock_client.calls) == 4
    called_files = [c["filename"] for c in mock_client.calls]
    assert "bench_ent_001_v2_a.wav" in called_files
    assert "bench_ent_001_v2_b.wav" in called_files
    assert "bench_tim_001_a.wav" in called_files
    assert "bench_tim_001_b.wav" in called_files
    # C. Old invalid pair bench_ent_001 was NEVER dispatched
    assert "bench_ent_001_a.wav" not in called_files
    assert "bench_ent_001_b.wav" not in called_files
    # D. Preserved historical request IDs from pilot are retained in pair scores
    qty_pair = next(p for p in summary["pairs"] if p["test_case_id"] == "bench_qty_001")
    assert qty_pair["request_id_a"] == "pilot_req_bench_qty_001_a"
    assert qty_pair["request_id_b"] == "pilot_req_bench_qty_001_b"


@pytest.mark.asyncio
async def test_benchmark_runner_partial_resume_and_failure_isolation(tmp_path: Path, monkeypatch):
    """Verifies that an incomplete pair resumes Utterance B without re-dispatching A, and failure of A skips B."""
    from scripts.benchmark_runner import BenchmarkRunner

    audio_dir = tmp_path / "benchmark_audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = tmp_path / "benchmark_checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # Create Pair 1 (where A is already done, B needs to run)
    wav_1a = audio_dir / "bench_p1_a.wav"
    wav_1b = audio_dir / "bench_p1_b.wav"
    wav_1a.write_bytes(_create_synthetic_pcm_wav(1.0, freq_mod=10))
    wav_1b.write_bytes(_create_synthetic_pcm_wav(1.0, freq_mod=11))

    # Create Pair 2 (where A will fail, so B must be skipped)
    wav_2a = audio_dir / "bench_p2_a.wav"
    wav_2b = audio_dir / "bench_p2_b.wav"
    wav_2a.write_bytes(_create_synthetic_pcm_wav(1.0, freq_mod=12))
    wav_2b.write_bytes(_create_synthetic_pcm_wav(1.0, freq_mod=13))

    items = [
        BatchManifestItem(
            pair_id="bench_p1",
            category="negation",
            audio_path_a="benchmark_audio/bench_p1_a.wav",
            audio_path_b="benchmark_audio/bench_p1_b.wav",
            reference_transcript_a="Cancel order",
            reference_transcript_b="Do not cancel",
            expected_critical_fields_a={"is_negated": False},
            expected_critical_fields_b={"is_negated": True},
            distinction_field="is_negated",
        ),
        BatchManifestItem(
            pair_id="bench_p2",
            category="command_intent",
            audio_path_a="benchmark_audio/bench_p2_a.wav",
            audio_path_b="benchmark_audio/bench_p2_b.wav",
            reference_transcript_a="Return item",
            reference_transcript_b="Modify item",
            expected_intent_a="return",
            expected_intent_b="modify",
            distinction_field="intent",
        ),
    ]

    manifest = BatchManifest(manifest_version="1.1.0", created_at="2026-10-21T00:00:00Z", items=items)
    manifest_file = tmp_path / "benchmark_manifest.json"
    save_manifest(manifest, manifest_file)

    # Seed checkpoint with bench_p1_a completed, but bench_p1_b incomplete
    cp_file = checkpoints_dir / "batch_partial_checkpoint.json"
    with open(cp_file, "w", encoding="utf-8") as f:
        json.dump({
            "bench_p1_a": {
                "utterance_id": "bench_p1_a",
                "pair_id": "bench_p1",
                "side": "a",
                "status": "completed",
                "request_id": "req_saved_p1_a",
                "actual_transcript": "Cancel order",
                "client_elapsed_ms": 50.0,
                "provider_processing_ms": None,
                "wer_raw": 0.0,
                "wer_normalized": 0.0,
                "resumed": False,
            }
        }, f)

    monkeypatch.setattr("scripts.benchmark_runner.WORKSPACE_DIR", tmp_path)
    runner = BenchmarkRunner(
        manifest_path=manifest_file,
        checkpoint_dir=checkpoints_dir,
        pacing_seconds=0.0,
    )
    runner.settings = Settings(
        live_evaluation_enabled=True,
        gnani_api_key="mock_test_key",
        gnani_token="mock_test_token",
    )

    # Mock client configured so bench_p2_a fails
    mock_client = MockGnaniClient(should_fail_on={"bench_p2_a.wav"})
    monkeypatch.setattr("scripts.benchmark_runner.GnaniPrismaClient", lambda settings: mock_client)

    summary = await runner.execute_batch(mode="live", max_pairs=50, authorize_live=True)

    # bench_p1_a was resumed (0 calls)
    # bench_p1_b was dispatched (1 call)
    # bench_p2_a was dispatched and failed (1 call)
    # bench_p2_b was dispatched independently (1 call)
    assert len(mock_client.calls) == 3
    assert mock_client.calls[0]["filename"] == "bench_p1_b.wav"
    assert mock_client.calls[1]["filename"] == "bench_p2_a.wav"
    assert mock_client.calls[2]["filename"] == "bench_p2_b.wav"
    assert summary["requests_resumed"] == 1  # bench_p1_a
    assert summary["requests_dispatched"] == 3  # bench_p1_b, bench_p2_a, bench_p2_b


def test_benchmark_runner_v2_batch_selection_and_guards(tmp_path: Path):
    """Verifies that BenchmarkRunner correctly handles v2 batching, namespaces, and blocks v1 reruns."""
    from scripts.benchmark_runner import BenchmarkRunner, WORKSPACE_DIR

    # 1. Test safety block when batch_index is used with historical manifest
    v1_manifest_file = tmp_path / "benchmark_manifest.json"
    v1_manifest_file.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="Historical v1 manifest cannot be run with batch indexing"):
        BenchmarkRunner(manifest_path=v1_manifest_file, batch_index=1)

    # 2. Test dedicated checkpoint dir for v2
    v2_manifest_file = tmp_path / "benchmark_manifest_v2.0.0.json"
    v2_manifest_file.write_text("{}", encoding="utf-8")
    runner = BenchmarkRunner(manifest_path=v2_manifest_file, batch_index=1)
    assert runner.is_v2 is True
    assert runner.checkpoint_dir.name == "v2"
    assert "benchmark_checkpoints" in str(runner.checkpoint_dir)


def test_benchmark_runner_v2_all_12_batches_partition_600_pairs():
    """Verifies that stratified batch selection partitions all 600 v2 pairs across 12 batches of 50 pairs without duplicates."""
    from scripts.benchmark_runner import BenchmarkRunner, WORKSPACE_DIR
    from app.evaluation.batch_manifest import load_manifest

    v2_manifest_path = WORKSPACE_DIR / "benchmark_manifest_v2.0.0.json"
    manifest = load_manifest(v2_manifest_path)
    runner = BenchmarkRunner(manifest_path=v2_manifest_path, batch_index=1)

    all_selected_ids = []
    for b in range(1, 13):
        batch_items = runner._select_batch_items(manifest.items, batch_index=b, batch_size=50, strategy="stratified")
        assert len(batch_items) == 50, f"Batch {b} expected 50 items, got {len(batch_items)}"

        # Verify all items are bench_v2_*
        for it in batch_items:
            assert it.pair_id.startswith("bench_v2_")
            assert not (it.pair_id.startswith("bench_qty_") or it.pair_id.startswith("bench_ent_"))

        all_selected_ids.extend([it.pair_id for it in batch_items])

    # Total must be exactly 600 unique items
    assert len(all_selected_ids) == 600
    assert len(set(all_selected_ids)) == 600
    manifest_ids = set(it.pair_id for it in manifest.items)
    assert set(all_selected_ids) == manifest_ids
