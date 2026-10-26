"""Isolated regression tests for evaluation resumption in BoliBench.

Verifies:
1. Resumption of incomplete live evaluation reuses completed Utterance A with 0 duplicate provider calls.
2. Only the failed utterance is retried, preventing duplicate credit consumption.
3. Resuming an already-completed run is idempotent and makes 0 provider calls.
4. Resuming with mismatched or invalid run ID is rejected.
5. Concurrent resumption of the same run ID is prevented.
6. Resumption works for human-recorded contrast pairs via evaluate_human_pair_live.
"""

from unittest.mock import AsyncMock, patch
import pytest

from app.evaluation.engine import EvaluationEngine
from app.evaluation.storage import StorageManager
from app.services.gnani_client import GnaniPrismaClient, GnaniTimeoutError
from app.services.recording_storage import RecordingStorage
from tests.test_human_recording_studio import create_pcm_wav_bytes


@pytest.fixture
def test_wav_bytes():
    return create_pcm_wav_bytes(duration_seconds=5.0)


@pytest.mark.asyncio
async def test_resumption_reuses_completed_utterance_a(tmp_path, test_wav_bytes):
    """When Utterance B previously failed, resuming reuses Utterance A and only calls B."""
    storage = StorageManager(db_path=tmp_path / "eval_resume_test.db")
    engine = EvaluationEngine(storage_manager=storage)

    mock_client = AsyncMock(spec=GnaniPrismaClient)
    # Attempt 1: Utterance A succeeds, Utterance B times out
    mock_client.transcribe.side_effect = [
        {
            "success": True,
            "transcript": "Do packet chahiye",
            "request_id": "req_first_attempt_a_111",
            "client_request_elapsed_ms": 120.0,
            "provider_processing_time_ms": 50.0,
        },
        GnaniTimeoutError(timeout_seconds=30.0),
    ]

    # Run 1: Should fail and persist incomplete run
    with pytest.raises(GnaniTimeoutError):
        await engine.evaluate_live_audio(
            test_case_id="pair_qty_001",
            audio_a_bytes=test_wav_bytes,
            filename_a="utt_a.wav",
            content_type_a="audio/wav",
            audio_b_bytes=test_wav_bytes,
            filename_b="utt_b.wav",
            content_type_b="audio/wav",
            client=mock_client,
        )

    assert mock_client.transcribe.call_count == 2
    runs = storage.list_runs()
    assert len(runs) == 1
    incomplete_run = runs[0]
    assert incomplete_run.overall_status == "incomplete"
    assert incomplete_run.utterance_a.status == "completed"
    assert incomplete_run.utterance_a.request_id == "req_first_attempt_a_111"
    assert incomplete_run.utterance_b.status == "failed"

    # Attempt 2: Resume the incomplete evaluation
    # Setup mock for resumption: only Utterance B should be called!
    mock_client.reset_mock()
    mock_client.transcribe.side_effect = [
        {
            "success": True,
            "transcript": "Teen packet chahiye",
            "request_id": "req_resumed_attempt_b_222",
            "client_request_elapsed_ms": 130.0,
            "provider_processing_time_ms": 55.0,
        }
    ]

    resumed_result = await engine.evaluate_live_audio(
        test_case_id="pair_qty_001",
        audio_a_bytes=test_wav_bytes,
        filename_a="utt_a.wav",
        content_type_a="audio/wav",
        audio_b_bytes=test_wav_bytes,
        filename_b="utt_b.wav",
        content_type_b="audio/wav",
        client=mock_client,
        resume_run_id=incomplete_run.run_id,
    )

    # CRUCIAL ASSERTION: transcribe was called exactly ONCE (for Utterance B only)!
    assert mock_client.transcribe.call_count == 1
    # Verify the call was for Utterance B
    call_args = mock_client.transcribe.call_args[1]
    assert call_args["filename"] == "utt_b.wav"

    # Verify completed result preserved Utterance A's request ID and combined with Utterance B
    assert resumed_result.run_id == incomplete_run.run_id
    assert resumed_result.overall_status == "completed"
    assert resumed_result.utterance_a.request_id == "req_first_attempt_a_111"
    assert resumed_result.utterance_a.actual_transcript == "Do packet chahiye"
    assert resumed_result.utterance_b.request_id == "req_resumed_attempt_b_222"
    assert resumed_result.utterance_b.actual_transcript == "Teen packet chahiye"
    assert resumed_result.semantic_distinction_preserved is True
    assert resumed_result.task_success is True

    # Database state: exactly 1 run (updated from incomplete to completed)
    all_runs = storage.list_runs()
    assert len(all_runs) == 1
    assert all_runs[0].overall_status == "completed"


@pytest.mark.asyncio
async def test_resuming_completed_run_is_idempotent(tmp_path, test_wav_bytes):
    """Resuming an already-completed run returns it immediately without provider calls."""
    storage = StorageManager(db_path=tmp_path / "eval_resume_idempotent.db")
    engine = EvaluationEngine(storage_manager=storage)

    mock_client = AsyncMock(spec=GnaniPrismaClient)
    mock_client.transcribe.side_effect = [
        {"success": True, "transcript": "Do packet chahiye", "request_id": "req_1"},
        {"success": True, "transcript": "Teen packet chahiye", "request_id": "req_2"},
    ]

    # Initial full evaluation
    result1 = await engine.evaluate_live_audio(
        test_case_id="pair_qty_001",
        audio_a_bytes=test_wav_bytes,
        filename_a="utt_a.wav",
        content_type_a="audio/wav",
        audio_b_bytes=test_wav_bytes,
        filename_b="utt_b.wav",
        content_type_b="audio/wav",
        client=mock_client,
    )
    assert result1.overall_status == "completed"
    assert mock_client.transcribe.call_count == 2

    # Attempt resumption of already completed run
    mock_client.reset_mock()
    result2 = await engine.evaluate_live_audio(
        test_case_id="pair_qty_001",
        audio_a_bytes=test_wav_bytes,
        filename_a="utt_a.wav",
        content_type_a="audio/wav",
        audio_b_bytes=test_wav_bytes,
        filename_b="utt_b.wav",
        content_type_b="audio/wav",
        client=mock_client,
        resume_run_id=result1.run_id,
    )

    # ZERO provider calls
    assert mock_client.transcribe.call_count == 0
    assert result2.run_id == result1.run_id
    assert result2.overall_status == "completed"


@pytest.mark.asyncio
async def test_resumption_rejects_invalid_or_mismatched_run(tmp_path, test_wav_bytes):
    """Resumption rejects non-existent run ID or test case mismatch."""
    storage = StorageManager(db_path=tmp_path / "eval_resume_invalid.db")
    engine = EvaluationEngine(storage_manager=storage)
    mock_client = AsyncMock(spec=GnaniPrismaClient)

    # 1. Non-existent run
    with pytest.raises(ValueError) as exc:
        await engine.evaluate_live_audio(
            test_case_id="pair_qty_001",
            audio_a_bytes=test_wav_bytes,
            filename_a="utt_a.wav",
            content_type_a="audio/wav",
            audio_b_bytes=test_wav_bytes,
            filename_b="utt_b.wav",
            content_type_b="audio/wav",
            client=mock_client,
            resume_run_id="non_existent_run_999",
        )
    assert "not found" in str(exc.value)

    # 2. Mismatched test case
    # Save a run under pair_neg_001
    mock_client.transcribe.side_effect = [
        {"success": True, "transcript": "Nahi", "request_id": "req_neg"},
        GnaniTimeoutError(timeout_seconds=30.0),
    ]
    with pytest.raises(GnaniTimeoutError):
        await engine.evaluate_live_audio(
            test_case_id="pair_neg_001",
            audio_a_bytes=test_wav_bytes,
            filename_a="utt_a.wav",
            content_type_a="audio/wav",
            audio_b_bytes=test_wav_bytes,
            filename_b="utt_b.wav",
            content_type_b="audio/wav",
            client=mock_client,
        )
    runs = storage.list_runs()
    neg_run_id = runs[0].run_id

    # Try to resume pair_neg_001 run with pair_qty_001 test case
    with pytest.raises(ValueError) as exc:
        await engine.evaluate_live_audio(
            test_case_id="pair_qty_001",
            audio_a_bytes=test_wav_bytes,
            filename_a="utt_a.wav",
            content_type_a="audio/wav",
            audio_b_bytes=test_wav_bytes,
            filename_b="utt_b.wav",
            content_type_b="audio/wav",
            client=mock_client,
            resume_run_id=neg_run_id,
        )
    assert "does not match" in str(exc.value)


@pytest.mark.asyncio
async def test_human_pair_live_evaluation_resumption(tmp_path, test_wav_bytes):
    """evaluate_human_pair_live safely reuses completed Utterance A on resumption."""
    storage = StorageManager(db_path=tmp_path / "eval_human_resume.db")
    rec_storage = RecordingStorage(base_dir=tmp_path / "recordings")
    engine = EvaluationEngine(storage_manager=storage)

    # Create contrastive pair with 2 utterances
    meta_a = rec_storage.save_recording(
        raw_bytes=test_wav_bytes,
        original_filename="a.wav",
        speaker_id="spk_res_01",
        language="hi-IN",
        category="quantities_and_numbers",
        reference_transcript="Do packet chahiye",
        verified_transcript="Do packet chahiye",
        is_transcript_verified=True,
        consent_given=True,
        consent_statement="Consent",
        pair_id="human_pair_res_001",
        utterance_role="utterance_a",
    )
    meta_b = rec_storage.save_recording(
        raw_bytes=test_wav_bytes,
        original_filename="b.wav",
        speaker_id="spk_res_01",
        language="hi-IN",
        category="quantities_and_numbers",
        reference_transcript="Teen packet chahiye",
        verified_transcript="Teen packet chahiye",
        is_transcript_verified=True,
        consent_given=True,
        consent_statement="Consent",
        pair_id="human_pair_res_001",
        utterance_role="utterance_b",
    )

    mock_client = AsyncMock(spec=GnaniPrismaClient)
    # Attempt 1: Utterance A succeeds, Utterance B fails
    mock_client.transcribe.side_effect = [
        {
            "success": True,
            "transcript": "Do packet chahiye",
            "request_id": "req_human_a_777",
            "client_request_elapsed_ms": 150.0,
            "provider_processing_time_ms": 80.0,
        },
        {"success": False, "error": "Provider 503 Service Unavailable"},
    ]

    with patch("app.services.recording_storage.RecordingStorage", return_value=rec_storage):
        run1 = await engine.evaluate_human_pair_live(
            pair_id="human_pair_res_001",
            client=mock_client,
        )

    assert run1.overall_status == "incomplete"
    assert run1.utterance_a.status == "completed"
    assert run1.utterance_a.request_id == "req_human_a_777"
    assert run1.utterance_b.status == "failed"
    assert mock_client.transcribe.call_count == 2

    # Attempt 2: Resume the incomplete human pair evaluation
    mock_client.reset_mock()
    mock_client.transcribe.side_effect = [
        {
            "success": True,
            "transcript": "Teen packet chahiye",
            "request_id": "req_human_b_888",
            "client_request_elapsed_ms": 160.0,
            "provider_processing_time_ms": 85.0,
        }
    ]

    with patch("app.services.recording_storage.RecordingStorage", return_value=rec_storage):
        resumed_run = await engine.evaluate_human_pair_live(
            pair_id="human_pair_res_001",
            client=mock_client,
            resume_run_id=run1.run_id,
        )

    # Transcribe called exactly once (for Utterance B only)!
    assert mock_client.transcribe.call_count == 1
    assert resumed_run.run_id == run1.run_id
    assert resumed_run.overall_status == "completed"
    assert resumed_run.utterance_a.request_id == "req_human_a_777"
    assert resumed_run.utterance_b.request_id == "req_human_b_888"
    assert resumed_run.dataset_source == "human_recording"


@pytest.mark.asyncio
async def test_concurrent_resumption_prevented(tmp_path, test_wav_bytes):
    """Parallel requests attempting to resume the same incomplete run concurrently are blocked."""
    import asyncio

    storage = StorageManager(db_path=tmp_path / "eval_concurrent_resume.db")
    engine = EvaluationEngine(storage_manager=storage)

    mock_client = AsyncMock(spec=GnaniPrismaClient)
    # Run 1 creates incomplete run
    mock_client.transcribe.side_effect = [
        {"success": True, "transcript": "Do packet", "request_id": "req_a"},
        GnaniTimeoutError(timeout_seconds=30.0),
    ]

    with pytest.raises(GnaniTimeoutError):
        await engine.evaluate_live_audio(
            test_case_id="pair_qty_001",
            audio_a_bytes=test_wav_bytes,
            filename_a="a.wav",
            content_type_a="audio/wav",
            audio_b_bytes=test_wav_bytes,
            filename_b="b.wav",
            content_type_b="audio/wav",
            client=mock_client,
        )

    incomplete_run_id = storage.list_runs()[0].run_id

    # Simulate slow transcribe on resume so another request can collide
    async def slow_transcribe(**kwargs):
        await asyncio.sleep(0.2)
        return {"success": True, "transcript": "Teen packet", "request_id": "req_b"}

    mock_client.reset_mock()
    mock_client.transcribe.side_effect = slow_transcribe

    async def attempt_resume():
        return await engine.evaluate_live_audio(
            test_case_id="pair_qty_001",
            audio_a_bytes=test_wav_bytes,
            filename_a="a.wav",
            content_type_a="audio/wav",
            audio_b_bytes=test_wav_bytes,
            filename_b="b.wav",
            content_type_b="audio/wav",
            client=mock_client,
            resume_run_id=incomplete_run_id,
        )

    # Launch two simultaneous resumptions
    t1 = asyncio.create_task(attempt_resume())
    await asyncio.sleep(0.02)  # Give t1 a moment to acquire the lock and start
    t2 = asyncio.create_task(attempt_resume())

    results = await asyncio.gather(t1, t2, return_exceptions=True)

    # One should succeed, the other should raise ValueError about concurrent resumption
    succeeded = [r for r in results if not isinstance(r, Exception)]
    errors = [r for r in results if isinstance(r, Exception)]

    assert len(succeeded) == 1
    assert len(errors) == 1
    assert "currently being resumed" in str(errors[0])
