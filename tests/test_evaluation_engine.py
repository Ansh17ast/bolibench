"""Automated tests for BoliBench Paired-Contrast Evaluation Engine.

Tests:
- Word Error Rate (WER) with exact mathematical verification
- Text normalization ensuring semantic tokens are preserved
- Critical field extraction: quantities, negation, corrections, entities, intent
- Semantic distinction preservation logic
- Downstream task success determination
- Explicit failure classification taxonomy
- SQLite persistence, querying, aggregate metrics, and exports
"""

from pathlib import Path
import pytest

from app.evaluation.dataset import load_seed_dataset
from app.evaluation.metrics import (
    calculate_wer,
    classify_failure,
    extract_all_critical_fields,
    extract_command_intent,
    extract_corrections,
    extract_entities,
    extract_negation,
    extract_quantities,
    normalize_text,
    score_pair,
)
from app.evaluation.schemas import (
    ExecutionMode,
    FailureCategory,
    PairEvaluationResult,
    TestCasePair,
    TestCategory,
    UtteranceEvaluationResult,
    UtteranceSpec,
)
from app.evaluation.storage import StorageManager


# -----------------------------------------------------------------------------
# 1. WER Calculation Tests
# -----------------------------------------------------------------------------

def test_wer_identical_strings():
    """Identical sentences produce 0.0 WER."""
    wer, s, d, i, r_len = calculate_wer("Do packet chahiye", "Do packet chahiye")
    assert wer == 0.0
    assert s == 0 and d == 0 and i == 0
    assert r_len == 3


def test_wer_single_substitution():
    """Replacing one word in a 3-word sentence yields 1/3 = 0.3333 WER."""
    wer, s, d, i, r_len = calculate_wer("Do packet chahiye", "Teen packet chahiye")
    assert wer == pytest.approx(0.3333, abs=1e-4)
    assert s == 1 and d == 0 and i == 0
    assert r_len == 3


def test_wer_deletion():
    """Omitting one word from a 3-word sentence yields 1/3 = 0.3333 WER."""
    wer, s, d, i, r_len = calculate_wer("Do packet chahiye", "Do chahiye")
    assert wer == pytest.approx(0.3333, abs=1e-4)
    assert s == 0 and d == 1 and i == 0
    assert r_len == 3


def test_wer_insertion():
    """Inserting one word into a 2-word sentence yields 1/2 = 0.5 WER."""
    wer, s, d, i, r_len = calculate_wer("Do packet", "Do packet extra")
    assert wer == 0.5
    assert s == 0 and d == 0 and i == 1
    assert r_len == 2


def test_wer_empty_reference_empty_hyp():
    """Empty reference and empty hypothesis produces 0.0 WER."""
    wer, s, d, i, r_len = calculate_wer("", "")
    assert wer == 0.0
    assert r_len == 0


def test_wer_empty_reference_nonempty_hyp():
    """Empty reference with non-empty hypothesis produces 1.0 WER."""
    wer, s, d, i, r_len = calculate_wer("", "extra words spoken")
    assert wer == 1.0
    assert i == 3


def test_wer_nonempty_reference_empty_hyp():
    """Non-empty reference with empty hypothesis produces 1.0 WER (all deleted)."""
    wer, s, d, i, r_len = calculate_wer("Do packet chahiye", "")
    assert wer == 1.0
    assert d == 3


# -----------------------------------------------------------------------------
# 2. Text Normalization Tests
# -----------------------------------------------------------------------------

def test_normalization_preserves_semantic_distinctions():
    """Punctuation and whitespace are removed without stripping numbers or negations."""
    raw = "  Do packet, chahiye!  "
    norm = normalize_text(raw)
    assert norm == "do packet chahiye"

    # Hindi Devanagari danda punctuation removal
    hindi_raw = "पाँच लाख रुपये। कृपया भेजें॥"
    hindi_norm = normalize_text(hindi_raw)
    assert "।" not in hindi_norm
    assert "॥" not in hindi_norm
    assert "पाँच लाख रुपये" in hindi_norm

    # Crucial rule: "do" vs "teen" must remain distinct tokens
    assert normalize_text("Do") != normalize_text("Teen")
    assert normalize_text("karna") != normalize_text("mat karna")


# -----------------------------------------------------------------------------
# 3. Critical Field Extraction Tests
# -----------------------------------------------------------------------------

def test_extract_quantities_small_counts():
    """Extracts integer quantities from Hindi number words and digits."""
    q_do = extract_quantities("Do packet chahiye")
    assert q_do.get("quantity") == 2

    q_teen = extract_quantities("Teen packet chahiye")
    assert q_teen.get("quantity") == 3

    q_char = extract_quantities("Chaar plates")
    assert q_char.get("quantity") == 4

    q_digit = extract_quantities("I need 5 packets")
    assert q_digit.get("quantity") == 5


def test_extract_quantities_financial_scale():
    """Extracts large financial magnitudes (5 Lakh vs 5 Thousand)."""
    q_lakh = extract_quantities("Paanch lakh rupaye bhejo")
    assert q_lakh.get("quantity") == 500000

    q_hazaar = extract_quantities("Paanch hazaar rupaye bhejo")
    assert q_hazaar.get("quantity") == 5000

    # Currency symbol extraction
    q_currency = extract_quantities("Payment of ₹500 confirmed")
    assert q_currency.get("quantity") == 500


def test_extract_negation():
    """Detects Hindi and English negation tokens."""
    neg_hindi = extract_negation("Nahi subscription cancel mat karo")
    assert neg_hindi["is_negated"] is True
    assert neg_hindi["action"] == "keep_booking"

    aff_hindi = extract_negation("Haan subscription cancel kar do")
    assert aff_hindi["is_negated"] is False
    assert aff_hindi["action"] == "cancel_booking"

    neg_en = extract_negation("Do not transfer the funds yet")
    assert neg_en["is_negated"] is True
    assert neg_en["action"] == "block_transfer"

    aff_en = extract_negation("Transfer the funds now")
    assert aff_en["is_negated"] is False
    assert aff_en["action"] == "execute_transfer"


def test_extract_corrections_change_of_mind():
    """Extracts final intended value after a change of mind."""
    # Destination correction
    corr_dest = extract_corrections("Book flight to Mumbai, no wait, to Delhi")
    assert corr_dest["correction_detected"] is True
    assert corr_dest["final_target"] == "delhi"

    # Food ordering count correction
    corr_qty = extract_corrections("Do plate biryani, nahi teen plate karo")
    assert corr_qty["correction_detected"] is True
    assert corr_qty["final_target"] == 3


def test_extract_entities_and_products():
    """Extracts named entities and product names."""
    ent_rahul = extract_entities("Send invoice to Rahul Sharma")
    assert ent_rahul.get("person_name") == "Rahul Sharma"

    ent_rohit = extract_entities("Send invoice to Rohit Sharma")
    assert ent_rohit.get("person_name") == "Rohit Sharma"

    prod_prisma = extract_entities("Prisma STT ka access enable karo")
    assert prod_prisma.get("product_name") == "Prisma STT"

    prod_krishna = extract_entities("Krishna STT ka access enable karo")
    assert prod_krishna.get("product_name") == "Krishna STT"


def test_extract_command_intent():
    """Classifies high-risk banking and command intents."""
    assert extract_command_intent("Mera account balance check karo") == "CHECK_BALANCE"
    assert extract_command_intent("Mera account band kar do") == "CLOSE_ACCOUNT"
    assert extract_command_intent("Pay five hundred rupees") == "TRANSFER_FUNDS"
    assert extract_command_intent("Request five hundred rupees") == "REQUEST_FUNDS"


# -----------------------------------------------------------------------------
# 4. Paired-Contrast Scoring & Failure Classification Tests
# -----------------------------------------------------------------------------

def test_score_pair_preserved_distinction():
    """When both utterances match references, semantic distinction is preserved."""
    seed_cases = load_seed_dataset()
    pair_qty = next(tc for tc in seed_cases if tc.id == "pair_qty_001")

    res = score_pair(
        run_id="run_test_001",
        test_case=pair_qty,
        transcript_a="Do packet chahiye",
        transcript_b="Teen packet chahiye",
        execution_mode="mock",
    )

    assert res.semantic_distinction_preserved is True
    assert res.task_success is True
    assert res.critical_field_accuracy == 1.0
    assert res.average_wer == 0.0
    assert res.primary_failure_category == FailureCategory.NONE


def test_score_pair_crashed_distinction_quantity_error():
    """When Utterance B collapses into Utterance A's quantity, failure is classified as QUANTITY_ERROR."""
    seed_cases = load_seed_dataset()
    pair_qty = next(tc for tc in seed_cases if tc.id == "pair_qty_001")

    # Simulate Utterance B ("Teen packet chahiye") misheard as "Do packet chahiye"
    res = score_pair(
        run_id="run_test_002",
        test_case=pair_qty,
        transcript_a="Do packet chahiye",
        transcript_b="Do packet chahiye",  # Crashed transcript!
        execution_mode="mock",
    )

    assert res.semantic_distinction_preserved is False
    assert res.task_success is False
    assert res.primary_failure_category == FailureCategory.QUANTITY_ERROR
    assert "Numeric distinction collapsed" in res.failure_explanation


def test_score_pair_crashed_distinction_missed_negation():
    """When negation is omitted, failure is classified as MISSED_NEGATION."""
    seed_cases = load_seed_dataset()
    pair_neg = next(tc for tc in seed_cases if tc.id == "pair_neg_001")

    # Utterance B ("Nahi subscription cancel mat karo") misheard as "Haan subscription cancel kar do"
    res = score_pair(
        run_id="run_test_003",
        test_case=pair_neg,
        transcript_a="Haan subscription cancel kar do",
        transcript_b="Haan subscription cancel kar do",  # Critical missed negation
        execution_mode="mock",
    )

    assert res.semantic_distinction_preserved is False
    assert res.task_success is False
    assert res.primary_failure_category == FailureCategory.MISSED_NEGATION
    assert "Voice agent missed critical negation" in res.failure_explanation


def test_score_pair_crashed_correction_error():
    """When correction is ignored in favor of initial value, failure is classified as CORRECTION_ERROR."""
    seed_cases = load_seed_dataset()
    pair_corr = next(tc for tc in seed_cases if tc.id == "pair_corr_001")

    # Utterance A ("Book flight to Mumbai no wait to Delhi") truncated to "Book flight to Mumbai"
    res = score_pair(
        run_id="run_test_004",
        test_case=pair_corr,
        transcript_a="Book flight to Mumbai",  # Truncated before correction
        transcript_b="Book flight to Mumbai",
        execution_mode="mock",
    )

    assert res.semantic_distinction_preserved is False
    assert res.task_success is False
    assert res.primary_failure_category == FailureCategory.CORRECTION_ERROR


# -----------------------------------------------------------------------------
# 5. SQLite Persistence and Export Tests
# -----------------------------------------------------------------------------

def test_storage_manager_lifecycle(tmp_path: Path):
    """Verifies SQLite save, retrieve, aggregate metrics, and exports."""
    db_file = tmp_path / "test_eval.db"
    storage = StorageManager(db_path=db_file)

    seed_cases = load_seed_dataset()
    pair = seed_cases[0]

    # Save a run
    res = score_pair(
        run_id="run_storage_001",
        test_case=pair,
        transcript_a="Do packet chahiye",
        transcript_b="Teen packet chahiye",
        execution_mode="mock",
        timestamp="2026-10-15T00:00:00Z",
    )
    storage.save_run(res)

    # Retrieve run
    fetched = storage.get_run("run_storage_001")
    assert fetched is not None
    assert fetched.run_id == "run_storage_001"
    assert fetched.semantic_distinction_preserved is True

    # Aggregate metrics
    metrics = storage.get_aggregate_metrics()
    assert metrics.total_runs == 1
    assert metrics.mock_runs == 1
    assert metrics.distinction_preservation_rate == 100.0

    # JSON export
    json_export = storage.export_json()
    assert "run_storage_001" in json_export

    # CSV export
    csv_export = storage.export_csv()
    assert "run_storage_001" in csv_export
    assert "pair_qty_001" in csv_export


def test_mock_vs_live_mode_distinction_and_metrics(tmp_path: Path):
    """Proves mock and live modes are distinct in storage, metrics, and latency fields."""
    db_file = tmp_path / "test_modes.db"
    storage = StorageManager(db_path=db_file)
    pair = load_seed_dataset()[0]

    # 1. Mock run with simulated latency
    mock_run = score_pair(
        run_id="run_mock_001",
        test_case=pair,
        transcript_a="Do packet chahiye",
        transcript_b="Teen packet chahiye",
        execution_mode="mock",
        simulated_latency_ms=12.5,
        client_latency_ms=999.0,  # Accidental client_latency must be ignored in mock
    )
    storage.save_run(mock_run)

    # 2. Live run with genuine measured latency
    live_run = score_pair(
        run_id="run_live_001",
        test_case=pair,
        transcript_a="Do packet chahiye",
        transcript_b="Teen packet chahiye",
        execution_mode="live",
        client_latency_ms=245.8,
        provider_latency_ms=190.2,
        request_id_a="req_a_123",
        request_id_b="req_b_456",
    )
    storage.save_run(live_run)

    # Assert mock run isolation
    saved_mock = storage.get_run("run_mock_001")
    assert saved_mock.execution_mode.value == "mock"
    assert saved_mock.client_measured_latency_ms is None
    assert saved_mock.client_request_elapsed_ms is None
    assert saved_mock.provider_processing_time_ms is None
    assert saved_mock.simulated_latency_ms == 12.5

    # Assert live run isolation
    saved_live = storage.get_run("run_live_001")
    assert saved_live.execution_mode.value == "live"
    assert saved_live.client_measured_latency_ms == 245.8
    assert saved_live.client_request_elapsed_ms == 245.8
    assert saved_live.provider_processing_time_ms == 190.2
    assert saved_live.simulated_latency_ms is None
    assert saved_live.utterance_a.request_id == "req_a_123"
    assert saved_live.utterance_b.request_id == "req_b_456"

    # Assert aggregate metrics separates mock and live
    metrics = storage.get_aggregate_metrics()
    assert metrics.total_runs == 2
    assert metrics.mock_runs == 1
    assert metrics.live_runs == 1


def test_live_paired_evaluation_preserves_null_provider_latency_when_missing():
    """Proves provider_processing_time_ms remains null when upstream provider omits processing duration."""
    pair = load_seed_dataset()[0]
    live_run = score_pair(
        run_id="run_live_null_prov",
        test_case=pair,
        transcript_a="Do packet chahiye",
        transcript_b="Teen packet chahiye",
        execution_mode="live",
        client_request_elapsed_ms=210.0,
        provider_latency_ms=None,  # Provider does not report server processing time
        request_id_a="req_a_789",
        request_id_b="req_b_012",
    )

    assert live_run.execution_mode.value == "live"
    assert live_run.client_measured_latency_ms == 210.0
    assert live_run.client_request_elapsed_ms == 210.0
    assert live_run.provider_processing_time_ms is None
    assert live_run.simulated_latency_ms is None


@pytest.mark.asyncio
async def test_engine_evaluate_live_audio_success(tmp_path, valid_wav_bytes):
    """Proves engine.evaluate_live_audio completes paired evaluation with format=transcribe."""
    from unittest.mock import AsyncMock, patch
    from app.evaluation.engine import EvaluationEngine
    from app.evaluation.storage import StorageManager
    from app.services.gnani_client import GnaniPrismaClient

    storage = StorageManager(db_path=tmp_path / "test_engine.db")
    engine = EvaluationEngine(storage_manager=storage)

    mock_client = AsyncMock(spec=GnaniPrismaClient)
    mock_client.transcribe.side_effect = [
        {
            "success": True,
            "transcript": "Do packet chahiye",
            "request_id": "req_a_111",
            "client_request_elapsed_ms": 120.0,
            "provider_processing_time_ms": None,
        },
        {
            "success": True,
            "transcript": "Teen packet chahiye",
            "request_id": "req_b_222",
            "client_request_elapsed_ms": 125.0,
            "provider_processing_time_ms": None,
        },
    ]

    result = await engine.evaluate_live_audio(
        test_case_id="pair_qty_001",
        audio_a_bytes=valid_wav_bytes,
        filename_a="utt_a.wav",
        content_type_a="audio/wav",
        audio_b_bytes=valid_wav_bytes,
        filename_b="utt_b.wav",
        content_type_b="audio/wav",
        client=mock_client,
        audio_condition="clean_recorded",
    )

    assert result.overall_status == "completed"
    assert result.semantic_distinction_preserved is True
    assert result.task_success is True
    assert mock_client.transcribe.call_count == 2

    # Assert contract parameters: format_mode="transcribe" and preferred_language="hi-IN"
    call_a = mock_client.transcribe.call_args_list[0].kwargs
    assert call_a["format_mode"] == "transcribe"
    assert call_a["preferred_language"] == "hi-IN"

    call_b = mock_client.transcribe.call_args_list[1].kwargs
    assert call_b["format_mode"] == "transcribe"
    assert call_b["preferred_language"] == "hi-IN"


@pytest.mark.asyncio
async def test_engine_evaluate_live_audio_a_fails(tmp_path, valid_wav_bytes):
    """Proves engine.evaluate_live_audio saves a failed run and skips Utterance B when A fails."""
    from unittest.mock import AsyncMock
    from app.evaluation.engine import EvaluationEngine
    from app.evaluation.storage import StorageManager
    from app.services.gnani_client import GnaniPrismaClient, GnaniServerError

    storage = StorageManager(db_path=tmp_path / "test_engine_fail_a.db")
    engine = EvaluationEngine(storage_manager=storage)

    mock_client = AsyncMock(spec=GnaniPrismaClient)
    mock_client.transcribe.side_effect = GnaniServerError("Gateway 502 Bad Gateway from Prisma", status_code=502)

    with pytest.raises(GnaniServerError):
        await engine.evaluate_live_audio(
            test_case_id="pair_qty_001",
            audio_a_bytes=valid_wav_bytes,
            filename_a="utt_a.wav",
            content_type_a="audio/wav",
            audio_b_bytes=valid_wav_bytes,
            filename_b="utt_b.wav",
            content_type_b="audio/wav",
            client=mock_client,
        )

    # Only 1 call was made to provider
    assert mock_client.transcribe.call_count == 1

    # Persisted failed run verified
    runs = storage.list_runs()
    assert len(runs) == 1
    assert runs[0].overall_status == "failed"
    assert runs[0].utterance_a.status == "failed"
    assert runs[0].utterance_b.status == "skipped"
    assert runs[0].task_success is False
    assert runs[0].primary_failure_category.value == "provider_error"


@pytest.mark.asyncio
async def test_engine_evaluate_live_audio_b_fails(tmp_path, valid_wav_bytes):
    """Proves engine.evaluate_live_audio saves an incomplete run with Utterance A completed when B fails."""
    from unittest.mock import AsyncMock
    from app.evaluation.engine import EvaluationEngine
    from app.evaluation.storage import StorageManager
    from app.services.gnani_client import GnaniPrismaClient, GnaniTimeoutError

    storage = StorageManager(db_path=tmp_path / "test_engine_fail_b.db")
    engine = EvaluationEngine(storage_manager=storage)

    mock_client = AsyncMock(spec=GnaniPrismaClient)
    mock_client.transcribe.side_effect = [
        {
            "success": True,
            "transcript": "Do packet chahiye",
            "request_id": "req_a_success_333",
            "client_request_elapsed_ms": 112.0,
            "provider_processing_time_ms": None,
        },
        GnaniTimeoutError(timeout_seconds=30.0),
    ]

    with pytest.raises(GnaniTimeoutError):
        await engine.evaluate_live_audio(
            test_case_id="pair_qty_001",
            audio_a_bytes=valid_wav_bytes,
            filename_a="utt_a.wav",
            content_type_a="audio/wav",
            audio_b_bytes=valid_wav_bytes,
            filename_b="utt_b.wav",
            content_type_b="audio/wav",
            client=mock_client,
        )

    # Exactly 2 calls made
    assert mock_client.transcribe.call_count == 2

    # Persisted incomplete run verified
    runs = storage.list_runs()
    assert len(runs) == 1
    assert runs[0].overall_status == "incomplete"
    assert runs[0].utterance_a.status == "completed"
    assert runs[0].utterance_a.request_id == "req_a_success_333"
    assert runs[0].utterance_b.status == "failed"
    assert runs[0].task_success is False
    assert runs[0].primary_failure_category.value == "provider_error"
    assert "Utterance A succeeded" in runs[0].failure_explanation
    assert "credits may have been consumed" in runs[0].failure_explanation


def test_aggregate_metrics_excludes_incomplete_and_failed_runs(tmp_path: Path):
    """Proves completed runs contribute to performance averages while failed/incomplete runs do not."""
    db_file = tmp_path / "test_clean_metrics.db"
    storage = StorageManager(db_path=db_file)
    pair = load_seed_dataset()[0]

    # 0. Empty dataset has total_runs=0 and None for averages (no invented scores)
    empty_metrics = storage.get_aggregate_metrics()
    assert empty_metrics.total_runs == 0
    assert empty_metrics.completed_runs == 0
    assert empty_metrics.distinction_preservation_rate is None
    assert empty_metrics.average_wer is None
    assert empty_metrics.task_success_rate is None
    assert empty_metrics.average_critical_field_accuracy is None

    # 1. Add Completed Run 1: Ideal run (WER=0.0, distinction preserved, task success)
    run_c1 = score_pair(
        run_id="run_comp_001",
        test_case=pair,
        transcript_a="Do packet chahiye",
        transcript_b="Teen packet chahiye",
        execution_mode="mock",
    )
    storage.save_run(run_c1)

    # 2. Add Completed Run 2: High WER run (WER=0.5, distinction collapsed, task failed)
    run_c2 = score_pair(
        run_id="run_comp_002",
        test_case=pair,
        transcript_a="Do packet chahiye",
        transcript_b="Do packet chahiye",
        execution_mode="mock",
    )
    storage.save_run(run_c2)

    # Completed baseline:
    # run 1: avg_wer = 0.0, distinction = 1, task = 1, field_acc = 1.0
    # run 2: avg_wer = 0.1667, distinction = 0, task = 0, field_acc = 0.5
    # expected averages across completed:
    # avg_wer = 0.0833, distinction = 50.0%, task = 50.0%, field_acc = 75.0%
    base_metrics = storage.get_aggregate_metrics()
    assert base_metrics.total_runs == 2
    assert base_metrics.completed_runs == 2
    assert base_metrics.failed_runs == 0
    assert base_metrics.incomplete_runs == 0
    assert base_metrics.average_wer == 0.0833
    assert base_metrics.distinction_preservation_rate == 50.0
    assert base_metrics.task_success_rate == 50.0
    assert base_metrics.average_critical_field_accuracy == 75.0

    # 3. Add an Operational Failed Run (Utterance A failed, e.g. provider error)
    failed_run = PairEvaluationResult(
        run_id="run_failed_ops",
        test_case_id=pair.id,
        category=pair.category.value,
        language_code=pair.language_code,
        execution_mode=ExecutionMode.LIVE,
        audio_condition="clean_recorded",
        timestamp="2026-10-20T04:00:00Z",
        overall_status="failed",
        utterance_a=UtteranceEvaluationResult(
            utterance_id="A",
            reference_transcript=pair.utterance_a.reference_transcript,
            actual_transcript="",
            wer=1.0,
            substitutions=0,
            deletions=3,
            insertions=0,
            reference_length=3,
            critical_fields_match=False,
            detected_intent="UNKNOWN",
            intent_match=False,
            task_success=False,
            status="failed",
            error_message="Connection timed out",
        ),
        utterance_b=UtteranceEvaluationResult(
            utterance_id="B",
            reference_transcript=pair.utterance_b.reference_transcript,
            actual_transcript="",
            wer=1.0,
            substitutions=0,
            deletions=3,
            insertions=0,
            reference_length=3,
            critical_fields_match=False,
            detected_intent="UNKNOWN",
            intent_match=False,
            task_success=False,
            status="skipped",
        ),
        wer_a=1.0,
        wer_b=1.0,
        average_wer=1.0,
        critical_field_accuracy=0.0,
        semantic_distinction_preserved=False,
        task_success=False,
        primary_failure_category=FailureCategory.PROVIDER_ERROR,
        failure_explanation="Provider request for Utterance A failed: Connection timed out.",
    )
    storage.save_run(failed_run)

    # 4. Add an Incomplete Run (Utterance A succeeded, Utterance B failed)
    incomplete_run = PairEvaluationResult(
        run_id="run_incomplete_ops",
        test_case_id=pair.id,
        category=pair.category.value,
        language_code=pair.language_code,
        execution_mode=ExecutionMode.LIVE,
        audio_condition="clean_recorded",
        timestamp="2026-10-20T04:05:00Z",
        overall_status="incomplete",
        utterance_a=UtteranceEvaluationResult(
            utterance_id="A",
            reference_transcript=pair.utterance_a.reference_transcript,
            actual_transcript="Do packet chahiye",
            wer=0.0,
            substitutions=0,
            deletions=0,
            insertions=0,
            reference_length=3,
            critical_fields_match=True,
            detected_intent="ORDER_PRODUCT",
            intent_match=True,
            task_success=True,
            status="completed",
            request_id="req_live_123",
        ),
        utterance_b=UtteranceEvaluationResult(
            utterance_id="B",
            reference_transcript=pair.utterance_b.reference_transcript,
            actual_transcript="",
            wer=1.0,
            substitutions=0,
            deletions=3,
            insertions=0,
            reference_length=3,
            critical_fields_match=False,
            detected_intent="UNKNOWN",
            intent_match=False,
            task_success=False,
            status="failed",
            error_message="HTTP 502 Bad Gateway",
        ),
        wer_a=0.0,
        wer_b=1.0,
        average_wer=0.5,
        critical_field_accuracy=0.5,
        semantic_distinction_preserved=False,
        task_success=False,
        primary_failure_category=FailureCategory.PROVIDER_ERROR,
        failure_explanation="Partial live evaluation failure: Utterance A succeeded, Utterance B failed.",
    )
    storage.save_run(incomplete_run)

    # 5. Verify Aggregate Metrics:
    # Total attempts = 4 (2 completed, 1 failed, 1 incomplete)
    # Mock attempts = 2, Live attempts = 2
    metrics = storage.get_aggregate_metrics()
    assert metrics.total_runs == 4
    assert metrics.completed_runs == 2
    assert metrics.failed_runs == 1
    assert metrics.incomplete_runs == 1
    assert metrics.mock_runs == 2
    assert metrics.live_runs == 2

    # CRITICAL ASSERTION: The 1.0 WER from failed run and 0.5 WER from incomplete run DO NOT contaminate!
    assert metrics.average_wer == 0.0833
    assert metrics.distinction_preservation_rate == 50.0
    assert metrics.task_success_rate == 50.0
    assert metrics.average_critical_field_accuracy == 75.0

    # Operational failures remain discoverable in failure breakdown
    assert metrics.failure_category_breakdown["provider_error"] == 2

    # Category performance averages also use completed evaluations only
    cat = metrics.category_breakdown[pair.category.value]
    assert cat["total"] == 4
    assert cat["completed"] == 2
    assert cat["failed"] == 1
    assert cat["incomplete"] == 1
    assert cat["average_wer"] == 0.0833
    assert cat["distinction_rate"] == 50.0
    assert cat["task_success_rate"] == 50.0


def test_sqlite_backward_compatible_migration_preserves_legacy_records_as_completed(tmp_path: Path):
    """Proves legacy SQLite database without overall_status migrates seamlessly preserving rows as completed."""
    import json
    import sqlite3
    db_file = tmp_path / "legacy_bolibench.db"

    # Create raw table WITHOUT overall_status (simulating pre-Phase-5.2 schema)
    with sqlite3.connect(str(db_file)) as conn:
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
                created_at TEXT NOT NULL,
                payload_json TEXT NOT NULL
            )
            """
        )
        legacy_payload = json.dumps({
            "run_id": "legacy_run_001",
            "test_case_id": "pair_qty_001",
            "category": "quantities_and_numbers",
            "language_code": "hi-IN",
            "execution_mode": "mock",
            "audio_condition": "text_only_simulation",
            "timestamp": "2026-10-18T10:00:00Z",
            "utterance_a": {
                "utterance_id": "A",
                "reference_transcript": "Do packet chahiye",
                "actual_transcript": "Do packet chahiye",
                "wer": 0.0,
                "substitutions": 0,
                "deletions": 0,
                "insertions": 0,
                "reference_length": 3,
                "critical_fields_match": True,
                "detected_intent": "ORDER_PRODUCT",
                "intent_match": True,
                "task_success": True,
                "status": "completed",
            },
            "utterance_b": {
                "utterance_id": "B",
                "reference_transcript": "Teen packet chahiye",
                "actual_transcript": "Teen packet chahiye",
                "wer": 0.0,
                "substitutions": 0,
                "deletions": 0,
                "insertions": 0,
                "reference_length": 3,
                "critical_fields_match": True,
                "detected_intent": "ORDER_PRODUCT",
                "intent_match": True,
                "task_success": True,
                "status": "completed",
            },
            "wer_a": 0.0,
            "wer_b": 0.0,
            "average_wer": 0.0,
            "critical_field_accuracy": 1.0,
            "semantic_distinction_preserved": True,
            "task_success": True,
            "primary_failure_category": "none",
            "failure_explanation": "None",
        })
        conn.execute(
            """
            INSERT INTO evaluation_runs VALUES (
                'legacy_run_001', 'pair_qty_001', 'quantities_and_numbers', 'hi-IN',
                'mock', 'text_only_simulation', 0.0, 0.0, 0.0, 1.0, 1, 1, 'none',
                'None', NULL, NULL, '2026-10-18T10:00:00Z', ?
            )
            """,
            (legacy_payload,),
        )
        conn.commit()

    # Now instantiate StorageManager on the legacy database
    storage = StorageManager(db_path=db_file)

    # 1. Assert column overall_status was added and row defaulted to 'completed'
    with sqlite3.connect(str(db_file)) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("PRAGMA table_info(evaluation_runs)")
        columns = [row["name"] for row in cur.fetchall()]
        assert "overall_status" in columns

        cur.execute("SELECT overall_status FROM evaluation_runs WHERE run_id = 'legacy_run_001'")
        row = cur.fetchone()
        assert row["overall_status"] == "completed"

    # 2. Assert get_run and list_runs expose overall_status == 'completed'
    run = storage.get_run("legacy_run_001")
    assert run is not None
    assert run.overall_status == "completed"

    runs = storage.list_runs()
    assert len(runs) == 1
    assert runs[0].overall_status == "completed"

    # 3. Assert metrics treat legacy record as a completed run
    metrics = storage.get_aggregate_metrics()
    assert metrics.total_runs == 1
    assert metrics.completed_runs == 1
    assert metrics.failed_runs == 0
    assert metrics.incomplete_runs == 0
    assert metrics.average_wer == 0.0
    assert metrics.distinction_preservation_rate == 100.0


def test_mock_and_live_metric_separation_and_empty_nulls(tmp_path: Path):
    """Proves mock demonstration metrics and live provider metrics are strictly separated and empty groups return null."""
    db_file = tmp_path / "test_separation.db"
    storage = StorageManager(db_path=db_file)
    pair = load_seed_dataset()[0]

    # 1. Empty database verification: all rates are None (JSON null)
    m_empty = storage.get_aggregate_metrics()
    assert m_empty.total_runs == 0
    assert m_empty.mock_runs == 0
    assert m_empty.live_runs == 0
    assert m_empty.mock.total_attempts == 0
    assert m_empty.mock.completed_runs == 0
    assert m_empty.mock.distinction_preservation_rate is None
    assert m_empty.mock.task_success_rate is None
    assert m_empty.mock.average_wer is None
    assert m_empty.mock.average_critical_field_accuracy is None
    assert m_empty.live.total_attempts == 0
    assert m_empty.live.completed_runs == 0
    assert m_empty.live.distinction_preservation_rate is None
    assert m_empty.live.task_success_rate is None
    assert m_empty.live.average_wer is None
    assert m_empty.live.average_critical_field_accuracy is None

    # 2. Add 1 Completed Mock Run (WER=0.0, distinction preserved=True, task success=True)
    mock_comp = score_pair(
        run_id="run_mock_sep_01",
        test_case=pair,
        transcript_a="Do packet chahiye",
        transcript_b="Teen packet chahiye",
        execution_mode="mock",
    )
    storage.save_run(mock_comp)

    m1 = storage.get_aggregate_metrics()
    assert m1.mock_runs == 1
    assert m1.live_runs == 0
    assert m1.mock.completed_runs == 1
    assert m1.mock.distinction_preservation_rate == 100.0
    assert m1.mock.task_success_rate == 100.0
    assert m1.mock.average_wer == 0.0
    assert m1.mock.average_critical_field_accuracy == 100.0
    # Live must remain untouched with null rates, NOT fabricated 0.0
    assert m1.live.completed_runs == 0
    assert m1.live.total_attempts == 0
    assert m1.live.distinction_preservation_rate is None
    assert m1.live.task_success_rate is None
    assert m1.live.average_wer is None
    assert m1.live.average_critical_field_accuracy is None

    # 3. Add Live Failed Run and Live Incomplete Run (operational errors)
    live_failed = PairEvaluationResult(
        run_id="run_live_fail_sep",
        test_case_id=pair.id,
        category=pair.category.value,
        language_code=pair.language_code,
        execution_mode=ExecutionMode.LIVE,
        audio_condition="clean_recorded",
        timestamp="2026-10-20T05:00:00Z",
        overall_status="failed",
        utterance_a=UtteranceEvaluationResult(
            utterance_id="A",
            reference_transcript=pair.utterance_a.reference_transcript,
            actual_transcript="",
            wer=1.0,
            substitutions=0,
            deletions=3,
            insertions=0,
            reference_length=3,
            critical_fields_match=False,
            detected_intent="UNKNOWN",
            intent_match=False,
            task_success=False,
            status="failed",
            error_message="HTTP 504 Gateway Timeout",
        ),
        utterance_b=UtteranceEvaluationResult(
            utterance_id="B",
            reference_transcript=pair.utterance_b.reference_transcript,
            actual_transcript="",
            wer=1.0,
            substitutions=0,
            deletions=3,
            insertions=0,
            reference_length=3,
            critical_fields_match=False,
            detected_intent="UNKNOWN",
            intent_match=False,
            task_success=False,
            status="skipped",
        ),
        wer_a=1.0,
        wer_b=1.0,
        average_wer=1.0,
        critical_field_accuracy=0.0,
        semantic_distinction_preserved=False,
        task_success=False,
        primary_failure_category=FailureCategory.PROVIDER_ERROR,
        failure_explanation="Provider 504 on Utterance A",
    )
    storage.save_run(live_failed)

    m2 = storage.get_aggregate_metrics()
    assert m2.total_runs == 2
    assert m2.mock_runs == 1
    assert m2.live_runs == 1
    assert m2.live.total_attempts == 1
    assert m2.live.completed_runs == 0
    assert m2.live.failed_runs == 1
    # Live still has 0 completed runs -> performance metrics remain None
    assert m2.live.distinction_preservation_rate is None
    assert m2.live.task_success_rate is None
    assert m2.live.average_wer is None
    # Mock performance is completely unaffected by live failed run
    assert m2.mock.completed_runs == 1
    assert m2.mock.distinction_preservation_rate == 100.0
    assert m2.mock.average_wer == 0.0

    # 4. Add 1 Completed Live Run (WER=0.25, distinction collapsed=False, task success=False)
    live_comp = score_pair(
        run_id="run_live_comp_sep",
        test_case=pair,
        transcript_a="Do packet chahiye",
        transcript_b="Do packet chahiye",
        execution_mode="live",
        client_request_elapsed_ms=310.0,
    )
    storage.save_run(live_comp)

    m3 = storage.get_aggregate_metrics()
    assert m3.total_runs == 3
    assert m3.completed_runs == 2
    assert m3.failed_runs == 1
    assert m3.mock_runs == 1
    assert m3.live_runs == 2

    # Verify Mock metrics (100% distinction, WER 0.0)
    assert m3.mock.total_attempts == 1
    assert m3.mock.completed_runs == 1
    assert m3.mock.failed_runs == 0
    assert m3.mock.distinction_preservation_rate == 100.0
    assert m3.mock.task_success_rate == 100.0
    assert m3.mock.average_wer == 0.0
    assert m3.mock.average_critical_field_accuracy == 100.0

    # Verify Live metrics (0% distinction, WER 0.1667)
    assert m3.live.total_attempts == 2
    assert m3.live.completed_runs == 1
    assert m3.live.failed_runs == 1
    assert m3.live.distinction_preservation_rate == 0.0
    assert m3.live.task_success_rate == 0.0
    assert m3.live.average_wer == round(live_comp.average_wer, 4)
    assert m3.live.average_critical_field_accuracy == 50.0
