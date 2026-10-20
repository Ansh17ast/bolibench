"""Regression and integration tests for BoliBench Phase 6 Judge-Ready Frontend Experience."""

import io
import pytest
from app.config import get_settings
from app.evaluation.dataset import load_seed_dataset
from app.evaluation.metrics import score_pair
from app.evaluation.schemas import ExecutionMode, FailureCategory, PairEvaluationResult, UtteranceEvaluationResult
from app.evaluation.storage import StorageManager
from app.routers.evaluation import engine


@pytest.mark.asyncio
async def test_frontend_page_structure_and_branding(async_client):
    """Proves index page renders the judge-ready branding, 6 navigation views, and UI controls."""
    resp = await async_client.get("/")
    assert resp.status_code == 200
    html = resp.text

    # 1. Branding & Header
    assert "BoliBench" in html
    assert "Indian Voice AI Crash-Test Lab" in html
    assert "Great Indian AI Internship Challenge" in html

    # 2. 6 Navigation Tabs
    assert 'id="tabBtnOverview"' in html
    assert 'id="tabBtnEvaluate"' in html
    assert 'id="tabBtnCatalogue"' in html
    assert 'id="tabBtnHistory"' in html
    assert 'id="tabBtnMethodology"' in html
    assert 'id="tabBtnTranscribe"' in html

    # 3. 6 Tab Panes
    assert 'id="overviewView"' in html
    assert 'id="evaluateView"' in html
    assert 'id="catalogueView"' in html
    assert 'id="historyView"' in html
    assert 'id="methodologyView"' in html
    assert 'id="transcribeView"' in html

    # 4. Mode Separation Elements in Overview
    assert 'id="liveEmptyStateBox"' in html
    assert 'id="liveStatsGrid"' in html
    assert 'id="mockEmptyStateBox"' in html
    assert 'id="mockStatsGrid"' in html
    assert 'id="healthTotalRuns"' in html
    assert 'id="healthCompletedRuns"' in html
    assert 'id="healthIncompleteRuns"' in html
    assert 'id="healthFailedRuns"' in html
    assert 'id="categoryBreakdownTable"' in html

    # 5. Audio Upload Previews and Format Notice in Paired Lab
    assert 'id="liveAudioPreviewA"' in html
    assert 'id="liveAudioPreviewB"' in html
    assert 'id="previewContainerA"' in html
    assert 'id="previewContainerB"' in html
    assert "WAV format only" in html
    assert "duration verifiably" in html or "&le; 60s" in html

    # 6. Status Filtering in History View
    assert 'data-status="all"' in html
    assert 'data-status="completed"' in html
    assert 'data-status="failed"' in html
    assert 'data-status="incomplete"' in html
    assert 'id="historyFullTableBody"' in html

    # 7. Benchmark Catalogue Elements
    assert 'id="catalogueSearchInput"' in html
    assert 'id="catalogueCategoryFilter"' in html
    assert 'id="catalogueGrid"' in html

    # 8. Methodology and Educational Section
    assert "The Indian Voice AI Evaluation Paradox" in html
    assert "Paired-Contrast Design" in html
    assert "Deterministic Scoring" in html
    assert "Transliteration &amp; Cross-Script WER Elevation" in html or "Transliteration & Cross-Script WER Elevation" in html


@pytest.mark.asyncio
async def test_catalogue_api_exposes_seed_cases_with_simulation_provenance(async_client):
    """Proves /api/eval/test-cases exposes all 24 curated seed cases with provenance and simulation classification."""
    resp = await async_client.get("/api/eval/test-cases")
    assert resp.status_code == 200
    cases = resp.json()
    assert len(cases) == 12
    for tc in cases:
        assert "id" in tc
        assert "category" in tc
        assert "language_code" in tc
        assert "utterance_a" in tc
        assert "utterance_b" in tc
        assert "expected_distinction" in tc
        assert "distinction_field" in tc
        assert "provenance" in tc
        assert tc["provenance"] == "hand_authored_text_simulation"
        assert tc["audio_condition"] == "text_only_simulation"


@pytest.mark.asyncio
async def test_metrics_separation_lifecycle(async_client, tmp_path):
    """Proves lifecycle of metric aggregation: empty state, mock runs, failed live, and completed live."""
    test_storage = StorageManager(db_path=tmp_path / "test_lifecycle_metrics.db")
    orig_storage = engine.storage
    engine.storage = test_storage

    try:
        pair = load_seed_dataset()[0]

        # Step 1: Empty state verification
        resp_empty = await async_client.get("/api/eval/metrics")
        assert resp_empty.status_code == 200
        d_empty = resp_empty.json()
        assert d_empty["total_runs"] == 0
        assert d_empty["completed_runs"] == 0
        assert d_empty["mock"]["completed_runs"] == 0
        assert d_empty["mock"]["average_wer"] is None
        assert d_empty["mock"]["distinction_preservation_rate"] is None
        assert d_empty["live"]["completed_runs"] == 0
        assert d_empty["live"]["average_wer"] is None
        assert d_empty["live"]["distinction_preservation_rate"] is None

        # Step 2: Add Completed Mock Run
        mock_run = score_pair(
            run_id="run_life_m1",
            test_case=pair,
            transcript_a="Do packet chahiye",
            transcript_b="Teen packet chahiye",
            execution_mode="mock",
        )
        test_storage.save_run(mock_run)

        resp_mock = await async_client.get("/api/eval/metrics")
        d_mock = resp_mock.json()
        assert d_mock["total_runs"] == 1
        assert d_mock["completed_runs"] == 1
        assert d_mock["mock"]["completed_runs"] == 1
        assert d_mock["mock"]["distinction_preservation_rate"] == 100.0
        assert d_mock["mock"]["average_wer"] == 0.0
        # Live remains with null rates
        assert d_mock["live"]["completed_runs"] == 0
        assert d_mock["live"]["distinction_preservation_rate"] is None

        # Step 3: Add Failed Live Run (e.g. 504 Timeout)
        failed_live = PairEvaluationResult(
            run_id="run_life_fail",
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
                error_message="Gateway Timeout 504",
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
            failure_explanation="Upstream gateway timeout on Utterance A",
        )
        test_storage.save_run(failed_live)

        resp_failed = await async_client.get("/api/eval/metrics")
        d_failed = resp_failed.json()
        assert d_failed["total_runs"] == 2
        assert d_failed["completed_runs"] == 1
        assert d_failed["failed_runs"] == 1
        assert d_failed["live"]["total_attempts"] == 1
        assert d_failed["live"]["completed_runs"] == 0
        assert d_failed["live"]["failed_runs"] == 1
        # The failed run must NOT set average_wer to 1.0!
        assert d_failed["live"]["average_wer"] is None
        assert d_failed["live"]["distinction_preservation_rate"] is None
        # Mock metrics remain completely isolated
        assert d_failed["mock"]["completed_runs"] == 1
        assert d_failed["mock"]["distinction_preservation_rate"] == 100.0

        # Step 4: Add Completed Live Run (e.g. simulated live response)
        live_completed = score_pair(
            run_id="run_life_live_comp",
            test_case=pair,
            transcript_a="Do packet chahiye",
            transcript_b="Teen packet chahiye",
            execution_mode="live",
            client_request_elapsed_ms=250.0,
        )
        test_storage.save_run(live_completed)

        resp_all = await async_client.get("/api/eval/metrics")
        d_all = resp_all.json()
        assert d_all["total_runs"] == 3
        assert d_all["completed_runs"] == 2
        assert d_all["failed_runs"] == 1
        assert d_all["mock_runs"] == 1
        assert d_all["live_runs"] == 2

        # Mock: 1 completed, 100% distinction
        assert d_all["mock"]["completed_runs"] == 1
        assert d_all["mock"]["distinction_preservation_rate"] == 100.0
        assert d_all["mock"]["average_wer"] == 0.0

        # Live: 2 attempts, 1 completed, 1 failed, 100% distinction across completed
        assert d_all["live"]["total_attempts"] == 2
        assert d_all["live"]["completed_runs"] == 1
        assert d_all["live"]["failed_runs"] == 1
        assert d_all["live"]["distinction_preservation_rate"] == 100.0
        assert d_all["live"]["average_wer"] == 0.0

    finally:
        engine.storage = orig_storage


@pytest.mark.asyncio
async def test_history_filtering_and_operational_error_isolation(async_client, tmp_path):
    """Proves history API supports status filtering and preserves operational error diagnostics."""
    test_storage = StorageManager(db_path=tmp_path / "test_hist_filter.db")
    orig_storage = engine.storage
    engine.storage = test_storage

    try:
        pair = load_seed_dataset()[0]

        # 1. Completed Mock Run
        r1 = score_pair(
            run_id="run_h_comp",
            test_case=pair,
            transcript_a="Do packet chahiye",
            transcript_b="Teen packet chahiye",
            execution_mode="mock",
        )
        test_storage.save_run(r1)

        # 2. Failed Live Run
        r2 = PairEvaluationResult(
            run_id="run_h_fail",
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
                error_message="HTTP 502 Bad Gateway from upstream",
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
            failure_explanation="Provider 502 Bad Gateway during Utterance A",
        )
        test_storage.save_run(r2)

        # 3. Incomplete Live Run
        r3 = PairEvaluationResult(
            run_id="run_h_incomp",
            test_case_id=pair.id,
            category=pair.category.value,
            language_code=pair.language_code,
            execution_mode=ExecutionMode.LIVE,
            audio_condition="clean_recorded",
            timestamp="2026-10-20T05:05:00Z",
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
                request_id="req_incomp_1",
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
                error_message="HTTP 504 Gateway Timeout",
            ),
            wer_a=0.0,
            wer_b=1.0,
            average_wer=0.5,
            critical_field_accuracy=0.5,
            semantic_distinction_preserved=False,
            task_success=False,
            primary_failure_category=FailureCategory.PROVIDER_ERROR,
            failure_explanation="Partial failure: Utterance A succeeded, Utterance B timed out",
        )
        test_storage.save_run(r3)

        # Test filtering
        resp_all = await async_client.get("/api/eval/results")
        assert len(resp_all.json()) == 3

        resp_comp = await async_client.get("/api/eval/results?status=completed")
        assert len(resp_comp.json()) == 1
        assert resp_comp.json()[0]["run_id"] == "run_h_comp"

        resp_failed = await async_client.get("/api/eval/results?status=failed")
        assert len(resp_failed.json()) == 1
        assert resp_failed.json()[0]["run_id"] == "run_h_fail"
        assert resp_failed.json()[0]["primary_failure_category"] == "provider_error"

        resp_incomp = await async_client.get("/api/eval/results?status=incomplete")
        assert len(resp_incomp.json()) == 1
        assert resp_incomp.json()[0]["run_id"] == "run_h_incomp"

    finally:
        engine.storage = orig_storage
