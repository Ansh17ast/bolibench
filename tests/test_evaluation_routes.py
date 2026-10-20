"""Route tests for BoliBench Paired-Contrast Evaluation API endpoints."""

import io
from unittest.mock import patch
import pytest

from fastapi import HTTPException
from app.config import get_settings
from app.services.gnani_client import GnaniServerError, GnaniTimeoutError
from app.services.live_guard import live_guard


@pytest.mark.asyncio
async def test_list_test_cases(async_client):
    """Test /api/eval/test-cases returns all 12 seed pairs."""
    response = await async_client.get("/api/eval/test-cases")
    assert response.status_code == 200

    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 12

    # Check required fields in schema
    first_item = data[0]
    assert "id" in first_item
    assert "category" in first_item
    assert "language_code" in first_item
    assert "utterance_a" in first_item
    assert "utterance_b" in first_item
    assert "distinction_field" in first_item


@pytest.mark.asyncio
async def test_get_single_test_case(async_client):
    """Test /api/eval/test-cases/{test_id} retrieves specific pair."""
    response = await async_client.get("/api/eval/test-cases/pair_qty_001")
    assert response.status_code == 200

    data = response.json()
    assert data["id"] == "pair_qty_001"
    assert data["distinction_field"] == "quantity"
    assert data["utterance_a"]["expected_critical_fields"]["quantity"] == 2
    assert data["utterance_b"]["expected_critical_fields"]["quantity"] == 3


@pytest.mark.asyncio
async def test_get_test_case_not_found(async_client):
    """Test /api/eval/test-cases/{test_id} returns 404 for unknown ID."""
    response = await async_client.get("/api/eval/test-cases/non_existent_id_999")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_run_mock_evaluation_ideal(async_client):
    """Test /api/eval/run-mock/{test_id} executes deterministic preserved evaluation."""
    response = await async_client.post("/api/eval/run-mock/pair_qty_001?scenario=ideal_preserved")
    assert response.status_code == 200

    data = response.json()
    assert data["execution_mode"] == "mock"
    assert data["semantic_distinction_preserved"] is True
    assert data["task_success"] is True
    assert data["average_wer"] == 0.0
    assert data["primary_failure_category"] == "none"


@pytest.mark.asyncio
async def test_run_mock_evaluation_crashed(async_client):
    """Test /api/eval/run-mock/{test_id} executes deterministic crashed evaluation."""
    response = await async_client.post("/api/eval/run-mock/pair_qty_001?scenario=crashed_distinction")
    assert response.status_code == 200

    data = response.json()
    assert data["execution_mode"] == "mock"
    assert data["semantic_distinction_preserved"] is False
    assert data["task_success"] is False
    assert data["primary_failure_category"] == "quantity_error"


@pytest.mark.asyncio
async def test_evaluate_custom_transcripts(async_client):
    """Test /api/eval/evaluate-transcripts scores arbitrary transcripts."""
    payload = {
        "test_case_id": "pair_neg_001",
        "transcript_a": "Haan subscription cancel kar do",
        "transcript_b": "Nahi subscription cancel mat karo",
        "execution_mode": "mock",
        "audio_condition": "clean_synthetic",
        "client_latency_ms": 150.0,
    }
    response = await async_client.post("/api/eval/evaluate-transcripts", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["semantic_distinction_preserved"] is True
    assert data["task_success"] is True
    assert data["primary_failure_category"] == "none"


@pytest.mark.asyncio
async def test_get_metrics(async_client):
    """Test /api/eval/metrics returns aggregate evaluation metrics with separated mock and live performance groups."""
    response = await async_client.get("/api/eval/metrics")
    assert response.status_code == 200

    data = response.json()
    assert "total_runs" in data
    assert "completed_runs" in data
    assert "failed_runs" in data
    assert "incomplete_runs" in data
    assert "mock_runs" in data
    assert "live_runs" in data
    assert "distinction_preservation_rate" in data
    assert "task_success_rate" in data
    assert "average_wer" in data
    assert "category_breakdown" in data

    # Verify separated mode metrics
    assert "mock" in data
    assert "live" in data
    for mode in ["mock", "live"]:
        assert "total_attempts" in data[mode]
        assert "completed_runs" in data[mode]
        assert "failed_runs" in data[mode]
        assert "incomplete_runs" in data[mode]
        assert "distinction_preservation_rate" in data[mode]
        assert "task_success_rate" in data[mode]
        assert "average_wer" in data[mode]
        assert "average_critical_field_accuracy" in data[mode]
        assert "category_breakdown" in data[mode]


@pytest.mark.asyncio
async def test_metrics_endpoint_isolates_mock_and_live_data(async_client, tmp_path):
    """Proves /api/eval/metrics correctly exposes isolated mock and live performance groups across completed runs."""
    from app.evaluation.storage import StorageManager
    from app.evaluation.metrics import score_pair
    from app.evaluation.dataset import load_seed_dataset
    from app.evaluation.schemas import PairEvaluationResult, ExecutionMode, FailureCategory, UtteranceEvaluationResult
    from app.routers.evaluation import engine

    test_storage = StorageManager(db_path=tmp_path / "test_api_metric_iso.db")
    orig_storage = engine.storage
    engine.storage = test_storage

    try:
        pair = load_seed_dataset()[0]

        # 1. Initial empty state via API
        resp = await async_client.get("/api/eval/metrics")
        assert resp.status_code == 200
        d_empty = resp.json()
        assert d_empty["total_runs"] == 0
        assert d_empty["mock"]["completed_runs"] == 0
        assert d_empty["mock"]["average_wer"] is None
        assert d_empty["mock"]["distinction_preservation_rate"] is None
        assert d_empty["live"]["completed_runs"] == 0
        assert d_empty["live"]["average_wer"] is None
        assert d_empty["live"]["distinction_preservation_rate"] is None

        # 2. Add Completed Mock Run
        mock_run = score_pair(
            run_id="run_api_m1",
            test_case=pair,
            transcript_a="Do packet chahiye",
            transcript_b="Teen packet chahiye",
            execution_mode="mock",
        )
        test_storage.save_run(mock_run)

        # 3. Add Incomplete Live Run (Utterance A succeeded, B failed)
        incomplete_live = PairEvaluationResult(
            run_id="run_api_incomp",
            test_case_id=pair.id,
            category=pair.category.value,
            language_code=pair.language_code,
            execution_mode=ExecutionMode.LIVE,
            audio_condition="clean_recorded",
            timestamp="2026-10-20T05:00:00Z",
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
                request_id="req_live_api_1",
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
            failure_explanation="Incomplete live evaluation",
        )
        test_storage.save_run(incomplete_live)

        resp2 = await async_client.get("/api/eval/metrics")
        assert resp2.status_code == 200
        d2 = resp2.json()
        assert d2["total_runs"] == 2
        assert d2["completed_runs"] == 1
        assert d2["incomplete_runs"] == 1
        assert d2["mock_runs"] == 1
        assert d2["live_runs"] == 1

        # Mock has 1 completed run with 100% distinction
        assert d2["mock"]["completed_runs"] == 1
        assert d2["mock"]["distinction_preservation_rate"] == 100.0
        assert d2["mock"]["average_wer"] == 0.0

        # Live has 0 completed runs -> performance averages remain null
        assert d2["live"]["total_attempts"] == 1
        assert d2["live"]["completed_runs"] == 0
        assert d2["live"]["incomplete_runs"] == 1
        assert d2["live"]["distinction_preservation_rate"] is None
        assert d2["live"]["task_success_rate"] is None
        assert d2["live"]["average_wer"] is None

    finally:
        engine.storage = orig_storage



@pytest.mark.asyncio
async def test_export_endpoints(async_client):
    """Test /api/eval/export returns JSON and CSV file downloads."""
    # JSON export
    resp_json = await async_client.get("/api/eval/export?format=json")
    assert resp_json.status_code == 200
    assert resp_json.headers.get("content-type") == "application/json"

    # CSV export
    resp_csv = await async_client.get("/api/eval/export?format=csv")
    assert resp_csv.status_code == 200
    assert "text/csv" in resp_csv.headers.get("content-type", "")
    assert "run_id,test_case_id" in resp_csv.text


# -----------------------------------------------------------------------------
# Phase 2: Telemetry Correction, Live Paired-Audio, and Credit Protection Tests
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_mock_latency_not_represented_as_real_measured_latency(async_client):
    """Proves mock runs never report fabricated numbers in client_measured_latency_ms."""
    response = await async_client.post("/api/eval/run-mock/pair_qty_001?scenario=ideal_preserved")
    assert response.status_code == 200
    data = response.json()

    # Measured latency fields must remain strictly null for mock runs
    assert data["client_measured_latency_ms"] is None
    assert data["provider_processing_time_ms"] is None
    assert data["utterance_a"]["latency_ms"] is None
    assert data["utterance_b"]["latency_ms"] is None
    assert data["utterance_a"]["request_id"] is None
    assert data["utterance_b"]["request_id"] is None

    # Custom transcript mock evaluation
    payload = {
        "test_case_id": "pair_qty_001",
        "transcript_a": "Do packet chahiye",
        "transcript_b": "Teen packet chahiye",
        "execution_mode": "mock",
        "client_latency_ms": 25.0,  # Simulated latency provided by caller
    }
    resp_custom = await async_client.post("/api/eval/evaluate-transcripts", json=payload)
    assert resp_custom.status_code == 200
    custom_data = resp_custom.json()

    # Must be categorized as simulated_latency_ms, NOT client_measured_latency_ms
    assert custom_data["client_measured_latency_ms"] is None
    assert custom_data["provider_processing_time_ms"] is None
    assert custom_data["simulated_latency_ms"] == 25.0


async def _get_live_auth(async_client, test_id: str = "pair_qty_001"):
    """Helper for tests to obtain an authorized single-use token and header."""
    resp = await async_client.post("/api/eval/authorize-live", json={"test_id": test_id})
    assert resp.status_code == 200, f"Auth failed: {resp.text}"
    token = resp.json()["confirmation_token"]
    headers = {"X-BoliBench-Confirm-Live": "CONFIRM_LIVE_EVALUATION"}
    data = {"confirmation_token": token}
    return headers, data


@pytest.mark.asyncio
async def test_run_live_audio_validates_both_uploads(async_client, valid_wav_bytes, long_wav_bytes, monkeypatch):
    """Proves POST /api/eval/run-live/{test_id} validates format, size, duration for both files."""
    monkeypatch.setenv("GNANI_API_KEY", "test_mock_secret_key_123")
    monkeypatch.setenv("LIVE_EVALUATION_ENABLED", "true")
    get_settings.cache_clear()

    with patch("app.services.gnani_client.GnaniPrismaClient.transcribe") as mock_transcribe:
        # 1. Reject invalid file extension on Utterance A
        headers, data = await _get_live_auth(async_client, "pair_qty_001")
        files_bad_a = {
            "audio_file_a": ("bad_a.py", io.BytesIO(b"print('invalid')"), "text/x-python"),
            "audio_file_b": ("good_b.wav", io.BytesIO(valid_wav_bytes), "audio/wav"),
        }
        resp = await async_client.post("/api/eval/run-live/pair_qty_001", files=files_bad_a, data=data, headers=headers)
        assert resp.status_code == 400
        assert resp.json()["error_type"] == "UNSUPPORTED_EXTENSION"

        # 2. Reject empty file on Utterance A
        headers, data = await _get_live_auth(async_client, "pair_qty_001")
        files_empty_a = {
            "audio_file_a": ("empty_a.wav", io.BytesIO(b""), "audio/wav"),
            "audio_file_b": ("good_b.wav", io.BytesIO(valid_wav_bytes), "audio/wav"),
        }
        resp = await async_client.post("/api/eval/run-live/pair_qty_001", files=files_empty_a, data=data, headers=headers)
        assert resp.status_code == 400
        assert resp.json()["error_type"] == "EMPTY_FILE"

        # 3. Reject invalid file extension on Utterance B
        headers, data = await _get_live_auth(async_client, "pair_qty_001")
        files_bad_b = {
            "audio_file_a": ("good_a.wav", io.BytesIO(valid_wav_bytes), "audio/wav"),
            "audio_file_b": ("bad_b.txt", io.BytesIO(b"not an audio"), "text/plain"),
        }
        resp = await async_client.post("/api/eval/run-live/pair_qty_001", files=files_bad_b, data=data, headers=headers)
        assert resp.status_code == 400
        assert resp.json()["error_type"] == "UNSUPPORTED_EXTENSION"

        # 4. Reject empty file on Utterance B
        headers, data = await _get_live_auth(async_client, "pair_qty_001")
        files_empty_b = {
            "audio_file_a": ("good_a.wav", io.BytesIO(valid_wav_bytes), "audio/wav"),
            "audio_file_b": ("empty_b.wav", io.BytesIO(b""), "audio/wav"),
        }
        resp = await async_client.post("/api/eval/run-live/pair_qty_001", files=files_empty_b, data=data, headers=headers)
        assert resp.status_code == 400
        assert resp.json()["error_type"] == "EMPTY_FILE"

        # 5. Reject duration > 60s on Utterance B
        headers, data = await _get_live_auth(async_client, "pair_qty_001")
        files_long_b = {
            "audio_file_a": ("good_a.wav", io.BytesIO(valid_wav_bytes), "audio/wav"),
            "audio_file_b": ("long_b.wav", io.BytesIO(long_wav_bytes), "audio/wav"),
        }
        resp = await async_client.post("/api/eval/run-live/pair_qty_001", files=files_long_b, data=data, headers=headers)
        assert resp.status_code == 400
        assert resp.json()["error_type"] == "DURATION_EXCEEDED"

        # 6. Reject non-WAV container (e.g. MP3) on live route because duration cannot be verified
        headers, data = await _get_live_auth(async_client, "pair_qty_001")
        mp3_bytes = b"ID3\x04\x00\x00\x00\x00\x00\x00" + b"\xff\xfb\x90d" + b"\x00" * 100
        files_mp3 = {
            "audio_file_a": ("speech_a.mp3", io.BytesIO(mp3_bytes), "audio/mpeg"),
            "audio_file_b": ("good_b.wav", io.BytesIO(valid_wav_bytes), "audio/wav"),
        }
        resp = await async_client.post("/api/eval/run-live/pair_qty_001", files=files_mp3, data=data, headers=headers)
        assert resp.status_code == 400
        assert resp.json()["error_type"] == "UNSUPPORTED_LIVE_FORMAT"
        assert "formats with verified duration" in resp.json()["message"]

        mock_transcribe.assert_not_called()



@pytest.mark.asyncio
async def test_run_live_invalid_test_id(async_client, valid_wav_bytes, monkeypatch):
    """Proves POST /api/eval/run-live/{test_id} returns 404 for unknown test cases on authorize."""
    monkeypatch.setenv("GNANI_API_KEY", "test_mock_secret_key_123")
    monkeypatch.setenv("LIVE_EVALUATION_ENABLED", "true")
    get_settings.cache_clear()

    # Authorizing unknown test returns 404
    resp = await async_client.post("/api/eval/authorize-live", json={"test_id": "non_existent_test_999"})
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()


@pytest.mark.asyncio
async def test_run_live_missing_credentials(async_client, valid_wav_bytes, monkeypatch):
    """Proves POST /api/eval/run-live/{test_id} returns 401 when API credentials are unset."""
    monkeypatch.setenv("GNANI_API_KEY", "")
    monkeypatch.setenv("LIVE_EVALUATION_ENABLED", "true")
    get_settings.cache_clear()

    headers, data = await _get_live_auth(async_client, "pair_qty_001")
    files = {
        "audio_file_a": ("clip_a.wav", io.BytesIO(valid_wav_bytes), "audio/wav"),
        "audio_file_b": ("clip_b.wav", io.BytesIO(valid_wav_bytes), "audio/wav"),
    }
    with patch("app.services.gnani_client.GnaniPrismaClient.transcribe") as mock_transcribe:
        resp = await async_client.post("/api/eval/run-live/pair_qty_001", files=files, data=data, headers=headers)
        assert resp.status_code == 401
        assert "not configured" in resp.json()["detail"].lower()
        mock_transcribe.assert_not_called()


@pytest.mark.asyncio
async def test_run_live_missing_credentials_safe_when_env_file_supplies_key(async_client, valid_wav_bytes, monkeypatch, tmp_path):
    """Regression test: Proves paired live evaluation remains safe with 401 response and ZERO
    provider calls, even when an underlying .env file supplies a real non-empty API key.
    """
    simulated_env = tmp_path / "simulated_user_eval.env"
    simulated_env.write_text(
        "GNANI_API_KEY=simulated_real_customer_key_abcdef123456\n"
        "LIVE_EVALUATION_ENABLED=true\n",
        encoding="utf-8",
    )

    from app.config import get_settings
    with patch("app.config.Settings.model_config", {"env_file": str(simulated_env), "env_file_encoding": "utf-8", "extra": "ignore"}):
        monkeypatch.setenv("GNANI_API_KEY", "")
        monkeypatch.setenv("LIVE_EVALUATION_ENABLED", "true")
        get_settings.cache_clear()

        safe_settings = get_settings()
        assert safe_settings.is_api_configured is False
        assert safe_settings.gnani_api_key == ""

        headers, data = await _get_live_auth(async_client, "pair_qty_001")
        files = {
            "audio_file_a": ("clip_a.wav", io.BytesIO(valid_wav_bytes), "audio/wav"),
            "audio_file_b": ("clip_b.wav", io.BytesIO(valid_wav_bytes), "audio/wav"),
        }
        with patch("app.services.gnani_client.GnaniPrismaClient.transcribe") as mock_transcribe:
            resp = await async_client.post("/api/eval/run-live/pair_qty_001", files=files, data=data, headers=headers)
            assert resp.status_code == 401
            assert "not configured" in resp.json()["detail"].lower()
            mock_transcribe.assert_not_called()


@pytest.mark.asyncio
async def test_run_live_success_with_mocked_provider(async_client, valid_wav_bytes, monkeypatch):
    """Proves paired live evaluation succeeds with deterministic mocked provider without network calls."""
    monkeypatch.setenv("GNANI_API_KEY", "test_mock_secret_key_123")
    monkeypatch.setenv("LIVE_EVALUATION_ENABLED", "true")
    get_settings.cache_clear()

    mock_resp_a = {
        "success": True,
        "transcript": "Do packet chahiye",
        "request_id": "req_live_test_a_001",
        "model": "gnani-prisma-v2.5",
        "client_request_elapsed_ms": 110.5,
        "server_timestamp": "20261020_040000.000",
        "language_code": "hi-IN",
        "format": "transcribe",
        "is_mocked": False,
    }
    mock_resp_b = {
        "success": True,
        "transcript": "Teen packet chahiye",
        "request_id": "req_live_test_b_002",
        "model": "gnani-prisma-v2.5",
        "client_request_elapsed_ms": 115.0,
        "server_timestamp": "20261020_040001.000",
        "language_code": "hi-IN",
        "format": "transcribe",
        "is_mocked": False,
    }

    headers, data = await _get_live_auth(async_client, "pair_qty_001")
    data["audio_condition"] = "clean_recorded"
    files = {
        "audio_file_a": ("utterance_a.wav", io.BytesIO(valid_wav_bytes), "audio/wav"),
        "audio_file_b": ("utterance_b.wav", io.BytesIO(valid_wav_bytes), "audio/wav"),
    }

    with patch("app.services.gnani_client.GnaniPrismaClient.transcribe", side_effect=[mock_resp_a, mock_resp_b]) as mock_transcribe:
        response = await async_client.post("/api/eval/run-live/pair_qty_001", files=files, data=data, headers=headers)
        assert response.status_code == 200
        assert mock_transcribe.call_count == 2

        # Verify contract compliance: format_mode="transcribe" and preferred_language="hi-IN"
        call_a = mock_transcribe.call_args_list[0].kwargs
        assert call_a["format_mode"] == "transcribe"
        assert call_a["preferred_language"] == "hi-IN"
        assert call_a["language_code"] == "hi-IN"

        call_b = mock_transcribe.call_args_list[1].kwargs
        assert call_b["format_mode"] == "transcribe"
        assert call_b["preferred_language"] == "hi-IN"
        assert call_b["language_code"] == "hi-IN"

        result = response.json()
        assert result["overall_status"] == "completed"
        assert result["execution_mode"] == "live"
        assert result["audio_condition"] == "clean_recorded"
        assert result["semantic_distinction_preserved"] is True
        assert result["task_success"] is True

        # Check telemetry precision: client-measured elapsed is populated, provider is null
        assert result["client_measured_latency_ms"] is not None
        assert result["client_measured_latency_ms"] == pytest.approx(225.5, abs=0.1)
        assert result["client_request_elapsed_ms"] == pytest.approx(225.5, abs=0.1)
        assert result["provider_processing_time_ms"] is None  # Gnani STT does not report provider latency
        assert result["simulated_latency_ms"] is None

        # Check utterance-level telemetry and request IDs
        assert result["utterance_a"]["request_id"] == "req_live_test_a_001"
        assert result["utterance_b"]["request_id"] == "req_live_test_b_002"
        assert result["utterance_a"]["latency_ms"] == 110.5
        assert result["utterance_a"]["client_request_elapsed_ms"] == 110.5
        assert result["utterance_a"]["provider_processing_time_ms"] is None
        assert result["utterance_b"]["latency_ms"] == 115.0
        assert result["utterance_b"]["client_request_elapsed_ms"] == 115.0
        assert result["utterance_b"]["provider_processing_time_ms"] is None


@pytest.mark.asyncio
async def test_run_live_partial_failure_first_fails(async_client, valid_wav_bytes, monkeypatch):
    """Proves failure of Utterance A never evaluates Utterance B, but records a safe failed audit run."""
    monkeypatch.setenv("GNANI_API_KEY", "test_mock_secret_key_123")
    monkeypatch.setenv("LIVE_EVALUATION_ENABLED", "true")
    get_settings.cache_clear()

    headers, data = await _get_live_auth(async_client, "pair_qty_001")
    files = {
        "audio_file_a": ("utterance_a.wav", io.BytesIO(valid_wav_bytes), "audio/wav"),
        "audio_file_b": ("utterance_b.wav", io.BytesIO(valid_wav_bytes), "audio/wav"),
    }

    with patch(
        "app.services.gnani_client.GnaniPrismaClient.transcribe",
        side_effect=GnaniServerError("Gnani recognition cluster error", status_code=500),
    ) as mock_transcribe:
        response = await async_client.post("/api/eval/run-live/pair_qty_001", files=files, data=data, headers=headers)
        assert response.status_code == 500
        assert mock_transcribe.call_count == 1  # Utterance B was never called

        from app.routers.evaluation import engine
        runs = engine.list_runs(limit=1)
        assert len(runs) >= 1
        latest = runs[0]
        assert latest.overall_status == "failed"
        assert latest.utterance_a.status == "failed"
        assert latest.utterance_b.status == "skipped"
        assert latest.task_success is False
        assert latest.semantic_distinction_preserved is False
        assert latest.primary_failure_category.value == "provider_error"


@pytest.mark.asyncio
async def test_run_live_partial_failure_second_fails(async_client, valid_wav_bytes, monkeypatch):
    """Proves if Utterance A succeeds but Utterance B fails, an incomplete audit run is safely recorded."""
    monkeypatch.setenv("GNANI_API_KEY", "test_mock_secret_key_123")
    monkeypatch.setenv("LIVE_EVALUATION_ENABLED", "true")
    get_settings.cache_clear()

    mock_resp_a = {
        "success": True,
        "transcript": "Do packet chahiye",
        "request_id": "req_live_test_a_ok",
        "model": "gnani-prisma-v2.5",
        "client_request_elapsed_ms": 105.0,
        "server_timestamp": "20261020_040000.000",
        "language_code": "hi-IN",
        "format": "transcribe",
        "is_mocked": False,
    }

    headers, data = await _get_live_auth(async_client, "pair_qty_001")
    files = {
        "audio_file_a": ("utterance_a.wav", io.BytesIO(valid_wav_bytes), "audio/wav"),
        "audio_file_b": ("utterance_b.wav", io.BytesIO(valid_wav_bytes), "audio/wav"),
    }

    with patch(
        "app.services.gnani_client.GnaniPrismaClient.transcribe",
        side_effect=[mock_resp_a, GnaniTimeoutError(timeout_seconds=30.0)],
    ) as mock_transcribe:
        response = await async_client.post("/api/eval/run-live/pair_qty_001", files=files, data=data, headers=headers)
        assert response.status_code == 504
        assert mock_transcribe.call_count == 2

        from app.routers.evaluation import engine
        runs = engine.list_runs(limit=1)
        assert len(runs) >= 1
        latest = runs[0]
        assert latest.overall_status == "incomplete"
        assert latest.utterance_a.status == "completed"
        assert latest.utterance_a.request_id == "req_live_test_a_ok"
        assert latest.utterance_b.status == "failed"
        assert latest.task_success is False
        assert latest.semantic_distinction_preserved is False
        assert latest.primary_failure_category.value == "provider_error"
        assert "Utterance A succeeded" in latest.failure_explanation
        assert "credits may have been consumed" in latest.failure_explanation


# -----------------------------------------------------------------------------
# Phase 3: Credit Guard, Anti-CSRF, and Authorization Hardening Tests
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_live_missing_confirmation_token_rejected(async_client, valid_wav_bytes, monkeypatch):
    """Proves live evaluation rejects requests without authorization token before calling provider."""
    monkeypatch.setenv("GNANI_API_KEY", "test_mock_secret_key_123")
    monkeypatch.setenv("LIVE_EVALUATION_ENABLED", "true")
    get_settings.cache_clear()

    files = {
        "audio_file_a": ("utterance_a.wav", io.BytesIO(valid_wav_bytes), "audio/wav"),
        "audio_file_b": ("utterance_b.wav", io.BytesIO(valid_wav_bytes), "audio/wav"),
    }
    # No confirmation_token provided
    headers = {"X-BoliBench-Confirm-Live": "CONFIRM_LIVE_EVALUATION"}
    with patch("app.services.gnani_client.GnaniPrismaClient.transcribe") as mock_transcribe:
        resp = await async_client.post("/api/eval/run-live/pair_qty_001", files=files, headers=headers)
        assert resp.status_code == 403
        assert "authorization token" in resp.json()["detail"].lower()
        mock_transcribe.assert_not_called()


@pytest.mark.asyncio
async def test_run_live_missing_csrf_header_rejected(async_client, valid_wav_bytes, monkeypatch):
    """Proves live evaluation rejects requests without X-BoliBench-Confirm-Live header."""
    monkeypatch.setenv("GNANI_API_KEY", "test_mock_secret_key_123")
    monkeypatch.setenv("LIVE_EVALUATION_ENABLED", "true")
    get_settings.cache_clear()

    _, data = await _get_live_auth(async_client, "pair_qty_001")
    files = {
        "audio_file_a": ("utterance_a.wav", io.BytesIO(valid_wav_bytes), "audio/wav"),
        "audio_file_b": ("utterance_b.wav", io.BytesIO(valid_wav_bytes), "audio/wav"),
    }
    # Missing header
    with patch("app.services.gnani_client.GnaniPrismaClient.transcribe") as mock_transcribe:
        resp = await async_client.post("/api/eval/run-live/pair_qty_001", files=files, data=data)
        assert resp.status_code == 403
        assert "authorization header" in resp.json()["detail"].lower()
        mock_transcribe.assert_not_called()


@pytest.mark.asyncio
async def test_run_live_wrong_csrf_header_value_rejected(async_client, valid_wav_bytes, monkeypatch):
    """Regression test: Proves live evaluation rejects legacy or mismatched confirmation header values.

    Specifically verifies that sending 'authorized-by-local-operator' instead of the required
    'CONFIRM_LIVE_EVALUATION' results in an immediate 403 Forbidden rejection before reading files
    or invoking provider APIs.
    """
    monkeypatch.setenv("GNANI_API_KEY", "test_mock_secret_key_123")
    monkeypatch.setenv("LIVE_EVALUATION_ENABLED", "true")
    get_settings.cache_clear()

    _, data = await _get_live_auth(async_client, "pair_qty_001")
    files = {
        "audio_file_a": ("utterance_a.wav", io.BytesIO(valid_wav_bytes), "audio/wav"),
        "audio_file_b": ("utterance_b.wav", io.BytesIO(valid_wav_bytes), "audio/wav"),
    }
    # Send the wrong header value that previously caused the frontend contract mismatch bug
    wrong_headers = {"X-BoliBench-Confirm-Live": "authorized-by-local-operator"}
    with patch("app.services.gnani_client.GnaniPrismaClient.transcribe") as mock_transcribe:
        resp = await async_client.post(
            "/api/eval/run-live/pair_qty_001",
            files=files,
            data=data,
            headers=wrong_headers,
        )
        assert resp.status_code == 403
        assert "Missing required authorization header" in resp.json()["detail"]
        assert "CONFIRM_LIVE_EVALUATION" in resp.json()["detail"]
        mock_transcribe.assert_not_called()


@pytest.mark.asyncio
async def test_authorize_live_exposes_explicit_header_contract(async_client, monkeypatch):
    """Proves /api/eval/authorize-live explicitly returns the required header name and expected value."""
    monkeypatch.setenv("LIVE_EVALUATION_ENABLED", "true")
    get_settings.cache_clear()

    resp = await async_client.post("/api/eval/authorize-live", json={"test_id": "pair_qty_001"})
    assert resp.status_code == 200
    payload = resp.json()
    assert "required_header" in payload
    assert payload["required_header"] == "X-BoliBench-Confirm-Live"
    assert "required_header_value" in payload
    assert payload["required_header_value"] == "CONFIRM_LIVE_EVALUATION"
    assert "confirmation_token" in payload


@pytest.mark.asyncio
async def test_run_live_invalid_token_rejected(async_client, valid_wav_bytes, monkeypatch):
    """Proves fabricated or invalid authorization tokens are rejected."""
    monkeypatch.setenv("GNANI_API_KEY", "test_mock_secret_key_123")
    monkeypatch.setenv("LIVE_EVALUATION_ENABLED", "true")
    get_settings.cache_clear()

    files = {
        "audio_file_a": ("utterance_a.wav", io.BytesIO(valid_wav_bytes), "audio/wav"),
        "audio_file_b": ("utterance_b.wav", io.BytesIO(valid_wav_bytes), "audio/wav"),
    }
    headers = {"X-BoliBench-Confirm-Live": "CONFIRM_LIVE_EVALUATION"}
    data = {"confirmation_token": "fabricated_token_abcdef123456"}

    with patch("app.services.gnani_client.GnaniPrismaClient.transcribe") as mock_transcribe:
        resp = await async_client.post("/api/eval/run-live/pair_qty_001", files=files, data=data, headers=headers)
        assert resp.status_code == 403
        assert "invalid" in resp.json()["detail"].lower()
        mock_transcribe.assert_not_called()


@pytest.mark.asyncio
async def test_run_live_token_reuse_rejected(async_client, valid_wav_bytes, monkeypatch):
    """Proves single-use tokens cannot be replayed for multiple live evaluations."""
    monkeypatch.setenv("GNANI_API_KEY", "test_mock_secret_key_123")
    monkeypatch.setenv("LIVE_EVALUATION_ENABLED", "true")
    get_settings.cache_clear()

    mock_resp = {
        "success": True,
        "transcript": "Sample transcript",
        "request_id": "req_1",
        "model": "gnani-prisma-v2.5",
        "client_request_elapsed_ms": 100.0,
        "server_timestamp": "20261020_040000.000",
        "language_code": "hi-IN",
        "format": "verbatim",
        "is_mocked": False,
    }

    headers, data = await _get_live_auth(async_client, "pair_qty_001")
    files_1 = {
        "audio_file_a": ("utterance_a.wav", io.BytesIO(valid_wav_bytes), "audio/wav"),
        "audio_file_b": ("utterance_b.wav", io.BytesIO(valid_wav_bytes), "audio/wav"),
    }

    with patch("app.services.gnani_client.GnaniPrismaClient.transcribe", return_value=mock_resp) as mock_transcribe:
        # First attempt: succeeds and consumes token
        resp1 = await async_client.post("/api/eval/run-live/pair_qty_001", files=files_1, data=data, headers=headers)
        assert resp1.status_code == 200
        assert mock_transcribe.call_count == 2

        # Second attempt with same token: rejected as consumed
        files_2 = {
            "audio_file_a": ("utterance_a.wav", io.BytesIO(valid_wav_bytes), "audio/wav"),
            "audio_file_b": ("utterance_b.wav", io.BytesIO(valid_wav_bytes), "audio/wav"),
        }
        resp2 = await async_client.post("/api/eval/run-live/pair_qty_001", files=files_2, data=data, headers=headers)
        assert resp2.status_code == 403
        assert "already been consumed" in resp2.json()["detail"].lower()
        assert mock_transcribe.call_count == 2  # No new calls made!


@pytest.mark.asyncio
async def test_run_live_cross_origin_rejected(async_client, valid_wav_bytes, monkeypatch):
    """Proves cross-origin requests from untrusted origins are strictly rejected with 403."""
    monkeypatch.setenv("GNANI_API_KEY", "test_mock_secret_key_123")
    monkeypatch.setenv("LIVE_EVALUATION_ENABLED", "true")
    get_settings.cache_clear()

    headers, data = await _get_live_auth(async_client, "pair_qty_001")
    # Attacker origin
    headers["Origin"] = "https://malicious-website.com"
    files = {
        "audio_file_a": ("utterance_a.wav", io.BytesIO(valid_wav_bytes), "audio/wav"),
        "audio_file_b": ("utterance_b.wav", io.BytesIO(valid_wav_bytes), "audio/wav"),
    }

    with patch("app.services.gnani_client.GnaniPrismaClient.transcribe") as mock_transcribe:
        resp = await async_client.post("/api/eval/run-live/pair_qty_001", files=files, data=data, headers=headers)
        assert resp.status_code == 403
        assert "untrusted origin" in resp.json()["detail"].lower()
        mock_transcribe.assert_not_called()


@pytest.mark.asyncio
async def test_run_live_sec_fetch_site_cross_site_rejected(async_client, valid_wav_bytes, monkeypatch):
    """Proves browser Sec-Fetch-Site: cross-site is immediately rejected with 403."""
    monkeypatch.setenv("GNANI_API_KEY", "test_mock_secret_key_123")
    monkeypatch.setenv("LIVE_EVALUATION_ENABLED", "true")
    get_settings.cache_clear()

    headers, data = await _get_live_auth(async_client, "pair_qty_001")
    headers["Sec-Fetch-Site"] = "cross-site"
    files = {
        "audio_file_a": ("utterance_a.wav", io.BytesIO(valid_wav_bytes), "audio/wav"),
        "audio_file_b": ("utterance_b.wav", io.BytesIO(valid_wav_bytes), "audio/wav"),
    }

    with patch("app.services.gnani_client.GnaniPrismaClient.transcribe") as mock_transcribe:
        resp = await async_client.post("/api/eval/run-live/pair_qty_001", files=files, data=data, headers=headers)
        assert resp.status_code == 403
        assert "cross-origin" in resp.json()["detail"].lower()
        mock_transcribe.assert_not_called()


@pytest.mark.asyncio
async def test_authorize_live_cross_origin_rejected(async_client, monkeypatch):
    """Proves authorization endpoint itself rejects cross-origin requests."""
    monkeypatch.setenv("LIVE_EVALUATION_ENABLED", "true")
    get_settings.cache_clear()
    resp = await async_client.post(
        "/api/eval/authorize-live",
        json={"test_id": "pair_qty_001"},
        headers={"Origin": "https://malicious-site.com"},
    )
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_run_live_token_mismatched_test_id(async_client, valid_wav_bytes, monkeypatch):
    """Proves a token issued for test_id A cannot be used to run test_id B."""
    monkeypatch.setenv("GNANI_API_KEY", "test_mock_secret_key_123")
    monkeypatch.setenv("LIVE_EVALUATION_ENABLED", "true")
    get_settings.cache_clear()

    headers, data = await _get_live_auth(async_client, "pair_qty_001")
    files = {
        "audio_file_a": ("utterance_a.wav", io.BytesIO(valid_wav_bytes), "audio/wav"),
        "audio_file_b": ("utterance_b.wav", io.BytesIO(valid_wav_bytes), "audio/wav"),
    }

    with patch("app.services.gnani_client.GnaniPrismaClient.transcribe") as mock_transcribe:
        # Token was issued for pair_qty_001, but calling pair_neg_001
        resp = await async_client.post("/api/eval/run-live/pair_neg_001", files=files, data=data, headers=headers)
        assert resp.status_code == 403
        assert "different test case" in resp.json()["detail"].lower()
        mock_transcribe.assert_not_called()


@pytest.mark.asyncio
async def test_live_latency_when_provider_returns_processing_duration(async_client, valid_wav_bytes, monkeypatch):
    """Proves provider_processing_time_ms is populated IF the provider genuinely reports it."""
    monkeypatch.setenv("GNANI_API_KEY", "test_mock_secret_key_123")
    monkeypatch.setenv("LIVE_EVALUATION_ENABLED", "true")
    get_settings.cache_clear()

    mock_resp_a = {
        "success": True,
        "transcript": "Do packet chahiye",
        "request_id": "req_a",
        "model": "gnani-prisma-v2.5",
        "client_request_elapsed_ms": 120.0,
        "provider_processing_time_ms": 45.0,  # Provider reported
        "server_timestamp": "20261020_040000.000",
        "language_code": "hi-IN",
        "format": "verbatim",
        "is_mocked": False,
    }
    mock_resp_b = {
        "success": True,
        "transcript": "Teen packet chahiye",
        "request_id": "req_b",
        "model": "gnani-prisma-v2.5",
        "client_request_elapsed_ms": 130.0,
        "provider_processing_time_ms": 50.0,  # Provider reported
        "server_timestamp": "20261020_040001.000",
        "language_code": "hi-IN",
        "format": "verbatim",
        "is_mocked": False,
    }

    headers, data = await _get_live_auth(async_client, "pair_qty_001")
    files = {
        "audio_file_a": ("utterance_a.wav", io.BytesIO(valid_wav_bytes), "audio/wav"),
        "audio_file_b": ("utterance_b.wav", io.BytesIO(valid_wav_bytes), "audio/wav"),
    }

    with patch("app.services.gnani_client.GnaniPrismaClient.transcribe", side_effect=[mock_resp_a, mock_resp_b]):
        resp = await async_client.post("/api/eval/run-live/pair_qty_001", files=files, data=data, headers=headers)
        assert resp.status_code == 200
        res = resp.json()
        assert res["client_measured_latency_ms"] == pytest.approx(250.0, abs=0.1)
        assert res["provider_processing_time_ms"] == pytest.approx(95.0, abs=0.1)
        assert res["utterance_a"]["provider_processing_time_ms"] == 45.0
        assert res["utterance_b"]["provider_processing_time_ms"] == 50.0


@pytest.mark.asyncio
async def test_csv_and_json_exports_preserve_latency_semantics(async_client):
    """Proves CSV and JSON exports contain client_request_elapsed_ms and preserve null provider processing time."""
    # Run a mock evaluation to have data
    await async_client.post("/api/eval/run-mock/pair_qty_001?scenario=ideal_preserved")

    # CSV export
    csv_resp = await async_client.get("/api/eval/export?format=csv")
    assert csv_resp.status_code == 200
    csv_text = csv_resp.text
    assert "client_request_elapsed_ms" in csv_text
    assert "client_measured_latency_ms" in csv_text
    assert "provider_processing_time_ms" in csv_text
    assert "simulated_latency_ms" in csv_text

    # JSON export
    json_resp = await async_client.get("/api/eval/export?format=json")
    assert json_resp.status_code == 200
    runs = json_resp.json()
    assert len(runs) > 0
    # For mock run: measured latency is None
    assert runs[0]["client_measured_latency_ms"] is None
    assert runs[0]["provider_processing_time_ms"] is None


@pytest.mark.asyncio
async def test_api_credentials_and_tokens_never_exposed(async_client, monkeypatch):
    """Proves API credentials and confirmation tokens never appear in API responses, exports, or Settings repr."""
    secret_key = "super_secret_gnani_key_xyz987"
    monkeypatch.setenv("GNANI_API_KEY", secret_key)
    monkeypatch.setenv("LIVE_EVALUATION_ENABLED", "true")
    get_settings.cache_clear()
    settings = get_settings()

    # Settings repr must mask secret
    assert secret_key not in repr(settings)
    assert "gnani_api_configured=True" in repr(settings)

    # Health check
    resp_health = await async_client.get("/health")
    assert secret_key not in resp_health.text

    # Test cases list
    resp_cases = await async_client.get("/api/eval/test-cases")
    assert secret_key not in resp_cases.text

    # Metrics
    resp_metrics = await async_client.get("/api/eval/metrics")
    assert secret_key not in resp_metrics.text

    # JSON export
    resp_export = await async_client.get("/api/eval/export?format=json")
    assert secret_key not in resp_export.text

    # Authorize endpoint returns confirmation token, but never leaks credentials
    auth_resp = await async_client.post("/api/eval/authorize-live", json={"test_id": "pair_qty_001"})
    assert auth_resp.status_code == 200
    token = auth_resp.json()["confirmation_token"]
    assert secret_key not in auth_resp.text

    # Stored runs never contain the confirmation token
    resp_runs = await async_client.get("/api/eval/results")
    assert token not in resp_runs.text


@pytest.mark.asyncio
async def test_programme_credits_protected_against_unauthorized_network(async_client, valid_wav_bytes, monkeypatch):
    """Proves automated tests cannot consume credits by verifying mocked client isolation."""
    monkeypatch.setenv("GNANI_API_KEY", "test_mock_secret_key_123")
    get_settings.cache_clear()

    # Verify mock mode uses 0 network calls
    files = {"audio_file": ("test.wav", io.BytesIO(valid_wav_bytes), "audio/wav")}
    data = {"language_code": "hi-IN", "mock_mode": "true"}
    resp = await async_client.post("/api/transcribe", files=files, data=data)
    assert resp.status_code == 200
    assert resp.json()["is_mocked"] is True


# =============================================================================
# Phase 4: Deployment Safety & Acoustic Condition Integrity Tests
# =============================================================================

@pytest.mark.asyncio
async def test_live_evaluation_disabled_by_default(async_client, valid_wav_bytes, monkeypatch):
    """Proves live evaluation is disabled by default and rejects requests before any provider calls."""
    monkeypatch.setenv("LIVE_EVALUATION_ENABLED", "false")
    monkeypatch.setenv("GNANI_API_KEY", "test_mock_secret_key_123")
    get_settings.cache_clear()
    settings = get_settings()
    assert settings.live_evaluation_enabled is False

    with patch("app.services.gnani_client.GnaniPrismaClient.transcribe") as mock_transcribe:
        # 1. Authorize endpoint rejected with 403
        resp_auth = await async_client.post("/api/eval/authorize-live", json={"test_id": "pair_qty_001"})
        assert resp_auth.status_code == 403
        assert "disabled by default" in resp_auth.json()["detail"].lower()

        # 2. Run-live endpoint rejected with 403 before file parsing or provider calls
        files = {
            "audio_file_a": ("clip_a.wav", io.BytesIO(valid_wav_bytes), "audio/wav"),
            "audio_file_b": ("clip_b.wav", io.BytesIO(valid_wav_bytes), "audio/wav"),
        }
        resp_run = await async_client.post("/api/eval/run-live/pair_qty_001", files=files)
        assert resp_run.status_code == 403
        assert "disabled by default" in resp_run.json()["detail"].lower()
        mock_transcribe.assert_not_called()


@pytest.mark.asyncio
async def test_disabled_live_execution_makes_zero_provider_calls(async_client, valid_wav_bytes, monkeypatch):
    """Proves disabled live mode makes zero provider calls and consumes 0 credits."""
    monkeypatch.setenv("LIVE_EVALUATION_ENABLED", "false")
    monkeypatch.setenv("GNANI_API_KEY", "test_mock_secret_key_123")
    get_settings.cache_clear()

    files = {
        "audio_file_a": ("clip_a.wav", io.BytesIO(valid_wav_bytes), "audio/wav"),
        "audio_file_b": ("clip_b.wav", io.BytesIO(valid_wav_bytes), "audio/wav"),
    }
    with patch("app.services.gnani_client.GnaniPrismaClient.transcribe") as mock_transcribe:
        resp = await async_client.post("/api/eval/run-live/pair_qty_001", files=files)
        assert resp.status_code == 403
        mock_transcribe.assert_not_called()


@pytest.mark.asyncio
async def test_live_audio_condition_validation(async_client, valid_wav_bytes, monkeypatch):
    """Proves audio_condition is strictly validated against AudioCondition enum, rejecting unknown and simulation labels."""
    monkeypatch.setenv("LIVE_EVALUATION_ENABLED", "true")
    monkeypatch.setenv("GNANI_API_KEY", "test_mock_secret_key_123")
    get_settings.cache_clear()

    # 1. Reject unknown audio_condition
    headers, data = await _get_live_auth(async_client, "pair_qty_001")
    data["audio_condition"] = "fabricated_acoustic_label"
    files = {
        "audio_file_a": ("clip_a.wav", io.BytesIO(valid_wav_bytes), "audio/wav"),
        "audio_file_b": ("clip_b.wav", io.BytesIO(valid_wav_bytes), "audio/wav"),
    }
    with patch("app.services.gnani_client.GnaniPrismaClient.transcribe") as mock_transcribe:
        resp = await async_client.post("/api/eval/run-live/pair_qty_001", files=files, data=data, headers=headers)
        assert resp.status_code == 400
        assert "invalid audio_condition" in resp.json()["detail"].lower()
        mock_transcribe.assert_not_called()

    # 2. Reject text_only_simulation on live audio route (simulation is not recorded audio)
    headers, data = await _get_live_auth(async_client, "pair_qty_001")
    data["audio_condition"] = "text_only_simulation"
    files = {
        "audio_file_a": ("clip_a.wav", io.BytesIO(valid_wav_bytes), "audio/wav"),
        "audio_file_b": ("clip_b.wav", io.BytesIO(valid_wav_bytes), "audio/wav"),
    }
    with patch("app.services.gnani_client.GnaniPrismaClient.transcribe") as mock_transcribe:
        resp = await async_client.post("/api/eval/run-live/pair_qty_001", files=files, data=data, headers=headers)
        assert resp.status_code == 400
        assert "cannot be used for live audio evaluation" in resp.json()["detail"].lower()
        mock_transcribe.assert_not_called()

    # 3. Accept valid condition (e.g. noisy_recorded) and verify persisted metadata
    headers, data = await _get_live_auth(async_client, "pair_qty_001")
    data["audio_condition"] = "noisy_recorded"
    files = {
        "audio_file_a": ("clip_a.wav", io.BytesIO(valid_wav_bytes), "audio/wav"),
        "audio_file_b": ("clip_b.wav", io.BytesIO(valid_wav_bytes), "audio/wav"),
    }
    mock_resp = {
        "success": True,
        "transcript": "Do packet chahiye",
        "request_id": "req_cond_test",
        "model": "gnani-prisma-v2.5",
        "client_request_elapsed_ms": 115.0,
        "server_timestamp": "20261020_040000.000",
        "language_code": "hi-IN",
        "format": "verbatim",
        "is_mocked": False,
    }
    with patch("app.services.gnani_client.GnaniPrismaClient.transcribe", return_value=mock_resp):
        resp = await async_client.post("/api/eval/run-live/pair_qty_001", files=files, data=data, headers=headers)
        assert resp.status_code == 200
        run_data = resp.json()
        assert run_data["audio_condition"] == "noisy_recorded"

        # Verify correctly persisted in SQLite storage
        run_id = run_data["run_id"]
        resp_get = await async_client.get(f"/api/eval/results/{run_id}")
        assert resp_get.status_code == 200
        assert resp_get.json()["audio_condition"] == "noisy_recorded"


@pytest.mark.asyncio
async def test_run_live_missing_origin_non_local_rejected(async_client, monkeypatch):
    """Proves missing Origin/Referer header on a non-local host is rejected with 403."""
    monkeypatch.setenv("LIVE_EVALUATION_ENABLED", "true")
    monkeypatch.setenv("GNANI_API_KEY", "test_mock_secret_key_123")
    get_settings.cache_clear()

    # Test live_guard.validate_same_origin directly with a mock request targeting a public host without Origin/Referer
    from starlette.requests import Request
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/eval/run-live/pair_qty_001",
        "headers": [
            (b"host", b"demo.bolibench.org"),
        ],
        "server": ("demo.bolibench.org", 443),
        "scheme": "https",
    }
    req = Request(scope)
    with pytest.raises(HTTPException) as exc_info:
        live_guard.validate_same_origin(req)
    assert exc_info.value.status_code == 403
    assert "missing origin or referer header" in exc_info.value.detail.lower()


def test_openapi_live_route_documentation_reflects_wav_only():
    """Asserts that OpenAPI schema for live paired evaluation documents WAV-only format."""
    from app.main import app
    openapi_schema = app.openapi()
    live_path = openapi_schema["paths"].get("/api/eval/run-live/{test_id}", {})
    post_op = live_path.get("post", {})
    assert post_op, "POST /api/eval/run-live/{test_id} not found in OpenAPI schema"

    # 1. Check description explains Audio Validation Policy
    doc = post_op.get("description", "")
    assert "WAV format only" in doc
    assert "<= 60 seconds" in doc or "<= 60s" in doc
    assert "General single-audio upload" in doc

    # 2. Check requestBody schema descriptions for audio files
    req_body = post_op.get("requestBody", {})
    form_schema = req_body.get("content", {}).get("multipart/form-data", {}).get("schema", {})
    ref = form_schema.get("$ref")
    if ref and ref.startswith("#/components/schemas/"):
        schema_name = ref.split("/")[-1]
        schema = openapi_schema["components"]["schemas"].get(schema_name, {})
    else:
        schema = form_schema

    props = schema.get("properties", {})
    desc_a = props.get("audio_file_a", {}).get("description", "")
    desc_b = props.get("audio_file_b", {}).get("description", "")

    assert "WAV format only" in desc_a
    assert "<= 60s" in desc_a or "<= 60 seconds" in desc_a
    assert "WAV format only" in desc_b
    assert "<= 60s" in desc_b or "<= 60 seconds" in desc_b

    # Verify stale format list (MP3, OGG, etc.) is removed from live upload description
    assert "MP3" not in desc_a
    assert "MP3" not in desc_b


@pytest.mark.asyncio
async def test_failed_and_incomplete_runs_remain_accessible_in_history(async_client, tmp_path):
    """Proves failed and incomplete operational runs remain discoverable in history with overall_status."""
    from app.evaluation.storage import StorageManager
    from app.evaluation.schemas import PairEvaluationResult, UtteranceEvaluationResult, ExecutionMode, FailureCategory
    from app.routers.evaluation import engine

    test_storage = StorageManager(db_path=tmp_path / "test_history_status.db")
    orig_storage = engine.storage
    engine.storage = test_storage

    try:
        # 1. Insert 1 completed, 1 failed, 1 incomplete run
        completed_run = PairEvaluationResult(
            run_id="hist_comp_001",
            test_case_id="pair_qty_001",
            category="quantities_and_numbers",
            language_code="hi-IN",
            execution_mode=ExecutionMode.MOCK,
            audio_condition="text_only_simulation",
            timestamp="2026-10-20T04:10:00Z",
            overall_status="completed",
            utterance_a=UtteranceEvaluationResult(
                utterance_id="A",
                reference_transcript="Do packet chahiye",
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
            ),
            utterance_b=UtteranceEvaluationResult(
                utterance_id="B",
                reference_transcript="Teen packet chahiye",
                actual_transcript="Teen packet chahiye",
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
            ),
            wer_a=0.0,
            wer_b=0.0,
            average_wer=0.0,
            critical_field_accuracy=1.0,
            semantic_distinction_preserved=True,
            task_success=True,
            primary_failure_category=FailureCategory.NONE,
            failure_explanation="Success",
        )
        failed_run = PairEvaluationResult(
            run_id="hist_failed_002",
            test_case_id="pair_qty_001",
            category="quantities_and_numbers",
            language_code="hi-IN",
            execution_mode=ExecutionMode.LIVE,
            audio_condition="clean_recorded",
            timestamp="2026-10-20T04:11:00Z",
            overall_status="failed",
            utterance_a=UtteranceEvaluationResult(
                utterance_id="A",
                reference_transcript="Do packet chahiye",
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
                error_message="Network connection reset",
            ),
            utterance_b=UtteranceEvaluationResult(
                utterance_id="B",
                reference_transcript="Teen packet chahiye",
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
            failure_explanation="Provider request for Utterance A failed: Network connection reset.",
        )
        incomplete_run = PairEvaluationResult(
            run_id="hist_incomp_003",
            test_case_id="pair_qty_001",
            category="quantities_and_numbers",
            language_code="hi-IN",
            execution_mode=ExecutionMode.LIVE,
            audio_condition="clean_recorded",
            timestamp="2026-10-20T04:12:00Z",
            overall_status="incomplete",
            utterance_a=UtteranceEvaluationResult(
                utterance_id="A",
                reference_transcript="Do packet chahiye",
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
                request_id="req_success_111",
            ),
            utterance_b=UtteranceEvaluationResult(
                utterance_id="B",
                reference_transcript="Teen packet chahiye",
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
                error_message="Rate limit 429",
            ),
            wer_a=0.0,
            wer_b=1.0,
            average_wer=0.5,
            critical_field_accuracy=0.5,
            semantic_distinction_preserved=False,
            task_success=False,
            primary_failure_category=FailureCategory.PROVIDER_ERROR,
            failure_explanation="Partial failure: Utterance A succeeded, Utterance B rate limited.",
        )
        test_storage.save_run(completed_run)
        test_storage.save_run(failed_run)
        test_storage.save_run(incomplete_run)

        # 2. Query all results via history endpoint
        resp = await async_client.get("/api/eval/results")
        assert resp.status_code == 200
        runs = resp.json()
        assert len(runs) == 3

        statuses = {r["run_id"]: r["overall_status"] for r in runs}
        assert statuses["hist_comp_001"] == "completed"
        assert statuses["hist_failed_002"] == "failed"
        assert statuses["hist_incomp_003"] == "incomplete"

        # 3. Test filtering by status
        resp_failed = await async_client.get("/api/eval/results?status=failed")
        assert resp_failed.status_code == 200
        failed_list = resp_failed.json()
        assert len(failed_list) == 1
        assert failed_list[0]["run_id"] == "hist_failed_002"
        assert failed_list[0]["overall_status"] == "failed"

        resp_incomp = await async_client.get("/api/eval/results?status=incomplete")
        assert resp_incomp.status_code == 200
        incomp_list = resp_incomp.json()
        assert len(incomp_list) == 1
        assert incomp_list[0]["run_id"] == "hist_incomp_003"
        assert incomp_list[0]["overall_status"] == "incomplete"

        # 4. Single run retrieval exposes overall_status and error details
        resp_single_failed = await async_client.get("/api/eval/results/hist_failed_002")
        assert resp_single_failed.status_code == 200
        s_failed = resp_single_failed.json()
        assert s_failed["overall_status"] == "failed"
        assert s_failed["primary_failure_category"] == "provider_error"
        assert "Network connection reset" in s_failed["failure_explanation"]
        assert s_failed["utterance_a"]["status"] == "failed"
        assert s_failed["utterance_b"]["status"] == "skipped"

        resp_single_incomp = await async_client.get("/api/eval/results/hist_incomp_003")
        assert resp_single_incomp.status_code == 200
        s_incomp = resp_single_incomp.json()
        assert s_incomp["overall_status"] == "incomplete"
        assert s_incomp["utterance_a"]["request_id"] == "req_success_111"
        assert s_incomp["utterance_b"]["status"] == "failed"

    finally:
        engine.storage = orig_storage


# -----------------------------------------------------------------------------
# Phase 6.9: SQLite Storage Isolation and Regression Proofs
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_mocked_live_run_persists_only_in_temporary_storage(
    async_client, valid_wav_bytes, monkeypatch
):
    """Proves a mocked successful /api/eval/run-live/{test_id} persists synthetic records
    only in temporary test storage, and does not touch the production SQLite database.
    """
    import sqlite3
    from pathlib import Path
    from app.routers.evaluation import engine

    prod_db_path = Path(__file__).resolve().parent.parent / "bolibench_evaluation.db"
    if not prod_db_path.exists():
        pytest.skip("Production database does not exist locally.")

    # Record row count of production database prior to test
    with sqlite3.connect(str(prod_db_path)) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM evaluation_runs")
        prod_count_before = cursor.fetchone()[0]

    monkeypatch.setenv("GNANI_API_KEY", "test_mock_secret_key_isolation")
    monkeypatch.setenv("LIVE_EVALUATION_ENABLED", "true")
    get_settings.cache_clear()

    mock_resp_a = {
        "success": True,
        "transcript": "Do packet chahiye",
        "request_id": "req_iso_proof_a_999",
        "model": "gnani-prisma-v2.5",
        "client_request_elapsed_ms": 100.0,
        "server_timestamp": "20261020_040000.000",
        "language_code": "hi-IN",
        "format": "transcribe",
        "is_mocked": False,
    }
    mock_resp_b = {
        "success": True,
        "transcript": "Teen packet chahiye",
        "request_id": "req_iso_proof_b_999",
        "model": "gnani-prisma-v2.5",
        "client_request_elapsed_ms": 105.0,
        "server_timestamp": "20261020_040001.000",
        "language_code": "hi-IN",
        "format": "transcribe",
        "is_mocked": False,
    }

    headers, data = await _get_live_auth(async_client, "pair_qty_001")
    data["audio_condition"] = "clean_recorded"
    files = {
        "audio_file_a": ("utterance_a.wav", io.BytesIO(valid_wav_bytes), "audio/wav"),
        "audio_file_b": ("utterance_b.wav", io.BytesIO(valid_wav_bytes), "audio/wav"),
    }

    with patch("app.services.gnani_client.GnaniPrismaClient.transcribe", side_effect=[mock_resp_a, mock_resp_b]):
        response = await async_client.post("/api/eval/run-live/pair_qty_001", files=files, data=data, headers=headers)
        assert response.status_code == 200
        res_data = response.json()
        live_run_id = res_data["run_id"]

        # 1. Assert record exists in isolated test storage
        test_run = engine.get_run(live_run_id)
        assert test_run is not None
        assert test_run.run_id == live_run_id
        assert test_run.utterance_a.request_id == "req_iso_proof_a_999"
        assert test_run.utterance_b.request_id == "req_iso_proof_b_999"

        # 2. Assert record is NOT present in production database
        with sqlite3.connect(str(prod_db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM evaluation_runs WHERE run_id = ?", (live_run_id,))
            match_count = cursor.fetchone()[0]
            assert match_count == 0, f"Synthetic test run {live_run_id} leaked into production database!"

            cursor.execute("SELECT COUNT(*) FROM evaluation_runs")
            prod_count_after = cursor.fetchone()[0]
            assert prod_count_after == prod_count_before, "Production database row count changed during test!"


@pytest.mark.asyncio
async def test_production_database_untouched_during_eval_test(async_client):
    """Proves the production database is neither selected nor modified during eval runs."""
    import sqlite3
    from pathlib import Path
    from app.routers.evaluation import engine

    prod_db_path = Path(__file__).resolve().parent.parent / "bolibench_evaluation.db"
    if not prod_db_path.exists():
        pytest.skip("Production database does not exist locally.")

    # Check that engine storage points to a temporary location, not prod
    assert Path(engine.storage.db_path).resolve() != prod_db_path.resolve()
    assert "tmp" in str(engine.storage.db_path).lower() or "temp" in str(engine.storage.db_path).lower() or "pytest" in str(engine.storage.db_path).lower()

    with sqlite3.connect(str(prod_db_path)) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM evaluation_runs")
        count_before = cursor.fetchone()[0]

    # Run mock evaluation
    resp = await async_client.post("/api/eval/run-mock/pair_qty_001?scenario=ideal_preserved")
    assert resp.status_code == 200
    mock_run_id = resp.json()["run_id"]

    # Verify mock run is in engine storage
    assert engine.get_run(mock_run_id) is not None

    # Verify production database was untouched
    with sqlite3.connect(str(prod_db_path)) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM evaluation_runs")
        count_after = cursor.fetchone()[0]
        assert count_after == count_before

        cursor.execute("SELECT COUNT(*) FROM evaluation_runs WHERE run_id = ?", (mock_run_id,))
        assert cursor.fetchone()[0] == 0


@pytest.mark.asyncio
async def test_history_isolation_between_tests_step1(async_client):
    """Proves that a test creates runs only in its own isolated test storage."""
    from app.routers.evaluation import engine

    # Storage should start clean for this test
    runs_before = engine.list_runs()
    assert len(runs_before) == 0

    resp = await async_client.post("/api/eval/run-mock/pair_qty_001?scenario=ideal_preserved")
    assert resp.status_code == 200
    run_id = resp.json()["run_id"]

    runs_after = engine.list_runs()
    assert len(runs_after) == 1
    assert runs_after[0].run_id == run_id


@pytest.mark.asyncio
async def test_history_isolation_between_tests_step2(async_client):
    """Proves that a subsequent test does not see runs created by previous tests."""
    from app.routers.evaluation import engine

    # Even though step1 created a run, this test receives a fresh isolated storage
    runs = engine.list_runs()
    assert len(runs) == 0


@pytest.mark.asyncio
async def test_existing_tests_overriding_engine_storage_still_behave_correctly(tmp_path):
    """Proves that tests which manually swap engine.storage can do so and restore cleanly."""
    from app.evaluation.storage import StorageManager
    from app.routers.evaluation import engine

    fixture_storage = engine.storage
    fixture_db_path = fixture_storage.db_path

    custom_storage = StorageManager(db_path=tmp_path / "custom_inner_override.db")
    orig_storage = engine.storage
    engine.storage = custom_storage

    try:
        assert engine.storage == custom_storage
        assert engine.storage.db_path == str(tmp_path / "custom_inner_override.db")
    finally:
        engine.storage = orig_storage

    assert engine.storage == fixture_storage
    assert engine.storage.db_path == fixture_db_path


@pytest.mark.asyncio
async def test_network_blocking_fixture_prevents_unmocked_outbound_httpx():
    """Proves the block_outbound_network_calls fixture intercepts unmocked outbound HTTPX calls."""
    import httpx

    with pytest.raises(RuntimeError, match="Outbound network request blocked in test suite"):
        async with httpx.AsyncClient() as client:
            await client.get("https://api.vachana.ai/stt/v3/health")
