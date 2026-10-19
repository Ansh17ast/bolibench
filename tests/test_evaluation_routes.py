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
    """Test /api/eval/metrics returns aggregate evaluation metrics."""
    response = await async_client.get("/api/eval/metrics")
    assert response.status_code == 200

    data = response.json()
    assert "total_runs" in data
    assert "distinction_preservation_rate" in data
    assert "task_success_rate" in data
    assert "average_wer" in data
    assert "category_breakdown" in data


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
    monkeypatch.delenv("GNANI_API_KEY", raising=False)
    monkeypatch.setenv("LIVE_EVALUATION_ENABLED", "true")
    get_settings.cache_clear()

    headers, data = await _get_live_auth(async_client, "pair_qty_001")
    files = {
        "audio_file_a": ("clip_a.wav", io.BytesIO(valid_wav_bytes), "audio/wav"),
        "audio_file_b": ("clip_b.wav", io.BytesIO(valid_wav_bytes), "audio/wav"),
    }
    resp = await async_client.post("/api/eval/run-live/pair_qty_001", files=files, data=data, headers=headers)
    assert resp.status_code == 401
    assert "not configured" in resp.json()["detail"].lower()


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
        "format": "verbatim",
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
        "format": "verbatim",
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

        result = response.json()
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
    """Proves failure of Utterance A never evaluates Utterance B or records a run."""
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


@pytest.mark.asyncio
async def test_run_live_partial_failure_second_fails(async_client, valid_wav_bytes, monkeypatch):
    """Proves if Utterance A succeeds but Utterance B fails, no partial success is recorded."""
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
        "format": "verbatim",
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
    monkeypatch.delenv("LIVE_EVALUATION_ENABLED", raising=False)
    get_settings.cache_clear()
    settings = get_settings()
    assert settings.live_evaluation_enabled is False

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


@pytest.mark.asyncio
async def test_disabled_live_execution_makes_zero_provider_calls(async_client, valid_wav_bytes, monkeypatch):
    """Proves disabled live mode makes zero provider calls and consumes 0 credits."""
    monkeypatch.delenv("LIVE_EVALUATION_ENABLED", raising=False)
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
