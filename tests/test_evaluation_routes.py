"""Route tests for BoliBench Paired-Contrast Evaluation API endpoints."""

import pytest


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
