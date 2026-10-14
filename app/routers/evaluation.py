"""API route handlers for Paired-Contrast Evaluation Engine."""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query, Response, status
from fastapi.responses import PlainTextResponse

from app.evaluation.engine import EvaluationEngine
from app.evaluation.schemas import (
    AggregateMetrics,
    EvaluateTranscriptsRequest,
    PairEvaluationResult,
    TestCasePair,
)

router = APIRouter(prefix="/api/eval", tags=["evaluation"])
engine = EvaluationEngine()


@router.get("/test-cases", response_model=List[TestCasePair])
async def list_test_cases(category: Optional[str] = Query(None, description="Optional category filter")):
    """Lists available seed paired-contrast test cases."""
    return engine.list_test_cases(category=category)


@router.get("/test-cases/{test_id}", response_model=TestCasePair)
async def get_test_case(test_id: str):
    """Retrieves metadata and specifications for a single test case pair."""
    test_case = engine.get_test_case(test_id)
    if not test_case:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Test case with ID '{test_id}' not found.",
        )
    return test_case


@router.post("/run-mock/{test_id}", response_model=PairEvaluationResult)
async def run_mock_evaluation(
    test_id: str,
    scenario: str = Query("ideal_preserved", description="'ideal_preserved' or 'crashed_distinction'"),
):
    """Executes a deterministic mock evaluation for an existing test case pair.

    Clearly labeled as execution_mode='mock'. Consumes 0 Gnani credits.
    """
    try:
        return engine.run_mock_evaluation(test_case_id=test_id, scenario=scenario)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        )


@router.post("/evaluate-transcripts", response_model=PairEvaluationResult)
async def evaluate_transcripts(req: EvaluateTranscriptsRequest):
    """Evaluates arbitrary spoken transcripts against reference expectations."""
    try:
        return engine.evaluate_transcripts(req)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        )


@router.get("/results", response_model=List[PairEvaluationResult])
async def list_evaluation_results(
    limit: int = Query(50, ge=1, le=500),
    category: Optional[str] = Query(None),
):
    """Lists recent evaluation runs stored in SQLite."""
    return engine.list_runs(limit=limit, category=category)


@router.get("/results/{run_id}", response_model=PairEvaluationResult)
async def get_evaluation_result(run_id: str):
    """Retrieves a single evaluation run by ID."""
    run = engine.get_run(run_id)
    if not run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Evaluation run with ID '{run_id}' not found.",
        )
    return run


@router.get("/metrics", response_model=AggregateMetrics)
async def get_aggregate_metrics():
    """Computes aggregate accuracy, pass rates, WER, and failure category breakdown."""
    return engine.get_aggregate_metrics()


@router.get("/export")
async def export_evaluation_results(
    format: str = Query("json", description="'json' or 'csv'"),
):
    """Exports evaluation runs as CSV or JSON file."""
    if format.lower() == "csv":
        csv_data = engine.export_csv()
        return Response(
            content=csv_data,
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=bolibench_evaluation_results.csv"},
        )
    else:
        json_data = engine.export_json()
        return Response(
            content=json_data,
            media_type="application/json",
            headers={"Content-Disposition": "attachment; filename=bolibench_evaluation_results.json"},
        )
