"""API route handlers for Paired-Contrast Evaluation Engine."""

from typing import List, Optional
from fastapi import APIRouter, File, Form, HTTPException, Query, Request, Response, UploadFile, status
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from app.config import get_settings
from app.evaluation.engine import EvaluationEngine
from app.evaluation.schemas import (
    AggregateMetrics,
    AudioCondition,
    EvaluateTranscriptsRequest,
    PairEvaluationResult,
    TestCasePair,
)
from app.services.audio_validator import AudioValidationError, AudioValidator
from app.services.gnani_client import GnaniClientError, GnaniPrismaClient, MissingCredentialsError
from app.services.live_guard import live_guard

router = APIRouter(prefix="/api/eval", tags=["evaluation"])
engine = EvaluationEngine()


class AuthorizeLiveRequest(BaseModel):
    test_id: str


@router.post("/authorize-live")
async def authorize_live_evaluation(
    req: AuthorizeLiveRequest,
    request: Request,
):
    """Authorizes an upcoming live evaluation run by issuing a single-use authorization token.

    Security & Credit Guard:
    - Rejects request if live evaluation is disabled (LIVE_EVALUATION_ENABLED=false).
    - Validates same-origin headers (Sec-Fetch-Site, Origin, Referer).
    - Requires target test case existence.
    - Issues a cryptographically random, single-use, 5-minute token.
    - NEVER makes any external network calls or consumes credits.
    - NEVER logs the issued token secret.
    """
    settings = get_settings()
    if not settings.live_evaluation_enabled:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Forbidden: Live evaluation is disabled by default to protect programme credits. Set LIVE_EVALUATION_ENABLED=true in .env for authorized local operator testing.",
        )

    test_case = engine.get_test_case(req.test_id)
    if not test_case:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Test case with ID '{req.test_id}' not found.",
        )

    token = live_guard.issue_authorization_token(test_id=req.test_id, request=request)
    return {
        "status": "authorized",
        "test_case_id": req.test_id,
        "confirmation_token": token,
        "expires_in_seconds": int(live_guard.TOKEN_TTL_SECONDS),
        "warning": "Authorizing live evaluation. This operation will invoke 2 real requests to Gnani Prisma STT v3 and consume programme credits.",
    }


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


@router.post("/run-live/{test_id}", response_model=PairEvaluationResult)
async def run_live_evaluation(
    test_id: str,
    request: Request,
    audio_file_a: UploadFile = File(..., description="Audio file for Utterance A (WAV, MP3, OGG, FLAC, AAC, M4A <= 60s)"),
    audio_file_b: UploadFile = File(..., description="Audio file for Utterance B (WAV, MP3, OGG, FLAC, AAC, M4A <= 60s)"),
    audio_condition: str = Form("clean_recorded", description="Acoustic environment metadata (e.g. clean_recorded, noisy_recorded)"),
    confirmation_token: Optional[str] = Form(None, description="Single-use confirmation authorization token from /api/eval/authorize-live"),
):
    """Executes a live paired-contrast evaluation using Gnani Prisma STT REST API.

    CREDIT PROTECTION & AUTHORIZATION GUARD:
    - Rejects request BEFORE reading uploads into memory or invoking provider if:
      - Live evaluation is disabled (LIVE_EVALUATION_ENABLED=false)
      - Origin is cross-site / cross-origin / untrusted
      - Anti-CSRF confirmation header X-BoliBench-Confirm-Live is missing or invalid
      - Single-use confirmation token is missing, invalid, expired, or already consumed
    - Validates audio_condition against AudioCondition enum (rejects text_only_simulation on live audio).
    - Requires GNANI_API_KEY to be configured in .env.
    """
    # 0. STRICT DEPLOYMENT SAFETY & CREDIT GUARD FIRST
    settings = get_settings()
    if not settings.live_evaluation_enabled:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Forbidden: Live evaluation is disabled by default to protect programme credits. Set LIVE_EVALUATION_ENABLED=true in .env for authorized local operator testing.",
        )

    # 1. Verify single-use token and anti-CSRF header
    live_guard.verify_authorization(
        test_id=test_id,
        confirmation_token=confirmation_token,
        request=request,
    )

    if not settings.is_api_configured:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Gnani API key is not configured. Set GNANI_API_KEY in .env to perform live evaluations.",
        )

    # 2. Validate acoustic condition metadata against AudioCondition enum
    try:
        cond_enum = AudioCondition(audio_condition)
    except ValueError:
        valid_conditions = [c.value for c in AudioCondition]
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid audio_condition '{audio_condition}'. Must be one of: {valid_conditions}",
        )

    if cond_enum == AudioCondition.TEXT_ONLY_SIMULATION:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Condition 'text_only_simulation' cannot be used for live audio evaluation. Please select an acoustic recording condition (e.g. 'clean_recorded', 'noisy_recorded', 'clean_synthetic', 'noisy_synthetic', 'pending_acoustic').",
        )

    test_case = engine.get_test_case(test_id)
    if not test_case:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Test case with ID '{test_id}' not found.",
        )

    validator = AudioValidator(max_size_bytes=settings.max_audio_size_bytes)

    # 3. Read & Validate Audio File A in memory
    try:
        content_a = await audio_file_a.read()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read audio file A: {str(exc)}",
        )
    ext_a, content_type_a = validator.validate(audio_file_a.filename, content_a)

    # 4. Read & Validate Audio File B in memory
    try:
        content_b = await audio_file_b.read()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read audio file B: {str(exc)}",
        )
    ext_b, content_type_b = validator.validate(audio_file_b.filename, content_b)

    # 5. Server-side Gnani client invocation
    client = GnaniPrismaClient(settings=settings)

    try:
        result = await engine.evaluate_live_audio(
            test_case_id=test_id,
            audio_a_bytes=content_a,
            filename_a=audio_file_a.filename or f"utterance_a{ext_a}",
            content_type_a=content_type_a,
            audio_b_bytes=content_b,
            filename_b=audio_file_b.filename or f"utterance_b{ext_b}",
            content_type_b=content_type_b,
            client=client,
            audio_condition=cond_enum.value,
        )
        return result
    except GnaniClientError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail=f"Gnani Prisma API error during paired evaluation: {exc.message}",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Live paired audio evaluation failed: {str(exc)}",
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
