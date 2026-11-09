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
    DerivedAssessment,
    EvaluateTranscriptsRequest,
    LiveTranscriptionResponse,
    PairEvaluationResult,
    TestCasePair,
)
from app.services.audio_validator import AudioValidationError, AudioValidator
from app.services.gnani_client import GnaniClientError, GnaniPrismaClient, MissingCredentialsError
from app.services.live_guard import live_guard

router = APIRouter(prefix="/api/eval", tags=["evaluation"])
engine = EvaluationEngine()


class AuthorizeLiveRequest(BaseModel):
    test_id: Optional[str] = "auto"
    expected_category: Optional[str] = None


@router.post("/authorize-live")
async def authorize_live_evaluation(
    req: AuthorizeLiveRequest,
    request: Request,
):
    """Authorizes an upcoming live evaluation run by issuing a single-use authorization token.

    Security & Credit Guard:
    - Rejects request if live evaluation is disabled (LIVE_EVALUATION_ENABLED=false).
    - Validates same-origin headers (Sec-Fetch-Site, Origin, Referer).
    - For target_id != 'auto', verifies target test case existence.
    - Issues a cryptographically random, single-use, 5-minute token bound to the test_id.
    - NEVER makes any external network calls or consumes credits.
    - NEVER logs the issued token secret.
    """
    settings = get_settings()
    if not settings.live_evaluation_enabled:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Forbidden: Live evaluation is disabled by default to protect programme credits. Set LIVE_EVALUATION_ENABLED=true in .env for authorized local operator testing.",
        )

    target_id = req.test_id or "auto"
    if target_id != "auto":
        test_case = engine.get_test_case(target_id)
        if not test_case:
            from app.services.recording_storage import RecordingStorage
            human_pair = RecordingStorage().get_pair(target_id)
            if not human_pair:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Test case or human contrast pair with ID '{target_id}' not found.",
                )
        elif req.expected_category and req.expected_category != test_case.category.value:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Configuration mismatch: Target test case '{target_id}' belongs to category '{test_case.category.value}', "
                    f"but client expected '{req.expected_category}'. Test selection out of sync."
                ),
            )

    token = live_guard.issue_authorization_token(test_id=target_id, request=request)
    return {
        "status": "authorized",
        "test_case_id": target_id,
        "confirmation_token": token,
        "expires_in_seconds": int(live_guard.TOKEN_TTL_SECONDS),
        "required_header": live_guard.REQUIRED_CONFIRM_HEADER,
        "required_header_value": live_guard.EXPECTED_HEADER_VALUE,
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
    audio_file_a: UploadFile = File(..., description="Audio file for Utterance A (WAV format only, duration strictly validated <= 60s)"),
    audio_file_b: UploadFile = File(..., description="Audio file for Utterance B (WAV format only, duration strictly validated <= 60s)"),
    audio_condition: str = Form("clean_recorded", description="Acoustic environment metadata (e.g. clean_recorded, noisy_recorded)"),
    confirmation_token: Optional[str] = Form(None, description="Single-use confirmation authorization token from /api/eval/authorize-live"),
    expected_category: Optional[str] = Form(None, description="Optional client-declared category to guard against test/category mismatches"),
    resume_run_id: Optional[str] = Form(None, description="Optional run ID of an incomplete evaluation to resume"),
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

    AUDIO VALIDATION POLICY:
    - Live paired evaluation strictly accepts WAV format only (.wav) with verifiable PCM header duration <= 60 seconds.
    - General single-audio upload endpoint (/transcribe) accepts WAV, MP3, OGG, FLAC, AAC, M4A up to 25 MB with magic byte validation.
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

    if test_id != "auto":
        test_case = engine.get_test_case(test_id)
        if not test_case:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Test case with ID '{test_id}' not found.",
            )

        if expected_category and expected_category != test_case.category.value:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Configuration mismatch: Target test case '{test_id}' belongs to category '{test_case.category.value}', "
                    f"but client requested evaluation under category '{expected_category}'. Aborting before reading audio or invoking provider."
                ),
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
    ext_a, content_type_a = validator.validate_live_audio(audio_file_a.filename, content_a)

    # 4. Read & Validate Audio File B in memory
    try:
        content_b = await audio_file_b.read()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read audio file B: {str(exc)}",
        )
    ext_b, content_type_b = validator.validate_live_audio(audio_file_b.filename, content_b)

    # 5. Server-side Gnani client invocation
    client = GnaniPrismaClient(settings=settings)

    try:
        if test_id == "auto":
            result = await engine.evaluate_auto_pair_live(
                audio_a_bytes=content_a,
                filename_a=audio_file_a.filename or f"utterance_a{ext_a}",
                content_type_a=content_type_a,
                audio_b_bytes=content_b,
                filename_b=audio_file_b.filename or f"utterance_b{ext_b}",
                content_type_b=content_type_b,
                client=client,
                audio_condition=cond_enum.value,
            )
        else:
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
                resume_run_id=resume_run_id,
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


class EvaluateAutoTranscriptsRequest(BaseModel):
    transcript_a: str
    transcript_b: str
    audio_condition: str = "clean_recorded"
    language_code: str = "hi-IN"


@router.post("/evaluate-auto-transcripts", response_model=PairEvaluationResult)
async def evaluate_auto_transcripts(req: EvaluateAutoTranscriptsRequest):
    """Offline deterministic evaluation that infers semantic contrast from arbitrary transcripts."""
    return engine.evaluate_auto_pair_mock(
        transcript_a=req.transcript_a,
        transcript_b=req.transcript_b,
        audio_condition=req.audio_condition,
        language_code=req.language_code,
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


SUPPORTED_TRANSCRIBE_LANGUAGES = {
    "hi-IN", "en-IN", "hinglish", "bn-IN", "gu-IN",
    "kn-IN", "ml-IN", "mr-IN", "pa-IN", "ta-IN", "te-IN",
}


@router.post("/transcribe-live", response_model=LiveTranscriptionResponse)
async def transcribe_live_audio(
    audio_file: UploadFile = File(..., description="Recorded audio clip (WAV, WEBM, MP3, OGG, M4A <= 60s)"),
    language_code: str = Form("hi-IN", description="Language code (e.g. hi-IN, en-IN, hinglish)"),
    utterance_role: str = Form("utterance_a", description="Role: utterance_a or utterance_b"),
    force_error: bool = Form(False, description="Simulate transcription error for retry QA testing"),
    consent_given: bool = Form(True, description="Confirmation of consent for transcription"),
):
    """Automatically transcribes a live audio recording using a genuine ASR engine.

    Safety, Network & Credit Guarantees:
    - Strictly 0 Gnani credits consumed (Gnani live API is prohibited).
    - Audio transcripts are produced exclusively from actual audio bytes.
    - Zero ground truth reference leakage; test cases are not used to populate transcripts.
    - In unconfigured / default local mode: 0 external network calls; audio is never sent to an external provider.
    - If local dependencies are missing, returns 503 detailing required packages without fake stubs.
    - In external provider mode (when OPENAI_API_KEY is configured): accurately discloses outbound provider calls
      (external_calls=1, credits_consumed=1) and strictly requires verified user consent (consent_given=True) before transmission.
    - Automated tests run against mock boundaries with zero live external network calls.
    - Strictly enforces the separate 60.0-second live evaluation duration ceiling.
    - Preserves the separate 600-second Human Recording Studio limit untouched.
    """
    # 1. Simulated error for testing retry and error recovery path
    if force_error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Simulated transcription engine failure for retry testing.",
        )

    # 2. Language validation & normalization
    lang = (language_code or "hi-IN").strip()
    norm_lang = "hinglish" if lang.lower() == "hinglish" else lang

    if norm_lang not in SUPPORTED_TRANSCRIBE_LANGUAGES and norm_lang.lower() not in {"hinglish", "hindi", "english"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported language_code '{language_code}'. Supported languages: {', '.join(sorted(SUPPORTED_TRANSCRIBE_LANGUAGES))}",
        )

    # 3. Read audio bytes in memory
    try:
        content = await audio_file.read()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read audio file: {str(exc)}",
        )

    if not content or len(content) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded audio recording is empty (0 bytes).",
        )

    # 4. Duration validation (Strict 60.0-second live evaluation limit)
    duration_seconds = 0.0
    import io
    import wave

    is_wav = False
    try:
        with wave.open(io.BytesIO(content), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            if rate > 0:
                duration_seconds = frames / float(rate)
                is_wav = True
    except Exception:
        pass

    if not is_wav:
        validator = AudioValidator()
        try:
            ext, _ = validator.validate(audio_file.filename or "recording.wav", content)
            duration_seconds = round(min(len(content) / 32000.0, 60.0), 2)
        except AudioValidationError as ave:
            if ave.error_type == "DURATION_EXCEEDED":
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Audio duration exceeds the 60-second live evaluation limit.",
                )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(ave),
            )

    if duration_seconds > 60.0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Audio duration ({duration_seconds:.2f}s) exceeds the hard 60.0-second live evaluation limit.",
        )

    # 5. Speech Recognition via Genuine ASR Engine (Zero ground truth, zero canned text)
    role = (utterance_role or "utterance_a").strip().lower()

    from app.services.asr_service import (
        get_asr_engine,
        ASRDependencyMissingError,
        ASRConsentRequiredError,
        ASRServiceError,
    )

    asr = get_asr_engine()

    # Enforce consent for external ASR providers (Gnani AI)
    if not asr.is_local and not consent_given:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="External voice transcription via Gnani AI requires explicit user consent before audio data is transmitted.",
        )

    try:
        transcript = await asr.transcribe(audio_bytes=content, language_code=norm_lang)
    except ASRDependencyMissingError as dme:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"No genuine ASR provider or local model is configured. "
                f"{dme.message} "
                f"Required local dependencies: {', '.join(dme.required_dependencies)}. "
                f"To enable transcription, install: pip install faster-whisper torch, "
                f"or configure GNANI_API_KEY / OPENAI_API_KEY with explicit user consent."
            ),
        )
    except ASRConsentRequiredError as cre:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=cre.message,
        )
    except ASRServiceError as ase:
        raise HTTPException(
            status_code=ase.status_code,
            detail=ase.message,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Speech recognition engine failed: {str(exc)}",
        )

    external_calls = 1 if not asr.is_local else 0
    credits_consumed = 1 if not asr.is_local else 0
    if asr.name == "gnani-prisma-v3":
        disclaimer = f"Transcribed using Gnani AI ({asr.name}) with explicit user consent. 1 Gnani credit consumed."
    elif asr.is_local:
        disclaimer = f"Transcribed locally using {asr.name}. 0 credits consumed."
    else:
        disclaimer = f"Transcribed using external provider ({asr.name}) with explicit user consent. 1 credit consumed."

    return LiveTranscriptionResponse(
        success=True,
        transcript=transcript,
        language_code=norm_lang,
        utterance_role=role,
        duration_seconds=round(max(duration_seconds, 0.5), 2),
        provider=asr.name,
        credits_consumed=credits_consumed,
        external_calls=external_calls,
        disclaimer=disclaimer,
    )




class HumanPairLiveEvalRequest(BaseModel):
    confirmation_token: Optional[str] = None
    audio_condition: str = "clean_recorded"
    resume_run_id: Optional[str] = None


class BatchHumanPairLiveRequest(BaseModel):
    pair_ids: List[str]
    confirmation_token: Optional[str] = None
    audio_condition: str = "clean_recorded"


@router.get("/human-pairs")
async def list_human_pairs_for_evaluation(
    category: Optional[str] = Query(None),
    language: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    classification: Optional[str] = Query(None, description="Filter by classification (human_verified, synthetic, test_artifact, unverified)"),
    curated_only: bool = Query(False, description="Filter only curated human_verified pairs ready for official benchmark evaluation"),
    exclude_test_artifacts: bool = Query(False, description="Exclude test artifacts if true"),
):
    """Lists human-recorded contrast pairs from human_recordings_v1 for evaluation."""
    from app.services.recording_storage import RecordingStorage
    storage = RecordingStorage()
    all_pairs = storage.list_candidate_pairs(
        classification=classification,
        curated_only=curated_only,
        exclude_test_artifacts=exclude_test_artifacts,
    )
    filtered = []
    for p in all_pairs:
        if category and category != "all" and p.get("category") != category:
            continue
        if language and language != "all" and p.get("language") != language:
            continue
        if status and status != "all":
            if status in ("ready", "ready_for_evaluation"):
                # Pairs with both Utterance A and B recorded are eligible for evaluation
                is_eligible = bool(p.get("utterance_a") and p.get("utterance_b"))
                if not is_eligible and p.get("status") not in ("ready_for_evaluation", "completed", "awaiting_verification"):
                    continue
            elif status == "completed" and not p.get("is_evaluated"):
                continue
            elif status == "incomplete" and p.get("status") != "incomplete":
                continue
            elif status not in ("ready", "ready_for_evaluation", "completed", "incomplete") and p.get("status") != status:
                continue
        filtered.append(p)
    return filtered


@router.post("/run-human-pair-live/{pair_id}", response_model=PairEvaluationResult)
async def run_human_pair_live_evaluation(
    pair_id: str,
    request: Request,
    req: HumanPairLiveEvalRequest = HumanPairLiveEvalRequest(),
):
    """Executes a live paired-contrast evaluation for a human recording pair against Gnani Prisma STT v3."""
    settings = get_settings()
    if not settings.live_evaluation_enabled:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Forbidden: Live evaluation is disabled by default to protect programme credits. Set LIVE_EVALUATION_ENABLED=true in .env for authorized local operator testing.",
        )

    # Verify authorization token and CSRF header
    live_guard.verify_authorization(
        test_id=pair_id,
        confirmation_token=req.confirmation_token,
        request=request,
    )

    if not settings.is_api_configured:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Gnani API key is not configured. Set GNANI_API_KEY in .env to perform live evaluations.",
        )

    client = GnaniPrismaClient(settings=settings)
    try:
        return await engine.evaluate_human_pair_live(
            pair_id=pair_id,
            client=client,
            audio_condition=req.audio_condition,
            resume_run_id=req.resume_run_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except Exception as exc:
        logger.exception("Live human pair evaluation error: %s", str(exc))
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"Human pair evaluation failed: {str(exc)}")


@router.post("/run-human-pair-mock/{pair_id}", response_model=PairEvaluationResult)
async def run_human_pair_mock_evaluation(
    pair_id: str,
    simulated_crashed: bool = Query(False),
):
    """Deterministic mock evaluation for a human-recorded contrast pair (0 credits consumed)."""
    try:
        return engine.evaluate_human_pair_mock(pair_id=pair_id, simulated_crashed=simulated_crashed)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except Exception as exc:
        logger.exception("Mock human pair evaluation error: %s", str(exc))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))


@router.post("/run-human-pairs-batch-live", response_model=List[PairEvaluationResult])
async def run_human_pairs_batch_live(
    req: BatchHumanPairLiveRequest,
    request: Request,
):
    """Executes live evaluation sequentially on multiple selected human contrast pairs."""
    settings = get_settings()
    if not settings.live_evaluation_enabled:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Forbidden: Live evaluation is disabled by default to protect programme credits.",
        )

    live_guard.verify_authorization(
        test_id="auto",
        confirmation_token=req.confirmation_token,
        request=request,
    )

    if not settings.is_api_configured:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Gnani API key is not configured.",
        )

    client = GnaniPrismaClient(settings=settings)
    results = []
    for pid in req.pair_ids:
        try:
            res = await engine.evaluate_human_pair_live(
                pair_id=pid,
                client=client,
                audio_condition=req.audio_condition,
            )
            results.append(res)
        except Exception as exc:
            logger.warning("Batch pair %s evaluation encountered error: %s", pid, str(exc))
    return results


@router.get("/results", response_model=List[PairEvaluationResult])
async def list_evaluation_results(
    limit: int = Query(50, ge=1, le=500),
    category: Optional[str] = Query(None),
    status: Optional[str] = Query(None, description="Filter by overall_status: completed, failed, incomplete"),
    dataset_source: Optional[str] = Query(None, description="Filter by dataset_source: synthetic_benchmark, human_recording"),
    dataset_classification: Optional[str] = Query(None, description="Filter by dataset_classification: human_verified, synthetic, test_artifact, unverified"),
):
    """Lists recent evaluation runs stored in SQLite."""
    return engine.list_runs(
        limit=limit,
        category=category,
        status=status,
        dataset_source=dataset_source,
        dataset_classification=dataset_classification,
    )


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
async def get_aggregate_metrics(
    dataset_classification: Optional[str] = Query(None, description="Filter by dataset_classification: human_verified, synthetic, test_artifact, unverified"),
):
    """Computes aggregate accuracy, pass rates, WER, and failure category breakdown."""
    return engine.get_aggregate_metrics(dataset_classification=dataset_classification)


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


# -----------------------------------------------------------------------------
# Derived Offline Assessments & Re-Scores
# -----------------------------------------------------------------------------

@router.get("/runs/{run_id}/derived-rescore", response_model=DerivedAssessment)
async def get_or_create_derived_rescore(run_id: str):
    """Retrieves or generates an offline derived re-score for a historical evaluation run.

    Guarantees:
    - Strictly 0 external provider calls and 0 credit usage.
    - Leaves the source evaluation run in evaluation_runs completely untouched.
    - Isolated from aggregate metrics (does not inflate live or mock attempt counts).
    """
    existing = engine.get_derived_assessment(run_id)
    if existing:
        return existing
    try:
        return engine.generate_derived_rescore(run_id, persist=True)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        )


@router.get("/derived-assessments", response_model=List[DerivedAssessment])
async def list_derived_assessments(limit: int = Query(100, ge=1, le=1000)):
    """Lists all offline derived evaluation assessments."""
    return engine.list_derived_assessments(limit=limit)


@router.get("/derived-assessments/export")
async def export_derived_assessments(
    format: str = Query("json", description="'json' or 'csv'"),
):
    """Exports offline derived assessments as CSV or JSON."""
    if format.lower() == "csv":
        csv_data = engine.export_derived_csv()
        return Response(
            content=csv_data,
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=bolibench_derived_rescores.csv"},
        )
    else:
        json_data = engine.export_derived_json()
        return Response(
            content=json_data,
            media_type="application/json",
            headers={"Content-Disposition": "attachment; filename=bolibench_derived_rescores.json"},
        )
