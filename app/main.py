"""FastAPI Application entrypoint for BoliBench."""

from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.config import get_settings
from app.schemas import (
    LANGUAGE_META,
    HealthResponse,
    LanguageCode,
    TranscriptFormat,
    TranscriptionResponse,
)
from app.services.audio_validator import AudioValidationError, AudioValidator
from app.services.gnani_client import (
    GnaniAuthenticationError,
    GnaniBadRequestError,
    GnaniClientError,
    GnaniNetworkError,
    GnaniPrismaClient,
    GnaniRateLimitError,
    GnaniServerError,
    GnaniServiceUnavailableError,
    GnaniTimeoutError,
    MissingCredentialsError,
)

# Setup logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("bolibench.app")

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(
    title="BoliBench — Indian Voice AI Crash-Test Lab",
    description="Evaluating task-critical speech recognition errors on Indian voice AI agents.",
    version="0.1.0",
)

# Restricted CORS middleware: Only allow trusted same-origin local development hosts
ALLOWED_ORIGINS = [
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Static files & Templates
static_dir = BASE_DIR / "static"
templates_dir = BASE_DIR / "templates"

if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

templates = Jinja2Templates(directory=str(templates_dir))

from app.routers.evaluation import router as evaluation_router
from app.evaluation.dataset import load_seed_dataset

app.include_router(evaluation_router)


# -----------------------------------------------------------------------------
# Exception Handlers
# -----------------------------------------------------------------------------

@app.exception_handler(AudioValidationError)
async def audio_validation_exception_handler(request: Request, exc: AudioValidationError):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "success": False,
            "error_type": exc.error_type,
            "message": exc.message,
            "details": exc.details,
        },
    )


@app.exception_handler(MissingCredentialsError)
async def missing_credentials_exception_handler(request: Request, exc: MissingCredentialsError):
    return JSONResponse(
        status_code=status.HTTP_401_UNAUTHORIZED,
        content={
            "success": False,
            "error_type": exc.error_type,
            "message": exc.message,
            "remediation": "Copy .env.example to .env and set your GNANI_API_KEY from https://app.gnani.ai/voice",
        },
    )


@app.exception_handler(GnaniClientError)
async def gnani_client_exception_handler(request: Request, exc: GnaniClientError):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error_type": exc.error_type,
            "message": exc.message,
            "details": exc.details,
        },
    )


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint exposing configuration status and capabilities."""
    settings = get_settings()
    return HealthResponse(
        status="ok",
        gnani_api_configured=settings.is_api_configured,
        app_environment=settings.app_env,
        supported_languages=[lang.value for lang in LanguageCode],
        supported_formats=[fmt.value for fmt in TranscriptFormat],
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.get("/", response_class=HTMLResponse)
async def index_page(request: Request):
    """Renders the interactive audio upload and transcription lab interface."""
    settings = get_settings()
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "is_configured": settings.is_api_configured,
            "languages": LANGUAGE_META,
            "max_mb": settings.max_audio_size_bytes // (1024 * 1024),
            "test_cases": load_seed_dataset(),
        },
    )


@app.post("/api/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    audio_file: UploadFile = File(..., description="Audio file (WAV, MP3, OGG, FLAC, AAC, M4A <= 60s)"),
    language_code: str = Form(..., description="Target language BCP-47 code (e.g. hi-IN, en-IN)"),
    format_mode: str = Form("verbatim", description="verbatim or transcribe (with ITN)"),
    itn_native_numerals: bool = Form(False, description="Native numerals when format=transcribe"),
    enable_substitution: bool = Form(False, description="Enable post-recognition find-and-replace"),
    substitution_map: Optional[str] = Form(None, description="JSON array of substitution rules"),
    bias_list: Optional[str] = Form(None, description="Words to boost during recognition"),
    bias_score: Optional[float] = Form(None, description="Boost strength (0.0 to 2.0)"),
    mock_mode: bool = Form(False, description="Run in mock mode for offline testing"),
):
    """Secure endpoint to validate and transcribe an audio clip via Gnani Prisma v2.5 STT."""
    settings = get_settings()

    # 1. Validate Language Code
    valid_languages = {lang.value for lang in LanguageCode}
    if language_code not in valid_languages:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported language_code '{language_code}'. Must be one of: {', '.join(sorted(valid_languages))}",
        )

    # 2. Read and Validate Audio File in Memory (Do NOT write to disk)
    try:
        content = await audio_file.read()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read uploaded audio file: {str(exc)}",
        )

    validator = AudioValidator(max_size_bytes=settings.max_audio_size_bytes)
    ext, content_type = validator.validate(audio_file.filename, content)

    # 3. Handle Offline / Mock Mode (For testing or when credentials aren't present)
    if mock_mode:
        meta = LANGUAGE_META.get(language_code, {"name": "Indian Language", "sample": "नमस्ते"})
        return TranscriptionResponse(
            success=True,
            transcript=f"[MOCK TRANSCRIPTION - {meta['name']}]: {meta['sample']}",
            request_id="mock_req_00000000",
            model="gnani-prisma-v2.5-mock",
            server_timestamp=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S.000"),
            language_code=language_code,
            format=format_mode,
            backend_latency_ms=15.5,
            is_mocked=True,
        )

    # 4. Execute Real Gnani Prisma Client Request
    client = GnaniPrismaClient(settings=settings)
    result = await client.transcribe(
        audio_bytes=content,
        filename=audio_file.filename or f"audio{ext}",
        content_type=content_type,
        language_code=language_code,
        format_mode=format_mode,
        itn_native_numerals=itn_native_numerals,
        enable_substitution=enable_substitution,
        substitution_map=substitution_map,
        bias_list=bias_list,
        bias_score=bias_score,
    )

    return TranscriptionResponse(**result)
