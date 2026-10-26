"""API endpoints for BoliBench In-Website Human Audio Recording Studio."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, Response, UploadFile, status
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from app.recording_schemas import (
    BenchmarkCategory,
    ContrastPairVerification,
    RecordingLanguage,
    RecordingMetadata,
    StudioConfigResponse,
    UtteranceRole,
)
from app.services.recording_storage import (
    RecordingStorage,
    StorageCollisionError,
    StorageSecurityError,
    StorageValidationError,
)
from app.services.admin_guard import verify_admin_authorization

logger = logging.getLogger("bolibench.studio_router")
router = APIRouter(prefix="/api/studio", tags=["recording_studio"])

# Default storage singleton
storage = RecordingStorage()

# Curated reference scenario templates to assist human speakers
SCENARIO_TEMPLATES = [
    {
        "id": "pair_qty_001",
        "category": "quantities_and_numbers",
        "pair_id": "pair_qty_001",
        "role": "utterance_a",
        "title": "E-Commerce: 2 Packets",
        "language": "hinglish",
        "reference_transcript": "Do packet chahiye",
        "distinction": "Quantity 2 vs Quantity 3",
    },
    {
        "id": "pair_qty_001_b",
        "category": "quantities_and_numbers",
        "pair_id": "pair_qty_001",
        "role": "utterance_b",
        "title": "E-Commerce: 3 Packets",
        "language": "hinglish",
        "reference_transcript": "Teen packet chahiye",
        "distinction": "Quantity 3 vs Quantity 2",
    },
    {
        "id": "pair_neg_001",
        "category": "negation",
        "pair_id": "pair_neg_001",
        "role": "utterance_a",
        "title": "Ride Booking: Retain Ride",
        "language": "hi-IN",
        "reference_transcript": "Nahi ride cancel mat karo",
        "distinction": "Retain ride (negated cancel) vs Cancel ride",
    },
    {
        "id": "pair_neg_001_b",
        "category": "negation",
        "pair_id": "pair_neg_001",
        "role": "utterance_b",
        "title": "Ride Booking: Cancel Ride",
        "language": "hi-IN",
        "reference_transcript": "Haan ride cancel kar do",
        "distinction": "Cancel ride vs Retain ride",
    },
    {
        "id": "pair_cor_001",
        "category": "corrections_and_mind_changes",
        "pair_id": "pair_cor_001",
        "role": "utterance_a",
        "title": "Travel: Mumbai to Delhi Correction",
        "language": "en-IN",
        "reference_transcript": "Book flight to Mumbai, no wait, to Delhi",
        "distinction": "Destination correction Delhi vs Destination Mumbai",
    },
    {
        "id": "pair_cor_001_b",
        "category": "corrections_and_mind_changes",
        "pair_id": "pair_cor_001",
        "role": "utterance_b",
        "title": "Travel: Mumbai Confirmed",
        "language": "en-IN",
        "reference_transcript": "Book flight to Mumbai, definitely Mumbai",
        "distinction": "Destination Mumbai vs Corrected destination",
    },
    {
        "id": "pair_ent_001",
        "category": "named_entities",
        "pair_id": "pair_ent_001",
        "role": "utterance_a",
        "title": "Banking: Send to Rahul Sharma",
        "language": "hinglish",
        "reference_transcript": "Rahul Sharma ko paise bhej do",
        "distinction": "Recipient Rahul vs Recipient Rohit",
    },
    {
        "id": "pair_ent_001_b",
        "category": "named_entities",
        "pair_id": "pair_ent_001",
        "role": "utterance_b",
        "title": "Banking: Send to Rohit Sharma",
        "language": "hinglish",
        "reference_transcript": "Rohit Sharma ko paise bhej do",
        "distinction": "Recipient Rohit vs Recipient Rahul",
    },
    {
        "id": "pair_cmd_001",
        "category": "command_intent",
        "pair_id": "pair_cmd_001",
        "role": "utterance_a",
        "title": "Account: Check Balance",
        "language": "hi-IN",
        "reference_transcript": "Mera account balance check karo",
        "distinction": "Check balance vs Close account",
    },
    {
        "id": "pair_cmd_001_b",
        "category": "command_intent",
        "pair_id": "pair_cmd_001",
        "role": "utterance_b",
        "title": "Account: Close Account",
        "language": "hi-IN",
        "reference_transcript": "Mera account band kar do",
        "distinction": "Close account vs Check balance",
    },
    {
        "id": "pair_tim_001",
        "category": "time_extraction",
        "pair_id": "pair_tim_001",
        "role": "utterance_a",
        "title": "Delivery: Afternoon 2 PM",
        "language": "hinglish",
        "reference_transcript": "Dopahar ko do baje deliver karna",
        "distinction": "Time 2:00 PM vs Time 3:00 PM",
    },
    {
        "id": "pair_tim_001_b",
        "category": "time_extraction",
        "pair_id": "pair_tim_001",
        "role": "utterance_b",
        "title": "Delivery: Afternoon 3 PM",
        "language": "hinglish",
        "reference_transcript": "Dopahar ko teen baje deliver karna",
        "distinction": "Time 3:00 PM vs Time 2:00 PM",
    },
]


@router.get("/config", response_model=StudioConfigResponse)
async def get_studio_config():
    """Returns studio capabilities, benchmark categories, languages, and scenario templates."""
    categories_meta = [
        {"value": "quantities_and_numbers", "label": "Quantities & Numbers (संख्या और परिमाण)"},
        {"value": "negation", "label": "Negation (नकार और प्रतिषेध)"},
        {"value": "corrections_and_mind_changes", "label": "Corrections & Mind Changes (सुधार)"},
        {"value": "named_entities", "label": "Named Entities (नाम और संस्था)"},
        {"value": "command_intent", "label": "Command Intent (आदेश और मंशा)"},
        {"value": "time_extraction", "label": "Time Extraction (समय निर्धारण)"},
    ]

    languages_meta = [
        {"value": "hi-IN", "label": "Hindi (हिन्दी)"},
        {"value": "en-IN", "label": "Indian English"},
        {"value": "hinglish", "label": "Hinglish (Code-Mixed / कोड-मिश्रित)"},
    ]

    return StudioConfigResponse(
        supported_categories=categories_meta,
        supported_languages=languages_meta,
        target_duration_seconds=50.0,
        standard_format="16 kHz, Mono, 16-bit PCM WAV",
        storage_directory="human_recordings_v1",
        scenario_templates=SCENARIO_TEMPLATES,
    )


@router.post("/upload", response_model=RecordingMetadata)
async def upload_recording(
    request: Request,
    audio_file: UploadFile = File(..., description="Raw audio recording stream (WebM/OGG/WAV)"),
    speaker_id: str = Form(..., description="Pseudonymous speaker identifier (e.g. spk_rajesh_01)"),
    language: str = Form(..., description="Language: hi-IN, en-IN, or hinglish"),
    category: str = Form(..., description="One of the six benchmark categories"),
    reference_transcript: str = Form(..., description="Intended reference transcript"),
    consent_given: bool = Form(..., description="Explicit confirmation of consent"),
    consent_statement: str = Form(..., description="Consent declaration text accepted by speaker"),
    recording_id: Optional[str] = Form(None, description="Optional unique recording ID"),
    scenario_id: Optional[str] = Form(None, description="Optional benchmark scenario ID"),
    pair_id: Optional[str] = Form(None, description="Optional contrastive A/B pair ID"),
    utterance_role: str = Form("standalone", description="utterance_a, utterance_b, or standalone"),
    verified_transcript: Optional[str] = Form(None, description="Manually verified transcript"),
    is_transcript_verified: bool = Form(False, description="Whether manual transcript verification is confirmed"),
    client_duration_seconds: Optional[float] = Form(None, description="Client measured duration"),
    idempotency_key: Optional[str] = Form(None, description="Optional caller idempotency key"),
):
    """Secure endpoint to ingest, standardize, validate, and store a human speech clip.

    Guarantees:
    - Zero external network calls (No Gnani API calls).
    - Preserves unmodified raw audio.
    - Generates 16kHz mono PCM WAV.
    - Runs acoustic diagnostics (sample rate, channels, silence, clipping, 50s completeness).
    - Writes to isolated human_recordings_v1/ store.
    - Upload idempotency supported via Form param or X-Idempotency-Key header.
    """
    if not consent_given:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Explicit consent is mandatory to record and store voice samples for evaluation.",
        )

    # Resolve idempotency key from Form or Header
    effective_idempotency_key = idempotency_key or request.headers.get("x-idempotency-key") or request.headers.get("idempotency-key")
    if effective_idempotency_key:
        effective_idempotency_key = effective_idempotency_key.strip()

    # Read audio content into memory
    try:
        content = await audio_file.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read uploaded audio stream: {str(e)}",
        )

    # Max upload limit guard (25MB)
    if len(content) > 26_214_400:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Uploaded audio exceeds maximum allowed limit of 25 MB.",
        )

    try:
        metadata = storage.save_recording(
            raw_bytes=content,
            original_filename=audio_file.filename or "recording.webm",
            speaker_id=speaker_id,
            language=language,
            category=category,
            reference_transcript=reference_transcript,
            consent_given=consent_given,
            consent_statement=consent_statement,
            recording_id=recording_id,
            scenario_id=scenario_id,
            pair_id=pair_id,
            utterance_role=utterance_role,
            verified_transcript=verified_transcript,
            is_transcript_verified=is_transcript_verified,
            client_duration_seconds=client_duration_seconds,
            idempotency_key=effective_idempotency_key,
        )
        return metadata
    except StorageSecurityError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except StorageCollisionError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    except StorageValidationError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected error saving recording: %s", str(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal error saving recording: {str(e)}")


@router.get("/recordings")
async def list_recordings(
    category: Optional[str] = Query(None, description="Filter by benchmark category"),
    speaker_id: Optional[str] = Query(None, description="Filter by speaker ID"),
    pair_id: Optional[str] = Query(None, description="Filter by pair ID"),
    is_complete_50s: Optional[bool] = Query(None, description="Filter by 50s completion status"),
    classification: Optional[str] = Query(None, description="Filter by classification cohort (human_verified, synthetic, test_artifact, unverified)"),
    exclude_test_artifacts: bool = Query(False, description="Exclude test artifacts if true"),
):
    """Lists saved recordings from human_recordings_v1/."""
    return storage.list_recordings(
        category=category,
        speaker_id=speaker_id,
        pair_id=pair_id,
        is_complete_50s=is_complete_50s,
        classification=classification,
        exclude_test_artifacts=exclude_test_artifacts,
    )


@router.get("/recordings/{recording_id}", response_model=RecordingMetadata)
async def get_recording(recording_id: str):
    """Retrieves metadata and acoustic diagnostics for an individual recording."""
    try:
        meta = storage.get_recording(recording_id)
        if not meta:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Recording '{recording_id}' not found.",
            )
        return meta
    except StorageSecurityError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.get("/recordings/{recording_id}/audio")
async def get_recording_audio(
    recording_id: str,
    raw: bool = Query(False, description="Stream raw uploaded audio if true, or standardized WAV copy if false"),
):
    """Streams the audio file (standardized 16kHz WAV by default, or original raw)."""
    try:
        audio_path = storage.get_audio_file_path(recording_id, raw=raw)
        media_type = "audio/wav"
        if raw:
            ext = audio_path.suffix.lower()
            if ext == ".webm":
                media_type = "audio/webm"
            elif ext == ".ogg":
                media_type = "audio/ogg"
            elif ext == ".mp3":
                media_type = "audio/mpeg"

        return FileResponse(
            path=str(audio_path),
            media_type=media_type,
            filename=audio_path.name,
        )
    except FileNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Audio for '{recording_id}' not found.")
    except StorageSecurityError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.get("/pairs")
async def list_candidate_pairs(
    classification: Optional[str] = Query(None, description="Filter by classification cohort"),
    curated_only: bool = Query(False, description="Filter only curated human_verified pairs"),
    exclude_test_artifacts: bool = Query(False, description="Exclude test artifacts"),
):
    """Lists candidate contrastive pairs and their verification status."""
    return storage.list_candidate_pairs(
        classification=classification,
        curated_only=curated_only,
        exclude_test_artifacts=exclude_test_artifacts,
    )


class RegisterSpeakerRequest(BaseModel):
    speaker_id: str


@router.get("/trusted-speakers")
async def list_trusted_speakers():
    """Lists registered trusted human speakers."""
    return {"trusted_speakers": storage.list_trusted_speakers()}


@router.post("/trusted-speakers", dependencies=[Depends(verify_admin_authorization)])
async def register_trusted_speaker(req: RegisterSpeakerRequest):
    """Registers a verified human speaker in the trusted speaker registry.

    Requires administrative authorization (ADMIN_API_KEY via Bearer or X-Admin-Token).
    """
    try:
        storage.register_trusted_speaker(req.speaker_id)
        return {"status": "registered", "speaker_id": req.speaker_id}
    except (StorageSecurityError, StorageValidationError) as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


class CreatePairRequest(BaseModel):
    pair_id: str
    category: str
    language: str
    speaker_id: str
    scenario_description: Optional[str] = None
    expected_distinction: Optional[str] = None
    distinction_field: Optional[str] = None


@router.post("/pairs/create")
async def create_contrast_pair(req: CreatePairRequest):
    """Creates or initializes a candidate contrast pair definition before audio recording."""
    try:
        return storage.create_pair_definition(
            pair_id=req.pair_id,
            category=req.category,
            language=req.language,
            speaker_id=req.speaker_id,
            scenario_description=req.scenario_description,
            expected_distinction=req.expected_distinction,
            distinction_field=req.distinction_field,
        )
    except StorageSecurityError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except StorageValidationError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.get("/pairs/{pair_id}")
async def get_contrast_pair(pair_id: str):
    """Retrieves full details, transcripts, and audio paths for a pair."""
    pair = storage.get_pair(pair_id)
    if not pair:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Pair '{pair_id}' not found.")
    return pair


class VerifyPairRequest(BaseModel):
    confirmed_semantic_distinction: bool
    distinction_field: Optional[str] = None
    expected_distinction_desc: Optional[str] = None
    operator_notes: Optional[str] = None
    verified_by: Optional[str] = "operator"


@router.post("/pairs/{pair_id}/verify", response_model=ContrastPairVerification)
async def verify_contrast_pair(pair_id: str, req: VerifyPairRequest):
    """Explicitly verifies and seals a matched contrastive A/B pair."""
    try:
        pair_record = storage.verify_contrast_pair(
            pair_id=pair_id,
            confirmed_semantic_distinction=req.confirmed_semantic_distinction,
            distinction_field=req.distinction_field,
            expected_distinction_desc=req.expected_distinction_desc,
            operator_notes=req.operator_notes,
            verified_by=req.verified_by,
        )
        return pair_record
    except StorageSecurityError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except StorageValidationError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.exception("Error verifying pair: %s", str(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
