"""Data models and schemas for BoliBench."""

from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class LanguageCode(str, Enum):
    """Supported 10 Indian language codes according to Gnani Prisma v2.5 STT specification."""
    BENGALI = "bn-IN"
    ENGLISH = "en-IN"
    GUJARATI = "gu-IN"
    HINDI = "hi-IN"
    KANNADA = "kn-IN"
    MALAYALAM = "ml-IN"
    MARATHI = "mr-IN"
    PUNJABI = "pa-IN"
    TAMIL = "ta-IN"
    TELUGU = "te-IN"


LANGUAGE_META: Dict[str, Dict[str, str]] = {
    "bn-IN": {"name": "Bengali", "native": "বাংলা", "sample": "আমি ভাত খাই"},
    "en-IN": {"name": "English (India)", "native": "English (India)", "sample": "I am going to the market"},
    "gu-IN": {"name": "Gujarati", "native": "ગુજરાતી", "sample": "હું બજાર જાઉં છું"},
    "hi-IN": {"name": "Hindi", "native": "हिन्दी", "sample": "मैं बाज़ार जा रहा हूँ"},
    "kn-IN": {"name": "Kannada", "native": "ಕನ್ನಡ", "sample": "ನಾನು ಮಾರುಕಟ್ಟೆಗೆ ಹೋಗುತ್ತೇನೆ"},
    "ml-IN": {"name": "Malayalam", "native": "മലയാളം", "sample": "ഞാൻ ചന്തയിലേക്ക് പോകുന്നു"},
    "mr-IN": {"name": "Marathi", "native": "मराठी", "sample": "मी बाजारात जातोय"},
    "pa-IN": {"name": "Punjabi", "native": "ਪੰਜਾਬੀ", "sample": "ਮੈਂ ਬਾਜ਼ਾਰ ਜਾ ਰਿਹਾ ਹਾਂ"},
    "ta-IN": {"name": "Tamil", "native": "தமிழ்", "sample": "நான் சந்தைக்கு செல்கிறேன்"},
    "te-IN": {"name": "Telugu", "native": "తెలుగు", "sample": "నేను మార్కెట్‌కి వెళ్తున్నాను"},
}


class TranscriptFormat(str, Enum):
    """Transcription formatting mode."""
    VERBATIM = "verbatim"
    TRANSCRIBE = "transcribe"


class HealthResponse(BaseModel):
    """Health status response."""
    status: str = "ok"
    gnani_api_configured: bool
    live_evaluation_enabled: bool = False
    app_environment: str
    supported_languages: List[str]
    supported_formats: List[str]
    timestamp: str


class TranscriptionResponse(BaseModel):
    """Transcription output model returned by BoliBench backend.

    Latency Semantics:
    - client_request_elapsed_ms: Locally measured round-trip elapsed time around the upstream Gnani HTTP request.
    - backend_latency_ms: Backwards-compatible alias for client_request_elapsed_ms.
    - provider_processing_time_ms: Internal server processing time reported by provider; null when not reported.
    """
    success: bool = True
    transcript: str = Field(..., description="Recognized speech transcript")
    request_id: Optional[str] = Field(None, description="Gnani API request ID or trace ID")
    model: Optional[str] = Field(None, description="Speech recognition model identifier if returned by provider")
    server_timestamp: Optional[str] = Field(None, description="Timestamp from Gnani server")
    language_code: str = Field(..., description="Language code of audio")
    format: str = Field("verbatim", description="verbatim or transcribe (ITN)")
    backend_latency_ms: float = Field(
        ...,
        description="Backwards-compatible alias for client_request_elapsed_ms: locally measured elapsed time in ms around the upstream Gnani HTTP request",
    )
    client_request_elapsed_ms: Optional[float] = Field(
        None,
        description="Locally measured elapsed time in ms around upstream Gnani HTTP request (not provider internal processing)",
    )
    provider_processing_time_ms: Optional[float] = Field(
        None,
        description="Provider-internal backend processing duration in ms if reported by upstream API; null/None when not reported by Gnani",
    )
    is_mocked: bool = Field(False, description="Flag indicating if response was produced by a mock engine")


class ErrorResponse(BaseModel):
    """Standardized error response model."""
    success: bool = False
    error_type: str = Field(..., description="Machine-readable error type identifier")
    message: str = Field(..., description="Human-readable explanation of error")
    details: Optional[dict] = Field(None, description="Optional diagnostic details")
