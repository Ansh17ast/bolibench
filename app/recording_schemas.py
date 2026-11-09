"""Pydantic schemas for the BoliBench Human Recording Studio.

Captures metadata, audio diagnostics, consent records, and contrastive pair definitions
for human speech recordings stored in human_recordings_v1/.
"""

from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class BenchmarkCategory(str, Enum):
    """The six standard BoliBench benchmark evaluation categories."""
    QUANTITIES_AND_NUMBERS = "quantities_and_numbers"
    NEGATION = "negation"
    CORRECTIONS_AND_MIND_CHANGES = "corrections_and_mind_changes"
    NAMED_ENTITIES = "named_entities"
    COMMAND_INTENT = "command_intent"
    TIME_EXTRACTION = "time_extraction"


class RecordingLanguage(str, Enum):
    """Supported recording languages for BoliBench."""
    HINDI = "hi-IN"
    ENGLISH = "en-IN"
    HINGLISH = "hinglish"


class UtteranceRole(str, Enum):
    """Role within a paired-contrast evaluation scenario."""
    UTTERANCE_A = "utterance_a"
    UTTERANCE_B = "utterance_b"
    STANDALONE = "standalone"


class ValidationStatus(str, Enum):
    """Quality and duration validation status of a recording."""
    VALID_50S = "valid_50s"
    VALID_RECORDING = "valid_recording"
    INCOMPLETE_EARLY_STOP = "incomplete_early_stop"
    DURATION_EXCEEDED = "duration_exceeded"
    SILENT_WARNING = "silent_warning"
    CLIPPED_WARNING = "clipped_warning"
    INVALID_FORMAT = "invalid_format"
    PENDING = "pending"


class DatasetClassification(str, Enum):
    """Provenance and verification cohort classification for BoliBench datasets."""
    HUMAN_VERIFIED = "human_verified"
    SYNTHETIC = "synthetic"
    TEST_ARTIFACT = "test_artifact"
    UNVERIFIED = "unverified"


class AudioDiagnostics(BaseModel):
    """Acoustic and technical validation metrics for a standardized recording."""
    sample_rate_hz: int = Field(..., description="Audio sample rate (must be 16000 Hz)")
    channels: int = Field(..., description="Audio channel count (must be 1 mono)")
    duration_seconds: float = Field(..., description="Exact duration in seconds")
    is_complete_50s: bool = Field(..., description="True if clip completed valid recording duration (<= 600s)")
    rms_amplitude: float = Field(..., description="Root-mean-square amplitude [0.0, 1.0]")
    peak_amplitude: float = Field(..., description="Maximum absolute sample value [0.0, 1.0]")
    is_silent: bool = Field(..., description="Flag indicating audio signal is below audible speech threshold")
    clipped_samples_count: int = Field(..., description="Count of samples reaching maximum digital ceiling")
    clipping_ratio: float = Field(..., description="Proportion of clipped samples to total samples")
    is_clipped: bool = Field(..., description="Flag indicating severe acoustic clipping / saturation")
    decoding_successful: bool = Field(..., description="True if audio container and frames decoded cleanly")
    converter_used: str = Field("ffmpeg", description="Conversion engine used (ffmpeg or wave_fallback)")


class RecordingMetadata(BaseModel):
    """Complete metadata schema for an individual human voice recording."""
    recording_id: str = Field(..., description="Unique alphanumeric identifier (e.g. rec_20261022_a1b2c3)")
    speaker_id: str = Field(..., description="Pseudonymous speaker identifier (e.g. spk_rajesh_01)")
    language: str = Field(..., description="Language code: hi-IN, en-IN, or hinglish")
    category: str = Field(..., description="One of the six BoliBench benchmark categories")
    scenario_id: Optional[str] = Field(None, description="Benchmark scenario identifier where applicable")
    pair_id: Optional[str] = Field(None, description="Contrastive A/B pair ID where applicable")
    utterance_role: UtteranceRole = Field(UtteranceRole.STANDALONE, description="Utterance role (A, B, or standalone)")
    reference_transcript: str = Field(..., description="Intended reference text the speaker aimed to read")
    verified_transcript: Optional[str] = Field(None, description="Manually verified transcript of spoken audio")
    is_transcript_verified: bool = Field(False, description="Whether manual transcript verification has been confirmed")
    consent_given: bool = Field(..., description="Explicit confirmation of consent to record and use audio for evaluation")
    consent_statement: str = Field(..., description="Consent declaration text accepted by speaker")
    consent_timestamp: str = Field(..., description="ISO-8601 UTC timestamp of consent grant")
    created_at: str = Field(..., description="ISO-8601 UTC timestamp of recording upload")
    client_duration_seconds: Optional[float] = Field(None, description="Duration measured by browser timer")
    diagnostics: Optional[AudioDiagnostics] = Field(None, description="Server-side audio validation diagnostics")
    validation_status: ValidationStatus = Field(ValidationStatus.PENDING, description="Overall validation classification")
    raw_audio_filename: str = Field(..., description="Filename of original unmodified uploaded audio")
    standardized_audio_filename: str = Field(..., description="Filename of standardized 16kHz mono PCM WAV copy")
    raw_audio_path: str = Field(..., description="Relative path to raw audio inside human_recordings_v1/")
    standardized_audio_path: str = Field(..., description="Relative path to standardized WAV inside human_recordings_v1/")
    idempotency_key: Optional[str] = Field(None, description="Optional caller idempotency key")
    content_sha256: Optional[str] = Field(None, description="SHA-256 fingerprint of the raw audio content")
    classification: Optional[str] = Field(DatasetClassification.UNVERIFIED.value, description="Dataset provenance classification (human_verified, synthetic, test_artifact, unverified)")


class PairStatus(str, Enum):
    """Lifecycle statuses for a human-recorded contrastive pair."""
    INCOMPLETE = "incomplete"
    AWAITING_VERIFICATION = "awaiting_verification"
    READY_FOR_EVALUATION = "ready_for_evaluation"
    EVALUATION_IN_PROGRESS = "evaluation_in_progress"
    COMPLETED = "completed"
    PROVIDER_FAILURE = "provider_failure"


class ContrastPairVerification(BaseModel):
    """Metadata and integrity verification record for a contrastive A/B test pair."""
    pair_id: str = Field(..., description="Unique contrastive pair ID (e.g. pair_human_qty_001)")
    category: str = Field(..., description="One of the six BoliBench benchmark categories")
    language: str = Field(..., description="Language code: hi-IN, en-IN, or hinglish")
    utterance_a_recording_id: str = Field(..., description="Recording ID for Utterance A")
    utterance_b_recording_id: str = Field(..., description="Recording ID for Utterance B")
    utterance_a_transcript: str = Field(..., description="Verified transcript for Utterance A")
    utterance_b_transcript: str = Field(..., description="Verified transcript for Utterance B")
    speaker_id: Optional[str] = Field(None, description="Pseudonymous speaker identifier")
    scenario_description: Optional[str] = Field(None, description="Benchmark scenario title / description")
    distinction_field: Optional[str] = Field(None, description="Field tested by contrast (e.g. quantity, negation)")
    expected_distinction_desc: Optional[str] = Field(None, description="Human description of what differentiates A and B")
    is_verified_pair: bool = Field(False, description="True only after operator confirms pair validity")
    confirmed_semantic_distinction: bool = Field(False, description="Operator verified that pair tests genuine semantic distinction")
    operator_notes: Optional[str] = Field(None, description="Auditor / operator notes on pair validity")
    verified_by: Optional[str] = Field(None, description="Pseudonym or identifier of verifying operator")
    verified_at: Optional[str] = Field(None, description="ISO-8601 UTC timestamp of pair verification")
    created_at: str = Field(..., description="ISO-8601 UTC timestamp of pair record creation")
    status: str = Field("ready_for_evaluation", description="Current lifecycle status of the pair")
    last_run_id: Optional[str] = Field(None, description="ID of most recent evaluation run on this pair")
    is_evaluated: bool = Field(False, description="True if pair has completed at least one evaluation run")
    classification: str = Field(DatasetClassification.UNVERIFIED.value, description="Dataset provenance classification (human_verified, synthetic, test_artifact, unverified)")


class StudioTranscriptionResponse(BaseModel):
    """Response returned when transcribing a studio audio recording."""
    success: bool = True
    transcript: str
    language_code: str
    utterance_role: str = "utterance_a"
    duration_seconds: float
    provider: str
    credits_consumed: int = 0
    external_calls: int = 0
    disclaimer: str


class StudioConfigResponse(BaseModel):
    """Configuration and capabilities exposed to the recording studio UI."""
    supported_categories: List[Dict[str, str]]
    supported_languages: List[Dict[str, str]]
    target_duration_seconds: float = 600.0
    max_duration_seconds: float = 600.0
    standard_format: str = "16 kHz, Mono, 16-bit PCM WAV"
    storage_directory: str = "human_recordings_v1"
    scenario_templates: List[Dict[str, str]]
    live_transcription_enabled: bool = False
    stt_provider_configured: bool = False
    stt_provider_name: Optional[str] = None
    max_stt_duration_seconds: float = 60.0
