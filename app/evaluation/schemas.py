"""Pydantic schemas for the Paired-Contrast Evaluation Engine."""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class TestCategory(str, Enum):
    """Evaluation categories for voice agent crash testing."""
    __test__ = False
    QUANTITIES_AND_NUMBERS = "quantities_and_numbers"
    NEGATION = "negation"
    CORRECTIONS_AND_MIND_CHANGES = "corrections_and_mind_changes"
    NAMED_ENTITIES = "named_entities"
    PRODUCT_NAMES = "product_names"
    COMMAND_INTENT = "command_intent"
    HINDI_ENGLISH_CODE_SWITCHING = "hindi_english_code_switching"


class CodeSwitchingType(str, Enum):
    """Linguistic and code-mixing classifications."""
    MONOLINGUAL_HINDI = "monolingual_hindi"
    MONOLINGUAL_ENGLISH = "monolingual_english"
    HINGLISH_CODE_MIXED = "hinglish_code_mixed"
    HINGLISH_CODE_SWITCHED = "hinglish_code_switched"


class FailureCategory(str, Enum):
    """Taxonomy of speech recognition & downstream voice agent failures."""
    QUANTITY_ERROR = "quantity_error"
    MISSED_NEGATION = "missed_negation"
    CORRECTION_ERROR = "correction_error"
    ENTITY_SUBSTITUTION = "entity_substitution"
    INTENT_ERROR = "intent_error"
    EXTRACTION_FAILURE = "extraction_failure"
    HIGH_WER_CORRUPTION = "high_wer_corruption"
    NONE = "none"


class ExecutionMode(str, Enum):
    """Execution mode distinguishing deterministic test mocks from live provider calls."""
    MOCK = "mock"
    LIVE = "live"


class AudioCondition(str, Enum):
    """Acoustic environment metadata."""
    CLEAN_SYNTHETIC = "clean_synthetic"
    CLEAN_RECORDED = "clean_recorded"
    NOISY_SYNTHETIC = "noisy_synthetic"
    NOISY_RECORDED = "noisy_recorded"
    PENDING_ACOUSTIC = "pending_acoustic"


class UtteranceSpec(BaseModel):
    """Specification of a single spoken utterance in an evaluation pair."""
    id: str = Field(..., description="Unique utterance ID within pair, e.g. A or B")
    text: str = Field(..., description="Spoken prompt text representation")
    reference_transcript: str = Field(..., description="Verified ground-truth transcript")
    expected_intent: str = Field(..., description="Expected downstream agent intent")
    expected_critical_fields: Dict[str, Any] = Field(
        default_factory=dict,
        description="Manually specified ground-truth critical fields (e.g. quantity, negation, destination)",
    )
    expected_action: str = Field(..., description="Downstream agent action that should be executed")
    simulated_crashed_transcript: Optional[str] = Field(
        None,
        description="Acoustically realistic error transcript used for deterministic crash demonstration",
    )


class TestCasePair(BaseModel):
    """Structured representation of a paired-contrast evaluation test."""
    __test__ = False
    id: str = Field(..., description="Unique test case identifier (e.g. pair_qty_001)")
    category: TestCategory = Field(..., description="Evaluation category")
    language_code: str = Field(..., description="BCP-47 language tag (e.g. hi-IN, en-IN)")
    code_switching_type: CodeSwitchingType = Field(..., description="Linguistic code-switching type")
    description: str = Field(..., description="Explanation of the critical distinction tested")
    utterance_a: UtteranceSpec = Field(..., description="First utterance in the contrast pair")
    utterance_b: UtteranceSpec = Field(..., description="Second utterance in the contrast pair")
    expected_distinction: str = Field(..., description="Expected semantic distinction between A and B")
    distinction_field: str = Field(..., description="Primary field name where distinction resides (e.g. quantity, is_negated)")
    audio_condition: AudioCondition = Field(AudioCondition.CLEAN_SYNTHETIC, description="Acoustic environment condition")
    provenance: str = Field("synthetic_curated", description="Data origin and authoring notes")
    consent_metadata: Optional[str] = Field("synthetic_no_pii", description="Consent and privacy declaration")


class UtteranceEvaluationResult(BaseModel):
    """Evaluation score for a single utterance."""
    utterance_id: str
    reference_transcript: str
    actual_transcript: str
    wer: float = Field(..., description="Word Error Rate (0.0 to 1.0+)")
    substitutions: int
    deletions: int
    insertions: int
    reference_length: int
    extracted_fields: Dict[str, Any] = Field(default_factory=dict)
    critical_fields_match: bool
    field_matches: Dict[str, bool] = Field(default_factory=dict)
    detected_intent: str
    intent_match: bool
    task_success: bool
    latency_ms: Optional[float] = None
    request_id: Optional[str] = None


class PairEvaluationResult(BaseModel):
    """Aggregate evaluation result for a paired-contrast test."""
    run_id: str = Field(..., description="Unique run identifier")
    test_case_id: str
    category: str
    language_code: str
    execution_mode: ExecutionMode
    audio_condition: str
    timestamp: str
    utterance_a: UtteranceEvaluationResult
    utterance_b: UtteranceEvaluationResult
    wer_a: float
    wer_b: float
    average_wer: float
    critical_field_accuracy: float = Field(..., description="Percentage of critical fields correctly extracted across pair (0.0 to 1.0)")
    semantic_distinction_preserved: bool = Field(..., description="True if the critical distinction between A and B was preserved")
    task_success: bool = Field(..., description="True if downstream actions for both utterances would be correct")
    primary_failure_category: FailureCategory
    failure_explanation: str
    client_measured_latency_ms: Optional[float] = None
    provider_processing_time_ms: Optional[float] = None


class AggregateMetrics(BaseModel):
    """Summary metrics across evaluation runs."""
    total_runs: int = 0
    mock_runs: int = 0
    live_runs: int = 0
    distinction_preservation_rate: float = 0.0
    task_success_rate: float = 0.0
    average_wer: float = 0.0
    average_critical_field_accuracy: float = 0.0
    category_breakdown: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    failure_category_breakdown: Dict[str, int] = Field(default_factory=dict)


class EvaluateTranscriptsRequest(BaseModel):
    """Request payload to score custom transcripts against a test case."""
    test_case_id: str
    transcript_a: str
    transcript_b: str
    execution_mode: ExecutionMode = ExecutionMode.MOCK
    audio_condition: AudioCondition = AudioCondition.CLEAN_SYNTHETIC
    client_latency_ms: Optional[float] = None
