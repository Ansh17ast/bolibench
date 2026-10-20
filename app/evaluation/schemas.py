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
    PROVIDER_ERROR = "provider_error"
    NONE = "none"


class ExecutionMode(str, Enum):
    """Execution mode distinguishing deterministic test mocks from live provider calls."""
    MOCK = "mock"
    LIVE = "live"


class AudioCondition(str, Enum):
    """Acoustic environment metadata and simulation classification."""
    TEXT_ONLY_SIMULATION = "text_only_simulation"
    CLEAN_RECORDED = "clean_recorded"
    NOISY_RECORDED = "noisy_recorded"
    CLEAN_SYNTHETIC = "clean_synthetic"
    NOISY_SYNTHETIC = "noisy_synthetic"
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
        description="Hand-authored simulated transcript corruption for crash demonstration (text-only simulation, not measured model error)",
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
    audio_condition: AudioCondition = Field(AudioCondition.TEXT_ONLY_SIMULATION, description="Acoustic environment condition or text simulation classification")
    provenance: str = Field("hand_authored_text_simulation", description="Data origin and authoring notes")
    consent_metadata: Optional[str] = Field("text_simulation_no_pii", description="Consent and privacy declaration")


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
    status: str = Field("completed", description="Request status: completed, failed, or skipped")
    error_message: Optional[str] = Field(None, description="Sanitized provider error message if failed")
    latency_ms: Optional[float] = Field(None, description="Actual measured client-side elapsed latency (ms) for live call; null for mock")
    client_request_elapsed_ms: Optional[float] = Field(None, description="Client-side measured HTTP round-trip elapsed time (ms) for this utterance; null in mock mode")
    provider_processing_time_ms: Optional[float] = Field(None, description="Provider-reported backend processing latency (ms); null when not reported by provider")
    simulated_latency_ms: Optional[float] = Field(None, description="Simulated latency (ms) for demonstration; null if not simulated")
    request_id: Optional[str] = Field(None, description="Provider request ID for live call; null for mock")
    normalized_wer: Optional[float] = Field(None, description="Supplementary cross-script-normalized Word Error Rate")


class PairEvaluationResult(BaseModel):
    """Aggregate evaluation result for a paired-contrast test."""
    run_id: str = Field(..., description="Unique run identifier")
    test_case_id: str
    category: str
    language_code: str
    execution_mode: ExecutionMode
    audio_condition: str
    timestamp: str
    overall_status: str = Field("completed", description="Overall run status: completed, incomplete, or failed")
    utterance_a: UtteranceEvaluationResult
    utterance_b: UtteranceEvaluationResult
    wer_a: float
    wer_b: float
    average_wer: float
    normalized_wer_a: Optional[float] = Field(None, description="Supplementary cross-script-normalized WER for Utterance A")
    normalized_wer_b: Optional[float] = Field(None, description="Supplementary cross-script-normalized WER for Utterance B")
    average_normalized_wer: Optional[float] = Field(None, description="Supplementary average cross-script-normalized WER across the pair")
    critical_field_accuracy: float = Field(..., description="Percentage of critical fields correctly extracted across pair (0.0 to 1.0)")
    semantic_distinction_preserved: bool = Field(..., description="True if the critical distinction between A and B was preserved")
    task_success: bool = Field(..., description="True if downstream actions for both utterances would be correct")
    primary_failure_category: FailureCategory
    failure_explanation: str
    client_measured_latency_ms: Optional[float] = Field(None, description="Actual measured total client elapsed latency (ms) for live requests; null in mock mode")
    client_request_elapsed_ms: Optional[float] = Field(None, description="Total client-side measured round-trip elapsed latency (ms) across both requests; alias for client_measured_latency_ms")
    provider_processing_time_ms: Optional[float] = Field(None, description="Actual backend processing latency (ms) reported by provider; null if not reported by provider or in mock mode")
    simulated_latency_ms: Optional[float] = Field(None, description="Simulated latency for demonstration purposes (clearly distinguished from measured latency)")


class ModePerformanceMetrics(BaseModel):
    """Performance metrics aggregated for a specific execution mode (completed runs only)."""
    total_attempts: int = Field(0, description="Total attempts initiated in this mode (completed, failed, incomplete)")
    completed_runs: int = Field(0, description="Count of successfully completed evaluations in this mode")
    failed_runs: int = Field(0, description="Count of operational failed runs in this mode")
    incomplete_runs: int = Field(0, description="Count of partial failures in this mode")
    distinction_preservation_rate: Optional[float] = Field(None, description="Semantic distinction preservation rate (0.0-100.0%) across completed runs in this mode; None if no completed runs")
    task_success_rate: Optional[float] = Field(None, description="Downstream task success rate (0.0-100.0%) across completed runs in this mode; None if no completed runs")
    average_wer: Optional[float] = Field(None, description="Average Word Error Rate across completed runs in this mode; None if no completed runs")
    average_normalized_wer: Optional[float] = Field(None, description="Average cross-script-normalized Word Error Rate across completed runs in this mode; None if no completed runs")
    average_critical_field_accuracy: Optional[float] = Field(None, description="Average critical field accuracy (0.0-100.0%) across completed runs in this mode; None if no completed runs")
    category_breakdown: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Performance metrics by category for this mode over completed runs")


class AggregateMetrics(BaseModel):
    """Summary metrics across evaluation runs with explicit separation between mock demonstration and live evaluation."""
    total_runs: int = Field(0, description="Total number of evaluation run attempts (completed, incomplete, and failed)")
    completed_runs: int = Field(0, description="Total number of successfully completed paired evaluations")
    failed_runs: int = Field(0, description="Total number of operational failed runs (aborted before completing pair)")
    incomplete_runs: int = Field(0, description="Total number of partial live failures (A completed, B failed)")
    mock_runs: int = Field(0, description="Total evaluation attempts executed in deterministic mock mode")
    live_runs: int = Field(0, description="Total evaluation attempts executed against live provider API")

    # Mode-isolated performance metrics
    mock: ModePerformanceMetrics = Field(default_factory=ModePerformanceMetrics, description="Isolated performance metrics for deterministic mock runs")
    live: ModePerformanceMetrics = Field(default_factory=ModePerformanceMetrics, description="Isolated performance metrics for live Gnani Prisma evaluations")

    # Overall / backward-compatible aggregates (completed runs across all modes)
    distinction_preservation_rate: Optional[float] = Field(None, description="Distinction preservation rate (0.0-100.0%) across completed runs; None if no completed runs exist")
    task_success_rate: Optional[float] = Field(None, description="Task success rate (0.0-100.0%) across completed runs; None if no completed runs exist")
    average_wer: Optional[float] = Field(None, description="Average Word Error Rate across completed runs; None if no completed runs exist")
    average_normalized_wer: Optional[float] = Field(None, description="Average cross-script-normalized Word Error Rate across completed runs; None if no completed runs exist")
    average_critical_field_accuracy: Optional[float] = Field(None, description="Average critical field accuracy (0.0-100.0%) across completed runs; None if no completed runs exist")
    category_breakdown: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Performance metrics by category computed over completed runs only")
    failure_category_breakdown: Dict[str, int] = Field(default_factory=dict, description="Taxonomy breakdown of all recorded failure categories")


class EvaluateTranscriptsRequest(BaseModel):
    """Request payload to score custom transcripts against a test case."""
    test_case_id: str
    transcript_a: str
    transcript_b: str
    execution_mode: ExecutionMode = ExecutionMode.MOCK
    audio_condition: AudioCondition = AudioCondition.TEXT_ONLY_SIMULATION
    client_latency_ms: Optional[float] = None
    simulated_latency_ms: Optional[float] = None


class DerivedAssessment(BaseModel):
    """Offline derived evaluation assessment for a historical evaluation run.

    Guarantees:
    - Never mutates or overwrites the source evaluation record.
    - Strictly preserves original provider request IDs, raw WER, latencies, and original outcomes.
    - Separates raw script-sensitive WER from supplementary cross-script-normalized WER.
    - Records explicit offline provenance, documenting 0 provider calls and 0 credits consumed.
    - Is completely isolated from live execution counts in aggregate metrics.
    """
    derived_id: str = Field(..., description="Unique derived assessment identifier (e.g. rescore_eval_live_...)")
    source_run_id: str = Field(..., description="ID of the original immutable source evaluation run")
    test_case_id: str
    category: str
    language_code: str
    provenance: str = Field("offline_deterministic_rescore", description="Audit provenance proving offline execution")
    created_at: str = Field(..., description="Timestamp when the offline re-scoring was performed")
    provider_calls_made: int = Field(0, description="Strictly 0 external network/provider calls made")
    credits_consumed: int = Field(0, description="Strictly 0 programme credits consumed")

    # Original historical evidence (verbatim from source live run)
    original_execution_mode: str
    original_overall_status: str
    original_raw_wer_a: float
    original_raw_wer_b: float
    original_average_raw_wer: float
    original_substitutions_a: int
    original_substitutions_b: int
    original_deletions_a: int
    original_deletions_b: int
    original_insertions_a: int
    original_insertions_b: int
    original_detected_intent_a: str
    original_detected_intent_b: str
    original_task_success: bool
    original_primary_failure_category: str
    original_failure_explanation: str
    request_id_a: Optional[str] = None
    request_id_b: Optional[str] = None
    client_request_elapsed_ms: Optional[float] = None
    latency_a_ms: Optional[float] = None
    latency_b_ms: Optional[float] = None

    # Text payloads
    reference_a: str
    reference_b: str
    actual_transcript_a: str
    actual_transcript_b: str

    # Derived Cross-Script Normalized Metrics
    normalized_wer_a: float
    normalized_wer_b: float
    average_normalized_wer: float
    normalized_substitutions_a: int = 0
    normalized_substitutions_b: int = 0
    normalized_deletions_a: int = 0
    normalized_deletions_b: int = 0
    normalized_insertions_a: int = 0
    normalized_insertions_b: int = 0

    # Corrected Deterministic Intents & Outcomes
    rescore_detected_intent_a: str
    rescore_detected_intent_b: str
    rescore_intent_match_a: bool
    rescore_intent_match_b: bool
    rescore_critical_fields_match_a: bool
    rescore_critical_fields_match_b: bool
    rescore_task_success_a: bool
    rescore_task_success_b: bool
    rescore_semantic_distinction_preserved: bool
    rescore_task_success: bool
    rescore_primary_failure_category: str
    rescore_failure_explanation: str
    rescore_notes: str = Field(
        "Offline deterministic re-scoring applying cross-script vocabulary canonicalization "
        "and Devanagari Hindi intent recognition to historical transcripts without provider calls."
    )
