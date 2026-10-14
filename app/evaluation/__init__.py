"""BoliBench Paired-Contrast Speech-to-Text Evaluation Engine."""

from app.evaluation.engine import EvaluationEngine
from app.evaluation.metrics import calculate_wer, normalize_text, score_pair
from app.evaluation.schemas import (
    AggregateMetrics,
    AudioCondition,
    CodeSwitchingType,
    EvaluateTranscriptsRequest,
    ExecutionMode,
    FailureCategory,
    PairEvaluationResult,
    TestCasePair,
    TestCategory,
    UtteranceEvaluationResult,
    UtteranceSpec,
)
from app.evaluation.storage import StorageManager

__all__ = [
    "EvaluationEngine",
    "StorageManager",
    "calculate_wer",
    "normalize_text",
    "score_pair",
    "TestCasePair",
    "UtteranceSpec",
    "UtteranceEvaluationResult",
    "PairEvaluationResult",
    "AggregateMetrics",
    "EvaluateTranscriptsRequest",
    "TestCategory",
    "CodeSwitchingType",
    "FailureCategory",
    "ExecutionMode",
    "AudioCondition",
]
