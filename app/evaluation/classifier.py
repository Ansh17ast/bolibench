"""Automatic semantic contrast detector for Indian voice AI speech transcripts.

Jointly inspects paired spoken utterances to infer the category of semantic contrast:
- Quantities and numbers
- Negation
- Corrections and changed intent
- Command intent (including return vs modify)
- Named entities (personal names, delivery recipients)
- Time extraction (clock times, delivery windows)
- Unknown or ambiguous (explicit explanation without fallback to default test cases)
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field

from app.evaluation.schemas import TestCategory, FailureCategory
from app.evaluation.dataset import load_seed_dataset
from app.evaluation.metrics import (
    normalize_text,
    extract_quantities,
    extract_negation,
    extract_corrections,
    extract_entities,
    extract_command_intent,
    extract_time,
    classify_failure,
    calculate_wer,
    calculate_cross_script_normalized_wer,
)


class DetectionConfidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    AMBIGUOUS = "ambiguous"


class ContrastInferenceResult(BaseModel):
    """Result of automatic semantic distinction detection on an arbitrary transcript pair."""
    category: str = Field(..., description="Inferred category name or 'unknown' / 'ambiguous'")
    distinction_field: str = Field(..., description="Key structured field carrying the business distinction")
    confidence: DetectionConfidence = DetectionConfidence.HIGH
    explanation: str = Field(..., description="Human-readable explanation of why this category was detected")
    supporting_evidence: Dict[str, Any] = Field(default_factory=dict, description="Audit trail of extracted signals")
    is_curated_reference: bool = Field(False, description="True if pair matches a known benchmark seed case")
    matched_test_case_id: Optional[str] = Field(None, description="Matched benchmark ID if known reference")
    reference_a: Optional[str] = None
    reference_b: Optional[str] = None
    value_a: Any = None
    value_b: Any = None
    semantic_distinction_preserved: bool = False
    task_success: bool = False
    primary_failure_category: str = "none"
    failure_explanation: str = ""
    fields_a: Dict[str, Any] = Field(default_factory=dict)
    fields_b: Dict[str, Any] = Field(default_factory=dict)
    intent_a: Optional[str] = None
    intent_b: Optional[str] = None
    wer_a: Optional[float] = None
    wer_b: Optional[float] = None
    average_wer: Optional[float] = None
    normalized_wer_a: Optional[float] = None
    normalized_wer_b: Optional[float] = None
    average_normalized_wer: Optional[float] = None


def _check_curated_benchmark_match(transcript_a: str, transcript_b: str) -> Optional[Tuple[str, Any]]:
    """Checks whether the spoken transcripts match an existing curated seed pair."""
    seed_cases = load_seed_dataset()
    norm_a = normalize_text(transcript_a)
    norm_b = normalize_text(transcript_b)

    for tc in seed_cases:
        ref_norm_a = normalize_text(tc.utterance_a.reference_transcript)
        ref_norm_b = normalize_text(tc.utterance_b.reference_transcript)

        # Cross-script check: either direct match or normalized WER == 0.0
        wer_a, _, _, _, _ = calculate_cross_script_normalized_wer(tc.utterance_a.reference_transcript, transcript_a)
        wer_b, _, _, _, _ = calculate_cross_script_normalized_wer(tc.utterance_b.reference_transcript, transcript_b)

        if (norm_a == ref_norm_a and norm_b == ref_norm_b) or (wer_a == 0.0 and wer_b == 0.0):
            return tc.id, tc

    return None


def detect_semantic_contrast(
    transcript_a: str,
    transcript_b: str,
    language_code: str = "hi-IN",
) -> ContrastInferenceResult:
    """Infers the semantic contrast between two spoken transcripts automatically.

    Guarantees:
    - Never falls back to 'pair_qty_001' or assumes quantities if not supported by speech signals.
    - Analyzes both utterances jointly.
    - Distinguishes time (09:00 vs 10:00) from purchase quantities.
    - Distinguishes negation (cancel vs keep order).
    - Resolves corrections to final targets.
    - Distinguishes return vs modify intents.
    - If ambiguous or unknown, returns explicit 'unknown' category with clear explanation.
    """
    raw_a = (transcript_a or "").strip()
    raw_b = (transcript_b or "").strip()

    if not raw_a or not raw_b:
        return ContrastInferenceResult(
            category="ambiguous",
            distinction_field="none",
            confidence=DetectionConfidence.AMBIGUOUS,
            explanation="One or both transcriptions were empty or silence was detected.",
            supporting_evidence={"transcript_a_empty": not bool(raw_a), "transcript_b_empty": not bool(raw_b)},
            semantic_distinction_preserved=False,
            task_success=False,
            primary_failure_category=FailureCategory.EXTRACTION_FAILURE.value,
            failure_explanation="Empty transcript received; unable to infer semantic contrast.",
        )

    # 1. First, check if this pair matches an existing curated benchmark seed case
    curated_match = _check_curated_benchmark_match(raw_a, raw_b)
    matched_id = curated_match[0] if curated_match else None
    matched_tc = curated_match[1] if curated_match else None

    # Compute candidate extractions for both utterances
    times_a = extract_time(raw_a)
    times_b = extract_time(raw_b)

    neg_a = extract_negation(raw_a)
    neg_b = extract_negation(raw_b)

    corr_a = extract_corrections(raw_a)
    corr_b = extract_corrections(raw_b)

    intent_a = extract_command_intent(raw_a)
    intent_b = extract_command_intent(raw_b)

    entities_a = extract_entities(raw_a)
    entities_b = extract_entities(raw_b)

    qty_a = extract_quantities(raw_a)
    qty_b = extract_quantities(raw_b)

    norm_a = normalize_text(raw_a)
    norm_b = normalize_text(raw_b)

    # -------------------------------------------------------------------------
    # RULE 1: Time Extraction
    # Detected if clock keywords ('बजे', 'baje', 'am', 'pm', 'सुबह', 'शाम') or HH:MM format present
    # -------------------------------------------------------------------------
    has_time_cue = any(k in norm_a or k in norm_b for k in ("बजे", "baje", "time", "सुबह", "subah", "शाम", "shaam", "am", "pm", "clock"))
    if has_time_cue or (times_a.get("delivery_time") and times_b.get("delivery_time")):
        time_val_a = times_a.get("delivery_time")
        time_val_b = times_b.get("delivery_time")

        if time_val_a or time_val_b:
            preserved = bool(time_val_a and time_val_b and time_val_a != time_val_b)
            task_succ = preserved

            fail_cat = "none"
            fail_expl = ""
            if not preserved:
                fail_cat = FailureCategory.TIME_ERROR.value
                fail_expl = f"Time distinction collapsed (Extracted: {time_val_a} vs {time_val_b}). Downstream scheduling agent would book incorrect slot."

            res = ContrastInferenceResult(
                category=TestCategory.TIME_EXTRACTION.value,
                distinction_field="delivery_time",
                confidence=DetectionConfidence.HIGH,
                explanation=f"Detected time-slot scheduling contrast based on clock and period expressions ({time_val_a} vs {time_val_b}).",
                supporting_evidence={"time_a": times_a, "time_b": times_b, "cues_found": has_time_cue},
                is_curated_reference=bool(matched_tc),
                matched_test_case_id=matched_id,
                value_a=time_val_a,
                value_b=time_val_b,
                semantic_distinction_preserved=preserved,
                task_success=task_succ,
                primary_failure_category=fail_cat,
                failure_explanation=fail_expl,
                fields_a=times_a,
                fields_b=times_b,
                intent_a=intent_a,
                intent_b=intent_b,
            )
            _populate_wer_if_reference(res, raw_a, raw_b, matched_tc, language_code)
            return res

    # -------------------------------------------------------------------------
    # RULE 2: In-Flight Corrections & Mind Changes
    # Detected if retraction cues ('नहीं', 'nahi', 'no wait', 'not two, three', 'sorry', 'actually') precede a secondary target
    # -------------------------------------------------------------------------
    has_correction_cue = bool(corr_a.get("correction_detected") or corr_b.get("correction_detected"))
    if has_correction_cue:
        target_a = corr_a.get("final_target") or qty_a.get("quantity")
        target_b = corr_b.get("final_target") or qty_b.get("quantity")

        preserved = bool(target_a is not None and target_b is not None and target_a != target_b)
        task_succ = preserved

        fail_cat = "none"
        fail_expl = ""
        if not preserved:
            fail_cat = FailureCategory.CORRECTION_ERROR.value
            fail_expl = f"Agent failed to resolve change-of-mind correction (Final target: {target_a} vs {target_b}). Downstream agent would execute rejected value."

        res = ContrastInferenceResult(
            category=TestCategory.CORRECTIONS_AND_MIND_CHANGES.value,
            distinction_field="final_target",
            confidence=DetectionConfidence.HIGH,
            explanation=f"Detected change-of-mind correction updating target from {corr_b.get('initial_value') or target_a} to {target_b}.",
            supporting_evidence={"correction_a": corr_a, "correction_b": corr_b},
            is_curated_reference=bool(matched_tc),
            matched_test_case_id=matched_id,
            value_a=target_a,
            value_b=target_b,
            semantic_distinction_preserved=preserved,
            task_success=task_succ,
            primary_failure_category=fail_cat,
            failure_explanation=fail_expl,
            fields_a=corr_a,
            fields_b=corr_b,
            intent_a=intent_a,
            intent_b=intent_b,
        )
        _populate_wer_if_reference(res, raw_a, raw_b, matched_tc, language_code)
        return res

    # -------------------------------------------------------------------------
    # RULE 3: Negation Contrast
    # Detected if explicit cancellation or negation markers differ between utterances
    # -------------------------------------------------------------------------
    neg_diff = (neg_a.get("is_negated") != neg_b.get("is_negated"))
    has_neg_cue = any(k in norm_a.split() or k in norm_b.split() for k in ("मत", "mat", "नहीं", "nahi", "nahin", "not", "dont", "don't", "no", "never"))
    has_cancel_cue = any(k in norm_a or k in norm_b for k in ("कैंसल", "cancel", "रद्द", "delete", "hatao", "band", "close"))

    if (has_neg_cue and has_cancel_cue) or (neg_diff and (has_neg_cue or neg_a.get("action") != neg_b.get("action"))):
        val_a = neg_a.get("is_negated", False)
        val_b = neg_b.get("is_negated", False)
        preserved = (val_a != val_b)
        task_succ = preserved

        fail_cat = "none"
        fail_expl = ""
        if not preserved:
            fail_cat = FailureCategory.MISSED_NEGATION.value
            fail_expl = f"Voice agent missed critical negation contrast (Extracted: is_negated={val_a} vs {val_b}). Downstream agent would execute irreversible action."

        res = ContrastInferenceResult(
            category=TestCategory.NEGATION.value,
            distinction_field="is_negated",
            confidence=DetectionConfidence.HIGH,
            explanation=f"Detected negation contrast ({'negated' if val_a else 'affirmative'} vs {'negated' if val_b else 'affirmative'}).",
            supporting_evidence={"negation_a": neg_a, "negation_b": neg_b},
            is_curated_reference=bool(matched_tc),
            matched_test_case_id=matched_id,
            value_a=val_a,
            value_b=val_b,
            semantic_distinction_preserved=preserved,
            task_success=task_succ,
            primary_failure_category=fail_cat,
            failure_explanation=fail_expl,
            fields_a=neg_a,
            fields_b=neg_b,
            intent_a=intent_a,
            intent_b=intent_b,
        )
        _populate_wer_if_reference(res, raw_a, raw_b, matched_tc, language_code)
        return res


    # -------------------------------------------------------------------------
    # RULE 4: Command Intent Disambiguation (e.g. Return vs Modify Order, Pay vs Request)
    # -------------------------------------------------------------------------
    if intent_a != intent_b and intent_a != "UNKNOWN_INTENT" and intent_b != "UNKNOWN_INTENT":
        # Clear distinction between business command handlers
        preserved = True
        task_succ = True

        res = ContrastInferenceResult(
            category=TestCategory.COMMAND_INTENT.value,
            distinction_field="intent",
            confidence=DetectionConfidence.HIGH,
            explanation=f"Detected diverging business command intents ({intent_a} vs {intent_b}).",
            supporting_evidence={"intent_a": intent_a, "intent_b": intent_b},
            is_curated_reference=bool(matched_tc),
            matched_test_case_id=matched_id,
            value_a=intent_a,
            value_b=intent_b,
            semantic_distinction_preserved=preserved,
            task_success=task_succ,
            primary_failure_category="none",
            failure_explanation="",
            fields_a={"intent": intent_a},
            fields_b={"intent": intent_b},
            intent_a=intent_a,
            intent_b=intent_b,
        )
        _populate_wer_if_reference(res, raw_a, raw_b, matched_tc, language_code)
        return res

    # -------------------------------------------------------------------------
    # RULE 5: Named Entities (Recipient / Person Name)
    # -------------------------------------------------------------------------
    person_a = entities_a.get("person_name")
    person_b = entities_b.get("person_name")
    if person_a or person_b:
        preserved = bool(person_a and person_b and person_a != person_b)
        task_succ = preserved

        fail_cat = "none"
        fail_expl = ""
        if not preserved:
            fail_cat = FailureCategory.ENTITY_SUBSTITUTION.value
            fail_expl = f"Entity substitution error (Recognized: {person_a} vs {person_b}). Downstream voice bot routed to wrong person."

        res = ContrastInferenceResult(
            category=TestCategory.NAMED_ENTITIES.value,
            distinction_field="person_name",
            confidence=DetectionConfidence.HIGH,
            explanation=f"Detected recipient personal name distinction ({person_a} vs {person_b}).",
            supporting_evidence={"entities_a": entities_a, "entities_b": entities_b},
            is_curated_reference=bool(matched_tc),
            matched_test_case_id=matched_id,
            value_a=person_a,
            value_b=person_b,
            semantic_distinction_preserved=preserved,
            task_success=task_succ,
            primary_failure_category=fail_cat,
            failure_explanation=fail_expl,
            fields_a=entities_a,
            fields_b=entities_b,
            intent_a=intent_a,
            intent_b=intent_b,
        )
        _populate_wer_if_reference(res, raw_a, raw_b, matched_tc, language_code)
        return res

    # -------------------------------------------------------------------------
    # RULE 6: Quantities & Numbers
    # -------------------------------------------------------------------------
    val_qty_a = qty_a.get("quantity")
    val_qty_b = qty_b.get("quantity")
    if val_qty_a is not None or val_qty_b is not None:
        preserved = bool(val_qty_a is not None and val_qty_b is not None and val_qty_a != val_qty_b)
        task_succ = preserved

        fail_cat = "none"
        fail_expl = ""
        if not preserved:
            fail_cat = FailureCategory.QUANTITY_ERROR.value
            fail_expl = f"Numeric distinction collapsed (Extracted: {val_qty_a} vs {val_qty_b}). Downstream agent would execute wrong amount."

        res = ContrastInferenceResult(
            category=TestCategory.QUANTITIES_AND_NUMBERS.value,
            distinction_field="quantity",
            confidence=DetectionConfidence.HIGH,
            explanation=f"Detected numeric quantity contrast ({val_qty_a} vs {val_qty_b}).",
            supporting_evidence={"quantity_a": qty_a, "quantity_b": qty_b},
            is_curated_reference=bool(matched_tc),
            matched_test_case_id=matched_id,
            value_a=val_qty_a,
            value_b=val_qty_b,
            semantic_distinction_preserved=preserved,
            task_success=task_succ,
            primary_failure_category=fail_cat,
            failure_explanation=fail_expl,
            fields_a=qty_a,
            fields_b=qty_b,
            intent_a=intent_a,
            intent_b=intent_b,
        )
        _populate_wer_if_reference(res, raw_a, raw_b, matched_tc, language_code)
        return res

    # -------------------------------------------------------------------------
    # RULE 7: Ambiguous or Unsupported Contrast
    # Never fall back silently to 'pair_qty_001'!
    # -------------------------------------------------------------------------
    return ContrastInferenceResult(
        category="ambiguous",
        distinction_field="unknown",
        confidence=DetectionConfidence.AMBIGUOUS,
        explanation=(
            "Could not automatically infer a distinct semantic contrast (numbers, negation, "
            "correction, intent, named entities, or time) between these two utterances. "
            "Downstream decision rules require a recognized contrast dimension."
        ),
        supporting_evidence={
            "transcript_a": raw_a,
            "transcript_b": raw_b,
            "detected_intent_a": intent_a,
            "detected_intent_b": intent_b,
        },
        is_curated_reference=False,
        semantic_distinction_preserved=False,
        task_success=False,
        primary_failure_category=FailureCategory.EXTRACTION_FAILURE.value,
        failure_explanation="No supported semantic contrast detected across critical business dimensions.",
        fields_a={},
        fields_b={},
        intent_a=intent_a,
        intent_b=intent_b,
    )


def _populate_wer_if_reference(
    res: ContrastInferenceResult,
    raw_a: str,
    raw_b: str,
    matched_tc: Any,
    language_code: str,
) -> None:
    """Populates benchmark reference WER only when transcripts match a known benchmark seed case."""
    if matched_tc:
        ref_a = matched_tc.utterance_a.reference_transcript
        ref_b = matched_tc.utterance_b.reference_transcript
        res.reference_a = ref_a
        res.reference_b = ref_b

        wer_a, _, _, _, _ = calculate_wer(ref_a, raw_a, language_code)
        wer_b, _, _, _, _ = calculate_wer(ref_b, raw_b, language_code)
        norm_a, _, _, _, _ = calculate_cross_script_normalized_wer(ref_a, raw_a, language_code)
        norm_b, _, _, _, _ = calculate_cross_script_normalized_wer(ref_b, raw_b, language_code)

        res.wer_a = wer_a
        res.wer_b = wer_b
        res.average_wer = round((wer_a + wer_b) / 2.0, 4)
        res.normalized_wer_a = norm_a
        res.normalized_wer_b = norm_b
        res.average_normalized_wer = round((norm_a + norm_b) / 2.0, 4)
    else:
        # For arbitrary unbenchmarked uploads, ground-truth reference transcripts are genuinely unavailable
        res.reference_a = None
        res.reference_b = None
        res.wer_a = None
        res.wer_b = None
        res.average_wer = None
        res.normalized_wer_a = None
        res.normalized_wer_b = None
        res.average_normalized_wer = None
