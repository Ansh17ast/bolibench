"""Computes derived re-evaluation of 50-pair live benchmark transcripts offline.

Strictly preserves original provider transcripts, request IDs, and historical results.
Applies updated deterministic extractors to produce a separate derived assessment.
"""

import json
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Any, Dict, List
from app.evaluation.batch_manifest import BatchManifest
from app.evaluation.schemas import (
    TestCasePair,
    UtteranceSpec,
    TestCategory,
    CodeSwitchingType,
    AudioCondition,
    FailureCategory,
)
from app.evaluation.metrics import (
    evaluate_utterance,
    score_pair,
    normalize_intent_alias,
)


def run_derived_reevaluation(
    results_path: str = "benchmark_results/batch_live_20261022_013841_results.json",
    manifest_path: str = "benchmark_manifest.json",
    output_path: str = "benchmark_results/batch_live_20261022_013841_derived_reevaluation.json",
) -> Dict[str, Any]:
    with open(results_path, "r", encoding="utf-8") as f:
        hist_data = json.load(f)

    manifest = BatchManifest.model_validate_json(Path(manifest_path).read_text(encoding="utf-8"))
    manifest_map = {item.pair_id: item for item in manifest.items}

    # Map utterances by utterance_id
    utt_map = {u["utterance_id"]: u for u in hist_data["utterances"]}

    derived_pairs: List[Dict[str, Any]] = []

    cat_stats: Dict[str, Dict[str, Any]] = {}
    for cat in TestCategory:
        cat_stats[cat.value] = {
            "total_pairs": 0,
            "completed_pairs": 0,
            "failed_pairs": 0,
            "original_preserved": 0,
            "derived_preserved": 0,
            "original_task_success": 0,
            "derived_task_success": 0,
            "total_critical_fields": 0,
            "matched_critical_fields": 0,
            "raw_wers": [],
            "norm_wers": [],
        }

    for p in hist_data["pairs"]:
        pair_id = p["test_case_id"]
        m_item = manifest_map.get(pair_id)
        if not m_item:
            continue

        cat_val = m_item.category.value if hasattr(m_item.category, "value") else str(m_item.category)
        cat_stats[cat_val]["total_pairs"] += 1

        utt_a = utt_map.get(f"{pair_id}_a")
        utt_b = utt_map.get(f"{pair_id}_b")

        is_completed = (
            utt_a is not None
            and utt_b is not None
            and utt_a.get("status") == "completed"
            and utt_b.get("status") == "completed"
            and utt_a.get("actual_transcript") is not None
            and utt_b.get("actual_transcript") is not None
        )

        orig_preserved = p.get("distinction_preserved", False)
        orig_task_success = p.get("task_success", False)
        if orig_preserved:
            cat_stats[cat_val]["original_preserved"] += 1
        if orig_task_success:
            cat_stats[cat_val]["original_task_success"] += 1

        if not is_completed:
            cat_stats[cat_val]["failed_pairs"] += 1
            derived_pairs.append({
                "pair_id": pair_id,
                "category": cat_val,
                "status": "incomplete",
                "failure_reason": f"Provider transcription failed on utterance(s): {utt_a.get('error') if utt_a else 'missing'} / {utt_b.get('error') if utt_b else 'missing'}",
                "original_preserved": orig_preserved,
                "derived_preserved": False,
                "derived_task_success": False,
                "critical_field_accuracy": 0.0,
                "primary_failure_category": FailureCategory.PROVIDER_ERROR.value,
                "request_id_a": utt_a.get("request_id") if utt_a else None,
                "request_id_b": utt_b.get("request_id") if utt_b else None,
            })
            continue

        cat_stats[cat_val]["completed_pairs"] += 1

        # Build TestCasePair object for scoring
        tc = TestCasePair(
            id=pair_id,
            category=TestCategory(cat_val),
            language_code=m_item.language_code,
            code_switching_type=CodeSwitchingType.HINGLISH_CODE_MIXED,
            description=m_item.description,
            expected_distinction=m_item.description,
            distinction_field=m_item.distinction_field,
            audio_condition=AudioCondition.CLEAN_RECORDED if m_item.audio_condition not in [c.value for c in AudioCondition] else AudioCondition(m_item.audio_condition),
            provenance=m_item.source_type.value if hasattr(m_item.source_type, "value") else str(m_item.source_type),
            utterance_a=UtteranceSpec(
                id=f"{pair_id}_a",
                text=m_item.reference_transcript_a,
                reference_transcript=m_item.reference_transcript_a,
                expected_critical_fields=m_item.expected_critical_fields_a or {},
                expected_intent=m_item.expected_intent_a or "UNKNOWN",
                expected_action=m_item.expected_action_a or "NONE",
            ),
            utterance_b=UtteranceSpec(
                id=f"{pair_id}_b",
                text=m_item.reference_transcript_b,
                reference_transcript=m_item.reference_transcript_b,
                expected_critical_fields=m_item.expected_critical_fields_b or {},
                expected_intent=m_item.expected_intent_b or "UNKNOWN",
                expected_action=m_item.expected_action_b or "NONE",
            ),
        )

        pair_score = score_pair(
            run_id=p.get("run_id", f"derived_{pair_id}"),
            test_case=tc,
            transcript_a=utt_a["actual_transcript"],
            transcript_b=utt_b["actual_transcript"],
            execution_mode="live",
            request_id_a=utt_a.get("request_id"),
            request_id_b=utt_b.get("request_id"),
            client_elapsed_a_ms=utt_a.get("client_elapsed_ms"),
            client_elapsed_b_ms=utt_b.get("client_elapsed_ms"),
        )

        if pair_score.semantic_distinction_preserved:
            cat_stats[cat_val]["derived_preserved"] += 1
        if pair_score.task_success:
            cat_stats[cat_val]["derived_task_success"] += 1

        cat_stats[cat_val]["raw_wers"].append(pair_score.average_wer)
        cat_stats[cat_val]["norm_wers"].append(pair_score.average_normalized_wer)

        num_fields = len(tc.utterance_a.expected_critical_fields) + len(tc.utterance_b.expected_critical_fields)
        matched_fields = sum(1 for m in pair_score.utterance_a.field_matches.values() if m) + sum(1 for m in pair_score.utterance_b.field_matches.values() if m)
        cat_stats[cat_val]["total_critical_fields"] += num_fields
        cat_stats[cat_val]["matched_critical_fields"] += matched_fields

        derived_pairs.append({
            "pair_id": pair_id,
            "category": cat_val,
            "status": "completed",
            "transcript_a": utt_a["actual_transcript"],
            "transcript_b": utt_b["actual_transcript"],
            "reference_a": m_item.reference_transcript_a,
            "reference_b": m_item.reference_transcript_b,
            "fields_a": pair_score.utterance_a.extracted_fields,
            "fields_b": pair_score.utterance_b.extracted_fields,
            "original_preserved": orig_preserved,
            "derived_preserved": pair_score.semantic_distinction_preserved,
            "original_task_success": orig_task_success,
            "derived_task_success": pair_score.task_success,
            "critical_field_accuracy": pair_score.critical_field_accuracy,
            "raw_wer_a": pair_score.utterance_a.wer,
            "raw_wer_b": pair_score.utterance_b.wer,
            "average_wer": pair_score.average_wer,
            "normalized_wer_a": pair_score.utterance_a.normalized_wer,
            "normalized_wer_b": pair_score.utterance_b.normalized_wer,
            "average_normalized_wer": pair_score.average_normalized_wer,
            "primary_failure_category": pair_score.primary_failure_category.value,
            "failure_explanation": pair_score.failure_explanation,
            "request_id_a": utt_a.get("request_id"),
            "request_id_b": utt_b.get("request_id"),
            "client_elapsed_ms_a": utt_a.get("client_elapsed_ms"),
            "client_elapsed_ms_b": utt_b.get("client_elapsed_ms"),
        })

    # Summary denominators and rates
    total_pairs = len(hist_data["pairs"])
    completed_pairs = sum(c["completed_pairs"] for c in cat_stats.values())
    orig_total_preserved = sum(c["original_preserved"] for c in cat_stats.values())
    derived_total_preserved = sum(c["derived_preserved"] for c in cat_stats.values())
    derived_total_task_success = sum(c["derived_task_success"] for c in cat_stats.values())

    total_fields = sum(c["total_critical_fields"] for c in cat_stats.values())
    matched_fields = sum(c["matched_critical_fields"] for c in cat_stats.values())
    overall_cfa = (matched_fields / total_fields) if total_fields > 0 else 0.0

    all_raw_wers = [w for c in cat_stats.values() for w in c["raw_wers"]]
    all_norm_wers = [w for c in cat_stats.values() for w in c["norm_wers"]]
    mean_raw_wer = sum(all_raw_wers) / len(all_raw_wers) if all_raw_wers else 0.0
    mean_norm_wer = sum(all_norm_wers) / len(all_norm_wers) if all_norm_wers else 0.0

    summary = {
        "assessment_type": "derived_offline_reevaluation",
        "original_batch_id": hist_data["batch_id"],
        "source_results_file": results_path,
        "manifest_version": manifest.manifest_version,
        "total_pairs_attempted": total_pairs,
        "completed_pairs": completed_pairs,
        "incomplete_pairs": total_pairs - completed_pairs,
        "original_distinction_preserved_count": orig_total_preserved,
        "original_distinction_preserved_rate_completed": orig_total_preserved / completed_pairs if completed_pairs else 0.0,
        "original_distinction_preserved_rate_total": orig_total_preserved / total_pairs,
        "derived_distinction_preserved_count": derived_total_preserved,
        "derived_distinction_preserved_rate_completed": derived_total_preserved / completed_pairs if completed_pairs else 0.0,
        "derived_distinction_preserved_rate_total": derived_total_preserved / total_pairs,
        "derived_task_success_count": derived_total_task_success,
        "derived_task_success_rate_completed": derived_total_task_success / completed_pairs if completed_pairs else 0.0,
        "derived_task_success_rate_total": derived_total_task_success / total_pairs,
        "critical_field_accuracy": round(overall_cfa, 4),
        "mean_raw_wer": round(mean_raw_wer, 4),
        "mean_normalized_wer": round(mean_norm_wer, 4),
        "categories": {
            cat_name: {
                "total_pairs": c["total_pairs"],
                "completed_pairs": c["completed_pairs"],
                "failed_pairs": c["failed_pairs"],
                "original_preserved": c["original_preserved"],
                "derived_preserved": c["derived_preserved"],
                "derived_preservation_rate_completed": (
                    c["derived_preserved"] / c["completed_pairs"] if c["completed_pairs"] > 0 else 0.0
                ),
                "derived_preservation_rate_total": c["derived_preserved"] / c["total_pairs"] if c["total_pairs"] > 0 else 0.0,
                "critical_field_accuracy": (
                    round(c["matched_critical_fields"] / c["total_critical_fields"], 4)
                    if c["total_critical_fields"] > 0
                    else 0.0
                ),
                "mean_raw_wer": round(sum(c["raw_wers"]) / len(c["raw_wers"]), 4) if c["raw_wers"] else 0.0,
                "mean_normalized_wer": round(sum(c["norm_wers"]) / len(c["norm_wers"]), 4) if c["norm_wers"] else 0.0,
            }
            for cat_name, c in cat_stats.items()
            if c["total_pairs"] > 0
        },
        "pairs": derived_pairs,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


if __name__ == "__main__":
    res = run_derived_reevaluation()
    print("Re-evaluation complete:")
    print(f"Total: {res['total_pairs_attempted']}, Completed: {res['completed_pairs']}")
    print(f"Original Preserved: {res['original_distinction_preserved_count']} ({res['original_distinction_preserved_rate_completed']:.1%})")
    print(f"Derived Preserved: {res['derived_distinction_preserved_count']} ({res['derived_distinction_preserved_rate_completed']:.1%})")
    print(f"Critical Field Accuracy: {res['critical_field_accuracy']:.1%}")
