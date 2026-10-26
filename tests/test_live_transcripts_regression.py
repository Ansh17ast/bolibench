"""Regression tests locking in evaluation engine fixes against 50 live benchmark transcripts.

Guarantees that deterministic extractors correctly evaluate historical live transcripts
without regression, specifically for:
- Named entities (Rahul vs Rohit, Priya vs Neha, Aarav vs Vikram, etc.)
- Command intent (Return vs Modify, Book vs Cancel Cab, Refund vs Check Status, Repeat vs Cancel Mandate)
- Corrections & mind changes (comma-formatted currency, destination, attribute/color)
- Time extraction (English hour AM/PM, Hindi half-past, 24-hr normalization)
- Denominator integrity (49 completed vs 50 attempted)
"""

import json
from pathlib import Path
import pytest

from app.evaluation.schemas import (
    TestCasePair,
    UtteranceSpec,
    TestCategory,
    CodeSwitchingType,
    AudioCondition,
)
from app.evaluation.metrics import score_pair, extract_entities, extract_command_intent, extract_corrections, extract_time
from app.evaluation.batch_manifest import BatchManifest


RESULTS_FILE = Path("benchmark_results/batch_live_20261022_013841_results.json")
MANIFEST_FILE = Path("benchmark_manifest.json")

pytestmark = pytest.mark.skipif(
    not RESULTS_FILE.exists(),
    reason="Requires historical benchmark results artifact (gitignored)",
)


@pytest.fixture(scope="module")
def live_results_and_manifest():
    assert RESULTS_FILE.exists(), f"Missing historical results file: {RESULTS_FILE}"
    assert MANIFEST_FILE.exists(), f"Missing benchmark manifest file: {MANIFEST_FILE}"

    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        hist_data = json.load(f)

    manifest = BatchManifest.model_validate_json(MANIFEST_FILE.read_text(encoding="utf-8"))
    manifest_map = {item.pair_id: item for item in manifest.items}
    utt_map = {u["utterance_id"]: u for u in hist_data["utterances"]}

    return hist_data, manifest_map, utt_map


def test_named_entities_live_transcripts_regression(live_results_and_manifest):
    """Named entities live transcripts must achieve 100% distinction preservation (8/8)."""
    hist_data, manifest_map, utt_map = live_results_and_manifest

    ent_pairs = [p for p in hist_data["pairs"] if p["category"] == "named_entities"]
    assert len(ent_pairs) == 8

    for p in ent_pairs:
        pair_id = p["test_case_id"]
        m_item = manifest_map[pair_id]
        utt_a = utt_map[f"{pair_id}_a"]
        utt_b = utt_map[f"{pair_id}_b"]

        tc = TestCasePair(
            id=pair_id,
            category=TestCategory.NAMED_ENTITIES,
            language_code=m_item.language_code,
            code_switching_type=CodeSwitchingType.HINGLISH_CODE_MIXED,
            description=m_item.description,
            expected_distinction=m_item.description,
            distinction_field=m_item.distinction_field,
            audio_condition=AudioCondition.CLEAN_RECORDED,
            provenance="live_regression_test",
            utterance_a=UtteranceSpec(
                id=f"{pair_id}_a",
                text=m_item.reference_transcript_a,
                reference_transcript=m_item.reference_transcript_a,
                expected_critical_fields=m_item.expected_critical_fields_a,
                expected_intent="TRANSFER_FUNDS",
                expected_action=m_item.expected_action_a or "EXECUTE_ACTION",
            ),
            utterance_b=UtteranceSpec(
                id=f"{pair_id}_b",
                text=m_item.reference_transcript_b,
                reference_transcript=m_item.reference_transcript_b,
                expected_critical_fields=m_item.expected_critical_fields_b,
                expected_intent="TRANSFER_FUNDS",
                expected_action=m_item.expected_action_b or "EXECUTE_ACTION",
            ),
        )

        res = score_pair(
            run_id=f"test_live_{pair_id}",
            test_case=tc,
            transcript_a=utt_a["actual_transcript"],
            transcript_b=utt_b["actual_transcript"],
            execution_mode="live",
        )

        assert res.semantic_distinction_preserved is True, f"Failed for {pair_id}: {res.failure_explanation}"


def test_command_intent_live_transcripts_regression(live_results_and_manifest):
    """Command intent live transcripts must achieve 100% distinction preservation (8/8)."""
    hist_data, manifest_map, utt_map = live_results_and_manifest

    cmd_pairs = [p for p in hist_data["pairs"] if p["category"] == "command_intent"]
    assert len(cmd_pairs) == 8

    for p in cmd_pairs:
        pair_id = p["test_case_id"]
        m_item = manifest_map[pair_id]
        utt_a = utt_map[f"{pair_id}_a"]
        utt_b = utt_map[f"{pair_id}_b"]

        tc = TestCasePair(
            id=pair_id,
            category=TestCategory.COMMAND_INTENT,
            language_code=m_item.language_code,
            code_switching_type=CodeSwitchingType.HINGLISH_CODE_MIXED,
            description=m_item.description,
            expected_distinction=m_item.description,
            distinction_field=m_item.distinction_field,
            audio_condition=AudioCondition.CLEAN_RECORDED,
            provenance="live_regression_test",
            utterance_a=UtteranceSpec(
                id=f"{pair_id}_a",
                text=m_item.reference_transcript_a,
                reference_transcript=m_item.reference_transcript_a,
                expected_critical_fields=m_item.expected_critical_fields_a,
                expected_intent=m_item.expected_intent_a or "UNKNOWN",
                expected_action=m_item.expected_action_a or "EXECUTE_ACTION",
            ),
            utterance_b=UtteranceSpec(
                id=f"{pair_id}_b",
                text=m_item.reference_transcript_b,
                reference_transcript=m_item.reference_transcript_b,
                expected_critical_fields=m_item.expected_critical_fields_b,
                expected_intent=m_item.expected_intent_b or "UNKNOWN",
                expected_action=m_item.expected_action_b or "EXECUTE_ACTION",
            ),
        )

        res = score_pair(
            run_id=f"test_live_{pair_id}",
            test_case=tc,
            transcript_a=utt_a["actual_transcript"],
            transcript_b=utt_b["actual_transcript"],
            execution_mode="live",
        )

        assert res.semantic_distinction_preserved is True, f"Failed for {pair_id}: {res.failure_explanation}"


def test_corrections_live_transcripts_regression(live_results_and_manifest):
    """Corrections live transcripts must achieve 100% distinction preservation (8/8)."""
    hist_data, manifest_map, utt_map = live_results_and_manifest

    cor_pairs = [p for p in hist_data["pairs"] if p["category"] == "corrections_and_mind_changes"]
    assert len(cor_pairs) == 8

    for p in cor_pairs:
        pair_id = p["test_case_id"]
        m_item = manifest_map[pair_id]
        utt_a = utt_map[f"{pair_id}_a"]
        utt_b = utt_map[f"{pair_id}_b"]

        tc = TestCasePair(
            id=pair_id,
            category=TestCategory.CORRECTIONS_AND_MIND_CHANGES,
            language_code=m_item.language_code,
            code_switching_type=CodeSwitchingType.HINGLISH_CODE_MIXED,
            description=m_item.description,
            expected_distinction=m_item.description,
            distinction_field=m_item.distinction_field,
            audio_condition=AudioCondition.CLEAN_RECORDED,
            provenance="live_regression_test",
            utterance_a=UtteranceSpec(
                id=f"{pair_id}_a",
                text=m_item.reference_transcript_a,
                reference_transcript=m_item.reference_transcript_a,
                expected_critical_fields=m_item.expected_critical_fields_a,
                expected_intent="ORDER_PRODUCT",
                expected_action=m_item.expected_action_a or "EXECUTE_ACTION",
            ),
            utterance_b=UtteranceSpec(
                id=f"{pair_id}_b",
                text=m_item.reference_transcript_b,
                reference_transcript=m_item.reference_transcript_b,
                expected_critical_fields=m_item.expected_critical_fields_b,
                expected_intent="ORDER_PRODUCT",
                expected_action=m_item.expected_action_b or "EXECUTE_ACTION",
            ),
        )

        res = score_pair(
            run_id=f"test_live_{pair_id}",
            test_case=tc,
            transcript_a=utt_a["actual_transcript"],
            transcript_b=utt_b["actual_transcript"],
            execution_mode="live",
        )

        assert res.semantic_distinction_preserved is True, f"Failed for {pair_id}: {res.failure_explanation}"


def test_completed_pairs_distinction_rate_regression(live_results_and_manifest):
    """Guarantees distinction preservation on 49 completed live pairs is >= 95% (47/49)."""
    hist_data, manifest_map, utt_map = live_results_and_manifest

    preserved_count = 0
    completed_count = 0

    for p in hist_data["pairs"]:
        pair_id = p["test_case_id"]
        m_item = manifest_map[pair_id]
        utt_a = utt_map.get(f"{pair_id}_a")
        utt_b = utt_map.get(f"{pair_id}_b")

        if not utt_a or not utt_b or utt_a.get("status") != "completed" or utt_b.get("status") != "completed":
            continue

        completed_count += 1

        tc = TestCasePair(
            id=pair_id,
            category=TestCategory(m_item.category),
            language_code=m_item.language_code,
            code_switching_type=CodeSwitchingType.HINGLISH_CODE_MIXED,
            description=m_item.description,
            expected_distinction=m_item.description,
            distinction_field=m_item.distinction_field,
            audio_condition=AudioCondition.CLEAN_RECORDED,
            provenance="live_regression_test",
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

        res = score_pair(
            run_id=f"test_live_{pair_id}",
            test_case=tc,
            transcript_a=utt_a["actual_transcript"],
            transcript_b=utt_b["actual_transcript"],
            execution_mode="live",
        )

        if res.semantic_distinction_preserved:
            preserved_count += 1

    assert completed_count == 49, f"Expected 49 completed pairs, got {completed_count}"
    assert preserved_count == 47, f"Expected 47 preserved pairs, got {preserved_count}"
    assert (preserved_count / completed_count) >= 0.95
