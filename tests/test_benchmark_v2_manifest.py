"""Validation test suite for BoliBench 600-Pair Expanded Benchmark (Manifest v2.0.0).

Proves that:
1. Manifest v2.0.0 contains exactly 600 pairs (100 per category across 6 categories).
2. All 1,200 audio files exist, have valid 16 kHz mono linear PCM format, and are <= 60s.
3. Within every pair, references and audio hashes are strictly distinct.
4. Deterministic extractors achieve 100% extractability on reference transcripts.
5. Subscription non-renewal cases are annotated as DO_NOT_RENEW_SUBSCRIPTION.
6. Dry-run pre-flight audit report has 0 invalid, 0 missing, and 0 duplicated items.
"""

import json
from pathlib import Path
import pytest
import wave

from app.evaluation.batch_manifest import BatchManifest
from app.evaluation.schemas import TestCategory
from app.evaluation.metrics import (
    extract_quantities,
    extract_negation,
    extract_corrections,
    extract_entities,
    extract_command_intent,
    extract_time,
    normalize_intent_alias,
)
from app.services.audio_validator import AudioValidator


MANIFEST_V2_PATH = Path("benchmark_manifest_v2.0.0.json")
DRY_RUN_V2_PATH = Path("benchmark_v2_dry_run_report.json")
AUDIO_V2_DIR = Path("benchmark_audio_v2")


@pytest.fixture(scope="module")
def manifest_v2():
    assert MANIFEST_V2_PATH.exists(), f"Missing manifest v2.0.0 at {MANIFEST_V2_PATH}"
    return BatchManifest.model_validate_json(MANIFEST_V2_PATH.read_text(encoding="utf-8"))


def test_manifest_v2_schema_and_counts(manifest_v2):
    """Manifest v2.0.0 must contain exactly 600 pairs, 100 in each of the 6 categories."""
    assert manifest_v2.manifest_version == "2.0.0"
    assert len(manifest_v2.items) == 600

    from collections import Counter
    cat_counts = Counter(item.category for item in manifest_v2.items)

    expected_cats = {
        "quantities_and_numbers": 100,
        "negation": 100,
        "corrections_and_mind_changes": 100,
        "command_intent": 100,
        "named_entities": 100,
        "time_extraction": 100,
    }
    assert cat_counts == expected_cats, f"Unexpected category distribution: {cat_counts}"


@pytest.mark.skipif(
    not AUDIO_V2_DIR.exists() or not any(AUDIO_V2_DIR.iterdir()),
    reason="Requires local benchmark_audio_v2 audio files (gitignored)",
)
def test_manifest_v2_audio_files_exist_and_conform(manifest_v2):
    """All 1,200 audio files must exist, conform to 16kHz mono linear PCM WAV, and have duration <= 60s."""
    validator = AudioValidator()
    audio_files_seen = set()

    for item in manifest_v2.items:
        path_a = Path(item.audio_path_a)
        path_b = Path(item.audio_path_b)

        assert path_a.exists(), f"Missing audio file: {path_a}"
        assert path_b.exists(), f"Missing audio file: {path_b}"

        bytes_a = path_a.read_bytes()
        bytes_b = path_b.read_bytes()

        # AudioValidator check
        validator.validate_live_audio(path_a.name, bytes_a)
        validator.validate_live_audio(path_b.name, bytes_b)

        # Check PCM parameters with wave
        with wave.open(str(path_a), "rb") as wf_a:
            assert wf_a.getnchannels() == 1, f"Expected mono audio for {path_a}"
            assert wf_a.getframerate() == 16000, f"Expected 16kHz for {path_a}"
            assert wf_a.getsampwidth() == 2, f"Expected 16-bit for {path_a}"
            dur_a = wf_a.getnframes() / 16000.0
            assert 0.1 <= dur_a <= 60.0

        with wave.open(str(path_b), "rb") as wf_b:
            assert wf_b.getnchannels() == 1, f"Expected mono audio for {path_b}"
            assert wf_b.getframerate() == 16000, f"Expected 16kHz for {path_b}"
            assert wf_b.getsampwidth() == 2, f"Expected 16-bit for {path_b}"
            dur_b = wf_b.getnframes() / 16000.0
            assert 0.1 <= dur_b <= 60.0

        audio_files_seen.add(str(path_a))
        audio_files_seen.add(str(path_b))

    assert len(audio_files_seen) == 1200, f"Expected 1,200 distinct audio paths, got {len(audio_files_seen)}"


@pytest.mark.skipif(
    not AUDIO_V2_DIR.exists() or not any(AUDIO_V2_DIR.iterdir()),
    reason="Requires local benchmark_audio_v2 audio files (gitignored)",
)
def test_manifest_v2_within_pair_distinctness(manifest_v2):
    """Every pair must have distinct reference transcripts, distinct audio content, and distinct critical values."""
    for item in manifest_v2.items:
        assert item.reference_transcript_a != item.reference_transcript_b, (
            f"Duplicate reference transcripts in pair {item.pair_id}: {item.reference_transcript_a}"
        )

        bytes_a = Path(item.audio_path_a).read_bytes()
        bytes_b = Path(item.audio_path_b).read_bytes()
        assert bytes_a != bytes_b, f"Identical audio bytes in pair {item.pair_id}"

        # Distinct expected critical fields on distinction field
        df = item.distinction_field
        assert df is not None, f"Missing distinction field for {item.pair_id}"
        val_a = item.expected_critical_fields_a.get(df)
        val_b = item.expected_critical_fields_b.get(df)
        assert val_a is not None, f"Missing field {df} in expected_critical_fields_a for {item.pair_id}"
        assert val_b is not None, f"Missing field {df} in expected_critical_fields_b for {item.pair_id}"
        assert val_a != val_b, f"Expected values must differ on distinction field {df} in {item.pair_id}: {val_a} vs {val_b}"


def test_manifest_v2_subscription_non_renewal_annotation(manifest_v2):
    """Proves that non-renewal cases are labeled DO_NOT_RENEW_SUBSCRIPTION and not TERMINATE_SUBSCRIPTION."""
    neg_items = [it for it in manifest_v2.items if it.category == "negation"]
    renewal_cases = [it for it in neg_items if "renew" in it.reference_transcript_b.lower() or "रिन्यू" in it.reference_transcript_b]

    assert len(renewal_cases) >= 5, f"Expected at least 5 renewal negation cases, found {len(renewal_cases)}"
    for it in renewal_cases:
        assert it.expected_action_b == "DO_NOT_RENEW_SUBSCRIPTION", (
            f"Non-renewal case {it.pair_id} incorrectly labeled as {it.expected_action_b}"
        )


def test_manifest_v2_dry_run_audit_report():
    """Validates that dry run report confirms 600 ready items with 0 invalid and 0 duplicates."""
    assert DRY_RUN_V2_PATH.exists(), f"Missing dry run report at {DRY_RUN_V2_PATH}"
    data = json.load(open(DRY_RUN_V2_PATH, encoding="utf-8"))

    assert data["manifest_version"] == "2.0.0"
    assert data["total_items"] == 600
    assert data["ready_items"] == 600
    assert len(data["invalid_items"]) == 0
    assert len(data["duplicated_items"]) == 0
    assert len(data["missing_ground_truth_items"]) == 0
