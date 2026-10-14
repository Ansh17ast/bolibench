"""Automated tests for BoliBench Paired-Contrast Evaluation Engine.

Tests:
- Word Error Rate (WER) with exact mathematical verification
- Text normalization ensuring semantic tokens are preserved
- Critical field extraction: quantities, negation, corrections, entities, intent
- Semantic distinction preservation logic
- Downstream task success determination
- Explicit failure classification taxonomy
- SQLite persistence, querying, aggregate metrics, and exports
"""

from pathlib import Path
import pytest

from app.evaluation.dataset import load_seed_dataset
from app.evaluation.metrics import (
    calculate_wer,
    classify_failure,
    extract_all_critical_fields,
    extract_command_intent,
    extract_corrections,
    extract_entities,
    extract_negation,
    extract_quantities,
    normalize_text,
    score_pair,
)
from app.evaluation.schemas import (
    FailureCategory,
    TestCasePair,
    TestCategory,
    UtteranceSpec,
)
from app.evaluation.storage import StorageManager


# -----------------------------------------------------------------------------
# 1. WER Calculation Tests
# -----------------------------------------------------------------------------

def test_wer_identical_strings():
    """Identical sentences produce 0.0 WER."""
    wer, s, d, i, r_len = calculate_wer("Do packet chahiye", "Do packet chahiye")
    assert wer == 0.0
    assert s == 0 and d == 0 and i == 0
    assert r_len == 3


def test_wer_single_substitution():
    """Replacing one word in a 3-word sentence yields 1/3 = 0.3333 WER."""
    wer, s, d, i, r_len = calculate_wer("Do packet chahiye", "Teen packet chahiye")
    assert wer == pytest.approx(0.3333, abs=1e-4)
    assert s == 1 and d == 0 and i == 0
    assert r_len == 3


def test_wer_deletion():
    """Omitting one word from a 3-word sentence yields 1/3 = 0.3333 WER."""
    wer, s, d, i, r_len = calculate_wer("Do packet chahiye", "Do chahiye")
    assert wer == pytest.approx(0.3333, abs=1e-4)
    assert s == 0 and d == 1 and i == 0
    assert r_len == 3


def test_wer_insertion():
    """Inserting one word into a 2-word sentence yields 1/2 = 0.5 WER."""
    wer, s, d, i, r_len = calculate_wer("Do packet", "Do packet extra")
    assert wer == 0.5
    assert s == 0 and d == 0 and i == 1
    assert r_len == 2


def test_wer_empty_reference_empty_hyp():
    """Empty reference and empty hypothesis produces 0.0 WER."""
    wer, s, d, i, r_len = calculate_wer("", "")
    assert wer == 0.0
    assert r_len == 0


def test_wer_empty_reference_nonempty_hyp():
    """Empty reference with non-empty hypothesis produces 1.0 WER."""
    wer, s, d, i, r_len = calculate_wer("", "extra words spoken")
    assert wer == 1.0
    assert i == 3


def test_wer_nonempty_reference_empty_hyp():
    """Non-empty reference with empty hypothesis produces 1.0 WER (all deleted)."""
    wer, s, d, i, r_len = calculate_wer("Do packet chahiye", "")
    assert wer == 1.0
    assert d == 3


# -----------------------------------------------------------------------------
# 2. Text Normalization Tests
# -----------------------------------------------------------------------------

def test_normalization_preserves_semantic_distinctions():
    """Punctuation and whitespace are removed without stripping numbers or negations."""
    raw = "  Do packet, chahiye!  "
    norm = normalize_text(raw)
    assert norm == "do packet chahiye"

    # Hindi Devanagari danda punctuation removal
    hindi_raw = "पाँच लाख रुपये। कृपया भेजें॥"
    hindi_norm = normalize_text(hindi_raw)
    assert "।" not in hindi_norm
    assert "॥" not in hindi_norm
    assert "पाँच लाख रुपये" in hindi_norm

    # Crucial rule: "do" vs "teen" must remain distinct tokens
    assert normalize_text("Do") != normalize_text("Teen")
    assert normalize_text("karna") != normalize_text("mat karna")


# -----------------------------------------------------------------------------
# 3. Critical Field Extraction Tests
# -----------------------------------------------------------------------------

def test_extract_quantities_small_counts():
    """Extracts integer quantities from Hindi number words and digits."""
    q_do = extract_quantities("Do packet chahiye")
    assert q_do.get("quantity") == 2

    q_teen = extract_quantities("Teen packet chahiye")
    assert q_teen.get("quantity") == 3

    q_char = extract_quantities("Chaar plates")
    assert q_char.get("quantity") == 4

    q_digit = extract_quantities("I need 5 packets")
    assert q_digit.get("quantity") == 5


def test_extract_quantities_financial_scale():
    """Extracts large financial magnitudes (5 Lakh vs 5 Thousand)."""
    q_lakh = extract_quantities("Paanch lakh rupaye bhejo")
    assert q_lakh.get("quantity") == 500000

    q_hazaar = extract_quantities("Paanch hazaar rupaye bhejo")
    assert q_hazaar.get("quantity") == 5000

    # Currency symbol extraction
    q_currency = extract_quantities("Payment of ₹500 confirmed")
    assert q_currency.get("quantity") == 500


def test_extract_negation():
    """Detects Hindi and English negation tokens."""
    neg_hindi = extract_negation("Nahi subscription cancel mat karo")
    assert neg_hindi["is_negated"] is True
    assert neg_hindi["action"] == "keep_booking"

    aff_hindi = extract_negation("Haan subscription cancel kar do")
    assert aff_hindi["is_negated"] is False
    assert aff_hindi["action"] == "cancel_booking"

    neg_en = extract_negation("Do not transfer the funds yet")
    assert neg_en["is_negated"] is True
    assert neg_en["action"] == "block_transfer"

    aff_en = extract_negation("Transfer the funds now")
    assert aff_en["is_negated"] is False
    assert aff_en["action"] == "execute_transfer"


def test_extract_corrections_change_of_mind():
    """Extracts final intended value after a change of mind."""
    # Destination correction
    corr_dest = extract_corrections("Book flight to Mumbai, no wait, to Delhi")
    assert corr_dest["correction_detected"] is True
    assert corr_dest["final_target"] == "delhi"

    # Food ordering count correction
    corr_qty = extract_corrections("Do plate biryani, nahi teen plate karo")
    assert corr_qty["correction_detected"] is True
    assert corr_qty["final_target"] == 3


def test_extract_entities_and_products():
    """Extracts named entities and product names."""
    ent_rahul = extract_entities("Send invoice to Rahul Sharma")
    assert ent_rahul.get("person_name") == "Rahul Sharma"

    ent_rohit = extract_entities("Send invoice to Rohit Sharma")
    assert ent_rohit.get("person_name") == "Rohit Sharma"

    prod_prisma = extract_entities("Prisma STT ka access enable karo")
    assert prod_prisma.get("product_name") == "Prisma STT"

    prod_krishna = extract_entities("Krishna STT ka access enable karo")
    assert prod_krishna.get("product_name") == "Krishna STT"


def test_extract_command_intent():
    """Classifies high-risk banking and command intents."""
    assert extract_command_intent("Mera account balance check karo") == "CHECK_BALANCE"
    assert extract_command_intent("Mera account band kar do") == "CLOSE_ACCOUNT"
    assert extract_command_intent("Pay five hundred rupees") == "TRANSFER_FUNDS"
    assert extract_command_intent("Request five hundred rupees") == "REQUEST_FUNDS"


# -----------------------------------------------------------------------------
# 4. Paired-Contrast Scoring & Failure Classification Tests
# -----------------------------------------------------------------------------

def test_score_pair_preserved_distinction():
    """When both utterances match references, semantic distinction is preserved."""
    seed_cases = load_seed_dataset()
    pair_qty = next(tc for tc in seed_cases if tc.id == "pair_qty_001")

    res = score_pair(
        run_id="run_test_001",
        test_case=pair_qty,
        transcript_a="Do packet chahiye",
        transcript_b="Teen packet chahiye",
        execution_mode="mock",
    )

    assert res.semantic_distinction_preserved is True
    assert res.task_success is True
    assert res.critical_field_accuracy == 1.0
    assert res.average_wer == 0.0
    assert res.primary_failure_category == FailureCategory.NONE


def test_score_pair_crashed_distinction_quantity_error():
    """When Utterance B collapses into Utterance A's quantity, failure is classified as QUANTITY_ERROR."""
    seed_cases = load_seed_dataset()
    pair_qty = next(tc for tc in seed_cases if tc.id == "pair_qty_001")

    # Simulate Utterance B ("Teen packet chahiye") misheard as "Do packet chahiye"
    res = score_pair(
        run_id="run_test_002",
        test_case=pair_qty,
        transcript_a="Do packet chahiye",
        transcript_b="Do packet chahiye",  # Crashed transcript!
        execution_mode="mock",
    )

    assert res.semantic_distinction_preserved is False
    assert res.task_success is False
    assert res.primary_failure_category == FailureCategory.QUANTITY_ERROR
    assert "Numeric distinction collapsed" in res.failure_explanation


def test_score_pair_crashed_distinction_missed_negation():
    """When negation is omitted, failure is classified as MISSED_NEGATION."""
    seed_cases = load_seed_dataset()
    pair_neg = next(tc for tc in seed_cases if tc.id == "pair_neg_001")

    # Utterance B ("Nahi subscription cancel mat karo") misheard as "Haan subscription cancel kar do"
    res = score_pair(
        run_id="run_test_003",
        test_case=pair_neg,
        transcript_a="Haan subscription cancel kar do",
        transcript_b="Haan subscription cancel kar do",  # Critical missed negation
        execution_mode="mock",
    )

    assert res.semantic_distinction_preserved is False
    assert res.task_success is False
    assert res.primary_failure_category == FailureCategory.MISSED_NEGATION
    assert "Voice agent missed critical negation" in res.failure_explanation


def test_score_pair_crashed_correction_error():
    """When correction is ignored in favor of initial value, failure is classified as CORRECTION_ERROR."""
    seed_cases = load_seed_dataset()
    pair_corr = next(tc for tc in seed_cases if tc.id == "pair_corr_001")

    # Utterance A ("Book flight to Mumbai no wait to Delhi") truncated to "Book flight to Mumbai"
    res = score_pair(
        run_id="run_test_004",
        test_case=pair_corr,
        transcript_a="Book flight to Mumbai",  # Truncated before correction
        transcript_b="Book flight to Mumbai",
        execution_mode="mock",
    )

    assert res.semantic_distinction_preserved is False
    assert res.task_success is False
    assert res.primary_failure_category == FailureCategory.CORRECTION_ERROR


# -----------------------------------------------------------------------------
# 5. SQLite Persistence and Export Tests
# -----------------------------------------------------------------------------

def test_storage_manager_lifecycle(tmp_path: Path):
    """Verifies SQLite save, retrieve, aggregate metrics, and exports."""
    db_file = tmp_path / "test_eval.db"
    storage = StorageManager(db_path=db_file)

    seed_cases = load_seed_dataset()
    pair = seed_cases[0]

    # Save a run
    res = score_pair(
        run_id="run_storage_001",
        test_case=pair,
        transcript_a="Do packet chahiye",
        transcript_b="Teen packet chahiye",
        execution_mode="mock",
        timestamp="2026-10-15T00:00:00Z",
    )
    storage.save_run(res)

    # Retrieve run
    fetched = storage.get_run("run_storage_001")
    assert fetched is not None
    assert fetched.run_id == "run_storage_001"
    assert fetched.semantic_distinction_preserved is True

    # Aggregate metrics
    metrics = storage.get_aggregate_metrics()
    assert metrics.total_runs == 1
    assert metrics.mock_runs == 1
    assert metrics.distinction_preservation_rate == 100.0

    # JSON export
    json_export = storage.export_json()
    assert "run_storage_001" in json_export

    # CSV export
    csv_export = storage.export_csv()
    assert "run_storage_001" in csv_export
    assert "pair_qty_001" in csv_export
