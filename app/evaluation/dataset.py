"""Seed dataset loader and validator for BoliBench paired-contrast evaluation."""

import json
from pathlib import Path
from typing import Dict, List, Optional

from app.evaluation.schemas import TestCasePair

DATA_FILE = Path(__file__).resolve().parent / "data" / "seed_pairs.json"


def load_seed_dataset(filepath: Optional[Path] = None) -> List[TestCasePair]:
    """Loads and validates the seed test-case pairs dataset from JSON."""
    target_path = filepath or DATA_FILE
    if not target_path.exists():
        return []

    with open(target_path, "r", encoding="utf-8") as f:
        raw_items = json.load(f)

    return [TestCasePair(**item) for item in raw_items]


def get_test_case_by_id(test_id: str, filepath: Optional[Path] = None) -> Optional[TestCasePair]:
    """Finds a single test-case pair by its unique ID."""
    test_cases = load_seed_dataset(filepath)
    for tc in test_cases:
        if tc.id == test_id:
            return tc
    return None


def get_test_cases_by_category(category: str, filepath: Optional[Path] = None) -> List[TestCasePair]:
    """Filters test-case pairs by evaluation category."""
    test_cases = load_seed_dataset(filepath)
    return [tc for tc in test_cases if tc.category.value == category]
