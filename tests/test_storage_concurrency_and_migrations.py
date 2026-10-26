"""Isolated regression tests for SQLite concurrency, WAL mode, and startup scan removal.

Verifies:
1. WAL mode is enabled and PRAGMA busy_timeout is set to 30000ms.
2. Concurrent readers and writers do not encounter 'database is locked'.
3. One-time schema migration tracking eliminates repeated startup table scans (O(1) init).
4. Safe column addition preserves existing columns without swallowing genuine database errors.
"""

import concurrent.futures
import json
import sqlite3
import pytest

from app.evaluation.schemas import (
    ExecutionMode,
    FailureCategory,
    PairEvaluationResult,
    UtteranceEvaluationResult,
)
from app.evaluation.storage import StorageManager


def make_dummy_result(run_id: str, test_case_id: str = "pair_qty_001", overall_status: str = "completed") -> PairEvaluationResult:
    return PairEvaluationResult(
        run_id=run_id,
        test_case_id=test_case_id,
        category="quantities_and_numbers",
        language_code="hi-IN",
        execution_mode=ExecutionMode.MOCK,
        audio_condition="clean_recorded",
        timestamp="2026-10-23T12:00:00Z",
        overall_status=overall_status,
        utterance_a=UtteranceEvaluationResult(
            utterance_id=f"{test_case_id}_a",
            reference_transcript="Do packet chahiye",
            actual_transcript="Do packet chahiye",
            wer=0.0,
            substitutions=0,
            deletions=0,
            insertions=0,
            reference_length=3,
            critical_fields_match=True,
            detected_intent="ORDER",
            intent_match=True,
            task_success=True,
            status="completed",
        ),
        utterance_b=UtteranceEvaluationResult(
            utterance_id=f"{test_case_id}_b",
            reference_transcript="Teen packet chahiye",
            actual_transcript="Teen packet chahiye",
            wer=0.0,
            substitutions=0,
            deletions=0,
            insertions=0,
            reference_length=3,
            critical_fields_match=True,
            detected_intent="ORDER",
            intent_match=True,
            task_success=True,
            status="completed",
        ),
        wer_a=0.0,
        wer_b=0.0,
        average_wer=0.0,
        critical_field_accuracy=1.0,
        semantic_distinction_preserved=True,
        task_success=True,
        primary_failure_category=FailureCategory.NONE,
        failure_explanation="Distinction preserved",
    )


def test_sqlite_pragmas_and_wal_mode(tmp_path):
    """Proves WAL mode and busy_timeout are configured on SQLite connections."""
    db_file = tmp_path / "test_pragmas.db"
    storage = StorageManager(db_path=db_file)

    with storage._get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("PRAGMA journal_mode")
        journal_mode = cursor.fetchone()[0].lower()
        assert journal_mode == "wal", f"Expected WAL mode, got: {journal_mode}"

        cursor.execute("PRAGMA busy_timeout")
        busy_timeout = cursor.fetchone()[0]
        assert busy_timeout == 30000, f"Expected busy_timeout=30000, got: {busy_timeout}"


def test_startup_scan_removed_after_migration(tmp_path, monkeypatch):
    """Proves startup scan runs once and is skipped on all subsequent initializations."""
    db_file = tmp_path / "test_startup_scan.db"

    # Step 1: Initial creation applies migration
    storage1 = StorageManager(db_path=db_file)
    with storage1._get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT version FROM schema_migrations WHERE version = '20261020_sync_overall_status'")
        row = cursor.fetchone()
        assert row is not None, "Migration must be recorded in schema_migrations"

    # Save a run with overall_status = 'completed'
    storage1.save_run(make_dummy_result("run_startup_001"))

    # Step 2: Spy on cursor.execute during subsequent StorageManager instantiation
    executed_queries = []
    real_connect = sqlite3.connect

    class SpyingCursor(sqlite3.Cursor):
        def execute(self, sql, *args, **kwargs):
            executed_queries.append(sql.strip() if isinstance(sql, str) else str(sql))
            return super().execute(sql, *args, **kwargs)

    class SpyingConnection(sqlite3.Connection):
        def cursor(self, *args, **kwargs):
            return super().cursor(factory=SpyingCursor)

    def mock_connect(*args, **kwargs):
        kwargs["factory"] = SpyingConnection
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", mock_connect)

    # Initialize a new StorageManager on the same database
    storage2 = StorageManager(db_path=db_file)

    # Verify that the unindexed scan query was NEVER executed
    scan_queries = [
        q for q in executed_queries
        if "SELECT run_id, payload_json FROM evaluation_runs WHERE overall_status = 'completed'" in q
    ]
    assert len(scan_queries) == 0, f"Startup scan query was executed {len(scan_queries)} times on warm init!"


def test_concurrent_sqlite_reads_and_writes(tmp_path):
    """Proves concurrent threads can read and write without lock contention errors."""
    db_file = tmp_path / "test_concurrency.db"
    storage = StorageManager(db_path=db_file)

    num_threads = 8
    runs_per_thread = 5

    def worker(thread_idx: int):
        for i in range(runs_per_thread):
            run_id = f"run_th_{thread_idx}_{i}"
            res = make_dummy_result(run_id)
            storage.save_run(res)
            # Read back
            fetched = storage.get_run(run_id)
            assert fetched is not None
            assert fetched.run_id == run_id

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(worker, idx) for idx in range(num_threads)]
        for f in concurrent.futures.as_completed(futures):
            f.result()  # Re-raises exceptions if any occurred

    # Total runs should equal num_threads * runs_per_thread
    all_runs = storage.list_runs()
    assert len(all_runs) == num_threads * runs_per_thread


def test_safe_add_column_behavior(tmp_path):
    """Proves _safe_add_column ignores duplicate columns but propagates genuine errors."""
    db_file = tmp_path / "test_safe_col.db"
    storage = StorageManager(db_path=db_file)

    with storage._get_connection() as conn:
        cursor = conn.cursor()
        # 1. Adding already existing column (duplicate) should not raise
        storage._safe_add_column(cursor, "evaluation_runs", "wer_a REAL")

        # 2. Syntax error or invalid table should raise sqlite3.OperationalError
        with pytest.raises(sqlite3.OperationalError):
            storage._safe_add_column(cursor, "non_existent_table", "col1 TEXT")
