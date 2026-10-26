"""SQLite persistence layer and export utilities for BoliBench evaluation results."""

import csv
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import sqlite3
import sys
from typing import Any, Dict, List, Optional

from app.evaluation.schemas import (
    AggregateMetrics,
    DerivedAssessment,
    ExecutionMode,
    FailureCategory,
    ModePerformanceMetrics,
    PairEvaluationResult,
)

PROD_DB_PATH = Path(__file__).resolve().parent.parent.parent / "bolibench_evaluation.db"
DEFAULT_DB_PATH = PROD_DB_PATH


def get_default_db_path() -> Path:
    """Resolves the active default SQLite database path.

    Loading priority:
    1. Explicit environment override `BOLIBENCH_DB_PATH` (used by test suites and custom deployments).
    2. Monkeypatched `DEFAULT_DB_PATH` module attribute (if different from PROD_DB_PATH).
    3. Test safety guard: If running under pytest and no override is set, redirects to a
       temporary session database to prevent any import or test from ever touching bolibench_evaluation.db.
    4. Default production database (`bolibench_evaluation.db`).
    """
    env_db = os.environ.get("BOLIBENCH_DB_PATH")
    if env_db:
        return Path(env_db).resolve()

    if Path(DEFAULT_DB_PATH).resolve() != PROD_DB_PATH.resolve():
        return Path(DEFAULT_DB_PATH).resolve()

    if "pytest" in sys.modules and not os.environ.get("BOLIBENCH_ALLOW_REAL_DB"):
        import tempfile
        temp_dir = Path(tempfile.gettempdir()) / "bolibench_pytest_isolated"
        temp_dir.mkdir(parents=True, exist_ok=True)
        return temp_dir / f"test_default_{os.getpid()}.db"

    return PROD_DB_PATH


class StorageManager:
    """Manages SQLite storage for evaluation runs and aggregate metrics."""

    def __init__(self, db_path: Optional[Path] = None):
        if db_path is not None:
            self.db_path = str(db_path)
        else:
            self.db_path = str(get_default_db_path())
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout = 30000")
        return conn

    def _safe_add_column(self, cursor: sqlite3.Cursor, table: str, column_def: str) -> None:
        """Safely executes ALTER TABLE ADD COLUMN, ignoring duplicate column errors while re-raising genuine failures."""
        try:
            cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column_def}")
        except sqlite3.OperationalError as exc:
            if "duplicate column name" in str(exc).lower():
                pass
            else:
                raise

    def _init_db(self) -> None:
        """Initializes database tables if they do not exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA journal_mode = WAL")
            cursor.execute("PRAGMA synchronous = NORMAL")

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version TEXT PRIMARY KEY,
                    applied_at TEXT NOT NULL
                )
                """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS evaluation_runs (
                    run_id TEXT PRIMARY KEY,
                    test_case_id TEXT NOT NULL,
                    category TEXT NOT NULL,
                    language_code TEXT NOT NULL,
                    execution_mode TEXT NOT NULL,
                    audio_condition TEXT NOT NULL,
                    overall_status TEXT NOT NULL DEFAULT 'completed',
                    wer_a REAL NOT NULL,
                    wer_b REAL NOT NULL,
                    average_wer REAL NOT NULL,
                    normalized_wer_a REAL,
                    normalized_wer_b REAL,
                    average_normalized_wer REAL,
                    critical_field_accuracy REAL NOT NULL,
                    distinction_preserved INTEGER NOT NULL,
                    task_success INTEGER NOT NULL,
                    primary_failure_category TEXT NOT NULL,
                    failure_explanation TEXT NOT NULL,
                    client_latency_ms REAL,
                    provider_latency_ms REAL,
                    simulated_latency_ms REAL,
                    created_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_eval_runs_test_case_id
                ON evaluation_runs (test_case_id)
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_eval_runs_category
                ON evaluation_runs (category)
                """
            )

            # Safe backward-compatible schema migrations with verified error handling
            self._safe_add_column(cursor, "evaluation_runs", "simulated_latency_ms REAL")
            self._safe_add_column(cursor, "evaluation_runs", "overall_status TEXT NOT NULL DEFAULT 'completed'")
            self._safe_add_column(cursor, "evaluation_runs", "normalized_wer_a REAL")
            self._safe_add_column(cursor, "evaluation_runs", "normalized_wer_b REAL")
            self._safe_add_column(cursor, "evaluation_runs", "average_normalized_wer REAL")
            self._safe_add_column(cursor, "evaluation_runs", "dataset_source TEXT NOT NULL DEFAULT 'synthetic_benchmark'")
            self._safe_add_column(cursor, "evaluation_runs", "dataset_classification TEXT NOT NULL DEFAULT 'synthetic'")

            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_eval_runs_dataset_source
                ON evaluation_runs (dataset_source)
                """
            )

            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_eval_runs_dataset_classification
                ON evaluation_runs (dataset_classification)
                """
            )

            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_eval_runs_overall_status
                ON evaluation_runs (overall_status)
                """
            )

            # Tracked one-time migration: sync overall_status from payload_json
            cursor.execute("SELECT 1 FROM schema_migrations WHERE version = '20261020_sync_overall_status'")
            if cursor.fetchone() is None:
                cursor.execute("UPDATE evaluation_runs SET overall_status = 'completed' WHERE overall_status IS NULL OR overall_status = ''")
                cursor.execute("SELECT run_id, payload_json FROM evaluation_runs WHERE overall_status = 'completed'")
                for row in cursor.fetchall():
                    try:
                        p = json.loads(row["payload_json"])
                        status = p.get("overall_status")
                        if status and status != "completed":
                            cursor.execute("UPDATE evaluation_runs SET overall_status = ? WHERE run_id = ?", (status, row["run_id"]))
                    except Exception:
                        pass
                cursor.execute(
                    "INSERT INTO schema_migrations (version, applied_at) VALUES (?, ?)",
                    ("20261020_sync_overall_status", datetime.now(timezone.utc).isoformat())
                )

            # Table for isolated offline derived assessments (re-scores)
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS derived_assessments (
                    derived_id TEXT PRIMARY KEY,
                    source_run_id TEXT NOT NULL,
                    test_case_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_derived_assessments_source_run_id
                ON derived_assessments (source_run_id)
                """
            )

            # Table for batch benchmark runs (isolated from single runs)
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS batch_benchmark_runs (
                    batch_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    execution_mode TEXT NOT NULL,
                    total_pairs INTEGER NOT NULL,
                    completed_pairs INTEGER NOT NULL,
                    failed_pairs INTEGER NOT NULL,
                    incomplete_pairs INTEGER NOT NULL,
                    skipped_pairs INTEGER NOT NULL,
                    attempted_requests INTEGER NOT NULL,
                    successful_requests INTEGER NOT NULL,
                    failed_requests INTEGER NOT NULL,
                    retried_requests INTEGER NOT NULL DEFAULT 0,
                    mean_raw_wer REAL,
                    mean_normalized_wer REAL,
                    critical_field_accuracy REAL,
                    distinction_preservation_rate REAL,
                    task_success_rate REAL,
                    summary_json TEXT NOT NULL
                )
                """
            )

            # Table for individual batch benchmark evaluated items
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS batch_benchmark_items (
                    item_id TEXT PRIMARY KEY,
                    batch_id TEXT NOT NULL,
                    pair_id TEXT NOT NULL,
                    category TEXT NOT NULL,
                    language_code TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    audio_condition TEXT NOT NULL,
                    status TEXT NOT NULL,
                    raw_wer_a REAL,
                    raw_wer_b REAL,
                    average_raw_wer REAL,
                    normalized_wer_a REAL,
                    normalized_wer_b REAL,
                    average_normalized_wer REAL,
                    critical_field_accuracy REAL,
                    distinction_preserved INTEGER,
                    task_success INTEGER,
                    primary_failure_category TEXT,
                    failure_explanation TEXT,
                    request_id_a TEXT,
                    request_id_b TEXT,
                    actual_transcript_a TEXT,
                    actual_transcript_b TEXT,
                    reference_transcript_a TEXT,
                    reference_transcript_b TEXT,
                    client_elapsed_a_ms REAL,
                    client_elapsed_b_ms REAL,
                    total_latency_ms REAL,
                    error_message TEXT,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_batch_items_batch_id
                ON batch_benchmark_items (batch_id)
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_batch_items_pair_id
                ON batch_benchmark_items (pair_id)
                """
            )

            conn.commit()

    def save_run(self, result: PairEvaluationResult) -> str:
        """Persists an evaluation run."""
        payload_json = json.dumps(result.model_dump(), ensure_ascii=False)
        overall_status = getattr(result, "overall_status", "completed") or "completed"
        dataset_source = getattr(result, "dataset_source", "synthetic_benchmark") or "synthetic_benchmark"
        dataset_classification = getattr(result, "dataset_classification", "synthetic") or "synthetic"
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO evaluation_runs (
                    run_id, test_case_id, category, language_code, execution_mode,
                    audio_condition, overall_status, wer_a, wer_b, average_wer,
                    normalized_wer_a, normalized_wer_b, average_normalized_wer,
                    critical_field_accuracy, distinction_preserved, task_success,
                    primary_failure_category, failure_explanation, client_latency_ms,
                    provider_latency_ms, simulated_latency_ms, created_at, payload_json, dataset_source, dataset_classification
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result.run_id,
                    result.test_case_id,
                    result.category,
                    result.language_code,
                    result.execution_mode.value if isinstance(result.execution_mode, ExecutionMode) else str(result.execution_mode),
                    result.audio_condition,
                    overall_status,
                    result.wer_a,
                    result.wer_b,
                    result.average_wer,
                    result.normalized_wer_a,
                    result.normalized_wer_b,
                    result.average_normalized_wer,
                    result.critical_field_accuracy,
                    1 if result.semantic_distinction_preserved else 0,
                    1 if result.task_success else 0,
                    result.primary_failure_category.value if isinstance(result.primary_failure_category, FailureCategory) else str(result.primary_failure_category),
                    result.failure_explanation,
                    result.client_measured_latency_ms,
                    result.provider_processing_time_ms,
                    result.simulated_latency_ms,
                    result.timestamp,
                    payload_json,
                    dataset_source,
                    dataset_classification,
                ),
            )
            conn.commit()
        return result.run_id

    def _row_to_result(self, row: sqlite3.Row) -> PairEvaluationResult:
        """Parses a database row into a PairEvaluationResult, adapting historical and batch runner formats."""
        data = json.loads(row["payload_json"]) if row["payload_json"] else {}
        if not data.get("run_id"):
            data["run_id"] = row["run_id"]
        if not data.get("test_case_id"):
            data["test_case_id"] = row["test_case_id"]
        if not data.get("category"):
            data["category"] = row["category"]
        if not data.get("language_code"):
            data["language_code"] = row["language_code"] or "hi-IN"
        if not data.get("overall_status"):
            data["overall_status"] = row["overall_status"] or "completed"
        if not data.get("execution_mode"):
            data["execution_mode"] = row["execution_mode"] or "live"
        if not data.get("audio_condition"):
            data["audio_condition"] = row["audio_condition"] or "clean_recorded"
        if not data.get("timestamp"):
            data["timestamp"] = data.get("created_at") or row["created_at"] or ""
        if "dataset_source" not in data:
            try:
                data["dataset_source"] = row["dataset_source"] or "synthetic_benchmark"
            except Exception:
                data["dataset_source"] = "synthetic_benchmark"
        if "dataset_classification" not in data:
            try:
                data["dataset_classification"] = row["dataset_classification"] or "synthetic"
            except Exception:
                data["dataset_classification"] = "synthetic"
        if "semantic_distinction_preserved" not in data:
            data["semantic_distinction_preserved"] = bool(row["distinction_preserved"]) if row["distinction_preserved"] is not None else bool(data.get("distinction_preserved", False))
        if "task_success" not in data:
            data["task_success"] = bool(row["task_success"]) if row["task_success"] is not None else False
        if "critical_field_accuracy" not in data:
            data["critical_field_accuracy"] = float(row["critical_field_accuracy"] or 0.0)
        if "wer_a" not in data:
            data["wer_a"] = float(row["wer_a"] or 0.0)
        if "wer_b" not in data:
            data["wer_b"] = float(row["wer_b"] or 0.0)
        if "average_wer" not in data:
            data["average_wer"] = float(row["average_wer"] or 0.0)
        p_fail = data.get("primary_failure_category") or row["primary_failure_category"] or "none"
        valid_cats = {fc.value for fc in FailureCategory}
        if p_fail not in valid_cats:
            if p_fail == "semantic_distinction_collapsed":
                p_fail = FailureCategory.EXTRACTION_FAILURE.value
            else:
                p_fail = FailureCategory.NONE.value
        data["primary_failure_category"] = p_fail
        if not data.get("failure_explanation"):
            data["failure_explanation"] = row["failure_explanation"] or ""
        if "utterance_a" not in data or not isinstance(data["utterance_a"], dict):
            data["utterance_a"] = {
                "utterance_id": f"{row['test_case_id']}_a",
                "reference_transcript": "",
                "actual_transcript": "",
                "wer": float(row["wer_a"] or 0.0),
                "substitutions": 0,
                "deletions": 0,
                "insertions": 0,
                "reference_length": 0,
                "critical_fields_match": bool(row["distinction_preserved"]),
                "detected_intent": "",
                "intent_match": True,
                "task_success": bool(row["task_success"]),
                "status": "completed" if row["overall_status"] == "completed" else "failed",
            }
        if "utterance_b" not in data or not isinstance(data["utterance_b"], dict):
            data["utterance_b"] = {
                "utterance_id": f"{row['test_case_id']}_b",
                "reference_transcript": "",
                "actual_transcript": "",
                "wer": float(row["wer_b"] or 0.0),
                "substitutions": 0,
                "deletions": 0,
                "insertions": 0,
                "reference_length": 0,
                "critical_fields_match": bool(row["distinction_preserved"]),
                "detected_intent": "",
                "intent_match": True,
                "task_success": bool(row["task_success"]),
                "status": "completed" if row["overall_status"] == "completed" else "failed",
            }
        return PairEvaluationResult(**data)

    def get_run(self, run_id: str) -> Optional[PairEvaluationResult]:
        """Retrieves a single evaluation run by ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM evaluation_runs WHERE run_id = ?", (run_id,))
            row = cursor.fetchone()
            if row:
                return self._row_to_result(row)
        return None

    def list_runs(
        self,
        limit: int = 100,
        category: Optional[str] = None,
        status: Optional[str] = None,
        dataset_source: Optional[str] = None,
        dataset_classification: Optional[str] = None,
    ) -> List[PairEvaluationResult]:
        """Lists recent evaluation runs, optionally filtered by category, status, dataset_source, or dataset_classification."""
        query = "SELECT * FROM evaluation_runs"
        params: List[Any] = []
        conditions: List[str] = []
        if category:
            conditions.append("category = ?")
            params.append(category)
        if status:
            conditions.append("overall_status = ?")
            params.append(status)
        if dataset_source:
            conditions.append("dataset_source = ?")
            params.append(dataset_source)
        if dataset_classification:
            conditions.append("dataset_classification = ?")
            params.append(dataset_classification)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        results: List[PairEvaluationResult] = []
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, tuple(params))
            for row in cursor.fetchall():
                results.append(self._row_to_result(row))
        return results

    def get_aggregate_metrics(self, dataset_classification: Optional[str] = None) -> AggregateMetrics:
        """Calculates aggregate metrics across all runs, separating mock vs live mode performance.
        Optionally filters by dataset_classification ('human_verified', 'synthetic', etc.).
        """
        where_clause = " WHERE dataset_classification = ?" if dataset_classification else ""
        params: Tuple[Any, ...] = (dataset_classification,) if dataset_classification else ()

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"""
                SELECT
                    COUNT(*) as total_runs,
                    SUM(CASE WHEN overall_status = 'completed' THEN 1 ELSE 0 END) as completed_runs,
                    SUM(CASE WHEN overall_status = 'failed' THEN 1 ELSE 0 END) as failed_runs,
                    SUM(CASE WHEN overall_status = 'incomplete' THEN 1 ELSE 0 END) as incomplete_runs,
                    SUM(CASE WHEN execution_mode = 'mock' THEN 1 ELSE 0 END) as mock_runs,
                    SUM(CASE WHEN execution_mode = 'live' THEN 1 ELSE 0 END) as live_runs,
                    AVG(CASE WHEN overall_status = 'completed' THEN distinction_preserved ELSE NULL END) as avg_distinction_preserved,
                    AVG(CASE WHEN overall_status = 'completed' THEN task_success ELSE NULL END) as avg_task_success,
                    AVG(CASE WHEN overall_status = 'completed' THEN average_wer ELSE NULL END) as avg_wer,
                    AVG(CASE WHEN overall_status = 'completed' THEN average_normalized_wer ELSE NULL END) as avg_norm_wer,
                    AVG(CASE WHEN overall_status = 'completed' THEN critical_field_accuracy ELSE NULL END) as avg_field_acc
                FROM evaluation_runs
                {where_clause}
                """,
                params,
            )
            summary = cursor.fetchone()

            total_runs = summary["total_runs"] or 0
            if total_runs == 0:
                return AggregateMetrics()

            completed_runs = summary["completed_runs"] or 0
            failed_runs = summary["failed_runs"] or 0
            incomplete_runs = summary["incomplete_runs"] or 0
            mock_runs = summary["mock_runs"] or 0
            live_runs = summary["live_runs"] or 0

            dist_rate = round(float(summary["avg_distinction_preserved"]) * 100.0, 2) if summary["avg_distinction_preserved"] is not None else None
            task_rate = round(float(summary["avg_task_success"]) * 100.0, 2) if summary["avg_task_success"] is not None else None
            avg_wer = round(float(summary["avg_wer"]), 4) if summary["avg_wer"] is not None else None
            avg_norm_wer = round(float(summary["avg_norm_wer"]), 4) if summary["avg_norm_wer"] is not None else None
            avg_acc = round(float(summary["avg_field_acc"]) * 100.0, 2) if summary["avg_field_acc"] is not None else None

            # Mode-specific performance metrics (completed runs only)
            cursor.execute(
                f"""
                SELECT
                    execution_mode,
                    COUNT(*) as total_attempts,
                    SUM(CASE WHEN overall_status = 'completed' THEN 1 ELSE 0 END) as completed_runs,
                    SUM(CASE WHEN overall_status = 'failed' THEN 1 ELSE 0 END) as failed_runs,
                    SUM(CASE WHEN overall_status = 'incomplete' THEN 1 ELSE 0 END) as incomplete_runs,
                    AVG(CASE WHEN overall_status = 'completed' THEN distinction_preserved ELSE NULL END) as avg_distinction_preserved,
                    AVG(CASE WHEN overall_status = 'completed' THEN task_success ELSE NULL END) as avg_task_success,
                    AVG(CASE WHEN overall_status = 'completed' THEN average_wer ELSE NULL END) as avg_wer,
                    AVG(CASE WHEN overall_status = 'completed' THEN average_normalized_wer ELSE NULL END) as avg_norm_wer,
                    AVG(CASE WHEN overall_status = 'completed' THEN critical_field_accuracy ELSE NULL END) as avg_field_acc
                FROM evaluation_runs
                {where_clause}
                GROUP BY execution_mode
                """,
                params,
            )
            mode_stats: Dict[str, Dict[str, Any]] = {}
            for row in cursor.fetchall():
                mode_stats[row["execution_mode"]] = dict(row)

            # Category breakdown per mode (completed runs only)
            cursor.execute(
                f"""
                SELECT
                    execution_mode,
                    category,
                    COUNT(*) as total,
                    SUM(CASE WHEN overall_status = 'completed' THEN 1 ELSE 0 END) as completed,
                    SUM(CASE WHEN overall_status = 'failed' THEN 1 ELSE 0 END) as failed,
                    SUM(CASE WHEN overall_status = 'incomplete' THEN 1 ELSE 0 END) as incomplete,
                    AVG(CASE WHEN overall_status = 'completed' THEN distinction_preserved ELSE NULL END) as dist_rate,
                    AVG(CASE WHEN overall_status = 'completed' THEN task_success ELSE NULL END) as task_rate,
                    AVG(CASE WHEN overall_status = 'completed' THEN average_wer ELSE NULL END) as avg_wer
                FROM evaluation_runs
                {where_clause}
                GROUP BY execution_mode, category
                """,
                params,
            )
            mode_cat_stats: Dict[str, Dict[str, Dict[str, Any]]] = {"mock": {}, "live": {}}
            for row in cursor.fetchall():
                m_mode = row["execution_mode"]
                if m_mode in mode_cat_stats:
                    mode_cat_stats[m_mode][row["category"]] = {
                        "total": row["total"],
                        "completed": row["completed"],
                        "failed": row["failed"],
                        "incomplete": row["incomplete"],
                        "distinction_rate": round(float(row["dist_rate"]) * 100.0, 2) if row["dist_rate"] is not None else None,
                        "task_success_rate": round(float(row["task_rate"]) * 100.0, 2) if row["task_rate"] is not None else None,
                        "average_wer": round(float(row["avg_wer"]), 4) if row["avg_wer"] is not None else None,
                    }

            def _build_mode_metrics(mode_key: str) -> ModePerformanceMetrics:
                data = mode_stats.get(mode_key, {})
                t_att = data.get("total_attempts", 0)
                comp = data.get("completed_runs", 0)
                fail = data.get("failed_runs", 0)
                incomp = data.get("incomplete_runs", 0)
                d_rate = round(float(data["avg_distinction_preserved"]) * 100.0, 2) if data.get("avg_distinction_preserved") is not None else None
                t_rate = round(float(data["avg_task_success"]) * 100.0, 2) if data.get("avg_task_success") is not None else None
                a_wer = round(float(data["avg_wer"]), 4) if data.get("avg_wer") is not None else None
                a_norm_wer = round(float(data["avg_norm_wer"]), 4) if data.get("avg_norm_wer") is not None else None
                a_acc = round(float(data["avg_field_acc"]) * 100.0, 2) if data.get("avg_field_acc") is not None else None
                return ModePerformanceMetrics(
                    total_attempts=t_att,
                    completed_runs=comp,
                    failed_runs=fail,
                    incomplete_runs=incomp,
                    distinction_preservation_rate=d_rate,
                    task_success_rate=t_rate,
                    average_wer=a_wer,
                    average_normalized_wer=a_norm_wer,
                    average_critical_field_accuracy=a_acc,
                    category_breakdown=mode_cat_stats.get(mode_key, {}),
                )

            mock_metrics = _build_mode_metrics("mock")
            live_metrics = _build_mode_metrics("live")

            # Failure category breakdown across all attempts (including provider operational errors)
            cursor.execute(
                f"""
                SELECT primary_failure_category, COUNT(*) as count
                FROM evaluation_runs
                {where_clause}
                GROUP BY primary_failure_category
                """,
                params,
            )
            failure_breakdown = {row["primary_failure_category"]: row["count"] for row in cursor.fetchall()}

            # Overall Category breakdown: counts all attempts, but computes performance averages over completed runs only
            cursor.execute(
                f"""
                SELECT
                    category,
                    COUNT(*) as total,
                    SUM(CASE WHEN overall_status = 'completed' THEN 1 ELSE 0 END) as completed,
                    SUM(CASE WHEN overall_status = 'failed' THEN 1 ELSE 0 END) as failed,
                    SUM(CASE WHEN overall_status = 'incomplete' THEN 1 ELSE 0 END) as incomplete,
                    AVG(CASE WHEN overall_status = 'completed' THEN distinction_preserved ELSE NULL END) as dist_rate,
                    AVG(CASE WHEN overall_status = 'completed' THEN task_success ELSE NULL END) as task_rate,
                    AVG(CASE WHEN overall_status = 'completed' THEN average_wer ELSE NULL END) as avg_wer
                FROM evaluation_runs
                {where_clause}
                GROUP BY category
                """,
                params,
            )
            cat_breakdown: Dict[str, Dict[str, Any]] = {}
            for row in cursor.fetchall():
                cat_breakdown[row["category"]] = {
                    "total": row["total"],
                    "completed": row["completed"],
                    "failed": row["failed"],
                    "incomplete": row["incomplete"],
                    "distinction_rate": round(float(row["dist_rate"]) * 100.0, 2) if row["dist_rate"] is not None else None,
                    "task_success_rate": round(float(row["task_rate"]) * 100.0, 2) if row["task_rate"] is not None else None,
                    "average_wer": round(float(row["avg_wer"]), 4) if row["avg_wer"] is not None else None,
                }

            return AggregateMetrics(
                total_runs=total_runs,
                completed_runs=completed_runs,
                failed_runs=failed_runs,
                incomplete_runs=incomplete_runs,
                mock_runs=mock_runs,
                live_runs=live_runs,
                mock=mock_metrics,
                live=live_metrics,
                distinction_preservation_rate=dist_rate,
                task_success_rate=task_rate,
                average_wer=avg_wer,
                average_normalized_wer=avg_norm_wer,
                average_critical_field_accuracy=avg_acc,
                category_breakdown=cat_breakdown,
                failure_category_breakdown=failure_breakdown,
            )

    def export_json(self) -> str:
        """Exports all evaluation runs as a formatted JSON string."""
        runs = self.list_runs(limit=1000)
        return json.dumps([r.model_dump() for r in runs], indent=2, ensure_ascii=False)

    def export_csv(self) -> str:
        """Exports all evaluation runs as CSV."""
        runs = self.list_runs(limit=1000)
        output = io.StringIO()
        fieldnames = [
            "run_id", "test_case_id", "category", "language_code", "execution_mode",
            "audio_condition", "overall_status", "timestamp", "wer_a", "wer_b", "average_wer",
            "normalized_wer_a", "normalized_wer_b", "average_normalized_wer",
            "critical_field_accuracy", "distinction_preserved", "task_success",
            "primary_failure_category", "failure_explanation",
            "client_request_elapsed_ms", "client_measured_latency_ms", "provider_processing_time_ms", "simulated_latency_ms",
            "actual_transcript_a", "actual_transcript_b"
        ]
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()

        for r in runs:
            writer.writerow({
                "run_id": r.run_id,
                "test_case_id": r.test_case_id,
                "category": r.category,
                "language_code": r.language_code,
                "execution_mode": r.execution_mode.value if isinstance(r.execution_mode, ExecutionMode) else str(r.execution_mode),
                "audio_condition": r.audio_condition,
                "overall_status": getattr(r, "overall_status", "completed") or "completed",
                "timestamp": r.timestamp,
                "wer_a": r.wer_a,
                "wer_b": r.wer_b,
                "average_wer": r.average_wer,
                "normalized_wer_a": r.normalized_wer_a,
                "normalized_wer_b": r.normalized_wer_b,
                "average_normalized_wer": r.average_normalized_wer,
                "critical_field_accuracy": r.critical_field_accuracy,
                "distinction_preserved": r.semantic_distinction_preserved,
                "task_success": r.task_success,
                "primary_failure_category": r.primary_failure_category.value if isinstance(r.primary_failure_category, FailureCategory) else str(r.primary_failure_category),
                "failure_explanation": r.failure_explanation,
                "client_request_elapsed_ms": r.client_request_elapsed_ms if r.client_request_elapsed_ms is not None else r.client_measured_latency_ms,
                "client_measured_latency_ms": r.client_measured_latency_ms,
                "provider_processing_time_ms": r.provider_processing_time_ms,
                "simulated_latency_ms": r.simulated_latency_ms,
                "actual_transcript_a": r.utterance_a.actual_transcript,
                "actual_transcript_b": r.utterance_b.actual_transcript,
            })

        return output.getvalue()

    # -------------------------------------------------------------------------
    # Offline Derived Assessments (Isolated Re-Scores)
    # -------------------------------------------------------------------------

    def save_derived_assessment(self, assessment: DerivedAssessment) -> str:
        """Persists an offline derived evaluation assessment without modifying source evaluation runs."""
        payload_json = json.dumps(assessment.model_dump(), ensure_ascii=False)
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO derived_assessments (
                    derived_id, source_run_id, test_case_id, created_at, payload_json
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    assessment.derived_id,
                    assessment.source_run_id,
                    assessment.test_case_id,
                    assessment.created_at,
                    payload_json,
                ),
            )
            conn.commit()
        return assessment.derived_id

    def get_derived_assessment(self, identifier: str) -> Optional[DerivedAssessment]:
        """Retrieves a derived assessment by derived_id or source_run_id."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT payload_json FROM derived_assessments WHERE derived_id = ? OR source_run_id = ? ORDER BY created_at DESC LIMIT 1",
                (identifier, identifier),
            )
            row = cursor.fetchone()
            if row:
                data = json.loads(row["payload_json"])
                return DerivedAssessment(**data)
        return None

    def list_derived_assessments(self, limit: int = 100) -> List[DerivedAssessment]:
        """Lists all offline derived evaluation assessments."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT payload_json FROM derived_assessments ORDER BY created_at DESC LIMIT ?",
                (limit,),
            )
            return [DerivedAssessment(**json.loads(row["payload_json"])) for row in cursor.fetchall()]

    def export_derived_assessments_json(self) -> str:
        """Exports all derived assessments as a formatted JSON string."""
        items = self.list_derived_assessments(limit=1000)
        return json.dumps([item.model_dump() for item in items], indent=2, ensure_ascii=False)

    def export_derived_assessments_csv(self) -> str:
        """Exports all derived assessments as CSV with explicit separation of original vs rescored metrics."""
        items = self.list_derived_assessments(limit=1000)
        output = io.StringIO()
        fieldnames = list(DerivedAssessment.model_fields.keys())
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for item in items:
            writer.writerow(item.model_dump())
        return output.getvalue()

    # -------------------------------------------------------------------------
    # Batch Benchmark Runs & Items Storage
    # -------------------------------------------------------------------------

    def save_batch_run(self, batch_data: Dict[str, Any]) -> str:
        """Persists a batch benchmark run summary."""
        batch_id = batch_data["batch_id"]
        summary_json = batch_data.get("summary_json") or json.dumps(batch_data, ensure_ascii=False)
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO batch_benchmark_runs (
                    batch_id, name, created_at, execution_mode, total_pairs,
                    completed_pairs, failed_pairs, incomplete_pairs, skipped_pairs,
                    attempted_requests, successful_requests, failed_requests,
                    retried_requests, mean_raw_wer, mean_normalized_wer,
                    critical_field_accuracy, distinction_preservation_rate,
                    task_success_rate, summary_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    batch_id,
                    batch_data.get("name", "Batch Benchmark Run"),
                    batch_data.get("created_at", ""),
                    batch_data.get("execution_mode", "dry_run"),
                    int(batch_data.get("total_pairs", 0)),
                    int(batch_data.get("completed_pairs", 0)),
                    int(batch_data.get("failed_pairs", 0)),
                    int(batch_data.get("incomplete_pairs", 0)),
                    int(batch_data.get("skipped_pairs", 0)),
                    int(batch_data.get("attempted_requests", 0)),
                    int(batch_data.get("successful_requests", 0)),
                    int(batch_data.get("failed_requests", 0)),
                    int(batch_data.get("retried_requests", 0)),
                    batch_data.get("mean_raw_wer"),
                    batch_data.get("mean_normalized_wer"),
                    batch_data.get("critical_field_accuracy"),
                    batch_data.get("distinction_preservation_rate"),
                    batch_data.get("task_success_rate"),
                    summary_json,
                ),
            )
            conn.commit()
        return batch_id

    def save_batch_item(self, item_data: Dict[str, Any]) -> str:
        """Persists an item from a batch benchmark run."""
        item_id = item_data.get("item_id") or f"{item_data['batch_id']}_{item_data['pair_id']}"
        payload_json = item_data.get("payload_json") or json.dumps(item_data, ensure_ascii=False)
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO batch_benchmark_items (
                    item_id, batch_id, pair_id, category, language_code,
                    source_type, audio_condition, status, raw_wer_a, raw_wer_b,
                    average_raw_wer, normalized_wer_a, normalized_wer_b,
                    average_normalized_wer, critical_field_accuracy,
                    distinction_preserved, task_success, primary_failure_category,
                    failure_explanation, request_id_a, request_id_b,
                    actual_transcript_a, actual_transcript_b,
                    reference_transcript_a, reference_transcript_b,
                    client_elapsed_a_ms, client_elapsed_b_ms, total_latency_ms,
                    error_message, payload_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item_id,
                    item_data["batch_id"],
                    item_data["pair_id"],
                    item_data.get("category", ""),
                    item_data.get("language_code", "hi-IN"),
                    item_data.get("source_type", "human_recorded"),
                    item_data.get("audio_condition", "clean_recorded"),
                    item_data.get("status", "completed"),
                    item_data.get("raw_wer_a"),
                    item_data.get("raw_wer_b"),
                    item_data.get("average_raw_wer"),
                    item_data.get("normalized_wer_a"),
                    item_data.get("normalized_wer_b"),
                    item_data.get("average_normalized_wer"),
                    item_data.get("critical_field_accuracy"),
                    1 if item_data.get("distinction_preserved") else 0,
                    1 if item_data.get("task_success") else 0,
                    item_data.get("primary_failure_category"),
                    item_data.get("failure_explanation"),
                    item_data.get("request_id_a"),
                    item_data.get("request_id_b"),
                    item_data.get("actual_transcript_a"),
                    item_data.get("actual_transcript_b"),
                    item_data.get("reference_transcript_a"),
                    item_data.get("reference_transcript_b"),
                    item_data.get("client_elapsed_a_ms"),
                    item_data.get("client_elapsed_b_ms"),
                    item_data.get("total_latency_ms"),
                    item_data.get("error_message"),
                    payload_json,
                    item_data.get("created_at", ""),
                ),
            )
            conn.commit()
        return item_id

    def get_batch_run(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a batch benchmark run summary by batch_id."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM batch_benchmark_runs WHERE batch_id = ?", (batch_id,))
            row = cursor.fetchone()
            if row:
                return dict(row)
        return None

    def list_batch_items(self, batch_id: str) -> List[Dict[str, Any]]:
        """Lists all items for a given batch benchmark run."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM batch_benchmark_items WHERE batch_id = ? ORDER BY created_at ASC", (batch_id,))
            return [dict(row) for row in cursor.fetchall()]

    def list_batch_runs(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Lists recent batch benchmark runs."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM batch_benchmark_runs ORDER BY created_at DESC LIMIT ?", (limit,))
            return [dict(row) for row in cursor.fetchall()]
