"""SQLite persistence layer and export utilities for BoliBench evaluation results."""

import csv
import io
import json
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional

from app.evaluation.schemas import (
    AggregateMetrics,
    ExecutionMode,
    FailureCategory,
    PairEvaluationResult,
)

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent.parent / "bolibench_evaluation.db"


class StorageManager:
    """Manages SQLite storage for evaluation runs and aggregate metrics."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = str(db_path or DEFAULT_DB_PATH)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Initializes database tables if they do not exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
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
            # Backward-compatible schema migration if simulated_latency_ms column is missing
            try:
                cursor.execute("ALTER TABLE evaluation_runs ADD COLUMN simulated_latency_ms REAL")
            except sqlite3.OperationalError:
                pass

            # Backward-compatible schema migration if overall_status column is missing
            try:
                cursor.execute("ALTER TABLE evaluation_runs ADD COLUMN overall_status TEXT NOT NULL DEFAULT 'completed'")
            except sqlite3.OperationalError:
                pass

            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_eval_runs_overall_status
                ON evaluation_runs (overall_status)
                """
            )

            # Ensure existing records without status migrate as completed
            cursor.execute("UPDATE evaluation_runs SET overall_status = 'completed' WHERE overall_status IS NULL OR overall_status = ''")

            # Synchronize overall_status from payload_json if an explicit status was saved in payload
            cursor.execute("SELECT run_id, payload_json FROM evaluation_runs WHERE overall_status = 'completed'")
            for row in cursor.fetchall():
                try:
                    p = json.loads(row["payload_json"])
                    status = p.get("overall_status")
                    if status and status != "completed":
                        cursor.execute("UPDATE evaluation_runs SET overall_status = ? WHERE run_id = ?", (status, row["run_id"]))
                except Exception:
                    pass

            conn.commit()

    def save_run(self, result: PairEvaluationResult) -> str:
        """Persists an evaluation run."""
        payload_json = json.dumps(result.model_dump(), ensure_ascii=False)
        overall_status = getattr(result, "overall_status", "completed") or "completed"
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO evaluation_runs (
                    run_id, test_case_id, category, language_code, execution_mode,
                    audio_condition, overall_status, wer_a, wer_b, average_wer, critical_field_accuracy,
                    distinction_preserved, task_success, primary_failure_category,
                    failure_explanation, client_latency_ms, provider_latency_ms,
                    simulated_latency_ms, created_at, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                ),
            )
            conn.commit()
        return result.run_id

    def get_run(self, run_id: str) -> Optional[PairEvaluationResult]:
        """Retrieves a single evaluation run by ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT payload_json, overall_status FROM evaluation_runs WHERE run_id = ?", (run_id,))
            row = cursor.fetchone()
            if row:
                data = json.loads(row["payload_json"])
                if not data.get("overall_status"):
                    data["overall_status"] = row["overall_status"] or "completed"
                return PairEvaluationResult(**data)
        return None

    def list_runs(
        self,
        limit: int = 100,
        category: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[PairEvaluationResult]:
        """Lists recent evaluation runs, optionally filtered by category or status."""
        query = "SELECT payload_json, overall_status FROM evaluation_runs"
        params: List[Any] = []
        conditions: List[str] = []
        if category:
            conditions.append("category = ?")
            params.append(category)
        if status:
            conditions.append("overall_status = ?")
            params.append(status)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        results: List[PairEvaluationResult] = []
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, tuple(params))
            for row in cursor.fetchall():
                data = json.loads(row["payload_json"])
                if not data.get("overall_status"):
                    data["overall_status"] = row["overall_status"] or "completed"
                results.append(PairEvaluationResult(**data))
        return results

    def get_aggregate_metrics(self) -> AggregateMetrics:
        """Calculates aggregate metrics across all runs, restricting performance averages to completed runs."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
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
                    AVG(CASE WHEN overall_status = 'completed' THEN critical_field_accuracy ELSE NULL END) as avg_field_acc
                FROM evaluation_runs
                """
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
            avg_acc = round(float(summary["avg_field_acc"]) * 100.0, 2) if summary["avg_field_acc"] is not None else None

            # Failure category breakdown across all attempts (including provider operational errors)
            cursor.execute(
                """
                SELECT primary_failure_category, COUNT(*) as count
                FROM evaluation_runs
                GROUP BY primary_failure_category
                """
            )
            failure_breakdown = {row["primary_failure_category"]: row["count"] for row in cursor.fetchall()}

            # Category breakdown: counts all attempts, but computes performance averages over completed runs only
            cursor.execute(
                """
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
                GROUP BY category
                """
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
                distinction_preservation_rate=dist_rate,
                task_success_rate=task_rate,
                average_wer=avg_wer,
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
