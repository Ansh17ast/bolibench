"""Reproducible offline-first batch benchmark runner for BoliBench.

Guarantees:
- Live execution is strictly disabled by default.
- Requires explicit --live, --max-pairs, confirmation, and feature flag.
- Pilot mode strictly caps live execution to 5 pairs (up to 10 STT requests).
- Sequential execution with configurable pacing delay between calls.
- Separate accounting of attempted, successful, failed, and retried requests (retries = 0).
- Sequential per-utterance failure isolation: if Utterance A fails, Utterance B is skipped.
- Resume support via persistent manifest / state to skip already completed utterances.
- Evaluates raw WER and cross-script normalized WER separately.
- Separates human-recorded, TTS-synthetic, and transformed acoustic variants.
- Never stores API secrets in manifests, reports, or logs.
- Disclaims statistical significance on small/correlated samples.
"""

import asyncio
import csv
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import uuid

from pydantic import BaseModel, Field

from app.config import Settings, get_settings
from app.evaluation.batch_manifest import (
    AudioSourceType,
    BatchManifest,
    BatchManifestItem,
    ManifestValidationReport,
    _resolve_audio_path,
    load_manifest,
    validate_manifest,
)
from app.evaluation.metrics import score_pair
from app.evaluation.schemas import (
    ExecutionMode,
    FailureCategory,
    PairEvaluationResult,
    TestCasePair,
    TestCategory,
    UtteranceEvaluationResult,
    UtteranceSpec,
)
from app.evaluation.storage import StorageManager
from app.services.audio_validator import AudioValidator
from app.services.gnani_client import GnaniClientError, GnaniPrismaClient

logger = logging.getLogger("bolibench.batch_runner")

MAX_PILOT_PAIRS = 5  # Strictly enforces the 5-pair pilot limit


class LiveExecutionBlockedError(Exception):
    """Raised when any required safeguard blocks live execution."""
    pass


class BatchExecutionConfig(BaseModel):
    """Configuration for batch benchmark execution."""
    live: bool = Field(False, description="Whether to make live provider calls. False by default.")
    max_pairs: Optional[int] = Field(None, description="Maximum number of pairs to evaluate.")
    pilot_mode: bool = Field(True, description="Enforce maximum 5-pair pilot limit.")
    pacing_delay_seconds: float = Field(1.0, description="Delay in seconds between sequential requests.")
    confirm: bool = Field(False, description="Explicit operator authorization confirmation.")
    resume: bool = Field(True, description="Skip already completed pairs in the batch run.")
    output_dir: Path = Field(default_factory=lambda: Path("benchmark_results"), description="Output destination directory.")
    batch_id: Optional[str] = Field(None, description="Stable batch identifier.")


class BatchDryRunReport(BaseModel):
    """Structured report returned by a dry-run execution."""
    batch_id: str
    manifest_name: str
    total_pairs: int
    ready_pairs: int
    planned_requests: int
    invalid_pairs: int
    duplicated_pairs: int
    missing_ground_truth: int
    counts_by_source: Dict[str, int]
    counts_by_category: Dict[str, int]
    output_dir: str
    validation_passed: bool
    invalid_details: List[Dict[str, Any]] = Field(default_factory=list)
    live_blocked_reason: Optional[str] = None
    disclaimer: str = (
        "DRY RUN ONLY: No network requests dispatched, no provider credits consumed. "
        "TTS-generated and transformed audio variants do not establish performance across independent human speakers."
    )

    def summary_markdown(self) -> str:
        lines = [
            "# Batch Benchmark Dry-Run Inspection Report",
            "",
            f"- **Batch ID**: `{self.batch_id}`",
            f"- **Manifest**: `{self.manifest_name}`",
            f"- **Total Manifest Pairs**: `{self.total_pairs}`",
            f"- **Ready Pairs (Eligible)**: `{self.ready_pairs}`",
            f"- **Planned Provider Requests**: `{self.planned_requests}` (2 requests per eligible pair)",
            f"- **Output Directory**: `{self.output_dir}`",
            f"- **Live Provider Calls Dispatched**: `0` (Zero credits consumed)",
            "",
            "### Pre-Flight Audio Validation",
            f"- **Valid & Ready**: `{self.ready_pairs}`",
            f"- **Invalid Audio / Headers**: `{self.invalid_pairs}`",
            f"- **Duplicates Detected**: `{self.duplicated_pairs}`",
            f"- **Missing Ground Truth**: `{self.missing_ground_truth}`",
            "",
            "### Source Provenance Breakdown",
            "| Source Type | Pairs | Share |",
            "| :--- | :---: | :---: |",
        ]
        for src, cnt in sorted(self.counts_by_source.items()):
            share = (cnt / max(1, self.total_pairs)) * 100
            lines.append(f"| `{src}` | {cnt} | {share:.1f}% |")

        lines.extend([
            "",
            "### Category Distribution",
            "| Category | Ready Pairs | Planned Calls |",
            "| :--- | :---: | :---: |",
        ])
        for cat, cnt in sorted(self.counts_by_category.items()):
            lines.append(f"| `{cat}` | {cnt} | {cnt * 2} |")

        lines.extend([
            "",
            "> [!NOTE]",
            f"> {self.disclaimer}",
        ])
        return "\n".join(lines)


class BatchBenchmarkRunner:
    """Orchestrates reproducible batch evaluations of curated audio pairs."""

    def __init__(
        self,
        config: Optional[BatchExecutionConfig] = None,
        settings: Optional[Settings] = None,
        client: Optional[GnaniPrismaClient] = None,
        storage: Optional[StorageManager] = None,
        audio_validator: Optional[AudioValidator] = None,
    ):
        self.config = config or BatchExecutionConfig()
        self.settings = settings or get_settings()
        self.storage = storage or StorageManager()
        self.audio_validator = audio_validator or AudioValidator()
        self._client = client

    @property
    def client(self) -> GnaniPrismaClient:
        if self._client is None:
            self._client = GnaniPrismaClient(settings=self.settings)
        return self._client

    def _generate_batch_id(self) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        suffix = uuid.uuid4().hex[:6]
        return f"batch_{timestamp}_{suffix}"

    def check_live_preflight_guards(self, pair_count: int) -> None:
        """Verifies every required guard before any live request can be sent.

        Guards:
        1. Explicit config.live must be True.
        2. Explicit max_pairs must be provided and > 0.
        3. In pilot mode, max_pairs strictly cannot exceed 5.
        4. Application setting LIVE_EVALUATION_ENABLED must be True.
        5. API credentials (GNANI_API_KEY) must be configured.
        6. Explicit confirmation flag (config.confirm) must be True.
        """
        if not self.config.live:
            raise LiveExecutionBlockedError(
                "Live execution is disabled by default to protect programme credits. "
                "Pass --live and an explicit --max-pairs limit to authorize live provider requests."
            )

        if self.config.max_pairs is None or self.config.max_pairs <= 0:
            raise LiveExecutionBlockedError(
                "Explicit --max-pairs limit is strictly required for live execution. "
                "Please specify a bounded number of pairs (e.g. --max-pairs 5)."
            )

        if self.config.pilot_mode and self.config.max_pairs > MAX_PILOT_PAIRS:
            raise LiveExecutionBlockedError(
                f"Pilot execution limit exceeded: Pilot runs are strictly capped at {MAX_PILOT_PAIRS} pairs "
                f"(up to {MAX_PILOT_PAIRS * 2} STT requests). Requested: {self.config.max_pairs}."
            )

        if not self.settings.live_evaluation_enabled:
            raise LiveExecutionBlockedError(
                "Forbidden: LIVE_EVALUATION_ENABLED is false in configuration. "
                "Set LIVE_EVALUATION_ENABLED=true in .env for authorized local operator testing."
            )

        if not self.settings.is_api_configured:
            raise LiveExecutionBlockedError(
                "Gnani API credentials are not configured. Set GNANI_API_KEY in .env before running live benchmarks."
            )

        if not self.config.confirm:
            expected_requests = min(pair_count, self.config.max_pairs) * 2
            raise LiveExecutionBlockedError(
                f"Explicit operator authorization confirmation required. "
                f"Planned execution will dispatch {expected_requests} live requests to Gnani Prisma STT v3. "
                "Pass --confirm to authorize dispatching requests."
            )

    def run_dry(
        self,
        manifest: Union[BatchManifest, str, Path],
        base_dir: Optional[Path] = None,
    ) -> BatchDryRunReport:
        """Performs a strict offline pre-flight validation and dry-run audit.

        Consumes 0 credits, makes 0 network calls.
        """
        report = validate_manifest(manifest, base_dir=base_dir, audio_validator=self.audio_validator)
        batch_id = self.config.batch_id or self._generate_batch_id()

        # Check output directory accessibility
        out_dir = Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        manifest_name = manifest.name if isinstance(manifest, BatchManifest) else str(manifest)

        return BatchDryRunReport(
            batch_id=batch_id,
            manifest_name=manifest_name,
            total_pairs=report.total_items,
            ready_pairs=report.ready_count,
            planned_requests=report.planned_requests_count,
            invalid_pairs=len(report.invalid_items),
            duplicated_pairs=len(report.duplicated_items),
            missing_ground_truth=len(report.missing_ground_truth_items),
            counts_by_source=report.counts_by_source,
            counts_by_category=report.counts_by_category,
            output_dir=str(out_dir.resolve()),
            validation_passed=report.is_valid,
            invalid_details=report.invalid_items,
        )

    async def run(
        self,
        manifest: Union[BatchManifest, str, Path],
        base_dir: Optional[Path] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """Executes the batch benchmark with auditable tracking and persistent resume support."""
        # 1. Pre-flight manifest validation
        report = validate_manifest(manifest, base_dir=base_dir, audio_validator=self.audio_validator)
        if not report.ready_items:
            raise ValueError(f"No ready pairs found in manifest ({len(report.invalid_items)} invalid items).")

        batch_id = self.config.batch_id or self._generate_batch_id()
        out_dir = Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        state_file = out_dir / f"{batch_id}_state.json"

        # 2. Check live execution guards if live mode is requested
        if self.config.live:
            self.check_live_preflight_guards(report.ready_count)

        # 3. Apply max_pairs limit
        eligible_items = report.ready_items
        if self.config.max_pairs is not None and self.config.max_pairs > 0:
            eligible_items = eligible_items[:self.config.max_pairs]

        # 4. Load existing state if resume is enabled
        completed_pairs: Dict[str, Dict[str, Any]] = {}
        if self.config.resume and state_file.exists():
            try:
                with open(state_file, "r", encoding="utf-8") as f:
                    saved_state = json.load(f)
                    completed_pairs = saved_state.get("items", {})
                    logger.info("Resuming batch '%s': found %d existing records in state", batch_id, len(completed_pairs))
            except Exception as exc:
                logger.warning("Could not read existing state file (%s), starting fresh: %s", state_file, exc)

        # 5. Initialize counters and metrics
        attempted_requests = 0
        successful_requests = 0
        failed_requests = 0
        retried_requests = 0  # Strictly 0 (automatic retries disabled to protect credits)

        evaluated_results: List[Dict[str, Any]] = []
        started_at = datetime.now(timezone.utc).isoformat()

        # 6. Sequential execution of selected items
        for index, item in enumerate(eligible_items):
            # Check resume state
            if self.config.resume and item.pair_id in completed_pairs:
                existing_item = completed_pairs[item.pair_id]
                if existing_item.get("status") == "completed":
                    logger.info("Skipping already completed pair '%s' on resume", item.pair_id)
                    evaluated_results.append(existing_item)
                    continue

            logger.info("Evaluating pair [%d/%d]: %s (%s)", index + 1, len(eligible_items), item.pair_id, item.category)

            # Construct mock TestCasePair for score_pair evaluation
            test_case = self._build_test_case(item)

            resolved_a = _resolve_audio_path(item.audio_path_a, base_dir or Path.cwd())
            resolved_b = _resolve_audio_path(item.audio_path_b, base_dir or Path.cwd())

            audio_a_bytes = resolved_a.read_bytes()
            audio_b_bytes = resolved_b.read_bytes()

            # Pacing delay between pairs (only when live)
            if self.config.live and index > 0 and self.config.pacing_delay_seconds > 0:
                await asyncio.sleep(self.config.pacing_delay_seconds)

            item_result = await self._evaluate_single_pair(
                item=item,
                test_case=test_case,
                audio_a_bytes=audio_a_bytes,
                filename_a=resolved_a.name,
                audio_b_bytes=audio_b_bytes,
                filename_b=resolved_b.name,
                batch_id=batch_id,
            )

            # Update request accounting
            attempted_requests += item_result["attempted_calls"]
            successful_requests += item_result["successful_calls"]
            failed_requests += item_result["failed_calls"]

            evaluated_results.append(item_result)
            completed_pairs[item.pair_id] = item_result

            # Persist incremental state to disk to survive unexpected interruption
            self._save_state(state_file, batch_id, completed_pairs)

            # Save item to SQLite storage
            self.storage.save_batch_item(item_result)

            if progress_callback:
                progress_callback({
                    "batch_id": batch_id,
                    "index": index + 1,
                    "total": len(eligible_items),
                    "pair_id": item.pair_id,
                    "status": item_result["status"],
                })

        finished_at = datetime.now(timezone.utc).isoformat()

        # 7. Compute aggregate statistics
        summary = self._compute_batch_summary(
            batch_id=batch_id,
            manifest_name=manifest.name if isinstance(manifest, BatchManifest) else str(manifest),
            items=evaluated_results,
            attempted_requests=attempted_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            retried_requests=retried_requests,
            started_at=started_at,
            finished_at=finished_at,
        )

        # 8. Persist batch run in SQLite
        self.storage.save_batch_run(summary)

        # 9. Generate file artifacts (CSV, JSON, Markdown)
        self._export_csv(out_dir / f"{batch_id}_results.csv", evaluated_results)
        self._export_json(out_dir / f"{batch_id}_results.json", summary, evaluated_results)
        self._export_markdown(out_dir / f"{batch_id}_summary.md", summary)

        return summary

    async def _evaluate_single_pair(
        self,
        item: BatchManifestItem,
        test_case: TestCasePair,
        audio_a_bytes: bytes,
        filename_a: str,
        audio_b_bytes: bytes,
        filename_b: str,
        batch_id: str,
    ) -> Dict[str, Any]:
        """Evaluates one pair with sequential failure isolation and pacing."""
        now = datetime.now(timezone.utc).isoformat()
        item_id = f"{batch_id}_{item.pair_id}"

        # If not live (dry-run or simulation), return deterministic baseline
        if not self.config.live:
            return {
                "item_id": item_id,
                "batch_id": batch_id,
                "pair_id": item.pair_id,
                "category": item.category,
                "language_code": item.language_code,
                "source_type": item.source_type.value,
                "audio_condition": item.audio_condition,
                "status": "dry_run",
                "attempted_calls": 0,
                "successful_calls": 0,
                "failed_calls": 0,
                "reference_transcript_a": item.reference_transcript_a,
                "reference_transcript_b": item.reference_transcript_b,
                "actual_transcript_a": item.reference_transcript_a,
                "actual_transcript_b": item.reference_transcript_b,
                "raw_wer_a": 0.0,
                "raw_wer_b": 0.0,
                "average_raw_wer": 0.0,
                "normalized_wer_a": 0.0,
                "normalized_wer_b": 0.0,
                "average_normalized_wer": 0.0,
                "critical_field_accuracy": 1.0,
                "distinction_preserved": True,
                "task_success": True,
                "primary_failure_category": None,
                "failure_explanation": "Dry-run inspection: No live calls made.",
                "request_id_a": "dry_run_a",
                "request_id_b": "dry_run_b",
                "client_elapsed_a_ms": 0.0,
                "client_elapsed_b_ms": 0.0,
                "total_latency_ms": 0.0,
                "error_message": None,
                "created_at": now,
                "payload_json": json.dumps({"mode": "dry_run"}),
            }

        # LIVE EXECUTION: Call 1 for Utterance A
        attempted_calls = 0
        successful_calls = 0
        failed_calls = 0

        # Transcribe Utterance A
        attempted_calls += 1
        start_a = time.perf_counter()
        try:
            resp_a = await self.client.transcribe(
                audio_bytes=audio_a_bytes,
                filename=filename_a,
                content_type="audio/wav",
                language_code=item.language_code,
                preferred_language=item.language_code,
                format_mode="transcribe",
            )
            elapsed_a_ms = round((time.perf_counter() - start_a) * 1000.0, 2)
            successful_calls += 1
            transcript_a = resp_a.get("transcript", "")
            req_id_a = resp_a.get("request_id")
        except Exception as exc:
            elapsed_a_ms = round((time.perf_counter() - start_a) * 1000.0, 2)
            failed_calls += 1
            safe_err = getattr(exc, "message", str(exc))
            logger.error("Utterance A failed for pair %s: %s (Utterance B skipped to protect credits)", item.pair_id, safe_err)
            return {
                "item_id": item_id,
                "batch_id": batch_id,
                "pair_id": item.pair_id,
                "category": item.category,
                "language_code": item.language_code,
                "source_type": item.source_type.value,
                "audio_condition": item.audio_condition,
                "status": "failed",
                "attempted_calls": attempted_calls,
                "successful_calls": successful_calls,
                "failed_calls": failed_calls,
                "reference_transcript_a": item.reference_transcript_a,
                "reference_transcript_b": item.reference_transcript_b,
                "actual_transcript_a": "",
                "actual_transcript_b": "",
                "raw_wer_a": 1.0,
                "raw_wer_b": 1.0,
                "average_raw_wer": 1.0,
                "normalized_wer_a": 1.0,
                "normalized_wer_b": 1.0,
                "average_normalized_wer": 1.0,
                "critical_field_accuracy": 0.0,
                "distinction_preserved": False,
                "task_success": False,
                "primary_failure_category": FailureCategory.PROVIDER_ERROR.value,
                "failure_explanation": f"Provider request failed for Utterance A: {safe_err}. Utterance B skipped.",
                "request_id_a": None,
                "request_id_b": None,
                "client_elapsed_a_ms": elapsed_a_ms,
                "client_elapsed_b_ms": None,
                "total_latency_ms": elapsed_a_ms,
                "error_message": safe_err,
                "created_at": now,
                "payload_json": json.dumps({"error": safe_err, "stage": "utterance_a"}),
            }

        # Pacing delay between Utterance A and Utterance B
        if self.config.pacing_delay_seconds > 0:
            await asyncio.sleep(self.config.pacing_delay_seconds)

        # Transcribe Utterance B
        attempted_calls += 1
        start_b = time.perf_counter()
        try:
            resp_b = await self.client.transcribe(
                audio_bytes=audio_b_bytes,
                filename=filename_b,
                content_type="audio/wav",
                language_code=item.language_code,
                preferred_language=item.language_code,
                format_mode="transcribe",
            )
            elapsed_b_ms = round((time.perf_counter() - start_b) * 1000.0, 2)
            successful_calls += 1
            transcript_b = resp_b.get("transcript", "")
            req_id_b = resp_b.get("request_id")
        except Exception as exc:
            elapsed_b_ms = round((time.perf_counter() - start_b) * 1000.0, 2)
            failed_calls += 1
            safe_err = getattr(exc, "message", str(exc))
            logger.error("Utterance B failed for pair %s: %s (Utterance A transcript preserved)", item.pair_id, safe_err)
            return {
                "item_id": item_id,
                "batch_id": batch_id,
                "pair_id": item.pair_id,
                "category": item.category,
                "language_code": item.language_code,
                "source_type": item.source_type.value,
                "audio_condition": item.audio_condition,
                "status": "incomplete",
                "attempted_calls": attempted_calls,
                "successful_calls": successful_calls,
                "failed_calls": failed_calls,
                "reference_transcript_a": item.reference_transcript_a,
                "reference_transcript_b": item.reference_transcript_b,
                "actual_transcript_a": transcript_a,
                "actual_transcript_b": "",
                "raw_wer_a": 1.0,
                "raw_wer_b": 1.0,
                "average_raw_wer": 1.0,
                "normalized_wer_a": 1.0,
                "normalized_wer_b": 1.0,
                "average_normalized_wer": 1.0,
                "critical_field_accuracy": 0.0,
                "distinction_preserved": False,
                "task_success": False,
                "primary_failure_category": FailureCategory.PROVIDER_ERROR.value,
                "failure_explanation": f"Utterance A succeeded, but Utterance B failed: {safe_err}. Incomplete run.",
                "request_id_a": req_id_a,
                "request_id_b": None,
                "client_elapsed_a_ms": elapsed_a_ms,
                "client_elapsed_b_ms": elapsed_b_ms,
                "total_latency_ms": elapsed_a_ms + elapsed_b_ms,
                "error_message": safe_err,
                "created_at": now,
                "payload_json": json.dumps({"error": safe_err, "stage": "utterance_b", "transcript_a": transcript_a}),
            }

        # Both utterances succeeded: Score the paired contrast
        pair_eval = score_pair(
            run_id=item_id,
            test_case=test_case,
            transcript_a=transcript_a,
            transcript_b=transcript_b,
            execution_mode="live",
            audio_condition=item.audio_condition,
            timestamp=now,
            client_elapsed_a_ms=elapsed_a_ms,
            client_elapsed_b_ms=elapsed_b_ms,
            request_id_a=req_id_a,
            request_id_b=req_id_b,
        )

        # Save to general evaluation_runs so history and dashboard reflect it
        self.storage.save_run(pair_eval)

        return {
            "item_id": item_id,
            "batch_id": batch_id,
            "pair_id": item.pair_id,
            "category": item.category,
            "language_code": item.language_code,
            "source_type": item.source_type.value,
            "audio_condition": item.audio_condition,
            "status": "completed",
            "attempted_calls": attempted_calls,
            "successful_calls": successful_calls,
            "failed_calls": failed_calls,
            "reference_transcript_a": item.reference_transcript_a,
            "reference_transcript_b": item.reference_transcript_b,
            "actual_transcript_a": transcript_a,
            "actual_transcript_b": transcript_b,
            "raw_wer_a": pair_eval.wer_a,
            "raw_wer_b": pair_eval.wer_b,
            "average_raw_wer": pair_eval.average_wer,
            "normalized_wer_a": pair_eval.normalized_wer_a,
            "normalized_wer_b": pair_eval.normalized_wer_b,
            "average_normalized_wer": pair_eval.average_normalized_wer,
            "critical_field_accuracy": pair_eval.critical_field_accuracy,
            "distinction_preserved": pair_eval.semantic_distinction_preserved,
            "task_success": pair_eval.task_success,
            "primary_failure_category": pair_eval.primary_failure_category.value if isinstance(pair_eval.primary_failure_category, FailureCategory) else str(pair_eval.primary_failure_category),
            "failure_explanation": pair_eval.failure_explanation,
            "request_id_a": req_id_a,
            "request_id_b": req_id_b,
            "client_elapsed_a_ms": elapsed_a_ms,
            "client_elapsed_b_ms": elapsed_b_ms,
            "total_latency_ms": elapsed_a_ms + elapsed_b_ms,
            "error_message": None,
            "created_at": now,
            "payload_json": json.dumps(pair_eval.model_dump(), ensure_ascii=False),
        }

    def _build_test_case(self, item: BatchManifestItem) -> TestCasePair:
        """Constructs a TestCasePair with the item's ground-truth reference transcripts and fields."""
        cat_enum = TestCategory(item.category) if item.category in [c.value for c in TestCategory] else TestCategory.QUANTITIES_AND_NUMBERS
        try:
            from app.evaluation.schemas import AudioCondition, CodeSwitchingType
            cond_enum = AudioCondition(item.audio_condition)
        except (ValueError, KeyError):
            cond_enum = AudioCondition.CLEAN_RECORDED

        dist_f = item.distinction_field
        if not dist_f:
            if item.expected_critical_fields_a:
                dist_f = list(item.expected_critical_fields_a.keys())[0]
            else:
                dist_f = "field"

        return TestCasePair(
            id=item.pair_id,
            category=cat_enum,
            language_code=item.language_code,
            code_switching_type=CodeSwitchingType.HINGLISH_CODE_MIXED,
            description=item.description or f"Batch evaluation pair: {item.pair_id}",
            expected_distinction=f"Preserve distinction across {dist_f}",
            distinction_field=dist_f,
            audio_condition=cond_enum,
            provenance=f"batch_benchmark_{item.source_type.value}",
            consent_metadata="batch_curated_pair",
            utterance_a=UtteranceSpec(
                id=f"{item.pair_id}_a",
                text=item.reference_transcript_a,
                reference_transcript=item.reference_transcript_a,
                expected_intent=item.expected_intent_a or "INTENT_A",
                expected_critical_fields=item.expected_critical_fields_a,
                expected_action=item.expected_action_a or "action_a",
            ),
            utterance_b=UtteranceSpec(
                id=f"{item.pair_id}_b",
                text=item.reference_transcript_b,
                reference_transcript=item.reference_transcript_b,
                expected_intent=item.expected_intent_b or "INTENT_B",
                expected_critical_fields=item.expected_critical_fields_b,
                expected_action=item.expected_action_b or "action_b",
            ),
        )

    def _save_state(self, state_file: Path, batch_id: str, completed_pairs: Dict[str, Any]) -> None:
        """Saves incremental execution state to avoid re-transcribing completed items on resume."""
        try:
            with open(state_file, "w", encoding="utf-8") as f:
                json.dump({
                    "batch_id": batch_id,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "items": completed_pairs,
                }, f, indent=2, ensure_ascii=False)
        except Exception as exc:
            logger.warning("Failed to save batch state file: %s", exc)

    def _compute_batch_summary(
        self,
        batch_id: str,
        manifest_name: str,
        items: List[Dict[str, Any]],
        attempted_requests: int,
        successful_requests: int,
        failed_requests: int,
        retried_requests: int,
        started_at: str,
        finished_at: str,
    ) -> Dict[str, Any]:
        """Calculates reproducible aggregate benchmark metrics."""
        total = len(items)
        completed = [it for it in items if it.get("status") == "completed"]
        failed = [it for it in items if it.get("status") == "failed"]
        incomplete = [it for it in items if it.get("status") == "incomplete"]
        skipped = [it for it in items if it.get("status") == "skipped"]

        comp_count = len(completed)

        mean_raw_wer = round(sum(it["average_raw_wer"] for it in completed) / max(1, comp_count), 4) if comp_count else None
        mean_norm_wer = round(sum(it["average_normalized_wer"] for it in completed) / max(1, comp_count), 4) if comp_count else None
        crit_acc = round(sum(it["critical_field_accuracy"] for it in completed) / max(1, comp_count), 4) if comp_count else None
        dist_rate = round(sum(1 for it in completed if it.get("distinction_preserved")) / max(1, comp_count), 4) if comp_count else None
        task_rate = round(sum(1 for it in completed if it.get("task_success")) / max(1, comp_count), 4) if comp_count else None

        # Breakdown by provenance
        provenance_breakdown: Dict[str, Dict[str, Any]] = {}
        for src_enum in AudioSourceType:
            src_val = src_enum.value
            src_items = [it for it in completed if it.get("source_type") == src_val]
            if src_items:
                provenance_breakdown[src_val] = {
                    "count": len(src_items),
                    "mean_raw_wer": round(sum(it["average_raw_wer"] for it in src_items) / len(src_items), 4),
                    "mean_normalized_wer": round(sum(it["average_normalized_wer"] for it in src_items) / len(src_items), 4),
                    "distinction_rate": round(sum(1 for it in src_items if it.get("distinction_preserved")) / len(src_items), 4),
                    "field_accuracy": round(sum(it["critical_field_accuracy"] for it in src_items) / len(src_items), 4),
                }

        # Breakdown by category
        category_breakdown: Dict[str, Dict[str, Any]] = {}
        for it in completed:
            cat = it.get("category", "unknown")
            if cat not in category_breakdown:
                category_breakdown[cat] = {"count": 0, "raw_wer": [], "norm_wer": [], "dist": 0, "field_acc": []}
            category_breakdown[cat]["count"] += 1
            category_breakdown[cat]["raw_wer"].append(it["average_raw_wer"])
            category_breakdown[cat]["norm_wer"].append(it["average_normalized_wer"])
            category_breakdown[cat]["field_acc"].append(it["critical_field_accuracy"])
            if it.get("distinction_preserved"):
                category_breakdown[cat]["dist"] += 1

        category_summary: Dict[str, Dict[str, Any]] = {}
        for cat, data in category_breakdown.items():
            cnt = data["count"]
            category_summary[cat] = {
                "count": cnt,
                "mean_raw_wer": round(sum(data["raw_wer"]) / cnt, 4),
                "mean_normalized_wer": round(sum(data["norm_wer"]) / cnt, 4),
                "distinction_rate": round(data["dist"] / cnt, 4),
                "field_accuracy": round(sum(data["field_acc"]) / cnt, 4),
            }

        # Latency statistics
        latencies = [it["total_latency_ms"] for it in completed if it.get("total_latency_ms")]
        latency_stats = {
            "mean_ms": round(sum(latencies) / len(latencies), 1) if latencies else None,
            "min_ms": round(min(latencies), 1) if latencies else None,
            "max_ms": round(max(latencies), 1) if latencies else None,
        }

        # Failure taxonomy
        failure_counts: Dict[str, int] = {}
        for it in items:
            fc = it.get("primary_failure_category")
            if fc and fc != FailureCategory.NONE.value:
                failure_counts[fc] = failure_counts.get(fc, 0) + 1

        return {
            "batch_id": batch_id,
            "name": f"Batch Benchmark - {manifest_name}",
            "execution_mode": "live" if self.config.live else "dry_run",
            "created_at": started_at,
            "finished_at": finished_at,
            "total_pairs": total,
            "completed_pairs": comp_count,
            "failed_pairs": len(failed),
            "incomplete_pairs": len(incomplete),
            "skipped_pairs": len(skipped),
            "attempted_requests": attempted_requests,
            "successful_requests": successful_requests,
            "failed_requests": failed_requests,
            "retried_requests": retried_requests,
            "mean_raw_wer": mean_raw_wer,
            "mean_normalized_wer": mean_norm_wer,
            "critical_field_accuracy": crit_acc,
            "distinction_preservation_rate": dist_rate,
            "task_success_rate": task_rate,
            "provenance_breakdown": provenance_breakdown,
            "category_breakdown": category_summary,
            "latency_stats": latency_stats,
            "failure_counts": failure_counts,
            "disclaimer": (
                "STATISTICAL VALIDITY NOTICE: Metrics are derived from bounded evaluation pairs and must not be used to claim "
                "universal general accuracy or statistical significance. TTS-generated and transformed audio variants do not "
                "establish performance across independent human speakers."
            ),
        }

    def _export_csv(self, path: Path, items: List[Dict[str, Any]]) -> None:
        """Exports item-level tabular results to CSV."""
        if not items:
            return
        fieldnames = [
            "pair_id", "category", "language_code", "source_type", "status",
            "reference_transcript_a", "actual_transcript_a",
            "reference_transcript_b", "actual_transcript_b",
            "raw_wer_a", "raw_wer_b", "average_raw_wer",
            "normalized_wer_a", "normalized_wer_b", "average_normalized_wer",
            "critical_field_accuracy", "distinction_preserved", "task_success",
            "primary_failure_category", "client_elapsed_a_ms", "client_elapsed_b_ms",
            "total_latency_ms", "request_id_a", "request_id_b", "error_message",
        ]
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for it in items:
                writer.writerow(it)

    def _export_json(self, path: Path, summary: Dict[str, Any], items: List[Dict[str, Any]]) -> None:
        """Exports complete JSON benchmark record."""
        data = {
            "summary": summary,
            "items": items,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _export_markdown(self, path: Path, summary: Dict[str, Any]) -> None:
        """Exports a human-readable markdown audit summary."""
        lines = [
            f"# BoliBench Batch Benchmark Audit Report: `{summary['batch_id']}`",
            "",
            f"- **Execution Mode**: `{summary['execution_mode'].upper()}`",
            f"- **Started**: `{summary['created_at']}`",
            f"- **Finished**: `{summary['finished_at']}`",
            f"- **Total Evaluated Pairs**: `{summary['total_pairs']}`",
            f"- **Completed Pairs**: `{summary['completed_pairs']}`",
            f"- **Failed Pairs**: `{summary['failed_pairs']}`",
            f"- **Incomplete Pairs**: `{summary['incomplete_pairs']}`",
            "",
            "### Provider Request Accounting",
            "| Metric | Count | Note |",
            "| :--- | :---: | :--- |",
            f"| **Attempted Provider Calls** | **{summary['attempted_requests']}** | Individual HTTP requests sent |",
            f"| **Successful Responses** | **{summary['successful_requests']}** | 200 OK transcribed |",
            f"| **Failed Calls** | **{summary['failed_requests']}** | Provider errors / timeouts |",
            f"| **Retried Calls** | **{summary['retried_requests']}** | Strictly 0 (no automatic retries) |",
            "",
            "### Aggregate Performance Metrics",
            "| Benchmark Metric | Value | Reference / Notes |",
            "| :--- | :---: | :--- |",
            f"| **Mean Raw WER** | `{summary['mean_raw_wer'] * 100:.2f}%` | Levenshtein word distance |" if summary['mean_raw_wer'] is not None else "| **Mean Raw WER** | `N/A` | No completed runs |",
            f"| **Mean Cross-Script Normalized WER** | `{summary['mean_normalized_wer'] * 100:.2f}%` | Canonicalized Hindi / numerals |" if summary['mean_normalized_wer'] is not None else "| **Mean Cross-Script Normalized WER** | `N/A` | No completed runs |",
            f"| **Critical Field Extraction Accuracy** | `{summary['critical_field_accuracy'] * 100:.1f}%` | Exact key-value slot matching |" if summary['critical_field_accuracy'] is not None else "| **Critical Field Extraction Accuracy** | `N/A` | No completed runs |",
            f"| **Semantic Distinction Preservation Rate** | `{summary['distinction_preservation_rate'] * 100:.1f}%` | Utterance A != Utterance B |" if summary['distinction_preservation_rate'] is not None else "| **Semantic Distinction Preservation Rate** | `N/A` | No completed runs |",
            f"| **Downstream Task Success Rate** | `{summary['task_success_rate'] * 100:.1f}%` | Correct agent action |" if summary['task_success_rate'] is not None else "| **Downstream Task Success Rate** | `N/A` | No completed runs |",
            "",
            "### Provenance & Audio Modality Breakdown",
            "| Modality | Pairs | Raw WER | Norm WER | Distinction Rate | Field Accuracy |",
            "| :--- | :---: | :---: | :---: | :---: | :---: |",
        ]
        for src, pdata in sorted(summary.get("provenance_breakdown", {}).items()):
            lines.append(
                f"| `{src}` | {pdata['count']} | {pdata['mean_raw_wer']*100:.1f}% | {pdata['mean_normalized_wer']*100:.1f}% | {pdata['distinction_rate']*100:.1f}% | {pdata['field_accuracy']*100:.1f}% |"
            )

        lines.extend([
            "",
            "### Category Breakdown",
            "| Category | Pairs | Raw WER | Norm WER | Distinction Rate | Field Accuracy |",
            "| :--- | :---: | :---: | :---: | :---: | :---: |",
        ])
        for cat, cdata in sorted(summary.get("category_breakdown", {}).items()):
            lines.append(
                f"| `{cat}` | {cdata['count']} | {cdata['mean_raw_wer']*100:.1f}% | {cdata['mean_normalized_wer']*100:.1f}% | {cdata['distinction_rate']*100:.1f}% | {cdata['field_accuracy']*100:.1f}% |"
            )

        if summary.get("latency_stats", {}).get("mean_ms"):
            lstats = summary["latency_stats"]
            lines.extend([
                "",
                "### Latency Statistics (Round-Trip Client Elapsed)",
                f"- **Mean Latency per Pair**: `{lstats['mean_ms']} ms`",
                f"- **Min Latency**: `{lstats['min_ms']} ms`",
                f"- **Max Latency**: `{lstats['max_ms']} ms`",
            ])

        lines.extend([
            "",
            "> [!NOTE]",
            f"> **Evaluation Integrity Notice:** {summary['disclaimer']}",
        ])

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
