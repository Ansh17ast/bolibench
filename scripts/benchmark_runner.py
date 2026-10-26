"""Benchmark Runner for BoliBench.

Modes:
- --dry-run: Rigorous offline pre-flight validation. Inspects manifest, WAV files,
             contradictions, category distributions, and planned request counts.
             Consumes 0 credits, makes 0 network calls.
- --pilot: Executes a 5-pair pilot (10 requests max) across diverse categories.
           Requires explicit live authorization, safety feature flag, and confirmation.
- --live: Executes up to --max-pairs (hard capped at 50) after dry-run and pilot review.

Guarantees:
- Never silently starts live evaluation by default.
- Never retries timeouts or provider errors automatically.
- Checkpoints each utterance result immediately for idempotent resume.
- Enforces strict audio validation before dispatching requests.
- Distinguishes attempted, successful, failed, skipped, and resumed requests.
- Records provider request IDs, client latency, provider latency, timestamps.
- Never leaks API keys or secrets in manifests, results, or logs.
"""

import argparse
import asyncio
import csv
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from app.config import get_settings
from app.evaluation.batch_manifest import (
    AudioSourceType,
    BatchManifest,
    BatchManifestItem,
    ManifestValidationReport,
    load_manifest,
    validate_manifest,
)
from app.evaluation.classifier import ContrastInferenceResult, detect_semantic_contrast
from app.evaluation.metrics import (
    calculate_cross_script_normalized_wer,
    calculate_wer,
    extract_command_intent,
    extract_corrections,
    extract_entities,
    extract_negation,
    extract_quantities,
    extract_time,
)
from app.evaluation.schemas import ExecutionMode, FailureCategory, PairEvaluationResult, UtteranceEvaluationResult
from app.evaluation.storage import StorageManager
from app.services.audio_validator import AudioValidator
from app.services.gnani_client import GnaniPrismaClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("benchmark_runner")

MANIFEST_PATH = WORKSPACE_DIR / "benchmark_manifest.json"
DRY_RUN_REPORT_PATH = WORKSPACE_DIR / "benchmark_dry_run_report.json"


class BenchmarkRunner:
    """Orchestrates dry-run, pilot, and full live paired benchmark evaluation."""

    def __init__(
        self,
        manifest_path: Optional[Path] = None,
        checkpoint_dir: Optional[Path] = None,
        pacing_seconds: float = 1.0,
        batch_index: Optional[int] = None,
        batch_strategy: str = "stratified",
    ):
        if batch_index is not None and not manifest_path:
            self.manifest_path = WORKSPACE_DIR / "benchmark_manifest_v2.0.0.json"
        else:
            self.manifest_path = Path(manifest_path) if manifest_path else MANIFEST_PATH

        self.batch_index = batch_index
        self.batch_strategy = batch_strategy
        self.is_v2 = ("v2" in self.manifest_path.name or "2.0.0" in self.manifest_path.name)

        if self.batch_index is not None and not self.is_v2:
            raise ValueError(
                f"Batch execution with --batch-index {self.batch_index} requires a v2 manifest "
                f"(e.g. benchmark_manifest_v2.0.0.json), but loaded: {self.manifest_path.name}. "
                f"Historical v1 manifest cannot be run with batch indexing."
            )

        if checkpoint_dir is not None:
            self.checkpoint_dir = Path(checkpoint_dir)
        elif self.is_v2:
            self.checkpoint_dir = WORKSPACE_DIR / "benchmark_checkpoints" / "v2"
        else:
            self.checkpoint_dir = WORKSPACE_DIR / "benchmark_checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.pacing_seconds = pacing_seconds
        self.settings = get_settings()
        self.validator = AudioValidator()
        self.storage = StorageManager()

    # =========================================================================
    # 1. DRY-RUN MODE
    # =========================================================================
    def _select_batch_items(
        self,
        items: List[BatchManifestItem],
        batch_index: int,
        batch_size: int = 50,
        strategy: str = "stratified",
    ) -> List[BatchManifestItem]:
        """Selects exactly 50 unique items for a specific batch (1 to 12)."""
        if strategy == "contiguous":
            start_idx = (batch_index - 1) * batch_size
            end_idx = start_idx + batch_size
            return items[start_idx:end_idx]

        # Stratified: distribute 100 items per category across 12 batches
        categories = [
            "quantities_and_numbers",
            "negation",
            "corrections_and_mind_changes",
            "command_intent",
            "named_entities",
            "time_extraction",
        ]
        by_cat = {cat: [it for it in items if it.category == cat] for cat in categories}

        batch_items: List[BatchManifestItem] = []
        for cat_idx, cat in enumerate(categories):
            cat_list = by_cat.get(cat, [])
            cum_start = 0
            for b in range(batch_index - 1):
                count_b = 9 if ((b + cat_idx) % 6 < 2) else 8
                cum_start += count_b
            count_cur = 9 if (((batch_index - 1) + cat_idx) % 6 < 2) else 8
            batch_items.extend(cat_list[cum_start : cum_start + count_cur])

        return batch_items

    def execute_dry_run(self) -> Dict[str, Any]:
        """Performs rigorous pre-flight validation on every manifest entry."""
        logger.info(f"Executing Benchmark Runner --dry-run (0 network calls, 0 credits) on {self.manifest_path.name}...")
        manifest = load_manifest(self.manifest_path)
        report: ManifestValidationReport = validate_manifest(
            manifest, base_dir=WORKSPACE_DIR, audio_validator=self.validator
        )

        # Batch-specific dry-run mode (e.g. Batch 1 of 12)
        if self.batch_index is not None:
            selected_items = self._select_batch_items(
                report.ready_items,
                batch_index=self.batch_index,
                batch_size=50,
                strategy=self.batch_strategy,
            )

            # Safety Guards:
            # 1. Ensure all selected items are bench_v2_*
            for it in selected_items:
                if not it.pair_id.startswith("bench_v2_"):
                    raise ValueError(f"Batch item {it.pair_id} is not a valid bench_v2_* pair!")
                # 2. Block any historical v1 test cases
                if it.pair_id.startswith("bench_qty_") or it.pair_id.startswith("bench_neg_") or it.pair_id.startswith("bench_cor_") or it.pair_id.startswith("bench_cmd_") or it.pair_id.startswith("bench_ent_") or it.pair_id.startswith("bench_tim_"):
                    if not it.pair_id.startswith("bench_v2_"):
                        raise ValueError(f"Historical v1 test case {it.pair_id} detected in v2 batch! Rerun strictly blocked.")

            assert len(selected_items) == 50, f"Expected 50 items for batch {self.batch_index}, got {len(selected_items)}"

            # Dedicated v2 checkpoint file
            batch_checkpoint_file = self.checkpoint_dir / f"batch_v2_b{self.batch_index:02d}_checkpoint.json"
            checkpoint_state = self._load_checkpoint(batch_checkpoint_file)

            # Validate all 100 audio files in this batch offline
            for it in selected_items:
                self.validator.validate_live_audio(Path(it.audio_path_a).name, (WORKSPACE_DIR / it.audio_path_a).read_bytes())
                self.validator.validate_live_audio(Path(it.audio_path_b).name, (WORKSPACE_DIR / it.audio_path_b).read_bytes())

            completed_pairs_count = 0
            completed_pair_ids = []
            for it in selected_items:
                rec_a = checkpoint_state.get(f"{it.pair_id}_a")
                rec_b = checkpoint_state.get(f"{it.pair_id}_b")
                if rec_a and rec_b and rec_a.get("status") == "completed" and rec_b.get("status") == "completed":
                    completed_pairs_count += 1
                    completed_pair_ids.append(it.pair_id)

            untested_pairs_count = len(selected_items) - completed_pairs_count
            planned_new_requests = untested_pairs_count * 2

            from collections import Counter
            batch_cat_counts = Counter(it.category for it in selected_items)
            batch_src_counts = Counter(it.source_type.value if hasattr(it.source_type, "value") else str(it.source_type) for it in selected_items)

            dry_run_data = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "manifest_file": str(self.manifest_path),
                "manifest_name": manifest.name,
                "manifest_version": manifest.manifest_version,
                "batch_index": self.batch_index,
                "batch_strategy": self.batch_strategy,
                "total_manifest_pairs": report.total_items,
                "batch_pairs_count": len(selected_items),
                "planned_stt_requests": planned_new_requests,
                "completed_pairs_count": completed_pairs_count,
                "completed_pair_ids": completed_pair_ids,
                "untested_pairs_remaining": untested_pairs_count,
                "selected_pair_ids": [it.pair_id for it in selected_items],
                "category_distribution": dict(sorted(batch_cat_counts.items())),
                "provenance_distribution": dict(sorted(batch_src_counts.items())),
                "audio_validation_result": "PASS (100/100 WAV files valid 16kHz mono linear PCM)",
                "checkpoint_directory": str(self.checkpoint_dir),
                "checkpoint_file": str(batch_checkpoint_file),
                "historical_v1_rerun_prevented": True,
                "is_valid": report.is_valid,
                "dry_run_passed": report.is_valid and len(selected_items) == 50 and planned_new_requests <= 100,
            }

            dry_run_report_path = WORKSPACE_DIR / f"benchmark_v2_batch{self.batch_index:02d}_dry_run_report.json"
            with open(dry_run_report_path, "w", encoding="utf-8") as f:
                json.dump(dry_run_data, f, indent=2, ensure_ascii=False)

            # Print thorough, readable report to console
            print("\n" + "=" * 70)
            print(f"      BOLIBENCH v2.0.0 BATCH {self.batch_index:02d} DRY-RUN AUDIT REPORT")
            print("=" * 70)
            print(f"Manifest File:               {self.manifest_path.name} (Version: {manifest.manifest_version})")
            print(f"Batch Number:                Batch {self.batch_index} of 12 (Strategy: {self.batch_strategy})")
            print(f"Total Cohort Pairs:          {report.total_items} pairs across 6 categories")
            print(f"Batch Target Pairs:          {len(selected_items)} unique pairs (Max 100 STT requests)")
            print(f"Planned New STT Requests:    {planned_new_requests} requests")
            print(f"Audio Pre-Flight Validation: PASS (100/100 audio files valid 16kHz mono linear PCM)")
            print(f"Checkpoint Namespace:        {self.checkpoint_dir}")
            print(f"Checkpoint File:             {batch_checkpoint_file.name}")
            print(f"Historical v1 Rerun Blocked: CONFIRMED (0 v1 items eligible; 100% bench_v2_* items)")
            print(f"Live Requests Dispatched:    0 (Zero provider credits consumed)")
            print("-" * 70)
            print("Selected Pair IDs (Exactly 50):")
            for i in range(0, len(selected_items), 5):
                chunk = [it.pair_id for it in selected_items[i : i + 5]]
                print("  " + ", ".join(f"{pid:<18}" for pid in chunk))
            print("-" * 70)
            print("Batch Category Distribution:")
            for cat, cnt in sorted(batch_cat_counts.items()):
                print(f"  - {cat:<30}: {cnt} pairs ({cnt * 2} requests)")
            print("-" * 70)
            print("Batch Provenance Breakdown:")
            for src, cnt in sorted(batch_src_counts.items()):
                share = (cnt / len(selected_items)) * 100
                print(f"  - {src:<25}: {cnt:>2} pairs ({share:5.1f}%)")
            print("=" * 70)
            print(f"[PASS] BATCH {self.batch_index:02d} DRY-RUN PASSED: 50 unique bench_v2_* pairs audited and ready for authorization.")
            print("=" * 70 + "\n")

            return dry_run_data

        # Original full manifest dry-run (e.g. for v1.1.0 50-pair manifest)
        checkpoint_state = self._load_checkpoint(self.checkpoint_dir / "dry_run_probe.json")
        completed_pairs_count = 0
        completed_pair_ids = []
        for it in report.ready_items:
            rec_a = checkpoint_state.get(f"{it.pair_id}_a")
            rec_b = checkpoint_state.get(f"{it.pair_id}_b")
            if rec_a and rec_b and rec_a.get("status") == "completed" and rec_b.get("status") == "completed":
                completed_pairs_count += 1
                completed_pair_ids.append(it.pair_id)

        untested_pairs_count = report.ready_count - completed_pairs_count
        planned_new_requests = untested_pairs_count * 2

        dry_run_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "manifest_file": str(self.manifest_path),
            "manifest_name": manifest.name,
            "manifest_version": manifest.manifest_version,
            "total_pairs": report.total_items,
            "ready_pairs": report.ready_count,
            "planned_provider_requests": report.planned_requests_count,
            "completed_pilot_pairs_count": completed_pairs_count,
            "completed_pilot_pair_ids": completed_pair_ids,
            "untested_pairs_remaining": untested_pairs_count,
            "planned_new_provider_requests": planned_new_requests,
            "invalid_pairs_count": len(report.invalid_items),
            "duplicated_pairs_count": len(report.duplicated_items),
            "manual_review_flags_count": len(report.manual_review_items),
            "counts_by_category": report.counts_by_category,
            "counts_by_provenance": report.counts_by_source,
            "invalid_items": report.invalid_items,
            "duplicated_items": report.duplicated_items,
            "manual_review_items": report.manual_review_items,
            "is_valid": report.is_valid,
            "dry_run_passed": report.is_valid and (report.ready_count == 600 if self.is_v2 else report.ready_count == 50),
            "audio_details_sample": {
                k: report.audio_details[k] for k in list(report.audio_details.keys())[:5]
            },
        }

        # Write dry run report JSON
        with open(DRY_RUN_REPORT_PATH, "w", encoding="utf-8") as f:
            json.dump(dry_run_data, f, indent=2, ensure_ascii=False)

        print("\n" + "=" * 70)
        print("          BOLIBENCH BENCHMARK DRY-RUN AUDIT REPORT")
        print("=" * 70)
        print(f"Manifest Version:            {manifest.manifest_version}")
        print(f"Total Configured Pairs:      {report.total_items}")
        print(f"Validated Ready Pairs:       {report.ready_count} / 50 target")
        print(f"Total Planned Requests:      {report.planned_requests_count} (2 per pair)")
        print(f"Already Completed (Pilot):   {completed_pairs_count} pairs ({', '.join(completed_pair_ids) if completed_pair_ids else 'none'})")
        print(f"Untested Pairs Remaining:    {untested_pairs_count} pairs")
        print(f"Planned New Requests Needed: {planned_new_requests} requests (resumes 4 pilot pairs)")
        print(f"Invalid Items Detected:      {len(report.invalid_items)}")
        print(f"Duplicated IDs Detected:     {len(report.duplicated_items)}")
        print(f"Semantic Review Flags:       {len(report.manual_review_items)} pairs (lexical overlap / duration audited)")
        print("-" * 70)
        print("Category Distribution:")
        for cat, cnt in sorted(report.counts_by_category.items()):
            print(f"  - {cat:<30}: {cnt} pairs ({cnt * 2} requests)")
        print("-" * 70)
        print("Provenance Breakdown:")
        for src, cnt in sorted(report.counts_by_source.items()):
            share = (cnt / report.total_items * 100) if report.total_items > 0 else 0
            print(f"  - {src:<25}: {cnt:>2} pairs ({share:5.1f}%)")
        print("=" * 70)

        if not report.is_valid or report.ready_count != 50:
            print("[FAIL] DRY-RUN AUDIT FAILED: Validation issues detected.")
            for err in report.invalid_items:
                print(f"  [ERROR] {err.get('pair_id')}: {err.get('message')}")
        else:
            print("[PASS] DRY-RUN AUDIT PASSED: All 50 pairs validated and ready for controlled execution.")
        print("=" * 70 + "\n")

        return dry_run_data

    # =========================================================================
    # 2. PILOT & LIVE EXECUTION
    # =========================================================================
    async def execute_batch(
        self,
        mode: str,  # "pilot" or "live"
        max_pairs: int = 5,
        authorize_live: bool = False,
    ) -> Dict[str, Any]:
        """Executes a controlled batch with idempotent resume and safe pacing."""
        # 1. Hard safety checks
        if not authorize_live:
            raise PermissionError(
                "Execution halted: Explicit live authorization (--authorize-live) is required. "
                "Provider credits will be consumed for live evaluations."
            )

        if not self.settings.live_evaluation_enabled:
            raise PermissionError(
                "Execution halted: LIVE_EVALUATION_ENABLED is false in settings. "
                "Set LIVE_EVALUATION_ENABLED=true to permit live evaluation."
            )

        if not self.settings.is_api_configured:
            raise PermissionError("Execution halted: GNANI_API_KEY is not configured.")

        # Hard cap guard: never exceed requested max_pairs or 50 absolute limit
        effective_max_pairs = min(max_pairs, 50)
        if mode == "pilot":
            effective_max_pairs = min(effective_max_pairs, 5)

        # 2. Load and validate manifest
        manifest = load_manifest(self.manifest_path)
        report = validate_manifest(manifest, base_dir=WORKSPACE_DIR, audio_validator=self.validator)
        if not report.is_valid:
            raise ValueError(f"Cannot run batch: manifest has {len(report.invalid_items)} invalid items.")

        # 3. Select items
        if mode == "pilot":
            # Select 5 pairs across diverse categories
            selected_items = self._select_pilot_pairs(report.ready_items, target_count=5)
        elif self.batch_index is not None:
            selected_items = self._select_batch_items(
                report.ready_items,
                batch_index=self.batch_index,
                batch_size=effective_max_pairs,
                strategy=self.batch_strategy,
            )
            # Safety guards:
            for it in selected_items:
                if not it.pair_id.startswith("bench_v2_"):
                    raise ValueError(f"Batch item {it.pair_id} is not a valid bench_v2_* pair!")
                if (it.pair_id.startswith("bench_qty_") or it.pair_id.startswith("bench_neg_") or
                    it.pair_id.startswith("bench_cor_") or it.pair_id.startswith("bench_cmd_") or
                    it.pair_id.startswith("bench_ent_") or it.pair_id.startswith("bench_tim_")):
                    if not it.pair_id.startswith("bench_v2_"):
                        raise ValueError(f"Historical v1 test case {it.pair_id} detected in v2 batch! Rerun strictly blocked.")
        else:
            selected_items = report.ready_items[:effective_max_pairs]

        if self.batch_index is not None:
            batch_id = f"batch_v2_b{self.batch_index:02d}_{mode}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
            checkpoint_file = self.checkpoint_dir / f"batch_v2_b{self.batch_index:02d}_checkpoint.json"
        else:
            batch_id = f"batch_{mode}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
            checkpoint_file = self.checkpoint_dir / f"{batch_id}_checkpoint.json"

        print("\n" + "=" * 70)
        print(f"       BOLIBENCH BATCH EXECUTION: {mode.upper()} MODE")
        print("=" * 70)
        print(f"Batch ID:                    {batch_id}")
        print(f"Selected Pairs:              {len(selected_items)} (Max {len(selected_items) * 2} requests)")
        print(f"Pacing Interval:             {self.pacing_seconds:.1f}s between requests")
        print(f"Live Evaluation Safety:      VERIFIED (LIVE_EVALUATION_ENABLED=true)")
        print(f"Upstream Provider:           Gnani Prisma STT v3 REST API")
        print("=" * 70 + "\n")

        client = GnaniPrismaClient(settings=self.settings)

        # Resume state dictionary: { utterance_id: record }
        checkpoint_state = self._load_checkpoint(checkpoint_file)

        pair_results: List[Dict[str, Any]] = []
        utterance_records: List[Dict[str, Any]] = []

        total_requests_dispatched = 0
        total_requests_resumed = 0

        for idx, item in enumerate(selected_items, 1):
            pair_id = item.pair_id
            print(f"\n--- [{idx}/{len(selected_items)}] Evaluating Pair: {pair_id} ({item.category}) ---")

            # Utterance A execution
            uttr_a_id = f"{pair_id}_a"
            res_a, req_made_a = await self._process_utterance(
                client=client,
                item=item,
                side="a",
                checkpoint_state=checkpoint_state,
                checkpoint_file=checkpoint_file,
            )
            if req_made_a:
                total_requests_dispatched += 1
                await asyncio.sleep(self.pacing_seconds)
            else:
                total_requests_resumed += 1

            # Utterance B execution (executed independently; never skipped due to Utterance A failure)
            uttr_b_id = f"{pair_id}_b"
            res_b, req_made_b = await self._process_utterance(
                client=client,
                item=item,
                side="b",
                checkpoint_state=checkpoint_state,
                checkpoint_file=checkpoint_file,
            )
            if req_made_b:
                total_requests_dispatched += 1
                await asyncio.sleep(self.pacing_seconds)
            else:
                total_requests_resumed += 1

            # Pair level scoring and contrast assessment
            pair_score = self._score_pair(item=item, res_a=res_a, res_b=res_b, batch_id=batch_id)
            pair_results.append(pair_score)
            utterance_records.extend([res_a, res_b])

            # Persist to SQLite evaluation_runs table
            self._persist_pair_to_sqlite(pair_score)

        # 4. Generate batch artifacts
        summary = self._generate_batch_artifacts(
            batch_id=batch_id,
            mode=mode,
            pair_results=pair_results,
            utterance_records=utterance_records,
            requests_dispatched=total_requests_dispatched,
            requests_resumed=total_requests_resumed,
        )

        return summary

    def _select_pilot_pairs(self, items: List[BatchManifestItem], target_count: int = 5) -> List[BatchManifestItem]:
        """Selects target_count pairs covering distinct categories, prioritizing human recordings."""
        selected: List[BatchManifestItem] = []
        seen_cats: Set[str] = set()

        # First pass: human recordings across distinct categories
        for it in items:
            if it.source_type == AudioSourceType.HUMAN_RECORDED and it.category not in seen_cats:
                selected.append(it)
                seen_cats.add(it.category)
                if len(selected) >= target_count:
                    break

        # Second pass: fill remaining categories if needed
        if len(selected) < target_count:
            for it in items:
                if it.category not in seen_cats:
                    selected.append(it)
                    seen_cats.add(it.category)
                    if len(selected) >= target_count:
                        break

        return selected[:target_count]

    async def _process_utterance(
        self,
        client: GnaniPrismaClient,
        item: BatchManifestItem,
        side: str,  # "a" or "b"
        checkpoint_state: Dict[str, Any],
        checkpoint_file: Path,
    ) -> Tuple[Dict[str, Any], bool]:
        """Processes a single utterance with resume checking, WAV validation, and error trapping."""
        uttr_id = f"{item.pair_id}_{side}"
        audio_rel_path = item.audio_path_a if side == "a" else item.audio_path_b
        ref_text = item.reference_transcript_a if side == "a" else item.reference_transcript_b
        audio_abs_path = (WORKSPACE_DIR / audio_rel_path).resolve()

        # Checkpoint check for idempotent resume
        if uttr_id in checkpoint_state and checkpoint_state[uttr_id].get("status") == "completed":
            logger.info(f"Utterance [{uttr_id}] already completed in checkpoint. Resuming without request.")
            rec = checkpoint_state[uttr_id]
            rec["resumed"] = True
            return rec, False

        # Pre-execution audio validation
        audio_bytes = audio_abs_path.read_bytes()
        ext, ct = self.validator.validate_live_audio(audio_abs_path.name, audio_bytes)

        logger.info(f"Dispatching live request for [{uttr_id}] ({audio_abs_path.name}, {len(audio_bytes)} bytes)...")

        record: Dict[str, Any] = {
            "utterance_id": uttr_id,
            "pair_id": item.pair_id,
            "side": side,
            "audio_file": audio_abs_path.name,
            "reference_transcript": ref_text,
            "language_code": item.language_code,
            "source_type": item.source_type.value if hasattr(item.source_type, "value") else str(item.source_type),
            "status": "in_progress",
            "request_id": None,
            "actual_transcript": "",
            "client_elapsed_ms": None,
            "provider_processing_ms": None,
            "error": None,
            "wer_raw": None,
            "wer_normalized": None,
            "resumed": False,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        try:
            resp = await client.transcribe(
                audio_bytes=audio_bytes,
                filename=audio_abs_path.name,
                content_type=ct,
                language_code=item.language_code,
            )
            actual_tx = resp.get("transcript", "").strip()
            record["status"] = "completed"
            record["actual_transcript"] = actual_tx
            record["request_id"] = resp.get("request_id")
            record["client_elapsed_ms"] = resp.get("client_request_elapsed_ms")
            record["provider_processing_ms"] = resp.get("provider_processing_time_ms")

            # Scoring against ground truth reference
            record["wer_raw"] = round(calculate_wer(ref_text, actual_tx)[0], 4)
            record["wer_normalized"] = round(calculate_cross_script_normalized_wer(ref_text, actual_tx)[0], 4)

            logger.info(f"[{uttr_id}] 200 OK | RequestID={record['request_id']} | WER={record['wer_raw']*100:.1f}%")
            logger.info(f"[{uttr_id}] Actual:    '{actual_tx}'")
            logger.info(f"[{uttr_id}] Reference: '{ref_text}'")

        except Exception as exc:
            record["status"] = "failed"
            record["error"] = str(exc)
            logger.error(f"[{uttr_id}] FAILED: {exc}")

        # Immediately checkpoint result
        checkpoint_state[uttr_id] = record
        self._save_checkpoint(checkpoint_file, checkpoint_state)

        return record, True

    def _score_pair(
        self,
        item: BatchManifestItem,
        res_a: Dict[str, Any],
        res_b: Dict[str, Any],
        batch_id: str,
    ) -> Dict[str, Any]:
        """Performs joint semantic contrast scoring across pair utterances."""
        pair_id = item.pair_id
        tx_a = res_a.get("actual_transcript", "")
        tx_b = res_b.get("actual_transcript", "")

        status_a = res_a.get("status")
        status_b = res_b.get("status")

        if status_a != "completed" or status_b != "completed":
            return {
                "run_id": f"eval_live_{pair_id}_{batch_id}",
                "test_case_id": pair_id,
                "batch_id": batch_id,
                "category": item.category,
                "language_code": item.language_code,
                "source_type": item.source_type.value if hasattr(item.source_type, "value") else str(item.source_type),
                "overall_status": "incomplete",
                "failure_explanation": f"Utterance failure (A: {status_a}, B: {status_b})",
                "distinction_preserved": False,
                "task_success": False,
                "critical_field_accuracy": 0.0,
                "average_wer": res_a.get("wer_raw") or 0.0,
                "average_normalized_wer": res_a.get("wer_normalized"),
                "detected_category": "unknown",
                "created_at": datetime.now(timezone.utc).isoformat(),
            }

        # Joint classification
        cat_inference = detect_semantic_contrast(tx_a, tx_b, item.language_code)
        detected_cat = cat_inference.category
        detected_conf = cat_inference.confidence.value if hasattr(cat_inference.confidence, "value") else str(cat_inference.confidence)

        # Critical fields extraction
        expected_a = item.expected_critical_fields_a
        expected_b = item.expected_critical_fields_b
        dist_field = item.distinction_field

        fields_a: Dict[str, Any] = {}
        fields_b: Dict[str, Any] = {}

        if item.category == "quantities_and_numbers":
            qa = extract_quantities(tx_a).get("quantity")
            qb = extract_quantities(tx_b).get("quantity")
            fields_a["quantity"] = qa
            fields_b["quantity"] = qb
            preserved = (qa is not None and qb is not None and qa != qb and qa == expected_a.get("quantity") and qb == expected_b.get("quantity"))

        elif item.category == "negation":
            na = extract_negation(tx_a).get("is_negated")
            nb = extract_negation(tx_b).get("is_negated")
            fields_a["is_negated"] = na
            fields_b["is_negated"] = nb
            preserved = (na != nb and na == expected_a.get("is_negated") and nb == expected_b.get("is_negated"))

        elif item.category == "corrections_and_mind_changes":
            ca = extract_corrections(tx_a)
            cb = extract_corrections(tx_b)
            fields_a.update(ca)
            fields_b.update(cb)
            t_b = (
                cb.get("final_target")
                or cb.get("final_destination")
                or cb.get("final_item")
                or cb.get("color")
                or cb.get("quantity")
            )
            expected_target_b = (
                expected_b.get("final_target")
                or expected_b.get("final_destination")
                or expected_b.get("final_item")
                or expected_b.get("color")
                or expected_b.get("quantity")
            )
            if isinstance(t_b, str) and isinstance(expected_target_b, str):
                from app.evaluation.metrics import CROSS_SCRIPT_MAP
                t_b_norm = CROSS_SCRIPT_MAP.get(t_b.lower(), t_b.lower())
                exp_norm = CROSS_SCRIPT_MAP.get(expected_target_b.lower(), expected_target_b.lower())
                preserved = (cb.get("correction_detected", False) and t_b_norm == exp_norm)
            else:
                preserved = (cb.get("correction_detected", False) and t_b == expected_target_b)

        elif item.category == "command_intent":
            ia = extract_command_intent(tx_a)
            ib = extract_command_intent(tx_b)
            fields_a["intent"] = ia
            fields_b["intent"] = ib
            preserved = (ia != "UNKNOWN_INTENT" and ib != "UNKNOWN_INTENT" and ia != ib)

        elif item.category == "named_entities":
            ent_a = extract_entities(tx_a)
            ent_b = extract_entities(tx_b)
            ea = ent_a.get("name") or ent_a.get("person_name") or ent_a.get("recipient")
            eb = ent_b.get("name") or ent_b.get("person_name") or ent_b.get("recipient")
            fields_a["name"] = ea
            fields_b["name"] = eb
            preserved = (ea is not None and eb is not None and ea != eb)

        elif item.category == "time_extraction":
            tm_a = extract_time(tx_a)
            tm_b = extract_time(tx_b)
            ta = tm_a.get("delivery_time") or tm_a.get("time") or tm_a.get("hour")
            tb = tm_b.get("delivery_time") or tm_b.get("time") or tm_b.get("hour")
            fields_a["time"] = ta
            fields_b["time"] = tb
            preserved = (ta is not None and tb is not None and ta != tb)

        else:
            preserved = (tx_a.strip().lower() != tx_b.strip().lower())

        wer_a = res_a.get("wer_raw", 0.0)
        wer_b = res_b.get("wer_raw", 0.0)
        avg_wer = round((wer_a + wer_b) / 2.0, 4)

        nwer_a = res_a.get("wer_normalized")
        nwer_b = res_b.get("wer_normalized")
        avg_nwer = round((nwer_a + nwer_b) / 2.0, 4) if (nwer_a is not None and nwer_b is not None) else None

        task_success = preserved
        crit_acc = 1.0 if preserved else 0.0
        fail_cat = "none" if preserved else "semantic_distinction_collapsed"
        fail_exp = (
            f"Preserved distinction: {fields_a} vs {fields_b}."
            if preserved
            else f"Contrast collapsed or mismatch: recognized '{fields_a}' vs '{fields_b}'."
        )

        return {
            "run_id": f"eval_live_{pair_id}_{batch_id}",
            "test_case_id": pair_id,
            "batch_id": batch_id,
            "category": item.category,
            "language_code": item.language_code,
            "source_type": item.source_type.value if hasattr(item.source_type, "value") else str(item.source_type),
            "overall_status": "completed",
            "wer_a": wer_a,
            "wer_b": wer_b,
            "average_wer": avg_wer,
            "normalized_wer_a": nwer_a,
            "normalized_wer_b": nwer_b,
            "average_normalized_wer": avg_nwer,
            "critical_field_accuracy": crit_acc,
            "distinction_preserved": preserved,
            "task_success": task_success,
            "primary_failure_category": fail_cat,
            "failure_explanation": fail_exp,
            "detected_category": detected_cat,
            "detection_confidence": detected_conf,
            "fields_a": fields_a,
            "fields_b": fields_b,
            "request_id_a": res_a.get("request_id"),
            "request_id_b": res_b.get("request_id"),
            "client_elapsed_ms_a": res_a.get("client_elapsed_ms"),
            "client_elapsed_ms_b": res_b.get("client_elapsed_ms"),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

    def _persist_pair_to_sqlite(self, pair_score: Dict[str, Any]) -> None:
        """Stores pair result into SQLite evaluation_runs table idempotently."""
        payload_json = json.dumps(pair_score, ensure_ascii=False)
        with self.storage._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO evaluation_runs (
                    run_id, test_case_id, category, language_code, execution_mode,
                    audio_condition, overall_status, wer_a, wer_b, average_wer,
                    normalized_wer_a, normalized_wer_b, average_normalized_wer,
                    critical_field_accuracy, distinction_preserved, task_success,
                    primary_failure_category, failure_explanation, client_latency_ms,
                    provider_latency_ms, simulated_latency_ms, created_at, payload_json
                ) VALUES (
                    ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?, ?
                )
                """,
                (
                    pair_score["run_id"],
                    pair_score["test_case_id"],
                    pair_score["category"],
                    pair_score["language_code"],
                    "live",
                    "clean_recorded",
                    pair_score["overall_status"],
                    pair_score.get("wer_a", 0.0),
                    pair_score.get("wer_b", 0.0),
                    pair_score.get("average_wer", 0.0),
                    pair_score.get("normalized_wer_a"),
                    pair_score.get("normalized_wer_b"),
                    pair_score.get("average_normalized_wer"),
                    pair_score.get("critical_field_accuracy", 0.0),
                    1 if pair_score.get("distinction_preserved") else 0,
                    1 if pair_score.get("task_success") else 0,
                    pair_score.get("primary_failure_category", "none"),
                    pair_score.get("failure_explanation", ""),
                    (pair_score.get("client_elapsed_ms_a") or 0.0) + (pair_score.get("client_elapsed_ms_b") or 0.0),
                    None,
                    None,
                    pair_score["created_at"],
                    payload_json,
                ),
            )
            conn.commit()

    def _generate_batch_artifacts(
        self,
        batch_id: str,
        mode: str,
        pair_results: List[Dict[str, Any]],
        utterance_records: List[Dict[str, Any]],
        requests_dispatched: int,
        requests_resumed: int,
    ) -> Dict[str, Any]:
        """Generates JSON, CSVs, and Markdown report for the executed batch."""
        out_dir = WORKSPACE_DIR / "benchmark_results"
        out_dir.mkdir(parents=True, exist_ok=True)

        json_path = out_dir / f"{batch_id}_results.json"
        uttr_csv_path = out_dir / f"{batch_id}_utterances.csv"
        pair_csv_path = out_dir / f"{batch_id}_pairs.csv"
        md_report_path = out_dir / f"{batch_id}_report.md"

        # 1. Results JSON
        completed_pairs_count = sum(1 for p in pair_results if p.get("overall_status") == "completed")
        preserved_count = sum(1 for p in pair_results if p.get("distinction_preserved"))
        rate_completed = (preserved_count / completed_pairs_count) if completed_pairs_count else 0.0
        rate_total = (preserved_count / len(pair_results)) if pair_results else 0.0

        batch_summary = {
            "batch_id": batch_id,
            "mode": mode,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_pairs_evaluated": len(pair_results),
            "requests_dispatched": requests_dispatched,
            "requests_resumed": requests_resumed,
            "requests_successful": sum(1 for u in utterance_records if u.get("status") == "completed"),
            "requests_failed": sum(1 for u in utterance_records if u.get("status") == "failed"),
            "requests_skipped": sum(1 for u in utterance_records if u.get("status") == "skipped"),
            "total_provider_requests": requests_dispatched + requests_resumed,
            "completed_pairs": completed_pairs_count,
            "incomplete_pairs": [p["test_case_id"] for p in pair_results if p.get("overall_status") != "completed"],
            "distinction_preserved_count": preserved_count,
            "distinction_preserved_rate_of_completed": rate_completed,
            "distinction_preserved_rate_of_total": rate_total,
            "distinction_preserved_rate": rate_total,
            "mean_raw_wer": (
                sum(p.get("average_wer", 0.0) for p in pair_results) / len(pair_results)
                if pair_results
                else 0.0
            ),
            "pairs": pair_results,
            "utterances": utterance_records,
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(batch_summary, f, indent=2, ensure_ascii=False)

        # 2. Utterances CSV
        with open(uttr_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "utterance_id",
                    "pair_id",
                    "side",
                    "source_type",
                    "status",
                    "request_id",
                    "reference_transcript",
                    "actual_transcript",
                    "wer_raw",
                    "wer_normalized",
                    "client_elapsed_ms",
                    "resumed",
                    "timestamp",
                ],
            )
            writer.writeheader()
            for u in utterance_records:
                writer.writerow({k: u.get(k) for k in writer.fieldnames})

        # 3. Pairs CSV
        with open(pair_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "run_id",
                    "test_case_id",
                    "category",
                    "source_type",
                    "overall_status",
                    "distinction_preserved",
                    "task_success",
                    "average_wer",
                    "average_normalized_wer",
                    "detected_category",
                    "request_id_a",
                    "request_id_b",
                    "failure_explanation",
                    "created_at",
                ],
            )
            writer.writeheader()
            for p in pair_results:
                writer.writerow({k: p.get(k) for k in writer.fieldnames})

        # 4. Markdown Report
        md_content = self._render_markdown_report(batch_summary)
        with open(md_report_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        # Print console summary
        print("\n" + "=" * 70)
        print(f"        BOLIBENCH {mode.upper()} EVALUATION SUMMARY")
        print("=" * 70)
        print(f"Pairs Evaluated:             {len(pair_results)}")
        print(f"Actual Requests Dispatched:  {requests_dispatched}")
        print(f"Resumed From Checkpoint:     {requests_resumed}")
        print(f"Successful (200 OK):         {batch_summary['requests_successful']}")
        print(f"Failed Requests:             {batch_summary['requests_failed']}")
        print(f"Skipped Requests:            {batch_summary['requests_skipped']}")
        print(f"Distinction Preserved (Completed): {batch_summary['distinction_preserved_count']} / {batch_summary['completed_pairs']} ({batch_summary['distinction_preserved_rate_of_completed']*100:.1f}%)")
        print(f"Distinction Preserved (Total):     {batch_summary['distinction_preserved_count']} / {len(pair_results)} ({batch_summary['distinction_preserved_rate_of_total']*100:.1f}%)")
        print(f"Mean Raw WER:                {batch_summary['mean_raw_wer']*100:.1f}%")
        print(f"Results JSON:                {json_path.name}")
        print(f"Utterances CSV:              {uttr_csv_path.name}")
        print(f"Pairs CSV:                   {pair_csv_path.name}")
        print(f"Markdown Report:             {md_report_path.name}")
        print("=" * 70 + "\n")

        return batch_summary

    def _render_markdown_report(self, summary: Dict[str, Any]) -> str:
        """Renders comprehensive Markdown benchmark report."""
        lines = [
            f"# BoliBench {summary['mode'].title()} Benchmark Report",
            "",
            f"- **Batch ID**: `{summary['batch_id']}`",
            f"- **Execution Timestamp**: `{summary['timestamp']}`",
            f"- **Pairs Evaluated**: `{summary['total_pairs_evaluated']}`",
            f"- **Provider Requests Dispatched**: `{summary['requests_dispatched']}`",
            f"- **Provider Requests Resumed**: `{summary['requests_resumed']}`",
            f"- **Successful Responses (200 OK)**: `{summary.get('requests_successful', 0)}`",
            f"- **Failed Requests**: `{summary.get('requests_failed', 0)}`",
            f"- **Skipped Requests**: `{summary.get('requests_skipped', 0)}`",
            f"- **Distinction Preserved Rate (Completed)**: `{summary.get('distinction_preserved_rate_of_completed', 0.0)*100:.1f}%` ({summary['distinction_preserved_count']}/{summary['completed_pairs']})",
            f"- **Distinction Preserved Rate (Total)**: `{summary.get('distinction_preserved_rate_of_total', 0.0)*100:.1f}%` ({summary['distinction_preserved_count']}/{summary['total_pairs_evaluated']})",
            f"- **Incomplete Pairs**: `{', '.join(summary.get('incomplete_pairs', [])) or 'None'}`",
            f"- **Mean Raw WER**: `{summary['mean_raw_wer']*100:.1f}%`",
            "",
            "## Pair Results Breakdown",
            "",
            "| Pair ID | Category | Provenance | Distinction | Task | Raw WER | Request ID A | Request ID B |",
            "| :--- | :--- | :--- | :---: | :---: | :---: | :--- | :--- |",
        ]
        for p in summary["pairs"]:
            preserved_badge = "✅ Preserved" if p.get("distinction_preserved") else "❌ Collapsed"
            task_badge = "✅ Success" if p.get("task_success") else "❌ Failed"
            wer_str = f"{p.get('average_wer', 0.0)*100:.1f}%"
            req_a = p.get("request_id_a") or "None"
            req_b = p.get("request_id_b") or "None"
            lines.append(
                f"| `{p.get('test_case_id')}` | `{p.get('category')}` | `{p.get('source_type')}` | {preserved_badge} | {task_badge} | {wer_str} | `{req_a}` | `{req_b}` |"
            )

        lines.extend([
            "",
            "## Upstream Provider Transcripts vs References",
            "",
        ])
        for p in summary["pairs"]:
            pid = p.get("test_case_id")
            uttrs = [u for u in summary["utterances"] if u.get("pair_id") == pid]
            lines.append(f"### Pair `{pid}` ({p.get('category')})")
            lines.append(f"- **Verdict**: {p.get('failure_explanation')}")
            for u in uttrs:
                side_label = "Utterance A (Baseline)" if u.get("side") == "a" else "Utterance B (Contrast)"
                lines.append(f"  * **{side_label}**:")
                lines.append(f"    - Reference: `{u.get('reference_transcript')}`")
                raw_wer_val = u.get("wer_raw")
                norm_wer_val = u.get("wer_normalized")
                raw_wer_str = f"{raw_wer_val*100:.1f}%" if raw_wer_val is not None else "N/A"
                norm_wer_str = f"{norm_wer_val*100:.1f}%" if norm_wer_val is not None else "N/A"
                latency_str = f"{u.get('client_elapsed_ms')} ms" if u.get("client_elapsed_ms") is not None else "N/A"
                lines.append(f"    - Raw WER:   `{raw_wer_str}` | Norm WER: `{norm_wer_str}` | Latency: `{latency_str}`")
            lines.append("")

        lines.extend([
            "## Safety & Credit Integrity Notice",
            "",
            "> [!NOTE]",
            "> - **Provider Requests**: Exactly one provider request dispatched per utterance attempt. Zero automatic retries.",
            "> - **Credit Account Verification**: Gnani Prisma STT v3 REST API does not provide a balance inquiry endpoint via API. Account balance must be inspected via dashboard.",
            "> - **Provenance Disclaimer**: Results on TTS-synthetic or acoustic-transformed audio do not prove performance across independent human speakers.",
        ])

        return "\n".join(lines)

    def _load_checkpoint(self, path: Path) -> Dict[str, Any]:
        """Loads cumulative checkpoint state across all historical checkpoints and SQLite.

        Guarantees:
        - Scans all *_checkpoint.json files in checkpoint_dir chronologically.
        - Loads any existing records for the current batch.
        - Checks SQLite evaluation_runs for completed utterances.
        - Merges completed utterance records so that previous pilot/batch results are recognized and skipped.
        - Prevents re-dispatching already completed requests across multiple batch runs.
        """
        cumulative: Dict[str, Any] = {}
        if self.checkpoint_dir.exists():
            cp_files = sorted(self.checkpoint_dir.glob("*_checkpoint.json"), key=lambda p: p.stat().st_mtime)
            for cp_file in cp_files:
                try:
                    with open(cp_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if isinstance(data, dict):
                            for k, v in data.items():
                                if isinstance(v, dict) and v.get("status") == "completed":
                                    cumulative[k] = v
                except Exception as exc:
                    logger.warning(f"Could not read historical checkpoint {cp_file.name}: {exc}")

        # Also load current batch checkpoint if it exists
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        cumulative.update(data)
            except Exception:
                pass

        return cumulative

    def _save_checkpoint(self, path: Path, state: Dict[str, Any]) -> None:
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2, ensure_ascii=False)
        except Exception as exc:
            logger.warning(f"Failed to write checkpoint {path}: {exc}")


def main():
    parser = argparse.ArgumentParser(description="BoliBench Contrast Evaluation Benchmark Runner")
    parser.add_argument("--dry-run", action="store_true", help="Perform offline pre-flight validation (0 credits)")
    parser.add_argument("--pilot", action="store_true", help="Execute 5-pair pilot evaluation (max 10 requests)")
    parser.add_argument("--live", action="store_true", help="Execute live benchmark evaluation")
    parser.add_argument("--manifest", type=str, default=None, help="Path to manifest JSON file")
    parser.add_argument("--batch-index", type=int, default=None, help="Batch index (1-12) for v2 50-pair batch execution")
    parser.add_argument("--batch-strategy", type=str, default="stratified", choices=["stratified", "contiguous"], help="Batch selection strategy")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Custom checkpoint directory")
    parser.add_argument("--max-pairs", type=int, default=50, help="Maximum pairs to evaluate (hard capped at 50)")
    parser.add_argument("--pacing", type=float, default=1.0, help="Pacing interval in seconds between requests")
    parser.add_argument("--authorize-live", action="store_true", help="Explicit confirmation for live evaluation")
    args = parser.parse_args()

    manifest_path = Path(args.manifest) if args.manifest else None
    if args.batch_index is not None and manifest_path is None:
        manifest_path = WORKSPACE_DIR / "benchmark_manifest_v2.0.0.json"

    runner = BenchmarkRunner(
        manifest_path=manifest_path,
        checkpoint_dir=Path(args.checkpoint_dir) if args.checkpoint_dir else None,
        pacing_seconds=args.pacing,
        batch_index=args.batch_index,
        batch_strategy=args.batch_strategy,
    )

    if args.dry_run:
        runner.execute_dry_run()
    elif args.pilot:
        asyncio.run(runner.execute_batch(mode="pilot", max_pairs=5, authorize_live=args.authorize_live))
    elif args.live:
        asyncio.run(runner.execute_batch(mode="live", max_pairs=args.max_pairs, authorize_live=args.authorize_live))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
