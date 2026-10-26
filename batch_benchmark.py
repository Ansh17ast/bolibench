#!/usr/bin/env python3
"""CLI runner for BoliBench reproducible offline-first batch benchmark.

Usage:
  # 1. Create a curated 5-pair sample manifest:
  python batch_benchmark.py --create-sample-manifest data/curated_batch_manifest.json

  # 2. Run offline dry-run pre-flight inspection (ZERO network calls, ZERO credits):
  python batch_benchmark.py --manifest data/curated_batch_manifest.json --dry-run

  # 3. Live 5-pair pilot evaluation (explicit authorization required):
  python batch_benchmark.py --manifest data/curated_batch_manifest.json --live --max-pairs 5 --confirm
"""

import argparse
import asyncio
import logging
from pathlib import Path
import sys
from typing import Optional

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Ensure UTF-8 output encoding for terminals
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from app.config import get_settings
from app.evaluation.batch_manifest import (
    AudioSourceType,
    BatchManifest,
    create_sample_manifest,
    load_manifest,
    validate_manifest,
)
from app.evaluation.batch_runner import (
    BatchBenchmarkRunner,
    BatchDryRunReport,
    BatchExecutionConfig,
    LiveExecutionBlockedError,
    MAX_PILOT_PAIRS,
)

logger = logging.getLogger("bolibench.cli")


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def print_banner() -> None:
    print("=" * 72)
    print("  BoliBench Paired-Contrast Batch Benchmark Runner (Gnani Prisma STT v3)")
    print("=" * 72)


def print_dry_run_report(report: BatchDryRunReport) -> None:
    print("\n" + "-" * 72)
    print("  DRY-RUN INSPECTION AUDIT REPORT (Zero Credits Consumed)")
    print("-" * 72)
    print(f"  Batch ID               : {report.batch_id}")
    print(f"  Manifest               : {report.manifest_name}")
    print(f"  Total Manifest Pairs   : {report.total_pairs}")
    print(f"  Ready Pairs (Audited)  : {report.ready_pairs}")
    print(f"  Planned Provider Calls : {report.planned_requests} (2 calls per eligible pair)")
    print(f"  Invalid Audio / Headers: {report.invalid_pairs}")
    print(f"  Duplicated Pairs       : {report.duplicated_pairs}")
    print(f"  Missing Ground Truth   : {report.missing_ground_truth}")
    print(f"  Output Directory       : {report.output_dir}")
    print(f"  Live Calls Dispatched  : 0 (Offline-first)")

    print("\n  [Audio Source Modality Breakdown]")
    for src, cnt in sorted(report.counts_by_source.items()):
        share = (cnt / max(1, report.total_pairs)) * 100
        print(f"    - {src:<24}: {cnt:>2} pairs ({share:>5.1f}%)")

    print("\n  [Scenario Category Distribution]")
    for cat, cnt in sorted(report.counts_by_category.items()):
        print(f"    - {cat:<28}: {cnt:>2} pairs -> {cnt * 2:>2} planned calls")

    print("\n  [Verification Status]")
    if report.validation_passed:
        print("  -> STATUS: PASS - Manifest and audio files conform to backend constraints.")
    else:
        print("  -> STATUS: WARNING - Some items were excluded due to validation errors.")
        if report.invalid_details:
            print("\n  [Excluded Items Detail]")
            for inv in report.invalid_details:
                print(f"    - {inv.get('pair_id', 'unknown')}: [{inv.get('error_type')}] {inv.get('message')}")

    print("\n  [Integrity Notice]")
    print(f"  NOTE: {report.disclaimer}")
    print("-" * 72 + "\n")


def print_batch_summary(summary: dict) -> None:
    print("\n" + "=" * 72)
    print(f"  BATCH BENCHMARK EXECUTION SUMMARY: {summary['batch_id']}")
    print("=" * 72)
    print(f"  Mode                       : {summary['execution_mode'].upper()}")
    print(f"  Total Pairs Evaluated      : {summary['total_pairs']}")
    print(f"  Completed Pairs            : {summary['completed_pairs']}")
    print(f"  Failed Pairs               : {summary['failed_pairs']}")
    print(f"  Incomplete Pairs           : {summary['incomplete_pairs']}")

    print("\n  [Request Accounting]")
    print(f"  Attempted Provider Calls   : {summary['attempted_requests']}")
    print(f"  Successful 200 OK Calls    : {summary['successful_requests']}")
    print(f"  Failed Calls               : {summary['failed_requests']}")
    print(f"  Retried Calls              : {summary['retried_requests']} (Automatic retries disabled)")

    print("\n  [Accuracy & Robustness Metrics]")
    raw_wer_str = f"{summary['mean_raw_wer'] * 100:.2f}%" if summary['mean_raw_wer'] is not None else "N/A"
    norm_wer_str = f"{summary['mean_normalized_wer'] * 100:.2f}%" if summary['mean_normalized_wer'] is not None else "N/A"
    crit_acc_str = f"{summary['critical_field_accuracy'] * 100:.1f}%" if summary['critical_field_accuracy'] is not None else "N/A"
    dist_str = f"{summary['distinction_preservation_rate'] * 100:.1f}%" if summary['distinction_preservation_rate'] is not None else "N/A"
    task_str = f"{summary['task_success_rate'] * 100:.1f}%" if summary['task_success_rate'] is not None else "N/A"

    print(f"  Mean Raw WER               : {raw_wer_str}")
    print(f"  Mean Cross-Script Norm WER : {norm_wer_str}")
    print(f"  Critical Field Accuracy    : {crit_acc_str}")
    print(f"  Distinction Preservation   : {dist_str}")
    print(f"  Downstream Task Success    : {task_str}")

    if summary.get("latency_stats", {}).get("mean_ms"):
        lstats = summary["latency_stats"]
        print("\n  [Latency (Round-Trip Client Elapsed)]")
        print(f"  Mean Latency / Pair        : {lstats['mean_ms']} ms")
        print(f"  Min / Max Latency          : {lstats['min_ms']} ms / {lstats['max_ms']} ms")

    if summary.get("failure_counts"):
        print("\n  [Failure Taxonomy]")
        for ftype, cnt in sorted(summary["failure_counts"].items()):
            print(f"    - {ftype:<24}: {cnt}")

    print("\n  [Integrity Notice]")
    print(f"  NOTE: {summary['disclaimer']}")
    print("=" * 72 + "\n")


async def run_cli() -> int:
    parser = argparse.ArgumentParser(
        description="BoliBench Reproducible Paired-Contrast Batch Benchmark Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("benchmark_manifest.json") if Path("benchmark_manifest.json").exists() else None,
        help="Path to the JSON batch manifest file (default: benchmark_manifest.json).",
    )
    parser.add_argument(
        "--create-sample-manifest",
        type=Path,
        metavar="OUTPUT_PATH",
        help="Generate a curated 5-pair sample manifest linking local test audio and exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform offline pre-flight validation and planned request accounting. ZERO provider calls.",
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Select five validated pairs across categories and execute pilot (max 10 requests).",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Execute live provider calls to Gnani Prisma STT v3. Capped at 5 pairs for pilot.",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Maximum number of test pairs to evaluate (pilot strictly capped at 5).",
    )
    parser.add_argument(
        "--confirm",
        "--authorize-live",
        dest="confirm",
        action="store_true",
        help="Explicit confirmation authorizing the planned number of provider calls.",
    )
    parser.add_argument(
        "--pacing-delay",
        type=float,
        default=1.0,
        help="Configurable delay (seconds) between sequential provider calls (default: 1.0s).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_results"),
        help="Output directory for CSV, JSON, and Markdown reports (default: ./benchmark_results).",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resume; do not skip already completed pairs from a previous run.",
    )
    parser.add_argument(
        "--batch-id",
        type=str,
        default=None,
        help="Optional custom stable batch ID.",
    )
    parser.add_argument(
        "--no-pilot",
        action="store_true",
        help="Override the 5-pair pilot limit if authorized for extended benchmarking.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable detailed debug logging.",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)
    print_banner()

    # If --pilot is passed, configure pilot execution mode
    if args.pilot:
        args.live = True
        args.max_pairs = 5

    # Step 1: Create sample manifest if requested
    if args.create_sample_manifest:
        out_p = args.create_sample_manifest
        manifest = create_sample_manifest(out_p)
        print(f"[+] Sample 5-pair manifest successfully created at: {out_p.resolve()}")
        print(f"    Total items: {len(manifest.items)}")
        print("    Categories : negation, corrections_and_mind_changes, command_intent, named_entities, time_extraction")
        print("\nNext step: Run an offline dry-run:")
        print(f"  python batch_benchmark.py --manifest {out_p} --dry-run")
        return 0

    if not args.manifest:
        parser.print_help()
        print("\n[!] Error: Please specify either --manifest PATH or --create-sample-manifest PATH.")
        return 1

    if not args.manifest.exists():
        print(f"\n[!] Error: Manifest file not found at: {args.manifest.resolve()}")
        return 1

    # Load settings
    settings = get_settings()

    config = BatchExecutionConfig(
        live=args.live,
        max_pairs=args.max_pairs,
        pilot_mode=not args.no_pilot,
        pacing_delay_seconds=args.pacing_delay,
        confirm=args.confirm,
        resume=not args.no_resume,
        output_dir=args.output_dir,
        batch_id=args.batch_id,
    )

    runner = BatchBenchmarkRunner(config=config, settings=settings)

    # Step 2: Dry-run mode
    if args.dry_run or not args.live:
        print("\n[*] Executing Offline Dry-Run Pre-Flight Validation...")
        try:
            dry_report = runner.run_dry(manifest=args.manifest, base_dir=args.manifest.parent)
            print_dry_run_report(dry_report)

            if not args.live:
                print("[*] Dry-run complete. Live execution was not requested (--live omitted).")
                print("    Zero provider calls dispatched. Zero credits consumed.\n")
                return 0
        except Exception as exc:
            print(f"\n[!] Dry-run validation failed: {exc}")
            return 1

    # Step 3: Live execution mode
    if args.live:
        print("\n[*] Live Execution Mode Requested.")
        try:
            # Pre-flight check
            report = validate_manifest(args.manifest, base_dir=args.manifest.parent)
            runner.check_live_preflight_guards(report.ready_count)
        except LiveExecutionBlockedError as exc:
            print(f"\n[!] BLOCKED: Live execution guard stopped execution:")
            print(f"    {exc}\n")
            return 2
        except Exception as exc:
            print(f"\n[!] Pre-flight verification error: {exc}\n")
            return 1

        target_pairs = min(report.ready_count, args.max_pairs) if args.max_pairs else report.ready_count
        expected_calls = target_pairs * 2
        print(f"    Target Pairs          : {target_pairs}")
        print(f"    Expected Provider Calls: {expected_calls} (sequential, {args.pacing_delay}s pacing)")
        print(f"    Pacing Delay          : {args.pacing_delay}s")
        print(f"    Output Directory      : {args.output_dir.resolve()}")
        print("\n[*] Dispatching live benchmark requests...")

        def on_progress(p: dict) -> None:
            print(f"    -> [{p['index']}/{p['total']}] Evaluated pair {p['pair_id']}: {p['status']}")

        summary = await runner.run(
            manifest=args.manifest,
            base_dir=args.manifest.parent,
            progress_callback=on_progress,
        )
        print_batch_summary(summary)
        bid = summary['batch_id']
        out_path = args.output_dir.resolve()
        print(f"[+] Results saved to directory: {out_path}")
        print(f"    - CSV     : {out_path / f'{bid}_results.csv'}")
        print(f"    - JSON    : {out_path / f'{bid}_results.json'}")
        print(f"    - Markdown: {out_path / f'{bid}_summary.md'}")
        return 0

    return 0


def main() -> None:
    try:
        exit_code = asyncio.run(run_cli())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n[!] Execution interrupted by operator. Partial results and state saved.")
        sys.exit(130)


if __name__ == "__main__":
    main()
