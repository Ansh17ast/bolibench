"""Sequential orchestrator for all remaining BoliBench v2 live batches (Batches 4 to 12)."""

import argparse
import asyncio
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.benchmark_runner import BenchmarkRunner, WORKSPACE_DIR
from app.evaluation.batch_manifest import load_manifest

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_all_v2_batches")

async def run_batch_range(start_batch: int = 4, end_batch: int = 12):
    manifest_path = WORKSPACE_DIR / "benchmark_manifest_v2.0.0.json"
    batch_summaries = []

    print("\n" + "=" * 70)
    print(f"   BOLIBENCH v2.0.0 FULL LIVE BENCHMARK ORCHESTRATOR")
    print(f"   Executing Batches {start_batch} to {end_batch} (Target: {(end_batch - start_batch + 1) * 50} pairs)")
    print("=" * 70 + "\n")

    for b in range(start_batch, end_batch + 1):
        print("\n" + "#" * 70)
        print(f"### LAUNCHING BATCH {b:02d} OF 12 (Pairs {(b-1)*50 + 1} to {b*50}) ###")
        print("#" * 70 + "\n")

        runner = BenchmarkRunner(
            manifest_path=manifest_path,
            batch_index=b,
            batch_strategy="stratified",
            pacing_seconds=1.0,
        )

        try:
            summary = await runner.execute_batch(mode="live", max_pairs=50, authorize_live=True)
            batch_summaries.append(summary)
            print(f"\n[PASS] Batch {b:02d} completed successfully. Distinction preserved: {summary['distinction_preserved_count']}/{summary['total_pairs_evaluated']}")
        except Exception as exc:
            logger.error(f"Batch {b:02d} encountered critical error: {exc}", exc_info=True)
            batch_summaries.append({
                "batch_index": b,
                "error": str(exc),
                "distinction_preserved_count": 0,
                "total_pairs_evaluated": 50,
            })

    print("\n" + "=" * 70)
    print("   BOLIBENCH v2.0.0 ALL BATCHES EXECUTION COMPLETE")
    print("=" * 70)
    for s in batch_summaries:
        b_id = s.get("batch_id") or f"batch_{s.get('batch_index', '?')}"
        pres = s.get("distinction_preserved_count", 0)
        tot = s.get("total_pairs_evaluated", 50)
        print(f"  - {b_id:<36}: {pres}/{tot} preserved")
    print("=" * 70 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Run all remaining BoliBench v2 live batches")
    parser.add_argument("--start-batch", type=int, default=4, help="Starting batch index (default: 4)")
    parser.add_argument("--end-batch", type=int, default=12, help="Ending batch index (default: 12)")
    args = parser.parse_args()

    asyncio.run(run_batch_range(start_batch=args.start_batch, end_batch=args.end_batch))

if __name__ == "__main__":
    main()
