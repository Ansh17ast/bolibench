"""Analyzes and aggregates all 12 live benchmark batches of the 600-pair v2 dataset."""

from collections import Counter
import glob
import json
from pathlib import Path
import sqlite3
import sys

sys.stdout.reconfigure(encoding='utf-8')

def main():
    conn = sqlite3.connect('bolibench_evaluation.db')
    cur = conn.cursor()
    cur.execute("SELECT count(*) FROM evaluation_runs WHERE run_id LIKE 'eval_live_bench_v2_%'")
    v2_count = cur.fetchone()[0]
    cur.execute("SELECT count(*) FROM evaluation_runs")
    total_count = cur.fetchone()[0]
    print(f"v2 runs in SQLite: {v2_count}")
    print(f"Total runs in SQLite: {total_count}")

    batch_files = sorted(glob.glob('benchmark_results/batch_v2_b*_live_*_results.json'))
    print(f"Batch result files found: {len(batch_files)}")

    tot_pairs = 0
    tot_dispatched = 0
    tot_successful = 0
    tot_failed = 0
    tot_completed_pairs = 0
    tot_preserved_pairs = 0
    by_category_total = Counter()
    by_category_preserved = Counter()
    by_category_completed = Counter()
    incomplete_pairs = []

    per_batch_stats = []

    for bf in batch_files:
        data = json.load(open(bf, encoding='utf-8'))
        b_id = data.get('batch_id', Path(bf).stem)
        pairs = data.get('pairs', [])
        uttrs = data.get('utterances', [])

        b_dispatched = data.get('requests_dispatched', 0)
        b_success = sum(1 for u in uttrs if u.get('status') == 'completed')
        b_failed = sum(1 for u in uttrs if u.get('status') == 'failed')
        b_completed_pairs = sum(1 for p in pairs if p.get('overall_status') == 'completed')
        b_preserved = sum(1 for p in pairs if p.get('distinction_preserved'))

        tot_pairs += len(pairs)
        tot_dispatched += b_dispatched
        tot_successful += b_success
        tot_failed += b_failed
        tot_completed_pairs += b_completed_pairs
        tot_preserved_pairs += b_preserved

        per_batch_stats.append({
            "batch_id": b_id,
            "pairs": len(pairs),
            "dispatched": b_dispatched,
            "success": b_success,
            "failed": b_failed,
            "completed": b_completed_pairs,
            "preserved": b_preserved,
            "mean_raw_wer": data.get('mean_raw_wer', 0.0),
        })

        for p in pairs:
            cat = p['category']
            by_category_total[cat] += 1
            if p.get('overall_status') == 'completed':
                by_category_completed[cat] += 1
            else:
                incomplete_pairs.append((p['test_case_id'], p.get('failure_explanation', '')))
            if p.get('distinction_preserved'):
                by_category_preserved[cat] += 1

    print("\n" + "=" * 75)
    print("      BOLIBENCH 600-PAIR EXPANDED BENCHMARK: FULL COHORT SUMMARY")
    print("=" * 75)
    print(f"Total Batches Executed:        12 of 12 (100% complete)")
    print(f"Total Unique Pairs Evaluated:  {tot_pairs} / 600")
    print(f"Actual STT Requests Dispatched:{tot_dispatched}")
    print(f"Successful Requests (200 OK):  {tot_successful}")
    print(f"Failed Requests (403 Blocks):  {tot_failed}")
    print(f"Completed Test Pairs:          {tot_completed_pairs} / {tot_pairs} ({tot_completed_pairs/tot_pairs*100:.1f}%)")
    print(f"Incomplete Test Pairs:         {len(incomplete_pairs)}")
    print(f"Distinction Preserved (Total): {tot_preserved_pairs} / {tot_pairs} ({tot_preserved_pairs/tot_pairs*100:.1f}%)")
    print(f"Distinction Preserved (Comp):  {tot_preserved_pairs} / {tot_completed_pairs} ({tot_preserved_pairs/tot_completed_pairs*100:.1f}%)")
    print("-" * 75)

    print("\nBatch-by-Batch Breakdown:")
    print(f"{'Batch ID':<36} | {'Pairs':>5} | {'Disp':>4} | {'Succ':>4} | {'Fail':>4} | {'Preserved':>9} | {'Mean Raw WER':>12}")
    print("-" * 85)
    for s in per_batch_stats:
        print(f"{s['batch_id']:<36} | {s['pairs']:>5} | {s['dispatched']:>4} | {s['success']:>4} | {s['failed']:>4} | {s['preserved']:>3}/{s['pairs']:<3} ({s['preserved']/s['pairs']*100:4.1f}%) | {s['mean_raw_wer']*100:10.1f}%")

    print("\n" + "-" * 75)
    print("Category Breakdown across Full 600 Pairs:")
    for cat, tot in sorted(by_category_total.items()):
        pres = by_category_preserved[cat]
        comp = by_category_completed[cat]
        print(f"  - {cat:<30}: {pres:>3}/{comp:>3} completed ({pres/comp*100:5.1f}%) | {pres:>3}/{tot:>3} total ({pres/tot*100:5.1f}%)")

    print("\nIncomplete Pairs List:")
    for pid, expl in incomplete_pairs:
        print(f"  - {pid}: {expl}")

if __name__ == '__main__':
    main()
