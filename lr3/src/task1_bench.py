"""Бенчмарк продуктивності рішень задачі 1: throughput vs кількість потоків."""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from task1_bank import (
    make_accounts, total_balance, run_threads,
    transfer_ordered_locks, transfer_global_lock, transfer_try_lock,
    N_TRANSFERS_PER_THREAD,
)

THREAD_COUNTS = [1, 2, 4, 8, 16, 64, 256, 1024]

SOLUTIONS = [
    ("ordered_locks", transfer_ordered_locks),
    ("global_lock", transfer_global_lock),
    ("try_lock", transfer_try_lock),
]


def bench():
    results = {"thread_counts": THREAD_COUNTS, "solutions": {}}
    for name, fn in SOLUTIONS:
        print(f"\n=== {name} ===")
        results["solutions"][name] = {}
        for n in THREAD_COUNTS:
            accounts = make_accounts()
            t_initial = total_balance(accounts)
            elapsed, ok, attempts = run_threads(
                accounts, fn, n_threads=n,
                n_transfers=N_TRANSFERS_PER_THREAD,
                deadlock_timeout=60.0)
            t_final = total_balance(accounts)
            integrity = abs(t_final - t_initial) < 0.01
            throughput = ok / elapsed if elapsed > 0 else 0
            print(f"  threads={n:5d}  t={elapsed:6.2f}s  ok={ok:6d}/{attempts}"
                  f"  thr={throughput:8.0f}/s  integrity={'✓' if integrity else '✗'}")
            results["solutions"][name][str(n)] = {
                "time": elapsed, "successful": ok, "attempts": attempts,
                "throughput": throughput, "integrity_ok": integrity,
            }
    Path("results/task1_bench.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False))
    print("\n✓ Saved: results/task1_bench.json")


if __name__ == "__main__":
    bench()
