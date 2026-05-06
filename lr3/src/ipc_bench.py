"""Запуск усіх 4 методів IPC і збереження зведених результатів."""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ipc_pipe import run_pipe
from ipc_shm import run_shm
from ipc_tcp import run_tcp
from ipc_uds import run_uds


def main():
    n_messages = 1000
    results = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_messages": n_messages,
        },
        "methods": [],
    }

    print("\n=== IPC Benchmark Suite ===\n")

    print("[1/4] multiprocessing.Pipe (Python ↔ Python)")
    res = run_pipe(n_messages)
    print(f"  avg RTT: {res['avg_rtt_us']:.1f} µs, total: {res['total_time_s']:.2f}s")
    results["methods"].append(res)

    print("\n[2/4] shared_memory + Event (Python ↔ Python)")
    res = run_shm(n_messages)
    print(f"  avg RTT: {res['avg_rtt_us']:.1f} µs, total: {res['total_time_s']:.2f}s")
    results["methods"].append(res)

    print("\n[3/4] Unix Domain Socket + JSON (Python ↔ Python)")
    res = run_uds(n_messages)
    print(f"  avg RTT: {res['avg_rtt_us']:.1f} µs, total: {res['total_time_s']:.2f}s")
    results["methods"].append(res)

    print("\n[4/4] TCP + JSON (Python ↔ Node.js КРОС-МОВНИЙ)")
    res = run_tcp(n_messages)
    print(f"  avg RTT: {res['avg_rtt_us']:.1f} µs, total: {res['total_time_s']:.2f}s")
    results["methods"].append(res)

    Path("results/ipc_all.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False))
    print("\n✓ Saved: results/ipc_all.json")

    # Зведена таблиця
    print("\n=== Підсумок ===")
    print(f"{'Метод':<35} {'Avg RTT (µs)':>14} {'Total (s)':>10}")
    for r in results["methods"]:
        print(f"{r['method']:<35} {r['avg_rtt_us']:>14.1f} {r['total_time_s']:>10.2f}")


if __name__ == "__main__":
    main()
