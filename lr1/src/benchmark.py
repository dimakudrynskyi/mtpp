"""Запуск бенчмарків і збереження у JSON."""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

sys.path.insert(0, str(Path(__file__).parent))
from cpu_bound import (
    pi_sequential, pi_parallel,
    factorize_sequential, factorize_parallel,
    primes_sequential, primes_parallel,
)
from memory_bound import make_matrix, transpose_sequential, transpose_parallel
from io_bound import count_sequential, count_parallel


WORKER_COUNTS = [1, 2, 4, 8]


def run_pi(total_points: int) -> dict:
    print(f"\n=== Monte-Carlo π, N={total_points:,} ===")
    res = {"task": "pi_montecarlo", "N": total_points,
           "sequential": None, "threads": {}, "processes": {}}
    pi, t = pi_sequential(total_points)
    print(f"  seq        π={pi:.5f}  t={t:.3f}s")
    res["sequential"] = {"value": pi, "time": t}
    for w in WORKER_COUNTS:
        pi, t = pi_parallel(total_points, w, ThreadPoolExecutor)
        print(f"  thr w={w}  π={pi:.5f}  t={t:.3f}s")
        res["threads"][str(w)] = {"value": pi, "time": t}
    for w in WORKER_COUNTS:
        pi, t = pi_parallel(total_points, w, ProcessPoolExecutor)
        print(f"  proc w={w} π={pi:.5f}  t={t:.3f}s")
        res["processes"][str(w)] = {"value": pi, "time": t}
    return res


def run_factorize(numbers: list[int]) -> dict:
    print(f"\n=== Factorization, {len(numbers)} numbers ===")
    res = {"task": "factorization", "N": len(numbers),
           "sequential": None, "threads": {}, "processes": {}}
    _, t = factorize_sequential(numbers)
    print(f"  seq        t={t:.3f}s")
    res["sequential"] = {"time": t}
    for w in WORKER_COUNTS:
        _, t = factorize_parallel(numbers, w, ThreadPoolExecutor)
        print(f"  thr w={w}  t={t:.3f}s")
        res["threads"][str(w)] = {"time": t}
    for w in WORKER_COUNTS:
        _, t = factorize_parallel(numbers, w, ProcessPoolExecutor)
        print(f"  proc w={w} t={t:.3f}s")
        res["processes"][str(w)] = {"time": t}
    return res


def run_primes(lo: int, hi: int) -> dict:
    print(f"\n=== Primes in [{lo}, {hi}) ===")
    res = {"task": "primes", "lo": lo, "hi": hi,
           "sequential": None, "threads": {}, "processes": {}}
    cnt, t = primes_sequential(lo, hi)
    print(f"  seq        cnt={cnt}  t={t:.3f}s")
    res["sequential"] = {"value": cnt, "time": t}
    for w in WORKER_COUNTS:
        cnt, t = primes_parallel(lo, hi, w, ThreadPoolExecutor)
        print(f"  thr w={w}  t={t:.3f}s")
        res["threads"][str(w)] = {"value": cnt, "time": t}
    for w in WORKER_COUNTS:
        cnt, t = primes_parallel(lo, hi, w, ProcessPoolExecutor)
        print(f"  proc w={w} t={t:.3f}s")
        res["processes"][str(w)] = {"value": cnt, "time": t}
    return res


def run_transpose(n: int, run_processes: bool = False) -> dict:
    print(f"\n=== Matrix transpose {n}x{n} ===")
    res = {"task": "transpose", "size": n,
           "sequential": None, "threads": {}, "processes": {}}
    M = make_matrix(n)
    _, t = transpose_sequential(M)
    print(f"  seq        t={t:.3f}s")
    res["sequential"] = {"time": t}
    for w in WORKER_COUNTS:
        _, t = transpose_parallel(M, w, ThreadPoolExecutor)
        print(f"  thr w={w}  t={t:.3f}s")
        res["threads"][str(w)] = {"time": t}
    if run_processes:
        for w in WORKER_COUNTS:
            _, t = transpose_parallel(M, w, ProcessPoolExecutor)
            print(f"  proc w={w} t={t:.3f}s")
            res["processes"][str(w)] = {"time": t}
    return res


def run_io(root: Path) -> dict:
    print(f"\n=== I/O word count in {root} ===")
    res = {"task": "io_wordcount", "root": str(root),
           "sequential": None, "threads": {}, "processes": {}}
    total, t = count_sequential(root)
    print(f"  seq        words={total}  t={t:.3f}s")
    res["sequential"] = {"value": total, "time": t}
    for w in WORKER_COUNTS:
        total, t = count_parallel(root, w, ThreadPoolExecutor)
        print(f"  thr w={w}  t={t:.3f}s")
        res["threads"][str(w)] = {"value": total, "time": t}
    for w in WORKER_COUNTS:
        total, t = count_parallel(root, w, ProcessPoolExecutor)
        print(f"  proc w={w} t={t:.3f}s")
        res["processes"][str(w)] = {"value": total, "time": t}
    return res
