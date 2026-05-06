"""Запускає всі бенчмарки і зберігає результати в JSON."""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np

import task1_html_tags
import task1_array_stats
import task1_matrix_mult
import task2_image_pipeline


WORKER_COUNTS = [1, 2, 4, 8]


def run_task1_html() -> dict:
    print("\n=== Task 1.1: HTML Tag Counting ===")
    root = Path("test_data/html")
    res = {"task": "html_tags", "n_files": len(task1_html_tags.list_html_files(root)),
           "sequential": None, "patterns": {}}

    c, t = task1_html_tags.run_sequential(root)
    print(f"  seq: t={t:.2f}s ({sum(c.values())} тегів)")
    res["sequential"] = {"time": t}

    for pattern_name, fn in [("map_reduce", task1_html_tags.run_map_reduce),
                              ("fork_join", task1_html_tags.run_fork_join),
                              ("worker_pool", task1_html_tags.run_worker_pool)]:
        res["patterns"][pattern_name] = {}
        for w in WORKER_COUNTS:
            c, t = fn(root, w)
            print(f"  {pattern_name:12s} w={w}: t={t:.2f}s")
            res["patterns"][pattern_name][str(w)] = {"time": t}
    return res


def run_task1_array() -> dict:
    print("\n=== Task 1.2: Array Statistics ===")
    arr = np.load("test_data/array.npy")
    res = {"task": "array_stats", "n_elements": int(arr.size),
           "sequential": None, "patterns": {}}

    s, t = task1_array_stats.run_sequential(arr)
    print(f"  seq: t={t:.2f}s (mean={s.mean:.2f})")
    res["sequential"] = {"time": t}

    for pattern_name, fn in [("map_reduce", task1_array_stats.run_map_reduce),
                              ("fork_join", task1_array_stats.run_fork_join),
                              ("worker_pool", task1_array_stats.run_worker_pool)]:
        res["patterns"][pattern_name] = {}
        for w in WORKER_COUNTS:
            s, t = fn(arr, w)
            print(f"  {pattern_name:12s} w={w}: t={t:.2f}s")
            res["patterns"][pattern_name][str(w)] = {"time": t}
    return res


def run_task1_matrix() -> dict:
    print("\n=== Task 1.3: Matrix Multiplication ===")
    A = np.load("test_data/matA.npy")
    B = np.load("test_data/matB.npy")
    res = {"task": "matrix_mult", "size": list(A.shape),
           "sequential": None, "patterns": {}}

    C_ref, t = task1_matrix_mult.run_sequential(A, B)
    print(f"  seq: t={t:.2f}s (C[0,0]={C_ref[0,0]:.4f})")
    res["sequential"] = {"time": t}

    for pattern_name, fn in [("map_reduce", task1_matrix_mult.run_map_reduce),
                              ("fork_join", task1_matrix_mult.run_fork_join),
                              ("worker_pool", task1_matrix_mult.run_worker_pool)]:
        res["patterns"][pattern_name] = {}
        for w in WORKER_COUNTS:
            C_par, t = fn(A, B, w)
            ok = np.allclose(C_par, C_ref)
            print(f"  {pattern_name:12s} w={w}: t={t:.2f}s {'✓' if ok else '✗'}")
            res["patterns"][pattern_name][str(w)] = {"time": t, "correct": bool(ok)}
    return res


def run_task2_images() -> dict:
    print("\n=== Task 2: Image Processing ===")
    in_dir = Path("test_data/images")
    out_dir = Path("test_data/output_images")
    files = sorted(in_dir.glob("*.jpg"))
    res = {"task": "image_processing", "n_images": len(files),
           "sequential": None, "patterns": {"pipeline": {}, "producer_consumer": {}}}

    n, t = task2_image_pipeline.run_sequential(in_dir, out_dir)
    print(f"  seq: {n} images, t={t:.2f}s")
    res["sequential"] = {"time": t}

    # Pipeline (4 етапи фіксовано, тестуємо як baseline)
    n, t = task2_image_pipeline.run_pipeline(in_dir, out_dir)
    print(f"  pipeline (4 stages): {n} images, t={t:.2f}s")
    res["patterns"]["pipeline"]["4"] = {"time": t}

    # Producer-Consumer для різних W
    for w in WORKER_COUNTS:
        n, t = task2_image_pipeline.run_producer_consumer(in_dir, out_dir, w)
        print(f"  producer-consumer w={w}: t={t:.2f}s")
        res["patterns"]["producer_consumer"][str(w)] = {"time": t}
    return res


def main():
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    # Виводимо змінну середовища
    import os
    print(f"IO_LATENCY_MS = {os.environ.get('IO_LATENCY_MS', '15')}")

    all_results = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "worker_counts": WORKER_COUNTS,
            "io_latency_ms": int(os.environ.get("IO_LATENCY_MS", "15")),
            "notes": ("Симульована I/O-затримка 15 мс/одиниця для всіх атомарних "
                      "операцій. Це моделює реалістичний серверний додаток із "
                      "I/O-операціями (web-scraping, streaming chunks, distributed "
                      "matrix mult). У такому режимі ThreadPoolExecutor дає лінійне "
                      "прискорення, бо під час time.sleep GIL вільний."),
        },
        "results": [
            run_task1_html(),
            run_task1_array(),
            run_task1_matrix(),
            run_task2_images(),
        ],
    }

    out_path = out_dir / "benchmark.json"
    out_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False))
    print(f"\n✓ Saved: {out_path}")


if __name__ == "__main__":
    main()
