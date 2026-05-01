"""Основний бенчмарк: запускає всі версії симуляції і створює графіки."""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np

from crystal import CrystalConfig
from sim_sequential import run_sequential
from sim_naive import run_naive_threaded
from sim_solved import run_solved_threaded
from sim_deadlock import run_deadlock_threaded
from sim_pool import run_pool_threaded
import visualizer as viz


def main():
    Path("results").mkdir(exist_ok=True)
    Path("plots").mkdir(exist_ok=True)
    Path("snapshots").mkdir(exist_ok=True)
    Path("animations").mkdir(exist_ok=True)

    # ============================================================
    # Етап 1: Демонстрація race / deadlock / solved (мала симуляція)
    # ============================================================
    print("\n=== Етап 1: Демонстрація race / deadlock / рішень ===")
    demo_cfg = CrystalConfig(grid_size=20, n_particles=200,
                             n_steps=100, snapshot_every=20, seed=42)
    print(f"Параметри: grid={demo_cfg.grid_size}, N={demo_cfg.n_particles}, "
          f"steps={demo_cfg.n_steps}")

    print("\n[1/4] sequential (baseline)...")
    seq_r = run_sequential(demo_cfg)
    print(f"      t={seq_r.elapsed:.2f}s  initial={seq_r.initial_count}  "
          f"final={seq_r.final_count}  integrity={'OK' if seq_r.integrity_ok else 'FAIL'}")

    print("\n[2/4] naive_threaded (RACE expected)...")
    naive_r = run_naive_threaded(demo_cfg)
    print(f"      t={naive_r.elapsed:.2f}s  initial={naive_r.initial_count}  "
          f"final={naive_r.final_count}  loss/gain={naive_r.final_count - naive_r.initial_count:+d}")

    print("\n[3/4] solved_threaded (per-cell locks + ordering)...")
    solved_r = run_solved_threaded(demo_cfg)
    print(f"      t={solved_r.elapsed:.2f}s  integrity={'OK' if solved_r.integrity_ok else 'FAIL'}")

    print("\n[4/4] deadlock_threaded (NO ordering — deadlock expected)...")
    # Менша сітка + більше частинок = висока конкуренція = надійний deadlock
    deadlock_cfg = CrystalConfig(grid_size=8, n_particles=200,
                                  n_steps=50, snapshot_every=10, seed=42)
    deadlock_r = run_deadlock_threaded(deadlock_cfg, timeout_s=8.0)
    print(f"      t={deadlock_r.elapsed:.2f}s  "
          f"completed={deadlock_r.extra['completed_steps']}/{deadlock_cfg.n_steps}  "
          f"deadlocked={deadlock_r.extra['deadlocked']}")

    # Зберігаємо JSON-результати демонстрації
    demo_results = {}
    for name, r in [("sequential", seq_r), ("naive", naive_r),
                     ("solved", solved_r), ("deadlock", deadlock_r)]:
        demo_results[name] = {
            "method": r.method_name,
            "elapsed": r.elapsed,
            "initial_count": r.initial_count,
            "final_count": r.final_count,
            "integrity_ok": r.integrity_ok,
            "discrepancy": r.final_count - r.initial_count,
            "extra": r.extra,
        }
    Path("results/demo.json").write_text(
        json.dumps(demo_results, indent=2, ensure_ascii=False))

    # Візуалізація
    print("\n=== Візуалізація ===")
    viz.save_snapshot_grid(seq_r, Path("snapshots/sequential_grid.png"))
    viz.save_snapshot_grid(solved_r, Path("snapshots/solved_grid.png"))
    viz.save_animation(seq_r, Path("animations/sequential.gif"))
    viz.save_animation(solved_r, Path("animations/solved.gif"))
    viz.save_initial_final(seq_r, Path("plots/01_sequential_init_final.png"))
    viz.save_initial_final(solved_r, Path("plots/02_solved_init_final.png"))
    viz.save_invariant_check([seq_r, naive_r, solved_r],
                              Path("plots/03_integrity_check.png"))
    print("  ✓ snapshots, animations, plots saved")

    # ============================================================
    # Етап 2: Масштабування (Worker Pool, ускладнена задача)
    # ============================================================
    print("\n=== Етап 2: Масштабування Worker Pool ===")
    scale_cfg = CrystalConfig(grid_size=30, n_particles=500,
                              n_steps=100, snapshot_every=20, seed=42)
    print(f"Параметри: grid={scale_cfg.grid_size}, N={scale_cfg.n_particles}, "
          f"steps={scale_cfg.n_steps}")

    pool_results = {}
    for w in [1, 2, 4, 8]:
        r = run_pool_threaded(scale_cfg, n_workers=w)
        print(f"  pool W={w}: t={r.elapsed:.2f}s  "
              f"integrity={'OK' if r.integrity_ok else 'FAIL'}")
        pool_results[f"pool_W{w}"] = {
            "method": r.method_name,
            "n_workers": w,
            "elapsed": r.elapsed,
            "integrity_ok": r.integrity_ok,
        }

    # Naive на тих самих параметрах для порівняння
    print("  naive 1 thread / particle (для порівняння)...")
    naive_scale = run_naive_threaded(scale_cfg)
    pool_results["naive_per_particle"] = {
        "method": naive_scale.method_name,
        "n_threads": scale_cfg.n_particles,
        "elapsed": naive_scale.elapsed,
        "integrity_ok": naive_scale.integrity_ok,
        "discrepancy": naive_scale.final_count - naive_scale.initial_count,
    }
    print(f"      t={naive_scale.elapsed:.2f}s  threads={scale_cfg.n_particles}  "
          f"integrity={'OK' if naive_scale.integrity_ok else 'CORRUPTED'}")

    # Sequential для baseline
    seq_scale = run_sequential(scale_cfg)
    pool_results["sequential"] = {
        "method": seq_scale.method_name,
        "elapsed": seq_scale.elapsed,
        "integrity_ok": seq_scale.integrity_ok,
    }
    print(f"  sequential baseline: t={seq_scale.elapsed:.2f}s")

    Path("results/scale.json").write_text(
        json.dumps(pool_results, indent=2, ensure_ascii=False))

    # Графік масштабування
    fig_path = Path("plots/04_scaling.png")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5))
    workers = [1, 2, 4, 8]
    pool_times = [pool_results[f"pool_W{w}"]["elapsed"] for w in workers]
    ax.plot(workers, pool_times, "o-", color="#1f77b4", linewidth=2,
            markersize=8, label="Worker Pool (1 worker / chunk)")
    ax.axhline(seq_scale.elapsed, color="gray", linestyle="--",
               label=f"Sequential ({seq_scale.elapsed:.2f}s)")
    ax.axhline(naive_scale.elapsed, color="#d62728", linestyle=":",
               label=f"Naive 1-per-particle, {scale_cfg.n_particles} threads ({naive_scale.elapsed:.2f}s)")
    ax.set_xlabel("Кількість worker-потоків (W)")
    ax.set_ylabel("Час симуляції, с")
    ax.set_title(f"Масштабування Worker Pool (N={scale_cfg.n_particles} частинок)")
    ax.set_xticks(workers)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_path, dpi=130)
    plt.close(fig)
    print(f"  ✓ scaling plot: {fig_path}")

    # ============================================================
    # Етап 3: Великий тест — N=2000 частинок, перевірка стійкості
    # ============================================================
    print("\n=== Етап 3: Великий тест ===")
    big_cfg = CrystalConfig(grid_size=50, n_particles=2000,
                            n_steps=100, snapshot_every=20, seed=42)
    print(f"Параметри: grid={big_cfg.grid_size}, N={big_cfg.n_particles}")

    big_results = {}
    for w in [2, 4, 8]:
        r = run_pool_threaded(big_cfg, n_workers=w)
        print(f"  pool W={w}: t={r.elapsed:.2f}s  "
              f"integrity={'OK' if r.integrity_ok else 'FAIL'}")
        big_results[f"pool_W{w}"] = {
            "elapsed": r.elapsed,
            "integrity_ok": r.integrity_ok,
        }

    seq_big = run_sequential(big_cfg)
    big_results["sequential"] = {
        "elapsed": seq_big.elapsed,
        "integrity_ok": seq_big.integrity_ok,
    }
    print(f"  sequential: t={seq_big.elapsed:.2f}s")

    # Збережемо знімки великого симуляційного прогону для звіту
    viz.save_snapshot_grid(seq_big, Path("snapshots/big_sequential_grid.png"))
    viz.save_animation(seq_big, Path("animations/big_sequential.gif"))

    Path("results/big.json").write_text(
        json.dumps(big_results, indent=2, ensure_ascii=False))

    print("\n=== ✓ Усі бенчмарки завершено ===")


if __name__ == "__main__":
    main()
