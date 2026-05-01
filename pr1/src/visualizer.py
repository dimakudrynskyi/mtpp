"""
Візуалізація результатів симуляції.

Створює:
  • Окремі PNG-файли для кожного знімка (для вставки у звіт)
  • Анімований GIF, що показує еволюцію кристалу з часом
  • Лінійний графік: кількість частинок vs крок (інваріант = плоска лінія)
  • Heatmap-порівняння початкового та кінцевого стану
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

from crystal import SimulationResult


def save_snapshot_grid(result: SimulationResult, out_path: Path, n_cols: int = 4):
    """Зберігає всі знімки як одну сітку зображень — для вставки у звіт."""
    snapshots = result.snapshots
    steps = result.snapshot_steps
    n = len(snapshots)
    n_rows = (n + n_cols - 1) // n_cols
    vmax = max(s.max() for s in snapshots)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.5 * n_cols, 2.5 * n_rows))
    axes = np.array(axes).reshape(n_rows, n_cols)
    for i, (snap, step) in enumerate(zip(snapshots, steps)):
        ax = axes[i // n_cols, i % n_cols]
        im = ax.imshow(snap, cmap="viridis", vmin=0, vmax=vmax,
                       interpolation="nearest")
        ax.set_title(f"Крок {step}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    # Сховати порожні
    for i in range(n, n_rows * n_cols):
        axes[i // n_cols, i % n_cols].set_visible(False)

    fig.suptitle(f"{result.method_name}\n"
                  f"GRID={result.config.grid_size}×{result.config.grid_size}, "
                  f"N={result.config.n_particles}, "
                  f"крок-знімка={result.config.snapshot_every}",
                  fontsize=11)
    fig.colorbar(im, ax=axes, fraction=0.025)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def save_animation(result: SimulationResult, out_path: Path, fps: int = 4):
    """Створює анімований GIF із послідовності знімків."""
    snapshots = result.snapshots
    steps = result.snapshot_steps
    vmax = max(s.max() for s in snapshots)

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(snapshots[0], cmap="viridis", vmin=0, vmax=vmax,
                   interpolation="nearest")
    title = ax.set_title("")
    fig.colorbar(im, ax=ax, fraction=0.046)

    def update(frame_idx):
        im.set_array(snapshots[frame_idx])
        title.set_text(f"Крок {steps[frame_idx]} — {result.method_name}")
        return [im, title]

    anim = FuncAnimation(fig, update, frames=len(snapshots), interval=1000 // fps)
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def save_invariant_check(results: list, out_path: Path):
    """
    Лінійний графік: кількість частинок vs крок для кожної версії.
    Має бути плоска лінія = N для коректних рішень. Розрив або зміна — race.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    for r in results:
        counts = [int(s.sum()) for s in r.snapshots]
        ax.plot(r.snapshot_steps, counts, marker="o", linewidth=2,
                markersize=6, label=r.method_name)
    if results:
        ax.axhline(results[0].initial_count, color="black", linestyle="--",
                   alpha=0.5, label=f"Інваріант (N={results[0].initial_count})")
    ax.set_xlabel("Крок симуляції")
    ax.set_ylabel("Кількість частинок у кристалі")
    ax.set_title("Цілісність даних: інваріант зберігання частинок")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def save_initial_final(result: SimulationResult, out_path: Path):
    """Side-by-side: початковий vs кінцевий стан."""
    snapshots = result.snapshots
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    vmax = max(snapshots[0].max(), snapshots[-1].max())
    for ax, snap, label in zip(axes, [snapshots[0], snapshots[-1]],
                                ["Початковий стан (крок 0)",
                                 f"Кінцевий стан (крок {result.snapshot_steps[-1]})"]):
        im = ax.imshow(snap, cmap="viridis", vmin=0, vmax=vmax, interpolation="nearest")
        ax.set_title(label)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes, fraction=0.025)
    fig.suptitle(f"{result.method_name}: початок ↔ кінець "
                  f"(N={result.initial_count} → {result.final_count})",
                  fontsize=11)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def save_throughput_comparison(results: dict, out_path: Path):
    """
    Bar-chart часу виконання для різних методів.
    results: dict[method_name → SimulationResult]
    """
    names = list(results.keys())
    times = [results[n].elapsed for n in names]
    integrity = [results[n].integrity_ok for n in names]
    deadlocked = [results[n].extra.get("deadlocked", False) for n in names]

    colors = []
    for ok, dl in zip(integrity, deadlocked):
        if dl:
            colors.append("#ff7f0e")
        elif not ok:
            colors.append("#d62728")
        else:
            colors.append("#2ca02c")

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(names, times, color=colors, edgecolor="black")
    for bar, t, ok, dl in zip(bars, times, integrity, deadlocked):
        if dl:
            label = f"  {t:.2f}s (DEADLOCK)"
        elif not ok:
            res = results[list(results.keys())[bars.index(bar) if hasattr(bars, 'index') else 0]]
            label = f"  {t:.2f}s (CORRUPTED)"
        else:
            label = f"  {t:.2f}s ✓"
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
                label, va="center", fontsize=10)
    ax.set_xlabel("Час виконання, секунд")
    ax.set_title("Порівняння часу симуляції за методами")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
