"""Графіки для Lab 2: порівняння часу/прискорення трьох патернів для кожної
задачі + окремий графік для Pipeline vs Producer-Consumer."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = json.loads(Path("results/benchmark.json").read_text())
WORKERS = DATA["metadata"]["worker_counts"]
PLOTS = Path("plots")
PLOTS.mkdir(exist_ok=True)

TASK_LABELS = {
    "html_tags": "HTML-теги (400 файлів)",
    "array_stats": "Статистика масиву (5·10⁶ чисел)",
    "matrix_mult": "Множення матриць 1024×1024",
    "image_processing": "Обробка 60 зображень 800×800",
}

PATTERN_STYLES = {
    "map_reduce": ("Map-Reduce", "#1f77b4", "o"),
    "fork_join":  ("Fork-Join",  "#d62728", "s"),
    "worker_pool":("Worker Pool","#2ca02c", "^"),
}


def plot_task1(task: dict, fname_stem: str):
    """Графік 3 патернів для одної задачі — час і speedup."""
    label = TASK_LABELS.get(task["task"], task["task"])
    seq_t = task["sequential"]["time"]

    # ---- Час ----
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(seq_t, color="gray", linestyle="--", label=f"Sequential ({seq_t:.2f}s)")
    for pname, (display, color, marker) in PATTERN_STYLES.items():
        if pname not in task["patterns"]:
            continue
        times = [task["patterns"][pname][str(w)]["time"] for w in WORKERS]
        ax.plot(WORKERS, times, marker + "-", color=color, label=display,
                linewidth=2, markersize=8)
    ax.set_xlabel("Кількість worker-ів")
    ax.set_ylabel("Час виконання, с")
    ax.set_title(f"Час виконання: {label}")
    ax.set_xticks(WORKERS)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS / f"{fname_stem}_time.png", dpi=130)
    plt.close(fig)

    # ---- Прискорення ----
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(1.0, color="gray", linestyle="--", label="Без прискорення (×1)")
    ax.plot(WORKERS, WORKERS, ":", color="green", alpha=0.5, label="Ідеальне (×N)")
    for pname, (display, color, marker) in PATTERN_STYLES.items():
        if pname not in task["patterns"]:
            continue
        times = [task["patterns"][pname][str(w)]["time"] for w in WORKERS]
        speedups = [seq_t / t for t in times]
        ax.plot(WORKERS, speedups, marker + "-", color=color, label=display,
                linewidth=2, markersize=8)
    ax.set_xlabel("Кількість worker-ів")
    ax.set_ylabel("Прискорення (T_seq / T_par)")
    ax.set_title(f"Прискорення: {label}")
    ax.set_xticks(WORKERS)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS / f"{fname_stem}_speedup.png", dpi=130)
    plt.close(fig)


def plot_task2(task: dict):
    """Pipeline vs Producer-Consumer."""
    label = TASK_LABELS.get(task["task"], task["task"])
    seq_t = task["sequential"]["time"]
    pipeline_t = task["patterns"]["pipeline"]["4"]["time"]

    # Час
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(seq_t, color="gray", linestyle="--", label=f"Sequential ({seq_t:.2f}s)")
    ax.axhline(pipeline_t, color="#ff7f0e", linestyle="-.",
               label=f"Pipeline 4-stage ({pipeline_t:.2f}s)")
    pc_times = [task["patterns"]["producer_consumer"][str(w)]["time"] for w in WORKERS]
    ax.plot(WORKERS, pc_times, "o-", color="#9467bd",
            label="Producer-Consumer", linewidth=2, markersize=8)
    ax.set_xlabel("Кількість consumer-потоків")
    ax.set_ylabel("Час виконання, с")
    ax.set_title(f"Час виконання: {label}")
    ax.set_xticks(WORKERS)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS / "06_images_time.png", dpi=130)
    plt.close(fig)

    # Прискорення
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(1.0, color="gray", linestyle="--", label="Без прискорення")
    ax.axhline(seq_t / pipeline_t, color="#ff7f0e", linestyle="-.",
               label=f"Pipeline 4-stage (×{seq_t/pipeline_t:.2f})")
    speedups = [seq_t / t for t in pc_times]
    ax.plot(WORKERS, speedups, "o-", color="#9467bd",
            label="Producer-Consumer", linewidth=2, markersize=8)
    ax.set_xlabel("Кількість consumer-потоків")
    ax.set_ylabel("Прискорення")
    ax.set_title(f"Прискорення: {label}")
    ax.set_xticks(WORKERS)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS / "06_images_speedup.png", dpi=130)
    plt.close(fig)


def plot_summary():
    """Зведене порівняння Map-Reduce/Fork-Join/Worker-Pool на всіх задачах."""
    task1_results = [r for r in DATA["results"] if r["task"] != "image_processing"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, task in zip(axes, task1_results):
        seq_t = task["sequential"]["time"]
        for pname, (display, color, marker) in PATTERN_STYLES.items():
            times = [task["patterns"][pname][str(w)]["time"] for w in WORKERS]
            ax.plot(WORKERS, [seq_t / t for t in times], marker + "-",
                    color=color, label=display, linewidth=2, markersize=7)
        ax.plot(WORKERS, WORKERS, ":", color="gray", alpha=0.5, label="Ідеальне")
        ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xticks(WORKERS)
        ax.set_xlabel("Worker-и")
        ax.grid(True, alpha=0.3)
        ax.set_title(TASK_LABELS.get(task["task"], task["task"]), fontsize=10)
    axes[0].set_ylabel("Прискорення")
    axes[-1].legend(loc="upper left", fontsize=9)
    fig.suptitle("Порівняння патернів на трьох задачах Task 1", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(PLOTS / "summary_task1.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    name_map = {
        "html_tags": "01_html",
        "array_stats": "02_array",
        "matrix_mult": "03_matrix",
    }
    for task in DATA["results"]:
        if task["task"] in name_map:
            plot_task1(task, name_map[task["task"]])
        elif task["task"] == "image_processing":
            plot_task2(task)
    plot_summary()
    print("Plots saved:")
    for p in sorted(PLOTS.iterdir()):
        print(f"  {p.name}")
