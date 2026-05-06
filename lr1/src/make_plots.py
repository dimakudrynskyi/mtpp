"""Генерація графіків часу виконання, прискорення та закону Амдала."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = json.loads(Path("results/benchmark.json").read_text())
WORKERS = DATA["metadata"]["worker_counts"]
PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True)

TASK_LABELS = {
    "pi_montecarlo": "Monte-Carlo π (5·10⁶ точок)",
    "factorization": "Факторизація 16 semiprimes (~10¹³)",
    "primes": "Прості числа в [1, 5·10⁵)",
    "transpose": "Транспонування 10000×10000 (int32)",
    "io_wordcount": "Підрахунок слів у 1000 файлах (5 мс/файл)",
}


def plot_task(task: dict, fname_stem: str):
    label = TASK_LABELS.get(task["task"], task["task"])
    seq_t = task["sequential"]["time"]
    thr = [task["threads"][str(w)]["time"] for w in WORKERS]
    has_proc = bool(task.get("processes"))
    proc = [task["processes"][str(w)]["time"] for w in WORKERS] if has_proc else None

    # ---- (1) час виконання ----
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(seq_t, color="gray", linestyle="--", label=f"Послідовно ({seq_t:.2f}s)")
    ax.plot(WORKERS, thr, "o-", color="#1f77b4",
            label="ThreadPoolExecutor", linewidth=2, markersize=8)
    if proc:
        ax.plot(WORKERS, proc, "s-", color="#d62728",
                label="ProcessPoolExecutor", linewidth=2, markersize=8)
    ax.set_xlabel("Кількість worker-ів")
    ax.set_ylabel("Час виконання, с")
    ax.set_title(f"Час виконання: {label}")
    ax.set_xticks(WORKERS)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"{fname_stem}_time.png", dpi=130)
    plt.close(fig)

    # ---- (2) прискорення (speedup) ----
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(1.0, color="gray", linestyle="--", label="Без прискорення (×1)")
    ax.plot(WORKERS, [seq_t / t for t in thr], "o-", color="#1f77b4",
            label="ThreadPoolExecutor", linewidth=2, markersize=8)
    if proc:
        ax.plot(WORKERS, [seq_t / t for t in proc], "s-", color="#d62728",
                label="ProcessPoolExecutor", linewidth=2, markersize=8)
    ax.plot(WORKERS, WORKERS, ":", color="green", alpha=0.5, label="Ідеальний (×N)")
    ax.set_xlabel("Кількість worker-ів")
    ax.set_ylabel("Прискорення (speedup)")
    ax.set_title(f"Прискорення: {label}")
    ax.set_xticks(WORKERS)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"{fname_stem}_speedup.png", dpi=130)
    plt.close(fig)


def plot_summary():
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for i, task in enumerate(DATA["results"]):
        seq_t = task["sequential"]["time"]
        best_times = []
        for w in WORKERS:
            t_thr = task["threads"][str(w)]["time"]
            best = t_thr
            if task.get("processes"):
                t_proc = task["processes"][str(w)]["time"]
                best = min(best, t_proc)
            best_times.append(best)
        ax.plot(WORKERS, [seq_t / t for t in best_times], "o-",
                color=colors[i % len(colors)], linewidth=2, markersize=8,
                label=TASK_LABELS.get(task["task"], task["task"]))
    ax.plot(WORKERS, WORKERS, ":", color="gray", alpha=0.5, label="Ідеальний (×N)")
    ax.set_xlabel("Кількість worker-ів")
    ax.set_ylabel("Прискорення (T_seq / T_par)")
    ax.set_title("Прискорення для всіх задач (краще з потоків/процесів)")
    ax.set_xticks(WORKERS)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "summary_speedup.png", dpi=130)
    plt.close(fig)


def plot_amdahl():
    fig, ax = plt.subplots(figsize=(8, 5))
    n_workers = list(range(1, 17))
    for p in [0.5, 0.75, 0.9, 0.95, 0.99]:
        speedup = [1 / ((1 - p) + p / n) for n in n_workers]
        ax.plot(n_workers, speedup, "o-", label=f"P = {p}", markersize=5)
    ax.set_xlabel("Кількість процесорів N")
    ax.set_ylabel("Максимальне прискорення S(N)")
    ax.set_title("Закон Амдала: S(N) = 1 / ((1−P) + P/N)")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Частка коду,\nщо паралелізується")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "amdahl.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    name_map = {
        "pi_montecarlo": "01_pi",
        "factorization": "02_factorize",
        "primes": "03_primes",
        "transpose": "04_transpose",
        "io_wordcount": "05_io",
    }
    for task in DATA["results"]:
        plot_task(task, name_map[task["task"]])
    plot_summary()
    plot_amdahl()
    print("Plots saved to plots/")
    for p in sorted(PLOTS_DIR.iterdir()):
        print(f"  {p.name}")
