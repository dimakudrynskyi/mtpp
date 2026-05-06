"""Графіки для Lab 3."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PLOTS = Path("plots")
PLOTS.mkdir(exist_ok=True)


# ---------- Task 1: Throughput vs threads ----------

def plot_task1_throughput():
    data = json.loads(Path("results/task1_bench.json").read_text())
    threads = data["thread_counts"]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = {"ordered_locks": "#1f77b4", "global_lock": "#d62728", "try_lock": "#2ca02c"}
    markers = {"ordered_locks": "o", "global_lock": "s", "try_lock": "^"}
    labels = {
        "ordered_locks": "Ordered Locks (рішення deadlock)",
        "global_lock":   "Global Lock (повна серіалізація)",
        "try_lock":      "Try-Lock (оптимістичний)",
    }
    for sol_name, sol_data in data["solutions"].items():
        thrs = [sol_data[str(t)]["throughput"] for t in threads]
        ax.plot(threads, thrs, markers[sol_name] + "-",
                color=colors[sol_name], label=labels[sol_name],
                linewidth=2, markersize=8)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Кількість потоків")
    ax.set_ylabel("Throughput, переказів/с")
    ax.set_title("Продуктивність рішень Race/Deadlock vs кількість потоків")
    ax.set_xticks(threads)
    ax.set_xticklabels([str(t) for t in threads])
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS / "01_task1_throughput.png", dpi=130)
    plt.close(fig)


def plot_task1_time():
    data = json.loads(Path("results/task1_bench.json").read_text())
    threads = data["thread_counts"]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = {"ordered_locks": "#1f77b4", "global_lock": "#d62728", "try_lock": "#2ca02c"}
    markers = {"ordered_locks": "o", "global_lock": "s", "try_lock": "^"}
    labels = {
        "ordered_locks": "Ordered Locks",
        "global_lock":   "Global Lock",
        "try_lock":      "Try-Lock",
    }
    for sol_name, sol_data in data["solutions"].items():
        times = [sol_data[str(t)]["time"] for t in threads]
        ax.plot(threads, times, markers[sol_name] + "-",
                color=colors[sol_name], label=labels[sol_name],
                linewidth=2, markersize=8)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Кількість потоків")
    ax.set_ylabel("Час виконання, с")
    ax.set_title("Час виконання залежно від кількості потоків")
    ax.set_xticks(threads)
    ax.set_xticklabels([str(t) for t in threads])
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS / "02_task1_time.png", dpi=130)
    plt.close(fig)


# ---------- Task 1: Demo результати — discrepancy ----------

def plot_task1_demo():
    data = json.loads(Path("results/task1_demo.json").read_text())
    names_map = {
        "naive_race (без синхронізації)": "Naive\n(race condition)",
        "naive_race (з yield-точкою)": "Naive\n(race condition)",
        "naive_deadlock (два замки)": "Two locks\n(deadlock)",
        "ordered_locks (рішення deadlock)": "Ordered\nlocks",
        "global_lock (повна серіалізація)": "Global\nlock",
        "try_lock (оптимістичний)": "Try-lock\n(optimistic)",
    }
    names, discrepancies, deadlocks = [], [], []
    for d in data:
        names.append(names_map.get(d["name"], d["name"][:15]))
        discrepancies.append(d["discrepancy"])
        deadlocks.append(d["deadlock"])

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = []
    for d, dl in zip(discrepancies, deadlocks):
        if dl:
            colors.append("#ff7f0e")  # помаранчевий — deadlock
        elif abs(d) > 0.01:
            colors.append("#d62728")  # червоний — race
        else:
            colors.append("#2ca02c")  # зелений — OK
    bars = ax.bar(names, discrepancies, color=colors, edgecolor="black")
    for bar, d, dl in zip(bars, discrepancies, deadlocks):
        h = bar.get_height()
        if dl:
            label = "DEADLOCK"
        elif abs(d) < 0.01:
            label = "OK"
        else:
            label = f"{d:+.0f}"
        ax.text(bar.get_x() + bar.get_width() / 2, h, label,
                ha="center", va="bottom" if h >= 0 else "top", fontsize=9)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Втрата/створення коштів (інваріант)")
    ax.set_title("Цілісність даних після 51 200 переказів (1024 потоки × 50)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS / "03_task1_integrity.png", dpi=130)
    plt.close(fig)


# ---------- Task 2: IPC RTT comparison ----------

def plot_ipc_rtt():
    data = json.loads(Path("results/ipc_all.json").read_text())
    methods = [m["method"] for m in data["methods"]]
    avg_rtts = [m["avg_rtt_us"] for m in data["methods"]]
    p50s = [m["p50_rtt_us"] for m in data["methods"]]
    p99s = [m["p99_rtt_us"] for m in data["methods"]]

    # Скорочені назви для осі
    short = []
    for m, lp in zip(methods, [m["language_pair"] for m in data["methods"]]):
        if "Pipe" in m:
            short.append("Pipe\n(Py↔Py)")
        elif "shared_memory" in m:
            short.append("Shared Memory\n(Py↔Py)")
        elif "Unix" in m:
            short.append("Unix Socket\n(Py↔Py)")
        elif "TCP" in m:
            short.append("TCP socket\n(Py↔Node.js)")
        else:
            short.append(m[:15])

    x = range(len(methods))
    width = 0.28
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar([i - width for i in x], p50s, width,
           label="p50 (median)", color="#1f77b4")
    ax.bar(list(x), avg_rtts, width,
           label="Середній RTT", color="#ff7f0e")
    ax.bar([i + width for i in x], p99s, width,
           label="p99 (хвіст)", color="#d62728")
    ax.set_xticks(list(x))
    ax.set_xticklabels(short)
    ax.set_yscale("log")
    ax.set_ylabel("Round-trip time, µs (лог. шкала)")
    ax.set_title("Порівняння методів IPC: RTT для одного повідомлення")
    ax.grid(True, axis="y", alpha=0.3, which="both")
    ax.legend()
    for i, val in enumerate(avg_rtts):
        ax.text(i, val * 1.1, f"{val:.0f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(PLOTS / "04_ipc_rtt.png", dpi=130)
    plt.close(fig)


def plot_ipc_total():
    data = json.loads(Path("results/ipc_all.json").read_text())
    methods = [m["method"] for m in data["methods"]]
    totals = [m["total_time_s"] for m in data["methods"]]
    short = []
    for m in data["methods"]:
        s = m["method"]
        if "Pipe" in s:
            short.append("Pipe (Py↔Py)")
        elif "shared_memory" in s:
            short.append("Shared Memory (Py↔Py)")
        elif "Unix" in s:
            short.append("Unix Socket (Py↔Py)")
        elif "TCP" in s:
            short.append("TCP (Py↔Node.js)")
        else:
            short.append(s)
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728"]
    bars = ax.barh(short, totals, color=colors, edgecolor="black")
    for bar, t in zip(bars, totals):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
                f"  {t:.2f} с", va="center", fontsize=10)
    ax.set_xlabel("Загальний час 1000 round-trip обмінів, с")
    ax.set_title("IPC — сумарний час передачі 1000 повідомлень")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS / "05_ipc_total.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    plot_task1_throughput()
    plot_task1_time()
    plot_task1_demo()
    plot_ipc_rtt()
    plot_ipc_total()
    print("Plots saved:")
    for p in sorted(PLOTS.iterdir()):
        print(f"  {p.name}")
