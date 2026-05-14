"""Графіки результатів бенчмарку."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

data = json.loads(Path("results/benchmark.json").read_text())
PLOTS = Path("plots")
PLOTS.mkdir(exist_ok=True)


def plot_throughput():
    rows = data["throughput"]
    Ns = [r["n_clients"] for r in rows]
    rates = [r["rate_msg_per_sec"] for r in rows]
    delivs = [r["delivery_rate"] * 100 for r in rows]

    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    ax1.set_xlabel("Кількість одночасних клієнтів N")
    ax1.set_ylabel("Throughput, повідомлень/с", color="#1f77b4")
    ax1.plot(Ns, rates, "o-", color="#1f77b4", linewidth=2, markersize=10,
             label="Throughput")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.grid(True, alpha=0.3)

    for n, r in zip(Ns, rates):
        ax1.annotate(f"{r:.0f}", (n, r), textcoords="offset points",
                     xytext=(0, 10), ha="center", fontsize=10)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Delivery rate, %", color="#d62728")
    ax2.plot(Ns, delivs, "s--", color="#d62728", linewidth=2, markersize=8,
             label="Delivery rate")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax2.set_ylim(0, 110)

    fig.suptitle("Throughput чат-сервера vs кількість клієнтів")
    fig.tight_layout()
    fig.savefig(PLOTS / "01_throughput.png", dpi=130)
    plt.close(fig)


def plot_latency():
    lat = data["latency"][0]
    metrics = ["min_ms", "median_ms", "avg_ms", "p99_ms", "max_ms"]
    labels = ["min", "median", "avg", "p99", "max"]
    values = [lat[m] for m in metrics]
    colors = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728", "#9467bd"]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, values, color=colors, edgecolor="black")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v,
                f"{v:.2f}", ha="center", va="bottom", fontsize=11)
    ax.set_ylabel("Round-trip time, мс")
    ax.set_title(f"Latency ping/pong ({lat['n_clients']} клієнтів × {lat['n_pings']} pings = "
                 f"{lat['samples']} вимірів)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS / "02_latency.png", dpi=130)
    plt.close(fig)


def plot_burst():
    burst = data["burst"][0]
    fig, ax = plt.subplots(figsize=(9, 5))
    metrics = ["expected", "actual"]
    values = [burst[m] for m in metrics]
    colors = ["#9ca3af", "#2ca02c"]
    bars = ax.bar(["Очікувано", "Доставлено"], values, color=colors, edgecolor="black")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v,
                f"{v}", ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.set_ylabel("Кількість повідомлень")
    ax.set_title(f"Burst broadcast: {burst['n_clients']} клієнтів × {burst['n_bursts']} broadcast'ів"
                 f" → {burst['rate_msg_per_sec']:.0f} msg/s, "
                 f"{burst['delivery_rate']*100:.1f}% delivery")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS / "03_burst.png", dpi=130)
    plt.close(fig)


def plot_stats():
    """Зведений bar-chart по статистиці."""
    test_path = Path("results/test_stats.json")
    if test_path.exists():
        stats = json.loads(test_path.read_text())["stats"]
    else:
        stats = data["final_stats"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Bar: типи повідомлень — log-scale, щоб маленькі категорії були видні
    labels = ["broadcasts", "private_messages", "group_messages", "files_sent"]
    values = [stats[k] for k in labels]
    pretty = ["Broadcast", "Приватні", "Групові", "Файли"]
    colors = ["#fbbf24", "#3b82f6", "#10b981", "#a855f7"]
    bars = axes[0].bar(pretty, values, color=colors, edgecolor="black")
    for bar, v in zip(bars, values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, max(v, 0.5),
                      str(v), ha="center", va="bottom", fontsize=11)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Кількість (лог. шкала)")
    axes[0].set_title(f"Типи повідомлень (всього {stats['messages_total']})")
    axes[0].grid(True, axis="y", alpha=0.3, which="both")

    # Bar: статистика конкурентної роботи
    bar_metrics = ["users_registered", "disconnects", "offline_stored",
                    "offline_delivered", "history_pairs", "groups"]
    bar_pretty = ["Реєстрації", "Відключення", "Збережено\nофлайн",
                   "Доставлено\nофлайн", "Пар у\nісторії", "Груп"]
    bar_values = [stats[k] for k in bar_metrics]
    axes[1].bar(bar_pretty, bar_values, color="#3b82f6", edgecolor="black")
    for i, v in enumerate(bar_values):
        axes[1].text(i, v, str(v), ha="center", va="bottom", fontsize=10)
    axes[1].set_title("Статистика користувачів та з'єднань")
    axes[1].grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(PLOTS / "04_stats.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    plot_throughput()
    plot_latency()
    plot_burst()
    plot_stats()
    print("Plots saved to plots/")
    for p in sorted(PLOTS.iterdir()):
        print(f"  {p.name}")
