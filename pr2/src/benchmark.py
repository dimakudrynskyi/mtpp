"""
Бенчмарк продуктивності чат-сервера.

Сценарії:
  1. Throughput vs кількість клієнтів (10, 50, 100, 200): кожен надсилає
     по 20 повідомлень — приватні + broadcast. Вимірюємо message rate.
  2. Latency vs кількість клієнтів: round-trip ping від кожного клієнта.
  3. Тест навантаження: 50 одночасних broadcast'ів — перевіряємо delivery rate.
"""
from __future__ import annotations
import json
import socket
import sys
import threading
import time
from pathlib import Path
from statistics import mean, median

sys.path.insert(0, str(Path(__file__).parent))
from server import start_servers
from test_e2e import TestClient


def bench_throughput(n_clients: int, messages_per_client: int = 20) -> dict:
    """N клієнтів, кожен надсилає M broadcast'ів. Вимірює rate."""
    print(f"\n[Throughput] {n_clients} clients × {messages_per_client} broadcasts...")
    clients = []
    for i in range(n_clients):
        clients.append(TestClient(f"throughput_{i}"))
        time.sleep(0.005)
    for c in clients:
        c.wait_for("welcome", timeout=5.0)
    time.sleep(0.5)

    expected_total = n_clients * messages_per_client * (n_clients - 1)
    received_before = sum(len(c.filter("broadcast_msg")) for c in clients)

    t0 = time.perf_counter()
    for c in clients:
        for j in range(messages_per_client):
            c.send({"type": "broadcast", "text": f"m{j}"})
    send_done = time.perf_counter() - t0

    # Ждати, поки всі повідомлення дійдуть
    time.sleep(min(2.0, n_clients * 0.05))
    elapsed = time.perf_counter() - t0
    received_after = sum(len(c.filter("broadcast_msg")) for c in clients)
    actual = received_after - received_before
    rate = actual / elapsed if elapsed > 0 else 0
    delivery_rate = actual / expected_total if expected_total else 0

    for c in clients:
        c.close()
    time.sleep(0.5)

    return {
        "n_clients": n_clients,
        "messages_per_client": messages_per_client,
        "send_time": send_done,
        "total_time": elapsed,
        "expected_deliveries": expected_total,
        "actual_deliveries": actual,
        "delivery_rate": delivery_rate,
        "rate_msg_per_sec": rate,
    }


def bench_latency(n_clients: int = 10, n_pings: int = 50) -> dict:
    """Вимірює RTT ping/pong для кожного клієнта."""
    print(f"\n[Latency] {n_clients} clients, {n_pings} pings each...")
    clients = []
    for i in range(n_clients):
        clients.append(TestClient(f"latency_{i}"))
        time.sleep(0.005)
    for c in clients:
        c.wait_for("welcome", timeout=5.0)
    time.sleep(0.3)

    all_rtts = []
    # Послідовні ping/pong від кожного клієнта
    for c in clients:
        rtts = []
        for _ in range(n_pings):
            t0 = time.perf_counter()
            c.send({"type": "ping"})
            pong = c.wait_for("pong", timeout=2.0)
            if pong:
                rtts.append((time.perf_counter() - t0) * 1000)  # ms
            # Очистити чергу pong'ів
            c.received = [m for m in c.received if m.get("type") != "pong"]
        all_rtts.extend(rtts)

    for c in clients:
        c.close()
    time.sleep(0.3)

    return {
        "n_clients": n_clients,
        "n_pings": n_pings,
        "samples": len(all_rtts),
        "avg_ms": mean(all_rtts),
        "median_ms": median(all_rtts),
        "min_ms": min(all_rtts),
        "max_ms": max(all_rtts),
        "p99_ms": sorted(all_rtts)[int(len(all_rtts) * 0.99)],
    }


def bench_burst_broadcast(n_clients: int = 30, n_bursts: int = 50) -> dict:
    """50 broadcast'ів за раз з одного клієнта на N глядачів — стрес."""
    print(f"\n[Burst] {n_clients} clients, {n_bursts} broadcasts...")
    clients = []
    for i in range(n_clients):
        clients.append(TestClient(f"burst_{i}"))
        time.sleep(0.005)
    for c in clients:
        c.wait_for("welcome", timeout=5.0)
    time.sleep(0.3)

    t0 = time.perf_counter()
    sender = clients[0]
    for j in range(n_bursts):
        sender.send({"type": "broadcast", "text": f"burst-{j}"})

    # Дочекатися доставки
    expected = (n_clients - 1) * n_bursts
    deadline = time.perf_counter() + 5.0
    while time.perf_counter() < deadline:
        actual = sum(len(c.filter("broadcast_msg")) for c in clients[1:])
        if actual >= expected:
            break
        time.sleep(0.05)
    elapsed = time.perf_counter() - t0
    actual = sum(len(c.filter("broadcast_msg")) for c in clients[1:])

    for c in clients:
        c.close()
    time.sleep(0.3)

    return {
        "n_clients": n_clients,
        "n_bursts": n_bursts,
        "expected": expected,
        "actual": actual,
        "delivery_rate": actual / expected,
        "elapsed_s": elapsed,
        "rate_msg_per_sec": actual / elapsed if elapsed > 0 else 0,
    }


def main():
    print("Starting servers...")
    state, stop = start_servers(tcp_port=9000, ws_port=9001)
    time.sleep(1.0)

    results = {"throughput": [], "latency": [], "burst": []}

    # 1. Throughput для різних N
    for n in [10, 30, 50, 100]:
        r = bench_throughput(n_clients=n, messages_per_client=10)
        print(f"  N={n}: rate={r['rate_msg_per_sec']:.0f} msg/s, "
              f"delivery={r['delivery_rate']*100:.1f}%")
        results["throughput"].append(r)

    # 2. Latency
    r = bench_latency(n_clients=10, n_pings=30)
    print(f"  latency avg={r['avg_ms']:.2f}ms median={r['median_ms']:.2f}ms p99={r['p99_ms']:.2f}ms")
    results["latency"].append(r)

    # 3. Burst broadcast
    r = bench_burst_broadcast(n_clients=30, n_bursts=50)
    print(f"  burst: {r['actual']}/{r['expected']} delivered in {r['elapsed_s']:.2f}s "
          f"({r['rate_msg_per_sec']:.0f} msg/s)")
    results["burst"].append(r)

    # 4. Final server stats
    results["final_stats"] = state.get_stats()

    Path("results").mkdir(exist_ok=True)
    Path("results/benchmark.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\n✓ Saved results/benchmark.json")

    stop.set()


if __name__ == "__main__":
    main()
