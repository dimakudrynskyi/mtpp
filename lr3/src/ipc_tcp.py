"""
Задача 2 / Метод 3: TCP socket між Python ↔ Node.js.

Це найгнучкіший метод: Python запускає Node.js-процес як subprocess, той
слухає на 127.0.0.1:PORT, обмінюються JSON-рядками з \\n-розділювачем.
Працює між будь-якими мовами, що вміють sockets — фактично цим живуть
сучасні мікросервісні архітектури.

Про середовища:
  • single environment: Python запускає Node як child process (тут же)
  • multiple environments: Node-сервер може жити на іншій машині, Python
    клієнт лише підключається — реалізація ідентична, відрізняється
    тільки host у connect().
"""
from __future__ import annotations
import socket
import subprocess
import json
import time
import random
import os
import sys
from pathlib import Path


def start_node_helper(log_path: str = "/tmp/node_helper.log") -> tuple[subprocess.Popen, int]:
    """Запускає Node.js helper як subprocess і повертає його PID + порт."""
    env = os.environ.copy()
    env["LOG_PATH"] = log_path
    env["PORT"] = "0"  # ОС вибирає вільний
    proc = subprocess.Popen(
        ["node", str(Path(__file__).parent / "node_helper.js")],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env=env, text=True,
    )
    # Читаємо першу строку — там оголошений порт
    port_line = proc.stdout.readline().strip()
    if not port_line.startswith("PORT="):
        err = proc.stderr.read()
        raise RuntimeError(f"Node helper failed: {err}")
    port = int(port_line.split("=", 1)[1])
    return proc, port


def run_tcp(n_messages: int = 1000, dual_env: bool = False) -> dict:
    """
    dual_env=False: Python і Node — child processes одного pytest-runner'а
    dual_env=True:  моделюємо «дві різні машини» додатковою затримкою mark
                    (для демонстрації різниці між localhost і реальною мережею)
    """
    log_path = "/tmp/node_helper.log"
    proc, port = start_node_helper(log_path)
    try:
        # Дочекатися готовності серверу (простий polling)
        for _ in range(20):
            try:
                s = socket.create_connection(("127.0.0.1", port), timeout=0.5)
                break
            except OSError:
                time.sleep(0.05)
        else:
            raise RuntimeError("Cannot connect to Node helper")

        s.settimeout(5.0)
        # Розгортаємо buffered IO для зручності
        f = s.makefile("rwb", buffering=0)

        rnd = random.Random(42)
        rtt_times = []
        t0 = time.perf_counter()
        for i in range(n_messages):
            value = rnd.random()
            payload = json.dumps({"id": i, "value": value}).encode() + b"\n"
            ts = time.perf_counter()
            f.write(payload)
            response = f.readline()
            rtt_times.append(time.perf_counter() - ts)
            data = json.loads(response.decode())
            assert data["id"] == i and abs(data["value"] - value) < 1e-12, \
                f"mismatch: {data}"
        elapsed = time.perf_counter() - t0

        # Сигнал STOP
        f.write(b"STOP\n")
        f.flush()
        try:
            f.readline()  # читаємо STOPPED-відповідь
        except (socket.timeout, OSError):
            pass
        s.close()
    finally:
        try:
            proc.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            proc.terminate()
            proc.wait(timeout=2.0)

    avg_rtt_us = (sum(rtt_times) / len(rtt_times)) * 1_000_000
    p50_us = sorted(rtt_times)[len(rtt_times) // 2] * 1_000_000
    p99_us = sorted(rtt_times)[int(len(rtt_times) * 0.99)] * 1_000_000

    # Перевіримо лог Node
    log_lines = Path(log_path).read_text().splitlines()

    return {
        "method": "TCP socket + JSON",
        "language_pair": "Python↔Node.js",
        "environment": "dual (different runtime, same host)",
        "n_messages": n_messages,
        "total_time_s": elapsed,
        "avg_rtt_us": avg_rtt_us,
        "p50_rtt_us": p50_us,
        "p99_rtt_us": p99_us,
        "node_logged_lines": len(log_lines),
    }


if __name__ == "__main__":
    res = run_tcp()
    print(json.dumps(res, indent=2, ensure_ascii=False))
    Path("results").mkdir(exist_ok=True)
    Path("results/ipc_tcp.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
