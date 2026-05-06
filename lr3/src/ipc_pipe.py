"""
Задача 2 / Метод 1: multiprocessing.Pipe між Python ↔ Python.

Pipe — це найшвидший IPC у Python: повна абстракція над OS-pipe, дані
передаються через pickle. Працює тільки між процесами одного середовища
(один host, обидва Python).

Сценарій: основний процес генерує 1000 випадкових чисел, передає їх у
допоміжний процес для логування, потім отримує їх назад. Вимірюємо
round-trip latency для кожної операції.
"""
from __future__ import annotations
import multiprocessing as mp
import random
import time
import json
from pathlib import Path


def helper_process(conn):
    """Допоміжний процес: читає число, логує, відправляє назад."""
    log = []
    while True:
        msg = conn.recv()
        if msg is None:
            break
        log.append(msg)
        conn.send(msg)
    conn.send(("LOG_LEN", len(log)))
    conn.close()


def run_pipe(n_messages: int = 1000) -> dict:
    parent_conn, child_conn = mp.Pipe()
    p = mp.Process(target=helper_process, args=(child_conn,))
    p.start()

    rnd = random.Random(42)
    rtt_times = []
    t0 = time.perf_counter()
    for _ in range(n_messages):
        value = rnd.random()
        ts = time.perf_counter()
        parent_conn.send(value)
        echoed = parent_conn.recv()
        rtt_times.append(time.perf_counter() - ts)
        assert echoed == value
    parent_conn.send(None)  # сигнал завершення
    log_info = parent_conn.recv()
    elapsed = time.perf_counter() - t0
    p.join()

    avg_rtt_us = (sum(rtt_times) / len(rtt_times)) * 1_000_000
    p50_us = sorted(rtt_times)[len(rtt_times) // 2] * 1_000_000
    p99_us = sorted(rtt_times)[int(len(rtt_times) * 0.99)] * 1_000_000
    return {
        "method": "multiprocessing.Pipe",
        "language_pair": "Python↔Python",
        "environment": "single (subprocess)",
        "n_messages": n_messages,
        "total_time_s": elapsed,
        "avg_rtt_us": avg_rtt_us,
        "p50_rtt_us": p50_us,
        "p99_rtt_us": p99_us,
        "logged": log_info[1],
    }


if __name__ == "__main__":
    res = run_pipe()
    print(json.dumps(res, indent=2, ensure_ascii=False))
    Path("results").mkdir(exist_ok=True)
    Path("results/ipc_pipe.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
