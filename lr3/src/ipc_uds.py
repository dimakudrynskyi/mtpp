"""
Задача 2 / Метод 4: Unix Domain Socket між Python ↔ Python.

Unix socket — як TCP, але через файл-сокет, без TCP/IP стека. Швидший за
TCP loopback, але працює тільки в межах одного хоста (не через мережу).
Однак працює між будь-якими процесами одного хоста, з будь-якими мовами,
що підтримують AF_UNIX (Python, Node.js, C, Go тощо).
"""
from __future__ import annotations
import socket
import os
import multiprocessing as mp
import struct
import random
import time
import json
import tempfile
from pathlib import Path


def helper_uds(socket_path: str, ready_event):
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(socket_path)
    server.listen(1)
    ready_event.set()
    conn, _ = server.accept()
    f = conn.makefile("rwb", buffering=0)
    log = []
    while True:
        line = f.readline()
        if not line or line.strip() == b"STOP":
            break
        msg = json.loads(line.decode())
        log.append(msg["value"])
        # Ехо
        f.write(json.dumps({"id": msg["id"], "value": msg["value"]}).encode() + b"\n")
    f.close()
    conn.close()
    server.close()
    os.unlink(socket_path)


def run_uds(n_messages: int = 1000) -> dict:
    socket_path = tempfile.mktemp(suffix=".sock")
    ready_event = mp.Event()
    p = mp.Process(target=helper_uds, args=(socket_path, ready_event))
    p.start()
    ready_event.wait(timeout=5.0)

    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.connect(socket_path)
    f = client.makefile("rwb", buffering=0)

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
        assert data["id"] == i and abs(data["value"] - value) < 1e-12
    elapsed = time.perf_counter() - t0
    f.write(b"STOP\n")
    client.close()
    p.join(timeout=2.0)
    if p.is_alive():
        p.terminate()

    avg_rtt_us = (sum(rtt_times) / len(rtt_times)) * 1_000_000
    p50_us = sorted(rtt_times)[len(rtt_times) // 2] * 1_000_000
    p99_us = sorted(rtt_times)[int(len(rtt_times) * 0.99)] * 1_000_000
    return {
        "method": "Unix Domain Socket + JSON",
        "language_pair": "Python↔Python",
        "environment": "single host (cross-language capable)",
        "n_messages": n_messages,
        "total_time_s": elapsed,
        "avg_rtt_us": avg_rtt_us,
        "p50_rtt_us": p50_us,
        "p99_rtt_us": p99_us,
    }


if __name__ == "__main__":
    res = run_uds()
    print(json.dumps(res, indent=2, ensure_ascii=False))
    Path("results").mkdir(exist_ok=True)
    Path("results/ipc_uds.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
