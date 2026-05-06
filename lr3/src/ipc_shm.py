"""
Задача 2 / Метод 2: shared_memory + threading.Event між Python ↔ Python.

multiprocessing.shared_memory — це POSIX shared memory (на Linux: /dev/shm).
Дані живуть у спільному регіоні пам'яті, доступному обом процесам без
копіювання. Це найшвидший спосіб обміну, але вимагає ручної синхронізації
(умовні події multiprocessing.Event).

Сценарій той самий: 1000 round-trip передач float'а.
"""
from __future__ import annotations
import multiprocessing as mp
from multiprocessing import shared_memory
import struct
import random
import time
import json
from pathlib import Path


SHM_SIZE = 16  # 8 байтів на float + 8 запасу


def helper_process(shm_name: str, req_event, resp_event, stop_event):
    """Допоміжний процес: чекає на req, читає float, ехає назад."""
    shm = shared_memory.SharedMemory(name=shm_name)
    log = []
    try:
        while not stop_event.is_set():
            if not req_event.wait(timeout=0.1):
                continue
            req_event.clear()
            if stop_event.is_set():
                break
            value = struct.unpack("d", bytes(shm.buf[:8]))[0]
            log.append(value)
            # Ехо: записуємо те саме значення (у реальному додатку міг би бути
            # результат обчислень)
            shm.buf[:8] = struct.pack("d", value)
            resp_event.set()
    finally:
        shm.close()


def run_shm(n_messages: int = 1000) -> dict:
    shm = shared_memory.SharedMemory(create=True, size=SHM_SIZE)
    req_event = mp.Event()
    resp_event = mp.Event()
    stop_event = mp.Event()
    try:
        p = mp.Process(target=helper_process,
                       args=(shm.name, req_event, resp_event, stop_event))
        p.start()

        rnd = random.Random(42)
        rtt_times = []
        t0 = time.perf_counter()
        for _ in range(n_messages):
            value = rnd.random()
            ts = time.perf_counter()
            shm.buf[:8] = struct.pack("d", value)
            req_event.set()
            resp_event.wait()
            resp_event.clear()
            echoed = struct.unpack("d", bytes(shm.buf[:8]))[0]
            rtt_times.append(time.perf_counter() - ts)
            assert echoed == value
        elapsed = time.perf_counter() - t0
        stop_event.set()
        req_event.set()  # розбудити, щоб побачив stop
        p.join(timeout=2.0)
        if p.is_alive():
            p.terminate()
    finally:
        shm.close()
        shm.unlink()

    avg_rtt_us = (sum(rtt_times) / len(rtt_times)) * 1_000_000
    p50_us = sorted(rtt_times)[len(rtt_times) // 2] * 1_000_000
    p99_us = sorted(rtt_times)[int(len(rtt_times) * 0.99)] * 1_000_000
    return {
        "method": "shared_memory + Event",
        "language_pair": "Python↔Python",
        "environment": "single (subprocess)",
        "n_messages": n_messages,
        "total_time_s": elapsed,
        "avg_rtt_us": avg_rtt_us,
        "p50_rtt_us": p50_us,
        "p99_rtt_us": p99_us,
    }


if __name__ == "__main__":
    res = run_shm()
    print(json.dumps(res, indent=2, ensure_ascii=False))
    Path("results").mkdir(exist_ok=True)
    Path("results/ipc_shm.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
