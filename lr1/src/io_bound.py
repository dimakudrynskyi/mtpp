"""
I/O-bound задача: рекурсивно пройти по директорії, відкрити всі текстові файли
і підрахувати загальну кількість слів.

Особливості:
  • час виконання визначається швидкістю диску, а не CPU;
  • під час open()/read() Python звільняє GIL, тому потоки тут реально працюють паралельно.

Примітка про середовище: в Docker-контейнері використовується tmpfs (RAM-FS),
де I/O затримки на порядки менші, ніж на реальному HDD/SSD. Тому ми додаємо
опційну змінну середовища IO_LATENCY_MS, щоб симулювати реалістичну затримку
і побачити характерну для I/O-bound поведінку.
"""
from __future__ import annotations
import os
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor


_IO_LATENCY = float(os.environ.get("IO_LATENCY_MS", "0")) / 1000.0


def find_text_files(root: Path) -> list[Path]:
    files = []
    for p in root.rglob("*.txt"):
        if p.is_file():
            files.append(p)
    return files


def count_words_in_file(path: Path) -> int:
    try:
        if _IO_LATENCY > 0:
            time.sleep(_IO_LATENCY)
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        return len(text.split())
    except OSError:
        return 0


def count_sequential(root: Path) -> tuple[int, float]:
    files = find_text_files(root)
    t0 = time.perf_counter()
    total = sum(count_words_in_file(p) for p in files)
    return total, time.perf_counter() - t0


def count_parallel(root: Path, workers: int, executor_cls) -> tuple[int, float]:
    files = find_text_files(root)
    t0 = time.perf_counter()
    with executor_cls(max_workers=workers) as ex:
        total = sum(ex.map(count_words_in_file, files,
                           chunksize=max(1, len(files) // (workers * 4))))
    return total, time.perf_counter() - t0
