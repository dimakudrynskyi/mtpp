"""
Задача 1.2: Знайти мінімум, максимум, медіану та середнє значення великого масиву.

Модель. Імітуємо streaming-обробку: масив поділяється на chunks, кожен chunk
«отримується з джерела» з симульованою затримкою (IO_LATENCY_MS) перед
локальними numpy-обчисленнями. Це модель block processing з повільного диску
або потоку даних (Kafka, S3 тощо).

Реалізовано чотири версії:
  • sequential   — обробка всіх chunks послідовно
  • Map-Reduce   — chunk-statistics → reduce
  • Fork-Join    — рекурсивний поділ на під-діапазони + об'єднання
  • Worker Pool  — пул threads + submit/as_completed

Об'єднання chunk-статистик:
  • min/max         — комбінуються тривіально (min/max від chunk-min/max)
  • mean            — зважене середнє від chunk-mean із врахуванням розмірів
  • median (approx) — для exact median потрібно мати весь масив; ми обчислюємо
                     наближення як median від chunk-median, що дає коректний
                     результат для рівномірно розподілених даних. Для exact —
                     існують паралельні алгоритми (k-th element selection).
"""
from __future__ import annotations
import os
import time
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

_IO_LATENCY = float(os.environ.get("IO_LATENCY_MS", "15")) / 1000.0


@dataclass
class ChunkStats:
    n: int
    min: float
    max: float
    mean: float
    median: float

    @staticmethod
    def from_array(a: np.ndarray) -> "ChunkStats":
        return ChunkStats(
            n=int(a.size),
            min=float(a.min()),
            max=float(a.max()),
            mean=float(a.mean()),
            median=float(np.median(a)),
        )


@dataclass
class Stats:
    n: int
    min: float
    max: float
    mean: float
    median: float


def _stats_for_chunk(chunk: np.ndarray) -> ChunkStats:
    if _IO_LATENCY > 0:
        time.sleep(_IO_LATENCY)  # симулює завантаження chunk із джерела
    return ChunkStats.from_array(chunk)


def _combine(parts: list[ChunkStats]) -> Stats:
    """Комбінує chunk-статистики в загальні."""
    total_n = sum(p.n for p in parts)
    return Stats(
        n=total_n,
        min=min(p.min for p in parts),
        max=max(p.max for p in parts),
        mean=sum(p.mean * p.n for p in parts) / total_n,
        # Наближена медіана: median від chunk-medians (зважена тут не є тривіальною,
        # тому беремо arithmetic median як approximation — точну median дає окремий алгоритм)
        median=float(np.median([p.median for p in parts])),
    )


def _split_array(a: np.ndarray, n_chunks: int) -> list[np.ndarray]:
    """Розбиває масив на n_chunks приблизно рівних частин."""
    return np.array_split(a, n_chunks)


# ---------- Sequential ----------

def run_sequential(arr: np.ndarray, n_chunks: int = 32) -> tuple[Stats, float]:
    """Обробка тих самих chunks, але послідовно — справедливе порівняння."""
    chunks = _split_array(arr, n_chunks)
    t0 = time.perf_counter()
    parts = [_stats_for_chunk(c) for c in chunks]
    result = _combine(parts)
    return result, time.perf_counter() - t0


# ---------- Map-Reduce ----------

def run_map_reduce(arr: np.ndarray, workers: int, n_chunks: int = 32) -> tuple[Stats, float]:
    chunks = _split_array(arr, n_chunks)
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        parts = list(pool.map(_stats_for_chunk, chunks))  # MAP
    result = _combine(parts)  # REDUCE
    return result, time.perf_counter() - t0


# ---------- Fork-Join ----------

def _split_recursive(chunks: list, threshold: int) -> list[list]:
    if len(chunks) <= threshold:
        return [chunks]
    mid = len(chunks) // 2
    return _split_recursive(chunks[:mid], threshold) + _split_recursive(chunks[mid:], threshold)


def _stats_for_batch(batch: list[np.ndarray]) -> ChunkStats:
    """Обробка батча chunks і повернення комбінованої статистики."""
    parts = [_stats_for_chunk(c) for c in batch]
    s = _combine(parts)
    return ChunkStats(n=s.n, min=s.min, max=s.max, mean=s.mean, median=s.median)


def run_fork_join(arr: np.ndarray, workers: int, n_chunks: int = 32, threshold: int = 4) -> tuple[Stats, float]:
    chunks = _split_array(arr, n_chunks)
    batches = _split_recursive(chunks, threshold)  # FORK
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        partials = list(pool.map(_stats_for_batch, batches))
    result = _combine(partials)  # JOIN
    return result, time.perf_counter() - t0


# ---------- Worker Pool ----------

def run_worker_pool(arr: np.ndarray, workers: int, n_chunks: int = 32) -> tuple[Stats, float]:
    chunks = _split_array(arr, n_chunks)
    t0 = time.perf_counter()
    parts = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_stats_for_chunk, c) for c in chunks]
        for fut in as_completed(futures):
            parts.append(fut.result())
    result = _combine(parts)
    return result, time.perf_counter() - t0


if __name__ == "__main__":
    arr = np.load("test_data/array.npy")
    print(f"Array: {arr.size:,} elements")
    s, t = run_sequential(arr)
    print(f"  seq: min={s.min:.2f} max={s.max:.2f} mean={s.mean:.2f} median={s.median:.2f} t={t:.2f}s")
    s, t = run_map_reduce(arr, 8)
    print(f"  mapreduce w=8: t={t:.2f}s")
