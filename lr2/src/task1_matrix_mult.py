"""
Задача 1.3: Множення двох великих матриць C = A × B.

Модель. Block matrix multiplication: матриця C поділяється на горизонтальні
смуги (блоки рядків). Для обчислення кожного блоку потрібно отримати відповідні
рядки A (симульована затримка завантаження). Це модель distributed matrix
multiplication, де блоки даних знаходяться на різних вузлах кластеру.

Усі патерни виконуються через ThreadPoolExecutor — numpy.dot реалізований
у нативному C-коді (BLAS) і звільняє GIL під час обчислень.
"""
from __future__ import annotations
import os
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

_IO_LATENCY = float(os.environ.get("IO_LATENCY_MS", "15")) / 1000.0


def _multiply_block(args) -> tuple[int, int, np.ndarray]:
    """Обчислює один блок рядків результату C[r0:r1, :] = A[r0:r1, :] @ B."""
    A, B, r0, r1 = args
    if _IO_LATENCY > 0:
        time.sleep(_IO_LATENCY)  # симулює завантаження A[r0:r1, :] із джерела
    C_block = A[r0:r1, :] @ B
    return r0, r1, C_block


def _make_blocks(n_rows: int, n_blocks: int):
    """Розбиває [0, n_rows) на n_blocks приблизно рівних діапазонів."""
    block = max(1, n_rows // n_blocks)
    out = []
    for i in range(n_blocks):
        r0 = i * block
        r1 = n_rows if i == n_blocks - 1 else (i + 1) * block
        if r0 < r1:
            out.append((r0, r1))
    return out


# ---------- Sequential ----------

def run_sequential(A: np.ndarray, B: np.ndarray, n_blocks: int = 32) -> tuple[np.ndarray, float]:
    blocks = _make_blocks(A.shape[0], n_blocks)
    C = np.empty((A.shape[0], B.shape[1]), dtype=A.dtype)
    t0 = time.perf_counter()
    for r0, r1 in blocks:
        _, _, Cb = _multiply_block((A, B, r0, r1))
        C[r0:r1, :] = Cb
    return C, time.perf_counter() - t0


# ---------- Map-Reduce ----------

def run_map_reduce(A: np.ndarray, B: np.ndarray, workers: int, n_blocks: int = 32) -> tuple[np.ndarray, float]:
    """Тут «reduce» вироджена — результати лише вкладаються в потрібні рядки C."""
    blocks = _make_blocks(A.shape[0], n_blocks)
    tasks = [(A, B, r0, r1) for r0, r1 in blocks]
    C = np.empty((A.shape[0], B.shape[1]), dtype=A.dtype)
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for r0, r1, Cb in pool.map(_multiply_block, tasks):  # MAP
            C[r0:r1, :] = Cb                                  # REDUCE (assemble)
    return C, time.perf_counter() - t0


# ---------- Fork-Join ----------

def _split_recursive(blocks, threshold):
    if len(blocks) <= threshold:
        return [blocks]
    mid = len(blocks) // 2
    return _split_recursive(blocks[:mid], threshold) + _split_recursive(blocks[mid:], threshold)


def _multiply_batch(args):
    """Обробляє батч блоків — повертає список результатів для join'у."""
    A, B, ranges = args
    out = []
    for r0, r1 in ranges:
        _, _, Cb = _multiply_block((A, B, r0, r1))
        out.append((r0, r1, Cb))
    return out


def run_fork_join(A: np.ndarray, B: np.ndarray, workers: int, n_blocks: int = 32, threshold: int = 4) -> tuple[np.ndarray, float]:
    blocks = _make_blocks(A.shape[0], n_blocks)
    batches = _split_recursive(blocks, threshold)  # FORK
    tasks = [(A, B, batch) for batch in batches]
    C = np.empty((A.shape[0], B.shape[1]), dtype=A.dtype)
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for partial in pool.map(_multiply_batch, tasks):
            for r0, r1, Cb in partial:    # JOIN: assemble blocks
                C[r0:r1, :] = Cb
    return C, time.perf_counter() - t0


# ---------- Worker Pool ----------

def run_worker_pool(A: np.ndarray, B: np.ndarray, workers: int, n_blocks: int = 32) -> tuple[np.ndarray, float]:
    blocks = _make_blocks(A.shape[0], n_blocks)
    tasks = [(A, B, r0, r1) for r0, r1 in blocks]
    C = np.empty((A.shape[0], B.shape[1]), dtype=A.dtype)
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_multiply_block, t) for t in tasks]
        for fut in as_completed(futures):
            r0, r1, Cb = fut.result()
            C[r0:r1, :] = Cb
    return C, time.perf_counter() - t0


if __name__ == "__main__":
    A = np.load("test_data/matA.npy")
    B = np.load("test_data/matB.npy")
    print(f"Matrices: A={A.shape} B={B.shape}")
    C_ref, t = run_sequential(A, B)
    print(f"  seq: t={t:.2f}s, C[0,0]={C_ref[0,0]:.4f}")
    C_par, t = run_map_reduce(A, B, 8)
    print(f"  mapreduce w=8: t={t:.2f}s, ok={np.allclose(C_par, C_ref)}")
