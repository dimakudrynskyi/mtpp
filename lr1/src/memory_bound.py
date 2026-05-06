"""
Memory-bound задача: транспонування матриці великої розмірності.

Особливість: ця задача обмежена не CPU, а пропускною здатністю пам'яті.
Транспонування — це фактично перестановка байтів між регіонами пам'яті,
тому додавання потоків понад певну межу не дає прискорення (а часто навпаки).
"""
from __future__ import annotations
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor


def make_matrix(n: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # int32 — 4 байти на елемент: для 10000x10000 це ≈ 400 МБ
    return rng.integers(0, 1_000_000, size=(n, n), dtype=np.int32)


def transpose_sequential(matrix: np.ndarray) -> tuple[np.ndarray, float]:
    """np.transpose повертає лише view; .copy() реально переміщує дані в пам'яті."""
    t0 = time.perf_counter()
    result = matrix.T.copy()
    return result, time.perf_counter() - t0


def _transpose_block(args):
    """Транспонує один горизонтальний блок матриці."""
    matrix, row_start, row_end = args
    return row_start, row_end, matrix[row_start:row_end, :].T.copy()


def transpose_parallel(matrix: np.ndarray, workers: int, executor_cls) -> tuple[np.ndarray, float]:
    n_rows, n_cols = matrix.shape
    block = max(1, n_rows // workers)
    tasks = []
    for i in range(workers):
        rs = i * block
        re = n_rows if i == workers - 1 else (i + 1) * block
        tasks.append((matrix, rs, re))

    result = np.empty((n_cols, n_rows), dtype=matrix.dtype)

    t0 = time.perf_counter()
    with executor_cls(max_workers=workers) as ex:
        for rs, re, sub_t in ex.map(_transpose_block, tasks):
            result[:, rs:re] = sub_t
    return result, time.perf_counter() - t0
