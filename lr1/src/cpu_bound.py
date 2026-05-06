"""
CPU-bound задачі:
  1. Обчислення числа π методом Монте-Карло
  2. Факторизація великих чисел
  3. Пошук простих чисел у заданому діапазоні

Реалізовано: послідовна версія, ThreadPoolExecutor (для демонстрації впливу GIL),
ProcessPoolExecutor (реальний паралелізм).
"""
from __future__ import annotations
import math
import random
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor


# ---------- 1. Monte-Carlo Pi ----------

def mc_pi_chunk(n_points: int, seed: int = 0) -> int:
    """Підрахунок точок, що потрапили в чверть кола, для одного шматка."""
    rnd = random.Random(seed)
    inside = 0
    for _ in range(n_points):
        x = rnd.random()
        y = rnd.random()
        if x * x + y * y <= 1.0:
            inside += 1
    return inside


def pi_sequential(total_points: int) -> tuple[float, float]:
    t0 = time.perf_counter()
    inside = mc_pi_chunk(total_points, seed=42)
    pi = 4.0 * inside / total_points
    return pi, time.perf_counter() - t0


def pi_parallel(total_points: int, workers: int, executor_cls) -> tuple[float, float]:
    chunk = total_points // workers
    chunks = [chunk] * workers
    chunks[-1] += total_points - chunk * workers
    seeds = [42 + i for i in range(workers)]

    t0 = time.perf_counter()
    with executor_cls(max_workers=workers) as ex:
        results = list(ex.map(mc_pi_chunk, chunks, seeds))
    pi = 4.0 * sum(results) / total_points
    return pi, time.perf_counter() - t0


# ---------- 2. Factorization ----------

def factorize(n: int) -> list[int]:
    """Trial division — навмисно «наївний» алгоритм для CPU-навантаження."""
    factors = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors


def factorize_sequential(numbers: list[int]) -> tuple[list[list[int]], float]:
    t0 = time.perf_counter()
    out = [factorize(n) for n in numbers]
    return out, time.perf_counter() - t0


def factorize_parallel(numbers: list[int], workers: int, executor_cls) -> tuple[list[list[int]], float]:
    t0 = time.perf_counter()
    with executor_cls(max_workers=workers) as ex:
        out = list(ex.map(factorize, numbers))
    return out, time.perf_counter() - t0


# ---------- 3. Primes in range ----------

def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False
    r = int(math.isqrt(n))
    for d in range(3, r + 1, 2):
        if n % d == 0:
            return False
    return True


def primes_in_subrange(args: tuple[int, int]) -> int:
    lo, hi = args
    return sum(1 for n in range(lo, hi) if is_prime(n))


def primes_sequential(lo: int, hi: int) -> tuple[int, float]:
    t0 = time.perf_counter()
    count = primes_in_subrange((lo, hi))
    return count, time.perf_counter() - t0


def primes_parallel(lo: int, hi: int, workers: int, executor_cls) -> tuple[int, float]:
    step = (hi - lo) // workers
    ranges = []
    cur = lo
    for i in range(workers):
        nxt = cur + step if i < workers - 1 else hi
        ranges.append((cur, nxt))
        cur = nxt

    t0 = time.perf_counter()
    with executor_cls(max_workers=workers) as ex:
        partials = list(ex.map(primes_in_subrange, ranges))
    return sum(partials), time.perf_counter() - t0
