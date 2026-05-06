"""
Задача 1.1: Підрахунок частоти HTML-тегів у наборі документів.

Модель. Імітуємо типовий веб-скрейпінг: для кожного документа спочатку
виконується «віддалений запит» (симульований time.sleep, IO_LATENCY_MS),
потім локальний CPU-парсинг через regex. У такому режимі більшу частину часу
потоки очікують на «мережу», тому всі патерни демонструють реальне прискорення
з ThreadPoolExecutor (під час time.sleep GIL вільний).

Реалізовано чотири версії:
  • sequential   — baseline
  • Map-Reduce   — рівне розбиття + Counter sum
  • Fork-Join    — рекурсивний бінарний поділ + flat-execution + reduce
  • Worker Pool  — фіксований пул, .submit() в циклі, as_completed
"""
from __future__ import annotations
import os
import re
import time
from collections import Counter
from functools import reduce
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

_IO_LATENCY = float(os.environ.get("IO_LATENCY_MS", "15")) / 1000.0

TAG_RE = re.compile(r"<\s*([a-zA-Z][a-zA-Z0-9]*)\b")
CLOSING_TAG_RE = re.compile(r"</\s*([a-zA-Z][a-zA-Z0-9]*)\s*>")


def count_tags_in_file(path: Path) -> Counter:
    """Атомарна одиниця роботи: «завантажуємо» + парсимо один файл."""
    if _IO_LATENCY > 0:
        time.sleep(_IO_LATENCY)
    text = Path(path).read_text(encoding="utf-8", errors="ignore").lower()
    c = Counter(TAG_RE.findall(text))
    c.update(CLOSING_TAG_RE.findall(text))
    return c


def list_html_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.html") if p.is_file())


def _count_batch(paths: list[Path]) -> Counter:
    total = Counter()
    for p in paths:
        total += count_tags_in_file(p)
    return total


def run_sequential(root: Path) -> tuple[Counter, float]:
    files = list_html_files(root)
    t0 = time.perf_counter()
    total = _count_batch(files)
    return total, time.perf_counter() - t0


def run_map_reduce(root: Path, workers: int) -> tuple[Counter, float]:
    files = list_html_files(root)
    chunk_size = max(1, len(files) // (workers * 4))
    chunks = [files[i:i + chunk_size] for i in range(0, len(files), chunk_size)]
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        partial_counts = list(pool.map(_count_batch, chunks))
    total = reduce(lambda a, b: a + b, partial_counts, Counter())
    return total, time.perf_counter() - t0


def _split_recursive(items: list, threshold: int) -> list[list]:
    if len(items) <= threshold:
        return [items]
    mid = len(items) // 2
    return _split_recursive(items[:mid], threshold) + _split_recursive(items[mid:], threshold)


def run_fork_join(root: Path, workers: int, threshold: int = 32) -> tuple[Counter, float]:
    files = list_html_files(root)
    chunks = _split_recursive(files, threshold)
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        partials = list(pool.map(_count_batch, chunks))
    total = reduce(lambda a, b: a + b, partials, Counter())
    return total, time.perf_counter() - t0


def run_worker_pool(root: Path, workers: int) -> tuple[Counter, float]:
    files = list_html_files(root)
    batch_size = max(1, len(files) // (workers * 8))
    batches = [files[i:i + batch_size] for i in range(0, len(files), batch_size)]
    t0 = time.perf_counter()
    total = Counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_count_batch, b) for b in batches]
        for fut in as_completed(futures):
            total += fut.result()
    return total, time.perf_counter() - t0
