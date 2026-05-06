"""
Задача 2: Обробка зображень — patterns Pipeline і Producer-Consumer.

Кожен кадр проходить чотири етапи:
  1. DECODE     — Image.open + load з диску (I/O + CPU)
  2. FILTER     — застосування Gaussian blur (CPU, Pillow звільняє GIL)
  3. WATERMARK  — додавання водяного знаку (CPU)
  4. ENCODE     — збереження як JPEG на диск (CPU + I/O)

PIPELINE: усі кадри проходять усі етапи послідовно, але різні етапи різних
кадрів можуть виконуватися паралельно — як конвеєр на заводі. Реалізовано
через ланцюжок queue.Queue, кожен етап — окремий потік.

PRODUCER-CONSUMER: producer створює задачі (paths), споживачі (worker threads)
беруть задачі з shared queue і виконують ВСІ етапи самостійно. Класичний
worker-pool pattern із спільною чергою.
"""
from __future__ import annotations
import os
import time
import queue
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from PIL import Image, ImageFilter, ImageDraw, ImageFont

_IO_LATENCY = float(os.environ.get("IO_LATENCY_MS", "10")) / 1000.0
SENTINEL = None  # маркер кінця черги


# ---------- Етапи обробки (атомарні операції) ----------

def stage_decode(path: Path) -> tuple[Path, Image.Image]:
    """Декодування JPEG з диску."""
    if _IO_LATENCY > 0:
        time.sleep(_IO_LATENCY)
    img = Image.open(path).convert("RGB")
    img.load()  # форсуємо повне декодування
    return path, img


def stage_filter(item: tuple[Path, Image.Image]) -> tuple[Path, Image.Image]:
    """Накладання Gaussian Blur."""
    path, img = item
    img = img.filter(ImageFilter.GaussianBlur(radius=2.5))
    return path, img


def stage_watermark(item: tuple[Path, Image.Image]) -> tuple[Path, Image.Image]:
    """Накладання водяного знаку."""
    path, img = item
    draw = ImageDraw.Draw(img)
    text = "© Lab 2 / 2026"
    # Розташовуємо в нижньому правому куті
    w, h = img.size
    draw.text((w - 200, h - 30), text, fill=(255, 255, 255))
    return path, img


def stage_encode(item: tuple[Path, Image.Image], out_dir: Path) -> Path:
    """Збереження як JPEG."""
    path, img = item
    out_path = out_dir / path.name
    img.save(out_path, "JPEG", quality=85)
    if _IO_LATENCY > 0:
        time.sleep(_IO_LATENCY)
    return out_path


def process_one_image(path: Path, out_dir: Path) -> Path:
    """Виконує всі 4 етапи послідовно — атомарна одиниця для Producer-Consumer."""
    item = stage_decode(path)
    item = stage_filter(item)
    item = stage_watermark(item)
    return stage_encode(item, out_dir)


# ---------- Sequential ----------

def run_sequential(in_dir: Path, out_dir: Path) -> tuple[int, float]:
    """Усі кадри обробляються один за одним."""
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(in_dir.glob("*.jpg"))
    t0 = time.perf_counter()
    for f in files:
        process_one_image(f, out_dir)
    return len(files), time.perf_counter() - t0


# ---------- PIPELINE ----------

def run_pipeline(in_dir: Path, out_dir: Path, stage_workers: int = 1) -> tuple[int, float]:
    """
    PIPELINE pattern: 4 етапи — 4 потоки (або групи потоків). Кожен етап має
    власну вхідну чергу. Кадри ллються через конвеєр: коли етап 1 закінчив
    кадр, він кладе його в чергу етапу 2 і одразу бере наступний.

    stage_workers — скільки worker'ів на КОЖНОМУ етапі. При stage_workers=1
    маємо чистий 4-stage pipeline.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(in_dir.glob("*.jpg"))

    # Черги між етапами; невеликий maxsize = backpressure
    q1 = queue.Queue(maxsize=8)  # decode -> filter
    q2 = queue.Queue(maxsize=8)  # filter -> watermark
    q3 = queue.Queue(maxsize=8)  # watermark -> encode
    done = threading.Event()
    counter = [0]
    counter_lock = threading.Lock()

    def stage_worker(in_q, out_q, fn):
        while True:
            item = in_q.get()
            if item is SENTINEL:
                in_q.task_done()
                # Передаємо sentinel далі ОДИН раз на групу
                if out_q is not None:
                    out_q.put(SENTINEL)
                break
            result = fn(item)
            if out_q is not None:
                out_q.put(result)
            else:
                with counter_lock:
                    counter[0] += 1
            in_q.task_done()

    t0 = time.perf_counter()

    # Запускаємо потоки для кожного етапу
    threads = []
    # Decode (читає з диску — в нас джерело — список files)
    def decoder():
        for f in files:
            q1.put(stage_decode(f))
        q1.put(SENTINEL)
    threads.append(threading.Thread(target=decoder, name="decoder"))

    threads.append(threading.Thread(target=stage_worker, args=(q1, q2, stage_filter), name="filter"))
    threads.append(threading.Thread(target=stage_worker, args=(q2, q3, stage_watermark), name="watermark"))
    threads.append(threading.Thread(target=stage_worker,
                                     args=(q3, None, lambda it: stage_encode(it, out_dir)),
                                     name="encoder"))

    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return counter[0], time.perf_counter() - t0


# ---------- PRODUCER-CONSUMER ----------

def run_producer_consumer(in_dir: Path, out_dir: Path, n_consumers: int) -> tuple[int, float]:
    """
    PRODUCER-CONSUMER: один producer кладе шляхи в чергу, n_consumers
    consumer-потоків забирають і обробляють кожен кадр повністю (всі 4 етапи).
    Класичний work-stealing підхід.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(in_dir.glob("*.jpg"))

    task_q = queue.Queue(maxsize=n_consumers * 2)
    counter = [0]
    counter_lock = threading.Lock()

    def consumer():
        while True:
            path = task_q.get()
            if path is SENTINEL:
                task_q.task_done()
                break
            process_one_image(path, out_dir)
            with counter_lock:
                counter[0] += 1
            task_q.task_done()

    t0 = time.perf_counter()

    consumers = [threading.Thread(target=consumer, name=f"consumer-{i}")
                 for i in range(n_consumers)]
    for c in consumers:
        c.start()

    # Producer: один потік (головний) кладе всі задачі
    for f in files:
        task_q.put(f)
    # Сигнал кінця для кожного consumer
    for _ in consumers:
        task_q.put(SENTINEL)

    for c in consumers:
        c.join()
    return counter[0], time.perf_counter() - t0


if __name__ == "__main__":
    in_dir = Path("test_data/images")
    out_dir = Path("test_data/output_images")
    n, t = run_sequential(in_dir, out_dir)
    print(f"  sequential: {n} images, t={t:.2f}s")
    n, t = run_pipeline(in_dir, out_dir)
    print(f"  pipeline:   {n} images, t={t:.2f}s")
    n, t = run_producer_consumer(in_dir, out_dir, n_consumers=4)
    print(f"  prod-cons w=4: {n} images, t={t:.2f}s")
